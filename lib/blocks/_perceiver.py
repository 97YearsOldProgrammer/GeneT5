from __future__ import annotations

import math
from dataclasses import dataclass

import torch                       as th
import torch.nn                    as nn
import torch.nn.functional         as F
import torch.utils.checkpoint      as ckpt

import triton
import triton.language             as tl

from lib.blocks._component         import LayerNorm


####################
#####  CONFIG  #####
####################


@dataclass
class PerceiverConfig:
    embed_dim:               int   = 768
    num_latents:             int   = 512
    num_heads:               int   = 12
    num_kv_heads:            int   = None
    num_layers:              int   = 2
    ff_dim:                  int   = None
    dropout:                 float = 0.0
    latent_init_std:         float = 0.02
    gradient_checkpointing:  bool  = False

    def __post_init__(self):

        if self.num_kv_heads is None:
            self.num_kv_heads = max(1, self.num_heads // 4)
        if self.ff_dim is None:
            self.ff_dim = self.embed_dim * 4


def _check_triton_support():
    """Check if Triton is supported"""

    import os
    if os.environ.get("GENET5_DISABLE_PERCEIVER_TRITON", "0") == "1":
        return False
    if not th.cuda.is_available():
        return False
    cap = th.cuda.get_device_capability()
    return cap[0] >= 7


##########################
##### TRITON KERNELS #####
##########################


@triton.jit
def cross_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    Mask_ptr,
    num_q_tokens, num_kv_tokens, head_dim,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kk, stride_kd,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    stride_mb, stride_mk,
    use_mask: tl.constexpr,
    softmax_scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
):
    """
    Cross-attention with online softmax (Flash Attention style)

    Grid: (batch, num_kv_heads, num_q_blocks)
    """

    pid_batch   = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    pid_q_block = tl.program_id(2)

    # Query block boundaries
    q_start = pid_q_block * BLOCK_Q
    offs_q  = q_start + tl.arange(0, BLOCK_Q)
    offs_d  = tl.arange(0, BLOCK_D)
    mask_q  = offs_q < num_q_tokens

    # Load Q for all heads in group
    q_base = Q_ptr + pid_batch * stride_qb + offs_q[:, None] * stride_qq + offs_d[None, :] * stride_qd

    q_0 = tl.load(q_base + (pid_kv_head * HEADS_PER_GROUP + 0) * stride_qh, mask=mask_q[:, None], other=0.0)
    q_1 = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    q_2 = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    q_3 = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    if HEADS_PER_GROUP > 1:
        q_1 = tl.load(q_base + (pid_kv_head * HEADS_PER_GROUP + 1) * stride_qh, mask=mask_q[:, None], other=0.0)
    if HEADS_PER_GROUP > 2:
        q_2 = tl.load(q_base + (pid_kv_head * HEADS_PER_GROUP + 2) * stride_qh, mask=mask_q[:, None], other=0.0)
    if HEADS_PER_GROUP > 3:
        q_3 = tl.load(q_base + (pid_kv_head * HEADS_PER_GROUP + 3) * stride_qh, mask=mask_q[:, None], other=0.0)

    # Initialize online softmax accumulators
    m_i_0 = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)
    m_i_1 = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)
    m_i_2 = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)
    m_i_3 = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)

    l_i_0 = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    l_i_1 = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    l_i_2 = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    l_i_3 = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    acc_0 = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    acc_1 = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    acc_2 = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    acc_3 = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    # Iterate over KV blocks
    num_kv_blocks = tl.cdiv(num_kv_tokens, BLOCK_KV)

    for kv_block_idx in range(num_kv_blocks):
        kv_start = kv_block_idx * BLOCK_KV
        offs_kv  = kv_start + tl.arange(0, BLOCK_KV)
        mask_kv  = offs_kv < num_kv_tokens

        # Load K and V (shared across all Q heads in group)
        k_ptrs = (K_ptr +
                  pid_batch * stride_kb +
                  pid_kv_head * stride_kh +
                  offs_kv[:, None] * stride_kk +
                  offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=mask_kv[:, None], other=0.0)

        v_ptrs = (V_ptr +
                  pid_batch * stride_vb +
                  pid_kv_head * stride_vh +
                  offs_kv[:, None] * stride_vk +
                  offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=mask_kv[:, None], other=0.0)

        k_t = tl.trans(k)

        # Base mask
        base_mask = mask_q[:, None] & mask_kv[None, :]

        # Apply attention mask if provided
        if use_mask:
            mask_ptrs  = Mask_ptr + pid_batch * stride_mb + offs_kv * stride_mk
            attn_mask  = tl.load(mask_ptrs, mask=mask_kv, other=0)
            valid_mask = attn_mask != 0
            base_mask  = base_mask & valid_mask[None, :]

        # Process head 0
        scores_0   = tl.dot(q_0, k_t) * softmax_scale
        scores_0   = tl.where(base_mask, scores_0, float('-inf'))
        m_ij_0     = tl.max(scores_0, axis=1)
        m_new_0    = tl.maximum(m_i_0, m_ij_0)
        both_inf_0 = (m_i_0 == float('-inf')) & (m_new_0 == float('-inf'))
        alpha_0    = tl.where(both_inf_0, 1.0, tl.exp(m_i_0 - m_new_0))
        p_0        = tl.where(both_inf_0[:, None], 0.0, tl.exp(scores_0 - m_new_0[:, None]))
        l_i_0      = alpha_0 * l_i_0 + tl.sum(p_0, axis=1)
        acc_0      = alpha_0[:, None] * acc_0 + tl.dot(p_0.to(v.dtype), v)
        m_i_0      = m_new_0

        if HEADS_PER_GROUP > 1:
            scores_1   = tl.dot(q_1, k_t) * softmax_scale
            scores_1   = tl.where(base_mask, scores_1, float('-inf'))
            m_ij_1     = tl.max(scores_1, axis=1)
            m_new_1    = tl.maximum(m_i_1, m_ij_1)
            both_inf_1 = (m_i_1 == float('-inf')) & (m_new_1 == float('-inf'))
            alpha_1    = tl.where(both_inf_1, 1.0, tl.exp(m_i_1 - m_new_1))
            p_1        = tl.where(both_inf_1[:, None], 0.0, tl.exp(scores_1 - m_new_1[:, None]))
            l_i_1      = alpha_1 * l_i_1 + tl.sum(p_1, axis=1)
            acc_1      = alpha_1[:, None] * acc_1 + tl.dot(p_1.to(v.dtype), v)
            m_i_1      = m_new_1

        if HEADS_PER_GROUP > 2:
            scores_2   = tl.dot(q_2, k_t) * softmax_scale
            scores_2   = tl.where(base_mask, scores_2, float('-inf'))
            m_ij_2     = tl.max(scores_2, axis=1)
            m_new_2    = tl.maximum(m_i_2, m_ij_2)
            both_inf_2 = (m_i_2 == float('-inf')) & (m_new_2 == float('-inf'))
            alpha_2    = tl.where(both_inf_2, 1.0, tl.exp(m_i_2 - m_new_2))
            p_2        = tl.where(both_inf_2[:, None], 0.0, tl.exp(scores_2 - m_new_2[:, None]))
            l_i_2      = alpha_2 * l_i_2 + tl.sum(p_2, axis=1)
            acc_2      = alpha_2[:, None] * acc_2 + tl.dot(p_2.to(v.dtype), v)
            m_i_2      = m_new_2

        if HEADS_PER_GROUP > 3:
            scores_3   = tl.dot(q_3, k_t) * softmax_scale
            scores_3   = tl.where(base_mask, scores_3, float('-inf'))
            m_ij_3     = tl.max(scores_3, axis=1)
            m_new_3    = tl.maximum(m_i_3, m_ij_3)
            both_inf_3 = (m_i_3 == float('-inf')) & (m_new_3 == float('-inf'))
            alpha_3    = tl.where(both_inf_3, 1.0, tl.exp(m_i_3 - m_new_3))
            p_3        = tl.where(both_inf_3[:, None], 0.0, tl.exp(scores_3 - m_new_3[:, None]))
            l_i_3      = alpha_3 * l_i_3 + tl.sum(p_3, axis=1)
            acc_3      = alpha_3[:, None] * acc_3 + tl.dot(p_3.to(v.dtype), v)
            m_i_3      = m_new_3

    # Finalize and store
    out_base = Out_ptr + pid_batch * stride_ob + offs_q[:, None] * stride_oq + offs_d[None, :] * stride_od

    zero_mask_0 = (l_i_0 == 0.0)[:, None]
    l_i_0       = tl.where(l_i_0 == 0.0, 1.0, l_i_0)
    acc_0       = tl.where(zero_mask_0, 0.0, acc_0 / l_i_0[:, None])
    tl.store(out_base + (pid_kv_head * HEADS_PER_GROUP + 0) * stride_oh, acc_0, mask=mask_q[:, None])

    if HEADS_PER_GROUP > 1:
        zero_mask_1 = (l_i_1 == 0.0)[:, None]
        l_i_1       = tl.where(l_i_1 == 0.0, 1.0, l_i_1)
        acc_1       = tl.where(zero_mask_1, 0.0, acc_1 / l_i_1[:, None])
        tl.store(out_base + (pid_kv_head * HEADS_PER_GROUP + 1) * stride_oh, acc_1, mask=mask_q[:, None])

    if HEADS_PER_GROUP > 2:
        zero_mask_2 = (l_i_2 == 0.0)[:, None]
        l_i_2       = tl.where(l_i_2 == 0.0, 1.0, l_i_2)
        acc_2       = tl.where(zero_mask_2, 0.0, acc_2 / l_i_2[:, None])
        tl.store(out_base + (pid_kv_head * HEADS_PER_GROUP + 2) * stride_oh, acc_2, mask=mask_q[:, None])

    if HEADS_PER_GROUP > 3:
        zero_mask_3 = (l_i_3 == 0.0)[:, None]
        l_i_3       = tl.where(l_i_3 == 0.0, 1.0, l_i_3)
        acc_3       = tl.where(zero_mask_3, 0.0, acc_3 / l_i_3[:, None])
        tl.store(out_base + (pid_kv_head * HEADS_PER_GROUP + 3) * stride_oh, acc_3, mask=mask_q[:, None])


@triton.jit
def self_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    seq_len, head_dim,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Self-attention with online softmax for latent refinement"""

    pid_batch = tl.program_id(0)
    pid_head  = tl.program_id(1)
    pid_block = tl.program_id(2)

    # Query block
    q_start = pid_block * BLOCK_M
    offs_m  = q_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, BLOCK_D)
    mask_m  = offs_m < seq_len

    # Load Q
    q_ptrs = (Q_ptr +
              pid_batch * stride_qb +
              pid_head * stride_qh +
              offs_m[:, None] * stride_qs +
              offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize accumulators
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Iterate over K,V blocks
    num_blocks = tl.cdiv(seq_len, BLOCK_N)

    for block_idx in range(num_blocks):
        kv_start = block_idx * BLOCK_N
        offs_n   = kv_start + tl.arange(0, BLOCK_N)
        mask_n   = offs_n < seq_len

        # Load K, V
        k_ptrs = (K_ptr +
                  pid_batch * stride_kb +
                  pid_head * stride_kh +
                  offs_n[:, None] * stride_ks +
                  offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        v_ptrs = (V_ptr +
                  pid_batch * stride_vb +
                  pid_head * stride_vh +
                  offs_n[:, None] * stride_vs +
                  offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        k_t = tl.trans(k)

        # Compute scores
        scores = tl.dot(q, k_t) * softmax_scale
        scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, float('-inf'))

        # Online softmax
        m_ij     = tl.max(scores, axis=1)
        m_new    = tl.maximum(m_i, m_ij)
        both_inf = (m_i == float('-inf')) & (m_new == float('-inf'))
        alpha    = tl.where(both_inf, 1.0, tl.exp(m_i - m_new))
        p        = tl.where(both_inf[:, None], 0.0, tl.exp(scores - m_new[:, None]))
        l_i      = alpha * l_i + tl.sum(p, axis=1)
        acc      = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)
        m_i      = m_new

    # Finalize
    zero_mask = (l_i == 0.0)[:, None]
    l_i       = tl.where(l_i == 0.0, 1.0, l_i)
    acc       = tl.where(zero_mask, 0.0, acc / l_i[:, None])

    # Store
    out_ptrs = (Out_ptr +
                pid_batch * stride_ob +
                pid_head * stride_oh +
                offs_m[:, None] * stride_os +
                offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


########################
#####  COMPONENTS  #####
########################


class CrossAttention(nn.Module):
    """Cross-attention with Triton kernel for memory efficiency"""

    BLOCK_Q  = 16   # Reduced for shared memory limits
    BLOCK_KV = 32

    def __init__(self, embed_dim, num_heads, num_kv_heads, dropout=0.0):

        super().__init__()

        self.embed_dim       = embed_dim
        self.num_heads       = num_heads
        self.num_kv_heads    = num_kv_heads
        self.head_dim        = embed_dim // num_heads
        self.scale           = 1.0 / math.sqrt(self.head_dim)
        self.heads_per_group = num_heads // num_kv_heads

        assert num_heads % num_kv_heads == 0
        assert self.heads_per_group <= 4, "Triton kernel supports up to 4 heads per group"

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self._triton_supported = _check_triton_support()

    def _forward_triton(self, q, k, v, attention_mask):
        """Triton kernel path"""

        B, L_q, H, D = q.shape
        L_kv         = k.shape[1]

        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        out = th.empty_like(q)

        use_mask = attention_mask is not None
        if use_mask:
            attention_mask = attention_mask.contiguous()
            stride_mb = attention_mask.stride(0)
            stride_mk = attention_mask.stride(1) if attention_mask.dim() > 1 else 1
        else:
            stride_mb = 0
            stride_mk = 0

        num_q_blocks = (L_q + self.BLOCK_Q - 1) // self.BLOCK_Q
        grid = (B, self.num_kv_heads, num_q_blocks)

        cross_attention_fwd_kernel[grid](
            q, k, v, out,
            attention_mask if use_mask else q,
            L_q, L_kv, self.head_dim,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            stride_mb, stride_mk,
            use_mask,
            self.scale,
            BLOCK_Q         = self.BLOCK_Q,
            BLOCK_KV        = self.BLOCK_KV,
            BLOCK_D         = self.head_dim,
            HEADS_PER_GROUP = self.heads_per_group,
        )

        out = out.permute(0, 2, 1, 3).reshape(B, L_q, self.num_heads * self.head_dim)

        return out

    def _expand_kv_for_gqa(self, k, v):
        """Expand KV heads for GQA using view/expand (no memory allocation)"""

        if self.heads_per_group == 1:
            return k, v

        B, H_kv, L, D = k.shape

        # Reshape to [B, num_kv_heads, 1, L, D] then expand to [B, num_kv_heads, heads_per_group, L, D]
        # Finally reshape to [B, num_heads, L, D]
        k = k.unsqueeze(2).expand(B, H_kv, self.heads_per_group, L, D)
        k = k.reshape(B, self.num_heads, L, D)

        v = v.unsqueeze(2).expand(B, H_kv, self.heads_per_group, L, D)
        v = v.reshape(B, self.num_heads, L, D)

        return k, v

    def _forward_pytorch(self, q, k, v, attention_mask):
        """PyTorch fallback with Flash Attention via SDPA"""

        B, L_q, H, D = q.shape

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Efficient GQA expansion using view/expand (no memory allocation)
        k, v = self._expand_kv_for_gqa(k, v)

        # Prepare attention mask for SDPA (additive mask)
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [B, L_kv] with 1=valid, 0=pad
            # SDPA additive mask: 0=attend, -inf=ignore
            attn_mask = th.zeros_like(attention_mask, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(attention_mask == 0, float('-inf'))
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L_kv]

        # Use SDPA for Flash Attention support (memory efficient)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask    = attn_mask,
            dropout_p    = self.dropout.p if self.training else 0.0,
            is_causal    = False,
            scale        = self.scale,
        )

        out = out.permute(0, 2, 1, 3).reshape(B, L_q, self.num_heads * self.head_dim)

        return out

    def forward(self, query, key_value, attention_mask=None):

        B, L_q, _ = query.shape
        L_kv      = key_value.shape[1]

        q = self.q_proj(query).view(B, L_q, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(B, L_kv, self.num_kv_heads, self.head_dim)
        v = self.v_proj(key_value).view(B, L_kv, self.num_kv_heads, self.head_dim)

        # Triton for inference, PyTorch for training (Triton lacks custom backward)
        use_triton = (
            self._triton_supported and
            query.is_cuda and
            not th.is_grad_enabled()
        )

        if use_triton:
            out = self._forward_triton(q, k, v, attention_mask)
        else:
            out = self._forward_pytorch(q, k, v, attention_mask)

        out = self.o_proj(out)

        return out


class SelfAttention(nn.Module):
    """Self-attention with Triton kernel for latent refinement"""

    BLOCK_M = 16   # Reduced for shared memory limits
    BLOCK_N = 16

    def __init__(self, embed_dim, num_heads, dropout=0.0):

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self._triton_supported = _check_triton_support()

    def _forward_triton(self, q, k, v):
        """Triton kernel path"""

        B, L, H, D = q.shape

        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        out = th.empty_like(q)

        num_blocks = (L + self.BLOCK_M - 1) // self.BLOCK_M
        grid = (B, self.num_heads, num_blocks)

        self_attention_fwd_kernel[grid](
            q, k, v, out,
            L, self.head_dim,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            self.scale,
            BLOCK_M = self.BLOCK_M,
            BLOCK_N = self.BLOCK_N,
            BLOCK_D = self.head_dim,
        )

        out = out.permute(0, 2, 1, 3).reshape(B, L, self.embed_dim)

        return out

    def _forward_pytorch(self, q, k, v):
        """PyTorch fallback with Flash Attention via SDPA"""

        B, L, H, D = q.shape

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Use SDPA for Flash Attention support (memory efficient)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = None,
            dropout_p = self.dropout.p if self.training else 0.0,
            is_causal = False,
            scale     = self.scale,
        )

        out = out.permute(0, 2, 1, 3).reshape(B, L, self.embed_dim)

        return out

    def forward(self, x):

        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        use_triton = (
            self._triton_supported and
            x.is_cuda and
            not th.is_grad_enabled()
        )

        if use_triton:
            out = self._forward_triton(q, k, v)
        else:
            out = self._forward_pytorch(q, k, v)

        out = self.o_proj(out)

        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU"""

    def __init__(self, embed_dim, ff_dim, dropout=0.0):

        super().__init__()

        self.fc1     = nn.Linear(embed_dim, ff_dim)
        self.fc2     = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class PerceiverLayer(nn.Module):
    """Single Perceiver layer: self-attention + feed-forward"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0):

        super().__init__()

        self.self_attn = SelfAttention(embed_dim, num_heads, dropout)
        self.norm1     = LayerNorm(embed_dim)
        self.ff        = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2     = LayerNorm(embed_dim)

    def forward(self, x):

        x = x + self.self_attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))

        return x


########################
#####  COMPRESSOR  #####
########################


class PerceiverCompressor(nn.Module):
    """
    Perceiver-style encoder compression using learned latent queries

    Compresses encoder output [B, N, D] to latent representation [B, L, D]
    where L << N. Decoder cross-attention then operates on L tokens instead of N.

    Uses Triton kernels for memory-efficient inference (no materialized attention matrix).
    Falls back to PyTorch with SDPA (Flash Attention) during training.
    Supports gradient checkpointing to reduce peak memory during backward pass.
    """

    def __init__(self, config):

        super().__init__()

        self.config                 = config
        self.embed_dim              = config.embed_dim
        self.num_latents            = config.num_latents
        self.num_heads              = config.num_heads
        self.num_kv_heads           = config.num_kv_heads
        self.gradient_checkpointing = config.gradient_checkpointing

        # Learned latent queries
        self.latents = nn.Parameter(th.empty(config.num_latents, config.embed_dim))

        # Cross-attention: latents query encoder
        self.cross_attn = CrossAttention(
            embed_dim    = config.embed_dim,
            num_heads    = config.num_heads,
            num_kv_heads = config.num_kv_heads,
            dropout      = config.dropout,
        )
        self.cross_norm = LayerNorm(config.embed_dim)

        # Self-attention layers for latent refinement
        self.layers = nn.ModuleList([
            PerceiverLayer(
                embed_dim = config.embed_dim,
                num_heads = config.num_heads,
                ff_dim    = config.ff_dim,
                dropout   = config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = LayerNorm(config.embed_dim)

        self._init_weights(config.latent_init_std)

    def _init_weights(self, latent_std):
        """Initialize weights following Perceiver paper"""

        nn.init.normal_(self.latents, mean=0.0, std=latent_std)

        for name, param in self.cross_attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)

    def set_gradient_checkpointing(self, enable):
        """Enable or disable gradient checkpointing"""

        self.gradient_checkpointing = enable

    def _checkpoint_forward(self, layer, x):
        """Wrapper for gradient checkpointing compatibility"""

        # use_reentrant=False is required for torch.compile compatibility
        return ckpt.checkpoint(layer, x, use_reentrant=False)

    def forward(self, encoder_hidden, encoder_mask=None):
        """
        Compress encoder output to latent representation

        Args:
            encoder_hidden: [B, N, D] encoder output
            encoder_mask:   [B, N] attention mask (1=valid, 0=pad)

        Returns:
            latents: [B, L, D] compressed representation
        """

        B = encoder_hidden.shape[0]

        # Expand latents for batch (contiguous for torch.compile)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Cross-attention: latents query encoder
        normed    = self.cross_norm(latents)
        cross_out = self.cross_attn(normed, encoder_hidden, encoder_mask)
        latents   = latents + cross_out

        # Self-attention refinement layers with optional gradient checkpointing
        use_ckpt = self.gradient_checkpointing and self.training

        for layer in self.layers:
            if use_ckpt:
                latents = self._checkpoint_forward(layer, latents)
            else:
                latents = layer(latents)

        latents = self.final_norm(latents)

        return latents

    def get_compression_ratio(self, encoder_len):
        """Return compression ratio for given encoder length"""

        return encoder_len / self.num_latents

    def get_memory_info(self, encoder_len):
        """Return memory usage comparison"""

        dense_cross  = self.num_latents * encoder_len * self.num_heads
        triton_cross = 0  # No materialization
        self_attn    = self.num_latents * self.num_latents * self.num_heads

        return {
            "dense_cross_scores":  dense_cross,
            "triton_cross_scores": triton_cross,
            "self_attn_scores":    self_attn,
            "total_dense":         dense_cross + self_attn,
            "total_triton":        triton_cross + self_attn,
        }


#######################
#####  UTILITIES  #####
#######################


def create_perceiver_compressor(
    embed_dim,
    num_latents            = 512,
    num_heads              = 12,
    num_kv_heads           = None,
    num_layers             = 2,
    ff_dim                 = None,
    dropout                = 0.0,
    latent_init_std        = 0.02,
    gradient_checkpointing = False,
):
    """Create PerceiverCompressor with given configuration"""

    config = PerceiverConfig(
        embed_dim              = embed_dim,
        num_latents            = num_latents,
        num_heads              = num_heads,
        num_kv_heads           = num_kv_heads,
        num_layers             = num_layers,
        ff_dim                 = ff_dim,
        dropout                = dropout,
        latent_init_std        = latent_init_std,
        gradient_checkpointing = gradient_checkpointing,
    )

    return PerceiverCompressor(config)


def estimate_memory_savings(encoder_len, decoder_len, num_latents, num_heads, dtype_bytes=2):
    """Estimate memory savings from Perceiver compression"""

    original_cross   = decoder_len * encoder_len * num_heads * dtype_bytes
    compression_cost = num_latents * encoder_len * num_heads * dtype_bytes
    perceiver_cross  = decoder_len * num_latents * num_heads * dtype_bytes

    return {
        'original_mb':       original_cross / 1e6,
        'compression_mb':    compression_cost / 1e6,
        'perceiver_mb':      perceiver_cross / 1e6,
        'savings_factor':    original_cross / perceiver_cross,
        'compression_ratio': encoder_len / num_latents,
    }
