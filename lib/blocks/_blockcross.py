from __future__ import annotations

import math
from dataclasses import dataclass
from collections import OrderedDict

import torch                as th
import torch.nn             as nn
import torch.nn.functional  as F

import triton
import triton.language      as tl


##################
##### CONFIG #####
##################


@dataclass
class BlockCrossAttentionConfig:
    embed_dim:    int   = 768
    num_heads:    int   = 12
    num_kv_heads: int   = None
    head_dim:     int   = None
    block_size:   int   = 16      # Same as decoder self-attention block_size
    dropout:      float = 0.0

    def __post_init__(self):

        if self.num_kv_heads is None:
            self.num_kv_heads = max(1, self.num_heads // 4)
        if self.head_dim is None:
            self.head_dim = self.embed_dim // self.num_heads


def _check_triton_support():
    """Check if Triton is supported on this device"""

    import os
    if os.environ.get("GENET5_DISABLE_CROSS_TRITON", "0") == "1":
        return False
    if not th.cuda.is_available():
        return False
    cap = th.cuda.get_device_capability()
    return cap[0] >= 7


##########################
##### TRITON KERNEL  #####
##########################


@triton.jit
def block_cross_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    Mask_ptr,
    num_blocks, num_kv_tokens, head_dim,
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
    Block cross-attention with online softmax (Flash Attention style)

    Processes query blocks attending to all KV tokens without materializing
    the full attention matrix. Uses online softmax for numerical stability.

    Grid: (batch, num_kv_heads, num_q_blocks)
    """

    pid_batch   = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    pid_q_block = tl.program_id(2)

    # Query block boundaries
    q_start = pid_q_block * BLOCK_Q
    offs_q  = q_start + tl.arange(0, BLOCK_Q)
    offs_d  = tl.arange(0, BLOCK_D)
    mask_q  = offs_q < num_blocks

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

    # Initialize accumulators for online softmax
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

        # Base validity mask
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

    # Finalize and store outputs
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


##########################
##### BLOCK CROSS    #####
##########################


class BlockCrossAttention(nn.Module):
    """
    Block-wise cross-attention for efficient decoder-encoder attention

    Instead of each decoder token attending to all encoder tokens independently,
    we group decoder tokens into blocks and compute ONE cross-attention per block.
    The result is broadcast to all tokens in the block.

    Uses Triton kernel with online softmax to avoid materializing full attention matrix.

    Memory reduction: seq_len/block_size factor
    - 171K tokens / 16 = 10.7K blocks
    - 10.7K × 512 latents = 5.5M vs 171K × 512 = 87M (16x reduction)
    - Plus: no materialized attention matrix (streaming computation)
    """

    # Kernel configuration
    BLOCK_Q  = 16   # Query block size for Triton grid
    BLOCK_KV = 64   # KV block size for iteration

    def __init__(self, config):

        super().__init__()

        self.config          = config
        self.embed_dim       = config.embed_dim
        self.num_heads       = config.num_heads
        self.num_kv_heads    = config.num_kv_heads
        self.head_dim        = config.head_dim
        self.block_size      = config.block_size
        self.heads_per_group = config.num_heads // config.num_kv_heads
        self.softmax_scale   = 1.0 / math.sqrt(self.head_dim)

        assert config.num_heads % config.num_kv_heads == 0
        assert self.heads_per_group <= 4, "Triton kernel supports up to 4 heads per group"

        # Query projection (for pooled decoder blocks)
        self.q = nn.Linear(config.embed_dim, config.num_heads * self.head_dim, bias=False)

        # Key/Value projection (for encoder)
        self.k = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.o = nn.Linear(config.num_heads * self.head_dim, config.embed_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        self._triton_supported = _check_triton_support()

    def _pool_to_blocks(self, x, pad_len):
        """Pool sequence into blocks by averaging

        Args:
            x:       [B, L_padded, D] input tensor (already padded)
            pad_len: padding length for tracking

        Returns:
            pooled: [B, num_blocks, D]
        """

        B, L_padded, D = x.shape
        num_blocks     = L_padded // self.block_size

        # Reshape and pool: [B, num_blocks, block_size, D] -> mean -> [B, num_blocks, D]
        # Contiguous view for efficient memory access
        x_blocks = x.view(B, num_blocks, self.block_size, D)
        pooled   = x_blocks.mean(dim=2)

        return pooled

    def _unpool_from_blocks(self, pooled, original_len):
        """Broadcast block representations back to token level

        Uses repeat_interleave which is more compile-friendly than expand+reshape
        """

        B, num_blocks, D = pooled.shape

        # Efficient broadcast: [B, num_blocks, D] -> [B, num_blocks * block_size, D]
        # repeat_interleave is well-optimized and compile-friendly
        unpooled = pooled.repeat_interleave(self.block_size, dim=1)

        # Slice to original length (compile-friendly static slicing when possible)
        unpooled = unpooled[:, :original_len, :]

        return unpooled

    def _forward_triton(self, q, k, v, attention_mask, num_blocks, L_enc):
        """Triton kernel path - memory efficient with online softmax"""

        B = q.shape[0]

        # Reshape for kernel: [B, num_heads, num_blocks, head_dim]
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

        # Grid: (batch, num_kv_heads, num_q_blocks)
        num_q_blocks = (num_blocks + self.BLOCK_Q - 1) // self.BLOCK_Q
        grid = (B, self.num_kv_heads, num_q_blocks)

        block_cross_attention_fwd_kernel[grid](
            q, k, v, out,
            attention_mask if use_mask else q,
            num_blocks, L_enc, self.head_dim,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            stride_mb, stride_mk,
            use_mask,
            self.softmax_scale,
            BLOCK_Q         = self.BLOCK_Q,
            BLOCK_KV        = self.BLOCK_KV,
            BLOCK_D         = self.head_dim,
            HEADS_PER_GROUP = self.heads_per_group,
        )

        # Reshape back: [B, num_blocks, num_heads * head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, num_blocks, self.num_heads * self.head_dim)

        return out

    def _forward_pytorch(self, q, k, v, attention_mask, num_blocks, L_enc):
        """PyTorch training path using F.scaled_dot_product_attention"""

        B = q.shape[0]

        # Reshape for SDPA: [B, num_heads, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)  # [B, num_heads, num_blocks, head_dim]
        k = k.permute(0, 2, 1, 3)  # [B, num_kv_heads, L_enc, head_dim]
        v = v.permute(0, 2, 1, 3)

        # GQA via expand (no memory copy, just view with stride tricks)
        # Reshape to [B, num_kv_heads, 1, L_enc, head_dim] then expand
        if self.heads_per_group > 1:
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.heads_per_group, L_enc, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.heads_per_group, L_enc, self.head_dim)
            # Reshape to [B, num_heads, L_enc, head_dim]
            k = k.reshape(B, self.num_heads, L_enc, self.head_dim)
            v = v.reshape(B, self.num_heads, L_enc, self.head_dim)

        # Build attention mask for SDPA
        # SDPA expects: [B, 1, num_blocks, L_enc] or [B, num_heads, num_blocks, L_enc]
        # Values: 0 = attend, -inf = mask out
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [B, L_enc] with 1=valid, 0=pad
            # Convert to: [B, 1, 1, L_enc] with 0.0=valid, -inf=pad
            attn_mask = th.where(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0,
                th.tensor(float('-inf'), device=q.device, dtype=q.dtype),
                th.tensor(0.0, device=q.device, dtype=q.dtype)
            )

        # Use scaled_dot_product_attention (flash attention when available)
        # Handles softmax, scaling, and dropout internally
        # Memory efficient: doesn't materialize full attention matrix
        dropout_p = self.config.dropout if self.training else 0.0
        attn_out  = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask   = attn_mask,
            dropout_p   = dropout_p,
            is_causal   = False,
            scale       = self.softmax_scale,
        )

        # Reshape: [B, num_blocks, num_heads * head_dim]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, num_blocks, self.num_heads * self.head_dim)

        return attn_out

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None):
        """
        Block-wise cross-attention

        Args:
            hidden_states:         [B, decoder_len, D] decoder hidden states
            encoder_hidden_states: [B, encoder_len, D] encoder output (or latents)
            attention_mask:        [B, encoder_len] optional mask (1=valid, 0=pad)

        Returns:
            output: [B, decoder_len, D] cross-attention output
        """

        B, L_dec, D = hidden_states.shape
        L_enc       = encoder_hidden_states.shape[1]

        # Compute padding for block alignment
        pad_len = (self.block_size - L_dec % self.block_size) % self.block_size

        # Pad decoder if necessary (fused with pooling)
        if pad_len > 0:
            hidden_padded = F.pad(hidden_states, (0, 0, 0, pad_len))
        else:
            hidden_padded = hidden_states

        L_padded   = L_dec + pad_len
        num_blocks = L_padded // self.block_size

        # Pool decoder to blocks
        pooled_dec = self._pool_to_blocks(hidden_padded, pad_len)

        # Project queries from pooled decoder blocks
        q = self.q(pooled_dec).view(B, num_blocks, self.num_heads, self.head_dim)

        # Project keys and values from encoder
        k = self.k(encoder_hidden_states).view(B, L_enc, self.num_kv_heads, self.head_dim)
        v = self.v(encoder_hidden_states).view(B, L_enc, self.num_kv_heads, self.head_dim)

        # Choose path: Triton for inference (no grad), PyTorch for training
        # Triton kernel doesn't have custom backward, so fall back to PyTorch when gradients needed
        use_triton = (
            self._triton_supported and
            hidden_states.is_cuda and
            not th.is_grad_enabled()
        )

        if use_triton:
            attn_out = self._forward_triton(q, k, v, attention_mask, num_blocks, L_enc)
        else:
            attn_out = self._forward_pytorch(q, k, v, attention_mask, num_blocks, L_enc)

        # Output projection
        attn_out = self.o(attn_out)

        # Unpool back to token level
        output = self._unpool_from_blocks(attn_out, L_dec)

        output = self.dropout(output)

        return output, None

    def get_memory_info(self, decoder_len, encoder_len):
        """Return memory usage comparison"""

        block_count    = (decoder_len + self.block_size - 1) // self.block_size
        dense_scores   = decoder_len * encoder_len * self.num_heads
        sparse_scores  = block_count * encoder_len * self.num_heads
        reduction      = dense_scores / max(sparse_scores, 1)

        # Triton kernel doesn't materialize scores at all
        triton_scores = 0  # Online softmax, no materialization

        return {
            "dense_scores":       dense_scores,
            "block_scores":       sparse_scores,
            "triton_scores":      triton_scores,
            "reduction_vs_dense": reduction,
            "reduction_factor":   reduction,
            "block_count":        block_count,
        }
