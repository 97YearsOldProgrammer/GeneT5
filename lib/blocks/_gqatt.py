from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn            as nn
import torch.nn.functional as F

import triton
import triton.language as tl


##################
##### CONFIG #####
##################


@dataclass
class GQAttentionConfig:
    embed_dim:     int   = 768
    num_heads:     int   = 12
    num_kv_heads:  int   = 4
    head_dim:      int   = None
    dropout:       float = 0.0
    use_alibi:     bool  = False
    max_seq_len:   int   = 8192
    is_cross_attn: bool  = True


######################
##### TRITON OPS #####
######################


@triton.jit
def gqa_cross_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    q_segment_ids_ptr, kv_segment_ids_ptr,
    q_len, kv_len, num_heads, num_kv_heads, head_dim,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_qsegb, stride_qsegs,
    stride_kvsegb, stride_kvsegs,
    heads_per_group,
    use_segment_mask: tl.constexpr,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Segment-aware GQA cross-attention"""

    pid_batch = tl.program_id(0)
    pid_head  = tl.program_id(1)
    pid_m     = tl.program_id(2)

    kv_head = pid_head // heads_per_group

    q_start = pid_m * BLOCK_M
    offs_m  = q_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, BLOCK_D)

    mask_m = offs_m < q_len

    q_ptrs = (Q_ptr +
              pid_batch * stride_qb +
              pid_head * stride_qh +
              offs_m[:, None] * stride_qs +
              offs_d[None, :] * stride_qd)
    q      = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    if use_segment_mask:
        q_seg_ptrs = q_segment_ids_ptr + pid_batch * stride_qsegb + offs_m * stride_qsegs
        q_seg_ids  = tl.load(q_seg_ptrs, mask=mask_m, other=-1)

    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, kv_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_len

        k_ptrs = (K_ptr +
                  pid_batch * stride_kb +
                  kv_head * stride_kh +
                  offs_n[:, None] * stride_ks +
                  offs_d[None, :] * stride_kd)
        k      = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k)) * softmax_scale

        if use_segment_mask:
            kv_seg_ptrs = kv_segment_ids_ptr + pid_batch * stride_kvsegb + offs_n * stride_kvsegs
            kv_seg_ids  = tl.load(kv_seg_ptrs, mask=mask_n, other=-1)
            same_seg    = (q_seg_ids[:, None] == kv_seg_ids[None, :]) & (q_seg_ids[:, None] >= 0)
            scores      = tl.where(same_seg & mask_m[:, None] & mask_n[None, :], scores, float('-inf'))
        else:
            scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, float('-inf'))

        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        beta  = tl.exp(m_ij - m_new)

        l_i = alpha * l_i + beta * tl.sum(tl.exp(scores - m_ij[:, None]), axis=1)

        v_ptrs = (V_ptr +
                  pid_batch * stride_vb +
                  kv_head * stride_vh +
                  offs_n[:, None] * stride_vs +
                  offs_d[None, :] * stride_vd)
        v      = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        p   = tl.exp(scores - m_ij[:, None])
        acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)

        m_i = m_new

    acc = acc / l_i[:, None]

    out_ptrs = (Out_ptr +
                pid_batch * stride_ob +
                pid_head * stride_oh +
                offs_m[:, None] * stride_os +
                offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


@triton.jit
def gqa_self_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    segment_ids_ptr,
    seq_len, num_heads, num_kv_heads, head_dim,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_segb, stride_segs,
    alibi_slopes_ptr,
    use_alibi: tl.constexpr,
    use_segment_mask: tl.constexpr,
    heads_per_group,
    softmax_scale,
    is_causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Segment-aware GQA self-attention"""

    pid_batch = tl.program_id(0)
    pid_head  = tl.program_id(1)
    pid_m     = tl.program_id(2)

    kv_head = pid_head // heads_per_group

    q_start = pid_m * BLOCK_M
    offs_m  = q_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, BLOCK_D)

    mask_m = offs_m < seq_len

    q_ptrs = (Q_ptr +
              pid_batch * stride_qb +
              pid_head * stride_qh +
              offs_m[:, None] * stride_qs +
              offs_d[None, :] * stride_qd)
    q      = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    if use_alibi:
        alibi_slope = tl.load(alibi_slopes_ptr + pid_head)

    if use_segment_mask:
        q_seg_ptrs = segment_ids_ptr + pid_batch * stride_segb + offs_m * stride_segs
        q_seg_ids  = tl.load(q_seg_ptrs, mask=mask_m, other=-1)

    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    kv_end = seq_len
    if is_causal:
        kv_end = min(seq_len, (pid_m + 1) * BLOCK_M)

    for n_start in range(0, kv_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        k_ptrs = (K_ptr +
                  pid_batch * stride_kb +
                  kv_head * stride_kh +
                  offs_n[:, None] * stride_ks +
                  offs_d[None, :] * stride_kd)
        k      = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k)) * softmax_scale

        if use_alibi:
            q_pos      = offs_m[:, None].to(tl.float32)
            k_pos      = offs_n[None, :].to(tl.float32)
            dist       = tl.abs(q_pos - k_pos)
            alibi_bias = -alibi_slope * dist
            scores     = scores + alibi_bias

        valid_mask = mask_m[:, None] & mask_n[None, :]

        if is_causal:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            valid_mask  = valid_mask & causal_mask

        if use_segment_mask:
            k_seg_ptrs = segment_ids_ptr + pid_batch * stride_segb + offs_n * stride_segs
            k_seg_ids  = tl.load(k_seg_ptrs, mask=mask_n, other=-1)
            same_seg   = (q_seg_ids[:, None] == k_seg_ids[None, :]) & (q_seg_ids[:, None] >= 0)
            valid_mask = valid_mask & same_seg

        scores = tl.where(valid_mask, scores, float('-inf'))

        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        beta  = tl.exp(m_ij - m_new)

        l_i = alpha * l_i + beta * tl.sum(tl.exp(scores - m_ij[:, None]), axis=1)

        v_ptrs = (V_ptr +
                  pid_batch * stride_vb +
                  kv_head * stride_vh +
                  offs_n[:, None] * stride_vs +
                  offs_d[None, :] * stride_vd)
        v      = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        p   = tl.exp(scores - m_ij[:, None])
        acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)

        m_i = m_new

    acc = acc / l_i[:, None]

    out_ptrs = (Out_ptr +
                pid_batch * stride_ob +
                pid_head * stride_oh +
                offs_m[:, None] * stride_os +
                offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


#####################################
#####  GROUPED QUERY ATTENTION  #####
#####################################


class GQAttention(nn.Module):
    """Grouped Query Attention with segment-aware masking"""

    def __init__(self, config, is_causal=False):
        super().__init__()

        self.config          = config
        self.is_causal       = is_causal
        self.is_cross_attn   = config.is_cross_attn
        self.embed_dim       = config.embed_dim
        self.num_heads       = config.num_heads
        self.num_kv_heads    = config.num_kv_heads
        self.head_dim        = config.head_dim if config.head_dim else config.embed_dim // config.num_heads
        self.heads_per_group = config.num_heads // config.num_kv_heads
        self.softmax_scale   = 1.0 / math.sqrt(self.head_dim)

        assert config.num_heads % config.num_kv_heads == 0, \
            "num_heads must be divisible by num_kv_heads"

        self.q = nn.Linear(config.embed_dim, config.num_heads * self.head_dim, bias=False)
        self.k = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(config.num_heads * self.head_dim, config.embed_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        if config.use_alibi and not config.is_cross_attn:
            slopes = self._compute_alibi_slopes(config.num_heads)
            self.register_buffer('alibi_slopes', slopes)
        else:
            self.alibi_slopes = None

        self._use_triton = True

    @staticmethod
    def _compute_alibi_slopes(num_heads):
        """Compute ALiBi slopes for position-aware attention"""

        def get_slopes(n):
            if n == 1:
                return torch.tensor([1.0])
            base = 2 ** (-2 ** -(math.log2(n) - 3))
            return torch.tensor([base ** i for i in range(1, n + 1)])

        if math.log2(num_heads).is_integer():
            return get_slopes(num_heads)
        else:
            closest_power = 2 ** math.floor(math.log2(num_heads))
            slopes        = torch.cat([
                get_slopes(closest_power),
                get_slopes(2 * closest_power)[0::2][:num_heads - closest_power]
            ])
            return slopes

    def _forward_pytorch_cross(self, hidden_states, key_value_states, attention_mask=None, q_segment_ids=None, kv_segment_ids=None):
        """PyTorch fallback for segment-aware cross-attention"""

        B, L_q, D = hidden_states.shape
        L_kv      = key_value_states.shape[1]

        q = self.q(hidden_states)
        k = self.k(key_value_states)
        v = self.v(key_value_states)

        q = q.view(B, L_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        q_grouped = q.view(B, self.num_kv_heads, self.heads_per_group, L_q, self.head_dim)
        scores    = torch.einsum('bghqd,bhkd->bghqk', q_grouped, k) * self.softmax_scale

        if q_segment_ids is not None and kv_segment_ids is not None:
            seg_mask = (q_segment_ids.unsqueeze(-1) == kv_segment_ids.unsqueeze(-2))
            seg_mask = seg_mask & (q_segment_ids.unsqueeze(-1) >= 0)
            seg_mask = seg_mask.unsqueeze(1).unsqueeze(2)
            scores   = scores.masked_fill(~seg_mask, float('-inf'))

        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(2)

        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = self.dropout(attn_weights)

        out = torch.einsum('bghqk,bhkd->bghqd', attn_weights, v)
        out = out.reshape(B, self.num_heads, L_q, self.head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, L_q, self.num_heads * self.head_dim)
        out = self.o(out)

        return out, None

    def _forward_pytorch_self(self, hidden_states, attention_mask=None, position_bias=None, segment_ids=None):
        """PyTorch fallback for segment-aware self-attention"""

        B, L, D = hidden_states.shape

        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        q = q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        q_grouped = q.view(B, self.num_kv_heads, self.heads_per_group, L, self.head_dim)
        scores    = torch.einsum('bghqd,bhkd->bghqk', q_grouped, k) * self.softmax_scale

        if self.alibi_slopes is not None and position_bias is None:
            positions     = torch.arange(L, device=hidden_states.device)
            dist_matrix   = torch.abs(positions[:, None] - positions[None, :]).float()
            alibi_slopes  = self.alibi_slopes.view(self.num_kv_heads, self.heads_per_group)
            position_bias = -alibi_slopes.view(1, self.num_kv_heads, self.heads_per_group, 1, 1) * \
                            dist_matrix.view(1, 1, 1, L, L)

        if position_bias is not None:
            scores = scores + position_bias

        if self.is_causal:
            causal_mask = torch.triu(torch.ones(L, L, device=scores.device), diagonal=1).bool()
            scores      = scores.masked_fill(causal_mask, float('-inf'))

        if segment_ids is not None:
            seg_mask = (segment_ids.unsqueeze(-1) == segment_ids.unsqueeze(-2))
            seg_mask = seg_mask & (segment_ids.unsqueeze(-1) >= 0)
            seg_mask = seg_mask.unsqueeze(1).unsqueeze(2)
            scores   = scores.masked_fill(~seg_mask, float('-inf'))

        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(2)

        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = self.dropout(attn_weights)

        out = torch.einsum('bghqk,bhkd->bghqd', attn_weights, v)
        out = out.reshape(B, self.num_heads, L, self.head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        out = self.o(out)

        return out, position_bias

    def _forward_triton_cross(self, hidden_states, key_value_states, attention_mask=None, q_segment_ids=None, kv_segment_ids=None):
        """Triton-optimized segment-aware cross-attention"""

        B, L_q, D = hidden_states.shape
        L_kv      = key_value_states.shape[1]

        q = self.q(hidden_states).view(B, L_q, self.num_heads, self.head_dim)
        k = self.k(key_value_states).view(B, L_kv, self.num_kv_heads, self.head_dim)
        v = self.v(key_value_states).view(B, L_kv, self.num_kv_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        out = torch.empty_like(q)

        use_segment_mask = q_segment_ids is not None and kv_segment_ids is not None

        if use_segment_mask:
            q_segment_ids  = q_segment_ids.contiguous()
            kv_segment_ids = kv_segment_ids.contiguous()
            stride_qsegb   = q_segment_ids.stride(0)
            stride_qsegs   = q_segment_ids.stride(1) if q_segment_ids.dim() > 1 else 1
            stride_kvsegb  = kv_segment_ids.stride(0)
            stride_kvsegs  = kv_segment_ids.stride(1) if kv_segment_ids.dim() > 1 else 1
        else:
            stride_qsegb  = 0
            stride_qsegs  = 0
            stride_kvsegb = 0
            stride_kvsegs = 0

        BLOCK_M = min(64, L_q)
        BLOCK_N = min(64, L_kv)
        BLOCK_D = self.head_dim

        num_m_blocks = (L_q + BLOCK_M - 1) // BLOCK_M
        grid         = (B, self.num_heads, num_m_blocks)

        gqa_cross_attention_fwd_kernel[grid](
            q, k, v, out,
            q_segment_ids if use_segment_mask else q,
            kv_segment_ids if use_segment_mask else k,
            L_q, L_kv, self.num_heads, self.num_kv_heads, self.head_dim,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            stride_qsegb, stride_qsegs,
            stride_kvsegb, stride_kvsegs,
            self.heads_per_group,
            use_segment_mask,
            self.softmax_scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

        out = out.permute(0, 2, 1, 3).reshape(B, L_q, self.num_heads * self.head_dim)
        out = self.o(out)
        out = self.dropout(out)

        return out, None

    def _forward_triton_self(self, hidden_states, attention_mask=None, segment_ids=None):
        """Triton-optimized segment-aware self-attention"""

        B, L, D = hidden_states.shape

        q = self.q(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        out = torch.empty_like(q)

        use_segment_mask = segment_ids is not None

        if use_segment_mask:
            segment_ids = segment_ids.contiguous()
            stride_segb = segment_ids.stride(0)
            stride_segs = segment_ids.stride(1) if segment_ids.dim() > 1 else 1
        else:
            stride_segb = 0
            stride_segs = 0

        BLOCK_M = min(64, L)
        BLOCK_N = min(64, L)
        BLOCK_D = self.head_dim

        num_m_blocks = (L + BLOCK_M - 1) // BLOCK_M
        grid         = (B, self.num_heads, num_m_blocks)

        alibi_ptr = self.alibi_slopes if self.alibi_slopes is not None else q

        gqa_self_attention_fwd_kernel[grid](
            q, k, v, out,
            segment_ids if use_segment_mask else q,
            L, self.num_heads, self.num_kv_heads, self.head_dim,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            stride_segb, stride_segs,
            alibi_ptr,
            self.alibi_slopes is not None,
            use_segment_mask,
            self.heads_per_group,
            self.softmax_scale,
            self.is_causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

        out = out.permute(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        out = self.o(out)
        out = self.dropout(out)

        return out, None

    def forward(self, hidden_states, key_value_states=None, attention_mask=None, position_bias=None, segment_ids=None, q_segment_ids=None, kv_segment_ids=None):
        """Forward with segment-aware masking for packed sequences"""

        if self.is_cross_attn and key_value_states is not None:
            if self._use_triton and hidden_states.is_cuda:
                try:
                    return self._forward_triton_cross(hidden_states, key_value_states, attention_mask, q_segment_ids, kv_segment_ids)
                except Exception:
                    pass
            return self._forward_pytorch_cross(hidden_states, key_value_states, attention_mask, q_segment_ids, kv_segment_ids)
        else:
            if self._use_triton and hidden_states.is_cuda:
                try:
                    return self._forward_triton_self(hidden_states, attention_mask, segment_ids)
                except Exception:
                    pass
            return self._forward_pytorch_self(hidden_states, attention_mask, position_bias, segment_ids)

    def get_kv_cache_size(self, batch_size, seq_len):
        """Return KV cache size in bytes (float16)"""

        return batch_size * seq_len * self.num_kv_heads * self.head_dim * 2 * 2