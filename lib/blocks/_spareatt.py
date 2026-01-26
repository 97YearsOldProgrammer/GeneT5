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
class SparseAttentionConfig:
    embed_dim:         int   = 768
    num_heads:         int   = 12
    block_size:        int   = 64
    window_size:       int   = 256
    num_global_tokens: int   = 64
    num_random_blocks: int   = 3
    dropout:           float = 0.0
    use_alibi:         bool  = True


######################
##### TRITON OPS #####
######################


@triton.jit
def sparse_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    Global_K_ptr, Global_V_ptr,
    block_indices_ptr,
    seq_len, num_heads, head_dim,
    num_global_tokens, blocks_per_query,
    start_block_idx,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_gkb, stride_gkh, stride_gks, stride_gkd,
    stride_gvb, stride_gvh, stride_gvs, stride_gvd,
    alibi_slopes_ptr,
    use_alibi: tl.constexpr,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """Sparse attention with bidirectional global token visibility"""

    pid_batch   = tl.program_id(0)
    pid_head    = tl.program_id(1)
    pid_q_block = tl.program_id(2) + start_block_idx

    q_block_start = pid_q_block * BLOCK_M
    offs_m        = q_block_start + tl.arange(0, BLOCK_M)
    offs_d        = tl.arange(0, BLOCK_D)

    mask_m = offs_m < seq_len

    q_ptrs = (Q_ptr +
              pid_batch * stride_qb +
              pid_head * stride_qh +
              offs_m[:, None] * stride_qs +
              offs_d[None, :] * stride_qd)
    q      = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    if use_alibi:
        alibi_slope = tl.load(alibi_slopes_ptr + pid_head)

    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for g_start in range(0, num_global_tokens, BLOCK_G):
        g_offs = g_start + tl.arange(0, BLOCK_G)
        g_mask = g_offs < num_global_tokens

        gk_ptrs = (Global_K_ptr +
                   pid_batch * stride_gkb +
                   pid_head * stride_gkh +
                   g_offs[:, None] * stride_gks +
                   offs_d[None, :] * stride_gkd)
        gk      = tl.load(gk_ptrs, mask=g_mask[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(gk)) * softmax_scale

        if use_alibi:
            q_pos      = offs_m[:, None].to(tl.float32)
            k_pos      = g_offs[None, :].to(tl.float32)
            dist       = tl.abs(q_pos - k_pos)
            alibi_bias = -alibi_slope * dist
            scores     = scores + alibi_bias

        scores = tl.where(mask_m[:, None] & g_mask[None, :], scores, float('-inf'))

        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        beta  = tl.exp(m_ij - m_new)

        l_i = alpha * l_i + beta * tl.sum(tl.exp(scores - m_ij[:, None]), axis=1)

        gv_ptrs = (Global_V_ptr +
                   pid_batch * stride_gvb +
                   pid_head * stride_gvh +
                   g_offs[:, None] * stride_gvs +
                   offs_d[None, :] * stride_gvd)
        gv      = tl.load(gv_ptrs, mask=g_mask[:, None], other=0.0)

        p   = tl.exp(scores - m_ij[:, None])
        acc = alpha[:, None] * acc + tl.dot(p.to(gv.dtype), gv)

        m_i = m_new

    for block_idx in range(blocks_per_query):
        kv_block_id = tl.load(block_indices_ptr + pid_q_block * blocks_per_query + block_idx)

        if kv_block_id >= 0:
            kv_block_start = kv_block_id * BLOCK_N
            offs_n         = kv_block_start + tl.arange(0, BLOCK_N)
            mask_n         = (offs_n < seq_len) & (offs_n >= num_global_tokens)

            k_ptrs = (K_ptr +
                      pid_batch * stride_kb +
                      pid_head * stride_kh +
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

            scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, float('-inf'))

            m_ij  = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)

            alpha = tl.exp(m_i - m_new)
            beta  = tl.exp(m_ij - m_new)

            l_i = alpha * l_i + beta * tl.sum(tl.exp(scores - m_ij[:, None]), axis=1)

            v_ptrs = (V_ptr +
                      pid_batch * stride_vb +
                      pid_head * stride_vh +
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
def global_token_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    seq_len, num_heads, head_dim,
    num_global_tokens,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    alibi_slopes_ptr,
    use_alibi: tl.constexpr,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Full attention for global tokens - they see ALL tokens"""

    pid_batch  = tl.program_id(0)
    pid_head   = tl.program_id(1)
    pid_g      = tl.program_id(2)

    g_start = pid_g * BLOCK_M
    offs_g  = g_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, BLOCK_D)

    mask_g = offs_g < num_global_tokens

    q_ptrs = (Q_ptr +
              pid_batch * stride_qb +
              pid_head * stride_qh +
              offs_g[:, None] * stride_qs +
              offs_d[None, :] * stride_qd)
    q      = tl.load(q_ptrs, mask=mask_g[:, None], other=0.0)

    if use_alibi:
        alibi_slope = tl.load(alibi_slopes_ptr + pid_head)

    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, seq_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        k_ptrs = (K_ptr +
                  pid_batch * stride_kb +
                  pid_head * stride_kh +
                  offs_n[:, None] * stride_ks +
                  offs_d[None, :] * stride_kd)
        k      = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k)) * softmax_scale

        if use_alibi:
            q_pos      = offs_g[:, None].to(tl.float32)
            k_pos      = offs_n[None, :].to(tl.float32)
            dist       = tl.abs(q_pos - k_pos)
            alibi_bias = -alibi_slope * dist
            scores     = scores + alibi_bias

        scores = tl.where(mask_g[:, None] & mask_n[None, :], scores, float('-inf'))

        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        beta  = tl.exp(m_ij - m_new)

        l_i = alpha * l_i + beta * tl.sum(tl.exp(scores - m_ij[:, None]), axis=1)

        v_ptrs = (V_ptr +
                  pid_batch * stride_vb +
                  pid_head * stride_vh +
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
                offs_g[:, None] * stride_os +
                offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc, mask=mask_g[:, None])


##############################
#####  SPARSE ATTENTION  #####
##############################


class SparseAttention(nn.Module):
    """BigBird-style sparse attention with bidirectional global tokens"""

    def __init__(self, config, is_causal=False):
        super().__init__()

        self.config            = config
        self.is_causal         = is_causal
        self.embed_dim         = config.embed_dim
        self.num_heads         = config.num_heads
        self.head_dim          = config.embed_dim // config.num_heads
        self.block_size        = config.block_size
        self.window_size       = config.window_size
        self.num_global_tokens = config.num_global_tokens
        self.num_random_blocks = config.num_random_blocks
        self.softmax_scale     = 1.0 / math.sqrt(self.head_dim)

        self.q = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.k = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.v = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.o = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        if config.use_alibi:
            slopes = self._compute_alibi_slopes(config.num_heads)
            self.register_buffer('alibi_slopes', slopes)
        else:
            self.alibi_slopes = None

        self._block_indices_cache = {}

    @staticmethod
    def _compute_alibi_slopes(num_heads):
        """Compute ALiBi slopes for positional bias"""

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

    def _compute_block_indices(self, seq_len, device):
        """Compute sparse block indices excluding global tokens"""

        cache_key = (seq_len, device)
        if cache_key in self._block_indices_cache:
            return self._block_indices_cache[cache_key]

        num_blocks        = seq_len // self.block_size
        num_global_blocks = self.num_global_tokens // self.block_size
        window_blocks     = self.window_size // self.block_size

        max_blocks = (2 * window_blocks + 1) + self.num_random_blocks

        block_indices = torch.full(
            (num_blocks, max_blocks), -1, dtype=torch.long, device=device
        )

        for q_block in range(num_blocks):
            idx      = 0
            attended = set()

            for g in range(num_global_blocks):
                attended.add(g)

            window_start = max(num_global_blocks, q_block - window_blocks)
            window_end   = min(num_blocks, q_block + window_blocks + 1)
            for w in range(window_start, window_end):
                if w not in attended:
                    block_indices[q_block, idx] = w
                    attended.add(w)
                    idx += 1

            available = [b for b in range(num_global_blocks, num_blocks) if b not in attended]
            if available:
                torch.manual_seed(q_block)
                perm = torch.randperm(len(available))[:self.num_random_blocks]
                for p in perm:
                    block_indices[q_block, idx] = available[p]
                    idx += 1

        self._block_indices_cache[cache_key] = block_indices
        return block_indices

    def forward(self, hidden_states, attention_mask=None):
        B, L, D = hidden_states.shape

        pad_len = (self.block_size - L % self.block_size) % self.block_size
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            L_padded      = L + pad_len
        else:
            L_padded = L

        q = self.q(hidden_states).view(B, L_padded, self.num_heads, self.head_dim)
        k = self.k(hidden_states).view(B, L_padded, self.num_heads, self.head_dim)
        v = self.v(hidden_states).view(B, L_padded, self.num_heads, self.head_dim)

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

        global_k = k[:, :, :self.num_global_tokens, :].contiguous()
        global_v = v[:, :, :self.num_global_tokens, :].contiguous()

        BLOCK_G       = min(64, self.num_global_tokens)
        num_g_blocks  = (self.num_global_tokens + BLOCK_G - 1) // BLOCK_G
        grid_global   = (B, self.num_heads, num_g_blocks)

        alibi_ptr = self.alibi_slopes if self.alibi_slopes is not None else q

        global_token_attention_kernel[grid_global](
            q, k, v, out,
            L_padded, self.num_heads, self.head_dim,
            self.num_global_tokens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            alibi_ptr,
            self.alibi_slopes is not None,
            self.softmax_scale,
            BLOCK_M=BLOCK_G,
            BLOCK_N=64,
            BLOCK_D=self.head_dim,
        )

        block_indices    = self._compute_block_indices(L_padded, hidden_states.device)
        num_blocks       = L_padded // self.block_size
        blocks_per_query = block_indices.shape[1]

        num_global_blocks    = self.num_global_tokens // self.block_size
        num_nonglobal_blocks = num_blocks - num_global_blocks
        grid_sparse          = (B, self.num_heads, num_nonglobal_blocks)

        sparse_attention_fwd_kernel[grid_sparse](
            q, k, v, out,
            global_k, global_v,
            block_indices,
            L_padded, self.num_heads, self.head_dim,
            self.num_global_tokens, blocks_per_query,
            num_global_blocks,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            global_k.stride(0), global_k.stride(1), global_k.stride(2), global_k.stride(3),
            global_v.stride(0), global_v.stride(1), global_v.stride(2), global_v.stride(3),
            alibi_ptr,
            self.alibi_slopes is not None,
            self.softmax_scale,
            BLOCK_M=self.block_size,
            BLOCK_N=self.block_size,
            BLOCK_D=self.head_dim,
            BLOCK_G=BLOCK_G,
        )

        out = out.permute(0, 2, 1, 3).reshape(B, L_padded, D)
        out = self.o(out)

        if pad_len > 0:
            out = out[:, :L, :]

        out = self.dropout(out)

        return out, None