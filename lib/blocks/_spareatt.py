from __future__ import annotations

import math
from dataclasses import dataclass

import torch                as th
import torch.nn             as nn
import torch.nn.functional  as F

import triton
import triton.language      as tl


##################
##### CONFIG #####
##################


@dataclass
class SparseAttentionConfig:
    embed_dim:    int   = 768
    num_heads:    int   = 12
    num_kv_heads: int   = None
    head_dim:     int   = None
    block_size:   int   = 64
    window_size:  int   = 256
    dropout:      float = 0.0
    use_alibi:    bool  = True
    max_seq_len:  int   = 8192

    def __post_init__(self):

        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            self.head_dim = self.embed_dim // self.num_heads


def _check_triton_support():
    """Check if Triton sparse attention is supported

    NOTE: Triton sparse attention requires sufficient shared memory.
    If you hit 'out of resource: shared memory' errors, reduce block_size.
    Set GENET5_ENABLE_SPARSE_TRITON=1 to force enable for testing.
    """

    import os
    if os.environ.get("GENET5_DISABLE_SPARSE_TRITON", "0") == "1":
        return False
    if not th.cuda.is_available():
        return False
    cap = th.cuda.get_device_capability()
    return cap[0] >= 7


####################
#####  Kernel  #####
####################


@triton.jit
def sparse_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    block_indices_ptr,
    segment_ids_ptr,
    seq_len, num_heads, num_kv_heads, head_dim,
    blocks_per_query,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_segb, stride_segs,
    stride_bi_q, stride_bi_b,
    alibi_slopes_ptr,
    heads_per_group,
    use_alibi: tl.constexpr,
    use_segment_mask: tl.constexpr,
    is_causal: tl.constexpr,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
):
    """Sparse window attention optimized to load KV once per group

    Fixed: KV loaded once per block, reused across all heads in group
    Added: Causal masking support
    """

    pid_batch   = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    pid_q_block = tl.program_id(2)

    q_block_start = pid_q_block * BLOCK_M
    offs_m        = q_block_start + tl.arange(0, BLOCK_M)
    offs_d        = tl.arange(0, BLOCK_D)
    mask_m        = offs_m < seq_len

    # Load segment IDs (shared across heads in group)
    if use_segment_mask:
        q_seg_ptrs = segment_ids_ptr + pid_batch * stride_segb + offs_m * stride_segs
        q_seg_ids  = tl.load(q_seg_ptrs, mask=mask_m, other=-1)

    # Initialize per-head accumulators
    m_i_0 = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    m_i_1 = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    m_i_2 = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    m_i_3 = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)

    l_i_0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    l_i_1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    l_i_2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    l_i_3 = tl.zeros((BLOCK_M,), dtype=tl.float32)

    acc_0 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_1 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_2 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_3 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Load Q heads upfront
    q_base = Q_ptr + pid_batch * stride_qb + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd

    q_0 = tl.load(q_base + (pid_kv_head * HEADS_PER_GROUP + 0) * stride_qh, mask=mask_m[:, None], other=0.0)
    q_1 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    q_2 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    q_3 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    if HEADS_PER_GROUP > 1:
        q_1 = tl.load(q_base + (pid_kv_head * HEADS_PER_GROUP + 1) * stride_qh, mask=mask_m[:, None], other=0.0)
    if HEADS_PER_GROUP > 2:
        q_2 = tl.load(q_base + (pid_kv_head * HEADS_PER_GROUP + 2) * stride_qh, mask=mask_m[:, None], other=0.0)
    if HEADS_PER_GROUP > 3:
        q_3 = tl.load(q_base + (pid_kv_head * HEADS_PER_GROUP + 3) * stride_qh, mask=mask_m[:, None], other=0.0)

    # Load ALiBi slopes
    alibi_slope_0 = 0.0
    alibi_slope_1 = 0.0
    alibi_slope_2 = 0.0
    alibi_slope_3 = 0.0

    if use_alibi:
        alibi_slope_0 = tl.load(alibi_slopes_ptr + pid_kv_head * HEADS_PER_GROUP + 0)
        if HEADS_PER_GROUP > 1:
            alibi_slope_1 = tl.load(alibi_slopes_ptr + pid_kv_head * HEADS_PER_GROUP + 1)
        if HEADS_PER_GROUP > 2:
            alibi_slope_2 = tl.load(alibi_slopes_ptr + pid_kv_head * HEADS_PER_GROUP + 2)
        if HEADS_PER_GROUP > 3:
            alibi_slope_3 = tl.load(alibi_slopes_ptr + pid_kv_head * HEADS_PER_GROUP + 3)

    # Process window blocks - load K,V once per block
    for block_idx in range(blocks_per_query):
        bi_ptr      = block_indices_ptr + pid_q_block * stride_bi_q + block_idx * stride_bi_b
        kv_block_id = tl.load(bi_ptr)

        if kv_block_id >= 0:
            kv_block_start = kv_block_id * BLOCK_N
            offs_n         = kv_block_start + tl.arange(0, BLOCK_N)
            mask_n         = offs_n < seq_len

            # Load K and V ONCE for this block
            k_ptrs = (K_ptr +
                      pid_batch * stride_kb +
                      pid_kv_head * stride_kh +
                      offs_n[:, None] * stride_ks +
                      offs_d[None, :] * stride_kd)
            k      = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

            v_ptrs = (V_ptr +
                      pid_batch * stride_vb +
                      pid_kv_head * stride_vh +
                      offs_n[:, None] * stride_vs +
                      offs_d[None, :] * stride_vd)
            v      = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            k_t = tl.trans(k)

            # Base mask
            base_mask = mask_m[:, None] & mask_n[None, :]

            if use_segment_mask:
                k_seg_ptrs = segment_ids_ptr + pid_batch * stride_segb + offs_n * stride_segs
                k_seg_ids  = tl.load(k_seg_ptrs, mask=mask_n, other=-1)
                same_seg   = (q_seg_ids[:, None] == k_seg_ids[None, :]) & (q_seg_ids[:, None] >= 0)
                base_mask  = base_mask & same_seg

            # Causal mask
            if is_causal:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                base_mask   = base_mask & causal_mask

            # ALiBi bias
            if use_alibi:
                q_pos    = offs_m[:, None].to(tl.float32)
                k_pos    = offs_n[None, :].to(tl.float32)
                pos_diff = tl.abs(q_pos - k_pos)

            # Process all heads with shared K,V
            scores_0    = tl.dot(q_0, k_t) * softmax_scale
            if use_alibi:
                scores_0 = scores_0 - alibi_slope_0 * pos_diff
            scores_0    = tl.where(base_mask, scores_0, float('-inf'))
            m_ij_0      = tl.max(scores_0, axis=1)
            m_new_0     = tl.maximum(m_i_0, m_ij_0)
            both_inf_0  = (m_i_0 == float('-inf')) & (m_new_0 == float('-inf'))
            alpha_0     = tl.where(both_inf_0, 1.0, tl.exp(m_i_0 - m_new_0))
            p_0         = tl.where(both_inf_0[:, None], 0.0, tl.exp(scores_0 - m_new_0[:, None]))
            l_i_0       = alpha_0 * l_i_0 + tl.sum(p_0, axis=1)
            acc_0       = alpha_0[:, None] * acc_0 + tl.dot(p_0.to(v.dtype), v)
            m_i_0       = m_new_0

            if HEADS_PER_GROUP > 1:
                scores_1    = tl.dot(q_1, k_t) * softmax_scale
                if use_alibi:
                    scores_1 = scores_1 - alibi_slope_1 * pos_diff
                scores_1    = tl.where(base_mask, scores_1, float('-inf'))
                m_ij_1      = tl.max(scores_1, axis=1)
                m_new_1     = tl.maximum(m_i_1, m_ij_1)
                both_inf_1  = (m_i_1 == float('-inf')) & (m_new_1 == float('-inf'))
                alpha_1     = tl.where(both_inf_1, 1.0, tl.exp(m_i_1 - m_new_1))
                p_1         = tl.where(both_inf_1[:, None], 0.0, tl.exp(scores_1 - m_new_1[:, None]))
                l_i_1       = alpha_1 * l_i_1 + tl.sum(p_1, axis=1)
                acc_1       = alpha_1[:, None] * acc_1 + tl.dot(p_1.to(v.dtype), v)
                m_i_1       = m_new_1

            if HEADS_PER_GROUP > 2:
                scores_2    = tl.dot(q_2, k_t) * softmax_scale
                if use_alibi:
                    scores_2 = scores_2 - alibi_slope_2 * pos_diff
                scores_2    = tl.where(base_mask, scores_2, float('-inf'))
                m_ij_2      = tl.max(scores_2, axis=1)
                m_new_2     = tl.maximum(m_i_2, m_ij_2)
                both_inf_2  = (m_i_2 == float('-inf')) & (m_new_2 == float('-inf'))
                alpha_2     = tl.where(both_inf_2, 1.0, tl.exp(m_i_2 - m_new_2))
                p_2         = tl.where(both_inf_2[:, None], 0.0, tl.exp(scores_2 - m_new_2[:, None]))
                l_i_2       = alpha_2 * l_i_2 + tl.sum(p_2, axis=1)
                acc_2       = alpha_2[:, None] * acc_2 + tl.dot(p_2.to(v.dtype), v)
                m_i_2       = m_new_2

            if HEADS_PER_GROUP > 3:
                scores_3    = tl.dot(q_3, k_t) * softmax_scale
                if use_alibi:
                    scores_3 = scores_3 - alibi_slope_3 * pos_diff
                scores_3    = tl.where(base_mask, scores_3, float('-inf'))
                m_ij_3      = tl.max(scores_3, axis=1)
                m_new_3     = tl.maximum(m_i_3, m_ij_3)
                both_inf_3  = (m_i_3 == float('-inf')) & (m_new_3 == float('-inf'))
                alpha_3     = tl.where(both_inf_3, 1.0, tl.exp(m_i_3 - m_new_3))
                p_3         = tl.where(both_inf_3[:, None], 0.0, tl.exp(scores_3 - m_new_3[:, None]))
                l_i_3       = alpha_3 * l_i_3 + tl.sum(p_3, axis=1)
                acc_3       = alpha_3[:, None] * acc_3 + tl.dot(p_3.to(v.dtype), v)
                m_i_3       = m_new_3

    # Finalize and store
    out_base = Out_ptr + pid_batch * stride_ob + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od

    # Guard against NaN: when l_i==0 (no valid attention targets), output zeros
    zero_mask_0 = (l_i_0 == 0.0)[:, None]
    l_i_0 = tl.where(l_i_0 == 0.0, 1.0, l_i_0)
    acc_0 = tl.where(zero_mask_0, 0.0, acc_0 / l_i_0[:, None])
    tl.store(out_base + (pid_kv_head * HEADS_PER_GROUP + 0) * stride_oh, acc_0, mask=mask_m[:, None])

    if HEADS_PER_GROUP > 1:
        zero_mask_1 = (l_i_1 == 0.0)[:, None]
        l_i_1 = tl.where(l_i_1 == 0.0, 1.0, l_i_1)
        acc_1 = tl.where(zero_mask_1, 0.0, acc_1 / l_i_1[:, None])
        tl.store(out_base + (pid_kv_head * HEADS_PER_GROUP + 1) * stride_oh, acc_1, mask=mask_m[:, None])

    if HEADS_PER_GROUP > 2:
        zero_mask_2 = (l_i_2 == 0.0)[:, None]
        l_i_2 = tl.where(l_i_2 == 0.0, 1.0, l_i_2)
        acc_2 = tl.where(zero_mask_2, 0.0, acc_2 / l_i_2[:, None])
        tl.store(out_base + (pid_kv_head * HEADS_PER_GROUP + 2) * stride_oh, acc_2, mask=mask_m[:, None])

    if HEADS_PER_GROUP > 3:
        zero_mask_3 = (l_i_3 == 0.0)[:, None]
        l_i_3 = tl.where(l_i_3 == 0.0, 1.0, l_i_3)
        acc_3 = tl.where(zero_mask_3, 0.0, acc_3 / l_i_3[:, None])
        tl.store(out_base + (pid_kv_head * HEADS_PER_GROUP + 3) * stride_oh, acc_3, mask=mask_m[:, None])


class BlockIndexBuilder:
    """Vectorized block index computation on GPU for window attention"""

    MAX_CACHE_SIZE = 32  # Limit cache to avoid memory leak

    def __init__(self, block_size, window_size, max_seq_len=8192):

        from collections import OrderedDict
        self.block_size  = block_size
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self._cache      = OrderedDict()  # LRU cache with O(1) operations

    def _compute_indices(self, seq_len, device):
        """Compute window block indices"""

        num_blocks        = seq_len // self.block_size
        window_blocks     = self.window_size // self.block_size
        max_window_blocks = 2 * window_blocks + 1

        all_blocks   = th.arange(num_blocks, device=device)
        query_blocks = all_blocks[:, None]
        kv_blocks    = all_blocks[None, :]
        block_dist   = th.abs(query_blocks - kv_blocks)

        # Window mask: blocks within window distance
        window_mask = block_dist <= window_blocks

        # Create scores for window blocks (closer blocks get higher scores)
        scores = th.where(window_mask, 1000.0 - block_dist.float(), -float('inf'))

        # Select top-k blocks (all window blocks)
        k_select    = min(max_window_blocks, num_blocks)
        _, selected = th.topk(scores, k=k_select, dim=-1)
        valid_mask  = window_mask.gather(1, selected)
        selected    = th.where(valid_mask, selected, -1)

        # Pad if necessary
        if selected.shape[1] < max_window_blocks:
            padding  = th.full((num_blocks, max_window_blocks - selected.shape[1]), -1, dtype=th.long, device=device)
            selected = th.cat([selected, padding], dim=-1)

        # Sort for coalesced memory access
        sort_keys    = th.where(selected >= 0, selected, th.tensor(num_blocks * 2, device=device))
        sorted_order = th.argsort(sort_keys, dim=-1)
        selected     = selected.gather(1, sorted_order)

        return selected.int()

    def get_indices(self, seq_len, device):
        """Get block indices with LRU caching (O(1) operations)"""

        cache_key = (seq_len, str(device))
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)  # O(1) LRU update
            return self._cache[cache_key]

        # Evict oldest entry if cache is full
        while len(self._cache) >= self.MAX_CACHE_SIZE:
            self._cache.popitem(last=False)  # O(1) remove oldest

        self._cache[cache_key] = self._compute_indices(seq_len, device)
        return self._cache[cache_key]

    def clear_cache(self):
        """Clear the index cache"""
        self._cache.clear()


class SparseAttention(nn.Module):
    """Window-based sparse attention with GQA support

    Fixed issues:
    - KV loading efficiency (load once per block, reuse across heads)
    - Causal masking now properly implemented
    """

    def __init__(self, config, is_causal=False):

        super().__init__()

        self.config          = config
        self.is_causal       = is_causal
        self.embed_dim       = config.embed_dim
        self.num_heads       = config.num_heads
        self.num_kv_heads    = config.num_kv_heads
        self.head_dim        = config.head_dim
        self.kv_dim          = config.head_dim * config.num_kv_heads
        self.block_size      = config.block_size
        self.window_size     = config.window_size
        self.softmax_scale   = 1.0 / math.sqrt(self.head_dim)
        self.heads_per_group = config.num_heads // config.num_kv_heads

        assert config.num_heads % config.num_kv_heads == 0
        assert self.heads_per_group <= 4, "Current kernel supports up to 4 heads per group"

        self.q       = nn.Linear(config.embed_dim, config.num_heads * self.head_dim, bias=False)
        self.k       = nn.Linear(config.embed_dim, self.kv_dim, bias=False)
        self.v       = nn.Linear(config.embed_dim, self.kv_dim, bias=False)
        self.o       = nn.Linear(config.num_heads * self.head_dim, config.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if config.use_alibi:
            slopes = self._compute_alibi_slopes(config.num_heads)
            self.register_buffer('alibi_slopes', slopes)
        else:
            self.alibi_slopes = None

        self.index_builder = BlockIndexBuilder(
            block_size  = config.block_size,
            window_size = config.window_size,
            max_seq_len = config.max_seq_len,
        )
        self._triton_supported = _check_triton_support()

    @staticmethod
    def _compute_alibi_slopes(num_heads):
        """Compute ALiBi slopes for positional bias"""

        def get_slopes(n):
            if n == 1:
                return th.tensor([1.0])
            base = 2 ** (-2 ** -(math.log2(n) - 3))
            return th.tensor([base ** i for i in range(1, n + 1)])

        if math.log2(num_heads).is_integer():
            return get_slopes(num_heads)
        else:
            closest_power = 2 ** math.floor(math.log2(num_heads))
            return th.cat([
                get_slopes(closest_power),
                get_slopes(2 * closest_power)[0::2][:num_heads - closest_power]
            ])

    def forward(self, hidden_states, attention_mask=None, segment_ids=None):
        """Forward with optional segment masking for packed sequences"""

        B, L, D    = hidden_states.shape
        use_triton = self._triton_supported and hidden_states.is_cuda

        if not use_triton:
            raise RuntimeError("SparseAttention requires CUDA with Triton support (compute capability >= 7.0)")

        pad_len = (self.block_size - L % self.block_size) % self.block_size
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            L_padded      = L + pad_len
            if segment_ids is not None:
                segment_ids = F.pad(segment_ids, (0, pad_len), value=-1)
        else:
            L_padded = L

        q = self.q(hidden_states).view(B, L_padded, self.num_heads, self.head_dim)
        k = self.k(hidden_states).view(B, L_padded, self.num_kv_heads, self.head_dim)
        v = self.v(hidden_states).view(B, L_padded, self.num_kv_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        out = th.empty_like(q)

        use_segment_mask = segment_ids is not None
        if use_segment_mask:
            segment_ids = segment_ids.contiguous()
            stride_segb = segment_ids.stride(0)
            stride_segs = segment_ids.stride(1) if segment_ids.dim() > 1 else 1
        else:
            stride_segb = 0
            stride_segs = 0

        alibi_ptr = self.alibi_slopes if self.alibi_slopes is not None else q

        # Process window blocks
        block_indices    = self.index_builder.get_indices(L_padded, hidden_states.device)
        num_blocks       = L_padded // self.block_size
        blocks_per_query = block_indices.shape[1]

        if num_blocks > 0:
            grid_sparse = (B, self.num_kv_heads, num_blocks)

            sparse_attention_fwd_kernel[grid_sparse](
                q, k, v, out,
                block_indices,
                segment_ids if use_segment_mask else q,
                L_padded, self.num_heads, self.num_kv_heads, self.head_dim,
                blocks_per_query,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                stride_segb, stride_segs,
                block_indices.stride(0), block_indices.stride(1),
                alibi_ptr,
                self.heads_per_group,
                self.alibi_slopes is not None,
                use_segment_mask,
                self.is_causal,
                self.softmax_scale,
                BLOCK_M         = self.block_size,
                BLOCK_N         = self.block_size,
                BLOCK_D         = self.head_dim,
                HEADS_PER_GROUP = self.heads_per_group,
            )

        out = out.permute(0, 2, 1, 3).reshape(B, L_padded, self.num_heads * self.head_dim)
        out = self.o(out)

        if pad_len > 0:
            out = out[:, :L, :]

        out = self.dropout(out)

        return out, None

    def get_kv_cache_size(self, batch_size, seq_len):
        """Return KV cache size in bytes for float16"""

        return batch_size * seq_len * self.num_kv_heads * self.head_dim * 2 * 2