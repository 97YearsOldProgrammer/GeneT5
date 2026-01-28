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
    embed_dim:         int   = 768
    num_heads:         int   = 12
    num_kv_heads:      int   = None
    head_dim:          int   = None
    block_size:        int   = 64
    window_size:       int   = 256
    num_global_tokens: int   = 64
    num_random_blocks: int   = 3
    dropout:           float = 0.0
    use_alibi:         bool  = True
    max_seq_len:       int   = 8192
    
    def __post_init__(self):

        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            self.head_dim = self.embed_dim // self.num_heads


def _check_triton_support():
    """Check if Triton is supported on current hardware"""

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
    Global_K_ptr, Global_V_ptr,
    block_indices_ptr,
    segment_ids_ptr,
    seq_len, num_heads, num_kv_heads, head_dim,
    num_global_tokens, blocks_per_query,
    start_block_idx,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_gkb, stride_gkh, stride_gks, stride_gkd,
    stride_gvb, stride_gvh, stride_gvs, stride_gvd,
    stride_segb, stride_segs,
    stride_bi_q, stride_bi_b,
    alibi_slopes_ptr,
    heads_per_group,
    use_alibi: tl.constexpr,
    use_segment_mask: tl.constexpr,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_G: tl.constexpr,
    NUM_RANDOM_BLOCKS: tl.constexpr,
):
    """Sparse attention optimized to load KV once per group"""

    pid_batch   = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    pid_q_block = tl.program_id(2) + start_block_idx
    
    q_block_start = pid_q_block * BLOCK_M
    offs_m        = q_block_start + tl.arange(0, BLOCK_M)
    offs_d        = tl.arange(0, BLOCK_D)
    mask_m        = offs_m < seq_len

    if use_segment_mask:
        q_seg_ptrs = segment_ids_ptr + pid_batch * stride_segb + offs_m * stride_segs
        q_seg_ids  = tl.load(q_seg_ptrs, mask=mask_m, other=-1)

    for local_head in range(heads_per_group):
        pid_head = pid_kv_head * heads_per_group + local_head

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
                       pid_kv_head * stride_gkh +
                       g_offs[:, None] * stride_gks +
                       offs_d[None, :] * stride_gkd)
            gk      = tl.load(gk_ptrs, mask=g_mask[:, None], other=0.0)
            scores  = tl.dot(q, tl.trans(gk)) * softmax_scale

            if use_alibi:
                q_pos      = offs_m[:, None].to(tl.float32)
                k_pos      = g_offs[None, :].to(tl.float32)
                alibi_bias = -alibi_slope * tl.abs(q_pos - k_pos)
                scores     = scores + alibi_bias

            if use_segment_mask:
                g_seg_ptrs = segment_ids_ptr + pid_batch * stride_segb + g_offs * stride_segs
                g_seg_ids  = tl.load(g_seg_ptrs, mask=g_mask, other=-1)
                same_seg   = (q_seg_ids[:, None] == g_seg_ids[None, :]) & (q_seg_ids[:, None] >= 0)
                scores     = tl.where(same_seg & mask_m[:, None] & g_mask[None, :], scores, float('-inf'))
            else:
                scores = tl.where(mask_m[:, None] & g_mask[None, :], scores, float('-inf'))

            m_ij  = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p     = tl.exp(scores - m_new[:, None])
            l_i   = alpha * l_i + tl.sum(p, axis=1)

            gv_ptrs = (Global_V_ptr +
                       pid_batch * stride_gvb +
                       pid_kv_head * stride_gvh +
                       g_offs[:, None] * stride_gvs +
                       offs_d[None, :] * stride_gvd)
            gv  = tl.load(gv_ptrs, mask=g_mask[:, None], other=0.0)
            acc = alpha[:, None] * acc + tl.dot(p.to(gv.dtype), gv)
            m_i = m_new

        for block_idx in range(blocks_per_query):
            bi_ptr      = block_indices_ptr + pid_q_block * stride_bi_q + block_idx * stride_bi_b
            kv_block_id = tl.load(bi_ptr)

            if kv_block_id >= 0:
                kv_block_start = kv_block_id * BLOCK_N
                offs_n         = kv_block_start + tl.arange(0, BLOCK_N)
                mask_n         = (offs_n < seq_len) & (offs_n >= num_global_tokens)

                k_ptrs = (K_ptr +
                          pid_batch * stride_kb +
                          pid_kv_head * stride_kh +
                          offs_n[:, None] * stride_ks +
                          offs_d[None, :] * stride_kd)
                k      = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
                scores = tl.dot(q, tl.trans(k)) * softmax_scale

                if use_alibi:
                    q_pos      = offs_m[:, None].to(tl.float32)
                    k_pos      = offs_n[None, :].to(tl.float32)
                    alibi_bias = -alibi_slope * tl.abs(q_pos - k_pos)
                    scores     = scores + alibi_bias

                if use_segment_mask:
                    k_seg_ptrs = segment_ids_ptr + pid_batch * stride_segb + offs_n * stride_segs
                    k_seg_ids  = tl.load(k_seg_ptrs, mask=mask_n, other=-1)
                    same_seg   = (q_seg_ids[:, None] == k_seg_ids[None, :]) & (q_seg_ids[:, None] >= 0)
                    scores     = tl.where(same_seg & mask_m[:, None] & mask_n[None, :], scores, float('-inf'))
                else:
                    scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, float('-inf'))

                m_ij  = tl.max(scores, axis=1)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p     = tl.exp(scores - m_new[:, None])
                l_i   = alpha * l_i + tl.sum(p, axis=1)

                v_ptrs = (V_ptr +
                          pid_batch * stride_vb +
                          pid_kv_head * stride_vh +
                          offs_n[:, None] * stride_vs +
                          offs_d[None, :] * stride_vd)
                v   = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
                acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)
                m_i = m_new

        l_i = tl.where(l_i == 0.0, 1.0, l_i)
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
    segment_ids_ptr,
    seq_len, num_heads, num_kv_heads, head_dim,
    num_global_tokens,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_segb, stride_segs,
    alibi_slopes_ptr,
    heads_per_group,
    use_alibi: tl.constexpr,
    use_segment_mask: tl.constexpr,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Global token attention optimized to load KV once per group"""

    pid_batch   = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    pid_g       = tl.program_id(2)
    
    g_start = pid_g * BLOCK_M
    offs_g  = g_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, BLOCK_D)
    mask_g  = offs_g < num_global_tokens

    if use_segment_mask:
        g_seg_ptrs = segment_ids_ptr + pid_batch * stride_segb + offs_g * stride_segs
        g_seg_ids  = tl.load(g_seg_ptrs, mask=mask_g, other=-1)

    for local_head in range(heads_per_group):
        pid_head = pid_kv_head * heads_per_group + local_head

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
                      pid_kv_head * stride_kh +
                      offs_n[:, None] * stride_ks +
                      offs_d[None, :] * stride_kd)
            k      = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k)) * softmax_scale

            if use_alibi:
                q_pos      = offs_g[:, None].to(tl.float32)
                k_pos      = offs_n[None, :].to(tl.float32)
                alibi_bias = -alibi_slope * tl.abs(q_pos - k_pos)
                scores     = scores + alibi_bias

            if use_segment_mask:
                k_seg_ptrs = segment_ids_ptr + pid_batch * stride_segb + offs_n * stride_segs
                k_seg_ids  = tl.load(k_seg_ptrs, mask=mask_n, other=-1)
                same_seg   = (g_seg_ids[:, None] == k_seg_ids[None, :]) & (g_seg_ids[:, None] >= 0)
                scores     = tl.where(same_seg & mask_g[:, None] & mask_n[None, :], scores, float('-inf'))
            else:
                scores = tl.where(mask_g[:, None] & mask_n[None, :], scores, float('-inf'))

            m_ij  = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p     = tl.exp(scores - m_new[:, None])
            l_i   = alpha * l_i + tl.sum(p, axis=1)

            v_ptrs = (V_ptr +
                      pid_batch * stride_vb +
                      pid_kv_head * stride_vh +
                      offs_n[:, None] * stride_vs +
                      offs_d[None, :] * stride_vd)
            v   = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        l_i = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i[:, None]

        out_ptrs = (Out_ptr +
                    pid_batch * stride_ob +
                    pid_head * stride_oh +
                    offs_g[:, None] * stride_os +
                    offs_d[None, :] * stride_od)
        tl.store(out_ptrs, acc, mask=mask_g[:, None])


class BlockIndexBuilder:
    """Vectorized block index computation on GPU"""

    def __init__(self, block_size, window_size, num_global_tokens, num_random_blocks, max_seq_len=8192):

        self.block_size        = block_size
        self.window_size       = window_size
        self.num_global_tokens = num_global_tokens
        self.num_random_blocks = num_random_blocks
        self.max_seq_len       = max_seq_len
        self._cache            = {}
    
    def _compute_indices(self, seq_len, device):
        """Compute sparse block indices via hash-based random selection"""

        num_blocks        = seq_len // self.block_size
        num_global_blocks = self.num_global_tokens // self.block_size
        window_blocks     = self.window_size // self.block_size
        max_window_blocks = 2 * window_blocks + 1
        max_blocks_query  = max_window_blocks + self.num_random_blocks
        
        all_blocks   = th.arange(num_blocks, device=device)
        query_blocks = all_blocks[:, None]
        kv_blocks    = all_blocks[None, :]
        block_dist   = th.abs(query_blocks - kv_blocks)
        
        window_mask    = (block_dist <= window_blocks) & (kv_blocks >= num_global_blocks)
        global_mask    = kv_blocks < num_global_blocks
        candidate_mask = ~(window_mask | global_mask)
        
        prime1      = 73856093
        prime2      = 19349663
        hash_vals   = ((query_blocks * prime1) ^ (kv_blocks * prime2)) % (2**31 - 1)
        rand_scores = hash_vals.float() / (2**31 - 1)
        rand_scores = rand_scores.masked_fill(~candidate_mask, -float('inf'))
        
        combined_scores  = th.zeros((num_blocks, num_blocks), device=device)
        combined_scores += th.where(window_mask, 1000.0 - block_dist.float(), 0.0)
        combined_scores += th.where(candidate_mask, rand_scores, 0.0)
        
        k_select    = min(max_blocks_query, num_blocks - num_global_blocks)
        _, selected = th.topk(combined_scores, k=k_select, dim=-1)
        valid_mask  = (window_mask | candidate_mask).gather(1, selected)
        selected    = th.where(valid_mask, selected, -1)
        
        if selected.shape[1] < max_blocks_query:
            padding  = th.full((num_blocks, max_blocks_query - selected.shape[1]), -1, dtype=th.long, device=device)
            selected = th.cat([selected, padding], dim=-1)
        
        sort_keys    = th.where(selected >= 0, selected, th.tensor(num_blocks * 2, device=device))
        sorted_order = th.argsort(sort_keys, dim=-1)
        selected     = selected.gather(1, sorted_order)
        
        return selected.int()
    
    def get_indices(self, seq_len, device):
        """Get block indices with caching"""

        cache_key = (seq_len, str(device))
        if cache_key not in self._cache:
            self._cache[cache_key] = self._compute_indices(seq_len, device)
        return self._cache[cache_key]
    
    def clear_cache(self):
        """Clear the index cache"""

        self._cache.clear()


class SparseAttention(nn.Module):
    """BigBird-style sparse attention with GQA support"""

    def __init__(self, config, is_causal=False):

        super().__init__()

        self.config            = config
        self.is_causal         = is_causal
        self.embed_dim         = config.embed_dim
        self.num_heads         = config.num_heads
        self.num_kv_heads      = config.num_kv_heads
        self.head_dim          = config.head_dim
        self.kv_dim            = config.head_dim * config.num_kv_heads
        self.block_size        = config.block_size
        self.window_size       = config.window_size
        self.num_global_tokens = config.num_global_tokens
        self.num_random_blocks = config.num_random_blocks
        self.softmax_scale     = 1.0 / math.sqrt(self.head_dim)
        self.heads_per_group   = config.num_heads // config.num_kv_heads

        assert config.num_heads % config.num_kv_heads == 0

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

        self.index_builder     = BlockIndexBuilder(
            block_size        = config.block_size,
            window_size       = config.window_size,
            num_global_tokens = config.num_global_tokens,
            num_random_blocks = config.num_random_blocks,
            max_seq_len       = config.max_seq_len,
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

        out      = th.empty_like(q)
        global_k = k[:, :, :self.num_global_tokens, :].contiguous()
        global_v = v[:, :, :self.num_global_tokens, :].contiguous()

        use_segment_mask = segment_ids is not None
        if use_segment_mask:
            segment_ids = segment_ids.contiguous()
            stride_segb = segment_ids.stride(0)
            stride_segs = segment_ids.stride(1) if segment_ids.dim() > 1 else 1
        else:
            stride_segb = 0
            stride_segs = 0

        alibi_ptr = self.alibi_slopes if self.alibi_slopes is not None else q
        BLOCK_G   = min(64, self.num_global_tokens) if self.num_global_tokens > 0 else 64

        if self.num_global_tokens > 0:
            num_g_blocks = (self.num_global_tokens + BLOCK_G - 1) // BLOCK_G
            grid_global  = (B, self.num_kv_heads, num_g_blocks)
            
            global_token_attention_kernel[grid_global](
                q, k, v, out,
                segment_ids if use_segment_mask else q,
                L_padded, self.num_heads, self.num_kv_heads, self.head_dim,
                self.num_global_tokens,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                stride_segb, stride_segs,
                alibi_ptr,
                self.heads_per_group,
                self.alibi_slopes is not None,
                use_segment_mask,
                self.softmax_scale,
                BLOCK_M = BLOCK_G,
                BLOCK_N = 64,
                BLOCK_D = self.head_dim,
            )

        block_indices        = self.index_builder.get_indices(L_padded, hidden_states.device)
        num_blocks           = L_padded // self.block_size
        blocks_per_query     = block_indices.shape[1]
        num_global_blocks    = self.num_global_tokens // self.block_size
        num_nonglobal_blocks = num_blocks - num_global_blocks

        if num_nonglobal_blocks > 0:
            grid_sparse = (B, self.num_kv_heads, num_nonglobal_blocks)

            sparse_attention_fwd_kernel[grid_sparse](
                q, k, v, out,
                global_k, global_v,
                block_indices,
                segment_ids if use_segment_mask else q,
                L_padded, self.num_heads, self.num_kv_heads, self.head_dim,
                self.num_global_tokens, blocks_per_query,
                num_global_blocks,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                global_k.stride(0), global_k.stride(1), global_k.stride(2), global_k.stride(3),
                global_v.stride(0), global_v.stride(1), global_v.stride(2), global_v.stride(3),
                stride_segb, stride_segs,
                block_indices.stride(0), block_indices.stride(1),
                alibi_ptr,
                self.heads_per_group,
                self.alibi_slopes is not None,
                use_segment_mask,
                self.softmax_scale,
                BLOCK_M           = self.block_size,
                BLOCK_N           = self.block_size,
                BLOCK_D           = self.head_dim,
                BLOCK_G           = BLOCK_G,
                NUM_RANDOM_BLOCKS = self.num_random_blocks,
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