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
class GQAttentionConfig:
    embed_dim:     int   = 768
    num_heads:     int   = 12
    num_kv_heads:  int   = 4
    head_dim:      int   = None
    dropout:       float = 0.0
    use_alibi:     bool  = False
    max_seq_len:   int   = 8192
    is_cross_attn: bool  = False

    def __post_init__(self):

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
    HEADS_PER_GROUP: tl.constexpr,
):
    """GQA cross-attention optimized to load KV once per group
    
    Fixed: KV loaded once per block, reused across all heads in group
    """

    pid_batch    = tl.program_id(0)
    pid_kv_head  = tl.program_id(1)
    pid_m        = tl.program_id(2)

    q_start = pid_m * BLOCK_M
    offs_m  = q_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, BLOCK_D)
    offs_n  = tl.arange(0, BLOCK_N)
    mask_m  = offs_m < q_len

    # Load segment IDs for queries (shared across heads in group)
    if use_segment_mask:
        q_seg_ptrs = q_segment_ids_ptr + pid_batch * stride_qsegb + offs_m * stride_qsegs
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

    # Load all Q heads for this group upfront
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

    # Iterate over KV blocks - load K,V ONCE per block
    for n_start in range(0, kv_len, BLOCK_N):
        offs_n_curr = n_start + offs_n
        mask_n      = offs_n_curr < kv_len

        # Load K and V ONCE for all heads in group
        k_ptrs = (K_ptr +
                  pid_batch * stride_kb +
                  pid_kv_head * stride_kh +
                  offs_n_curr[:, None] * stride_ks +
                  offs_d[None, :] * stride_kd)
        k      = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        
        v_ptrs = (V_ptr +
                  pid_batch * stride_vb +
                  pid_kv_head * stride_vh +
                  offs_n_curr[:, None] * stride_vs +
                  offs_d[None, :] * stride_vd)
        v      = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        k_t = tl.trans(k)

        # Compute base mask (shared across heads)
        if use_segment_mask:
            kv_seg_ptrs = kv_segment_ids_ptr + pid_batch * stride_kvsegb + offs_n_curr * stride_kvsegs
            kv_seg_ids  = tl.load(kv_seg_ptrs, mask=mask_n, other=-1)
            same_seg    = (q_seg_ids[:, None] == kv_seg_ids[None, :]) & (q_seg_ids[:, None] >= 0)
            base_mask   = same_seg & mask_m[:, None] & mask_n[None, :]
        else:
            base_mask = mask_m[:, None] & mask_n[None, :]

        # Process head 0
        scores_0    = tl.dot(q_0, k_t) * softmax_scale
        scores_0    = tl.where(base_mask, scores_0, float('-inf'))
        m_ij_0      = tl.max(scores_0, axis=1)
        m_new_0     = tl.maximum(m_i_0, m_ij_0)
        both_inf_0  = (m_i_0 == float('-inf')) & (m_new_0 == float('-inf'))
        alpha_0     = tl.where(both_inf_0, 1.0, tl.exp(m_i_0 - m_new_0))
        p_0         = tl.where(both_inf_0[:, None], 0.0, tl.exp(scores_0 - m_new_0[:, None]))
        l_i_0       = alpha_0 * l_i_0 + tl.sum(p_0, axis=1)
        acc_0       = alpha_0[:, None] * acc_0 + tl.dot(p_0.to(v.dtype), v)
        m_i_0       = m_new_0

        # Process head 1
        if HEADS_PER_GROUP > 1:
            scores_1    = tl.dot(q_1, k_t) * softmax_scale
            scores_1    = tl.where(base_mask, scores_1, float('-inf'))
            m_ij_1      = tl.max(scores_1, axis=1)
            m_new_1     = tl.maximum(m_i_1, m_ij_1)
            both_inf_1  = (m_i_1 == float('-inf')) & (m_new_1 == float('-inf'))
            alpha_1     = tl.where(both_inf_1, 1.0, tl.exp(m_i_1 - m_new_1))
            p_1         = tl.where(both_inf_1[:, None], 0.0, tl.exp(scores_1 - m_new_1[:, None]))
            l_i_1       = alpha_1 * l_i_1 + tl.sum(p_1, axis=1)
            acc_1       = alpha_1[:, None] * acc_1 + tl.dot(p_1.to(v.dtype), v)
            m_i_1       = m_new_1

        # Process head 2
        if HEADS_PER_GROUP > 2:
            scores_2    = tl.dot(q_2, k_t) * softmax_scale
            scores_2    = tl.where(base_mask, scores_2, float('-inf'))
            m_ij_2      = tl.max(scores_2, axis=1)
            m_new_2     = tl.maximum(m_i_2, m_ij_2)
            both_inf_2  = (m_i_2 == float('-inf')) & (m_new_2 == float('-inf'))
            alpha_2     = tl.where(both_inf_2, 1.0, tl.exp(m_i_2 - m_new_2))
            p_2         = tl.where(both_inf_2[:, None], 0.0, tl.exp(scores_2 - m_new_2[:, None]))
            l_i_2       = alpha_2 * l_i_2 + tl.sum(p_2, axis=1)
            acc_2       = alpha_2[:, None] * acc_2 + tl.dot(p_2.to(v.dtype), v)
            m_i_2       = m_new_2

        # Process head 3
        if HEADS_PER_GROUP > 3:
            scores_3    = tl.dot(q_3, k_t) * softmax_scale
            scores_3    = tl.where(base_mask, scores_3, float('-inf'))
            m_ij_3      = tl.max(scores_3, axis=1)
            m_new_3     = tl.maximum(m_i_3, m_ij_3)
            both_inf_3  = (m_i_3 == float('-inf')) & (m_new_3 == float('-inf'))
            alpha_3     = tl.where(both_inf_3, 1.0, tl.exp(m_i_3 - m_new_3))
            p_3         = tl.where(both_inf_3[:, None], 0.0, tl.exp(scores_3 - m_new_3[:, None]))
            l_i_3       = alpha_3 * l_i_3 + tl.sum(p_3, axis=1)
            acc_3       = alpha_3[:, None] * acc_3 + tl.dot(p_3.to(v.dtype), v)
            m_i_3       = m_new_3

    # Finalize and store outputs
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
    HEADS_PER_GROUP: tl.constexpr,
):
    """GQA self-attention optimized to load KV once per group
    
    Fixed: KV loaded once per block, reused across all heads in group
    """

    pid_batch   = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    pid_m       = tl.program_id(2)

    q_start = pid_m * BLOCK_M
    offs_m  = q_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, BLOCK_D)
    offs_n  = tl.arange(0, BLOCK_N)
    mask_m  = offs_m < seq_len

    if use_segment_mask:
        q_seg_ptrs = segment_ids_ptr + pid_batch * stride_segb + offs_m * stride_segs
        q_seg_ids  = tl.load(q_seg_ptrs, mask=mask_m, other=-1)

    kv_end = seq_len
    if is_causal:
        kv_end = tl.minimum(seq_len, (pid_m + 1) * BLOCK_M)

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

    # Load all Q heads for this group upfront
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

    # Load ALiBi slopes if needed
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

    # Iterate over KV blocks - load K,V ONCE per block
    for n_start in range(0, kv_end, BLOCK_N):
        offs_n_curr = n_start + offs_n
        mask_n      = offs_n_curr < seq_len

        # Load K and V ONCE for all heads in group
        k_ptrs = (K_ptr +
                  pid_batch * stride_kb +
                  pid_kv_head * stride_kh +
                  offs_n_curr[:, None] * stride_ks +
                  offs_d[None, :] * stride_kd)
        k      = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        
        v_ptrs = (V_ptr +
                  pid_batch * stride_vb +
                  pid_kv_head * stride_vh +
                  offs_n_curr[:, None] * stride_vs +
                  offs_d[None, :] * stride_vd)
        v      = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        k_t = tl.trans(k)

        # Compute base mask (shared across heads)
        valid_mask = mask_m[:, None] & mask_n[None, :]

        if is_causal:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            valid_mask  = valid_mask & causal_mask

        if use_segment_mask:
            k_seg_ptrs = segment_ids_ptr + pid_batch * stride_segb + offs_n_curr * stride_segs
            k_seg_ids  = tl.load(k_seg_ptrs, mask=mask_n, other=-1)
            same_seg   = (q_seg_ids[:, None] == k_seg_ids[None, :]) & (q_seg_ids[:, None] >= 0)
            valid_mask = valid_mask & same_seg

        # Precompute position differences for ALiBi
        if use_alibi:
            q_pos = offs_m[:, None].to(tl.float32)
            k_pos = offs_n_curr[None, :].to(tl.float32)
            pos_diff = tl.abs(q_pos - k_pos)

        # Process head 0
        scores_0    = tl.dot(q_0, k_t) * softmax_scale
        if use_alibi:
            scores_0 = scores_0 - alibi_slope_0 * pos_diff
        scores_0    = tl.where(valid_mask, scores_0, float('-inf'))
        m_ij_0      = tl.max(scores_0, axis=1)
        m_new_0     = tl.maximum(m_i_0, m_ij_0)
        both_inf_0  = (m_i_0 == float('-inf')) & (m_new_0 == float('-inf'))
        alpha_0     = tl.where(both_inf_0, 1.0, tl.exp(m_i_0 - m_new_0))
        p_0         = tl.where(both_inf_0[:, None], 0.0, tl.exp(scores_0 - m_new_0[:, None]))
        l_i_0       = alpha_0 * l_i_0 + tl.sum(p_0, axis=1)
        acc_0       = alpha_0[:, None] * acc_0 + tl.dot(p_0.to(v.dtype), v)
        m_i_0       = m_new_0

        # Process head 1
        if HEADS_PER_GROUP > 1:
            scores_1    = tl.dot(q_1, k_t) * softmax_scale
            if use_alibi:
                scores_1 = scores_1 - alibi_slope_1 * pos_diff
            scores_1    = tl.where(valid_mask, scores_1, float('-inf'))
            m_ij_1      = tl.max(scores_1, axis=1)
            m_new_1     = tl.maximum(m_i_1, m_ij_1)
            both_inf_1  = (m_i_1 == float('-inf')) & (m_new_1 == float('-inf'))
            alpha_1     = tl.where(both_inf_1, 1.0, tl.exp(m_i_1 - m_new_1))
            p_1         = tl.where(both_inf_1[:, None], 0.0, tl.exp(scores_1 - m_new_1[:, None]))
            l_i_1       = alpha_1 * l_i_1 + tl.sum(p_1, axis=1)
            acc_1       = alpha_1[:, None] * acc_1 + tl.dot(p_1.to(v.dtype), v)
            m_i_1       = m_new_1

        # Process head 2
        if HEADS_PER_GROUP > 2:
            scores_2    = tl.dot(q_2, k_t) * softmax_scale
            if use_alibi:
                scores_2 = scores_2 - alibi_slope_2 * pos_diff
            scores_2    = tl.where(valid_mask, scores_2, float('-inf'))
            m_ij_2      = tl.max(scores_2, axis=1)
            m_new_2     = tl.maximum(m_i_2, m_ij_2)
            both_inf_2  = (m_i_2 == float('-inf')) & (m_new_2 == float('-inf'))
            alpha_2     = tl.where(both_inf_2, 1.0, tl.exp(m_i_2 - m_new_2))
            p_2         = tl.where(both_inf_2[:, None], 0.0, tl.exp(scores_2 - m_new_2[:, None]))
            l_i_2       = alpha_2 * l_i_2 + tl.sum(p_2, axis=1)
            acc_2       = alpha_2[:, None] * acc_2 + tl.dot(p_2.to(v.dtype), v)
            m_i_2       = m_new_2

        # Process head 3
        if HEADS_PER_GROUP > 3:
            scores_3    = tl.dot(q_3, k_t) * softmax_scale
            if use_alibi:
                scores_3 = scores_3 - alibi_slope_3 * pos_diff
            scores_3    = tl.where(valid_mask, scores_3, float('-inf'))
            m_ij_3      = tl.max(scores_3, axis=1)
            m_new_3     = tl.maximum(m_i_3, m_ij_3)
            both_inf_3  = (m_i_3 == float('-inf')) & (m_new_3 == float('-inf'))
            alpha_3     = tl.where(both_inf_3, 1.0, tl.exp(m_i_3 - m_new_3))
            p_3         = tl.where(both_inf_3[:, None], 0.0, tl.exp(scores_3 - m_new_3[:, None]))
            l_i_3       = alpha_3 * l_i_3 + tl.sum(p_3, axis=1)
            acc_3       = alpha_3[:, None] * acc_3 + tl.dot(p_3.to(v.dtype), v)
            m_i_3       = m_new_3

    # Finalize and store outputs
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
        self.head_dim        = config.head_dim
        self.heads_per_group = config.num_heads // config.num_kv_heads
        self.softmax_scale   = 1.0 / math.sqrt(self.head_dim)

        assert config.num_heads % config.num_kv_heads == 0
        assert self.heads_per_group <= 4, "Current kernel supports up to 4 heads per group"

        self.q       = nn.Linear(config.embed_dim, config.num_heads * self.head_dim, bias=False)
        self.k       = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v       = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.o       = nn.Linear(config.num_heads * self.head_dim, config.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if config.use_alibi and not config.is_cross_attn:
            slopes = self._compute_alibi_slopes(config.num_heads)
            self.register_buffer('alibi_slopes', slopes)
        else:
            self.alibi_slopes = None

        self._triton_supported = _check_triton_support()

    @staticmethod
    def _compute_alibi_slopes(num_heads):
        """Compute ALiBi slopes for position-aware attention"""

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

    def _validate_inputs(self, hidden_states, key_value_states=None, segment_ids=None, q_segment_ids=None, kv_segment_ids=None):
        """Validate input tensors"""

        B, L, D = hidden_states.shape
        assert D == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {D}"

        if self.is_cross_attn:
            assert key_value_states is not None, "Cross-attention requires key_value_states"
            assert key_value_states.shape[0] == B, "Batch size mismatch"
            assert key_value_states.shape[2] == D, "Embed dim mismatch"

        if segment_ids is not None:
            assert segment_ids.shape[0] == B, "Segment IDs batch size mismatch"
            assert segment_ids.shape[1] == L, "Segment IDs length mismatch"

        if q_segment_ids is not None and kv_segment_ids is not None:
            assert q_segment_ids.shape[0] == B, "Q segment IDs batch size mismatch"
            assert kv_segment_ids.shape[0] == B, "KV segment IDs batch size mismatch"

    def _forward_pytorch_cross(self, hidden_states, key_value_states, attention_mask=None, q_segment_ids=None, kv_segment_ids=None):
        """PyTorch fallback for segment-aware cross-attention"""

        B, L_q, _ = hidden_states.shape
        L_kv      = key_value_states.shape[1]

        q = self.q(hidden_states).view(B, L_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(key_value_states).view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(key_value_states).view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        q_grouped = q.view(B, self.num_kv_heads, self.heads_per_group, L_q, self.head_dim)
        k_exp     = k.unsqueeze(2)
        scores    = th.matmul(q_grouped, k_exp.transpose(-2, -1)) * self.softmax_scale

        if q_segment_ids is not None and kv_segment_ids is not None:
            seg_mask = (q_segment_ids.unsqueeze(-1) == kv_segment_ids.unsqueeze(-2))
            seg_mask = seg_mask & (q_segment_ids.unsqueeze(-1) >= 0)
            seg_mask = seg_mask.unsqueeze(1).unsqueeze(2)
            scores   = scores.masked_fill(~seg_mask, float('-inf'))

        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(2)

        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = self.dropout(attn_weights)

        v_exp = v.unsqueeze(2)
        out   = th.matmul(attn_weights, v_exp)
        out   = out.reshape(B, self.num_heads, L_q, self.head_dim)
        out   = out.permute(0, 2, 1, 3).reshape(B, L_q, self.num_heads * self.head_dim)
        out   = self.o(out)

        return out, None

    def _forward_pytorch_self(self, hidden_states, attention_mask=None, position_bias=None, segment_ids=None):
        """PyTorch fallback for segment-aware self-attention"""

        B, L, _ = hidden_states.shape

        q = self.q(hidden_states).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        q_grouped = q.view(B, self.num_kv_heads, self.heads_per_group, L, self.head_dim)
        k_exp     = k.unsqueeze(2)
        scores    = th.matmul(q_grouped, k_exp.transpose(-2, -1)) * self.softmax_scale

        if self.alibi_slopes is not None and position_bias is None:
            positions     = th.arange(L, device=hidden_states.device)
            dist_matrix   = th.abs(positions[:, None] - positions[None, :]).float()
            alibi_slopes  = self.alibi_slopes.view(self.num_kv_heads, self.heads_per_group)
            position_bias = -alibi_slopes.view(1, self.num_kv_heads, self.heads_per_group, 1, 1) * \
                            dist_matrix.view(1, 1, 1, L, L)

        if position_bias is not None:
            scores = scores + position_bias

        if self.is_causal:
            causal_mask = th.triu(th.ones(L, L, device=scores.device), diagonal=1).bool()
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

        v_exp = v.unsqueeze(2)
        out   = th.matmul(attn_weights, v_exp)
        out   = out.reshape(B, self.num_heads, L, self.head_dim)
        out   = out.permute(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        out   = self.o(out)

        return out, position_bias

    def _forward_triton_cross(self, hidden_states, key_value_states, attention_mask=None, q_segment_ids=None, kv_segment_ids=None):
        """Triton-optimized segment-aware cross-attention"""

        B, L_q, _ = hidden_states.shape
        L_kv      = key_value_states.shape[1]

        q = self.q(hidden_states).view(B, L_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = self.k(key_value_states).view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = self.v(key_value_states).view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        out              = th.empty_like(q)
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

        BLOCK_M      = 64
        BLOCK_N      = 64
        BLOCK_D      = self.head_dim
        num_m_blocks = (L_q + BLOCK_M - 1) // BLOCK_M
        grid         = (B, self.num_kv_heads, num_m_blocks)

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
            BLOCK_M         = BLOCK_M,
            BLOCK_N         = BLOCK_N,
            BLOCK_D         = BLOCK_D,
            HEADS_PER_GROUP = self.heads_per_group,
        )

        out = out.permute(0, 2, 1, 3).reshape(B, L_q, self.num_heads * self.head_dim)
        out = self.o(out)
        out = self.dropout(out)

        return out, None

    def _forward_triton_self(self, hidden_states, attention_mask=None, segment_ids=None):
        """Triton-optimized segment-aware self-attention"""

        B, L, _ = hidden_states.shape

        q = self.q(hidden_states).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = self.k(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = self.v(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        out              = th.empty_like(q)
        use_segment_mask = segment_ids is not None

        if use_segment_mask:
            segment_ids = segment_ids.contiguous()
            stride_segb = segment_ids.stride(0)
            stride_segs = segment_ids.stride(1) if segment_ids.dim() > 1 else 1
        else:
            stride_segb = 0
            stride_segs = 0

        BLOCK_M      = 64
        BLOCK_N      = 64
        BLOCK_D      = self.head_dim
        num_m_blocks = (L + BLOCK_M - 1) // BLOCK_M
        grid         = (B, self.num_kv_heads, num_m_blocks)

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
            BLOCK_M         = BLOCK_M,
            BLOCK_N         = BLOCK_N,
            BLOCK_D         = BLOCK_D,
            HEADS_PER_GROUP = self.heads_per_group,
        )

        out = out.permute(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        out = self.o(out)
        out = self.dropout(out)

        return out, None

    def forward(self, hidden_states, key_value_states=None, attention_mask=None, position_bias=None, segment_ids=None, q_segment_ids=None, kv_segment_ids=None):
        """Forward with segment-aware masking for packed sequences"""

        self._validate_inputs(hidden_states, key_value_states, segment_ids, q_segment_ids, kv_segment_ids)
        use_triton = self._triton_supported and hidden_states.is_cuda

        if self.is_cross_attn and key_value_states is not None:
            if use_triton:
                return self._forward_triton_cross(hidden_states, key_value_states, attention_mask, q_segment_ids, kv_segment_ids)
            return self._forward_pytorch_cross(hidden_states, key_value_states, attention_mask, q_segment_ids, kv_segment_ids)
        else:
            if use_triton:
                return self._forward_triton_self(hidden_states, attention_mask, segment_ids)
            return self._forward_pytorch_self(hidden_states, attention_mask, position_bias, segment_ids)

    def get_kv_cache_size(self, batch_size, seq_len):
        """Return KV cache size in bytes for float16"""

        return batch_size * seq_len * self.num_kv_heads * self.head_dim * 2 * 2