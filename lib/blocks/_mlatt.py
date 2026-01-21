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
class MLAttentionConfig:
    embed_dim:        int   = 768
    num_heads:        int   = 12
    kv_lora_rank:     int   = 64
    q_lora_rank:      int   = 128
    rope_head_dim:    int   = 32
    v_head_dim:       int   = None
    dropout:          float = 0.0
    use_alibi:        bool  = True
    max_seq_len:      int   = 8192


########################
#####  TRITON OPS  #####
########################


@triton.jit
def mla_compress_kv_kernel(
    input_ptr, compressed_ptr, proj_ptr,
    seq_len, embed_dim, lora_rank,
    stride_ib, stride_is, stride_id,
    stride_cb, stride_cs, stride_cd,
    stride_ph, stride_pd,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Compress input into low-rank latent space for KV cache"""
    
    pid_batch = tl.program_id(0)
    pid_seq   = tl.program_id(1)
    
    seq_offset = pid_seq * BLOCK_S
    offs_s     = seq_offset + tl.arange(0, BLOCK_S)
    mask_s     = offs_s < seq_len
    
    for r in range(0, lora_rank, BLOCK_R):
        offs_r = r + tl.arange(0, BLOCK_R)
        mask_r = offs_r < lora_rank
        
        acc = tl.zeros((BLOCK_S, BLOCK_R), dtype=tl.float32)
        
        for d in range(0, embed_dim, BLOCK_D):
            offs_d = d + tl.arange(0, BLOCK_D)
            mask_d = offs_d < embed_dim
            
            input_ptrs = (input_ptr +
                          pid_batch * stride_ib +
                          offs_s[:, None] * stride_is +
                          offs_d[None, :] * stride_id)
            x = tl.load(input_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)
            
            proj_ptrs = (proj_ptr +
                         offs_d[:, None] * stride_ph +
                         offs_r[None, :] * stride_pd)
            w = tl.load(proj_ptrs, mask=mask_d[:, None] & mask_r[None, :], other=0.0)
            
            acc += tl.dot(x, w)
        
        out_ptrs = (compressed_ptr +
                    pid_batch * stride_cb +
                    offs_s[:, None] * stride_cs +
                    offs_r[None, :] * stride_cd)
        tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_r[None, :])


@triton.jit
def mla_attention_fwd_kernel(
    Q_ptr, C_ptr, K_up_ptr, V_up_ptr, Out_ptr,
    seq_len, num_heads, head_dim, lora_rank,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_cb, stride_cs, stride_cr,
    stride_kh, stride_kr, stride_kd,
    stride_vh, stride_vr, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    alibi_slopes_ptr,
    use_alibi: tl.constexpr,
    softmax_scale,
    is_causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Forward pass for Multi-Head Latent Attention with compressed KV"""
    
    pid_batch = tl.program_id(0)
    pid_head  = tl.program_id(1)
    pid_m     = tl.program_id(2)
    
    q_start = pid_m * BLOCK_M
    offs_m  = q_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, BLOCK_D)
    
    mask_m = offs_m < seq_len
    
    q_ptrs = (Q_ptr +
              pid_batch * stride_qb +
              pid_head * stride_qh +
              offs_m[:, None] * stride_qs +
              offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    if use_alibi:
        alibi_slope = tl.load(alibi_slopes_ptr + pid_head)
    
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    for n_start in range(0, seq_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        if is_causal:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
        
        k_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        v_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        
        for r in range(0, lora_rank, BLOCK_R):
            offs_r = r + tl.arange(0, BLOCK_R)
            mask_r = offs_r < lora_rank
            
            c_ptrs = (C_ptr +
                      pid_batch * stride_cb +
                      offs_n[:, None] * stride_cs +
                      offs_r[None, :] * stride_cr)
            c = tl.load(c_ptrs, mask=mask_n[:, None] & mask_r[None, :], other=0.0)
            
            k_up_ptrs = (K_up_ptr +
                        pid_head * stride_kh +
                        offs_r[:, None] * stride_kr +
                        offs_d[None, :] * stride_kd)
            k_up = tl.load(k_up_ptrs, mask=mask_r[:, None], other=0.0)
            
            v_up_ptrs = (V_up_ptr +
                        pid_head * stride_vh +
                        offs_r[:, None] * stride_vr +
                        offs_d[None, :] * stride_vd)
            v_up = tl.load(v_up_ptrs, mask=mask_r[:, None], other=0.0)
            
            k_acc += tl.dot(c, k_up)
            v_acc += tl.dot(c, v_up)
        
        scores = tl.dot(q, tl.trans(k_acc)) * softmax_scale
        
        if use_alibi:
            q_pos      = offs_m[:, None].to(tl.float32)
            k_pos      = offs_n[None, :].to(tl.float32)
            dist       = tl.abs(q_pos - k_pos)
            alibi_bias = -alibi_slope * dist
            scores     = scores + alibi_bias
        
        if is_causal:
            scores = tl.where(causal_mask & mask_m[:, None] & mask_n[None, :],
                              scores, float('-inf'))
        else:
            scores = tl.where(mask_m[:, None] & mask_n[None, :],
                              scores, float('-inf'))
        
        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        beta  = tl.exp(m_ij - m_new)
        
        l_i = alpha * l_i + beta * tl.sum(tl.exp(scores - m_ij[:, None]), axis=1)
        
        p   = tl.exp(scores - m_ij[:, None])
        acc = alpha[:, None] * acc + tl.dot(p.to(v_acc.dtype), v_acc)
        
        m_i = m_new
    
    acc = acc / l_i[:, None]
    
    out_ptrs = (Out_ptr +
                pid_batch * stride_ob +
                pid_head * stride_oh +
                offs_m[:, None] * stride_os +
                offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


#########################################
#####  MULTI-HEAD LATENT ATTENTION  #####
#########################################


class MLAttention(nn.Module):
    """
    Multi-Head Latent Attention with Low-Rank KV Compression
    
    Compresses key-value pairs into a shared latent space, then projects
    per-head using up-projection matrices, reducing KV cache memory
    """
    
    def __init__(self, config, is_causal=True):
        super().__init__()
        
        self.config       = config
        self.is_causal    = is_causal
        self.embed_dim    = config.embed_dim
        self.num_heads    = config.num_heads
        self.head_dim     = config.embed_dim // config.num_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank  = config.q_lora_rank
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_down = nn.Linear(config.embed_dim, config.q_lora_rank, bias=False)
        self.q_up   = nn.Linear(config.q_lora_rank, config.num_heads * self.head_dim, bias=False)
        
        self.kv_down = nn.Linear(config.embed_dim, config.kv_lora_rank, bias=False)
        
        self.k_up = nn.Parameter(torch.empty(config.num_heads, config.kv_lora_rank, self.head_dim))
        self.v_up = nn.Parameter(torch.empty(config.num_heads, config.kv_lora_rank, self.head_dim))
        
        self.o = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
        
        if config.use_alibi:
            slopes = self._compute_alibi_slopes(config.num_heads)
            self.register_buffer('alibi_slopes', slopes)
        else:
            self.alibi_slopes = None
        
        self._use_triton = True
    
    def _init_weights(self):
        """Initialize weights with small variance for stable training"""
        nn.init.normal_(self.q_down.weight, std=0.02)
        nn.init.normal_(self.q_up.weight, std=0.02)
        nn.init.normal_(self.kv_down.weight, std=0.02)
        nn.init.normal_(self.k_up, std=0.02)
        nn.init.normal_(self.v_up, std=0.02)
        nn.init.normal_(self.o.weight, std=0.02)
    
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
            slopes = torch.cat([
                get_slopes(closest_power),
                get_slopes(2 * closest_power)[0::2][:num_heads - closest_power]
            ])
            return slopes
    
    def _forward_pytorch(self, hidden_states, attention_mask=None, position_bias=None):
        """PyTorch fallback implementation for MLA"""
        
        B, L, D = hidden_states.shape
        
        q_compressed = self.q_down(hidden_states)
        q            = self.q_up(q_compressed)
        q            = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        kv_compressed = self.kv_down(hidden_states)
        
        k = torch.einsum('bsr,hrd->bshd', kv_compressed, self.k_up)
        v = torch.einsum('bsr,hrd->bshd', kv_compressed, self.v_up)
        
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale
        
        if self.alibi_slopes is not None and position_bias is None:
            positions   = torch.arange(L, device=hidden_states.device)
            dist_matrix = torch.abs(positions[:, None] - positions[None, :]).float()
            position_bias = -self.alibi_slopes.view(1, -1, 1, 1) * dist_matrix.view(1, 1, L, L)
        
        if position_bias is not None:
            scores = scores + position_bias
        
        if self.is_causal:
            causal_mask = torch.triu(torch.ones(L, L, device=scores.device), diagonal=1).bool()
            scores      = scores.masked_fill(causal_mask, float('-inf'))
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o(out)
        
        return out, position_bias
    
    def _forward_triton(self, hidden_states, attention_mask=None):
        """Triton-optimized forward pass for MLA"""
        
        B, L, D = hidden_states.shape
        
        q_compressed = self.q_down(hidden_states)
        q            = self.q_up(q_compressed)
        q            = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
        kv_compressed = self.kv_down(hidden_states)
        
        out = torch.empty_like(q)
        
        BLOCK_M = min(64, L)
        BLOCK_N = min(64, L)
        BLOCK_D = self.head_dim
        BLOCK_R = min(32, self.kv_lora_rank)
        
        num_m_blocks = (L + BLOCK_M - 1) // BLOCK_M
        grid         = (B, self.num_heads, num_m_blocks)
        
        alibi_ptr = self.alibi_slopes if self.alibi_slopes is not None else q
        
        mla_attention_fwd_kernel[grid](
            q, kv_compressed, self.k_up, self.v_up, out,
            L, self.num_heads, self.head_dim, self.kv_lora_rank,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            kv_compressed.stride(0), kv_compressed.stride(1), kv_compressed.stride(2),
            self.k_up.stride(0), self.k_up.stride(1), self.k_up.stride(2),
            self.v_up.stride(0), self.v_up.stride(1), self.v_up.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            alibi_ptr,
            self.alibi_slopes is not None,
            self.softmax_scale,
            self.is_causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            BLOCK_R=BLOCK_R,
        )
        
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o(out)
        out = self.dropout(out)
        
        return out, None
    
    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        """Forward pass for Multi-Head Latent Attention"""
        
        if self._use_triton and hidden_states.is_cuda:
            try:
                return self._forward_triton(hidden_states, attention_mask)
            except Exception:
                pass
        
        return self._forward_pytorch(hidden_states, attention_mask, position_bias)
    
    def get_kv_cache_size(self, batch_size, seq_len):
        """Return compressed KV cache size in bytes (float16)"""
        return batch_size * seq_len * self.kv_lora_rank * 2