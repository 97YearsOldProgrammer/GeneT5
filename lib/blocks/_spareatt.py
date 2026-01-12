from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class SparseAttentionConfig:
    """Configuration for BigBird sparse attention"""
    
    embed_dim:          int     = 768
    num_heads:          int     = 12
    block_size:         int     = 64
    window_size:        int     = 256       # Local window (tokens, not blocks)
    num_global_tokens:  int     = 64        # Global tokens at start
    num_random_blocks:  int     = 3         # Random blocks per query block
    dropout:            float   = 0.0
    use_alibi:          bool    = True


#############################
##### BACKEND DETECTION #####
#############################

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    from torch.nn.attention.flex_attention import (
        flex_attention, 
        create_block_mask,
    )
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False

MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


##############################
#####  SPARSE ATTENTION  #####
##############################


class SparseAttention(nn.Module):

    
    def __init__(self, config, is_causal=False, force_backend=None):
        super().__init__()
        
        self.config     = config
        self.is_causal  = is_causal
        
        # Select backend following the same pattern as MoE
        if force_backend == "triton":
            if not TRITON_AVAILABLE:
                raise RuntimeError("Triton not available. Install: pip install triton")
            self.backend = "triton"
            self.attention = TritonSparseAttention(config)
            
        elif force_backend == "flex":
            if not FLEX_ATTENTION_AVAILABLE:
                raise RuntimeError("FlexAttention not available. Requires PyTorch 2.5+")
            self.backend = "flex"
            self.attention = FlexAttentionSparse(config, is_causal=is_causal)
            
        elif force_backend is None:
            # Auto-select backend based on availability
            if TRITON_AVAILABLE and torch.cuda.is_available():
                self.backend = "triton"
                self.attention = TritonSparseAttention(config)
            elif FLEX_ATTENTION_AVAILABLE:
                # FlexAttention works on MPS and other backends
                self.backend = "flex"
                self.attention = FlexAttentionSparse(config, is_causal=is_causal)
            else:
                # Fallback error message
                raise RuntimeError(
                    "No sparse attention backend available. "
                    "Install either: pip install triton (for CUDA) "
                    "or upgrade to PyTorch 2.5+ (for FlexAttention on MPS/CPU)"
                )
        else:
            raise ValueError(f"Unknown backend: {force_backend}")
        
        print(f"SparseAttention initialized with backend: {self.backend}")
    
    def forward(self, hidden_states, attention_mask=None):
        return self.attention(hidden_states, attention_mask)


##################
#####  CUDA  #####
##################


if TRITON_AVAILABLE:

    @triton.jit
    def _sparse_attention_fwd_kernel(
        
        # Pointers
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        
        # Block indices (which K/V blocks each Q block attends to)
        block_indices_ptr,  # (num_q_blocks, blocks_per_query)
        
        # Dimensions
        seq_len, num_heads, head_dim,
        blocks_per_query,   # How many K/V blocks each Q block attends to
        
        # Strides for Q, K, V: (batch, head, seq, dim)
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        
        # ALiBi slopes (optional)
        alibi_slopes_ptr,
        use_alibi: tl.constexpr,
        
        # Softmax scale
        softmax_scale,
        # Block sizes
        BLOCK_M: tl.constexpr,  # Query block size
        BLOCK_N: tl.constexpr,  # Key/Value block size
        BLOCK_D: tl.constexpr,  # Head dimension block
    ):


        # Program IDs
        pid_batch   = tl.program_id(0)
        pid_head    = tl.program_id(1)
        pid_q_block = tl.program_id(2)
        
        # Compute offsets
        q_block_start   = pid_q_block * BLOCK_M
        offs_m          = q_block_start + tl.arange(0, BLOCK_M)
        offs_d          = tl.arange(0, BLOCK_D)
        
        # Mask for valid query positions
        mask_m = offs_m < seq_len
        
        # Load query block: (BLOCK_M, BLOCK_D)
        q_ptrs = (Q_ptr + 
                  pid_batch * stride_qb + 
                  pid_head * stride_qh + 
                  offs_m[:, None] * stride_qs + 
                  offs_d[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        
        # ALiBi slope for this head
        if use_alibi:
            alibi_slope = tl.load(alibi_slopes_ptr + pid_head)
        
        # Initialize accumulators
        m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)  # Running max
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                # Running sum
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)        # Output accumulator
        
        # Iterate over K/V blocks that this Q block attends to
        for block_idx in range(blocks_per_query):
            # Load which K/V block to attend to
            kv_block_id = tl.load(
                block_indices_ptr + pid_q_block * blocks_per_query + block_idx
            )
            
            # Skip invalid blocks (marked as -1)
            if kv_block_id >= 0:
                kv_block_start = kv_block_id * BLOCK_N
                offs_n = kv_block_start + tl.arange(0, BLOCK_N)
                mask_n = offs_n < seq_len
                
                # Load K block: (BLOCK_N, BLOCK_D)
                k_ptrs = (K_ptr + 
                          pid_batch * stride_kb + 
                          pid_head * stride_kh + 
                          offs_n[:, None] * stride_ks + 
                          offs_d[None, :] * stride_kd)
                k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
                
                # Compute attention scores: (BLOCK_M, BLOCK_N)
                scores = tl.dot(q, tl.trans(k)) * softmax_scale
                
                # Apply ALiBi bias
                if use_alibi:
                    # Compute |query_pos - key_pos| for each pair
                    q_pos       = offs_m[:, None].to(tl.float32)
                    k_pos       = offs_n[None, :].to(tl.float32)
                    dist        = tl.abs(q_pos - k_pos)
                    alibi_bias  = -alibi_slope * dist
                    scores      = scores + alibi_bias
                
                # Mask invalid positions
                scores = tl.where(
                    mask_m[:, None] & mask_n[None, :],
                    scores,
                    float('-inf')
                )
                
                # Online softmax update
                m_ij    = tl.max(scores, axis=1)
                m_new   = tl.maximum(m_i, m_ij)
                
                # Correction factors
                alpha   = tl.exp(m_i - m_new)
                beta    = tl.exp(m_ij - m_new)
                
                # Update running sum
                l_i = alpha * l_i + beta * tl.sum(tl.exp(scores - m_ij[:, None]), axis=1)
                
                # Load V block: (BLOCK_N, BLOCK_D)
                v_ptrs = (V_ptr + 
                          pid_batch * stride_vb + 
                          pid_head * stride_vh + 
                          offs_n[:, None] * stride_vs + 
                          offs_d[None, :] * stride_vd)
                v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
                
                # Update accumulator
                p   = tl.exp(scores - m_ij[:, None])
                acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)
                
                # Update max
                m_i = m_new
        
        # Final normalization
        acc = acc / l_i[:, None]
        
        # Store output
        out_ptrs = (Out_ptr + 
                    pid_batch * stride_ob + 
                    pid_head * stride_oh + 
                    offs_m[:, None] * stride_os + 
                    offs_d[None, :] * stride_od)
        tl.store(out_ptrs, acc, mask=mask_m[:, None])


class TritonSparseAttention(nn.Module):
    """BigBird sparse attention using Triton kernels"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config             = config
        self.embed_dim          = config.embed_dim
        self.num_heads          = config.num_heads
        self.head_dim           = config.embed_dim // config.num_heads
        self.block_size         = config.block_size
        self.window_size        = config.window_size
        self.num_global_tokens  = config.num_global_tokens
        self.num_random_blocks  = config.num_random_blocks
        self.softmax_scale      = 1.0 / math.sqrt(self.head_dim)
        
        # Projections
        self.q = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.k = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.v = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.o = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # ALiBi slopes
        if config.use_alibi:
            slopes = self._compute_alibi_slopes(config.num_heads)
            self.register_buffer('alibi_slopes', slopes)
        else:
            self.alibi_slopes = None
        
        # Cache for block indices
        self._block_indices_cache = {}
    
    @staticmethod
    def _compute_alibi_slopes(num_heads):
        """Compute ALiBi slopes for each head"""
        
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
    
    def _compute_block_indices(self, seq_len, device):
        """Compute which K/V blocks each Q block should attend to (BigBird pattern)"""
        cache_key = (seq_len, device)
        if cache_key in self._block_indices_cache:
            return self._block_indices_cache[cache_key]
        
        num_blocks          = seq_len // self.block_size
        num_global_blocks   = self.num_global_tokens // self.block_size
        window_blocks       = self.window_size // self.block_size
        
        # Maximum blocks any query could attend to
        max_blocks = (
            num_global_blocks +         # Global blocks
            (2 * window_blocks + 1) +   # Window blocks
            self.num_random_blocks      # Random blocks
        )
        
        block_indices = torch.full(
            (num_blocks, max_blocks), -1, dtype=torch.long, device=device
        )
        
        for q_block in range(num_blocks):
            idx = 0
            attended = set()
            
            # Global blocks (always attend to first blocks)
            for g in range(num_global_blocks):
                if g not in attended:
                    block_indices[q_block, idx] = g
                    attended.add(g)
                    idx += 1
            
            # Window blocks
            window_start    = max(0, q_block - window_blocks)
            window_end      = min(num_blocks, q_block + window_blocks + 1)
            for w in range(window_start, window_end):
                if w not in attended:
                    block_indices[q_block, idx] = w
                    attended.add(w)
                    idx += 1
            
            # Random blocks
            available = [b for b in range(num_blocks) if b not in attended]
            if available:
                torch.manual_seed(q_block)  # Reproducible randomness
                perm = torch.randperm(len(available))[:self.num_random_blocks]
                for p in perm:
                    block_indices[q_block, idx] = available[p]
                    idx += 1
        
        self._block_indices_cache[cache_key] = block_indices
        return block_indices
    
    def forward(self, hidden_states, attention_mask=None):
        """Forward pass with BigBird sparse attention"""
        B, L, D = hidden_states.shape
        
        # Pad to block size
        pad_len = (self.block_size - L % self.block_size) % self.block_size
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            L_padded = L + pad_len
        else:
            L_padded = L
        
        # Project Q, K, V
        q = self.q(hidden_states).view(B, L_padded, self.num_heads, self.head_dim)
        k = self.k(hidden_states).view(B, L_padded, self.num_heads, self.head_dim)
        v = self.v(hidden_states).view(B, L_padded, self.num_heads, self.head_dim)
        
        # Transpose to (B, H, L, D)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        # Allocate output
        out = torch.empty_like(q)
        
        # Get block indices
        block_indices = self._compute_block_indices(L_padded, hidden_states.device)
        num_blocks = L_padded // self.block_size
        blocks_per_query = block_indices.shape[1]
        
        # Launch kernel
        BLOCK_M = self.block_size
        BLOCK_N = self.block_size
        BLOCK_D = self.head_dim
        
        grid = (B, self.num_heads, num_blocks)
        
        # Pass a valid pointer for alibi_slopes_ptr
        alibi_ptr = self.alibi_slopes if self.alibi_slopes is not None else q
        
        _sparse_attention_fwd_kernel[grid](
            q, k, v, out,
            block_indices,
            L_padded, self.num_heads, self.head_dim,
            blocks_per_query,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            alibi_ptr,  # FIXED: Always pass a valid pointer
            self.alibi_slopes is not None,
            self.softmax_scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
        
        # Reshape output
        out = out.transpose(1, 2).reshape(B, L_padded, D)
        out = self.o(out)
        
        # Remove padding
        if pad_len > 0:
            out = out[:, :L, :]
        
        out = self.dropout(out)
        
        return out, None


############################
#####  FLEX ATTENTION  #####
############################


if FLEX_ATTENTION_AVAILABLE:

    class FlexAttentionSparse(nn.Module):
        """BigBird sparse attention using FlexAttention"""

        def __init__(self, config, is_causal=False):
            super().__init__()
            
            self.config             = config
            self.embed_dim          = config.embed_dim
            self.num_heads          = config.num_heads
            self.head_dim           = config.embed_dim // config.num_heads
            self.block_size         = config.block_size
            self.window_size        = config.window_size
            self.num_global_tokens  = config.num_global_tokens
            self.num_random_blocks  = config.num_random_blocks
            self.is_causal          = is_causal
            
            # Projections
            self.q = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            self.k = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            self.v = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            self.o = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            
            self.dropout = nn.Dropout(config.dropout)
            
            # ALiBi slopes
            if config.use_alibi:
                slopes = self._compute_alibi_slopes(config.num_heads)
                self.register_buffer('alibi_slopes', slopes)
            else:
                self.alibi_slopes = None
            
            # Cached masks
            self._mask_cache = {}
        
        @staticmethod
        def _compute_alibi_slopes(num_heads):
            """Compute ALiBi slopes"""
            
            def get_slopes(n):
                if n == 1:
                    return torch.tensor([1.0])
                base = 2 ** (-2 ** -(math.log2(n) - 3))
                return torch.tensor([base ** i for i in range(1, n + 1)])
            
            if math.log2(num_heads).is_integer():
                return get_slopes(num_heads)
            else:
                closest = 2 ** math.floor(math.log2(num_heads))
                return torch.cat([
                    get_slopes(closest),
                    get_slopes(2 * closest)[0::2][:num_heads - closest]
                ])
        
        def _create_score_mod(self):
            """Score modification function for FlexAttention"""
            alibi_slopes = self.alibi_slopes
            
            def score_mod(score, batch, head, q_idx, kv_idx):
                """Apply ALiBi positional bias"""
                if alibi_slopes is not None:
                    slope       = alibi_slopes[head]
                    distance    = torch.abs(q_idx - kv_idx).float()
                    score       = score - slope * distance
                return score
            
            return score_mod
        
        def _create_bigbird_mask_mod(self, seq_len, device):
            """Create BigBird attention mask for FlexAttention (Vectorized)"""
            window      = self.window_size
            num_global  = self.num_global_tokens
            block_size  = self.block_size
            num_random  = self.num_random_blocks
            is_causal   = self.is_causal
            
            num_blocks = (seq_len + block_size - 1) // block_size
            
            # Precompute Random Block Connections as a Dense Tensor
            random_mask = torch.zeros((num_blocks, num_blocks), dtype=torch.bool, device=device)
            
            for q_block in range(num_blocks):
                torch.manual_seed(q_block * 31337)
                available = list(range(num_blocks))
                if q_block in available:
                    available.remove(q_block)
                
                # Select random blocks
                if available:
                    perm = torch.randperm(len(available))[:num_random]
                    targets = [available[i] for i in perm]
                    random_mask[q_block, targets] = True

            def mask_mod(b, h, q_idx, kv_idx):
                """Returns True if q_idx should attend to kv_idx"""
                
                # Global attention (First N tokens)
                is_global = (q_idx < num_global) | (kv_idx < num_global)
                
                # Sliding Window attention
                is_window = (q_idx - kv_idx).abs() <= window
                
                # Random attention (Block Lookup)
                q_block  = q_idx // block_size
                kv_block = kv_idx // block_size
                
                # Instead of iterating over dictionary, use direct tensor indexing
                is_random = random_mask[q_block, kv_block]
                
                mask = is_global | is_window | is_random
                
                if is_causal:
                    mask = mask & (kv_idx <= q_idx)
                
                return mask
            
            return mask_mod
        
        def _get_block_mask(self, seq_len, device):
            """Get or create cached block mask"""
            cache_key = (seq_len, device)
            
            if cache_key not in self._mask_cache:
                # Pass device to _create_bigbird_mask_mod so the random_mask
                mask_mod = self._create_bigbird_mask_mod(seq_len, device)
                
                # Create block mask for efficient sparse computation
                block_mask = create_block_mask(
                    mask_mod,
                    B=None,  # Broadcast over batch
                    H=None,  # Broadcast over heads
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=device,
                    BLOCK_SIZE=self.block_size,
                )
                
                self._mask_cache[cache_key] = block_mask
            
            return self._mask_cache[cache_key]
        
        def forward(self, hidden_states, attention_mask=None):
            """Forward pass using FlexAttention with BigBird pattern"""
            B, L, D = hidden_states.shape
            
            # Project Q, K, V
            q = self.q(hidden_states).view(B, L, self.num_heads, self.head_dim)
            k = self.k(hidden_states).view(B, L, self.num_heads, self.head_dim)
            v = self.v(hidden_states).view(B, L, self.num_heads, self.head_dim)
            
            # Transpose to (B, H, L, D)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Get block mask and score mod
            block_mask  = self._get_block_mask(L, hidden_states.device)
            score_mod   = self._create_score_mod()
            
            # Apply FlexAttention
            out = flex_attention(
                q, k, v,
                score_mod=score_mod,
                block_mask=block_mask,
            )
            
            # Reshape output
            out = out.transpose(1, 2).reshape(B, L, D)
            out = self.o(out)
            out = self.dropout(out)
            
            return out, None