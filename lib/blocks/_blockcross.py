from __future__ import annotations

import math
from dataclasses import dataclass

import torch                as th
import torch.nn             as nn
import torch.nn.functional  as F


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


##########################
##### BLOCK CROSS    #####
##########################


class BlockCrossAttention(nn.Module):
    """
    Block-wise cross-attention for efficient decoder-encoder attention

    Groups decoder tokens into blocks and computes ONE cross-attention per block.
    The result is broadcast to all tokens in the block.

    Memory reduction: seq_len/block_size factor
    - 171K tokens / 16 = 10.7K blocks
    - 10.7K x 512 latents = 5.5M vs 171K x 512 = 87M (16x reduction)
    """

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

        self.q = nn.Linear(config.embed_dim, config.num_heads * self.head_dim, bias=False)
        self.k = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(config.num_heads * self.head_dim, config.embed_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

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

        return {
            "dense_scores":       dense_scores,
            "block_scores":       sparse_scores,
            "reduction_vs_dense": reduction,
            "reduction_factor":   reduction,
            "block_count":        block_count,
        }
