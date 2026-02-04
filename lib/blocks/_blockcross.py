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

    Instead of each decoder token attending to all encoder tokens independently,
    we group decoder tokens into blocks and compute ONE cross-attention per block.
    The result is broadcast to all tokens in the block.

    This matches the block_size convention from sparse self-attention:
    - Decoder self-attention: 16 tokens = 1 block (sparse window)
    - Cross-attention: 16 tokens = 1 block (shared encoder context)

    Memory reduction: seq_len/block_size factor
    - 171K tokens / 16 = 10.7K blocks
    - 10.7K × 512 latents = 5.5M vs 171K × 512 = 87M (16x reduction)
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

        # Query projection (for pooled decoder blocks)
        self.q = nn.Linear(config.embed_dim, config.num_heads * self.head_dim, bias=False)

        # Key/Value projection (for encoder)
        self.k = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.o = nn.Linear(config.num_heads * self.head_dim, config.embed_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def _pool_to_blocks(self, x):
        """Pool sequence into blocks by averaging"""

        B, L, D = x.shape
        block_size = self.block_size

        # Pad if necessary
        pad_len = (block_size - L % block_size) % block_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            L_padded = L + pad_len
        else:
            L_padded = L

        # Reshape and pool: [B, num_blocks, block_size, D] → [B, num_blocks, D]
        num_blocks = L_padded // block_size
        x_blocks   = x.view(B, num_blocks, block_size, D)
        pooled     = x_blocks.mean(dim=2)

        return pooled, L, pad_len

    def _unpool_from_blocks(self, pooled, original_len, pad_len):
        """Broadcast block representations back to token level"""

        B, num_blocks, D = pooled.shape

        # Expand: [B, num_blocks, D] → [B, num_blocks, block_size, D]
        expanded = pooled.unsqueeze(2).expand(-1, -1, self.block_size, -1)

        # Reshape: [B, L_padded, D]
        unpooled = expanded.reshape(B, num_blocks * self.block_size, D)

        # Remove padding
        if pad_len > 0:
            unpooled = unpooled[:, :original_len, :]

        return unpooled

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

        B, L_dec, _ = hidden_states.shape
        L_enc       = encoder_hidden_states.shape[1]

        # Pool decoder to blocks
        pooled_dec, original_len, pad_len = self._pool_to_blocks(hidden_states)
        num_blocks = pooled_dec.shape[1]

        # Project queries from pooled decoder blocks
        q = self.q(pooled_dec).view(B, num_blocks, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [B, num_heads, num_blocks, head_dim]

        # Project keys and values from encoder
        k = self.k(encoder_hidden_states).view(B, L_enc, self.num_kv_heads, self.head_dim)
        v = self.v(encoder_hidden_states).view(B, L_enc, self.num_kv_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # [B, num_kv_heads, L_enc, head_dim]
        v = v.permute(0, 2, 1, 3)

        # Expand KV for grouped query attention
        if self.heads_per_group > 1:
            k = k.repeat_interleave(self.heads_per_group, dim=1)
            v = v.repeat_interleave(self.heads_per_group, dim=1)

        # Attention scores: [B, num_heads, num_blocks, L_enc]
        scores = th.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [B, L_enc] → [B, 1, 1, L_enc]
            mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output: [B, num_heads, num_blocks, head_dim]
        attn_out = th.matmul(attn_weights, v)

        # Reshape: [B, num_blocks, num_heads * head_dim]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, num_blocks, self.num_heads * self.head_dim)

        # Output projection
        attn_out = self.o(attn_out)

        # Unpool back to token level
        output = self._unpool_from_blocks(attn_out, original_len, pad_len)

        return output, None

    def get_memory_info(self, decoder_len, encoder_len):
        """Return memory usage comparison"""

        block_count    = (decoder_len + self.block_size - 1) // self.block_size
        dense_scores   = decoder_len * encoder_len * self.num_heads
        sparse_scores  = block_count * encoder_len * self.num_heads

        return {
            "dense_scores":     dense_scores,
            "block_scores":     sparse_scores,
            "reduction_factor": dense_scores / sparse_scores,
            "block_count":      block_count,
        }
