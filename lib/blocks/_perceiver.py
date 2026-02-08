from __future__ import annotations

import math
from dataclasses import dataclass

import torch                       as th
import torch.nn                    as nn
import torch.nn.functional         as F
import torch.utils.checkpoint      as ckpt

from lib.blocks._component         import LayerNorm


####################
#####  CONFIG  #####
####################


@dataclass
class PerceiverConfig:
    embed_dim:               int   = 768
    num_latents:             int   = 512
    num_heads:               int   = 12
    num_kv_heads:            int   = None
    num_layers:              int   = 2
    ff_dim:                  int   = None
    dropout:                 float = 0.0
    latent_init_std:         float = 0.02
    gradient_checkpointing:  bool  = False

    def __post_init__(self):

        if self.num_kv_heads is None:
            self.num_kv_heads = max(1, self.num_heads // 4)
        if self.ff_dim is None:
            self.ff_dim = self.embed_dim * 4


########################
#####  COMPONENTS  #####
########################


class CrossAttention(nn.Module):
    """Cross-attention with GQA via SDPA (Flash Attention)"""

    def __init__(self, embed_dim, num_heads, num_kv_heads, dropout=0.0):

        super().__init__()

        self.embed_dim       = embed_dim
        self.num_heads       = num_heads
        self.num_kv_heads    = num_kv_heads
        self.head_dim        = embed_dim // num_heads
        self.scale           = 1.0 / math.sqrt(self.head_dim)
        self.heads_per_group = num_heads // num_kv_heads

        assert num_heads % num_kv_heads == 0

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _expand_kv_for_gqa(self, k, v):
        """Expand KV heads for GQA using view/expand (no memory allocation)"""

        if self.heads_per_group == 1:
            return k, v

        B, H_kv, L, D = k.shape

        # Reshape to [B, num_kv_heads, 1, L, D] then expand to [B, num_kv_heads, heads_per_group, L, D]
        # Finally reshape to [B, num_heads, L, D]
        k = k.unsqueeze(2).expand(B, H_kv, self.heads_per_group, L, D)
        k = k.reshape(B, self.num_heads, L, D)

        v = v.unsqueeze(2).expand(B, H_kv, self.heads_per_group, L, D)
        v = v.reshape(B, self.num_heads, L, D)

        return k, v

    def _forward_pytorch(self, q, k, v, attention_mask):
        """PyTorch fallback with Flash Attention via SDPA"""

        B, L_q, H, D = q.shape

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Efficient GQA expansion using view/expand (no memory allocation)
        k, v = self._expand_kv_for_gqa(k, v)

        # Prepare attention mask for SDPA (additive mask)
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [B, L_kv] with 1=valid, 0=pad
            # SDPA additive mask: 0=attend, -inf=ignore
            attn_mask = th.zeros_like(attention_mask, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(attention_mask == 0, float('-inf'))
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L_kv]

        # Use SDPA for Flash Attention support (memory efficient)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask    = attn_mask,
            dropout_p    = self.dropout.p if self.training else 0.0,
            is_causal    = False,
            scale        = self.scale,
        )

        out = out.permute(0, 2, 1, 3).reshape(B, L_q, self.num_heads * self.head_dim)

        return out

    def forward(self, query, key_value, attention_mask=None):

        B, L_q, _ = query.shape
        L_kv      = key_value.shape[1]

        q = self.q_proj(query).view(B, L_q, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(B, L_kv, self.num_kv_heads, self.head_dim)
        v = self.v_proj(key_value).view(B, L_kv, self.num_kv_heads, self.head_dim)

        out = self._forward_pytorch(q, k, v, attention_mask)
        out = self.o_proj(out)

        return out


class SelfAttention(nn.Module):
    """Self-attention via SDPA (Flash Attention)"""

    def __init__(self, embed_dim, num_heads, dropout=0.0):

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _forward_pytorch(self, q, k, v):
        """PyTorch fallback with Flash Attention via SDPA"""

        B, L, H, D = q.shape

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Use SDPA for Flash Attention support (memory efficient)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = None,
            dropout_p = self.dropout.p if self.training else 0.0,
            is_causal = False,
            scale     = self.scale,
        )

        out = out.permute(0, 2, 1, 3).reshape(B, L, self.embed_dim)

        return out

    def forward(self, x):

        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        out = self._forward_pytorch(q, k, v)
        out = self.o_proj(out)

        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU"""

    def __init__(self, embed_dim, ff_dim, dropout=0.0):

        super().__init__()

        self.fc1     = nn.Linear(embed_dim, ff_dim)
        self.fc2     = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class PerceiverLayer(nn.Module):
    """Single Perceiver layer: self-attention + feed-forward"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0):

        super().__init__()

        self.self_attn = SelfAttention(embed_dim, num_heads, dropout)
        self.norm1     = LayerNorm(embed_dim)
        self.ff        = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2     = LayerNorm(embed_dim)

    def forward(self, x):

        x = x + self.self_attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))

        return x


########################
#####  COMPRESSOR  #####
########################


class PerceiverCompressor(nn.Module):
    """
    Perceiver-style encoder compression using learned latent queries

    Compresses encoder output [B, N, D] to latent representation [B, L, D]
    where L << N. Decoder cross-attention then operates on L tokens instead of N.

    Uses SDPA (Flash Attention) for memory-efficient attention.
    Supports gradient checkpointing to reduce peak memory during backward pass.
    """

    def __init__(self, config):

        super().__init__()

        self.config                 = config
        self.embed_dim              = config.embed_dim
        self.num_latents            = config.num_latents
        self.num_heads              = config.num_heads
        self.num_kv_heads           = config.num_kv_heads
        self.gradient_checkpointing = config.gradient_checkpointing

        # Learned latent queries
        self.latents = nn.Parameter(th.empty(config.num_latents, config.embed_dim))

        # Cross-attention: latents query encoder
        self.cross_attn = CrossAttention(
            embed_dim    = config.embed_dim,
            num_heads    = config.num_heads,
            num_kv_heads = config.num_kv_heads,
            dropout      = config.dropout,
        )
        self.cross_norm = LayerNorm(config.embed_dim)

        # Self-attention layers for latent refinement
        self.layers = nn.ModuleList([
            PerceiverLayer(
                embed_dim = config.embed_dim,
                num_heads = config.num_heads,
                ff_dim    = config.ff_dim,
                dropout   = config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = LayerNorm(config.embed_dim)

        self._init_weights(config.latent_init_std)

    def _init_weights(self, latent_std):
        """Initialize weights following Perceiver paper"""

        nn.init.normal_(self.latents, mean=0.0, std=latent_std)

        for name, param in self.cross_attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)

    def set_gradient_checkpointing(self, enable):
        """Enable or disable gradient checkpointing"""

        self.gradient_checkpointing = enable

    def _checkpoint_forward(self, layer, x):
        """Wrapper for gradient checkpointing compatibility"""

        # use_reentrant=False is required for torch.compile compatibility
        return ckpt.checkpoint(layer, x, use_reentrant=False)

    def forward(self, encoder_hidden, encoder_mask=None):
        """
        Compress encoder output to latent representation

        Args:
            encoder_hidden: [B, N, D] encoder output
            encoder_mask:   [B, N] attention mask (1=valid, 0=pad)

        Returns:
            latents: [B, L, D] compressed representation
        """

        B = encoder_hidden.shape[0]

        # Expand latents for batch (contiguous for torch.compile)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Cross-attention: latents query encoder
        normed    = self.cross_norm(latents)
        cross_out = self.cross_attn(normed, encoder_hidden, encoder_mask)
        latents   = latents + cross_out

        # Self-attention refinement layers with optional gradient checkpointing
        use_ckpt = self.gradient_checkpointing and self.training

        for layer in self.layers:
            if use_ckpt:
                latents = self._checkpoint_forward(layer, latents)
            else:
                latents = layer(latents)

        latents = self.final_norm(latents)

        return latents

    def get_compression_ratio(self, encoder_len):
        """Return compression ratio for given encoder length"""

        return encoder_len / self.num_latents

    def get_memory_info(self, encoder_len):
        """Return memory usage comparison"""

        dense_cross = self.num_latents * encoder_len * self.num_heads
        self_attn   = self.num_latents * self.num_latents * self.num_heads

        return {
            "cross_attn_scores": dense_cross,
            "self_attn_scores":  self_attn,
            "total_scores":      dense_cross + self_attn,
        }


#######################
#####  UTILITIES  #####
#######################


def create_perceiver_compressor(
    embed_dim,
    num_latents            = 512,
    num_heads              = 12,
    num_kv_heads           = None,
    num_layers             = 2,
    ff_dim                 = None,
    dropout                = 0.0,
    latent_init_std        = 0.02,
    gradient_checkpointing = False,
):
    """Create PerceiverCompressor with given configuration"""

    config = PerceiverConfig(
        embed_dim              = embed_dim,
        num_latents            = num_latents,
        num_heads              = num_heads,
        num_kv_heads           = num_kv_heads,
        num_layers             = num_layers,
        ff_dim                 = ff_dim,
        dropout                = dropout,
        latent_init_std        = latent_init_std,
        gradient_checkpointing = gradient_checkpointing,
    )

    return PerceiverCompressor(config)


def estimate_memory_savings(encoder_len, decoder_len, num_latents, num_heads, dtype_bytes=2):
    """Estimate memory savings from Perceiver compression"""

    original_cross   = decoder_len * encoder_len * num_heads * dtype_bytes
    compression_cost = num_latents * encoder_len * num_heads * dtype_bytes
    perceiver_cross  = decoder_len * num_latents * num_heads * dtype_bytes

    return {
        'original_mb':       original_cross / 1e6,
        'compression_mb':    compression_cost / 1e6,
        'perceiver_mb':      perceiver_cross / 1e6,
        'savings_factor':    original_cross / perceiver_cross,
        'compression_ratio': encoder_len / num_latents,
    }
