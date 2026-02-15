import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from flash_attn import flash_attn_func


@dataclass
class CrossAttentionConfig:
    embed_dim:    int   = 768
    num_heads:    int   = 12
    num_kv_heads: int   = None
    head_dim:     int   = None
    dropout:      float = 0.0

    def __post_init__(self):

        if self.num_kv_heads is None:
            self.num_kv_heads = max(1, self.num_heads // 4)
        if self.head_dim is None:
            self.head_dim = self.embed_dim // self.num_heads


class CrossAttention(nn.Module):
    """Full cross-attention using flash_attn with GQA"""

    def __init__(self, config):

        super().__init__()

        self.config       = config
        self.embed_dim    = config.embed_dim
        self.num_heads    = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim     = config.head_dim
        self.kv_dim       = config.head_dim * config.num_kv_heads
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        assert config.num_heads % config.num_kv_heads == 0

        self.q       = nn.Linear(config.embed_dim, config.num_heads * self.head_dim, bias=False)
        self.k       = nn.Linear(config.embed_dim, self.kv_dim, bias=False)
        self.v       = nn.Linear(config.embed_dim, self.kv_dim, bias=False)
        self.o       = nn.Linear(config.num_heads * self.head_dim, config.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None):
        """Cross-attention: Q from decoder, K/V from encoder"""

        B, L_dec, D = hidden_states.shape
        L_enc       = encoder_hidden_states.shape[1]

        q = self.q(hidden_states).view(B, L_dec, self.num_heads, self.head_dim)
        k = self.k(encoder_hidden_states).view(B, L_enc, self.num_kv_heads, self.head_dim)
        v = self.v(encoder_hidden_states).view(B, L_enc, self.num_kv_heads, self.head_dim)

        out = flash_attn_func(
            q, k, v,
            dropout_p     = self.dropout.p if self.training else 0.0,
            softmax_scale = self.softmax_scale,
            causal        = False,
            window_size   = (-1, -1),
        )

        out = out.reshape(B, L_dec, self.embed_dim)
        out = self.o(out)
        out = self.dropout(out)

        return out, None
