import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from flash_attn import flash_attn_func, flash_attn_varlen_func


#####################
####   CONFIG    ####
#####################


@dataclass
class FlashAttentionConfig:
    embed_dim:    int   = 768
    num_heads:    int   = 12
    num_kv_heads: int   = None
    head_dim:     int   = None
    dropout:      float = 0.0
    window_size:  tuple = (-1, -1)

    def __post_init__(self):

        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            self.head_dim = self.embed_dim // self.num_heads


#####################
####  ATTENTION  ####
#####################


class FlashAttention(nn.Module):
    """Full bidirectional attention using flash_attn with GQA"""

    def __init__(self, config):

        super().__init__()

        self.config        = config
        self.embed_dim     = config.embed_dim
        self.num_heads     = config.num_heads
        self.num_kv_heads  = config.num_kv_heads
        self.head_dim      = config.head_dim
        self.kv_dim        = config.head_dim * config.num_kv_heads
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        assert config.num_heads % config.num_kv_heads == 0

        self.q       = nn.Linear(config.embed_dim, config.num_heads * self.head_dim, bias=False)
        self.k       = nn.Linear(config.embed_dim, self.kv_dim, bias=False)
        self.v       = nn.Linear(config.embed_dim, self.kv_dim, bias=False)
        self.o       = nn.Linear(config.num_heads * self.head_dim, config.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, cu_seqlens=None, max_seqlen=None):
        """Bidirectional flash attention with optional varlen packed mode"""

        B, L, D = hidden_states.shape

        q = self.q(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)

        drop_p = self.dropout.p if self.training else 0.0

        if cu_seqlens is not None:
            q   = q.reshape(-1, self.num_heads, self.head_dim)
            k   = k.reshape(-1, self.num_kv_heads, self.head_dim)
            v   = v.reshape(-1, self.num_kv_heads, self.head_dim)
            out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q  = cu_seqlens,
                cu_seqlens_k  = cu_seqlens,
                max_seqlen_q  = max_seqlen,
                max_seqlen_k  = max_seqlen,
                dropout_p     = drop_p,
                softmax_scale = self.softmax_scale,
                causal        = False,
            )
            out = out.reshape(B, L, self.embed_dim)
        else:
            out = flash_attn_func(
                q, k, v,
                dropout_p     = drop_p,
                softmax_scale = self.softmax_scale,
                causal        = False,
            )
            out = out.reshape(B, L, self.embed_dim)

        out = self.o(out)
        out = self.dropout(out)

        return out, None
