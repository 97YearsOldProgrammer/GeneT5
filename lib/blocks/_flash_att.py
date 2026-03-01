import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from flash_attn import flash_attn_func


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
    use_alibi:    bool  = True
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
    """Full bidirectional attention using flash_attn with GQA and ALiBi"""

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

        if config.use_alibi:
            slopes = self._compute_alibi_slopes(config.num_heads).float()
            self.register_buffer('alibi_slopes', slopes, persistent=False)
        else:
            self.alibi_slopes = None

    @staticmethod
    def _compute_alibi_slopes(num_heads):
        """Compute ALiBi slopes for positional bias"""

        def get_slopes(n):
            if n == 1:
                return torch.tensor([1.0])
            base = 2 ** (-2 ** -(math.log2(n) - 3))
            return torch.tensor([base ** i for i in range(1, n + 1)])

        if math.log2(num_heads).is_integer():
            return get_slopes(num_heads)
        else:
            closest_power = 2 ** math.floor(math.log2(num_heads))
            return torch.cat([
                get_slopes(closest_power),
                get_slopes(2 * closest_power)[0::2][:num_heads - closest_power]
            ])

    def forward(self, hidden_states):
        """Bidirectional flash attention"""

        B, L, D = hidden_states.shape

        q = self.q(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)

        alibi  = self.alibi_slopes.float() if self.alibi_slopes is not None else None
        drop_p = self.dropout.p if self.training else 0.0

        out = flash_attn_func(
            q, k, v,
            dropout_p     = drop_p,
            softmax_scale = self.softmax_scale,
            causal        = False,
            alibi_slopes  = alibi,
        )

        out = out.reshape(B, L, self.embed_dim)
        out = self.o(out)
        out = self.dropout(out)

        return out, None
