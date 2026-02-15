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

    def __post_init__(self):

        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            self.head_dim = self.embed_dim // self.num_heads


#####################
####  ATTENTION  ####
#####################


class FlashAttention(nn.Module):
    """Full attention using flash_attn with GQA and ALiBi"""

    def __init__(self, config, is_causal=False):

        super().__init__()

        self.config        = config
        self.is_causal     = is_causal
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

    def forward(self, hidden_states, attention_mask=None):
        """Forward pass with full flash attention"""

        B, L, D = hidden_states.shape

        q = self.q(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)

        alibi = self.alibi_slopes.float() if self.alibi_slopes is not None else None

        out = flash_attn_func(
            q, k, v,
            dropout_p     = self.dropout.p if self.training else 0.0,
            softmax_scale = self.softmax_scale,
            causal        = self.is_causal,
            window_size   = (-1, -1),
            alibi_slopes  = alibi,
        )

        out = out.reshape(B, L, self.embed_dim)
        out = self.o(out)
        out = self.dropout(out)

        return out, None

    def get_kv_cache_size(self, batch_size, seq_len):
        """Return KV cache size in bytes for float16"""

        return batch_size * seq_len * self.num_kv_heads * self.head_dim * 2 * 2
