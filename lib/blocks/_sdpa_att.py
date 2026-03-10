import math
from dataclasses import dataclass

import torch
import torch.nn              as nn
import torch.nn.functional   as F


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


def _build_varlen_mask(cu_seqlens, total_len, device):
    """Block-diagonal mask from cu_seqlens for packed sequences"""

    num_seqs = cu_seqlens.shape[0] - 1
    seq_ids  = torch.zeros(total_len, dtype=torch.int32, device=device)
    for i in range(num_seqs):
        seq_ids[cu_seqlens[i]:cu_seqlens[i+1]] = i
    return seq_ids.unsqueeze(0) == seq_ids.unsqueeze(1)


class FlashAttention(nn.Module):
    """Full bidirectional attention using PyTorch SDPA with GQA

    Drop-in replacement for flash_attn-based FlashAttention.
    Works on any GPU architecture (Ampere, Hopper, Blackwell).
    """

    def __init__(self, config):

        super().__init__()

        self.config        = config
        self.embed_dim     = config.embed_dim
        self.num_heads     = config.num_heads
        self.num_kv_heads  = config.num_kv_heads
        self.head_dim      = config.head_dim
        self.kv_dim        = config.head_dim * config.num_kv_heads
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        self.gqa_ratio     = config.num_heads // config.num_kv_heads

        assert config.num_heads % config.num_kv_heads == 0

        self.q       = nn.Linear(config.embed_dim, config.num_heads * self.head_dim, bias=False)
        self.k       = nn.Linear(config.embed_dim, self.kv_dim, bias=False)
        self.v       = nn.Linear(config.embed_dim, self.kv_dim, bias=False)
        self.o       = nn.Linear(config.num_heads * self.head_dim, config.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, cu_seqlens=None, max_seqlen=None):
        """Bidirectional attention with optional varlen packed mode"""

        B, L, D = hidden_states.shape

        q = self.q(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)

        # GQA: expand KV heads to match Q heads
        if self.gqa_ratio > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.gqa_ratio, -1)
            k = k.reshape(B, L, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.gqa_ratio, -1)
            v = v.reshape(B, L, self.num_heads, self.head_dim)

        # SDPA expects [B, num_heads, L, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        drop_p = self.dropout.p if self.training else 0.0

        if cu_seqlens is not None:
            mask = _build_varlen_mask(cu_seqlens, L, hidden_states.device)
            out  = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask     = mask,
                dropout_p     = drop_p,
                scale         = self.softmax_scale,
                is_causal     = False,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p     = drop_p,
                scale         = self.softmax_scale,
                is_causal     = False,
            )

        out = out.transpose(1, 2).reshape(B, L, self.embed_dim)
        out = self.o(out)
        out = self.dropout(out)

        return out, None
