import math

import torch
import torch.nn                 as nn
import torch.utils.checkpoint   as ckpt

from lib.blocks._component      import LayerNorm, FeedForward, fused_add_rmsnorm

import os as _os
_ATTN_BACKEND = _os.environ.get("GENET5_ATTN_BACKEND", "auto")

def _load_attention():
    """FA4 > FA3 > FA2 > SDPA fallback, or explicit via GENET5_ATTN_BACKEND"""

    if _ATTN_BACKEND == "fa4":
        from lib.blocks._flash_attn4 import FlashAttention, FlashAttentionConfig
        return FlashAttention, FlashAttentionConfig
    if _ATTN_BACKEND == "fa3":
        from lib.blocks._flash_attn3 import FlashAttention, FlashAttentionConfig
        return FlashAttention, FlashAttentionConfig
    if _ATTN_BACKEND == "fa2":
        from lib.blocks._flash_att   import FlashAttention, FlashAttentionConfig
        return FlashAttention, FlashAttentionConfig
    if _ATTN_BACKEND == "sdpa":
        from lib.blocks._sdpa_att    import FlashAttention, FlashAttentionConfig
        return FlashAttention, FlashAttentionConfig

    # auto: try best available
    for mod in ("_flash_attn4", "_flash_attn3", "_flash_att", "_sdpa_att"):
        try:
            m = __import__(f"lib.blocks.{mod}", fromlist=["FlashAttention", "FlashAttentionConfig"])
            return m.FlashAttention, m.FlashAttentionConfig
        except ImportError:
            continue
    raise ImportError("No attention backend available")

FlashAttention, FlashAttentionConfig = _load_attention()


####  ENCODER  ####


class EncoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout        = 0.0,
        attn_dropout   = 0.0,
        num_kv_heads   = None,
        fused_add_norm = False,
        residual_scale = 1.0,
    ):
        super().__init__()

        self.fused_add_norm = fused_add_norm
        self.residual_scale = residual_scale

        flash_config = FlashAttentionConfig(
            embed_dim    = embed_dim,
            num_heads    = num_heads,
            num_kv_heads = num_kv_heads,
            dropout      = attn_dropout,
        )
        self.self_attn = FlashAttention(config=flash_config)
        self.norm1     = LayerNorm(embed_dim)
        self.ff        = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2     = LayerNorm(embed_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, hidden_states, cu_seqlens=None, max_seqlen=None):

        alpha = self.residual_scale

        normed             = self.norm1(hidden_states)
        attn_output, _     = self.self_attn(normed, cu_seqlens, max_seqlen)

        if self.fused_add_norm:
            normed, hidden_states = fused_add_rmsnorm(
                self.dropout(attn_output) * alpha, hidden_states,
                self.norm2.weight, self.norm2.variance_epsilon)
        else:
            hidden_states = hidden_states + self.dropout(attn_output) * alpha
            normed        = self.norm2(hidden_states)

        ff_output     = self.ff(normed)
        hidden_states = hidden_states + self.dropout(ff_output) * alpha
        return hidden_states


class Encoder(nn.Module):

    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        ff_dim,
        dropout        = 0.0,
        attn_dropout   = 0.0,
        num_kv_heads   = None,
        fused_add_norm = False,
        depth_scaling  = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(
                embed_dim      = embed_dim,
                num_heads      = num_heads,
                ff_dim         = ff_dim,
                dropout        = dropout,
                attn_dropout   = attn_dropout,
                num_kv_heads   = num_kv_heads,
                fused_add_norm = fused_add_norm,
                residual_scale = (1.0 / math.sqrt(i + 1)) if depth_scaling else 1.0,
            )
            for i in range(num_layers)
        ])

        self.final_norm = LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)

        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    def forward(self, hidden_states, cu_seqlens=None, max_seqlen=None):

        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                hidden_states = ckpt.checkpoint(
                    layer,
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, cu_seqlens, max_seqlen)

        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
