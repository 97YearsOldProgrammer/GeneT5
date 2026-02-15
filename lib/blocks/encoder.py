import torch
import torch.nn                 as nn
import torch.utils.checkpoint   as ckpt

from lib.blocks._component  import LayerNorm, FeedForward
from lib.blocks._flash_att  import FlashAttention, FlashAttentionConfig


####  ENCODER  ####


class EncoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout      = 0.0,
        attn_dropout = 0.0,
        use_alibi    = True,
    ):
        super().__init__()

        flash_config = FlashAttentionConfig(
            embed_dim = embed_dim,
            num_heads = num_heads,
            dropout   = attn_dropout,
            use_alibi = use_alibi
        )
        self.self_attn = FlashAttention(
            config    = flash_config,
            is_causal = False
        )

        self.norm1   = LayerNorm(embed_dim)
        self.ff      = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2   = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):

        normed             = self.norm1(hidden_states)
        attn_output, _     = self.self_attn(normed, attention_mask)
        hidden_states      = hidden_states + self.dropout(attn_output)

        normed             = self.norm2(hidden_states)
        ff_output          = self.ff(normed)
        hidden_states      = hidden_states + self.dropout(ff_output)

        return hidden_states


class Encoder(nn.Module):

    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        ff_dim,
        dropout      = 0.0,
        attn_dropout = 0.0,
        use_alibi    = True,
    ):
        super().__init__()

        self.use_alibi = use_alibi

        self.layers = nn.ModuleList([
            EncoderBlock(
                embed_dim    = embed_dim,
                num_heads    = num_heads,
                ff_dim       = ff_dim,
                dropout      = dropout,
                attn_dropout = attn_dropout,
                use_alibi    = use_alibi,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)

        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None):

        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                hidden_states = ckpt.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states