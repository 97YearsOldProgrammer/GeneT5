import torch
import torch.nn                 as nn
import torch.utils.checkpoint   as ckpt

from lib.blocks._component      import LayerNorm, FeedForward
from lib.blocks._flash_att      import FlashAttention, FlashAttentionConfig
from lib.blocks._moe            import MoE, MoEConfig


####  DECODER  ####


class DecoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout          = 0.0,
        attn_dropout     = 0.0,
        use_alibi        = True,
        use_moe          = False,
        num_experts      = 8,
        moe_top_k        = 2,
        moe_load_balance = 0.01,
        moe_router_z     = 0.001,
        num_kv_heads     = None,
    ):
        super().__init__()

        self.use_moe = use_moe

        flash_config = FlashAttentionConfig(
            embed_dim    = embed_dim,
            num_heads    = num_heads,
            num_kv_heads = num_kv_heads,
            dropout      = attn_dropout,
            use_alibi    = use_alibi,
        )
        self.self_attn = FlashAttention(config=flash_config)
        self.norm1     = LayerNorm(embed_dim)

        # Feed-forward: MoE or standard
        if use_moe:
            moe_config = MoEConfig(
                embed_dim            = embed_dim,
                ff_dim               = ff_dim,
                num_experts          = num_experts,
                top_k                = moe_top_k,
                dropout              = dropout,
                load_balance_weight  = moe_load_balance,
                router_z_loss_weight = moe_router_z,
            )
            self.ff = MoE(config=moe_config)
        else:
            self.ff = FeedForward(embed_dim, ff_dim, dropout)

        self.norm2   = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):

        # Self-attention
        normed             = self.norm1(hidden_states)
        attn_output, _     = self.self_attn(normed)
        hidden_states      = hidden_states + self.dropout(attn_output)

        # Feed-forward
        normed = self.norm2(hidden_states)

        if self.use_moe:
            ff_output, moe_aux_loss = self.ff(normed)
            hidden_states           = hidden_states + self.dropout(ff_output)
            return hidden_states, moe_aux_loss
        else:
            ff_output     = self.ff(normed)
            hidden_states = hidden_states + self.dropout(ff_output)
            return hidden_states, None


class Decoder(nn.Module):

    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        ff_dim,
        dropout          = 0.0,
        attn_dropout     = 0.0,
        use_alibi        = True,
        use_moe          = True,
        num_experts      = 8,
        moe_top_k        = 2,
        moe_load_balance = 0.01,
        moe_router_z     = 0.001,
        num_kv_heads     = None,
    ):
        super().__init__()

        self.use_moe   = use_moe
        self.use_alibi = use_alibi

        self.layers = nn.ModuleList([
            DecoderBlock(
                embed_dim        = embed_dim,
                num_heads        = num_heads,
                ff_dim           = ff_dim,
                dropout          = dropout,
                attn_dropout     = attn_dropout,
                use_alibi        = use_alibi,
                use_moe          = use_moe,
                num_experts      = num_experts,
                moe_top_k        = moe_top_k,
                moe_load_balance = moe_load_balance,
                moe_router_z     = moe_router_z,
                num_kv_heads     = num_kv_heads,
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

    def forward(self, hidden_states):

        total_moe_loss = 0.0 if self.use_moe else None

        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                hidden_states, moe_aux_loss = ckpt.checkpoint(
                    layer,
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states, moe_aux_loss = layer(hidden_states)

            if self.use_moe and moe_aux_loss is not None:
                total_moe_loss = total_moe_loss + moe_aux_loss

        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states, total_moe_loss
