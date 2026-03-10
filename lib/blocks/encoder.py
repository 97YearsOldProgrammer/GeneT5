import torch
import torch.nn                 as nn
import torch.utils.checkpoint   as ckpt

from lib.blocks._component      import LayerNorm, FeedForward, fused_add_rmsnorm
from lib.blocks._moe            import MoE, MoEConfig

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
        dropout          = 0.0,
        attn_dropout     = 0.0,
        use_moe          = False,
        num_experts      = 16,
        moe_top_k        = 4,
        moe_load_balance = 0.01,
        moe_router_z     = 0.001,
        num_kv_heads     = None,
        fused_add_norm   = False,
    ):
        super().__init__()

        self.use_moe        = use_moe
        self.fused_add_norm = fused_add_norm

        flash_config = FlashAttentionConfig(
            embed_dim    = embed_dim,
            num_heads    = num_heads,
            num_kv_heads = num_kv_heads,
            dropout      = attn_dropout,
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

    def forward(self, hidden_states, cu_seqlens=None, max_seqlen=None):

        # Self-attention
        normed             = self.norm1(hidden_states)
        attn_output, _     = self.self_attn(normed, cu_seqlens, max_seqlen)

        # Residual + norm2: fused or standard
        if self.fused_add_norm:
            normed, hidden_states = fused_add_rmsnorm(
                self.dropout(attn_output), hidden_states,
                self.norm2.weight, self.norm2.variance_epsilon)
        else:
            hidden_states = hidden_states + self.dropout(attn_output)
            normed        = self.norm2(hidden_states)

        # Feed-forward
        if self.use_moe:
            ff_output, moe_aux_loss = self.ff(normed)
            hidden_states           = hidden_states + self.dropout(ff_output)
            return hidden_states, moe_aux_loss
        else:
            ff_output     = self.ff(normed)
            hidden_states = hidden_states + self.dropout(ff_output)
            return hidden_states, None


class Encoder(nn.Module):

    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        ff_dim,
        dense_ff_dim     = None,
        dropout          = 0.0,
        attn_dropout     = 0.0,
        use_moe          = True,
        moe_layers       = None,
        num_experts      = 16,
        moe_top_k        = 4,
        moe_load_balance = 0.01,
        moe_router_z     = 0.001,
        num_kv_heads     = None,
        fused_add_norm   = False,
    ):
        super().__init__()

        if dense_ff_dim is None:
            dense_ff_dim = ff_dim

        # Per-layer MoE assignment
        if moe_layers is not None:
            moe_set = set(moe_layers)
        elif use_moe:
            moe_set = set(range(num_layers))
        else:
            moe_set = set()

        self.use_moe = len(moe_set) > 0

        self.layers = nn.ModuleList([
            EncoderBlock(
                embed_dim        = embed_dim,
                num_heads        = num_heads,
                ff_dim           = ff_dim if i in moe_set else dense_ff_dim,
                dropout          = dropout,
                attn_dropout     = attn_dropout,
                use_moe          = i in moe_set,
                num_experts      = num_experts,
                moe_top_k        = moe_top_k,
                moe_load_balance = moe_load_balance,
                moe_router_z     = moe_router_z,
                num_kv_heads     = num_kv_heads,
                fused_add_norm   = fused_add_norm,
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

        total_moe_loss = 0.0 if self.use_moe else None

        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                hidden_states, moe_aux_loss = ckpt.checkpoint(
                    layer,
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    use_reentrant=False,
                )
            else:
                hidden_states, moe_aux_loss = layer(hidden_states, cu_seqlens, max_seqlen)

            if self.use_moe and moe_aux_loss is not None:
                total_moe_loss = total_moe_loss + moe_aux_loss

        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states, total_moe_loss
