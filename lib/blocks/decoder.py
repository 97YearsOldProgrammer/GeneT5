import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lib.blocks._component import LayerNorm, FeedForward
from lib.blocks._spareatt  import SparseAttention, SparseAttentionConfig
from lib.blocks._gqatt     import GQAttention, GQAttentionConfig
from lib.blocks._moe       import MoE, MoEConfig


#################
#### Decoder ####
#################


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
        block_size       = 64,
        window_size      = 256,
    ):
        super().__init__()

        self.use_moe = use_moe

        if num_kv_heads is None:
            num_kv_heads = max(1, num_heads // 4)

        # Self-attention: SparseAttention (weight-compatible with standard attention)
        sparse_config = SparseAttentionConfig(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            block_size  = block_size,
            window_size = window_size,
            dropout     = attn_dropout,
            use_alibi   = use_alibi,
        )
        self.self_attn = SparseAttention(config=sparse_config, is_causal=True)
        self.norm1     = LayerNorm(embed_dim)
        
        # Cross-attention: GQAttention
        gqa_config = GQAttentionConfig(
            embed_dim     = embed_dim,
            num_heads     = num_heads,
            num_kv_heads  = num_kv_heads,
            dropout       = attn_dropout,
            use_alibi     = False,
            is_cross_attn = True,
        )
        self.cross_attn = GQAttention(config=gqa_config, is_causal=False)
        self.norm2      = LayerNorm(embed_dim)
        
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
        
        self.norm3   = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        hidden_states, 
        encoder_hidden_states, 
        attention_mask         = None, 
        encoder_attention_mask = None, 
        position_bias          = None
    ):
        # Self-attention
        normed = self.norm1(hidden_states)
        attn_output, _ = self.self_attn(normed, attention_mask)
        hidden_states  = hidden_states + self.dropout(attn_output)
        
        # Cross-attention
        normed = self.norm2(hidden_states)
        cross_output, _ = self.cross_attn(
            normed,
            key_value_states = encoder_hidden_states,
            attention_mask   = encoder_attention_mask
        )
        hidden_states = hidden_states + self.dropout(cross_output)
        
        # Feed-forward
        normed = self.norm3(hidden_states)
        
        if self.use_moe:
            ff_output, moe_aux_loss = self.ff(normed)
            hidden_states           = hidden_states + self.dropout(ff_output)
            return hidden_states, None, moe_aux_loss
        else:
            ff_output     = self.ff(normed)
            hidden_states = hidden_states + self.dropout(ff_output)
            return hidden_states, None, None


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
        block_size       = 64,
        window_size      = 256,
    ):
        super().__init__()

        self.use_moe   = use_moe
        self.use_alibi = use_alibi

        if num_kv_heads is None:
            num_kv_heads = max(1, num_heads // 4)

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
                block_size       = block_size,
                window_size      = window_size,
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(
        self, 
        hidden_states, 
        encoder_hidden_states, 
        attention_mask         = None, 
        encoder_attention_mask = None
    ):
        total_moe_loss = 0.0 if self.use_moe else None
        
        for layer in self.layers:
            hidden_states, _, moe_aux_loss = layer(
                hidden_states,
                encoder_hidden_states  = encoder_hidden_states,
                attention_mask         = attention_mask,
                encoder_attention_mask = encoder_attention_mask,
            )
            
            if self.use_moe and moe_aux_loss is not None:
                total_moe_loss = total_moe_loss + moe_aux_loss
        
        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states, total_moe_loss
    
    def get_kv_cache_info(self, batch_size, seq_len, encoder_seq_len):
        """Return KV cache memory usage info in bytes"""
        
        if len(self.layers) == 0:
            return {}
        
        layer      = self.layers[0]
        num_layers = len(self.layers)
        
        # SparseAttention: standard KV cache
        num_heads = layer.self_attn.num_heads
        head_dim  = layer.self_attn.head_dim
        self_kv   = batch_size * seq_len * num_heads * head_dim * 2 * 2  # K+V, float16
        
        # GQAttention: reduced KV cache
        cross_kv = layer.cross_attn.get_kv_cache_size(batch_size, encoder_seq_len)
        
        return {
            "self_attn_per_layer":  self_kv,
            "cross_attn_per_layer": cross_kv,
            "total_self":           self_kv * num_layers,
            "total_cross":          cross_kv * num_layers,
            "total_kv_cache":       (self_kv + cross_kv) * num_layers,
        }