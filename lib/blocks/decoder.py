import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lib.blocks._component  import LayerNorm, Attention, FeedForward
from lib.blocks._moe        import MoE, MoEConfig


#################
#### Decoder ####
#################


class DecoderBlock(nn.Module):
    
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        ff_dim, 
        dropout         =0.0, 
        attn_dropout    =0.0, 
        use_alibi       =True,
        use_moe         =False,
        num_experts     =8,
        moe_top_k       =2,
        moe_load_balance=0.01,
        moe_router_z    =0.001,
        use_deepspeed   =False,
        force_backend   =None
    ):
        super().__init__()
        
        self.use_moe = use_moe
        
        self.self_attn = Attention(
            embed_dim           =embed_dim,
            num_heads           =num_heads,
            dropout             =attn_dropout,
            is_decoder          =True,
            is_cross_attention  =False,
            use_alibi           =use_alibi
        )
        self.norm1 = LayerNorm(embed_dim)
        
        self.cross_attn = Attention(
            embed_dim           =embed_dim,
            num_heads           =num_heads,
            dropout             =attn_dropout,
            is_decoder          =True,
            is_cross_attention  =True,
            use_alibi           =False
        )
        self.norm2 = LayerNorm(embed_dim)
        
        if use_moe:
            moe_config = MoEConfig(
                embed_dim               =embed_dim,
                ff_dim                  =ff_dim,
                num_experts             =num_experts,
                top_k                   =moe_top_k,
                dropout                 =dropout,
                load_balance_weight     =moe_load_balance,
                router_z_loss_weight    =moe_router_z,
                use_deepspeed           =use_deepspeed
            )
            self.ff = MoE(config=moe_config, force_backend=force_backend)
        else:
            self.ff = FeedForward(embed_dim, ff_dim, dropout)
        
        self.norm3      = LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None, position_bias=None):
        
        normed = self.norm1(hidden_states)
        attn_output, position_bias = self.self_attn(
            normed,
            attention_mask  =attention_mask,
            position_bias   =position_bias
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        
        normed = self.norm2(hidden_states)
        cross_output, _ = self.cross_attn(
            normed,
            key_value_states    =encoder_hidden_states,
            attention_mask      =encoder_attention_mask
        )
        hidden_states = hidden_states + self.dropout(cross_output)
        
        normed = self.norm3(hidden_states)
        
        if self.use_moe:
            ff_output, moe_aux_loss = self.ff(normed)
            hidden_states           = hidden_states + self.dropout(ff_output)
            return hidden_states, position_bias, moe_aux_loss
        else:
            ff_output       = self.ff(normed)
            hidden_states   = hidden_states + self.dropout(ff_output)
            return hidden_states, position_bias, None


class Decoder(nn.Module):
    
    def __init__(
        self, 
        num_layers, 
        embed_dim, 
        num_heads, 
        ff_dim, 
        dropout         =0.0, 
        attn_dropout    =0.0,
        use_alibi       =True,
        use_moe         =True,
        num_experts     =8,
        moe_top_k       =2,
        moe_load_balance=0.01,
        moe_router_z    =0.001,
        use_deepspeed   =False,
        force_backend   =None
    ):
        super().__init__()
        
        self.use_moe    = use_moe
        self.use_alibi  = use_alibi
        
        self.layers = nn.ModuleList([
            DecoderBlock(
                embed_dim           =embed_dim,
                num_heads           =num_heads,
                ff_dim              =ff_dim,
                dropout             =dropout,
                attn_dropout        =attn_dropout,
                use_alibi           =use_alibi,
                use_moe             =use_moe,
                num_experts         =num_experts,
                moe_top_k           =moe_top_k,
                moe_load_balance    =moe_load_balance,
                moe_router_z        =moe_router_z,
                use_deepspeed       =use_deepspeed,
                force_backend       =force_backend
            )
            for i in range(num_layers)
        ])
        
        self.final_norm = LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None):

        position_bias   = None
        total_moe_loss  = 0.0 if self.use_moe else None
        
        for layer in self.layers:
            result = layer(
                hidden_states,
                encoder_hidden_states   =encoder_hidden_states,
                attention_mask          =attention_mask,
                encoder_attention_mask  =encoder_attention_mask,
                position_bias           =position_bias
            )
            
            if self.use_moe:
                hidden_states, position_bias, moe_aux_loss = result
                if moe_aux_loss is not None:
                    total_moe_loss = total_moe_loss + moe_aux_loss
            else:
                hidden_states, position_bias, _ = result
        
        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states, total_moe_loss