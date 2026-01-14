import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lib.blocks._component import LayerNorm, Attention, FeedForward
from lib.blocks._spareatt  import SparseAttention, SparseAttentionConfig


#################
#### Encoder ####
#################


class EncoderBlock(nn.Module):
    
    def __init__(
        self, 
        embed_dim,
        num_heads, 
        ff_dim, 
        dropout            =0.0, 
        attn_dropout       =0.0, 
        use_alibi          =False,
        use_bigbird_sparse =False,
        block_size         =64,
        window_size        =256,
        num_global_tokens  =64,
        num_random_blocks  =3
    ):
        super().__init__()
        
        self.use_bigbird_sparse = use_bigbird_sparse
        
        if use_bigbird_sparse:
            sparse_config = SparseAttentionConfig(
                embed_dim         =embed_dim,
                num_heads         =num_heads,
                block_size        =block_size,
                window_size       =window_size,
                num_global_tokens =num_global_tokens,
                num_random_blocks =num_random_blocks,
                dropout           =attn_dropout,
                use_alibi         =use_alibi
            )
            self.self_attn = SparseAttention(
                config    =sparse_config,
                is_causal =False
            )
        else:
            self.self_attn = Attention(
                embed_dim          =embed_dim,
                num_heads          =num_heads,
                dropout            =attn_dropout,
                is_decoder         =False,
                is_cross_attention =False,
                use_alibi          =use_alibi
            )
        
        self.norm1   = LayerNorm(embed_dim)
        self.ff      = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2   = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        normed = self.norm1(hidden_states)
        
        if self.use_bigbird_sparse:
            attn_output, _ = self.self_attn(normed, attention_mask)
            position_bias  = None
        else:
            attn_output, position_bias = self.self_attn(
                normed,
                attention_mask =attention_mask,
                position_bias  =position_bias
            )
        
        hidden_states = hidden_states + self.dropout(attn_output)
        
        normed        = self.norm2(hidden_states)
        ff_output     = self.ff(normed)
        hidden_states = hidden_states + self.dropout(ff_output)
        
        return hidden_states, position_bias


class Encoder(nn.Module):
    
    def __init__(
        self, 
        num_layers, 
        embed_dim, 
        num_heads, 
        ff_dim, 
        dropout            =0.0, 
        attn_dropout       =0.0,
        use_alibi          =False,
        use_bigbird_sparse =False,
        block_size         =64,
        window_size        =256,
        num_global_tokens  =64,
        num_random_blocks  =3
    ):
        super().__init__()
        
        self.use_bigbird_sparse = use_bigbird_sparse
        self.use_alibi          = use_alibi
        
        self.layers = nn.ModuleList([
            EncoderBlock(
                embed_dim          =embed_dim,
                num_heads          =num_heads,
                ff_dim             =ff_dim,
                dropout            =dropout,
                attn_dropout       =attn_dropout,
                use_alibi          =use_alibi,
                use_bigbird_sparse =use_bigbird_sparse,
                block_size         =block_size,
                window_size        =window_size,
                num_global_tokens  =num_global_tokens,
                num_random_blocks  =num_random_blocks
            )
            for i in range(num_layers)
        ])
        
        self.final_norm = LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        position_bias = None
        
        for layer in self.layers:
            hidden_states, position_bias = layer(
                hidden_states,
                attention_mask =attention_mask,
                position_bias  =position_bias if not self.use_bigbird_sparse else None
            )
        
        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states