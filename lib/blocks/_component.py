import torch
import torch.nn as nn
import torch.nn.functional as F
import math


####################
#### Layer Norm ####
####################


class LayerNorm(nn.Module):
    """RMSNorm without mean centering"""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps    = eps
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x        = x * torch.rsqrt(variance + self.eps)
        
        return self.weight * x


###################
#### Attention ####
###################


class Attention(nn.Module):
    """Generalized Attention Module"""
    
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout            = 0.0,
        is_decoder         = False,
        is_cross_attention = False,
        use_alibi          = False,
        use_flash          = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim          = embed_dim
        self.num_heads          = num_heads
        self.head_dim           = embed_dim // num_heads
        self.is_decoder         = is_decoder
        self.is_cross_attention = is_cross_attention
        self.use_alibi          = use_alibi
        self.use_flash          = use_flash
        
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        if not is_cross_attention and use_alibi:
            self.position_bias_module = ALiBi(num_heads=num_heads)
        else:
            self.position_bias_module = None
    
    def forward(self, hidden_states, key_value_states=None, attention_mask=None, position_bias=None):
        
        B, L, D = hidden_states.shape
        
        q = self.q(hidden_states)
        
        if self.is_cross_attention and key_value_states is not None:
            k    = self.k(key_value_states)
            v    = self.v(key_value_states)
            L_kv = key_value_states.shape[1]
        else:
            k    = self.k(hidden_states)
            v    = self.v(hidden_states)
            L_kv = L
        
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.position_bias_module is not None and position_bias is None:
            position_bias = self.position_bias_module(L, L_kv, hidden_states.device)
        
        use_flash_attn = (
            self.use_flash 
            and hasattr(F, 'scaled_dot_product_attention')
            and position_bias is None 
            and attention_mask is None
        )
        
        if use_flash_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout.p if self.training else 0.0,
                is_causal = self.is_decoder and not self.is_cross_attention
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if position_bias is not None:
                scores = scores + position_bias
            
            if attention_mask is not None:
                scores = scores + attention_mask
            
            attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
            attn_weights = self.dropout(attn_weights)
            out          = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o(out)
        
        return out, position_bias


######################
#### Feed Forward ####
######################


class FeedForward(nn.Module):
    """Gated MLP based on GeGLU"""
    
    def __init__(self, embed_dim, ff_dim, dropout=0.0, activation='gelu_new'):
        super().__init__()
        
        self.wi_0    = nn.Linear(embed_dim, ff_dim, bias=False)
        self.wi_1    = nn.Linear(embed_dim, ff_dim, bias=False)
        self.wo      = nn.Linear(ff_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu_new':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()
    
    def forward(self, x):
        
        hidden = self.act(self.wi_0(x)) * self.wi_1(x)
        hidden = self.dropout(hidden)
        output = self.wo(hidden)
        
        return output


###############
#### ALiBi ####
###############


class ALiBi(nn.Module):
    """
    Attention with Linear Biases - Memory Efficient Version
    
    Reference: "Train Short, Test Long" (Press et al., 2021)
    """
    
    def __init__(self, num_heads, max_seq_len=8192):
        super().__init__()
        
        self.num_heads = num_heads
        
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
    
    @staticmethod
    def _get_slopes(num_heads):
        
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return torch.tensor([start * (ratio ** i) for i in range(n)])
        
        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = torch.cat([
                get_slopes_power_of_2(closest_power_of_2),
                get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]
            ])
            return slopes
    
    def _build_bias(self, query_len, key_len, device):
        
        context_position  = torch.arange(key_len, device=device)[None, :]
        memory_position   = torch.arange(query_len, device=device)[:, None]
        relative_position = torch.abs(memory_position - context_position)
        
        slopes     = self.slopes.view(1, -1, 1, 1)
        alibi_bias = -slopes * relative_position.unsqueeze(0).unsqueeze(0).float()
        
        return alibi_bias
    
    def forward(self, query_length, key_length, device):
        
        return self._build_bias(query_length, key_length, device)