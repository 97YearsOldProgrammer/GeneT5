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
        variance    = x.pow(2).mean(-1, keepdim=True)
        x           = x * torch.rsqrt(variance + self.eps)

        return self.weight * x


###################
#### Attention ####
###################


class Attention(nn.Module):
    """ Class for Generalized Attention """
    
    def __init__(
        self,
        embed_dim           : int,
        num_heads           : int,
        dropout             : float = 0.0,
        is_decoder          : bool = False,
        is_cross_attention  : bool = False,
        use_alibi           : bool = False,
        use_flash           : bool = True
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
        
        # No Bias in Attention Projection
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Position bias - only ALiBi for non-cross attention
        if not is_cross_attention and use_alibi:
            self.position_bias_module = ALiBi(num_heads=num_heads)
        else:
            self.position_bias_module = None
    
    def forward(self, hidden_states, key_value_states=None, attention_mask=None, position_bias=None):
        """
        Args:
            hidden_states   : (B, L, D) query states
            key_value_states: (B, L_kv, D) for cross-attention, None for self-attention
            attention_mask  : (B, 1, L, L_kv) additive mask
            position_bias   : precomputed position bias for reuse
        
        Returns:
            output          : (B, L, D)
            position_bias   : for reuse in subsequent layers
        """
        
        B, L, D = hidden_states.shape
        
        # Query from hidden_states
        q = self.q(hidden_states)
        
        # k&v from cross-attn
        if self.is_cross_attention and key_value_states is not None:
            k       = self.k(key_value_states)
            v       = self.v(key_value_states)
            L_kv    = key_value_states.shape[1]
        # k&v from self-attn
        else:
            k       = self.k(hidden_states)
            v       = self.v(hidden_states)
            L_kv    = L
        
        # Reshape: (B, L, D) -> (B, num_heads, L, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute position bias
        if self.position_bias_module is not None and position_bias is None:
            position_bias = self.position_bias_module(L, L_kv, hidden_states.device)
        
        # Use Flash Attention if available
        use_flash_attn = (
            self.use_flash 
            and hasattr(F, 'scaled_dot_product_attention')
            and position_bias is None 
            and attention_mask is None
        )
        
        if use_flash_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=self.is_decoder and not self.is_cross_attention
            )
        else:
            # Compute Attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, L, L_kv)
            
            # Apply relative position bias
            if position_bias is not None:
                scores = scores + position_bias
            
            # Apply attention mask
            if attention_mask is not None:
                scores = scores + attention_mask
            
            # Softmax and dropout
            attn_weights    = F.softmax(scores.float(), dim=-1).type_as(scores)
            attn_weights    = self.dropout(attn_weights)
            out             = torch.matmul(attn_weights, v)
        
        # Reshape back: (B, num_heads, L, head_dim) -> (B, L, D)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o(out)
        
        return out, position_bias


######################
#### Feed Forward ####
######################


class FeedForward(nn.Module):
    """ Gated MLP based on GeGLU """
    
    def __init__(self, embed_dim, ff_dim, dropout=0.0, activation='gelu_new'):
        super().__init__()
        
        # Gated linear unit style
        self.wi_0       = nn.Linear(embed_dim, ff_dim, bias=False)  # Gate projection
        self.wi_1       = nn.Linear(embed_dim, ff_dim, bias=False)  # Up projection
        self.wo         = nn.Linear(ff_dim, embed_dim, bias=False)  # Down projection
        self.dropout    = nn.Dropout(dropout)
        
        if activation == 'gelu_new':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()
    
    def forward(self, x):

        # Gated activation: act(gate) * up
        hidden = self.act(self.wi_0(x)) * self.wi_1(x)
        hidden = self.dropout(hidden)
        output = self.wo(hidden)
        return output


###############
#### ALiBi ####
###############


class ALiBi(nn.Module):
    """
    Attention with Linear Biases
    
    Reference: "Train Short, Test Long" (Press et al., 2021)
    """
    
    def __init__(self, num_heads, max_seq_len=8192):
        super().__init__()
        
        self.num_heads      = num_heads
        self.max_seq_len    = max_seq_len
        
        # Compute slopes for each head (geometric sequence)
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        
        # Precompute bias for max_seq_len
        alibi_bias = self._build_alibi_bias(max_seq_len, slopes)
        self.register_buffer('alibi_bias', alibi_bias)
    
    @staticmethod
    def _get_slopes(num_heads):

        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return torch.tensor([start * (ratio ** i) for i in range(n)])
        
        # Handle non-power-of-2 num_heads
        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            # Closest power of 2
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = torch.cat([
                get_slopes_power_of_2(closest_power_of_2),
                get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]
            ])
            return slopes
    
    @staticmethod
    def _build_alibi_bias(seq_len, slopes):
        """ Build ALiBi bias matrix: -m * |i - j| """
        
        # Distance matrix |i - j|
        context_position    = torch.arange(seq_len)[:, None]
        memory_position     = torch.arange(seq_len)[None, :]
        relative_position   = torch.abs(memory_position - context_position)  # (seq_len, seq_len)
        
        # Apply head-specific slopes: -m * |i - j|
        slopes              = slopes[:, None, None]
        relative_position   = relative_position[None, :, :].float()
        
        alibi_bias = -slopes * relative_position    # (num_heads, seq_len, seq_len)
        alibi_bias = alibi_bias.unsqueeze(0)        # (1, num_heads, seq_len, seq_len)
        
        return alibi_bias
    
    def forward(self, query_length, key_length, device):

        # Use cached bias if within max_seq_len
        if query_length <= self.max_seq_len and key_length <= self.max_seq_len:
            bias = self.alibi_bias[:, :, :query_length, :key_length].to(device)
        else:
            # Recompute for longer sequences
            slopes = self.slopes.to(device)
            bias = self._build_alibi_bias(max(query_length, key_length), slopes)
            bias = bias[:, :, :query_length, :key_length].to(device)
        
        return bias