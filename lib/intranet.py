# -----------------------------------------------------------------------------
# This file implements components of the T5 architecture:
#     Raffel et al., "Exploring the Limits of Transfer Learning with a
#     Unified Text-to-Text Transformer", JMLR 2020.
#
# Original reference code:
#   - Google T5 (Apache-2.0):
#       https://github.com/google-research/text-to-text-transfer-transformer
#   - HuggingFace Transformers T5 (Apache-2.0):
#       https://github.com/huggingface/transformers
#
# Portions of this implementation (relative position bias, RMSNorm, GEGLU)
# are adapted from the above works. 
#
# Apache License 2.0 allows modification and redistribution with attribution.
# -----------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


################################
#### Relative Position Bias ####
################################


class RelativePositionBias(nn.Module):
    
    def __init__(self, num_heads , num_buckets=32, max_distance=128, bidirectional=True):
        super().__init__()
        
        self.num_heads      = num_heads
        self.num_buckets    = num_buckets
        self.max_distance   = max_distance
        # Encoder is bidirectional
        self.bidirectional  = bidirectional
        
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
    
    """
        Compute bucket indices for relative positions
        
        T5 uses a mix of:
            - Exact positions for small distances
            - Log-spaced bins for larger distances
    """
    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance, bidirectional=True):

        relative_buckets = 0
        
        if bidirectional:
            num_buckets     //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, 
                torch.zeros_like(relative_position)
            )
        
        # Half buckets for exact positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # Half buckets for log-spaced bins
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        
        return relative_buckets
    
    def forward(self, query_length, key_length, device):

        context_position    = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position     = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position   = memory_position - context_position
        
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        
        values = self.relative_attention_bias(relative_position_bucket)
        # (1, num_heads, query_len, key_len)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        
        return values


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
    """
    T5-style multi-head attention with relative position bias
    
    Key differences from standard attention:
        - No bias in linear projections
        - Relative position bias instead of absolute positional encoding
        - Supports both self-attention and cross-attention
    """
    
    def __init__(
        self,
        embed_dim           : int,
        num_heads           : int,
        dropout             : float = 0.0,
        is_decoder          : bool = False,
        is_cross_attention  : bool = False,
        has_relative_bias   : bool = True,
        use_flash           : bool = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim          = embed_dim
        self.num_heads          = num_heads
        self.head_dim           = embed_dim // num_heads
        self.is_decoder         = is_decoder
        self.is_cross_attention = is_cross_attention
        self.has_relative_bias  = has_relative_bias
        self.use_flash          = use_flash
        
        # No Bias in Attention Projection
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Relative position bias (only for self-attention)
        if has_relative_bias and not is_cross_attention:
            self.relative_bias = RelativePositionBias(
                num_heads=num_heads,
                # Encoder is bidirectional
                bidirectional=not is_decoder
            )
        else:
            self.relative_bias = None
    
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
        
        # Compute relative position bias if needed
        if self.relative_bias is not None and position_bias is None:
            position_bias = self.relative_bias(L, L_kv, hidden_states.device)
        
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
            scores = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, L, L_kv)
            
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
    """Gated MLP based on GeGLU"""
    
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


#################
#### Encoder ####
#################


class EncoderBlock(nn.Module):
    """
    Encoder block with pre-norm architecture
    
    Structure:
    - LayerNorm -> Self-Attention -> Residual
    - LayerNorm -> FeedForward -> Residual
    """
    
    def __init__(self, embed_dim ,num_heads, ff_dim, dropout=0.0, attn_dropout=0.0, has_relative_bias=True):
        super().__init__()
        
        self.self_attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            is_decoder=False,
            is_cross_attention=False,
            has_relative_bias=has_relative_bias
        )
        
        self.norm1      = LayerNorm(embed_dim)
        self.ff         = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2      = LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        
        # Pre-norm
        normed = self.norm1(hidden_states)
        
        # Self-attention
        attn_output, position_bias = self.self_attn(
            normed,
            attention_mask=attention_mask,
            position_bias=position_bias
        )
        
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # Pre-norm
        normed = self.norm2(hidden_states)
        
        # Feed forward
        ff_output       = self.ff(normed)
        hidden_states   = hidden_states + self.dropout(ff_output)
        
        return hidden_states, position_bias


class Encoder(nn.Module):
    """Encoder stack with shared relative position bias"""
    
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        
        # Only first layer computes relative position bias
        self.layers = nn.ModuleList([
            EncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                has_relative_bias=(i == 0)
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
                attention_mask=attention_mask,
                position_bias=position_bias
            )
        
        hidden_states = self.final_norm(hidden_states)
        # (B, L, D)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


#################
#### Decoder ####
#################


class DecoderBlock(nn.Module):
    """
    Decoder block with pre-norm architecture
    
    Structure:
    - LayerNorm -> Causal Self-Attention -> Residual
    - LayerNorm -> Cross-Attention -> Residual
    - LayerNorm -> FeedForward -> Residual
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0, attn_dropout=0.0, has_relative_bias=True):
        super().__init__()
        
        # Self-attention
        self.self_attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            is_decoder=True,
            is_cross_attention=False,
            has_relative_bias=has_relative_bias
        )
        self.norm1 = LayerNorm(embed_dim)
        
        # Cross-attention to encoder
        self.cross_attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            is_decoder=True,
            is_cross_attention=True,
            # No relative bias for cross attention
            has_relative_bias=False
        )
        self.norm2 = LayerNorm(embed_dim)
        
        # Feed-forward
        self.ff     = FeedForward(embed_dim, ff_dim, dropout)
        self.norm3  = LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None, position_bias=None):
        
        # Self-attention
        normed = self.norm1(hidden_states)
        attn_output, position_bias = self.self_attn(
            normed,
            attention_mask=attention_mask,
            position_bias=position_bias
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # Cross-attention
        normed = self.norm2(hidden_states)
        cross_output, _ = self.cross_attn(
            normed,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask
        )
        hidden_states = hidden_states + self.dropout(cross_output)
        
        # Feed-forward
        normed = self.norm3(hidden_states)
        ff_output = self.ff(normed)
        hidden_states = hidden_states + self.dropout(ff_output)
        
        return hidden_states, position_bias


class Decoder(nn.Module):
    """Decoder stack with shared relative position bias"""
    
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                has_relative_bias=(i == 0)
            )
            for i in range(num_layers)
        ])
        
        self.final_norm = LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None):
        """
        Args:
            hidden_states           : (B, L_dec, D) embedded decoder input
            encoder_hidden_states   : (B, L_enc, D) encoder output
            attention_mask          : (B, 1, L_dec, L_dec) causal + padding mask
            encoder_attention_mask  : (B, 1, 1, L_enc) encoder padding mask
        
        Returns:
            hidden_states           : (B, L_dec, D) decoded output
        """
        position_bias = None
        
        for layer in self.layers:
            hidden_states, position_bias = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                position_bias=position_bias
            )
        
        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


##########################
#### Main Model Class ####
##########################


class IntraNet(nn.Module):
    """
    T5-style encoder-decoder transformer for sequence-to-sequence tasks
    
    Architecture:
    1. Shared token embedding (encoder and decoder)
    2. T5 Encoder with relative position bias
    3. T5 Decoder with causal self-attention and cross-attention
    4. Output projection head (tied with embedding)
    
    Input: Integer token IDs
    Output: Logits over vocabulary
    """
    
    def __init__(
        self,
        vocab_size                  = 4096,
        embed_dim                   = 768,
        num_encoder_layers          = 12,
        num_decoder_layers          = 12,
        num_heads                   = 12,
        ff_dim                      = 768*4,
        dropout                     = 0.1,
        attn_dropout                = 0.0,
        pad_token_id                = 0,
        eos_token_id                = 1,
        decoder_start_token_id      = 0
    ):
        super().__init__()
        
        # MLP ratio = 4
        self.ff_dim                 = 4*embed_dim,
        
        self.vocab_size             = vocab_size
        self.embed_dim              = embed_dim
        self.pad_token_id           = pad_token_id
        self.eos_token_id           = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        
        # Shared embedding for encoder and decoder
        self.shared_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Embedding scale (T5 uses this)
        self.embed_scale = embed_dim ** 0.5
        
        # Encoder
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            attn_dropout=attn_dropout
        )
        
        # Decoder
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            attn_dropout=attn_dropout
        )
        
        # Output projection (tied with embedding weights)
        self.lm_head        = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):

        for module in self.modules():
            
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
            elif isinstance(module, LayerNorm):
                nn.init.ones_(module.weight)
    
    def _create_encoder_attention_mask(self, input_ids):
        """
        Create attention mask from input ids (mask padding tokens)
        
        Returns:
            mask: (B, 1, 1, L) with -inf for padding positions
        """
        mask = (input_ids == self.pad_token_id).float()
        mask = mask[:, None, None, :]
        mask = mask * torch.finfo(mask.dtype).min
        return mask
    
    def _create_decoder_attention_mask(self, decoder_input_ids):
        """
        Create causal + padding mask for decoder
        
        Returns:
            mask: (B, 1, L, L) with -inf for masked positions
        """
        B, L = decoder_input_ids.shape
        device = decoder_input_ids.device
        
        # Causal mask (upper triangular)
        causal_mask = torch.triu(
            torch.ones(L, L, device=device),
            diagonal=1
        )
        causal_mask = causal_mask * torch.finfo(causal_mask.dtype).min
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, L, L)
        
        # Padding mask
        padding_mask = (decoder_input_ids == self.pad_token_id).float()
        padding_mask = padding_mask[:, None, None, :]  # (B, 1, 1, L)
        padding_mask = padding_mask * torch.finfo(padding_mask.dtype).min
        
        # Combine: broadcast addition
        combined_mask = causal_mask + padding_mask
        
        return combined_mask
    
    def get_input_embeddings(self):
        return self.shared_embedding
    
    def set_input_embeddings(self, embeddings):
        self.shared_embedding = embeddings
        self.lm_head.weight = embeddings.weight
    
    def encode(self, input_ids, attention_mask=None):
        """
        Encode input sequence
        
        Args:
            input_ids       : (B, L) input token ids
            attention_mask  : (B, 1, 1, L) optional attention mask
        
        Returns:
            encoder_output  : (B, L, D) encoded representations
        """
        # Embed inputs
        hidden_states = self.shared_embedding(input_ids)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self._create_encoder_attention_mask(input_ids)
        
        # Encode
        encoder_output = self.encoder(hidden_states, attention_mask)
        
        return encoder_output
    
    def decode(self, decoder_input_ids, encoder_hidden_states, decoder_attention_mask=None, encoder_attention_mask=None):
        """
        Decode with encoder outputs
        
        Args:
            decoder_input_ids       : (B, L_dec) decoder token ids
            encoder_hidden_states   : (B, L_enc, D) encoder output
            decoder_attention_mask  : (B, 1, L_dec, L_dec) causal mask
            encoder_attention_mask  : (B, 1, 1, L_enc) encoder mask
        
        Returns:
            decoder_output          : (B, L_dec, D) decoded representations
        """
        # Embed decoder inputs
        hidden_states = self.shared_embedding(decoder_input_ids)
        
        # Create causal mask if not provided
        if decoder_attention_mask is None:
            decoder_attention_mask = self._create_decoder_attention_mask(decoder_input_ids)
        
        # Decode
        decoder_output = self.decoder(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask
        )
        
        return decoder_output
    
    def forward(
        self,
        input_ids               : torch.Tensor,
        decoder_input_ids       : torch.Tensor,
        attention_mask          : Optional[torch.Tensor] = None,
        decoder_attention_mask  : Optional[torch.Tensor] = None,
        labels                  : Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass through encoder-decoder
        
        Args:
            input_ids               : (B, L_enc) encoder input token ids
            decoder_input_ids       : (B, L_dec) decoder input token ids
            attention_mask          : (B, 1, 1, L_enc) encoder attention mask
            decoder_attention_mask  : (B, 1, L_dec, L_dec) decoder attention mask
            labels                  : (B, L_dec) target token ids for loss computation
        
        Returns:
            dict with:
                logits  : (B, L_dec, vocab_size) output logits
                loss    : scalar loss (if labels provided)
        """
        # Create encoder attention mask
        if attention_mask is None:
            attention_mask = self._create_encoder_attention_mask(input_ids)
        
        # Encode
        encoder_hidden_states = self.encode(input_ids, attention_mask)
        
        # Decode
        decoder_hidden_states = self.decode(
            decoder_input_ids,
            encoder_hidden_states,
            decoder_attention_mask,
            attention_mask
        )
        
        # Project to vocabulary
        logits = self.lm_head(decoder_hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fn(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'encoder_hidden_states': encoder_hidden_states
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids       : torch.Tensor,
        max_length      : int = 128,
        temperature     : float = 1.0,
        top_k           : Optional[int] = None,
        top_p           : Optional[float] = None,
        do_sample       : bool = False
    ) -> torch.Tensor:
        """
        Autoregressive generation
        
        Args:
            input_ids   : (B, L_enc) encoder input token ids
            max_length  : maximum generation length
            temperature : sampling temperature
            top_k       : top-k sampling
            top_p       : nucleus sampling threshold
            do_sample   : whether to sample or use greedy decoding
        
        Returns:
            generated   : (B, L_gen) generated token ids
        """
        B = input_ids.shape[0]
        device = input_ids.device
        
        # Encode input
        encoder_attention_mask = self._create_encoder_attention_mask(input_ids)
        encoder_hidden_states = self.encode(input_ids, encoder_attention_mask)
        
        # Initialize decoder with start token
        decoder_input_ids = torch.full(
            (B, 1), 
            self.decoder_start_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # Generate tokens autoregressively
        for _ in range(max_length - 1):
            # Decode
            decoder_output = self.decode(
                decoder_input_ids,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
            
            # Get logits for last position
            logits = self.lm_head(decoder_output[:, -1:, :])  # (B, 1, vocab_size)
            logits = logits[:, 0, :]  # (B, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Sample or greedy
            if do_sample:
                # Top-k filtering
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == self.eos_token_id).all():
                break
        
        return decoder_input_ids


##################
#### Variants ####
##################


def intranet_t5_small(vocab_size: int = 32000, max_seq_length: int = 4096):
    """T5-Small configuration (~60M params)"""
    return IntraNet(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        embed_dim=512,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1
    )


def intranet_t5_base(vocab_size: int = 32000, max_seq_length: int = 4096):
    """T5-Base configuration (~220M params)"""
    return IntraNet(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        embed_dim=768,
        num_encoder_layers=12,
        num_decoder_layers=12,
        num_heads=12,
        ff_dim=3072,
        dropout=0.1
    )


def intranet_t5_large(vocab_size: int = 32000, max_seq_length: int = 4096):
    """T5-Large configuration (~770M params)"""
    return IntraNet(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        embed_dim=1024,
        num_encoder_layers=24,
        num_decoder_layers=24,
        num_heads=16,
        ff_dim=4096,
        dropout=0.1
    )


def intranet_t5_xl(vocab_size: int = 32000, max_seq_length: int = 4096):
    """T5-XL configuration (~3B params)"""
    return IntraNet(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        embed_dim=2048,
        num_encoder_layers=24,
        num_decoder_layers=24,
        num_heads=32,
        ff_dim=5120,
        dropout=0.1
    )


##################
#### Auxiliary ####
##################


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the model with random input"""
    print("=" * 60)
    print("Testing IntraNet T5")
    print("=" * 60)
    
    model = intranet_t5_base(vocab_size=32000, max_seq_length=4096)
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # Test forward pass
    input_ids = torch.randint(0, 32000, (2, 128))
    decoder_input_ids = torch.randint(0, 32000, (2, 64))
    labels = torch.randint(0, 32000, (2, 64))
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Decoder input shape: {decoder_input_ids.shape}")
    
    output = model(input_ids, decoder_input_ids, labels=labels)
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    
    # Test generation
    print("\nTesting generation...")
    generated = model.generate(input_ids, max_length=32)
    print(f"Generated shape: {generated.shape}")
    
    # Test different model sizes
    print("\n" + "=" * 60)
    print("Model Size Comparison")
    print("=" * 60)
    
    for name, model_fn in [
        ("T5-Small", intranet_t5_small),
        ("T5-Base", intranet_t5_base),
        ("T5-Large", intranet_t5_large),
    ]:
        model = model_fn()
        params = count_parameters(model)
        print(f"{name}: {params:,} parameters ({params/1e6:.1f}M)")
    
    print("\nAll tests passed!")
    return model


if __name__ == "__main__":
    test_model()