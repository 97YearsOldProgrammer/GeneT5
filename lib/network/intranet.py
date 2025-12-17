import torch
import torch.nn as nn
import torch.nn.functional as F

from network import _blocks as util


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
        embed_dim                   = 1536,
        num_encoder_layers          = 12,
        num_decoder_layers          = 12,
        num_heads                   = 12,
        ff_dim                      = 1536*4,
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
        self.encoder = util.Encoder(
            num_layers=num_encoder_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            attn_dropout=attn_dropout
        )
        
        # Decoder
        self.decoder = util.Decoder(
            num_layers=num_decoder_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            attn_dropout=attn_dropout
        )
        
        # Output projection
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
                
            elif isinstance(module, util.LayerNorm):
                nn.init.ones_(module.weight)
    
    def _create_encoder_attention_mask(self, input_ids):
        """Mask padding tokens"""
        mask = (input_ids == self.pad_token_id).float()
        mask = mask[:, None, None, :]
        mask = mask * torch.finfo(mask.dtype).min
        return mask
    
    def _create_decoder_attention_mask(self, decoder_input_ids):
        """Mask Self Attention + Mask padding tokens"""
        B, L    = decoder_input_ids.shape
        device  = decoder_input_ids.device
        
        # Upper triangular for self attention
        causal_mask = torch.triu(
            torch.ones(L, L, device=device),
            diagonal=1
        )
        causal_mask = causal_mask * torch.finfo(causal_mask.dtype).min
        causal_mask = causal_mask[None, None, :, :]     # (1, 1, L, L)
        
        # Padding mask
        padding_mask    = (decoder_input_ids == self.pad_token_id).float()
        padding_mask    = padding_mask[:, None, None, :]   # (B, 1, 1, L)
        padding_mask    = padding_mask * torch.finfo(padding_mask.dtype).min
        combined_mask   = causal_mask + padding_mask
        
        return combined_mask
    
    def get_input_embeddings(self):
        return self.shared_embedding
    
    def set_input_embeddings(self, embeddings):
        self.shared_embedding   = embeddings
        self.lm_head.weight     = embeddings.weight
    
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

        # Embed decoder inputs
        hidden_states = self.shared_embedding(decoder_input_ids)
        
        # Masked Self Attention
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
    
    def forward(self, input_ids, decoder_input_ids, attention_mask=None, decoder_attention_mask=None, labels=None):

        # Mask padding token in encoder
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
    def generate(self, input_ids, max_length=128, temperature=1.0, top_k=None, top_p=None, do_sample=False):

        B       = input_ids.shape[0]
        device  = input_ids.device
        
        # Encode input
        encoder_attention_mask  = self._create_encoder_attention_mask(input_ids)
        encoder_hidden_states   = self.encode(input_ids, encoder_attention_mask)
        
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