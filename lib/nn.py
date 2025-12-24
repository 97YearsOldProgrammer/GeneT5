import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer
from pathlib import Path

from lib.blocks import Encoder, Decoder


class GeneT5(nn.Module):
    """
    Encoder: DNABERT-2 weights converted to BigBird sparse attention
    Decoder: Self-attention copied from encoder, cross-attention random init
    """
    
    def __init__(
        self,
        encoder,
        encoder_embed_dim,
        decoder_num_layers      = 6,
        decoder_num_heads       = 8,
        decoder_ff_dim          = 2048,
        decoder_dropout         = 0.1,
        decoder_attn_dropout    = 0.1,
        decoder_use_alibi       = True,
        decoder_use_moe         = False,
        decoder_num_experts     = 8,
        decoder_moe_top_k       = 2,
        decoder_moe_load_balance= 0.01,
        vocab_size              = 4096,
        freeze_encoder          = True
    ):
        super().__init__()
        
        self.encoder_embed_dim = encoder_embed_dim
        self.freeze_encoder    = freeze_encoder
        
        # Encoder (DNABERT-2 converted to BigBird)
        self.encoder = encoder
        
        # Decoder
        self.decoder = Decoder(
            num_layers      = decoder_num_layers,
            embed_dim       = encoder_embed_dim,
            num_heads       = decoder_num_heads,
            ff_dim          = decoder_ff_dim,
            dropout         = decoder_dropout,
            attn_dropout    = decoder_attn_dropout,
            use_alibi       = decoder_use_alibi,
            use_moe         = decoder_use_moe,
            num_experts     = decoder_num_experts,
            moe_top_k       = decoder_moe_top_k,
            moe_load_balance= decoder_moe_load_balance
        )
        
        # Embeddings
        self.decoder_embed         = nn.Embedding(vocab_size, encoder_embed_dim)
        self.decoder_embed_dropout = nn.Dropout(decoder_dropout)
        
        # Output head
        self.lm_head = nn.Linear(encoder_embed_dim, vocab_size, bias=False)
        
        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freeze_encoder = True
    
    def _unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
    
    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attention_mask  = None,
        decoder_attention_mask  = None,
        encoder_hidden_states   = None,
        return_encoder_hidden   = False
    ):
        # Encode
        if encoder_hidden_states is None:
            encoder_hidden_states = self.encoder(
                encoder_input_ids,
                attention_mask = encoder_attention_mask
            )
        
        # Embed decoder inputs
        decoder_embeds = self.decoder_embed(decoder_input_ids)
        decoder_embeds = self.decoder_embed_dropout(decoder_embeds)
        
        # Decode
        decoder_output, moe_loss = self.decoder(
            hidden_states          = decoder_embeds,
            encoder_hidden_states  = encoder_hidden_states,
            attention_mask         = decoder_attention_mask,
            encoder_attention_mask = encoder_attention_mask
        )
        
        # Project to vocab
        logits = self.lm_head(decoder_output)
        
        outputs = {"logits": logits}
        
        if moe_loss is not None:
            outputs["moe_loss"] = moe_loss
        
        if return_encoder_hidden:
            outputs["encoder_hidden_states"] = encoder_hidden_states
        
        return outputs
    
    def get_param_stats(self):
        enc_train  = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        enc_frozen = sum(p.numel() for p in self.encoder.parameters() if not p.requires_grad)
        dec_train  = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        emb_train  = sum(p.numel() for p in self.decoder_embed.parameters() if p.requires_grad)
        head_train = sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad)
        
        return {
            "encoder_trainable": enc_train,
            "encoder_frozen":    enc_frozen,
            "decoder_trainable": dec_train,
            "embed_trainable":   emb_train,
            "head_trainable":    head_train,
            "total_trainable":   enc_train + dec_train + emb_train + head_train,
            "total_frozen":      enc_frozen
        }
    
    def save_weights(self, save_dir, save_encoder=True, save_decoder=True):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {"config": {"encoder_embed_dim": self.encoder_embed_dim}}
        
        if save_encoder:
            checkpoint["encoder_state_dict"] = self.encoder.state_dict()
        
        if save_decoder:
            checkpoint["decoder_state_dict"]       = self.decoder.state_dict()
            checkpoint["decoder_embed_state_dict"] = self.decoder_embed.state_dict()
            checkpoint["lm_head_state_dict"]       = self.lm_head.state_dict()
        
        save_file = save_path / "checkpoint.pt"
        torch.save(checkpoint, save_file)
        
        return str(save_file)
    
    def load_weights(self, checkpoint_path, load_encoder=True, load_decoder=True, strict=True):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if load_encoder and "encoder_state_dict" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=strict)
        
        if load_decoder:
            if "decoder_state_dict" in checkpoint:
                self.decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=strict)
            if "decoder_embed_state_dict" in checkpoint:
                self.decoder_embed.load_state_dict(checkpoint["decoder_embed_state_dict"], strict=strict)
            if "lm_head_state_dict" in checkpoint:
                self.lm_head.load_state_dict(checkpoint["lm_head_state_dict"], strict=strict)
        
        return checkpoint


def create_gt5(
    dnabert_model_name  = "zhihan1996/DNABERT-2-117M",
    block_size          = 64,
    num_rand_blocks     = 3,
    decoder_num_layers  = 6,
    decoder_num_heads   = 8,
    decoder_ff_dim      = 2048,
    decoder_dropout     = 0.1,
    decoder_use_alibi   = True,
    decoder_use_moe     = False,
    vocab_size          = 4096,
    freeze_encoder      = True,
    device              = None
):
    """
    Factory function to create DPG model
    
    Weight Transfer Strategy:
        - Encoder Self-Attention:  COPY from DNABERT-2
        - Decoder Self-Attention:  COPY from Encoder (reuse attention logic)
        - Cross-Attention:         RANDOM INIT (new connection)
        - Layer Norms:             COPY (keeps math stable)
        - Output Head:             RANDOM INIT
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Creating DPG Model")
    print("=" * 60)
    
    # Load DNABERT-2
    print(f"\n[1] Loading DNABERT-2: {dnabert_model_name}")
    tokenizer      = AutoTokenizer.from_pretrained(dnabert_model_name, trust_remote_code=True)
    original_model = AutoModel.from_pretrained(dnabert_model_name, trust_remote_code=True)
    config         = original_model.config
    
    print(f"    Hidden:       {config.hidden_size}")
    print(f"    Layers:       {config.num_hidden_layers}")
    print(f"    Heads:        {config.num_attention_heads}")
    print(f"    Intermediate: {config.intermediate_size}")
    
    # Create BigBird encoder
    print(f"\n[2] Creating BigBird Encoder (block_size={block_size})")
    encoder = Encoder(
        num_layers         = config.num_hidden_layers,
        embed_dim          = config.hidden_size,
        num_heads          = config.num_attention_heads,
        ff_dim             = config.intermediate_size,
        dropout            = config.hidden_dropout_prob,
        attn_dropout       = config.attention_probs_dropout_prob,
        use_alibi          = True,
        use_bigbird_sparse = True,
        block_size         = block_size,
        num_random_blocks  = num_rand_blocks
    )
    
    # Transfer encoder weights from DNABERT-2
    print("\n[3] Transferring Encoder Weights")
    original_layers = original_model.bert.encoder.layer
    
    for idx, (orig, new) in enumerate(zip(original_layers, encoder.layers)):
        orig_attn = orig.attention.self
        new_attn  = new.self_attn
        
        # Self-attention Q, K, V, O
        new_attn.q.weight.data.copy_(orig_attn.query.weight.data)
        new_attn.k.weight.data.copy_(orig_attn.key.weight.data)
        new_attn.v.weight.data.copy_(orig_attn.value.weight.data)
        new_attn.o.weight.data.copy_(orig.attention.output.dense.weight.data)
        
        # Layer norm after attention
        new.norm1.weight.data.copy_(orig.attention.output.LayerNorm.weight.data)
        
        # FFN
        new.ff.wi_0.weight.data.copy_(orig.intermediate.dense.weight.data)
        new.ff.wi_1.weight.data.copy_(orig.intermediate.dense.weight.data)
        new.ff.wo.weight.data.copy_(orig.output.dense.weight.data)
        
        # Layer norm after FFN
        new.norm2.weight.data.copy_(orig.output.LayerNorm.weight.data)
    
    # Final norm
    encoder.final_norm.weight.data.copy_(original_model.bert.encoder.LayerNorm.weight.data)
    print("    ✓ Encoder self-attention copied")
    print("    ✓ Encoder layer norms copied")
    print("    ✓ Encoder FFN copied")
    
    # Create decoder
    print(f"\n[4] Creating Decoder (layers={decoder_num_layers})")
    decoder = Decoder(
        num_layers       = decoder_num_layers,
        embed_dim        = config.hidden_size,
        num_heads        = decoder_num_heads,
        ff_dim           = decoder_ff_dim,
        dropout          = decoder_dropout,
        attn_dropout     = decoder_dropout,
        use_alibi        = decoder_use_alibi,
        use_moe          = decoder_use_moe
    )
    
    # Transfer decoder self-attention from encoder (reuse attention logic)
    print("\n[5] Transferring Decoder Self-Attention from Encoder")
    num_layers_to_copy = min(decoder_num_layers, len(encoder.layers))
    
    for idx in range(num_layers_to_copy):
        enc_attn = encoder.layers[idx].self_attn
        dec_attn = decoder.layers[idx].self_attn
        
        # Copy Q, K, V, O
        dec_attn.q.weight.data.copy_(enc_attn.q.weight.data)
        dec_attn.k.weight.data.copy_(enc_attn.k.weight.data)
        dec_attn.v.weight.data.copy_(enc_attn.v.weight.data)
        dec_attn.o.weight.data.copy_(enc_attn.o.weight.data)
        
        # Copy layer norm
        decoder.layers[idx].norm1.weight.data.copy_(encoder.layers[idx].norm1.weight.data)
    
    print(f"    ✓ Decoder self-attention copied from encoder ({num_layers_to_copy} layers)")
    
    # Random init cross-attention (already random from Decoder init)
    print("\n[6] Cross-Attention: Random Initialized")
    for idx, layer in enumerate(decoder.layers):
        nn.init.xavier_uniform_(layer.cross_attn.q.weight)
        nn.init.xavier_uniform_(layer.cross_attn.k.weight)
        nn.init.xavier_uniform_(layer.cross_attn.v.weight)
        nn.init.xavier_uniform_(layer.cross_attn.o.weight)
        nn.init.ones_(layer.norm2.weight)
    print("    ✓ Cross-attention randomly initialized")
    
    # Random init FFN in decoder (or copy if same dimensions)
    print("\n[7] Decoder FFN: Random Initialized")
    for layer in decoder.layers:
        if hasattr(layer.ff, 'wi_0'):
            nn.init.xavier_uniform_(layer.ff.wi_0.weight)
            nn.init.xavier_uniform_(layer.ff.wi_1.weight)
            nn.init.xavier_uniform_(layer.ff.wo.weight)
        nn.init.ones_(layer.norm3.weight)
    print("    ✓ Decoder FFN randomly initialized")
    
    # Final decoder norm
    nn.init.ones_(decoder.final_norm.weight)
    
    # Create full model
    print("\n[8] Assembling DPG Model")
    model = GeneT5(
        encoder                  = encoder,
        encoder_embed_dim        = config.hidden_size,
        decoder_num_layers       = decoder_num_layers,
        decoder_num_heads        = decoder_num_heads,
        decoder_ff_dim           = decoder_ff_dim,
        decoder_dropout          = decoder_dropout,
        decoder_use_alibi        = decoder_use_alibi,
        decoder_use_moe          = decoder_use_moe,
        vocab_size               = vocab_size,
        freeze_encoder           = freeze_encoder
    )
    
    # Replace decoder with our initialized one
    model.decoder = decoder
    
    # Random init embeddings and output head
    nn.init.normal_(model.decoder_embed.weight, mean=0.0, std=0.02)
    nn.init.xavier_uniform_(model.lm_head.weight)
    print("    ✓ Embeddings randomly initialized")
    print("    ✓ Output head randomly initialized")
    
    model = model.to(device)
    
    # Print stats
    print(f"\n[9] Model on device: {device}")
    stats = model.get_param_stats()
    print(f"    Encoder trainable: {stats['encoder_trainable']:,}")
    print(f"    Encoder frozen:    {stats['encoder_frozen']:,}")
    print(f"    Decoder trainable: {stats['decoder_trainable']:,}")
    print(f"    Total trainable:   {stats['total_trainable']:,}")
    
    # Cleanup
    del original_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("✓ DPG Model Created")
    print("=" * 60)
    
    return model, tokenizer