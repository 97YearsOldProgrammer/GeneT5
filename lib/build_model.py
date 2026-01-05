import torch
import torch.nn as nn
import json

from transformers import AutoModel, AutoTokenizer
from pathlib import Path

from lib.blocks import Encoder, Decoder


def build_gt5(
    dnabert_model_name  = "zhihan1996/DNABERT-2-117M",
    save_dir            = "./checkpoints/genet5_init",
    block_size          = 64,
    num_rand_blocks     = 3,
    decoder_num_layers  = None,
    decoder_num_heads   = None,
    decoder_ff_dim      = None,
    decoder_dropout     = 0.1,
    decoder_use_alibi   = True,
    decoder_use_moe     = False,
    decoder_num_experts = 8,
    decoder_moe_top_k   = 2,
    vocab_size          = 4096,
    tie_weights         = True
):
    """
    Build GeneT5 from DNABERT-2 and save clean checkpoint.
    
    Weight Transfer:
        Encoder Self-Attention  -> COPY from DNABERT-2 (unzip fused Wqkv)
        Decoder Self-Attention  -> COPY from Encoder
        Cross-Attention         -> RANDOM INIT
        Layer Norms             -> COPY
        Embeddings              -> RANDOM INIT
        Output Head             -> TIED or RANDOM INIT
    """
    print("=" * 60)
    print("Building GeneT5 from DNABERT-2")
    print("=" * 60)
    
    # Load Pre-trained Model
    print(f"\n[1] Loading DNABERT-2: {dnabert_model_name}")
    tokenizer      = AutoTokenizer.from_pretrained(dnabert_model_name, trust_remote_code=True)
    original_model = AutoModel.from_pretrained(dnabert_model_name, trust_remote_code=True)
    dna_config     = original_model.config
    
    # Create Symmetry Dimension
    if decoder_num_layers is None:
        decoder_num_layers = dna_config.num_hidden_layers
    if decoder_num_heads is None:
        decoder_num_heads = dna_config.num_attention_heads
    if decoder_ff_dim is None:
        decoder_ff_dim = dna_config.intermediate_size
    
    print(f"    DNABERT-2 Config:")
    print(f"      hidden_size:     {dna_config.hidden_size}")
    print(f"      num_layers:      {dna_config.num_hidden_layers}")
    print(f"      num_heads:       {dna_config.num_attention_heads}")
    print(f"      intermediate:    {dna_config.intermediate_size}")
    print(f"    Decoder Config:")
    print(f"      num_layers:      {decoder_num_layers}")
    print(f"      num_heads:       {decoder_num_heads}")
    print(f"      ff_dim:          {decoder_ff_dim}")
    print(f"      use_alibi:       {decoder_use_alibi}")
    print(f"      use_moe:         {decoder_use_moe}")
    
    # Build Encoder
    print(f"\n[2] Building BigBird Encoder (block_size={block_size})")
    encoder = Encoder(
        num_layers         = dna_config.num_hidden_layers,
        embed_dim          = dna_config.hidden_size,
        num_heads          = dna_config.num_attention_heads,
        ff_dim             = dna_config.intermediate_size,
        dropout            = dna_config.hidden_dropout_prob,
        attn_dropout       = dna_config.attention_probs_dropout_prob,
        use_alibi          = True,
        use_bigbird_sparse = True,
        block_size         = block_size,
        num_random_blocks  = num_rand_blocks
    )
    
    # Transfer Encoder Weights
    print("\n[3] Transferring Encoder Weights from DNABERT-2")
    original_layers = original_model.encoder.layer
    
    for idx, (orig, new) in enumerate(zip(original_layers, encoder.layers)):
        orig_attn = orig.attention.self
        new_attn  = new.self_attn.attention  # Access nested attention module
        
        # === UNZIP FUSED Wqkv ===
        if hasattr(orig_attn, 'Wqkv'):
            # DNABERT-2 uses fused QKV projection
            Wqkv_weight = orig_attn.Wqkv.weight.data  # Shape: (3*hidden, hidden)
            
            # Split into Q, K, V chunks along dimension 0
            q_weight, k_weight, v_weight = Wqkv_weight.chunk(3, dim=0)
            
            # Copy to separate Q, K, V layers
            new_attn.q.weight.data.copy_(q_weight)
            new_attn.k.weight.data.copy_(k_weight)
            new_attn.v.weight.data.copy_(v_weight)
            
            # Handle biases if they exist
            if hasattr(orig_attn.Wqkv, 'bias') and orig_attn.Wqkv.bias is not None:
                Wqkv_bias = orig_attn.Wqkv.bias.data
                q_bias, k_bias, v_bias = Wqkv_bias.chunk(3, dim=0)
                
                # Check if new attention has biases
                if new_attn.q.bias is not None:
                    new_attn.q.bias.data.copy_(q_bias)
                    new_attn.k.bias.data.copy_(k_bias)
                    new_attn.v.bias.data.copy_(v_bias)
        
        else:
            # Fallback: Standard BERT with separate query/key/value
            new_attn.q.weight.data.copy_(orig_attn.query.weight.data)
            new_attn.k.weight.data.copy_(orig_attn.key.weight.data)
            new_attn.v.weight.data.copy_(orig_attn.value.weight.data)
            
            if orig_attn.query.bias is not None and new_attn.q.bias is not None:
                new_attn.q.bias.data.copy_(orig_attn.query.bias.data)
                new_attn.k.bias.data.copy_(orig_attn.key.bias.data)
                new_attn.v.bias.data.copy_(orig_attn.value.bias.data)
        
        # Copy output projection
        new_attn.o.weight.data.copy_(orig.attention.output.dense.weight.data)
        if hasattr(orig.attention.output.dense, 'bias') and orig.attention.output.dense.bias is not None:
            if new_attn.o.bias is not None:
                new_attn.o.bias.data.copy_(orig.attention.output.dense.bias.data)
        
        # Copy layer norm after attention
        new.norm1.weight.data.copy_(orig.attention.output.LayerNorm.weight.data)
        if hasattr(orig.attention.output.LayerNorm, 'bias') and orig.attention.output.LayerNorm.bias is not None:
            if hasattr(new.norm1, 'bias') and new.norm1.bias is not None:
                new.norm1.bias.data.copy_(orig.attention.output.LayerNorm.bias.data)
        
        # Copy FFN weights
        new.ff.wi_0.weight.data.copy_(orig.intermediate.dense.weight.data)
        new.ff.wi_1.weight.data.copy_(orig.intermediate.dense.weight.data)
        new.ff.wo.weight.data.copy_(orig.output.dense.weight.data)
        
        # Copy biases for FFN if they exist
        if hasattr(orig.intermediate.dense, 'bias') and orig.intermediate.dense.bias is not None:
            if hasattr(new.ff.wi_0, 'bias') and new.ff.wi_0.bias is not None:
                new.ff.wi_0.bias.data.copy_(orig.intermediate.dense.bias.data)
                new.ff.wi_1.bias.data.copy_(orig.intermediate.dense.bias.data)
        
        if hasattr(orig.output.dense, 'bias') and orig.output.dense.bias is not None:
            if hasattr(new.ff.wo, 'bias') and new.ff.wo.bias is not None:
                new.ff.wo.bias.data.copy_(orig.output.dense.bias.data)
        
        # Copy layer norm after FFN
        new.norm2.weight.data.copy_(orig.output.LayerNorm.weight.data)
        if hasattr(orig.output.LayerNorm, 'bias') and orig.output.LayerNorm.bias is not None:
            if hasattr(new.norm2, 'bias') and new.norm2.bias is not None:
                new.norm2.bias.data.copy_(orig.output.LayerNorm.bias.data)
    
    # Copy final encoder norm
    encoder.final_norm.weight.data.copy_(original_model.encoder.LayerNorm.weight.data)
    if hasattr(original_model.encoder.LayerNorm, 'bias') and original_model.encoder.LayerNorm.bias is not None:
        if hasattr(encoder.final_norm, 'bias') and encoder.final_norm.bias is not None:
            encoder.final_norm.bias.data.copy_(original_model.encoder.LayerNorm.bias.data)
    
    print("    ✓ Encoder weights copied (Wqkv unzipped successfully)")
    
    # Build Decoder
    print(f"\n[4] Building Decoder (layers={decoder_num_layers}, moe={decoder_use_moe})")
    decoder = Decoder(
        num_layers       = decoder_num_layers,
        embed_dim        = dna_config.hidden_size,
        num_heads        = decoder_num_heads,
        ff_dim           = decoder_ff_dim,
        dropout          = decoder_dropout,
        attn_dropout     = decoder_dropout,
        use_alibi        = decoder_use_alibi,
        use_moe          = decoder_use_moe,
        num_experts      = decoder_num_experts,
        moe_top_k        = decoder_moe_top_k
    )
    
    # Transfer Decoder Self-Attention from Encoder
    print("\n[5] Transferring Decoder Self-Attention from Encoder")
    num_copy = min(decoder_num_layers, len(encoder.layers))
    
    for idx in range(num_copy):
        enc_attn = encoder.layers[idx].self_attn.attention
        dec_attn = decoder.layers[idx].self_attn
        
        dec_attn.q.weight.data.copy_(enc_attn.q.weight.data)
        dec_attn.k.weight.data.copy_(enc_attn.k.weight.data)
        dec_attn.v.weight.data.copy_(enc_attn.v.weight.data)
        dec_attn.o.weight.data.copy_(enc_attn.o.weight.data)
        
        # Copy biases if they exist
        if enc_attn.q.bias is not None and dec_attn.q.bias is not None:
            dec_attn.q.bias.data.copy_(enc_attn.q.bias.data)
            dec_attn.k.bias.data.copy_(enc_attn.k.bias.data)
            dec_attn.v.bias.data.copy_(enc_attn.v.bias.data)
        
        if enc_attn.o.bias is not None and dec_attn.o.bias is not None:
            dec_attn.o.bias.data.copy_(enc_attn.o.bias.data)
        
        decoder.layers[idx].norm1.weight.data.copy_(encoder.layers[idx].norm1.weight.data)
        if hasattr(encoder.layers[idx].norm1, 'bias') and encoder.layers[idx].norm1.bias is not None:
            if hasattr(decoder.layers[idx].norm1, 'bias') and decoder.layers[idx].norm1.bias is not None:
                decoder.layers[idx].norm1.bias.data.copy_(encoder.layers[idx].norm1.bias.data)
    
    print(f"    ✓ Copied {num_copy} layers")
    
    # Random Init Cross-Attention
    print("\n[6] Random Init Cross-Attention")
    for layer in decoder.layers:
        nn.init.xavier_uniform_(layer.cross_attn.q.weight)
        nn.init.xavier_uniform_(layer.cross_attn.k.weight)
        nn.init.xavier_uniform_(layer.cross_attn.v.weight)
        nn.init.xavier_uniform_(layer.cross_attn.o.weight)
        nn.init.ones_(layer.norm2.weight)
        
        # Initialize biases if they exist
        if layer.cross_attn.q.bias is not None:
            nn.init.zeros_(layer.cross_attn.q.bias)
            nn.init.zeros_(layer.cross_attn.k.bias)
            nn.init.zeros_(layer.cross_attn.v.bias)
        if layer.cross_attn.o.bias is not None:
            nn.init.zeros_(layer.cross_attn.o.bias)
        if hasattr(layer.norm2, 'bias') and layer.norm2.bias is not None:
            nn.init.zeros_(layer.norm2.bias)
    
    print("    ✓ Cross-attention initialized")
    
    # Random Init Decoder FFN (or MoE)
    print("\n[7] Random Init Decoder FFN")
    for layer in decoder.layers:
        if decoder_use_moe:
            # MoE has different structure
            if hasattr(layer.ff, 'experts'):
                for expert in layer.ff.experts:
                    if hasattr(expert, 'wi_0'):
                        nn.init.xavier_uniform_(expert.wi_0.weight)
                        nn.init.xavier_uniform_(expert.wi_1.weight)
                        nn.init.xavier_uniform_(expert.wo.weight)
                        if expert.wi_0.bias is not None:
                            nn.init.zeros_(expert.wi_0.bias)
                            nn.init.zeros_(expert.wi_1.bias)
                        if expert.wo.bias is not None:
                            nn.init.zeros_(expert.wo.bias)
            if hasattr(layer.ff, 'router'):
                nn.init.xavier_uniform_(layer.ff.router.weight)
        else:
            if hasattr(layer.ff, 'wi_0'):
                nn.init.xavier_uniform_(layer.ff.wi_0.weight)
                nn.init.xavier_uniform_(layer.ff.wi_1.weight)
                nn.init.xavier_uniform_(layer.ff.wo.weight)
                if layer.ff.wi_0.bias is not None:
                    nn.init.zeros_(layer.ff.wi_0.bias)
                    nn.init.zeros_(layer.ff.wi_1.bias)
                if layer.ff.wo.bias is not None:
                    nn.init.zeros_(layer.ff.wo.bias)
        
        nn.init.ones_(layer.norm3.weight)
        if hasattr(layer.norm3, 'bias') and layer.norm3.bias is not None:
            nn.init.zeros_(layer.norm3.bias)
    
    nn.init.ones_(decoder.final_norm.weight)
    if hasattr(decoder.final_norm, 'bias') and decoder.final_norm.bias is not None:
        nn.init.zeros_(decoder.final_norm.bias)
    
    print("    ✓ FFN initialized")
    
    # Build Embeddings and LM Head
    print("\n[8] Building Embeddings and LM Head")
    decoder_embed = nn.Embedding(vocab_size, dna_config.hidden_size)
    lm_head       = nn.Linear(dna_config.hidden_size, vocab_size, bias=False)
    
    nn.init.normal_(decoder_embed.weight, mean=0.0, std=0.02)
    
    if tie_weights:
        lm_head.weight = decoder_embed.weight
        print("    ✓ Weights tied (embed = lm_head)")
    else:
        nn.init.xavier_uniform_(lm_head.weight)
        print("    ✓ Separate weights initialized")
    
    # Save Checkpoint
    print(f"\n[9] Saving Checkpoint to {save_dir}")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Config
    config = {
        "embed_dim":          dna_config.hidden_size,
        "encoder_num_layers": dna_config.num_hidden_layers,
        "encoder_num_heads":  dna_config.num_attention_heads,
        "encoder_ff_dim":     dna_config.intermediate_size,
        "decoder_num_layers": decoder_num_layers,
        "decoder_num_heads":  decoder_num_heads,
        "decoder_ff_dim":     decoder_ff_dim,
        "decoder_dropout":    decoder_dropout,
        "decoder_use_alibi":  decoder_use_alibi,
        "decoder_use_moe":    decoder_use_moe,
        "decoder_num_experts":decoder_num_experts,
        "decoder_moe_top_k":  decoder_moe_top_k,
        "vocab_size":         vocab_size,
        "tie_weights":        tie_weights,
        "block_size":         block_size,
        "num_rand_blocks":    num_rand_blocks
    }
    
    # Save config.json
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("    ✓ config.json saved")
    
    # Save weights
    checkpoint = {
        "encoder":       encoder.state_dict(),
        "decoder":       decoder.state_dict(),
        "decoder_embed": decoder_embed.state_dict(),
        "lm_head":       lm_head.state_dict()
    }
    torch.save(checkpoint, save_path / "pytorch_model.bin")
    print("    ✓ pytorch_model.bin saved")
    
    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    print("    ✓ tokenizer saved")
    
    # Cleanup
    del original_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Stats
    total_params = sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in decoder.parameters())
    total_params += sum(p.numel() for p in decoder_embed.parameters())
    total_params += sum(p.numel() for p in lm_head.parameters())
    
    print("\n" + "=" * 60)
    print(f"✓ GeneT5 Built Successfully")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Saved to: {save_path}")
    print("=" * 60)
    
    return str(save_path)