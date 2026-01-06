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
    decoder_num_experts = 1,
    decoder_moe_top_k   = 1,
    vocab_size          = None,
    tie_weights         = True,
    new_tokens_list     = None
):
    """
    Build GeneT5 from DNABERT-2 and save clean checkpoint.
    
    Weight Transfer:
        Encoder Self-Attention  -> COPY from DNABERT-2
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
    
    # Expand Tokenizer
    if new_tokens_list:
        print(f"\n[1.5] Expanding Tokenizer")
        print(f"      Original vocab size: {len(tokenizer)}")
        num_added = tokenizer.add_tokens(new_tokens_list)
        print(f"      Tokens added: {num_added}")
        print(f"      New vocab size: {len(tokenizer)}")
    
    # Auto-detect vocab size from tokenizer if not specified
    if vocab_size is None:
        vocab_size = len(tokenizer)
        print(f"      Using vocab_size: {vocab_size}")
    
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
    
    # DNABERT-2's AutoModel returns BertModel directly (no .bert wrapper)
    # Could be original_model.encoder or original_model.bert.encoder depending on version
    if hasattr(original_model, 'bert'):
        bert = original_model.bert
    else:
        bert = original_model
    
    original_layers = bert.encoder.layer
    
    # Debug: print layer 0 structure
    layer0 = original_layers[0]
    print(f"    DNABERT-2 Layer Structure:")
    print(f"      Attention: {'Wqkv (fused)' if hasattr(layer0.attention.self, 'Wqkv') else 'Q/K/V (separate)'}")
    print(f"      FFN: {'mlp' if hasattr(layer0, 'mlp') else 'intermediate/output'}")
    if hasattr(layer0, 'mlp'):
        mlp = layer0.mlp
        print(f"      MLP attrs: {[n for n,_ in mlp.named_children()]}")
        if hasattr(mlp, 'gated_layers'):
            print(f"      gated_layers shape: {mlp.gated_layers.weight.shape}")
        if hasattr(mlp, 'wo'):
            print(f"      wo shape: {mlp.wo.weight.shape}")
        if hasattr(mlp, 'fc1'):
            print(f"      fc1 shape: {mlp.fc1.weight.shape}")
        if hasattr(mlp, 'fc2'):
            print(f"      fc2 shape: {mlp.fc2.weight.shape}")
    
    for idx, (orig, new) in enumerate(zip(original_layers, encoder.layers)):
        orig_attn = orig.attention.self
        
        # Your SparseAttention is a wrapper - actual Q/K/V are in .attention
        if hasattr(new.self_attn, 'attention'):
            new_attn = new.self_attn.attention  # TritonSparseAttention or FlexAttentionSparse
        else:
            new_attn = new.self_attn
        
        # DNABERT-2 uses Flash Attention with fused Wqkv projection
        # BertUnpadSelfAttention stores Q,K,V in a single fused tensor: Wqkv
        # Shape: (3 * hidden_size, hidden_size) - Q, K, V stacked along dim 0
        if hasattr(orig_attn, 'Wqkv'):
            # Fused QKV projection - split into separate Q, K, V
            Wqkv = orig_attn.Wqkv.weight.data  # (3*H, H)
            hidden_size = Wqkv.shape[1]
            q_weight, k_weight, v_weight = Wqkv.chunk(3, dim=0)
            
            new_attn.q.weight.data.copy_(q_weight)
            new_attn.k.weight.data.copy_(k_weight)
            new_attn.v.weight.data.copy_(v_weight)
            
            # Handle bias if present
            if orig_attn.Wqkv.bias is not None:
                q_bias, k_bias, v_bias = orig_attn.Wqkv.bias.data.chunk(3, dim=0)
                if hasattr(new_attn.q, 'bias') and new_attn.q.bias is not None:
                    new_attn.q.bias.data.copy_(q_bias)
                    new_attn.k.bias.data.copy_(k_bias)
                    new_attn.v.bias.data.copy_(v_bias)
        else:
            # Standard BERT with separate Q, K, V (fallback for other models)
            new_attn.q.weight.data.copy_(orig_attn.query.weight.data)
            new_attn.k.weight.data.copy_(orig_attn.key.weight.data)
            new_attn.v.weight.data.copy_(orig_attn.value.weight.data)
        
        # Output projection - same structure in both Flash and standard
        new_attn.o.weight.data.copy_(orig.attention.output.dense.weight.data)
        
        # Layer norm after attention
        new.norm1.weight.data.copy_(orig.attention.output.LayerNorm.weight.data)
        
        # FFN - DNABERT-2 uses mlp with potentially fused gated projections
        # Standard BERT: intermediate.dense -> output.dense
        # DNABERT-2 Flash: mlp.fc1 (or Wi) -> mlp.fc2 (or Wo), possibly gated
        if hasattr(orig, 'mlp'):
            mlp = orig.mlp
            
            # DNABERT-2 uses gated_layers (fused gate+up) + wo structure
            if hasattr(mlp, 'gated_layers') and hasattr(mlp, 'wo'):
                # gated_layers is fused: (2 * ff_dim, hidden_size) -> split into gate & up
                gated_weight = mlp.gated_layers.weight.data  # (2*ff_dim, hidden)
                gate_weight, up_weight = gated_weight.chunk(2, dim=0)
                
                new.ff.wi_0.weight.data.copy_(gate_weight)
                new.ff.wi_1.weight.data.copy_(up_weight)
                new.ff.wo.weight.data.copy_(mlp.wo.weight.data)
                
                # LayerNorm inside MLP
                if hasattr(mlp, 'layernorm'):
                    new.norm2.weight.data.copy_(mlp.layernorm.weight.data)
                else:
                    nn.init.ones_(new.norm2.weight)
                    
            # Check for gated linear unit (GLU) style - fc1 might be fused gate+up
            elif hasattr(mlp, 'fc1') and hasattr(mlp, 'fc2'):
                fc1_weight = mlp.fc1.weight.data  # Could be (2*ff_dim, hidden) for GLU
                fc2_weight = mlp.fc2.weight.data  # (hidden, ff_dim)
                
                # Check if fc1 is fused gate+up (GLU style)
                if fc1_weight.shape[0] == 2 * fc2_weight.shape[1]:
                    # Fused: split into gate and up projections
                    ff_dim = fc1_weight.shape[0] // 2
                    gate_weight, up_weight = fc1_weight.chunk(2, dim=0)
                    new.ff.wi_0.weight.data.copy_(gate_weight)
                    new.ff.wi_1.weight.data.copy_(up_weight)
                else:
                    # Not fused - use same weights for both (standard FFN)
                    new.ff.wi_0.weight.data.copy_(fc1_weight)
                    new.ff.wi_1.weight.data.copy_(fc1_weight)
                
                new.ff.wo.weight.data.copy_(fc2_weight)
                
                # MLP layer norm
                if hasattr(orig, 'norm2'):
                    new.norm2.weight.data.copy_(orig.norm2.weight.data)
                else:
                    nn.init.ones_(new.norm2.weight)
                    
            elif hasattr(mlp, 'Wi') and hasattr(mlp, 'Wo'):
                # Alternative naming convention
                new.ff.wi_0.weight.data.copy_(mlp.Wi.weight.data)
                new.ff.wi_1.weight.data.copy_(mlp.Wi.weight.data)
                new.ff.wo.weight.data.copy_(mlp.Wo.weight.data)
                nn.init.ones_(new.norm2.weight)
            else:
                print(f"    WARNING: Unknown MLP structure in layer {idx}, using random init")
                nn.init.xavier_uniform_(new.ff.wi_0.weight)
                nn.init.xavier_uniform_(new.ff.wi_1.weight)
                nn.init.xavier_uniform_(new.ff.wo.weight)
                nn.init.ones_(new.norm2.weight)
        else:
            # Standard BERT structure (fallback)
            new.ff.wi_0.weight.data.copy_(orig.intermediate.dense.weight.data)
            new.ff.wi_1.weight.data.copy_(orig.intermediate.dense.weight.data)
            new.ff.wo.weight.data.copy_(orig.output.dense.weight.data)
            new.norm2.weight.data.copy_(orig.output.LayerNorm.weight.data)
    
    # Final encoder layer norm
    if hasattr(bert.encoder, 'LayerNorm'):
        encoder.final_norm.weight.data.copy_(bert.encoder.LayerNorm.weight.data)
    elif hasattr(bert, 'LayerNorm'):
        encoder.final_norm.weight.data.copy_(bert.LayerNorm.weight.data)
    else:
        # No final LN found, init to ones
        nn.init.ones_(encoder.final_norm.weight)
    print("    ✓ Encoder weights copied")
    
    # CRITICAL: Free DNABERT memory BEFORE building decoder
    print("    Freeing DNABERT memory...")
    del original_model
    del bert
    del original_layers
    del layer0
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
        # Encoder uses SparseAttention wrapper - actual weights in .attention
        if hasattr(encoder.layers[idx].self_attn, 'attention'):
            enc_attn = encoder.layers[idx].self_attn.attention
        else:
            enc_attn = encoder.layers[idx].self_attn
        
        # Decoder uses standard Attention (no wrapper)
        dec_attn = decoder.layers[idx].self_attn
        
        dec_attn.q.weight.data.copy_(enc_attn.q.weight.data)
        dec_attn.k.weight.data.copy_(enc_attn.k.weight.data)
        dec_attn.v.weight.data.copy_(enc_attn.v.weight.data)
        dec_attn.o.weight.data.copy_(enc_attn.o.weight.data)
        
        decoder.layers[idx].norm1.weight.data.copy_(encoder.layers[idx].norm1.weight.data)
    
    print(f"    ✓ Copied {num_copy} layers")
    
    # Random Init Cross-Attention
    print("\n[6] Random Init Cross-Attention")
    for layer in decoder.layers:
        nn.init.xavier_uniform_(layer.cross_attn.q.weight)
        nn.init.xavier_uniform_(layer.cross_attn.k.weight)
        nn.init.xavier_uniform_(layer.cross_attn.v.weight)
        nn.init.xavier_uniform_(layer.cross_attn.o.weight)
        nn.init.ones_(layer.norm2.weight)
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
            if hasattr(layer.ff, 'router'):
                nn.init.xavier_uniform_(layer.ff.router.weight)
        else:
            if hasattr(layer.ff, 'wi_0'):
                nn.init.xavier_uniform_(layer.ff.wi_0.weight)
                nn.init.xavier_uniform_(layer.ff.wi_1.weight)
                nn.init.xavier_uniform_(layer.ff.wo.weight)
        nn.init.ones_(layer.norm3.weight)
    nn.init.ones_(decoder.final_norm.weight)
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
    
    # Save tokenizer (includes new tokens if added)
    tokenizer.save_pretrained(save_path)
    print("    ✓ tokenizer saved")
    
    # Cleanup
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