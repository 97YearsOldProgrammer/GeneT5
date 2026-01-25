import torch
import torch.nn as nn
import json
import gc

from transformers import AutoModel, AutoTokenizer
from pathlib      import Path

from lib.blocks import Encoder, Decoder


def build_gt5(
    dnabert_model_name    = "zhihan1996/DNABERT-2-117M",
    save_dir              = "./checkpoints/genet5_init",
    block_size            = 64,
    num_rand_blocks       = 3,
    window_size           = 256,
    num_global_tokens     = 64,
    decoder_num_layers    = None,
    decoder_num_heads     = None,
    decoder_ff_dim        = None,
    decoder_dropout       = 0.1,
    decoder_use_alibi     = True,
    decoder_use_moe       = True,
    decoder_num_experts   = 8,
    decoder_moe_top_k     = 2,
    decoder_num_kv_heads  = None,
    vocab_size            = None,
    tie_weights           = True,
    new_tokens_list       = None,
    init_std              = 0.02,
    init_embed_std        = None,
    init_ffn_std          = None,
    init_attn_std         = None,
    init_moe_router_std   = None,
):
    """
    Build GeneT5 from DNABERT-2 and save clean checkpoint.
    
    Architecture:
        Encoder: BigBird sparse attention (from DNABERT-2)
        Decoder: Sparse self-attention + GQA cross-attention + MoE FFN
    
    Weight Transfer:
        Encoder Embedding       -> COPY from DNABERT-2
        Encoder Self-Attention  -> COPY from DNABERT-2
        Decoder Self-Attention  -> COPY from Encoder (sparse attention compatible)
        Cross-Attention         -> RANDOM INIT
        Decoder FFN / MoE       -> RANDOM INIT
        Layer Norms             -> COPY where applicable
        Output Head             -> TIED or RANDOM INIT
    """
    
    # Set component-specific stds
    embed_std  = init_embed_std      if init_embed_std      is not None else init_std
    ffn_std    = init_ffn_std        if init_ffn_std        is not None else init_std
    attn_std   = init_attn_std       if init_attn_std       is not None else init_std
    router_std = init_moe_router_std if init_moe_router_std is not None else init_std
    
    print("=" * 60)
    print("Building GeneT5 from DNABERT-2")
    print("=" * 60)
    print(f"\nInitialization Config:")
    print(f"  Default std:    {init_std}")
    print(f"  Embedding std:  {embed_std}")
    print(f"  FFN std:        {ffn_std}")
    print(f"  Attention std:  {attn_std}")
    print(f"  MoE Router std: {router_std}")
    
    # Load Pre-trained Model
    print(f"\n[1] Loading DNABERT-2: {dnabert_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(dnabert_model_name, trust_remote_code=True)
    
    # Expand Tokenizer
    if new_tokens_list:
        print(f"\n[1.5] Expanding Tokenizer")
        print(f"      Original vocab size: {len(tokenizer)}")
        num_added = tokenizer.add_tokens(new_tokens_list)
        print(f"      Tokens added: {num_added}")
        print(f"      New vocab size: {len(tokenizer)}")
    
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
    if decoder_num_kv_heads is None:
        decoder_num_kv_heads = max(1, decoder_num_heads // 4)
    
    print(f"\n    DNABERT-2 Config:")
    print(f"      hidden_size:     {dna_config.hidden_size}")
    print(f"      num_layers:      {dna_config.num_hidden_layers}")
    print(f"      num_heads:       {dna_config.num_attention_heads}")
    print(f"      intermediate:    {dna_config.intermediate_size}")
    print(f"\n    Decoder Config:")
    print(f"      num_layers:      {decoder_num_layers}")
    print(f"      num_heads:       {decoder_num_heads}")
    print(f"      num_kv_heads:    {decoder_num_kv_heads}")
    print(f"      ff_dim:          {decoder_ff_dim}")
    print(f"      use_alibi:       {decoder_use_alibi}")
    print(f"      use_moe:         {decoder_use_moe}")
    print(f"      num_experts:     {decoder_num_experts}")
    print(f"      moe_top_k:       {decoder_moe_top_k}")
    print(f"\n    Sparse Attention Config:")
    print(f"      block_size:      {block_size}")
    print(f"      window_size:     {window_size}")
    print(f"      num_global:      {num_global_tokens}")
    print(f"      num_random:      {num_rand_blocks}")
    
    # Get BERT backbone
    if hasattr(original_model, 'bert'):
        bert = original_model.bert
    else:
        bert = original_model
    
    # Extract Encoder Embedding
    print("\n[2] Extracting Encoder Embedding from DNABERT-2")
    encoder_embed = nn.Embedding(vocab_size, dna_config.hidden_size)
    
    if hasattr(bert, 'embeddings') and hasattr(bert.embeddings, 'word_embeddings'):
        orig_embed      = bert.embeddings.word_embeddings.weight.data
        orig_vocab_size = orig_embed.shape[0]
        
        copy_size = min(orig_vocab_size, vocab_size)
        encoder_embed.weight.data[:copy_size].copy_(orig_embed[:copy_size])
        
        if vocab_size > orig_vocab_size:
            nn.init.normal_(encoder_embed.weight.data[orig_vocab_size:], mean=0.0, std=embed_std)
            print(f"    ✓ Copied {copy_size} embeddings, random init for {vocab_size - orig_vocab_size} new tokens")
        else:
            print(f"    ✓ Copied {copy_size} embeddings")
    else:
        nn.init.normal_(encoder_embed.weight, mean=0.0, std=embed_std)
        print(f"    ! No embedding found, using random init")
    
    # Build Encoder
    print(f"\n[3] Building BigBird Encoder")
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
        window_size        = window_size,
        num_global_tokens  = num_global_tokens,
        num_random_blocks  = num_rand_blocks,
    )
    
    # Transfer Encoder Weights
    print("\n[4] Transferring Encoder Weights from DNABERT-2")
    
    original_layers = bert.encoder.layer
    layer0          = original_layers[0]
    
    print(f"    DNABERT-2 Layer Structure:")
    print(f"      Attention: {'Wqkv (fused)' if hasattr(layer0.attention.self, 'Wqkv') else 'Q/K/V (separate)'}")
    print(f"      FFN: {'mlp' if hasattr(layer0, 'mlp') else 'intermediate/output'}")
    
    for idx, (orig, new) in enumerate(zip(original_layers, encoder.layers)):
        orig_attn = orig.attention.self
        
        # Get actual attention module (may be wrapped in SparseAttention)
        if hasattr(new.self_attn, 'q'):
            new_attn = new.self_attn
        else:
            new_attn = new.self_attn
        
        # Transfer Q/K/V weights
        if hasattr(orig_attn, 'Wqkv'):
            # DNABERT-2 fused QKV
            Wqkv                         = orig_attn.Wqkv.weight.data
            q_weight, k_weight, v_weight = Wqkv.chunk(3, dim=0)
            
            new_attn.q.weight.data.copy_(q_weight)
            new_attn.k.weight.data.copy_(k_weight)
            new_attn.v.weight.data.copy_(v_weight)
            
            if orig_attn.Wqkv.bias is not None:
                q_bias, k_bias, v_bias = orig_attn.Wqkv.bias.data.chunk(3, dim=0)
                if hasattr(new_attn.q, 'bias') and new_attn.q.bias is not None:
                    new_attn.q.bias.data.copy_(q_bias)
                    new_attn.k.bias.data.copy_(k_bias)
                    new_attn.v.bias.data.copy_(v_bias)
        else:
            # Standard BERT separate Q/K/V
            new_attn.q.weight.data.copy_(orig_attn.query.weight.data)
            new_attn.k.weight.data.copy_(orig_attn.key.weight.data)
            new_attn.v.weight.data.copy_(orig_attn.value.weight.data)
        
        # Output projection
        new_attn.o.weight.data.copy_(orig.attention.output.dense.weight.data)
        
        # Layer norm after attention
        new.norm1.weight.data.copy_(orig.attention.output.LayerNorm.weight.data)
        
        # Transfer FFN weights
        if hasattr(orig, 'mlp'):
            mlp = orig.mlp
            
            if hasattr(mlp, 'gated_layers') and hasattr(mlp, 'wo'):
                # DNABERT-2 gated MLP
                gated_weight           = mlp.gated_layers.weight.data
                gate_weight, up_weight = gated_weight.chunk(2, dim=0)
                
                new.ff.wi_0.weight.data.copy_(gate_weight)
                new.ff.wi_1.weight.data.copy_(up_weight)
                new.ff.wo.weight.data.copy_(mlp.wo.weight.data)
                
                if hasattr(mlp, 'layernorm'):
                    new.norm2.weight.data.copy_(mlp.layernorm.weight.data)
                else:
                    nn.init.ones_(new.norm2.weight)
                    
            elif hasattr(mlp, 'fc1') and hasattr(mlp, 'fc2'):
                fc1_weight = mlp.fc1.weight.data
                fc2_weight = mlp.fc2.weight.data
                
                if fc1_weight.shape[0] == 2 * fc2_weight.shape[1]:
                    gate_weight, up_weight = fc1_weight.chunk(2, dim=0)
                    new.ff.wi_0.weight.data.copy_(gate_weight)
                    new.ff.wi_1.weight.data.copy_(up_weight)
                else:
                    new.ff.wi_0.weight.data.copy_(fc1_weight)
                    new.ff.wi_1.weight.data.copy_(fc1_weight)
                
                new.ff.wo.weight.data.copy_(fc2_weight)
                nn.init.ones_(new.norm2.weight)
            else:
                print(f"    WARNING: Unknown MLP structure in layer {idx}, using random init")
                nn.init.normal_(new.ff.wi_0.weight, mean=0.0, std=ffn_std)
                nn.init.normal_(new.ff.wi_1.weight, mean=0.0, std=ffn_std)
                nn.init.normal_(new.ff.wo.weight, mean=0.0, std=ffn_std)
                nn.init.ones_(new.norm2.weight)
        else:
            # Standard BERT FFN
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
        nn.init.ones_(encoder.final_norm.weight)
    
    print("    ✓ Encoder weights transferred")
    
    # Free DNABERT memory
    print("    Freeing DNABERT memory...")
    del original_model
    del bert
    del original_layers
    del layer0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Build Decoder
    print(f"\n[5] Building Decoder (sparse self-attn + GQA cross-attn + MoE)")
    decoder = Decoder(
        num_layers        = decoder_num_layers,
        embed_dim         = dna_config.hidden_size,
        num_heads         = decoder_num_heads,
        ff_dim            = decoder_ff_dim,
        dropout           = decoder_dropout,
        attn_dropout      = decoder_dropout,
        use_alibi         = decoder_use_alibi,
        use_moe           = decoder_use_moe,
        num_experts       = decoder_num_experts,
        moe_top_k         = decoder_moe_top_k,
        num_kv_heads      = decoder_num_kv_heads,
        block_size        = block_size,
        window_size       = window_size,
        num_global_tokens = num_global_tokens,
        num_random_blocks = num_rand_blocks,
    )
    
    # Transfer Decoder Self-Attention
    print("\n[6] Transferring Decoder Self-Attention from Encoder")
    num_copy = min(decoder_num_layers, len(encoder.layers))
    
    for idx in range(num_copy):
        # Get encoder attention (SparseAttention has direct q,k,v,o)
        enc_attn = encoder.layers[idx].self_attn
        dec_attn = decoder.layers[idx].self_attn
        
        # Direct copy - both use standard q,k,v,o projections
        dec_attn.q.weight.data.copy_(enc_attn.q.weight.data)
        dec_attn.k.weight.data.copy_(enc_attn.k.weight.data)
        dec_attn.v.weight.data.copy_(enc_attn.v.weight.data)
        dec_attn.o.weight.data.copy_(enc_attn.o.weight.data)
        
        # Copy layer norm
        decoder.layers[idx].norm1.weight.data.copy_(encoder.layers[idx].norm1.weight.data)
    
    print(f"    ✓ Copied {num_copy} layers (direct weight transfer)")
    
    # Random Init Cross-Attention
    print(f"\n[7] Random Init Cross-Attention (GQA, std={attn_std})")
    for layer in decoder.layers:
        nn.init.normal_(layer.cross_attn.q.weight, mean=0.0, std=attn_std)
        nn.init.normal_(layer.cross_attn.k.weight, mean=0.0, std=attn_std)
        nn.init.normal_(layer.cross_attn.v.weight, mean=0.0, std=attn_std)
        nn.init.normal_(layer.cross_attn.o.weight, mean=0.0, std=attn_std)
        nn.init.ones_(layer.norm2.weight)
    print("    ✓ Cross-attention initialized")
    
    # Random Init Decoder FFN
    print(f"\n[8] Random Init Decoder FFN (std={ffn_std}, router_std={router_std})")
    for layer in decoder.layers:
        if decoder_use_moe:
            # MoE experts
            if hasattr(layer.ff, 'experts'):
                for expert in layer.ff.experts:
                    if hasattr(expert, 'wi_gate'):
                        nn.init.normal_(expert.wi_gate.weight, mean=0.0, std=ffn_std)
                        nn.init.normal_(expert.wi_up.weight, mean=0.0, std=ffn_std)
                        nn.init.normal_(expert.wo.weight, mean=0.0, std=ffn_std)
            # MoE router
            if hasattr(layer.ff, 'gate'):
                nn.init.normal_(layer.ff.gate.weight, mean=0.0, std=router_std)
        else:
            # Standard FFN
            if hasattr(layer.ff, 'wi_0'):
                nn.init.normal_(layer.ff.wi_0.weight, mean=0.0, std=ffn_std)
                nn.init.normal_(layer.ff.wi_1.weight, mean=0.0, std=ffn_std)
                nn.init.normal_(layer.ff.wo.weight, mean=0.0, std=ffn_std)
        nn.init.ones_(layer.norm3.weight)
    
    nn.init.ones_(decoder.final_norm.weight)
    print("    ✓ FFN initialized")
    
    # Build Embeddings and LM Head
    print(f"\n[9] Building Decoder Embeddings and LM Head")
    decoder_embed = nn.Embedding(vocab_size, dna_config.hidden_size)
    lm_head       = nn.Linear(dna_config.hidden_size, vocab_size, bias=False)
    
    nn.init.normal_(decoder_embed.weight, mean=0.0, std=embed_std)
    
    if tie_weights:
        lm_head.weight = decoder_embed.weight
        print("    ✓ Weights tied (decoder_embed = lm_head)")
    else:
        nn.init.normal_(lm_head.weight, mean=0.0, std=embed_std)
        print("    ✓ Separate weights initialized")
    
    # Save Checkpoint
    print(f"\n[10] Saving Checkpoint to {save_dir}")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        "embed_dim":            dna_config.hidden_size,
        "encoder_num_layers":   dna_config.num_hidden_layers,
        "encoder_num_heads":    dna_config.num_attention_heads,
        "encoder_ff_dim":       dna_config.intermediate_size,
        "decoder_num_layers":   decoder_num_layers,
        "decoder_num_heads":    decoder_num_heads,
        "decoder_num_kv_heads": decoder_num_kv_heads,
        "decoder_ff_dim":       decoder_ff_dim,
        "decoder_dropout":      decoder_dropout,
        "decoder_use_alibi":    decoder_use_alibi,
        "decoder_use_moe":      decoder_use_moe,
        "decoder_num_experts":  decoder_num_experts,
        "decoder_moe_top_k":    decoder_moe_top_k,
        "vocab_size":           vocab_size,
        "tie_weights":          tie_weights,
        "block_size":           block_size,
        "window_size":          window_size,
        "num_global_tokens":    num_global_tokens,
        "num_rand_blocks":      num_rand_blocks,
        "init_std":             init_std,
        "init_embed_std":       embed_std,
        "init_ffn_std":         ffn_std,
        "init_attn_std":        attn_std,
        "init_moe_router_std":  router_std,
    }
    
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("    ✓ config.json saved")
    
    checkpoint = {
        "encoder_embed": encoder_embed.state_dict(),
        "encoder":       encoder.state_dict(),
        "decoder":       decoder.state_dict(),
        "decoder_embed": decoder_embed.state_dict(),
        "lm_head":       lm_head.state_dict()
    }
    torch.save(checkpoint, save_path / "pytorch_model.bin")
    print("    ✓ pytorch_model.bin saved")
    
    tokenizer.save_pretrained(save_path)
    print("    ✓ tokenizer saved")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Stats
    total_params  = sum(p.numel() for p in encoder_embed.parameters())
    total_params += sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in decoder.parameters())
    total_params += sum(p.numel() for p in decoder_embed.parameters())
    total_params += sum(p.numel() for p in lm_head.parameters())
    
    print("\n" + "=" * 60)
    print(f"✓ GeneT5 Built Successfully")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Saved to:         {save_path}")
    print("=" * 60)
    
    return str(save_path)