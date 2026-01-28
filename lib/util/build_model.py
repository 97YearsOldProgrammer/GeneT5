import torch
import torch.nn as nn
import json
import gc

from transformers import AutoModel, AutoTokenizer
from pathlib      import Path

from lib.blocks import Encoder, Decoder


#####################
#####  Configs  #####
#####################


ENCODER_DEFAULTS = {
    "block_size":        64,
    "window_size":       256,
    "num_global_tokens": 64,
    "num_rand_blocks":   3,
}

DECODER_DEFAULTS = {
    "block_size":        64,
    "window_size":       256,
    "num_global_tokens": 64,
    "num_rand_blocks":   3,
    "dropout":           0.1,
    "use_alibi":         True,
    "use_moe":           True,
    "num_experts":       8,
    "moe_top_k":         2,
}

INIT_DEFAULTS = {
    "std":        0.02,
    "embed_std":  0.02,
    "ffn_std":    0.02,
    "attn_std":   0.02,
    "router_std": 0.006,
}


def build_gt5(
    dnabert_model_name   = "zhihan1996/DNABERT-2-117M",
    save_dir             = "./checkpoints/genet5_init",
    # Encoder sparse attention
    encoder_block_size        = ENCODER_DEFAULTS["block_size"],
    encoder_window_size       = ENCODER_DEFAULTS["window_size"],
    encoder_num_global_tokens = ENCODER_DEFAULTS["num_global_tokens"],
    encoder_num_rand_blocks   = ENCODER_DEFAULTS["num_rand_blocks"],
    # Decoder sparse attention
    decoder_block_size        = DECODER_DEFAULTS["block_size"],
    decoder_window_size       = DECODER_DEFAULTS["window_size"],
    decoder_num_global_tokens = DECODER_DEFAULTS["num_global_tokens"],
    decoder_num_rand_blocks   = DECODER_DEFAULTS["num_rand_blocks"],
    # Decoder architecture
    decoder_num_layers   = None,  # defaults to encoder
    decoder_num_heads    = None,  # defaults to encoder
    decoder_num_kv_heads = None,  # defaults to heads // 4
    decoder_ff_dim       = None,  # defaults to encoder
    decoder_dropout      = DECODER_DEFAULTS["dropout"],
    decoder_use_alibi    = DECODER_DEFAULTS["use_alibi"],
    decoder_use_moe      = DECODER_DEFAULTS["use_moe"],
    decoder_num_experts  = DECODER_DEFAULTS["num_experts"],
    decoder_moe_top_k    = DECODER_DEFAULTS["moe_top_k"],
    # Vocab
    vocab_size      = None,
    tie_weights     = True,
    new_tokens_list = None,
    # Init
    init_std             = INIT_DEFAULTS["std"],
    init_embed_std       = INIT_DEFAULTS["embed_std"],
    init_ffn_std         = INIT_DEFAULTS["ffn_std"],
    init_attn_std        = INIT_DEFAULTS["attn_std"],
    init_moe_router_std  = INIT_DEFAULTS["router_std"],
):
    """
    Build GeneT5 from DNABERT-2 and save clean checkpoint.
    
    Architecture:
        Encoder: BigBird sparse attention (from DNABERT-2)
        Decoder: Sparse self-attention + GQA cross-attention + MoE FFN
    
    Weight Transfer:
        Encoder Embedding       -> COPY from DNABERT-2
        Encoder Self-Attention  -> COPY from DNABERT-2
        Decoder Self-Attention  -> COPY from Encoder
        Cross-Attention         -> RANDOM INIT
        Decoder FFN / MoE       -> RANDOM INIT
        Layer Norms             -> COPY where applicable
        Output Head             -> TIED or RANDOM INIT
    """
    
    print("=" * 60)
    print("Building GeneT5 from DNABERT-2")
    print("=" * 60)
    print(f"\nInit std: {init_std}, embed: {init_embed_std}, ffn: {init_ffn_std}, attn: {init_attn_std}, router: {init_moe_router_std}")
    
    # Load Pre-trained Model
    print(f"\n[1] Loading DNABERT-2: {dnabert_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(dnabert_model_name, trust_remote_code=True)
    
    # Expand Tokenizer
    if new_tokens_list:
        print(f"\n[1.5] Expanding Tokenizer: {len(tokenizer)} -> ", end="")
        tokenizer.add_tokens(new_tokens_list)
        print(f"{len(tokenizer)}")
    
    if vocab_size is None:
        vocab_size = len(tokenizer)
    print(f"      vocab_size: {vocab_size}")
    
    original_model = AutoModel.from_pretrained(dnabert_model_name, trust_remote_code=True)
    dna_config     = original_model.config
    
    # Decoder defaults to encoder
    if decoder_num_layers is None:
        decoder_num_layers = dna_config.num_hidden_layers
    if decoder_num_heads is None:
        decoder_num_heads = dna_config.num_attention_heads
    if decoder_ff_dim is None:
        decoder_ff_dim = dna_config.intermediate_size
    if decoder_num_kv_heads is None:
        decoder_num_kv_heads = max(1, decoder_num_heads // 4)
    
    print(f"\n    DNABERT-2: hidden={dna_config.hidden_size}, layers={dna_config.num_hidden_layers}, heads={dna_config.num_attention_heads}")
    print(f"    Decoder:   layers={decoder_num_layers}, heads={decoder_num_heads}, kv_heads={decoder_num_kv_heads}, moe={decoder_use_moe}")
    print(f"    Encoder Sparse: block={encoder_block_size}, window={encoder_window_size}, global={encoder_num_global_tokens}, rand={encoder_num_rand_blocks}")
    print(f"    Decoder Sparse: block={decoder_block_size}, window={decoder_window_size}, global={decoder_num_global_tokens}, rand={decoder_num_rand_blocks}")
    
    # Get BERT backbone
    bert = original_model.bert if hasattr(original_model, 'bert') else original_model
    
    # Extract Encoder Embedding
    print("\n[2] Extracting Encoder Embedding")
    encoder_embed = nn.Embedding(vocab_size, dna_config.hidden_size)
    
    if hasattr(bert, 'embeddings') and hasattr(bert.embeddings, 'word_embeddings'):
        orig_embed = bert.embeddings.word_embeddings.weight.data
        copy_size  = min(orig_embed.shape[0], vocab_size)
        encoder_embed.weight.data[:copy_size].copy_(orig_embed[:copy_size])
        if vocab_size > orig_embed.shape[0]:
            nn.init.normal_(encoder_embed.weight.data[orig_embed.shape[0]:], mean=0.0, std=init_embed_std)
        print(f"    ✓ Copied {copy_size} embeddings")
    else:
        nn.init.normal_(encoder_embed.weight, mean=0.0, std=init_embed_std)
        print(f"    ! Random init")
    
    # Build Encoder
    print(f"\n[3] Building Encoder")
    encoder = Encoder(
        num_layers         = dna_config.num_hidden_layers,
        embed_dim          = dna_config.hidden_size,
        num_heads          = dna_config.num_attention_heads,
        ff_dim             = dna_config.intermediate_size,
        dropout            = dna_config.hidden_dropout_prob,
        attn_dropout       = dna_config.attention_probs_dropout_prob,
        use_alibi          = True,
        use_bigbird_sparse = True,
        block_size         = encoder_block_size,
        window_size        = encoder_window_size,
        num_global_tokens  = encoder_num_global_tokens,
        num_random_blocks  = encoder_num_rand_blocks,
    )
    
    # Transfer Encoder Weights
    print("\n[4] Transferring Encoder Weights")
    original_layers = bert.encoder.layer
    
    for idx, (orig, new) in enumerate(zip(original_layers, encoder.layers)):
        orig_attn = orig.attention.self
        new_attn  = new.self_attn
        
        # Q/K/V weights
        if hasattr(orig_attn, 'Wqkv'):
            Wqkv = orig_attn.Wqkv.weight.data
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
            new_attn.q.weight.data.copy_(orig_attn.query.weight.data)
            new_attn.k.weight.data.copy_(orig_attn.key.weight.data)
            new_attn.v.weight.data.copy_(orig_attn.value.weight.data)
        
        new_attn.o.weight.data.copy_(orig.attention.output.dense.weight.data)
        new.norm1.weight.data.copy_(orig.attention.output.LayerNorm.weight.data)
        
        # FFN weights
        if hasattr(orig, 'mlp'):
            mlp = orig.mlp
            if hasattr(mlp, 'gated_layers') and hasattr(mlp, 'wo'):
                gated_weight = mlp.gated_layers.weight.data
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
                nn.init.normal_(new.ff.wi_0.weight, mean=0.0, std=init_ffn_std)
                nn.init.normal_(new.ff.wi_1.weight, mean=0.0, std=init_ffn_std)
                nn.init.normal_(new.ff.wo.weight, mean=0.0, std=init_ffn_std)
                nn.init.ones_(new.norm2.weight)
        else:
            new.ff.wi_0.weight.data.copy_(orig.intermediate.dense.weight.data)
            new.ff.wi_1.weight.data.copy_(orig.intermediate.dense.weight.data)
            new.ff.wo.weight.data.copy_(orig.output.dense.weight.data)
            new.norm2.weight.data.copy_(orig.output.LayerNorm.weight.data)
    
    # Final encoder norm
    if hasattr(bert.encoder, 'LayerNorm'):
        encoder.final_norm.weight.data.copy_(bert.encoder.LayerNorm.weight.data)
    elif hasattr(bert, 'LayerNorm'):
        encoder.final_norm.weight.data.copy_(bert.LayerNorm.weight.data)
    else:
        nn.init.ones_(encoder.final_norm.weight)
    
    print("    ✓ Done")
    
    # Free DNABERT memory
    del original_model, bert, original_layers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Build Decoder
    print(f"\n[5] Building Decoder")
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
        block_size        = decoder_block_size,
        window_size       = decoder_window_size,
        num_global_tokens = decoder_num_global_tokens,
        num_random_blocks = decoder_num_rand_blocks,
    )
    
    # Transfer Decoder Self-Attention from Encoder
    print("\n[6] Transferring Decoder Self-Attention")
    num_copy = min(decoder_num_layers, len(encoder.layers))
    for idx in range(num_copy):
        enc_attn = encoder.layers[idx].self_attn
        dec_attn = decoder.layers[idx].self_attn
        dec_attn.q.weight.data.copy_(enc_attn.q.weight.data)
        dec_attn.k.weight.data.copy_(enc_attn.k.weight.data)
        dec_attn.v.weight.data.copy_(enc_attn.v.weight.data)
        dec_attn.o.weight.data.copy_(enc_attn.o.weight.data)
        decoder.layers[idx].norm1.weight.data.copy_(encoder.layers[idx].norm1.weight.data)
    print(f"    ✓ Copied {num_copy} layers")
    
    # Random Init Cross-Attention
    print(f"\n[7] Random Init Cross-Attention")
    for layer in decoder.layers:
        nn.init.normal_(layer.cross_attn.q.weight, mean=0.0, std=init_attn_std)
        nn.init.normal_(layer.cross_attn.k.weight, mean=0.0, std=init_attn_std)
        nn.init.normal_(layer.cross_attn.v.weight, mean=0.0, std=init_attn_std)
        nn.init.normal_(layer.cross_attn.o.weight, mean=0.0, std=init_attn_std)
        nn.init.ones_(layer.norm2.weight)
    
    # Random Init Decoder FFN
    print(f"\n[8] Random Init Decoder FFN")
    for layer in decoder.layers:
        if decoder_use_moe:
            if hasattr(layer.ff, 'expert_weights'):
                nn.init.normal_(layer.ff.expert_weights.gate_weights, mean=0.0, std=init_ffn_std)
                nn.init.normal_(layer.ff.expert_weights.up_weights, mean=0.0, std=init_ffn_std)
                nn.init.normal_(layer.ff.expert_weights.down_weights, mean=0.0, std=init_ffn_std)
            if hasattr(layer.ff, 'gate'):
                nn.init.normal_(layer.ff.gate.weight, mean=0.0, std=init_moe_router_std)
        else:
            if hasattr(layer.ff, 'wi_0'):
                nn.init.normal_(layer.ff.wi_0.weight, mean=0.0, std=init_ffn_std)
                nn.init.normal_(layer.ff.wi_1.weight, mean=0.0, std=init_ffn_std)
                nn.init.normal_(layer.ff.wo.weight, mean=0.0, std=init_ffn_std)
        nn.init.ones_(layer.norm3.weight)
    nn.init.ones_(decoder.final_norm.weight)
    
    # Embeddings and LM Head
    print(f"\n[9] Building Decoder Embeddings and LM Head")
    decoder_embed = nn.Embedding(vocab_size, dna_config.hidden_size)
    lm_head       = nn.Linear(dna_config.hidden_size, vocab_size, bias=False)
    nn.init.normal_(decoder_embed.weight, mean=0.0, std=init_embed_std)
    
    if tie_weights:
        lm_head.weight = decoder_embed.weight
        print("    ✓ Weights tied")
    else:
        nn.init.normal_(lm_head.weight, mean=0.0, std=init_embed_std)
        print("    ✓ Separate weights")
    
    # Save
    print(f"\n[10] Saving to {save_dir}")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        "embed_dim":                 dna_config.hidden_size,
        "encoder_num_layers":        dna_config.num_hidden_layers,
        "encoder_num_heads":         dna_config.num_attention_heads,
        "encoder_ff_dim":            dna_config.intermediate_size,
        "decoder_num_layers":        decoder_num_layers,
        "decoder_num_heads":         decoder_num_heads,
        "decoder_num_kv_heads":      decoder_num_kv_heads,
        "decoder_ff_dim":            decoder_ff_dim,
        "decoder_dropout":           decoder_dropout,
        "decoder_use_alibi":         decoder_use_alibi,
        "decoder_use_moe":           decoder_use_moe,
        "decoder_num_experts":       decoder_num_experts,
        "decoder_moe_top_k":         decoder_moe_top_k,
        "vocab_size":                vocab_size,
        "tie_weights":               tie_weights,
        "encoder_block_size":        encoder_block_size,
        "encoder_window_size":       encoder_window_size,
        "encoder_num_global_tokens": encoder_num_global_tokens,
        "encoder_num_rand_blocks":   encoder_num_rand_blocks,
        "decoder_block_size":        decoder_block_size,
        "decoder_window_size":       decoder_window_size,
        "decoder_num_global_tokens": decoder_num_global_tokens,
        "decoder_num_rand_blocks":   decoder_num_rand_blocks,
    }
    
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    checkpoint = {
        "encoder_embed": encoder_embed.state_dict(),
        "encoder":       encoder.state_dict(),
        "decoder":       decoder.state_dict(),
        "decoder_embed": decoder_embed.state_dict(),
        "lm_head":       lm_head.state_dict()
    }
    torch.save(checkpoint, save_path / "pytorch_model.bin")
    tokenizer.save_pretrained(save_path)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    total_params  = sum(p.numel() for p in encoder_embed.parameters())
    total_params += sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in decoder.parameters())
    total_params += sum(p.numel() for p in decoder_embed.parameters())
    total_params += sum(p.numel() for p in lm_head.parameters())
    
    print("\n" + "=" * 60)
    print(f"✓ GeneT5 Built: {total_params:,} params")
    print(f"  Saved to: {save_path}")
    print("=" * 60)
    
    return str(save_path)