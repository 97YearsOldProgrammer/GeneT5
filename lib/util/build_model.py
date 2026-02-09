import torch
import torch.nn as nn
import json
import gc

from transformers import AutoModel, AutoConfig
from pathlib      import Path

from lib.blocks    import Encoder, Decoder, PerceiverCompressor, PerceiverConfig
from lib.tokenizer import GeneTokenizer


#####################
#####  Configs  #####
#####################


ENCODER_DEFAULTS = {
    "window_size": -1,
}

DECODER_DEFAULTS = {
    "block_size":   64,
    "window_size":  256,
    "dropout":      0.1,
    "use_alibi":    True,
    "use_moe":      True,
    "num_experts":  8,
    "moe_top_k":    2,
}

INIT_DEFAULTS = {
    "std":        0.02,
    "embed_std":  0.02,
    "ffn_std":    0.02,
    "attn_std":   0.02,
    "router_std": 0.006,
}

def build_gt5(
    dnabert_model_name  = "zhihan1996/DNABERT-2-117M",
    save_dir            = "./checkpoints/genet5_init",
    # Encoder sliding window
    encoder_window_size = ENCODER_DEFAULTS["window_size"],
    # Decoder sparse attention
    decoder_block_size  = DECODER_DEFAULTS["block_size"],
    decoder_window_size = DECODER_DEFAULTS["window_size"],
    # Decoder architecture
    decoder_num_layers  = None,  # defaults to encoder
    decoder_num_heads   = None,  # defaults to encoder
    decoder_num_kv_heads= None,  # defaults to heads // 4
    decoder_ff_dim      = None,  # defaults to encoder
    decoder_dropout     = DECODER_DEFAULTS["dropout"],
    decoder_use_alibi   = DECODER_DEFAULTS["use_alibi"],
    decoder_use_moe     = DECODER_DEFAULTS["use_moe"],
    decoder_num_experts = DECODER_DEFAULTS["num_experts"],
    decoder_moe_top_k   = DECODER_DEFAULTS["moe_top_k"],
    # Vocab
    vocab_size          = None,
    tie_weights         = True,
    # Perceiver
    num_latents         = 1024,
    perceiver_layers    = 2,
    # Init
    init_std            = INIT_DEFAULTS["std"],
    init_embed_std      = INIT_DEFAULTS["embed_std"],
    init_ffn_std        = INIT_DEFAULTS["ffn_std"],
    init_attn_std       = INIT_DEFAULTS["attn_std"],
    init_moe_router_std = INIT_DEFAULTS["router_std"],
):
    """
    Build GeneT5 from DNABERT-2 and save clean checkpoint.

    Architecture:
        Encoder: BigBird sparse attention (from DNABERT-2)
        Decoder: Sparse self-attention + block cross-attention + MoE FFN

    Weight Transfer:
        Encoder Embedding       -> COPY from DNABERT-2
        Encoder Self-Attention  -> COPY from DNABERT-2
        Decoder Self-Attention  -> COPY from Encoder
        Cross-Attention         -> RANDOM INIT
        Decoder FFN / MoE       -> RANDOM INIT
        Layer Norms             -> COPY where applicable
        Output Head             -> TIED or RANDOM INIT
    """
    
    # Handle None values - fall back to defaults
    if decoder_block_size is None:
        decoder_block_size = DECODER_DEFAULTS["block_size"]
    if decoder_window_size is None:
        decoder_window_size = DECODER_DEFAULTS["window_size"]
    if init_embed_std is None:
        init_embed_std = init_std
    if init_ffn_std is None:
        init_ffn_std = init_std
    if init_attn_std is None:
        init_attn_std = init_std

    print("=" * 60)
    print("Building GeneT5 from DNABERT-2")
    print("=" * 60)
    print(f"\nInit std: {init_std}, embed: {init_embed_std}, ffn: {init_ffn_std}, attn: {init_attn_std}, router: {init_moe_router_std}")
    
    # Build tokenizer with hardcoded gene finder tokens
    print(f"\n[1] Building tokenizer from DNABERT-2: {dnabert_model_name}")
    tokenizer = GeneTokenizer.from_dnabert(dnabert_model_name, save_dir=save_dir)

    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    print(f"      vocab_size: {vocab_size}")

    # Load DNABERT-2 config and raw weights (bypass model construction for NGC compat)
    dna_config = AutoConfig.from_pretrained(dnabert_model_name, trust_remote_code=True)

    if decoder_num_layers is None:
        decoder_num_layers = dna_config.num_hidden_layers
    if decoder_num_heads is None:
        decoder_num_heads = dna_config.num_attention_heads
    if decoder_ff_dim is None:
        decoder_ff_dim = dna_config.intermediate_size
    if decoder_num_kv_heads is None:
        decoder_num_kv_heads = max(1, decoder_num_heads // 4)

    embed_dim  = dna_config.hidden_size
    num_layers = dna_config.num_hidden_layers
    num_heads  = dna_config.num_attention_heads
    ff_dim     = dna_config.intermediate_size

    print(f"\n    DNABERT-2: hidden={embed_dim}, layers={num_layers}, heads={num_heads}")
    print(f"    Decoder:   layers={decoder_num_layers}, heads={decoder_num_heads}, kv_heads={decoder_num_kv_heads}, moe={decoder_use_moe}")
    print(f"    Encoder: window={encoder_window_size}")
    print(f"    Decoder Sparse: block={decoder_block_size}, window={decoder_window_size}")

    # Load raw state dict (no model construction needed)
    print(f"\n[2] Loading DNABERT-2 weights")
    from huggingface_hub import hf_hub_download
    weight_path = hf_hub_download(dnabert_model_name, "pytorch_model.bin")
    sd = torch.load(weight_path, map_location="cpu", weights_only=False)
    print(f"    ✓ {len(sd)} tensors loaded")

    # Extract Encoder Embedding
    print(f"\n[3] Extracting Encoder Embedding")
    encoder_embed = nn.Embedding(vocab_size, embed_dim)
    orig_embed    = sd["bert.embeddings.word_embeddings.weight"]
    copy_size     = min(orig_embed.shape[0], vocab_size)
    encoder_embed.weight.data[:copy_size].copy_(orig_embed[:copy_size])
    if vocab_size > orig_embed.shape[0]:
        nn.init.normal_(encoder_embed.weight.data[orig_embed.shape[0]:], mean=0.0, std=init_embed_std)
    print(f"    ✓ Copied {copy_size}, random init {max(0, vocab_size - copy_size)} new")

    # Build Encoder
    print(f"\n[4] Building Encoder")
    encoder = Encoder(
        num_layers   = num_layers,
        embed_dim    = embed_dim,
        num_heads    = num_heads,
        ff_dim       = ff_dim,
        dropout      = dna_config.hidden_dropout_prob,
        attn_dropout = dna_config.attention_probs_dropout_prob,
        use_alibi    = True,
        window_size  = encoder_window_size,
    )

    # Transfer Encoder Weights from raw state dict
    print(f"\n[5] Transferring Encoder Weights")
    for idx in range(num_layers):
        layer  = encoder.layers[idx]
        prefix = f"bert.encoder.layer.{idx}"

        # Q/K/V from fused Wqkv
        Wqkv = sd[f"{prefix}.attention.self.Wqkv.weight"]
        q_w, k_w, v_w = Wqkv.chunk(3, dim=0)
        layer.self_attn.q.weight.data.copy_(q_w)
        layer.self_attn.k.weight.data.copy_(k_w)
        layer.self_attn.v.weight.data.copy_(v_w)

        # Output projection
        layer.self_attn.o.weight.data.copy_(sd[f"{prefix}.attention.output.dense.weight"])

        # Post-attention LayerNorm
        layer.norm1.weight.data.copy_(sd[f"{prefix}.attention.output.LayerNorm.weight"])

        # FFN: gated_layers = [gate; up] fused, wo = down
        gated = sd[f"{prefix}.mlp.gated_layers.weight"]
        gate_w, up_w = gated.chunk(2, dim=0)
        layer.ff.wi_0.weight.data.copy_(gate_w)
        layer.ff.wi_1.weight.data.copy_(up_w)
        layer.ff.wo.weight.data.copy_(sd[f"{prefix}.mlp.wo.weight"])
        layer.norm2.weight.data.copy_(sd[f"{prefix}.mlp.layernorm.weight"])

    nn.init.ones_(encoder.final_norm.weight)
    print(f"    ✓ {num_layers} layers transferred")

    del sd
    gc.collect()
    
    # Build Perceiver Compressor
    print(f"\n[5] Building Perceiver Compressor (latents={num_latents}, layers={perceiver_layers})")
    compressor = PerceiverCompressor(PerceiverConfig(
        embed_dim              = dna_config.hidden_size,
        num_latents            = num_latents,
        num_heads              = dna_config.num_attention_heads,
        num_kv_heads           = max(1, dna_config.num_attention_heads // 4),
        num_layers             = perceiver_layers,
        ff_dim                 = dna_config.intermediate_size,
        dropout                = decoder_dropout,
        gradient_checkpointing = True,
    ))
    print("    + Random init (no pretrained weights for perceiver)")

    # Build Decoder
    print(f"\n[6] Building Decoder")
    decoder = Decoder(
        num_layers   = decoder_num_layers,
        embed_dim    = dna_config.hidden_size,
        num_heads    = decoder_num_heads,
        ff_dim       = decoder_ff_dim,
        dropout      = decoder_dropout,
        attn_dropout = decoder_dropout,
        use_alibi    = decoder_use_alibi,
        use_moe      = decoder_use_moe,
        num_experts  = decoder_num_experts,
        moe_top_k    = decoder_moe_top_k,
        num_kv_heads = decoder_num_kv_heads,
        block_size   = decoder_block_size,
        window_size  = decoder_window_size,
    )
    
    # Transfer Decoder Self-Attention from Encoder
    print("\n[7] Transferring Decoder Self-Attention")
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
    print(f"\n[8] Random Init Cross-Attention")
    for layer in decoder.layers:
        nn.init.normal_(layer.cross_attn.q.weight, mean=0.0, std=init_attn_std)
        nn.init.normal_(layer.cross_attn.k.weight, mean=0.0, std=init_attn_std)
        nn.init.normal_(layer.cross_attn.v.weight, mean=0.0, std=init_attn_std)
        nn.init.normal_(layer.cross_attn.o.weight, mean=0.0, std=init_attn_std)
        nn.init.ones_(layer.norm2.weight)
    
    # Random Init Decoder FFN
    print(f"\n[9] Random Init Decoder FFN")
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
    print(f"\n[10] Building Decoder Embeddings and LM Head")
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
    print(f"\n[11] Saving to {save_dir}")
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
        "encoder_window_size":  encoder_window_size,
        "decoder_block_size":   decoder_block_size,
        "decoder_window_size":  decoder_window_size,
        "num_latents":          num_latents,
        "perceiver_layers":     perceiver_layers,
    }
    
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    checkpoint = {
        "encoder_embed": encoder_embed.state_dict(),
        "encoder":       encoder.state_dict(),
        "compressor":    compressor.state_dict(),
        "decoder":       decoder.state_dict(),
        "decoder_embed": decoder_embed.state_dict(),
        "lm_head":       lm_head.state_dict()
    }
    torch.save(checkpoint, save_path / "pytorch_model.bin")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    total_params  = sum(p.numel() for p in encoder_embed.parameters())
    total_params += sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in compressor.parameters())
    total_params += sum(p.numel() for p in decoder.parameters())
    total_params += sum(p.numel() for p in decoder_embed.parameters())
    total_params += sum(p.numel() for p in lm_head.parameters())
    
    print("\n" + "=" * 60)
    print(f"✓ GeneT5 Built: {total_params:,} params")
    print(f"  Saved to: {save_path}")
    print("=" * 60)
    
    return str(save_path)