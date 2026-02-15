import torch
import torch.nn as nn
import json
import gc

from transformers import AutoConfig
from pathlib      import Path

from lib.blocks    import Decoder
from lib.tokenizer import GeneTokenizer


DEFAULTS = {
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
    # Architecture
    num_layers          = None,
    num_heads           = None,
    ff_dim              = None,
    dropout             = DEFAULTS["dropout"],
    use_alibi           = DEFAULTS["use_alibi"],
    use_moe             = DEFAULTS["use_moe"],
    num_experts         = DEFAULTS["num_experts"],
    moe_top_k           = DEFAULTS["moe_top_k"],
    # Vocab
    vocab_size          = None,
    tie_weights         = True,
    # Init
    init_std            = INIT_DEFAULTS["std"],
    init_embed_std      = INIT_DEFAULTS["embed_std"],
    init_ffn_std        = INIT_DEFAULTS["ffn_std"],
    init_attn_std       = INIT_DEFAULTS["attn_std"],
    init_moe_router_std = INIT_DEFAULTS["router_std"],
):
    """Build decoder-only GeneT5 from DNABERT-2 and save clean checkpoint"""

    if init_embed_std is None:
        init_embed_std = init_std
    if init_ffn_std is None:
        init_ffn_std = init_std
    if init_attn_std is None:
        init_attn_std = init_std

    print("=" * 60)
    print("Building GeneT5 (decoder-only) from DNABERT-2")
    print("=" * 60)
    print(f"\nInit std: {init_std}, embed: {init_embed_std}, ffn: {init_ffn_std}, attn: {init_attn_std}, router: {init_moe_router_std}")

    # Build tokenizer
    print(f"\n[1] Building tokenizer from DNABERT-2: {dnabert_model_name}")
    tokenizer = GeneTokenizer.from_dnabert(dnabert_model_name, save_dir=save_dir)

    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    print(f"      vocab_size: {vocab_size}")

    # Load DNABERT-2 config
    dna_config = AutoConfig.from_pretrained(dnabert_model_name, trust_remote_code=True)

    if num_layers is None:
        num_layers = dna_config.num_hidden_layers
    if num_heads is None:
        num_heads = dna_config.num_attention_heads
    if ff_dim is None:
        ff_dim = dna_config.intermediate_size

    embed_dim       = dna_config.hidden_size
    dna_num_layers  = dna_config.num_hidden_layers

    print(f"\n    DNABERT-2: hidden={embed_dim}, layers={dna_num_layers}, heads={dna_config.num_attention_heads}")
    print(f"    Decoder:   layers={num_layers}, heads={num_heads}, moe={use_moe}, experts={num_experts}")

    # Load raw state dict
    print(f"\n[2] Loading DNABERT-2 weights")
    from huggingface_hub import hf_hub_download
    weight_path = hf_hub_download(dnabert_model_name, "pytorch_model.bin")
    sd = torch.load(weight_path, map_location="cpu", weights_only=False)
    print(f"    loaded {len(sd)} tensors")

    # Build embedding
    print(f"\n[3] Building Embedding")
    embed     = nn.Embedding(vocab_size, embed_dim)
    orig_embed = sd["bert.embeddings.word_embeddings.weight"]
    copy_size  = min(orig_embed.shape[0], vocab_size)
    embed.weight.data[:copy_size].copy_(orig_embed[:copy_size])
    if vocab_size > orig_embed.shape[0]:
        nn.init.normal_(embed.weight.data[orig_embed.shape[0]:], mean=0.0, std=init_embed_std)
    print(f"    copied {copy_size}, random init {max(0, vocab_size - copy_size)} new")

    # Build decoder stack
    print(f"\n[4] Building Decoder Stack")
    decoder = Decoder(
        num_layers   = num_layers,
        embed_dim    = embed_dim,
        num_heads    = num_heads,
        ff_dim       = ff_dim,
        dropout      = dropout,
        attn_dropout = dropout,
        use_alibi    = use_alibi,
        use_moe      = use_moe,
        num_experts  = num_experts,
        moe_top_k    = moe_top_k,
    )

    # Transfer weights from DNABERT-2
    print(f"\n[5] Transferring DNABERT-2 weights to decoder")
    num_copy = min(num_layers, dna_num_layers)
    for idx in range(num_copy):
        layer  = decoder.layers[idx]
        prefix = f"bert.encoder.layer.{idx}"

        # Self-attention: Q/K/V from fused Wqkv
        Wqkv            = sd[f"{prefix}.attention.self.Wqkv.weight"]
        q_w, k_w, v_w   = Wqkv.chunk(3, dim=0)
        layer.self_attn.q.weight.data.copy_(q_w)
        layer.self_attn.k.weight.data.copy_(k_w)
        layer.self_attn.v.weight.data.copy_(v_w)

        # Output projection
        layer.self_attn.o.weight.data.copy_(sd[f"{prefix}.attention.output.dense.weight"])

        # Post-attention LayerNorm
        layer.norm1.weight.data.copy_(sd[f"{prefix}.attention.output.LayerNorm.weight"])

        # MoE: clone DNABERT-2 FFN into ALL experts
        gated            = sd[f"{prefix}.mlp.gated_layers.weight"]
        gate_w, up_w     = gated.chunk(2, dim=0)
        down_w           = sd[f"{prefix}.mlp.wo.weight"]

        if use_moe:
            for e in range(num_experts):
                layer.ff.expert_weights.gate_weights.data[e].copy_(gate_w.T)
                layer.ff.expert_weights.up_weights.data[e].copy_(up_w.T)
                layer.ff.expert_weights.down_weights.data[e].copy_(down_w.T)

            # Router: random small init
            nn.init.normal_(layer.ff.gate.weight, std=init_moe_router_std)
        else:
            layer.ff.wi_0.weight.data.copy_(gate_w)
            layer.ff.wi_1.weight.data.copy_(up_w)
            layer.ff.wo.weight.data.copy_(down_w)

        # FFN norm
        layer.norm2.weight.data.copy_(sd[f"{prefix}.mlp.layernorm.weight"])

    nn.init.ones_(decoder.final_norm.weight)
    print(f"    transferred {num_copy} layers (self-attn + FFN cloned to {num_experts} experts)")

    del sd
    gc.collect()

    # LM Head
    print(f"\n[6] Building LM Head")
    lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    if tie_weights:
        lm_head.weight = embed.weight
        print("    weights tied")
    else:
        nn.init.normal_(lm_head.weight, mean=0.0, std=init_embed_std)
        print("    separate weights")

    # Save
    print(f"\n[7] Saving to {save_dir}")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    config = {
        "embed_dim":    embed_dim,
        "num_layers":   num_layers,
        "num_heads":    num_heads,
        "ff_dim":       ff_dim,
        "dropout":      dropout,
        "use_alibi":    use_alibi,
        "use_moe":      use_moe,
        "num_experts":  num_experts,
        "moe_top_k":    moe_top_k,
        "vocab_size":   vocab_size,
        "tie_weights":  tie_weights,
    }

    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    checkpoint = {
        "embed":   embed.state_dict(),
        "decoder": decoder.state_dict(),
        "lm_head": lm_head.state_dict(),
    }
    torch.save(checkpoint, save_path / "pytorch_model.bin")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_params  = sum(p.numel() for p in embed.parameters())
    total_params += sum(p.numel() for p in decoder.parameters())
    total_params += sum(p.numel() for p in lm_head.parameters())

    print("\n" + "=" * 60)
    print(f"GeneT5 Built: {total_params:,} params")
    print(f"  Saved to: {save_path}")
    print("=" * 60)

    return str(save_path)
