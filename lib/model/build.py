import torch
import torch.nn as nn
import json
import gc

from transformers import AutoConfig
from pathlib      import Path

from lib.blocks         import Decoder
from lib.tokenizer.hf   import GeneTokenizer


UPCYCLE_NOISE_STD  = 0.001   # symmetry-breaking for tiled copies
UPCYCLE_FRESH_STD  = 0.06    # random init for fresh experts
UPCYCLE_ROUTER_STD = 0.001   # near-zero router init
NUM_BASE_EXPERTS   = 4       # experts sliced from dense FFN (3072 / 768 = 4)
NUM_FRESH_EXPERTS  = 4       # randomly initialized experts for new patterns


def _upcycle_moe_layer(layer, sd, prefix, ff_dim, num_experts, init_std):
    """Upcycle dense DNABERT-2 FFN into MoE experts

    Slices the dense [3072] FFN into 4 base experts of [768] each,
    tiles to fill 12 slots, leaves 4 fresh random experts for new patterns
    """

    gated        = sd[f"{prefix}.mlp.gated_layers.weight"]   # [6144, 768]
    gate_w, up_w = gated.chunk(2, dim=0)                     # each [3072, 768]
    down_w       = sd[f"{prefix}.mlp.wo.weight"]             # [768, 3072]

    dense_ff  = gate_w.shape[0]                              # 3072
    embed_dim = gate_w.shape[1]                              # 768
    chunk_dim = dense_ff // NUM_BASE_EXPERTS                 # 768

    # Transpose to match MoE layout: MoE does x @ W, so W is [embed_dim, ff_dim]
    gate_wT = gate_w.T     # [768, 3072]
    up_wT   = up_w.T       # [768, 3072]
    down_wT = down_w.T     # [3072, 768]

    # Slice into 4 base experts
    base_gate_up = []
    base_down    = []
    for i in range(NUM_BASE_EXPERTS):
        s = i * chunk_dim
        e = s + chunk_dim
        gu = torch.cat([gate_wT[:, s:e], up_wT[:, s:e]], dim=1)   # [768, 1536]
        base_gate_up.append(gu)
        base_down.append(down_wT[s:e, :].clone())                 # [768, 768]

    # Fill all experts
    gate_up_all = layer.ff.expert_weights.gate_up_weights.data   # [E, 768, 2*ff_dim]
    down_all    = layer.ff.expert_weights.down_weights.data      # [E, ff_dim, 768]

    num_tiled = num_experts - NUM_FRESH_EXPERTS
    for slot in range(num_tiled):
        base_idx = slot % NUM_BASE_EXPERTS
        tile_num = slot // NUM_BASE_EXPERTS

        gate_up_all[slot].copy_(base_gate_up[base_idx])
        down_all[slot].copy_(base_down[base_idx])

        # Symmetry-breaking noise on copies (keep first tile exact)
        if tile_num > 0:
            gate_up_all[slot].add_(torch.randn_like(gate_up_all[slot]) * UPCYCLE_NOISE_STD)
            down_all[slot].add_(torch.randn_like(down_all[slot]) * UPCYCLE_NOISE_STD)

    # Fresh random experts for learning new structure
    for slot in range(num_tiled, num_experts):
        nn.init.normal_(gate_up_all[slot], std=UPCYCLE_FRESH_STD)
        nn.init.normal_(down_all[slot], std=UPCYCLE_FRESH_STD)

    # Router: near-zero so early training approximates dense behavior
    nn.init.normal_(layer.ff.gate.weight, std=UPCYCLE_ROUTER_STD)


DEFAULTS = {
    "dropout":      0.1,
    "use_alibi":    True,
    "use_moe":      True,
    "num_experts":  16,
    "moe_top_k":    2,
    "init_std":     0.006,
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
    init_std            = DEFAULTS["init_std"],
):
    """Build decoder-only GeneT5 from DNABERT-2 and save clean checkpoint"""

    print("=" * 60)
    print("Building GeneT5 (decoder-only) from DNABERT-2")
    print("=" * 60)
    print(f"\nInit std: {init_std}")

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
    print(f"    Decoder:   layers={num_layers}, heads={num_heads}, moe={use_moe}, experts={num_experts}, ff_dim={ff_dim}")

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
        nn.init.normal_(embed.weight.data[orig_embed.shape[0]:], mean=0.0, std=init_std)
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

        # FFN weights
        if use_moe:
            _upcycle_moe_layer(layer, sd, prefix, ff_dim, num_experts, init_std)
        else:
            gated        = sd[f"{prefix}.mlp.gated_layers.weight"]
            gate_w, up_w = gated.chunk(2, dim=0)
            down_w       = sd[f"{prefix}.mlp.wo.weight"]
            layer.ff.wi_0.weight.data.copy_(gate_w)
            layer.ff.wi_1.weight.data.copy_(up_w)
            layer.ff.wo.weight.data.copy_(down_w)

        # FFN norm
        layer.norm2.weight.data.copy_(sd[f"{prefix}.mlp.layernorm.weight"])

    nn.init.ones_(decoder.final_norm.weight)
    if use_moe:
        print(f"    transferred {num_copy} layers (self-attn + norms from DNABERT-2)")
        print(f"    MoE upcycled: {NUM_BASE_EXPERTS} base experts from dense FFN, "
              f"tiled to {num_experts - NUM_FRESH_EXPERTS}, "
              f"{NUM_FRESH_EXPERTS} fresh (std={UPCYCLE_FRESH_STD})")
    else:
        print(f"    transferred {num_copy} layers (self-attn + FFN + norms from DNABERT-2)")

    del sd
    gc.collect()

    # LM Head
    print(f"\n[6] Building LM Head")
    lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    if tie_weights:
        lm_head.weight = embed.weight
        print("    weights tied")
    else:
        nn.init.normal_(lm_head.weight, mean=0.0, std=init_std)
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
