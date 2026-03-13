import torch
import torch.nn as nn
import json
import gc

from transformers import AutoConfig
from pathlib      import Path

from lib.blocks       import Encoder
from lib.tokenizer.hf import GeneTokenizer


def _transfer_layer(layer, sd, prefix):
    """Transfer DNABERT-2 weights into an EncoderBlock"""

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
    gated        = sd[f"{prefix}.mlp.gated_layers.weight"]
    gate_w, up_w = gated.chunk(2, dim=0)
    down_w       = sd[f"{prefix}.mlp.wo.weight"]
    layer.ff.wi_0.weight.data.copy_(gate_w)
    layer.ff.wi_1.weight.data.copy_(up_w)
    layer.ff.wo.weight.data.copy_(down_w)

    # FFN norm
    layer.norm2.weight.data.copy_(sd[f"{prefix}.mlp.layernorm.weight"])


def _zero_init_outputs(layer):
    """Zero output projections so layer acts as identity through residual"""

    nn.init.zeros_(layer.self_attn.o.weight)
    nn.init.zeros_(layer.ff.wo.weight)


def build_gt5(
    dnabert_model_name  = "zhihan1996/DNABERT-2-117M",
    save_dir            = "./checkpoints/genet5_init",
    num_layers          = None,
    num_heads           = None,
    ff_dim              = None,
    dropout             = 0.1,
    vocab_size          = None,
    tie_weights         = True,
    init_std            = 0.006,
    depth_scaling       = False,
):
    """Build GeneT5 from DNABERT-2 with optional G_stack depth upscaling"""

    print("=" * 60)
    print("Building GeneT5 from DNABERT-2")
    print("=" * 60)

    # Build tokenizer
    print(f"\n[1] Building tokenizer from DNABERT-2: {dnabert_model_name}")
    tokenizer = GeneTokenizer.from_dnabert(dnabert_model_name, save_dir=save_dir)

    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    print(f"      vocab_size: {vocab_size}")

    # Load DNABERT-2 config
    dna_config = AutoConfig.from_pretrained(dnabert_model_name, trust_remote_code=True)

    dna_num_layers = dna_config.num_hidden_layers
    if num_layers is None:
        num_layers = dna_num_layers
    if num_heads is None:
        num_heads = dna_config.num_attention_heads
    if ff_dim is None:
        ff_dim = dna_config.intermediate_size

    embed_dim = dna_config.hidden_size
    g_stack   = num_layers > dna_num_layers

    print(f"\n    DNABERT-2: hidden={embed_dim}, layers={dna_num_layers}, heads={dna_config.num_attention_heads}")
    print(f"    GeneT5:    layers={num_layers}, heads={num_heads}, ff_dim={ff_dim}")
    if g_stack:
        print(f"    G_stack:   {dna_num_layers} → {num_layers} (duplicate + zero-init outputs)")
    if depth_scaling:
        print(f"    LayerNorm Scaling: residual *= 1/sqrt(depth)")

    # Load raw state dict
    print(f"\n[2] Loading DNABERT-2 weights")
    from huggingface_hub import hf_hub_download
    weight_path = hf_hub_download(dnabert_model_name, "pytorch_model.bin")
    sd = torch.load(weight_path, map_location="cpu", weights_only=False)
    print(f"    loaded {len(sd)} tensors")

    # Build embedding
    print(f"\n[3] Building Embedding")
    embed      = nn.Embedding(vocab_size, embed_dim)
    orig_embed = sd["bert.embeddings.word_embeddings.weight"]
    copy_size  = min(orig_embed.shape[0], vocab_size)
    embed.weight.data[:copy_size].copy_(orig_embed[:copy_size])
    if vocab_size > orig_embed.shape[0]:
        nn.init.normal_(embed.weight.data[orig_embed.shape[0]:], mean=0.0, std=init_std)
    print(f"    copied {copy_size}, random init {max(0, vocab_size - copy_size)} new")

    # Build encoder stack
    print(f"\n[4] Building Encoder Stack ({num_layers} layers)")
    encoder = Encoder(
        num_layers    = num_layers,
        embed_dim     = embed_dim,
        num_heads     = num_heads,
        ff_dim        = ff_dim,
        dropout       = dropout,
        attn_dropout  = dropout,
        depth_scaling = depth_scaling,
    )

    # Transfer DNABERT-2 weights
    print(f"\n[5] Transferring weights")
    for idx in range(min(num_layers, dna_num_layers)):
        src_idx = idx
        prefix  = f"bert.encoder.layer.{src_idx}"
        _transfer_layer(encoder.layers[idx], sd, prefix)

    # G_stack: duplicate layers beyond base, zero-init outputs
    if g_stack:
        print(f"    Layers 0-{dna_num_layers-1}: direct DNABERT-2 transfer")
        num_new = 0
        for idx in range(dna_num_layers, num_layers):
            src_idx = idx % dna_num_layers
            prefix  = f"bert.encoder.layer.{src_idx}"
            _transfer_layer(encoder.layers[idx], sd, prefix)
            _zero_init_outputs(encoder.layers[idx])
            num_new += 1
        print(f"    Layers {dna_num_layers}-{num_layers-1}: duplicated from 0-{dna_num_layers-1}, outputs zeroed ({num_new} new)")
    else:
        print(f"    transferred {min(num_layers, dna_num_layers)} layers")

    nn.init.ones_(encoder.final_norm.weight)

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
        "embed_dim":     embed_dim,
        "num_layers":    num_layers,
        "num_heads":     num_heads,
        "ff_dim":        ff_dim,
        "dropout":       dropout,
        "vocab_size":    vocab_size,
        "tie_weights":   tie_weights,
        "depth_scaling": depth_scaling,
        "base_layers":   dna_num_layers,
    }

    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    checkpoint = {
        "embed":   embed.state_dict(),
        "encoder": encoder.state_dict(),
        "lm_head": lm_head.state_dict(),
    }
    torch.save(checkpoint, save_path / "pytorch_model.bin")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_params  = sum(p.numel() for p in embed.parameters())
    total_params += sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in lm_head.parameters())

    print("\n" + "=" * 60)
    print(f"GeneT5 Built: {total_params:,} params ({num_layers}L, {embed_dim}d)")
    if g_stack:
        print(f"  G_stack: {dna_num_layers}→{num_layers} layers, new layer outputs zeroed")
    print(f"  Saved to: {save_path}")
    print("=" * 60)

    return str(save_path)
