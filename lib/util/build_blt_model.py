import json

import torch

from pathlib        import Path
from lib.blt_model  import GeneBLT


def build_gene_blt(
    genet5_checkpoint_dir,
    save_dir,
    local_dim        = 256,
    local_num_layers = 4,
    local_num_heads  = 4,
    local_ff_dim     = 1024,
    patch_size       = 8,
    enc_window_size  = (256, 256),
    ngram_sizes      = tuple(range(3, 21)),
    hash_table_size  = 4096,
):
    """Build GeneBLT by loading existing GeneT5 global decoder weights"""

    genet5_path = Path(genet5_checkpoint_dir)

    with open(genet5_path / "config.json", "r") as f:
        gt5_config = json.load(f)

    model = GeneBLT(
        byte_vocab_size   = 14,
        local_dim         = local_dim,
        global_dim        = gt5_config["embed_dim"],
        local_num_layers  = local_num_layers,
        local_num_heads   = local_num_heads,
        local_ff_dim      = local_ff_dim,
        global_num_layers = gt5_config["num_layers"],
        global_num_heads  = gt5_config["num_heads"],
        global_ff_dim     = gt5_config["ff_dim"],
        patch_size        = patch_size,
        enc_window_size   = enc_window_size,
        dropout           = gt5_config.get("dropout", 0.1),
        use_moe           = gt5_config.get("use_moe", True),
        num_experts       = gt5_config.get("num_experts", 8),
        moe_top_k         = gt5_config.get("moe_top_k", 2),
        num_kv_heads      = gt5_config.get("num_kv_heads"),
        ngram_sizes       = ngram_sizes,
        hash_table_size   = hash_table_size,
    )

    gt5_ckpt = torch.load(
        genet5_path / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )

    model.global_transformer.load_state_dict(gt5_ckpt["decoder"], strict=True)
    print(f"Loaded global transformer weights from {genet5_path}")

    matched = sum(
        1 for k in gt5_ckpt["decoder"]
        if k in dict(model.global_transformer.named_parameters())
    )
    print(f"  Matched {matched}/{len(gt5_ckpt['decoder'])} decoder parameters")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    blt_config = {
        "byte_vocab_size":   14,
        "local_dim":         local_dim,
        "global_dim":        gt5_config["embed_dim"],
        "local_num_layers":  local_num_layers,
        "local_num_heads":   local_num_heads,
        "local_ff_dim":      local_ff_dim,
        "global_num_layers": gt5_config["num_layers"],
        "global_num_heads":  gt5_config["num_heads"],
        "global_ff_dim":     gt5_config["ff_dim"],
        "patch_size":        patch_size,
        "enc_window_size":   list(enc_window_size),
        "dropout":           gt5_config.get("dropout", 0.1),
        "use_moe":           gt5_config.get("use_moe", True),
        "num_experts":       gt5_config.get("num_experts", 8),
        "moe_top_k":         gt5_config.get("moe_top_k", 2),
        "num_kv_heads":      gt5_config.get("num_kv_heads"),
        "ngram_sizes":       list(ngram_sizes),
        "hash_table_size":   hash_table_size,
        "source_genet5":     str(genet5_path),
    }

    with open(save_path / "blt_config.json", "w") as f:
        json.dump(blt_config, f, indent=2)

    model.save(save_path / "pytorch_model.bin")

    stats = model.get_param_stats()
    print(f"\nGeneBLT built successfully:")
    print(f"  Local encoder:      {stats['local_encoder_trainable']:,} params")
    print(f"  Global transformer: {stats['global_transformer_trainable']:,} params")
    print(f"  Local decoder:      {stats['local_decoder_trainable']:,} params")
    print(f"  Total:              {stats['total_trainable']:,} params")
    print(f"  Saved to: {save_path}")

    return model
