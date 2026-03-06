import pathlib
import json

import lib.data as ds


############################
#####  Stats Collection ####
############################


def collect_species_stats(baked_dir, species_name):
    """Collect stats from baked species data"""

    species_dir = pathlib.Path(baked_dir) / species_name
    train_path  = species_dir / "training.bin"
    val_path    = species_dir / "validation.bin"

    stats = {
        "species":       species_name,
        "has_data":      False,
        "train_chunks":  0,
        "val_chunks":    0,
        "train_size":    0,
        "val_size":      0,
        "raw_count":     0,
        "aug_count":     0,
    }

    if not train_path.exists():
        return stats

    stats["has_data"] = True

    # Get training stats
    try:
        train_info          = ds.get_binary_info(train_path)
        stats["train_chunks"] = train_info["num_chunks"]
        stats["train_size"]   = train_path.stat().st_size

        # Count raw vs augmented
        chunks            = ds.read_binary(train_path)
        stats["raw_count"] = sum(1 for c in chunks if not c.is_augmented)
        stats["aug_count"] = sum(1 for c in chunks if c.is_augmented)
    except Exception:
        pass

    # Get validation stats
    if val_path.exists():
        try:
            val_info           = ds.get_binary_info(val_path)
            stats["val_chunks"] = val_info["num_chunks"]
            stats["val_size"]   = val_path.stat().st_size
        except Exception:
            pass

    return stats


def write_bake_summary(log_path, run_config, species_results, species_stats, tokenizer_result=None, compact_result=None):
    """Write comprehensive bake summary log"""

    import datetime

    log_path = pathlib.Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"{'GeneT5 Data Bake Summary':^70}\n")
        f.write(f"{'='*70}\n")
        f.write(f"  Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")

        # Run configuration
        f.write(f"{' Run Configuration ':=^70}\n")
        f.write(f"  Raw directory:    {run_config.get('raw_dir', 'N/A')}\n")
        f.write(f"  Baked directory:  {run_config.get('baked_dir', 'N/A')}\n")
        f.write(f"  Species parallel: {run_config.get('species_parallel', 'N/A')}\n")
        f.write(f"  Workers/species:  {run_config.get('n_workers', 'N/A')}\n")
        f.write(f"  Tokenizer:        {run_config.get('tokenizer', 'None')}\n")
        f.write(f"  Compact:          {run_config.get('compact', False)}\n")
        if run_config.get('compact'):
            f.write(f"  Compact target:   {run_config.get('compact_target', 8192)}\n")
        f.write(f"\n")

        # Species results
        success_count = sum(1 for r in species_results if r.get("success"))
        failed_count  = len(species_results) - success_count

        f.write(f"{' Species Processing ':=^70}\n")
        f.write(f"  Total species:   {len(species_results)}\n")
        f.write(f"  Successful:      {success_count}\n")
        f.write(f"  Failed:          {failed_count}\n")
        f.write(f"\n")

        # Failed species details
        failed = [r for r in species_results if not r.get("success")]
        if failed:
            f.write(f"  Failed Species:\n")
            for r in failed:
                error = r.get("error", "Unknown error")
                f.write(f"    ✗ {r['species']}: {error}\n")
            f.write(f"\n")

        # Aggregated stats
        f.write(f"{' Aggregated Statistics ':=^70}\n")

        total_train_chunks = sum(s["train_chunks"] for s in species_stats)
        total_val_chunks   = sum(s["val_chunks"] for s in species_stats)
        total_raw          = sum(s["raw_count"] for s in species_stats)
        total_aug          = sum(s["aug_count"] for s in species_stats)
        total_train_size   = sum(s["train_size"] for s in species_stats)
        total_val_size     = sum(s["val_size"] for s in species_stats)

        f.write(f"\n  Training Data:\n")
        f.write(f"    Total chunks:     {total_train_chunks:,}\n")
        f.write(f"    Raw chunks:       {total_raw:,}\n")
        f.write(f"    Augmented chunks: {total_aug:,}\n")
        f.write(f"    Total size:       {ds.format_size(total_train_size)}\n")

        f.write(f"\n  Validation Data:\n")
        f.write(f"    Total chunks:     {total_val_chunks:,}\n")
        f.write(f"    Total size:       {ds.format_size(total_val_size)}\n")
        f.write(f"\n")

        # Per-species breakdown
        f.write(f"{' Per-Species Breakdown ':=^70}\n")
        f.write(f"\n")
        f.write(f"  {'Species':<20} {'Train':>8} {'Val':>6} {'Raw':>8} {'Aug':>8} {'Size':>10}\n")
        f.write(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*10}\n")

        for s in sorted(species_stats, key=lambda x: x["species"]):
            if s["has_data"]:
                f.write(f"  {s['species']:<20} {s['train_chunks']:>8,} {s['val_chunks']:>6} "
                        f"{s['raw_count']:>8,} {s['aug_count']:>8,} {ds.format_size(s['train_size']):>10}\n")
            else:
                f.write(f"  {s['species']:<20} {'--':>8} {'--':>6} {'--':>8} {'--':>8} {'--':>10}\n")
        f.write(f"\n")

        # Tokenizer expansion
        if tokenizer_result:
            f.write(f"{' Tokenizer Expansion ':=^70}\n")
            if tokenizer_result.get("success"):
                f.write(f"  Status:  Success\n")
                f.write(f"  Tokens:  {tokenizer_result.get('tokens_in_file', 'N/A')}\n")
            else:
                f.write(f"  Status:  Failed\n")
                f.write(f"  Error:   {tokenizer_result.get('error', 'Unknown')}\n")
            f.write(f"\n")

        # Compacting
        if compact_result:
            f.write(f"{' Compacting ':=^70}\n")
            if compact_result.get("success"):
                f.write(f"  Status:       Success\n")
                f.write(f"  Output:       {compact_result.get('output', 'N/A')}\n")
                f.write(f"  Output size:  {compact_result.get('size', 'N/A')}\n")
            else:
                f.write(f"  Status:  Failed\n")
                f.write(f"  Error:   {compact_result.get('error', 'Unknown')}\n")
            f.write(f"\n")

        f.write(f"{'='*70}\n")
        f.write(f"End of Summary\n")
        f.write(f"{'='*70}\n")

    return log_path


########################################
#####  Augmentation Status Report  #####
########################################


def report_augmentation_status(baked_dir):
    """Read merged bins and print nosing/augmentation ratios."""

    baked_dir = pathlib.Path(baked_dir)

    print(f"\n{'='*50}")
    print(f"{'Augmentation Status':^50}")
    print(f"{'='*50}")

    for label, filename in [("training", "training.bin"), ("validation", "validation.bin")]:
        bin_path = baked_dir / filename
        if not bin_path.exists():
            continue

        chunks    = ds.read_binary(bin_path)
        total     = len(chunks)
        raw_count = sum(1 for c in chunks if not c.is_augmented)
        aug_count = total - raw_count

        raw_pct = (raw_count / total * 100) if total else 0
        aug_pct = (aug_count / total * 100) if total else 0

        print(f"  {label}:")
        print(f"    Total:      {total:,} chunks")
        print(f"    Ab initio:  {raw_count:,} ({raw_pct:.1f}%)")
        print(f"    With hints: {aug_count:,} ({aug_pct:.1f}%)")

    eval_path = baked_dir / "eval.json"
    if eval_path.exists():
        with open(eval_path, 'r') as f:
            eval_data = json.load(f)
        print(f"  eval:")
        print(f"    Total: {len(eval_data)} samples (100% ab initio)")
