import pathlib


#######################
#####  Utilities  #####
#######################


def append_tokens_to_txt(tokens, txt_path):
    """Append new tokens to txt file if not present"""

    txt_path = pathlib.Path(txt_path)

    existing = set()
    if txt_path.exists():
        with open(txt_path, 'r') as f:
            for line in f:
                existing.add(line.strip())

    new_tokens = [t for t in tokens if t not in existing]

    if new_tokens:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, 'a') as f:
            for token in new_tokens:
                f.write(token + '\n')

    return new_tokens


def print_run_stats(run_stats, chunk_stats, validation, output_path):
    """Print comprehensive run statistics"""

    print(f"\n{'=' * 60}")
    print("Run Statistics")
    print(f"{'=' * 60}")

    print(f"\n  Chunks:")
    print(f"    Raw:       {run_stats['raw_count']}")
    print(f"    Augmented: {run_stats['aug_count']}")
    print(f"    Total:     {run_stats['total_samples']}")

    if chunk_stats:
        print(f"\n  Window Scanning:")
        print(f"    Scanned:   {chunk_stats.get('windows_scanned', 0)}")
        print(f"    Empty:     {chunk_stats.get('windows_empty', 0)}")
        print(f"    N-heavy:   {chunk_stats.get('windows_n_heavy', 0)}")
        print(f"    Kept:      {chunk_stats.get('windows_kept', 0)}")

        if chunk_stats.get("features_per_chunk"):
            avg_feat = sum(chunk_stats["features_per_chunk"]) / len(chunk_stats["features_per_chunk"])
            print(f"    Avg feat:  {avg_feat:.1f}")

    if validation:
        print(f"\n  Validation:")
        print(f"    Complex loci: {len(validation.get('complex_loci', []))}")
        print(f"    Normal genes: {len(validation.get('normal_genes', []))}")
        print(f"    Easy samples: {len(validation.get('easy_samples', []))}")
        print(f"    Total genes:  {len(validation.get('all_ids', []))}")
        print(f"    Scenarios:    {len(validation.get('scenarios', []))}")

    if "compact_stats" in run_stats:
        cs = run_stats["compact_stats"]
        print(f"\n  Compacting:")
        print(f"    Groups:       {cs['total_groups']}")
        print(f"    Utilization:  {cs['avg_utilization']*100:.1f}%")
        print(f"    Overflow:     {cs['overflow_count']}")
        print(f"    Singletons:   {cs['singleton_count']}")

    file_size = run_stats.get("file_size", 0)
    print(f"\n  Output:")
    print(f"    Path: {output_path}")
    print(f"    Size: {file_size / 1024 / 1024:.2f} MB")


def format_size(size_bytes):
    """Format byte size to human readable string"""

    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"