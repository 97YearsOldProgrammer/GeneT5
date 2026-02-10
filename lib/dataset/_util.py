import pathlib


#######################
#####  Utilities  #####
#######################


def print_run_stats(run_stats, chunk_stats, val_count, output_path):
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

    if val_count:
        print(f"\n  Validation:  {val_count} chunks")

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
