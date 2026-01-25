#!/usr/bin/env python3

import argparse
import pathlib

import lib.dataset as ds


def main():

    args = parse_args()

    print(f"\n{' Binary Merge & Compact ':=^60}")

    all_chunks = load_all_binaries(args.inputs)

    if args.compact_target:
        all_chunks, stats = compact_merged(all_chunks, args)
    else:
        stats = None

    write_output(all_chunks, args, stats)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


def parse_args():

    parser = argparse.ArgumentParser(description="Merge and compact binary files")

    parser.add_argument("inputs", nargs="+", help="Input .bin files")
    parser.add_argument("-o", "--output",   required=True, help="Output .bin file")
    parser.add_argument("--compact_target", type=int,   default=None)
    parser.add_argument("--hard_limit",     type=int,   default=None)
    parser.add_argument("--bp_per_token",   type=float, default=4.5)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--compress",       action="store_true", default=True)
    parser.add_argument("--no_compress",    action="store_false", dest="compress")

    return parser.parse_args()


def load_all_binaries(input_paths):

    print(f"\n  Loading {len(input_paths)} input file(s)...")

    all_chunks   = []
    total_loaded = 0

    for path in input_paths:
        path = pathlib.Path(path)
        if not path.exists():
            print(f"    WARNING: {path} not found, skipping")
            continue

        info   = ds.get_binary_info(path)
        chunks = ds.read_binary(path)

        all_chunks.extend(chunks)
        total_loaded += len(chunks)

        print(f"    {path.name}: {len(chunks)} chunks ({ds.format_size(info['total_size'])})")

    print(f"  Total loaded: {total_loaded} chunks")

    return all_chunks


def compact_merged(all_chunks, args):

    print(f"\n{' Compacting ':=^60}")

    hard_limit = args.hard_limit or int(args.compact_target * 1.1)

    print(f"  Input chunks:  {len(all_chunks)}")
    print(f"  Target length: {args.compact_target} tokens")
    print(f"  Hard limit:    {hard_limit} tokens")

    compacted_groups, stats = ds.compact_chunks(
        all_chunks, args.compact_target, hard_limit, args.bp_per_token, args.seed
    )

    all_chunks = ds.flatten_groups(compacted_groups)

    print(f"\n  Results:")
    print(f"    Groups:      {stats['total_groups']}")
    print(f"    Utilization: {stats['avg_utilization']*100:.1f}%")
    print(f"    Overflow:    {stats['overflow_count']}")
    print(f"    Singletons:  {stats['singleton_count']}")

    return all_chunks, stats


def write_output(all_chunks, args, stats):

    print(f"\n{' Writing Output ':=^60}")

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds.write_binary(all_chunks, output_path, args.compress)

    file_size = output_path.stat().st_size
    raw_count = sum(1 for c in all_chunks if not c.is_augmented)
    aug_count = sum(1 for c in all_chunks if c.is_augmented)

    print(f"  Output: {output_path}")
    print(f"  Size:   {ds.format_size(file_size)}")
    print(f"  Chunks: {len(all_chunks)} (raw: {raw_count}, aug: {aug_count})")


if __name__ == "__main__":
    main()