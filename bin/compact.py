#!/usr/bin/env python3

"""Merge and compact multiple .bin files into consolidated binary"""

import argparse as ap
import pathlib  as pl

import lib.binary     as bi
import lib.compacting as co
import lib.util       as ut


#####################  Main Entry  #####################


def main():
    """Execute merge and compact pipeline"""
    
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


#####################  Argument Parsing  #####################


def parse_args():
    """Parse command line arguments"""
    
    parser = ap.ArgumentParser(
        description="Merge multiple .bin files and compact to target length"
    )
    
    parser.add_argument("inputs", nargs="+",
        help="Input .bin files to merge")
    parser.add_argument("-o", "--output", required=True,
        help="Output .bin file path")
    parser.add_argument("--compact_target", type=int, default=None,
        help="Target input length for compacting (soft limit in tokens)")
    parser.add_argument("--hard_limit", type=int, default=None,
        help="Hard limit for compacting (default: 1.1x target)")
    parser.add_argument("--bp_per_token", type=float, default=4.5,
        help="Base pairs per token estimate (default: 4.5)")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for compacting")
    parser.add_argument("--compress", action="store_true", default=True,
        help="Compress output (default: True)")
    parser.add_argument("--no_compress", action="store_false", dest="compress",
        help="Disable compression")
    
    return parser.parse_args()


#####################  Loading  #####################


def load_all_binaries(input_paths):
    """Load and merge all input binary files"""
    
    print(f"\n  Loading {len(input_paths)} input file(s)...")
    
    all_chunks   = []
    total_loaded = 0
    
    for path in input_paths:
        path = pl.Path(path)
        if not path.exists():
            print(f"    WARNING: {path} not found, skipping")
            continue
        
        info   = bi.get_binary_info(path)
        chunks = bi.read_binary(path)
        
        all_chunks.extend(chunks)
        total_loaded += len(chunks)
        
        print(f"    {path.name}: {len(chunks)} chunks ({ut.format_size(info['total_size'])})")
    
    print(f"  Total loaded: {total_loaded} chunks")
    
    return all_chunks


#####################  Compacting  #####################


def compact_merged(all_chunks, args):
    """Compact merged chunks to target length"""
    
    print(f"\n{' Compacting ':=^60}")
    print(f"  Input chunks:  {len(all_chunks)}")
    print(f"  Target length: {args.compact_target} tokens")
    
    hard_limit = args.hard_limit or int(args.compact_target * 1.1)
    print(f"  Hard limit:    {hard_limit} tokens")
    print(f"  BP per token:  {args.bp_per_token}")
    
    compacted_groups, stats = co.compact_chunks(
        all_chunks,
        args.compact_target,
        hard_limit,
        args.bp_per_token,
        args.seed,
    )
    
    all_chunks = co.flatten_groups(compacted_groups)
    
    print(f"\n  Results:")
    print(f"    Groups:       {stats['total_groups']}")
    print(f"    Utilization:  {stats['avg_utilization']*100:.1f}%")
    print(f"    Min util:     {stats['min_utilization']*100:.1f}%")
    print(f"    Max util:     {stats['max_utilization']*100:.1f}%")
    print(f"    Overflow:     {stats['overflow_count']}")
    print(f"    Singletons:   {stats['singleton_count']}")
    
    return all_chunks, stats


#####################  Output  #####################


def write_output(all_chunks, args, stats):
    """Write merged/compacted output"""
    
    print(f"\n{' Writing Output ':=^60}")
    
    output_path = pl.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    bi.write_binary(all_chunks, output_path, args.compress)
    
    file_size = output_path.stat().st_size
    
    raw_count = sum(1 for c in all_chunks if not c.is_augmented)
    aug_count = sum(1 for c in all_chunks if c.is_augmented)
    
    print(f"  Output: {output_path}")
    print(f"  Size:   {ut.format_size(file_size)}")
    print(f"  Chunks: {len(all_chunks)} (raw: {raw_count}, aug: {aug_count})")
    
    if stats:
        print(f"\n  Compacting Summary:")
        print(f"    Groups:      {stats['total_groups']}")
        print(f"    Utilization: {stats['avg_utilization']*100:.1f}%")


#####################  Script Entry  #####################


if __name__ == "__main__":
    main()
