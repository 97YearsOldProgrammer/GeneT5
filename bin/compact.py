#!/usr/bin/env python3

import argparse
import pathlib

import lib.dataset as ds


parser = argparse.ArgumentParser(
    description='Merge and compact binary files',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Merge without compacting
  python bin/compact.py data1.bin data2.bin -o merged.bin
  
  # Compact with tokenizer (recommended)
  python bin/compact.py training.bin -o compacted.bin \\
      --tokenizer models/geneT5 --compact_target 8192
  
  # Compact with hard limit
  python bin/compact.py training.bin -o compacted.bin \\
      --tokenizer models/geneT5 --compact_target 8192 --hard_limit 10000
""")

parser.add_argument('inputs', type=str, nargs='+', metavar='<bin>',
    help='input .bin files')
parser.add_argument('-o', '--output', required=True, type=str,
    metavar='<file>', help='output .bin file')
parser.add_argument('--tokenizer', required=False, type=str, default=None,
    metavar='<path>', help='tokenizer path (required for compacting)')
parser.add_argument('--compact_target', required=False, type=int, default=None,
    metavar='<int>', help='target tokens for compacting')
parser.add_argument('--hard_limit', required=False, type=int, default=None,
    metavar='<int>', help='hard limit for compacting (default: target * 1.1)')
parser.add_argument('--seed', required=False, type=int, default=42,
    metavar='<int>', help='random seed [%(default)i]')
parser.add_argument('--compress', action='store_true', default=True,
    help='compress output binary')
parser.add_argument('--no_compress', action='store_false', dest='compress',
    help='do not compress output binary')

args = parser.parse_args()

print(f"\n{' Binary Merge & Compact ':=^60}")

# load tokenizer if compacting
tokenizer = None
if args.compact_target:
    if args.tokenizer is None:
        print("\n  ERROR: --tokenizer required for compacting")
        print("  Compacting needs actual token counts for accuracy.")
        print("  Without tokenizer, use rough estimates which may be inaccurate.")
        exit(1)

    from transformers import AutoTokenizer
    print(f"\n  Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"  Vocab size: {len(tokenizer)}")

# load all binaries
print(f"\n  Loading {len(args.inputs)} input file(s)...")

all_chunks   = []
total_loaded = 0

for path in args.inputs:
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

# compact if requested
stats = None
if args.compact_target:
    print(f"\n{' Compacting ':=^60}")

    hard_limit = args.hard_limit or int(args.compact_target * 1.1)

    print(f"  Input chunks:  {len(all_chunks)}")
    print(f"  Target length: {args.compact_target} tokens")
    print(f"  Hard limit:    {hard_limit} tokens")
    print(f"  Separator:     [SEP] (block-diagonal decoder attention)")

    compacted_groups, stats = ds.compact_chunks(
        all_chunks, args.compact_target, hard_limit, tokenizer, args.seed
    )

    all_chunks = ds.flatten_groups(compacted_groups)

    print(f"\n  Results:")
    print(f"    Groups:      {stats['total_groups']}")
    print(f"    Utilization: {stats['avg_utilization']*100:.1f}%")
    print(f"    Overflow:    {stats['overflow_count']}")
    print(f"    Singletons:  {stats['singleton_count']}")

# write output
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

if args.compact_target:
    print(f"\n  NOTE: Use CompactingCollator with block-diagonal attention")
    print(f"        for proper decoder masking during training.")

print(f"\n{'='*60}")
print('Done!')
print(f"{'='*60}")