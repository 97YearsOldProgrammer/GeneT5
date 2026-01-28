import argparse
import pathlib
import time
import multiprocessing as mp

import lib.dataset as ds
import lib.util    as util


parser = argparse.ArgumentParser(
    description='Streaming merge and compact binary files (memory-efficient)')
parser.add_argument('inputs', type=str, nargs='+', metavar='<bin>',
    help='input .bin files')
parser.add_argument('-o', '--output', required=True, type=str,
    metavar='<file>', help='output .bin file')
parser.add_argument('--tokenizer', required=True, type=str,
    metavar='<path>', help='tokenizer path (required)')
parser.add_argument('--compact_target', required=True, type=int,
    metavar='<int>', help='target tokens per packed sequence')
parser.add_argument('--hard_limit', required=False, type=int, default=None,
    metavar='<int>', help='hard limit for compacting (default: target * 1.1)')
parser.add_argument('--seed', required=False, type=int, default=42,
    metavar='<int>', help='random seed [%(default)i]')
parser.add_argument('--compress', action='store_true', default=False,
    help='compress output binary (slower for training)')
parser.add_argument('--workers', required=False, type=int, default=None,
    metavar='<int>', help=f'number of parallel workers per file [min(8, CPU count)]')
parser.add_argument('--file_parallel', required=False, type=int, default=1,
    metavar='<int>', help='number of files to process in parallel [%(default)i]')
parser.add_argument('--batch_size', required=False, type=int, default=1000,
    metavar='<int>', help='tokenization batch size [%(default)i]')

args = parser.parse_args()


total_start = time.time()
print(f"\n{' Streaming Compact Pipeline ':=^70}")
print(f"\n  Memory-efficient: processes one file at a time")
print(f"  Peak memory: ~size of largest input file + ~40MB metadata")

# Validate inputs
file_paths = []
for path in args.inputs:
    path = pathlib.Path(path)
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        continue
    file_paths.append(path)

if not file_paths:
    print("\n  ERROR: No valid input files!")
    exit(1)

# Configuration
n_workers = args.workers or min(8, mp.cpu_count())
hard_limit = args.hard_limit or int(args.compact_target * 1.1)

print(f"\n  Configuration:")
print(f"    Input files:   {len(file_paths)}")
print(f"    Target length: {args.compact_target:,} tokens")
print(f"    Hard limit:    {hard_limit:,} tokens")
print(f"    Workers/file:  {n_workers}")
print(f"    File parallel: {args.file_parallel}")
print(f"    Batch size:    {args.batch_size}")
print(f"    Compress:      {args.compress}")

# Load tokenizer
print(f"\n{' Loading Tokenizer ':=^70}")
from transformers import AutoTokenizer
print(f"  Path: {args.tokenizer}")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
print(f"  Vocab size: {len(tokenizer)}")


# ============================================
# Phase 1: Extract metadata (streaming)
# ============================================
print(f"\n{' Phase 1: Extract Metadata ':=^70}")
print(f"  Streaming through files, tokenizing, extracting lengths...")
print(f"  (Each file loaded → tokenized → freed)\n")

phase1_start = time.time()

metadata = ds.stream_extract_metadata(
    file_paths,
    tokenizer,
    batch_size=args.batch_size,
    n_workers=n_workers,
    file_parallel=args.file_parallel,
)

phase1_time = time.time() - phase1_start
print(f"\n  Phase 1 time: {util.format_time(phase1_time)}")


# ============================================
# Phase 2: Bin packing (integers only)
# ============================================
print(f"\n{' Phase 2: Bin Packing ':=^70}")
print(f"  Running FFD algorithm on metadata (no chunk data in memory)...\n")

phase2_start = time.time()

group_assignments, stats = ds.pack_from_metadata(
    metadata,
    args.compact_target,
    hard_limit=hard_limit,
    seed=args.seed,
)

phase2_time = time.time() - phase2_start

print(f"\n  Results:")
print(f"    Groups:      {stats['total_groups']:,}")
print(f"    Utilization: {stats['avg_utilization']*100:.1f}% avg "
      f"(min: {stats['min_utilization']*100:.1f}%, max: {stats['max_utilization']*100:.1f}%)")
print(f"    Overflow:    {stats['overflow_count']:,} (chunks > hard limit)")
print(f"    Singletons:  {stats['singleton_count']:,} (single-chunk bins)")
print(f"  Phase 2 time: {util.format_time(phase2_time)}")

# Free metadata
del metadata


# ============================================
# Phase 3: Write output (streaming)
# ============================================
print(f"\n{' Phase 3: Write Output ':=^70}")
print(f"  Re-reading files and writing with group assignments...")
print(f"  (Each file loaded → chunks written → freed)\n")

phase3_start = time.time()

output_path = pathlib.Path(args.output)

ds.stream_write_compacted(
    file_paths,
    group_assignments,
    output_path,
    compress=args.compress,
)

phase3_time = time.time() - phase3_start

# Get output stats
file_size = output_path.stat().st_size
total_chunks = len(group_assignments)

print(f"\n  Output: {output_path}")
print(f"  Size:   {ds.format_size(file_size)}")
print(f"  Chunks: {total_chunks:,}")
print(f"  Phase 3 time: {util.format_time(phase3_time)}")


# ============================================
# Summary
# ============================================
total_time = time.time() - total_start

print(f"\n{'='*70}")
print(f"  Phase 1 (metadata): {util.format_time(phase1_time)}")
print(f"  Phase 2 (packing):  {util.format_time(phase2_time)}")
print(f"  Phase 3 (writing):  {util.format_time(phase3_time)}")
print(f"  Total time:         {util.format_time(total_time)}")
print(f"{'='*70}")

print(f"\n  NOTE: Use CompactingCollator with block-diagonal attention")
print(f"        for proper decoder masking during training.")
print(f"\n{'='*70}")
print('Done!')
