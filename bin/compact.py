import argparse
import pathlib
import time
import multiprocessing as mp

import lib.dataset as ds
import lib.util    as util


parser = argparse.ArgumentParser(
    description='Merge and compact binary files')
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
parser.add_argument('--workers', required=False, type=int, default=None,
    metavar='<int>', help=f'number of parallel workers [min(8, CPU count)]')
parser.add_argument('--batch_size', required=False, type=int, default=1000,
    metavar='<int>', help='tokenization batch size [%(default)i]')

args = parser.parse_args()


total_start = time.time()
print(f"\n{' Optimized Binary Merge & Compact ':=^70}")

# Determine worker count
n_workers = args.workers or min(8, mp.cpu_count())
print(f"\n  Workers: {n_workers}, Batch size: {args.batch_size}")

# Load tokenizer if compacting
tokenizer = None
tokenizer_path = None
if args.compact_target:
    if args.tokenizer is None:
        print("\n  ERROR: --tokenizer required for compacting")
        print("  Compacting needs actual token counts for accuracy.")
        exit(1)
    
    from transformers import AutoTokenizer
    print(f"\n  Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer_path = args.tokenizer
    print(f"  Vocab size: {len(tokenizer)}")

# Load all binaries with progress
print(f"\n{' Loading Files ':=^70}")
print(f"  Loading {len(args.inputs)} input file(s)...\n")

load_start = time.time()
all_chunks   = []
total_loaded = 0
total_size   = 0

for i, path in enumerate(args.inputs):
    path = pathlib.Path(path)
    if not path.exists():
        print(f"    [{i+1}/{len(args.inputs)}] WARNING: {path} not found, skipping")
        continue
    
    file_start = time.time()
    info   = ds.get_binary_info(path)
    chunks = ds.read_binary(path)
    file_time = time.time() - file_start
    
    all_chunks.extend(chunks)
    total_loaded += len(chunks)
    total_size   += info['total_size']
    
    print(f"    [{i+1}/{len(args.inputs)}] {path.name}: {len(chunks):,} chunks "
          f"({ds.format_size(info['total_size'])}) [{util.format_time(file_time)}]")

load_time = time.time() - load_start
print(f"\n  Total loaded: {total_loaded:,} chunks ({ds.format_size(total_size)})")
print(f"  Load time: {util.format_time(load_time)} ({util.format_rate(total_loaded, load_time)})")

# Compact if requested
stats = None
if args.compact_target:
    print(f"\n{' Compacting ':=^70}")
    
    hard_limit = args.hard_limit or int(args.compact_target * 1.1)
    
    print(f"\n  Configuration:")
    print(f"    Input chunks:  {len(all_chunks):,}")
    print(f"    Target length: {args.compact_target:,} tokens")
    print(f"    Hard limit:    {hard_limit:,} tokens")
    print(f"    Workers:       {n_workers}")
    print(f"    Batch size:    {args.batch_size}")
    
    compact_start = time.time()
    
    compacted_groups, stats = ds.compact_chunks(
        all_chunks, 
        args.compact_target, 
        hard_limit, 
        tokenizer, 
        args.seed,
        tokenizer_path=tokenizer_path,
        n_workers=n_workers,
        batch_size=args.batch_size
    )
    
    all_chunks = ds.flatten_groups(compacted_groups)
    
    compact_time = time.time() - compact_start
    
    print(f"\n  Results:")
    print(f"    Groups:      {stats['total_groups']:,}")
    print(f"    Utilization: {stats['avg_utilization']*100:.1f}% avg "
          f"(min: {stats['min_utilization']*100:.1f}%, max: {stats['max_utilization']*100:.1f}%)")
    print(f"    Overflow:    {stats['overflow_count']:,} (chunks > hard limit)")
    print(f"    Singletons:  {stats['singleton_count']:,} (single-chunk bins)")
    print(f"    Time:        {util.format_time(compact_time)}")

# Write output
print(f"\n{' Writing Output ':=^70}")

write_start = time.time()

output_path = pathlib.Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

ds.write_binary(all_chunks, output_path, args.compress)

write_time = time.time() - write_start

file_size = output_path.stat().st_size
raw_count = sum(1 for c in all_chunks if not c.is_augmented)
aug_count = sum(1 for c in all_chunks if c.is_augmented)

print(f"\n  Output: {output_path}")
print(f"  Size:   {ds.format_size(file_size)}")
print(f"  Chunks: {len(all_chunks):,} (raw: {raw_count:,}, aug: {aug_count:,})")
print(f"  Write time: {util.format_time(write_time)}")

if args.compact_target:
    print(f"\n  NOTE: Use CompactingCollator with block-diagonal attention")
    print(f"        for proper decoder masking during training.")

total_time = time.time() - total_start

print(f"\n{'='*70}")
print(f"  Total time: {util.format_time(total_time)}")
print(f"  Throughput: {util.format_rate(total_loaded, total_time)}")
print(f"{'='*70}")
print('Done!')