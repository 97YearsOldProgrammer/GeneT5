import argparse
import pathlib
import time

import lib.dataset as ds
import lib.util    as util


parser = argparse.ArgumentParser(
    description='Compact chunks into pre-packed training data')
parser.add_argument('inputs', type=str, nargs='+', metavar='<bin>',
    help='input .bin files')
parser.add_argument('-o', '--output', required=True, type=str,
    metavar='<file>', help='output .packed file')
parser.add_argument('--tokenizer', required=True, type=str,
    metavar='<path>', help='tokenizer path')
parser.add_argument('--target', required=True, type=int,
    metavar='<int>', help='target tokens per packed sequence')
parser.add_argument('--hard_limit', required=False, type=int, default=None,
    metavar='<int>', help='hard limit (default: target * 1.1)')
parser.add_argument('--max_target_len', required=False, type=int, default=None,
    metavar='<int>', help='max cumulative target tokens per packed sample')
parser.add_argument('--target_hard_limit', required=False, type=int, default=None,
    metavar='<int>', help='target hard limit (default: max_target_len * 1.1)')
parser.add_argument('--block_size', required=False, type=int, default=64,
    metavar='<int>', help='attention block size [%(default)i]')
parser.add_argument('--window_size', required=False, type=int, default=256,
    metavar='<int>', help='attention window size [%(default)i]')
parser.add_argument('--seed', required=False, type=int, default=42,
    metavar='<int>', help='random seed [%(default)i]')
parser.add_argument('--file_parallel', required=False, type=int, default=1,
    metavar='<int>', help='files to process in parallel [%(default)i]')
parser.add_argument('--workers', required=False, type=int, default=1,
    metavar='<int>', help='parallel workers for Phase 3 [%(default)i]')

args       = parser.parse_args()
hard_limit = args.hard_limit or int(args.target * 1.1)


####################
#####  Header  #####
####################


total_start = time.time()

print(f"\n{'='*70}")
print(f"{'Compact Pipeline':^70}")
print(f"{'='*70}")
print(f"  Output format: Pre-packed sequences (ready to stream)")
print(f"  No runtime packing overhead during training")


######################
#####  Validate  #####
######################


file_paths = []
for path in args.inputs:
    path = pathlib.Path(path)
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        continue
    file_paths.append(path)

if not file_paths:
    print("\n  ERROR: No valid input files")
    exit(1)


########################
#####  Tokenizer   #####
########################


print(f"\n{' Tokenizer ':=^70}")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

print(f"  Path:       {args.tokenizer}")
print(f"  Vocab size: {len(tokenizer):,}")
print(f"  Pad token:  {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

if "[SEP]" not in tokenizer.get_vocab():
    tokenizer.add_tokens(["[SEP]"])
    print(f"  Added:      [SEP] (id={tokenizer.convert_tokens_to_ids('[SEP]')})")


########################
#####  Config      #####
########################


print(f"\n{' Configuration ':=^70}")
print(f"  Input files:     {len(file_paths)}")
print(f"  Target length:   {args.target:,} tokens")
print(f"  Hard limit:      {hard_limit:,} tokens")
if args.max_target_len:
    target_hard = args.target_hard_limit or int(args.max_target_len * 1.1)
    print(f"  Max target len:  {args.max_target_len:,} tokens")
    print(f"  Target hard lim: {target_hard:,} tokens")
print(f"  Block size:      {args.block_size}")
print(f"  Window size:     {args.window_size}")
print(f"  Phase 3 workers: {args.workers}")


##############################
#####  Phase 1: Metadata #####
##############################


print(f"\n{' Phase 1: Extract Metadata ':=^70}")
print(f"  Reading pre-computed token lengths...\n")

phase1_start = time.time()

metadata = ds.stream_extract_metadata(
    file_paths,
    file_parallel = args.file_parallel,
    block_size    = args.block_size,
    window_size   = args.window_size,
)

phase1_time = time.time() - phase1_start
print(f"\n  Time: {util.format_time(phase1_time)}")


##############################
#####  Phase 2: Packing  #####
##############################


print(f"\n{' Phase 2: Bin Packing (FFD) ':=^70}")
print(f"  Assigning chunks to bins...\n")

phase2_start = time.time()

group_assignments, pack_stats = ds.pack_from_metadata(
    metadata,
    args.target,
    hard_limit        = hard_limit,
    max_target_len    = args.max_target_len,
    target_hard_limit = args.target_hard_limit,
    seed              = args.seed,
)

phase2_time = time.time() - phase2_start

print(f"\n  Results:")
print(f"    Groups:           {pack_stats['total_groups']:,}")
print(f"    Input util:       {pack_stats['avg_utilization']*100:.1f}% avg")
if 'avg_target_utilization' in pack_stats:
    print(f"    Target util:      {pack_stats['avg_target_utilization']*100:.1f}% avg")
print(f"    Singletons:       {pack_stats['singleton_count']:,}")
if pack_stats['overflow_count'] > 0:
    print(f"    Input overflow:   {pack_stats['overflow_count']:,}")
if pack_stats['target_overflow_count'] > 0:
    print(f"    Target overflow:  {pack_stats['target_overflow_count']:,}")
print(f"  Time: {util.format_time(phase2_time)}")


##############################
#####  Phase 3: Write    #####
##############################


print(f"\n{' Phase 3: Tokenize & Write (Streaming) ':=^70}")

phase3_start = time.time()

# Build reverse mapping: group_id -> [(file_idx, chunk_idx), ...]
groups = {}
for (file_idx, chunk_idx), group_id in group_assignments.items():
    if group_id not in groups:
        groups[group_id] = []
    groups[group_id].append((file_idx, chunk_idx))

for group_id in groups:
    groups[group_id].sort()

num_groups = len(groups)
print(f"  Groups to pack: {num_groups:,}")
print(f"  Workers: {args.workers}")
print(f"  Memory mode: streaming\n")


def process_group(work_item):
    """Worker function: load chunks and pack into sample"""

    group_id, members = work_item

    group_chunks = []
    for file_idx, chunk_idx in members:
        chunk = ds.read_chunk_at_index(file_paths[file_idx], chunk_idx)
        group_chunks.append(chunk)

    packed_sample = ds.pack_chunks_to_sample(
        group_chunks,
        tokenizer,
        block_size  = args.block_size,
        window_size = args.window_size,
    )

    return packed_sample


# Streaming: parallel workers + write in order as results come in
output_path     = pathlib.Path(args.output)
group_ids       = sorted(groups.keys())
input_lengths   = []
label_lengths   = []
segment_counts  = []

from concurrent.futures import ThreadPoolExecutor

with ds.PackedWriter(output_path, num_groups) as writer:
    with ThreadPoolExecutor(max_workers=args.workers) as executor:

        # Prepare all work items
        work_items = [(gid, groups[gid]) for gid in group_ids]

        # executor.map() returns results in order, streams as completed
        for progress, packed_sample in enumerate(executor.map(process_group, work_items)):

            input_lengths.append(packed_sample.total_input_len)
            label_lengths.append(packed_sample.total_label_len)
            segment_counts.append(packed_sample.num_segments)

            writer.write_sample(packed_sample)

            if (progress + 1) % 10000 == 0:
                pct = 100 * (progress + 1) / num_groups
                print(f"    {progress + 1:,}/{num_groups:,} ({pct:.1f}%)", end='\r')

print(f"    {num_groups:,}/{num_groups:,} (100.0%)")

phase3_time = time.time() - phase3_start
file_size   = output_path.stat().st_size

print(f"\n  Output:  {output_path}")
print(f"  Size:    {ds.format_size(file_size)}")
print(f"  Samples: {num_groups:,}")
print(f"  Time:    {util.format_time(phase3_time)}")


##############################
#####  Stats             #####
##############################


print(f"\n{' Packed Stats ':=^70}")

# Stats collected during streaming (no need to reload)
print(f"  Input tokens:  max {max(input_lengths):,}, avg {sum(input_lengths)/len(input_lengths):,.0f}")
print(f"  Label tokens:  max {max(label_lengths):,}, avg {sum(label_lengths)/len(label_lengths):,.0f}")
print(f"  Segments/sample: max {max(segment_counts)}, avg {sum(segment_counts)/len(segment_counts):.1f}")

print(f"\n  Token distribution:")
for threshold in [8192, 16384, 20000, 22000]:
    count = sum(1 for x in input_lengths if x > threshold)
    pct   = 100 * count / len(input_lengths)
    print(f"    > {threshold:,}: {count:,} ({pct:.2f}%)")


##############################
#####  Summary           #####
##############################


total_time = time.time() - total_start

print(f"\n{'='*70}")
print(f"  Phase 1 (metadata): {util.format_time(phase1_time)}")
print(f"  Phase 2 (packing):  {util.format_time(phase2_time)}")
print(f"  Phase 3 (write):    {util.format_time(phase3_time)}")
print(f"  Total:              {util.format_time(total_time)}")
print(f"{'='*70}")
print(f"Done!")
