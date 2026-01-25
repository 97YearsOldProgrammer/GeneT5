import argparse
import sys
from pathlib import Path

from lib import util as ut
import lib.dataset   as ds
import lib.tokenizer as tk


parser = argparse.ArgumentParser(
    description="Parse GFF3/FASTA for GeneT5 with dynamic chunking")
parser.add_argument("fasta",
    help="Path to FASTA file")
parser.add_argument("gff",
    help="Path to GFF3 annotation file")
parser.add_argument("output_dir",
    help="Output directory")
parser.add_argument("--limit", type=int, default=25000,
    help="Window size limit in bp (default: 25000)")
parser.add_argument("--overlap_ratio", type=float, default=1/2.718281828,
    help="Overlap ratio relative to window size (default: 1/e ≈ 0.368)")
parser.add_argument("--anchor_pad_ratio", type=float, default=1/2.718281828,
    help="Padding ratio relative to window size for first gene (default: 1/e ≈ 0.368)")
parser.add_argument("--hint_ratio", type=float, default=0.5,
    help="Ratio of samples with hints (default: 0.5)")
parser.add_argument("--seed", type=int, default=42,
    help="Random seed")
parser.add_argument("--extract_tokens", type=str, default=None,
    help="Path to txt file for extracted tokens")
parser.add_argument("--validation_jsonl", type=str, default=None,
    help="Path to validation JSONL file (append mode)")
parser.add_argument("--long_threshold", type=int, default=50000,
    help="Threshold for long genes (default: 50000)")
parser.add_argument("--top_k_complex", type=int, default=5,
    help="Number of complex loci for validation (default: 5)")
parser.add_argument("--num_rare", type=int, default=10,
    help="Number of rare samples for validation (default: 10)")
parser.add_argument("--num_easy", type=int, default=10,
    help="Number of easy samples for validation (default: 10)")
parser.add_argument("--compress", action="store_true", default=True,
    help="Compress binary output")
parser.add_argument("--no_compress", action="store_false", dest="compress",
    help="Disable compression")
parser.add_argument("--compact_target", type=int, default=None,
    help="Target input length for compacting (soft limit)")
parser.add_argument("--compact_hard_limit", type=int, default=None,
    help="Hard limit for compacting (default: 1.1x target)")
parser.add_argument("--bp_per_token", type=float, default=4.5,
    help="Base pairs per token estimate (default: 4.5)")
parser.add_argument("--val_compact_target", type=int, default=None,
    help="Target input length for validation compacting")
args = parser.parse_args()


print(f"\n{' GFF3/FASTA Processing ':=^60}")

print("\nParsing FASTA...")
sequences = ds.parse_fasta(args.fasta)
print(f"  Found {len(sequences)} sequence(s): {list(sequences.keys())[:5]}...")

print("\nParsing GFF3...")
features = ds.parse_gff(args.gff)
print(f"  Found {len(features)} features")

type_counts = {}
for feat in features:
    ftype             = feat["type"]
    type_counts[ftype] = type_counts.get(ftype, 0) + 1
print(f"  Feature types: {dict(sorted(type_counts.items(), key=lambda x: -x[1])[:10])}")

print("\nBuilding gene index...")
gene_index = ds.build_gene_index(features)
print(f"  Found {len(gene_index)} genes")

if args.extract_tokens:
    print(f"\n{' Token Extraction ':=^60}")
    
    feature_types = ds.extract_feature_types(features)
    biotypes      = ds.extract_biotypes(features)
    all_types     = feature_types | biotypes
    
    print(f"  Feature types: {sorted(feature_types)}")
    print(f"  Biotypes:      {sorted(biotypes)}")
    
    added = tk.append_tokens_to_txt(sorted(all_types), args.extract_tokens)
    if added:
        print(f"  Added {len(added)} new tokens to {args.extract_tokens}")


###########################################
#####  Building Validation Set        #####
###########################################


print(f"\n{' Building Validation Set ':=^60}")

existing_val_ids = set()
if args.validation_jsonl and Path(args.validation_jsonl).exists():
    existing_val_ids = ds.get_existing_validation_ids(args.validation_jsonl)
    print(f"  Found {len(existing_val_ids)} existing validation gene IDs")

remaining_genes = {
    gid: gdata for gid, gdata in gene_index.items()
    if gid not in existing_val_ids
}

validation = ds.build_validation_set(
    remaining_genes,
    args.long_threshold,
    args.top_k_complex,
    args.num_rare,
    args.num_easy,
    args.seed,
)

overlap    = int(args.limit * args.overlap_ratio)
anchor_pad = int(args.limit * args.anchor_pad_ratio)

print(f"  New validation genes: {len(validation['all_ids'])}")
print(f"  New scenarios:        {len(validation['scenarios'])}")


###########################################
#####  Validation Compacting & Write  #####
###########################################


output_dir     = Path(args.output_dir)
val_jsonl_path = args.validation_jsonl or (output_dir / "validation.jsonl")

if validation["scenarios"]:
    val_compact_target = args.val_compact_target or args.compact_target or 4096
    val_hard_limit     = int(val_compact_target * 1.1)
    
    print(f"\n{' Validation Compacting ':=^60}")
    print(f"  Target length: {val_compact_target} tokens")
    print(f"  Hard limit:    {val_hard_limit} tokens")
    
    compacted_val, val_compact_stats = ds.compact_validation_scenarios(
        validation["scenarios"],
        val_compact_target,
        val_hard_limit,
        args.bp_per_token,
        args.seed,
    )
    
    print(f"  Compacted groups:   {val_compact_stats['num_groups']}")
    print(f"  Avg utilization:    {val_compact_stats['avg_utilization']*100:.1f}%")
    
    flattened_scenarios = [s for group in compacted_val for s in group]
    
    append_mode = args.validation_jsonl and Path(args.validation_jsonl).exists()
    
    if append_mode:
        ds.append_validation_jsonl(flattened_scenarios, val_jsonl_path, sequences)
        print(f"  Appended {len(flattened_scenarios)} scenarios to {val_jsonl_path}")
    else:
        ds.write_validation_jsonl(flattened_scenarios, val_jsonl_path, sequences)
        print(f"  Wrote {len(flattened_scenarios)} scenarios to {val_jsonl_path}")


###########################################
#####  Dynamic Chunking               #####
###########################################


print(f"\n{' Dynamic Chunking ':=^60}")
print(f"  Limit:   {args.limit/1000:.1f} kb")
print(f"  Overlap: {overlap/1000:.1f} kb ({args.overlap_ratio:.3f})")
print(f"  Step:    {(args.limit - overlap)/1000:.1f} kb")

all_validation_ids = validation["all_ids"] | existing_val_ids

train_gene_index = {
    gid: gdata for gid, gdata in gene_index.items()
    if gid not in all_validation_ids
}

print(f"  Training genes: {len(train_gene_index)}")

chunks, chunk_stats = ds.dynamic_chunking(
    sequences,
    train_gene_index,
    args.limit,
    overlap,
    anchor_pad,
)

print(f"  Created {len(chunks)} raw chunks")


###########################################
#####  Data Augmentation              #####
###########################################


print(f"\n{' Data Augmentation ':=^60}")

all_chunks = ds.augment_with_hints(chunks, args.hint_ratio, args.seed)

raw_count = sum(1 for c in all_chunks if not c.is_augmented)
aug_count = sum(1 for c in all_chunks if c.is_augmented)
total     = len(all_chunks)

if total > 0:
    print(f"  Raw:       {raw_count} ({raw_count/total*100:.1f}%)")
    print(f"  Augmented: {aug_count} ({aug_count/total*100:.1f}%)")
else:
    print("  No chunks created (sequences may be shorter than window size)")


#####################################################
#####  Input-Length Based Compacting Workflow   #####
#####################################################


compact_stats = None

if args.compact_target:
    print(f"\n{' Input-Length Compacting ':=^60}")
    print(f"  Target length: {args.compact_target} tokens")
    
    hard_limit = args.compact_hard_limit or int(args.compact_target * 1.1)
    print(f"  Hard limit:    {hard_limit} tokens")
    print(f"  BP per token:  {args.bp_per_token}")
    
    compacted_groups, compact_stats = ds.compact_to_target_length(
        all_chunks,
        args.compact_target,
        hard_limit,
        args.bp_per_token,
        args.seed,
    )
    
    all_chunks = ds.flatten_compacted_groups(compacted_groups)
    
    print(f"\n  Compacting Results:")
    print(f"    Total groups:     {compact_stats['total_groups']}")
    print(f"    Avg utilization:  {compact_stats['avg_utilization']*100:.1f}%")
    print(f"    Min utilization:  {compact_stats['min_utilization']*100:.1f}%")
    print(f"    Max utilization:  {compact_stats['max_utilization']*100:.1f}%")
    print(f"    Overflow chunks:  {compact_stats['overflow_count']}")
    print(f"    Singleton groups: {compact_stats['singleton_count']}")

else:
    print(f"\n{' Smart Compacting ':=^60}")
    
    efficiency = ds.estimate_compacting_efficiency(all_chunks)
    print(f"  Groups:          {efficiency['num_groups']}")
    print(f"  Avg utilization: {efficiency['avg_utilization']*100:.1f}%")


###########################################
#####  Write Output                   #####
###########################################


output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "train_data.gt5b"
ds.write_binary_dataset(all_chunks, output_path, args.compress)

file_size = output_path.stat().st_size

val_meta_output = output_dir / "validation_meta.json"
ds.save_validation_set(validation, val_meta_output)

run_stats = {
    "raw_count":     raw_count,
    "aug_count":     aug_count,
    "total_samples": len(all_chunks),
    "file_size":     file_size,
}

if compact_stats:
    run_stats["compact_stats"] = compact_stats

ut.print_run_stats(run_stats, chunk_stats, validation, output_path)

print(f"\n{'='*60}")
print("Done!")
print(f"  Training data:    {output_path}")
print(f"  Validation JSONL: {val_jsonl_path if validation['scenarios'] else 'N/A'}")
print(f"  Validation meta:  {val_meta_output}")
print(f"{'='*60}")