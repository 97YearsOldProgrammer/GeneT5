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
parser.add_argument("--overlap", type=int, default=5000,
    help="Overlap in bp (default: 5000)")
parser.add_argument("--anchor_pad", type=int, default=5000,
    help="Padding before first gene (default: 5000)")

parser.add_argument("--hint_ratio", type=float, default=0.5,
    help="Ratio of samples with hints (default: 0.5)")
parser.add_argument("--seed", type=int, default=42,
    help="Random seed")

parser.add_argument("--extract_tokens", type=str, default=None,
    help="Path to txt file for extracted tokens")

parser.add_argument("--validation_file", type=str, default=None,
    help="Path to existing validation file to extend")
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

print(f"\n{' Building Validation Set ':=^60}")

validation = ds.build_validation_set(
    gene_index,
    args.long_threshold,
    args.top_k_complex,
    args.num_rare,
    args.num_easy,
    args.seed,
)

if args.validation_file and Path(args.validation_file).exists():
    existing   = ds.load_validation_set(args.validation_file)
    validation = ds.extend_validation_set(existing, validation)
    print(f"  Extended existing validation set")

print(f"\n{' Dynamic Chunking ':=^60}")
print(f"  Limit:   {args.limit/1000:.1f} kb")
print(f"  Overlap: {args.overlap/1000:.1f} kb")
print(f"  Step:    {(args.limit - args.overlap)/1000:.1f} kb")

train_gene_index = {
    gid: gdata for gid, gdata in gene_index.items()
    if gid not in validation["all_ids"]
}

chunks, chunk_stats = ds.dynamic_chunking(
    sequences,
    train_gene_index,
    args.limit,
    args.overlap,
    args.anchor_pad,
)

print(f"  Created {len(chunks)} raw chunks")

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

print(f"\n{' Smart Compacting ':=^60}")

efficiency = ds.estimate_compacting_efficiency(all_chunks)
print(f"  Groups:          {efficiency['num_groups']}")
print(f"  Avg utilization: {efficiency['avg_utilization']*100:.1f}%")

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "train_data.gt5b"
ds.write_binary_dataset(all_chunks, output_path, args.compress)

file_size = output_path.stat().st_size

val_output = output_dir / "validation_set.json"
ds.save_validation_set(validation, val_output)

run_stats = {
    "raw_count":     raw_count,
    "aug_count":     aug_count,
    "total_samples": len(all_chunks),
    "file_size":     file_size,
}

ut.print_run_stats(run_stats, chunk_stats, validation, output_path)

print(f"\n{'='*60}")
print("Done!")
print(f"  Training data:   {output_path}")
print(f"  Validation set:  {val_output}")
print(f"{'='*60}")