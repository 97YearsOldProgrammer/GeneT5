import argparse
from pathlib import Path

from lib import dataset
from lib import tokenizer as tk


parser = argparse.ArgumentParser(
    description="Parse GFF3/FASTA for GeneT5 fine-tuning tasks.")
parser.add_argument("fasta",
    help="Path to FASTA file (can be multi-chromosome).")
parser.add_argument("gff",
    help="Path to GFF3 annotation file.")
parser.add_argument("output_dir",
    help="Output directory for processed datasets.")
parser.add_argument("--extract_tokens", type=str, default=None,
    help="Path to txt file to append extracted types/biotypes as tokens.")
parser.add_argument("--window_size", type=int, default=None,
    help="Window size for sliding window (None for transcript-level).")
parser.add_argument("--stride", type=int, default=None,
    help="Stride for sliding window (default: window_size // 2).")
parser.add_argument("--gene_token", default="[ATT]",
    help="Special token for gene prediction task.")
parser.add_argument("--bos_token", default="<BOS>",
    help="Beginning of sequence token.")
parser.add_argument("--eos_token", default="<EOS>",
    help="End of sequence token.")
parser.add_argument("--gene_context_pad", type=int, default=0,
    help="Context padding around gene features (bp).")
parser.add_argument("--max_gff_lines", type=int, default=1000,
    help="Maximum GFF lines per sample before chunking (~2000 tokens).")
parser.add_argument("--overlap_bp", type=int, default=500,
    help="Overlap in bp for sequence chunking to avoid splitting genes.")
parser.add_argument("--overlap_lines", type=int, default=30,
    help="Overlap in GFF lines for annotation chunking.")
args = parser.parse_args()


# parse input files
print(f"\n{' GFF3/FASTA Parsing ':=^60}")
print("Parsing FASTA...")
sequences = dataset.parse_fasta(args.fasta)
print(f"  Found {len(sequences)} sequence(s): {list(sequences.keys())[:5]}...")

print("Parsing GFF3...")
features          = dataset.parse_gff(args.gff)
features_by_seqid = dataset.group_features_by_seqid(features)
print(f"  Found {len(features)} features across {len(features_by_seqid)} seqid(s)")

# count feature types
type_counts = {}
for feat in features:
    ftype = feat["type"]
    type_counts[ftype] = type_counts.get(ftype, 0) + 1
print(f"  Feature types: {dict(sorted(type_counts.items(), key=lambda x: -x[1])[:10])}")


# extract tokens if requested
if args.extract_tokens:
    print(f"\n{' Token Extraction ':=^60}")
    
    feature_types = dataset.extract_feature_types(features)
    biotypes      = dataset.extract_biotypes(features)
    
    all_types = feature_types | biotypes
    
    print(f"  Feature types found: {sorted(feature_types)}")
    print(f"  Biotypes found:      {sorted(biotypes)}")
    print(f"  Total unique:        {len(all_types)}")
    
    # append to txt file
    added = tk.append_tokens_to_txt(sorted(all_types), args.extract_tokens)
    
    if added:
        print(f"  Added {len(added)} new tokens to {args.extract_tokens}")
        for t in added[:10]:
            print(f"    + {t}")
        if len(added) > 10:
            print(f"    ... and {len(added) - 10} more")
    else:
        print(f"  All tokens already in {args.extract_tokens}")


output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


# gene prediction dataset
print(f"\n{'=' * 60}")
print("Creating gene prediction dataset...")
print(f"  Grouping by parent ID, filtering: {dataset.GENE_FEATURE_TYPES}")
print(f"  Chunking params: max_gff_lines={args.max_gff_lines}, overlap_lines={args.overlap_lines}")
if args.window_size:
    print(f"  Sliding window: size={args.window_size}, stride={args.stride or args.window_size // 2}")

gene_dataset = dataset.create_gene_prediction_dataset_with_chunking(
    sequences,
    features_by_seqid,
    args.window_size,
    args.stride,
    args.gene_token,
    args.bos_token,
    args.eos_token,
    args.gene_context_pad,
    args.max_gff_lines,
    args.overlap_bp,
    args.overlap_lines,
)

dataset.save_dataset(gene_dataset, output_dir / "gene_prediction.jsonl")

# show sample and stats
if gene_dataset:
    sample = gene_dataset[0]
    print(f"\n  Sample entry:")
    print(f"    parent_id:   {sample.get('parent_id', 'N/A')}")
    print(f"    seqid:       {sample['seqid']}")
    print(f"    span:        {sample['start']}-{sample['end']}")
    print(f"    gene_index:  {sample.get('gene_index', 'N/A')}")
    print(f"    biotype:     {sample.get('biotype', 'N/A')}")
    print(f"    input len:   {len(sample['input'])} chars")
    print(f"    target preview:\n{sample['target'][:400]}...")
    
    # biotype distribution
    biotype_dist = {}
    for s in gene_dataset:
        bt = s.get("biotype", "unknown")
        biotype_dist[bt] = biotype_dist.get(bt, 0) + 1
    print(f"\n  Biotype distribution: {biotype_dist}")
    
    chunked_count = sum(1 for s in gene_dataset if s.get('is_chunked', False))
    if chunked_count > 0:
        print(f"\n  Chunking stats:")
        print(f"    Chunked samples:  {chunked_count}")
        print(f"    Original samples: {len(gene_dataset) - chunked_count}")


print(f"\n{'=' * 60}")
print("Done!")
print(f"  Gene prediction: {len(gene_dataset)} samples")
print(f"  Output directory: {output_dir}")
if args.extract_tokens:
    print(f"  Tokens file: {args.extract_tokens}")