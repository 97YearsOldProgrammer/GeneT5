import argparse
import json
from pathlib import Path

from lib import tuning


parser = argparse.ArgumentParser(
    description="Parse GFF3/FASTA for GeneT5 fine-tuning tasks.")
parser.add_argument("fasta",
    help="Path to FASTA file (can be multi-chromosome).")
parser.add_argument("gff",
    help="Path to GFF3 annotation file.")
parser.add_argument("output_dir",
    help="Output directory for processed datasets.")

# gene prediction args
parser.add_argument("--window_size", type=int, default=None,
    help="Window size for sliding window (None for transcript-level).")
parser.add_argument("--stride", type=int, default=None,
    help="Stride for sliding window (default: window_size).")
parser.add_argument("--gene_token", default="[ATT]",
    help="Special token for gene prediction task.")
parser.add_argument("--bos_token", default="<BOS>",
    help="Beginning of sequence token.")
parser.add_argument("--eos_token", default="<EOS>",
    help="End of sequence token.")
parser.add_argument("--gene_context_pad", type=int, default=0,
    help="Context padding around gene features (bp).")
# rna classification args
parser.add_argument("--cls_token", default="[CLS]",
    help="Classification token for RNA task.")
parser.add_argument("--rna_context_pad", type=int, default=50,
    help="Context padding around RNA features (bp).")
args = parser.parse_args()


# parse input files
print(f"\n{' GFF3/FASTA Parsing ':=^60}")
print("Parsing FASTA...")
sequences = tuning.parse_fasta(args.fasta)
print(f"  Found {len(sequences)} sequence(s): {list(sequences.keys())[:5]}...")

print("Parsing GFF3...")
features          = tuning.parse_gff(args.gff)
features_by_seqid = tuning.group_features_by_seqid(features)
print(f"  Found {len(features)} features across {len(features_by_seqid)} seqid(s)")

# count feature types
type_counts = {}
for feat in features:
    ftype = feat["type"]
    type_counts[ftype] = type_counts.get(ftype, 0) + 1
print(f"  Feature types: {dict(sorted(type_counts.items(), key=lambda x: -x[1])[:10])}")

output_dir = Path(args.output_dir)


# task 1: gene prediction
print(f"\n{'=' * 60}")
print("Creating gene prediction dataset...")
print(f"  Grouping by parent ID, filtering: {tuning.GENE_FEATURE_TYPES}")

gene_dataset = tuning.create_gene_prediction_dataset(
    sequences         = sequences,
    features_by_seqid = features_by_seqid,
    window_size       = args.window_size,
    stride            = args.stride,
    gene_token        = args.gene_token,
    bos_token         = args.bos_token,
    eos_token         = args.eos_token,
    context_pad       = args.gene_context_pad,
)

tuning.save_dataset(gene_dataset, output_dir / "gene_prediction.jsonl")

# show sample
if gene_dataset:
    sample = gene_dataset[0]
    print(f"\n  Sample entry:")
    print(f"    parent_id: {sample.get('parent_id', 'N/A')}")
    print(f"    seqid:     {sample['seqid']}")
    print(f"    span:      {sample['start']}-{sample['end']}")
    print(f"    input len: {len(sample['input'])} chars")
    print(f"    target preview:\n{sample['target'][:300]}...")


# task 2: rna classification
print(f"\n{'=' * 60}")
print("Creating RNA classification dataset...")

rna_dataset = tuning.create_rna_classification_dataset(
    sequences         = sequences,
    features_by_seqid = features_by_seqid,
    cls_token         = args.cls_token,
    context_pad       = args.rna_context_pad,
)

tuning.save_dataset(rna_dataset, output_dir / "rna_classification.jsonl")

# save label mapping
label_map_path = output_dir / "rna_labels.json"
with open(label_map_path, "w") as f:
    json.dump(tuning.RNA_CLASSES, f, indent=2)
print(f"Saved label mapping to {label_map_path}")

# show distribution
if rna_dataset:
    label_dist = {}
    for sample in rna_dataset:
        lbl = sample["label_str"]
        label_dist[lbl] = label_dist.get(lbl, 0) + 1
    print(f"  Label distribution: {label_dist}")


print(f"\n{'=' * 60}")
print("Done!")