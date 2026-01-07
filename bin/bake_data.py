
import argparse
from pathlib import Path
import json

from lib import tuning

parser = argparse.ArgumentParser(
    description="Parse GFF3/FASTA for GeneT5 fine-tuning tasks.")
parser.add_argument("fasta", type=str,
    help="Path to FASTA file (can be multi-chromosome).")
parser.add_argument("gff", type=str,
    help="Path to GFF3 annotation file.")
parser.add_argument("--output_dir", type=str, default="./data/processed",
    help="Output directory for processed datasets.")
parser.add_argument("--task", type=str, default="both",
    choices=["gene", "rna", "both"],
    help="Which task to prepare: gene prediction, rna classification, or both.")
parser.add_argument("--window_size", type=int, default=None,
    help="Window size for sliding window (None for full sequence).")
parser.add_argument("--stride", type=int, default=None,
    help="Stride for sliding window (default: window_size).")
parser.add_argument("--gene_token", type=str, default="[ATT]",
    help="Special token for gene prediction task.")
parser.add_argument("--cls_token", type=str, default="[CLS]",
    help="Classification token for RNA task.")
parser.add_argument("--context_pad", type=int, default=50,
    help="Context padding around RNA features (bp).")
args = parser.parse_args()


# Parse input files
print(f"\n{' GFF3/FASTA Parsing ':=^60}")
print("Parsing FASTA...")
sequences = tuning.parse_fasta(args.fasta)
print(f"  Found {len(sequences)} sequence(s): {list(sequences.keys())[:5]}...")

print("Parsing GFF3...")
features          = tuning.parse_gff(args.gff)
features_by_seqid = tuning.group_features_by_seqid(features)
print(f"  Found {len(features)} features across {len(features_by_seqid)} seqid(s)")

output_dir = Path(args.output_dir)

# Task 1: Gene Prediction
if args.task in ["gene", "both"]:
    print(f"\n{'=' * 60}")
    print("Creating gene prediction dataset...")
    gene_dataset = tuning.create_gene_prediction_dataset(
        sequences         = sequences,
        features_by_seqid = features_by_seqid,
        window_size       = args.window_size,
        stride            = args.stride,
        gene_token        = args.gene_token,
    )
    tuning.save_dataset(gene_dataset, output_dir / "gene_prediction.jsonl")

# Task 2: RNA Classification
if args.task in ["rna", "both"]:
    print(f"\n{'=' * 60}")
    print("Creating RNA classification dataset...")
    rna_dataset = tuning.create_rna_classification_dataset(
        sequences         = sequences,
        features_by_seqid = features_by_seqid,
        cls_token         = args.cls_token,
        context_pad       = args.context_pad,
    )
    tuning.save_dataset(rna_dataset, output_dir / "rna_classification.jsonl")
    
    # Save label mapping
    label_map_path = output_dir / "rna_labels.json"
    with open(label_map_path, "w") as f:
        json.dump(tuning.RNA_CLASSES, f, indent=2)
    print(f"Saved label mapping to {label_map_path}")

print(f"\n{'=' * 60}")
print("Done!")