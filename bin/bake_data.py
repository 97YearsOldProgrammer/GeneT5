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
parser.add_argument("--tokenizer_config", type=str, default=None,
    help="Path to tokenizer.json to check for missing RNA tokens.")
parser.add_argument("--update_tokenizer", action="store_true",
    help="If set, append missing RNA tokens to tokenizer config.")
parser.add_argument("--tokenizer_output", type=str, default=None,
    help="Output path for updated tokenizer (default: same as input).")
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
parser.add_argument("--max_gff_lines", type=int, default=400,
    help="Maximum GFF lines per sample before chunking (~2000 tokens).")
parser.add_argument("--overlap_bp", type=int, default=50,
    help="Overlap in bp for sequence chunking to avoid splitting genes.")
parser.add_argument("--overlap_lines", type=int, default=20,
    help="Overlap in GFF lines for annotation chunking.")
parser.add_argument("--cls_token", default="[CLS]",
    help="Classification token for RNA task.")
parser.add_argument("--rna_context_pad", type=int, default=50,
    help="Context padding around RNA features (bp).")
parser.add_argument("--skip_ncrna", action="store_true",
    help="Skip ncRNA in classification (use when ncRNA encompasses other types).")
parser.add_argument("--include_ncrna", action="store_true",
    help="Explicitly include ncRNA in classification.")
parser.add_argument("--rna_classes_file", type=str, default=None,
    help="JSON file with custom RNA class mapping.")
args = parser.parse_args()


# determine ncRNA inclusion
include_ncrna = False
if args.include_ncrna:
    include_ncrna = True
elif args.skip_ncrna:
    include_ncrna = False

# load custom RNA classes if provided
rna_classes = tuning.RNA_CLASSES.copy()
if args.rna_classes_file:
    print(f"\nLoading custom RNA classes from {args.rna_classes_file}")
    with open(args.rna_classes_file) as f:
        rna_classes = json.load(f)
    print(f"  Loaded {len(rna_classes)} RNA classes")

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
output_dir.mkdir(parents=True, exist_ok=True)


# tokenizer check and update
if args.tokenizer_config:
    print(f"\n{' Tokenizer Check ':=^60}")
    tokenizer_output = args.tokenizer_output or args.tokenizer_config
    
    if args.update_tokenizer:
        config, added_tokens = tuning.update_tokenizer_with_rna_classes(
            tokenizer_path=args.tokenizer_config,
            rna_classes=rna_classes,
            output_path=tokenizer_output
        )
        if added_tokens:
            print(f"  Added tokens: {added_tokens}")
            tokens_file = output_dir / "added_rna_tokens.txt"
            with open(tokens_file, 'w') as f:
                for token in added_tokens:
                    f.write(f"{token}\n")
            print(f"  Saved token list to {tokens_file}")
    else:
        config  = tuning.load_tokenizer_config(args.tokenizer_config)
        missing = tuning.find_missing_rna_tokens(config, rna_classes)
        if missing:
            print(f"  Missing RNA tokens: {missing}")
            print(f"  Run with --update_tokenizer to add them")
        else:
            print(f"  All RNA tokens present in tokenizer")


# task 1: gene prediction
print(f"\n{'=' * 60}")
print("Creating gene prediction dataset...")
print(f"  Grouping by parent ID, filtering: {tuning.GENE_FEATURE_TYPES}")
print(f"  Chunking params: max_gff_lines={args.max_gff_lines}, overlap_lines={args.overlap_lines}")
if args.window_size:
    print(f"  Sliding window: size={args.window_size}, stride={args.stride or args.window_size // 2}")

gene_dataset = tuning.create_gene_prediction_dataset_with_chunking(
    sequences         = sequences,
    features_by_seqid = features_by_seqid,
    window_size       = args.window_size,
    stride            = args.stride,
    gene_token        = args.gene_token,
    bos_token         = args.bos_token,
    eos_token         = args.eos_token,
    context_pad       = args.gene_context_pad,
    max_gff_lines     = args.max_gff_lines,
    overlap_bp        = args.overlap_bp,
    overlap_lines     = args.overlap_lines,
)

tuning.save_dataset(gene_dataset, output_dir / "gene_prediction.jsonl")

# show sample and chunking stats
if gene_dataset:
    sample = gene_dataset[0]
    print(f"\n  Sample entry:")
    print(f"    parent_id: {sample.get('parent_id', 'N/A')}")
    print(f"    seqid:     {sample['seqid']}")
    print(f"    span:      {sample['start']}-{sample['end']}")
    print(f"    input len: {len(sample['input'])} chars")
    print(f"    target preview:\n{sample['target'][:300]}...")
    
    chunked_count = sum(1 for s in gene_dataset if s.get('is_chunked', False))
    if chunked_count > 0:
        print(f"\n  Chunking stats:")
        print(f"    Chunked samples:  {chunked_count}")
        print(f"    Original samples: {len(gene_dataset) - chunked_count}")


# task 2: rna classification
print(f"\n{'=' * 60}")
print("Creating RNA classification dataset...")
print(f"  include_ncrna: {include_ncrna}")
print(f"  RNA classes:   {len(rna_classes)}")

rna_dataset = tuning.create_rna_classification_dataset(
    sequences         = sequences,
    features_by_seqid = features_by_seqid,
    cls_token         = args.cls_token,
    context_pad       = args.rna_context_pad,
    include_ncrna     = include_ncrna,
    rna_classes       = rna_classes,
)

tuning.save_dataset(rna_dataset, output_dir / "rna_classification.jsonl")

# save label mapping
label_map_path = output_dir / "rna_labels.json"
with open(label_map_path, "w") as f:
    json.dump(rna_classes, f, indent=2)
print(f"  Saved label mapping to {label_map_path}")

# show distribution
if rna_dataset:
    label_dist = {}
    for sample in rna_dataset:
        lbl = sample["label_str"]
        label_dist[lbl] = label_dist.get(lbl, 0) + 1
    print(f"  Label distribution: {label_dist}")


print(f"\n{'=' * 60}")
print("Done!")
print(f"  Gene prediction:    {len(gene_dataset)} samples")
print(f"  RNA classification: {len(rna_dataset)} samples")
print(f"  Output directory:   {output_dir}")