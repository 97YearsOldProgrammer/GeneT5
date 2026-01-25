#!/usr/bin/env python3
"""Parse GFF3/FASTA for GeneT5 and output training.bin + validation.bin"""

import argparse as ap
import pathlib  as pl

import lib.parser     as ps
import lib.binary     as bi
import lib.chunking   as ch
import lib.compacting as co
import lib.validation as vl
import lib.util       as ut


#####################  Main Entry  #####################


def main():
    """Execute data baking pipeline"""
    
    args = parse_args()
    
    print(f"\n{' GFF3/FASTA Processing ':=^60}")
    
    sequences  = load_sequences(args)
    features   = load_features(args)
    gene_index = build_index(features)
    
    if args.extract_tokens:
        extract_tokens(features, args)
    
    validation, existing_val_ids = build_validation(gene_index, args)
    
    train_chunks, chunk_stats = build_training_chunks(
        sequences, gene_index, validation, existing_val_ids, args
    )
    
    all_chunks = augment_chunks(train_chunks, args)
    
    if args.compact_target:
        all_chunks, compact_stats = compact_training(all_chunks, args)
    else:
        compact_stats = None
    
    write_outputs(all_chunks, validation, sequences, args, chunk_stats, compact_stats)
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


#####################  Argument Parsing  #####################


def parse_args():
    """Parse command line arguments"""
    
    parser = ap.ArgumentParser(description="Parse GFF3/FASTA for GeneT5 with dynamic chunking")
    
    parser.add_argument("fasta",      help="Path to FASTA file")
    parser.add_argument("gff",        help="Path to GFF3 annotation file")
    parser.add_argument("output_dir", help="Output directory")
    
    parser.add_argument("--limit", type=int, default=25000,
        help="Window size limit in bp (default: 25000)")
    parser.add_argument("--overlap_ratio", type=float, default=1/2.718281828,
        help="Overlap ratio relative to window size (default: 1/e)")
    parser.add_argument("--anchor_pad_ratio", type=float, default=1/2.718281828,
        help="Padding ratio relative to window size for first gene (default: 1/e)")
    parser.add_argument("--hint_ratio", type=float, default=0.5,
        help="Ratio of samples with hints (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed")
    parser.add_argument("--extract_tokens", type=str, default=None,
        help="Path to txt file for extracted tokens")
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
    
    return parser.parse_args()


#####################  Loading Steps  #####################


def load_sequences(args):
    """Load FASTA sequences"""
    
    print("\nParsing FASTA...")
    sequences = ps.parse_fasta(args.fasta)
    print(f"  Found {len(sequences)} sequence(s): {list(sequences.keys())[:5]}...")
    return sequences


def load_features(args):
    """Load GFF3 features"""
    
    print("\nParsing GFF3...")
    features = ps.parse_gff(args.gff)
    print(f"  Found {len(features)} features")
    
    type_counts = {}
    for feat in features:
        ftype             = feat["type"]
        type_counts[ftype] = type_counts.get(ftype, 0) + 1
    print(f"  Feature types: {dict(sorted(type_counts.items(), key=lambda x: -x[1])[:10])}")
    
    return features


def build_index(features):
    """Build gene index from features"""
    
    print("\nBuilding gene index...")
    gene_index = ps.build_gene_index(features)
    print(f"  Found {len(gene_index)} genes")
    return gene_index


#####################  Token Extraction  #####################


def extract_tokens(features, args):
    """Extract and save new tokens"""
    
    print(f"\n{' Token Extraction ':=^60}")
    
    feature_types = ps.extract_feature_types(features)
    biotypes      = ps.extract_biotypes(features)
    all_types     = feature_types | biotypes
    
    print(f"  Feature types: {sorted(feature_types)}")
    print(f"  Biotypes:      {sorted(biotypes)}")
    
    added = ut.append_tokens_to_txt(sorted(all_types), args.extract_tokens)
    if added:
        print(f"  Added {len(added)} new tokens to {args.extract_tokens}")


#####################  Validation Building  #####################


def build_validation(gene_index, args):
    """Build validation set"""
    
    print(f"\n{' Building Validation Set ':=^60}")
    
    existing_val_ids = set()
    
    validation = vl.build_validation_set(
        gene_index,
        args.long_threshold,
        args.top_k_complex,
        args.num_rare,
        args.num_easy,
        args.seed,
    )
    
    print(f"  Validation genes:  {len(validation['all_ids'])}")
    print(f"  Validation scenarios: {len(validation['scenarios'])}")
    
    return validation, existing_val_ids


#####################  Training Chunks  #####################


def build_training_chunks(sequences, gene_index, validation, existing_val_ids, args):
    """Build training chunks via dynamic chunking"""
    
    print(f"\n{' Dynamic Chunking ':=^60}")
    
    overlap    = int(args.limit * args.overlap_ratio)
    anchor_pad = int(args.limit * args.anchor_pad_ratio)
    
    print(f"  Limit:   {args.limit/1000:.1f} kb")
    print(f"  Overlap: {overlap/1000:.1f} kb ({args.overlap_ratio:.3f})")
    print(f"  Step:    {(args.limit - overlap)/1000:.1f} kb")
    
    all_validation_ids = validation["all_ids"] | existing_val_ids
    
    train_gene_index = {
        gid: gdata for gid, gdata in gene_index.items()
        if gid not in all_validation_ids
    }
    
    print(f"  Training genes: {len(train_gene_index)}")
    
    chunks, chunk_stats = ch.dynamic_chunking(
        sequences,
        train_gene_index,
        args.limit,
        overlap,
        anchor_pad,
    )
    
    print(f"  Created {len(chunks)} raw chunks")
    
    return chunks, chunk_stats


#####################  Augmentation  #####################


def augment_chunks(chunks, args):
    """Augment chunks with hints"""
    
    print(f"\n{' Data Augmentation ':=^60}")
    
    all_chunks = ch.augment_with_hints(chunks, args.hint_ratio, args.seed)
    
    raw_count = sum(1 for c in all_chunks if not c.is_augmented)
    aug_count = sum(1 for c in all_chunks if c.is_augmented)
    total     = len(all_chunks)
    
    if total > 0:
        print(f"  Raw:       {raw_count} ({raw_count/total*100:.1f}%)")
        print(f"  Augmented: {aug_count} ({aug_count/total*100:.1f}%)")
    else:
        print("  No chunks created (sequences may be shorter than window size)")
    
    return all_chunks


#####################  Compacting  #####################


def compact_training(all_chunks, args):
    """Compact training chunks to target length"""
    
    print(f"\n{' Input-Length Compacting ':=^60}")
    print(f"  Target length: {args.compact_target} tokens")
    
    hard_limit = args.compact_hard_limit or int(args.compact_target * 1.1)
    print(f"  Hard limit:    {hard_limit} tokens")
    print(f"  BP per token:  {args.bp_per_token}")
    
    compacted_groups, compact_stats = co.compact_chunks(
        all_chunks,
        args.compact_target,
        hard_limit,
        args.bp_per_token,
        args.seed,
    )
    
    all_chunks = co.flatten_groups(compacted_groups)
    
    print(f"\n  Compacting Results:")
    print(f"    Total groups:     {compact_stats['total_groups']}")
    print(f"    Avg utilization:  {compact_stats['avg_utilization']*100:.1f}%")
    print(f"    Min utilization:  {compact_stats['min_utilization']*100:.1f}%")
    print(f"    Max utilization:  {compact_stats['max_utilization']*100:.1f}%")
    print(f"    Overflow chunks:  {compact_stats['overflow_count']}")
    print(f"    Singleton groups: {compact_stats['singleton_count']}")
    
    return all_chunks, compact_stats


#####################  Output Writing  #####################


def write_outputs(all_chunks, validation, sequences, args, chunk_stats, compact_stats):
    """Write all output files"""
    
    output_dir = pl.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{' Writing Outputs ':=^60}")
    
    # Write training binary
    train_path = output_dir / "training.bin"
    bi.write_binary(all_chunks, train_path, args.compress)
    train_size = train_path.stat().st_size
    print(f"  Training:   {train_path} ({ut.format_size(train_size)})")
    
    # Build and write validation binary
    val_chunks = build_validation_chunks(validation, sequences)
    val_path   = output_dir / "validation.bin"
    bi.write_binary(val_chunks, val_path, args.compress)
    val_size = val_path.stat().st_size
    print(f"  Validation: {val_path} ({ut.format_size(val_size)})")
    
    # Save validation metadata
    val_meta_path = output_dir / "validation_meta.json"
    vl.save_validation_set(validation, val_meta_path)
    print(f"  Val meta:   {val_meta_path}")
    
    # Print stats
    raw_count = sum(1 for c in all_chunks if not c.is_augmented)
    aug_count = sum(1 for c in all_chunks if c.is_augmented)
    
    run_stats = {
        "raw_count":     raw_count,
        "aug_count":     aug_count,
        "total_samples": len(all_chunks),
        "file_size":     train_size,
    }
    
    if compact_stats:
        run_stats["compact_stats"] = compact_stats
    
    ut.print_run_stats(run_stats, chunk_stats, validation, train_path)


def build_validation_chunks(validation, sequences):
    """Convert validation scenarios to binary chunks"""
    
    val_chunks = []
    
    for scenario in validation.get("scenarios", []):
        gene_id  = scenario.get("gene_id", "unknown")
        start    = scenario.get("start", 0)
        end      = scenario.get("end", 0)
        strand   = scenario.get("strand", "+")
        features = scenario.get("features", [])
        hints    = scenario.get("hints", [])
        stype    = scenario.get("scenario_type", "unknown")
        
        # Get sequence for this gene region from sequences dict
        seq = ""
        for seqid, full_seq in sequences.items():
            if start < len(full_seq) and end <= len(full_seq):
                seq = full_seq[start:end]
                break
        
        chunk = bi.BinaryChunk(
            seqid        = gene_id,
            start        = start,
            end          = end,
            strand       = strand,
            sequence     = seq,
            features     = features,
            biotype      = stype,
            gene_ids     = [gene_id],
            has_hints    = len(hints) > 0,
            hints        = hints,
            chunk_index  = 0,
            is_augmented = False,
        )
        val_chunks.append(chunk)
    
    return val_chunks


#####################  Script Entry  #####################


if __name__ == "__main__":
    main()
