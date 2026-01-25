#!/usr/bin/env python3

import argparse
import pathlib

import lib.dataset as ds


def main():

    args = parse_args()

    print(f"\n{' GFF3/FASTA Processing ':=^60}")

    sequences  = ds.parse_fasta(args.fasta)
    features   = ds.parse_gff(args.gff)
    gene_index = ds.build_gene_index(features)

    print(f"\n  Sequences: {len(sequences)}")
    print(f"  Features:  {len(features)}")
    print(f"  Genes:     {len(gene_index)}")

    if args.extract_tokens:
        feature_types = ds.extract_feature_types(features)
        biotypes      = ds.extract_biotypes(features)
        all_types     = feature_types | biotypes
        added         = ds.append_tokens_to_txt(sorted(all_types), args.extract_tokens)
        if added:
            print(f"  Added {len(added)} new tokens")

    validation, existing_val_ids = build_validation(gene_index, args)

    train_chunks, chunk_stats = build_training_chunks(
        sequences, gene_index, validation, existing_val_ids, args
    )

    all_chunks = ds.augment_with_hints(train_chunks, args.hint_ratio, args.seed)

    raw_count = sum(1 for c in all_chunks if not c.is_augmented)
    aug_count = sum(1 for c in all_chunks if c.is_augmented)
    print(f"\n  Raw chunks:       {raw_count}")
    print(f"  Augmented chunks: {aug_count}")

    if args.compact_target:
        all_chunks, compact_stats = compact_training(all_chunks, args)
    else:
        compact_stats = None

    write_outputs(all_chunks, validation, sequences, args, chunk_stats, compact_stats)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


def parse_args():

    parser = argparse.ArgumentParser(description="Parse GFF3/FASTA for GeneT5")

    parser.add_argument("fasta",      help="Path to FASTA file")
    parser.add_argument("gff",        help="Path to GFF3 annotation file")
    parser.add_argument("output_dir", help="Output directory")

    parser.add_argument("--limit",            type=int,   default=25000)
    parser.add_argument("--overlap_ratio",    type=float, default=1/2.718281828)
    parser.add_argument("--anchor_pad_ratio", type=float, default=1/2.718281828)
    parser.add_argument("--hint_ratio",       type=float, default=0.5)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--extract_tokens",   type=str,   default=None)
    parser.add_argument("--long_threshold",   type=int,   default=50000)
    parser.add_argument("--top_k_complex",    type=int,   default=5)
    parser.add_argument("--num_rare",         type=int,   default=10)
    parser.add_argument("--num_easy",         type=int,   default=10)
    parser.add_argument("--compress",         action="store_true", default=True)
    parser.add_argument("--no_compress",      action="store_false", dest="compress")
    parser.add_argument("--compact_target",   type=int,   default=None)
    parser.add_argument("--compact_hard_limit", type=int, default=None)
    parser.add_argument("--bp_per_token",     type=float, default=4.5)

    return parser.parse_args()


def build_validation(gene_index, args):

    print(f"\n{' Building Validation Set ':=^60}")

    validation = ds.build_validation_set(
        gene_index,
        args.long_threshold,
        args.top_k_complex,
        args.num_rare,
        args.num_easy,
        args.seed,
    )

    print(f"  Validation genes:     {len(validation['all_ids'])}")
    print(f"  Validation scenarios: {len(validation['scenarios'])}")

    return validation, set()


def build_training_chunks(sequences, gene_index, validation, existing_val_ids, args):

    print(f"\n{' Dynamic Chunking ':=^60}")

    overlap    = int(args.limit * args.overlap_ratio)
    anchor_pad = int(args.limit * args.anchor_pad_ratio)

    print(f"  Limit:   {args.limit/1000:.1f} kb")
    print(f"  Overlap: {overlap/1000:.1f} kb")

    all_validation_ids = validation["all_ids"] | existing_val_ids
    train_gene_index   = {
        gid: gdata for gid, gdata in gene_index.items()
        if gid not in all_validation_ids
    }

    print(f"  Training genes: {len(train_gene_index)}")

    chunks, chunk_stats = ds.dynamic_chunking(
        sequences, train_gene_index, args.limit, overlap, anchor_pad
    )

    print(f"  Created {len(chunks)} raw chunks")

    return chunks, chunk_stats


def compact_training(all_chunks, args):

    print(f"\n{' Input-Length Compacting ':=^60}")

    hard_limit = args.compact_hard_limit or int(args.compact_target * 1.1)

    print(f"  Target: {args.compact_target} tokens")
    print(f"  Hard:   {hard_limit} tokens")

    compacted_groups, compact_stats = ds.compact_chunks(
        all_chunks, args.compact_target, hard_limit, args.bp_per_token, args.seed
    )

    all_chunks = ds.flatten_groups(compacted_groups)

    print(f"  Groups:      {compact_stats['total_groups']}")
    print(f"  Utilization: {compact_stats['avg_utilization']*100:.1f}%")

    return all_chunks, compact_stats


def write_outputs(all_chunks, validation, sequences, args, chunk_stats, compact_stats):

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{' Writing Outputs ':=^60}")

    train_path = output_dir / "training.bin"
    ds.write_binary(all_chunks, train_path, args.compress)
    train_size = train_path.stat().st_size
    print(f"  Training:   {train_path} ({ds.format_size(train_size)})")

    val_chunks = build_validation_chunks(validation, sequences)
    val_path   = output_dir / "validation.bin"
    ds.write_binary(val_chunks, val_path, args.compress)
    val_size = val_path.stat().st_size
    print(f"  Validation: {val_path} ({ds.format_size(val_size)})")

    val_meta_path = output_dir / "validation_meta.json"
    ds.save_validation_set(validation, val_meta_path)
    print(f"  Val meta:   {val_meta_path}")

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

    ds.print_run_stats(run_stats, chunk_stats, validation, train_path)


def build_validation_chunks(validation, sequences):

    val_chunks = []

    for scenario in validation.get("scenarios", []):
        gene_id  = scenario.get("gene_id", "unknown")
        start    = scenario.get("start", 0)
        end      = scenario.get("end", 0)
        strand   = scenario.get("strand", "+")
        features = scenario.get("features", [])
        hints    = scenario.get("hints", [])
        stype    = scenario.get("scenario_type", "unknown")

        seq = ""
        for seqid, full_seq in sequences.items():
            if start < len(full_seq) and end <= len(full_seq):
                seq = full_seq[start:end]
                break

        chunk = ds.BinaryChunk(
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


if __name__ == "__main__":
    main()