#!/usr/bin/env python3

import argparse
import pathlib
import multiprocessing
import gc

import lib.dataset as ds


parser = argparse.ArgumentParser(
    description='Parse GFF3/FASTA for GeneT5')
parser.add_argument('fasta', type=str, metavar='<fasta>',
    help='path to FASTA file')
parser.add_argument('gff', type=str, metavar='<gff>',
    help='path to GFF3 annotation file')
parser.add_argument('output_dir', type=str, metavar='<output>',
    help='output directory')
parser.add_argument('--limit', required=False, type=int, default=25000,
    metavar='<int>', help='chunk size limit in bp [%(default)i]')
parser.add_argument('--overlap_ratio', required=False, type=float, default=1/2.718281828,
    metavar='<float>', help='overlap ratio [%(default).4f]')
parser.add_argument('--anchor_pad_ratio', required=False, type=float, default=1/2.718281828,
    metavar='<float>', help='anchor padding ratio [%(default).4f]')
parser.add_argument('--hint_ratio', required=False, type=float, default=0.5,
    metavar='<float>', help='hint augmentation ratio [%(default).2f]')
parser.add_argument('--seed', required=False, type=int, default=42,
    metavar='<int>', help='random seed [%(default)i]')
parser.add_argument('--extract_tokens', required=False, type=str, default=None,
    metavar='<file>', help='extract tokens to file')
parser.add_argument('--top_k_long', required=False, type=int, default=5,
    metavar='<int>', help='top K long genes for validation [%(default)i]')
parser.add_argument('--top_k_complex', required=False, type=int, default=5,
    metavar='<int>', help='top K complex genes for validation [%(default)i]')
parser.add_argument('--num_normal', required=False, type=int, default=5,
    metavar='<int>', help='number of mean-complexity genes [%(default)i]')
parser.add_argument('--num_easy', required=False, type=int, default=5,
    metavar='<int>', help='number of easy genes [%(default)i]')
parser.add_argument('--n_workers', required=False, type=int, default=None,
    metavar='<int>', help='parallel workers [auto]')
parser.add_argument('--compress', action='store_true', default=True,
    help='compress output binary')
parser.add_argument('--no_compress', action='store_false', dest='compress',
    help='do not compress output binary')
parser.add_argument('--streaming', action='store_true', default=False,
    help='force streaming mode for large genomes')
parser.add_argument('--memory_limit_gb', required=False, type=float, default=8.0,
    metavar='<float>', help='auto-enable streaming above this estimated size [%(default).1f]')

args = parser.parse_args()

# Set workers
n_workers = args.n_workers or max(1, multiprocessing.cpu_count() - 1)

print(f"\n{' GFF3/FASTA Processing ':=^60}")
print(f"  Workers: {n_workers}")

# Estimate genome size and decide on streaming
estimated_size = ds.estimate_fasta_size(args.fasta)
estimated_gb   = estimated_size / (1024 ** 3)

print(f"  Estimated size: {estimated_gb:.1f} GB")

use_streaming = args.streaming or (estimated_gb > args.memory_limit_gb)

if use_streaming:
    print(f"  Mode: STREAMING (memory-efficient)")
else:
    print(f"  Mode: Standard (in-memory)")


###################################
#####  Streaming Mode (Large) #####
###################################

if use_streaming:
    # Parse GFF grouped by seqid (still need all features for gene index)
    print(f"\n{' Parsing GFF (grouped by seqid) ':=^60}")
    features_by_seqid = ds.parse_gff_by_seqid(args.gff)
    
    total_features = sum(len(f) for f in features_by_seqid.values())
    print(f"  Sequences with features: {len(features_by_seqid)}")
    print(f"  Total features: {total_features}")
    
    # Extract tokens if requested (from all features)
    if args.extract_tokens:
        all_features  = [f for feats in features_by_seqid.values() for f in feats]
        feature_types = ds.extract_feature_types(all_features)
        biotypes      = ds.extract_biotypes(all_features)
        all_types     = feature_types | biotypes
        added         = ds.append_tokens_to_txt(sorted(all_types), args.extract_tokens)
        if added:
            print(f"  Added {len(added)} new tokens")
        del all_features
    
    # Build global gene index for validation selection
    print(f"\n{' Building Global Gene Index ':=^60}")
    all_features = [f for feats in features_by_seqid.values() for f in feats]
    gene_index   = ds.build_gene_index(all_features)
    print(f"  Total genes: {len(gene_index)}")
    del all_features
    gc.collect()
    
    # Build validation set (needs global view)
    print(f"\n{' Building Validation Set ':=^60}")
    validation = ds.build_validation_set(
        gene_index,
        args.top_k_long,
        args.top_k_complex,
        args.num_normal,
        args.num_easy,
        args.seed,
    )
    print(f"  Validation genes:     {len(validation['all_ids'])}")
    print(f"  Validation scenarios: {len(validation['scenarios'])}")
    
    all_validation_ids = validation['all_ids']
    
    # Process chromosomes one at a time
    print(f"\n{' Streaming Chunking (Per-Chromosome) ':=^60}")
    
    overlap    = int(args.limit * args.overlap_ratio)
    anchor_pad = int(args.limit * args.anchor_pad_ratio)
    
    print(f"  Limit:   {args.limit/1000:.1f} kb")
    print(f"  Overlap: {overlap/1000:.1f} kb")
    
    all_chunks = []
    combined_stats = {
        "total_chunks":    0,
        "backtrack_count": 0,
        "genes_per_chunk": [],
        "chunk_sizes":     [],
    }
    
    chr_count = 0
    for seqid, sequence in ds.stream_fasta(args.fasta):
        chr_count += 1
        
        # Get features for this chromosome
        chr_features = features_by_seqid.get(seqid, [])
        if not chr_features:
            continue
        
        # Build gene index for this chromosome
        chr_gene_index = ds.build_gene_index(chr_features)
        
        # Filter out validation genes
        train_gene_index = {
            gid: gdata for gid, gdata in chr_gene_index.items()
            if gid not in all_validation_ids
        }
        
        if not train_gene_index:
            continue
        
        # Chunk this chromosome
        chr_chunks, chr_stats = ds.dynamic_chunking(
            {seqid: sequence},
            train_gene_index,
            args.limit,
            overlap,
            anchor_pad,
            n_workers=1,  # Single worker per chromosome in streaming mode
        )
        
        all_chunks.extend(chr_chunks)
        combined_stats["total_chunks"]    += chr_stats["total_chunks"]
        combined_stats["backtrack_count"] += chr_stats["backtrack_count"]
        combined_stats["genes_per_chunk"].extend(chr_stats["genes_per_chunk"])
        combined_stats["chunk_sizes"].extend(chr_stats["chunk_sizes"])
        
        # Progress
        if chr_count % 10 == 0:
            print(f"    Processed {chr_count} chromosomes, {len(all_chunks)} chunks...")
        
        # Clear chromosome data
        del sequence, chr_features, chr_gene_index, train_gene_index, chr_chunks
        gc.collect()
    
    print(f"  Processed {chr_count} chromosomes")
    print(f"  Created {len(all_chunks)} raw chunks")
    
    # Re-index chunks globally
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i
    
    chunk_stats  = combined_stats
    train_chunks = all_chunks
    
    # Clear features_by_seqid
    del features_by_seqid
    gc.collect()

# Stand Alone not into Memory
else:
    sequences  = ds.parse_fasta(args.fasta)
    features   = ds.parse_gff(args.gff)
    gene_index = ds.build_gene_index(features)

    print(f"\n  Sequences: {len(sequences)}")
    print(f"  Features:  {len(features)}")
    print(f"  Genes:     {len(gene_index)}")

    # Extract tokens if requested
    if args.extract_tokens:
        feature_types = ds.extract_feature_types(features)
        biotypes      = ds.extract_biotypes(features)
        all_types     = feature_types | biotypes
        added         = ds.append_tokens_to_txt(sorted(all_types), args.extract_tokens)
        if added:
            print(f"  Added {len(added)} new tokens")

    # Build validation set
    print(f"\n{' Building Validation Set ':=^60}")

    validation = ds.build_validation_set(
        gene_index,
        args.top_k_long,
        args.top_k_complex,
        args.num_normal,
        args.num_easy,
        args.seed,
    )

    print(f"  Validation genes:     {len(validation['all_ids'])}")
    print(f"  Validation scenarios: {len(validation['scenarios'])}")

    existing_val_ids = set()

    # Build training chunks
    print(f"\n{' Dynamic Chunking (Parallel) ':=^60}")

    overlap    = int(args.limit * args.overlap_ratio)
    anchor_pad = int(args.limit * args.anchor_pad_ratio)

    print(f"  Limit:   {args.limit/1000:.1f} kb")
    print(f"  Overlap: {overlap/1000:.1f} kb")

    all_validation_ids = validation['all_ids'] | existing_val_ids
    train_gene_index   = {
        gid: gdata for gid, gdata in gene_index.items()
        if gid not in all_validation_ids
    }

    print(f"  Training genes: {len(train_gene_index)}")

    train_chunks, chunk_stats = ds.dynamic_chunking(
        sequences, train_gene_index, args.limit, overlap, anchor_pad, n_workers
    )

    print(f"  Created {len(train_chunks)} raw chunks")


###############################
#####  Common Processing  #####
###############################

# Augment with hints (parallel)
print(f"\n{' Hint Augmentation (Parallel) ':=^60}")

all_chunks = ds.augment_with_hints(train_chunks, args.hint_ratio, args.seed, n_workers)

raw_count = sum(1 for c in all_chunks if not c.is_augmented)
aug_count = sum(1 for c in all_chunks if c.is_augmented)
print(f"  Raw chunks:       {raw_count}")
print(f"  Augmented chunks: {aug_count}")

# Write outputs
output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{' Writing Outputs ':=^60}")

# Write training
train_path = output_dir / 'training.bin'
ds.write_binary(all_chunks, train_path, args.compress)
train_size = train_path.stat().st_size
print(f"  Training:   {train_path} ({ds.format_size(train_size)})")

# Build validation chunks
# Need sequences for validation - reload if streaming
if use_streaming:
    # Reload sequences for validation (only needed genes)
    val_sequences = {}
    for seqid, sequence in ds.stream_fasta(args.fasta):
        # Check if any validation gene is on this seqid
        needs_seq = False
        for scenario in validation.get('scenarios', []):
            gene_id = scenario.get('gene_id', '')
            if gene_id in gene_index and gene_index[gene_id].get('seqid') == seqid:
                needs_seq = True
                break
        if needs_seq:
            val_sequences[seqid] = sequence
else:
    val_sequences = sequences

val_chunks = []
for scenario in validation.get('scenarios', []):
    gene_id  = scenario.get('gene_id', 'unknown')
    start    = scenario.get('start', 0)
    end      = scenario.get('end', 0)
    strand   = scenario.get('strand', '+')
    feats    = scenario.get('features', [])
    hints    = scenario.get('hints', [])
    stype    = scenario.get('scenario_type', 'unknown')

    seq = ''
    for seqid, full_seq in val_sequences.items():
        if start < len(full_seq) and end <= len(full_seq):
            seq = full_seq[start:end]
            break

    chunk = ds.BinaryChunk(
        seqid        = gene_id,
        start        = start,
        end          = end,
        strand       = strand,
        sequence     = seq,
        features     = feats,
        biotype      = stype,
        gene_ids     = [gene_id],
        has_hints    = len(hints) > 0,
        hints        = hints,
        chunk_index  = 0,
        is_augmented = False,
    )
    val_chunks.append(chunk)

# Write validation
val_path = output_dir / 'validation.bin'
ds.write_binary(val_chunks, val_path, args.compress)
val_size = val_path.stat().st_size
print(f"  Validation: {val_path} ({ds.format_size(val_size)})")

# Write validation metadata
val_meta_path = output_dir / 'validation.json'
ds.save_validation_set(validation, val_meta_path)

# Print stats
run_stats = {
    'raw_count':     raw_count,
    'aug_count':     aug_count,
    'total_samples': len(all_chunks),
    'file_size':     train_size,
}

ds.print_run_stats(run_stats, chunk_stats, validation, train_path)

print(f"\n{'='*60}")
print('Done!')
print()
print('Next steps:')
print('  1. Update tokenizer with extracted tokens (if --extract_tokens used)')
print('  2. Optionally compact with: bin/compact.py --tokenizer <path>')
print(f"{'='*60}")