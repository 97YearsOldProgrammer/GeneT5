import argparse
import pathlib
import multiprocessing
import json

import pyfaidx

import lib.dataset as ds


parser = argparse.ArgumentParser(
    description='Parse GFF3/FASTA for GeneT5')
parser.add_argument('fasta', type=str, metavar='<fasta>',
    help='path to FASTA file')
parser.add_argument('gff', type=str, metavar='<gff>',
    help='path to GFF3 annotation file')
parser.add_argument('output_dir', type=str, metavar='<output>',
    help='output directory')
parser.add_argument('--limit', required=False, type=int, default=20000,
    metavar='<int>', help='chunk size limit in bp [%(default)i]')
parser.add_argument('--overlap_ratio', required=False, type=float, default=1/2.718281828,
    metavar='<float>', help='step ratio: fraction of window as new territory (1/e) [%(default).4f]')
parser.add_argument('--max_n_ratio', required=False, type=float, default=0.1,
    metavar='<float>', help='max N ratio before skipping window [%(default).2f]')
parser.add_argument('--hint_ratio', required=False, type=float, default=0.5,
    metavar='<float>', help='hint augmentation ratio [%(default).2f]')
parser.add_argument('--seed', required=False, type=int, default=42,
    metavar='<int>', help='random seed [%(default)i]')
parser.add_argument('--n_workers', required=False, type=int, default=None,
    metavar='<int>', help='parallel workers [auto]')
parser.add_argument('--tokenizer', required=False, type=str, default=None,
    metavar='<path>', help='tokenizer path for storing token lengths')
parser.add_argument('--compress', required=False, type=str, default=None,
    choices=['zlib', 'zstd'], help='compress output with zlib or zstd')
# Kept as CLI flags so _databaker.py can pass them, but always-on in practice
parser.add_argument('--canonical_only', action='store_true', default=True,
    help=argparse.SUPPRESS)
parser.add_argument('--fast_tokenizer', action='store_true', default=True,
    help=argparse.SUPPRESS)

args = parser.parse_args()

# Set workers
n_workers = args.n_workers or max(1, multiprocessing.cpu_count() - 1)

print(f"\n{' GFF3/FASTA Processing ':=^60}")
print(f"  Workers: {n_workers}")

gene_index = ds.parse_gff_to_gene_index(args.gff)
ds.filter_canonical_transcripts(gene_index)

# Save gene_index sidecar (avoids re-parsing GFF for eval)
output_dir = pathlib.Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
ds.save_gene_index(gene_index, output_dir / "gene_index.json")

# Brief open to count sequences
fasta     = pyfaidx.Fasta(args.fasta, as_raw=True, sequence_always_upper=True)
n_seqids  = len(fasta.keys())
fasta.close()

print(f"\n  Sequences: {n_seqids}")
print(f"  Genes:     {len(gene_index)}")
print(f"  Mode:      canonical only (one transcript per gene)")
print(f"  Saved:     gene_index.json sidecar")

# Sliding window chunking on ALL genes
print(f"\n{' Sliding Window Chunking ':=^60}")

print(f"  Window size:   {args.limit/1000:.1f} kb")
print(f"  Overlap ratio: {args.overlap_ratio:.4f} (1/e)")
print(f"  Max N ratio:   {args.max_n_ratio*100:.0f}%")

all_raw_chunks, chunk_stats = ds.sliding_window_chunking(
    args.fasta,
    gene_index,
    window_size   = args.limit,
    overlap_ratio = args.overlap_ratio,
    max_n_ratio   = args.max_n_ratio,
    n_workers     = n_workers,
)

print(f"\n  Windows scanned: {chunk_stats['windows_scanned']:,}")
print(f"  Windows empty:   {chunk_stats['windows_empty']:,}")
print(f"  Windows N-heavy: {chunk_stats['windows_n_heavy']:,}")
print(f"  Windows kept:    {chunk_stats['windows_kept']:,}")

if chunk_stats['features_per_chunk']:
    avg_feat = sum(chunk_stats['features_per_chunk']) / len(chunk_stats['features_per_chunk'])
    print(f"  Avg features:    {avg_feat:.1f}")

# Augment with hints (parallel)
print(f"\n{' Hint Augmentation (Parallel) ':=^60}")

all_chunks = ds.augment_with_hints(all_raw_chunks, args.hint_ratio, args.seed, n_workers)

raw_count = sum(1 for c in all_chunks if not c.is_augmented)
aug_count = sum(1 for c in all_chunks if c.is_augmented)
print(f"  Raw chunks:       {raw_count}")
print(f"  Augmented chunks: {aug_count}")

# Tokenize — always use fast tokenizer (Rust backend, ~50ms load)
tokenizer = None
if args.tokenizer:
    print(f"\n{' Batch Tokenizing Chunks ':=^60}")

    from tokenizers import Tokenizer as FastTokenizer
    tok_json = pathlib.Path(args.tokenizer) / "tokenizer.json"
    if not tok_json.exists():
        tok_json = pathlib.Path(args.tokenizer)
    tokenizer = FastTokenizer.from_file(str(tok_json))
    print(f"  Vocab size: {tokenizer.get_vocab_size()} (fast tokenizer)")

    batch_size   = 1000
    total_chunks = len(all_chunks)

    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch     = all_chunks[batch_start:batch_end]

        input_texts  = [c.get_input_text() for c in batch]
        target_texts = [c.get_target_text() for c in batch]

        input_enc  = tokenizer.encode_batch(input_texts, add_special_tokens=False)
        target_enc = tokenizer.encode_batch(target_texts, add_special_tokens=False)

        for i, chunk in enumerate(batch):
            chunk.input_ids  = input_enc[i].ids
            chunk.target_ids = target_enc[i].ids
            chunk.input_len  = len(chunk.input_ids)
            chunk.target_len = len(chunk.target_ids)

        pct = 100 * batch_end / total_chunks
        print(f"    {batch_end:,}/{total_chunks:,} ({pct:.1f}%)", end='\r')

    print(f"    {total_chunks:,}/{total_chunks:,} (100.0%)")

# Write output
print(f"\n{' Writing Output ':=^60}")

train_path = output_dir / 'training.bin'
ds.write_binary(all_chunks, train_path, compress=args.compress)
train_size = train_path.stat().st_size
print(f"  training.bin: {train_path} ({ds.format_size(train_size)})")

# Print stats
run_stats = {
    'raw_count':     raw_count,
    'aug_count':     aug_count,
    'total_samples': len(all_chunks),
    'file_size':     train_size,
}

ds.print_run_stats(run_stats, chunk_stats, 0, train_path)

# Write stats JSON
stats_path = output_dir / 'stats.json'
stats_json = {
    'run_stats':   run_stats,
    'chunk_stats': {k: v for k, v in chunk_stats.items() if k != 'features_per_chunk'},
}
with open(stats_path, 'w') as f:
    json.dump(stats_json, f, indent=2)

print(f"\n{'='*60}")
print('Done!')
print(f"{'='*60}")
