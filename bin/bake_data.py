#!/usr/bin/env python3

import sys
import argparse
import subprocess
import multiprocessing
import pathlib
import concurrent.futures

import lib.util as util

parser = argparse.ArgumentParser(
    description="Bake all species data with taxa-specific parameters")
parser.add_argument("--raw_dir",   type=str, default="../raw",
    help="Directory containing species subdirectories [%(default)s]")
parser.add_argument("--baked_dir", type=str, default="../baked",
    help="Output directory for baked data [%(default)s]")
parser.add_argument("--log_dir",   type=str, default="../logs/baker",
    help="Directory for log files [%(default)s]")
parser.add_argument("--taxa",      type=str, nargs='+', default=None,
    help="Process only specific taxa (default: all)")
parser.add_argument("--species",   type=str, nargs='+', default=None,
    help="Process only specific species (default: all)")
parser.add_argument("--n_workers", type=int, default=None,
    help="Parallel workers for species processing [auto]")
parser.add_argument("--n_workers_per_species", type=int, default=1,
    help="Workers per species for chunking [%(default)s]")
parser.add_argument("--token_file", type=str, default="data/new_tokens.txt",
    help="File to collect extracted tokens [%(default)s]")
parser.add_argument("--tokenizer",  type=str, default=None,
    help="Tokenizer path for expansion and compacting")
parser.add_argument("--skip_tokenizer_expansion", action="store_true",
    help="Skip tokenizer expansion after processing")
parser.add_argument("--compact",        action="store_true",
    help="Compact all training.bin files after baking")
parser.add_argument("--compact_target", type=int, default=8192,
    help="Target tokens for compacting [%(default)s]")

args = parser.parse_args()


n_workers          = args.n_workers or max(1, multiprocessing.cpu_count() - 1)
species_to_process = util.build_species_list(args.species, args.taxa)

if not species_to_process:
    print("No species to process!")
    sys.exit(0)

raw_dir    = pathlib.Path(args.raw_dir)
baked_dir  = pathlib.Path(args.baked_dir)
log_dir    = pathlib.Path(args.log_dir)
token_file = pathlib.Path(args.token_file)

baked_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
token_file.parent.mkdir(parents=True, exist_ok=True)

print(f"\n{' GeneT5 Data Baker ':=^60}")
print(f"  Raw directory:   {raw_dir}")
print(f"  Baked directory: {baked_dir}")
print(f"  Log directory:   {log_dir}")
print(f"  Token file:      {token_file}")
print(f"  Workers:         {n_workers}")
print(f"  Species:         {len(species_to_process)}")

print(f"\n{' Taxa Summary ':=^60}")
taxa_counts = {}
for sp, limit, taxa in species_to_process:
    if taxa not in taxa_counts:
        taxa_counts[taxa] = {"count": 0, "limit": limit}
    taxa_counts[taxa]["count"] += 1

for taxa, info in taxa_counts.items():
    print(f"  {taxa:15s}: {info['count']:2d} species @ {info['limit']:,} bp")

work_items = [
    (sp, raw_dir, baked_dir, log_dir, limit, str(token_file), args.n_workers_per_species)
    for sp, limit, taxa in species_to_process
]

print(f"\n{' Processing Species ':=^60}")

results = []
success = 0
failed  = 0

with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures = {executor.submit(util.process_species, item): item[0] for item in work_items}
    
    for future in concurrent.futures.as_completed(futures):
        species_name = futures[future]
        try:
            result = future.result()
            results.append(result)
            
            if result["success"]:
                success += 1
                print(f"  ✓ {species_name}")
            else:
                failed += 1
                error = result.get("error", "Unknown error")
                print(f"  ✗ {species_name}: {error}")
        
        except Exception as e:
            failed += 1
            print(f"  ✗ {species_name}: {e}")

print(f"\n{' Processing Results ':=^60}")
print(f"  Success: {success}")
print(f"  Failed:  {failed}")

if args.tokenizer and not args.skip_tokenizer_expansion and success > 0:
    print(f"\n{' Tokenizer Expansion ':=^60}")
    
    if not token_file.exists():
        print(f"  WARNING: Token file not found: {token_file}")
    else:
        with open(token_file, 'r') as f:
            tokens = [line.strip() for line in f if line.strip()]
        print(f"  Tokens extracted: {len(tokens)}")
        
        if tokens:
            expansion_result = util.run_tokenizer_expansion(
                str(token_file),
                args.tokenizer,
            )
            
            if expansion_result["success"]:
                print(f"  ✓ Tokenizer expansion complete")
                if expansion_result.get("stdout"):
                    for line in expansion_result["stdout"].split('\n'):
                        if line.strip() and not line.startswith('='):
                            print(f"    {line.strip()}")
            else:
                error = expansion_result.get("error", "Unknown error")
                print(f"  ✗ Tokenizer expansion failed: {error}")
                if expansion_result.get("stderr"):
                    print(f"    {expansion_result['stderr']}")
        else:
            print(f"  No new tokens to add")

if args.compact and success > 0:
    print(f"\n{' Compacting ':=^60}")
    
    training_files = list(baked_dir.glob("*/training.bin"))
    
    if not training_files:
        print("  No training.bin files found!")
    else:
        print(f"  Found {len(training_files)} training files")
        
        compact_cmd = [
            "python3", "bin/compact.py",
        ] + [str(f) for f in training_files] + [
            "-o", str(baked_dir / "all_training.bin"),
        ]
        
        if args.tokenizer:
            compact_cmd.extend([
                "--tokenizer", args.tokenizer,
                "--compact_target", str(args.compact_target),
            ])
        
        print(f"  Running compact...")
        result = subprocess.run(compact_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✓ Compacting complete")
        else:
            print(f"  ✗ Compacting failed")
            if result.stderr:
                print(f"    {result.stderr}")

print(f"\n{'='*60}")
print("Done!")

if args.tokenizer and not args.skip_tokenizer_expansion:
    print("\nNext steps:")
    print("  1. Run resize_model.py to resize model embeddings")
    print(f"     python bin/resize_model.py {args.tokenizer}")

print(f"{'='*60}")