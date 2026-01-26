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
parser.add_argument("--memory_limit_pct", type=float, default=80.0,
    help="Max RAM usage percent before throttling [%(default)s]")
parser.add_argument("--memory_check_interval", type=float, default=2.0,
    help="Seconds between RAM checks when throttling [%(default)s]")
parser.add_argument("--memory_wait_timeout", type=float, default=600.0,
    help="Max seconds to wait for RAM to free up [%(default)s]")
parser.add_argument("--no_memory_limit", action="store_true",
    help="Disable RAM monitoring (not recommended)")

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

# Check RAM monitoring
use_memory_limit = not args.no_memory_limit
if use_memory_limit and not util.HAS_PSUTIL:
    print("  WARNING: psutil not installed, RAM monitoring disabled")
    print("           Install with: pip install psutil")
    use_memory_limit = False

print(f"\n{' GeneT5 Data Baker ':=^60}")
print(f"  Raw directory:   {raw_dir}")
print(f"  Baked directory: {baked_dir}")
print(f"  Log directory:   {log_dir}")
print(f"  Token file:      {token_file}")
print(f"  Workers:         {n_workers}")
print(f"  Species:         {len(species_to_process)}")

if use_memory_limit:
    mem_info = util.get_memory_info()
    print(f"\n{' Memory Configuration ':=^60}")
    print(f"  Total RAM:       {mem_info['total_gb']:.1f} GB")
    print(f"  Available RAM:   {mem_info['available_gb']:.1f} GB")
    print(f"  Current usage:   {mem_info['used_pct']:.1f}%")
    print(f"  Throttle at:     {args.memory_limit_pct:.1f}%")
    print(f"  Check interval:  {args.memory_check_interval:.1f}s")
    print(f"  Wait timeout:    {args.memory_wait_timeout:.1f}s")

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
throttle_events = 0

# Use memory-aware processing
if use_memory_limit:
    # Submit jobs one at a time with memory checks
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        pending_futures = {}
        work_queue      = list(work_items)
        
        while work_queue or pending_futures:
            # Check if we can submit more work
            while work_queue and len(pending_futures) < n_workers:
                # Check RAM before submitting
                if not util.can_submit_new_work(args.memory_limit_pct):
                    mem_pct = util.get_memory_usage_pct()
                    print(f"  ⏸ RAM at {mem_pct:.1f}%, waiting...")
                    throttle_events += 1
                    
                    if not util.wait_for_memory(
                        args.memory_limit_pct,
                        args.memory_check_interval,
                        args.memory_wait_timeout,
                    ):
                        print(f"  ⚠ RAM timeout, continuing anyway...")
                    else:
                        print(f"  ▶ RAM freed, resuming...")
                
                # Submit next item
                item   = work_queue.pop(0)
                future = executor.submit(util.process_species, item)
                pending_futures[future] = item[0]  # species_name
            
            # Wait for at least one future to complete
            if pending_futures:
                done, _ = concurrent.futures.wait(
                    pending_futures.keys(),
                    timeout     = args.memory_check_interval,
                    return_when = concurrent.futures.FIRST_COMPLETED,
                )
                
                for future in done:
                    species_name = pending_futures.pop(future)
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
                        results.append({"species": species_name, "success": False, "error": str(e)})
                        print(f"  ✗ {species_name}: {e}")

else:
    # Original behavior without memory monitoring
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
                results.append({"species": species_name, "success": False, "error": str(e)})
                print(f"  ✗ {species_name}: {e}")

print(f"\n{' Processing Results ':=^60}")
print(f"  Success: {success}")
print(f"  Failed:  {failed}")
if use_memory_limit and throttle_events > 0:
    print(f"  Throttle events: {throttle_events}")

# Track tokenizer and compact results for summary
tokenizer_result = None
compact_result   = None

if args.tokenizer and not args.skip_tokenizer_expansion and success > 0:
    print(f"\n{' Tokenizer Expansion ':=^60}")
    
    if not token_file.exists():
        print(f"  WARNING: Token file not found: {token_file}")
        tokenizer_result = {"success": False, "error": "Token file not found"}
    else:
        with open(token_file, 'r') as f:
            tokens = [line.strip() for line in f if line.strip()]
        print(f"  Tokens extracted: {len(tokens)}")
        
        if tokens:
            tokenizer_result = util.run_tokenizer_expansion(
                str(token_file),
                args.tokenizer,
            )
            
            if tokenizer_result["success"]:
                print(f"  ✓ Tokenizer expansion complete")
                if tokenizer_result.get("stdout"):
                    for line in tokenizer_result["stdout"].split('\n'):
                        if line.strip() and not line.startswith('='):
                            print(f"    {line.strip()}")
            else:
                error = tokenizer_result.get("error", "Unknown error")
                print(f"  ✗ Tokenizer expansion failed: {error}")
                if tokenizer_result.get("stderr"):
                    print(f"    {tokenizer_result['stderr']}")
        else:
            print(f"  No new tokens to add")
            tokenizer_result = {"success": True, "tokens_in_file": 0, "message": "No tokens"}

if args.compact and success > 0:
    print(f"\n{' Compacting ':=^60}")
    
    # Check RAM before compacting (can be memory intensive)
    if use_memory_limit:
        mem_pct = util.get_memory_usage_pct()
        if mem_pct > args.memory_limit_pct:
            print(f"  ⏸ RAM at {mem_pct:.1f}%, waiting before compact...")
            if util.wait_for_memory(
                args.memory_limit_pct,
                args.memory_check_interval,
                args.memory_wait_timeout,
            ):
                print(f"  ▶ RAM freed, proceeding with compact...")
            else:
                print(f"  ⚠ RAM timeout, proceeding anyway...")
    
    training_files = list(baked_dir.glob("*/training.bin"))
    
    if not training_files:
        print("  No training.bin files found!")
        compact_result = {"success": False, "error": "No training.bin files found"}
    else:
        print(f"  Found {len(training_files)} training files")
        
        output_path = baked_dir / "all_training.bin"
        compact_cmd = [
            "python3", "bin/compact.py",
        ] + [str(f) for f in training_files] + [
            "-o", str(output_path),
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
            output_size = output_path.stat().st_size if output_path.exists() else 0
            compact_result = {
                "success": True,
                "output":  str(output_path),
                "size":    f"{output_size / 1024 / 1024:.2f} MB",
            }
        else:
            print(f"  ✗ Compacting failed")
            if result.stderr:
                print(f"    {result.stderr}")
            compact_result = {"success": False, "error": result.stderr or "Unknown error"}

# Collect stats and write summary
print(f"\n{' Collecting Statistics ':=^60}")

species_stats = []
for sp, limit, taxa in species_to_process:
    stats = util.collect_species_stats(baked_dir, sp)
    stats["taxa"]  = taxa
    stats["limit"] = limit
    species_stats.append(stats)

# Build run config for summary
run_config = {
    "raw_dir":          str(raw_dir),
    "baked_dir":        str(baked_dir),
    "n_workers":        n_workers,
    "tokenizer":        args.tokenizer,
    "compact":          args.compact,
    "compact_target":   args.compact_target,
    "memory_limit_pct": args.memory_limit_pct if use_memory_limit else None,
    "throttle_events":  throttle_events if use_memory_limit else 0,
}

# Write comprehensive summary
summary_path = log_dir / "bake_summary.log"
util.write_bake_summary(
    summary_path,
    run_config,
    results,
    species_stats,
    tokenizer_result,
    compact_result,
)

print(f"  Summary written: {summary_path}")

# Print quick stats
total_chunks = sum(s["train_chunks"] for s in species_stats)
total_val    = sum(s["val_chunks"] for s in species_stats)
total_long   = sum(s["long_genes"] for s in species_stats)
total_complex = sum(s["complex_loci"] for s in species_stats)

print(f"\n{' Quick Summary ':=^60}")
print(f"  Species processed: {success}/{len(species_to_process)}")
print(f"  Training chunks:   {total_chunks:,}")
print(f"  Validation chunks: {total_val:,}")
print(f"  Long genes:        {total_long}")
print(f"  Complex loci:      {total_complex}")
if use_memory_limit:
    final_mem = util.get_memory_info()
    print(f"  Final RAM usage:   {final_mem['used_pct']:.1f}%")
    if throttle_events > 0:
        print(f"  Throttle events:   {throttle_events}")

print(f"\n{'='*60}")
print("Done!")

if args.tokenizer and not args.skip_tokenizer_expansion:
    print("\nNext steps:")
    print("  1. Run resize_model.py to resize model embeddings")
    print(f"     python bin/resize_model.py {args.tokenizer}")

print(f"{'='*60}")