#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def main():
    
    parser = argparse.ArgumentParser(
        description="Bake all species data with taxa-specific parameters", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", type=str, default="../raw",
        help="Directory containing species subdirectories [%(default)s]")
    parser.add_argument("--baked_dir", type=str, default="../baked",
        help="Output directory for baked data [%(default)s]")
    parser.add_argument("--log_dir", type=str, default="../logs/baker",
        help="Directory for log files [%(default)s]")
    parser.add_argument("--taxa", type=str, nargs='+', default=None,
        help="Process only specific taxa (default: all)")
    parser.add_argument("--species", type=str, nargs='+', default=None,
        help="Process only specific species (default: all)")
    parser.add_argument("--n_workers", type=int, default=None,
        help="Parallel workers for species processing [auto]")
    parser.add_argument("--n_workers_per_species", type=int, default=1,
        help="Workers per species for chunking [%(default)s]")
    parser.add_argument("--compact", action="store_true",
        help="Compact all training.bin files after baking")
    parser.add_argument("--tokenizer", type=str, default=None,
        help="Tokenizer path for compacting")
    parser.add_argument("--compact_target", type=int, default=8192,
        help="Target tokens for compacting [%(default)s]")
    
    args = parser.parse_args()
    
    # Determine number of workers
    n_workers = args.n_workers or max(1, multiprocessing.cpu_count() - 1)
    
    # Build species list
    species_to_process = []
    
    if args.species:
        # Specific species requested
        for sp in args.species:
            if sp in SPECIES_LOOKUP:
                taxa, limit = SPECIES_LOOKUP[sp]
                species_to_process.append((sp, limit, taxa))
            else:
                print(f"  WARNING: Unknown species '{sp}', skipping")
    
    elif args.taxa:
        # Specific taxa requested
        for taxa in args.taxa:
            if taxa in TAXA_CONFIG:
                config = TAXA_CONFIG[taxa]
                for sp in config["species"]:
                    species_to_process.append((sp, config["limit"], taxa))
            else:
                print(f"  WARNING: Unknown taxa '{taxa}', skipping")
    
    else:
        # All species
        for taxa, config in TAXA_CONFIG.items():
            for sp in config["species"]:
                species_to_process.append((sp, config["limit"], taxa))
    
    if not species_to_process:
        print("No species to process!")
        return
    
    # Setup directories
    raw_dir   = Path(args.raw_dir)
    baked_dir = Path(args.baked_dir)
    log_dir   = Path(args.log_dir)
    
    baked_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{' GeneT5 Data Baker ':=^60}")
    print(f"  Raw directory:   {raw_dir}")
    print(f"  Baked directory: {baked_dir}")
    print(f"  Log directory:   {log_dir}")
    print(f"  Workers:         {n_workers}")
    print(f"  Species:         {len(species_to_process)}")
    
    # Print taxa summary
    print(f"\n{' Taxa Summary ':=^60}")
    taxa_counts = {}
    for sp, limit, taxa in species_to_process:
        if taxa not in taxa_counts:
            taxa_counts[taxa] = {"count": 0, "limit": limit}
        taxa_counts[taxa]["count"] += 1
    
    for taxa, info in taxa_counts.items():
        print(f"  {taxa:15s}: {info['count']:2d} species @ {info['limit']:,} bp")
    
    # Prepare work items
    work_items = [
        (sp, raw_dir, baked_dir, log_dir, limit, args.n_workers_per_species)
        for sp, limit, taxa in species_to_process
    ]
    
    # Process species in parallel
    print(f"\n{' Processing Species ':=^60}")
    
    results  = []
    success  = 0
    failed   = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_species, item): item[0] for item in work_items}
        
        for future in as_completed(futures):
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
    
    print(f"\n{' Results ':=^60}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    
    # Compact if requested
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
            subprocess.run(compact_cmd)
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
    

####################
#####  Config  #####
####################


TAXA_CONFIG = {
    "Prokaryotes": {
        "limit": 9000,
        "species": [
            "E.coli",
            "B.subtilis", 
            "C.crescentus",
            "PCC6803",
            "H.archaea",
            "V.fischeri",
        ],
    },
    "Unicellular": {
        "limit": 22500,
        "species": [
            "S.cerevisiae",
            "S.pombe",
            "C.reinhardtii",
            "N.crassa",
            "D.discoideum",
            "T.thermophila",
        ],
    },
    "Invertebrates": {
        "limit": 45000,
        "species": [
            "C.elegan",
            "Fly",
            "S.anemone",
            "S.urchin",
            "H.vulgaris",
            "Bee",
            "Silkworm",
        ],
    },
    "Vertebrates": {
        "limit": 90000,
        "species": [
            "Axolotl",
            "C.jacchus",
            "Chicken",
            "C.porcellus",
            "Frog",
            "Human",
            "Medaka",
            "Mouse",
            "Rat",
            "Zebrafish",
        ],
    },
    "Plants": {
        "limit": 45000,
        "species": [
            "Earthmoss",
            "Maize",
            "M.truncatula",
            "Rice",
            "T.cress",
        ],
    },
}

SPECIES_LOOKUP = build_species_lookup()


#######################
#####  Utilities  #####
#######################


def build_species_lookup():
    """Build species -> (taxa, limit) lookup"""
    
    lookup = {}
    for taxa, config in TAXA_CONFIG.items():
        limit = config["limit"]
        for species in config["species"]:
            lookup[species] = (taxa, limit)
    return lookup


def find_genome_files(species_dir):
    """Find GFF and FASTA files in species directory"""
    
    species_dir = Path(species_dir)
    
    gff_file   = None
    fasta_file = None
    
    for f in species_dir.iterdir():
        name_lower = f.name.lower()
        if name_lower.endswith('.gff.gz') or name_lower.endswith('.gff3.gz'):
            gff_file = f
        elif name_lower.endswith('.fna.gz') or name_lower.endswith('.fasta.gz') or name_lower.endswith('.fa.gz'):
            fasta_file = f
    
    # Fallback: check for common names
    if gff_file is None:
        for name in ['gff.gz', 'annotation.gff.gz', 'genes.gff.gz']:
            candidate = species_dir / name
            if candidate.exists():
                gff_file = candidate
                break
    
    if fasta_file is None:
        for name in ['fna.gz', 'fasta.gz', 'genome.fna.gz', 'genome.fasta.gz']:
            candidate = species_dir / name
            if candidate.exists():
                fasta_file = candidate
                break
    
    return fasta_file, gff_file


def run_parse_data(species_name, fasta_path, gff_path, output_dir, limit, log_dir, n_workers=1):
    """Run parse_data.py for a single species"""
    
    cmd = [
        "python3", "bin/parse_data.py",
        str(fasta_path),
        str(gff_path),
        str(output_dir),
        "--limit", str(limit),
        "--n_workers", str(n_workers),
        "--extract_tokens", "data/new_tokens.txt",
    ]
    
    log_file = log_dir / f"{species_name}.log"
    
    try:
        with open(log_file, 'w') as log:
            log.write(f"{'='*20} PARSING: {species_name} {'='*20}\n")
            log.write(f"FASTA: {fasta_path}\n")
            log.write(f"GFF:   {gff_path}\n")
            log.write(f"Limit: {limit} bp\n")
            log.write(f"{'='*60}\n\n")
            log.flush()
            
            result = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        
        return {
            "species":  species_name,
            "success":  result.returncode == 0,
            "log_file": str(log_file),
            "output":   str(output_dir),
        }
    
    except Exception as e:
        return {
            "species":  species_name,
            "success":  False,
            "error":    str(e),
            "log_file": str(log_file),
        }


def process_species(args):
    """Worker function for parallel species processing"""
    
    species_name, raw_dir, baked_dir, log_dir, limit, n_workers = args
    
    species_raw_dir = Path(raw_dir) / species_name
    
    if not species_raw_dir.exists():
        return {
            "species": species_name,
            "success": False,
            "error":   f"Directory not found: {species_raw_dir}",
        }
    
    fasta_file, gff_file = find_genome_files(species_raw_dir)
    
    if fasta_file is None or gff_file is None:
        return {
            "species": species_name,
            "success": False,
            "error":   f"Missing FASTA or GFF in: {species_raw_dir}",
        }
    
    output_dir = Path(baked_dir) / species_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return run_parse_data(
        species_name, fasta_file, gff_file, output_dir, limit, log_dir, n_workers
    )