import argparse
import os
import sys
import subprocess
from pathlib import Path


def main():

    parser = argparse.ArgumentParser(
        description="Run bake_data.py for multiple species.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("species", nargs='+', 
        help="List of species names (e.g., H.archaea E.coli)")
    parser.add_argument("--limit", type=int, default=25000, 
        help="Chunk size limit in bp")
    parser.add_argument("--top_k_long", type=int, default=5, 
        help="Top K long genes for validation")
    
    args = parser.parse_args()

    for species_name in args.species:
        print(f"\n{' Processing: ' + species_name + ' ':=^60}")

        raw_dir   = Path("../raw") / species_name
        fna_path  = raw_dir / "fna.gz"
        gff_path  = raw_dir / "gff.gz"
        
        baked_dir = Path("../baked") / species_name
        log_dir   = Path("../logs/baker")
        log_file  = log_dir / f"{species_name}.txt"

        if not raw_dir.exists():
            print(f"  [SKIP] Directory not found: {raw_dir}")
            continue
        
        if not fna_path.exists() or not gff_path.exists():
            print(f"  [SKIP] Missing fna.gz or gff.gz in: {raw_dir}")
            continue

        baked_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python3", "bin/bake_data.py",
            str(fna_path),
            str(gff_path),
            str(baked_dir),
            "--limit", str(args.limit),
            "--top_k_long", str(args.top_k_long),
            "--extract_tokens", "data/new_tokens.txt",
        ]

        print(f"  -> Baking to {baked_dir}...")
        try:
            with open(log_file, "a") as log:
                log.write(f"\n{'='*20} RUNNING BAKE: {species_name} {'='*20}\n")
                log.flush()

                result = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            if result.returncode == 0:
                print(f"  [SUCCESS] Done. Log: {log_file}")
            else:
                print(f"  [ERROR] Script failed (Exit code {result.returncode}). Check logs.")

        except Exception as e:
            print(f"  [EXCEPTION] Critical error: {e}")

if __name__ == "__main__":
    main()