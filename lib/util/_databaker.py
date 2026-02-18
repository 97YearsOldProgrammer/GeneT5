import subprocess
import pathlib
import tempfile
import shutil
import json
import os
from dataclasses import dataclass, field

import lib.dataset as ds


############################
#####  Pre-Decompress  #####
############################


def decompress_to_temp(gz_path, temp_dir=None):
    """
    Decompress gzipped file to temp location for faster parsing

    Uses pigz if available (parallel), falls back to gunzip
    Returns path to decompressed file
    """
    gz_path = pathlib.Path(gz_path)

    if not str(gz_path).endswith('.gz'):
        return gz_path

    if temp_dir is None:
        temp_dir = pathlib.Path(tempfile.gettempdir())
    else:
        temp_dir = pathlib.Path(temp_dir)

    # Output filename without .gz
    out_name = gz_path.name[:-3]
    out_path = temp_dir / out_name

    # Try pigz first (parallel, much faster), then gzip
    try:
        # Check if pigz is available
        result = subprocess.run(['which', 'pigz'], capture_output=True)
        if result.returncode == 0:
            subprocess.run(
                ['pigz', '-d', '-k', '-c', str(gz_path)],
                stdout=open(out_path, 'wb'),
                check=True
            )
        else:
            subprocess.run(
                ['gunzip', '-c', str(gz_path)],
                stdout=open(out_path, 'wb'),
                check=True
            )
        return out_path
    except Exception:
        # Fallback to Python gzip
        import gzip
        with gzip.open(gz_path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return out_path


############################
#####  Auto-Discovery  #####
############################


def discover_species(raw_dir):
    """
    Scan raw_dir for valid species (dirs containing fna.gz + gff.gz).

    Returns list of (species_name, raw_path) tuples sorted by name.
    Warns about invalid dirs (missing files, common-name duplicates, etc.).
    """

    raw_dir = pathlib.Path(raw_dir)
    valid   = []
    skipped = []

    for entry in sorted(raw_dir.iterdir()):
        if not entry.is_dir():
            continue

        name = entry.name

        # Skip hidden dirs
        if name.startswith('.'):
            continue

        # Check for genome files
        has_fna = False
        has_gff = False

        for f in entry.iterdir():
            fn = f.name.lower()
            if fn.endswith(('.fna.gz', '.fasta.gz', '.fa.gz')) or fn == 'fna.gz':
                has_fna = True
            elif fn.endswith(('.gff.gz', '.gff3.gz')) or fn == 'gff.gz':
                has_gff = True

        if has_fna and has_gff:
            valid.append((name, entry))
        else:
            missing = []
            if not has_fna:
                missing.append("fna.gz")
            if not has_gff:
                missing.append("gff.gz")
            skipped.append((name, missing))

    if skipped:
        print(f"  Skipped {len(skipped)} invalid dirs:")
        for name, missing in skipped:
            print(f"    {name}: missing {', '.join(missing)}")

    return valid


##############################
#####  BakeJob Dataclass #####
##############################


@dataclass
class BakeJob:
    """Work item for species processing — replaces fragile tuples"""

    species:       str = ""
    raw_dir:       str = ""
    output_dir:    str = ""
    log_dir:       str = ""
    window_size:   int = 20000
    tokenizer:     str = None
    n_workers:     int = 1
    compress:      str = None
    output_format: str = "binary"


######################
#####  Utilities #####
######################


def find_genome_files(species_dir):
    """Find GFF and FASTA files in species directory"""
    
    species_dir = pathlib.Path(species_dir)
    
    gff_file   = None
    fasta_file = None
    
    for f in species_dir.iterdir():
        # Skip hidden files and macOS resource forks (._*)
        if f.name.startswith('.'):
            continue
        
        name_lower = f.name.lower()
        if name_lower.endswith('.gff.gz') or name_lower.endswith('.gff3.gz'):
            gff_file = f
        elif name_lower.endswith('.fna.gz') or name_lower.endswith('.fasta.gz') or name_lower.endswith('.fa.gz'):
            fasta_file = f
    
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


#################################
#####  Species Processing  ######
#################################


def run_parse_data(species_name, fasta_path, gff_path, output_dir, limit, log_dir, token_file=None, tokenizer_path=None, n_workers=1, compress=None):
    """Run parse_data.py for a single species"""

    cmd = [
        "python3", "bin/parse_data.py",
        str(fasta_path),
        str(gff_path),
        str(output_dir),
        "--limit", str(limit),
        "--n_workers", str(n_workers),
        "--canonical_only",
        "--fast_tokenizer",
    ]

    if token_file:
        cmd.extend(["--extract_tokens", token_file])

    if tokenizer_path:
        cmd.extend(["--tokenizer", str(tokenizer_path)])

    if compress:
        cmd.extend(["--compress", compress])

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
                stdout = log,
                stderr = subprocess.STDOUT,
                text   = True,
            )
        
        # Extract error from log if failed
        error_msg = None
        if result.returncode != 0:
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                
                # Filter out header lines
                content_lines = [l.strip() for l in lines if l.strip() and not l.startswith('=')]
                
                # Check if only header present (crashed immediately)
                if len(content_lines) <= 4:
                    error_msg = "Crashed immediately (likely OOM or file error)"
                else:
                    # Look for traceback or error
                    for i, line in enumerate(lines):
                        if 'Error' in line or 'Exception' in line or 'Killed' in line:
                            error_msg = line.strip()[:100]
                            break
                        if 'MemoryError' in line:
                            error_msg = "MemoryError (OOM)"
                            break
                    
                    # Fallback: get last non-empty line
                    if error_msg is None:
                        non_empty = [l.strip() for l in lines if l.strip()]
                        error_msg = non_empty[-1][:100] if non_empty else f"Exit code {result.returncode}"
            except Exception:
                error_msg = f"Exit code {result.returncode}"
        
        return {
            "species":  species_name,
            "success":  result.returncode == 0,
            "error":    error_msg,
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


def convert_binary_to_tar(binary_path, species_name, maxcount=50000):
    """Convert training.bin to WebDataset tar shards, return sample count"""

    import webdataset as wds

    binary_path = pathlib.Path(binary_path)
    shard_dir   = binary_path.parent
    pattern     = str(shard_dir / f"{species_name}-%06d.tar")

    total_written = 0
    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        for idx, chunk in enumerate(ds.iter_binary(binary_path)):
            key = f"{chunk.seqid}_{chunk.start}_{chunk.end}_{idx:06d}"

            meta = {
                "seqid":    chunk.seqid,
                "start":    chunk.start,
                "end":      chunk.end,
                "gene_ids": chunk.gene_ids,
            }

            sink.write({
                "__key__":    key,
                "input.txt":  chunk.get_input_text(),
                "target.txt": chunk.get_target_text(),
                "meta.json":  json.dumps(meta),
            })
            total_written += 1

    # Remove the binary after successful conversion
    binary_path.unlink()

    return total_written


def process_species(job):
    """Worker function for parallel species processing

    Accepts a BakeJob dataclass instance
    """

    species_raw_dir = pathlib.Path(job.raw_dir) / job.species

    if not species_raw_dir.exists():
        return {
            "species": job.species,
            "success": False,
            "error":   f"Directory not found: {species_raw_dir}",
        }

    fasta_file, gff_file = find_genome_files(species_raw_dir)

    if fasta_file is None or gff_file is None:
        return {
            "species": job.species,
            "success": False,
            "error":   f"Missing FASTA or GFF in: {species_raw_dir}",
        }

    output_dir = pathlib.Path(job.output_dir) / job.species
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-decompress gzipped files for faster parsing
    decompressed_files = []
    try:
        if str(fasta_file).endswith('.gz'):
            fasta_file = decompress_to_temp(fasta_file, output_dir)
            decompressed_files.append(fasta_file)
        if str(gff_file).endswith('.gz'):
            gff_file = decompress_to_temp(gff_file, output_dir)
            decompressed_files.append(gff_file)
    except Exception:
        pass

    result = run_parse_data(
        job.species, fasta_file, gff_file, output_dir, job.window_size,
        pathlib.Path(job.log_dir), None, job.tokenizer, job.n_workers,
        job.compress,
    )

    # Clean up decompressed files
    for f in decompressed_files:
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass

    # Convert binary to tar shards if requested and parse succeeded
    if result["success"] and job.output_format == "tar":
        binary_path = output_dir / "training.bin"
        if binary_path.exists():
            try:
                total = convert_binary_to_tar(binary_path, job.species)
                result["total_samples"] = total
            except Exception as e:
                result["success"] = False
                result["error"]   = f"Tar conversion failed: {e}"

    return result


####################################
#####  Tokenizer Expansion  ########
####################################


def run_tokenizer_expansion(token_file, tokenizer_path, output_path=None, dry_run=False):
    """Run append_tk.py to expand tokenizer with new tokens"""
    
    token_path = pathlib.Path(token_file)
    
    if not token_path.exists():
        return {
            "success": False,
            "error":   f"Token file not found: {token_file}",
        }
    
    with open(token_path, 'r') as f:
        tokens = [line.strip() for line in f if line.strip()]
    
    if not tokens:
        return {
            "success": True,
            "added":   0,
            "message": "No tokens to add (file empty)",
        }
    
    cmd = [
        "python3", "bin/append_tk.py",
        token_file,
        tokenizer_path,
    ]
    
    if output_path:
        cmd.extend(["--output", output_path])
    
    if dry_run:
        cmd.append("--dry_run")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output = True,
            text           = True,
        )
        
        return {
            "success":        result.returncode == 0,
            "stdout":         result.stdout,
            "stderr":         result.stderr,
            "tokens_in_file": len(tokens),
        }
    
    except Exception as e:
        return {
            "success": False,
            "error":   str(e),
        }


############################
#####  Stats Collection ####
############################


def collect_species_stats(baked_dir, species_name):
    """Collect stats from baked species data"""
    
    import json
    
    species_dir = pathlib.Path(baked_dir) / species_name
    train_path  = species_dir / "training.bin"
    val_path    = species_dir / "validation.bin"
    
    stats = {
        "species":       species_name,
        "has_data":      False,
        "train_chunks":  0,
        "val_chunks":    0,
        "train_size":    0,
        "val_size":      0,
        "raw_count":     0,
        "aug_count":     0,
    }
    
    if not train_path.exists():
        return stats
    
    stats["has_data"] = True
    
    # Get training stats
    try:
        train_info          = ds.get_binary_info(train_path)
        stats["train_chunks"] = train_info["num_chunks"]
        stats["train_size"]   = train_path.stat().st_size
        
        # Count raw vs augmented
        chunks            = ds.read_binary(train_path)
        stats["raw_count"] = sum(1 for c in chunks if not c.is_augmented)
        stats["aug_count"] = sum(1 for c in chunks if c.is_augmented)
    except Exception:
        pass
    
    # Get validation stats
    if val_path.exists():
        try:
            val_info           = ds.get_binary_info(val_path)
            stats["val_chunks"] = val_info["num_chunks"]
            stats["val_size"]   = val_path.stat().st_size
        except Exception:
            pass
    
    return stats


def write_bake_summary(log_path, run_config, species_results, species_stats, tokenizer_result=None, compact_result=None):
    """Write comprehensive bake summary log"""
    
    import datetime
    
    log_path = pathlib.Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"{'GeneT5 Data Bake Summary':^70}\n")
        f.write(f"{'='*70}\n")
        f.write(f"  Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")
        
        # Run configuration
        f.write(f"{' Run Configuration ':=^70}\n")
        f.write(f"  Raw directory:    {run_config.get('raw_dir', 'N/A')}\n")
        f.write(f"  Baked directory:  {run_config.get('baked_dir', 'N/A')}\n")
        f.write(f"  Species parallel: {run_config.get('species_parallel', 'N/A')}\n")
        f.write(f"  Workers/species:  {run_config.get('n_workers', 'N/A')}\n")
        f.write(f"  Tokenizer:        {run_config.get('tokenizer', 'None')}\n")
        f.write(f"  Compact:          {run_config.get('compact', False)}\n")
        if run_config.get('compact'):
            f.write(f"  Compact target:   {run_config.get('compact_target', 8192)}\n")
        f.write(f"\n")
        
        # Species results
        success_count = sum(1 for r in species_results if r.get("success"))
        failed_count  = len(species_results) - success_count
        
        f.write(f"{' Species Processing ':=^70}\n")
        f.write(f"  Total species:   {len(species_results)}\n")
        f.write(f"  Successful:      {success_count}\n")
        f.write(f"  Failed:          {failed_count}\n")
        f.write(f"\n")
        
        # Failed species details
        failed = [r for r in species_results if not r.get("success")]
        if failed:
            f.write(f"  Failed Species:\n")
            for r in failed:
                error = r.get("error", "Unknown error")
                f.write(f"    ✗ {r['species']}: {error}\n")
            f.write(f"\n")
        
        # Aggregated stats
        f.write(f"{' Aggregated Statistics ':=^70}\n")
        
        total_train_chunks = sum(s["train_chunks"] for s in species_stats)
        total_val_chunks   = sum(s["val_chunks"] for s in species_stats)
        total_raw          = sum(s["raw_count"] for s in species_stats)
        total_aug          = sum(s["aug_count"] for s in species_stats)
        total_train_size   = sum(s["train_size"] for s in species_stats)
        total_val_size     = sum(s["val_size"] for s in species_stats)
        
        f.write(f"\n  Training Data:\n")
        f.write(f"    Total chunks:     {total_train_chunks:,}\n")
        f.write(f"    Raw chunks:       {total_raw:,}\n")
        f.write(f"    Augmented chunks: {total_aug:,}\n")
        f.write(f"    Total size:       {ds.format_size(total_train_size)}\n")

        f.write(f"\n  Validation Data:\n")
        f.write(f"    Total chunks:     {total_val_chunks:,}\n")
        f.write(f"    Total size:       {ds.format_size(total_val_size)}\n")
        f.write(f"\n")
        
        # Per-species breakdown
        f.write(f"{' Per-Species Breakdown ':=^70}\n")
        f.write(f"\n")
        f.write(f"  {'Species':<20} {'Train':>8} {'Val':>6} {'Raw':>8} {'Aug':>8} {'Size':>10}\n")
        f.write(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*10}\n")
        
        for s in sorted(species_stats, key=lambda x: x["species"]):
            if s["has_data"]:
                f.write(f"  {s['species']:<20} {s['train_chunks']:>8,} {s['val_chunks']:>6} "
                        f"{s['raw_count']:>8,} {s['aug_count']:>8,} {ds.format_size(s['train_size']):>10}\n")
            else:
                f.write(f"  {s['species']:<20} {'--':>8} {'--':>6} {'--':>8} {'--':>8} {'--':>10}\n")
        f.write(f"\n")
        
        # Tokenizer expansion
        if tokenizer_result:
            f.write(f"{' Tokenizer Expansion ':=^70}\n")
            if tokenizer_result.get("success"):
                f.write(f"  Status:  Success\n")
                f.write(f"  Tokens:  {tokenizer_result.get('tokens_in_file', 'N/A')}\n")
            else:
                f.write(f"  Status:  Failed\n")
                f.write(f"  Error:   {tokenizer_result.get('error', 'Unknown')}\n")
            f.write(f"\n")
        
        # Compacting
        if compact_result:
            f.write(f"{' Compacting ':=^70}\n")
            if compact_result.get("success"):
                f.write(f"  Status:       Success\n")
                f.write(f"  Output:       {compact_result.get('output', 'N/A')}\n")
                f.write(f"  Output size:  {compact_result.get('size', 'N/A')}\n")
            else:
                f.write(f"  Status:  Failed\n")
                f.write(f"  Error:   {compact_result.get('error', 'Unknown')}\n")
            f.write(f"\n")
        
        f.write(f"{'='*70}\n")
        f.write(f"End of Summary\n")
        f.write(f"{'='*70}\n")
    
    return log_path


########################################
#####  Augmentation Status Report  #####
########################################


def report_augmentation_status(baked_dir):
    """Read merged bins and print nosing/augmentation ratios."""

    baked_dir = pathlib.Path(baked_dir)

    print(f"\n{'='*50}")
    print(f"{'Augmentation Status':^50}")
    print(f"{'='*50}")

    for label, filename in [("training", "training.bin"), ("validation", "validation.bin")]:
        bin_path = baked_dir / filename
        if not bin_path.exists():
            continue

        chunks    = ds.read_binary(bin_path)
        total     = len(chunks)
        raw_count = sum(1 for c in chunks if not c.is_augmented)
        aug_count = total - raw_count

        raw_pct = (raw_count / total * 100) if total else 0
        aug_pct = (aug_count / total * 100) if total else 0

        print(f"  {label}:")
        print(f"    Total:      {total:,} chunks")
        print(f"    Ab initio:  {raw_count:,} ({raw_pct:.1f}%)")
        print(f"    With hints: {aug_count:,} ({aug_pct:.1f}%)")

    eval_path = baked_dir / "eval.json"
    if eval_path.exists():
        import json
        with open(eval_path, 'r') as f:
            eval_data = json.load(f)
        print(f"  eval:")
        print(f"    Total: {len(eval_data)} samples (100% ab initio)")