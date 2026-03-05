import subprocess
import pathlib
import tempfile
import shutil
from dataclasses import dataclass


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
            with open(out_path, 'wb') as out_f:
                subprocess.run(
                    ['pigz', '-d', '-k', '-c', str(gz_path)],
                    stdout=out_f, check=True,
                )
        else:
            with open(out_path, 'wb') as out_f:
                subprocess.run(
                    ['gunzip', '-c', str(gz_path)],
                    stdout=out_f, check=True,
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
    pack_seq_len:  int = 8192


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
