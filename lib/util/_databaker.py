import subprocess
import pathlib

import lib.dataset as ds


####################
#####  Config  #####
####################


TAXA_CONFIG = {
    "Prokaryotes": {
        "limit":   9000,
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
        "limit":   22500,
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
        "limit":   45000,
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
        "limit":   90000,
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
        "limit":   45000,
        "species": [
            "Earthmoss",
            "Maize",
            "M.truncatula",
            "Rice",
            "T.cress",
        ],
    },
}


def build_species_lookup():
    """Build species -> (taxa, limit) lookup"""
    
    lookup = {}
    for taxa, config in TAXA_CONFIG.items():
        limit = config["limit"]
        for species in config["species"]:
            lookup[species] = (taxa, limit)
    return lookup


SPECIES_LOOKUP = build_species_lookup()


######################
#####  Utilities #####
######################


def find_genome_files(species_dir):
    """Find GFF and FASTA files in species directory"""
    
    species_dir = pathlib.Path(species_dir)
    
    gff_file   = None
    fasta_file = None
    
    for f in species_dir.iterdir():
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


def run_parse_data(species_name, fasta_path, gff_path, output_dir, limit, log_dir, token_file=None, n_workers=1):
    """Run parse_data.py for a single species"""
    
    cmd = [
        "python3", "bin/parse_data.py",
        str(fasta_path),
        str(gff_path),
        str(output_dir),
        "--limit", str(limit),
        "--n_workers", str(n_workers),
    ]
    
    if token_file:
        cmd.extend(["--extract_tokens", token_file])
    
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
    
    species_name, raw_dir, baked_dir, log_dir, limit, token_file, n_workers = args
    
    species_raw_dir = pathlib.Path(raw_dir) / species_name
    
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
    
    output_dir = pathlib.Path(baked_dir) / species_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return run_parse_data(
        species_name, fasta_file, gff_file, output_dir, limit, log_dir, token_file, n_workers
    )


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


#######################
#####  Bake Stats #####
#######################


def print_run_stats(stats, chunk_stats, validation, output_path):
    """Print comprehensive run statistics"""
    
    print(f"\n{'='*60}")
    print("Run Statistics")
    print(f"{'='*60}")
    
    print(f"\nChunking:")
    print(f"  Total chunks:     {chunk_stats['total_chunks']}")
    print(f"  Backtrack events: {chunk_stats['backtrack_count']}")
    
    if chunk_stats["genes_per_chunk"]:
        avg_genes = sum(chunk_stats["genes_per_chunk"]) / len(chunk_stats["genes_per_chunk"])
        print(f"  Avg genes/chunk:  {avg_genes:.2f}")
    
    if chunk_stats["chunk_sizes"]:
        avg_size = sum(chunk_stats["chunk_sizes"]) / len(chunk_stats["chunk_sizes"])
        print(f"  Avg chunk size:   {avg_size/1000:.1f} kb")
    
    print(f"\nData Augmentation:")
    print(f"  Raw chunks:       {stats['raw_count']}")
    print(f"  Augmented chunks: {stats['aug_count']}")
    print(f"  Total samples:    {stats['total_samples']}")
    
    print(f"\nOutput:")
    print(f"  Binary file:      {output_path}")
    print(f"  File size:        {stats.get('file_size', 0) / 1024:.1f} KB")
    
    if validation:
        ds.print_validation_stats(validation)


def build_species_list(species=None, taxa=None):
    """Build list of species to process"""
    
    species_to_process = []
    
    if species:
        for sp in species:
            if sp in SPECIES_LOOKUP:
                taxa_name, limit = SPECIES_LOOKUP[sp]
                species_to_process.append((sp, limit, taxa_name))
            else:
                print(f"  WARNING: Unknown species '{sp}', skipping")
    
    elif taxa:
        for taxa_name in taxa:
            if taxa_name in TAXA_CONFIG:
                config = TAXA_CONFIG[taxa_name]
                for sp in config["species"]:
                    species_to_process.append((sp, config["limit"], taxa_name))
            else:
                print(f"  WARNING: Unknown taxa '{taxa_name}', skipping")
    
    else:
        for taxa_name, config in TAXA_CONFIG.items():