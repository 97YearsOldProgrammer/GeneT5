import gzip
import json
import random
import pathlib

import pyfaidx


###################
#####  FASTA  #####
###################


def lazy_fasta_open(fasta_path):
    """Open FASTA with pyfaidx for lazy indexed access"""

    return pyfaidx.Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)


#################
#####  GFF  #####
#################


def parse_gff_attributes(attr_string):
    """Parse GFF3 attribute string into dict"""

    attrs = {}
    if not attr_string or attr_string == '.':
        return attrs

    for part in attr_string.split(';'):
        part = part.strip()
        if not part:
            continue

        if '=' in part:
            key, value = part.split('=', 1)
            attrs[key] = value

    return attrs


def _iter_gff_features(gff_path):
    """Yield parsed feature dicts from a GFF file (handles gzipped)"""

    open_func = gzip.open if str(gff_path).endswith('.gz') else open
    mode      = 'rt' if str(gff_path).endswith('.gz') else 'r'

    with open_func(gff_path, mode) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 9:
                continue

            yield {
                "seqid":      parts[0],
                "source":     parts[1],
                "type":       parts[2],
                "start":      int(parts[3]),
                "end":        int(parts[4]),
                "score":      parts[5],
                "strand":     parts[6],
                "phase":      parts[7],
                "attributes": parse_gff_attributes(parts[8]),
            }


########################
#####  Gene Index  #####
########################


TRANSCRIPT_TYPES = frozenset({
    "mrna", "transcript", "ncrna", "lncrna",
    "rrna", "trna", "snorna", "mirna", "pseudogenic_transcript",
})

CHILD_TYPES = frozenset({
    "exon", "cds", "five_prime_utr", "three_prime_utr",
    "start_codon", "stop_codon", "intron",
})


def parse_gff_to_gene_index(gff_path):
    """Stream GFF directly into gene_index — no intermediate features list"""

    gene_index     = {}
    transcript_map = {}

    # Pass 1: collect genes and transcripts
    for feat in _iter_gff_features(gff_path):
        ftype = feat["type"].lower()
        attrs = feat["attributes"]

        if ftype == "gene":
            gene_id = attrs.get("ID", "")
            if gene_id:
                gene_index[gene_id] = {
                    "seqid":       feat["seqid"],
                    "start":       feat["start"],
                    "end":         feat["end"],
                    "strand":      feat["strand"],
                    "transcripts": {},
                    "features":    [],
                    "attributes":  attrs,
                }

        elif ftype in TRANSCRIPT_TYPES:
            transcript_id = attrs.get("ID", "")
            parent_gene   = attrs.get("Parent", "")

            if transcript_id and parent_gene:
                biotype = attrs.get("biotype", attrs.get("transcript_biotype", ftype))

                transcript_map[transcript_id] = {
                    "gene_id":  parent_gene,
                    "biotype":  biotype,
                    "start":    feat["start"],
                    "end":      feat["end"],
                    "strand":   feat["strand"],
                }

    # Assign transcripts to genes (iterates transcript_map, not file)
    for t_id, t_data in transcript_map.items():
        gene_id = t_data["gene_id"]
        if gene_id in gene_index:
            gene_index[gene_id]["transcripts"][t_id] = {
                "biotype":  t_data["biotype"],
                "start":    t_data["start"],
                "end":      t_data["end"],
                "strand":   t_data["strand"],
                "features": [],
            }

    # Pass 2: assign child features to transcripts and genes
    for feat in _iter_gff_features(gff_path):
        ftype = feat["type"].lower()

        if ftype not in CHILD_TYPES:
            continue

        attrs  = feat["attributes"]
        parent = attrs.get("Parent", "")

        if not parent:
            continue

        # Parent could be transcript
        if parent in transcript_map:
            t_data  = transcript_map[parent]
            gene_id = t_data["gene_id"]

            if gene_id in gene_index:
                gene_index[gene_id]["features"].append(feat)

                if parent in gene_index[gene_id]["transcripts"]:
                    gene_index[gene_id]["transcripts"][parent]["features"].append(feat)

        # Parent could be gene directly (some GFF formats)
        elif parent in gene_index:
            gene_index[parent]["features"].append(feat)

    # Merge CDS info into exons
    _merge_cds_into_exons(gene_index)

    return gene_index


def filter_canonical_transcripts(gene_index):
    """Keep only the canonical (longest) transcript per gene"""

    biotype_priority = {
        "mrna": 0, "protein_coding": 0,
        "lncrna": 1, "lnc_rna": 1,
        "rrna": 2, "trna": 2, "snorna": 2, "snrna": 2, "mirna": 2,
    }

    for gene_id, gene_data in gene_index.items():
        transcripts = gene_data.get("transcripts", {})

        if len(transcripts) <= 1:
            continue

        best_tid   = None
        best_score = (999, 0)

        for tid, tdata in transcripts.items():
            biotype  = tdata.get("biotype", "unknown").lower()
            priority = biotype_priority.get(biotype, 10)
            span     = tdata.get("end", 0) - tdata.get("start", 0)
            score    = (priority, -span)

            if best_tid is None or score < best_score:
                best_tid   = tid
                best_score = score

        # Remove non-canonical transcripts
        keep_features = []
        for feat in gene_data.get("features", []):
            parent = feat.get("attributes", {}).get("Parent", "")
            if parent == best_tid or parent == gene_id or parent == "":
                keep_features.append(feat)

        gene_data["features"]    = keep_features
        gene_data["transcripts"] = {best_tid: transcripts[best_tid]}

    return gene_index


def _merge_cds_into_exons(gene_index):
    """Merge CDS phase and boundaries into matching exon features"""

    for gene_id, gene_data in gene_index.items():
        for transcript_id, transcript_data in gene_data.get("transcripts", {}).items():
            features = transcript_data.get("features", [])

            exons     = [f for f in features if f["type"].lower() == "exon"]
            cdses     = [f for f in features if f["type"].lower() == "cds"]
            utr5_list = [f for f in features if f["type"].lower() == "five_prime_utr"]
            utr3_list = [f for f in features if f["type"].lower() == "three_prime_utr"]

            if not cdses and not utr5_list and not utr3_list:
                continue

            for exon in exons:
                e_start = exon["start"]
                e_end   = exon["end"]

                # First check for explicit UTR features
                for utr5 in utr5_list:
                    u_start = utr5["start"]
                    u_end   = utr5["end"]

                    # UTR overlaps this exon - CDS starts after UTR
                    if u_start >= e_start and u_end <= e_end:
                        exon["cds_start"] = u_end + 1
                        break

                for utr3 in utr3_list:
                    u_start = utr3["start"]
                    u_end   = utr3["end"]

                    # UTR overlaps this exon - CDS ends before UTR
                    if u_start >= e_start and u_end <= e_end:
                        exon["cds_end"] = u_start - 1
                        break

                # Then check CDS for phase and boundaries
                for cds in cdses:
                    c_start = cds["start"]
                    c_end   = cds["end"]

                    # Check if CDS overlaps this exon
                    if c_start > e_end or c_end < e_start:
                        continue

                    # CDS overlaps exon - copy phase
                    exon["phase"] = cds.get("phase", ".")

                    # Only set cds_start/cds_end if not already set by explicit UTR
                    if "cds_start" not in exon and c_start > e_start:
                        exon["cds_start"] = c_start

                    if "cds_end" not in exon and c_end < e_end:
                        exon["cds_end"] = c_end

                    break


###########################
#####  Token Extract  #####
###########################


def extract_feature_types(features):
    """Extract unique feature types from features list"""

    types = set()
    for feat in features:
        ftype = feat.get("type", "").lower()
        if ftype:
            types.add(ftype)
    return types


def extract_biotypes(features):
    """Extract unique biotypes from features list"""

    biotypes = set()
    for feat in features:
        attrs = feat.get("attributes", {})

        for key in ("biotype", "transcript_biotype", "gene_biotype"):
            bt = attrs.get(key, "")
            if bt:
                biotypes.add(bt.lower())

    return biotypes


#########################
#####  File Utils   #####
#########################


def find_genome_files(species_dir):
    """Find FASTA and GFF files in species directory, preferring uncompressed"""

    species_dir = pathlib.Path(species_dir)
    fna_path    = None
    gff_path    = None

    fna_gz = ('.fna.gz', '.fa.gz', '.fasta.gz')
    fna_raw = ('.fna', '.fa', '.fasta')
    gff_gz  = ('.gff.gz', '.gff3.gz')
    gff_raw = ('.gff', '.gff3')

    for f in species_dir.iterdir():
        name = f.name.lower()

        if name.endswith(fna_raw) or name in ('fna',):
            fna_path = f
        elif name.endswith(fna_gz) or name in ('fna.gz',):
            if fna_path is None:
                fna_path = f

        if name.endswith(gff_raw) or name in ('gff',):
            gff_path = f
        elif name.endswith(gff_gz) or name in ('gff.gz',):
            if gff_path is None:
                gff_path = f

    if not fna_path:
        raise FileNotFoundError(f"No FASTA file found in {species_dir}")
    if not gff_path:
        raise FileNotFoundError(f"No GFF file found in {species_dir}")

    return fna_path, gff_path


##############################
#####  Gene Index I/O    #####
##############################


def save_gene_index(gene_index, path):
    """Save gene_index to JSON sidecar for later reuse (avoids re-parsing GFF)"""

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(gene_index, f)


def load_gene_index(path):
    """Load gene_index from JSON sidecar"""

    path = pathlib.Path(path)
    if not path.exists():
        return None

    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


########################
#####  Eval Prep   #####
########################


def extract_coding_genes(gene_index):
    """Filter gene index to protein-coding genes with exons"""

    coding_genes = {}

    for gene_id, gene_data in gene_index.items():
        attrs   = gene_data.get("attributes", {})
        biotype = attrs.get("gene_biotype", attrs.get("biotype", "")).lower()

        if biotype != "protein_coding":
            continue

        exons = [f for f in gene_data.get("features", []) if f["type"].lower() == "exon"]
        if not exons:
            continue

        coding_genes[gene_id] = gene_data

    return coding_genes


def build_eval_sample(gene_id, gene_data, sequences, window_size):
    """Build a single eval sample with genomic window around a gene"""

    seqid  = gene_data["seqid"]
    strand = gene_data["strand"]

    if seqid not in sequences:
        return None

    seq_len    = len(sequences[seqid])
    gene_start = gene_data["start"]
    gene_end   = gene_data["end"]
    gene_len   = gene_end - gene_start + 1

    if gene_len > window_size:
        return None

    # Center gene in window
    padding      = (window_size - gene_len) // 2
    window_start = max(1, gene_start - padding)
    window_end   = min(seq_len, window_start + window_size - 1)
    window_start = max(1, window_end - window_size + 1)

    # Extract sequence (1-based GFF coords -> 0-based python slice)
    sequence = sequences[seqid][window_start - 1:window_end]

    if len(sequence) < 1000:
        return None

    # Build reference features relative to window
    ref_features = []
    for feat in gene_data.get("features", []):
        if feat["type"].lower() != "exon":
            continue

        feat_start = feat["start"] - window_start
        feat_end   = feat["end"] - window_start

        if feat_start < 0 or feat_end >= len(sequence):
            continue

        ref_features.append({
            "start":   feat_start,
            "end":     feat_end,
            "strand":  strand,
            "type":    "exon",
            "phase":   feat.get("phase", "."),
        })

    if not ref_features:
        return None

    return {
        "sequence":     sequence,
        "ref_features": sorted(ref_features, key=lambda f: f["start"]),
        "seqid":        seqid,
        "gene_id":      gene_id,
        "window_start": window_start,
        "window_end":   window_end,
        "strand":       strand,
        "num_exons":    len(ref_features),
    }


def select_diverse_samples(samples, n):
    """Select samples from diverse chromosomes via round-robin"""

    by_chr = {}
    for s in samples:
        seqid = s["seqid"]
        if seqid not in by_chr:
            by_chr[seqid] = []
        by_chr[seqid].append(s)

    selected  = []
    chr_names = sorted(by_chr.keys())

    idx = 0
    while len(selected) < n and any(by_chr[c] for c in chr_names):
        c = chr_names[idx % len(chr_names)]
        if by_chr[c]:
            sample = random.choice(by_chr[c])
            by_chr[c].remove(sample)
            selected.append(sample)
        idx += 1

    return selected[:n]
