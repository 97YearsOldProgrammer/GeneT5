import gzip
import re
from collections import defaultdict


###################
#####  FASTA  #####
###################


def parse_fasta(fasta_path):
    """Parse FASTA file (handles gzipped files)"""

    sequences = {}
    current_id = None
    current_seq = []

    open_func = gzip.open if str(fasta_path).endswith('.gz') else open
    mode = 'rt' if str(fasta_path).endswith('.gz') else 'r'

    with open_func(fasta_path, mode) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)

                # Parse header: >seqid description
                header = line[1:].split()[0]
                current_id = header
                current_seq = []
            else:
                current_seq.append(line.upper())

    # Add last sequence
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)

    return sequences


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


def parse_gff(gff_path):
    """Parse GFF3 file (handles gzipped files)"""

    features = []

    open_func = gzip.open if str(gff_path).endswith('.gz') else open
    mode = 'rt' if str(gff_path).endswith('.gz') else 'r'

    with open_func(gff_path, mode) as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 9:
                continue

            seqid   = parts[0]
            source  = parts[1]
            ftype   = parts[2]
            start   = int(parts[3])
            end     = int(parts[4])
            score   = parts[5]
            strand  = parts[6]
            phase   = parts[7]
            attrs   = parse_gff_attributes(parts[8])

            feature = {
                "seqid":      seqid,
                "source":     source,
                "type":       ftype,
                "start":      start,
                "end":        end,
                "score":      score,
                "strand":     strand,
                "phase":      phase,
                "attributes": attrs,
            }

            features.append(feature)

    return features


########################
#####  Gene Index  #####
########################


def build_gene_index(features):
    """Build hierarchical gene index from GFF features"""

    gene_index = {}
    transcript_map = {}

    # First pass: collect genes and transcripts
    for feat in features:
        ftype = feat["type"].lower()
        attrs = feat.get("attributes", {})

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

        elif ftype in ("mrna", "transcript", "ncrna", "lncrna", "rrna", "trna", "snorna", "mirna", "pseudogenic_transcript"):
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
                    "features": [],
                }

    # Second pass: assign transcripts to genes
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

    # Third pass: assign child features (exons, CDS, etc.) to transcripts and genes
    child_types = {"exon", "cds", "five_prime_utr", "three_prime_utr", "start_codon", "stop_codon", "intron"}

    for feat in features:
        ftype = feat["type"].lower()

        if ftype not in child_types:
            continue

        attrs  = feat.get("attributes", {})
        parent = attrs.get("Parent", "")

        if not parent:
            continue

        # Parent could be transcript
        if parent in transcript_map:
            t_data  = transcript_map[parent]
            gene_id = t_data["gene_id"]

            if gene_id in gene_index:
                # Add to gene's features
                gene_index[gene_id]["features"].append(feat)

                # Add to transcript's features
                if parent in gene_index[gene_id]["transcripts"]:
                    gene_index[gene_id]["transcripts"][parent]["features"].append(feat)

        # Parent could be gene directly (some GFF formats)
        elif parent in gene_index:
            gene_index[parent]["features"].append(feat)

    return gene_index


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