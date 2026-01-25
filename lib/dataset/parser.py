"""Parsing utilities for FASTA and GFF3 files"""

import gzip as gz
import json as js
import sys
import pathlib as pl


#####################  Constants  #####################


GENE_FEATURE_TYPES = {
    "exon", "CDS", "cds",
    "five_prime_UTR", "three_prime_UTR", "utr5", "utr3",
}


#####################  Sequence Utilities  #####################


def anti(seq):
    """Reverse complement a DNA sequence"""
    
    comp     = str.maketrans('ACGTRYMKBDHVNacgtrymkbdhvn', 'TGCAYRKMVHDBNtgcayrkmvhdbn')
    anti_seq = seq.translate(comp)[::-1]
    return anti_seq


#####################  File I/O  #####################


def get_filepointer(filename):
    """Get file pointer handling gzip and stdin"""
    
    fp = None
    if   filename.endswith('.gz'): fp = gz.open(filename, 'rt')
    elif filename == '-':          fp = sys.stdin
    else:                          fp = open(filename)
    return fp


def parse_fasta(fasta_path):
    """Parse FASTA file into dict of sequences"""
    
    sequences   = {}
    current_id  = None
    current_seq = []
    
    fp = get_filepointer(fasta_path)
    for line in fp:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('>'):
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
            
            header      = line[1:].split()[0]
            current_id  = header
            current_seq = []
        else:
            current_seq.append(line.upper())
    
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)
    
    fp.close()
    return sequences


def parse_gff(gff_path):
    """Parse GFF3 file into list of feature dicts"""
    
    features = []
    
    fp = get_filepointer(gff_path)
    for line in fp:
        line = line.strip()
        
        if not line or line.startswith('#'):
            continue
        
        parts = line.split('\t')
        if len(parts) < 9:
            continue
        
        attrs = {}
        for attr in parts[8].split(';'):
            if '=' in attr:
                key, value = attr.split('=', 1)
                attrs[key.strip()] = value.strip()
        
        feature = {
            "seqid":      parts[0],
            "source":     parts[1],
            "type":       parts[2],
            "start":      int(parts[3]),
            "end":        int(parts[4]),
            "score":      parts[5],
            "strand":     parts[6],
            "phase":      parts[7],
            "attributes": attrs,
        }
        
        features.append(feature)
    
    fp.close()
    return features


#####################  Feature Grouping  #####################


def group_by_seqid(features):
    """Group features by their seqid (chromosome/contig)"""
    
    grouped = {}
    for feat in features:
        seqid = feat["seqid"]
        if seqid not in grouped:
            grouped[seqid] = []
        grouped[seqid].append(feat)
    return grouped


def group_by_parent(features):
    """Group features by parent ID"""
    
    grouped = {}
    orphans = []
    
    for feat in features:
        parent = feat["attributes"].get("Parent", feat["attributes"].get("ID"))
        if parent:
            if parent not in grouped:
                grouped[parent] = []
            grouped[parent].append(feat)
        else:
            orphans.append(feat)
    
    return grouped, orphans


#####################  Gene Index Building  #####################


def build_gene_index(features):
    """Build gene hierarchy from features"""
    
    gene_info          = {}
    transcript_info    = {}
    transcript_to_gene = {}
    
    sorted_feats = sorted(features, key=lambda x: x["start"])
    
    for feat in sorted_feats:
        ftype = feat["type"].lower()
        attrs = feat["attributes"]
        
        if ftype == "gene":
            gene_id = attrs.get("ID", "")
            if gene_id:
                gene_info[gene_id] = {
                    "seqid":       feat["seqid"],
                    "start":       feat["start"],
                    "end":         feat["end"],
                    "strand":      feat["strand"],
                    "transcripts": {},
                    "features":    [],
                }
        
        elif ftype in {"mrna", "transcript", "rrna", "trna", "ncrna",
                       "snrna", "snorna", "lncrna", "mirna", "guide_rna", "lnc_rna"}:
            t_id   = attrs.get("ID", "")
            parent = attrs.get("Parent", "")
            
            biotype = attrs.get("biotype", ftype)
            if biotype.lower() == "protein_coding":
                biotype = "mRNA"
            
            if t_id:
                transcript_info[t_id] = {
                    "parent":  parent,
                    "biotype": biotype,
                    "start":   feat["start"],
                    "end":     feat["end"],
                }
                if parent:
                    transcript_to_gene[t_id] = parent
    
    for t_id, t_info in transcript_info.items():
        gene_id = t_info["parent"]
        if gene_id and gene_id in gene_info:
            gene_info[gene_id]["transcripts"][t_id] = {
                "biotype":  t_info["biotype"],
                "start":    t_info["start"],
                "end":      t_info["end"],
                "features": [],
            }
    
    for feat in sorted_feats:
        ftype = feat["type"]
        if ftype not in GENE_FEATURE_TYPES and ftype.lower() not in GENE_FEATURE_TYPES:
            continue
        
        parent = feat["attributes"].get("Parent", "")
        if not parent:
            continue
        
        gene_id = transcript_to_gene.get(parent, parent)
        
        if gene_id in gene_info:
            gene_info[gene_id]["features"].append(feat)
            
            if parent in gene_info[gene_id]["transcripts"]:
                gene_info[gene_id]["transcripts"][parent]["features"].append(feat)
    
    return gene_info


#####################  Token Extraction  #####################


def extract_feature_types(features):
    """Extract unique feature types for tokenizer"""
    
    types = set()
    
    for feat in features:
        ftype = feat["type"].lower()
        if ftype in {"exon", "cds", "five_prime_utr", "three_prime_utr"}:
            types.add(ftype)
    
    return types


def extract_biotypes(features):
    """Extract unique biotypes for tokenizer"""
    
    biotypes = set()
    
    for feat in features:
        attrs   = feat.get("attributes", {})
        biotype = attrs.get("biotype", "")
        ftype   = feat["type"].lower()
        
        if biotype:
            biotypes.add(biotype.lower())
        
        if ftype in {"mrna", "rrna", "trna", "ncrna", "snrna", "snorna",
                     "lncrna", "mirna", "pseudogene"}:
            biotypes.add(ftype)
    
    return biotypes


#####################  Target Formatting  #####################


def format_target(features, gene_index_map, bos_token="<BOS>", eos_token="<EOS>"):
    """Format features as target annotation string"""
    
    lines = [bos_token]
    
    gene_indices = {}
    gene_counter = 1
    
    for feat in sorted(features, key=lambda x: x["start"]):
        gene_id = feat.get("gene_id", "unknown")
        
        if gene_id not in gene_indices:
            gene_indices[gene_id] = gene_counter
            gene_counter += 1
        
        gene_idx = gene_indices[gene_id]
        biotype  = gene_index_map.get(gene_id, {}).get("biotype", ".")
        
        line = f"{feat['type']}\t{feat['start']}\t{feat['end']}\t{feat['strand']}\t{feat['phase']}\t{gene_idx}\t{biotype}"
        lines.append(line)
    
    lines.append(eos_token)
    
    return "\n".join(lines)


#####################  Dataset I/O  #####################


def save_jsonl(dataset, output_path):
    """Save dataset to JSONL file"""
    
    output_path = pl.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in dataset:
            sample_out = {k: v for k, v in sample.items() if k != "features"}
            f.write(js.dumps(sample_out) + '\n')
    
    return len(dataset)


def load_jsonl(input_path):
    """Load dataset from JSONL file"""
    
    dataset = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                dataset.append(js.loads(line))
    return dataset
