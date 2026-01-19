import gzip
import json
import sys
from pathlib import Path
from typing  import Dict, List, Optional, Tuple, Any


def anti(seq):
    comp     = str.maketrans('ACGTRYMKBDHVNacgtrymkbdhvn', 'TGCAYRKMVHDBNtgcayrkmvhdbn')
    anti_seq = seq.translate(comp)[::-1]
    return anti_seq


#######################
#####  Constants  #####
#######################


GENE_FEATURE_TYPES = {
    "exon", "intron", "CDS", "cds",
    "five_prime_UTR", "three_prime_UTR", "utr5", "utr3",
}


################################
#####  Parsing Functions   #####
################################


def get_filepointer(filename):
    fp = None
    if   filename.endswith('.gz'): fp = gzip.open(filename, 'rt')
    elif filename == '-':          fp = sys.stdin
    else:                          fp = open(filename)
    return fp


def parse_fasta(fasta_path):
    """
    Parse FASTA file, handling multi-chromosome genomes.
    The seqid is the first word after '>' in the header.
    """
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
    """
    Parse GFF3 file.
    Column 1 (seqid) is used to match with FASTA sequences.
    """
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
            "raw_line":   line
        }
        
        features.append(feature)
    
    fp.close()
    return features


def group_features_by_seqid(features):
    """Group features by their seqid (chromosome/contig)."""
    grouped = {}
    for feat in features:
        seqid = feat["seqid"]
        if seqid not in grouped:
            grouped[seqid] = []
        grouped[seqid].append(feat)
    return grouped


def group_features_by_parent(features):
    """Group features by their Parent attribute."""
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


################################
#####  Hierarchy Building  #####
################################


def build_transcript_map(features):
    """
    Build maps for gene->transcript and transcript->biotype relationships.
    
    Returns:
        gene_to_transcripts: {gene_id: [transcript_ids in order]}
        transcript_info:     {transcript_id: {"parent": gene_id, "biotype": str}}
    """
    gene_to_transcripts = {}
    transcript_info     = {}
    
    # sort by start position to maintain order
    sorted_feats = sorted(features, key=lambda x: x["start"])
    
    for feat in sorted_feats:
        ftype = feat["type"].lower()
        attrs = feat["attributes"]
        
        # transcript-level features
        if ftype in {"mrna", "transcript", "rrna", "trna", "ncrna", 
                     "snrna", "snorna", "lncrna", "mirna", "guide_rna", "lnc_rna"}:
            t_id   = attrs.get("ID", "")
            parent = attrs.get("Parent", "")
            
            # get biotype from attributes or infer from type
            biotype = attrs.get("biotype", ftype)
            if biotype.lower() == "protein_coding":
                biotype = "mRNA"
            
            if t_id:
                transcript_info[t_id] = {
                    "parent":  parent,
                    "biotype": biotype,
                    "start":   feat["start"],
                }
                
                if parent:
                    if parent not in gene_to_transcripts:
                        gene_to_transcripts[parent] = []
                    if t_id not in gene_to_transcripts[parent]:
                        gene_to_transcripts[parent].append(t_id)
    
    return gene_to_transcripts, transcript_info


def build_feature_hierarchy(features):
    """
    Build complete hierarchy: gene -> transcript -> exon/CDS
    
    Returns:
        hierarchy: {
            gene_id: {
                "transcripts": {
                    transcript_id: {
                        "biotype":  str,
                        "features": [feature_dicts]
                    }
                },
                "start": int,
                "end":   int,
                "strand": str
            }
        }
    """
    gene_to_transcripts, transcript_info = build_transcript_map(features)
    
    # build parent lookup for exon/CDS -> transcript
    hierarchy = {}
    
    # first pass: initialize genes
    for feat in features:
        if feat["type"].lower() == "gene":
            gene_id = feat["attributes"].get("ID", "")
            if gene_id:
                hierarchy[gene_id] = {
                    "transcripts": {},
                    "start":       feat["start"],
                    "end":         feat["end"],
                    "strand":      feat["strand"],
                }
    
    # second pass: add transcripts to genes
    for t_id, t_info in transcript_info.items():
        gene_id = t_info["parent"]
        
        # create gene entry if it doesn't exist
        if gene_id and gene_id not in hierarchy:
            hierarchy[gene_id] = {
                "transcripts": {},
                "start":       t_info["start"],
                "end":         t_info["start"],
                "strand":      "+",
            }
        
        if gene_id:
            hierarchy[gene_id]["transcripts"][t_id] = {
                "biotype":  t_info["biotype"],
                "features": [],
            }
    
    # third pass: add exon/CDS features to transcripts
    for feat in features:
        ftype = feat["type"].lower()
        if ftype not in {"exon", "cds", "intron", "five_prime_utr", "three_prime_utr", "utr"}:
            continue
        
        parent = feat["attributes"].get("Parent", "")
        if not parent:
            continue
        
        # find which gene this belongs to
        if parent in transcript_info:
            gene_id = transcript_info[parent]["parent"]
            if gene_id in hierarchy and parent in hierarchy[gene_id]["transcripts"]:
                hierarchy[gene_id]["transcripts"][parent]["features"].append(feat)
        else:
            # parent might be gene directly (some GFF formats)
            if parent in hierarchy:
                # add to first transcript or create one
                if not hierarchy[parent]["transcripts"]:
                    hierarchy[parent]["transcripts"][parent] = {
                        "biotype":  ".",
                        "features": [],
                    }
                first_t = list(hierarchy[parent]["transcripts"].keys())[0]
                hierarchy[parent]["transcripts"][first_t]["features"].append(feat)
    
    return hierarchy


################################
#####  Dataset Creation    #####
################################


def format_annotation_target(gene_features, gene_index, biotype, bos_token="<BOS>", eos_token="<EOS>"):
    """
    Format annotation target with new condensed format.
    
    Output format per line: type start end strand phase gene_index biotype
    Example: exon 149658 149776 + . 1 rRNA
    """
    if not gene_features:
        return f"{bos_token}{eos_token}"
    
    lines = []
    for feat in sorted(gene_features, key=lambda x: x["start"]):
        ftype  = feat["type"].lower()
        strand = feat["strand"]
        start  = feat["start"]
        end    = feat["end"]
        phase  = feat["phase"]
        
        lines.append(f"{ftype}\t{start}\t{end}\t{strand}\t{phase}\t{gene_index}\t{biotype}")
    
    return f"{bos_token}\n" + "\n".join(lines) + f"\n{eos_token}"


def create_gene_prediction_dataset(sequences, features_by_seqid, gene_token="[ATT]",
                                    bos_token="<BOS>", eos_token="<EOS>", context_pad=0):
    """
    Create gene prediction dataset with proper hierarchy tracking.
    
    Each sample contains features from one gene, with gene_index and biotype.
    """
    dataset = []
    
    for seqid, sequence in sequences.items():
        features = features_by_seqid.get(seqid, [])
        
        if not features:
            continue
        
        # build hierarchy
        hierarchy = build_feature_hierarchy(features)
        
        if not hierarchy:
            continue
        
        # process genes in order of position
        sorted_genes = sorted(hierarchy.items(), key=lambda x: x[1]["start"])
        
        for gene_index, (gene_id, gene_data) in enumerate(sorted_genes, start=1):
            transcripts = gene_data["transcripts"]
            
            if not transcripts:
                continue
            
            # collect all features and determine primary biotype
            all_features = []
            biotypes     = []
            
            for t_id, t_data in transcripts.items():
                biotypes.append(t_data["biotype"])
                all_features.extend(t_data["features"])
            
            if not all_features:
                continue
            
            # use most common biotype or "." for unknown
            primary_biotype = biotypes[0] if biotypes else "."
            
            # get span
            min_start = min(f["start"] for f in all_features)
            max_end   = max(f["end"] for f in all_features)
            strand    = gene_data["strand"]
            
            # extract sequence with context
            seq_start = max(0, min_start - 1 - context_pad)
            seq_end   = min(len(sequence), max_end + context_pad)
            chunk_seq = sequence[seq_start:seq_end]
            
            if strand == "-":
                chunk_seq = anti(chunk_seq)
            
            # adjust coordinates relative to extracted sequence
            adjusted_features = []
            for f in all_features:
                adj_f          = f.copy()
                adj_f["start"] = f["start"] - seq_start
                adj_f["end"]   = f["end"] - seq_start
                adjusted_features.append(adj_f)
            
            input_text  = f"{gene_token} {chunk_seq}"
            target_text = format_annotation_target(
                adjusted_features, gene_index, primary_biotype, bos_token, eos_token
            )
            
            sample = {
                "seqid":        seqid,
                "parent_id":    gene_id,
                "start":        min_start,
                "end":          max_end,
                "strand":       strand,
                "gene_index":   gene_index,
                "biotype":      primary_biotype,
                "input":        input_text,
                "target":       target_text,
                "num_features": len(all_features),
            }
            
            dataset.append(sample)
    
    print(f"  Created {len(dataset)} gene prediction samples")
    return dataset


################################
#####  Type Extraction     #####
################################


def extract_feature_types(features):
    """
    Extract all feature types that belong to valid parent/transcript hierarchy.
    
    Returns:
        set: unique feature types found under valid genes/transcripts
    """
    # build parent chain
    id_to_parent = {}
    valid_parents = set()
    
    for feat in features:
        attrs   = feat["attributes"]
        feat_id = attrs.get("ID", "")
        parent  = attrs.get("Parent", "")
        ftype   = feat["type"].lower()
        
        if feat_id:
            id_to_parent[feat_id] = parent
        
        # genes and transcripts are valid roots
        if ftype in {"gene", "mrna", "transcript", "rrna", "trna", "ncrna",
                     "snrna", "snorna", "lncrna", "mirna", "lnc_rna", "guide_rna"}:
            valid_parents.add(feat_id)
    
    # find all IDs that descend from valid parents
    def is_valid_descendant(feat_id):
        if not feat_id:
            return False
        if feat_id in valid_parents:
            return True
        parent = id_to_parent.get(feat_id)
        if parent:
            return is_valid_descendant(parent)
        return False
    
    # collect types from features with valid parents
    types = set()
    for feat in features:
        attrs  = feat["attributes"]
        parent = attrs.get("Parent", "")
        ftype  = feat["type"]
        
        # include if parent is valid or if it's a transcript-level feature
        if parent and (parent in valid_parents or is_valid_descendant(parent)):
            types.add(ftype.lower())
        elif ftype.lower() in {"gene", "mrna", "transcript", "rrna", "trna", "ncrna",
                               "snrna", "snorna", "lncrna", "mirna"}:
            types.add(ftype.lower())
    
    return types


def extract_biotypes(features):
    """
    Extract all biotypes from transcript-level features.
    
    Returns:
        set: unique biotypes found
    """
    biotypes = set()
    
    for feat in features:
        attrs   = feat["attributes"]
        biotype = attrs.get("biotype", "")
        ftype   = feat["type"].lower()
        
        if biotype:
            biotypes.add(biotype.lower())
        
        # also add feature type as potential biotype for transcript-level
        if ftype in {"mrna", "rrna", "trna", "ncrna", "snrna", "snorna", 
                     "lncrna", "mirna", "pseudogene"}:
            biotypes.add(ftype)
    
    return biotypes


################################
#####  I/O Functions       #####
################################


def save_dataset(dataset, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in dataset:
            f.write(json.dumps(sample) + '\n')
    
    print(f"  Saved {len(dataset)} samples to {output_path}")


def load_dataset(input_path):
    dataset = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset