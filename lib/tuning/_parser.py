import gzip
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


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

RNA_CLASSES = {
    "pseudogene":               0,
    "ncrna":                    1,
    "trna":                     2,
    "rrna":                     3,
    "tmrna":                    4,
    "srna":                     5,
    "misc_rna":                 6,
    "antisense_rna":            7,
    "rnase_p_rna":              8,
    "srp_rna":                  9,
    "snorna":                   10,
    "snrna":                    11,
    "transposable_element":     12,
    "origin_of_replication":    13,
    "mobile_genetic_element":   14,
}

RNA_FEATURE_TYPES = {
    "pseudogene", "ncRNA", "ncrna", "tRNA", "trna", 
    "rRNA", "rrna", "tmRNA", "tmrna", "sRNA", "srna",
    "misc_RNA", "misc_rna", "antisense_RNA", "antisense_rna",
    "RNase_P_RNA", "rnase_p_rna", "SRP_RNA", "srp_rna",
    "snoRNA", "snorna", "snRNA", "snrna",
    "transposable_element", "origin_of_replication", 
    "mobile_genetic_element"
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
            
            # extract seqid: first word after '>'
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
#####  Dataset Creation    #####
################################


def format_annotation_target(features, gene_token="[ATT]", bos_token="<BOS>", eos_token="<EOS>"):
    if not features:
        return f"{bos_token}{eos_token}"
    
    lines = []
    for feat in sorted(features, key=lambda x: x["start"]):
        ftype  = feat["type"].lower()
        strand = feat["strand"]
        start  = feat["start"]
        end    = feat["end"]
        lines.append(f"{ftype}\t{strand}\t{start}\t{end}")
    
    return f"{bos_token}\n" + "\n".join(lines) + f"\n{eos_token}"


def build_transcript_to_gene_map(features):
    """
    Build a map of transcript/mRNA ID -> gene ID.
    This allows resolving exon/CDS parents to their gene.
    """
    parent_map = {}
    for f in features:
        ftype = f["type"].lower()
        if ftype in {"mrna", "transcript", "guide_rna", "lnc_rna", "ncrna"}:
            t_id     = f["attributes"].get("ID")
            g_parent = f["attributes"].get("Parent")
            if t_id and g_parent:
                parent_map[t_id] = g_parent
    return parent_map


def create_gene_prediction_dataset(sequences, features_by_seqid, gene_token="[ATT]",
                                    bos_token="<BOS>", eos_token="<EOS>", context_pad=0):
    """
    Create gene prediction dataset by grouping features by Gene ID.
    Resolves transcript->gene relationships to ensure CDS and exons stay together.
    """
    dataset = []
    
    for seqid, sequence in sequences.items():
        features = features_by_seqid.get(seqid, [])
        
        # 1. build transcript -> gene map from ALL features (before filtering)
        parent_map = build_transcript_to_gene_map(features)
        
        # 2. filter for features we want to predict (exon, CDS, etc.)
        gene_features = [
            f for f in features 
            if f["type"] in GENE_FEATURE_TYPES or f["type"].lower() in GENE_FEATURE_TYPES
        ]
        
        if not gene_features:
            continue
        
        # 3. group features by gene ID (resolving through transcript if needed)
        grouped = {}
        for feat in gene_features:
            direct_parent = feat["attributes"].get("Parent", feat["attributes"].get("ID"))
            
            # resolve to gene ID if parent is a transcript
            group_id = parent_map.get(direct_parent, direct_parent)
            
            if group_id:
                if group_id not in grouped:
                    grouped[group_id] = []
                grouped[group_id].append(feat)
        
        # 4. create samples from groups
        for group_id, group_feats in grouped.items():
            if not group_feats:
                continue
            
            min_start = min(f["start"] for f in group_feats)
            max_end   = max(f["end"] for f in group_feats)
            strand    = group_feats[0]["strand"]
            
            seq_start = max(0, min_start - 1 - context_pad)
            seq_end   = min(len(sequence), max_end + context_pad)
            
            chunk_seq = sequence[seq_start:seq_end]
            
            if strand == "-":
                chunk_seq = anti(chunk_seq)
            
            adjusted_features = []
            for f in group_feats:
                adj_f          = f.copy()
                adj_f["start"] = f["start"] - seq_start
                adj_f["end"]   = f["end"] - seq_start
                adjusted_features.append(adj_f)
            
            input_text  = f"{gene_token} {chunk_seq}"
            target_text = format_annotation_target(
                adjusted_features, gene_token, bos_token, eos_token
            )
            
            sample = {
                "seqid":        seqid,
                "parent_id":    group_id,
                "start":        min_start,
                "end":          max_end,
                "strand":       strand,
                "input":        input_text,
                "target":       target_text,
                "num_features": len(group_feats)
            }
            
            dataset.append(sample)
    
    print(f"  Created {len(dataset)} gene prediction samples")
    return dataset


def create_rna_classification_dataset(sequences, features_by_seqid, cls_token="[CLS]",
                                       context_pad=50, include_ncrna=False, rna_classes=None):
    if rna_classes is None:
        rna_classes = RNA_CLASSES
    
    dataset       = []
    skipped_ncrna = 0
    
    for seqid, sequence in sequences.items():
        features = features_by_seqid.get(seqid, [])
        
        for feat in features:
            feat_type       = feat["type"]
            feat_type_lower = feat_type.lower()
            
            if feat_type not in RNA_FEATURE_TYPES and feat_type_lower not in RNA_FEATURE_TYPES:
                continue
            
            if not include_ncrna and feat_type_lower in {"ncrna", "ncrna"}:
                skipped_ncrna += 1
                continue
            
            label     = None
            label_str = None
            
            for class_name, class_id in rna_classes.items():
                if feat_type_lower == class_name.lower():
                    label     = class_id
                    label_str = class_name
                    break
            
            if label is None:
                for class_name, class_id in rna_classes.items():
                    if class_name.lower() in feat_type_lower or feat_type_lower in class_name.lower():
                        label     = class_id
                        label_str = class_name
                        break
            
            if label is None:
                print(f"  Warning: Unknown RNA type '{feat_type}', skipping")
                continue
            
            start   = max(0, feat["start"] - 1 - context_pad)
            end     = min(len(sequence), feat["end"] + context_pad)
            rna_seq = sequence[start:end]
            
            if feat["strand"] == "-":
                rna_seq = anti(rna_seq)
            
            input_text = f"{cls_token} {rna_seq}"
            
            sample = {
                "seqid":        seqid,
                "feature_id":   feat["attributes"].get("ID", f"{seqid}_{feat['start']}"),
                "start":        feat["start"],
                "end":          feat["end"],
                "strand":       feat["strand"],
                "input":        input_text,
                "label":        label,
                "label_str":    label_str,
                "original_type": feat_type
            }
            
            dataset.append(sample)
    
    if skipped_ncrna > 0:
        print(f"  Skipped {skipped_ncrna} ncRNA features (include_ncrna=False)")
    
    print(f"  Created {len(dataset)} RNA classification samples")
    return dataset


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