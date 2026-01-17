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

# ============================================================
# NEW: Biological region support for Ensembl GFF
# ============================================================

# Known biological_region types extracted from logic_name
# These map logic_name values to classification labels
BIOLOGICAL_REGION_TYPES = {
    "intron":   "intron",
    "promoter": "promoter",
    # Add more as discovered
}


def extract_biological_region_type(feature):
    """
    Extract the actual feature type from biological_region's logic_name attribute.
    
    Ensembl GFF uses 'biological_region' as a generic type, with the actual
    type stored in the logic_name attribute.
    
    Example attribute: logic_name=intron -> returns "intron"
    Example attribute: logic_name=promoter -> returns "promoter"
    
    Returns:
        tuple: (extracted_type, is_known)
               extracted_type: the logic_name value or None if not found
               is_known: True if the type is in BIOLOGICAL_REGION_TYPES
    """
    attrs = feature.get("attributes", {})
    logic_name = attrs.get("logic_name")
    
    if logic_name:
        logic_name_lower = logic_name.lower()
        is_known = logic_name_lower in BIOLOGICAL_REGION_TYPES
        return logic_name_lower, is_known
    
    return None, False


def discover_biological_region_types(features):
    """
    Scan features and discover all biological_region types from logic_name.
    
    Returns:
        dict: {logic_name: count} for all biological_region features
    """
    discovered = {}
    
    for feat in features:
        if feat["type"].lower() == "biological_region":
            extracted_type, _ = extract_biological_region_type(feat)
            if extracted_type:
                discovered[extracted_type] = discovered.get(extracted_type, 0) + 1
    
    return discovered


def report_biological_region_types(features, known_rna_classes=None):
    """
    Report discovered biological_region types and categorize them.
    
    Args:
        features: List of parsed GFF features
        known_rna_classes: Dict of known RNA classes (for checking coverage)
    
    Returns:
        dict: {
            'known': {type: count},      # Types in BIOLOGICAL_REGION_TYPES
            'unknown': {type: count},    # Types NOT in BIOLOGICAL_REGION_TYPES
            'in_rna_classes': {type: count},  # Unknown but present in rna_classes
            'needs_token': {type: count}      # Truly new types needing tokens
        }
    """
    if known_rna_classes is None:
        known_rna_classes = RNA_CLASSES
    
    discovered = discover_biological_region_types(features)
    
    result = {
        'known': {},
        'unknown': {},
        'in_rna_classes': {},
        'needs_token': {}
    }
    
    known_rna_lower = {k.lower() for k in known_rna_classes.keys()}
    
    for typ, count in discovered.items():
        typ_lower = typ.lower()
        
        if typ_lower in BIOLOGICAL_REGION_TYPES:
            result['known'][typ] = count
        else:
            result['unknown'][typ] = count
            
            # Check if it's already in RNA classes
            if typ_lower in known_rna_lower:
                result['in_rna_classes'][typ] = count
            else:
                result['needs_token'][typ] = count
    
    return result

# ============================================================
# END NEW SECTION
# ============================================================


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
                                       context_pad=50, include_ncrna=False, rna_classes=None,
                                       include_biological_region=True, 
                                       biological_region_classes=None):
    """
    Create RNA classification dataset.
    
    NEW: Handles biological_region features by extracting type from logic_name.
    
    Args:
        sequences: Dict of seqid -> sequence
        features_by_seqid: Dict of seqid -> list of features
        cls_token: Classification token prefix
        context_pad: Base pairs to pad around feature
        include_ncrna: Whether to include ncRNA features
        rna_classes: Dict of class_name -> class_id
        include_biological_region: Whether to process biological_region features
        biological_region_classes: Dict of logic_name -> class_id for biological_region
                                   If None, uses rna_classes for matching
    
    Returns:
        tuple: (dataset, discovered_bio_types)
               dataset: list of classification samples
               discovered_bio_types: dict of discovered biological_region types
    """
    if rna_classes is None:
        rna_classes = RNA_CLASSES
    
    dataset              = []
    skipped_ncrna        = 0
    discovered_bio_types = {}  # NEW: track discovered biological_region types
    bio_region_used      = {}  # NEW: track which bio types were actually used
    bio_region_skipped   = {}  # NEW: track which bio types were skipped (no class mapping)
    
    for seqid, sequence in sequences.items():
        features = features_by_seqid.get(seqid, [])
        
        for feat in features:
            feat_type       = feat["type"]
            feat_type_lower = feat_type.lower()
            
            # ============================================================
            # NEW: Handle biological_region specially
            # ============================================================
            if feat_type_lower == "biological_region" and include_biological_region:
                extracted_type, is_known = extract_biological_region_type(feat)
                
                if extracted_type:
                    # Track discovery
                    discovered_bio_types[extracted_type] = discovered_bio_types.get(extracted_type, 0) + 1
                    
                    # Try to find a class mapping
                    label     = None
                    label_str = None
                    
                    # First check biological_region_classes if provided
                    if biological_region_classes and extracted_type in biological_region_classes:
                        label     = biological_region_classes[extracted_type]
                        label_str = extracted_type
                    else:
                        # Fall back to rna_classes
                        for class_name, class_id in rna_classes.items():
                            if extracted_type == class_name.lower():
                                label     = class_id
                                label_str = class_name
                                break
                    
                    if label is not None:
                        bio_region_used[extracted_type] = bio_region_used.get(extracted_type, 0) + 1
                        
                        # Create the sample
                        start   = max(0, feat["start"] - 1 - context_pad)
                        end     = min(len(sequence), feat["end"] + context_pad)
                        rna_seq = sequence[start:end]
                        
                        if feat["strand"] == "-":
                            rna_seq = anti(rna_seq)
                        
                        input_text = f"{cls_token} {rna_seq}"
                        
                        sample = {
                            "seqid":          seqid,
                            "feature_id":     feat["attributes"].get("ID", f"{seqid}_{feat['start']}"),
                            "start":          feat["start"],
                            "end":            feat["end"],
                            "strand":         feat["strand"],
                            "input":          input_text,
                            "label":          label,
                            "label_str":      label_str,
                            "original_type":  feat_type,
                            "bio_region_type": extracted_type,  # NEW: track original bio type
                        }
                        
                        dataset.append(sample)
                    else:
                        bio_region_skipped[extracted_type] = bio_region_skipped.get(extracted_type, 0) + 1
                
                continue  # Done with this biological_region feature
            # ============================================================
            # END NEW biological_region handling
            # ============================================================
            
            # Original logic for non-biological_region features
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
    
    # ============================================================
    # NEW: Report biological_region discovery
    # ============================================================
    if discovered_bio_types:
        print(f"\n  [biological_region discovery]")
        print(f"    Total biological_region features: {sum(discovered_bio_types.values())}")
        print(f"    Unique types found: {len(discovered_bio_types)}")
        
        if bio_region_used:
            print(f"    Types USED (had class mapping):")
            for typ, cnt in sorted(bio_region_used.items(), key=lambda x: -x[1]):
                print(f"      {typ}: {cnt}")
        
        if bio_region_skipped:
            print(f"    Types SKIPPED (no class mapping - NEED TOKENS):")
            for typ, cnt in sorted(bio_region_skipped.items(), key=lambda x: -x[1]):
                print(f"      {typ}: {cnt}  <-- ADD TO RNA_CLASSES")
    # ============================================================
    
    if skipped_ncrna > 0:
        print(f"  Skipped {skipped_ncrna} ncRNA features (include_ncrna=False)")
    
    print(f"  Created {len(dataset)} RNA classification samples")
    
    # Return both dataset and discovery info
    return dataset, {
        'discovered': discovered_bio_types,
        'used': bio_region_used,
        'skipped': bio_region_skipped
    }


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