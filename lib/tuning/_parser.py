import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any



def anti(seq):
    """Reverse complement of DNA sequence."""
    comp = str.maketrans('ACGTRYMKBDHVNacgtrymkbdhvn', 'TGCAYRKMVHDBNtgcayrkmvhdbn')
    anti_seq = seq.translate(comp)[::-1]
    return anti_seq



#######################
#####  Constants  #####
#######################


# Gene feature types for annotation task
GENE_FEATURE_TYPES = {
    "gene", "exon", "intron", "CDS", "cds",
    "five_prime_UTR", "three_prime_UTR", "utr5", "utr3",
}

# RNA classes for classification task
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

# RNA feature types to extract from GFF
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


def parse_fasta(fasta_path: str) -> Dict[str, str]:
    """
    Parse FASTA file into dictionary of sequences.
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        Dict mapping sequence ID to sequence string
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                # Save previous sequence
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                
                # Parse new header
                header = line[1:].split()[0]
                current_id = header
                current_seq = []
            else:
                current_seq.append(line.upper())
        
        # Save last sequence
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def parse_gff(gff_path: str) -> List[Dict[str, Any]]:
    """
    Parse GFF3 file into list of feature dictionaries.
    
    Args:
        gff_path: Path to GFF3 file
        
    Returns:
        List of feature dictionaries with keys:
        seqid, source, type, start, end, score, strand, phase, attributes
    """
    features = []
    
    with open(gff_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 9:
                continue
            
            # Parse attributes
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
    
    return features


def group_features_by_seqid(features: List[Dict]) -> Dict[str, List[Dict]]:
    """Group features by their sequence ID."""
    grouped = {}
    for feat in features:
        seqid = feat["seqid"]
        if seqid not in grouped:
            grouped[seqid] = []
        grouped[seqid].append(feat)
    return grouped


def group_features_by_parent(features: List[Dict]) -> Dict[str, List[Dict]]:
    """Group features by their parent ID for hierarchical annotation."""
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


def format_annotation_target(
    features: List[Dict],
    gene_token: str = "[ATT]",
    bos_token: str = "<BOS>",
    eos_token: str = "<EOS>"
) -> str:
    """
    Format features into annotation target string.
    
    Format: <BOS> type strand start end <tab> type strand start end ... <EOS>
    """
    if not features:
        return f"{bos_token}{eos_token}"
    
    lines = []
    for feat in sorted(features, key=lambda x: x["start"]):
        ftype = feat["type"].lower()
        strand = feat["strand"]
        start = feat["start"]
        end = feat["end"]
        lines.append(f"{ftype}\t{strand}\t{start}\t{end}")
    
    return f"{bos_token}\n" + "\n".join(lines) + f"\n{eos_token}"


def create_gene_prediction_dataset(
    sequences: Dict[str, str],
    features_by_seqid: Dict[str, List[Dict]],
    gene_token: str = "[ATT]",
    bos_token: str = "<BOS>",
    eos_token: str = "<EOS>",
    context_pad: int = 0
) -> List[Dict]:
    """
    Create gene prediction dataset (basic version without chunking).
    
    Args:
        sequences: Dict of seqid -> sequence
        features_by_seqid: Dict of seqid -> features
        gene_token: Special token for annotation task
        bos_token: Beginning of sequence token
        eos_token: End of sequence token
        context_pad: Context padding around features (bp)
        
    Returns:
        List of dataset samples
    """
    dataset = []
    
    for seqid, sequence in sequences.items():
        features = features_by_seqid.get(seqid, [])
        
        # Filter to gene-related features
        gene_features = [
            f for f in features 
            if f["type"] in GENE_FEATURE_TYPES or f["type"].lower() in GENE_FEATURE_TYPES
        ]
        
        if not gene_features:
            continue
        
        # Group by parent for hierarchical processing
        grouped, orphans = group_features_by_parent(gene_features)
        
        for parent_id, group_feats in grouped.items():
            if not group_feats:
                continue
            
            # Get span
            min_start = min(f["start"] for f in group_feats)
            max_end = max(f["end"] for f in group_feats)
            strand = group_feats[0]["strand"]
            
            # Extract sequence with padding
            seq_start = max(0, min_start - 1 - context_pad)
            seq_end = min(len(sequence), max_end + context_pad)
            
            chunk_seq = sequence[seq_start:seq_end]
            
            # Handle reverse strand
            if strand == "-":
                chunk_seq = anti(chunk_seq)
            
            # Adjust feature coordinates
            adjusted_features = []
            for f in group_feats:
                adj_f = f.copy()
                adj_f["start"] = f["start"] - seq_start
                adj_f["end"] = f["end"] - seq_start
                adjusted_features.append(adj_f)
            
            # Format input and target
            input_text = f"{gene_token} {chunk_seq}"
            target_text = format_annotation_target(
                adjusted_features, gene_token, bos_token, eos_token
            )
            
            sample = {
                "seqid":        seqid,
                "parent_id":    parent_id,
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


def create_rna_classification_dataset(
    sequences: Dict[str, str],
    features_by_seqid: Dict[str, List[Dict]],
    cls_token: str = "[CLS]",
    context_pad: int = 50,
    include_ncrna: bool = False,
    rna_classes: Dict[str, int] = None
) -> List[Dict]:
    """
    Create RNA classification dataset.
    
    Args:
        sequences: Dict of seqid -> sequence
        features_by_seqid: Dict of seqid -> features
        cls_token: Classification token
        context_pad: Context padding around RNA features (bp)
        include_ncrna: Whether to include generic ncRNA class
        rna_classes: Custom RNA classes dict (uses default if None)
        
    Returns:
        List of dataset samples
    """
    if rna_classes is None:
        rna_classes = RNA_CLASSES
    
    dataset = []
    skipped_ncrna = 0
    
    for seqid, sequence in sequences.items():
        features = features_by_seqid.get(seqid, [])
        
        for feat in features:
            feat_type = feat["type"]
            feat_type_lower = feat_type.lower()
            
            # Check if it's an RNA feature
            if feat_type not in RNA_FEATURE_TYPES and feat_type_lower not in RNA_FEATURE_TYPES:
                continue
            
            # Skip ncRNA if configured
            if not include_ncrna and feat_type_lower in {"ncrna", "ncrna"}:
                skipped_ncrna += 1
                continue
            
            # Determine class label
            label = None
            label_str = None
            
            for class_name, class_id in rna_classes.items():
                if feat_type_lower == class_name.lower():
                    label = class_id
                    label_str = class_name
                    break
            
            if label is None:
                # Try partial match
                for class_name, class_id in rna_classes.items():
                    if class_name.lower() in feat_type_lower or feat_type_lower in class_name.lower():
                        label = class_id
                        label_str = class_name
                        break
            
            if label is None:
                print(f"  Warning: Unknown RNA type '{feat_type}', skipping")
                continue
            
            # Extract sequence with context
            start = max(0, feat["start"] - 1 - context_pad)
            end = min(len(sequence), feat["end"] + context_pad)
            rna_seq = sequence[start:end]
            
            # Handle reverse strand
            if feat["strand"] == "-":
                rna_seq = anti(rna_seq)
            
            # Format input
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


def save_dataset(dataset: List[Dict], output_path: str) -> None:
    """Save dataset to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in dataset:
            f.write(json.dumps(sample) + '\n')
    
    print(f"  Saved {len(dataset)} samples to {output_path}")


def load_dataset(input_path: str) -> List[Dict]:
    """Load dataset from JSONL file."""
    dataset = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset