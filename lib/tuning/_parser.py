
import re
import json
import gzip
import sys
from pathlib import Path
from contextlib import closing
from collections import defaultdict



def anti(seq):
    """Reverse complement of DNA sequence."""
    comp = str.maketrans('ACGTRYMKBDHVNacgtrymkbdhvn', 'TGCAYRKMVHDBNtgcayrkmvhdbn')
    anti = seq.translate(comp)[::-1]
    return anti



#####################
#####  Utility  #####
#####################


def getfp(filename):

    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt', encoding='ISO-8859-1')
    elif filename == '-':
        return sys.stdin
    return open(filename)

def read_fasta(filename):

    name = None
    seqs = []
    
    fp = getfp(filename)
    
    for line in fp:
        line = line.rstrip()
        if line.startswith('>'):
            if len(seqs) > 0:
                seq = ''.join(seqs)
                yield(name, seq)
                name = line[1:].split()[0]
                seqs = []
            else:
                name = line[1:].split()[0]
        else:
            seqs.append(line)
    
    if name:
        yield(name, ''.join(seqs))
    fp.close()

def parse_fasta(filename):
    
    fp = getfp(filename)
    
    chms = {}
    chm = None
    seq = []
    
    with closing(fp):
        for line in fp:
            line = line.strip()
            if line.startswith('>'):
                if chm is not None:
                    chms[chm] = ''.join(seq)
                chm = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.upper())
        
        if chm is not None:
            chms[chm] = ''.join(seq)
    
    return chms

def parse_att(att):

    attributes = {}
    
    if not att or att == ".":
        return attributes
    
    for stuff in att.split(";"):
        stuff = stuff.strip()
        if not stuff:
            continue
        if "=" in stuff:
            key, value = stuff.split("=", 1)
        elif " " in stuff:
            key, value = stuff.split(" ", 1)
        else:
            key, value = stuff, ""
        attributes[key.strip()] = value.strip()
    
    return attributes

def parse_gff(filename):

    fp = getfp(filename)
    features = []
    
    with closing(fp):
        for line in fp:
            line = line.strip()
            
            if not line or line.startswith("#"):
                continue
            
            parts = line.split("\t")
            
            if len(parts) == 8:
                parts.append(".")
            elif len(parts) != 9:
                continue
            
            seqid, source, ftype, start, end, score, strand, phase, attrs = parts
            
            feature = {
                "seqid":      seqid,
                "source":     source,
                "type":       ftype.lower(),
                "start":      int(start),
                "end":        int(end),
                "score":      None if score == "." else float(score),
                "strand":     strand,
                "phase":      "." if phase == "." else int(phase),
                "attributes": parse_att(attrs),
            }
            features.append(feature)
    
    return features

def group_features_by_seqid(features):

    grouped = defaultdict(list)
    for feat in features:
        grouped[feat["seqid"]].append(feat)
    return dict(grouped)

def group_features_by_parent(features, filter_types=None):

    grouped = defaultdict(list)
    
    for feat in features:
        if filter_types and feat["type"] not in filter_types:
            continue
        
        # Choose parent ID
        attrs = feat["attributes"]
        if "Parent" in attrs:
            parent_id = attrs["Parent"].split(",")[0]
        elif "ID" in attrs:
            parent_id = attrs["ID"]
        else:
            parent_id = feat["seqid"]
        
        grouped[parent_id].append(feat)
    
    return dict(grouped)


#######################
#####  Ab initio  #####
#######################


def format_stripped_gff(features):
    """
    Format features into stripped GFF for prediction target.
    Output columns: type, start, end, strand, phase
    """
    lines = []
    for feat in sorted(features, key=lambda x: x["start"]):
        line = "\t".join([
            feat["type"],
            str(feat["start"]),
            str(feat["end"]),
            feat["strand"],
            str(feat["phase"]),
        ])
        lines.append(line)
    return "\n".join(lines)

def create_gene_prediction_dataset(sequences, features_by_seqid, 
                                   window_size=None, stride=None,
                                   gene_token="[GENE]"):
    """
    Create gene prediction dataset.
    
    If window_size is set, slides a window across each chromosome.
    Otherwise uses full sequences (for smaller genomes like E. coli).
    """
    dataset = []
    
    for seqid, seq in sequences.items():
        if seqid not in features_by_seqid:
            continue
        
        feats = features_by_seqid[seqid]
        
        if window_size is None:
            # Full sequence mode
            input_seq  = f"{gene_token} {seq}"
            target_gff = format_stripped_gff(feats)
            
            dataset.append({
                "seqid":  seqid,
                "start":  1,
                "end":    len(seq),
                "input":  input_seq,
                "target": target_gff,
            })
        else:
            # Sliding window mode for large genomes
            step = stride or window_size
            seq_len = len(seq)
            
            for win_start in range(0, seq_len, step):
                win_end    = min(win_start + window_size, seq_len)
                win_seq    = seq[win_start:win_end]
                
                # GFF uses 1-based coords
                gff_start  = win_start + 1
                gff_end    = win_end
                
                # Filter features within window
                win_feats  = []
                for f in feats:
                    if f["start"] >= gff_start and f["end"] <= gff_end:
                        # Adjust coords relative to window
                        adjusted = f.copy()
                        adjusted["start"] = f["start"] - win_start
                        adjusted["end"]   = f["end"] - win_start
                        win_feats.append(adjusted)
                
                if not win_feats:
                    continue
                
                input_seq  = f"{gene_token} {win_seq}"
                target_gff = format_stripped_gff(win_feats)
                
                dataset.append({
                    "seqid":  seqid,
                    "start":  gff_start,
                    "end":    gff_end,
                    "input":  input_seq,
                    "target": target_gff,
                })
    
    return dataset


############################
#####  Classification  #####
############################


RNA_CLASSES = {
    "pseudogene":   0,
    "ncrna":        1,
    "trna":         2,
    "rrna":         3,
    "tmrna":        4,
    "srna":         5,
    "misc_rna":     6,
      
    # Prokaryotic
    "origin_of_replication":  7,
    "mobile_genetic_element": 8,
}

def extract_rna_features(features):

    rna_types = set(RNA_CLASSES.keys())
    rna_feats = []
    
    for feat in features:
        ftype = feat["type"]
        
        # Direct match
        if ftype in rna_types:
            rna_feats.append(feat)
            continue
        
        # Check for ncRNA subtype in attributes
        if ftype == "ncrna" or "ncrna" in ftype:
            rna_feats.append(feat)
    
    return rna_feats

def create_rna_classification_dataset(sequences, features_by_seqid,
                                      cls_token="[CLS]", context_pad=0):

    dataset = []
    
    for seqid, seq in sequences.items():
        if seqid not in features_by_seqid:
            continue
        
        rna_feats = extract_rna_features(features_by_seqid[seqid])
        
        for feat in rna_feats:
            # Convert to 0-based for slicing
            start = max(0, feat["start"] - 1 - context_pad)
            end   = min(len(seq), feat["end"] + context_pad)
            
            dna_seq = seq[start:end]
            
            # Handle reverse strand
            if feat["strand"] == "-":
                dna_seq = anti(dna_seq)
            
            # Determine label
            ftype = feat["type"]
            if ftype in RNA_CLASSES:
                label = RNA_CLASSES[ftype]
            else:
                label = RNA_CLASSES.get("ncrna", 0)
            
            input_seq = f"{cls_token} {dna_seq}"
            
            dataset.append({
                "seqid":     seqid,
                "start":     feat["start"],
                "end":       feat["end"],
                "strand":    feat["strand"],
                "input":     input_seq,
                "label":     label,
                "label_str": ftype,
            })
    
    return dataset


####################
#####  Output  #####
####################


def save_dataset(dataset, output_path):
    """Save dataset as JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    
    print(f"Saved {len(dataset)} samples to {output_path}")


def load_dataset(data_path):
    """Load dataset from JSONL."""
    samples = []
    with open(data_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples