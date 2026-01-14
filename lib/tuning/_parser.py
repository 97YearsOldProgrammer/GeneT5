import re
import json
import gzip
import sys
from pathlib     import Path
from contextlib  import closing
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
    chm  = None
    seq  = []
    
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

    fp       = getfp(filename)
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

def get_parent_id(feat):

    attrs = feat["attributes"]
    if "Parent" in attrs:
        return attrs["Parent"].split(",")[0]
    if "ID" in attrs:
        return attrs["ID"]
    return None

def group_features_by_parent(features, filter_types=None):

    grouped = defaultdict(list)
    
    for feat in features:
        if filter_types and feat["type"] not in filter_types:
            continue
        
        parent_id = get_parent_id(feat)
        if parent_id is None:
            continue
        
        grouped[parent_id].append(feat)
    
    return dict(grouped)


#######################
#####  Ab initio  #####
#######################


GENE_FEATURE_TYPES = {
    "exon", "intron", "cds",
    "five_prime_utr", "three_prime_utr",
    "start_codon", "stop_codon",
}

def format_stripped_gff(features, offset=0):

    lines = []
    for feat in sorted(features, key=lambda x: x["start"]):
        # adjust coords relative to extracted region
        rel_start = feat["start"] - offset
        rel_end   = feat["end"] - offset
        
        phase = feat["phase"] if feat["phase"] != "." else "."
        
        line = "\t".join([
            feat["type"],
            str(rel_start),
            str(rel_end),
            feat["strand"],
            str(phase),
        ])
        lines.append(line)
    
    return "\n".join(lines)

def create_gene_prediction_dataset(sequences, features_by_seqid,
                                   window_size=None, stride=None,
                                   gene_token="[GENE]", bos_token="<BOS>",
                                   eos_token="<EOS>", context_pad=0):

    dataset = []
    
    for seqid, seq in sequences.items():
        if seqid not in features_by_seqid:
            continue
        
        feats = features_by_seqid[seqid]
        
        # group features by parent (transcript/gene level)
        grouped = group_features_by_parent(feats, filter_types=GENE_FEATURE_TYPES)
        
        # also collect gene-level features for span reference
        gene_spans = {}
        for feat in feats:
            if feat["type"] == "gene":
                gene_id = feat["attributes"].get("ID")
                if gene_id:
                    gene_spans[gene_id] = (feat["start"], feat["end"], feat["strand"])
        
        for parent_id, group_feats in grouped.items():
            if not group_feats:
                continue
            
            # sort by position
            group_feats = sorted(group_feats, key=lambda x: x["start"])
            
            # get span from features
            span_start = min(f["start"] for f in group_feats)
            span_end   = max(f["end"] for f in group_feats)
            strand     = group_feats[0]["strand"]
            
            # check if we have gene-level span
            if parent_id in gene_spans:
                span_start, span_end, strand = gene_spans[parent_id]
            
            # apply context padding
            ext_start = max(1, span_start - context_pad)
            ext_end   = min(len(seq), span_end + context_pad)
            
            # extract sequence (0-based slicing)
            dna_seq = seq[ext_start - 1 : ext_end]
            
            # handle reverse strand
            if strand == "-":
                dna_seq = anti(dna_seq)
            
            # format stripped gff (coords relative to ext_start)
            target_gff = format_stripped_gff(group_feats, offset=ext_start - 1)
            
            # build input with tokens
            input_seq = f"{bos_token} {gene_token} {dna_seq} {eos_token}"
            target    = f"{bos_token}\n{target_gff}\n{eos_token}"
            
            dataset.append({
                "parent_id": parent_id,
                "seqid":     seqid,
                "start":     span_start,
                "end":       span_end,
                "strand":    strand,
                "input":     input_seq,
                "target":    target,
            })
    
    # windowed mode for large genomes
    if window_size is not None:
        dataset = create_windowed_dataset(
            sequences, features_by_seqid,
            window_size, stride,
            gene_token, bos_token, eos_token
        )
    
    return dataset

def create_windowed_dataset(sequences, features_by_seqid,
                            window_size, stride,
                            gene_token, bos_token, eos_token):

    dataset = []
    step    = stride or window_size
    
    for seqid, seq in sequences.items():
        if seqid not in features_by_seqid:
            continue
        
        feats   = features_by_seqid[seqid]
        seq_len = len(seq)
        
        # group features by parent first
        grouped = group_features_by_parent(feats, filter_types=GENE_FEATURE_TYPES)
        
        for win_start in range(0, seq_len, step):
            win_end   = min(win_start + window_size, seq_len)
            win_seq   = seq[win_start:win_end]
            
            # gff uses 1-based coords
            gff_start = win_start + 1
            gff_end   = win_end
            
            # collect complete transcripts within window
            win_groups = {}
            
            for parent_id, group_feats in grouped.items():
                # check if all features of this group fall within window
                group_start = min(f["start"] for f in group_feats)
                group_end   = max(f["end"] for f in group_feats)
                
                if group_start >= gff_start and group_end <= gff_end:
                    # adjust coords relative to window
                    adjusted = []
                    for f in group_feats:
                        adj        = f.copy()
                        adj["start"] = f["start"] - win_start
                        adj["end"]   = f["end"] - win_start
                        adjusted.append(adj)
                    win_groups[parent_id] = adjusted
            
            if not win_groups:
                continue
            
            # format all groups in window
            all_feats = []
            for grp in win_groups.values():
                all_feats.extend(grp)
            
            target_gff = format_stripped_gff_raw(all_feats)
            
            input_seq = f"{bos_token} {gene_token} {win_seq} {eos_token}"
            target    = f"{bos_token}\n{target_gff}\n{eos_token}"
            
            dataset.append({
                "seqid":  seqid,
                "start":  gff_start,
                "end":    gff_end,
                "input":  input_seq,
                "target": target,
            })
    
    return dataset

def format_stripped_gff_raw(features):

    lines = []
    for feat in sorted(features, key=lambda x: x["start"]):
        phase = feat["phase"] if feat["phase"] != "." else "."
        
        line = "\t".join([
            feat["type"],
            str(feat["start"]),
            str(feat["end"]),
            feat["strand"],
            str(phase),
        ])
        lines.append(line)
    
    return "\n".join(lines)


############################
#####  Classification  #####
############################


RNA_CLASSES = {
    "pseudogene":       0,
    "ncrna":            1,
    "trna":             2,
    "rrna":             3,
    "tmrma":            4,
    "srna":             5,
    "misc_rna":         6,
    "antisense_rna":    7,
    "rnase_p_rna":      8,
    "srp_rna":          9,
    
    # prokaryotic
    "origin_of_replication":  7,
    "mobile_genetic_element": 8,
}

def extract_rna_features(features):

    rna_types = set(RNA_CLASSES.keys())
    rna_feats = []
    
    for feat in features:
        ftype = feat["type"]
        
        # direct match
        if ftype in rna_types:
            rna_feats.append(feat)
            continue
        
        # check for ncrna subtype
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
            # convert to 0-based for slicing
            start = max(0, feat["start"] - 1 - context_pad)
            end   = min(len(seq), feat["end"] + context_pad)
            
            dna_seq = seq[start:end]
            
            # handle reverse strand
            if feat["strand"] == "-":
                dna_seq = anti(dna_seq)
            
            # determine label
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    
    print(f"Saved {len(dataset)} samples to {output_path}")


def load_dataset(data_path):

    samples = []
    with open(data_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples
