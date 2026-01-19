import json
from pathlib import Path


################################
#####  Chunking Functions  #####
################################


def estimate_tokens(text, chars_per_token=4.0):
    return int(len(text) / chars_per_token)


def estimate_gff_tokens(gff_lines, tokens_per_line=5.0):
    return int(len(gff_lines) * tokens_per_line)


def find_gene_boundaries(features, start_bp, end_bp):
    gene_features = [
        f for f in features 
        if f["type"].lower() in {"gene", "mrna", "transcript"}
    ]
    
    adjusted_start = start_bp
    adjusted_end   = end_bp
    
    for gene in gene_features:
        gene_start = gene["start"]
        gene_end   = gene["end"]
        
        if gene_start < adjusted_start < gene_end:
            adjusted_start = gene_start
        
        if gene_start < adjusted_end < gene_end:
            adjusted_end = gene_end
    
    return adjusted_start, adjusted_end


def chunk_sequence_with_overlap(sequence, features, window_size, stride, respect_gene_boundaries=True):
    seq_len = len(sequence)
    chunks  = []
    
    if seq_len <= window_size:
        chunk_features = [f for f in features if f["start"] >= 1 and f["end"] <= seq_len]
        return [(0, seq_len, sequence, chunk_features)]
    
    start = 0
    while start < seq_len:
        end = min(start + window_size, seq_len)
        
        if respect_gene_boundaries and features:
            adjusted_start, adjusted_end = find_gene_boundaries(features, start, end)
            
            if adjusted_end - adjusted_start <= window_size * 1.5:
                start, end = adjusted_start, adjusted_end
        
        chunk_seq = sequence[start:end]
        
        chunk_features = [
            f for f in features
            if f["start"] >= start + 1 and f["end"] <= end
        ]
        
        chunks.append((start, end, chunk_seq, chunk_features))
        
        start += stride
        
        if end >= seq_len:
            break
    
    return chunks


def chunk_gff_with_overlap(features, max_lines, overlap_lines):
    if len(features) <= max_lines:
        return [features]
    
    chunks = []
    start  = 0
    
    while start < len(features):
        end   = min(start + max_lines, len(features))
        chunk = features[start:end]
        chunks.append(chunk)
        
        start = end - overlap_lines
        
        if end >= len(features):
            break
    
    return chunks


def should_chunk_annotation(features, max_lines, max_tokens=2000):
    if len(features) > max_lines:
        return True
    
    estimated_tokens = estimate_gff_tokens([f.get("raw_line", "") for f in features])
    return estimated_tokens > max_tokens


#######################
#####  Validation #####
#######################


def validate_chunks(chunks, original_features):
    all_chunk_features = []
    for _, _, _, chunk_features in chunks:
        all_chunk_features.extend(chunk_features)
    
    unique_features = {
        (f["seqid"], f["start"], f["end"], f["type"]) 
        for f in all_chunk_features
    }
    
    original_set = {
        (f["seqid"], f["start"], f["end"], f["type"]) 
        for f in original_features
    }
    
    missing = original_set - unique_features
    extra   = unique_features - original_set
    
    return {
        "num_chunks":       len(chunks),
        "original_count":   len(original_features),
        "chunk_count":      len(all_chunk_features),
        "unique_count":     len(unique_features),
        "missing":          len(missing),
        "missing_features": list(missing)[:10],
        "valid":            len(missing) == 0
    }


#######################################
#####  Hierarchy Building         #####
#######################################


def build_transcript_to_gene_map(features):
    """
    Build a map of transcript/mRNA ID -> gene ID.
    """
    parent_map = {}
    for f in features:
        ftype = f["type"].lower()
        if ftype in {"mrna", "transcript", "rrna", "trna", "ncrna", 
                     "snrna", "snorna", "guide_rna", "lnc_rna"}:
            t_id     = f["attributes"].get("ID")
            g_parent = f["attributes"].get("Parent")
            if t_id and g_parent:
                parent_map[t_id] = g_parent
    return parent_map


def build_transcript_info(features):
    """
    Build transcript info map with biotype.
    
    Returns:
        {transcript_id: {"parent": gene_id, "biotype": str, "start": int}}
    """
    transcript_info = {}
    
    sorted_feats = sorted(features, key=lambda x: x["start"])
    
    for feat in sorted_feats:
        ftype = feat["type"].lower()
        attrs = feat["attributes"]
        
        if ftype in {"mrna", "transcript", "rrna", "trna", "ncrna",
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
                }
    
    return transcript_info


def group_features_by_gene_with_biotype(features):
    """
    Group features by gene ID with biotype tracking.
    
    Returns:
        {gene_id: {"features": [...], "biotype": str, "start": int, "end": int, "strand": str}}
    """
    try:
        from ._parser import GENE_FEATURE_TYPES
    except ImportError:
        from _parser import GENE_FEATURE_TYPES
    
    transcript_info = build_transcript_info(features)
    parent_map      = build_transcript_to_gene_map(features)
    
    # filter to gene-related features
    gene_features = [
        f for f in features
        if f["type"] in GENE_FEATURE_TYPES or f["type"].lower() in GENE_FEATURE_TYPES
    ]
    
    if not gene_features:
        return {}
    
    # track biotype per gene
    gene_biotypes = {}
    for t_id, t_info in transcript_info.items():
        gene_id = t_info["parent"]
        if gene_id:
            if gene_id not in gene_biotypes:
                gene_biotypes[gene_id] = []
            gene_biotypes[gene_id].append(t_info["biotype"])
    
    # group features
    groups = {}
    for feat in gene_features:
        direct_parent = feat["attributes"].get("Parent", feat["attributes"].get("ID"))
        
        gene_id = parent_map.get(direct_parent, direct_parent)
        
        if gene_id:
            if gene_id not in groups:
                groups[gene_id] = {
                    "features": [],
                    "biotype":  "unknown",
                    "start":    feat["start"],
                    "end":      feat["end"],
                    "strand":   feat["strand"],
                }
            
            groups[gene_id]["features"].append(feat)
            groups[gene_id]["start"]  = min(groups[gene_id]["start"], feat["start"])
            groups[gene_id]["end"]    = max(groups[gene_id]["end"], feat["end"])
    
    # assign biotypes
    for gene_id in groups:
        if gene_id in gene_biotypes and gene_biotypes[gene_id]:
            groups[gene_id]["biotype"] = gene_biotypes[gene_id][0]
    
    return groups


#######################
#####  Main Entry #####
#######################


def create_gene_prediction_dataset_with_chunking(sequences, features_by_seqid, window_size=None,
                                                  stride=None, gene_token="[ATT]", bos_token="<BOS>",
                                                  eos_token="<EOS>", context_pad=0, 
                                                  max_gff_lines=400, overlap_bp=50, overlap_lines=20):
    try:
        from ._parser import GENE_FEATURE_TYPES, anti
    except ImportError:
        from _parser import GENE_FEATURE_TYPES, anti
    
    dataset       = []
    total_genes   = 0
    chunked_genes = 0
    
    if stride is None and window_size is not None:
        stride = window_size // 2
    
    for seqid, sequence in sequences.items():
        features = features_by_seqid.get(seqid, [])
        
        if not features:
            continue
        
        # group features by gene with biotype
        gene_groups = group_features_by_gene_with_biotype(features)
        
        if not gene_groups:
            continue
        
        # sort genes by position for consistent indexing
        sorted_genes = sorted(gene_groups.items(), key=lambda x: x[1]["start"])
        total_genes += len(sorted_genes)
        
        for gene_index, (gene_id, gene_data) in enumerate(sorted_genes, start=1):
            group_feats = gene_data["features"]
            biotype     = gene_data["biotype"]
            
            if not group_feats:
                continue
            
            min_start = gene_data["start"]
            max_end   = gene_data["end"]
            strand    = gene_data["strand"]
            
            # extract sequence with context
            seq_start = max(0, min_start - 1 - context_pad)
            seq_end   = min(len(sequence), max_end + context_pad)
            chunk_seq = sequence[seq_start:seq_end]
            
            if strand == "-":
                chunk_seq = anti(chunk_seq)
            
            # adjust coordinates
            adjusted_features = []
            for f in group_feats:
                adj_f          = f.copy()
                adj_f["start"] = f["start"] - seq_start
                adj_f["end"]   = f["end"] - seq_start
                adjusted_features.append(adj_f)
            
            # check chunking needs
            if window_size is not None and len(chunk_seq) > window_size:
                chunks = chunk_sequence_with_overlap(
                    chunk_seq, adjusted_features, window_size, stride, False
                )
                chunked_genes += 1
            elif should_chunk_annotation(adjusted_features, max_gff_lines):
                gff_chunks = chunk_gff_with_overlap(adjusted_features, max_gff_lines, overlap_lines)
                chunks     = [(0, len(chunk_seq), chunk_seq, gc) for gc in gff_chunks]
                chunked_genes += 1
            else:
                chunks = [(0, len(chunk_seq), chunk_seq, adjusted_features)]
            
            # create samples
            for chunk_idx, (c_start, c_end, c_seq, c_feats) in enumerate(chunks):
                if not c_feats:
                    continue
                
                input_text  = f"{gene_token} {c_seq}"
                target_text = format_annotation_target_chunked(
                    c_feats, gene_index, biotype, bos_token, eos_token
                )
                
                sample = {
                    "seqid":        seqid,
                    "parent_id":    gene_id,
                    "start":        min_start + c_start,
                    "end":          min_start + c_end,
                    "strand":       strand,
                    "gene_index":   gene_index,
                    "biotype":      biotype,
                    "chunk_idx":    chunk_idx,
                    "input":        input_text,
                    "target":       target_text,
                    "num_features": len(c_feats),
                    "is_chunked":   len(chunks) > 1
                }
                
                dataset.append(sample)
    
    print(f"  Found {total_genes} genes across {len(sequences)} sequence(s)")
    print(f"  Created {len(dataset)} gene prediction samples")
    if chunked_genes > 0:
        print(f"  Chunked {chunked_genes} large genes")
    
    return dataset


def format_annotation_target_chunked(gene_features, gene_index, biotype, bos_token, eos_token):
    """
    Format annotation target with condensed format.
    
    Output: type start end strand phase gene_index biotype
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