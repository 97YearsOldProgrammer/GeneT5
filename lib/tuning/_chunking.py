import json
from pathlib import Path


################################
#####  Tokenizer Functions #####
################################


def load_tokenizer_config(tokenizer_path):
    path = Path(tokenizer_path)
    
    if path.is_dir():
        config_file = path / "tokenizer.json"
    else:
        config_file = path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Tokenizer config not found: {config_file}")
    
    with open(config_file, 'r') as f:
        return json.load(f)


def get_existing_tokens(tokenizer_config):
    tokens = set()
    
    if "model" in tokenizer_config and "vocab" in tokenizer_config["model"]:
        vocab = tokenizer_config["model"]["vocab"]
        if isinstance(vocab, dict):
            tokens.update(vocab.keys())
        elif isinstance(vocab, list):
            tokens.update(vocab)
    
    if "added_tokens" in tokenizer_config:
        for token_info in tokenizer_config["added_tokens"]:
            if isinstance(token_info, dict) and "content" in token_info:
                tokens.add(token_info["content"])
            elif isinstance(token_info, str):
                tokens.add(token_info)
    
    return tokens


def find_missing_rna_tokens(tokenizer_config, rna_classes):
    existing = get_existing_tokens(tokenizer_config)
    missing  = []
    
    for rna_type in rna_classes.keys():
        token_variants = [
            rna_type,
            rna_type.lower(),
            rna_type.upper(),
            f"[{rna_type}]",
            f"[{rna_type.upper()}]",
        ]
        
        found = any(t in existing for t in token_variants)
        if not found:
            missing.append(rna_type.lower())
    
    return missing


def append_tokens_to_config(tokenizer_config, new_tokens, output_path=None):
    if not new_tokens:
        return tokenizer_config
    
    max_id = 0
    if "added_tokens" in tokenizer_config:
        for token_info in tokenizer_config["added_tokens"]:
            if isinstance(token_info, dict) and "id" in token_info:
                max_id = max(max_id, token_info["id"])
    
    if "model" in tokenizer_config and "vocab" in tokenizer_config["model"]:
        vocab = tokenizer_config["model"]["vocab"]
        if isinstance(vocab, dict):
            max_id = max(max_id, max(vocab.values()) if vocab else 0)
    
    if "added_tokens" not in tokenizer_config:
        tokenizer_config["added_tokens"] = []
    
    for i, token in enumerate(new_tokens):
        new_id      = max_id + 1 + i
        token_entry = {
            "id":         new_id,
            "content":    token,
            "single_word": False,
            "lstrip":     False,
            "rstrip":     False,
            "normalized": False,
            "special":    False
        }
        tokenizer_config["added_tokens"].append(token_entry)
    
    if output_path:
        output_file = Path(output_path)
        if output_file.is_dir():
            output_file = output_file / "tokenizer.json"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        print(f"  Saved updated tokenizer to {output_file}")
    
    return tokenizer_config


def update_tokenizer_with_rna_classes(tokenizer_path, rna_classes, output_path=None):
    print(f"  Loading tokenizer from {tokenizer_path}")
    config = load_tokenizer_config(tokenizer_path)
    
    existing = get_existing_tokens(config)
    print(f"  Found {len(existing)} existing tokens")
    
    missing = find_missing_rna_tokens(config, rna_classes)
    
    if missing:
        print(f"  Missing RNA tokens: {missing}")
        config = append_tokens_to_config(config, missing, output_path)
        print(f"  Added {len(missing)} new tokens")
    else:
        print(f"  All RNA tokens already present")
    
    return config, missing


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
#####  Gene Hierarchy Functions   #####
#######################################


def build_gene_hierarchy(features):
    """
    Build gene hierarchy from GFF features.
    Groups features by gene ID, handling the gene -> mRNA -> exon/CDS structure.
    
    Returns dict: {gene_id: [list of all child features including gene itself]}
    """
    genes      = {}
    id_to_gene = {}
    
    # first pass: find all gene-level features and create mapping
    for feat in features:
        feat_type = feat["type"].lower()
        attrs     = feat["attributes"]
        feat_id   = attrs.get("ID", "")
        
        if feat_type == "gene":
            gene_id = feat_id
            if gene_id not in genes:
                genes[gene_id] = []
            genes[gene_id].append(feat)
            id_to_gene[feat_id] = gene_id
    
    # second pass: find mRNA/transcript and map to parent gene
    for feat in features:
        feat_type = feat["type"].lower()
        attrs     = feat["attributes"]
        feat_id   = attrs.get("ID", "")
        parent    = attrs.get("Parent", "")
        
        if feat_type in {"mrna", "transcript"}:
            if parent in id_to_gene:
                gene_id = id_to_gene[parent]
            elif parent in genes:
                gene_id = parent
            else:
                gene_id = parent if parent else feat_id
                if gene_id not in genes:
                    genes[gene_id] = []
            
            genes[gene_id].append(feat)
            id_to_gene[feat_id] = gene_id
    
    # third pass: add exon/CDS/UTR features to their parent's gene
    for feat in features:
        feat_type = feat["type"].lower()
        attrs     = feat["attributes"]
        parent    = attrs.get("Parent", "")
        
        if feat_type in {"exon", "cds", "five_prime_utr", "three_prime_utr", "utr", "intron"}:
            if parent in id_to_gene:
                gene_id = id_to_gene[parent]
                genes[gene_id].append(feat)
            elif parent:
                # parent might be the gene itself
                for gid in genes:
                    if parent == gid or parent.endswith(gid.split(":")[-1] if ":" in gid else gid):
                        genes[gid].append(feat)
                        break
    
    return genes


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


def group_features_by_gene_simple(features):
    """
    Group features by gene ID, resolving transcript->gene relationships.
    Ensures exons and CDS from the same gene stay together.
    """
    try:
        from ._parser import GENE_FEATURE_TYPES
    except ImportError:
        from _parser import GENE_FEATURE_TYPES
    
    # 1. build transcript -> gene map from ALL features
    parent_map = build_transcript_to_gene_map(features)
    
    # 2. filter to gene-related features only
    gene_features = [
        f for f in features
        if f["type"] in GENE_FEATURE_TYPES or f["type"].lower() in GENE_FEATURE_TYPES
    ]
    
    if not gene_features:
        return {}
    
    # 3. group by gene ID (resolving through transcript if needed)
    groups = {}
    for feat in gene_features:
        direct_parent = feat["attributes"].get("Parent", feat["attributes"].get("ID"))
        
        # resolve to gene ID if parent is a transcript
        group_id = parent_map.get(direct_parent, direct_parent)
        
        if group_id:
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(feat)
    
    return groups


#######################
#####  Main Entry #####
#######################


def create_gene_prediction_dataset_with_chunking(sequences, features_by_seqid, window_size=None,
                                                  stride=None, gene_token="[ATT]", bos_token="<BOS>",
                                                  eos_token="<EOS>", context_pad=0, 
                                                  max_gff_lines=400, overlap_bp=50, overlap_lines=20):
    try:
        from ._parser import GENE_FEATURE_TYPES, format_annotation_target, anti
    except ImportError:
        from _parser import GENE_FEATURE_TYPES, format_annotation_target, anti
    
    dataset       = []
    total_genes   = 0
    chunked_genes = 0
    
    if stride is None and window_size is not None:
        stride = window_size // 2
    
    for seqid, sequence in sequences.items():
        features = features_by_seqid.get(seqid, [])
        
        if not features:
            continue
        
        # group features by gene (using Parent hierarchy)
        gene_groups = group_features_by_gene_simple(features)
        
        if not gene_groups:
            continue
        
        total_genes += len(gene_groups)
        
        for gene_id, group_feats in gene_groups.items():
            if not group_feats:
                continue
            
            # get gene span
            min_start = min(f["start"] for f in group_feats)
            max_end   = max(f["end"] for f in group_feats)
            strand    = group_feats[0]["strand"]
            
            # extract sequence with context padding
            seq_start = max(0, min_start - 1 - context_pad)
            seq_end   = min(len(sequence), max_end + context_pad)
            chunk_seq = sequence[seq_start:seq_end]
            
            # reverse complement if minus strand
            if strand == "-":
                chunk_seq = anti(chunk_seq)
            
            # adjust feature coordinates relative to extracted sequence
            adjusted_features = []
            for f in group_feats:
                adj_f          = f.copy()
                adj_f["start"] = f["start"] - seq_start
                adj_f["end"]   = f["end"] - seq_start
                adjusted_features.append(adj_f)
            
            # check if this gene needs chunking (very large genes)
            if window_size is not None and len(chunk_seq) > window_size:
                # use sliding window for very large genes
                chunks = chunk_sequence_with_overlap(
                    sequence=chunk_seq,
                    features=adjusted_features,
                    window_size=window_size,
                    stride=stride,
                    respect_gene_boundaries=False
                )
                chunked_genes += 1
            elif should_chunk_annotation(adjusted_features, max_gff_lines):
                # chunk by GFF lines if too many features
                gff_chunks = chunk_gff_with_overlap(adjusted_features, max_gff_lines, overlap_lines)
                chunks     = [(0, len(chunk_seq), chunk_seq, gc) for gc in gff_chunks]
                chunked_genes += 1
            else:
                # single chunk for this gene
                chunks = [(0, len(chunk_seq), chunk_seq, adjusted_features)]
            
            # create samples for each chunk
            for chunk_idx, (c_start, c_end, c_seq, c_feats) in enumerate(chunks):
                if not c_feats:
                    continue
                
                input_text  = f"{gene_token} {c_seq}"
                target_text = format_annotation_target(c_feats, gene_token, bos_token, eos_token)
                
                sample = {
                    "seqid":        seqid,
                    "parent_id":    gene_id,
                    "start":        min_start + c_start,
                    "end":          min_start + c_end,
                    "strand":       strand,
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