import json
from pathlib import Path

from ._parser import GENE_FEATURE_TYPES, format_annotation_target


################################
#####  Constants           #####
################################


DEFAULT_WINDOW_SIZE    = 10000
DEFAULT_STRIDE         = 5000
DEFAULT_OVERLAP_TOKENS = 50
DEFAULT_MAX_GFF_LINES  = 400


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


def chunk_sequence_with_overlap(sequence, features, window_size=DEFAULT_WINDOW_SIZE, 
                                 stride=DEFAULT_STRIDE, respect_gene_boundaries=True):
    
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


def chunk_gff_with_overlap(features, max_lines=DEFAULT_MAX_GFF_LINES, overlap_lines=20):
    
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


def should_chunk_annotation(features, max_lines=DEFAULT_MAX_GFF_LINES, max_tokens=2000):
    if len(features) > max_lines:
        return True
    
    estimated_tokens = estimate_gff_tokens([f["raw_line"] for f in features])
    return estimated_tokens > max_tokens


################################
#####  Validation          #####
################################


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


################################
#####  Advanced Chunking   #####
################################


def create_gene_prediction_dataset_with_chunking(sequences, features_by_seqid, window_size=None,
                                                  stride=None, gene_token="[ATT]", bos_token="<BOS>",
                                                  eos_token="<EOS>", context_pad=0, 
                                                  max_gff_lines=DEFAULT_MAX_GFF_LINES, 
                                                  overlap_bp=50, overlap_lines=20):
    
    dataset = []
    
    if stride is None and window_size is not None:
        stride = window_size // 2
    
    for seqid, sequence in sequences.items():
        features = features_by_seqid.get(seqid, [])
        
        gene_features = [
            f for f in features 
            if f["type"] in GENE_FEATURE_TYPES or f["type"].lower() in GENE_FEATURE_TYPES
        ]
        
        if not gene_features:
            continue
        
        if window_size is not None:
            chunks = chunk_sequence_with_overlap(
                sequence=sequence,
                features=gene_features,
                window_size=window_size,
                stride=stride,
                respect_gene_boundaries=True
            )
        else:
            if should_chunk_annotation(gene_features, max_gff_lines):
                gff_chunks = chunk_gff_with_overlap(gene_features, max_gff_lines, overlap_lines)
                chunks     = []
                
                for gff_chunk in gff_chunks:
                    if not gff_chunk:
                        continue
                    
                    min_start = min(f["start"] for f in gff_chunk)
                    max_end   = max(f["end"] for f in gff_chunk)
                    
                    seq_start = max(0, min_start - 1 - context_pad)
                    seq_end   = min(len(sequence), max_end + context_pad)
                    
                    chunk_seq = sequence[seq_start:seq_end]
                    
                    adjusted_features = []
                    for f in gff_chunk:
                        adj_f          = f.copy()
                        adj_f["start"] = f["start"] - seq_start
                        adj_f["end"]   = f["end"] - seq_start
                        adjusted_features.append(adj_f)
                    
                    chunks.append((seq_start, seq_end, chunk_seq, adjusted_features))
            else:
                chunks = [(0, len(sequence), sequence, gene_features)]
        
        for chunk_idx, (start, end, chunk_seq, chunk_features) in enumerate(chunks):
            if not chunk_features:
                continue
            
            input_text  = f"{gene_token} {chunk_seq}"
            target_text = format_annotation_target(
                chunk_features, gene_token, bos_token, eos_token
            )
            
            parent_id = None
            if chunk_features:
                parent_id = chunk_features[0]["attributes"].get(
                    "Parent", 
                    chunk_features[0]["attributes"].get("ID", f"{seqid}_chunk{chunk_idx}")
                )
            
            sample = {
                "seqid":        seqid,
                "parent_id":    parent_id,
                "start":        start,
                "end":          end,
                "chunk_idx":    chunk_idx,
                "input":        input_text,
                "target":       target_text,
                "num_features": len(chunk_features),
                "is_chunked":   len(chunks) > 1
            }
            
            dataset.append(sample)
    
    print(f"  Created {len(dataset)} gene prediction samples (with chunking)")
    return dataset