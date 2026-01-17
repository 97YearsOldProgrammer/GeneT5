import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


################################
#####  Constants           #####
################################


# Default chunking parameters
DEFAULT_WINDOW_SIZE     = 10000      # bp
DEFAULT_STRIDE          = 5000       # bp (50% overlap)
DEFAULT_OVERLAP_TOKENS  = 50         # tokens for GFF overlap
DEFAULT_MAX_GFF_LINES   = 400        # ~2000 tokens target limit


################################
#####  Tokenizer Functions #####
################################


def load_tokenizer_config(tokenizer_path: str) -> Dict[str, Any]:
    """
    Load tokenizer configuration from file.
    
    Args:
        tokenizer_path: Path to tokenizer.json or directory containing it
        
    Returns:
        Tokenizer config dictionary
    """
    path = Path(tokenizer_path)
    
    if path.is_dir():
        config_file = path / "tokenizer.json"
    else:
        config_file = path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Tokenizer config not found: {config_file}")
    
    with open(config_file, 'r') as f:
        return json.load(f)


def get_existing_tokens(tokenizer_config: Dict[str, Any]) -> set:
    """
    Extract all existing tokens from tokenizer config.
    
    Args:
        tokenizer_config: Loaded tokenizer configuration
        
    Returns:
        Set of existing token strings
    """
    tokens = set()
    
    # Check vocab in model
    if "model" in tokenizer_config and "vocab" in tokenizer_config["model"]:
        vocab = tokenizer_config["model"]["vocab"]
        if isinstance(vocab, dict):
            tokens.update(vocab.keys())
        elif isinstance(vocab, list):
            tokens.update(vocab)
    
    # Check added_tokens
    if "added_tokens" in tokenizer_config:
        for token_info in tokenizer_config["added_tokens"]:
            if isinstance(token_info, dict) and "content" in token_info:
                tokens.add(token_info["content"])
            elif isinstance(token_info, str):
                tokens.add(token_info)
    
    return tokens


def find_missing_rna_tokens(
    tokenizer_config: Dict[str, Any],
    rna_classes: Dict[str, int]
) -> List[str]:
    """
    Find RNA class tokens missing from tokenizer vocabulary.
    
    Args:
        tokenizer_config: Loaded tokenizer configuration
        rna_classes: Dictionary of RNA class names to IDs
        
    Returns:
        List of missing token strings
    """
    existing = get_existing_tokens(tokenizer_config)
    missing = []
    
    for rna_type in rna_classes.keys():
        # Check various token formats
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


def append_tokens_to_config(
    tokenizer_config: Dict[str, Any],
    new_tokens: List[str],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Append new tokens to tokenizer configuration.
    
    Args:
        tokenizer_config: Loaded tokenizer configuration
        new_tokens: List of new tokens to add
        output_path: Optional path to save updated config
        
    Returns:
        Updated tokenizer configuration
    """
    if not new_tokens:
        return tokenizer_config
    
    # Get current max ID
    max_id = 0
    if "added_tokens" in tokenizer_config:
        for token_info in tokenizer_config["added_tokens"]:
            if isinstance(token_info, dict) and "id" in token_info:
                max_id = max(max_id, token_info["id"])
    
    if "model" in tokenizer_config and "vocab" in tokenizer_config["model"]:
        vocab = tokenizer_config["model"]["vocab"]
        if isinstance(vocab, dict):
            max_id = max(max_id, max(vocab.values()) if vocab else 0)
    
    # Add new tokens
    if "added_tokens" not in tokenizer_config:
        tokenizer_config["added_tokens"] = []
    
    for i, token in enumerate(new_tokens):
        new_id = max_id + 1 + i
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
    
    # Save if output path provided
    if output_path:
        output_file = Path(output_path)
        if output_file.is_dir():
            output_file = output_file / "tokenizer.json"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        print(f"  Saved updated tokenizer to {output_file}")
    
    return tokenizer_config


def update_tokenizer_with_rna_classes(
    tokenizer_path: str,
    rna_classes: Dict[str, int],
    output_path: Optional[str] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Check tokenizer config and append missing RNA tokens.
    
    Args:
        tokenizer_path: Path to tokenizer config
        rna_classes: Dictionary of RNA class names
        output_path: Optional path for updated config
        
    Returns:
        Tuple of (updated config, list of added tokens)
    """
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

def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count from character count."""
    return int(len(text) / chars_per_token)


def estimate_gff_tokens(gff_lines: List[str], tokens_per_line: float = 5.0) -> int:
    """Estimate token count for GFF annotation lines."""
    return int(len(gff_lines) * tokens_per_line)


def find_gene_boundaries(
    features: List[Dict],
    start_bp: int,
    end_bp: int
) -> Tuple[int, int]:
    """
    Find safe chunk boundaries that don't split genes.
    
    Args:
        features: List of features in region
        start_bp: Proposed start position (bp)
        end_bp: Proposed end position (bp)
        
    Returns:
        Tuple of (adjusted_start, adjusted_end) that don't split genes
    """
    # Find all gene-level features in region
    gene_features = [
        f for f in features 
        if f["type"].lower() in {"gene", "mrna", "transcript"}
    ]
    
    adjusted_start = start_bp
    adjusted_end = end_bp
    
    for gene in gene_features:
        gene_start = gene["start"]
        gene_end = gene["end"]
        
        # If chunk start is inside a gene, move it before the gene
        if gene_start < adjusted_start < gene_end:
            adjusted_start = gene_start
        
        # If chunk end is inside a gene, extend to include whole gene
        if gene_start < adjusted_end < gene_end:
            adjusted_end = gene_end
    
    return adjusted_start, adjusted_end


def chunk_sequence_with_overlap(
    sequence: str,
    features: List[Dict],
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    respect_gene_boundaries: bool = True
) -> List[Tuple[int, int, str, List[Dict]]]:
    """
    Chunk sequence with sliding window overlap.
    
    Avoids chopping genes in half by adjusting boundaries.
    
    Args:
        sequence: DNA sequence string
        features: List of features for this sequence
        window_size: Size of each chunk in bp
        stride: Step size between chunks in bp
        respect_gene_boundaries: Whether to adjust boundaries to not split genes
        
    Returns:
        List of tuples: (start_bp, end_bp, sequence_chunk, features_in_chunk)
    """
    seq_len = len(sequence)
    chunks = []
    
    if seq_len <= window_size:
        # Sequence fits in one chunk
        chunk_features = [f for f in features if f["start"] >= 1 and f["end"] <= seq_len]
        return [(0, seq_len, sequence, chunk_features)]
    
    start = 0
    while start < seq_len:
        end = min(start + window_size, seq_len)
        
        # Adjust boundaries to not split genes
        if respect_gene_boundaries and features:
            adjusted_start, adjusted_end = find_gene_boundaries(features, start, end)
            
            # Don't let adjustment grow chunk too much
            if adjusted_end - adjusted_start <= window_size * 1.5:
                start, end = adjusted_start, adjusted_end
        
        # Extract sequence chunk (0-indexed)
        chunk_seq = sequence[start:end]
        
        # Find features within chunk (GFF is 1-indexed)
        chunk_features = [
            f for f in features
            if f["start"] >= start + 1 and f["end"] <= end
        ]
        
        chunks.append((start, end, chunk_seq, chunk_features))
        
        # Move to next chunk
        start += stride
        
        # Stop if we've covered the whole sequence
        if end >= seq_len:
            break
    
    return chunks


def chunk_gff_with_overlap(
    features: List[Dict],
    max_lines: int = DEFAULT_MAX_GFF_LINES,
    overlap_lines: int = 20
) -> List[List[Dict]]:
    """
    Chunk GFF features with overlap to avoid information loss.
    
    Args:
        features: List of feature dictionaries
        max_lines: Maximum lines per chunk
        overlap_lines: Number of lines to overlap between chunks
        
    Returns:
        List of feature chunks
    """
    if len(features) <= max_lines:
        return [features]
    
    chunks = []
    start = 0
    
    while start < len(features):
        end = min(start + max_lines, len(features))
        chunk = features[start:end]
        chunks.append(chunk)
        
        # Move forward with overlap
        start = end - overlap_lines
        
        if end >= len(features):
            break
    
    return chunks


def should_chunk_annotation(
    features: List[Dict],
    max_lines: int = DEFAULT_MAX_GFF_LINES,
    max_tokens: int = 2000
) -> bool:
    """Check if annotation needs chunking."""
    if len(features) > max_lines:
        return True
    
    estimated_tokens = estimate_gff_tokens([f["raw_line"] for f in features])
    return estimated_tokens > max_tokens


################################
#####  Validation          #####
################################

def validate_chunks(chunks: List[Tuple], original_features: List[Dict]) -> Dict[str, Any]:
    """
    Validate that chunking didn't lose any features.
    
    Args:
        chunks: List of chunk tuples from chunk_sequence_with_overlap
        original_features: Original list of features
        
    Returns:
        Validation report dict
    """
    all_chunk_features = []
    for _, _, _, chunk_features in chunks:
        all_chunk_features.extend(chunk_features)
    
    # Remove duplicates (from overlap)
    unique_features = {
        (f["seqid"], f["start"], f["end"], f["type"]) 
        for f in all_chunk_features
    }
    
    original_set = {
        (f["seqid"], f["start"], f["end"], f["type"]) 
        for f in original_features
    }
    
    missing = original_set - unique_features
    extra = unique_features - original_set
    
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

def create_gene_prediction_dataset_with_chunking(
    sequences: Dict[str, str],
    features_by_seqid: Dict[str, List[Dict]],
    window_size: int = None,
    stride: int = None,
    gene_token: str = "[ATT]",
    bos_token: str = "<BOS>",
    eos_token: str = "<EOS>",
    context_pad: int = 0,
    max_gff_lines: int = DEFAULT_MAX_GFF_LINES,
    overlap_bp: int = 50,
    overlap_lines: int = 20
) -> List[Dict]:
    """
    Create gene prediction dataset with advanced chunking support.
    
    This function handles very long sequences by breaking them into overlapping
    windows, and handles annotations with many features by chunking the GFF.
    
    Args:
        sequences: Dict of seqid -> sequence
        features_by_seqid: Dict of seqid -> features
        window_size: Sliding window size (None for whole sequence)
        stride: Sliding window stride (None = window_size)
        gene_token: Special token for annotation task
        bos_token: Beginning of sequence token
        eos_token: End of sequence token
        context_pad: Context padding around features (bp)
        max_gff_lines: Maximum GFF lines per sample before chunking
        overlap_bp: Overlap in bp for sequence chunking
        overlap_lines: Overlap in lines for GFF chunking
        
    Returns:
        List of dataset samples
    """
    from ._parser import GENE_FEATURE_TYPES, format_annotation_target
    
    dataset = []
    
    # Default stride
    if stride is None and window_size is not None:
        stride = window_size // 2  # 50% overlap by default
    
    for seqid, sequence in sequences.items():
        features = features_by_seqid.get(seqid, [])
        
        # Filter to gene-related features
        gene_features = [
            f for f in features 
            if f["type"] in GENE_FEATURE_TYPES or f["type"].lower() in GENE_FEATURE_TYPES
        ]
        
        if not gene_features:
            continue
        
        # Determine if we need to chunk
        if window_size is not None:
            # Use sliding window chunking
            chunks = chunk_sequence_with_overlap(
                sequence=sequence,
                features=gene_features,
                window_size=window_size,
                stride=stride,
                respect_gene_boundaries=True
            )
        else:
            # Check if we need to chunk based on GFF size
            if should_chunk_annotation(gene_features, max_gff_lines):
                # Chunk the GFF and find corresponding sequence regions
                gff_chunks = chunk_gff_with_overlap(gene_features, max_gff_lines, overlap_lines)
                chunks = []
                
                for gff_chunk in gff_chunks:
                    if not gff_chunk:
                        continue
                    
                    # Find sequence region for this GFF chunk
                    min_start = min(f["start"] for f in gff_chunk)
                    max_end = max(f["end"] for f in gff_chunk)
                    
                    # Add context padding
                    seq_start = max(0, min_start - 1 - context_pad)
                    seq_end = min(len(sequence), max_end + context_pad)
                    
                    chunk_seq = sequence[seq_start:seq_end]
                    
                    # Adjust feature coordinates relative to chunk
                    adjusted_features = []
                    for f in gff_chunk:
                        adj_f = f.copy()
                        adj_f["start"] = f["start"] - seq_start
                        adj_f["end"] = f["end"] - seq_start
                        adjusted_features.append(adj_f)
                    
                    chunks.append((seq_start, seq_end, chunk_seq, adjusted_features))
            else:
                # Single chunk for whole sequence
                chunks = [(0, len(sequence), sequence, gene_features)]
        
        # Create samples from chunks
        for chunk_idx, (start, end, chunk_seq, chunk_features) in enumerate(chunks):
            if not chunk_features:
                continue
            
            # Format input and target
            input_text = f"{gene_token} {chunk_seq}"
            target_text = format_annotation_target(
                chunk_features, gene_token, bos_token, eos_token
            )
            
            # Group by parent for hierarchical info
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