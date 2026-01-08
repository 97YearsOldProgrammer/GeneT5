

def anti(seq):
    """Reverse complement of DNA sequence."""
    comp = str.maketrans('ACGTRYMKBDHVNacgtrymkbdhvn', 'TGCAYRKMVHDBNtgcayrkmvhdbn')
    anti = seq.translate(comp)[::-1]
    return anti


def estimate_tokens(tokenizer, text):
    """Get token count without padding."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def parse_stripped_gff(gff_text):
    """Parse stripped GFF format back into feature list."""
    features = []
    
    for line in gff_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("<"):
            continue
        
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        
        feat = {
            "type":   parts[0],
            "start":  int(parts[1]),
            "end":    int(parts[2]),
            "strand": parts[3],
            "phase":  parts[4] if len(parts) > 4 else ".",
        }
        features.append(feat)
    
    return features


def format_features_to_gff(features):
    """Convert feature list back to stripped GFF format."""
    lines = []
    
    for feat in sorted(features, key=lambda x: x["start"]):
        phase = feat["phase"] if feat["phase"] != "." else "."
        line  = "\t".join([
            feat["type"],
            str(feat["start"]),
            str(feat["end"]),
            feat["strand"],
            str(phase),
        ])
        lines.append(line)
    
    return "\n".join(lines)


def find_chunk_boundary(features, max_pos, min_pos=0):
    """
    Find a good position to split based on feature boundaries.
    Returns position after the last complete feature before max_pos.
    """
    # sort by end position
    sorted_feats = sorted(features, key=lambda x: x["end"])
    
    best_end = min_pos
    for feat in sorted_feats:
        if feat["end"] <= max_pos:
            best_end = feat["end"]
        else:
            break
    
    return best_end


def extract_dna_from_input(input_text, gene_token="[GENE]", bos_token="<BOS>", eos_token="<EOS>"):
    """Extract raw DNA sequence from formatted input."""
    text = input_text
    text = text.replace(bos_token, "").replace(eos_token, "")
    text = text.replace(gene_token, "").strip()
    return text


def extract_gff_from_target(target_text, bos_token="<BOS>", eos_token="<EOS>"):
    """Extract GFF content from formatted target."""
    text = target_text
    text = text.replace(bos_token, "").replace(eos_token, "")
    return text.strip()


def chunk_gene_prediction_sample(sample, tokenizer, max_input_len, max_target_len,
                                  gene_token="[GENE]", bos_token="<BOS>", eos_token="<EOS>"):
    """
    Chunk a single gene prediction sample if it exceeds token limits.
    Returns list of chunked samples.
    """
    input_text  = sample["input"]
    target_text = sample["target"]
    
    # check if chunking needed
    input_tokens  = estimate_tokens(tokenizer, input_text)
    target_tokens = estimate_tokens(tokenizer, target_text)
    
    if input_tokens <= max_input_len and target_tokens <= max_target_len:
        return [sample]
    
    # extract components
    dna_seq  = extract_dna_from_input(input_text, gene_token, bos_token, eos_token)
    gff_text = extract_gff_from_target(target_text, bos_token, eos_token)
    features = parse_stripped_gff(gff_text)
    
    if not features:
        return [sample]
    
    # estimate overhead tokens
    overhead      = f"{bos_token} {gene_token}  {eos_token}"
    overhead_toks = estimate_tokens(tokenizer, overhead)
    
    # available tokens for DNA
    avail_dna_tokens = max_input_len - overhead_toks - 10  # small buffer
    
    # estimate chars per token (rough)
    dna_tokens    = estimate_tokens(tokenizer, dna_seq)
    chars_per_tok = len(dna_seq) / max(dna_tokens, 1)
    
    # target chunk size in base pairs
    chunk_bp = int(avail_dna_tokens * chars_per_tok)
    
    chunks     = []
    seq_len    = len(dna_seq)
    chunk_idx  = 0
    current_bp = 0
    
    while current_bp < seq_len:
        # find boundary
        target_end = min(current_bp + chunk_bp, seq_len)
        
        # get features in this region
        region_feats = [f for f in features if f["start"] > current_bp]
        
        if region_feats and target_end < seq_len:
            # find natural boundary
            boundary = find_chunk_boundary(region_feats, target_end, current_bp)
            if boundary > current_bp:
                target_end = boundary
        
        # extract chunk
        chunk_dna = dna_seq[current_bp:target_end]
        
        # get features for this chunk (adjust coordinates)
        chunk_feats = []
        for f in features:
            if f["start"] > current_bp and f["end"] <= target_end:
                adj_feat = f.copy()
                adj_feat["start"] = f["start"] - current_bp
                adj_feat["end"]   = f["end"] - current_bp
                chunk_feats.append(adj_feat)
        
        # only create chunk if it has features
        if chunk_feats:
            chunk_gff   = format_features_to_gff(chunk_feats)
            chunk_input = f"{bos_token} {gene_token} {chunk_dna} {eos_token}"
            chunk_target = f"{bos_token}\n{chunk_gff}\n{eos_token}"
            
            chunk_sample = {
                "parent_id": f"{sample.get('parent_id', 'unk')}_chunk{chunk_idx}",
                "seqid":     sample.get("seqid", ""),
                "start":     sample.get("start", 0) + current_bp,
                "end":       sample.get("start", 0) + target_end,
                "strand":    sample.get("strand", "+"),
                "input":     chunk_input,
                "target":    chunk_target,
            }
            chunks.append(chunk_sample)
            chunk_idx += 1
        
        current_bp = target_end
        
        # safety check
        if target_end == current_bp and current_bp < seq_len:
            current_bp += chunk_bp // 2
    
    return chunks if chunks else [sample]


def chunk_classification_sample(sample, tokenizer, max_len, cls_token="[CLS]"):
    """
    Chunk a classification sample if needed.
    For classification, we keep the same label for all chunks.
    """
    input_text = sample["input"]
    
    input_tokens = estimate_tokens(tokenizer, input_text)
    
    if input_tokens <= max_len:
        return [sample]
    
    # extract DNA
    dna_seq = input_text.replace(cls_token, "").strip()
    
    # estimate chunk size
    overhead_toks = estimate_tokens(tokenizer, f"{cls_token} ")
    avail_tokens  = max_len - overhead_toks - 5
    
    dna_tokens    = estimate_tokens(tokenizer, dna_seq)
    chars_per_tok = len(dna_seq) / max(dna_tokens, 1)
    chunk_bp      = int(avail_tokens * chars_per_tok)
    
    chunks    = []
    seq_len   = len(dna_seq)
    chunk_idx = 0
    
    for start in range(0, seq_len, chunk_bp):
        end       = min(start + chunk_bp, seq_len)
        chunk_dna = dna_seq[start:end]
        
        chunk_sample = {
            "seqid":     sample.get("seqid", ""),
            "start":     sample.get("start", 0) + start,
            "end":       sample.get("start", 0) + end,
            "strand":    sample.get("strand", "+"),
            "input":     f"{cls_token} {chunk_dna}",
            "label":     sample["label"],
            "label_str": sample.get("label_str", ""),
            "chunk_idx": chunk_idx,
        }
        chunks.append(chunk_sample)
        chunk_idx += 1
    
    return chunks


def chunk_dataset(samples, tokenizer, max_input_len, max_target_len=None,
                  task="gene_prediction", gene_token="[GENE]", cls_token="[CLS]",
                  bos_token="<BOS>", eos_token="<EOS>"):
    """
    Chunk all samples in dataset that exceed token limits.
    
    task: "gene_prediction" or "classification"
    """
    chunked = []
    
    for sample in samples:
        if task == "gene_prediction":
            chunks = chunk_gene_prediction_sample(
                sample, tokenizer, max_input_len, max_target_len or 1024,
                gene_token, bos_token, eos_token
            )
        else:
            chunks = chunk_classification_sample(
                sample, tokenizer, max_input_len, cls_token
            )
        chunked.extend(chunks)
    
    return chunked


def preprocess_and_chunk(data_path, tokenizer, output_path,
                         max_input_len, max_target_len=None, task="gene_prediction"):
    """
    Load dataset, chunk, and save preprocessed version.
    """
    from ._parser import load_dataset, save_dataset
    
    samples = load_dataset(data_path)
    chunked = chunk_dataset(
        samples, tokenizer, max_input_len, max_target_len, task
    )
    save_dataset(chunked, output_path)
    
    print(f"Preprocessed {len(samples)} -> {len(chunked)} samples")
    return chunked