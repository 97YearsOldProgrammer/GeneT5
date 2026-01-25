import struct
import json
import zlib
import random
from pathlib    import Path
from collections import defaultdict


MAGIC_HEADER   = b"GT5B"
FORMAT_VERSION = 1


class BinaryChunk:
    """Binary representation of a single training chunk"""
    
    def __init__(
        self,
        seqid,
        start,
        end,
        strand,
        sequence,
        features,
        biotype      = ".",
        gene_ids     = None,
        has_hints    = False,
        hints        = None,
        chunk_index  = 0,
        is_augmented = False,
    ):
        self.seqid        = seqid
        self.start        = start
        self.end          = end
        self.strand       = strand
        self.sequence     = sequence
        self.features     = features
        self.biotype      = biotype
        self.gene_ids     = gene_ids or []
        self.has_hints    = has_hints
        self.hints        = hints or []
        self.chunk_index  = chunk_index
        self.is_augmented = is_augmented
    
    def to_bytes(self):
        """Serialize chunk to bytes"""
        
        meta = {
            "seqid":        self.seqid,
            "start":        self.start,
            "end":          self.end,
            "strand":       self.strand,
            "biotype":      self.biotype,
            "gene_ids":     self.gene_ids,
            "has_hints":    self.has_hints,
            "chunk_index":  self.chunk_index,
            "is_augmented": self.is_augmented,
        }
        
        meta_json  = json.dumps(meta).encode('utf-8')
        seq_bytes  = self.sequence.encode('utf-8')
        feat_json  = json.dumps(self.features).encode('utf-8')
        hints_json = json.dumps(self.hints).encode('utf-8')
        
        meta_len   = len(meta_json)
        seq_len    = len(seq_bytes)
        feat_len   = len(feat_json)
        hints_len  = len(hints_json)
        
        header = struct.pack('<4I', meta_len, seq_len, feat_len, hints_len)
        data   = header + meta_json + seq_bytes + feat_json + hints_json
        
        return data
    
    @classmethod
    def from_bytes(cls, data):
        """Deserialize chunk from bytes"""
        
        header_size = struct.calcsize('<4I')
        meta_len, seq_len, feat_len, hints_len = struct.unpack('<4I', data[:header_size])
        
        offset     = header_size
        meta_json  = data[offset:offset + meta_len].decode('utf-8')
        offset    += meta_len
        
        seq_bytes  = data[offset:offset + seq_len].decode('utf-8')
        offset    += seq_len
        
        feat_json  = data[offset:offset + feat_len].decode('utf-8')
        offset    += feat_len
        
        hints_json = data[offset:offset + hints_len].decode('utf-8')
        
        meta     = json.loads(meta_json)
        features = json.loads(feat_json)
        hints    = json.loads(hints_json)
        
        return cls(
            seqid        = meta["seqid"],
            start        = meta["start"],
            end          = meta["end"],
            strand       = meta["strand"],
            sequence     = seq_bytes,
            features     = features,
            biotype      = meta.get("biotype", "."),
            gene_ids     = meta.get("gene_ids", []),
            has_hints    = meta.get("has_hints", False),
            hints        = hints,
            chunk_index  = meta.get("chunk_index", 0),
            is_augmented = meta.get("is_augmented", False),
        )
    
    def estimate_tokens(self, bp_per_token=4.5):
        """Estimate token count for this chunk"""
        
        seq_tokens  = len(self.sequence) / bp_per_token
        feat_tokens = len(self.features) * 7
        hint_tokens = len(self.hints) * 5 if self.has_hints else 0
        overhead    = 10
        
        return int(seq_tokens + feat_tokens + hint_tokens + overhead)
    
    def estimate_input_tokens(self, bp_per_token=4.5):
        """Estimate input token count (sequence + hints only)"""
        
        seq_tokens  = len(self.sequence) / bp_per_token
        hint_tokens = len(self.hints) * 5 if self.has_hints else 0
        overhead    = 5
        
        return int(seq_tokens + hint_tokens + overhead)


def check_cut_inside_gene(gene_index, seqid, cut_pos):
    """Check if cut position falls inside a gene body"""
    
    for gene_id, gene_data in gene_index.items():
        if gene_data["seqid"] != seqid:
            continue
        
        g_start = gene_data["start"]
        g_end   = gene_data["end"]
        
        if g_start < cut_pos < g_end:
            return True, gene_id
    
    return False, None


def find_genes_in_range(gene_index, seqid, start, end):
    """Find all genes overlapping a genomic range"""
    
    genes = []
    
    for gene_id, gene_data in gene_index.items():
        if gene_data["seqid"] != seqid:
            continue
        
        g_start = gene_data["start"]
        g_end   = gene_data["end"]
        
        if g_start <= end and g_end >= start:
            genes.append((gene_id, gene_data))
    
    return genes


def dynamic_chunking(
    sequences,
    gene_index,
    limit_bp   = 25000,
    overlap_bp = 5000,
    anchor_pad = 5000,
):
    """Gene-centric sliding window chunking"""
    
    chunks    = []
    step_size = limit_bp - overlap_bp
    
    stats = {
        "total_chunks":    0,
        "backtrack_count": 0,
        "genes_per_chunk": [],
        "chunk_sizes":     [],
    }
    
    for seqid, sequence in sequences.items():
        seq_len = len(sequence)
        
        seqid_genes = [
            (gid, gdata) for gid, gdata in gene_index.items()
            if gdata["seqid"] == seqid
        ]
        
        if not seqid_genes:
            continue
        
        seqid_genes.sort(key=lambda x: x[1]["start"])
        
        first_gene_start = seqid_genes[0][1]["start"]
        window_start     = max(0, first_gene_start - anchor_pad)
        chunk_index      = 0
        
        while window_start < seq_len:
            window_end  = min(window_start + limit_bp, seq_len)
            cut_pos     = window_start + step_size
            backtracked = False
            
            if cut_pos < seq_len:
                is_inside, blocking_gene = check_cut_inside_gene(gene_index, seqid, cut_pos)
                
                if is_inside:
                    new_cut      = cut_pos - overlap_bp
                    still_inside = check_cut_inside_gene(gene_index, seqid, new_cut)[0]
                    
                    if not still_inside and new_cut > window_start:
                        cut_pos     = new_cut
                        window_end  = cut_pos
                        backtracked = True
                        stats["backtrack_count"] += 1
            
            genes_in_chunk = find_genes_in_range(gene_index, seqid, window_start, window_end)
            chunk_seq      = sequence[window_start:window_end]
            chunk_features = []
            gene_ids       = []
            
            for gene_id, gene_data in genes_in_chunk:
                gene_ids.append(gene_id)
                
                for feat in gene_data.get("features", []):
                    adj_start = feat["start"] - window_start
                    adj_end   = feat["end"] - window_start
                    
                    if adj_start < 0 or adj_end > (window_end - window_start):
                        continue
                    
                    chunk_features.append({
                        "type":    feat["type"].lower(),
                        "start":   adj_start,
                        "end":     adj_end,
                        "strand":  feat["strand"],
                        "phase":   feat.get("phase", "."),
                        "gene_id": gene_id,
                    })
            
            biotypes = []
            for gene_id, gene_data in genes_in_chunk:
                for t_id, t_data in gene_data.get("transcripts", {}).items():
                    bt = t_data.get("biotype", "")
                    if bt:
                        biotypes.append(bt)
            
            primary_biotype = biotypes[0] if biotypes else "."
            
            chunk = BinaryChunk(
                seqid        = seqid,
                start        = window_start,
                end          = window_end,
                strand       = "+",
                sequence     = chunk_seq,
                features     = chunk_features,
                biotype      = primary_biotype,
                gene_ids     = gene_ids,
                has_hints    = False,
                hints        = [],
                chunk_index  = chunk_index,
                is_augmented = False,
            )
            
            chunks.append(chunk)
            
            stats["total_chunks"] += 1
            stats["genes_per_chunk"].append(len(gene_ids))
            stats["chunk_sizes"].append(window_end - window_start)
            
            chunk_index += 1
            
            if backtracked:
                window_start = cut_pos
            else:
                window_start += step_size
            
            if window_start >= seq_len:
                break
    
    return chunks, stats


def generate_hints_from_features(features, noise_rate=0.1):
    """Generate noised hints from features (simulating extrinsic evidence)"""
    
    hints = []
    
    for feat in features:
        if random.random() < noise_rate:
            continue
        
        jitter_start = int(random.gauss(0, 15))
        jitter_end   = int(random.gauss(0, 15))
        
        hint = {
            "type":   feat["type"],
            "start":  max(0, feat["start"] + jitter_start),
            "end":    feat["end"] + jitter_end,
            "strand": feat["strand"],
        }
        hints.append(hint)
    
    if random.random() < 0.05 and features:
        max_pos    = max(f["end"] for f in features)
        fake_start = random.randint(0, max(0, max_pos - 200))
        fake_end   = fake_start + random.randint(50, 200)
        
        hints.append({
            "type":   "exon",
            "start":  fake_start,
            "end":    fake_end,
            "strand": random.choice(["+", "-"]),
        })
    
    return hints


def augment_with_hints(chunks, hint_ratio=0.5, seed=42):
    """Create augmented copies with hints"""
    
    random.seed(seed)
    
    num_to_augment = int(len(chunks) * hint_ratio)
    indices        = random.sample(range(len(chunks)), min(num_to_augment, len(chunks)))
    
    augmented = []
    
    for idx in indices:
        original = chunks[idx]
        hints    = generate_hints_from_features(original.features)
        
        aug_chunk = BinaryChunk(
            seqid        = original.seqid,
            start        = original.start,
            end          = original.end,
            strand       = original.strand,
            sequence     = original.sequence,
            features     = original.features,
            biotype      = original.biotype,
            gene_ids     = original.gene_ids,
            has_hints    = True,
            hints        = hints,
            chunk_index  = original.chunk_index,
            is_augmented = True,
        )
        
        augmented.append(aug_chunk)
    
    return chunks + augmented


def smart_compacting(chunks, max_tokens=10000, bp_per_token=4.5):
    """Pack chunks efficiently into compacted groups"""
    
    raw_chunks = [c for c in chunks if not c.is_augmented]
    aug_chunks = [c for c in chunks if c.is_augmented]
    
    compacted = []
    
    for chunk_list in [raw_chunks, aug_chunks]:
        if not chunk_list:
            continue
        
        with_tokens = [(c, c.estimate_tokens(bp_per_token)) for c in chunk_list]
        with_tokens.sort(key=lambda x: -x[1])
        
        used   = set()
        groups = []
        
        for i, (chunk1, tokens1) in enumerate(with_tokens):
            if i in used:
                continue
            
            used.add(i)
            group        = [chunk1]
            group_tokens = tokens1
            
            for j in range(len(with_tokens) - 1, i, -1):
                if j in used:
                    continue
                
                chunk2, tokens2 = with_tokens[j]
                
                if group_tokens + tokens2 <= max_tokens:
                    group.append(chunk2)
                    group_tokens += tokens2
                    used.add(j)
                    break
            
            groups.append(group)
        
        compacted.extend(groups)
    
    return compacted


#####################################################
#####  Input-Length Based Compacting Workflow   #####
#####################################################


def estimate_chunk_input_length(chunk, bp_per_token=4.5):
    """Estimate input length in tokens for a single chunk"""
    
    seq_tokens  = len(chunk.sequence) / bp_per_token
    hint_tokens = len(chunk.hints) * 5 if chunk.has_hints else 0
    overhead    = 5
    
    return int(seq_tokens + hint_tokens + overhead)


def compact_to_target_length(
    chunks,
    target_length,
    hard_limit    = None,
    bp_per_token  = 4.5,
    seed          = 42,
):
    """
    Compact chunks to converge toward target input length with minimal padding
    
    Uses first-fit-decreasing bin packing to maximize utilization
    """
    
    random.seed(seed)
    
    hard_limit = hard_limit or int(target_length * 1.1)
    
    raw_chunks = [c for c in chunks if not c.is_augmented]
    aug_chunks = [c for c in chunks if c.is_augmented]
    
    all_compacted = []
    compact_stats = {
        "total_groups":     0,
        "total_input_toks": 0,
        "utilizations":     [],
        "overflow_count":   0,
        "singleton_count":  0,
    }
    
    for chunk_list in [raw_chunks, aug_chunks]:
        if not chunk_list:
            continue
        
        with_lengths = [
            (c, estimate_chunk_input_length(c, bp_per_token))
            for c in chunk_list
        ]
        with_lengths.sort(key=lambda x: -x[1])
        
        bins       = []
        bin_totals = []
        
        for chunk, length in with_lengths:
            if length > hard_limit:
                bins.append([chunk])
                bin_totals.append(length)
                compact_stats["overflow_count"] += 1
                continue
            
            best_bin   = -1
            best_space = hard_limit + 1
            
            for i, total in enumerate(bin_totals):
                remaining = target_length - total
                if remaining >= length and remaining < best_space:
                    best_bin   = i
                    best_space = remaining
            
            if best_bin == -1:
                for i, total in enumerate(bin_totals):
                    remaining = hard_limit - total
                    if remaining >= length and remaining < best_space:
                        best_bin   = i
                        best_space = remaining
            
            if best_bin >= 0:
                bins[best_bin].append(chunk)
                bin_totals[best_bin] += length
            else:
                bins.append([chunk])
                bin_totals.append(length)
        
        for i, group in enumerate(bins):
            all_compacted.append(group)
            
            total_len   = bin_totals[i]
            utilization = total_len / target_length if target_length > 0 else 0
            
            compact_stats["total_groups"]     += 1
            compact_stats["total_input_toks"] += total_len
            compact_stats["utilizations"].append(utilization)
            
            if len(group) == 1:
                compact_stats["singleton_count"] += 1
    
    if compact_stats["utilizations"]:
        compact_stats["avg_utilization"] = sum(compact_stats["utilizations"]) / len(compact_stats["utilizations"])
        compact_stats["min_utilization"] = min(compact_stats["utilizations"])
        compact_stats["max_utilization"] = max(compact_stats["utilizations"])
    else:
        compact_stats["avg_utilization"] = 0
        compact_stats["min_utilization"] = 0
        compact_stats["max_utilization"] = 0
    
    return all_compacted, compact_stats


def flatten_compacted_groups(compacted_groups):
    """Flatten compacted groups back to chunk list with group markers"""
    
    flattened = []
    
    for group_idx, group in enumerate(compacted_groups):
        for chunk in group:
            chunk.compact_group = group_idx
            flattened.append(chunk)
    
    return flattened


def write_binary_dataset(chunks, output_path, compress=True):
    """Write chunks to binary file with optional compression"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(MAGIC_HEADER)
        f.write(struct.pack('<B', FORMAT_VERSION))
        f.write(struct.pack('<B', 1 if compress else 0))
        f.write(struct.pack('<I', len(chunks)))
        
        offsets    = []
        data_start = f.tell() + len(chunks) * 8
        
        all_data       = []
        current_offset = data_start
        
        for chunk in chunks:
            chunk_bytes = chunk.to_bytes()
            
            if compress:
                chunk_bytes = zlib.compress(chunk_bytes, level=6)
            
            offsets.append((current_offset, len(chunk_bytes)))
            all_data.append(chunk_bytes)
            current_offset += len(chunk_bytes)
        
        for offset, length in offsets:
            f.write(struct.pack('<II', offset, length))
        
        for data in all_data:
            f.write(data)
    
    return output_path


def read_binary_dataset(input_path):
    """Read chunks from binary file"""
    
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}")
        
        version    = struct.unpack('<B', f.read(1))[0]
        compressed = struct.unpack('<B', f.read(1))[0]
        num_chunks = struct.unpack('<I', f.read(4))[0]
        
        offsets = []
        for _ in range(num_chunks):
            offset, length = struct.unpack('<II', f.read(8))
            offsets.append((offset, length))
        
        chunks = []
        for offset, length in offsets:
            f.seek(offset)
            data = f.read(length)
            
            if compressed:
                data = zlib.decompress(data)
            
            chunk = BinaryChunk.from_bytes(data)
            chunks.append(chunk)
    
    return chunks


def get_chunk_at_index(input_path, index):
    """Read single chunk by index without loading entire file"""
    
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}")
        
        version    = struct.unpack('<B', f.read(1))[0]
        compressed = struct.unpack('<B', f.read(1))[0]
        num_chunks = struct.unpack('<I', f.read(4))[0]
        
        if index >= num_chunks:
            raise IndexError(f"Index {index} out of range (total: {num_chunks})")
        
        f.seek(10 + index * 8)
        offset, length = struct.unpack('<II', f.read(8))
        
        f.seek(offset)
        data = f.read(length)
        
        if compressed:
            data = zlib.decompress(data)
        
        return BinaryChunk.from_bytes(data)


def get_dataset_info(input_path):
    """Get metadata about binary dataset without loading all chunks"""
    
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}")
        
        version    = struct.unpack('<B', f.read(1))[0]
        compressed = struct.unpack('<B', f.read(1))[0]
        num_chunks = struct.unpack('<I', f.read(4))[0]
        
        offsets = []
        for _ in range(num_chunks):
            offset, length = struct.unpack('<II', f.read(8))
            offsets.append((offset, length))
        
        total_size = sum(length for _, length in offsets)
    
    return {
        "version":    version,
        "compressed": bool(compressed),
        "num_chunks": num_chunks,
        "total_size": total_size,
        "file_path":  str(input_path),
    }


def estimate_compacting_efficiency(chunks, max_tokens=10000):
    """Estimate efficiency of compacting strategy"""
    
    total_tokens = sum(c.estimate_tokens() for c in chunks)
    compacted    = smart_compacting(chunks, max_tokens)
    num_groups   = len(compacted)
    group_tokens = [sum(c.estimate_tokens() for c in g) for g in compacted]
    
    avg_utilization = sum(group_tokens) / (num_groups * max_tokens) if num_groups > 0 else 0
    
    return {
        "total_chunks":    len(chunks),
        "total_tokens":    total_tokens,
        "num_groups":      num_groups,
        "avg_group_size":  len(chunks) / num_groups if num_groups > 0 else 0,
        "avg_utilization": avg_utilization,
        "wasted_tokens":   (num_groups * max_tokens) - total_tokens,
    }