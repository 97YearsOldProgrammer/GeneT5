import struct
import json
import zlib
from pathlib import Path
from collections import defaultdict


MAGIC_HEADER   = b"GT5B"
FORMAT_VERSION = 1


class BinaryChunk:
    """
    Binary representation of a single training chunk
    """
    
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
        """
        Serialize chunk to bytes
        """
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
        
        meta_json    = json.dumps(meta).encode('utf-8')
        seq_bytes    = self.sequence.encode('utf-8')
        feat_json    = json.dumps(self.features).encode('utf-8')
        hints_json   = json.dumps(self.hints).encode('utf-8')
        
        meta_len     = len(meta_json)
        seq_len      = len(seq_bytes)
        feat_len     = len(feat_json)
        hints_len    = len(hints_json)
        
        header = struct.pack(
            '<4I',
            meta_len,
            seq_len,
            feat_len,
            hints_len
        )
        
        data = header + meta_json + seq_bytes + feat_json + hints_json
        
        return data
    
    @classmethod
    def from_bytes(cls, data):
        """
        Deserialize chunk from bytes
        """
        header_size = struct.calcsize('<4I')
        meta_len, seq_len, feat_len, hints_len = struct.unpack('<4I', data[:header_size])
        
        offset    = header_size
        meta_json = data[offset:offset + meta_len].decode('utf-8')
        offset   += meta_len
        
        seq_bytes = data[offset:offset + seq_len].decode('utf-8')
        offset   += seq_len
        
        feat_json = data[offset:offset + feat_len].decode('utf-8')
        offset   += feat_len
        
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
        """
        Estimate token count for this chunk
        """
        seq_tokens  = len(self.sequence) / bp_per_token
        feat_tokens = len(self.features) * 7
        hint_tokens = len(self.hints) * 5 if self.has_hints else 0
        overhead    = 10
        
        return int(seq_tokens + feat_tokens + hint_tokens + overhead)


def smart_compacting(chunks, max_tokens=10000, bp_per_token=4.5):
    """
    Pack chunks efficiently into compacted groups
    
    Strategy:
    - Sort by estimated token count
    - Greedy bin packing: pair large with small
    - Separate raw and augmented data
    """
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


def write_binary_dataset(chunks, output_path, compress=True):
    """
    Write chunks to binary file with optional compression
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(MAGIC_HEADER)
        f.write(struct.pack('<B', FORMAT_VERSION))
        f.write(struct.pack('<B', 1 if compress else 0))
        f.write(struct.pack('<I', len(chunks)))
        
        offsets = []
        data_start = f.tell() + len(chunks) * 8
        
        all_data   = []
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
    """
    Read chunks from binary file
    """
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
    """
    Read single chunk by index without loading entire file
    """
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
    """
    Get metadata about binary dataset without loading all chunks
    """
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
    """
    Estimate efficiency of compacting strategy
    """
    total_tokens = sum(c.estimate_tokens() for c in chunks)
    
    compacted = smart_compacting(chunks, max_tokens)
    
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