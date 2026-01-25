import struct
import json
import zlib
import pathlib as pl


#####################  Constants  #####################


MAGIC_HEADER   = b"GT5B"
FORMAT_VERSION = 2


##########################
#####  Chunk Class  ######
##########################


class BinaryChunk:
    """Single training chunk with serialization support"""
    
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
        compact_group = None,
    ):
        self.seqid         = seqid
        self.start         = start
        self.end           = end
        self.strand        = strand
        self.sequence      = sequence
        self.features      = features
        self.biotype       = biotype
        self.gene_ids      = gene_ids or []
        self.has_hints     = has_hints
        self.hints         = hints or []
        self.chunk_index   = chunk_index
        self.is_augmented  = is_augmented
        self.compact_group = compact_group
    
    def to_bytes(self):
        """Serialize chunk to bytes"""
        
        meta = {
            "seqid":         self.seqid,
            "start":         self.start,
            "end":           self.end,
            "strand":        self.strand,
            "biotype":       self.biotype,
            "gene_ids":      self.gene_ids,
            "has_hints":     self.has_hints,
            "chunk_index":   self.chunk_index,
            "is_augmented":  self.is_augmented,
            "compact_group": self.compact_group,
        }
        
        meta_json  = js.dumps(meta).encode('utf-8')
        seq_bytes  = self.sequence.encode('utf-8')
        feat_json  = js.dumps(self.features).encode('utf-8')
        hints_json = js.dumps(self.hints).encode('utf-8')
        
        meta_len  = len(meta_json)
        seq_len   = len(seq_bytes)
        feat_len  = len(feat_json)
        hints_len = len(hints_json)
        
        header = pack('<4I', meta_len, seq_len, feat_len, hints_len)
        data   = header + meta_json + seq_bytes + feat_json + hints_json
        
        return data
    
    @classmethod
    def from_bytes(cls, data):
        """Deserialize chunk from bytes"""
        
        header_size                            = calcsize('<4I')
        meta_len, seq_len, feat_len, hints_len = unpack('<4I', data[:header_size])
        
        offset     = header_size
        meta_json  = data[offset:offset + meta_len].decode('utf-8')
        offset    += meta_len
        
        seq_bytes  = data[offset:offset + seq_len].decode('utf-8')
        offset    += seq_len
        
        feat_json  = data[offset:offset + feat_len].decode('utf-8')
        offset    += feat_len
        
        hints_json = data[offset:offset + hints_len].decode('utf-8')
        
        meta     = js.loads(meta_json)
        features = js.loads(feat_json)
        hints    = js.loads(hints_json)
        
        return cls(
            seqid         = meta["seqid"],
            start         = meta["start"],
            end           = meta["end"],
            strand        = meta["strand"],
            sequence      = seq_bytes,
            features      = features,
            biotype       = meta.get("biotype", "."),
            gene_ids      = meta.get("gene_ids", []),
            has_hints     = meta.get("has_hints", False),
            hints         = hints,
            chunk_index   = meta.get("chunk_index", 0),
            is_augmented  = meta.get("is_augmented", False),
            compact_group = meta.get("compact_group"),
        )
    
    def estimate_input_tokens(self, bp_per_token=4.5):
        """Estimate input token count for this chunk"""
        
        seq_tokens  = len(self.sequence) / bp_per_token
        hint_tokens = len(self.hints) * 5 if self.has_hints else 0
        overhead    = 5
        
        return int(seq_tokens + hint_tokens + overhead)


#####################  Writing  #####################


def write_binary(chunks, output_path, compress=True):
    """Write chunks to binary file with optional compression"""
    
    output_path = pl.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(MAGIC_HEADER)
        f.write(pack('<B', FORMAT_VERSION))
        f.write(pack('<B', 1 if compress else 0))
        f.write(pack('<I', len(chunks)))
        
        offsets    = []
        data_start = f.tell() + len(chunks) * 8
        
        all_data       = []
        current_offset = data_start
        
        for chunk in chunks:
            chunk_bytes = chunk.to_bytes()
            
            if compress:
                chunk_bytes = zl.compress(chunk_bytes, level=6)
            
            offsets.append((current_offset, len(chunk_bytes)))
            all_data.append(chunk_bytes)
            current_offset += len(chunk_bytes)
        
        for offset, length in offsets:
            f.write(st.pack('<II', offset, length))
        
        for data in all_data:
            f.write(data)
    
    return output_path


#####################
#####  Reading  #####
#####################


def read_binary(input_path):
    """Read all chunks from binary file"""
    
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}")
        
        version    = unpack('<B', f.read(1))[0]
        compressed = unpack('<B', f.read(1))[0]
        num_chunks = unpack('<I', f.read(4))[0]
        
        offsets = []
        for _ in range(num_chunks):
            offset, length = unpack('<II', f.read(8))
            offsets.append((offset, length))
        
        chunks = []
        for offset, length in offsets:
            f.seek(offset)
            data = f.read(length)
            
            if compressed:
                data = zl.decompress(data)
            
            chunk = BinaryChunk.from_bytes(data)
            chunks.append(chunk)
    
    return chunks


def read_chunk_at_index(input_path, index):
    """Read single chunk by index without loading entire file"""
    
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}")
        
        version    = unpack('<B', f.read(1))[0]
        compressed = unpack('<B', f.read(1))[0]
        num_chunks = unpack('<I', f.read(4))[0]
        
        if index >= num_chunks:
            raise IndexError(f"Index {index} out of range (total: {num_chunks})")
        
        f.seek(10 + index * 8)
        offset, length = unpack('<II', f.read(8))
        
        f.seek(offset)
        data = f.read(length)
        
        if compressed:
            data = zl.decompress(data)
        
        return BinaryChunk.from_bytes(data)


def get_binary_info(input_path):
    """Get metadata about binary dataset without loading all chunks"""
    
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}")
        
        version    = unpack('<B', f.read(1))[0]
        compressed = unpack('<B', f.read(1))[0]
        num_chunks = unpack('<I', f.read(4))[0]
        
        offsets = []
        for _ in range(num_chunks):
            offset, length = unpack('<II', f.read(8))
            offsets.append((offset, length))
        
        total_size = sum(length for _, length in offsets)
    
    return {
        "version":    version,
        "compressed": bool(compressed),
        "num_chunks": num_chunks,
        "total_size": total_size,
        "file_path":  str(input_path),
    }


#####################  Merging  #####################


def merge_binary_files(input_paths, output_path, compress=True):
    """Merge multiple binary files into one"""
    
    all_chunks = []
    
    for path in input_paths:
        chunks = read_binary(path)
        all_chunks.extend(chunks)
    
    write_binary(all_chunks, output_path, compress)
    
    return len(all_chunks)
