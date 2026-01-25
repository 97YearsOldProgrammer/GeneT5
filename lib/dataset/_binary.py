import struct
import json
import zlib
import pathlib


#######################
#####  Constants  #####
#######################


MAGIC_HEADER   = b'GT5B'  # GeneT5 Binary
FORMAT_VERSION = 1


#########################
#####  BinaryChunk  #####
#########################


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
        biotype       = ".",
        gene_ids      = None,
        has_hints     = False,
        hints         = None,
        chunk_index   = 0,
        is_augmented  = False,
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

        meta_json  = json.dumps(meta).encode('utf-8')
        seq_bytes  = self.sequence.encode('utf-8')
        feat_json  = json.dumps(self.features).encode('utf-8')
        hints_json = json.dumps(self.hints).encode('utf-8')

        meta_len  = len(meta_json)
        seq_len   = len(seq_bytes)
        feat_len  = len(feat_json)
        hints_len = len(hints_json)

        header = struct.pack('<4I', meta_len, seq_len, feat_len, hints_len)
        data   = header + meta_json + seq_bytes + feat_json + hints_json

        return data

    @classmethod
    def from_bytes(cls, data):
        """Deserialize chunk from bytes"""

        header_size                            = struct.calcsize('<4I')
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

    def get_input_text(self):
        """Format chunk as input text for tokenization"""

        input_text = self.sequence

        if self.has_hints and self.hints:
            input_text += "\n[HIT]"
            for h in sorted(self.hints, key=lambda x: x.get("start", 0)):
                htype   = h.get("type", "exon").lower()
                hstart  = h.get("start", 0)
                hend    = h.get("end", 0)
                hstrand = h.get("strand", "+")
                input_text += f"\n{htype}\t{hstart}\t{hend}\t{hstrand}"

        return input_text

    def get_target_text(self):
        """Format chunk as target text for tokenization"""

        target_text = "<BOS>"

        for f in sorted(self.features, key=lambda x: x.get("start", 0)):
            ftype   = f.get("type", "exon").lower()
            fstart  = f.get("start", 0)
            fend    = f.get("end", 0)
            fstrand = f.get("strand", "+")
            fphase  = f.get("phase", ".")
            target_text += f"\n{ftype}\t{fstart}\t{fend}\t{fstrand}\t{fphase}"

        target_text += "\n<EOS>"

        return target_text

    def estimate_input_tokens(self, tokenizer=None):
        """
        Estimate input token count for this chunk.
        
        If tokenizer provided: returns exact count
        Otherwise: rough estimate (not recommended)
        """

        input_text = self.get_input_text()

        if tokenizer is not None:
            return len(tokenizer.encode(input_text, add_special_tokens=False))

        # Fallback: rough estimate
        seq_tokens  = len(self.sequence) / 4.5
        hint_tokens = len(self.hints) * 5 if self.has_hints else 0
        overhead    = 5

        return int(seq_tokens + hint_tokens + overhead)


#################
#####  I/O  #####
#################


def write_binary(chunks, output_path, compress=True):
    """Write chunks to binary file with optional compression"""

    output_path = pathlib.Path(output_path)
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


def read_binary(input_path):
    """Read all chunks from binary file"""

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


def read_chunk_at_index(input_path, index):
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


def get_binary_info(input_path):
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


def merge_binary_files(input_paths, output_path, compress=True):
    """Merge multiple binary files into one"""

    all_chunks = []

    for path in input_paths:
        chunks = read_binary(path)
        all_chunks.extend(chunks)

    write_binary(all_chunks, output_path, compress)

    return len(all_chunks)