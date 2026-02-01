import struct
import json
import pathlib


#######################
#####  Constants  #####
#######################


MAGIC_HEADER   = b'GT5B'
FORMAT_VERSION = 2  # v2: 64-bit offsets (QI) for files >4GB

# Offset table entry sizes by version
OFFSET_ENTRY_SIZE_V1 = 8   # II: 32-bit offset, 32-bit length
OFFSET_ENTRY_SIZE_V2 = 12  # QI: 64-bit offset, 32-bit length


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
        input_len     = None,
        target_len    = None,
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
        self.input_len     = input_len
        self.target_len    = target_len

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
            "input_len":     self.input_len,
            "target_len":    self.target_len,
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
            input_len     = meta.get("input_len"),
            target_len    = meta.get("target_len"),
        )

    def _build_gene_transcript_indices(self):
        """Build gene and transcript index mappings for features"""

        gene_indices       = {}
        transcript_indices = {}
        gene_counter       = 1

        sorted_features = sorted(self.features, key=lambda x: x.get("start", 0))

        for f in sorted_features:
            gene_id       = f.get("gene_id", "")
            transcript_id = f.get("transcript_id", "")

            if gene_id and gene_id not in gene_indices:
                gene_indices[gene_id]       = gene_counter
                transcript_indices[gene_id] = {}
                gene_counter               += 1

            if gene_id and transcript_id:
                if transcript_id not in transcript_indices[gene_id]:
                    t_idx = len(transcript_indices[gene_id]) + 1
                    transcript_indices[gene_id][transcript_id] = t_idx

        return gene_indices, transcript_indices

    def _format_gene_idx(self, gene_id, transcript_id, gene_indices, transcript_indices):
        """Format gene index with optional transcript index for alt splicing"""

        gene_idx = gene_indices.get(gene_id, 1)

        # Check if gene has multiple transcripts
        if gene_id and transcript_id:
            trans_map = transcript_indices.get(gene_id, {})
            if len(trans_map) > 1:
                t_idx = trans_map.get(transcript_id, 1)
                return f"{gene_idx}.{t_idx}"

        return str(gene_idx)

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
                input_text += f"\n{htype}{hstart}\t{hend}{hstrand}"

        return input_text

    def get_target_text(self):
        """Format chunk as target text for tokenization"""

        gene_indices, transcript_indices = self._build_gene_transcript_indices()

        target_text     = "<BOS>"
        sorted_features = sorted(self.features, key=lambda x: x.get("start", 0))

        for f in sorted_features:
            ftype   = f.get("type", "exon").lower()
            fstart  = f.get("start", 0)
            fend    = f.get("end", 0)
            fstrand = f.get("strand", "+")
            fphase  = f.get("phase", ".")
            biotype = f.get("biotype", ".")

            gene_id       = f.get("gene_id", "")
            transcript_id = f.get("transcript_id", "")

            gene_idx_str = self._format_gene_idx(
                gene_id, transcript_id, gene_indices, transcript_indices
            )

            # Format: type start \t end strand phase biotype gene_idx
            # biotype before gene_idx to avoid phase+gene_idx tokenizer confusion
            target_text += f"\n{ftype}{fstart}\t{fend}{fstrand}{fphase}{biotype}{gene_idx_str}"

        target_text += "\n<EOS>"

        return target_text

    def estimate_input_tokens(self, tokenizer=None):
        """Estimate input token count for this chunk"""

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


def write_binary(chunks, output_path, show_progress=True):
    """
    Write chunks to binary file (uncompressed for fast streaming)

    Uses streaming write with seek-based offset table to minimize memory:
    - Memory: O(n) for offset array (~12MB per 1M chunks) + O(1) chunk buffer
    - Single pass through chunks, single serialization per chunk
    - v3 format: 64-bit offsets, uncompressed
    """

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_chunks = len(chunks)

    with open(output_path, 'wb') as f:
        f.write(MAGIC_HEADER)
        f.write(struct.pack('<B', FORMAT_VERSION))
        f.write(struct.pack('<B', 0))
        f.write(struct.pack('<I', num_chunks))

        offset_table_pos = f.tell()
        f.write(b'\x00' * (num_chunks * OFFSET_ENTRY_SIZE_V2))

        offsets = []

        for i, chunk in enumerate(chunks):
            current_offset = f.tell()
            chunk_bytes    = chunk.to_bytes()

            f.write(chunk_bytes)
            offsets.append((current_offset, len(chunk_bytes)))

            if show_progress and (i + 1) % 50000 == 0:
                pct = 100 * (i + 1) / num_chunks
                print(f"    Writing: {i + 1:,}/{num_chunks:,} ({pct:.1f}%)", end='\r')

        if show_progress and num_chunks > 50000:
            print(f"    Writing: {num_chunks:,}/{num_chunks:,} (100.0%)")

        f.seek(offset_table_pos)
        for offset, length in offsets:
            f.write(struct.pack('<QI', offset, length))

    return output_path


def read_binary(input_path):
    """Read all chunks from binary file (supports v1-v3 formats)"""

    import zlib

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}")

        version    = struct.unpack('<B', f.read(1))[0]
        compressed = struct.unpack('<B', f.read(1))[0]
        num_chunks = struct.unpack('<I', f.read(4))[0]

        offsets = []
        if version >= 2:
            for _ in range(num_chunks):
                offset, length = struct.unpack('<QI', f.read(12))
                offsets.append((offset, length))
        else:
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

    import zlib

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}")

        version    = struct.unpack('<B', f.read(1))[0]
        compressed = struct.unpack('<B', f.read(1))[0]
        num_chunks = struct.unpack('<I', f.read(4))[0]

        if index >= num_chunks:
            raise IndexError(f"Index {index} out of range (total: {num_chunks})")

        if version >= 2:
            f.seek(10 + index * 12)
            offset, length = struct.unpack('<QI', f.read(12))
        else:
            f.seek(10 + index * 8)
            offset, length = struct.unpack('<II', f.read(8))

        f.seek(offset)
        data = f.read(length)

        if compressed:
            data = zlib.decompress(data)

        return BinaryChunk.from_bytes(data)


def get_binary_info(input_path):
    """Get metadata about binary dataset without loading all chunks (supports v1 and v2)"""

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}")

        version    = struct.unpack('<B', f.read(1))[0]
        compressed = struct.unpack('<B', f.read(1))[0]
        num_chunks = struct.unpack('<I', f.read(4))[0]

        # Read offset table based on version
        offsets = []
        if version >= 2:
            for _ in range(num_chunks):
                offset, length = struct.unpack('<QI', f.read(12))
                offsets.append((offset, length))
        else:
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