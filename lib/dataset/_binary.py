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

# Compression types
COMPRESS_NONE = 0
COMPRESS_ZLIB = 1
COMPRESS_ZSTD = 2


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
        input_ids     = None,
        target_ids    = None,
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
        self.input_ids     = input_ids      # Pre-tokenized input IDs
        self.target_ids    = target_ids     # Pre-tokenized target IDs

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

        # Pack token IDs if present (for pre-tokenized chunks)
        if self.input_ids is not None:
            input_ids_bytes = struct.pack(f'<{len(self.input_ids)}i', *self.input_ids)
        else:
            input_ids_bytes = b''

        if self.target_ids is not None:
            target_ids_bytes = struct.pack(f'<{len(self.target_ids)}i', *self.target_ids)
        else:
            target_ids_bytes = b''

        meta_len       = len(meta_json)
        seq_len        = len(seq_bytes)
        feat_len       = len(feat_json)
        hints_len      = len(hints_json)
        input_ids_len  = len(self.input_ids) if self.input_ids else 0
        target_ids_len = len(self.target_ids) if self.target_ids else 0

        # v2 format: 6 length fields
        header = struct.pack('<6I', meta_len, seq_len, feat_len, hints_len, input_ids_len, target_ids_len)
        data   = header + meta_json + seq_bytes + feat_json + hints_json + input_ids_bytes + target_ids_bytes

        return data

    @classmethod
    def from_bytes(cls, data):
        """Deserialize chunk from bytes (v2 format with optional pre-tokenized IDs)"""

        # Try v2 format first (6I header), fall back to v1 (4I header)
        header_size_v2 = struct.calcsize('<6I')
        header_size_v1 = struct.calcsize('<4I')

        # Check if we have enough data for v2 header and if input_ids_len/target_ids_len are reasonable
        if len(data) >= header_size_v2:
            meta_len, seq_len, feat_len, hints_len, input_ids_len, target_ids_len = struct.unpack(
                '<6I', data[:header_size_v2]
            )

            # Sanity check: if token counts are unreasonably large, it's v1 format
            if input_ids_len > 1000000 or target_ids_len > 1000000:
                # Fall back to v1 parsing
                meta_len, seq_len, feat_len, hints_len = struct.unpack('<4I', data[:header_size_v1])
                input_ids_len  = 0
                target_ids_len = 0
                offset         = header_size_v1
            else:
                offset = header_size_v2
        else:
            # v1 format
            meta_len, seq_len, feat_len, hints_len = struct.unpack('<4I', data[:header_size_v1])
            input_ids_len  = 0
            target_ids_len = 0
            offset         = header_size_v1

        meta_json  = data[offset:offset + meta_len].decode('utf-8')
        offset    += meta_len

        seq_bytes  = data[offset:offset + seq_len].decode('utf-8')
        offset    += seq_len

        feat_json  = data[offset:offset + feat_len].decode('utf-8')
        offset    += feat_len

        hints_json = data[offset:offset + hints_len].decode('utf-8')
        offset    += hints_len

        # Unpack pre-tokenized IDs if present
        input_ids  = None
        target_ids = None

        if input_ids_len > 0:
            input_ids_bytes = data[offset:offset + input_ids_len * 4]
            input_ids       = list(struct.unpack(f'<{input_ids_len}i', input_ids_bytes))
            offset         += input_ids_len * 4

        if target_ids_len > 0:
            target_ids_bytes = data[offset:offset + target_ids_len * 4]
            target_ids       = list(struct.unpack(f'<{target_ids_len}i', target_ids_bytes))

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
            input_ids     = input_ids,
            target_ids    = target_ids,
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
        """Format gene index (one transcript per gene, no sub-indices)"""

        gene_idx = gene_indices.get(gene_id, 1)
        return str(gene_idx)

    def get_input_text(self):
        """Format chunk as input text for tokenization"""

        input_text = self.sequence

        if self.has_hints and self.hints:
            input_text += r"[\n][HIT]"
            for h in sorted(self.hints, key=lambda x: x.get("start", 0)):
                htype   = h.get("type", "exon").lower()
                hstart  = h.get("start", 0)
                hend    = h.get("end", 0)
                hstrand = h.get("strand", "+")
                input_text += rf"[\n]{htype}{hstart}[\t]{hend}{hstrand}"

        return input_text

    def get_target_text(self):
        """Format chunk as target text for tokenization (compressed format)"""

        gene_indices, transcript_indices = self._build_gene_transcript_indices()

        target_text     = "<bos>"
        sorted_features = sorted(self.features, key=lambda x: x.get("start", 0))

        for f in sorted_features:
            ftype = f.get("type", "exon").lower()

            # Skip CDS features - info merged into exon
            if ftype == "cds":
                continue

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

            # Build trailing - UTR lines have [\t]{cds_coord}
            # 5'UTR: cds_start (where CDS begins)
            # 3'UTR: cds_end (where CDS ends)
            cds_start = f.get("cds_start")
            cds_end   = f.get("cds_end")

            if cds_start is not None and cds_start > fstart:
                trailing = rf"[\t]{cds_start}"
            elif cds_end is not None and cds_end < fend:
                trailing = rf"[\t]{cds_end}"
            else:
                trailing = ""

            # Format: {start}[\t]{end}{strand}{phase}{biotype}{gene_idx}[[\t]{cds_coord}][\n]
            target_text += rf"{fstart}[\t]{fend}{fstrand}{fphase}{biotype}{gene_idx_str}{trailing}[\n]"

        target_text += "<eos>"

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


def write_binary(chunks, output_path, show_progress=True, compress=None):
    """
    Write chunks to binary file with optional compression

    Uses streaming write with seek-based offset table to minimize memory:
    - Memory: O(n) for offset array (~12MB per 1M chunks) + O(1) chunk buffer
    - Single pass through chunks, single serialization per chunk
    - v2 format: 64-bit offsets

    Args:
        compress: None (no compression), 'zlib', or 'zstd'
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_chunks = len(chunks)

    # Determine compression
    if compress == 'zstd':
        try:
            import zstd
            compress_byte = COMPRESS_ZSTD
            compressor    = lambda data: zstd.compress(data, 3)
        except ImportError:
            print("  WARNING: zstd not available, falling back to zlib")
            import zlib
            compress_byte = COMPRESS_ZLIB
            compressor    = lambda data: zlib.compress(data, 6)
    elif compress == 'zlib':
        import zlib
        compress_byte = COMPRESS_ZLIB
        compressor    = lambda data: zlib.compress(data, 6)
    else:
        compress_byte = COMPRESS_NONE
        compressor    = None

    with open(output_path, 'wb') as f:
        f.write(MAGIC_HEADER)
        f.write(struct.pack('<B', FORMAT_VERSION))
        f.write(struct.pack('<B', compress_byte))
        f.write(struct.pack('<I', num_chunks))

        offset_table_pos = f.tell()
        f.write(b'\x00' * (num_chunks * OFFSET_ENTRY_SIZE_V2))

        offsets = []

        for i, chunk in enumerate(chunks):
            current_offset = f.tell()
            chunk_bytes    = chunk.to_bytes()

            if compressor:
                chunk_bytes = compressor(chunk_bytes)

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


def _get_decompressor(compress_type):
    """Get decompression function for given compression type"""

    if compress_type == COMPRESS_ZSTD:
        try:
            import zstd
            return zstd.decompress
        except ImportError:
            raise ImportError("zstd required to read this file. Install with: pip install zstd")
    elif compress_type == COMPRESS_ZLIB:
        import zlib
        return zlib.decompress
    else:
        return None


def read_binary(input_path):
    """Read all chunks from binary file (supports v1-v2 formats, zlib/zstd)"""

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}")

        version      = struct.unpack('<B', f.read(1))[0]
        compress_type = struct.unpack('<B', f.read(1))[0]
        num_chunks   = struct.unpack('<I', f.read(4))[0]

        decompressor = _get_decompressor(compress_type)

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

            if decompressor:
                data = decompressor(data)

            chunk = BinaryChunk.from_bytes(data)
            chunks.append(chunk)

    return chunks


def read_chunk_at_index(input_path, index):
    """Read single chunk by index without loading entire file"""

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}")

        version       = struct.unpack('<B', f.read(1))[0]
        compress_type = struct.unpack('<B', f.read(1))[0]
        num_chunks    = struct.unpack('<I', f.read(4))[0]

        if index >= num_chunks:
            raise IndexError(f"Index {index} out of range (total: {num_chunks})")

        decompressor = _get_decompressor(compress_type)

        if version >= 2:
            f.seek(10 + index * 12)
            offset, length = struct.unpack('<QI', f.read(12))
        else:
            f.seek(10 + index * 8)
            offset, length = struct.unpack('<II', f.read(8))

        f.seek(offset)
        data = f.read(length)

        if decompressor:
            data = decompressor(data)

        return BinaryChunk.from_bytes(data)


def iter_binary(input_path):
    """
    Iterate chunks from binary file without loading all into memory

    Yields (chunk_idx, chunk) tuples. More efficient than read_binary for
    sequential processing, and more efficient than repeated read_chunk_at_index.
    """
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}")

        version       = struct.unpack('<B', f.read(1))[0]
        compress_type = struct.unpack('<B', f.read(1))[0]
        num_chunks    = struct.unpack('<I', f.read(4))[0]

        decompressor = _get_decompressor(compress_type)

        # Read offset table
        offsets = []
        if version >= 2:
            for _ in range(num_chunks):
                offset, length = struct.unpack('<QI', f.read(12))
                offsets.append((offset, length))
        else:
            for _ in range(num_chunks):
                offset, length = struct.unpack('<II', f.read(8))
                offsets.append((offset, length))

        # Yield chunks sequentially
        for chunk_idx, (offset, length) in enumerate(offsets):
            f.seek(offset)
            data = f.read(length)

            if decompressor:
                data = decompressor(data)

            yield chunk_idx, BinaryChunk.from_bytes(data)


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