import os
import struct
import json
import pathlib
import atexit


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

    def get_input_text(self):
        """Format chunk as input text with optional intron hint DNA"""

        input_text = self.sequence

        if self.has_hints and self.hints:
            input_text += "<hints>"
            for h in sorted(self.hints, key=lambda x: x.get("start", 0)):
                htype = h.get("type", "intron_hc")
                tag   = "<hc>" if "hc" in htype else "<lc>"
                input_text += tag + self.sequence[h["start"]:h["end"]]

        return input_text

    def get_target_text(self):
        """Format target as exon DNA sequences grouped by gene"""

        sorted_features = sorted(self.features, key=lambda x: x.get("start", 0))

        genes = {}
        for f in sorted_features:
            if f.get("type", "").lower() != "exon":
                continue
            gid = f.get("gene_id", "unknown")
            if gid not in genes:
                genes[gid] = {"strand": f.get("strand", "+"), "exons": [], "pos": f["start"]}
            genes[gid]["exons"].append(f)

        target = "<bos>"
        for gid, g in sorted(genes.items(), key=lambda x: x[1]["pos"]):
            target += "<+>" if g["strand"] == "+" else "<->"
            for i, exon in enumerate(sorted(g["exons"], key=lambda e: e["start"])):
                if i > 0:
                    target += "<exon>"
                target += self.sequence[exon["start"]:exon["end"]]
        target += "<eos>"

        return target

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


################################
#####  File Handle Cache   #####
################################


_FILE_CACHE = {}


def _get_cached_handle(path):
    """Get or create persistent file handle with pre-loaded offset table"""

    path = str(path)
    if path in _FILE_CACHE:
        return _FILE_CACHE[path]

    f = open(path, 'rb')

    magic = f.read(4)
    if magic != MAGIC_HEADER:
        f.close()
        raise ValueError(f"Invalid magic header: {magic}")

    version       = struct.unpack('<B', f.read(1))[0]
    compress_type = struct.unpack('<B', f.read(1))[0]
    num_chunks    = struct.unpack('<I', f.read(4))[0]

    decompressor = _get_decompressor(compress_type)

    offsets = []
    if version >= 2:
        raw = f.read(num_chunks * OFFSET_ENTRY_SIZE_V2)
        for i in range(num_chunks):
            off = i * OFFSET_ENTRY_SIZE_V2
            offset, length = struct.unpack('<QI', raw[off:off + OFFSET_ENTRY_SIZE_V2])
            offsets.append((offset, length))
    else:
        raw = f.read(num_chunks * OFFSET_ENTRY_SIZE_V1)
        for i in range(num_chunks):
            off = i * OFFSET_ENTRY_SIZE_V1
            offset, length = struct.unpack('<II', raw[off:off + OFFSET_ENTRY_SIZE_V1])
            offsets.append((offset, length))

    # Advise kernel: deprioritize pages from this file for eviction
    try:
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_NOREUSE)
    except (AttributeError, OSError):
        pass

    _FILE_CACHE[path] = (f, offsets, decompressor)
    return _FILE_CACHE[path]


def _close_all_handles():
    """Close all cached file handles on interpreter exit"""

    for _, (f, _, _) in _FILE_CACHE.items():
        try:
            f.close()
        except Exception:
            pass
    _FILE_CACHE.clear()


atexit.register(_close_all_handles)


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
    """Read single chunk by index using persistent file handle"""

    f, offsets, decompressor = _get_cached_handle(input_path)

    if index >= len(offsets):
        raise IndexError(f"Index {index} out of range (total: {len(offsets)})")

    offset, length = offsets[index]
    f.seek(offset)
    data = f.read(length)

    # Drop just-read pages from kernel cache
    try:
        os.posix_fadvise(f.fileno(), offset, length, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass

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


def merge_binary_files(input_paths, output_path, compress=None, show_progress=True, max_chunks=None, seed=42):
    """
    Stream-merge multiple .bin files into one unified file

    Memory: O(1) per chunk — reads raw bytes from each source file
    and writes directly to output without deserializing BinaryChunk

    Handles mixed compression: decompresses if needed, optionally recompresses
    When max_chunks is set, randomly subsamples to cap total output
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_paths = [pathlib.Path(p) for p in input_paths]
    input_paths = [p for p in input_paths if p.exists()]

    if not input_paths:
        return output_path

    # First pass: count total chunks from all files (header-only reads)
    file_metas   = []
    total_chunks = 0

    for path in input_paths:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != MAGIC_HEADER:
                continue

            version       = struct.unpack('<B', f.read(1))[0]
            compress_type = struct.unpack('<B', f.read(1))[0]
            num_chunks    = struct.unpack('<I', f.read(4))[0]

            if num_chunks == 0:
                continue

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

            file_metas.append({
                'path':         path,
                'version':      version,
                'compress':     compress_type,
                'num_chunks':   num_chunks,
                'offsets':      offsets,
            })
            total_chunks += num_chunks

    if total_chunks == 0:
        return output_path

    # Subsample if total exceeds max_chunks
    selected = None
    if max_chunks is not None and max_chunks > 0 and total_chunks > max_chunks:
        import random
        random.seed(seed)

        # Build flat list of (file_idx, chunk_idx) then sample
        all_indices = []
        for fi, meta in enumerate(file_metas):
            for ci in range(meta['num_chunks']):
                all_indices.append((fi, ci))

        chosen       = set(random.sample(all_indices, max_chunks))
        selected     = {}
        for fi, ci in chosen:
            selected.setdefault(fi, set()).add(ci)

        if show_progress:
            print(f"  Subsampling {total_chunks:,} -> {max_chunks:,} chunks")
        total_chunks = max_chunks

    # Determine output compression
    if compress == 'zstd':
        try:
            import zstd
            out_compress   = COMPRESS_ZSTD
            compressor     = lambda data: zstd.compress(data, 3)
        except ImportError:
            import zlib
            out_compress   = COMPRESS_ZLIB
            compressor     = lambda data: zlib.compress(data, 6)
    elif compress == 'zlib':
        import zlib
        out_compress = COMPRESS_ZLIB
        compressor   = lambda data: zlib.compress(data, 6)
    else:
        out_compress = COMPRESS_NONE
        compressor   = None

    if show_progress:
        print(f"  Merging {len(file_metas)} files -> {output_path.name} ({total_chunks:,} chunks)")

    # Second pass: stream-write merged file
    with open(output_path, 'wb') as out:
        out.write(MAGIC_HEADER)
        out.write(struct.pack('<B', FORMAT_VERSION))
        out.write(struct.pack('<B', out_compress))
        out.write(struct.pack('<I', total_chunks))

        # Placeholder offset table
        offset_table_pos = out.tell()
        out.write(b'\x00' * (total_chunks * OFFSET_ENTRY_SIZE_V2))

        out_offsets   = []
        chunks_done   = 0

        for fi, meta in enumerate(file_metas):
            src_decomp = _get_decompressor(meta['compress'])

            # Can we skip decompression+recompression?
            passthrough = (meta['compress'] == out_compress)
            file_selected = selected.get(fi) if selected else None

            with open(meta['path'], 'rb') as src:
                for ci, (src_offset, src_length) in enumerate(meta['offsets']):
                    if file_selected is not None and ci not in file_selected:
                        continue

                    src.seek(src_offset)
                    raw_bytes = src.read(src_length)

                    if passthrough:
                        write_bytes = raw_bytes
                    else:
                        # Decompress from source format
                        if src_decomp:
                            plain = src_decomp(raw_bytes)
                        else:
                            plain = raw_bytes

                        # Recompress to output format
                        if compressor:
                            write_bytes = compressor(plain)
                        else:
                            write_bytes = plain

                    current_offset = out.tell()
                    out.write(write_bytes)
                    out_offsets.append((current_offset, len(write_bytes)))

                    chunks_done += 1
                    if show_progress and chunks_done % 50000 == 0:
                        pct = 100 * chunks_done / total_chunks
                        print(f"    {chunks_done:,}/{total_chunks:,} ({pct:.1f}%)", end='\r')

        if show_progress and total_chunks > 50000:
            print(f"    {total_chunks:,}/{total_chunks:,} (100.0%)")

        # Write real offset table
        out.seek(offset_table_pos)
        for offset, length in out_offsets:
            out.write(struct.pack('<QI', offset, length))

    file_size = output_path.stat().st_size
    if show_progress:
        if file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        elif file_size < 1024 ** 3:
            size_str = f"{file_size / 1024 / 1024:.1f} MB"
        else:
            size_str = f"{file_size / 1024 ** 3:.2f} GB"
        print(f"  Done: {size_str}")

    return output_path


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