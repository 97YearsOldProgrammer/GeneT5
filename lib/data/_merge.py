import struct
import pathlib

from lib.data._binary import (
    MAGIC_HEADER,
    FORMAT_VERSION,
    OFFSET_ENTRY_SIZE_V2,
    COMPRESS_NONE,
    COMPRESS_ZLIB,
    COMPRESS_ZSTD,
    _get_decompressor,
)


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
