import random
import math
import heapq

from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any


#######################
#####  Utilities  #####
#######################


def align_to_block(length, block_size=64):
    """Align length to next block boundary"""
    return math.ceil(length / block_size) * block_size


def compute_effective_length(raw_length, block_size=64, window_size=256):
    """Compute effective length including isolation padding"""
    aligned   = align_to_block(raw_length, block_size)
    isolation = window_size + block_size
    return aligned + isolation


##########################################
#####  Streaming Compaction Pipeline #####
##########################################


# Lightweight metadata - ~40 bytes per chunk instead of ~17KB
ChunkMeta = namedtuple('ChunkMeta', ['file_idx', 'chunk_idx', 'token_length', 'eff_length', 'is_augmented'])


def _process_single_file(args):
    """Process a single file and return its metadata. Used by parallel extraction."""
    file_idx, file_path, tokenizer, batch_size, n_workers, block_size, isolation_pad = args
    from lib.dataset._binary import read_binary

    # Load and tokenize
    chunks = read_binary(file_path)
    file_lengths = _tokenize_file_chunks(chunks, tokenizer, batch_size, n_workers)

    # Extract metadata
    file_metadata = []
    raw_count = 0
    aug_count = 0

    for chunk_idx, (chunk, token_len) in enumerate(zip(chunks, file_lengths)):
        eff_len = align_to_block(token_len, block_size) + isolation_pad
        meta = ChunkMeta(
            file_idx=file_idx,
            chunk_idx=chunk_idx,
            token_length=token_len,
            eff_length=eff_len,
            is_augmented=chunk.is_augmented,
        )
        file_metadata.append(meta)

        if chunk.is_augmented:
            aug_count += 1
        else:
            raw_count += 1

    chunk_count = len(chunks)

    # Free memory
    del chunks
    del file_lengths

    return file_idx, file_path.name, chunk_count, raw_count, aug_count, file_metadata


def stream_extract_metadata(
    file_paths,
    tokenizer,
    batch_size=1000,
    n_workers=4,
    file_parallel=1,
    block_size=64,
    window_size=256,
):
    """
    Phase 1: Stream through files, extract only metadata.

    Memory: O(file_parallel * max_file_size) + O(n) for metadata
    With file_parallel=2, loads 2 files simultaneously.

    Args:
        file_paths: List of input file paths
        tokenizer: HuggingFace tokenizer
        batch_size: Tokenization batch size
        n_workers: Workers per file for tokenization
        file_parallel: Number of files to process in parallel (default 1)
        block_size: Alignment block size
        window_size: Attention window size
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    isolation_pad = window_size + block_size
    total_files = len(file_paths)

    if file_parallel > 1:
        print(f"    Processing {file_parallel} files in parallel...")
        print(f"    (Memory: ~{file_parallel}x largest file size)\n")

        # Prepare arguments for each file
        work_items = [
            (file_idx, file_path, tokenizer, batch_size, n_workers, block_size, isolation_pad)
            for file_idx, file_path in enumerate(file_paths)
        ]

        # Process files in parallel
        all_metadata = []
        total_chunks = 0
        raw_count = 0
        aug_count = 0
        completed = 0

        # Use thread pool - tokenizers release GIL so this works well
        with ThreadPoolExecutor(max_workers=file_parallel) as executor:
            futures = {executor.submit(_process_single_file, item): item[0] for item in work_items}

            for future in as_completed(futures):
                file_idx, file_name, chunk_count, file_raw, file_aug, file_metadata = future.result()
                all_metadata.extend(file_metadata)
                total_chunks += chunk_count
                raw_count += file_raw
                aug_count += file_aug
                completed += 1
                print(f"    [{completed}/{total_files}] {file_name}: {chunk_count:,} chunks")

        # Sort metadata by file_idx to maintain consistent ordering
        all_metadata.sort(key=lambda m: (m.file_idx, m.chunk_idx))

    else:
        # Sequential processing (original behavior)
        all_metadata = []
        total_chunks = 0
        raw_count = 0
        aug_count = 0

        for file_idx, file_path in enumerate(file_paths):
            print(f"    [{file_idx + 1}/{total_files}] {file_path.name}...", end=' ', flush=True)

            result = _process_single_file(
                (file_idx, file_path, tokenizer, batch_size, n_workers, block_size, isolation_pad)
            )
            _, _, chunk_count, file_raw, file_aug, file_metadata = result

            all_metadata.extend(file_metadata)
            total_chunks += chunk_count
            raw_count += file_raw
            aug_count += file_aug
            print(f"{chunk_count:,} chunks")

    print(f"\n  Total: {total_chunks:,} chunks (raw: {raw_count:,}, aug: {aug_count:,})")
    print(f"  Metadata size: ~{len(all_metadata) * 40 / 1024 / 1024:.1f} MB")

    return all_metadata


def _tokenize_file_chunks(chunks, tokenizer, batch_size, n_workers):
    """Tokenize chunks from a single file using parallel batches."""
    total = len(chunks)
    if total == 0:
        return []

    lengths = [0] * total

    def process_batch(start_idx):
        end_idx = min(start_idx + batch_size, total)
        texts = [chunks[i].get_input_text() for i in range(start_idx, end_idx)]
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
        return start_idx, [len(ids) for ids in encoded['input_ids']]

    batch_starts = list(range(0, total, batch_size))

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_batch, start) for start in batch_starts]
        for future in as_completed(futures):
            start_idx, batch_lengths = future.result()
            lengths[start_idx:start_idx + len(batch_lengths)] = batch_lengths

    return lengths


def pack_from_metadata(
    metadata_list: List[ChunkMeta],
    target_length: int,
    hard_limit: int = None,
    seed: int = 42,
) -> Tuple[Dict[Tuple[int, int], int], dict]:
    """
    Phase 2: Run FFD bin packing on metadata only (integers, no chunk data).

    Memory: O(n) for metadata + O(n) for group assignments dict
    All operations on integers, no chunk objects in memory.

    Returns:
        group_assignments: dict mapping (file_idx, chunk_idx) -> group_id
        stats: packing statistics
    """
    random.seed(seed)
    hard_limit = hard_limit or int(target_length * 1.1)

    # Separate by augmentation status
    raw_metas = [m for m in metadata_list if not m.is_augmented]
    aug_metas = [m for m in metadata_list if m.is_augmented]

    print(f"    Raw: {len(raw_metas):,}, Augmented: {len(aug_metas):,}")

    # Group assignments: (file_idx, chunk_idx) -> group_id
    group_assignments = {}
    current_group_id = 0

    stats = {
        "total_groups": 0,
        "utilizations": [],
        "overflow_count": 0,
        "singleton_count": 0,
    }

    for metas, category in [(raw_metas, "raw"), (aug_metas, "augmented")]:
        if not metas:
            continue

        print(f"\n  Packing {len(metas):,} {category} chunks...")

        # Sort by effective length descending (FFD)
        sorted_metas = sorted(metas, key=lambda m: -m.eff_length)

        # Heap-based bin packing - O(n log n)
        bins = []           # List of lists of (file_idx, chunk_idx)
        bin_totals = []     # Effective length totals

        target_heap = []    # Bins with space under target
        overflow_heap = []  # Bins with space only under hard_limit

        total = len(sorted_metas)

        for progress_idx, meta in enumerate(sorted_metas):
            if progress_idx % 50000 == 0:
                pct = 100 * progress_idx / total
                print(f"    Packing: {progress_idx:,}/{total:,} ({pct:.1f}%) - {len(bins):,} bins", end='\r')

            eff_len = meta.eff_length
            chunk_key = (meta.file_idx, meta.chunk_idx)

            # Oversized - standalone bin
            if eff_len > hard_limit:
                bin_idx = len(bins)
                bins.append([chunk_key])
                bin_totals.append(eff_len)
                group_assignments[chunk_key] = current_group_id + bin_idx
                stats["overflow_count"] += 1
                continue

            placed = False

            # Try target heap first
            while target_heap and not placed:
                neg_remaining, bin_idx = heapq.heappop(target_heap)
                current_total = bin_totals[bin_idx]
                remaining = target_length - current_total

                if remaining >= eff_len:
                    bins[bin_idx].append(chunk_key)
                    bin_totals[bin_idx] += eff_len
                    group_assignments[chunk_key] = current_group_id + bin_idx

                    new_remaining = target_length - bin_totals[bin_idx]
                    if new_remaining > 0:
                        heapq.heappush(target_heap, (-new_remaining, bin_idx))
                    else:
                        hard_remaining = hard_limit - bin_totals[bin_idx]
                        if hard_remaining > 0:
                            heapq.heappush(overflow_heap, (-hard_remaining, bin_idx))
                    placed = True
                else:
                    hard_remaining = hard_limit - current_total
                    if hard_remaining > 0:
                        heapq.heappush(overflow_heap, (-hard_remaining, bin_idx))

            # Try overflow heap
            while overflow_heap and not placed:
                neg_remaining, bin_idx = heapq.heappop(overflow_heap)
                current_total = bin_totals[bin_idx]
                remaining = hard_limit - current_total

                if remaining >= eff_len:
                    bins[bin_idx].append(chunk_key)
                    bin_totals[bin_idx] += eff_len
                    group_assignments[chunk_key] = current_group_id + bin_idx

                    new_remaining = hard_limit - bin_totals[bin_idx]
                    if new_remaining > 0:
                        heapq.heappush(overflow_heap, (-new_remaining, bin_idx))
                    placed = True

            # Create new bin
            if not placed:
                bin_idx = len(bins)
                bins.append([chunk_key])
                bin_totals.append(eff_len)
                group_assignments[chunk_key] = current_group_id + bin_idx

                remaining = target_length - eff_len
                if remaining > 0:
                    heapq.heappush(target_heap, (-remaining, bin_idx))
                else:
                    hard_remaining = hard_limit - eff_len
                    if hard_remaining > 0:
                        heapq.heappush(overflow_heap, (-hard_remaining, bin_idx))

        print(f"    Packing: {total:,}/{total:,} (100.0%) - {len(bins):,} bins")

        # Update stats
        for bin_idx, bin_keys in enumerate(bins):
            total_len = bin_totals[bin_idx]
            utilization = total_len / target_length if target_length > 0 else 0
            stats["utilizations"].append(utilization)
            stats["total_groups"] += 1
            if len(bin_keys) == 1:
                stats["singleton_count"] += 1

        current_group_id += len(bins)

    # Compute summary stats
    if stats["utilizations"]:
        stats["avg_utilization"] = sum(stats["utilizations"]) / len(stats["utilizations"])
        stats["min_utilization"] = min(stats["utilizations"])
        stats["max_utilization"] = max(stats["utilizations"])
    else:
        stats["avg_utilization"] = 0
        stats["min_utilization"] = 0
        stats["max_utilization"] = 0

    return group_assignments, stats


def stream_write_compacted(
    file_paths,
    group_assignments: Dict[Tuple[int, int], int],
    output_path,
    compress=False,
    show_progress=True,
):
    """
    Phase 3: Re-read chunks and write to output with group assignments.

    Memory: O(file_size) - only one input file loaded at a time.
    Chunks are written immediately then the file is freed.
    Uses v2 format with 64-bit offsets for files >4GB.
    """
    import pathlib
    import struct
    from lib.dataset._binary import read_binary, MAGIC_HEADER, FORMAT_VERSION, OFFSET_ENTRY_SIZE_V2

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = len(group_assignments)

    print(f"    Writing {total_chunks:,} chunks to {output_path.name}...")

    with open(output_path, 'wb') as f:
        # Write header
        f.write(MAGIC_HEADER)
        f.write(struct.pack('<B', FORMAT_VERSION))  # v2
        f.write(struct.pack('<B', 1 if compress else 0))
        f.write(struct.pack('<I', total_chunks))

        # Placeholder offset table (will seek back to fill)
        # v2: 12 bytes per entry (QI: 64-bit offset + 32-bit length)
        offset_table_pos = f.tell()
        f.write(b'\x00' * (total_chunks * OFFSET_ENTRY_SIZE_V2))

        # Write chunks in order sorted by (file_idx, chunk_idx)
        chunk_order = sorted(group_assignments.keys())
        offsets = []

        written = 0
        current_file_idx = -1
        current_chunks = None

        for file_idx, chunk_idx in chunk_order:
            # Load new file if needed
            if file_idx != current_file_idx:
                if current_chunks is not None:
                    del current_chunks
                print(f"    Loading file {file_idx + 1}/{len(file_paths)}...", end='\r')
                current_chunks = read_binary(file_paths[file_idx])
                current_file_idx = file_idx

            # Get chunk and set group
            chunk = current_chunks[chunk_idx]
            chunk.compact_group = group_assignments[(file_idx, chunk_idx)]

            # Serialize and write immediately
            current_offset = f.tell()
            chunk_bytes = chunk.to_bytes()
            if compress:
                import zlib
                chunk_bytes = zlib.compress(chunk_bytes, level=6)

            f.write(chunk_bytes)
            offsets.append((current_offset, len(chunk_bytes)))

            written += 1
            if show_progress and written % 50000 == 0:
                pct = 100 * written / total_chunks
                print(f"    Written: {written:,}/{total_chunks:,} ({pct:.1f}%)", end='\r')

        # Free last file
        if current_chunks is not None:
            del current_chunks

        if show_progress:
            print(f"    Written: {total_chunks:,}/{total_chunks:,} (100.0%)")

        # Seek back and write offset table (v2: QI format)
        f.seek(offset_table_pos)
        for offset, length in offsets:
            f.write(struct.pack('<QI', offset, length))

    return output_path


############################
#####  Segment Packing #####
############################


def pack_with_isolation(token_lists, pad_id, sep_id, block_size=64, window_size=256):
    """Pack token sequences with window-based isolation between segments"""

    packed         = []
    segment_starts = []
    segment_ends   = []

    for i, tokens in enumerate(token_lists):
        segment_starts.append(len(packed))
        packed.extend(tokens)
        segment_ends.append(len(packed))

        if i < len(token_lists) - 1:
            packed.append(sep_id)

            current    = len(packed)
            target_gap = window_size + 1
            next_start = align_to_block(current + target_gap, block_size)
            pad_needed = next_start - current

            packed.extend([pad_id] * pad_needed)

    return packed, segment_starts, segment_ends


def verify_isolation(segment_starts, segment_ends, window_size=256):
    """Verify all segments are isolated by window constraint"""

    for i in range(len(segment_starts) - 1):
        gap = segment_starts[i + 1] - segment_ends[i]
        if gap <= window_size:
            return False, i, i + 1, gap

    return True, None, None, None


def build_segment_mask(segment_starts, segment_ends, seq_len):
    """Build segment membership array for masking"""

    segment_ids = [-1] * seq_len

    for seg_idx, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        for pos in range(start, end):
            segment_ids[pos] = seg_idx

    return segment_ids
