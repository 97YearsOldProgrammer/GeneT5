import random
import math
import heapq

from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
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


# Lightweight metadata - ~48 bytes per chunk instead of ~17KB
ChunkMeta = namedtuple('ChunkMeta', ['file_idx', 'chunk_idx', 'token_length', 'eff_length', 'is_augmented', 'target_len'])


def _process_single_file(args):
    """Process a single file and return its metadata (streaming, low memory)"""

    file_idx, file_path, block_size, isolation_pad = args
    from lib.dataset._binary import iter_binary

    file_metadata = []
    raw_count     = 0
    aug_count     = 0
    chunk_count   = 0

    # Stream chunks sequentially (single file open, no repeated seeks)
    for chunk_idx, chunk in iter_binary(file_path):
        if chunk.input_len is None:
            raise ValueError(f"Chunk {chunk_idx} in {file_path} missing input_len - run parse_data first")

        token_len = chunk.input_len
        eff_len   = align_to_block(token_len, block_size) + isolation_pad

        meta = ChunkMeta(
            file_idx     = file_idx,
            chunk_idx    = chunk_idx,
            token_length = token_len,
            eff_length   = eff_len,
            is_augmented = chunk.is_augmented,
            target_len   = chunk.target_len or 0,
        )
        file_metadata.append(meta)

        if chunk.is_augmented:
            aug_count += 1
        else:
            raw_count += 1

        chunk_count += 1

    return file_idx, file_path.name, chunk_count, raw_count, aug_count, file_metadata


def stream_extract_metadata(
    file_paths,
    file_parallel=1,
    block_size=64,
    window_size=256,
):
    """
    Phase 1: Stream through files, extract only metadata

    Memory: O(file_parallel * max_file_size) + O(n) for metadata
    Requires pre-tokenized data with input_len stored in chunks
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    isolation_pad = window_size + block_size
    total_files   = len(file_paths)

    # Prepare arguments for each file
    work_items = [
        (file_idx, file_path, block_size, isolation_pad)
        for file_idx, file_path in enumerate(file_paths)
    ]

    all_metadata = []
    total_chunks = 0
    raw_count    = 0
    aug_count    = 0

    if file_parallel > 1:
        print(f"    Processing {file_parallel} files in parallel...\n")
        completed = 0

        with ThreadPoolExecutor(max_workers=file_parallel) as executor:
            futures = {executor.submit(_process_single_file, item): item[0] for item in work_items}

            for future in as_completed(futures):
                file_idx, file_name, chunk_count, file_raw, file_aug, file_metadata = future.result()
                all_metadata.extend(file_metadata)
                total_chunks += chunk_count
                raw_count    += file_raw
                aug_count    += file_aug
                completed    += 1
                print(f"    [{completed}/{total_files}] {file_name}: {chunk_count:,} chunks")

        all_metadata.sort(key=lambda m: (m.file_idx, m.chunk_idx))

    else:
        for file_idx, file_path in enumerate(file_paths):
            print(f"    [{file_idx + 1}/{total_files}] {file_path.name}...", end=' ', flush=True)

            result = _process_single_file(
                (file_idx, file_path, block_size, isolation_pad)
            )
            _, _, chunk_count, file_raw, file_aug, file_metadata = result

            all_metadata.extend(file_metadata)
            total_chunks += chunk_count
            raw_count    += file_raw
            aug_count    += file_aug

            print(f"{chunk_count:,} chunks")

    print(f"\n  Total: {total_chunks:,} chunks (raw: {raw_count:,}, aug: {aug_count:,})")
    print(f"  Metadata size: ~{len(all_metadata) * 40 / 1024 / 1024:.1f} MB")

    return all_metadata



def pack_from_metadata(
    metadata_list: List[ChunkMeta],
    target_length: int,
    hard_limit: int = None,
    max_target_len: int = None,
    target_hard_limit: int = None,
    seed: int = 42,
) -> Tuple[Dict[Tuple[int, int], int], dict]:
    """
    Phase 2: Run FFD bin packing on metadata only (integers, no chunk data).

    Memory: O(n) for metadata + O(n) for group assignments dict
    All operations on integers, no chunk objects in memory.

    Dual-constraint packing:
        - input constraint: target_length / hard_limit (effective input tokens)
        - target constraint: max_target_len / target_hard_limit (label tokens)

    Returns:
        group_assignments: dict mapping (file_idx, chunk_idx) -> group_id
        stats: packing statistics
    """
    random.seed(seed)
    hard_limit        = hard_limit or int(target_length * 1.1)
    target_hard_limit = target_hard_limit or (int(max_target_len * 1.1) if max_target_len else None)

    # Token length distribution stats
    lengths     = [m.token_length for m in metadata_list]
    thresholds  = [1000, 2000, 3000, 4000, 5000]
    print(f"    Total chunks: {len(metadata_list):,}")
    print(f"    Token lengths: min={min(lengths):,}, max={max(lengths):,}, avg={sum(lengths)//len(lengths):,}")
    for t in thresholds:
        count = sum(1 for l in lengths if l <= t)
        pct   = 100 * count / len(lengths)
        print(f"      ≤{t:,}: {count:,} ({pct:.1f}%)")

    # Group assignments: (file_idx, chunk_idx) -> group_id
    group_assignments = {}

    stats = {
        "total_groups":          0,
        "utilizations":          [],
        "overflow_count":        0,
        "singleton_count":       0,
        "target_overflow_count": 0,
        "target_utilizations":   [],
    }

    # Pack all chunks together (no separation by augmentation)
    metas = metadata_list
    print(f"\n  Packing {len(metas):,} chunks...")

    # Sort by effective length descending (FFD)
    sorted_metas = sorted(metas, key=lambda m: -m.eff_length)

    bins              = []    # List of lists of (file_idx, chunk_idx)
    bin_totals        = []    # Effective length totals
    bin_target_totals = []    # Target length totals

    target_heap   = []    # For single-constraint mode
    overflow_heap = []

    total    = len(sorted_metas)
    use_dual = max_target_len is not None

    # For dual-constraint: use numpy for vectorized bin search
    if use_dual:
        import numpy as np
        # Estimate bin count: avg ~3 chunks per bin, add 50% margin
        estimated_bins = max(total // 2, 10000)
        np_input_total = np.zeros(estimated_bins, dtype=np.int32)
        np_tgt_total   = np.zeros(np_input_total.shape, dtype=np.int32)
        np_open        = np.ones(np_input_total.shape, dtype=np.bool_)
        n_bins         = 0

    for progress_idx, meta in enumerate(sorted_metas):
        if progress_idx % 50000 == 0:
            pct = 100 * progress_idx / total
            n_b = n_bins if use_dual else len(bins)
            print(f"    Packing: {progress_idx:,}/{total:,} ({pct:.1f}%) - {n_b:,} bins", end='\r')

        eff_len   = meta.eff_length
        tgt_len   = meta.target_len
        chunk_key = (meta.file_idx, meta.chunk_idx)

        # Oversized input - standalone bin
        if eff_len > hard_limit:
            bin_idx = len(bins)
            bins.append([chunk_key])
            bin_totals.append(eff_len)
            bin_target_totals.append(tgt_len)
            group_assignments[chunk_key] = bin_idx
            stats["overflow_count"] += 1
            if use_dual:
                if n_bins >= len(np_input_total):
                    new_size       = len(np_input_total) * 2
                    new_input      = np.zeros(new_size, dtype=np.int32)
                    new_tgt        = np.zeros(new_size, dtype=np.int32)
                    new_open       = np.ones(new_size, dtype=np.bool_)
                    new_input[:n_bins] = np_input_total[:n_bins]
                    new_tgt[:n_bins]   = np_tgt_total[:n_bins]
                    new_open[:n_bins]  = np_open[:n_bins]
                    np_input_total, np_tgt_total, np_open = new_input, new_tgt, new_open
                np_input_total[n_bins] = eff_len
                np_tgt_total[n_bins]   = tgt_len
                np_open[n_bins]        = False
                n_bins += 1
            continue

        # Oversized target - standalone bin
        if use_dual and tgt_len > target_hard_limit:
            bin_idx = len(bins)
            bins.append([chunk_key])
            bin_totals.append(eff_len)
            bin_target_totals.append(tgt_len)
            group_assignments[chunk_key] = bin_idx
            stats["target_overflow_count"] += 1
            if n_bins >= len(np_input_total):
                new_size       = len(np_input_total) * 2
                new_input      = np.zeros(new_size, dtype=np.int32)
                new_tgt        = np.zeros(new_size, dtype=np.int32)
                new_open       = np.ones(new_size, dtype=np.bool_)
                new_input[:n_bins] = np_input_total[:n_bins]
                new_tgt[:n_bins]   = np_tgt_total[:n_bins]
                new_open[:n_bins]  = np_open[:n_bins]
                np_input_total, np_tgt_total, np_open = new_input, new_tgt, new_open
            np_input_total[n_bins] = eff_len
            np_tgt_total[n_bins]   = tgt_len
            np_open[n_bins]        = False
            n_bins += 1
            continue

        placed = False

        if not use_dual:
            # Single constraint: use original heap algorithm
            while target_heap and not placed:
                neg_remaining, bin_idx = heapq.heappop(target_heap)
                remaining = target_length - bin_totals[bin_idx]

                if remaining >= eff_len:
                    bins[bin_idx].append(chunk_key)
                    bin_totals[bin_idx]        += eff_len
                    bin_target_totals[bin_idx] += tgt_len
                    group_assignments[chunk_key] = bin_idx

                    new_remaining = target_length - bin_totals[bin_idx]
                    if new_remaining > 0:
                        heapq.heappush(target_heap, (-new_remaining, bin_idx))
                    else:
                        hard_remaining = hard_limit - bin_totals[bin_idx]
                        if hard_remaining > 0:
                            heapq.heappush(overflow_heap, (-hard_remaining, bin_idx))
                    placed = True
                else:
                    hard_remaining = hard_limit - bin_totals[bin_idx]
                    if hard_remaining > 0:
                        heapq.heappush(overflow_heap, (-hard_remaining, bin_idx))

            while overflow_heap and not placed:
                neg_remaining, bin_idx = heapq.heappop(overflow_heap)
                remaining = hard_limit - bin_totals[bin_idx]

                if remaining >= eff_len:
                    bins[bin_idx].append(chunk_key)
                    bin_totals[bin_idx]        += eff_len
                    bin_target_totals[bin_idx] += tgt_len
                    group_assignments[chunk_key] = bin_idx

                    new_remaining = hard_limit - bin_totals[bin_idx]
                    if new_remaining > 0:
                        heapq.heappush(overflow_heap, (-new_remaining, bin_idx))
                    placed = True

        else:
            # Dual constraint: vectorized numpy search
            if n_bins > 0:
                # Find bins where both constraints fit (vectorized)
                input_fits  = (hard_limit - np_input_total[:n_bins]) >= eff_len
                target_fits = (target_hard_limit - np_tgt_total[:n_bins]) >= tgt_len
                candidates  = np.where(np_open[:n_bins] & input_fits & target_fits)[0]

                if len(candidates) > 0:
                    # Best-fit: pick bin with least remaining input space
                    best_idx = candidates[np.argmax(np_input_total[candidates])]
                    bin_idx  = int(best_idx)

                    bins[bin_idx].append(chunk_key)
                    bin_totals[bin_idx]        += eff_len
                    bin_target_totals[bin_idx] += tgt_len
                    np_input_total[bin_idx]    += eff_len
                    np_tgt_total[bin_idx]      += tgt_len
                    group_assignments[chunk_key] = bin_idx

                    # Mark closed if nearly full
                    if (hard_limit - np_input_total[bin_idx] < 500 or
                        target_hard_limit - np_tgt_total[bin_idx] < 500):
                        np_open[bin_idx] = False

                    placed = True

        # Create new bin
        if not placed:
            bin_idx = len(bins)
            bins.append([chunk_key])
            bin_totals.append(eff_len)
            bin_target_totals.append(tgt_len)
            group_assignments[chunk_key] = bin_idx

            if not use_dual:
                remaining = target_length - eff_len
                if remaining > 0:
                    heapq.heappush(target_heap, (-remaining, bin_idx))
                else:
                    hard_remaining = hard_limit - eff_len
                    if hard_remaining > 0:
                        heapq.heappush(overflow_heap, (-hard_remaining, bin_idx))
            else:
                if n_bins >= len(np_input_total):
                    new_size       = len(np_input_total) * 2
                    new_input      = np.zeros(new_size, dtype=np.int32)
                    new_tgt        = np.zeros(new_size, dtype=np.int32)
                    new_open       = np.ones(new_size, dtype=np.bool_)
                    new_input[:n_bins] = np_input_total[:n_bins]
                    new_tgt[:n_bins]   = np_tgt_total[:n_bins]
                    new_open[:n_bins]  = np_open[:n_bins]
                    np_input_total, np_tgt_total, np_open = new_input, new_tgt, new_open
                np_input_total[n_bins] = eff_len
                np_tgt_total[n_bins]   = tgt_len
                np_open[n_bins]        = True
                n_bins += 1

    print(f"    Packing: {total:,}/{total:,} (100.0%) - {len(bins):,} bins")

    # Update stats
    for bin_idx, bin_keys in enumerate(bins):
        total_len   = bin_totals[bin_idx]
        utilization = total_len / target_length if target_length > 0 else 0
        stats["utilizations"].append(utilization)
        stats["total_groups"] += 1
        if len(bin_keys) == 1:
            stats["singleton_count"] += 1

        if max_target_len is not None:
            tgt_total = bin_target_totals[bin_idx]
            tgt_util  = tgt_total / max_target_len if max_target_len > 0 else 0
            stats["target_utilizations"].append(tgt_util)

    # Compute summary stats
    if stats["utilizations"]:
        stats["avg_utilization"] = sum(stats["utilizations"]) / len(stats["utilizations"])
        stats["min_utilization"] = min(stats["utilizations"])
        stats["max_utilization"] = max(stats["utilizations"])
    else:
        stats["avg_utilization"] = 0
        stats["min_utilization"] = 0
        stats["max_utilization"] = 0

    if stats["target_utilizations"]:
        stats["avg_target_utilization"] = sum(stats["target_utilizations"]) / len(stats["target_utilizations"])
        stats["min_target_utilization"] = min(stats["target_utilizations"])
        stats["max_target_utilization"] = max(stats["target_utilizations"])

    return group_assignments, stats


def stream_write_compacted(
    file_paths,
    group_assignments: Dict[Tuple[int, int], int],
    output_path,
    show_progress=True,
):
    """
    Phase 3: Re-read chunks and write to output with group assignments

    Memory: O(file_size) - only one input file loaded at a time
    """

    import pathlib
    import struct
    from lib.dataset._binary import read_binary, MAGIC_HEADER, FORMAT_VERSION, OFFSET_ENTRY_SIZE_V2

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = len(group_assignments)

    print(f"    Writing {total_chunks:,} chunks to {output_path.name}...")

    with open(output_path, 'wb') as f:
        f.write(MAGIC_HEADER)
        f.write(struct.pack('<B', FORMAT_VERSION))
        f.write(struct.pack('<B', 0))
        f.write(struct.pack('<I', total_chunks))

        offset_table_pos = f.tell()
        f.write(b'\x00' * (total_chunks * OFFSET_ENTRY_SIZE_V2))

        chunk_order = sorted(group_assignments.keys())
        offsets     = []
        written     = 0

        current_file_idx = -1
        current_chunks   = None

        for file_idx, chunk_idx in chunk_order:
            if file_idx != current_file_idx:
                if current_chunks is not None:
                    del current_chunks
                print(f"    Loading file {file_idx + 1}/{len(file_paths)}...", end='\r')
                current_chunks   = read_binary(file_paths[file_idx])
                current_file_idx = file_idx

            chunk              = current_chunks[chunk_idx]
            chunk.compact_group = group_assignments[(file_idx, chunk_idx)]

            current_offset = f.tell()
            chunk_bytes    = chunk.to_bytes()

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


def build_loss_labels(input_ids, segment_ids):
    """
    Build labels for loss computation with padding/separator masked

    Positions with segment_id == -1 (padding, separator) get label -100,
    which PyTorch CrossEntropyLoss ignores during gradient computation.
    """

    labels = list(input_ids)

    for pos, seg_id in enumerate(segment_ids):
        if seg_id == -1:
            labels[pos] = -100

    return labels
