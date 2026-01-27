import random
import math


#######################
#####  Utilities  #####
#######################


def estimate_chunk_tokens(chunk, tokenizer=None):
    """Get actual token count for a chunk"""

    if tokenizer is not None:
        input_text = chunk.get_input_text()
        input_ids  = tokenizer.encode(input_text, add_special_tokens=False)
        return len(input_ids)

    seq_tokens  = len(chunk.sequence) / 4.5
    hint_tokens = len(chunk.hints) * 5 if chunk.has_hints else 0
    overhead    = 5

    return int(seq_tokens + hint_tokens + overhead)


def align_to_block(length, block_size=64):
    """Align length to next block boundary"""

    return math.ceil(length / block_size) * block_size


def compute_effective_length(raw_length, block_size=64, window_size=256):
    """Compute effective length including isolation padding"""

    aligned   = align_to_block(raw_length, block_size)
    isolation = window_size + block_size
    return aligned + isolation


#########################
#####  Bin Packing  #####
#########################


def compact_chunks(chunks, target_length, hard_limit=None, tokenizer=None, seed=42, block_size=64, window_size=256):
    """Compact chunks using first-fit-decreasing with window-aware isolation"""

    random.seed(seed)

    hard_limit    = hard_limit or int(target_length * 1.1)
    isolation_pad = window_size + block_size

    if tokenizer is None:
        print("  WARNING: No tokenizer provided, using rough estimates")

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

        with_lengths = []
        for c in chunk_list:
            raw_len = estimate_chunk_tokens(c, tokenizer)
            eff_len = align_to_block(raw_len, block_size) + isolation_pad
            with_lengths.append((c, raw_len, eff_len))

        with_lengths.sort(key=lambda x: -x[2])

        bins           = []
        bin_totals     = []
        bin_raw_totals = []

        for chunk, raw_len, eff_len in with_lengths:
            if eff_len > hard_limit:
                bins.append([chunk])
                bin_totals.append(raw_len)
                bin_raw_totals.append(raw_len)
                compact_stats["overflow_count"] += 1
                continue

            best_bin   = -1
            best_space = hard_limit + 1

            for i, total in enumerate(bin_totals):
                remaining = target_length - total
                if remaining >= eff_len and remaining < best_space:
                    best_bin   = i
                    best_space = remaining

            if best_bin == -1:
                for i, total in enumerate(bin_totals):
                    remaining = hard_limit - total
                    if remaining >= eff_len and remaining < best_space:
                        best_bin   = i
                        best_space = remaining

            if best_bin >= 0:
                bins[best_bin].append(chunk)
                bin_totals[best_bin]     += eff_len
                bin_raw_totals[best_bin] += raw_len
            else:
                bins.append([chunk])
                bin_totals.append(eff_len)
                bin_raw_totals.append(raw_len)

        for i, group in enumerate(bins):
            all_compacted.append(group)

            total_len   = bin_totals[i]
            utilization = total_len / target_length if target_length > 0 else 0

            compact_stats["total_groups"]     += 1
            compact_stats["total_input_toks"] += bin_raw_totals[i]
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


def flatten_groups(compacted_groups):
    """Flatten compacted groups back to chunk list with group markers"""

    flattened = []

    for group_idx, group in enumerate(compacted_groups):
        for chunk in group:
            chunk.compact_group = group_idx
            flattened.append(chunk)

    return flattened


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