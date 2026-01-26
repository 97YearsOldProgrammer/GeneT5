import random


#######################
#####  Utilities  #####
#######################


def estimate_chunk_tokens(chunk, tokenizer=None):
    """Get actual token count for a chunk"""

    if tokenizer is not None:
        input_text = chunk.get_input_text()
        input_ids  = tokenizer.encode(input_text, add_special_tokens=False)
        return len(input_ids)

    # Fallback: rough estimate
    seq_tokens  = len(chunk.sequence) / 4.5
    hint_tokens = len(chunk.hints) * 5 if chunk.has_hints else 0
    overhead    = 5

    return int(seq_tokens + hint_tokens + overhead)


def compact_chunks(chunks, target_length, hard_limit=None, tokenizer=None, seed=42):
    """Compact chunks using first-fit-decreasing bin packing"""

    random.seed(seed)

    hard_limit = hard_limit or int(target_length * 1.1)

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

        with_lengths = [
            (c, estimate_chunk_tokens(c, tokenizer))
            for c in chunk_list
        ]
        with_lengths.sort(key=lambda x: -x[1])

        bins       = []
        bin_totals = []

        for chunk, length in with_lengths:
            effective_length = length + 1

            if effective_length > hard_limit:
                bins.append([chunk])
                bin_totals.append(length)
                compact_stats["overflow_count"] += 1
                continue

            best_bin   = -1
            best_space = hard_limit + 1

            for i, total in enumerate(bin_totals):
                remaining = target_length - total
                if remaining >= effective_length and remaining < best_space:
                    best_bin   = i
                    best_space = remaining

            if best_bin == -1:
                for i, total in enumerate(bin_totals):
                    remaining = hard_limit - total
                    if remaining >= effective_length and remaining < best_space:
                        best_bin   = i
                        best_space = remaining

            if best_bin >= 0:
                bins[best_bin].append(chunk)
                bin_totals[best_bin] += effective_length
            else:
                bins.append([chunk])
                bin_totals.append(length)

        for i, group in enumerate(bins):
            all_compacted.append(group)

            total_len   = bin_totals[i]
            utilization = total_len / target_length if target_length > 0 else 0

            compact_stats["total_groups"]     += 1
            compact_stats["total_input_toks"] += total_len
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