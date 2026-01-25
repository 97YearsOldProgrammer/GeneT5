import random


#######################
#####  Utilities  #####
#######################


def estimate_chunk_tokens(chunk, bp_per_token=4.5):
    """Estimate input tokens for a single chunk"""

    seq_tokens  = len(chunk.sequence) / bp_per_token
    hint_tokens = len(chunk.hints) * 5 if chunk.has_hints else 0
    overhead    = 5

    return int(seq_tokens + hint_tokens + overhead)


def compact_chunks(chunks, target_length, hard_limit=None, bp_per_token=4.5, seed=42):
    """Compact chunks using first-fit-decreasing bin packing"""

    random.seed(seed)

    hard_limit = hard_limit or int(target_length * 1.1)

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
            (c, estimate_chunk_tokens(c, bp_per_token))
            for c in chunk_list
        ]
        with_lengths.sort(key=lambda x: -x[1])

        bins       = []
        bin_totals = []

        for chunk, length in with_lengths:
            if length > hard_limit:
                bins.append([chunk])
                bin_totals.append(length)
                compact_stats["overflow_count"] += 1
                continue

            best_bin   = -1
            best_space = hard_limit + 1

            for i, total in enumerate(bin_totals):
                remaining = target_length - total
                if remaining >= length and remaining < best_space:
                    best_bin   = i
                    best_space = remaining

            if best_bin == -1:
                for i, total in enumerate(bin_totals):
                    remaining = hard_limit - total
                    if remaining >= length and remaining < best_space:
                        best_bin   = i
                        best_space = remaining

            if best_bin >= 0:
                bins[best_bin].append(chunk)
                bin_totals[best_bin] += length
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


def estimate_efficiency(chunks, target_length=10000, bp_per_token=4.5):
    """Estimate efficiency of compacting without actually compacting"""

    total_tokens = sum(estimate_chunk_tokens(c, bp_per_token) for c in chunks)

    compacted, stats = compact_chunks(chunks, target_length, bp_per_token=bp_per_token)
    num_groups       = len(compacted)

    return {
        "total_chunks":    len(chunks),
        "total_tokens":    total_tokens,
        "num_groups":      num_groups,
        "avg_group_size":  len(chunks) / num_groups if num_groups > 0 else 0,
        "avg_utilization": stats["avg_utilization"],
        "wasted_tokens":   (num_groups * target_length) - total_tokens,
    }


def estimate_scenario_tokens(scenario, bp_per_token=4.5):
    """Estimate input tokens for a validation scenario"""

    seq_len     = scenario.get("end", 0) - scenario.get("start", 0)
    seq_tokens  = seq_len / bp_per_token
    hint_tokens = len(scenario.get("hints", [])) * 5
    overhead    = 10

    return int(seq_tokens + hint_tokens + overhead)


def compact_scenarios(scenarios, target_length, hard_limit=None, bp_per_token=4.5, seed=42):
    """Compact validation scenarios using first-fit-decreasing bin packing"""

    random.seed(seed)

    hard_limit = hard_limit or int(target_length * 1.1)

    with_lengths = [
        (s, estimate_scenario_tokens(s, bp_per_token))
        for s in scenarios
    ]
    with_lengths.sort(key=lambda x: -x[1])

    bins       = []
    bin_totals = []

    for scenario, length in with_lengths:
        if length > hard_limit:
            bins.append([scenario])
            bin_totals.append(length)
            continue

        best_bin   = -1
        best_space = hard_limit + 1

        for i, total in enumerate(bin_totals):
            remaining = target_length - total
            if remaining >= length and remaining < best_space:
                best_bin   = i
                best_space = remaining

        if best_bin == -1:
            for i, total in enumerate(bin_totals):
                remaining = hard_limit - total
                if remaining >= length and remaining < best_space:
                    best_bin   = i
                    best_space = remaining

        if best_bin >= 0:
            bins[best_bin].append(scenario)
            bin_totals[best_bin] += length
        else:
            bins.append([scenario])
            bin_totals.append(length)

    compact_stats = {
        "num_groups":      len(bins),
        "total_scenarios": len(scenarios),
        "utilizations":    [t / target_length for t in bin_totals],
    }

    if compact_stats["utilizations"]:
        compact_stats["avg_utilization"] = sum(compact_stats["utilizations"]) / len(compact_stats["utilizations"])
    else:
        compact_stats["avg_utilization"] = 0

    return bins, compact_stats