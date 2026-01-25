import random


#######################
#####  Constants  #####
#######################


COMPACT_SEP = "[SEP]"


#######################
#####  Utilities  #####
#######################


def estimate_chunk_tokens(chunk, tokenizer=None):
    """
    Get actual token count for a chunk.
    
    If tokenizer provided: returns exact count
    Otherwise: rough estimate (not recommended for compacting)
    """

    if tokenizer is not None:
        # Build input text same as in dataloader
        input_text = chunk.sequence
        if chunk.has_hints and chunk.hints:
            input_text += "\n[HIT]"
            for h in sorted(chunk.hints, key=lambda x: x.get("start", 0)):
                htype   = h.get("type", "exon").lower()
                hstart  = h.get("start", 0)
                hend    = h.get("end", 0)
                hstrand = h.get("strand", "+")
                input_text += f"\n{htype}\t{hstart}\t{hend}\t{hstrand}"

        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        return len(input_ids)

    # Fallback: rough estimate (discouraged)
    seq_tokens  = len(chunk.sequence) / 4.5
    hint_tokens = len(chunk.hints) * 5 if chunk.has_hints else 0
    overhead    = 5

    return int(seq_tokens + hint_tokens + overhead)


def compact_chunks(chunks, target_length, hard_limit=None, tokenizer=None, seed=42):
    """
    Compact chunks using first-fit-decreasing bin packing.
    
    Args:
        chunks:        List of BinaryChunk objects
        target_length: Target token count per group
        hard_limit:    Maximum token count (default: target * 1.1)
        tokenizer:     Tokenizer for accurate token counting (recommended)
        seed:          Random seed for reproducibility
    
    Returns:
        compacted_groups: List of chunk groups
        compact_stats:    Statistics dict
    
    Note: When using compacted data, add COMPACT_SEP between samples
          and use block-diagonal decoder attention mask.
    """

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

        # Get actual token lengths
        with_lengths = [
            (c, estimate_chunk_tokens(c, tokenizer))
            for c in chunk_list
        ]
        with_lengths.sort(key=lambda x: -x[1])

        bins       = []
        bin_totals = []

        for chunk, length in with_lengths:
            # Account for separator token between samples
            effective_length = length + 1  # +1 for [SEP]

            if effective_length > hard_limit:
                bins.append([chunk])
                bin_totals.append(length)
                compact_stats["overflow_count"] += 1
                continue

            best_bin   = -1
            best_space = hard_limit + 1

            # Try to fit in existing bin (prefer target over hard limit)
            for i, total in enumerate(bin_totals):
                remaining = target_length - total
                if remaining >= effective_length and remaining < best_space:
                    best_bin   = i
                    best_space = remaining

            # If no fit under target, try hard limit
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

        # Record stats
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


def estimate_efficiency(chunks, target_length, tokenizer=None):
    """Estimate efficiency of compacting without actually compacting"""

    total_tokens = sum(estimate_chunk_tokens(c, tokenizer) for c in chunks)

    compacted, stats = compact_chunks(chunks, target_length, tokenizer=tokenizer)
    num_groups       = len(compacted)

    return {
        "total_chunks":    len(chunks),
        "total_tokens":    total_tokens,
        "num_groups":      num_groups,
        "avg_group_size":  len(chunks) / num_groups if num_groups > 0 else 0,
        "avg_utilization": stats["avg_utilization"],
        "wasted_tokens":   (num_groups * target_length) - total_tokens,
    }


###############################
#####  Validation Compact #####
###############################


def estimate_scenario_tokens(scenario, tokenizer=None):
    """Estimate input tokens for a validation scenario"""

    if tokenizer is not None:
        # Build input text
        seq_len = scenario.get("end", 0) - scenario.get("start", 0)
        # For validation, we don't have the actual sequence in scenario
        # Just estimate based on length
        input_text = "N" * seq_len  # Placeholder

        hints = scenario.get("hints", [])
        if hints:
            input_text += "\n[HIT]"
            for h in sorted(hints, key=lambda x: x.get("start", 0)):
                htype   = h.get("type", "exon").lower()
                hstart  = h.get("start", 0)
                hend    = h.get("end", 0)
                hstrand = h.get("strand", "+")
                input_text += f"\n{htype}\t{hstart}\t{hend}\t{hstrand}"

        return len(tokenizer.encode(input_text, add_special_tokens=False))

    # Fallback estimate
    seq_len     = scenario.get("end", 0) - scenario.get("start", 0)
    seq_tokens  = seq_len / 4.5
    hint_tokens = len(scenario.get("hints", [])) * 5
    overhead    = 10

    return int(seq_tokens + hint_tokens + overhead)


def compact_scenarios(scenarios, target_length, hard_limit=None, tokenizer=None, seed=42):
    """Compact validation scenarios using first-fit-decreasing bin packing"""

    random.seed(seed)

    hard_limit = hard_limit or int(target_length * 1.1)

    with_lengths = [
        (s, estimate_scenario_tokens(s, tokenizer))
        for s in scenarios
    ]
    with_lengths.sort(key=lambda x: -x[1])

    bins       = []
    bin_totals = []

    for scenario, length in with_lengths:
        effective_length = length + 1  # +1 for separator

        if effective_length > hard_limit:
            bins.append([scenario])
            bin_totals.append(length)
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
            bins[best_bin].append(scenario)
            bin_totals[best_bin] += effective_length
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