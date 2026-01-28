import random
import math
import heapq

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Any


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


def estimate_chunk_tokens_fallback(chunk):
    """Rough estimate without tokenizer"""
    seq_tokens  = len(chunk.sequence) / 4.5
    hint_tokens = len(chunk.hints) * 5 if chunk.has_hints else 0
    overhead    = 5
    return int(seq_tokens + hint_tokens + overhead)


#################################
#####  Batch Tokenization   #####
#################################


def _tokenize_batch(texts, tokenizer):
    """Tokenize a batch of texts, return list of lengths"""
    encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
    return [len(ids) for ids in encoded['input_ids']]


def estimate_tokens_batch_parallel(
    chunks: List[Any],
    tokenizer,
    batch_size: int = 1000,
    n_workers: int = 4,
) -> List[int]:
    """
    Batch tokenize chunks in parallel using threads.
    HuggingFace tokenizers release GIL so threading works well.
    """
    # Extract all texts
    print(f"    Extracting texts from {len(chunks):,} chunks...")
    texts = [c.get_input_text() for c in chunks]
    
    # Create batches
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append(texts[i:i + batch_size])
    
    print(f"    Tokenizing {len(batches):,} batches with {n_workers} workers...")
    
    all_lengths = []
    completed = 0
    
    def process_batch(batch_texts):
        return _tokenize_batch(batch_texts, tokenizer)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all batches
        futures = [executor.submit(process_batch, batch) for batch in batches]
        
        # Collect results in order
        for i, future in enumerate(futures):
            batch_lengths = future.result()
            all_lengths.extend(batch_lengths)
            completed += len(batch_lengths)
            
            if (i + 1) % 100 == 0 or i == len(futures) - 1:
                pct = 100 * completed / len(texts)
                print(f"    Tokenized: {completed:,}/{len(texts):,} ({pct:.1f}%)", end='\r')
    
    print()  # newline
    return all_lengths


#######################################
#####  Heap-Based Bin Packing     #####
#######################################


def compact_chunks(
    chunks,
    target_length,
    hard_limit=None,
    tokenizer=None,
    seed=42,
    block_size=64,
    window_size=256,
    tokenizer_path=None,  # unused but kept for API compat
    n_workers=4,
    batch_size=1000
):
    """
    Compact chunks using first-fit-decreasing with HEAP-based bin selection.
    
    O(n log n) instead of O(n²) for bin packing.
    """
    random.seed(seed)
    
    hard_limit    = hard_limit or int(target_length * 1.1)
    isolation_pad = window_size + block_size
    
    # Separate raw and augmented
    raw_indices = [i for i, c in enumerate(chunks) if not c.is_augmented]
    aug_indices = [i for i, c in enumerate(chunks) if c.is_augmented]
    
    print(f"    Raw chunks: {len(raw_indices):,}, Augmented: {len(aug_indices):,}")
    
    # Batch tokenize ALL chunks at once
    print(f"\n  Tokenizing {len(chunks):,} chunks...")
    
    if tokenizer is not None:
        all_lengths = estimate_tokens_batch_parallel(
            chunks, 
            tokenizer,
            batch_size=batch_size,
            n_workers=n_workers,
        )
    else:
        print("    WARNING: No tokenizer, using rough estimates")
        all_lengths = [estimate_chunk_tokens_fallback(c) for c in chunks]
    
    # Compute effective lengths (with alignment + isolation)
    effective_lengths = [
        align_to_block(raw_len, block_size) + isolation_pad 
        for raw_len in all_lengths
    ]
    
    # Pack each category
    all_compacted = []
    compact_stats = {
        "total_groups":     0,
        "total_input_toks": 0,
        "utilizations":     [],
        "overflow_count":   0,
        "singleton_count":  0,
    }
    
    for indices, category in [(raw_indices, "raw"), (aug_indices, "augmented")]:
        if not indices:
            continue
        
        print(f"\n  Packing {len(indices):,} {category} chunks (heap-based)...")
        
        # Create (chunk_idx, raw_len, eff_len) tuples
        with_lengths = [
            (idx, all_lengths[idx], effective_lengths[idx])
            for idx in indices
        ]
        
        # Sort by effective length descending (FFD algorithm)
        with_lengths.sort(key=lambda x: -x[2])
        
        # Heap-based bin packing
        # Each bin: [list of chunk indices]
        bins: List[List[int]] = []
        bin_totals: List[int] = []      # effective length totals
        bin_raw_totals: List[int] = []  # raw length totals
        
        # Two heaps: one for bins under target, one for bins between target and hard_limit
        # Heap entries: (negative_remaining, bin_idx) - negative because heapq is min-heap
        target_heap = []   # bins with remaining space under target
        overflow_heap = [] # bins with remaining space only under hard_limit
        
        total = len(with_lengths)
        
        for progress_idx, (chunk_idx, raw_len, eff_len) in enumerate(with_lengths):
            if progress_idx % 50000 == 0:
                pct = 100 * progress_idx / total
                print(f"    Packing: {progress_idx:,}/{total:,} ({pct:.1f}%) - {len(bins):,} bins", end='\r')
            
            # Oversized chunk - must be standalone
            if eff_len > hard_limit:
                bin_idx = len(bins)
                bins.append([chunk_idx])
                bin_totals.append(eff_len)
                bin_raw_totals.append(raw_len)
                compact_stats["overflow_count"] += 1
                continue
            
            placed = False
            
            # Try target heap first (prefer bins close to target)
            while target_heap and not placed:
                neg_remaining, bin_idx = heapq.heappop(target_heap)
                current_total = bin_totals[bin_idx]
                remaining_to_target = target_length - current_total
                
                if remaining_to_target >= eff_len:
                    # Fits under target!
                    bins[bin_idx].append(chunk_idx)
                    bin_totals[bin_idx] += eff_len
                    bin_raw_totals[bin_idx] += raw_len
                    
                    new_remaining = target_length - bin_totals[bin_idx]
                    if new_remaining > 0:
                        heapq.heappush(target_heap, (-new_remaining, bin_idx))
                    else:
                        # Bin is at/over target, move to overflow heap if space left
                        hard_remaining = hard_limit - bin_totals[bin_idx]
                        if hard_remaining > 0:
                            heapq.heappush(overflow_heap, (-hard_remaining, bin_idx))
                    placed = True
                else:
                    # Bin no longer fits this chunk under target
                    # Check if it should go to overflow heap
                    hard_remaining = hard_limit - current_total
                    if hard_remaining > 0:
                        heapq.heappush(overflow_heap, (-hard_remaining, bin_idx))
            
            # Try overflow heap (between target and hard_limit)
            while overflow_heap and not placed:
                neg_remaining, bin_idx = heapq.heappop(overflow_heap)
                current_total = bin_totals[bin_idx]
                remaining_to_hard = hard_limit - current_total
                
                if remaining_to_hard >= eff_len:
                    # Fits under hard limit
                    bins[bin_idx].append(chunk_idx)
                    bin_totals[bin_idx] += eff_len
                    bin_raw_totals[bin_idx] += raw_len
                    
                    new_remaining = hard_limit - bin_totals[bin_idx]
                    if new_remaining > 0:
                        heapq.heappush(overflow_heap, (-new_remaining, bin_idx))
                    placed = True
                # else: bin can't fit this chunk, discard from heap
            
            # Create new bin if couldn't place
            if not placed:
                bin_idx = len(bins)
                bins.append([chunk_idx])
                bin_totals.append(eff_len)
                bin_raw_totals.append(raw_len)
                
                remaining_to_target = target_length - eff_len
                if remaining_to_target > 0:
                    heapq.heappush(target_heap, (-remaining_to_target, bin_idx))
                else:
                    remaining_to_hard = hard_limit - eff_len
                    if remaining_to_hard > 0:
                        heapq.heappush(overflow_heap, (-remaining_to_hard, bin_idx))
        
        print(f"    Packing: {total:,}/{total:,} (100.0%) - {len(bins):,} bins")
        
        # Convert bin indices to actual chunk groups
        for bin_idx, chunk_indices in enumerate(bins):
            group = [chunks[idx] for idx in chunk_indices]
            all_compacted.append(group)
            
            total_len   = bin_totals[bin_idx]
            utilization = total_len / target_length if target_length > 0 else 0
            
            compact_stats["total_groups"]     += 1
            compact_stats["total_input_toks"] += bin_raw_totals[bin_idx]
            compact_stats["utilizations"].append(utilization)
            
            if len(group) == 1:
                compact_stats["singleton_count"] += 1
    
    # Compute summary stats
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