import random
import multiprocessing
from  concurrent.futures import ProcessPoolExecutor, as_completed

import lib.dataset._binary as binary
import lib.nosing.nosing   as nosing


#######################
#####  Utilities  #####
#######################


def check_cut_inside_gene(gene_index, seqid, cut_pos):
    """Check if cut position falls inside a gene body"""

    for gene_id, gene_data in gene_index.items():
        if gene_data["seqid"] != seqid:
            continue

        g_start = gene_data["start"]
        g_end   = gene_data["end"]

        if g_start < cut_pos < g_end:
            return True, gene_id

    return False, None


def find_genes_in_range(gene_index, seqid, start, end):
    """Find all genes overlapping a genomic range"""

    genes = []

    for gene_id, gene_data in gene_index.items():
        if gene_data["seqid"] != seqid:
            continue

        g_start = gene_data["start"]
        g_end   = gene_data["end"]

        if g_start <= end and g_end >= start:
            genes.append((gene_id, gene_data))

    return genes


#################################
#####  Single Chromosome    #####
#################################


def _chunk_single_chromosome(args):
    """Process a single chromosome - designed for parallel execution"""

    seqid, sequence, chr_gene_index, limit_bp, overlap_bp, anchor_pad = args

    chunks    = []
    step_size = limit_bp - overlap_bp
    seq_len   = len(sequence)

    stats = {
        "total_chunks":    0,
        "backtrack_count": 0,
        "genes_per_chunk": [],
        "chunk_sizes":     [],
    }

    seqid_genes = [
        (gid, gdata) for gid, gdata in chr_gene_index.items()
        if gdata["seqid"] == seqid
    ]

    if not seqid_genes:
        return chunks, stats

    seqid_genes.sort(key=lambda x: x[1]["start"])

    first_gene_start = seqid_genes[0][1]["start"]
    window_start     = max(0, first_gene_start - anchor_pad)
    chunk_index      = 0

    while window_start < seq_len:
        window_end  = min(window_start + limit_bp, seq_len)
        cut_pos     = window_start + step_size
        backtracked = False

        if cut_pos < seq_len:
            is_inside, blocking_gene = check_cut_inside_gene(chr_gene_index, seqid, cut_pos)

            if is_inside:
                new_cut      = cut_pos - overlap_bp
                still_inside = check_cut_inside_gene(chr_gene_index, seqid, new_cut)[0]

                if not still_inside and new_cut > window_start:
                    cut_pos     = new_cut
                    window_end  = cut_pos
                    backtracked = True
                    stats["backtrack_count"] += 1

        genes_in_chunk = find_genes_in_range(chr_gene_index, seqid, window_start, window_end)
        chunk_seq      = sequence[window_start:window_end]
        chunk_features = []
        gene_ids       = []

        for gene_id, gene_data in genes_in_chunk:
            gene_ids.append(gene_id)
            transcripts = gene_data.get("transcripts", {})

            for feat in gene_data.get("features", []):
                adj_start = feat["start"] - window_start
                adj_end   = feat["end"] - window_start

                if adj_start < 0 or adj_end > (window_end - window_start):
                    continue

                transcript_id = feat.get("attributes", {}).get("Parent", "")

                biotype = "."
                if transcript_id and transcript_id in transcripts:
                    biotype = transcripts[transcript_id].get("biotype", ".")

                chunk_features.append({
                    "type":          feat["type"].lower(),
                    "start":         adj_start,
                    "end":           adj_end,
                    "strand":        feat["strand"],
                    "phase":         feat.get("phase", "."),
                    "gene_id":       gene_id,
                    "transcript_id": transcript_id,
                    "biotype":       biotype,
                })

        biotypes = []
        for gene_id, gene_data in genes_in_chunk:
            for t_id, t_data in gene_data.get("transcripts", {}).items():
                bt = t_data.get("biotype", "")
                if bt:
                    biotypes.append(bt)

        primary_biotype = biotypes[0] if biotypes else "."

        chunk = binary.BinaryChunk(
            seqid        = seqid,
            start        = window_start,
            end          = window_end,
            strand       = "+",
            sequence     = chunk_seq,
            features     = chunk_features,
            biotype      = primary_biotype,
            gene_ids     = gene_ids,
            has_hints    = False,
            hints        = [],
            chunk_index  = chunk_index,
            is_augmented = False,
        )

        chunks.append(chunk)

        stats["total_chunks"] += 1
        stats["genes_per_chunk"].append(len(gene_ids))
        stats["chunk_sizes"].append(window_end - window_start)

        chunk_index += 1

        if backtracked:
            window_start = cut_pos
        else:
            window_start += step_size

        if window_start >= seq_len:
            break

    return chunks, stats


#################################
#####  Parallel Chunking    #####
#################################


def dynamic_chunking(sequences, gene_index, limit_bp=25000, overlap_bp=5000, anchor_pad=5000, n_workers=None):
    """Gene-centric sliding window chunking with parallel processing per chromosome"""

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    # Build per-chromosome gene indices
    chr_gene_indices = {}
    for gene_id, gene_data in gene_index.items():
        seqid = gene_data["seqid"]
        if seqid not in chr_gene_indices:
            chr_gene_indices[seqid] = {}
        chr_gene_indices[seqid][gene_id] = gene_data

    # Prepare arguments for parallel execution
    work_items = []
    for seqid, sequence in sequences.items():
        chr_genes = chr_gene_indices.get(seqid, {})
        if chr_genes:
            work_items.append((seqid, sequence, chr_genes, limit_bp, overlap_bp, anchor_pad))

    all_chunks = []
    combined_stats = {
        "total_chunks":    0,
        "backtrack_count": 0,
        "genes_per_chunk": [],
        "chunk_sizes":     [],
    }

    # Use single process for small datasets, parallel for larger
    if len(work_items) <= 2 or n_workers == 1:
        for args in work_items:
            chunks, stats = _chunk_single_chromosome(args)
            all_chunks.extend(chunks)
            combined_stats["total_chunks"]    += stats["total_chunks"]
            combined_stats["backtrack_count"] += stats["backtrack_count"]
            combined_stats["genes_per_chunk"].extend(stats["genes_per_chunk"])
            combined_stats["chunk_sizes"].extend(stats["chunk_sizes"])
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_chunk_single_chromosome, args): args[0] for args in work_items}

            for future in as_completed(futures):
                seqid = futures[future]
                try:
                    chunks, stats = future.result()
                    all_chunks.extend(chunks)
                    combined_stats["total_chunks"]    += stats["total_chunks"]
                    combined_stats["backtrack_count"] += stats["backtrack_count"]
                    combined_stats["genes_per_chunk"].extend(stats["genes_per_chunk"])
                    combined_stats["chunk_sizes"].extend(stats["chunk_sizes"])
                except Exception as e:
                    print(f"  WARNING: Failed to process {seqid}: {e}")

    # Sort by seqid and start position for deterministic order
    all_chunks.sort(key=lambda c: (c.seqid, c.start))

    # Re-index chunks globally
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i

    return all_chunks, combined_stats


###################
#####  Hints  #####
###################


def _augment_chunk_batch(args):
    """Augment a batch of chunks with hints - for parallel execution"""

    chunk_batch, seed_offset = args

    noiser    = nosing.GFFNoiser()
    augmented = []

    for i, chunk in enumerate(chunk_batch):
        random.seed(seed_offset + i)
        hints, scenario, _ = noiser.noise_features(chunk.features, chunk.sequence)

        has_hints = len(hints) > 0

        aug_chunk = binary.BinaryChunk(
            seqid        = chunk.seqid,
            start        = chunk.start,
            end          = chunk.end,
            strand       = chunk.strand,
            sequence     = chunk.sequence,
            features     = chunk.features,
            biotype      = chunk.biotype,
            gene_ids     = chunk.gene_ids,
            has_hints    = has_hints,
            hints        = hints,
            chunk_index  = chunk.chunk_index,
            is_augmented = True,
        )

        augmented.append(aug_chunk)

    return augmented


def augment_with_hints(chunks, hint_ratio=0.5, seed=42, n_workers=None, batch_size=100):
    """Create augmented copies with hints using parallel processing"""

    random.seed(seed)

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    num_to_augment = int(len(chunks) * hint_ratio)
    indices        = random.sample(range(len(chunks)), min(num_to_augment, len(chunks)))
    selected       = [chunks[i] for i in indices]

    # Batch chunks for parallel processing
    batches = []
    for i in range(0, len(selected), batch_size):
        batch       = selected[i:i + batch_size]
        seed_offset = seed + i
        batches.append((batch, seed_offset))

    all_augmented = []

    if len(batches) <= 2 or n_workers == 1:
        for args in batches:
            augmented = _augment_chunk_batch(args)
            all_augmented.extend(augmented)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_augment_chunk_batch, args) for args in batches]

            for future in as_completed(futures):
                try:
                    augmented = future.result()
                    all_augmented.extend(augmented)
                except Exception as e:
                    print(f"  WARNING: Augmentation batch failed: {e}")

    return chunks + all_augmented