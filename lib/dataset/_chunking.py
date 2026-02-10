import random
import math
import bisect
import multiprocessing
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed

import lib.dataset._binary as binary
import lib.nosing.nosing   as nosing


########################################
#####  Gene Index Partitioning     #####
########################################


def partition_genes_by_seqid(gene_index):
    """
    Partition gene_index by chromosome/seqid

    Returns dict: {seqid: {gene_id: gene_data, ...}, ...}
    """

    partitioned = {}

    for gene_id, gene_data in gene_index.items():
        seqid = gene_data["seqid"]

        if seqid not in partitioned:
            partitioned[seqid] = {}

        partitioned[seqid][gene_id] = gene_data

    return partitioned


########################################
#####  Interval Index (O(log G))   #####
########################################


class GeneIntervalIndex:
    """
    Binary-search based gene interval index

    Enables O(log G) lookups instead of O(G) scans
    """

    def __init__(self, chr_gene_index):
        """Build sorted index from chromosome-partitioned genes"""

        self.genes       = []
        self.starts      = []
        self.ends        = []
        self.gene_id_map = {}

        for gene_id, gene_data in chr_gene_index.items():
            self.genes.append((gene_id, gene_data))

        self.genes.sort(key=lambda x: x[1]["start"])

        for i, (gene_id, gene_data) in enumerate(self.genes):
            self.starts.append(gene_data["start"])
            self.ends.append(gene_data["end"])
            self.gene_id_map[gene_id] = i

    def find_genes_in_range(self, start, end):
        """Find genes overlapping [start, end] - O(log G + k) where k = matches"""

        if not self.genes:
            return []

        # Find first gene that could overlap (gene.end >= start)
        # Binary search for leftmost gene whose end >= start
        left = bisect.bisect_left(self.ends, start)

        results = []
        for i in range(left, len(self.genes)):
            gene_id, gene_data = self.genes[i]
            g_start            = gene_data["start"]
            g_end              = gene_data["end"]

            # Gene starts after our range ends - done
            if g_start > end:
                break

            # Gene overlaps our range
            if g_end >= start:
                results.append((gene_id, gene_data))

        return results

    def check_position_inside_gene(self, pos):
        """Check if position is inside any gene body - O(log G)"""

        if not self.genes:
            return False, None

        # Find first gene that could contain pos (gene.end >= pos)
        idx = bisect.bisect_left(self.ends, pos)

        for i in range(idx, len(self.genes)):
            gene_id, gene_data = self.genes[i]
            g_start            = gene_data["start"]
            g_end              = gene_data["end"]

            if g_start > pos:
                break

            if g_start <= pos <= g_end:
                return True, gene_id

        return False, None

    def get_first_gene_start(self):
        """Get start position of first gene"""

        if not self.genes:
            return 0
        return self.genes[0][1]["start"]

    def get_last_gene_end(self):
        """Get end position of last gene"""

        if not self.genes:
            return 0
        return max(self.ends) if self.ends else 0


########################################
#####  Feature Index Builder       #####
########################################


def build_feature_list(chr_gene_index):
    """
    Build sorted feature list from chromosome genes

    Returns list of feature dicts sorted by start position
    """

    features = []

    for gene_id, gene_data in chr_gene_index.items():
        transcripts = gene_data.get("transcripts", {})

        for feat in gene_data.get("features", []):
            transcript_id = feat.get("attributes", {}).get("Parent", "")

            biotype = "."
            if transcript_id and transcript_id in transcripts:
                biotype = transcripts[transcript_id].get("biotype", ".")

            features.append({
                "start":         feat["start"],
                "end":           feat["end"],
                "type":          feat["type"].lower(),
                "strand":        feat["strand"],
                "phase":         feat.get("phase", "."),
                "cds_start":     feat.get("cds_start"),
                "cds_end":       feat.get("cds_end"),
                "gene_id":       gene_id,
                "transcript_id": transcript_id,
                "biotype":       biotype,
            })

    features.sort(key=lambda f: f["start"])
    return features


########################################
#####  Gene-Centric Chunking       #####
########################################


def _chunk_chromosome_genecentric(args):
    """
    Gene-centric chunking with O(log G) operations

    Slides window across gene regions only, uses binary search for lookups
    """

    seqid, sequence, chr_gene_index, window_size, step_bp, anchor_pad = args

    seq_len = len(sequence)
    chunks  = []

    stats = {
        "windows_scanned":    0,
        "windows_empty":      0,
        "windows_kept":       0,
        "backtrack_count":    0,
        "features_per_chunk": [],
    }

    if not chr_gene_index:
        return chunks, stats

    # Build interval index - O(G log G) once
    gene_idx     = GeneIntervalIndex(chr_gene_index)
    all_features = build_feature_list(chr_gene_index)

    if not all_features:
        return chunks, stats

    # Build feature starts array for binary search
    feat_starts = [f["start"] for f in all_features]
    feat_ends   = [f["end"] for f in all_features]

    # Start at first gene with padding
    first_gene_start = gene_idx.get_first_gene_start()
    last_gene_end    = gene_idx.get_last_gene_end()
    window_start     = max(0, first_gene_start - anchor_pad)
    step_size        = step_bp

    chunk_index = 0

    while window_start < seq_len and window_start < last_gene_end + anchor_pad:
        window_end = min(window_start + window_size, seq_len)
        stats["windows_scanned"] += 1

        # Check if window cut would split a gene - O(log G)
        cut_pos = window_start + step_size
        if cut_pos < seq_len:
            is_inside, blocking_gene = gene_idx.check_position_inside_gene(cut_pos)

            if is_inside and blocking_gene:
                # Backtrack to just before the blocking gene's start
                gene_start = chr_gene_index[blocking_gene]["start"]
                new_cut    = max(window_start + 1, gene_start - 1)
                still_inside = gene_idx.check_position_inside_gene(new_cut)[0]

                if not still_inside and new_cut > window_start:
                    window_end = new_cut
                    stats["backtrack_count"] += 1

        # Find features in window using binary search - O(log F + k)
        left_idx  = bisect.bisect_left(feat_ends, window_start)
        chunk_features = []
        gene_ids       = set()
        biotypes       = []

        for i in range(left_idx, len(all_features)):
            feat = all_features[i]

            if feat["start"] > window_end:
                break

            # Feature must be COMPLETELY inside window
            if feat["start"] >= window_start and feat["end"] <= window_end:
                adj_feat = {
                    "type":          feat["type"],
                    "start":         feat["start"] - window_start,
                    "end":           feat["end"] - window_start,
                    "strand":        feat["strand"],
                    "phase":         feat["phase"],
                    "gene_id":       feat["gene_id"],
                    "transcript_id": feat["transcript_id"],
                    "biotype":       feat["biotype"],
                }

                # Adjust CDS coordinates if present
                if feat.get("cds_start") is not None:
                    adj_feat["cds_start"] = feat["cds_start"] - window_start
                if feat.get("cds_end") is not None:
                    adj_feat["cds_end"] = feat["cds_end"] - window_start

                chunk_features.append(adj_feat)
                gene_ids.add(feat["gene_id"])

                if feat["biotype"] != ".":
                    biotypes.append(feat["biotype"])

        if not chunk_features:
            stats["windows_empty"] += 1
            window_start += step_size
            continue

        chunk_seq       = sequence[window_start:window_end]
        primary_biotype = biotypes[0] if biotypes else "."

        chunk = binary.BinaryChunk(
            seqid        = seqid,
            start        = window_start,
            end          = window_end,
            strand       = "+",
            sequence     = chunk_seq,
            features     = chunk_features,
            biotype      = primary_biotype,
            gene_ids     = list(gene_ids),
            has_hints    = False,
            hints        = [],
            chunk_index  = chunk_index,
            is_augmented = False,
        )

        chunks.append(chunk)
        stats["windows_kept"] += 1
        stats["features_per_chunk"].append(len(chunk_features))
        chunk_index += 1

        window_start += step_size

    return chunks, stats


########################################
#####  Sliding Window Chunking     #####
########################################


def sliding_window_chunking(sequences, gene_index, window_size=20000, overlap_ratio=None, max_n_ratio=0.1, n_workers=None):
    """
    Gene-centric sliding window chunking

    Pre-partitions genes by chromosome, uses O(log G) binary search for lookups.
    Only processes regions around genes, not the entire genome.
    """

    if overlap_ratio is None:
        overlap_ratio = 1 / math.e

    # overlap_ratio is the STEP ratio: fraction of window that's new each slide
    # step = W/e, overlap = W(1-1/e), so 2 adjacent windows cover W(1+1/e) unique bp
    step_bp    = int(window_size * overlap_ratio)
    anchor_pad = window_size // 5

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    # Pre-partition genes by chromosome - O(G) once
    genes_by_seqid = partition_genes_by_seqid(gene_index)

    all_chunks     = []
    combined_stats = {
        "windows_scanned":    0,
        "windows_empty":      0,
        "windows_n_heavy":    0,
        "windows_kept":       0,
        "backtrack_count":    0,
        "features_per_chunk": [],
    }

    # Only process chromosomes with genes
    work_seqids = [seqid for seqid in sequences.keys() if seqid in genes_by_seqid]

    # Build work items with pre-partitioned gene indices
    work_items = []
    for seqid in work_seqids:
        sequence       = sequences[seqid]
        chr_gene_index = genes_by_seqid[seqid]
        work_items.append((seqid, sequence, chr_gene_index, window_size, step_bp, anchor_pad))

    if len(work_items) <= 2 or n_workers == 1:
        for args in work_items:
            chunks, stats = _chunk_chromosome_genecentric(args)
            all_chunks.extend(chunks)
            _merge_stats(combined_stats, stats)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_chunk_chromosome_genecentric, args): args[0] for args in work_items}

            for future in as_completed(futures):
                seqid = futures[future]
                try:
                    chunks, stats = future.result()
                    all_chunks.extend(chunks)
                    _merge_stats(combined_stats, stats)
                except Exception as e:
                    print(f"  WARNING: Failed to process {seqid}: {e}")

    # Sort and re-index
    all_chunks.sort(key=lambda c: (c.seqid, c.start))
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i

    return all_chunks, combined_stats


def _merge_stats(combined, new):
    """Merge stats from worker into combined stats"""

    combined["windows_scanned"] += new.get("windows_scanned", 0)
    combined["windows_empty"]   += new.get("windows_empty", 0)
    combined["windows_kept"]    += new.get("windows_kept", 0)
    combined["backtrack_count"] += new.get("backtrack_count", 0)
    combined["features_per_chunk"].extend(new.get("features_per_chunk", []))


########################################
#####  Legacy Compatibility        #####
########################################


def dynamic_chunking(sequences, gene_index, limit_bp=25000, overlap_bp=5000, anchor_pad=5000, n_workers=None):
    """Legacy wrapper - redirects to sliding_window_chunking"""

    overlap_ratio = overlap_bp / limit_bp if limit_bp > 0 else 1 / math.e

    return sliding_window_chunking(
        sequences,
        gene_index,
        window_size   = limit_bp,
        overlap_ratio = overlap_ratio,
        max_n_ratio   = 0.1,
        n_workers     = n_workers,
    )


############################
#####  Quality Filter  #####
############################


def filter_n_heavy_chunks(chunks, max_n_ratio=0.1, require_features=True):
    """Filter out chunks with high N content or no features"""

    filtered = []
    stats    = {
        "total_input":      len(chunks),
        "n_heavy_count":    0,
        "empty_seq_count":  0,
        "zero_feat_count":  0,
        "passed_count":     0,
        "n_ratios":         [],
    }

    for chunk in chunks:
        seq     = chunk.sequence
        seq_len = len(seq)

        if seq_len == 0:
            stats["empty_seq_count"] += 1
            continue

        n_count = seq.count('N') + seq.count('n')
        n_ratio = n_count / seq_len

        stats["n_ratios"].append(n_ratio)

        if n_ratio > max_n_ratio:
            stats["n_heavy_count"] += 1
            continue

        if require_features and len(chunk.features) == 0:
            stats["zero_feat_count"] += 1
            continue

        filtered.append(chunk)
        stats["passed_count"] += 1

    return filtered, stats


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
    del indices  # Free memory

    batches = []
    for i in range(0, len(selected), batch_size):
        batch       = selected[i:i + batch_size]
        seed_offset = seed + i
        batches.append((batch, seed_offset))

    del selected  # Free memory

    if len(batches) <= 2 or n_workers == 1:
        for args in batches:
            augmented = _augment_chunk_batch(args)
            chunks.extend(augmented)  # Extend in place, no copy
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_augment_chunk_batch, args) for args in batches]

            for future in as_completed(futures):
                try:
                    augmented = future.result()
                    chunks.extend(augmented)  # Extend in place
                except Exception as e:
                    print(f"  WARNING: Augmentation batch failed: {e}")

    return chunks
