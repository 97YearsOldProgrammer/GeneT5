import random

import lib.dataset._binary as binary


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


def dynamic_chunking(sequences, gene_index, limit_bp=25000, overlap_bp=5000, anchor_pad=5000):
    """Gene-centric sliding window chunking"""

    chunks    = []
    step_size = limit_bp - overlap_bp

    stats = {
        "total_chunks":    0,
        "backtrack_count": 0,
        "genes_per_chunk": [],
        "chunk_sizes":     [],
    }

    for seqid, sequence in sequences.items():
        seq_len = len(sequence)

        seqid_genes = [
            (gid, gdata) for gid, gdata in gene_index.items()
            if gdata["seqid"] == seqid
        ]

        if not seqid_genes:
            continue

        seqid_genes.sort(key=lambda x: x[1]["start"])

        first_gene_start = seqid_genes[0][1]["start"]
        window_start     = max(0, first_gene_start - anchor_pad)
        chunk_index      = 0

        while window_start < seq_len:
            window_end  = min(window_start + limit_bp, seq_len)
            cut_pos     = window_start + step_size
            backtracked = False

            if cut_pos < seq_len:
                is_inside, blocking_gene = check_cut_inside_gene(gene_index, seqid, cut_pos)

                if is_inside:
                    new_cut      = cut_pos - overlap_bp
                    still_inside = check_cut_inside_gene(gene_index, seqid, new_cut)[0]

                    if not still_inside and new_cut > window_start:
                        cut_pos     = new_cut
                        window_end  = cut_pos
                        backtracked = True
                        stats["backtrack_count"] += 1

            genes_in_chunk = find_genes_in_range(gene_index, seqid, window_start, window_end)
            chunk_seq      = sequence[window_start:window_end]
            chunk_features = []
            gene_ids       = []

            for gene_id, gene_data in genes_in_chunk:
                gene_ids.append(gene_id)

                for feat in gene_data.get("features", []):
                    adj_start = feat["start"] - window_start
                    adj_end   = feat["end"] - window_start

                    if adj_start < 0 or adj_end > (window_end - window_start):
                        continue

                    chunk_features.append({
                        "type":    feat["type"].lower(),
                        "start":   adj_start,
                        "end":     adj_end,
                        "strand":  feat["strand"],
                        "phase":   feat.get("phase", "."),
                        "gene_id": gene_id,
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


###################
#####  Hints  #####
###################


def generate_hints_from_features(features, noise_rate=0.1):
    """Generate noised hints from features"""

    hints = []

    for feat in features:
        if random.random() < noise_rate:
            continue

        jitter_start = int(random.gauss(0, 15))
        jitter_end   = int(random.gauss(0, 15))

        hint = {
            "type":   feat["type"],
            "start":  max(0, feat["start"] + jitter_start),
            "end":    feat["end"] + jitter_end,
            "strand": feat["strand"],
        }
        hints.append(hint)

    if random.random() < 0.05 and features:
        max_pos    = max(f["end"] for f in features)
        fake_start = random.randint(0, max(0, max_pos - 200))
        fake_end   = fake_start + random.randint(50, 200)

        hints.append({
            "type":   "exon",
            "start":  fake_start,
            "end":    fake_end,
            "strand": random.choice(["+", "-"]),
        })

    return hints


def augment_with_hints(chunks, hint_ratio=0.5, seed=42):
    """Create augmented copies with hints"""

    random.seed(seed)

    num_to_augment = int(len(chunks) * hint_ratio)
    indices        = random.sample(range(len(chunks)), min(num_to_augment, len(chunks)))

    augmented = []

    for idx in indices:
        original = chunks[idx]
        hints    = generate_hints_from_features(original.features)

        aug_chunk = binary.BinaryChunk(
            seqid        = original.seqid,
            start        = original.start,
            end          = original.end,
            strand       = original.strand,
            sequence     = original.sequence,
            features     = original.features,
            biotype      = original.biotype,
            gene_ids     = original.gene_ids,
            has_hints    = True,
            hints        = hints,
            chunk_index  = original.chunk_index,
            is_augmented = True,
        )

        augmented.append(aug_chunk)

    return chunks + augmented