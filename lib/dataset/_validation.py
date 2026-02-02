import json
import math
import random
import pathlib
import statistics

import lib.nosing.nosing as nosing


########################
#####  Complexity  #####
########################


def entropy(probs):
    """Calculate Shannon entropy of probability distribution"""

    if not probs:
        return 0.0

    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h


def calculate_complexity(gene_data):
    """Calculate complexity score using entropy-based approach"""

    transcripts = gene_data.get("transcripts", {})
    features    = gene_data.get("features", [])
    num_trans   = len(transcripts)

    if num_trans == 0:
        return 0.0

    if num_trans == 1:
        num_exons = sum(1 for f in features if f.get("type", "").lower() == "exon")
        return num_exons * 0.1

    weights = []
    total   = 0.0

    for t_id, t_data in transcripts.items():
        t_features = t_data.get("features", [])
        exon_count = sum(1 for f in t_features if f.get("type", "").lower() == "exon")
        cds_count  = sum(1 for f in t_features if f.get("type", "").lower() == "cds")

        score = 1.0 + exon_count * 0.5 + cds_count * 0.3
        w     = 2 ** score
        weights.append(w)
        total += w

    if total == 0:
        return float(num_trans)

    probs      = [w / total for w in weights]
    h          = entropy(probs)
    complexity = h * num_trans

    gene_start = gene_data.get("start", 0)
    gene_end   = gene_data.get("end", 0)
    gene_span  = gene_end - gene_start + 1

    if gene_span > 50000:
        complexity *= 1.5

    return complexity


def compute_all_complexities(gene_groups):
    """Compute complexity scores for all genes (cached for reuse)"""

    return {
        gene_id: calculate_complexity(gene_data)
        for gene_id, gene_data in gene_groups.items()
    }


#######################
#####  Selection  #####
#######################


def identify_complex_loci(gene_groups, complexity_cache, top_k=5):
    """Identify top-K most complex loci using precomputed complexity scores"""

    scored = [
        (gene_id, gene_data, complexity_cache[gene_id])
        for gene_id, gene_data in gene_groups.items()
    ]

    scored.sort(key=lambda x: -x[2])

    return scored[:top_k]


def identify_easy_genes(gene_groups, complexity_cache, num_samples=5, seed=42):
    """Identify easy genes with simple structure"""

    random.seed(seed)

    easy_candidates = []

    for gene_id, gene_data in gene_groups.items():
        transcripts  = gene_data.get("transcripts", {})
        features     = gene_data.get("features", [])
        num_trans    = len(transcripts)
        num_features = len(features)

        gene_start = gene_data.get("start", 0)
        gene_end   = gene_data.get("end", 0)
        gene_len   = gene_end - gene_start + 1

        if num_trans <= 1 and num_features <= 6 and gene_len < 10000:
            complexity = complexity_cache[gene_id]
            easy_candidates.append((gene_id, gene_data, complexity))

    easy_candidates.sort(key=lambda x: x[2])

    if len(easy_candidates) <= num_samples:
        return [(gid, gdata) for gid, gdata, _ in easy_candidates]

    selected = random.sample(easy_candidates[:num_samples * 3], num_samples)
    return [(gid, gdata) for gid, gdata, _ in selected]


def identify_mean_complexity_genes(gene_groups, complexity_cache, exclude_ids, num_samples=5, seed=42):
    """Select genes closest to mean complexity for representative normal samples"""

    random.seed(seed)

    candidates = {
        gid: gdata for gid, gdata in gene_groups.items()
        if gid not in exclude_ids
    }

    if not candidates:
        return []

    candidate_scores = [
        (gid, gdata, complexity_cache[gid])
        for gid, gdata in candidates.items()
    ]

    if len(candidate_scores) < num_samples:
        return [(gid, gdata) for gid, gdata, _ in candidate_scores]

    scores_only = [s for _, _, s in candidate_scores]
    mean_score  = statistics.mean(scores_only)

    # Sort by distance from mean
    candidate_scores.sort(key=lambda x: abs(x[2] - mean_score))

    # Take genes closest to mean with small pool for diversity
    pool_size = min(num_samples * 3, len(candidate_scores))
    pool      = candidate_scores[:pool_size]

    if len(pool) <= num_samples:
        return [(gid, gdata) for gid, gdata, _ in pool]

    selected = random.sample(pool, num_samples)
    return [(gid, gdata) for gid, gdata, _ in selected]


#####################
#####  Scenario #####
#####################


def build_scenarios(gene_id, gene_data, seed=42):
    """Build validation scenarios: 50% ab initio, 50% hinted"""

    random.seed(seed)

    noiser      = nosing.GFFNoiser()
    features    = gene_data.get("features", [])
    transcripts = gene_data.get("transcripts", {})
    scenarios   = []

    enriched_features = []
    for feat in features:
        transcript_id = feat.get("attributes", {}).get("Parent", "")
        biotype       = "."
        if transcript_id and transcript_id in transcripts:
            biotype = transcripts[transcript_id].get("biotype", ".")

        enriched_features.append({
            "type":          feat.get("type", "exon").lower(),
            "start":         feat.get("start", 0),
            "end":           feat.get("end", 0),
            "strand":        feat.get("strand", "+"),
            "phase":         feat.get("phase", "."),
            "gene_id":       gene_id,
            "transcript_id": transcript_id,
            "biotype":       biotype,
        })

    # Ab initio scenario (no hints)
    scenarios.append({
        "gene_id":       gene_id,
        "scenario_type": "empty",
        "features":      [f.copy() for f in enriched_features],
        "hints":         [],
        "start":         gene_data.get("start", 0),
        "end":           gene_data.get("end", 0),
        "strand":        gene_data.get("strand", "+"),
    })

    # Hinted scenario (noised hints from ab initio)
    random.seed(seed + hash(gene_id) % 10000)
    hints, _, _ = noiser.noise_features(enriched_features, "")

    scenarios.append({
        "gene_id":       gene_id,
        "scenario_type": "hinted",
        "features":      [f.copy() for f in enriched_features],
        "hints":         hints,
        "start":         gene_data.get("start", 0),
        "end":           gene_data.get("end", 0),
        "strand":        gene_data.get("strand", "+"),
    })

    return scenarios


def build_validation_set(
    gene_groups,
    num_complex     = 5,
    num_normal      = 5,
    num_easy        = 5,
    seed            = 42,
    max_gene_length = None,
):
    """
    Build validation set: 50% ab initio, 50% hinted per gene category

    Filters genes by max_gene_length FIRST, then selects by complexity
    """

    # Filter genes by length before selection
    if max_gene_length is not None:
        filtered_groups = {}
        filtered_count  = 0

        for gene_id, gene_data in gene_groups.items():
            gene_start = gene_data.get("start", 0)
            gene_end   = gene_data.get("end", 0)
            gene_len   = gene_end - gene_start + 1

            if gene_len <= max_gene_length:
                filtered_groups[gene_id] = gene_data
            else:
                filtered_count += 1

        if filtered_count > 0:
            print(f"  Filtered {filtered_count} genes exceeding {max_gene_length} bp")

        gene_groups = filtered_groups

    complexity_cache = compute_all_complexities(gene_groups)

    validation = {
        "complex_loci": [],
        "normal_genes": [],
        "easy_samples": [],
        "all_ids":      set(),
        "scenarios":    [],
    }

    # Get top K complex loci
    complex_loci = identify_complex_loci(gene_groups, complexity_cache, num_complex)
    for gene_id, gene_data, score in complex_loci:
        validation["complex_loci"].append({
            "gene_id":    gene_id,
            "complexity": score,
            "start":      gene_data.get("start", 0),
            "end":        gene_data.get("end", 0),
        })
        validation["all_ids"].add(gene_id)
        validation["scenarios"].extend(build_scenarios(gene_id, gene_data, seed))

    # Get easy samples
    remaining    = {gid: gdata for gid, gdata in gene_groups.items() if gid not in validation["all_ids"]}
    easy_samples = identify_easy_genes(remaining, complexity_cache, num_easy, seed)
    for gene_id, gene_data in easy_samples:
        validation["easy_samples"].append({
            "gene_id": gene_id,
            "start":   gene_data.get("start", 0),
            "end":     gene_data.get("end", 0),
        })
        validation["all_ids"].add(gene_id)
        validation["scenarios"].extend(build_scenarios(gene_id, gene_data, seed))

    # Get normal genes (around mean complexity)
    normal_genes = identify_mean_complexity_genes(
        gene_groups, complexity_cache, validation["all_ids"], num_normal, seed
    )
    for gene_id, gene_data in normal_genes:
        validation["normal_genes"].append({
            "gene_id":    gene_id,
            "complexity": complexity_cache[gene_id],
            "start":      gene_data.get("start", 0),
            "end":        gene_data.get("end", 0),
        })
        validation["all_ids"].add(gene_id)
        validation["scenarios"].extend(build_scenarios(gene_id, gene_data, seed))

    return validation


#################
#####  I/O  #####
#################


def save_validation_set(validation, output_path):
    """Save validation set to JSON file"""

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "complex_loci": validation["complex_loci"],
        "normal_genes": validation["normal_genes"],
        "easy_samples": validation["easy_samples"],
        "all_ids":      list(validation["all_ids"]),
        "scenarios":    validation["scenarios"],
    }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    n_genes     = len(validation['all_ids'])
    n_scenarios = len(validation['scenarios'])

    print(f"\n  Validation Set:")
    print(f"    Complex loci:  {len(validation['complex_loci'])}")
    print(f"    Normal genes:  {len(validation['normal_genes'])}")
    print(f"    Easy samples:  {len(validation['easy_samples'])}")
    print(f"    Total genes:   {n_genes}")
    print(f"    Scenarios:     {n_scenarios} ({n_scenarios//2} ab initio + {n_scenarios//2} hinted)")

    return output_path


def print_validation_stats(validation):
    """Print validation statistics"""

    n_genes     = len(validation.get('all_ids', []))
    n_scenarios = len(validation.get('scenarios', []))

    print(f"\n  Validation:")
    print(f"    Complex loci: {len(validation.get('complex_loci', []))}")
    print(f"    Normal genes: {len(validation.get('normal_genes', []))}")
    print(f"    Easy samples: {len(validation.get('easy_samples', []))}")
    print(f"    Total genes:  {n_genes}")
    print(f"    Scenarios:    {n_scenarios} ({n_scenarios//2} ab initio + {n_scenarios//2} hinted)")