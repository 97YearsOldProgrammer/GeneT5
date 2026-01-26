import json
import math
import random
import pathlib

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


#######################
#####  Selection  #####
#######################


def identify_long_genes(gene_groups, top_k=5):
    """Identify top-k longest genes"""

    scored = []

    for gene_id, gene_data in gene_groups.items():
        gene_start = gene_data.get("start", 0)
        gene_end   = gene_data.get("end", 0)
        gene_len   = gene_end - gene_start + 1

        scored.append((gene_id, gene_data, gene_len))

    scored.sort(key=lambda x: -x[2])

    return scored[:top_k]


def identify_complex_loci(gene_groups, top_k=5):
    """Identify top-K most complex loci based on entropy complexity score"""

    scored = []

    for gene_id, gene_data in gene_groups.items():
        score = calculate_complexity(gene_data)
        scored.append((gene_id, gene_data, score))

    scored.sort(key=lambda x: -x[2])

    return scored[:top_k]


def identify_easy_genes(gene_groups, num_samples=10, seed=42):
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
            complexity = calculate_complexity(gene_data)
            easy_candidates.append((gene_id, gene_data, complexity))

    easy_candidates.sort(key=lambda x: x[2])

    if len(easy_candidates) <= num_samples:
        return [(gid, gdata) for gid, gdata, _ in easy_candidates]

    selected = random.sample(easy_candidates[:num_samples * 3], num_samples)
    return [(gid, gdata) for gid, gdata, _ in selected]


def select_rare_samples(gene_groups, exclude_ids, num_samples=10, seed=42):
    """Randomly select rare samples excluding specified gene IDs"""

    random.seed(seed)

    candidates = {
        gid: gdata for gid, gdata in gene_groups.items()
        if gid not in exclude_ids
    }

    if not candidates:
        return []

    rare_biotypes = {"pseudogene", "lncrna", "snorna", "mirna", "guide_rna", "rrna", "trna"}

    rarity_scored = []
    for gene_id, gene_data in candidates.items():
        features     = gene_data.get("features", [])
        num_features = len(features)

        biotypes = set()
        for t_id, t_data in gene_data.get("transcripts", {}).items():
            bt = t_data.get("biotype", "").lower()
            if bt:
                biotypes.add(bt)

        is_rare_biotype = bool(biotypes & rare_biotypes)

        rarity = 1.0 / max(num_features, 1)
        if is_rare_biotype:
            rarity *= 3.0

        rarity_scored.append((gene_id, gene_data, rarity))

    rarity_scored.sort(key=lambda x: -x[2])
    top_rare = rarity_scored[:min(num_samples * 3, len(rarity_scored))]

    if len(top_rare) <= num_samples:
        return [(gid, gdata) for gid, gdata, _ in top_rare]

    selected = random.sample(top_rare, num_samples)
    return [(gid, gdata) for gid, gdata, _ in selected]


#####################
#####  Scenario #####
#####################


def build_scenarios(gene_id, gene_data, seed=42):
    """Build multiple validation scenarios for a single gene"""

    random.seed(seed)

    noiser         = nosing.GFFNoiser()
    features       = gene_data.get("features", [])
    transcripts    = gene_data.get("transcripts", {})
    scenarios      = []
    scenario_types = ["perfect", "good", "bad", "empty"]

    # Enrich features with transcript_id and biotype
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

    for stype in scenario_types:
        random.seed(seed + hash(stype) % 10000)

        if stype == "empty":
            hints = []
        elif stype == "perfect":
            hints = [f.copy() for f in enriched_features]
        else:
            hints, _, _ = noiser.noise_features(enriched_features, "", scenario=stype)

        scenarios.append({
            "gene_id":       gene_id,
            "scenario_type": stype,
            "features":      [f.copy() for f in enriched_features],
            "hints":         hints,
            "start":         gene_data.get("start", 0),
            "end":           gene_data.get("end", 0),
            "strand":        gene_data.get("strand", "+"),
        })

    return scenarios


def build_validation_set(
    gene_groups,
    top_k_long       = 5,
    top_k_complex    = 5,
    num_rare_samples = 10,
    num_easy_samples = 10,
    seed             = 42,
):
    """Build validation set using feature-based selection"""

    validation = {
        "long_genes":   [],
        "complex_loci": [],
        "rare_samples": [],
        "easy_samples": [],
        "all_ids":      set(),
        "scenarios":    [],
    }

    # Get top K long genes
    long_genes = identify_long_genes(gene_groups, top_k_long)
    for gene_id, gene_data, length in long_genes:
        validation["long_genes"].append({
            "gene_id": gene_id,
            "length":  length,
            "start":   gene_data.get("start", 0),
            "end":     gene_data.get("end", 0),
        })
        validation["all_ids"].add(gene_id)

        scenarios = build_scenarios(gene_id, gene_data, seed)
        validation["scenarios"].extend(scenarios)

    remaining = {
        gid: gdata for gid, gdata in gene_groups.items()
        if gid not in validation["all_ids"]
    }
    complex_loci = identify_complex_loci(remaining, top_k_complex)
    for gene_id, gene_data, score in complex_loci:
        validation["complex_loci"].append({
            "gene_id":    gene_id,
            "complexity": score,
            "start":      gene_data.get("start", 0),
            "end":        gene_data.get("end", 0),
        })
        validation["all_ids"].add(gene_id)

        scenarios = build_scenarios(gene_id, gene_data, seed)
        validation["scenarios"].extend(scenarios)

    remaining = {
        gid: gdata for gid, gdata in gene_groups.items()
        if gid not in validation["all_ids"]
    }
    easy_samples = identify_easy_genes(remaining, num_easy_samples, seed)
    for gene_id, gene_data in easy_samples:
        validation["easy_samples"].append({
            "gene_id": gene_id,
            "start":   gene_data.get("start", 0),
            "end":     gene_data.get("end", 0),
        })
        validation["all_ids"].add(gene_id)

        scenarios = build_scenarios(gene_id, gene_data, seed)
        validation["scenarios"].extend(scenarios)

    rare_samples = select_rare_samples(
        gene_groups,
        validation["all_ids"],
        num_rare_samples,
        seed,
    )
    for gene_id, gene_data in rare_samples:
        validation["rare_samples"].append({
            "gene_id": gene_id,
            "start":   gene_data.get("start", 0),
            "end":     gene_data.get("end", 0),
        })
        validation["all_ids"].add(gene_id)

        scenarios = build_scenarios(gene_id, gene_data, seed)
        validation["scenarios"].extend(scenarios)

    return validation


#################
#####  I/O  #####
#################


def save_validation_set(validation, output_path):
    """Save validation set to JSON file"""

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "long_genes":   validation["long_genes"],
        "complex_loci": validation["complex_loci"],
        "rare_samples": validation["rare_samples"],
        "easy_samples": validation["easy_samples"],
        "all_ids":      list(validation["all_ids"]),
        "scenarios":    validation["scenarios"],
    }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Validation Set:")
    print(f"    Long genes:     {len(validation['long_genes'])}")
    print(f"    Complex loci:   {len(validation['complex_loci'])}")
    print(f"    Rare samples:   {len(validation['rare_samples'])}")
    print(f"    Easy samples:   {len(validation['easy_samples'])}")
    print(f"    Total genes:    {len(validation['all_ids'])}")
    print(f"    Scenarios:      {len(validation['scenarios'])}")

    return output_path