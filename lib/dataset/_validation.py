import json
import random
from pathlib    import Path
from collections import defaultdict


def calculate_locus_complexity(gene_data):
    """
    Calculate complexity score for a gene locus
    
    Factors: AS degree, exon variance, overlaps, span ratio
    """
    transcripts     = gene_data.get("transcripts", {})
    features        = gene_data.get("features", [])
    num_transcripts = len(transcripts)
    
    if num_transcripts == 0:
        return 0.0
    
    as_score = num_transcripts * 2.0
    
    exon_counts = []
    for t_id, t_data in transcripts.items():
        t_features = t_data.get("features", [])
        exon_count = sum(1 for f in t_features if f.get("type", "").lower() == "exon")
        exon_counts.append(exon_count)
    
    exon_variance = 0.0
    if len(exon_counts) > 1:
        mean_exons    = sum(exon_counts) / len(exon_counts)
        exon_variance = sum((x - mean_exons) ** 2 for x in exon_counts) / len(exon_counts)
    
    overlap_score   = 0.0
    sorted_features = sorted(features, key=lambda x: x.get("start", 0))
    for i in range(len(sorted_features) - 1):
        curr_end   = sorted_features[i].get("end", 0)
        next_start = sorted_features[i + 1].get("start", 0)
        if curr_end >= next_start:
            overlap_score += 1.0
    
    gene_start = gene_data.get("start", 0)
    gene_end   = gene_data.get("end", 0)
    gene_span  = gene_end - gene_start + 1
    
    coding_len = sum(
        f.get("end", 0) - f.get("start", 0) + 1
        for f in features
        if f.get("type", "").lower() in {"exon", "cds"}
    )
    
    span_ratio = (gene_span / max(coding_len, 1)) if coding_len > 0 else 1.0
    complexity = as_score + exon_variance * 0.5 + overlap_score * 1.5 + span_ratio * 0.1
    
    return complexity


def identify_long_genes(gene_groups, threshold_bp=50000):
    """Identify genes longer than threshold"""
    long_genes = []
    
    for gene_id, gene_data in gene_groups.items():
        gene_start = gene_data.get("start", 0)
        gene_end   = gene_data.get("end", 0)
        gene_len   = gene_end - gene_start + 1
        
        if gene_len > threshold_bp:
            long_genes.append((gene_id, gene_data, gene_len))
    
    return sorted(long_genes, key=lambda x: -x[2])


def identify_complex_loci(gene_groups, top_k=5):
    """Identify top-K most complex loci based on complexity score"""
    scored = []
    
    for gene_id, gene_data in gene_groups.items():
        score = calculate_locus_complexity(gene_data)
        scored.append((gene_id, gene_data, score))
    
    scored.sort(key=lambda x: -x[2])
    
    return scored[:top_k]


def identify_easy_genes(gene_groups, num_samples=10, seed=42):
    """
    Identify easy genes with simple structure
    
    Easy genes: single transcript, few exons, short span
    """
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
            complexity = calculate_locus_complexity(gene_data)
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


def generate_validation_hints(features, scenario="mixed", seed=None):
    """
    Generate noised hints for validation scenarios
    
    Scenarios:
        perfect  - exact copy of features (no noise)
        good     - minor boundary jitter only
        bad      - heavy noise, drops, fake additions
        mixed    - random mix of good and bad
        empty    - no hints at all
    """
    if seed is not None:
        random.seed(seed)
    
    if scenario == "empty":
        return [], "empty"
    
    if scenario == "perfect":
        return [f.copy() for f in features], "perfect"
    
    if scenario == "mixed":
        scenario = random.choice(["good", "bad", "perfect", "empty"])
        if scenario == "empty":
            return [], "empty"
        if scenario == "perfect":
            return [f.copy() for f in features], "perfect"
    
    hints      = []
    is_good    = (scenario == "good")
    drop_rate  = 0.05 if is_good else 0.30
    jitter_std = 5    if is_good else 30
    fake_rate  = 0.02 if is_good else 0.15
    
    for feat in features:
        if random.random() < drop_rate:
            continue
        
        hint = feat.copy()
        
        jitter_start  = int(random.gauss(0, jitter_std))
        jitter_end    = int(random.gauss(0, jitter_std))
        hint["start"] = max(1, hint["start"] + jitter_start)
        hint["end"]   = hint["end"] + jitter_end
        
        if hint["end"] <= hint["start"]:
            hint["end"] = hint["start"] + 10
        
        hints.append(hint)
    
    if features and random.random() < fake_rate:
        max_pos    = max(f["end"] for f in features)
        fake_start = random.randint(1, max(1, max_pos - 200))
        fake_end   = fake_start + random.randint(50, 200)
        
        hints.append({
            "type":   random.choice(["exon", "intron", "CDS"]),
            "start":  fake_start,
            "end":    fake_end,
            "strand": random.choice(["+", "-"]),
            "fake":   True,
        })
    
    return hints, scenario


def build_validation_scenarios(gene_id, gene_data, seed=42):
    """
    Build multiple validation scenarios for a single gene
    
    Returns list of scenarios with different hint qualities
    """
    random.seed(seed)
    
    features  = gene_data.get("features", [])
    scenarios = []
    
    scenario_types = ["perfect", "good", "bad", "empty"]
    
    for stype in scenario_types:
        hints, actual_type = generate_validation_hints(
            features,
            scenario = stype,
            seed     = seed + hash(stype) % 10000,
        )
        
        scenarios.append({
            "gene_id":       gene_id,
            "scenario_type": actual_type,
            "features":      [f.copy() for f in features],
            "hints":         hints,
            "start":         gene_data.get("start", 0),
            "end":           gene_data.get("end", 0),
            "strand":        gene_data.get("strand", "+"),
        })
    
    return scenarios


def build_validation_set(
    gene_groups,
    long_threshold   = 50000,
    top_k_complex    = 5,
    num_rare_samples = 10,
    num_easy_samples = 10,
    seed             = 42,
):
    """
    Build validation set using feature-based selection
    
    Includes: long genes, complex loci, rare samples, easy samples
    Each with multiple noise scenarios
    """
    validation = {
        "long_genes":   [],
        "complex_loci": [],
        "rare_samples": [],
        "easy_samples": [],
        "all_ids":      set(),
        "scenarios":    [],
    }
    
    long_genes = identify_long_genes(gene_groups, long_threshold)
    for gene_id, gene_data, length in long_genes:
        validation["long_genes"].append({
            "gene_id": gene_id,
            "length":  length,
            "start":   gene_data.get("start", 0),
            "end":     gene_data.get("end", 0),
        })
        validation["all_ids"].add(gene_id)
        
        scenarios = build_validation_scenarios(gene_id, gene_data, seed)
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
        
        scenarios = build_validation_scenarios(gene_id, gene_data, seed)
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
        
        scenarios = build_validation_scenarios(gene_id, gene_data, seed)
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
        
        scenarios = build_validation_scenarios(gene_id, gene_data, seed)
        validation["scenarios"].extend(scenarios)
    
    return validation


def save_validation_set(validation, output_path):
    """Save validation set to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        "long_genes":   validation["long_genes"],
        "complex_loci": validation["complex_loci"],
        "rare_samples": validation["rare_samples"],
        "easy_samples": validation["easy_samples"],
        "all_ids":      list(validation["all_ids"]),
        "scenarios":    validation["scenarios"],
        "stats": {
            "num_long":      len(validation["long_genes"]),
            "num_complex":   len(validation["complex_loci"]),
            "num_rare":      len(validation["rare_samples"]),
            "num_easy":      len(validation["easy_samples"]),
            "num_scenarios": len(validation["scenarios"]),
            "total":         len(validation["all_ids"]),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    return output_path


def load_validation_set(input_path):
    """Load existing validation set from JSON file"""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    data["all_ids"] = set(data.get("all_ids", []))
    
    return data


def extend_validation_set(existing, new_validation):
    """Extend existing validation set with new entries"""
    for category in ["long_genes", "complex_loci", "rare_samples", "easy_samples"]:
        existing_ids = {item["gene_id"] for item in existing.get(category, [])}
        for item in new_validation.get(category, []):
            if item["gene_id"] not in existing_ids:
                existing[category].append(item)
                existing["all_ids"].add(item["gene_id"])
    
    if "scenarios" not in existing:
        existing["scenarios"] = []
    existing["scenarios"].extend(new_validation.get("scenarios", []))
    
    return existing


def print_validation_stats(validation):
    """Print validation set statistics"""
    print(f"\n{'='*60}")
    print("Validation Set Statistics")
    print(f"{'='*60}")
    print(f"  Long genes (>50kb):  {len(validation['long_genes'])}")
    print(f"  Complex loci:        {len(validation['complex_loci'])}")
    print(f"  Rare samples:        {len(validation['rare_samples'])}")
    print(f"  Easy samples:        {len(validation.get('easy_samples', []))}")
    print(f"  Total scenarios:     {len(validation.get('scenarios', []))}")
    print(f"  Total unique genes:  {len(validation['all_ids'])}")
    
    if validation["long_genes"]:
        print(f"\n  Longest genes:")
        for item in validation["long_genes"][:5]:
            print(f"    {item['gene_id']}: {item['length']:,} bp")
    
    if validation["complex_loci"]:
        print(f"\n  Most complex loci:")
        for item in validation["complex_loci"][:5]:
            print(f"    {item['gene_id']}: complexity={item['complexity']:.2f}")
    
    if validation.get("scenarios"):
        scenario_counts = {}
        for s in validation["scenarios"]:
            stype = s.get("scenario_type", "unknown")
            scenario_counts[stype] = scenario_counts.get(stype, 0) + 1
        print(f"\n  Scenario distribution: {scenario_counts}")