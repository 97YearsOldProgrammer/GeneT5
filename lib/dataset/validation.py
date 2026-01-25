"""Validation set building and management"""

import json    as js
import math    as mt
import random  as rnd
import pathlib as pl


#####################  Entropy Calculation  #####################


def entropy(probs):
    """Calculate Shannon entropy of probability distribution"""
    
    if not probs:
        return 0.0
    
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * mt.log2(p)
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


#####################  Gene Identifiers  #####################


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
    """Identify top-K most complex loci based on entropy complexity score"""
    
    scored = []
    
    for gene_id, gene_data in gene_groups.items():
        score = calculate_complexity(gene_data)
        scored.append((gene_id, gene_data, score))
    
    scored.sort(key=lambda x: -x[2])
    
    return scored[:top_k]


def identify_easy_genes(gene_groups, num_samples=10, seed=42):
    """Identify easy genes with simple structure"""
    
    rnd.seed(seed)
    
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
    
    selected = rnd.sample(easy_candidates[:num_samples * 3], num_samples)
    return [(gid, gdata) for gid, gdata, _ in selected]


def select_rare_samples(gene_groups, exclude_ids, num_samples=10, seed=42):
    """Randomly select rare samples excluding specified gene IDs"""
    
    rnd.seed(seed)
    
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
    
    selected = rnd.sample(top_rare, num_samples)
    return [(gid, gdata) for gid, gdata, _ in selected]


#####################  Hint Generation  #####################


def generate_hints(features, scenario="mixed", seed=None):
    """Generate noised hints for validation scenarios"""
    
    if seed is not None:
        rnd.seed(seed)
    
    if scenario == "empty":
        return [], "empty"
    
    if scenario == "perfect":
        return [f.copy() for f in features], "perfect"
    
    if scenario == "mixed":
        scenario = rnd.choice(["good", "bad", "perfect", "empty"])
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
        if rnd.random() < drop_rate:
            continue
        
        hint = feat.copy()
        
        jitter_start  = int(rnd.gauss(0, jitter_std))
        jitter_end    = int(rnd.gauss(0, jitter_std))
        hint["start"] = max(1, hint["start"] + jitter_start)
        hint["end"]   = hint["end"] + jitter_end
        
        if hint["end"] <= hint["start"]:
            hint["end"] = hint["start"] + 10
        
        hints.append(hint)
    
    if features and rnd.random() < fake_rate:
        max_pos    = max(f["end"] for f in features)
        fake_start = rnd.randint(1, max(1, max_pos - 200))
        fake_end   = fake_start + rnd.randint(50, 200)
        
        hints.append({
            "type":   rnd.choice(["exon", "intron", "CDS"]),
            "start":  fake_start,
            "end":    fake_end,
            "strand": rnd.choice(["+", "-"]),
            "fake":   True,
        })
    
    return hints, scenario


#####################  Scenario Building  #####################


def build_scenarios(gene_id, gene_data, seed=42):
    """Build multiple validation scenarios for a single gene"""
    
    rnd.seed(seed)
    
    features       = gene_data.get("features", [])
    scenarios      = []
    scenario_types = ["perfect", "good", "bad", "empty"]
    
    for stype in scenario_types:
        hints, actual_type = generate_hints(
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
    """Build validation set using feature-based selection"""
    
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


#####################  I/O  #####################


def save_validation_set(validation, output_path):
    """Save validation set to JSON file"""
    
    output_path = pl.Path(output_path)
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
        js.dump(save_data, f, indent=2)
    
    return output_path


def load_validation_set(input_path):
    """Load existing validation set from JSON file"""
    
    with open(input_path, 'r') as f:
        data = js.load(f)
    
    data["all_ids"] = set(data.get("all_ids", []))
    
    return data


def get_existing_ids(jsonl_path):
    """Get gene IDs already in validation JSONL file"""
    
    path = pl.Path(jsonl_path)
    if not path.exists():
        return set()
    
    ids = set()
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                record = js.loads(line)
                gid    = record.get("gene_id", "")
                if gid:
                    ids.add(gid)
    
    return ids


def scenario_to_record(scenario, sequences=None):
    """Convert a validation scenario to JSONL record format"""
    
    gene_id   = scenario.get("gene_id", "unknown")
    start     = scenario.get("start", 0)
    end       = scenario.get("end", 0)
    strand    = scenario.get("strand", "+")
    features  = scenario.get("features", [])
    hints     = scenario.get("hints", [])
    stype     = scenario.get("scenario_type", "unknown")
    
    input_parts = []
    
    if sequences and gene_id in sequences:
        seq = sequences.get(gene_id, "")
    else:
        seq = scenario.get("sequence", "")
    
    if seq:
        input_parts.append(seq)
    
    if hints:
        input_parts.append("[HIT]")
        for h in sorted(hints, key=lambda x: x.get("start", 0)):
            htype   = h.get("type", "exon").lower()
            hstart  = h.get("start", 0)
            hend    = h.get("end", 0)
            hstrand = h.get("strand", "+")
            input_parts.append(f"{htype}\t{hstart}\t{hend}\t{hstrand}")
    
    target_parts = ["<BOS>"]
    for f in sorted(features, key=lambda x: x.get("start", 0)):
        ftype   = f.get("type", "exon").lower()
        fstart  = f.get("start", 0)
        fend    = f.get("end", 0)
        fstrand = f.get("strand", "+")
        fphase  = f.get("phase", ".")
        target_parts.append(f"{ftype}\t{fstart}\t{fend}\t{fstrand}\t{fphase}")
    target_parts.append("<EOS>")
    
    return {
        "input":         "\n".join(input_parts),
        "target":        "\n".join(target_parts),
        "gene_id":       gene_id,
        "scenario_type": stype,
        "start":         start,
        "end":           end,
        "strand":        strand,
    }
