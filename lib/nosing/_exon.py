import random


def apply_exon_boundary_extension(exon, config):

    ext_start = int(random.expovariate(config.exon_boundary_lambda))
    ext_end   = int(random.expovariate(config.exon_boundary_lambda))
    ext_start = min(ext_start, config.exon_boundary_max)
    ext_end   = min(ext_end, config.exon_boundary_max)
    
    return {
        "type":   "exon",
        "start":  max(1, exon["start"] - ext_start),
        "end":    exon["end"] + ext_end,
        "strand": exon["strand"],
    }


def noise_real_exons(exons, config, degraded=False):

    if not exons:
        return []
    
    noised       = []
    drop_mult    = config.degraded_drop_mult if degraded else 1.0
    sorted_exons = sorted(exons, key=lambda x: x["start"])
    i            = 0
    
    while i < len(sorted_exons):
        exon = sorted_exons[i]
        
        if random.random() < config.exon_drop_prob * drop_mult:
            i += 1
            continue
        
        if i < len(sorted_exons) - 1 and random.random() < config.exon_merge_prob:
            next_exon = sorted_exons[i + 1]
            merged    = {
                "type":   "exon",
                "start":  exon["start"],
                "end":    next_exon["end"],
                "strand": exon["strand"],
            }
            noised.append(apply_exon_boundary_extension(merged, config))
            i += 2
            continue
        
        noised.append(apply_exon_boundary_extension(exon, config))
        i += 1
    
    return noised


def generate_fake_exon(sequence, existing_features, config):

    feature_regions = []
    for feat in existing_features:
        feature_regions.append((feat["start"], feat["end"]))
    
    if not feature_regions:
        start  = random.randint(1, max(1, len(sequence) - 500))
        length = random.randint(50, 300)
        return {
            "type":   "exon",
            "start":  start,
            "end":    min(start + length, len(sequence)),
            "strand": random.choice(["+", "-"]),
            "fake":   True,
        }
    
    sorted_regions = sorted(feature_regions)
    gaps           = []
    prev_end       = 0
    
    for start, end in sorted_regions:
        if start - prev_end > 200:
            gaps.append((prev_end + 1, start - 1))
        prev_end = end
    
    if len(sequence) - prev_end > 200:
        gaps.append((prev_end + 1, len(sequence)))
    
    if not gaps:
        return None
    
    gap_start, gap_end = random.choice(gaps)
    gap_size           = gap_end - gap_start
    
    if gap_size < 100:
        return None
    
    exon_length = random.randint(50, min(300, gap_size - 50))
    exon_start  = random.randint(gap_start, gap_end - exon_length)
    
    return {
        "type":   "exon",
        "start":  exon_start,
        "end":    exon_start + exon_length,
        "strand": random.choice(["+", "-"]),
        "fake":   True,
    }