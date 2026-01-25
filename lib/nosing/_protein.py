import random



#######################
#####  Utilities  #####
#######################


def apply_cds_jitter(cds, config, is_terminal=False):
    """Apply jitter to CDS coordinates"""

    sigma        = config.cds_jitter_sigma
    jitter_start = int(random.gauss(0, sigma))
    jitter_end   = int(random.gauss(0, sigma))

    if not is_terminal:
        jitter_start = (jitter_start // 3) * 3
        jitter_end   = (jitter_end // 3) * 3

    new_start = max(1, cds["start"] + jitter_start)
    new_end   = max(new_start + 3, cds["end"] + jitter_end)

    return {
        "type":   "CDS",
        "start":  new_start,
        "end":    new_end,
        "strand": cds["strand"],
        "phase":  cds.get("phase", "."),
    }


def apply_cds_truncation(cds, config, position):
    """Apply truncation to CDS at specified position"""

    result = cds.copy()

    if position == "5prime":
        max_trunc = min(config.cds_truncate_5_max, (cds["end"] - cds["start"]) // 2)
        if max_trunc > 3:
            trunc           = random.randint(3, max_trunc)
            trunc           = (trunc // 3) * 3
            result["start"] = cds["start"] + trunc
    else:
        max_trunc = min(config.cds_truncate_3_max, (cds["end"] - cds["start"]) // 2)
        if max_trunc > 3:
            trunc         = random.randint(3, max_trunc)
            trunc         = (trunc // 3) * 3
            result["end"] = cds["end"] - trunc

    return result


def apply_frameshift(cds, config):
    """Apply frameshift to CDS"""

    shift           = random.choice([-2, -1, 1, 2])
    result          = cds.copy()
    result["start"] = max(1, cds["start"] + shift)
    return result


def noise_real_cds(cds_features, config, degraded=False):
    """Apply noise to real CDS features"""

    if not cds_features:
        return []

    noised     = []
    noise_mult = config.degraded_noise_mult if degraded else 1.0
    sorted_cds = sorted(cds_features, key=lambda x: x["start"])

    for i, cds in enumerate(sorted_cds):
        is_first = (i == 0)
        is_last  = (i == len(sorted_cds) - 1)
        result   = cds.copy()
        result   = apply_cds_jitter(result, config, is_terminal=(is_first or is_last))

        if is_first and random.random() < config.cds_truncate_5_prob * noise_mult:
            result = apply_cds_truncation(result, config, "5prime")

        if is_last and random.random() < config.cds_truncate_3_prob * noise_mult:
            result = apply_cds_truncation(result, config, "3prime")

        if random.random() < config.cds_frameshift_prob * noise_mult:
            result = apply_frameshift(result, config)

        noised.append(result)

    return noised


def generate_fake_cds(sequence, existing_features, config):
    """Generate a fake CDS using start/stop codons"""

    seq_upper     = sequence.upper()
    atg_positions = []
    start         = 0

    while True:
        pos = seq_upper.find("ATG", start)
        if pos == -1:
            break
        atg_positions.append(pos)
        start = pos + 1

    if not atg_positions:
        return None

    stop_codons    = ["TAA", "TAG", "TGA"]
    coding_regions = []

    for feat in existing_features:
        ftype = feat.get("type", "").lower()
        if ftype in {"cds", "exon"}:
            coding_regions.append((feat["start"], feat["end"]))

    for _ in range(30):
        atg_pos = random.choice(atg_positions)

        for offset in range(atg_pos + 3, min(atg_pos + 3000, len(sequence) - 2), 3):
            codon = seq_upper[offset:offset + 3]
            if codon in stop_codons:
                cds_start = atg_pos + 1
                cds_end   = offset + 3
                overlaps  = False

                for cs, ce in coding_regions:
                    if not (cds_end < cs or cds_start > ce):
                        overlaps = True
                        break

                if not overlaps and cds_end - cds_start >= 60:
                    return {
                        "type":   "CDS",
                        "start":  cds_start,
                        "end":    cds_end,
                        "strand": "+",
                        "phase":  "0",
                        "fake":   True,
                    }
                break

    return None