import random
import math


#######################
#####  Utilities  #####
#######################


DONOR_SITE    = "GT"
ACCEPTOR_SITE = "AG"


def compute_introns_from_exons(exons, strand):
    """Compute intron positions from exon gaps"""

    if len(exons) < 2:
        return []

    sorted_exons = sorted(exons, key=lambda x: x["start"])
    introns      = []

    for i in range(len(sorted_exons) - 1):
        exon1        = sorted_exons[i]
        exon2        = sorted_exons[i + 1]
        intron_start = exon1["end"] + 1
        intron_end   = exon2["start"] - 1

        if intron_end > intron_start:
            introns.append({
                "type":   "intron",
                "start":  intron_start,
                "end":    intron_end,
                "strand": strand,
            })

    return introns


def group_exons_by_parent(features):
    """Group exons by their parent transcript"""

    groups = {}

    for feat in features:
        if feat.get("type", "").lower() != "exon":
            continue

        parent = feat.get("attributes", {}).get("Parent", "unknown")
        if parent not in groups:
            groups[parent] = []
        groups[parent].append(feat)

    return groups


def drop_intron_by_anchor(intron, exons, config):
    """Determine if intron should be dropped based on anchor length"""

    intron_start      = intron["start"]
    intron_end        = intron["end"]
    upstream_anchor   = 0
    downstream_anchor = 0

    for exon in exons:
        if exon["end"] == intron_start - 1:
            upstream_anchor = exon["end"] - exon["start"] + 1
        if exon["start"] == intron_end + 1:
            downstream_anchor = exon["end"] - exon["start"] + 1

    L_anchor = min(upstream_anchor, downstream_anchor) if upstream_anchor and downstream_anchor else 10
    alpha    = config.intron_anchor_alpha
    beta     = config.intron_anchor_beta
    gamma    = config.intron_anchor_gamma
    p_drop   = alpha + beta * math.exp(-gamma * L_anchor)

    return random.random() < p_drop


def noise_real_introns(introns, exons, config, degraded=False):
    """Apply noise to real introns"""

    surviving = []
    drop_mult = config.degraded_drop_mult if degraded else 1.0

    for intron in introns:
        if random.random() < config.intron_drop_base * drop_mult:
            continue

        if drop_intron_by_anchor(intron, exons, config):
            continue

        surviving.append({
            "type":   "intron",
            "start":  intron["start"],
            "end":    intron["end"],
            "strand": intron["strand"],
        })

    return surviving


def find_splice_sites(sequence, site_type="donor"):
    """Find potential splice sites in sequence"""

    pattern   = DONOR_SITE if site_type == "donor" else ACCEPTOR_SITE
    positions = []
    seq_upper = sequence.upper()
    start     = 0

    while True:
        pos = seq_upper.find(pattern, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1

    return positions


def generate_fake_intron(sequence, existing_features, config, strand="+"):
    """Generate a fake intron using canonical splice sites"""

    donors    = find_splice_sites(sequence, "donor")
    acceptors = find_splice_sites(sequence, "acceptor")

    if not donors or not acceptors:
        return None

    exon_regions = []
    for feat in existing_features:
        if feat.get("type", "").lower() == "exon":
            exon_regions.append((feat["start"], feat["end"]))

    max_attempts = 50

    for _ in range(max_attempts):
        donor_pos       = random.choice(donors) + 1
        valid_acceptors = [
            a + 1 for a in acceptors
            if config.intron_min_length <= (a + 1 - donor_pos) <= config.intron_max_length
            and a > donor_pos
        ]

        if not valid_acceptors:
            continue

        acceptor_pos = random.choice(valid_acceptors)
        overlaps     = False

        for exon_start, exon_end in exon_regions:
            if not (acceptor_pos < exon_start or donor_pos > exon_end):
                overlaps = True
                break

        if overlaps:
            continue

        return {
            "type":   "intron",
            "start":  donor_pos,
            "end":    acceptor_pos + 1,
            "strand": strand,
            "fake":   True,
        }

    return None


def generate_fake_introns(sequence, real_introns, existing_features, config):
    """Generate fake introns based on hallucination rate"""

    n_real = len(real_introns)
    n_fake = int(n_real * config.intron_hallucinate_rate)

    if n_real > 0 and n_fake == 0:
        n_fake = 1 if random.random() < config.intron_hallucinate_rate else 0

    fake_introns = []
    strands      = ["+", "-"]

    for _ in range(n_fake):
        strand = random.choice(strands)
        fake   = generate_fake_intron(sequence, existing_features, config, strand)
        if fake:
            fake_introns.append(fake)

    return fake_introns