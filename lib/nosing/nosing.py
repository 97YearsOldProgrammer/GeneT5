import random
from dataclasses import dataclass, field

import lib.nosing._exon    as exon
import lib.nosing._intron  as intron
import lib.nosing._protein as protein


####################
#####  Config  #####
####################


@dataclass
class NoisingConfig:
    """Configuration for GFF noising during training"""

    scenario_weights: dict = field(default_factory=lambda: {
        'full_mix':    0.40,
        'intron_only': 0.25,
        'cds_only':    0.20,
        'degraded':    0.10,
        'ab_initio':   0.05,
    })

    intron_drop_base:        float = 0.10
    intron_anchor_alpha:     float = 0.05
    intron_anchor_beta:      float = 0.40
    intron_anchor_gamma:     float = 0.15
    intron_hallucinate_rate: float = 0.03
    intron_min_length:       int   = 20
    intron_max_length:       int   = 50000

    cds_jitter_sigma:      float = 20.0
    cds_truncate_5_prob:   float = 0.12
    cds_truncate_3_prob:   float = 0.10
    cds_truncate_5_max:    int   = 60
    cds_truncate_3_max:    int   = 45
    cds_frameshift_prob:   float = 0.02
    cds_wrong_strand_prob: float = 0.28

    exon_boundary_lambda: float = 0.1
    exon_boundary_max:    int   = 50
    exon_merge_prob:      float = 0.02
    exon_drop_prob:       float = 0.05

    degraded_drop_mult:  float = 3.0
    degraded_noise_mult: float = 2.0


#######################
#####  Utilities  #####
#######################


def select_scenario(config):
    """Select a noise scenario based on configured weights"""

    weights   = config.scenario_weights
    scenarios = list(weights.keys())
    probs     = [weights[s] for s in scenarios]
    return random.choices(scenarios, weights=probs, k=1)[0]


class GFFNoiser:
    """Noiser for generating training hints from GFF features"""

    def __init__(self, config=None):

        self.config = config or NoisingConfig()

    def extract_features_by_type(self, features):
        """Separate features into exons and CDS"""

        exons    = []
        cds_list = []

        for feat in features:
            ftype = feat.get("type", "").lower()
            if ftype == "exon":
                exons.append(feat)
            elif ftype == "cds":
                cds_list.append(feat)

        return exons, cds_list

    def noise_features(self, features, sequence, scenario=None):
        """Apply noise to features based on scenario"""

        if scenario is None:
            scenario = select_scenario(self.config)

        exons, cds_features = self.extract_features_by_type(features)
        hints               = []
        degraded            = (scenario == "degraded")

        noise_log = {
            "scenario":       scenario,
            "original_exons": len(exons),
            "original_cds":   len(cds_features),
        }

        if scenario == "ab_initio" or scenario == "empty":
            noise_log["hint_introns"] = 0
            noise_log["hint_exons"]   = 0
            noise_log["hint_cds"]     = 0
            return [], scenario, noise_log

        if scenario == "perfect":
            hints = [f.copy() for f in features]
            noise_log["hint_introns"] = len([h for h in hints if h.get("type", "").lower() == "intron"])
            noise_log["hint_exons"]   = len([h for h in hints if h.get("type", "").lower() == "exon"])
            noise_log["hint_cds"]     = len([h for h in hints if h.get("type", "").lower() == "cds"])
            return hints, scenario, noise_log

        if scenario == "good":
            hints = self._apply_good_noise(exons, cds_features)
            noise_log["hint_introns"] = len([h for h in hints if h.get("type", "").lower() == "intron"])
            noise_log["hint_exons"]   = len([h for h in hints if h.get("type", "").lower() == "exon"])
            noise_log["hint_cds"]     = len([h for h in hints if h.get("type", "").lower() == "cds"])
            return hints, scenario, noise_log

        if scenario == "bad":
            hints = self._apply_bad_noise(exons, cds_features, sequence)
            noise_log["hint_introns"] = len([h for h in hints if h.get("type", "").lower() == "intron"])
            noise_log["hint_exons"]   = len([h for h in hints if h.get("type", "").lower() == "exon"])
            noise_log["hint_cds"]     = len([h for h in hints if h.get("type", "").lower() == "cds"])
            return hints, scenario, noise_log

        if scenario in ("full_mix", "intron_only", "degraded"):
            strand       = exons[0]["strand"] if exons else "+"
            real_introns = intron.compute_introns_from_exons(exons, strand)
            hint_introns = intron.noise_real_introns(real_introns, exons, self.config, degraded)
            hints.extend(hint_introns)

            fake_introns = intron.generate_fake_introns(sequence, real_introns, features, self.config)
            hints.extend(fake_introns)

            noise_log["hint_introns"] = len([h for h in hints if h["type"] == "intron"])

        if scenario in ("full_mix", "cds_only", "degraded"):
            hint_exons = exon.noise_real_exons(exons, self.config, degraded)
            hints.extend(hint_exons)

            fake_exon = exon.generate_fake_exon(sequence, features, self.config)
            if fake_exon and random.random() < 0.05:
                hints.append(fake_exon)

            hint_cds = protein.noise_real_cds(cds_features, self.config, degraded)
            hints.extend(hint_cds)

            fake_cds = protein.generate_fake_cds(sequence, features, self.config)
            if fake_cds and random.random() < 0.03:
                hints.append(fake_cds)

            noise_log["hint_exons"] = len([h for h in hints if h.get("type", "").lower() == "exon"])
            noise_log["hint_cds"]   = len([h for h in hints if h.get("type", "").lower() == "cds"])

        return hints, scenario, noise_log

    def _apply_good_noise(self, exons, cds_features):
        """Apply light noise for good quality hints"""

        hints  = []
        strand = exons[0]["strand"] if exons else "+"

        real_introns = intron.compute_introns_from_exons(exons, strand)
        for intr in real_introns:
            if random.random() < 0.05:
                continue
            hints.append(intr.copy())

        for ex in exons:
            if random.random() < 0.05:
                continue

            jitter_start = int(random.gauss(0, 5))
            jitter_end   = int(random.gauss(0, 5))

            hints.append({
                "type":   "exon",
                "start":  max(1, ex["start"] + jitter_start),
                "end":    ex["end"] + jitter_end,
                "strand": ex["strand"],
            })

        for cds in cds_features:
            jitter_start = int(random.gauss(0, 5))
            jitter_end   = int(random.gauss(0, 5))
            jitter_start = (jitter_start // 3) * 3
            jitter_end   = (jitter_end // 3) * 3

            hints.append({
                "type":   "CDS",
                "start":  max(1, cds["start"] + jitter_start),
                "end":    cds["end"] + jitter_end,
                "strand": cds["strand"],
                "phase":  cds.get("phase", "."),
            })

        return hints

    def _apply_bad_noise(self, exons, cds_features, sequence):
        """Apply heavy noise for bad quality hints"""

        hints  = []
        strand = exons[0]["strand"] if exons else "+"

        real_introns = intron.compute_introns_from_exons(exons, strand)
        for intr in real_introns:
            if random.random() < 0.30:
                continue
            hints.append(intr.copy())

        for ex in exons:
            if random.random() < 0.30:
                continue

            jitter_start = int(random.gauss(0, 30))
            jitter_end   = int(random.gauss(0, 30))

            hints.append({
                "type":   "exon",
                "start":  max(1, ex["start"] + jitter_start),
                "end":    ex["end"] + jitter_end,
                "strand": ex["strand"],
            })

        for cds in cds_features:
            if random.random() < 0.15:
                continue

            jitter_start = int(random.gauss(0, 30))
            jitter_end   = int(random.gauss(0, 30))

            hints.append({
                "type":   "CDS",
                "start":  max(1, cds["start"] + jitter_start),
                "end":    cds["end"] + jitter_end,
                "strand": cds["strand"],
                "phase":  cds.get("phase", "."),
            })

        all_features = exons + cds_features
        if all_features and random.random() < 0.15:
            max_pos    = max(f["end"] for f in all_features)
            fake_start = random.randint(1, max(1, max_pos - 200))
            fake_end   = fake_start + random.randint(50, 200)

            hints.append({
                "type":   random.choice(["exon", "intron", "CDS"]),
                "start":  fake_start,
                "end":    fake_end,
                "strand": random.choice(["+", "-"]),
                "fake":   True,
            })

        return hints

    def format_hints_for_input(self, hints, hint_token="[HIT]"):
        """Format hints as input string"""

        if not hints:
            return ""

        lines = [hint_token]

        for hint in sorted(hints, key=lambda x: x["start"]):
            ftype  = hint["type"].lower()
            start  = hint["start"]
            end    = hint["end"]
            strand = hint.get("strand", "+")
            lines.append(f"{ftype}\t{start}\t{end}\t{strand}")

        return "\n".join(lines)