import random
from dataclasses import dataclass, field

import lib.nosing._intron as intron


####################
#####  Config  #####
####################


@dataclass
class NoisingConfig:
    """Configuration for intron-focused hint generation"""

    scenario_weights: dict = field(default_factory=lambda: {
        'intron_rich':    0.40,
        'intron_sparse':  0.30,
        'intron_noisy':   0.20,
        'intron_minimal': 0.10,
    })

    intron_drop_base:        float = 0.10
    intron_anchor_alpha:     float = 0.05
    intron_anchor_beta:      float = 0.40
    intron_anchor_gamma:     float = 0.15
    intron_hallucinate_rate: float = 0.03
    intron_min_length:       int   = 20
    intron_max_length:       int   = 50000

    max_hints:            int = 300
    complexity_threshold: int = 500


#######################
#####  Utilities  #####
#######################


def select_scenario(config, feature_count=0):
    """Select scenario based on weights and complexity"""

    if feature_count > config.complexity_threshold:
        return 'intron_minimal'

    weights   = config.scenario_weights
    scenarios = list(weights.keys())
    probs     = [weights[s] for s in scenarios]

    return random.choices(scenarios, weights=probs, k=1)[0]


class GFFNoiser:
    """Intron-focused noiser for training hints"""

    def __init__(self, config=None):

        self.config = config or NoisingConfig()

    def extract_exons(self, features):
        """Extract exons from features"""

        exons = []

        for feat in features:
            ftype = feat.get("type", "").lower()
            if ftype == "exon":
                exons.append(feat)

        return exons

    def noise_features(self, features, sequence, scenario=None):
        """Generate intron hints based on scenario"""

        exons        = self.extract_exons(features)
        feature_count = len(features)

        if scenario is None:
            scenario = select_scenario(self.config, feature_count)

        noise_log = {
            "scenario":        scenario,
            "original_exons":  len(exons),
            "feature_count":   feature_count,
        }

        if scenario == "ab_initio":
            noise_log["intron_hc"] = 0
            noise_log["intron_lc"] = 0
            return [], scenario, noise_log

        strand       = exons[0]["strand"] if exons else "+"
        real_introns = intron.compute_introns_from_exons(exons, strand)

        if scenario == "intron_rich":
            hints = self._generate_rich_hints(real_introns, exons, sequence)

        elif scenario == "intron_sparse":
            hints = self._generate_sparse_hints(real_introns, exons, sequence)

        elif scenario == "intron_noisy":
            hints = self._generate_noisy_hints(real_introns, exons, sequence)

        elif scenario == "intron_minimal":
            hints = self._generate_minimal_hints(real_introns, exons)

        else:
            hints = self._generate_sparse_hints(real_introns, exons, sequence)

        if len(hints) > self.config.max_hints:
            hints = random.sample(hints, self.config.max_hints)

        noise_log["intron_hc"] = sum(1 for h in hints if h["type"] == "intron_hc")
        noise_log["intron_lc"] = sum(1 for h in hints if h["type"] == "intron_lc")

        return hints, scenario, noise_log

    def _generate_rich_hints(self, real_introns, exons, sequence):
        """Deep RNA-seq simulation: many high-confidence intron hints"""

        hints    = []
        drop_rate = 0.05

        for intr in real_introns:
            if random.random() < drop_rate:
                continue

            if intron.drop_intron_by_anchor(intr, exons, self.config):
                if random.random() < 0.7:
                    continue

            hints.append({
                "type":   "intron_hc",
                "start":  intr["start"],
                "end":    intr["end"],
                "strand": intr["strand"],
            })

        fake_introns = intron.generate_fake_introns(
            sequence, real_introns, exons, self.config
        )
        for fake in fake_introns:
            fake["type"] = "intron_lc"
            hints.append(fake)

        return hints

    def _generate_sparse_hints(self, real_introns, exons, sequence):
        """Shallow RNA-seq simulation: mix of hc and lc introns"""

        hints     = []
        drop_rate = 0.20

        for intr in real_introns:
            if random.random() < drop_rate:
                continue

            if intron.drop_intron_by_anchor(intr, exons, self.config):
                continue

            if random.random() < 0.7:
                hint_type = "intron_hc"
            else:
                hint_type = "intron_lc"

            hints.append({
                "type":   hint_type,
                "start":  intr["start"],
                "end":    intr["end"],
                "strand": intr["strand"],
            })

        fake_count = max(1, int(len(real_introns) * 0.05))
        fake_introns = intron.generate_fake_introns(
            sequence, real_introns, exons, self.config
        )[:fake_count]

        for fake in fake_introns:
            fake["type"] = "intron_lc"
            hints.append(fake)

        return hints

    def _generate_noisy_hints(self, real_introns, exons, sequence):
        """Degraded RNA-seq simulation: mostly lc with more fakes"""

        hints     = []
        drop_rate = 0.35

        for intr in real_introns:
            if random.random() < drop_rate:
                continue

            if intron.drop_intron_by_anchor(intr, exons, self.config):
                continue

            if random.random() < 0.3:
                hint_type = "intron_hc"
            else:
                hint_type = "intron_lc"

            hints.append({
                "type":   hint_type,
                "start":  intr["start"],
                "end":    intr["end"],
                "strand": intr["strand"],
            })

        fake_count = max(2, int(len(real_introns) * 0.10))
        for _ in range(fake_count):
            strand = random.choice(["+", "-"])
            fake   = intron.generate_fake_intron(sequence, exons, self.config, strand)
            if fake:
                fake["type"] = "intron_lc"
                hints.append(fake)

        return hints

    def _generate_minimal_hints(self, real_introns, exons):
        """Very sparse hints: few introns only"""

        if not real_introns:
            return []

        keep_count = max(1, int(len(real_introns) * 0.15))
        selected   = random.sample(real_introns, min(keep_count, len(real_introns)))

        hints = []

        for intr in selected:
            hints.append({
                "type":   "intron_hc",
                "start":  intr["start"],
                "end":    intr["end"],
                "strand": intr["strand"],
            })

        return hints
