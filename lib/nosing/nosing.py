import random
from dataclasses import dataclass, field


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
        exons = []
        cds   = []
        
        for feat in features:
            ftype = feat.get("type", "").lower()
            if ftype == "exon":
                exons.append(feat)
            elif ftype == "cds":
                cds.append(feat)
        
        return exons, cds
    
    def compute_introns_from_exons(self, exons, strand):
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
    
    def apply_exon_noise(self, exon, degraded=False):
        """Apply boundary jitter to exon"""
        drop_mult = self.config.degraded_drop_mult if degraded else 1.0
        
        if random.random() < self.config.exon_drop_prob * drop_mult:
            return None
        
        ext_start = int(random.expovariate(self.config.exon_boundary_lambda))
        ext_end   = int(random.expovariate(self.config.exon_boundary_lambda))
        ext_start = min(ext_start, self.config.exon_boundary_max)
        ext_end   = min(ext_end, self.config.exon_boundary_max)
        
        return {
            "type":   "exon",
            "start":  max(1, exon["start"] - ext_start),
            "end":    exon["end"] + ext_end,
            "strand": exon["strand"],
        }
    
    def apply_cds_noise(self, cds, is_terminal=False, degraded=False):
        """Apply jitter and truncation to CDS"""
        noise_mult = self.config.degraded_noise_mult if degraded else 1.0
        sigma      = self.config.cds_jitter_sigma
        
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
    
    def noise_features(self, features, sequence, scenario=None):
        """
        Apply noise to features based on scenario
        
        Returns: (hints, scenario, noise_log)
        """
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
        
        if scenario == "ab_initio":
            noise_log["hint_introns"] = 0
            noise_log["hint_exons"]   = 0
            noise_log["hint_cds"]     = 0
            return [], scenario, noise_log
        
        if scenario in ("full_mix", "intron_only", "degraded"):
            strand  = exons[0]["strand"] if exons else "+"
            introns = self.compute_introns_from_exons(exons, strand)
            
            for intron in introns:
                if random.random() < self.config.intron_drop_base * (self.config.degraded_drop_mult if degraded else 1.0):
                    continue
                hints.append(intron)
            
            noise_log["hint_introns"] = len([h for h in hints if h["type"] == "intron"])
        
        if scenario in ("full_mix", "cds_only", "degraded"):
            for exon in exons:
                noised = self.apply_exon_noise(exon, degraded)
                if noised:
                    hints.append(noised)
            
            sorted_cds = sorted(cds_features, key=lambda x: x["start"])
            for i, cds in enumerate(sorted_cds):
                is_terminal = (i == 0 or i == len(sorted_cds) - 1)
                noised      = self.apply_cds_noise(cds, is_terminal, degraded)
                hints.append(noised)
            
            noise_log["hint_exons"] = len([h for h in hints if h["type"] == "exon"])
            noise_log["hint_cds"]   = len([h for h in hints if h["type"] == "CDS"])
        
        return hints, scenario, noise_log
    
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