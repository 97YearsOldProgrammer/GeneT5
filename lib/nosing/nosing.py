import random
from dataclasses import dataclass, field

from lib.nosing import _intron
from lib.nosing import _exon
from lib.nosing import _cds
from lib.nosing import _config


##################
##### Config #####
##################


@dataclass
class NoisingConfig:
    
    scenario_weights = field(default_factory=lambda: {
        'full_mix':    0.40,
        'intron_only': 0.25,
        'cds_only':    0.20,
        'degraded':    0.10,
        'ab_initio':   0.05,
    })
    
    intron_drop_base        = 0.10
    intron_anchor_alpha     = 0.05
    intron_anchor_beta      = 0.40
    intron_anchor_gamma     = 0.15
    intron_hallucinate_rate = 0.03
    intron_min_length       = 20
    intron_max_length       = 50000
    
    cds_jitter_sigma       = 20.0
    cds_truncate_5_prob    = 0.12
    cds_truncate_3_prob    = 0.10
    cds_truncate_5_max     = 60
    cds_truncate_3_max     = 45
    cds_frameshift_prob    = 0.02
    cds_wrong_strand_prob  = 0.28
    
    exon_boundary_lambda = 0.1
    exon_boundary_max    = 50
    exon_merge_prob      = 0.02
    exon_drop_prob       = 0.05
    
    degraded_drop_mult  = 3.0
    degraded_noise_mult = 2.0

def select_scenario(config):

    weights   = config.scenario_weights
    scenarios = list(weights.keys())
    probs     = [weights[s] for s in scenarios]
    return random.choices(scenarios, weights=probs, k=1)[0]


##################
##### Nosier #####
##################


class GFFNoiser:
    
    def __init__(self, config=None):
        self.config = config or _config.NoisingConfig()
    
    def extract_features_by_type(self, features):

        exons = []
        cds   = []
        
        for feat in features:
            ftype = feat.get("type", "").lower()
            if ftype == "exon":
                exons.append(feat)
            elif ftype == "cds":
                cds.append(feat)
        
        return exons, cds
    
    def compute_all_introns(self, features):

        all_introns = []
        exon_groups = _intron.group_exons_by_parent(features)
        
        for parent_id, exons in exon_groups.items():
            if len(exons) < 2:
                continue
            
            strand  = exons[0].get("strand", "+")
            introns = _intron.compute_introns_from_exons(exons, strand)
            all_introns.extend(introns)
        
        return all_introns
    
    def noise_features(self, features, sequence, scenario=None):

        if scenario is None:
            scenario = _config.select_scenario(self.config)
        
        exons, cds_features = self.extract_features_by_type(features)
        introns             = self.compute_all_introns(features)
        hints               = []
        
        noise_log = {
            "scenario":         scenario,
            "original_exons":   len(exons),
            "original_cds":     len(cds_features),
            "computed_introns": len(introns),
        }
        
        degraded = (scenario == "degraded")
        
        if scenario == "ab_initio":
            noise_log["hint_introns"] = 0
            noise_log["hint_exons"]   = 0
            noise_log["hint_cds"]     = 0
            return [], scenario, noise_log
        
        if scenario in ("full_mix", "intron_only", "degraded"):
            noised_introns = _intron.noise_real_introns(introns, exons, self.config, degraded)
            fake_introns   = _intron.generate_fake_introns(sequence, introns, features, self.config)
            hints.extend(noised_introns)
            hints.extend(fake_introns)
            noise_log["hint_introns"] = len(noised_introns)
            noise_log["fake_introns"] = len(fake_introns)
        
        if scenario in ("full_mix", "cds_only", "degraded"):
            noised_cds = _cds.noise_real_cds(cds_features, self.config, degraded)
            hints.extend(noised_cds)
            noise_log["hint_cds"] = len(noised_cds)
            
            noised_exons = _exon.noise_real_exons(exons, self.config, degraded)
            hints.extend(noised_exons)
            noise_log["hint_exons"] = len(noised_exons)
        
        return hints, scenario, noise_log
    
    def format_hints_for_input(self, hints, hint_token="[HIT]"):

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