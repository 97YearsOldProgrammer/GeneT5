import random
import math
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


#####################
##### CONSTANTS #####
#####################


# Splice site consensus sequences
DONOR_SITE    = "GT"  # 5' splice site
ACCEPTOR_SITE = "AG"  # 3' splice site

# Feature types we care about
HINT_FEATURE_TYPES = {"exon", "cds", "CDS"}


####################
##### CONFIG   #####
####################


@dataclass
class NoisingConfig:
    """Configuration for biological noise parameters."""
    
    # Scenario mixing weights
    scenario_weights: Dict[str, float] = field(default_factory=lambda: {
        'full_mix':    0.40,   # Both intron + CDS hints
        'intron_only': 0.25,   # RNA-seq only
        'cds_only':    0.20,   # Protein only
        'degraded':    0.10,   # Heavy noise on both
        'ab_initio':   0.05,   # NO hints
    })
    
    # Intron parameters (RNA-seq origin)
    intron_drop_base:        float = 0.10
    intron_anchor_alpha:     float = 0.05
    intron_anchor_beta:      float = 0.40
    intron_anchor_gamma:     float = 0.15
    intron_hallucinate_rate: float = 0.03
    intron_min_length:       int   = 20
    intron_max_length:       int   = 50000
    
    # CDS parameters (protein alignment origin)
    cds_jitter_sigma:       float = 20.0
    cds_truncate_5_prob:    float = 0.12
    cds_truncate_3_prob:    float = 0.10
    cds_truncate_5_max:     int   = 60
    cds_truncate_3_max:     int   = 45
    cds_frameshift_prob:    float = 0.02
    cds_wrong_strand_prob:  float = 0.28
    
    # Exon parameters (coverage-based)
    exon_boundary_lambda:   float = 0.1
    exon_boundary_max:      int   = 50
    exon_merge_prob:        float = 0.02
    exon_drop_prob:         float = 0.05
    
    # Degraded scenario multipliers
    degraded_drop_mult:     float = 3.0
    degraded_noise_mult:    float = 2.0


###############################
##### INTRON COMPUTATION  #####
###############################


def compute_introns_from_exons(exons: List[Dict], strand: str) -> List[Dict]:
    """
    Compute intron positions from sorted exon list.
    
    Introns are the gaps between consecutive exons of the same transcript.
    
    Args:
        exons: List of exon features, sorted by start position
        strand: Strand of the transcript ('+' or '-')
    
    Returns:
        List of intron features (start, end, strand)
    """
    if len(exons) < 2:
        return []
    
    # Sort exons by start position
    sorted_exons = sorted(exons, key=lambda x: x["start"])
    
    introns = []
    for i in range(len(sorted_exons) - 1):
        exon1 = sorted_exons[i]
        exon2 = sorted_exons[i + 1]
        
        # Intron is the gap between exons
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


def group_exons_by_parent(features: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group exon features by their parent transcript/gene.
    
    Returns:
        Dict mapping parent_id to list of exon features
    """
    groups = {}
    
    for feat in features:
        if feat.get("type", "").lower() != "exon":
            continue
        
        parent = feat.get("attributes", {}).get("Parent", "unknown")
        if parent not in groups:
            groups[parent] = []
        groups[parent].append(feat)
    
    return groups


#######################
##### REAL INTRON #####
#######################


def drop_intron_by_anchor(intron: Dict, exons: List[Dict], config: NoisingConfig) -> bool:
    """
    Decide whether to drop an intron based on anchor length.
    
    RNA-seq splice junction detection fails when read anchors are too short.
    
    P(drop | L_anchor) = α + β * exp(-γ * L_anchor)
    
    Args:
        intron: Intron feature dict
        exons: List of flanking exons
        config: Noising configuration
    
    Returns:
        True if intron should be dropped
    """
    # Find flanking exons
    intron_start = intron["start"]
    intron_end   = intron["end"]
    
    upstream_anchor   = 0
    downstream_anchor = 0
    
    for exon in exons:
        # Upstream exon (ends at intron start - 1)
        if exon["end"] == intron_start - 1:
            upstream_anchor = exon["end"] - exon["start"] + 1
        # Downstream exon (starts at intron end + 1)
        if exon["start"] == intron_end + 1:
            downstream_anchor = exon["end"] - exon["start"] + 1
    
    # Use minimum anchor length
    L_anchor = min(upstream_anchor, downstream_anchor) if upstream_anchor and downstream_anchor else 10
    
    # Calculate drop probability
    alpha = config.intron_anchor_alpha
    beta  = config.intron_anchor_beta
    gamma = config.intron_anchor_gamma
    
    p_drop = alpha + beta * math.exp(-gamma * L_anchor)
    
    return random.random() < p_drop


def noise_real_introns(
    introns: List[Dict],
    exons: List[Dict],
    config: NoisingConfig,
    degraded: bool = False
) -> List[Dict]:
    """
    Apply RNA-seq style noise to real introns.
    
    Key properties:
    - NO boundary jitter (aligners snap to GT/AG)
    - Expression-dependent dropout
    - Anchor-length dependent dropout
    
    Args:
        introns: List of real intron features
        exons: List of exon features (for anchor calculation)
        config: Noising configuration
        degraded: Whether to apply extra degradation
    
    Returns:
        List of surviving intron hints
    """
    surviving = []
    
    drop_mult = config.degraded_drop_mult if degraded else 1.0
    
    for intron in introns:
        # Base dropout (expression-dependent simulation)
        if random.random() < config.intron_drop_base * drop_mult:
            continue
        
        # Anchor-length dependent dropout
        if drop_intron_by_anchor(intron, exons, config):
            continue
        
        # Intron survives - NO jitter applied
        surviving.append({
            "type":   "intron",
            "start":  intron["start"],
            "end":    intron["end"],
            "strand": intron["strand"],
        })
    
    return surviving


#######################
##### FAKE INTRON #####
#######################


def find_splice_sites(sequence: str, site_type: str = "donor") -> List[int]:
    """
    Find all GT (donor) or AG (acceptor) sites in sequence.
    
    Args:
        sequence: DNA sequence string
        site_type: 'donor' for GT, 'acceptor' for AG
    
    Returns:
        List of 0-based positions
    """
    pattern = DONOR_SITE if site_type == "donor" else ACCEPTOR_SITE
    positions = []
    
    seq_upper = sequence.upper()
    start = 0
    
    while True:
        pos = seq_upper.find(pattern, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    
    return positions


def generate_fake_intron(
    sequence: str,
    existing_features: List[Dict],
    config: NoisingConfig,
    strand: str = "+"
) -> Optional[Dict]:
    """
    Generate a plausible false positive intron.
    
    Requirements:
    1. Must have GT..AG consensus in DNA
    2. Should not overlap real exons
    3. Should have reasonable length
    
    Args:
        sequence: DNA sequence
        existing_features: List of real features to avoid
        config: Noising configuration
        strand: Strand for the fake intron
    
    Returns:
        Fake intron feature dict, or None if generation fails
    """
    # Find all GT and AG sites
    donors    = find_splice_sites(sequence, "donor")
    acceptors = find_splice_sites(sequence, "acceptor")
    
    if not donors or not acceptors:
        return None
    
    # Get real exon regions to avoid
    exon_regions = []
    for feat in existing_features:
        if feat.get("type", "").lower() == "exon":
            exon_regions.append((feat["start"], feat["end"]))
    
    # Try to find valid fake intron
    max_attempts = 50
    
    for _ in range(max_attempts):
        # Random GT site (convert to 1-based)
        donor_pos = random.choice(donors) + 1
        
        # Find AG sites downstream within valid range
        valid_acceptors = [
            a + 1 for a in acceptors
            if config.intron_min_length <= (a + 1 - donor_pos) <= config.intron_max_length
            and a > donor_pos
        ]
        
        if not valid_acceptors:
            continue
        
        acceptor_pos = random.choice(valid_acceptors)
        
        # Check overlap with exons
        overlaps = False
        for exon_start, exon_end in exon_regions:
            # Fake intron should be in intergenic or intronic region
            if not (acceptor_pos < exon_start or donor_pos > exon_end):
                overlaps = True
                break
        
        if overlaps:
            continue
        
        # Valid fake intron found
        return {
            "type":   "intron",
            "start":  donor_pos,
            "end":    acceptor_pos + 1,  # AG is 2bp
            "strand": strand,
            "fake":   True,
        }
    
    return None


def generate_fake_introns(
    sequence: str,
    real_introns: List[Dict],
    existing_features: List[Dict],
    config: NoisingConfig
) -> List[Dict]:
    """
    Generate hallucinated introns based on hallucination rate.
    
    Args:
        sequence: DNA sequence
        real_introns: List of real introns (for count calculation)
        existing_features: All features (for overlap avoidance)
        config: Noising configuration
    
    Returns:
        List of fake intron features
    """
    n_real = len(real_introns)
    n_fake = int(n_real * config.intron_hallucinate_rate)
    
    # At least try to generate some if we have real introns
    if n_real > 0 and n_fake == 0:
        n_fake = 1 if random.random() < config.intron_hallucinate_rate else 0
    
    fake_introns = []
    strands = ["+", "-"]
    
    for _ in range(n_fake):
        strand = random.choice(strands)
        fake = generate_fake_intron(sequence, existing_features, config, strand)
        if fake:
            fake_introns.append(fake)
    
    return fake_introns


####################
##### REAL CDS #####
####################


def apply_cds_jitter(cds: Dict, config: NoisingConfig, is_terminal: bool = False) -> Dict:
    """
    Apply boundary jitter to CDS feature.
    
    Protein alignments have fuzzy terminal boundaries.
    Internal jitter must be codon-aligned (multiple of 3).
    
    Args:
        cds: CDS feature dict
        config: Noising configuration
        is_terminal: Whether this is a terminal CDS (start/stop codon region)
    
    Returns:
        Jittered CDS feature
    """
    sigma = config.cds_jitter_sigma
    
    # Sample jitter
    jitter_start = int(random.gauss(0, sigma))
    jitter_end   = int(random.gauss(0, sigma))
    
    # Codon-align for internal boundaries
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


def apply_cds_truncation(cds: Dict, config: NoisingConfig, position: str) -> Dict:
    """
    Apply truncation to CDS (simulates incomplete protein alignment).
    
    Args:
        cds: CDS feature dict
        config: Noising configuration
        position: '5prime' or '3prime'
    
    Returns:
        Truncated CDS feature
    """
    result = cds.copy()
    
    if position == "5prime":
        max_trunc = min(config.cds_truncate_5_max, (cds["end"] - cds["start"]) // 2)
        if max_trunc > 3:
            trunc = random.randint(3, max_trunc)
            trunc = (trunc // 3) * 3  # Codon align
            result["start"] = cds["start"] + trunc
    else:
        max_trunc = min(config.cds_truncate_3_max, (cds["end"] - cds["start"]) // 2)
        if max_trunc > 3:
            trunc = random.randint(3, max_trunc)
            trunc = (trunc // 3) * 3
            result["end"] = cds["end"] - trunc
    
    return result


def apply_frameshift(cds: Dict, config: NoisingConfig) -> Dict:
    """
    Apply frameshift error to CDS.
    
    Args:
        cds: CDS feature dict
        config: Noising configuration
    
    Returns:
        Frameshifted CDS feature
    """
    shift = random.choice([-2, -1, 1, 2])
    
    result = cds.copy()
    # Apply shift to start (simulates insertion/deletion)
    result["start"] = max(1, cds["start"] + shift)
    
    return result


def noise_real_cds(
    cds_features: List[Dict],
    config: NoisingConfig,
    degraded: bool = False
) -> List[Dict]:
    """
    Apply protein alignment style noise to real CDS features.
    
    Key properties:
    - Boundary jitter (codon-aligned for internal)
    - 5' and 3' truncation
    - Rare frameshift errors
    
    Args:
        cds_features: List of CDS features
        config: Noising configuration
        degraded: Whether to apply extra degradation
    
    Returns:
        List of noised CDS hints
    """
    if not cds_features:
        return []
    
    noised = []
    noise_mult = config.degraded_noise_mult if degraded else 1.0
    
    # Sort by position
    sorted_cds = sorted(cds_features, key=lambda x: x["start"])
    
    for i, cds in enumerate(sorted_cds):
        is_first = (i == 0)
        is_last  = (i == len(sorted_cds) - 1)
        
        result = cds.copy()
        
        # Apply jitter
        result = apply_cds_jitter(result, config, is_terminal=(is_first or is_last))
        
        # 5' truncation (more likely for first CDS)
        if is_first and random.random() < config.cds_truncate_5_prob * noise_mult:
            result = apply_cds_truncation(result, config, "5prime")
        
        # 3' truncation (more likely for last CDS)
        if is_last and random.random() < config.cds_truncate_3_prob * noise_mult:
            result = apply_cds_truncation(result, config, "3prime")
        
        # Rare frameshift
        if random.random() < config.cds_frameshift_prob * noise_mult:
            result = apply_frameshift(result, config)
        
        noised.append(result)
    
    return noised


####################
##### FAKE CDS #####
####################


def generate_fake_cds(
    sequence: str,
    existing_features: List[Dict],
    config: NoisingConfig
) -> Optional[Dict]:
    """
    Generate a plausible false positive CDS.
    
    Look for ORF-like regions (ATG...stop) in non-coding regions.
    
    Args:
        sequence: DNA sequence
        existing_features: List of real features
        config: Noising configuration
    
    Returns:
        Fake CDS feature or None
    """
    # Find ATG start codons
    seq_upper = sequence.upper()
    atg_positions = []
    
    start = 0
    while True:
        pos = seq_upper.find("ATG", start)
        if pos == -1:
            break
        atg_positions.append(pos)
        start = pos + 1
    
    if not atg_positions:
        return None
    
    # Stop codons
    stop_codons = ["TAA", "TAG", "TGA"]
    
    # Get coding regions to avoid
    coding_regions = []
    for feat in existing_features:
        ftype = feat.get("type", "").lower()
        if ftype in {"cds", "exon"}:
            coding_regions.append((feat["start"], feat["end"]))
    
    # Try to find valid fake CDS
    for _ in range(30):
        atg_pos = random.choice(atg_positions)
        
        # Look for stop codon in same frame
        for offset in range(atg_pos + 3, min(atg_pos + 3000, len(sequence) - 2), 3):
            codon = seq_upper[offset:offset + 3]
            if codon in stop_codons:
                # Found potential ORF
                cds_start = atg_pos + 1  # 1-based
                cds_end   = offset + 3
                
                # Check if in non-coding region
                overlaps = False
                for cs, ce in coding_regions:
                    if not (cds_end < cs or cds_start > ce):
                        overlaps = True
                        break
                
                if not overlaps and cds_end - cds_start >= 60:  # Min 20 codons
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


#####################
##### REAL EXON #####
#####################


def apply_exon_boundary_extension(exon: Dict, config: NoisingConfig) -> Dict:
    """
    Apply boundary extension to exon (coverage bleeds into introns).
    
    Extension follows exponential distribution.
    
    Args:
        exon: Exon feature dict
        config: Noising configuration
    
    Returns:
        Extended exon feature
    """
    # Exponential extension
    ext_start = int(random.expovariate(config.exon_boundary_lambda))
    ext_end   = int(random.expovariate(config.exon_boundary_lambda))
    
    # Cap at maximum
    ext_start = min(ext_start, config.exon_boundary_max)
    ext_end   = min(ext_end, config.exon_boundary_max)
    
    return {
        "type":   "exon",
        "start":  max(1, exon["start"] - ext_start),
        "end":    exon["end"] + ext_end,
        "strand": exon["strand"],
    }


def noise_real_exons(
    exons: List[Dict],
    config: NoisingConfig,
    degraded: bool = False
) -> List[Dict]:
    """
    Apply coverage-based noise to real exons.
    
    Key properties:
    - Boundary extension (coverage bleed)
    - Occasional dropout
    - Merge adjacent exons (intron retention)
    
    Args:
        exons: List of exon features
        config: Noising configuration
        degraded: Whether to apply extra degradation
    
    Returns:
        List of noised exon hints
    """
    if not exons:
        return []
    
    noised = []
    drop_mult = config.degraded_drop_mult if degraded else 1.0
    
    sorted_exons = sorted(exons, key=lambda x: x["start"])
    
    i = 0
    while i < len(sorted_exons):
        exon = sorted_exons[i]
        
        # Dropout
        if random.random() < config.exon_drop_prob * drop_mult:
            i += 1
            continue
        
        # Merge with next exon (intron retention)
        if i < len(sorted_exons) - 1 and random.random() < config.exon_merge_prob:
            next_exon = sorted_exons[i + 1]
            merged = {
                "type":   "exon",
                "start":  exon["start"],
                "end":    next_exon["end"],
                "strand": exon["strand"],
            }
            noised.append(apply_exon_boundary_extension(merged, config))
            i += 2
            continue
        
        # Normal processing with boundary extension
        noised.append(apply_exon_boundary_extension(exon, config))
        i += 1
    
    return noised


#####################
##### FAKE EXON #####
#####################


def generate_fake_exon(
    sequence: str,
    existing_features: List[Dict],
    config: NoisingConfig
) -> Optional[Dict]:
    """
    Generate a plausible false positive exon.
    
    Place in intergenic regions with reasonable length.
    
    Args:
        sequence: DNA sequence
        existing_features: List of real features
        config: Noising configuration
    
    Returns:
        Fake exon feature or None
    """
    # Get existing feature regions
    feature_regions = []
    for feat in existing_features:
        feature_regions.append((feat["start"], feat["end"]))
    
    # Find gaps between features
    if not feature_regions:
        # No features - can place anywhere
        start = random.randint(1, max(1, len(sequence) - 500))
        length = random.randint(50, 300)
        return {
            "type":   "exon",
            "start":  start,
            "end":    min(start + length, len(sequence)),
            "strand": random.choice(["+", "-"]),
            "fake":   True,
        }
    
    # Sort regions
    sorted_regions = sorted(feature_regions)
    
    # Find gaps
    gaps = []
    prev_end = 0
    for start, end in sorted_regions:
        if start - prev_end > 200:
            gaps.append((prev_end + 1, start - 1))
        prev_end = end
    
    # Gap after last feature
    if len(sequence) - prev_end > 200:
        gaps.append((prev_end + 1, len(sequence)))
    
    if not gaps:
        return None
    
    # Pick random gap
    gap_start, gap_end = random.choice(gaps)
    gap_size = gap_end - gap_start
    
    if gap_size < 100:
        return None
    
    # Place fake exon in gap
    exon_length = random.randint(50, min(300, gap_size - 50))
    exon_start  = random.randint(gap_start, gap_end - exon_length)
    
    return {
        "type":   "exon",
        "start":  exon_start,
        "end":    exon_start + exon_length,
        "strand": random.choice(["+", "-"]),
        "fake":   True,
    }


###############################
##### SCENARIO MIXING     #####
###############################


def select_scenario(config: NoisingConfig) -> str:
    """
    Randomly select a training scenario based on weights.
    
    Returns:
        Scenario name: 'full_mix', 'intron_only', 'cds_only', 'degraded', or 'ab_initio'
    """
    weights = config.scenario_weights
    scenarios = list(weights.keys())
    probs = [weights[s] for s in scenarios]
    
    return random.choices(scenarios, weights=probs, k=1)[0]


###############################
##### MAIN NOISER CLASS   #####
###############################


class GFFNoiser:
    """
    Main noiser class that applies biological noise to GFF annotations.
    
    Generates noised hints for model training, forcing the model to learn
    when to trust hints vs DNA signals.
    """
    
    def __init__(self, config: Optional[NoisingConfig] = None):
        """
        Initialize the noiser.
        
        Args:
            config: Noising configuration (uses defaults if None)
        """
        self.config = config or NoisingConfig()
    
    def extract_features_by_type(
        self, 
        features: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract exons and CDS features from feature list.
        
        Args:
            features: List of all GFF features
        
        Returns:
            Tuple of (exon_features, cds_features)
        """
        exons = []
        cds   = []
        
        for feat in features:
            ftype = feat.get("type", "").lower()
            if ftype == "exon":
                exons.append(feat)
            elif ftype == "cds":
                cds.append(feat)
        
        return exons, cds
    
    def compute_all_introns(self, features: List[Dict]) -> List[Dict]:
        """
        Compute all introns from exon features.
        
        Groups exons by parent and computes introns for each transcript.
        
        Args:
            features: List of all GFF features
        
        Returns:
            List of computed intron features
        """
        all_introns = []
        
        # Group exons by parent
        exon_groups = group_exons_by_parent(features)
        
        for parent_id, exons in exon_groups.items():
            if len(exons) < 2:
                continue
            
            # Get strand from first exon
            strand = exons[0].get("strand", "+")
            
            # Compute introns
            introns = compute_introns_from_exons(exons, strand)
            all_introns.extend(introns)
        
        return all_introns
    
    def noise_features(
        self,
        features: List[Dict],
        sequence: str,
        scenario: Optional[str] = None
    ) -> Tuple[List[Dict], str, Dict[str, Any]]:
        """
        Apply biological noise to features based on scenario.
        
        Args:
            features: List of original GFF features
            sequence: DNA sequence for fake feature generation
            scenario: Specific scenario (random if None)
        
        Returns:
            Tuple of (noised_hints, scenario_name, noise_log)
        """
        # Select scenario
        if scenario is None:
            scenario = select_scenario(self.config)
        
        # Extract feature types
        exons, cds_features = self.extract_features_by_type(features)
        introns = self.compute_all_introns(features)
        
        # Initialize result
        hints = []
        noise_log = {
            "scenario":       scenario,
            "original_exons": len(exons),
            "original_cds":   len(cds_features),
            "computed_introns": len(introns),
        }
        
        degraded = (scenario == "degraded")
        
        # Apply noise based on scenario
        if scenario == "ab_initio":
            # No hints at all
            noise_log["hint_introns"] = 0
            noise_log["hint_exons"]   = 0
            noise_log["hint_cds"]     = 0
            return [], scenario, noise_log
        
        if scenario in ("full_mix", "intron_only", "degraded"):
            # Add intron hints
            noised_introns = noise_real_introns(introns, exons, self.config, degraded)
            fake_introns   = generate_fake_introns(sequence, introns, features, self.config)
            hints.extend(noised_introns)
            hints.extend(fake_introns)
            noise_log["hint_introns"]      = len(noised_introns)
            noise_log["fake_introns"]      = len(fake_introns)
        
        if scenario in ("full_mix", "cds_only", "degraded"):
            # Add CDS hints
            noised_cds = noise_real_cds(cds_features, self.config, degraded)
            hints.extend(noised_cds)
            noise_log["hint_cds"] = len(noised_cds)
            
            # Add exon hints (often from same source as CDS)
            noised_exons = noise_real_exons(exons, self.config, degraded)
            hints.extend(noised_exons)
            noise_log["hint_exons"] = len(noised_exons)
        
        return hints, scenario, noise_log
    
    def format_hints_for_input(self, hints: List[Dict], hint_token: str = "[HIT]") -> str:
        """
        Format hints as input string for the model.
        
        Format: [HIT] type start end strand
        
        Args:
            hints: List of hint features
            hint_token: Token to mark hint section
        
        Returns:
            Formatted hint string
        """
        if not hints:
            return ""
        
        lines = [hint_token]
        
        for hint in sorted(hints, key=lambda x: x["start"]):
            ftype  = hint["type"].lower()
            start  = hint["start"]
            end    = hint["end"]
            strand = hint.get("strand", "+")
            
            # Format: type start end strand
            lines.append(f"{ftype}\t{start}\t{end}\t{strand}")
        
        return "\n".join(lines)