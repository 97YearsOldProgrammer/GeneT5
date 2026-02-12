from collections import Counter

import lib.util._output as output_lib


def exon_f1(pred_features, ref_features):
    """Compute exon boundary F1 between predicted and reference features"""

    pred_coords = {(pf.start, pf.end) for pf in pred_features if pf.feature_type == "exon"}
    ref_coords  = {(rf["start"], rf["end"]) for rf in ref_features}

    if not ref_coords:
        return 1.0 if not pred_coords else 0.0

    tp = len(pred_coords & ref_coords)
    fp = len(pred_coords - ref_coords)
    fn = len(ref_coords - pred_coords)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def gene_f1(pred_features, ref_features):
    """Compute gene-structure F1 using exon-set signatures per gene"""

    pred_gene_map = {}
    for pf in pred_features:
        if pf.feature_type == "exon":
            gkey = pf.gene_idx
            if gkey not in pred_gene_map:
                pred_gene_map[gkey] = set()
            pred_gene_map[gkey].add((pf.start, pf.end))

    ref_gene_map = {}
    for rf in ref_features:
        gkey = rf.get("gene_idx", 0)
        if gkey not in ref_gene_map:
            ref_gene_map[gkey] = set()
        ref_gene_map[gkey].add((rf["start"], rf["end"]))

    pred_sigs = Counter(frozenset(exons) for exons in pred_gene_map.values() if exons)
    ref_sigs  = Counter(frozenset(exons) for exons in ref_gene_map.values() if exons)

    if not ref_sigs:
        return 1.0 if not pred_sigs else 0.0

    common = pred_sigs & ref_sigs
    tp     = sum(common.values())
    fp     = sum((pred_sigs - ref_sigs).values())
    fn     = sum((ref_sigs - pred_sigs).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def composite_reward(pred_text, ref_features, exon_weight=0.6, gene_weight=0.4):
    """Compute composite reward from predicted text and reference features"""

    parser          = output_lib.ModelOutputParser(strict=False)
    parsed_seqs     = parser.parse_sequence(pred_text)
    parsed_features = parsed_seqs[0] if parsed_seqs else []

    e_f1 = exon_f1(parsed_features, ref_features)
    g_f1 = gene_f1(parsed_features, ref_features)

    return exon_weight * e_f1 + gene_weight * g_f1


def batch_rewards(pred_texts, ref_features_list, exon_weight=0.6, gene_weight=0.4):
    """Compute rewards for a batch of predictions"""

    rewards = []
    for pred_text, ref_features in zip(pred_texts, ref_features_list):
        r = composite_reward(pred_text, ref_features, exon_weight, gene_weight)
        rewards.append(r)

    return rewards
