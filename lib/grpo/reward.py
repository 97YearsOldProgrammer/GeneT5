from collections import Counter

import lib.inference.output as output_lib


def exon_f1(parsed_genes, ref_features, sequence):
    """Compute exon DNA F1 between predicted genes and reference features"""

    pred_exon_dna = {exon_seq for gene in parsed_genes for exon_seq in gene.exons}
    ref_exon_dna  = set()
    for rf in ref_features:
        if rf.get("type", "exon").lower() != "exon":
            continue
        ref_exon_dna.add(sequence[rf["start"]:rf["end"]])

    if not ref_exon_dna:
        return 1.0 if not pred_exon_dna else 0.0

    tp = len(pred_exon_dna & ref_exon_dna)
    fp = len(pred_exon_dna - ref_exon_dna)
    fn = len(ref_exon_dna - pred_exon_dna)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def gene_f1(parsed_genes, ref_features, sequence):
    """Compute gene-structure F1 using exon DNA set signatures per gene"""

    pred_gene_map = {}
    for gi, gene in enumerate(parsed_genes):
        if gene.exons:
            pred_gene_map[gi] = set(gene.exons)

    ref_gene_map = {}
    for rf in ref_features:
        if rf.get("type", "exon").lower() != "exon":
            continue
        dna  = sequence[rf["start"]:rf["end"]]
        gkey = rf.get("gene_idx", rf.get("gene_id", 0))
        if gkey not in ref_gene_map:
            ref_gene_map[gkey] = set()
        ref_gene_map[gkey].add(dna)

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


def composite_reward(pred_text, ref_features, sequence, exon_weight=0.6, gene_weight=0.4):
    """Compute composite reward from predicted text and reference features"""

    parser       = output_lib.ModelOutputParser(strict=False)
    parsed_genes = parser.parse_sequence(pred_text)

    e_f1 = exon_f1(parsed_genes, ref_features, sequence)
    g_f1 = gene_f1(parsed_genes, ref_features, sequence)

    return exon_weight * e_f1 + gene_weight * g_f1


def batch_rewards(pred_texts, ref_features_list, sequences, exon_weight=0.6, gene_weight=0.4):
    """Compute rewards for a batch of predictions"""

    rewards = []
    for pred_text, ref_features, sequence in zip(pred_texts, ref_features_list, sequences):
        r = composite_reward(pred_text, ref_features, sequence, exon_weight, gene_weight)
        rewards.append(r)

    return rewards
