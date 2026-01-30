from .gff_parser import parse_gff, extract_features_by_type, build_gene_structure


def _coords_to_positions(coords_list):
    """
    Convert list of (start, end) coords to set of positions.
    """

    positions = set()
    for start, end in coords_list:
        positions.update(range(start, end + 1))
    return positions


def _calculate_metrics(tp, fp, fn):
    """
    Calculate precision, recall (sensitivity), and F1.
    """

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "sensitivity": recall,
        "precision":   precision,
        "f1":          f1,
        "tp":          tp,
        "fp":          fp,
        "fn":          fn
    }


def nucleotide_metrics(ref_gff, pred_gff):
    """
    Calculate nucleotide-level metrics.

    Compares coding base positions between reference and predicted GFF files.

    Args:
        ref_gff:  Path to reference GFF file
        pred_gff: Path to predicted GFF file

    Returns:
        Dict with sensitivity, precision, f1, tp, fp, fn
    """

    ref_features  = parse_gff(ref_gff)
    pred_features = parse_gff(pred_gff)

    ref_cds  = extract_features_by_type(ref_features, "CDS")
    pred_cds = extract_features_by_type(pred_features, "CDS")

    if not ref_cds:
        ref_cds = extract_features_by_type(ref_features, "exon")
    if not pred_cds:
        pred_cds = extract_features_by_type(pred_features, "exon")

    all_seqids = set(ref_cds.keys()) | set(pred_cds.keys())

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for seqid in all_seqids:
        ref_coords  = ref_cds.get(seqid, [])
        pred_coords = pred_cds.get(seqid, [])

        ref_positions  = _coords_to_positions(ref_coords)
        pred_positions = _coords_to_positions(pred_coords)

        tp = len(ref_positions & pred_positions)
        fp = len(pred_positions - ref_positions)
        fn = len(ref_positions - pred_positions)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    return _calculate_metrics(total_tp, total_fp, total_fn)


def exon_metrics(ref_gff, pred_gff):
    """
    Calculate exon-level metrics.

    Both exon boundaries must match exactly for a true positive.

    Args:
        ref_gff:  Path to reference GFF file
        pred_gff: Path to predicted GFF file

    Returns:
        Dict with sensitivity, precision, f1, tp, fp, fn
    """

    ref_features  = parse_gff(ref_gff)
    pred_features = parse_gff(pred_gff)

    ref_exons  = extract_features_by_type(ref_features, "exon")
    pred_exons = extract_features_by_type(pred_features, "exon")

    all_seqids = set(ref_exons.keys()) | set(pred_exons.keys())

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for seqid in all_seqids:
        ref_set  = set(ref_exons.get(seqid, []))
        pred_set = set(pred_exons.get(seqid, []))

        tp = len(ref_set & pred_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    return _calculate_metrics(total_tp, total_fp, total_fn)


def gene_metrics(ref_gff, pred_gff):
    """
    Calculate gene-level metrics.

    Complete gene structure (all exons) must match for a true positive.

    Args:
        ref_gff:  Path to reference GFF file
        pred_gff: Path to predicted GFF file

    Returns:
        Dict with sensitivity, precision, f1, tp, fp, fn
    """

    ref_features  = parse_gff(ref_gff)
    pred_features = parse_gff(pred_gff)

    ref_genes  = build_gene_structure(ref_features)
    pred_genes = build_gene_structure(pred_features)

    def gene_signature(gene):
        return (gene.seqid, gene.strand, tuple(gene.exons))

    ref_signatures  = set(gene_signature(g) for g in ref_genes.values() if g.exons)
    pred_signatures = set(gene_signature(g) for g in pred_genes.values() if g.exons)

    tp = len(ref_signatures & pred_signatures)
    fp = len(pred_signatures - ref_signatures)
    fn = len(ref_signatures - pred_signatures)

    return _calculate_metrics(tp, fp, fn)


def calculate_f1(ref_gff, pred_gff, level="exon"):
    """
    Calculate F1 score at specified level.

    Args:
        ref_gff:  Path to reference GFF file
        pred_gff: Path to predicted GFF file
        level:    Comparison level - "nucleotide", "exon", or "gene"

    Returns:
        Dict with sensitivity, precision, f1, tp, fp, fn

    Raises:
        ValueError: If level is not one of nucleotide, exon, gene
    """

    level = level.lower()

    if level == "nucleotide":
        return nucleotide_metrics(ref_gff, pred_gff)
    elif level == "exon":
        return exon_metrics(ref_gff, pred_gff)
    elif level == "gene":
        return gene_metrics(ref_gff, pred_gff)
    else:
        raise ValueError(f"Unknown level: {level}. Use 'nucleotide', 'exon', or 'gene'")
