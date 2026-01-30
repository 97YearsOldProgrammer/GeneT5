from dataclasses import dataclass, field
import re


@dataclass
class GeneStructure:
    """
    Represents a gene with its exons and CDS regions.
    """

    gene_id: str
    seqid:   str
    strand:  str
    start:   int
    end:     int
    exons:   list = field(default_factory=list)
    cds:     list = field(default_factory=list)


def parse_gff(gff_path):
    """
    Parse a GFF3 file into a list of feature dictionaries.

    Args:
        gff_path: Path to GFF3 file

    Returns:
        List of dicts with keys: seqid, source, type, start, end,
        score, strand, phase, attributes
    """

    features = []

    with open(gff_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) != 9:
                continue

            attrs = _parse_attributes(parts[8])

            feature = {
                'seqid':      parts[0],
                'source':     parts[1],
                'type':       parts[2],
                'start':      int(parts[3]),
                'end':        int(parts[4]),
                'score':      parts[5] if parts[5] != '.' else None,
                'strand':     parts[6],
                'phase':      int(parts[7]) if parts[7] != '.' else None,
                'attributes': attrs
            }
            features.append(feature)

    return features


def _parse_attributes(attr_string):
    """
    Parse GFF3 attribute string (col 9) into a dictionary.
    """

    attrs = {}
    for item in attr_string.split(';'):
        item = item.strip()
        if not item:
            continue
        if '=' in item:
            key, value = item.split('=', 1)
            attrs[key] = value
    return attrs


def extract_features_by_type(features, feature_type="exon"):
    """
    Extract features of a specific type grouped by sequence ID.

    Args:
        features:     List of feature dicts from parse_gff
        feature_type: GFF feature type to extract (exon, CDS, gene, etc.)

    Returns:
        Dict mapping seqid -> list of (start, end) coordinates
    """

    result = {}

    for feat in features:
        if feat['type'] == feature_type:
            seqid = feat['seqid']
            if seqid not in result:
                result[seqid] = []
            result[seqid].append((feat['start'], feat['end']))

    for seqid in result:
        result[seqid].sort()

    return result


def build_gene_structure(features):
    """
    Build gene structures from GFF features.

    Args:
        features: List of feature dicts from parse_gff

    Returns:
        Dict mapping gene_id -> GeneStructure
    """

    genes = {}

    for feat in features:
        if feat['type'] == 'gene':
            gene_id = feat['attributes'].get('ID', '')
            if gene_id:
                genes[gene_id] = GeneStructure(
                    gene_id = gene_id,
                    seqid   = feat['seqid'],
                    strand  = feat['strand'],
                    start   = feat['start'],
                    end     = feat['end']
                )

    for feat in features:
        parent     = feat['attributes'].get('Parent', '')
        parent_ids = parent.split(',') if parent else []

        for parent_id in parent_ids:
            gene_id = _find_gene_parent(parent_id, features, genes)

            if gene_id and gene_id in genes:
                coords = (feat['start'], feat['end'])
                if feat['type'] == 'exon':
                    if coords not in genes[gene_id].exons:
                        genes[gene_id].exons.append(coords)
                elif feat['type'] == 'CDS':
                    if coords not in genes[gene_id].cds:
                        genes[gene_id].cds.append(coords)

    for gene in genes.values():
        gene.exons.sort()
        gene.cds.sort()

    return genes


def _find_gene_parent(feature_id, features, genes):
    """
    Recursively find the gene ID that is the ancestor of a feature.
    """

    if feature_id in genes:
        return feature_id

    for feat in features:
        if feat['attributes'].get('ID') == feature_id:
            parent = feat['attributes'].get('Parent', '')
            if parent:
                return _find_gene_parent(parent.split(',')[0], features, genes)

    return None
