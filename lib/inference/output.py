import re

from dataclasses import dataclass, field


@dataclass
class ParsedExon:
    """Single exon or UTR parsed from model output"""

    kind: str
    dna:  str


class ModelOutputParser:
    """Parser for GeneT5 diffusion output format (flat exon/UTR list)"""

    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, strict=False):

        self.strict = strict

    def parse_sequence(self, text):
        """Parse model output into list of ParsedExon"""

        text = text.strip()

        if text.startswith(self.BOS_TOKEN):
            text = text[len(self.BOS_TOKEN):]
        if text.endswith(self.EOS_TOKEN):
            text = text[:-len(self.EOS_TOKEN)]

        if not text:
            return []

        # Split on <exon> and <UTR> tokens, keeping delimiters
        parts  = re.split(r'(<exon>|<UTR>)', text)
        result = []

        i = 0
        while i < len(parts):
            token = parts[i].strip()

            if token == "<exon>":
                dna = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if dna:
                    result.append(ParsedExon(kind="exon", dna=dna))
                i += 2
            elif token == "<UTR>":
                dna = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if dna:
                    result.append(ParsedExon(kind="utr", dna=dna))
                i += 2
            else:
                # Leading DNA before first delimiter (treat as exon)
                if token:
                    result.append(ParsedExon(kind="exon", dna=token))
                i += 1

        return result


def parse_model_output(text, strict=False):
    """Parse model output and return list of ParsedExon"""

    parser = ModelOutputParser(strict=strict)
    return parser.parse_sequence(text)


def locate_exon_in_input(input_sequence, exon_dna):
    """Find exon position in input sequence by exact match"""

    pos = input_sequence.find(exon_dna)
    if pos >= 0:
        return pos, pos + len(exon_dna)
    return None, None


def features_to_gff3(features, input_sequence, seqid="seq", source="GeneT5", offset=0):
    """Convert parsed features to GFF3 lines by locating DNA in input"""

    lines      = []
    exon_count = 0

    for feat in features:
        start, end = locate_exon_in_input(input_sequence, feat.dna)
        if start is None:
            continue

        exon_count += 1
        feat_type   = "CDS" if feat.kind == "exon" else "UTR"
        feat_id     = f"{feat_type.lower()}_{exon_count:04d}"

        lines.append(
            f"{seqid}\t{source}\t{feat_type}\t{start + offset}\t{end + offset}"
            f"\t.\t.\t.\tID={feat_id}"
        )

    return lines


def write_gff3(lines, output_path, append=False):
    """Write GFF3 lines to file"""

    mode = "a" if append else "w"

    with open(output_path, mode) as f:
        if not append:
            f.write("##gff-version 3\n")
        for line in lines:
            f.write(line + "\n")


#############################
#####  Codon Translation ####
#############################

_CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def _translate_dna(dna):
    """Translate DNA to protein using standard genetic code"""

    dna    = dna.upper().replace('U', 'T')
    codons = [dna[i:i+3] for i in range(0, len(dna) - 2, 3)]
    aa     = []

    for codon in codons:
        if len(codon) < 3:
            break
        residue = _CODON_TABLE.get(codon, 'X')
        if residue == '*':
            break
        aa.append(residue)

    return ''.join(aa)


################################
#####  Locating Features   #####
################################


def _locate_features(features, input_sequence, offset=0):
    """Locate all features in input and return (feat, start, end) triples"""

    located = []
    count   = 0

    for feat in features:
        start, end = locate_exon_in_input(input_sequence, feat.dna)
        if start is None:
            continue
        count += 1
        located.append((feat, start + offset, end + offset, count))

    return located


#############################
#####  FASTA Output     #####
#############################


def features_to_fasta(features, input_sequence, seqid="seq"):
    """Extract exon/UTR DNA sequences as FASTA records"""

    records = []
    located = _locate_features(features, input_sequence)

    for feat, start, end, idx in located:
        feat_type = "CDS" if feat.kind == "exon" else "UTR"
        header    = f"{seqid}_{feat_type.lower()}_{idx:04d} {start}-{end} {feat_type}"
        records.append((header, feat.dna))

    return records


def features_to_protein(features, input_sequence, seqid="seq"):
    """Translate CDS features to protein FASTA records"""

    records = []
    located = _locate_features(features, input_sequence)

    for feat, start, end, idx in located:
        if feat.kind != "exon":
            continue
        protein = _translate_dna(feat.dna)
        if not protein:
            continue
        header = f"{seqid}_protein_{idx:04d} {start}-{end} translated_CDS"
        records.append((header, protein))

    return records


def write_fasta(records, output_path, line_width=80):
    """Write list of (header, sequence) tuples as FASTA"""

    with open(output_path, 'w') as f:
        for header, seq in records:
            f.write(f">{header}\n")
            for i in range(0, len(seq), line_width):
                f.write(seq[i:i+line_width] + "\n")


#############################
#####  GTF Output       #####
#############################


def features_to_gtf(features, input_sequence, seqid="seq", source="GeneT5"):
    """Convert parsed features to GTF lines"""

    lines   = []
    located = _locate_features(features, input_sequence)
    gene_id = f"{seqid}_gene"
    tx_id   = f"{seqid}_tx"

    for feat, start, end, idx in located:
        feat_type = "CDS" if feat.kind == "exon" else "UTR"
        attrs     = f'gene_id "{gene_id}"; transcript_id "{tx_id}"; exon_number "{idx}";'

        # GTF uses 1-based inclusive coordinates
        lines.append(
            f"{seqid}\t{source}\t{feat_type}\t{start + 1}\t{end}"
            f"\t.\t.\t.\t{attrs}"
        )

    return lines


def write_gtf(lines, output_path):
    """Write GTF lines to file"""

    with open(output_path, 'w') as f:
        for line in lines:
            f.write(line + "\n")


#############################
#####  BED Output       #####
#############################


def features_to_bed(features, input_sequence, seqid="seq"):
    """Convert parsed features to BED6 lines (0-based half-open)"""

    lines   = []
    located = _locate_features(features, input_sequence)

    for feat, start, end, idx in located:
        feat_type = "CDS" if feat.kind == "exon" else "UTR"
        name      = f"{feat_type.lower()}_{idx:04d}"
        lines.append(f"{seqid}\t{start}\t{end}\t{name}\t0\t.")

    return lines


def write_bed(lines, output_path):
    """Write BED lines to file"""

    with open(output_path, 'w') as f:
        for line in lines:
            f.write(line + "\n")


#############################
#####  Format Dispatch  #####
#############################

FORMAT_WRITERS = {
    'gff3':    (features_to_gff3,    write_gff3,  '.gff3'),
    'fasta':   (features_to_fasta,   write_fasta, '.fa'),
    'protein': (features_to_protein, write_fasta, '.protein.fa'),
    'gtf':     (features_to_gtf,     write_gtf,   '.gtf'),
    'bed':     (features_to_bed,     write_bed,   '.bed'),
}


def write_all_formats(features, input_sequence, seqid, output_dir,
                      formats=None, source="GeneT5", offset=0):
    """Write features in multiple formats, return dict of {fmt: path}"""

    import pathlib as pl

    formats    = formats or ['gff3']
    output_dir = pl.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for fmt in formats:
        if fmt not in FORMAT_WRITERS:
            continue

        converter, writer, ext = FORMAT_WRITERS[fmt]

        if fmt == 'gff3':
            data = converter(features, input_sequence, seqid=seqid,
                             source=source, offset=offset)
        elif fmt == 'gtf':
            data = converter(features, input_sequence, seqid=seqid,
                             source=source)
        else:
            data = converter(features, input_sequence, seqid=seqid)

        path = output_dir / f"{seqid}{ext}"
        writer(data, path)
        paths[fmt] = path

    return paths
