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
