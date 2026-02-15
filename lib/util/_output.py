import re
import pathlib as pl

from dataclasses import dataclass, field


@dataclass
class ParsedGene:
    """Single gene parsed from model output"""

    strand: str
    exons:  list = field(default_factory=list)


class ModelOutputParser:
    """Parser for GeneT5 DNA-to-DNA output format"""

    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, strict=False):
        """Initialize parser"""

        self.strict = strict

    def parse_sequence(self, text):
        """Parse model output into list of ParsedGene"""

        text = text.strip()

        if text.startswith(self.BOS_TOKEN):
            text = text[len(self.BOS_TOKEN):]
        if text.endswith(self.EOS_TOKEN):
            text = text[:-len(self.EOS_TOKEN)]

        if not text:
            return []

        gene_blocks = re.split(r'(<\+>|<->)', text)

        genes   = []
        i       = 0
        while i < len(gene_blocks):
            block = gene_blocks[i].strip()

            if block in ("<+>", "<->"):
                strand   = "+" if block == "<+>" else "-"
                exon_dna = gene_blocks[i + 1] if i + 1 < len(gene_blocks) else ""
                exons    = [e for e in exon_dna.split("<exon>") if e.strip()]
                if exons:
                    genes.append(ParsedGene(strand=strand, exons=exons))
                i += 2
            else:
                i += 1

        return genes


def parse_model_output(text, strict=False):
    """Parse model output and return list of ParsedGene"""

    parser = ModelOutputParser(strict=strict)
    return parser.parse_sequence(text)


def locate_exon_in_input(input_sequence, exon_dna):
    """Find exon position in input sequence by exact match"""

    pos = input_sequence.find(exon_dna)
    if pos >= 0:
        return pos, pos + len(exon_dna)
    return None, None


def genes_to_gff3(genes, input_sequence, seqid="seq", source="GeneT5", offset=0):
    """Convert parsed genes to GFF3 lines by locating exon DNA in input"""

    lines      = []
    gene_count = 0

    for gene in genes:
        gene_count  += 1
        gene_id      = f"gene_{gene_count:04d}"
        exon_coords  = []

        for ei, exon_dna in enumerate(gene.exons, 1):
            start, end = locate_exon_in_input(input_sequence, exon_dna)
            if start is None:
                continue
            exon_coords.append((start + offset, end + offset))
            lines.append(
                f"{seqid}\t{source}\texon\t{start + offset}\t{end + offset}"
                f"\t.\t{gene.strand}\t.\tID={gene_id}.exon{ei};Parent={gene_id}"
            )

        if exon_coords:
            gene_start = min(s for s, e in exon_coords)
            gene_end   = max(e for s, e in exon_coords)
            lines.insert(-len(exon_coords),
                f"{seqid}\t{source}\tgene\t{gene_start}\t{gene_end}"
                f"\t.\t{gene.strand}\t.\tID={gene_id}"
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
