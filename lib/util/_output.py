import re
import json
import pathlib as pl

from dataclasses import dataclass, field


####################
#####  Format  #####
####################


@dataclass
class GFFFeature:
    """Represents a single GFF3 feature"""
    
    seqid:      str
    source:     str
    type:       str
    start:      int
    end:        int
    score:      str            = "."
    strand:     str            = "+"
    phase:      str            = "."
    attributes: dict           = field(default_factory=dict)
    
    def to_gff_line(self):
        """Convert to GFF3 line format"""
        
        attr_str = ";".join(f"{k}={v}" for k, v in self.attributes.items())
        return f"{self.seqid}\t{self.source}\t{self.type}\t{self.start}\t{self.end}\t{self.score}\t{self.strand}\t{self.phase}\t{attr_str}"
    
    def __str__(self):
        return self.to_gff_line()


@dataclass
class ParsedFeature:
    """Intermediate parsed feature from model output"""
    
    feature_type:   str
    start:          int
    end:            int
    strand:         str
    phase:          str
    biotype:        str
    gene_idx:       int
    transcript_idx: int  = None
    raw_line:       str  = ""


#####################
#####  Wrapper  #####
#####################


class ModelOutputParser:
    """Parser for GeneT5 model output format (compressed: no type prefix)"""

    # Compressed format: {start}[\t]{end}{strand}{phase}{biotype}{gene_idx}[[\t]{cds_coord}]
    # Non-UTR: 100[\t]200+.protein_coding1
    # UTR:     100[\t]200+0protein_coding1[\t]150
    FEATURE_PATTERN = re.compile(
        r'^(?P<start>\d+)'
        r'\[\\t\]'
        r'(?P<end>\d+)'
        r'(?P<strand>[+\-])'
        r'(?P<phase>[.\d])'
        r'(?P<biotype>[a-zA-Z_]+)'
        r'(?P<gene_idx>\d+)'
        r'(?:\.(?P<transcript_idx>\d+))?'
        r'(?:\[\\t\](?P<cds_coord>\d+))?$'
    )

    # Legacy format with type prefix (for backwards compatibility)
    FEATURE_PATTERN_LEGACY = re.compile(
        r'^(?P<type>[a-zA-Z_]+)'
        r'(?P<start>\d+)'
        r'(?:\[\\t\]|\t|\s+)'
        r'(?P<end>\d+)'
        r'(?P<strand>[+\-])'
        r'(?P<phase>[.\d])'
        r'(?P<biotype>[a-zA-Z_]+)'
        r'(?P<gene_idx>\d+)'
        r'(?:\.(?P<transcript_idx>\d+))?$'
    )

    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    
    def __init__(self, strict=False):
        """Initialize parser with optional strict mode"""

        self.strict = strict

    def parse_line(self, line):
        """Parse a single line from model output"""

        line = line.strip()

        if not line or line in (self.BOS_TOKEN, self.EOS_TOKEN, "<BOS>", "<EOS>"):
            return None

        # Try compressed format first (no type prefix)
        match = self.FEATURE_PATTERN.match(line)
        if match:
            groups = match.groupdict()
            return ParsedFeature(
                feature_type   = "exon",
                start          = int(groups["start"]),
                end            = int(groups["end"]),
                strand         = groups["strand"],
                phase          = groups["phase"],
                biotype        = groups["biotype"],
                gene_idx       = int(groups["gene_idx"]),
                transcript_idx = int(groups["transcript_idx"]) if groups.get("transcript_idx") else None,
                raw_line       = line,
            )

        # Try legacy format (with type prefix)
        match = self.FEATURE_PATTERN_LEGACY.match(line)
        if match:
            groups = match.groupdict()
            return ParsedFeature(
                feature_type   = groups["type"].lower(),
                start          = int(groups["start"]),
                end            = int(groups["end"]),
                strand         = groups["strand"],
                phase          = groups["phase"],
                biotype        = groups["biotype"],
                gene_idx       = int(groups["gene_idx"]),
                transcript_idx = int(groups["transcript_idx"]) if groups.get("transcript_idx") else None,
                raw_line       = line,
            )

        if self.strict:
            raise ValueError(f"Failed to parse line: {line}")
        return None
    
    def parse_sequence(self, text):
        """Parse complete model output with multiple sequences"""

        sequences   = []
        current_seq = []

        # Split on [\n] token (new format) or actual newline (legacy)
        if r'[\n]' in text:
            lines = text.strip().split(r'[\n]')
        else:
            lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Handle <bos> - may be standalone or attached to first feature
            if line.lower().startswith(("<bos>", "<BOS>")):
                if current_seq:
                    sequences.append(current_seq)
                current_seq = []
                # Strip <bos> prefix if attached to content
                if len(line) > 5:
                    line = line[5:]
                else:
                    continue

            # Handle <eos>
            if line.lower() in ("<eos>", "<EOS>", "eos"):
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = []
                continue

            if not line:
                continue

            feature = self.parse_line(line)
            if feature:
                current_seq.append(feature)

        if current_seq:
            sequences.append(current_seq)

        return sequences


class GFFConverter:
    """Converts parsed model output to GFF3 format"""

    def __init__(self, seqid="seq", source="GeneT5", offset=0, score=".", id_prefix="gene", include_introns=False):
        """Initialize converter with output settings"""

        self.seqid          = seqid
        self.source         = source
        self.offset         = offset
        self.score          = score
        self.id_prefix      = id_prefix
        self.include_introns = include_introns
    
    def _make_gene_id(self, gene_idx):
        """Generate gene ID string"""
        
        return f"{self.id_prefix}_{gene_idx:04d}"
    
    def _make_transcript_id(self, gene_idx, transcript_idx):
        """Generate transcript ID string"""
        
        gene_id = self._make_gene_id(gene_idx)
        if transcript_idx is not None:
            return f"{gene_id}.t{transcript_idx}"
        return f"{gene_id}.t1"
    
    def _make_feature_id(self, gene_idx, transcript_idx, feature_type, feature_num):
        """Generate feature ID string"""

        transcript_id = self._make_transcript_id(gene_idx, transcript_idx)
        return f"{transcript_id}.{feature_type}{feature_num}"

    def _compute_introns(self, exons, gene_idx, transcript_idx):
        """Compute introns as intervals between consecutive exons"""

        if len(exons) < 2:
            return []

        sorted_exons = sorted(exons, key=lambda e: e.start)
        introns      = []

        for i in range(len(sorted_exons) - 1):
            curr_exon = sorted_exons[i]
            next_exon = sorted_exons[i + 1]

            intron_start = curr_exon.end + 1
            intron_end   = next_exon.start - 1

            if intron_start <= intron_end:
                introns.append(ParsedFeature(
                    feature_type   = "intron",
                    start          = intron_start,
                    end            = intron_end,
                    strand         = curr_exon.strand,
                    phase          = ".",
                    biotype        = curr_exon.biotype,
                    gene_idx       = gene_idx,
                    transcript_idx = transcript_idx,
                    raw_line       = "",
                ))

        return introns
    
    def convert_feature(self, feature, feature_num=1):
        """Convert a single ParsedFeature to GFFFeature"""
        
        gene_id       = self._make_gene_id(feature.gene_idx)
        transcript_id = self._make_transcript_id(feature.gene_idx, feature.transcript_idx)
        feature_id    = self._make_feature_id(
            feature.gene_idx,
            feature.transcript_idx,
            feature.feature_type,
            feature_num
        )
        
        attributes = {
            "ID":      feature_id,
            "Parent":  transcript_id,
            "gene_id": gene_id,
            "biotype": feature.biotype,
        }
        
        if feature.transcript_idx is not None:
            attributes["transcript_id"] = transcript_id
        
        return GFFFeature(
            seqid      = self.seqid,
            source     = self.source,
            type       = feature.feature_type.upper() if feature.feature_type == "cds" else feature.feature_type,
            start      = feature.start + self.offset,
            end        = feature.end + self.offset,
            score      = self.score,
            strand     = feature.strand,
            phase      = feature.phase,
            attributes = attributes,
        )
    
    def convert_sequence(self, features):
        """Convert a sequence of ParsedFeatures to GFFFeatures"""
        
        if not features:
            return []
        
        gff_features  = []
        gene_features = {}
        
        for f in features:
            if f.gene_idx not in gene_features:
                gene_features[f.gene_idx] = {}
            if f.transcript_idx not in gene_features[f.gene_idx]:
                gene_features[f.gene_idx][f.transcript_idx] = []
            gene_features[f.gene_idx][f.transcript_idx].append(f)
        
        for gene_idx in sorted(gene_features.keys()):
            transcripts         = gene_features[gene_idx]
            all_features_in_gene = [f for t in transcripts.values() for f in t]
            
            gene_start   = min(f.start for f in all_features_in_gene)
            gene_end     = max(f.end for f in all_features_in_gene)
            gene_strand  = all_features_in_gene[0].strand
            gene_biotype = all_features_in_gene[0].biotype
            gene_id      = self._make_gene_id(gene_idx)
            
            gff_features.append(GFFFeature(
                seqid      = self.seqid,
                source     = self.source,
                type       = "gene",
                start      = gene_start + self.offset,
                end        = gene_end + self.offset,
                score      = self.score,
                strand     = gene_strand,
                phase      = ".",
                attributes = {"ID": gene_id, "biotype": gene_biotype},
            ))
            
            for transcript_idx in sorted(transcripts.keys(), key=lambda x: x or 0):
                transcript_features = transcripts[transcript_idx]
                trans_start         = min(f.start for f in transcript_features)
                trans_end           = max(f.end for f in transcript_features)
                transcript_id       = self._make_transcript_id(gene_idx, transcript_idx)
                
                gff_features.append(GFFFeature(
                    seqid      = self.seqid,
                    source     = self.source,
                    type       = transcript_features[0].biotype,
                    start      = trans_start + self.offset,
                    end        = trans_end + self.offset,
                    score      = self.score,
                    strand     = transcript_features[0].strand,
                    phase      = ".",
                    attributes = {"ID": transcript_id, "Parent": gene_id},
                ))
                
                # Compute introns from exons if enabled
                if self.include_introns:
                    exons   = [f for f in transcript_features if f.feature_type == "exon"]
                    introns = self._compute_introns(exons, gene_idx, transcript_idx)
                    transcript_features = transcript_features + introns

                # Sort features by position for consistent output
                transcript_features = sorted(transcript_features, key=lambda f: (f.start, f.feature_type))

                feature_counts = {}
                for f in transcript_features:
                    ftype                 = f.feature_type
                    feature_counts[ftype] = feature_counts.get(ftype, 0) + 1
                    gff_features.append(self.convert_feature(f, feature_counts[ftype]))

        return gff_features


##########################
#####  Functions     #####
##########################


def parse_model_output(text, seqid="seq", source="GeneT5", offset=0, strict=False, include_introns=False):
    """Parse model output and convert to GFF features"""

    parser           = ModelOutputParser(strict=strict)
    converter        = GFFConverter(seqid=seqid, source=source, offset=offset, include_introns=include_introns)
    parsed_sequences = parser.parse_sequence(text)
    gff_sequences    = [converter.convert_sequence(seq) for seq in parsed_sequences]

    return gff_sequences


def write_gff3(features, output_path, append=False):
    """Write GFF features to file in GFF3 format"""
    
    mode = "a" if append else "w"
    
    with open(output_path, mode) as f:
        if not append:
            f.write("##gff-version 3\n")
        for feature in features:
            f.write(feature.to_gff_line() + "\n")


def model_output_to_gff3(model_output, output_path, seqid="seq", source="GeneT5", offset=0, strict=False, include_introns=False):
    """Complete pipeline to parse model output and write to GFF3 file"""

    sequences    = parse_model_output(
        text            = model_output,
        seqid           = seqid,
        source          = source,
        offset          = offset,
        strict          = strict,
        include_introns = include_introns,
    )
    all_features = [f for seq in sequences for f in seq]

    write_gff3(all_features, output_path)

    return len(all_features)


def process_batch_outputs(outputs, seqids=None, output_dir=".", source="GeneT5", offsets=None, include_introns=False):
    """Process multiple model outputs to individual GFF3 files"""

    output_dir = pl.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seqids is None:
        seqids = [f"seq_{i}" for i in range(len(outputs))]

    if offsets is None:
        offsets = [0] * len(outputs)

    output_paths = []

    for i, (output, seqid, offset) in enumerate(zip(outputs, seqids, offsets)):
        output_path = output_dir / f"{seqid}.gff3"
        model_output_to_gff3(
            model_output    = output,
            output_path     = output_path,
            seqid           = seqid,
            source          = source,
            offset          = offset,
            include_introns = include_introns,
        )
        output_paths.append(output_path)

    return output_paths