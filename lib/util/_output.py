"""
GFF Converter Module

Parses GeneT5 model output format and converts to standard GFF3 format.

Model output format:
    {type}{start}\t{end}{strand}{phase}{biotype}{gene_idx}[.transcript_idx]
    
Examples:
    exon100\t200+.mRNA1       -> exon, 100-200, +, phase=., biotype=mRNA, gene=1
    cds120\t180+0mRNA1        -> cds, 120-180, +, phase=0, biotype=mRNA, gene=1
    exon100\t200+.mRNA1.2     -> exon, 100-200, +, phase=., biotype=mRNA, gene=1, transcript=2

GFF3 format:
    seqid  source  type  start  end  score  strand  phase  attributes
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path


@dataclass
class GFFFeature:
    """Represents a single GFF3 feature"""
    seqid:      str
    source:     str
    type:       str
    start:      int
    end:        int
    score:      str = "."
    strand:     str = "+"
    phase:      str = "."
    attributes: Dict[str, str] = field(default_factory=dict)
    
    def to_gff_line(self) -> str:
        """Convert to GFF3 line format"""
        attr_str = ";".join(f"{k}={v}" for k, v in self.attributes.items())
        return f"{self.seqid}\t{self.source}\t{self.type}\t{self.start}\t{self.end}\t{self.score}\t{self.strand}\t{self.phase}\t{attr_str}"
    
    def __str__(self) -> str:
        return self.to_gff_line()


@dataclass
class ParsedFeature:
    """Intermediate parsed feature from model output"""
    feature_type:    str
    start:           int
    end:             int
    strand:          str
    phase:           str
    biotype:         str
    gene_idx:        int
    transcript_idx:  Optional[int] = None
    raw_line:        str = ""


class ModelOutputParser:
    """
    Parser for GeneT5 model output format.
    
    Format: {type}{start}\t{end}{strand}{phase}{biotype}{gene_idx}[.transcript_idx]
    
    Regex breakdown:
        - type: letters (exon, cds, gene, etc.)
        - start: digits
        - \t: tab separator
        - end: digits
        - strand: + or -
        - phase: . or 0/1/2
        - biotype: letters (mRNA, ncRNA, etc.)
        - gene_idx: digits
        - transcript_idx: optional .digits
    """
    
    # Pattern for model output format
    FEATURE_PATTERN = re.compile(
        r'^(?P<type>[a-zA-Z_]+)'           # feature type (exon, cds, gene, etc.)
        r'(?P<start>\d+)'                   # start position
        r'\t'                               # tab separator
        r'(?P<end>\d+)'                     # end position
        r'(?P<strand>[+\-])'                # strand
        r'(?P<phase>[.\d])'                 # phase (. or 0/1/2)
        r'(?P<biotype>[a-zA-Z_]+)'          # biotype (mRNA, ncRNA, etc.)
        r'(?P<gene_idx>\d+)'                # gene index
        r'(?:\.(?P<transcript_idx>\d+))?$'  # optional transcript index
    )
    
    # Alternative pattern for space-separated format
    FEATURE_PATTERN_SPACE = re.compile(
        r'^(?P<type>[a-zA-Z_]+)'
        r'(?P<start>\d+)'
        r'\s+'
        r'(?P<end>\d+)'
        r'(?P<strand>[+\-])'
        r'(?P<phase>[.\d])'
        r'(?P<biotype>[a-zA-Z_]+)'
        r'(?P<gene_idx>\d+)'
        r'(?:\.(?P<transcript_idx>\d+))?$'
    )
    
    # Special tokens
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    
    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, raise errors on parse failures. 
                    If False, skip unparseable lines.
        """
        self.strict = strict
    
    def parse_line(self, line: str) -> Optional[ParsedFeature]:
        """
        Parse a single line from model output.
        
        Args:
            line: Raw line from model output
            
        Returns:
            ParsedFeature if successful, None if line is special token or unparseable
        """
        line = line.strip()
        
        # Skip empty lines and special tokens
        if not line or line in (self.BOS_TOKEN, self.EOS_TOKEN, "<bos>", "<eos>"):
            return None
        
        # Try tab-separated pattern first
        match = self.FEATURE_PATTERN.match(line)
        if not match:
            # Try space-separated pattern
            match = self.FEATURE_PATTERN_SPACE.match(line)
        
        if not match:
            if self.strict:
                raise ValueError(f"Failed to parse line: {line}")
            return None
        
        groups = match.groupdict()
        
        return ParsedFeature(
            feature_type   = groups["type"].lower(),
            start          = int(groups["start"]),
            end            = int(groups["end"]),
            strand         = groups["strand"],
            phase          = groups["phase"],
            biotype        = groups["biotype"],
            gene_idx       = int(groups["gene_idx"]),
            transcript_idx = int(groups["transcript_idx"]) if groups["transcript_idx"] else None,
            raw_line       = line,
        )
    
    def parse_sequence(self, text: str) -> List[List[ParsedFeature]]:
        """
        Parse complete model output, potentially containing multiple sequences.
        
        Sequences are delimited by <BOS>/<EOS> tokens.
        
        Args:
            text: Complete model output text
            
        Returns:
            List of sequences, each containing list of ParsedFeatures
        """
        sequences = []
        current_seq = []
        
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Start new sequence
            if line.upper() in ("<BOS>", "BOS"):
                if current_seq:
                    sequences.append(current_seq)
                current_seq = []
                continue
            
            # End current sequence
            if line.upper() in ("<EOS>", "EOS"):
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = []
                continue
            
            # Parse feature line
            feature = self.parse_line(line)
            if feature:
                current_seq.append(feature)
        
        # Don't forget last sequence if no EOS
        if current_seq:
            sequences.append(current_seq)
        
        return sequences


class GFFConverter:
    """
    Converts parsed model output to GFF3 format.
    """
    
    def __init__(
        self,
        seqid:         str = "seq",
        source:        str = "GeneT5",
        offset:        int = 0,
        score:         str = ".",
        id_prefix:     str = "gene",
    ):
        """
        Args:
            seqid:     Sequence ID for GFF output (chromosome/contig name)
            source:    Source field for GFF (tool name)
            offset:    Position offset to add to all coordinates
            score:     Score field value (use "." for inference output)
            id_prefix: Prefix for gene/transcript IDs
        """
        self.seqid     = seqid
        self.source    = source
        self.offset    = offset
        self.score     = score
        self.id_prefix = id_prefix
    
    def _make_gene_id(self, gene_idx: int) -> str:
        return f"{self.id_prefix}_{gene_idx:04d}"
    
    def _make_transcript_id(self, gene_idx: int, transcript_idx: Optional[int]) -> str:
        gene_id = self._make_gene_id(gene_idx)
        if transcript_idx is not None:
            return f"{gene_id}.t{transcript_idx}"
        return f"{gene_id}.t1"
    
    def _make_feature_id(self, gene_idx: int, transcript_idx: Optional[int], 
                         feature_type: str, feature_num: int) -> str:
        transcript_id = self._make_transcript_id(gene_idx, transcript_idx)
        return f"{transcript_id}.{feature_type}{feature_num}"
    
    def convert_feature(self, feature: ParsedFeature, feature_num: int = 1) -> GFFFeature:
        """
        Convert a single ParsedFeature to GFFFeature.
        
        Args:
            feature:     Parsed feature from model output
            feature_num: Feature number for ID generation
            
        Returns:
            GFFFeature ready for output
        """
        gene_id       = self._make_gene_id(feature.gene_idx)
        transcript_id = self._make_transcript_id(feature.gene_idx, feature.transcript_idx)
        feature_id    = self._make_feature_id(
            feature.gene_idx, 
            feature.transcript_idx,
            feature.feature_type,
            feature_num
        )
        
        attributes = {
            "ID":        feature_id,
            "Parent":    transcript_id,
            "gene_id":   gene_id,
            "biotype":   feature.biotype,
        }
        
        # Add transcript info if present
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
    
    def convert_sequence(self, features: List[ParsedFeature]) -> List[GFFFeature]:
        """
        Convert a sequence of ParsedFeatures to GFFFeatures.
        
        Also generates parent gene and transcript features.
        
        Args:
            features: List of parsed features from one sequence
            
        Returns:
            List of GFFFeatures including gene/transcript parents
        """
        if not features:
            return []
        
        gff_features = []
        
        # Group features by gene and transcript
        gene_features: Dict[int, Dict[Optional[int], List[ParsedFeature]]] = {}
        
        for f in features:
            if f.gene_idx not in gene_features:
                gene_features[f.gene_idx] = {}
            if f.transcript_idx not in gene_features[f.gene_idx]:
                gene_features[f.gene_idx][f.transcript_idx] = []
            gene_features[f.gene_idx][f.transcript_idx].append(f)
        
        # Generate hierarchical GFF structure
        for gene_idx in sorted(gene_features.keys()):
            transcripts = gene_features[gene_idx]
            
            # Compute gene boundaries
            all_features_in_gene = [f for t in transcripts.values() for f in t]
            gene_start  = min(f.start for f in all_features_in_gene)
            gene_end    = max(f.end for f in all_features_in_gene)
            gene_strand = all_features_in_gene[0].strand
            gene_biotype = all_features_in_gene[0].biotype
            
            gene_id = self._make_gene_id(gene_idx)
            
            # Add gene feature
            gff_features.append(GFFFeature(
                seqid      = self.seqid,
                source     = self.source,
                type       = "gene",
                start      = gene_start + self.offset,
                end        = gene_end + self.offset,
                score      = self.score,
                strand     = gene_strand,
                phase      = ".",
                attributes = {
                    "ID":      gene_id,
                    "biotype": gene_biotype,
                },
            ))
            
            # Add transcript(s) and their features
            for transcript_idx in sorted(transcripts.keys(), key=lambda x: x or 0):
                transcript_features = transcripts[transcript_idx]
                
                trans_start = min(f.start for f in transcript_features)
                trans_end   = max(f.end for f in transcript_features)
                
                transcript_id = self._make_transcript_id(gene_idx, transcript_idx)
                
                # Add transcript/mRNA feature
                gff_features.append(GFFFeature(
                    seqid      = self.seqid,
                    source     = self.source,
                    type       = transcript_features[0].biotype,
                    start      = trans_start + self.offset,
                    end        = trans_end + self.offset,
                    score      = self.score,
                    strand     = transcript_features[0].strand,
                    phase      = ".",
                    attributes = {
                        "ID":     transcript_id,
                        "Parent": gene_id,
                    },
                ))
                
                # Add child features (exon, CDS, etc.)
                feature_counts: Dict[str, int] = {}
                for f in transcript_features:
                    ftype = f.feature_type
                    feature_counts[ftype] = feature_counts.get(ftype, 0) + 1
                    gff_features.append(self.convert_feature(f, feature_counts[ftype]))
        
        return gff_features


def parse_model_output(
    text:         str,
    seqid:        str = "seq",
    source:       str = "GeneT5",
    offset:       int = 0,
    strict:       bool = False,
) -> List[List[GFFFeature]]:
    """
    Convenience function to parse model output and convert to GFF features.
    
    Args:
        text:   Raw model output text
        seqid:  Sequence ID for GFF
        source: Source field for GFF
        offset: Position offset to add
        strict: Whether to raise errors on parse failures
        
    Returns:
        List of sequences, each containing GFF features
    """
    parser    = ModelOutputParser(strict=strict)
    converter = GFFConverter(seqid=seqid, source=source, offset=offset)
    
    parsed_sequences = parser.parse_sequence(text)
    gff_sequences    = [converter.convert_sequence(seq) for seq in parsed_sequences]
    
    return gff_sequences


def write_gff3(
    features:    List[GFFFeature],
    output_path: str | Path,
    append:      bool = False,
):
    """
    Write GFF features to file in GFF3 format.
    
    Args:
        features:    List of GFF features to write
        output_path: Output file path
        append:      Whether to append to existing file
    """
    mode = "a" if append else "w"
    
    with open(output_path, mode) as f:
        if not append:
            f.write("##gff-version 3\n")
        
        for feature in features:
            f.write(feature.to_gff_line() + "\n")


def model_output_to_gff3(
    model_output: str,
    output_path:  str | Path,
    seqid:        str = "seq",
    source:       str = "GeneT5",
    offset:       int = 0,
    strict:       bool = False,
) -> int:
    """
    Complete pipeline: parse model output and write to GFF3 file.
    
    Args:
        model_output: Raw model output text
        output_path:  Output GFF3 file path
        seqid:        Sequence ID for GFF
        source:       Source field for GFF
        offset:       Position offset
        strict:       Whether to raise errors on parse failures
        
    Returns:
        Number of features written
    """
    sequences = parse_model_output(
        text   = model_output,
        seqid  = seqid,
        source = source,
        offset = offset,
        strict = strict,
    )
    
    all_features = [f for seq in sequences for f in seq]
    write_gff3(all_features, output_path)
    
    return len(all_features)


# Batch processing utilities

def process_batch_outputs(
    outputs:      List[str],
    seqids:       Optional[List[str]] = None,
    output_dir:   str | Path = ".",
    source:       str = "GeneT5",
    offsets:      Optional[List[int]] = None,
) -> List[Path]:
    """
    Process multiple model outputs to individual GFF3 files.
    
    Args:
        outputs:    List of model output strings
        seqids:     List of sequence IDs (defaults to seq_0, seq_1, ...)
        output_dir: Directory for output files
        source:     Source field for GFF
        offsets:    List of position offsets
        
    Returns:
        List of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if seqids is None:
        seqids = [f"seq_{i}" for i in range(len(outputs))]
    
    if offsets is None:
        offsets = [0] * len(outputs)
    
    output_paths = []
    
    for i, (output, seqid, offset) in enumerate(zip(outputs, seqids, offsets)):
        output_path = output_dir / f"{seqid}.gff3"
        model_output_to_gff3(
            model_output = output,
            output_path  = output_path,
            seqid        = seqid,
            source       = source,
            offset       = offset,
        )
        output_paths.append(output_path)
    
    return output_paths