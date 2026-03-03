import pathlib as pl
from typing import List, Optional

import typer
from rich.console import Console

from cli import app

console = Console()


@app.command()
def convert(
    input_gff: str            = typer.Option(..., "--input", "-i", help="Input GFF3 file"),
    fasta:     Optional[str]  = typer.Option(None, "--fasta", help="Genome FASTA (needed for protein translation)"),
    formats:   List[str]      = typer.Option(["gtf"], "--format", "-f", help="Output formats: gtf bed fasta protein"),
    output:    Optional[str]  = typer.Option(None, "--output", "-o", help="Output directory (default: same as input)"),
    source:    str            = typer.Option("GeneT5", "--source", help="Source field for GTF"),
):
    """Convert GFF3 predictions to other formats"""

    import lib.inference.output as output_lib

    gff_path = pl.Path(input_gff)
    if not gff_path.exists():
        console.print(f"[red]GFF3 file not found:[/red] {gff_path}")
        raise typer.Exit(1)

    valid_fmts = {'gtf', 'bed', 'fasta', 'protein'}
    for fmt in formats:
        if fmt not in valid_fmts:
            console.print(f"[red]Unknown format:[/red] {fmt}  (valid: {', '.join(sorted(valid_fmts))})")
            raise typer.Exit(1)

    # Parse GFF3 to recover features
    features, seqids = _parse_gff3_to_features(gff_path)
    if not features:
        console.print("[yellow]No features found in GFF3 file[/yellow]")
        raise typer.Exit(0)

    console.print(f"Parsed [bold]{len(features)}[/bold] features from {gff_path.name}")

    # Load genome FASTA if needed for protein translation
    genome_seq = None
    if ('protein' in formats or 'fasta' in formats) and fasta:
        genome_seq = _load_fasta_sequence(pl.Path(fasta))
    elif 'protein' in formats and not fasta:
        console.print("[red]--fasta required for protein format[/red]")
        raise typer.Exit(1)

    out_dir = pl.Path(output) if output else gff_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = gff_path.stem

    written = []

    for fmt in formats:
        if fmt == 'gtf':
            lines = output_lib.features_to_gtf(
                features, genome_seq or "", seqid=stem, source=source)
            out_path = out_dir / f"{stem}.gtf"
            output_lib.write_gtf(lines, out_path)
            written.append(out_path)

        elif fmt == 'bed':
            lines = output_lib.features_to_bed(features, genome_seq or "", seqid=stem)
            out_path = out_dir / f"{stem}.bed"
            output_lib.write_bed(lines, out_path)
            written.append(out_path)

        elif fmt == 'fasta' and genome_seq:
            records = output_lib.features_to_fasta(features, genome_seq, seqid=stem)
            out_path = out_dir / f"{stem}.fa"
            output_lib.write_fasta(records, out_path)
            written.append(out_path)

        elif fmt == 'protein' and genome_seq:
            records = output_lib.features_to_protein(features, genome_seq, seqid=stem)
            out_path = out_dir / f"{stem}.protein.fa"
            output_lib.write_fasta(records, out_path)
            written.append(out_path)

    for p in written:
        console.print(f"  [green]Wrote[/green] {p}")

    console.print(f"\n[bold green]Done![/bold green] {len(written)} file(s) written")


def _parse_gff3_to_features(gff_path):
    """Parse GFF3 back into ParsedExon-like objects for format conversion"""

    import lib.inference.output as output_lib

    features = []
    seqids   = set()

    with open(gff_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 9:
                continue

            seqid     = parts[0]
            feat_type = parts[2]

            seqids.add(seqid)

            if feat_type in ('CDS', 'exon'):
                kind = 'exon'
            elif feat_type in ('UTR', 'five_prime_UTR', 'three_prime_UTR'):
                kind = 'utr'
            else:
                continue

            start = int(parts[3])
            end   = int(parts[4])

            features.append(output_lib.ParsedExon(kind=kind, dna=f"{start}-{end}"))

    return features, list(seqids)


def _load_fasta_sequence(fasta_path):
    """Load first sequence from FASTA file"""

    lines = []

    with open(fasta_path) as f:
        started = False
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if started:
                    break
                started = True
                continue
            lines.append(line)

    return ''.join(lines)
