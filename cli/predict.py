import time
import pathlib as pl
from typing import List, Optional

import typer
from rich.console   import Console
from rich.panel     import Panel
from rich.table     import Table
from rich.progress  import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from cli import app

console = Console()


@app.command()
def predict(
    model:       str            = typer.Option(..., "--model",  "-m", help="Model checkpoint path"),
    input_path:  str            = typer.Option(..., "--input",  "-i", help="Input FASTA file or sequence"),
    output:      str            = typer.Option("output", "--output", "-o", help="Output directory"),
    formats:     List[str]      = typer.Option(["gff3"], "--format", "-f", help="Output formats: gff3 fasta protein gtf bed"),
    tokenizer:   Optional[str]  = typer.Option(None, "--tokenizer", help="Tokenizer path (default: same as model)"),
    source:      str            = typer.Option("GeneT5", "--source", help="GFF source field"),
    max_length:  int            = typer.Option(512,  "--max-length",  help="Max generation length"),
    temperature: float          = typer.Option(1.0,  "--temperature", help="Sampling temperature"),
    top_k:       int            = typer.Option(50,   "--top-k",       help="Top-k sampling"),
    top_p:       float          = typer.Option(0.9,  "--top-p",       help="Top-p sampling"),
    batch_size:  int            = typer.Option(1,    "--batch-size",  help="Batch size"),
    device:      Optional[str]  = typer.Option(None, "--device",      help="Device (auto-detect if omitted)"),
    seed:        int            = typer.Option(42,   "--seed",        help="Random seed"),
    diff_steps:  int            = typer.Option(32,   "--diffusion-steps", help="Diffusion denoising steps"),
):
    """Run gene prediction on input sequences"""

    import torch
    import lib.inference.engine as engine_lib
    import lib.inference.output as output_lib

    # Seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Validate formats
    valid_fmts = set(output_lib.FORMAT_WRITERS.keys())
    for fmt in formats:
        if fmt not in valid_fmts:
            console.print(f"[red]Unknown format:[/red] {fmt}  (valid: {', '.join(sorted(valid_fmts))})")
            raise typer.Exit(1)

    # Config panel
    config_table = Table.grid(padding=(0, 2))
    config_table.add_column(style="bold cyan", justify="right")
    config_table.add_column()
    config_table.add_row("Model",       model)
    config_table.add_row("Input",       input_path)
    config_table.add_row("Output",      output)
    config_table.add_row("Formats",     ", ".join(formats))
    config_table.add_row("Max length",  str(max_length))
    config_table.add_row("Temperature", str(temperature))
    config_table.add_row("Batch size",  str(batch_size))
    config_table.add_row("Diff steps",  str(diff_steps))
    config_table.add_row("Device",      device or "auto")

    console.print(Panel(config_table, title="[bold]GeneT5 Predict[/bold]", border_style="blue"))

    # Load model
    with console.status("[bold green]Loading model..."):
        target_device  = torch.device(device) if device else None
        tokenizer_path = tokenizer or model
        inferencer     = engine_lib.GeneT5Inference.from_pretrained(
            checkpoint_path = model,
            tokenizer_path  = tokenizer_path,
            device          = target_device,
        )

    # Load input
    sequences, seqids = engine_lib.read_input(input_path)
    console.print(f"Loaded [bold]{len(sequences)}[/bold] sequence(s)")

    # Generation config
    gen_config = engine_lib.GenerationConfig(
        max_length  = max_length,
        temperature = temperature,
        top_k       = top_k,
        top_p       = top_p,
    )

    # Run inference with progress bar
    output_dir = pl.Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    t0          = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Predicting", total=len(sequences))

        for batch_start in range(0, len(sequences), batch_size):
            batch_end  = min(batch_start + batch_size, len(sequences))
            batch_seqs = sequences[batch_start:batch_end]
            batch_ids  = seqids[batch_start:batch_end]

            results = inferencer.predict(
                sequences  = batch_seqs,
                seqids     = batch_ids,
                output_dir = None,
                source     = source,
                gen_config = gen_config,
                batch_size = batch_size,
            )

            # Write each result in requested formats
            for r in results:
                sid = r.metadata['seqid']
                output_lib.write_all_formats(
                    features       = r.parsed_features,
                    input_sequence = r.input_sequence,
                    seqid          = sid,
                    output_dir     = output_dir,
                    formats        = formats,
                    source         = source,
                    offset         = r.metadata.get('offset', 0),
                )

            all_results.extend(results)
            progress.update(task, advance=len(batch_seqs))

    elapsed = time.time() - t0

    # Results table
    table = Table(title="Results", show_lines=False)
    table.add_column("Sequence",   style="cyan")
    table.add_column("Exons",      justify="right")
    table.add_column("UTRs",       justify="right")
    table.add_column("Output",     style="dim")

    total_exons = 0
    total_utrs  = 0

    for r in all_results:
        sid       = r.metadata['seqid']
        n_exons   = sum(1 for f in r.parsed_features if f.kind == "exon")
        n_utrs    = sum(1 for f in r.parsed_features if f.kind == "utr")
        total_exons += n_exons
        total_utrs  += n_utrs
        table.add_row(sid, str(n_exons), str(n_utrs), str(output_dir / sid))

    console.print(table)
    console.print(
        f"\n[bold green]Done![/bold green] "
        f"{len(all_results)} sequences, {total_exons} exons, {total_utrs} UTRs "
        f"in {elapsed:.1f}s"
    )
