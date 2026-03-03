import sys
import pathlib as pl
from typing import List, Optional

import typer
from rich.console import Console
from rich.table   import Table

from cli import app

console = Console()


@app.command(name="eval")
def evaluate(
    reference: str            = typer.Option(..., "--reference", "-r", help="Reference GFF3 annotation"),
    predicted: List[str]      = typer.Option(..., "--predicted", "-p", help="Predicted GFF3 file(s)"),
    level:     str            = typer.Option("all", "--level", "-l", help="Evaluation level: nucleotide, exon, gene, all"),
    output:    Optional[str]  = typer.Option(None, "--output", "-o", help="Output JSON report file"),
    parallel:  int            = typer.Option(1, "--parallel", help="Parallel evaluations"),
):
    """Evaluate predictions against reference annotations"""

    import json
    import concurrent.futures

    sys.path.insert(0, str(pl.Path(__file__).resolve().parent.parent / "eval"))
    import eval as ev

    ref_path   = pl.Path(reference)
    pred_paths = [pl.Path(p) for p in predicted]

    if not ref_path.exists():
        console.print(f"[red]Reference not found:[/red] {ref_path}")
        raise typer.Exit(1)

    for p in pred_paths:
        if not p.exists():
            console.print(f"[red]Predicted file not found:[/red] {p}")
            raise typer.Exit(1)

    levels = ['nucleotide', 'exon', 'gene'] if level == 'all' else [level]

    console.print(f"Reference:  [cyan]{ref_path}[/cyan]")
    console.print(f"Predicted:  [cyan]{len(pred_paths)}[/cyan] file(s)")
    console.print(f"Levels:     [cyan]{', '.join(levels)}[/cyan]\n")

    def evaluate_file(pred_path):

        results = {'file': str(pred_path.name)}
        for lv in levels:
            try:
                metrics     = ev.calculate_f1(str(ref_path), str(pred_path), level=lv)
                results[lv] = metrics
            except Exception as e:
                results[lv] = {'error': str(e)}
        return results

    all_results = []

    with console.status("[bold green]Evaluating..."):
        if parallel > 1 and len(pred_paths) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
                futures = {executor.submit(evaluate_file, p): p for p in pred_paths}
                for future in concurrent.futures.as_completed(futures):
                    all_results.append(future.result())
        else:
            for p in pred_paths:
                all_results.append(evaluate_file(p))

    # Display results
    for result in all_results:
        table = Table(title=result['file'], show_lines=False)
        table.add_column("Level",       style="cyan", width=12)
        table.add_column("F1",          justify="right")
        table.add_column("Precision",   justify="right")
        table.add_column("Sensitivity", justify="right")
        table.add_column("TP",          justify="right", style="green")
        table.add_column("FP",          justify="right", style="red")
        table.add_column("FN",          justify="right", style="yellow")

        for lv in levels:
            if lv not in result:
                continue
            m = result[lv]
            if 'error' in m:
                table.add_row(lv, f"[red]ERROR: {m['error']}[/red]", "", "", "", "", "")
            else:
                table.add_row(
                    lv,
                    f"{m['f1']:.4f}",
                    f"{m['precision']:.4f}",
                    f"{m['sensitivity']:.4f}",
                    str(m['tp']),
                    str(m['fp']),
                    str(m['fn']),
                )

        console.print(table)
        console.print()

    # Write JSON report
    if output:
        output_path = pl.Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {'reference': str(ref_path), 'levels': levels, 'results': all_results}
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        console.print(f"Report written: [cyan]{output_path}[/cyan]")
