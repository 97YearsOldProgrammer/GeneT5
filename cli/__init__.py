import typer

app = typer.Typer(
    name="genet5",
    help="GeneT5 — Gene prediction with diffusion transformers",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def version_callback(value):

    if value:
        from rich.console import Console
        Console().print("[bold]GeneT5[/bold] v0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-V",
        callback=version_callback, is_eager=True,
        help="Show version and exit",
    ),
):
    """GeneT5 gene prediction toolkit"""


import cli.predict  # noqa: E402, F401
import cli.eval     # noqa: E402, F401
import cli.formats  # noqa: E402, F401
