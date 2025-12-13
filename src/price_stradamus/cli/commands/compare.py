"""Compare command - Compare multiple models side-by-side."""

from __future__ import annotations

import typer
from rich.console import Console

# Create console for output
console = Console()


def compare(
    models: list[str] = typer.Option(
        ..., "--model", "-m", help="Model names to compare (repeat for multiple)"
    ),
) -> None:
    """Compare multiple models side-by-side.

    This command will be fully implemented in Phase 2 with:
    - Parallel model training
    - Side-by-side metrics comparison
    - Visual comparison charts
    - Statistical significance tests

    Examples:
        # Compare three models
        price-stradamus compare -m nbeats -m lstm -m xgboost

        # Compare with saved models
        price-stradamus compare --model-path models/nbeats.pkl --model-path models/lstm.pkl
    """
    console.print(f"[bold cyan]Comparing models: {', '.join(models)}...[/bold cyan]")
    console.print(
        "[yellow]Model comparison will be fully implemented in Phase 2[/yellow]"
    )
    console.print(
        "[dim]This feature will include parallel training, metrics comparison, and visualization[/dim]"
    )
