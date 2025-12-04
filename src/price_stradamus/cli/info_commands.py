"""System information CLI commands for Price Stradamus.

This module handles informational commands:
- Listing available models
- Showing system configuration
- Displaying version information
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from price_stradamus.config.settings import settings
from price_stradamus.models.registry import ModelRegistry

# Create console for output
console = Console()


def register_info_commands(app: typer.Typer) -> None:
    """Register all info-related commands to the main app.

    Args:
        app: Main Typer application instance
    """
    app.command(name="list-models")(list_models)
    app.command()(info)


def list_models() -> None:
    """List all available prediction models.

    Displays a table of all registered models with:
    - Model name
    - Class name
    - Brief description

    Examples:
        price-stradamus list-models
    """
    console.print("[bold cyan]Available Models:[/bold cyan]\n")

    models = ModelRegistry.list_models()

    if not models:
        console.print("[yellow]No models registered[/yellow]")
        return

    # Create table
    table = Table(title=f"Registered Models ({len(models)} total)")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Class", style="magenta")
    table.add_column("Description", style="white")

    for model_name in models:
        info_dict = ModelRegistry.get_model_info(model_name)
        # Get first line of docstring
        desc = (
            info_dict["docstring"].split("\n")[0]
            if info_dict["docstring"]
            else "No description"
        )
        table.add_row(
            model_name,
            info_dict["class"],
            desc[:60] + "..." if len(desc) > 60 else desc,
        )

    console.print(table)


def info() -> None:
    """Show system information and configuration.

    Displays:
    - Database connection details
    - Binance API configuration
    - Training default parameters
    - List of registered models

    Examples:
        price-stradamus info
    """
    console.print("[bold cyan]Price Stradamus - System Information[/bold cyan]\n")

    # Database info
    console.print("[bold]Database:[/bold]")
    console.print(f"  URL: {settings.database_url}")

    # Binance API info
    console.print("\n[bold]Binance API:[/bold]")
    console.print(f"  URL: {settings.binance_base_url}")

    # Training defaults
    console.print("\n[bold]Training Defaults:[/bold]")
    console.print(f"  Lookback window: {settings.default_lookback_candles}")
    console.print(f"  Prediction steps: {settings.default_prediction_steps}")
    console.print(f"  Batch size: {settings.batch_size}")
    console.print(f"  Learning rate: {settings.learning_rate}")
    console.print(f"  Epochs: {settings.epochs}")

    # Models
    console.print("\n[bold]Registered Models:[/bold]")
    models = ModelRegistry.list_models()
    for model in models:
        console.print(f"  • {model}")
