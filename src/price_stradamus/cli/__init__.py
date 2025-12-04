"""Price Stradamus CLI - Main entry point.

This module sets up the main Typer application and registers all command groups:
- Data commands (fetch)
- Model commands (train, predict, evaluate, compare)
- Info commands (list-models, info)
"""

from __future__ import annotations

import typer

from price_stradamus.cli.data_commands import register_data_commands
from price_stradamus.cli.info_commands import register_info_commands
from price_stradamus.cli.model_commands import register_model_commands

# Create main Typer app
app = typer.Typer(
    name="price-stradamus",
    help="Bitcoin Price Prediction System - Professional ML-powered cryptocurrency forecasting",
    add_completion=False,
    rich_markup_mode="rich",
)

# Register all command groups
register_data_commands(app)
register_model_commands(app)
register_info_commands(app)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
