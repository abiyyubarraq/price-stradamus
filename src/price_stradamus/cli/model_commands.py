"""Model training and prediction CLI commands for Price Stradamus.

This module handles registration of all ML model operations:
- Training models
- Making predictions
- Model evaluation and backtesting
- Model comparison

The actual command implementations are in the commands/ subdirectory.
"""

from __future__ import annotations

import typer

from price_stradamus.cli.commands import compare, evaluate, predict, train


def register_model_commands(app: typer.Typer) -> None:
    """Register all model-related commands to the main app.

    Args:
        app: Main Typer application instance
    """
    app.command()(train)
    app.command()(predict)
    app.command()(evaluate)
    app.command()(compare)
