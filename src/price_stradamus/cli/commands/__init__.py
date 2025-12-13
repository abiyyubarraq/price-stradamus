"""Model command implementations.

This package contains individual command implementations for model operations:
- train: Train prediction models
- predict: Make predictions with trained models
- evaluate: Evaluate model performance with backtesting
- compare: Compare multiple models side-by-side
"""

from __future__ import annotations

from price_stradamus.cli.commands.compare import compare
from price_stradamus.cli.commands.evaluate import evaluate
from price_stradamus.cli.commands.predict import predict
from price_stradamus.cli.commands.train import train

__all__ = ["train", "predict", "evaluate", "compare"]
