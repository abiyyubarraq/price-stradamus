"""Price Stradamus models package.

This package contains all prediction models and the model registry.
Importing this package will automatically register all models.
"""

from __future__ import annotations

# Import all model subpackages to trigger @ModelRegistry.register() decorators
from price_stradamus.models import ml  # noqa: F401

__all__ = ["ml"]
