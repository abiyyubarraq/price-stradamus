"""Configuration module."""

from __future__ import annotations

from price_stradamus.config.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOOKBACK_WINDOW,
    DEFAULT_PREDICTION_HORIZON,
    RANDOM_SEED,
    ModelType,
    NormalizationMethod,
    Timeframe,
)
from price_stradamus.config.settings import Settings, get_settings, settings

__all__ = [
    # Settings
    "Settings",
    "settings",
    "get_settings",
    # Enums
    "Timeframe",
    "ModelType",
    "NormalizationMethod",
    # Constants
    "DEFAULT_LOOKBACK_WINDOW",
    "DEFAULT_PREDICTION_HORIZON",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_EPOCHS",
    "RANDOM_SEED",
]
