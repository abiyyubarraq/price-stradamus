"""Custom exceptions for Price Stradamus models."""

from __future__ import annotations


class PriceStradamusError(Exception):
    """Base exception for Price Stradamus."""


class ModelError(PriceStradamusError):
    """Base exception for model-related errors."""


class ModelNotFittedError(ModelError):
    """Model must be fitted before prediction or other operations."""


class ModelTrainingError(ModelError):
    """Error during model training."""


class InvalidDataError(ModelError):
    """Invalid training or prediction data."""


class InvalidHyperparameterError(ModelError):
    """Invalid hyperparameter value."""


class ModelLoadError(ModelError):
    """Error loading model from disk."""


class ModelSaveError(ModelError):
    """Error saving model to disk."""
