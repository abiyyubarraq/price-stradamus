"""Abstract base model interface for all prediction models.

This module defines the BaseModel abstract base class that all time series
prediction models must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

from darts import TimeSeries
from loguru import logger


class BaseModel(ABC):
    """Abstract base class for all prediction models.

    All models (neural, classical, ML) must inherit from this class and
    implement the abstract methods. This ensures a consistent interface
    across all model types.

    Attributes:
        name: Model name/identifier
        input_chunk_length: Number of past timesteps to use for prediction
        output_chunk_length: Number of future timesteps to predict
        hyperparameters: Dictionary of model hyperparameters
        is_fitted: Whether model has been trained

    Example:
        class MyModel(BaseModel):
            def fit(self, train_data, val_data=None, **kwargs):
                # Training logic here
                self._is_fitted = True

            def predict(self, n, series=None):
                # Prediction logic here
                return predictions

            def save(self, path):
                # Save model to disk
                pass

            def load(self, path):
                # Load model from disk
                return self
    """

    def __init__(
        self,
        name: str,
        input_chunk_length: int,
        output_chunk_length: int,
        **kwargs: Any,
    ):
        """Initialize base model.

        Args:
            name: Model name/identifier
            input_chunk_length: Number of past timesteps to use (lookback window)
            output_chunk_length: Number of future timesteps to predict (forecast horizon)
            **kwargs: Additional hyperparameters (model-specific)
        """
        self.name = name
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.hyperparameters = kwargs
        self._is_fitted = False
        self._model: Any | None = None  # Underlying model object

        logger.debug(
            f"Initialized {self.__class__.__name__} "
            f"(input={input_chunk_length}, output={output_chunk_length})"
        )

    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained.

        Returns:
            True if model has been fitted, False otherwise
        """
        return self._is_fitted

    @abstractmethod
    def fit(
        self,
        train_data: TimeSeries,
        val_data: TimeSeries | None = None,
        **kwargs: Any,
    ) -> None:
        """Train the model on time series data.

        Args:
            train_data: Training time series data
            val_data: Validation time series data (optional, for early stopping)
            **kwargs: Additional training parameters (model-specific)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def predict(
        self,
        n: int,
        series: TimeSeries | None = None,
    ) -> TimeSeries:
        """Predict n steps ahead.

        Args:
            n: Number of timesteps to predict (should match output_chunk_length)
            series: Historical time series data to predict from (optional)
                    If None, uses the last training data

        Returns:
            TimeSeries with predictions

        Raises:
            ValueError: If model not fitted or invalid parameters
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save trained model to disk.

        Args:
            path: File path to save model

        Raises:
            ValueError: If model not fitted
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> Self:
        """Load trained model from disk.

        Args:
            path: File path to load model from

        Returns:
            Self (the loaded model instance)

        Raises:
            FileNotFoundError: If model file doesn't exist
            NotImplementedError: If not implemented by subclass
        """
        pass

    def get_params(self) -> dict[str, Any]:
        """Get model hyperparameters.

        Returns:
            Dictionary with all hyperparameters including base parameters
        """
        return {
            "name": self.name,
            "input_chunk_length": self.input_chunk_length,
            "output_chunk_length": self.output_chunk_length,
            **self.hyperparameters,
        }

    def set_params(self, **params: Any) -> None:
        """Set model hyperparameters.

        Args:
            **params: Parameters to set (name=value)

        Note:
            Setting parameters after fitting may require retraining.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Set parameter {key}={value}")
            else:
                self.hyperparameters[key] = value
                logger.debug(f"Added hyperparameter {key}={value}")

    def validate_fit_params(self, train_data: TimeSeries) -> None:
        """Validate training data and parameters.

        Args:
            train_data: Training time series

        Raises:
            ValueError: If validation fails
        """
        if len(train_data) < self.input_chunk_length + self.output_chunk_length:
            raise ValueError(
                f"Training data too short: need at least "
                f"{self.input_chunk_length + self.output_chunk_length} timesteps, "
                f"but have {len(train_data)}"
            )

    def validate_predict_params(self, n: int) -> None:
        """Validate prediction parameters.

        Args:
            n: Number of steps to predict

        Raises:
            ValueError: If validation fails
        """
        if not self.is_fitted:
            raise ValueError(
                f"Model '{self.name}' must be fitted before prediction. Call fit() first."
            )

        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        if n != self.output_chunk_length:
            logger.warning(
                f"Requested {n} steps but model is configured for "
                f"{self.output_chunk_length} steps"
            )

    def __repr__(self) -> str:
        """String representation of model.

        Returns:
            String describing the model
        """
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"input_length={self.input_chunk_length}, "
            f"output_length={self.output_chunk_length}, "
            f"status={status})"
        )

    def __str__(self) -> str:
        """User-friendly string representation.

        Returns:
            String describing the model
        """
        return self.__repr__()
