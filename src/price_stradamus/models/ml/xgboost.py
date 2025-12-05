"""XGBoost model for time series forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from darts import TimeSeries
from darts.models import XGBModel as DartsXGBModel
from loguru import logger

from price_stradamus.models.base import BaseModel
from price_stradamus.models.registry import ModelRegistry
from price_stradamus.utils.helpers import set_random_seeds


@ModelRegistry.register("xgboost")
class XGBoostModel(BaseModel):
    """XGBoost model for time series forecasting.

    Gradient boosting for time series with lag features. XGBoost is a fast and
    efficient tree-based model that works well as a baseline for time series prediction.

    Paper: https://arxiv.org/abs/1603.02754

    Args:
        name: Model name (default: "xgboost")
        input_chunk_length: Lookback window size (default: 60)
        output_chunk_length: Number of future timesteps to predict (default: 5)
        n_estimators: Number of boosting rounds (default: 100)
        max_depth: Maximum tree depth (default: 6)
        learning_rate: Boosting learning rate (default: 0.1)
        min_child_weight: Minimum sum of instance weight in child (default: 1)
        subsample: Subsample ratio of training data (default: 0.8)
        colsample_bytree: Subsample ratio of columns per tree (default: 0.8)
        random_state: Random seed for reproducibility (default: 42)
        **kwargs: Additional hyperparameters

    Example:
        >>> model = XGBoostModel(
        ...     name="xgboost_btc",
        ...     input_chunk_length=60,
        ...     output_chunk_length=5,
        ...     n_estimators=100,
        ... )
        >>> model.fit(train_series)
        >>> predictions = model.predict(n=5)
        >>> model.save(Path("models/xgboost_btc.pkl"))
    """

    def __init__(
        self,
        name: str = "xgboost",
        input_chunk_length: int = 60,
        output_chunk_length: int = 5,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """Initialize XGBoost model."""
        super().__init__(name, input_chunk_length, output_chunk_length, **kwargs)

        # Model hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        # Store all hyperparameters
        self.hyperparameters.update(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "min_child_weight": min_child_weight,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "random_state": random_state,
            }
        )

        logger.info(
            f"Initialized XGBoostModel (n_estimators={n_estimators}, "
            f"max_depth={max_depth}, learning_rate={learning_rate})"
        )

    def fit(
        self,
        train_data: TimeSeries,
        val_data: TimeSeries | None = None,
        **kwargs: Any,
    ) -> None:
        """Train XGBoost model.

        XGBoost uses lag features instead of sliding windows. It fits once on all
        data without mini-batch training or epochs.

        Args:
            train_data: Training time series data
            val_data: Validation time series data (optional, not used by XGBoost)
            **kwargs: Additional training parameters

        Raises:
            ValueError: If training data is invalid
        """
        # Validate training data
        self.validate_fit_params(train_data)

        # Set random seed for reproducibility
        set_random_seeds(self.random_state)

        logger.info(
            f"Training {self.name} on {len(train_data)} timesteps "
            f"(validation: {len(val_data) if val_data else 0} timesteps)"
        )

        # Check GPU availability
        tree_method = "gpu_hist" if torch.cuda.is_available() else "auto"
        logger.info(f"Using tree_method: {tree_method}")

        # Get additional kwargs
        verbose = kwargs.get("verbose", False)

        # Create XGBoost model
        self._model = DartsXGBModel(
            lags=self.input_chunk_length,  # XGBoost uses lags instead of input_chunk_length
            output_chunk_length=self.output_chunk_length,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method=tree_method,
            random_state=self.random_state,
        )

        # Train model
        logger.info("Starting training...")
        self._model.fit(
            series=train_data,
            verbose=verbose,
        )

        # Mark as fitted
        self._is_fitted = True
        logger.info(f"Training complete for {self.name}")

    def predict(
        self,
        n: int,
        series: TimeSeries | None = None,
    ) -> TimeSeries:
        """Predict n steps ahead.

        Args:
            n: Number of timesteps to predict
            series: Historical time series data to predict from (optional)
                   If None, uses the last training data

        Returns:
            TimeSeries with predictions

        Raises:
            ValueError: If model not fitted or invalid parameters
        """
        # Validate prediction parameters
        self.validate_predict_params(n)

        # Handle mismatch between requested and configured output length
        if n != self.output_chunk_length:
            logger.warning(
                f"Requested {n} steps but model outputs {self.output_chunk_length}. "
                f"Will predict {self.output_chunk_length} steps."
            )
            n = self.output_chunk_length

        logger.debug(f"Predicting {n} steps ahead")

        # Make prediction
        predictions = self._model.predict(n=n, series=series)  # type: ignore[union-attr]

        logger.info(f"Generated {len(predictions)} predictions")
        return predictions

    def save(self, path: Path) -> None:
        """Save trained model to disk.

        Args:
            path: File path to save model (will create parent directories)

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")

        # Ensure parent directory exists
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Darts models have built-in save method (uses pickle for XGBoost)
        self._model.save(str(path))  # type: ignore[union-attr]

        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> Self:
        """Load trained model from disk.

        Args:
            path: File path to load model from

        Returns:
            Self (this model instance with loaded weights)

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        logger.info(f"Loading model from {path}")

        # Load Darts XGBoost model
        self._model = DartsXGBModel.load(str(path))

        # Update fitted status
        self._is_fitted = True

        # Extract parameters from loaded model
        # Note: XGBoost models may not expose all parameters after loading
        logger.info(f"Model loaded successfully from {path}")
        return self

    def get_model_summary(self) -> str:
        """Get summary of model configuration.

        Returns:
            String summary of model configuration

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        summary = [
            f"XGBoost Model: {self.name}",
            "=" * 50,
            "Model Type: Gradient Boosting (Tree-based)",
            "",
            "Input/Output:",
            f"  Input chunk length (lags): {self.input_chunk_length}",
            f"  Output chunk length: {self.output_chunk_length}",
            "",
            "Hyperparameters:",
            f"  Number of estimators: {self.n_estimators}",
            f"  Max depth: {self.max_depth}",
            f"  Learning rate: {self.learning_rate}",
            f"  Min child weight: {self.min_child_weight}",
            f"  Subsample: {self.subsample}",
            f"  Column subsample: {self.colsample_bytree}",
            "",
            f"Tree method: {'gpu_hist (GPU)' if torch.cuda.is_available() else 'auto (CPU)'}",
            "=" * 50,
        ]

        return "\n".join(summary)
