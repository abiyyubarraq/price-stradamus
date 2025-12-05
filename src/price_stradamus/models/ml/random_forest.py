"""Random Forest model for time series forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from darts import TimeSeries
from darts.models import RandomForest as DartsRandomForest
from loguru import logger

from price_stradamus.models.base import BaseModel
from price_stradamus.models.registry import ModelRegistry
from price_stradamus.utils.helpers import set_random_seeds


@ModelRegistry.register("random_forest")
class RandomForestModel(BaseModel):
    """Random Forest model for time series forecasting.

    Ensemble of decision trees for time series prediction. Random Forest is a
    fast and robust baseline model that works well for many time series tasks.

    Args:
        name: Model name (default: "random_forest")
        input_chunk_length: Lookback window size (default: 60)
        output_chunk_length: Number of future timesteps to predict (default: 5)
        n_estimators: Number of trees in the forest (default: 100)
        max_depth: Maximum tree depth (default: 10)
        min_samples_split: Minimum samples required to split node (default: 2)
        min_samples_leaf: Minimum samples required at leaf node (default: 1)
        max_features: Number of features to consider for best split (default: "sqrt")
        random_state: Random seed for reproducibility (default: 42)
        n_jobs: Number of parallel jobs (-1 uses all cores, default: -1)
        **kwargs: Additional hyperparameters

    Example:
        >>> model = RandomForestModel(
        ...     name="rf_btc",
        ...     input_chunk_length=60,
        ...     output_chunk_length=5,
        ...     n_estimators=100,
        ...     n_jobs=-1,
        ... )
        >>> model.fit(train_series)
        >>> predictions = model.predict(n=5)
        >>> model.save(Path("models/rf_btc.pkl"))
    """

    def __init__(
        self,
        name: str = "random_forest",
        input_chunk_length: int = 60,
        output_chunk_length: int = 5,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ):
        """Initialize Random Forest model."""
        super().__init__(name, input_chunk_length, output_chunk_length, **kwargs)

        # Model hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Store all hyperparameters
        self.hyperparameters.update(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "random_state": random_state,
                "n_jobs": n_jobs,
            }
        )

        logger.info(
            f"Initialized RandomForestModel (n_estimators={n_estimators}, "
            f"max_depth={max_depth}, n_jobs={n_jobs})"
        )

    def fit(
        self,
        train_data: TimeSeries,
        val_data: TimeSeries | None = None,
        **kwargs: Any,
    ) -> None:
        """Train Random Forest model.

        Random Forest uses lag features instead of sliding windows. It fits once
        on all data without mini-batch training or epochs.

        Args:
            train_data: Training time series data
            val_data: Validation time series data (optional, not used by Random Forest)
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
        logger.info(f"Using n_jobs={self.n_jobs} for parallel training")

        # Get additional kwargs
        verbose = kwargs.get("verbose", False)

        # Create Random Forest model
        self._model = DartsRandomForest(
            lags=self.input_chunk_length,  # RF uses lags instead of input_chunk_length
            output_chunk_length=self.output_chunk_length,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
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

        # Darts models have built-in save method (uses pickle for Random Forest)
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

        # Load Darts Random Forest model
        self._model = DartsRandomForest.load(str(path))

        # Update fitted status
        self._is_fitted = True

        # Extract parameters from loaded model
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
            f"Random Forest Model: {self.name}",
            "=" * 50,
            "Model Type: Ensemble (Tree-based)",
            "",
            "Input/Output:",
            f"  Input chunk length (lags): {self.input_chunk_length}",
            f"  Output chunk length: {self.output_chunk_length}",
            "",
            "Hyperparameters:",
            f"  Number of estimators: {self.n_estimators}",
            f"  Max depth: {self.max_depth}",
            f"  Min samples split: {self.min_samples_split}",
            f"  Min samples leaf: {self.min_samples_leaf}",
            f"  Max features: {self.max_features}",
            "",
            f"Parallel jobs: {self.n_jobs}",
            "=" * 50,
        ]

        return "\n".join(summary)
