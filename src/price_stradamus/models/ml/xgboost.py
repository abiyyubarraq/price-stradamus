"""XGBoost model for time series forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

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

        # Returns and covariates configuration
        self.use_returns = True  # Always use returns for ML models
        self.use_covariates = True  # Always use CORE_FEATURES
        self.last_known_price: float | None = None  # Store for inverse transform

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
                "use_returns": self.use_returns,
                "use_covariates": self.use_covariates,
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
        past_covariates: TimeSeries | None = None,
        last_known_price: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Train XGBoost model with optional covariates.

        XGBoost uses lag features instead of sliding windows. It fits once on all
        data without mini-batch training or epochs.

        Args:
            train_data: Training time series data (returns if use_returns=True)
            val_data: Validation time series data (optional, not used by XGBoost)
            past_covariates: Covariate features (e.g., technical indicators)
            last_known_price: Last known price for inverse transformation (required if use_returns=True)
            **kwargs: Additional training parameters

        Raises:
            ValueError: If training data is invalid or last_known_price is missing when needed
        """
        # Validate training data
        self.validate_fit_params(train_data)

        # Set random seed for reproducibility
        set_random_seeds(self.random_state)

        logger.info(
            f"Training {self.name} on {len(train_data)} timesteps "
            f"(validation: {len(val_data) if val_data else 0} timesteps)"
        )

        # Check GPU availability for XGBoost (not PyTorch CUDA!)
        from price_stradamus.utils.gpu_utils import get_recommended_xgboost_tree_method

        tree_method = get_recommended_xgboost_tree_method()
        device = "GPU" if tree_method == "gpu_hist" else "CPU"
        logger.info(f"Using tree_method: {tree_method} ({device})")

        # Get additional kwargs
        verbose = kwargs.get("verbose", False)

        # Create XGBoost model
        model_kwargs = {
            "lags": self.input_chunk_length,  # XGBoost uses lags instead of input_chunk_length
            "output_chunk_length": self.output_chunk_length,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "tree_method": tree_method,
            "random_state": self.random_state,
        }

        # Add covariate lags if using covariates
        if self.use_covariates and past_covariates is not None:
            model_kwargs["lags_past_covariates"] = self.input_chunk_length
            logger.info(f"Using {past_covariates.n_components} past covariates")

        self._model = DartsXGBModel(**model_kwargs)

        # Store last known price for inverse transform
        self.last_known_price = last_known_price
        if self.last_known_price:
            logger.info(f"Stored last known price: ${self.last_known_price:.2f}")

        # Train model
        logger.info("Starting training...")
        fit_kwargs = {"series": train_data, "verbose": verbose}
        if self.use_covariates and past_covariates is not None:
            fit_kwargs["past_covariates"] = past_covariates

        self._model.fit(**fit_kwargs)

        # Mark as fitted
        self._is_fitted = True
        logger.info(f"Training complete for {self.name}")

    def predict(
        self,
        n: int,
        series: TimeSeries | None = None,
        past_covariates: TimeSeries | None = None,
    ) -> TimeSeries:
        """Predict n steps ahead with returns-to-price conversion.

        For predictions longer than output_chunk_length, uses iterative
        multi-step prediction where predictions are recursively added
        to the input series.

        Args:
            n: Number of timesteps to predict
            series: Historical time series data to predict from (optional)
                   If None, uses the last training data
            past_covariates: Covariate features for prediction (must align with series)

        Returns:
            TimeSeries with predictions (in price space if use_returns=True)

        Raises:
            ValueError: If model not fitted or invalid parameters
        """
        # Validate prediction parameters
        self.validate_predict_params(n)

        logger.debug(f"Predicting {n} steps ahead")

        # Build predict kwargs
        predict_kwargs = {"n": n, "series": series}
        if self.use_covariates and past_covariates is not None:
            predict_kwargs["past_covariates"] = past_covariates

        # If requested steps <= output_chunk_length, predict directly
        if n <= self.output_chunk_length:
            predictions = self._model.predict(**predict_kwargs)  # type: ignore[union-attr]
        else:
            # For longer predictions, use iterative approach
            logger.info(
                f"Using iterative prediction: {n} steps in chunks of {self.output_chunk_length}"
            )

            # Start with historical data
            current_series = series
            current_covariates = past_covariates
            all_predictions = []

            steps_remaining = n
            iteration = 0

            while steps_remaining > 0:
                iteration += 1
                # Predict next chunk
                chunk_size = min(self.output_chunk_length, steps_remaining)

                chunk_kwargs = {"n": chunk_size, "series": current_series}
                if self.use_covariates and current_covariates is not None:
                    chunk_kwargs["past_covariates"] = current_covariates

                chunk_pred = self._model.predict(**chunk_kwargs)  # type: ignore[union-attr]

                all_predictions.append(chunk_pred)

                # Append predictions to series for next iteration
                if steps_remaining > chunk_size:
                    current_series = current_series.append(chunk_pred)  # type: ignore[union-attr]

                steps_remaining -= chunk_size
                logger.debug(
                    f"Iteration {iteration}: predicted {chunk_size} steps, {steps_remaining} remaining"
                )

            # Concatenate all prediction chunks
            predictions = all_predictions[0]
            for pred_chunk in all_predictions[1:]:
                predictions = predictions.append(pred_chunk)

        # Convert returns to prices if needed
        if self.use_returns and self.last_known_price is not None:
            from price_stradamus.utils.helpers import returns_to_prices

            returns_array = predictions.values().flatten()
            prices_array = returns_to_prices(
                returns_array,
                initial_price=self.last_known_price,
                cumulative=False,
            )

            # Create new TimeSeries with prices
            predictions = TimeSeries.from_times_and_values(
                times=predictions.time_index,
                values=prices_array.reshape(-1, 1),
                columns=["close"],
            )
            logger.debug(f"Converted {len(returns_array)} returns to prices")

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

        # Save additional metadata
        import json

        metadata = {
            "use_returns": self.use_returns,
            "use_covariates": self.use_covariates,
            "last_known_price": self.last_known_price,
        }
        metadata_path = path.with_suffix(".meta.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        logger.info(f"Model metadata saved to {metadata_path}")

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

        # Load additional metadata
        import json

        metadata_path = path.with_suffix(".meta.json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.use_returns = metadata.get("use_returns", True)
            self.use_covariates = metadata.get("use_covariates", True)
            self.last_known_price = metadata.get("last_known_price")
            logger.info("Loaded model metadata")

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

        from price_stradamus.utils.gpu_utils import check_xgboost_gpu

        gpu_available = check_xgboost_gpu()
        tree_method = "gpu_hist (GPU)" if gpu_available else "hist (CPU)"

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
            f"Tree method: {tree_method}",
            "=" * 50,
        ]

        return "\n".join(summary)
