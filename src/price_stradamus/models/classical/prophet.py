"""Prophet model for time series forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from darts import TimeSeries
from darts.models import Prophet as DartsProphet
from loguru import logger

from price_stradamus.models.base import BaseModel
from price_stradamus.models.registry import ModelRegistry
from price_stradamus.utils.helpers import set_random_seeds


@ModelRegistry.register("prophet")
class ProphetModel(BaseModel):
    """Prophet model for time series forecasting.

    Facebook's Prophet algorithm for time series with strong seasonality.
    Prophet decomposes time series into trend, seasonal, and holiday components.

    Paper: https://peerj.com/preprints/3190/

    Note: Prophet uses all available history, not just input_chunk_length.
    The input_chunk_length parameter is kept for API consistency but not
    used internally by Prophet.

    Args:
        name: Model name (default: "prophet")
        input_chunk_length: Lookback window (default: 60, for API consistency only)
        output_chunk_length: Number of future timesteps to predict (default: 5)
        growth: Trend growth type - "linear" or "logistic" (default: "linear")
        changepoint_prior_scale: Flexibility of trend changes (default: 0.05)
        seasonality_prior_scale: Flexibility of seasonality (default: 10.0)
        seasonality_mode: "additive" or "multiplicative" (default: "multiplicative")
        yearly_seasonality: Enable yearly seasonality (default: False)
        weekly_seasonality: Enable weekly seasonality (default: True)
        daily_seasonality: Enable daily seasonality (default: True)
        **kwargs: Additional hyperparameters

    Example:
        >>> model = ProphetModel(
        ...     name="prophet_btc",
        ...     output_chunk_length=5,
        ...     seasonality_mode="multiplicative",
        ...     daily_seasonality=True,
        ... )
        >>> model.fit(train_series)
        >>> predictions = model.predict(n=5)
        >>> model.save(Path("models/prophet_btc.pkl"))
    """

    def __init__(
        self,
        name: str = "prophet",
        input_chunk_length: int = 60,
        output_chunk_length: int = 5,
        growth: str = "linear",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """Initialize Prophet model."""
        super().__init__(name, input_chunk_length, output_chunk_length, **kwargs)

        # Model hyperparameters
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.random_state = random_state

        # Store all hyperparameters
        self.hyperparameters.update(
            {
                "growth": growth,
                "changepoint_prior_scale": changepoint_prior_scale,
                "seasonality_prior_scale": seasonality_prior_scale,
                "seasonality_mode": seasonality_mode,
                "yearly_seasonality": yearly_seasonality,
                "weekly_seasonality": weekly_seasonality,
                "daily_seasonality": daily_seasonality,
                "random_state": random_state,
            }
        )

        logger.info(
            f"Initialized ProphetModel (growth={growth}, "
            f"seasonality_mode={seasonality_mode})"
        )

    def fit(
        self,
        train_data: TimeSeries,
        val_data: TimeSeries | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Train Prophet model.

        Prophet fits on all available data at once. It does not use mini-batch
        training or epochs. Validation data is not used by Prophet.

        Args:
            train_data: Training time series data
            val_data: Validation time series data (optional, not used by Prophet)
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
        logger.info(
            f"Seasonality: yearly={self.yearly_seasonality}, "
            f"weekly={self.weekly_seasonality}, daily={self.daily_seasonality}"
        )

        # Warn if dataset is very large (Prophet can be slow)
        if len(train_data) > 10000:
            logger.warning(
                f"Training on {len(train_data)} samples. Prophet may be slow "
                "on large datasets (>10k samples)."
            )

        # Create Prophet model
        self._model = DartsProphet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
        )

        # Train model (Prophet uses all data at once)
        logger.info("Starting training...")
        self._model.fit(series=train_data)

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

        # Prophet can predict any number of steps, but we prefer output_chunk_length
        if n != self.output_chunk_length:
            logger.warning(
                f"Requested {n} steps but model was configured for {self.output_chunk_length}. "
                f"Will predict {n} steps (Prophet supports variable horizon)."
            )

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

        # Darts models have built-in save method (uses pickle for Prophet)
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

        # Load Darts Prophet model
        self._model = DartsProphet.load(str(path))

        # Update fitted status
        self._is_fitted = True

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
            f"Prophet Model: {self.name}",
            "=" * 50,
            "Model Type: Statistical (Decomposition)",
            "",
            "Input/Output:",
            "  Uses all available history (not limited by input_chunk_length)",
            f"  Output chunk length: {self.output_chunk_length}",
            "",
            "Trend:",
            f"  Growth: {self.growth}",
            f"  Changepoint flexibility: {self.changepoint_prior_scale}",
            "",
            "Seasonality:",
            f"  Mode: {self.seasonality_mode}",
            f"  Seasonality flexibility: {self.seasonality_prior_scale}",
            f"  Yearly: {self.yearly_seasonality}",
            f"  Weekly: {self.weekly_seasonality}",
            f"  Daily: {self.daily_seasonality}",
            "=" * 50,
        ]

        return "\n".join(summary)
