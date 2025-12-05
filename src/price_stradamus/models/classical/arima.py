"""ARIMA model for time series forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from darts import TimeSeries
from darts.models import ARIMA as DartsARIMA
from loguru import logger

from price_stradamus.models.base import BaseModel
from price_stradamus.models.registry import ModelRegistry
from price_stradamus.utils.helpers import set_random_seeds


@ModelRegistry.register("arima")
class ARIMAModel(BaseModel):
    """ARIMA model for time series forecasting.

    AutoRegressive Integrated Moving Average (ARIMA) is a classic statistical
    approach for univariate time series prediction.

    Note: ARIMA uses all available history, not just input_chunk_length.
    The input_chunk_length parameter is kept for API consistency.
    Training can be slow for large (p, q) values or long series.

    Args:
        name: Model name (default: "arima")
        input_chunk_length: Lookback window (default: 60, for API consistency only)
        output_chunk_length: Number of future timesteps to predict (default: 5)
        p: AR (autoregressive) order (default: 5)
        d: Differencing order (default: 1)
        q: MA (moving average) order (default: 5)
        seasonal: Enable seasonal ARIMA (default: False)
        P: Seasonal AR order (default: 1)
        D: Seasonal differencing order (default: 0)
        Q: Seasonal MA order (default: 1)
        m: Seasonal period in timesteps (default: 60)
        random_state: Random seed for reproducibility (default: 42)
        **kwargs: Additional hyperparameters

    Example:
        >>> model = ARIMAModel(
        ...     name="arima_btc",
        ...     p=5,
        ...     d=1,
        ...     q=5,
        ...     output_chunk_length=5,
        ... )
        >>> model.fit(train_series)
        >>> predictions = model.predict(n=5)
        >>> model.save(Path("models/arima_btc.pkl"))
    """

    def __init__(
        self,
        name: str = "arima",
        input_chunk_length: int = 60,
        output_chunk_length: int = 5,
        p: int = 5,
        d: int = 1,
        q: int = 5,
        seasonal: bool = False,
        P: int = 1,
        D: int = 0,
        Q: int = 1,
        m: int = 60,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """Initialize ARIMA model."""
        super().__init__(name, input_chunk_length, output_chunk_length, **kwargs)

        # Model hyperparameters
        self.p = p
        self.d = d
        self.q = q
        self.seasonal = seasonal
        self.P = P
        self.D = D
        self.Q = Q
        self.m = m
        self.random_state = random_state

        # Store all hyperparameters
        self.hyperparameters.update(
            {
                "p": p,
                "d": d,
                "q": q,
                "seasonal": seasonal,
                "P": P,
                "D": D,
                "Q": Q,
                "m": m,
                "random_state": random_state,
            }
        )

        if seasonal:
            logger.info(f"Initialized ARIMA({p},{d},{q})x({P},{D},{Q},{m}) model")
        else:
            logger.info(f"Initialized ARIMA({p},{d},{q}) model")

    def fit(
        self,
        train_data: TimeSeries,
        val_data: TimeSeries | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Train ARIMA model.

        ARIMA fits on all available data at once using maximum likelihood estimation.
        No mini-batch training. Can be slow for large (p, q) or long series.

        Args:
            train_data: Training time series data
            val_data: Validation time series data (optional, not used by ARIMA)
            **kwargs: Additional training parameters

        Raises:
            ValueError: If training data is invalid or too short
        """
        # Validate training data
        self.validate_fit_params(train_data)

        # Set random seed for reproducibility
        set_random_seeds(self.random_state)

        # Check minimum data length for ARIMA
        min_length = self.p + self.q + self.d
        if len(train_data) < min_length:
            raise ValueError(
                f"Training data too short for ARIMA({self.p},{self.d},{self.q}). "
                f"Need at least {min_length} samples, got {len(train_data)}."
            )

        logger.info(
            f"Training {self.name} on {len(train_data)} timesteps "
            f"(validation: {len(val_data) if val_data else 0} timesteps)"
        )

        # Warn if p or q is large (slow training)
        if self.p > 10 or self.q > 10:
            logger.warning(
                f"Large ARIMA parameters (p={self.p}, q={self.q}). "
                "Training may be slow."
            )

        # Create seasonal order tuple
        if self.seasonal:
            seasonal_order = (self.P, self.D, self.Q, self.m)
            logger.info(f"Using seasonal ARIMA with period m={self.m}")
        else:
            seasonal_order = (0, 0, 0, 0)

        # Create ARIMA model
        self._model = DartsARIMA(
            p=self.p,
            d=self.d,
            q=self.q,
            seasonal_order=seasonal_order,
            random_state=self.random_state,
        )

        # Train model
        logger.info("Starting training...")

        # ARIMA may raise convergence warnings - these are usually fine
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
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

        # ARIMA can predict any number of steps, but we prefer output_chunk_length
        if n != self.output_chunk_length:
            logger.warning(
                f"Requested {n} steps but model was configured for {self.output_chunk_length}. "
                f"Will predict {n} steps (ARIMA supports variable horizon)."
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

        # Darts models have built-in save method (uses pickle for ARIMA)
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

        # Load Darts ARIMA model
        self._model = DartsARIMA.load(str(path))

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
            f"ARIMA Model: {self.name}",
            "=" * 50,
            "Model Type: Statistical (ARIMA)",
            "",
            "Input/Output:",
            "  Uses all available history (not limited by input_chunk_length)",
            f"  Output chunk length: {self.output_chunk_length}",
            "",
            "ARIMA Parameters:",
            f"  AR order (p): {self.p}",
            f"  Differencing (d): {self.d}",
            f"  MA order (q): {self.q}",
        ]

        if self.seasonal:
            summary.extend(
                [
                    "",
                    "Seasonal ARIMA Parameters:",
                    f"  Seasonal AR (P): {self.P}",
                    f"  Seasonal differencing (D): {self.D}",
                    f"  Seasonal MA (Q): {self.Q}",
                    f"  Seasonal period (m): {self.m}",
                ]
            )

        summary.append("=" * 50)

        return "\n".join(summary)
