"""TCN model for time series forecasting.

TCN (Temporal Convolutional Network) uses causal dilated convolutions
for fast and effective sequence modeling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from darts import TimeSeries
from darts.models import TCNModel as DartsTCN
from loguru import logger

from price_stradamus.models.base import BaseModel
from price_stradamus.models.registry import ModelRegistry
from price_stradamus.utils.helpers import set_random_seeds


@ModelRegistry.register("tcn")
class TCNModel(BaseModel):
    """TCN model for time series forecasting.

    TCN (Temporal Convolutional Network) uses 1D causal dilated convolutions
    to capture long-range dependencies in time series data. Key advantages:
    - Parallel computation (faster than RNNs)
    - Flexible receptive field through dilated convolutions
    - No vanishing gradient problems
    - Memory efficient

    Paper: https://arxiv.org/abs/1803.01271

    Args:
        name: Model name (default: "tcn")
        input_chunk_length: Number of past timesteps to use (default: 60)
        output_chunk_length: Number of future timesteps to predict (default: 5)
        kernel_size: Size of convolutional kernel (default: 3)
        num_filters: Number of filters in convolutional layers (default: 64)
        num_layers: Number of convolutional layers (default: 3)
        dilation_base: Base for exponential dilation (default: 2)
        dropout: Dropout rate for regularization (default: 0.1)
        learning_rate: Learning rate for optimizer (default: 0.001)
        batch_size: Batch size for training (default: 32)
        n_epochs: Number of training epochs (default: 100)
        random_state: Random seed for reproducibility (default: 42)

    Example:
        model = TCNModel(
            name="tcn_btc",
            input_chunk_length=60,
            output_chunk_length=5,
            num_filters=64,
            num_layers=3,
            n_epochs=100,
        )

        model.fit(train_series, val_series)
        predictions = model.predict(n=5)

        model.save(Path("models/tcn_btc.pth"))
    """

    def __init__(
        self,
        name: str = "tcn",
        input_chunk_length: int = 60,
        output_chunk_length: int = 5,
        kernel_size: int = 3,
        num_filters: int = 64,
        num_layers: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 100,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """Initialize TCN model."""
        super().__init__(name, input_chunk_length, output_chunk_length, **kwargs)

        # Model architecture parameters
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.dilation_base = dilation_base
        self.dropout = dropout

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state

        # Returns and covariates configuration
        self.use_returns = True  # Always use returns for neural models
        self.use_covariates = True  # Always use CORE_FEATURES
        self.last_known_price: float | None = None  # Store for inverse transform

        # Store all hyperparameters
        self.hyperparameters.update(
            {
                "kernel_size": kernel_size,
                "num_filters": num_filters,
                "num_layers": num_layers,
                "dilation_base": dilation_base,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "random_state": random_state,
                "use_returns": self.use_returns,
                "use_covariates": self.use_covariates,
            }
        )

        logger.info(
            f"Initialized TCNModel (filters={num_filters}, layers={num_layers}, "
            f"kernel_size={kernel_size})"
        )

    def fit(
        self,
        train_data: TimeSeries,
        val_data: TimeSeries | None = None,
        past_covariates: TimeSeries | None = None,
        last_known_price: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Train TCN model with optional covariates.

        Args:
            train_data: Training time series data (returns if use_returns=True)
            val_data: Validation time series data (optional, for early stopping)
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

        # Store last known price for inverse transform
        self.last_known_price = last_known_price
        if self.last_known_price:
            logger.info(f"Stored last known price: ${self.last_known_price:.2f}")

        # Log covariate information
        if self.use_covariates and past_covariates is not None:
            logger.info(f"Using {past_covariates.n_components} past covariates")

        # Check GPU availability
        device = "gpu" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Get additional kwargs
        verbose = kwargs.get("verbose", True)

        # Create Darts TCN model
        self._model = DartsTCN(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            kernel_size=self.kernel_size,
            num_filters=self.num_filters,
            num_layers=self.num_layers,
            dilation_base=self.dilation_base,
            dropout=self.dropout,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            optimizer_kwargs={"lr": self.learning_rate},
            pl_trainer_kwargs={
                "accelerator": device,
                "enable_progress_bar": verbose,
                "enable_model_summary": verbose,
            },
            model_name=self.name,
            random_state=self.random_state,
            force_reset=True,
            save_checkpoints=False,
        )

        # Train model
        logger.info("Starting training...")
        fit_kwargs = {"series": train_data, "val_series": val_data, "verbose": verbose}
        if self.use_covariates and past_covariates is not None:
            fit_kwargs["past_covariates"] = past_covariates

        self._model.fit(**fit_kwargs)

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
        """Save trained model to disk with metadata.

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

        # Darts models have built-in save method
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
        """Load trained model from disk with metadata.

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

        # Load Darts TCN model
        self._model = DartsTCN.load(str(path))

        # Update fitted status
        self._is_fitted = True

        # Extract parameters from loaded model
        self.input_chunk_length = self._model.input_chunk_length
        self.output_chunk_length = self._model.output_chunk_length

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
        else:
            logger.warning(f"No metadata file found at {metadata_path}, using defaults")

        logger.info(f"Model loaded successfully from {path}")
        return self

    def get_model_summary(self) -> str:
        """Get summary of model architecture.

        Returns:
            String summary of model architecture

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        pytorch_model = self._model.model  # type: ignore[union-attr]
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        trainable_params = sum(
            p.numel() for p in pytorch_model.parameters() if p.requires_grad
        )

        # Calculate receptive field
        receptive_field = 1 + 2 * (self.kernel_size - 1) * sum(
            self.dilation_base**i for i in range(self.num_layers)
        )

        summary = [
            f"TCN Model: {self.name}",
            "=" * 50,
            "Architecture:",
            f"  Kernel size: {self.kernel_size}",
            f"  Num filters: {self.num_filters}",
            f"  Num layers: {self.num_layers}",
            f"  Dilation base: {self.dilation_base}",
            f"  Dropout: {self.dropout}",
            f"  Receptive field: {receptive_field}",
            "",
            "Input/Output:",
            f"  Input chunk length: {self.input_chunk_length}",
            f"  Output chunk length: {self.output_chunk_length}",
            "",
            "Training:",
            f"  Epochs: {self.n_epochs}",
            f"  Batch size: {self.batch_size}",
            f"  Learning rate: {self.learning_rate}",
            "",
            "Parameters:",
            f"  Total: {total_params:,}",
            f"  Trainable: {trainable_params:,}",
            "=" * 50,
        ]

        return "\n".join(summary)
