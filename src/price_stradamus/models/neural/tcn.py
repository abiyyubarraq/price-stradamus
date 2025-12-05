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
        **kwargs: Any,
    ) -> None:
        """Train TCN model.

        Args:
            train_data: Training time series data
            val_data: Validation time series data (optional, for early stopping)
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
        self._model.fit(
            series=train_data,
            val_series=val_data,
            verbose=verbose,
        )

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

        Returns:
            TimeSeries with predictions

        Raises:
            ValueError: If model not fitted or invalid parameters
        """
        # Validate prediction parameters
        self.validate_predict_params(n)

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
            path: File path to save model

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

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

        self._model = DartsTCN.load(str(path))
        self._is_fitted = True

        self.input_chunk_length = self._model.input_chunk_length
        self.output_chunk_length = self._model.output_chunk_length

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
