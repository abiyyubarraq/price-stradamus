"""N-BEATS model for time series forecasting.

N-BEATS (Neural Basis Expansion Analysis for Time Series) is a deep learning
architecture specifically designed for univariate time series forecasting.

Paper: https://arxiv.org/abs/1905.10437
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from darts import TimeSeries
from darts.models import NBEATSModel as DartsNBEATS
from loguru import logger

from price_stradamus.models.base import BaseModel
from price_stradamus.models.registry import ModelRegistry
from price_stradamus.utils.helpers import set_random_seeds


@ModelRegistry.register("nbeats")
class NBEATSModel(BaseModel):
    """N-BEATS model for time series forecasting.

    N-BEATS (Neural Basis Expansion Analysis) is a deep learning architecture
    specifically designed for univariate time series forecasting. It uses
    backward and forward residual connections and a deep stack of fully
    connected layers.

    The model achieves strong performance without requiring extensive
    hyperparameter tuning, making it a good default choice for time series
    prediction.

    Paper: https://arxiv.org/abs/1905.10437

    Args:
        name: Model name (default: "nbeats")
        input_chunk_length: Number of past timesteps to use (default: 60)
        output_chunk_length: Number of future timesteps to predict (default: 5)
        num_stacks: Number of stacks in the architecture (default: 30)
        num_blocks: Number of blocks per stack (default: 1)
        num_layers: Number of fully connected layers per block (default: 4)
        layer_widths: Width of fully connected layers (default: 256)
        learning_rate: Learning rate for optimizer (default: 0.001)
        batch_size: Batch size for training (default: 32)
        n_epochs: Number of training epochs (default: 100)
        random_state: Random seed for reproducibility (default: 42)

    Example:
        model = NBEATSModel(
            name="nbeats_btc",
            input_chunk_length=60,
            output_chunk_length=5,
            n_epochs=100,
        )

        model.fit(train_series, val_series)
        predictions = model.predict(n=5)

        model.save(Path("models/nbeats_btc.pth"))
    """

    def __init__(
        self,
        name: str = "nbeats",
        input_chunk_length: int = 60,
        output_chunk_length: int = 5,
        num_stacks: int = 30,
        num_blocks: int = 1,
        num_layers: int = 4,
        layer_widths: int = 256,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 100,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """Initialize N-BEATS model."""
        super().__init__(name, input_chunk_length, output_chunk_length, **kwargs)

        # Model architecture parameters
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state

        # Store all hyperparameters
        self.hyperparameters.update(
            {
                "num_stacks": num_stacks,
                "num_blocks": num_blocks,
                "num_layers": num_layers,
                "layer_widths": layer_widths,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "random_state": random_state,
            }
        )

        logger.info(
            f"Initialized NBEATSModel (stacks={num_stacks}, blocks={num_blocks}, "
            f"layers={num_layers}, width={layer_widths})"
        )

    def fit(
        self,
        train_data: TimeSeries,
        val_data: TimeSeries | None = None,
        **kwargs: Any,
    ) -> None:
        """Train N-BEATS model.

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

        # Create Darts N-BEATS model
        self._model = DartsNBEATS(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            num_stacks=self.num_stacks,
            num_blocks=self.num_blocks,
            num_layers=self.num_layers,
            layer_widths=self.layer_widths,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            optimizer_kwargs={"lr": self.learning_rate},
            pl_trainer_kwargs={
                "accelerator": device,
                "enable_progress_bar": verbose,
                "enable_model_summary": verbose,
                # Early stopping and model checkpoint can be added here
                # "callbacks": [early_stopping_callback, checkpoint_callback],
            },
            model_name=self.name,
            random_state=self.random_state,
            force_reset=True,
            save_checkpoints=True,  # Enable automatic checkpointing every epoch
            work_dir="modelsResults/checkpoints",  # Save checkpoints here
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
                   If None, uses the last training data

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

        # Load Darts model
        self._model = DartsNBEATS.load(str(path))

        # Update fitted status
        self._is_fitted = True

        # Extract parameters from loaded model
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

        # Get PyTorch model
        pytorch_model = self._model.model  # type: ignore[union-attr]

        # Count parameters
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        trainable_params = sum(
            p.numel() for p in pytorch_model.parameters() if p.requires_grad
        )

        summary = [
            f"N-BEATS Model: {self.name}",
            "=" * 50,
            "Architecture:",
            f"  Stacks: {self.num_stacks}",
            f"  Blocks per stack: {self.num_blocks}",
            f"  Layers per block: {self.num_layers}",
            f"  Layer width: {self.layer_widths}",
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

    def get_training_history(self) -> dict[str, list[float]]:
        """Get training history (loss curves).

        Returns:
            Dictionary with training metrics

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Darts models store training history in trainer
        if (
            self._model is not None
            and hasattr(self._model, "trainer")
            and self._model.trainer is not None
            and hasattr(self._model.trainer, "callback_metrics")
        ):
            return dict(self._model.trainer.callback_metrics)

        logger.warning("Training history not available")
        return {}
