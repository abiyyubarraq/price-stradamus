"""Temporal Fusion Transformer model for time series forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from darts import TimeSeries
from darts.models import TFTModel as DartsTFTModel
from loguru import logger

from price_stradamus.models.base import BaseModel
from price_stradamus.models.registry import ModelRegistry
from price_stradamus.utils.helpers import set_random_seeds


@ModelRegistry.register("tft")
class TFTModel(BaseModel):
    """Temporal Fusion Transformer for time series forecasting.

    State-of-the-art attention-based model with variable selection and
    interpretable multi-horizon forecasting. TFT combines LSTM for local
    processing with multi-head attention for long-term dependencies.

    Paper: https://arxiv.org/abs/1912.09363

    Most complex and computationally expensive model in the suite.
    Provides interpretable attention weights for feature importance.
    Can leverage covariates (future work) for enhanced predictions.

    Args:
        name: Model name (default: "tft")
        input_chunk_length: Lookback window size (default: 60)
        output_chunk_length: Number of future timesteps to predict (default: 5)
        hidden_size: Hidden state size (default: 64)
        lstm_layers: Number of LSTM layers (default: 2)
        num_attention_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.1)
        hidden_continuous_size: Hidden size for continuous variables (default: 8)
        learning_rate: Learning rate (default: 0.001)
        batch_size: Batch size for training (default: 32)
        n_epochs: Number of training epochs (default: 100)
        random_state: Random seed for reproducibility (default: 42)
        **kwargs: Additional hyperparameters

    Example:
        >>> model = TFTModel(
        ...     name="tft_btc",
        ...     input_chunk_length=60,
        ...     output_chunk_length=5,
        ...     hidden_size=64,
        ...     n_epochs=100,
        ... )
        >>> model.fit(train_series, val_series)
        >>> predictions = model.predict(n=5)
        >>> model.save(Path("models/tft_btc.pth"))
    """

    def __init__(
        self,
        name: str = "tft",
        input_chunk_length: int = 60,
        output_chunk_length: int = 5,
        hidden_size: int = 64,
        lstm_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 100,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """Initialize TFT model."""
        super().__init__(name, input_chunk_length, output_chunk_length, **kwargs)

        # Model architecture parameters
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state

        # Store all hyperparameters
        self.hyperparameters.update(
            {
                "hidden_size": hidden_size,
                "lstm_layers": lstm_layers,
                "num_attention_heads": num_attention_heads,
                "dropout": dropout,
                "hidden_continuous_size": hidden_continuous_size,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "random_state": random_state,
            }
        )

        logger.info(
            f"Initialized TFTModel (hidden_size={hidden_size}, "
            f"lstm_layers={lstm_layers}, attention_heads={num_attention_heads})"
        )

    def fit(
        self,
        train_data: TimeSeries,
        val_data: TimeSeries | None = None,
        **kwargs: Any,
    ) -> None:
        """Train TFT model.

        TFT uses attention mechanisms and can handle covariates.
        Requires significant GPU memory (reduce batch_size if OOM).

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

        if device == "cpu":
            logger.warning(
                "TFT is slow on CPU and memory-intensive. "
                "Consider using GPU or a simpler model (LSTM, TCN)."
            )

        # Get additional kwargs
        verbose = kwargs.get("verbose", True)

        # Create TFT model
        self._model = DartsTFTModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
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
            save_checkpoints=False,  # We handle saving manually
        )

        # Train model
        logger.info("Starting training...")
        self._model.fit(
            series=train_data,
            val_series=val_data,
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

        # Load Darts TFT model
        self._model = DartsTFTModel.load(str(path))

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
            f"Temporal Fusion Transformer: {self.name}",
            "=" * 50,
            "Model Type: Transformer (Attention-based)",
            "",
            "Architecture:",
            f"  Hidden size: {self.hidden_size}",
            f"  LSTM layers: {self.lstm_layers}",
            f"  Attention heads: {self.num_attention_heads}",
            f"  Dropout: {self.dropout}",
            f"  Hidden continuous size: {self.hidden_continuous_size}",
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
            "",
            f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}",
            "=" * 50,
        ]

        return "\n".join(summary)

    def get_attention_weights(self) -> dict[str, Any]:
        """Get attention weights for interpretability.

        Note: Extracting attention weights requires a forward pass with data.
        This is an optional utility method for model interpretability.

        Returns:
            Dictionary with note about attention weight extraction

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Note: Full attention weight extraction requires forward pass
        # This would need input data to compute
        return {
            "note": "Attention weights require forward pass with data",
            "model_has_attention": True,
            "num_attention_heads": self.num_attention_heads,
        }
