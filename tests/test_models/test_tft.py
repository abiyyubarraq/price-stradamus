"""Tests for Temporal Fusion Transformer model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from darts import TimeSeries

from price_stradamus.models.neural.tft import TFTModel
from price_stradamus.models.registry import ModelRegistry


class TestTFTModel:
    """Test TFTModel implementation."""

    def test_model_registered(self):
        """Test that TFT model is registered in the registry."""
        assert ModelRegistry.is_registered("tft")
        assert "tft" in ModelRegistry.list_models()

    def test_initialization(self):
        """Test TFT model initialization with default parameters."""
        model = ModelRegistry.create_model(
            "tft",
            input_chunk_length=60,
            output_chunk_length=5,
        )

        assert model.name == "tft"
        assert model.input_chunk_length == 60
        assert model.output_chunk_length == 5
        assert not model.is_fitted

    def test_initialization_custom_params(self):
        """Test TFT initialization with custom hyperparameters."""
        model = TFTModel(
            name="tft_custom",
            input_chunk_length=120,
            output_chunk_length=10,
            hidden_size=128,
            lstm_layers=3,
            num_attention_heads=8,
            dropout=0.2,
            n_epochs=50,
        )

        assert model.name == "tft_custom"
        assert model.input_chunk_length == 120
        assert model.output_chunk_length == 10
        assert model.hidden_size == 128
        assert model.lstm_layers == 3
        assert model.num_attention_heads == 8
        assert model.dropout == 0.2
        assert model.n_epochs == 50

    def test_hyperparameters_stored(self):
        """Test that hyperparameters are correctly stored."""
        model = ModelRegistry.create_model(
            "tft",
            hidden_size=96,
            num_attention_heads=6,
            learning_rate=0.0005,
        )

        params = model.get_params()
        assert params["hidden_size"] == 96
        assert params["num_attention_heads"] == 6
        assert params["learning_rate"] == 0.0005

    @pytest.mark.slow
    def test_fit_predict(self, sample_timeseries):
        """Test training and prediction with TFT."""
        model = ModelRegistry.create_model(
            "tft",
            input_chunk_length=20,
            output_chunk_length=5,
            hidden_size=32,  # Smaller for faster testing
            lstm_layers=1,
            num_attention_heads=2,
            n_epochs=2,  # Very few epochs for testing
            batch_size=16,
        )

        # Split data
        train = sample_timeseries[:500]

        # Train
        model.fit(train)
        assert model.is_fitted

        # Predict
        predictions = model.predict(n=5)
        assert len(predictions) == 5
        assert isinstance(predictions, TimeSeries)

    @pytest.mark.slow
    def test_fit_with_validation_data(self, sample_timeseries):
        """Test training with validation data."""
        model = ModelRegistry.create_model(
            "tft",
            input_chunk_length=20,
            output_chunk_length=5,
            hidden_size=32,
            lstm_layers=1,
            num_attention_heads=2,
            n_epochs=2,
            batch_size=16,
        )

        train = sample_timeseries[:400]
        val = sample_timeseries[400:500]

        # TFT can use validation data for early stopping
        model.fit(train, val)
        assert model.is_fitted

    @pytest.mark.slow
    def test_predict_with_series(self, sample_timeseries):
        """Test prediction with custom series."""
        model = ModelRegistry.create_model(
            "tft",
            input_chunk_length=20,
            output_chunk_length=5,
            hidden_size=32,
            lstm_layers=1,
            num_attention_heads=2,
            n_epochs=2,
            batch_size=16,
        )

        train = sample_timeseries[:500]
        test_series = sample_timeseries[500:600]

        model.fit(train)

        # Predict from different series
        predictions = model.predict(n=5, series=test_series)
        assert len(predictions) == 5

    @pytest.mark.slow
    def test_save_load(self, tmp_path, sample_timeseries):
        """Test model saving and loading."""
        model = ModelRegistry.create_model(
            "tft",
            input_chunk_length=20,
            output_chunk_length=5,
            hidden_size=32,
            lstm_layers=1,
            num_attention_heads=2,
            n_epochs=2,
            batch_size=16,
        )

        train = sample_timeseries[:500]
        model.fit(train)

        # Save
        save_path = tmp_path / "tft_test.pth"
        model.save(save_path)
        assert save_path.exists()

        # Load
        new_model = ModelRegistry.create_model("tft")
        new_model.load(save_path)
        assert new_model.is_fitted

        # Predict with loaded model
        predictions = new_model.predict(n=5)
        assert len(predictions) == 5

    def test_predict_before_fit_raises(self):
        """Test that predicting before fitting raises ValueError."""
        model = ModelRegistry.create_model(
            "tft",
            input_chunk_length=20,
            output_chunk_length=5,
        )

        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(n=5)

    def test_save_before_fit_raises(self, tmp_path):
        """Test that saving before fitting raises ValueError."""
        model = ModelRegistry.create_model(
            "tft",
            input_chunk_length=20,
            output_chunk_length=5,
        )

        save_path = tmp_path / "model.pth"
        with pytest.raises(ValueError, match="Cannot save unfitted model"):
            model.save(save_path)

    def test_invalid_training_data_length(self):
        """Test error with insufficient training data."""
        model = ModelRegistry.create_model(
            "tft",
            input_chunk_length=100,
            output_chunk_length=5,
        )

        # Create series that's too short
        short_series = TimeSeries.from_values(np.random.randn(50))

        with pytest.raises(ValueError, match="Training data too short"):
            model.fit(short_series)

    def test_load_nonexistent_file_raises(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        model = ModelRegistry.create_model("tft")

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            model.load(Path("/nonexistent/path/model.pth"))

    @pytest.mark.slow
    def test_model_summary(self, sample_timeseries):
        """Test model summary generation."""
        model = ModelRegistry.create_model(
            "tft",
            input_chunk_length=20,
            output_chunk_length=5,
            hidden_size=64,
            lstm_layers=2,
            num_attention_heads=4,
            n_epochs=2,
            batch_size=16,
        )

        train = sample_timeseries[:500]
        model.fit(train)

        summary = model.get_model_summary()
        assert "Temporal Fusion Transformer" in summary
        assert "64" in summary  # hidden_size
        assert "2" in summary  # lstm_layers
        assert "4" in summary  # num_attention_heads

    def test_model_summary_before_fit_raises(self):
        """Test that getting summary before fitting raises ValueError."""
        model = ModelRegistry.create_model("tft")

        with pytest.raises(ValueError, match="Model not fitted"):
            model.get_model_summary()

    @pytest.mark.slow
    def test_cpu_warning(self, caplog, sample_timeseries):
        """Test warning when training on CPU."""
        import torch

        # Skip if GPU is available
        if torch.cuda.is_available():
            pytest.skip("GPU available, CPU warning not triggered")

        model = ModelRegistry.create_model(
            "tft",
            input_chunk_length=20,
            output_chunk_length=5,
            hidden_size=32,
            lstm_layers=1,
            num_attention_heads=2,
            n_epochs=1,
            batch_size=16,
        )

        train = sample_timeseries[:200]
        model.fit(train)

        # Should warn about slow CPU training
        assert "TFT is slow on CPU" in caplog.text

    @pytest.mark.slow
    def test_attention_weights(self, sample_timeseries):
        """Test attention weight extraction utility."""
        model = ModelRegistry.create_model(
            "tft",
            input_chunk_length=20,
            output_chunk_length=5,
            hidden_size=32,
            lstm_layers=1,
            num_attention_heads=2,
            n_epochs=2,
            batch_size=16,
        )

        train = sample_timeseries[:500]
        model.fit(train)

        # Get attention weights info
        attention_info = model.get_attention_weights()
        assert attention_info["model_has_attention"] is True
        assert attention_info["num_attention_heads"] == 2

    def test_attention_weights_before_fit_raises(self):
        """Test that getting attention weights before fitting raises ValueError."""
        model = ModelRegistry.create_model("tft")

        with pytest.raises(ValueError, match="Model not fitted"):
            model.get_attention_weights()

    @pytest.mark.slow
    def test_prediction_output_mismatch_warning(self, sample_timeseries, caplog):
        """Test warning when requested predictions != output_chunk_length."""
        model = ModelRegistry.create_model(
            "tft",
            input_chunk_length=20,
            output_chunk_length=5,
            hidden_size=32,
            lstm_layers=1,
            num_attention_heads=2,
            n_epochs=2,
            batch_size=16,
        )

        train = sample_timeseries[:500]
        model.fit(train)

        # Request different number of steps (should warn and use 5)
        predictions = model.predict(n=10)

        assert len(predictions) == 5  # Should use output_chunk_length
        assert "Requested 10 steps but model outputs 5" in caplog.text

    @pytest.mark.slow
    def test_different_architectures(self, sample_timeseries):
        """Test TFT with different architecture configurations."""
        train = sample_timeseries[:500]

        # Small architecture
        model1 = ModelRegistry.create_model(
            "tft",
            input_chunk_length=20,
            output_chunk_length=5,
            hidden_size=32,
            lstm_layers=1,
            num_attention_heads=2,
            n_epochs=2,
            batch_size=16,
        )
        model1.fit(train)
        assert model1.is_fitted

        # Larger architecture
        model2 = ModelRegistry.create_model(
            "tft",
            input_chunk_length=20,
            output_chunk_length=5,
            hidden_size=64,
            lstm_layers=2,
            num_attention_heads=4,
            n_epochs=2,
            batch_size=16,
        )
        model2.fit(train)
        assert model2.is_fitted

    @pytest.mark.slow
    def test_parameter_count(self, sample_timeseries):
        """Test that model summary includes parameter count."""
        model = ModelRegistry.create_model(
            "tft",
            input_chunk_length=20,
            output_chunk_length=5,
            hidden_size=32,
            lstm_layers=1,
            num_attention_heads=2,
            n_epochs=2,
            batch_size=16,
        )

        train = sample_timeseries[:500]
        model.fit(train)

        summary = model.get_model_summary()
        assert "Parameters:" in summary
        assert "Total:" in summary
        assert "Trainable:" in summary

    @pytest.mark.slow
    def test_different_dropout_rates(self, sample_timeseries):
        """Test TFT with different dropout rates."""
        train = sample_timeseries[:500]

        for dropout in [0.0, 0.1, 0.3]:
            model = ModelRegistry.create_model(
                "tft",
                input_chunk_length=20,
                output_chunk_length=5,
                hidden_size=32,
                lstm_layers=1,
                num_attention_heads=2,
                dropout=dropout,
                n_epochs=2,
                batch_size=16,
            )
            model.fit(train)
            assert model.is_fitted
            predictions = model.predict(n=5)
            assert len(predictions) == 5
