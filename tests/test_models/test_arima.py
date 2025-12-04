"""Tests for ARIMA model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from darts import TimeSeries

from price_stradamus.models.classical.arima import ARIMAModel
from price_stradamus.models.registry import ModelRegistry


class TestARIMAModel:
    """Test ARIMAModel implementation."""

    def test_model_registered(self):
        """Test that ARIMA model is registered in the registry."""
        assert ModelRegistry.is_registered("arima")
        assert "arima" in ModelRegistry.list_models()

    def test_initialization(self):
        """Test ARIMA model initialization with default parameters."""
        model = ModelRegistry.create_model(
            "arima",
            output_chunk_length=5,
        )

        assert model.name == "arima"
        assert model.output_chunk_length == 5
        assert not model.is_fitted

    def test_initialization_custom_params(self):
        """Test ARIMA initialization with custom hyperparameters."""
        model = ARIMAModel(
            name="arima_custom",
            output_chunk_length=10,
            p=3,
            d=2,
            q=3,
            seasonal=True,
            P=2,
            m=24,
        )

        assert model.name == "arima_custom"
        assert model.output_chunk_length == 10
        assert model.p == 3
        assert model.d == 2
        assert model.q == 3
        assert model.seasonal is True
        assert model.P == 2
        assert model.m == 24

    def test_hyperparameters_stored(self):
        """Test that hyperparameters are correctly stored."""
        model = ModelRegistry.create_model(
            "arima",
            p=7,
            d=1,
            q=7,
        )

        params = model.get_params()
        assert params["p"] == 7
        assert params["d"] == 1
        assert params["q"] == 7

    @pytest.mark.slow
    def test_fit_predict(self, sample_timeseries):
        """Test training and prediction with ARIMA."""
        model = ModelRegistry.create_model(
            "arima",
            p=2,
            d=1,
            q=2,
            output_chunk_length=5,
        )

        # Use smaller dataset for faster testing
        train = sample_timeseries[:300]

        # Train
        model.fit(train)
        assert model.is_fitted

        # Predict
        predictions = model.predict(n=5)
        assert len(predictions) == 5
        assert isinstance(predictions, TimeSeries)

    @pytest.mark.slow
    def test_fit_with_validation_data(self, sample_timeseries):
        """Test training with validation data (not used by ARIMA)."""
        model = ModelRegistry.create_model(
            "arima",
            p=2,
            d=1,
            q=2,
            output_chunk_length=5,
        )

        train = sample_timeseries[:250]
        val = sample_timeseries[250:300]

        # ARIMA doesn't use validation data but should accept it
        model.fit(train, val)
        assert model.is_fitted

    @pytest.mark.slow
    def test_predict_with_series(self, sample_timeseries):
        """Test prediction with custom series."""
        model = ModelRegistry.create_model(
            "arima",
            p=2,
            d=1,
            q=2,
            output_chunk_length=5,
        )

        train = sample_timeseries[:300]
        test_series = sample_timeseries[300:400]

        model.fit(train)

        # Predict from different series
        predictions = model.predict(n=5, series=test_series)
        assert len(predictions) == 5

    @pytest.mark.slow
    def test_variable_prediction_horizon(self, sample_timeseries):
        """Test ARIMA with variable prediction horizons."""
        model = ModelRegistry.create_model(
            "arima",
            p=2,
            d=1,
            q=2,
            output_chunk_length=5,
        )

        train = sample_timeseries[:300]
        model.fit(train)

        # ARIMA supports variable horizon
        for n in [3, 5, 10]:
            predictions = model.predict(n=n)
            assert len(predictions) == n

    def test_save_load(self, tmp_path, sample_timeseries):
        """Test model saving and loading."""
        model = ModelRegistry.create_model(
            "arima",
            p=2,
            d=1,
            q=2,
            output_chunk_length=5,
        )

        train = sample_timeseries[:300]
        model.fit(train)

        # Save
        save_path = tmp_path / "arima_test.pkl"
        model.save(save_path)
        assert save_path.exists()

        # Load
        new_model = ModelRegistry.create_model("arima")
        new_model.load(save_path)
        assert new_model.is_fitted

        # Predict with loaded model
        predictions = new_model.predict(n=5)
        assert len(predictions) == 5

    def test_predict_before_fit_raises(self):
        """Test that predicting before fitting raises ValueError."""
        model = ModelRegistry.create_model(
            "arima",
            output_chunk_length=5,
        )

        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(n=5)

    def test_save_before_fit_raises(self, tmp_path):
        """Test that saving before fitting raises ValueError."""
        model = ModelRegistry.create_model(
            "arima",
            output_chunk_length=5,
        )

        save_path = tmp_path / "model.pkl"
        with pytest.raises(ValueError, match="Cannot save unfitted model"):
            model.save(save_path)

    def test_insufficient_data_length_raises(self):
        """Test error when training data is too short for ARIMA parameters."""
        model = ModelRegistry.create_model(
            "arima",
            p=10,
            d=1,
            q=10,
            output_chunk_length=5,
        )

        # Create series too short for p=10, d=1, q=10 (needs >= 21 samples)
        short_series = TimeSeries.from_values(np.random.randn(15))

        with pytest.raises(ValueError, match="Training data too short"):
            model.fit(short_series)

    def test_minimum_data_length_validation(self):
        """Test that minimum data length is correctly calculated."""
        model = ModelRegistry.create_model(
            "arima",
            p=5,
            d=1,
            q=3,
            output_chunk_length=5,
        )

        # Minimum length should be p + d + q = 5 + 1 + 3 = 9
        # This should work (10 samples >= 9)
        valid_series = TimeSeries.from_values(np.random.randn(100))
        model.fit(valid_series)
        assert model.is_fitted

    def test_load_nonexistent_file_raises(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        model = ModelRegistry.create_model("arima")

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            model.load(Path("/nonexistent/path/model.pkl"))

    @pytest.mark.slow
    def test_model_summary(self, sample_timeseries):
        """Test model summary generation."""
        model = ModelRegistry.create_model(
            "arima",
            p=5,
            d=1,
            q=5,
            output_chunk_length=5,
        )

        train = sample_timeseries[:300]
        model.fit(train)

        summary = model.get_model_summary()
        assert "ARIMA Model" in summary
        assert "5" in summary  # p and q values
        assert "1" in summary  # d value

    def test_model_summary_before_fit_raises(self):
        """Test that getting summary before fitting raises ValueError."""
        model = ModelRegistry.create_model("arima")

        with pytest.raises(ValueError, match="Model not fitted"):
            model.get_model_summary()

    @pytest.mark.slow
    def test_seasonal_arima(self, sample_timeseries):
        """Test ARIMA with seasonal components."""
        model = ModelRegistry.create_model(
            "arima",
            p=2,
            d=1,
            q=2,
            seasonal=True,
            P=1,
            D=0,
            Q=1,
            m=12,  # Seasonal period
            output_chunk_length=5,
        )

        train = sample_timeseries[:300]
        model.fit(train)
        assert model.is_fitted

        predictions = model.predict(n=5)
        assert len(predictions) == 5

    @pytest.mark.slow
    def test_seasonal_arima_summary(self, sample_timeseries):
        """Test that seasonal ARIMA summary includes seasonal parameters."""
        model = ModelRegistry.create_model(
            "arima",
            p=2,
            d=1,
            q=2,
            seasonal=True,
            P=1,
            D=0,
            Q=1,
            m=24,
            output_chunk_length=5,
        )

        train = sample_timeseries[:300]
        model.fit(train)

        summary = model.get_model_summary()
        assert "Seasonal ARIMA" in summary
        assert "24" in summary  # Seasonal period

    @pytest.mark.slow
    def test_large_parameters_warning(self, caplog):
        """Test warning when p or q is large."""
        model = ModelRegistry.create_model(
            "arima",
            p=15,
            d=1,
            q=15,
            output_chunk_length=5,
        )

        train = TimeSeries.from_values(np.random.randn(100))
        model.fit(train)

        # Should warn about slow training
        assert "Training may be slow" in caplog.text

    @pytest.mark.slow
    def test_different_differencing_orders(self, sample_timeseries):
        """Test ARIMA with different differencing orders."""
        train = sample_timeseries[:300]

        # No differencing
        model1 = ModelRegistry.create_model(
            "arima",
            p=2,
            d=0,
            q=2,
            output_chunk_length=5,
        )
        model1.fit(train)
        pred1 = model1.predict(n=5)
        assert len(pred1) == 5

        # First-order differencing
        model2 = ModelRegistry.create_model(
            "arima",
            p=2,
            d=1,
            q=2,
            output_chunk_length=5,
        )
        model2.fit(train)
        pred2 = model2.predict(n=5)
        assert len(pred2) == 5

        # Second-order differencing
        model3 = ModelRegistry.create_model(
            "arima",
            p=2,
            d=2,
            q=2,
            output_chunk_length=5,
        )
        model3.fit(train)
        pred3 = model3.predict(n=5)
        assert len(pred3) == 5

    @pytest.mark.slow
    def test_uses_all_history(self, sample_timeseries):
        """Test that ARIMA uses all available history, not just input_chunk_length."""
        # input_chunk_length should not affect ARIMA's training
        model = ModelRegistry.create_model(
            "arima",
            input_chunk_length=20,  # This should not limit ARIMA
            p=2,
            d=1,
            q=2,
            output_chunk_length=5,
        )

        # Train on 300 samples (much more than input_chunk_length)
        train = sample_timeseries[:300]
        model.fit(train)

        assert model.is_fitted
        predictions = model.predict(n=5)
        assert len(predictions) == 5

    @pytest.mark.slow
    def test_convergence_warnings_handled(self, sample_timeseries):
        """Test that ARIMA handles convergence warnings gracefully."""
        # Use parameters that might cause convergence issues
        model = ModelRegistry.create_model(
            "arima",
            p=8,
            d=1,
            q=8,
            output_chunk_length=5,
        )

        train = sample_timeseries[:200]

        # Should not raise exceptions even if there are warnings
        model.fit(train)
        assert model.is_fitted

        predictions = model.predict(n=5)
        assert len(predictions) == 5
