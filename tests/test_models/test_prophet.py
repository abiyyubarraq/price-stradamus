"""Tests for Prophet model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from darts import TimeSeries

from price_stradamus.models.classical.prophet import ProphetModel
from price_stradamus.models.registry import ModelRegistry


class TestProphetModel:
    """Test ProphetModel implementation."""

    def test_model_registered(self):
        """Test that Prophet model is registered in the registry."""
        assert ModelRegistry.is_registered("prophet")
        assert "prophet" in ModelRegistry.list_models()

    def test_initialization(self):
        """Test Prophet model initialization with default parameters."""
        model = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
        )

        assert model.name == "prophet"
        assert model.output_chunk_length == 5
        assert not model.is_fitted

    def test_initialization_custom_params(self):
        """Test Prophet initialization with custom hyperparameters."""
        model = ProphetModel(
            name="prophet_custom",
            output_chunk_length=10,
            growth="logistic",
            seasonality_mode="additive",
            yearly_seasonality=True,
        )

        assert model.name == "prophet_custom"
        assert model.output_chunk_length == 10
        assert model.growth == "logistic"
        assert model.seasonality_mode == "additive"
        assert model.yearly_seasonality is True

    def test_hyperparameters_stored(self):
        """Test that hyperparameters are correctly stored."""
        model = ModelRegistry.create_model(
            "prophet",
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=15.0,
            daily_seasonality=False,
        )

        params = model.get_params()
        assert params["changepoint_prior_scale"] == 0.1
        assert params["seasonality_prior_scale"] == 15.0
        assert params["daily_seasonality"] is False

    @pytest.mark.slow
    def test_fit_predict(self, sample_timeseries):
        """Test training and prediction with Prophet."""
        model = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
            daily_seasonality=False,  # Disable for faster testing
            weekly_seasonality=False,
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
        """Test training with validation data (not used by Prophet)."""
        model = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
            daily_seasonality=False,
            weekly_seasonality=False,
        )

        train = sample_timeseries[:250]
        val = sample_timeseries[250:300]

        # Prophet doesn't use validation data but should accept it
        model.fit(train, val)
        assert model.is_fitted

    @pytest.mark.slow
    def test_predict_with_series(self, sample_timeseries):
        """Test prediction with custom series."""
        model = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
            daily_seasonality=False,
            weekly_seasonality=False,
        )

        train = sample_timeseries[:300]
        test_series = sample_timeseries[300:400]

        model.fit(train)

        # Predict from different series
        predictions = model.predict(n=5, series=test_series)
        assert len(predictions) == 5

    @pytest.mark.slow
    def test_variable_prediction_horizon(self, sample_timeseries):
        """Test Prophet with variable prediction horizons."""
        model = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
            daily_seasonality=False,
            weekly_seasonality=False,
        )

        train = sample_timeseries[:300]
        model.fit(train)

        # Prophet supports variable horizon
        for n in [3, 5, 10]:
            predictions = model.predict(n=n)
            assert len(predictions) == n

    def test_save_load(self, tmp_path, sample_timeseries):
        """Test model saving and loading."""
        model = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
            daily_seasonality=False,
            weekly_seasonality=False,
        )

        train = sample_timeseries[:300]
        model.fit(train)

        # Save
        save_path = tmp_path / "prophet_test.pkl"
        model.save(save_path)
        assert save_path.exists()

        # Load
        new_model = ModelRegistry.create_model("prophet")
        new_model.load(save_path)
        assert new_model.is_fitted

        # Predict with loaded model
        predictions = new_model.predict(n=5)
        assert len(predictions) == 5

    def test_predict_before_fit_raises(self):
        """Test that predicting before fitting raises ValueError."""
        model = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
        )

        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(n=5)

    def test_save_before_fit_raises(self, tmp_path):
        """Test that saving before fitting raises ValueError."""
        model = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
        )

        save_path = tmp_path / "model.pkl"
        with pytest.raises(ValueError, match="Cannot save unfitted model"):
            model.save(save_path)

    def test_invalid_training_data_length(self):
        """Test error with insufficient training data."""
        model = ModelRegistry.create_model(
            "prophet",
            input_chunk_length=100,
            output_chunk_length=5,
        )

        # Create series that's too short
        short_series = TimeSeries.from_values(np.random.randn(50))

        with pytest.raises(ValueError, match="Training data too short"):
            model.fit(short_series)

    def test_load_nonexistent_file_raises(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        model = ModelRegistry.create_model("prophet")

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            model.load(Path("/nonexistent/path/model.pkl"))

    @pytest.mark.slow
    def test_model_summary(self, sample_timeseries):
        """Test model summary generation."""
        model = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
            growth="linear",
            seasonality_mode="multiplicative",
            daily_seasonality=False,
            weekly_seasonality=False,
        )

        train = sample_timeseries[:300]
        model.fit(train)

        summary = model.get_model_summary()
        assert "Prophet Model" in summary
        assert "linear" in summary  # growth
        assert "multiplicative" in summary  # seasonality_mode

    def test_model_summary_before_fit_raises(self):
        """Test that getting summary before fitting raises ValueError."""
        model = ModelRegistry.create_model("prophet")

        with pytest.raises(ValueError, match="Model not fitted"):
            model.get_model_summary()

    @pytest.mark.slow
    def test_seasonality_settings(self, sample_timeseries):
        """Test Prophet with different seasonality settings."""
        # Test with daily seasonality
        model = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False,
        )

        train = sample_timeseries[:300]
        model.fit(train)
        assert model.is_fitted

        # Test multiplicative seasonality
        model2 = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
            seasonality_mode="multiplicative",
            daily_seasonality=False,
            weekly_seasonality=False,
        )
        model2.fit(train)
        assert model2.is_fitted

    @pytest.mark.slow
    def test_large_dataset_warning(self, caplog):
        """Test warning when training on large datasets."""
        model = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
            daily_seasonality=False,
            weekly_seasonality=False,
        )

        # Create large dataset
        large_series = TimeSeries.from_values(np.random.randn(11000))

        model.fit(large_series)

        # Should warn about slow training
        assert "Prophet may be slow" in caplog.text

    @pytest.mark.slow
    def test_growth_modes(self, sample_timeseries):
        """Test Prophet with different growth modes."""
        train = sample_timeseries[:300]

        # Linear growth
        model1 = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
            growth="linear",
            daily_seasonality=False,
            weekly_seasonality=False,
        )
        model1.fit(train)
        pred1 = model1.predict(n=5)
        assert len(pred1) == 5

        # Logistic growth
        model2 = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
            growth="logistic",
            daily_seasonality=False,
            weekly_seasonality=False,
        )
        model2.fit(train)
        pred2 = model2.predict(n=5)
        assert len(pred2) == 5

    @pytest.mark.slow
    def test_changepoint_flexibility(self, sample_timeseries):
        """Test Prophet with different changepoint flexibility."""
        train = sample_timeseries[:300]

        # Low flexibility (more stable trend)
        model1 = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
            changepoint_prior_scale=0.01,
            daily_seasonality=False,
            weekly_seasonality=False,
        )
        model1.fit(train)
        assert model1.is_fitted

        # High flexibility (more responsive to changes)
        model2 = ModelRegistry.create_model(
            "prophet",
            output_chunk_length=5,
            changepoint_prior_scale=0.5,
            daily_seasonality=False,
            weekly_seasonality=False,
        )
        model2.fit(train)
        assert model2.is_fitted

    @pytest.mark.slow
    def test_uses_all_history(self, sample_timeseries):
        """Test that Prophet uses all available history, not just input_chunk_length."""
        # input_chunk_length should not affect Prophet's training
        model = ModelRegistry.create_model(
            "prophet",
            input_chunk_length=20,  # This should not limit Prophet
            output_chunk_length=5,
            daily_seasonality=False,
            weekly_seasonality=False,
        )

        # Train on 300 samples (much more than input_chunk_length)
        train = sample_timeseries[:300]
        model.fit(train)

        assert model.is_fitted
        predictions = model.predict(n=5)
        assert len(predictions) == 5
