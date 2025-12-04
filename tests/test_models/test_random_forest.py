"""Tests for Random Forest model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from darts import TimeSeries

from price_stradamus.models.ml.random_forest import RandomForestModel
from price_stradamus.models.registry import ModelRegistry


class TestRandomForestModel:
    """Test RandomForestModel implementation."""

    def test_model_registered(self):
        """Test that Random Forest model is registered in the registry."""
        assert ModelRegistry.is_registered("random_forest")
        assert "random_forest" in ModelRegistry.list_models()

    def test_initialization(self):
        """Test Random Forest model initialization with default parameters."""
        model = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=60,
            output_chunk_length=5,
        )

        assert model.name == "random_forest"
        assert model.input_chunk_length == 60
        assert model.output_chunk_length == 5
        assert not model.is_fitted

    def test_initialization_custom_params(self):
        """Test Random Forest initialization with custom hyperparameters."""
        model = RandomForestModel(
            name="rf_custom",
            input_chunk_length=120,
            output_chunk_length=10,
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            n_jobs=2,
        )

        assert model.name == "rf_custom"
        assert model.input_chunk_length == 120
        assert model.output_chunk_length == 10
        assert model.n_estimators == 200
        assert model.max_depth == 12
        assert model.min_samples_split == 5
        assert model.n_jobs == 2

    def test_hyperparameters_stored(self):
        """Test that hyperparameters are correctly stored."""
        model = ModelRegistry.create_model(
            "random_forest",
            n_estimators=150,
            max_depth=8,
            max_features="log2",
        )

        params = model.get_params()
        assert params["n_estimators"] == 150
        assert params["max_depth"] == 8
        assert params["max_features"] == "log2"

    @pytest.mark.slow
    def test_fit_predict(self, sample_timeseries):
        """Test training and prediction with Random Forest."""
        model = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=20,
            output_chunk_length=5,
            n_estimators=50,  # Fewer for faster testing
            n_jobs=1,
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
        """Test training with validation data (not used by Random Forest)."""
        model = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=20,
            output_chunk_length=5,
            n_estimators=50,
            n_jobs=1,
        )

        train = sample_timeseries[:400]
        val = sample_timeseries[400:500]

        # Random Forest doesn't use validation data but should accept it
        model.fit(train, val)
        assert model.is_fitted

    @pytest.mark.slow
    def test_predict_with_series(self, sample_timeseries):
        """Test prediction with custom series."""
        model = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=20,
            output_chunk_length=5,
            n_estimators=50,
            n_jobs=1,
        )

        train = sample_timeseries[:500]
        test_series = sample_timeseries[500:600]

        model.fit(train)

        # Predict from different series
        predictions = model.predict(n=5, series=test_series)
        assert len(predictions) == 5

    def test_save_load(self, tmp_path, sample_timeseries):
        """Test model saving and loading."""
        model = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=20,
            output_chunk_length=5,
            n_estimators=50,
            n_jobs=1,
        )

        train = sample_timeseries[:500]
        model.fit(train)

        # Save
        save_path = tmp_path / "random_forest_test.pkl"
        model.save(save_path)
        assert save_path.exists()

        # Load
        new_model = ModelRegistry.create_model("random_forest")
        new_model.load(save_path)
        assert new_model.is_fitted

        # Predict with loaded model
        predictions = new_model.predict(n=5)
        assert len(predictions) == 5

    def test_predict_before_fit_raises(self):
        """Test that predicting before fitting raises ValueError."""
        model = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=20,
            output_chunk_length=5,
        )

        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(n=5)

    def test_save_before_fit_raises(self, tmp_path):
        """Test that saving before fitting raises ValueError."""
        model = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=20,
            output_chunk_length=5,
        )

        save_path = tmp_path / "model.pkl"
        with pytest.raises(ValueError, match="Cannot save unfitted model"):
            model.save(save_path)

    def test_invalid_training_data_length(self):
        """Test error with insufficient training data."""
        model = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=100,
            output_chunk_length=5,
        )

        # Create series that's too short
        short_series = TimeSeries.from_values(np.random.randn(50))

        with pytest.raises(ValueError, match="Training data too short"):
            model.fit(short_series)

    def test_load_nonexistent_file_raises(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        model = ModelRegistry.create_model("random_forest")

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            model.load(Path("/nonexistent/path/model.pkl"))

    @pytest.mark.slow
    def test_model_summary(self, sample_timeseries):
        """Test model summary generation."""
        model = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=20,
            output_chunk_length=5,
            n_estimators=100,
            n_jobs=1,
        )

        train = sample_timeseries[:500]
        model.fit(train)

        summary = model.get_model_summary()
        assert "Random Forest Model" in summary
        assert "100" in summary  # n_estimators
        assert "20" in summary  # input_chunk_length

    def test_model_summary_before_fit_raises(self):
        """Test that getting summary before fitting raises ValueError."""
        model = ModelRegistry.create_model("random_forest")

        with pytest.raises(ValueError, match="Model not fitted"):
            model.get_model_summary()

    @pytest.mark.slow
    def test_parallel_training(self, sample_timeseries):
        """Test that parallel training works with n_jobs."""
        model = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=20,
            output_chunk_length=5,
            n_estimators=50,
            n_jobs=2,  # Use 2 parallel jobs
        )

        train = sample_timeseries[:500]
        model.fit(train)

        assert model.is_fitted
        predictions = model.predict(n=5)
        assert len(predictions) == 5

    @pytest.mark.slow
    def test_prediction_output_mismatch_warning(self, sample_timeseries, caplog):
        """Test warning when requested predictions != output_chunk_length."""
        model = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=20,
            output_chunk_length=5,
            n_estimators=50,
            n_jobs=1,
        )

        train = sample_timeseries[:500]
        model.fit(train)

        # Request different number of steps (should warn and use 5)
        predictions = model.predict(n=10)

        assert len(predictions) == 5  # Should use output_chunk_length
        assert "Requested 10 steps but model outputs 5" in caplog.text

    @pytest.mark.slow
    def test_deterministic_with_random_state(self, sample_timeseries):
        """Test that predictions are deterministic with same random_state."""
        model1 = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=20,
            output_chunk_length=5,
            n_estimators=50,
            random_state=42,
            n_jobs=1,  # Use single job for determinism
        )

        model2 = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=20,
            output_chunk_length=5,
            n_estimators=50,
            random_state=42,
            n_jobs=1,
        )

        train = sample_timeseries[:500]

        # Train both models
        model1.fit(train)
        model2.fit(train)

        # Get predictions
        pred1 = model1.predict(n=5)
        pred2 = model2.predict(n=5)

        # Should be identical with same random state
        np.testing.assert_array_equal(pred1.values(), pred2.values())

    @pytest.mark.slow
    def test_lags_parameter_used(self, sample_timeseries):
        """Test that Random Forest uses lags parameter correctly."""
        model = ModelRegistry.create_model(
            "random_forest",
            input_chunk_length=30,  # This should map to lags
            output_chunk_length=5,
            n_estimators=50,
            n_jobs=1,
        )

        train = sample_timeseries[:500]
        model.fit(train)

        # Model should be fitted successfully with lags=30
        assert model.is_fitted
        predictions = model.predict(n=5)
        assert len(predictions) == 5

    @pytest.mark.slow
    def test_different_max_features(self, sample_timeseries):
        """Test Random Forest with different max_features settings."""
        for max_features in ["sqrt", "log2"]:
            model = ModelRegistry.create_model(
                "random_forest",
                input_chunk_length=20,
                output_chunk_length=5,
                n_estimators=50,
                max_features=max_features,
                n_jobs=1,
            )

            train = sample_timeseries[:500]
            model.fit(train)

            assert model.is_fitted
            predictions = model.predict(n=5)
            assert len(predictions) == 5
