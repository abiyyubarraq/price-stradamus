"""Tests for metrics calculator."""

from __future__ import annotations

import numpy as np

from price_stradamus.evaluation.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Test MetricsCalculator class."""

    def test_mae_calculation(self):
        """Test MAE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        mae = MetricsCalculator.mae(y_true, y_pred)

        # MAE should be 0.1
        assert np.isclose(mae, 0.1, atol=0.01)

    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])  # Perfect prediction

        rmse = MetricsCalculator.rmse(y_true, y_pred)

        # RMSE should be 0 for perfect prediction
        assert rmse == 0.0

    def test_directional_accuracy(self):
        """Test directional accuracy calculation."""
        # Sequence going: up, up, down, up
        y_true = np.array([1, 2, 3, 2, 3])

        # Predictions: up, down, up, up (2 out of 4 correct)
        y_pred = np.array([1, 2.5, 2.5, 2.5, 3])

        acc = MetricsCalculator.directional_accuracy(y_true, y_pred)

        # Should be 50% accurate
        assert acc == 50.0

    def test_mape_with_zeros(self):
        """Test MAPE handles zeros in true values."""
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([1, 1, 2, 3])

        mape = MetricsCalculator.mape(y_true, y_pred)

        # Should handle zeros gracefully (skip them)
        assert mape >= 0

    def test_calculate_all(self, sample_predictions):
        """Test calculating all metrics at once."""
        y_true, y_pred = sample_predictions

        metrics = MetricsCalculator.calculate_all(y_true, y_pred)

        # Check all expected metrics are present
        expected_metrics = [
            "mae",
            "mse",
            "rmse",
            "mape",
            "smape",
            "r2",
            "directional_accuracy",
            "max_error",
        ]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)

    def test_format_metrics(self, sample_predictions):
        """Test formatting metrics as DataFrame."""
        y_true, y_pred = sample_predictions

        metrics = MetricsCalculator.calculate_all(y_true, y_pred)
        df = MetricsCalculator.format_metrics(metrics)

        # Check DataFrame structure
        assert "Metric" in df.columns
        assert "Value" in df.columns
        assert len(df) == len(metrics)

    def test_compare_models(self):
        """Test comparing multiple models."""
        y_true = np.array([1, 2, 3, 4, 5])
        predictions = {
            "model_a": np.array([1.1, 2.0, 3.1, 3.9, 5.0]),
            "model_b": np.array([0.9, 2.1, 2.9, 4.1, 5.1]),
        }

        comparison = MetricsCalculator.compare_models(y_true, predictions)

        # Check DataFrame structure
        assert "model" in comparison.columns
        assert len(comparison) == 2

        # Should be sorted by RMSE
        assert comparison["rmse"].is_monotonic_increasing
