"""Tests for base model interface."""

from __future__ import annotations

import pytest

from price_stradamus.models.base import BaseModel


class TestBaseModel:
    """Test BaseModel abstract class."""

    def test_cannot_instantiate_base_model(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel(name="test", input_chunk_length=60, output_chunk_length=5)

    def test_concrete_model_must_implement_methods(self):
        """Test that concrete models must implement abstract methods."""

        # Create incomplete concrete model
        class IncompleteModel(BaseModel):
            def fit(self, train_data, val_data=None, **kwargs):
                pass

            # Missing: predict, save, load

        with pytest.raises(TypeError):
            IncompleteModel(
                name="incomplete", input_chunk_length=60, output_chunk_length=5
            )

    def test_get_params(self):
        """Test getting model parameters."""

        class DummyModel(BaseModel):
            def fit(self, train_data, val_data=None, **kwargs):
                pass

            def predict(self, n, series=None):
                pass

            def save(self, path):
                pass

            def load(self, path):
                return self

        model = DummyModel(
            name="dummy",
            input_chunk_length=60,
            output_chunk_length=5,
            learning_rate=0.001,
        )

        params = model.get_params()

        assert params["name"] == "dummy"
        assert params["input_chunk_length"] == 60
        assert params["output_chunk_length"] == 5
        assert params["learning_rate"] == 0.001

    def test_set_params(self):
        """Test setting model parameters."""

        class DummyModel(BaseModel):
            def fit(self, train_data, val_data=None, **kwargs):
                pass

            def predict(self, n, series=None):
                pass

            def save(self, path):
                pass

            def load(self, path):
                return self

        model = DummyModel(name="dummy", input_chunk_length=60, output_chunk_length=5)

        model.set_params(learning_rate=0.002, new_param="test")

        params = model.get_params()
        assert params["learning_rate"] == 0.002
        assert params["new_param"] == "test"

    def test_is_fitted_property(self):
        """Test is_fitted property."""

        class DummyModel(BaseModel):
            def fit(self, train_data, val_data=None, **kwargs):
                self._is_fitted = True

            def predict(self, n, series=None):
                pass

            def save(self, path):
                pass

            def load(self, path):
                return self

        model = DummyModel(name="dummy", input_chunk_length=60, output_chunk_length=5)

        assert not model.is_fitted

        model.fit(None)

        assert model.is_fitted

    def test_validate_fit_params(self, sample_timeseries):
        """Test validation of training data."""

        class DummyModel(BaseModel):
            def fit(self, train_data, val_data=None, **kwargs):
                pass

            def predict(self, n, series=None):
                pass

            def save(self, path):
                pass

            def load(self, path):
                return self

        model = DummyModel(name="dummy", input_chunk_length=60, output_chunk_length=5)

        # Should pass with sufficient data
        long_series = sample_timeseries[:100]
        model.validate_fit_params(long_series)

        # Should fail with insufficient data
        short_series = sample_timeseries[:50]
        with pytest.raises(ValueError, match="Training data too short"):
            model.validate_fit_params(short_series)

    def test_validate_predict_params(self):
        """Test validation of prediction parameters."""

        class DummyModel(BaseModel):
            def fit(self, train_data, val_data=None, **kwargs):
                self._is_fitted = True

            def predict(self, n, series=None):
                pass

            def save(self, path):
                pass

            def load(self, path):
                return self

        model = DummyModel(name="dummy", input_chunk_length=60, output_chunk_length=5)

        # Should fail if not fitted
        with pytest.raises(ValueError, match="must be fitted"):
            model.validate_predict_params(5)

        # Fit model
        model.fit(None)

        # Should pass
        model.validate_predict_params(5)

        # Should fail for invalid n
        with pytest.raises(ValueError, match="n must be >= 1"):
            model.validate_predict_params(0)
