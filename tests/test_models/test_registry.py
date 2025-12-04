"""Tests for model registry."""

from __future__ import annotations

import pytest

from price_stradamus.models.base import BaseModel
from price_stradamus.models.registry import ModelRegistry


class TestModelRegistry:
    """Test ModelRegistry class."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear registry before each test."""
        ModelRegistry.clear_registry()
        yield
        ModelRegistry.clear_registry()

    def test_register_model(self):
        """Test registering a model."""

        @ModelRegistry.register("test_model")
        class TestModel(BaseModel):
            def fit(self, train_data, val_data=None, **kwargs):
                pass

            def predict(self, n, series=None):
                pass

            def save(self, path):
                pass

            def load(self, path):
                return self

        # Check model is registered
        assert ModelRegistry.is_registered("test_model")
        assert "test_model" in ModelRegistry.list_models()

    def test_get_model_class(self):
        """Test getting model class by name."""

        @ModelRegistry.register("test_model")
        class TestModel(BaseModel):
            def fit(self, train_data, val_data=None, **kwargs):
                pass

            def predict(self, n, series=None):
                pass

            def save(self, path):
                pass

            def load(self, path):
                return self

        model_class = ModelRegistry.get_model_class("test_model")
        assert model_class == TestModel

    def test_get_nonexistent_model(self):
        """Test getting non-existent model raises error."""
        with pytest.raises(ValueError, match="not found"):
            ModelRegistry.get_model_class("nonexistent")

    def test_create_model(self):
        """Test creating model instance."""

        @ModelRegistry.register("test_model")
        class TestModel(BaseModel):
            def fit(self, train_data, val_data=None, **kwargs):
                pass

            def predict(self, n, series=None):
                pass

            def save(self, path):
                pass

            def load(self, path):
                return self

        model = ModelRegistry.create_model(
            "test_model",
            input_chunk_length=60,
            output_chunk_length=5,
        )

        assert isinstance(model, TestModel)
        assert model.name == "test_model"
        assert model.input_chunk_length == 60
        assert model.output_chunk_length == 5

    def test_list_models(self):
        """Test listing all models."""

        @ModelRegistry.register("model_a")
        class ModelA(BaseModel):
            def fit(self, train_data, val_data=None, **kwargs):
                pass

            def predict(self, n, series=None):
                pass

            def save(self, path):
                pass

            def load(self, path):
                return self

        @ModelRegistry.register("model_b")
        class ModelB(BaseModel):
            def fit(self, train_data, val_data=None, **kwargs):
                pass

            def predict(self, n, series=None):
                pass

            def save(self, path):
                pass

            def load(self, path):
                return self

        models = ModelRegistry.list_models()

        assert len(models) == 2
        assert "model_a" in models
        assert "model_b" in models

    def test_get_model_info(self):
        """Test getting model information."""

        @ModelRegistry.register("test_model")
        class TestModel(BaseModel):
            """Test model docstring."""

            def fit(self, train_data, val_data=None, **kwargs):
                pass

            def predict(self, n, series=None):
                pass

            def save(self, path):
                pass

            def load(self, path):
                return self

        info = ModelRegistry.get_model_info("test_model")

        assert info["name"] == "test_model"
        assert info["class"] == "TestModel"
        assert "Test model docstring" in info["docstring"]

    def test_is_registered(self):
        """Test checking if model is registered."""

        @ModelRegistry.register("test_model")
        class TestModel(BaseModel):
            def fit(self, train_data, val_data=None, **kwargs):
                pass

            def predict(self, n, series=None):
                pass

            def save(self, path):
                pass

            def load(self, path):
                return self

        assert ModelRegistry.is_registered("test_model")
        assert not ModelRegistry.is_registered("nonexistent")
