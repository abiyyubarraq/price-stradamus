"""Model registry for centralized model management.

This module provides a registry pattern for registering and retrieving
prediction models by name.
"""

from __future__ import annotations

from loguru import logger

from price_stradamus.models.base import BaseModel


class ModelRegistry:
    """Registry for all available prediction models.

    Provides decorator-based registration and factory methods for
    creating model instances by name.

    Example:
        # Register a model
        @ModelRegistry.register("my_model")
        class MyModel(BaseModel):
            ...

        # Create model instance
        model = ModelRegistry.create_model("my_model", input_chunk_length=60, output_chunk_length=5)

        # List all models
        available = ModelRegistry.list_models()
    """

    _models: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class.

        Args:
            name: Model name/identifier (must be unique)

        Returns:
            Decorator function

        Raises:
            ValueError: If model name already registered

        Example:
            @ModelRegistry.register("nbeats")
            class NBEATSModel(BaseModel):
                ...
        """

        def decorator(model_class: type[BaseModel]) -> type[BaseModel]:
            if name in cls._models:
                logger.warning(
                    f"Model '{name}' already registered, overwriting with {model_class.__name__}"
                )

            cls._models[name] = model_class
            logger.info(f"Registered model '{name}' -> {model_class.__name__}")
            return model_class

        return decorator

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a model.

        Args:
            name: Model name to unregister

        Raises:
            KeyError: If model not found
        """
        if name not in cls._models:
            raise KeyError(f"Model '{name}' not registered")

        model_class = cls._models.pop(name)
        logger.info(f"Unregistered model '{name}' ({model_class.__name__})")

    @classmethod
    def get_model_class(cls, name: str) -> type[BaseModel]:
        """Get model class by name.

        Args:
            name: Model name

        Returns:
            Model class (not instance)

        Raises:
            ValueError: If model not found
        """
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available}")

        return cls._models[name]

    @classmethod
    def create_model(cls, name: str, **kwargs) -> BaseModel:
        """Create model instance by name (factory method).

        Args:
            name: Model name
            **kwargs: Parameters to pass to model constructor

        Returns:
            Instantiated model

        Raises:
            ValueError: If model not found
            TypeError: If invalid parameters provided

        Example:
            model = ModelRegistry.create_model(
                "nbeats",
                input_chunk_length=60,
                output_chunk_length=5,
                num_stacks=30,
            )
        """
        model_class = cls.get_model_class(name)

        # Set name if not provided
        if "name" not in kwargs:
            kwargs["name"] = name

        try:
            model = model_class(**kwargs)
            logger.info(f"Created model instance: {name}")
            return model
        except TypeError as e:
            raise TypeError(
                f"Failed to create model '{name}': {e}. "
                f"Check that all required parameters are provided."
            ) from e

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model names.

        Returns:
            List of model names (sorted alphabetically)
        """
        return sorted(cls._models.keys())

    @classmethod
    def get_model_info(cls, name: str) -> dict[str, str]:
        """Get detailed information about a model.

        Args:
            name: Model name

        Returns:
            Dictionary with model information:
            - name: Model name
            - class: Class name
            - module: Module path
            - docstring: Class docstring

        Raises:
            ValueError: If model not found
        """
        model_class = cls.get_model_class(name)

        return {
            "name": name,
            "class": model_class.__name__,
            "module": model_class.__module__,
            "docstring": model_class.__doc__ or "No description available",
        }

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered.

        Args:
            name: Model name

        Returns:
            True if model is registered, False otherwise
        """
        return name in cls._models

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered models.

        Warning:
            This is mainly for testing purposes. Use with caution.
        """
        cls._models.clear()
        logger.warning("Cleared all registered models")

    @classmethod
    def get_registry_summary(cls) -> str:
        """Get human-readable summary of registered models.

        Returns:
            Formatted string with all registered models
        """
        if not cls._models:
            return "No models registered"

        lines = ["Registered Models:", "=" * 50]

        for name in sorted(cls._models.keys()):
            model_class = cls._models[name]
            lines.append(f"  {name:20s} -> {model_class.__name__}")

        lines.append("=" * 50)
        lines.append(f"Total: {len(cls._models)} models")

        return "\n".join(lines)

    def __repr__(cls) -> str:
        """String representation of registry.

        Returns:
            Summary string
        """
        return cls.get_registry_summary()
