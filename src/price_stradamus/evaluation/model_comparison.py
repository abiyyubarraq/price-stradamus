"""Utility for comparing multiple models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from darts import TimeSeries
from loguru import logger

from price_stradamus.evaluation.backtester import Backtester
from price_stradamus.models.registry import ModelRegistry


def compare_models(
    model_names: list[str],
    data: TimeSeries,
    test_size: float = 0.2,
    val_size: float = 0.1,
    model_kwargs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Compare multiple models on same dataset.

    Trains and evaluates multiple models using walk-forward backtesting,
    providing a comprehensive comparison of their performance.

    Args:
        model_names: List of registered model names to compare
        data: Time series data for training and evaluation
        test_size: Proportion of data for testing (default: 0.2)
        val_size: Proportion of data for validation (default: 0.1)
        model_kwargs: Model-specific hyperparameters. Dictionary mapping
            model names to their kwargs. Example:
            {
                "nbeats": {"n_epochs": 50, "hidden_size": 64},
                "xgboost": {"n_estimators": 200}
            }

    Returns:
        DataFrame with model comparison metrics, sorted by MAE (ascending).
        Columns include: model, mae, rmse, mape, directional_accuracy, etc.

    Raises:
        ValueError: If model_names is empty or contains unregistered models

    Example:
        >>> from darts import TimeSeries
        >>> import numpy as np
        >>> data = TimeSeries.from_values(np.random.randn(1000))
        >>> results = compare_models(
        ...     model_names=["nbeats", "lstm", "xgboost"],
        ...     data=data,
        ...     test_size=0.2,
        ...     model_kwargs={
        ...         "nbeats": {"n_epochs": 10},
        ...         "lstm": {"n_epochs": 10},
        ...         "xgboost": {"n_estimators": 50}
        ...     }
        ... )
        >>> print(results)
           model       mae      rmse      mape  directional_accuracy
        0  nbeats  1.234     1.456     5.678                 0.567
        1  lstm    1.345     1.567     6.789                 0.556
        2  xgboost 1.456     1.678     7.890                 0.545
    """
    if not model_names:
        raise ValueError("model_names cannot be empty")

    # Verify all models are registered
    available_models = ModelRegistry.list_models()
    for model_name in model_names:
        if model_name not in available_models:
            raise ValueError(
                f"Model '{model_name}' not registered. "
                f"Available models: {available_models}"
            )

    if model_kwargs is None:
        model_kwargs = {}

    results = []

    logger.info(f"Comparing {len(model_names)} models: {', '.join(model_names)}")

    for model_name in model_names:
        logger.info(f"Evaluating {model_name}...")

        try:
            # Create model with specific kwargs
            kwargs = model_kwargs.get(model_name, {})
            model = ModelRegistry.create_model(model_name, **kwargs)

            # Run backtest
            backtester = Backtester(
                model=model,
                test_size=test_size,
                val_size=val_size,
            )
            result = backtester.run_expanding_window(data)

            # Store results
            results.append(
                {
                    "model": model_name,
                    **result["metrics"],
                    "train_size": result["train_size"],
                    "val_size": result["val_size"],
                    "test_size": result["test_size"],
                }
            )

            logger.info(
                f"{model_name} complete - MAE: {result['metrics']['mae']:.4f}, "
                f"Dir Acc: {result['metrics']['directional_accuracy']:.2%}"
            )

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            # Add error entry
            results.append(
                {
                    "model": model_name,
                    "mae": float("inf"),
                    "rmse": float("inf"),
                    "mse": float("inf"),
                    "mape": float("inf"),
                    "smape": float("inf"),
                    "max_error": float("inf"),
                    "r2": float("-inf"),
                    "directional_accuracy": 0.0,
                    "train_size": 0,
                    "val_size": 0,
                    "test_size": 0,
                    "error": str(e),
                }
            )

    # Create DataFrame and sort by MAE
    df = pd.DataFrame(results)
    df = df.sort_values("mae", ascending=True).reset_index(drop=True)

    logger.info("Model comparison complete")

    return df


def compare_models_detailed(
    model_names: list[str],
    data: TimeSeries,
    test_size: float = 0.2,
    val_size: float = 0.1,
    model_kwargs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare models with detailed results including predictions.

    Similar to compare_models but returns full backtest results including
    predictions and timestamps for visualization.

    Args:
        model_names: List of registered model names to compare
        data: Time series data for training and evaluation
        test_size: Proportion of data for testing (default: 0.2)
        val_size: Proportion of data for validation (default: 0.1)
        model_kwargs: Model-specific hyperparameters

    Returns:
        Dictionary mapping model names to their full backtest results:
        {
            "nbeats": {
                "predictions": np.ndarray,
                "actuals": np.ndarray,
                "timestamps": np.ndarray,
                "metrics": dict,
                ...
            },
            ...
        }

    Raises:
        ValueError: If model_names is empty or contains unregistered models

    Example:
        >>> results = compare_models_detailed(
        ...     model_names=["nbeats", "xgboost"],
        ...     data=data,
        ...     test_size=0.2,
        ... )
        >>> # Plot predictions
        >>> import matplotlib.pyplot as plt
        >>> for model_name, result in results.items():
        ...     plt.plot(result["actuals"], label=f"{model_name} actual")
        ...     plt.plot(result["predictions"], label=f"{model_name} pred")
        >>> plt.legend()
        >>> plt.show()
    """
    if not model_names:
        raise ValueError("model_names cannot be empty")

    # Verify all models are registered
    available_models = ModelRegistry.list_models()
    for model_name in model_names:
        if model_name not in available_models:
            raise ValueError(
                f"Model '{model_name}' not registered. "
                f"Available models: {available_models}"
            )

    if model_kwargs is None:
        model_kwargs = {}

    results = {}

    logger.info(f"Comparing {len(model_names)} models with detailed results")

    for model_name in model_names:
        logger.info(f"Evaluating {model_name}...")

        try:
            # Create model with specific kwargs
            kwargs = model_kwargs.get(model_name, {})
            model = ModelRegistry.create_model(model_name, **kwargs)

            # Run backtest
            backtester = Backtester(
                model=model,
                test_size=test_size,
                val_size=val_size,
            )
            result = backtester.run_expanding_window(data)

            # Store full results
            results[model_name] = result

            logger.info(f"{model_name} complete - MAE: {result['metrics']['mae']:.4f}")

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    logger.info("Detailed model comparison complete")

    return results
