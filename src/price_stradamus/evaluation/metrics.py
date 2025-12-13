"""Performance metrics for time series prediction evaluation.

This module provides comprehensive metrics for evaluating time series
prediction models, including error metrics and directional accuracy.

For complete evaluation, combine with financial_metrics.py to get:
- Error metrics (MAE, RMSE, etc.) - from this module
- Financial metrics (P&L, Sharpe, etc.) - from financial_metrics

Example:
    # Get all metrics including financial
    metrics = MetricsCalculator.calculate_all_with_financial(
        y_true=actual_prices,
        y_pred=predicted_prices,
        include_financial=True,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if TYPE_CHECKING:
    from price_stradamus.evaluation.financial_metrics import TradingCosts


class MetricsCalculator:
    """Calculate time series prediction metrics.

    Provides various metrics for evaluating prediction quality:
    - Error metrics: MAE, MSE, RMSE, MAPE, sMAPE
    - Statistical metrics: R², Max Error
    - Trading metrics: Directional Accuracy

    All methods are static and can be called without instantiation.

    Example:
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        mae = MetricsCalculator.mae(y_true, y_pred)
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        dir_acc = MetricsCalculator.directional_accuracy(y_true, y_pred)

        # Or calculate all metrics at once
        all_metrics = MetricsCalculator.calculate_all(y_true, y_pred)
    """

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error.

        MAE measures the average magnitude of errors in predictions,
        giving equal weight to all errors.

        Formula: MAE = mean(|y_true - y_pred|)

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Mean absolute error (lower is better)
        """
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error.

        MSE gives more weight to large errors (squared).

        Formula: MSE = mean((y_true - y_pred)²)

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Mean squared error (lower is better)
        """
        return float(mean_squared_error(y_true, y_pred))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error.

        RMSE is in the same units as the target variable.
        More sensitive to large errors than MAE.

        Formula: RMSE = sqrt(mean((y_true - y_pred)²))

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Root mean squared error (lower is better)
        """
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error.

        MAPE expresses error as a percentage of the true values.
        Undefined when true values are zero.

        Formula: MAPE = mean(|y_true - y_pred| / |y_true|) * 100

        Args:
            y_true: True values (must be non-zero)
            y_pred: Predicted values

        Returns:
            Mean absolute percentage error in % (lower is better)
        """
        # Avoid division by zero
        mask = y_true != 0

        if not mask.any():
            logger.warning("All true values are zero, MAPE undefined. Returning inf.")
            return float("inf")

        mape_value = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return float(mape_value)

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error.

        sMAPE is a symmetric version of MAPE that treats over-predictions
        and under-predictions equally. Bounded between 0% and 100%.

        Formula: sMAPE = mean(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2)) * 100

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Symmetric mean absolute percentage error in % (lower is better)
        """
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

        # Avoid division by zero
        mask = denominator != 0

        if not mask.any():
            logger.warning("All values are zero, sMAPE undefined. Returning 0.")
            return 0.0

        smape_value = np.mean(numerator[mask] / denominator[mask]) * 100
        return float(smape_value)

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² (coefficient of determination) score.

        R² represents the proportion of variance in the dependent variable
        that is predictable from the independent variable.

        - R² = 1: Perfect prediction
        - R² = 0: Model performs as well as mean baseline
        - R² < 0: Model performs worse than mean baseline

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            R² score (higher is better, max 1.0)
        """
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate percentage of correct direction predictions.

        For trading/finance applications, correctly predicting whether
        the price will go up or down is often more important than
        the exact magnitude of the change.

        Compares direction of change between consecutive timesteps:
        - If both true and predicted increase: correct
        - If both true and predicted decrease: correct
        - Otherwise: incorrect

        Args:
            y_true: True values (must have at least 2 values)
            y_pred: Predicted values (must have at least 2 values)

        Returns:
            Directional accuracy in % (0-100, higher is better)
        """
        if len(y_true) < 2 or len(y_pred) < 2:
            logger.warning("Need at least 2 values to calculate directional accuracy")
            return 0.0

        # Calculate direction of change (positive = up, negative = down, zero = flat)
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))

        # Calculate accuracy (percentage of matching directions)
        accuracy = np.mean(true_direction == pred_direction) * 100

        return float(accuracy)

    @staticmethod
    def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate maximum absolute error.

        Identifies the worst prediction in the set.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Maximum absolute error (lower is better)
        """
        return float(np.max(np.abs(y_true - y_pred)))

    @staticmethod
    def calculate_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Calculate all metrics at once.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with all metrics:
            - mae: Mean Absolute Error
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - mape: Mean Absolute Percentage Error (%)
            - smape: Symmetric MAPE (%)
            - r2: R² score
            - directional_accuracy: Directional accuracy (%)
            - max_error: Maximum absolute error
        """
        logger.debug(f"Calculating all metrics for {len(y_true)} predictions")

        metrics = {
            "mae": MetricsCalculator.mae(y_true, y_pred),
            "mse": MetricsCalculator.mse(y_true, y_pred),
            "rmse": MetricsCalculator.rmse(y_true, y_pred),
            "mape": MetricsCalculator.mape(y_true, y_pred),
            "smape": MetricsCalculator.smape(y_true, y_pred),
            "r2": MetricsCalculator.r2(y_true, y_pred),
            "directional_accuracy": MetricsCalculator.directional_accuracy(
                y_true, y_pred
            ),
            "max_error": MetricsCalculator.max_error(y_true, y_pred),
        }

        logger.debug(
            f"Calculated metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
            f"Directional Accuracy={metrics['directional_accuracy']:.2f}%"
        )

        return metrics

    @staticmethod
    def calculate_all_with_financial(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        include_financial: bool = True,
        trading_costs: "TradingCosts | None" = None,
    ) -> dict[str, float]:
        """Calculate all metrics including financial performance metrics.

        This method combines error metrics with financial metrics for a complete
        evaluation that accounts for realistic trading costs.

        Args:
            y_true: True values (prices)
            y_pred: Predicted values (prices)
            include_financial: Whether to include financial metrics (default True)
            trading_costs: Trading cost configuration. If None, uses defaults
                          (0.1% commission + 0.05% slippage)

        Returns:
            Dictionary with all metrics:
            - Error metrics: mae, mse, rmse, mape, smape, r2, max_error
            - directional_accuracy: Direction prediction accuracy (%)
            - Financial metrics (if include_financial=True):
              - fin_gross_pnl: P&L before costs
              - fin_net_pnl: P&L after costs
              - fin_total_costs: Total trading costs
              - fin_return_pct: Return percentage
              - fin_sharpe_ratio: Risk-adjusted return
              - fin_sortino_ratio: Downside risk-adjusted return
              - fin_max_drawdown_pct: Maximum peak-to-trough decline
              - fin_win_rate: Percentage of winning trades
              - fin_profit_factor: Gross profit / gross loss
              - fin_avg_win: Average winning trade
              - fin_avg_loss: Average losing trade

        Example:
            y_true = np.array([100, 101, 99, 102, 100])
            y_pred = np.array([100.5, 100.8, 99.5, 101.5, 100.2])

            # With default costs
            metrics = MetricsCalculator.calculate_all_with_financial(
                y_true, y_pred
            )

            # With custom costs
            from price_stradamus.evaluation.financial_metrics import TradingCosts
            costs = TradingCosts(commission_pct=0.0005, slippage_pct=0.0002)
            metrics = MetricsCalculator.calculate_all_with_financial(
                y_true, y_pred, trading_costs=costs
            )
        """
        # Calculate error metrics (existing)
        metrics = MetricsCalculator.calculate_all(y_true, y_pred)

        # Add financial metrics if requested
        if include_financial:
            from price_stradamus.evaluation.financial_metrics import (
                FinancialMetricsCalculator,
                TradingCosts,
            )

            # Use provided costs or defaults
            costs = trading_costs or TradingCosts()
            fin_calc = FinancialMetricsCalculator(costs=costs)

            # Calculate financial metrics
            financial = fin_calc.calculate_all(y_true, y_pred)

            # Add with 'fin_' prefix to distinguish from error metrics
            for key, value in financial.items():
                metrics[f"fin_{key}"] = value

            logger.debug(
                f"Financial metrics: Net P&L=${financial['net_pnl']:.2f}, "
                f"Sharpe={financial['sharpe_ratio']:.2f}, "
                f"Win Rate={financial['win_rate']:.1f}%"
            )

        return metrics

    @staticmethod
    def format_metrics(metrics: dict[str, float]) -> pd.DataFrame:
        """Format metrics as a nice DataFrame for display.

        Args:
            metrics: Dictionary of metric name -> value

        Returns:
            DataFrame with two columns: Metric and Value
        """
        # Format values based on metric type
        formatted_values = []
        for key, value in metrics.items():
            if key in ["mape", "smape", "directional_accuracy"]:
                # Percentage metrics
                formatted_values.append(f"{value:.2f}%")
            elif key == "r2":
                # R² score (can be negative)
                formatted_values.append(f"{value:.4f}")
            else:
                # Error metrics
                formatted_values.append(f"{value:.4f}")

        df = pd.DataFrame(
            {
                "Metric": list(metrics.keys()),
                "Value": formatted_values,
            }
        )

        return df

    @staticmethod
    def compare_models(
        y_true: np.ndarray,
        predictions: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Compare multiple models' predictions.

        Args:
            y_true: True values
            predictions: Dictionary mapping model name to predictions

        Returns:
            DataFrame with metrics for each model, sorted by RMSE

        Example:
            y_true = np.array([1, 2, 3, 4, 5])
            predictions = {
                "model_a": np.array([1.1, 2.0, 3.1, 3.9, 5.0]),
                "model_b": np.array([0.9, 2.1, 2.9, 4.1, 5.1]),
            }
            comparison = MetricsCalculator.compare_models(y_true, predictions)
        """
        logger.info(f"Comparing {len(predictions)} models")

        results = []

        for model_name, y_pred in predictions.items():
            metrics = MetricsCalculator.calculate_all(y_true, y_pred)
            metrics["model"] = model_name
            results.append(metrics)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Reorder columns (model first, then metrics)
        cols = ["model"] + [col for col in df.columns if col != "model"]
        df = df[cols]

        # Sort by RMSE (lower is better)
        df = df.sort_values("rmse")

        logger.info(f"Comparison complete. Best model: {df.iloc[0]['model']}")

        return df
