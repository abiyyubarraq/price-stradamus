"""Walk-forward backtesting for time series models.

This module provides backtesting functionality with proper time series
cross-validation to avoid lookahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from darts import TimeSeries
from loguru import logger

from price_stradamus.evaluation.metrics import MetricsCalculator
from price_stradamus.models.base import BaseModel


class Backtester:
    """Walk-forward backtesting for time series models.

    Implements proper time series cross-validation methods:
    - Walk-forward validation: Train on increasing/sliding window, test on next period
    - Expanding window: Train on all historical data up to test point
    - Optional retraining at regular intervals

    This avoids lookahead bias and provides realistic performance estimates.

    Example:
        model = NBEATSModel(input_chunk_length=60, output_chunk_length=5)
        backtester = Backtester(
            model=model,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        results = backtester.run_expanding_window(data)
        print(results["metrics"])
    """

    def __init__(
        self,
        model: BaseModel,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        retrain_frequency: int | None = None,
    ):
        """Initialize backtester.

        Args:
            model: Model instance to evaluate
            train_ratio: Proportion of data for training (default: 0.7)
            val_ratio: Proportion of data for validation (default: 0.15)
            test_ratio: Proportion of data for testing (default: 0.15)
            retrain_frequency: How often to retrain in walk-forward (None = no retraining)

        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        self.model = model
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.retrain_frequency = retrain_frequency

        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total_ratio}"
            )

        logger.info(
            f"Initialized Backtester (train={train_ratio:.1%}, val={val_ratio:.1%}, "
            f"test={test_ratio:.1%}, retrain_freq={retrain_frequency})"
        )

    def run_walk_forward(
        self,
        data: TimeSeries,
        initial_train_size: int,
        step_size: int = 1,
    ) -> dict:
        """Run walk-forward validation.

        At each step:
        1. Train model on historical data up to current point
        2. Predict next N steps
        3. Move forward by step_size
        4. Optionally retrain

        This simulates real-world forecasting where you continually
        update predictions as new data arrives.

        Args:
            data: Full time series data
            initial_train_size: Number of samples for initial training
            step_size: Number of steps to move forward each iteration (default: 1)

        Returns:
            Dictionary with:
            - predictions: Array of predicted values
            - actuals: Array of actual values
            - timestamps: Array of timestamps
            - metrics: Performance metrics
            - num_iterations: Number of prediction iterations

        Raises:
            ValueError: If data too small or parameters invalid
        """
        if len(data) < initial_train_size + self.model.output_chunk_length:
            raise ValueError(
                f"Data too small: need at least {initial_train_size + self.model.output_chunk_length} "
                f"timesteps, but have {len(data)}"
            )

        logger.info(
            f"Starting walk-forward validation (initial_train={initial_train_size}, "
            f"step={step_size}, output_length={self.model.output_chunk_length})"
        )

        predictions_list = []
        actuals_list = []
        timestamps_list = []

        # Initial training
        logger.info("Training initial model...")
        train_data = data[:initial_train_size]
        self.model.fit(train_data)

        # Walk forward
        current_idx = initial_train_size
        iteration = 0

        while current_idx + self.model.output_chunk_length <= len(data):
            # Get historical data up to current point
            historical = data[:current_idx]

            # Predict next N steps
            try:
                pred = self.model.predict(
                    n=self.model.output_chunk_length, series=historical
                )
            except Exception as e:
                logger.error(f"Prediction failed at iteration {iteration}: {e}")
                break

            # Get actual values
            actual_start = current_idx
            actual_end = current_idx + self.model.output_chunk_length
            actual = data[actual_start:actual_end]

            # Store results
            predictions_list.append(pred.values().flatten())
            actuals_list.append(actual.values().flatten())
            timestamps_list.append(actual.time_index.to_numpy())

            # Move forward
            current_idx += step_size
            iteration += 1

            # Retrain if needed
            if self.retrain_frequency and iteration % self.retrain_frequency == 0:
                logger.info(f"Retraining model at iteration {iteration}")
                train_data = data[:current_idx]
                self.model.fit(train_data)

            # Progress logging
            if iteration % 10 == 0:
                logger.debug(f"Walk-forward iteration {iteration}")

        if not predictions_list:
            raise ValueError("No predictions were generated")

        # Combine results
        predictions = np.concatenate(predictions_list)
        actuals = np.concatenate(actuals_list)
        timestamps = np.concatenate(timestamps_list)

        # Calculate metrics
        metrics = MetricsCalculator.calculate_all(actuals, predictions)

        logger.info(
            f"Walk-forward validation complete: {iteration} iterations, "
            f"{len(predictions)} predictions"
        )
        logger.info(
            f"Performance: RMSE={metrics['rmse']:.4f}, "
            f"Directional Accuracy={metrics['directional_accuracy']:.2f}%"
        )

        return {
            "predictions": predictions,
            "actuals": actuals,
            "timestamps": timestamps,
            "metrics": metrics,
            "num_iterations": iteration,
        }

    def run_expanding_window(
        self,
        data: TimeSeries,
    ) -> dict:
        """Run expanding window backtesting.

        Simple train/val/test split with expanding training window:
        1. Train on train_ratio of data
        2. Validate on val_ratio of data (optional, for early stopping)
        3. Test on test_ratio of data

        This is faster than walk-forward but provides less granular results.

        Args:
            data: Full time series data

        Returns:
            Dictionary with:
            - predictions: Array of predicted values
            - actuals: Array of actual values
            - timestamps: Array of timestamps
            - metrics: Performance metrics
            - train_size, val_size, test_size: Split sizes

        Raises:
            ValueError: If data too small
        """
        # Split data
        train_size = int(len(data) * self.train_ratio)
        val_size = int(len(data) * self.val_ratio)

        train_data = data[:train_size]
        val_data = data[train_size : train_size + val_size] if val_size > 0 else None
        test_data = data[train_size + val_size :]

        if len(test_data) < self.model.output_chunk_length:
            raise ValueError(
                f"Test set too small: {len(test_data)} timesteps "
                f"(need at least {self.model.output_chunk_length})"
            )

        logger.info(
            f"Running expanding window backtest: "
            f"train={len(train_data)}, val={len(val_data) if val_data else 0}, "
            f"test={len(test_data)}"
        )

        # Train model
        logger.info("Training model...")
        self.model.fit(train_data, val_data)

        # Predict on test set (iteratively)
        logger.info("Generating predictions on test set...")
        predictions_list = []
        actuals_list = []
        timestamps_list = []

        current_idx = 0
        while current_idx + self.model.output_chunk_length <= len(test_data):
            # Historical data includes train + val + test up to current point
            historical_end = train_size + val_size + current_idx
            historical = data[:historical_end]

            # Predict next N steps
            pred = self.model.predict(
                n=self.model.output_chunk_length, series=historical
            )

            # Get actual values from test set
            actual_start = current_idx
            actual_end = current_idx + self.model.output_chunk_length
            actual = test_data[actual_start:actual_end]

            predictions_list.append(pred.values().flatten())
            actuals_list.append(actual.values().flatten())
            timestamps_list.append(actual.time_index.to_numpy())

            # Move forward by one output_chunk_length (non-overlapping predictions)
            current_idx += self.model.output_chunk_length

        # Combine results
        predictions = np.concatenate(predictions_list)
        actuals = np.concatenate(actuals_list)
        timestamps = np.concatenate(timestamps_list)

        # Calculate metrics
        metrics = MetricsCalculator.calculate_all(actuals, predictions)

        logger.info(
            f"Expanding window backtest complete: {len(predictions)} predictions"
        )
        logger.info(
            f"Performance: RMSE={metrics['rmse']:.4f}, "
            f"Directional Accuracy={metrics['directional_accuracy']:.2f}%"
        )

        return {
            "predictions": predictions,
            "actuals": actuals,
            "timestamps": timestamps,
            "metrics": metrics,
            "train_size": len(train_data),
            "val_size": len(val_data) if val_data else 0,
            "test_size": len(test_data),
        }

    def get_summary(self, results: dict) -> pd.DataFrame:
        """Get formatted summary of backtest results.

        Args:
            results: Results dictionary from run_walk_forward or run_expanding_window

        Returns:
            DataFrame with formatted metrics
        """
        return MetricsCalculator.format_metrics(results["metrics"])

    def plot_results(
        self,
        results: dict,
        save_path: str | None = None,
    ) -> None:
        """Plot backtest results (requires matplotlib).

        Args:
            results: Results dictionary from backtesting
            save_path: Path to save plot (optional, if None: displays plot)

        Note:
            This method requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error(
                "matplotlib not installed. Install with: pip install matplotlib"
            )
            return

        timestamps = results["timestamps"]
        actuals = results["actuals"]
        predictions = results["predictions"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Actual vs Predicted
        ax1.plot(timestamps, actuals, label="Actual", alpha=0.7, linewidth=1)
        ax1.plot(timestamps, predictions, label="Predicted", alpha=0.7, linewidth=1)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")
        ax1.set_title(f"Backtest Results: {self.model.name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Prediction Errors
        errors = actuals - predictions
        ax2.plot(timestamps, errors, label="Error", alpha=0.7, linewidth=1, color="red")
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Error (Actual - Predicted)")
        ax2.set_title("Prediction Errors")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
