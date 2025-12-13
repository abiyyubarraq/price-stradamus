"""Walk-forward validation to prevent data leakage.

This module implements proper temporal train/test splits for time series models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator

import pandas as pd
from darts import TimeSeries
from loguru import logger


@dataclass
class TimeWindow:
    """Single train/test window for walk-forward validation."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def __post_init__(self):
        """Validate temporal ordering."""
        if self.train_start >= self.train_end:
            raise ValueError("Train start must be before train end")
        if self.test_start >= self.test_end:
            raise ValueError("Test start must be before test end")
        if self.test_start <= self.train_end:
            raise ValueError(
                f"Data leakage detected! Test starts ({self.test_start}) "
                f"before training ends ({self.train_end})"
            )

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"TimeWindow(\n"
            f"  Train: {self.train_start} to {self.train_end}\n"
            f"  Test:  {self.test_start} to {self.test_end}\n"
            f")"
        )


class WalkForwardSplitter:
    """Generate walk-forward validation windows.

    Prevents data leakage by ensuring test data is always AFTER training data.

    Example:
        >>> splitter = WalkForwardSplitter(
        ...     train_window_days=60,
        ...     test_window_days=10,
        ...     step_days=10,
        ... )
        >>> for window in splitter.split(start_date, end_date):
        ...     print(window)
    """

    def __init__(
        self,
        train_window_days: int = 60,
        test_window_days: int = 10,
        step_days: int = 10,
        gap_days: int = 0,
    ):
        """Initialize walk-forward splitter.

        Args:
            train_window_days: Size of training window
            test_window_days: Size of test window
            step_days: How many days to step forward each iteration
            gap_days: Gap between train and test (for data processing delays)
        """
        self.train_window = timedelta(days=train_window_days)
        self.test_window = timedelta(days=test_window_days)
        self.step = timedelta(days=step_days)
        self.gap = timedelta(days=gap_days)

    def split(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Iterator[TimeWindow]:
        """Generate walk-forward windows.

        Args:
            start_date: Earliest available data
            end_date: Latest available data

        Yields:
            TimeWindow objects with train/test splits
        """
        current = start_date

        while True:
            train_start = current
            train_end = current + self.train_window

            test_start = train_end + self.gap
            test_end = test_start + self.test_window

            # Stop if we run out of data
            if test_end > end_date:
                break

            window = TimeWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

            logger.debug(f"Generated {window}")
            yield window

            # Step forward
            current += self.step

    def split_dataframe(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Split DataFrame using walk-forward windows.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Yields:
            (train_df, test_df) tuples
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Column '{timestamp_col}' not found in DataFrame")

        # Ensure timestamp is datetime in UTC
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

        # Convert min/max to datetime explicitly
        start_date_val = df[timestamp_col].min()
        end_date_val = df[timestamp_col].max()

        # Handle pandas Timestamp objects
        if isinstance(start_date_val, pd.Timestamp):
            start_date = start_date_val.to_pydatetime()
        else:
            start_date = pd.to_datetime(start_date_val, utc=True).to_pydatetime()

        if isinstance(end_date_val, pd.Timestamp):
            end_date = end_date_val.to_pydatetime()
        else:
            end_date = pd.to_datetime(end_date_val, utc=True).to_pydatetime()

        for window in self.split(start_date, end_date):
            train_mask = (df[timestamp_col] >= window.train_start) & (
                df[timestamp_col] < window.train_end
            )
            test_mask = (df[timestamp_col] >= window.test_start) & (
                df[timestamp_col] < window.test_end
            )

            train_result = df[train_mask].copy()
            test_result = df[test_mask].copy()

            # Type narrowing: ensure we have DataFrames
            if not isinstance(train_result, pd.DataFrame):
                logger.warning(f"Train result is not a DataFrame for window {window}")
                continue

            if not isinstance(test_result, pd.DataFrame):
                logger.warning(f"Test result is not a DataFrame for window {window}")
                continue

            if len(train_result) == 0:
                logger.warning(f"Empty training set for window {window}")
                continue

            if len(test_result) == 0:
                logger.warning(f"Empty test set for window {window}")
                continue

            yield train_result, test_result


@dataclass
class DataLeakageChecker:
    """Check for data leakage in train/test splits."""

    @staticmethod
    def check_temporal_leakage(
        train_data: pd.DataFrame | TimeSeries,
        test_data: pd.DataFrame | TimeSeries,
        timestamp_col: str = "timestamp",
    ) -> dict[str, bool | datetime]:
        """Check if test data comes after training data.

        Args:
            train_data: Training data
            test_data: Test data
            timestamp_col: Name of timestamp column

        Returns:
            Dictionary with leakage check results
        """
        # Convert TimeSeries to DataFrame if needed
        if isinstance(train_data, TimeSeries):
            # Use to_dataframe() method to convert to DataFrame
            train_df = train_data.to_dataframe().reset_index()  # type: ignore[attr-defined]
            timestamp_col = str(train_df.columns[0])
        else:
            train_df = train_data

        if isinstance(test_data, TimeSeries):
            # Use to_dataframe() method to convert to DataFrame
            test_df = test_data.to_dataframe().reset_index()  # type: ignore[attr-defined]
        else:
            test_df = test_data

        train_max = train_df[timestamp_col].max()
        test_min = test_df[timestamp_col].min()

        has_leakage = test_min <= train_max

        result = {
            "has_leakage": has_leakage,
            "train_max": train_max,
            "test_min": test_min,
            "gap": (test_min - train_max).total_seconds() / 3600,  # hours
        }

        if has_leakage:
            logger.error(
                f"DATA LEAKAGE DETECTED!\n"
                f"  Training ends: {train_max}\n"
                f"  Testing starts: {test_min}\n"
                f"  Test data overlaps training by {-result['gap']:.1f} hours"
            )
        else:
            logger.info(
                f"✓ No data leakage. Gap between train/test: {result['gap']:.1f} hours"
            )

        return result

    @staticmethod
    def check_feature_leakage(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        future_cols: list[str],
    ) -> dict[str, list[str]]:
        """Check if test set contains features that shouldn't be known at prediction time.

        Args:
            train_data: Training data
            test_data: Test data
            future_cols: Columns that represent future information

        Returns:
            Dictionary with columns that leak future information
        """
        leaking_cols = [col for col in future_cols if col in test_data.columns]

        if leaking_cols:
            logger.error(
                f"Feature leakage detected! Test set contains future columns: {leaking_cols}"
            )
        else:
            logger.info("✓ No feature leakage detected")

        return {"leaking_columns": leaking_cols}


# Example usage
if __name__ == "__main__":
    # Example: Generate walk-forward windows
    splitter = WalkForwardSplitter(
        train_window_days=60,
        test_window_days=10,
        step_days=10,
        gap_days=1,  # 1-day gap to simulate real-world delay
    )

    start = datetime(2025, 8, 1)
    end = datetime(2025, 12, 4)

    print("Walk-Forward Validation Windows:")
    print("=" * 80)
    for i, window in enumerate(splitter.split(start, end), 1):
        print(f"\nFold {i}:")
        print(window)

    # Example: Check for data leakage
    print("\n\nData Leakage Check Example:")
    print("=" * 80)

    # Simulate train/test data
    train_df = pd.DataFrame(
        {"timestamp": pd.date_range("2025-11-01", "2025-11-30", freq="1h")}
    )
    test_df = pd.DataFrame(
        {"timestamp": pd.date_range("2025-12-01", "2025-12-10", freq="1h")}
    )

    checker = DataLeakageChecker()
    result = checker.check_temporal_leakage(train_df, test_df)
