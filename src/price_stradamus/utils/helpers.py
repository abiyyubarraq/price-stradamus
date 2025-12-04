"""Helper utility functions."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from price_stradamus.config.constants import TIMEFRAME_TO_MS, Timeframe


def timeframe_to_milliseconds(timeframe: str | Timeframe) -> int:
    """Convert timeframe string to milliseconds.

    Args:
        timeframe: Timeframe string (e.g., "1m", "1h", "1d")

    Returns:
        Milliseconds for the timeframe

    Raises:
        ValueError: If timeframe is invalid
    """
    if isinstance(timeframe, str):
        try:
            timeframe = Timeframe(timeframe)
        except ValueError as e:
            msg = f"Invalid timeframe: {timeframe}"
            raise ValueError(msg) from e

    return TIMEFRAME_TO_MS[timeframe]


def timeframe_to_timedelta(timeframe: str | Timeframe) -> timedelta:
    """Convert timeframe to timedelta.

    Args:
        timeframe: Timeframe string or enum

    Returns:
        Timedelta object
    """
    ms = timeframe_to_milliseconds(timeframe)
    return timedelta(milliseconds=ms)


def ensure_directory(path: str | Path) -> Path:
    """Ensure directory exists, create if not.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def datetime_to_unix_ms(dt: datetime) -> int:
    """Convert datetime to Unix timestamp in milliseconds.

    Args:
        dt: Datetime object

    Returns:
        Unix timestamp in milliseconds
    """
    return int(dt.timestamp() * 1000)


def unix_ms_to_datetime(ts: int) -> datetime:
    """Convert Unix timestamp in milliseconds to datetime.

    Args:
        ts: Unix timestamp in milliseconds

    Returns:
        Datetime object
    """
    return datetime.fromtimestamp(ts / 1000)


def calculate_splits(
    total_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[int, int, int]:
    """Calculate train/val/test split sizes.

    Args:
        total_size: Total number of samples
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        Tuple of (train_size, val_size, test_size)

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        msg = "Split ratios must sum to 1.0"
        raise ValueError(msg)

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    return train_size, val_size, test_size


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string

    Examples:
        >>> format_duration(65)
        '1m 5s'
        >>> format_duration(3665)
        '1h 1m 5s'
    """
    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"

    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m {seconds}s"

    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h {minutes}m"


def validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Raises:
        ValueError: If required columns are missing
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        msg = f"DataFrame missing required columns: {missing}"
        raise ValueError(msg)


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Calculate DataFrame memory usage in MB.

    Args:
        df: DataFrame

    Returns:
        Memory usage in megabytes
    """
    return df.memory_usage(deep=True).sum() / 1024**2


__all__ = [
    "timeframe_to_milliseconds",
    "timeframe_to_timedelta",
    "ensure_directory",
    "datetime_to_unix_ms",
    "unix_ms_to_datetime",
    "calculate_splits",
    "set_random_seeds",
    "format_duration",
    "validate_dataframe",
    "memory_usage_mb",
]
