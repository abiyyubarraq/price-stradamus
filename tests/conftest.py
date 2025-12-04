"""Pytest configuration and shared fixtures.

This module provides shared fixtures for testing Price Stradamus components.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries

from price_stradamus.config.settings import settings


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Generate sample OHLCV data for testing.

    Returns:
        DataFrame with 1000 rows of synthetic OHLCV data
    """
    np.random.seed(42)

    # Generate dates (1-minute frequency)
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")

    # Generate synthetic price data (random walk)
    close = 50000 + np.cumsum(np.random.randn(1000) * 100)

    # Generate OHLC from close
    high = close + np.abs(np.random.randn(1000) * 50)
    low = close - np.abs(np.random.randn(1000) * 50)
    open_price = np.roll(close, 1)  # Previous close
    open_price[0] = close[0]

    # Generate volume
    volume = np.random.randint(10, 1000, size=1000).astype(float)
    quote_volume = volume * close
    num_trades = np.random.randint(100, 500, size=1000)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": quote_volume,
            "num_trades": num_trades,
        }
    )


@pytest.fixture
def sample_timeseries(sample_ohlcv_df: pd.DataFrame) -> TimeSeries:
    """Generate sample Darts TimeSeries from OHLCV data.

    Args:
        sample_ohlcv_df: OHLCV DataFrame fixture

    Returns:
        Darts TimeSeries with close prices
    """
    df = sample_ohlcv_df.set_index("timestamp")
    return TimeSeries.from_dataframe(df, value_cols=["close"])


@pytest.fixture
def sample_features_df(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Generate sample DataFrame with features.

    Args:
        sample_ohlcv_df: OHLCV DataFrame fixture

    Returns:
        DataFrame with OHLCV and some simple features
    """
    df = sample_ohlcv_df.copy()

    # Add simple features
    df["returns"] = df["close"].pct_change()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["std_20"] = df["close"].rolling(window=20).std()

    # Drop NaN rows
    df = df.dropna()

    return df


@pytest.fixture
def sample_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample predictions and actuals for testing metrics.

    Returns:
        Tuple of (y_true, y_pred) arrays
    """
    np.random.seed(42)

    y_true = np.array([100, 102, 105, 103, 107, 110, 108, 112, 115, 113])
    y_pred = y_true + np.random.randn(10) * 2  # Add small noise

    return y_true, y_pred


@pytest.fixture
async def test_db_url() -> str:
    """Get test database URL.

    Returns:
        Test database connection string
    """
    # Use test database or same database for now
    # In production, you'd use a separate test database
    return settings.DATABASE_URL


@pytest.fixture
def mock_binance_response() -> list[list]:
    """Generate mock Binance API response.

    Returns:
        List of kline data in Binance format
    """
    base_time = int(datetime(2024, 1, 1).timestamp() * 1000)

    klines = []
    for i in range(10):
        timestamp = base_time + (i * 60000)  # 1 minute intervals
        klines.append(
            [
                timestamp,  # Open time
                "50000.00",  # Open
                "50100.00",  # High
                "49900.00",  # Low
                "50050.00",  # Close
                "100.5",  # Volume
                timestamp + 59999,  # Close time
                "5025000.00",  # Quote volume
                250,  # Number of trades
                "50.2",  # Taker buy base asset volume
                "2512500.00",  # Taker buy quote asset volume
                "0",  # Ignore
            ]
        )

    return klines


@pytest.fixture
def model_hyperparameters() -> dict:
    """Get standard model hyperparameters for testing.

    Returns:
        Dictionary with model parameters
    """
    return {
        "input_chunk_length": 20,
        "output_chunk_length": 5,
        "n_epochs": 2,  # Small number for fast testing
        "batch_size": 16,
        "learning_rate": 0.001,
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "asyncio: marks tests as async")
