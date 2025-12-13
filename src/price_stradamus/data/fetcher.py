"""Binance API data fetcher with async support.

This module provides an async client for fetching OHLCV (candlestick) data
from the Binance API with rate limiting, retry logic, and error handling.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Any

import aiohttp
import pandas as pd
from loguru import logger

from price_stradamus.config.constants import OHLCV_COLUMNS, Timeframe
from price_stradamus.config.settings import settings
from price_stradamus.utils.helpers import datetime_to_unix_ms, timeframe_to_milliseconds


class RateLimiter:
    """Token bucket rate limiter for API requests.

    Implements a token bucket algorithm to limit requests per minute.
    """

    def __init__(self, max_requests: int = 1200, time_window: int = 60):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed per time window
            time_window: Time window in seconds (default: 60s = 1 minute)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.tokens = max_requests
        self.last_update = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            # Refill tokens based on time elapsed
            now = datetime.now(timezone.utc)
            elapsed = (now - self.last_update).total_seconds()
            self.tokens = min(
                self.max_requests,
                self.tokens + (elapsed * self.max_requests / self.time_window),
            )
            self.last_update = now

            # Wait if no tokens available
            if self.tokens < 1:
                wait_time = (1 - self.tokens) * self.time_window / self.max_requests
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 1

            # Consume a token
            self.tokens -= 1


class BinanceDataFetcher:
    """Async Binance API client for fetching OHLCV data.

    Features:
    - Async HTTP requests using aiohttp
    - Token bucket rate limiting (1200 req/min)
    - Exponential backoff retry logic
    - Comprehensive error handling
    - Data validation and conversion to pandas DataFrames

    Example:
        async with BinanceDataFetcher() as fetcher:
            df = await fetcher.fetch_historical_range(
                symbol="BTCUSDT",
                interval="1m",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
            )
    """

    def __init__(
        self,
        base_url: str | None = None,
        max_requests_per_minute: int = 1200,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """Initialize Binance data fetcher.

        Args:
            base_url: Binance API base URL (default from settings)
            max_requests_per_minute: Rate limit (default: 1200)
            max_retries: Maximum number of retry attempts (default: 3)
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url or settings.binance_base_url
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        self.session: aiohttp.ClientSession | None = None
        self.rate_limiter = RateLimiter(max_requests=max_requests_per_minute)

        logger.info(
            f"Initialized BinanceDataFetcher (rate limit: {max_requests_per_minute} req/min)"
        )

    async def __aenter__(self) -> BinanceDataFetcher:
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _request_with_retry(
        self,
        url: str,
        params: dict[str, Any],
    ) -> Any:
        """Make HTTP request with exponential backoff retry logic.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            aiohttp.ClientError: If all retries fail
        """
        if not self.session:
            raise RuntimeError(
                "Session not initialized. Use 'async with' context manager."
            )

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Wait for rate limiter
                await self.rate_limiter.acquire()

                # Make request
                async with self.session.get(url, params=params) as response:
                    # Check for HTTP errors
                    if response.status == 429:
                        # Rate limit hit
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limit hit, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue

                    response.raise_for_status()

                    # Parse JSON
                    data = await response.json()

                    # Binance API error codes
                    if isinstance(data, dict) and "code" in data:
                        error_code = data["code"]
                        error_msg = data.get("msg", "Unknown error")
                        logger.error(f"Binance API error {error_code}: {error_msg}")
                        raise aiohttp.ClientError(f"Binance API error: {error_msg}")

                    return data

            except (TimeoutError, aiohttp.ClientError) as e:
                last_exception = e
                backoff = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"All {self.max_retries} retry attempts failed")

        # All retries failed
        raise last_exception or aiohttp.ClientError("Request failed after all retries")

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch OHLCV klines (candlestick) data from Binance API.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "1m", "5m", "1h", "1d")
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)
            limit: Number of candles to fetch (max 1000, default 1000)

        Returns:
            List of OHLCV dictionaries with keys:
            - timestamp: Open time (datetime)
            - open, high, low, close, volume: Price/volume data
            - quote_volume: Quote asset volume
            - num_trades: Number of trades

        Raises:
            aiohttp.ClientError: If API request fails
        """
        endpoint = f"{self.base_url}/api/v3/klines"

        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),  # Binance max is 1000
        }

        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        logger.debug(f"Fetching klines: {symbol} {interval} (limit={limit})")

        # Make request with retry logic
        data = await self._request_with_retry(endpoint, params)

        # Parse response
        # Binance response format: [
        #   [timestamp, open, high, low, close, volume, close_time,
        #    quote_volume, num_trades, taker_buy_base, taker_buy_quote, ignore]
        # ]
        klines = []
        for item in data:
            klines.append(
                {
                    "timestamp": pd.to_datetime(item[0], unit="ms", utc=True),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "quote_volume": float(item[7]),
                    "num_trades": int(item[8]),
                }
            )

        logger.info(f"Fetched {len(klines)} candles for {symbol} {interval}")
        return klines

    async def fetch_historical_range(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch historical data for a date range in chunks.

        Binance API limits requests to 1000 candles each. This method
        automatically splits the date range into chunks and fetches them.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "1m", "5m", "1h")
            start_date: Start datetime
            end_date: End datetime

        Returns:
            pandas DataFrame with OHLCV data, sorted by timestamp

        Raises:
            ValueError: If invalid timeframe or date range
        """
        # Validate timeframe
        try:
            timeframe = Timeframe(interval)
        except ValueError:
            valid_timeframes = [tf.value for tf in Timeframe]
            raise ValueError(
                f"Invalid timeframe '{interval}'. Valid options: {valid_timeframes}"
            ) from None

        # Calculate timeframe duration in milliseconds
        interval_ms = timeframe_to_milliseconds(timeframe)

        # Calculate total candles needed
        total_duration_ms = int((end_date - start_date).total_seconds() * 1000)
        total_candles = total_duration_ms // interval_ms

        logger.info(
            f"Fetching {total_candles} candles of {symbol} {interval} "
            f"from {start_date} to {end_date}"
        )

        # Fetch data in chunks of 1000 candles
        all_klines = []
        current_start = start_date
        chunk_duration = timedelta(milliseconds=interval_ms * 1000)  # 1000 candles

        while current_start < end_date:
            current_end = min(current_start + chunk_duration, end_date)

            # Convert to milliseconds
            start_ms = datetime_to_unix_ms(current_start)
            end_ms = datetime_to_unix_ms(current_end)

            # Fetch chunk
            klines = await self.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_ms,
                end_time=end_ms,
                limit=1000,
            )

            if not klines:
                logger.warning(f"No data returned for {current_start} to {current_end}")
                break

            all_klines.extend(klines)

            # Move to next chunk
            # Use the last candle's timestamp + 1 interval to avoid duplicates
            last_timestamp = klines[-1]["timestamp"]
            current_start = last_timestamp + timedelta(milliseconds=interval_ms)

            # Small delay to be nice to the API
            await asyncio.sleep(0.1)

        if not all_klines:
            logger.warning(f"No data found for {symbol} {interval}")
            return pd.DataFrame(columns=list(OHLCV_COLUMNS))  # type: ignore[call-overload]

        # Convert to DataFrame
        df = pd.DataFrame(all_klines)

        # Remove duplicates (can happen at chunk boundaries)
        df = df.drop_duplicates(subset=["timestamp"], keep="first")

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Validate data
        self._validate_ohlcv(df)

        logger.info(
            f"Successfully fetched {len(df)} candles for {symbol} {interval} "
            f"(requested {total_candles})"
        )

        return df

    async def get_latest_kline(self, symbol: str, interval: str) -> dict[str, Any]:
        """Get the most recent completed candle.

        Args:
            symbol: Trading pair symbol
            interval: Candle interval

        Returns:
            Dict with latest OHLCV data

        Raises:
            ValueError: If no data returned
        """
        klines = await self.fetch_klines(symbol=symbol, interval=interval, limit=1)

        if not klines:
            raise ValueError(f"No data returned for {symbol} {interval}")

        return klines[0]

    def _validate_ohlcv(self, df: pd.DataFrame) -> None:
        """Validate OHLCV data integrity.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            return

        # Check required columns exist
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check for null values
        has_nulls = df[required_cols].isnull().any()
        if has_nulls.any():  # type: ignore[union-attr]
            null_counts = df[required_cols].isnull().sum()
            raise ValueError(
                f"Found null values: {null_counts[null_counts > 0].to_dict()}"
            )

        # Check OHLCV relationships
        invalid_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
        invalid_low = df["low"] > df[["open", "close", "high"]].min(axis=1)

        if invalid_high.any():
            logger.warning(
                f"Found {invalid_high.sum()} candles where high < max(open, close)"
            )

        if invalid_low.any():
            logger.warning(
                f"Found {invalid_low.sum()} candles where low > min(open, close)"
            )

        # Check volume is non-negative
        if (df["volume"] < 0).any():
            raise ValueError("Found negative volume values")

        logger.debug(f"OHLCV validation passed for {len(df)} candles")
