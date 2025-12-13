"""Async PostgreSQL database manager.

This module provides an async database manager for storing and retrieving
time series data, features, predictions, and model metadata.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class DatabaseManager:
    """Async PostgreSQL database manager with connection pooling.

    Provides async operations for:
    - OHLCV raw data storage and retrieval
    - Technical indicator feature storage
    - Model prediction storage
    - Model metadata storage

    Example:
        db = DatabaseManager(settings.database_url)
        await db.initialize()

        # Save OHLCV data
        df = ... # pandas DataFrame with OHLCV data
        rows_saved = await db.save_ohlcv(df, "BTCUSDT", "1m")

        # Retrieve data
        df = await db.get_ohlcv("BTCUSDT", "1m", limit=1000)

        await db.close()
    """

    def __init__(self, connection_string: str):
        """Initialize database manager.

        Args:
            connection_string: PostgreSQL connection string (asyncpg format)
        """
        self.connection_string = connection_string
        self.engine: AsyncEngine | None = None
        self.sessionmaker: async_sessionmaker[AsyncSession] | None = None

        logger.info("DatabaseManager initialized")

    async def initialize(self) -> None:
        """Create async engine and session maker with connection pooling."""
        if self.engine:
            logger.warning("Engine already initialized")
            return

        # Create async engine with connection pooling
        self.engine = create_async_engine(
            self.connection_string,
            pool_size=10,  # Base number of connections
            max_overflow=20,  # Additional connections when pool is full
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False,  # Set to True for SQL logging
        )

        # Create async session maker
        self.sessionmaker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Don't expire objects after commit
        )

        logger.info(
            "Database engine and session maker initialized (pool_size=10, max_overflow=20)"
        )

    async def close(self) -> None:
        """Close engine and dispose of connection pool."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.sessionmaker = None
            logger.info("Database connections closed")

    async def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> int:
        """Insert OHLCV data, skipping duplicates.

        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume,
                quote_volume (optional), num_trades (optional)
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (e.g., "1m", "5m", "1h")

        Returns:
            Number of rows inserted (excludes duplicates)

        Raises:
            RuntimeError: If engine not initialized
            ValueError: If DataFrame is invalid
        """
        if not self.sessionmaker:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        if df.empty:
            logger.warning("Empty DataFrame provided, nothing to save")
            return 0

        # Validate required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Saving {len(df)} OHLCV records for {symbol} {timeframe}")

        # Prepare records for insertion
        records = []
        for _, row in df.iterrows():
            record = {
                "timestamp": row["timestamp"],
                "symbol": symbol,
                "timeframe": timeframe,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "quote_volume": float(row.get("quote_volume") or 0),
                "num_trades": int(row.get("num_trades") or 0),
            }
            records.append(record)

        # Use async session
        async with self.sessionmaker() as session:
            # Build INSERT ... ON CONFLICT DO NOTHING query
            # This skips duplicates based on unique constraint (symbol, timeframe, timestamp)
            query = text("""
                INSERT INTO market_data.ohlcv_raw
                (timestamp, symbol, timeframe, open, high, low, close, volume, quote_volume, num_trades)
                VALUES
                (:timestamp, :symbol, :timeframe, :open, :high, :low, :close, :volume, :quote_volume, :num_trades)
                ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
            """)

            # Execute batch insert
            result = await session.execute(query, records)
            await session.commit()

            rows_inserted = result.rowcount  # type: ignore[attr-defined]

        logger.info(
            f"Inserted {rows_inserted} new OHLCV records (skipped {len(df) - rows_inserted} duplicates)"
        )
        return rows_inserted

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Retrieve OHLCV data as DataFrame.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Filter records after this date (optional)
            end_date: Filter records before this date (optional)
            limit: Maximum number of records to return (optional)

        Returns:
            DataFrame with OHLCV data, sorted by timestamp ascending

        Raises:
            RuntimeError: If engine not initialized
        """
        if not self.sessionmaker:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        logger.debug(f"Querying OHLCV data for {symbol} {timeframe}")

        # Build query
        query = """
            SELECT timestamp, open, high, low, close, volume, quote_volume, num_trades
            FROM market_data.ohlcv_raw
            WHERE symbol = :symbol AND timeframe = :timeframe
        """
        params: dict[str, Any] = {"symbol": symbol, "timeframe": timeframe}

        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY timestamp ASC"

        if limit:
            query += " LIMIT :limit"
            params["limit"] = limit

        # Execute query
        async with self.sessionmaker() as session:
            result = await session.execute(text(query), params)
            rows = result.fetchall()

        # Convert to DataFrame
        columns_list = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "num_trades",
        ]

        if not rows:
            logger.warning(f"No OHLCV data found for {symbol} {timeframe}")
            return pd.DataFrame(columns=columns_list)

        df = pd.DataFrame(rows, columns=columns_list)

        # Convert Decimal columns to float for NumPy compatibility
        numeric_columns = ["open", "high", "low", "close", "volume", "quote_volume"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert num_trades to int
        df["num_trades"] = df["num_trades"].astype(int)

        logger.info(f"Retrieved {len(df)} OHLCV records for {symbol} {timeframe}")
        return df

    async def get_latest_timestamp(
        self,
        symbol: str,
        timeframe: str,
    ) -> datetime | None:
        """Get timestamp of most recent candle.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Most recent timestamp or None if no data exists
        """
        if not self.sessionmaker:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        query = text("""
            SELECT MAX(timestamp) as latest
            FROM market_data.ohlcv_raw
            WHERE symbol = :symbol AND timeframe = :timeframe
        """)

        async with self.sessionmaker() as session:
            result = await session.execute(
                query, {"symbol": symbol, "timeframe": timeframe}
            )
            row = result.fetchone()

        if row and row[0]:
            logger.debug(f"Latest timestamp for {symbol} {timeframe}: {row[0]}")
            return row[0]

        logger.debug(f"No data found for {symbol} {timeframe}")
        return None

    async def save_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        feature_version: str | None = None,
    ) -> int:
        """Save computed technical indicator features.

        Args:
            df: DataFrame with timestamp column and feature columns
            symbol: Trading symbol
            timeframe: Timeframe
            feature_version: Feature computation version string (optional)

        Returns:
            Number of rows inserted

        Raises:
            RuntimeError: If engine not initialized
            ValueError: If DataFrame is invalid
        """
        if not self.sessionmaker:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        if df.empty:
            logger.warning("Empty DataFrame provided, nothing to save")
            return 0

        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        logger.info(f"Saving features for {len(df)} timestamps ({symbol} {timeframe})")

        # Prepare records
        records = []
        feature_columns = [col for col in df.columns if col != "timestamp"]

        for _, row in df.iterrows():
            # Store all features in JSONB column
            features_dict = {}
            for col in feature_columns:
                value = row[col]
                # Check for any type of missing value (None, NaN, pd.NA, etc.)
                if value is None or pd.isna(value):
                    features_dict[col] = None
                else:
                    features_dict[col] = float(value)

            record = {
                "timestamp": row["timestamp"],
                "symbol": symbol,
                "timeframe": timeframe,
                "features": json.dumps(features_dict),
                "feature_version": feature_version,
            }
            records.append(record)

        # Insert with ON CONFLICT DO UPDATE to replace existing features
        query = text("""
            INSERT INTO market_data.features
            (timestamp, symbol, timeframe, features, feature_version)
            VALUES
            (:timestamp, :symbol, :timeframe, :features, :feature_version)
            ON CONFLICT (symbol, timeframe, timestamp)
            DO UPDATE SET features = EXCLUDED.features, feature_version = EXCLUDED.feature_version
        """)

        async with self.sessionmaker() as session:
            await session.execute(query, records)
            await session.commit()

        logger.info(f"Saved features for {len(records)} timestamps")
        return len(records)

    async def get_features(
        self,
        symbol: str,
        timeframe: str,
        feature_names: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Retrieve computed features as DataFrame.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            feature_names: List of specific feature names to retrieve (optional, retrieves all if None)
            start_date: Filter records after this date (optional)
            end_date: Filter records before this date (optional)

        Returns:
            DataFrame with timestamp and feature columns

        Raises:
            RuntimeError: If engine not initialized
        """
        if not self.sessionmaker:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        logger.debug(f"Querying features for {symbol} {timeframe}")

        # Build query
        query = """
            SELECT timestamp, features
            FROM market_data.features
            WHERE symbol = :symbol AND timeframe = :timeframe
        """
        params: dict[str, Any] = {"symbol": symbol, "timeframe": timeframe}

        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY timestamp ASC"

        # Execute query
        async with self.sessionmaker() as session:
            result = await session.execute(text(query), params)
            rows = result.fetchall()

        if not rows:
            logger.warning(f"No features found for {symbol} {timeframe}")
            return pd.DataFrame()

        # Parse JSONB and create DataFrame
        records = []
        for row in rows:
            timestamp, features_json = row
            # PostgreSQL JSONB columns are already deserialized to dict by the driver
            features = features_json if isinstance(features_json, dict) else json.loads(features_json)
            record = {"timestamp": timestamp, **features}
            records.append(record)

        df = pd.DataFrame(records)

        # Filter to specific features if requested
        if feature_names:
            available_features = set(df.columns) - {"timestamp"}
            requested_features = set(feature_names)
            missing_features = requested_features - available_features

            if missing_features:
                logger.warning(f"Requested features not found: {missing_features}")

            # Select only available requested features
            columns_to_keep = ["timestamp"] + [
                f for f in feature_names if f in df.columns
            ]
            df = df[columns_to_keep]

        logger.info(
            f"Retrieved features for {len(df)} timestamps ({len(df.columns) - 1} features)"
        )
        # Ensure we return DataFrame type
        assert isinstance(df, pd.DataFrame)
        return df

    async def save_predictions(
        self,
        predictions: list[dict[str, Any]],
    ) -> int:
        """Save model predictions.

        Args:
            predictions: List of prediction dictionaries with keys:
                - model_name, prediction_timestamp, target_timestamp, symbol, timeframe,
                  steps_ahead, predicted_close, actual_close (optional),
                  predicted_direction (optional), actual_direction (optional)

        Returns:
            Number of rows inserted

        Raises:
            RuntimeError: If engine not initialized
        """
        if not self.sessionmaker:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        if not predictions:
            logger.warning("Empty predictions list, nothing to save")
            return 0

        logger.info(f"Saving {len(predictions)} predictions")

        query = text("""
            INSERT INTO ml_data.predictions
            (model_name, prediction_timestamp, target_timestamp, symbol, timeframe,
             steps_ahead, predicted_close, actual_close, predicted_direction, actual_direction, error)
            VALUES
            (:model_name, :prediction_timestamp, :target_timestamp, :symbol, :timeframe,
             :steps_ahead, :predicted_close, :actual_close, :predicted_direction, :actual_direction, :error)
        """)

        async with self.sessionmaker() as session:
            await session.execute(query, predictions)
            await session.commit()

        logger.info(f"Saved {len(predictions)} predictions")
        return len(predictions)

    async def save_model_metadata(
        self,
        model_name: str,
        model_type: str,
        model_class: str,
        hyperparameters: dict[str, Any],
        metrics: dict[str, float],
        training_info: dict[str, Any],
    ) -> int:
        """Save model training metadata.

        Args:
            model_name: Unique model name
            model_type: Model type (e.g., "neural", "classical", "ml", "automl")
            model_class: Model class name (e.g., "NBEATSModel")
            hyperparameters: Model hyperparameters dictionary
            metrics: Performance metrics dictionary (MAE, RMSE, etc.)
            training_info: Additional training info:
                - trained_at, training_duration_seconds, data_start, data_end,
                  num_samples, symbol, timeframe, feature_list, version, file_path, notes

        Returns:
            1 if successful

        Raises:
            RuntimeError: If engine not initialized
        """
        if not self.sessionmaker:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        logger.info(f"Saving metadata for model '{model_name}'")

        # Prepare record
        record = {
            "model_name": model_name,
            "model_type": model_type,
            "model_class": model_class,
            "hyperparameters": json.dumps(hyperparameters),
            "metrics": json.dumps(metrics),
            "trained_at": training_info.get("trained_at", datetime.now(timezone.utc)),
            "training_duration_seconds": training_info.get("training_duration_seconds"),
            "data_start": training_info.get("data_start"),
            "data_end": training_info.get("data_end"),
            "num_samples": training_info.get("num_samples"),
            "symbol": training_info.get("symbol"),
            "timeframe": training_info.get("timeframe"),
            "feature_list": json.dumps(training_info.get("feature_list", [])),
            "version": training_info.get("version"),
            "file_path": training_info.get("file_path"),
            "notes": training_info.get("notes"),
        }

        # Insert or update (ON CONFLICT UPDATE)
        query = text("""
            INSERT INTO ml_data.model_metadata
            (model_name, model_type, model_class, hyperparameters, metrics, trained_at,
             training_duration_seconds, data_start, data_end, num_samples, symbol, timeframe,
             feature_list, version, file_path, notes)
            VALUES
            (:model_name, :model_type, :model_class, :hyperparameters, :metrics, :trained_at,
             :training_duration_seconds, :data_start, :data_end, :num_samples, :symbol, :timeframe,
             :feature_list, :version, :file_path, :notes)
            ON CONFLICT (model_name)
            DO UPDATE SET
                model_type = EXCLUDED.model_type,
                model_class = EXCLUDED.model_class,
                hyperparameters = EXCLUDED.hyperparameters,
                metrics = EXCLUDED.metrics,
                trained_at = EXCLUDED.trained_at,
                training_duration_seconds = EXCLUDED.training_duration_seconds,
                data_start = EXCLUDED.data_start,
                data_end = EXCLUDED.data_end,
                num_samples = EXCLUDED.num_samples,
                symbol = EXCLUDED.symbol,
                timeframe = EXCLUDED.timeframe,
                feature_list = EXCLUDED.feature_list,
                version = EXCLUDED.version,
                file_path = EXCLUDED.file_path,
                notes = EXCLUDED.notes,
                updated_at = CURRENT_TIMESTAMP
        """)

        async with self.sessionmaker() as session:
            await session.execute(query, record)
            await session.commit()

        logger.info(f"Saved metadata for model '{model_name}'")
        return 1

    async def get_model_metadata(
        self,
        model_name: str,
    ) -> dict[str, Any] | None:
        """Get metadata for a specific model.

        Args:
            model_name: Model name

        Returns:
            Dictionary with model metadata or None if not found

        Raises:
            RuntimeError: If engine not initialized
        """
        if not self.sessionmaker:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        query = text("""
            SELECT model_name, model_type, model_class, hyperparameters, metrics, trained_at,
                   training_duration_seconds, data_start, data_end, num_samples, symbol, timeframe,
                   feature_list, version, file_path, notes, created_at, updated_at
            FROM ml_data.model_metadata
            WHERE model_name = :model_name
        """)

        async with self.sessionmaker() as session:
            result = await session.execute(query, {"model_name": model_name})
            row = result.fetchone()

        if not row:
            logger.warning(f"Model metadata not found for '{model_name}'")
            return None

        # Parse result
        metadata = {
            "model_name": row[0],
            "model_type": row[1],
            "model_class": row[2],
            "hyperparameters": json.loads(row[3]),
            "metrics": json.loads(row[4]),
            "trained_at": row[5],
            "training_duration_seconds": float(row[6]) if row[6] else None,
            "data_start": row[7],
            "data_end": row[8],
            "num_samples": row[9],
            "symbol": row[10],
            "timeframe": row[11],
            "feature_list": json.loads(row[12]) if row[12] else [],
            "version": row[13],
            "file_path": row[14],
            "notes": row[15],
            "created_at": row[16],
            "updated_at": row[17],
        }

        logger.debug(f"Retrieved metadata for model '{model_name}'")
        return metadata
