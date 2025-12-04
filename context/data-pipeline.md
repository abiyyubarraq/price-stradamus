# Price Stradamus - Data Pipeline Documentation

## Overview

The data pipeline handles the complete flow from raw market data to ML-ready features. It ensures data quality, consistency, and proper temporal ordering for time series forecasting.

## Pipeline Stages

```
Binance API → Raw Storage → Validation → Preprocessing → Feature Engineering → Model Input
```

---

## Stage 1: Data Fetching (Binance API)

### API Endpoints Used

**Klines (Candlestick) Endpoint**:
```
GET /api/v3/klines
Parameters:
  - symbol: Trading pair (e.g., "BTCUSDT")
  - interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
  - startTime: Unix timestamp (ms)
  - endTime: Unix timestamp (ms)
  - limit: Max 1000 candles per request
```

**Response Format**:
```json
[
  [
    1499040000000,      // Open time
    "0.01634790",       // Open
    "0.80000000",       // High
    "0.01575800",       // Low
    "0.01577100",       // Close
    "148976.11427815",  // Volume
    1499644799999,      // Close time
    "2434.19055334",    // Quote asset volume
    308,                // Number of trades
    "1756.87402397",    // Taker buy base asset volume
    "28.46694368",      // Taker buy quote asset volume
    "17928899.62484339" // Ignore
  ]
]
```

### Rate Limiting Strategy

**Binance Limits**:
- Weight limits: 1200 per minute, 6000 per 5 minutes
- Raw request limit: 5000 per 5 minutes
- Each klines request = weight 1

**Our Strategy**:
```python
class RateLimiter:
    def __init__(self):
        self.max_requests_per_minute = 1000  # Buffer below 1200
        self.requests = []

    async def acquire(self):
        now = time.time()
        # Remove requests older than 1 minute
        self.requests = [t for t in self.requests if now - t < 60]

        if len(self.requests) >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.requests[0])
            await asyncio.sleep(wait_time)

        self.requests.append(now)
```

### Retry Logic

**Exponential Backoff**:
```python
async def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict:
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get("Retry-After", 60))
                        await asyncio.sleep(retry_after)
                    elif response.status >= 500:  # Server error
                        delay = base_delay * (2 ** attempt)  # Exponential
                        await asyncio.sleep(delay)
                    else:
                        raise DataFetchError(f"HTTP {response.status}")
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(base_delay * (2 ** attempt))

    raise DataFetchError("Max retries exceeded")
```

### Historical Data Chunking

**Problem**: Binance limits to 1000 candles per request

**Solution**: Break into chunks
```python
async def fetch_historical_data(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    chunk_size = timeframe_to_ms(timeframe) * 1000  # 1000 candles
    chunks = []

    current_start = start
    while current_start < end:
        current_end = min(current_start + chunk_size, end)
        chunk = await fetch_chunk(symbol, timeframe, current_start, current_end)
        chunks.append(chunk)
        current_start = current_end

    return pd.concat(chunks, ignore_index=True)
```

**Timeframe to Milliseconds**:
- 1m = 60,000 ms
- 5m = 300,000 ms
- 1h = 3,600,000 ms
- 1d = 86,400,000 ms

---

## Stage 2: Raw Storage (PostgreSQL)

### Database Schema

**Table: market_data.ohlcv_raw**
```sql
CREATE TABLE market_data.ohlcv_raw (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    num_trades INT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_candle UNIQUE(timestamp, symbol, timeframe)
);

-- Indexes for fast queries
CREATE INDEX idx_ohlcv_lookup
ON market_data.ohlcv_raw(symbol, timeframe, timestamp DESC);

-- Partitioning by month for large datasets
CREATE TABLE market_data.ohlcv_raw_2025_01
PARTITION OF market_data.ohlcv_raw
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

### Upsert Strategy

**Handle Duplicates**:
```python
async def upsert_ohlcv(conn, data: pd.DataFrame):
    """Insert or update on conflict."""
    query = """
        INSERT INTO market_data.ohlcv_raw (
            timestamp, symbol, timeframe, open, high, low, close, volume
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (timestamp, symbol, timeframe)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume;
    """
    await conn.executemany(query, data.to_records(index=False))
```

---

## Stage 3: Validation

### Data Quality Checks

**1. OHLC Consistency**
```python
def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure high/low bounds are respected."""
    invalid_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
    invalid_low = df["low"] > df[["open", "close", "high"]].min(axis=1)

    if invalid_high.any() or invalid_low.any():
        logger.warning(f"Found {invalid_high.sum() + invalid_low.sum()} invalid candles")
        # Option 1: Fix by adjusting high/low
        df.loc[invalid_high, "high"] = df.loc[invalid_high, ["open", "close", "low"]].max(axis=1)
        df.loc[invalid_low, "low"] = df.loc[invalid_low, ["open", "close", "high"]].min(axis=1)
        # Option 2: Drop invalid rows
        # df = df[~(invalid_high | invalid_low)]

    return df
```

**2. Volume Validation**
```python
def validate_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure volume is non-negative."""
    invalid = df["volume"] < 0
    if invalid.any():
        logger.error(f"Found {invalid.sum()} negative volume candles")
        df = df[~invalid]
    return df
```

**3. Timestamp Continuity**
```python
def validate_timestamps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Check for gaps in timestamps."""
    df = df.sort_values("timestamp")
    expected_delta = timeframe_to_timedelta(timeframe)

    actual_deltas = df["timestamp"].diff()
    gaps = actual_deltas[actual_deltas > expected_delta * 1.5]

    if not gaps.empty:
        logger.warning(f"Found {len(gaps)} gaps in data")
        # Log gap details
        for idx, delta in gaps.items():
            logger.debug(f"Gap at {df.loc[idx, 'timestamp']}: {delta}")

    return df
```

**4. Outlier Detection**
```python
from scipy import stats

def detect_outliers(df: pd.DataFrame, columns: list[str], z_threshold: float = 5.0) -> pd.DataFrame:
    """Detect outliers using z-score."""
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = z_scores > z_threshold

        if outliers.any():
            logger.warning(f"Found {outliers.sum()} outliers in {col}")
            # Option 1: Cap at threshold
            df.loc[outliers, col] = df[col].median()
            # Option 2: Remove outliers
            # df = df[~outliers]

    return df
```

---

## Stage 4: Preprocessing

### Missing Value Handling

**Forward Fill**:
```python
def fill_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Fill missing values using forward fill."""
    missing_before = df.isnull().sum().sum()

    if method == "ffill":
        df = df.fillna(method="ffill")
    elif method == "interpolate":
        df = df.interpolate(method="linear", limit=5)  # Max 5 consecutive
    elif method == "drop":
        df = df.dropna()

    missing_after = df.isnull().sum().sum()
    logger.info(f"Filled {missing_before - missing_after} missing values")

    return df
```

### Normalization

**MinMaxScaler** (Default):
```python
from sklearn.preprocessing import MinMaxScaler

def normalize_minmax(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Normalize to [0, 1] range."""
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler
```

**StandardScaler**:
```python
from sklearn.preprocessing import StandardScaler

def normalize_standard(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Normalize to zero mean, unit variance."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler
```

**RobustScaler** (Recommended for financial data):
```python
from sklearn.preprocessing import RobustScaler

def normalize_robust(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, RobustScaler]:
    """Normalize using median and IQR (robust to outliers)."""
    scaler = RobustScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler
```

---

## Stage 5: Feature Engineering

### Technical Indicators (pandas-ta)

**Price-Based Features**:
```python
import pandas_ta as ta

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-based features."""
    # Returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Price changes
    df["price_change"] = df["close"] - df["open"]
    df["price_range"] = df["high"] - df["low"]

    # Body/wick ratios
    df["body_ratio"] = abs(df["close"] - df["open"]) / df["price_range"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    return df
```

**Moving Averages**:
```python
def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA and EMA."""
    periods = [7, 14, 30, 50, 200]

    for period in periods:
        df[f"sma_{period}"] = ta.sma(df["close"], length=period)
        df[f"ema_{period}"] = ta.ema(df["close"], length=period)

    # Crossovers
    df["sma_7_14_cross"] = (df["sma_7"] > df["sma_14"]).astype(int)
    df["ema_7_14_cross"] = (df["ema_7"] > df["ema_14"]).astype(int)

    return df
```

**Volatility Indicators**:
```python
def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility indicators."""
    # ATR
    df[["atr_14"]] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Bollinger Bands
    bbands = ta.bbands(df["close"], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)

    # Standard deviation
    df["std_14"] = df["returns"].rolling(14).std()
    df["std_30"] = df["returns"].rolling(30).std()

    # Historical volatility
    df["hvol_14"] = df["log_returns"].rolling(14).std() * np.sqrt(252)

    return df
```

**Momentum Indicators**:
```python
def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators."""
    # RSI
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    # MACD
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    # Stochastic
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    df = pd.concat([df, stoch], axis=1)

    # ROC
    df["roc_12"] = ta.roc(df["close"], length=12)

    # MOM
    df["mom_10"] = ta.mom(df["close"], length=10)

    return df
```

**Volume Indicators**:
```python
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume indicators."""
    # OBV
    df["obv"] = ta.obv(df["close"], df["volume"])

    # VWAP
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

    # Volume SMA
    df["volume_sma_20"] = ta.sma(df["volume"], length=20)
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    return df
```

**Trend Indicators**:
```python
def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend indicators."""
    # ADX
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    df = pd.concat([df, adx], axis=1)

    # Aroon
    aroon = ta.aroon(df["high"], df["low"], length=25)
    df = pd.concat([df, aroon], axis=1)

    # CCI
    df["cci_20"] = ta.cci(df["high"], df["low"], df["close"], length=20)

    return df
```

### Lag Features

```python
def add_lag_features(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    """Add lagged features."""
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

# Usage
lags = [1, 2, 3, 5, 10, 15, 30]
lag_columns = ["close", "volume", "rsi_14", "macd"]
df = add_lag_features(df, lag_columns, lags)
```

### Time-Based Features

```python
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df
```

---

## Stage 6: Conversion to Darts TimeSeries

```python
from darts import TimeSeries

def to_timeseries(df: pd.DataFrame, value_cols: list[str] | None = None) -> TimeSeries:
    """Convert DataFrame to Darts TimeSeries."""
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    ts = TimeSeries.from_dataframe(
        df,
        time_col="timestamp",
        value_cols=value_cols,
        fill_missing_dates=True,
        freq="T",  # Minute frequency
    )

    return ts
```

---

## Train/Validation/Test Split

### Temporal Split (Required for Time Series)

```python
def temporal_split(
    data: TimeSeries,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[TimeSeries, TimeSeries, TimeSeries]:
    """Split data temporally (no shuffle!)."""
    assert train_ratio + val_ratio + test_ratio == 1.0

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    logger.info(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    return train, val, test
```

---

## Walk-Forward Validation

```python
def walk_forward_split(
    data: TimeSeries,
    train_size: int,
    test_size: int,
    step: int = 1,
) -> list[tuple[TimeSeries, TimeSeries]]:
    """Generate walk-forward splits."""
    splits = []

    for start in range(0, len(data) - train_size - test_size, step):
        train_end = start + train_size
        test_end = train_end + test_size

        train = data[start:train_end]
        test = data[train_end:test_end]

        splits.append((train, test))

    logger.info(f"Generated {len(splits)} walk-forward splits")

    return splits
```

---

## Data Quality Metrics

Track these metrics for monitoring:

- **Completeness**: % of non-null values
- **Consistency**: % of valid OHLC candles
- **Continuity**: % of candles without gaps
- **Outlier Rate**: % of outliers detected
- **Fetch Success Rate**: % of successful API calls
- **Storage Success Rate**: % of successful DB inserts

---

## Data Versioning

### DVC Integration

Data Version Control (DVC) enables Git-like versioning for datasets and ML pipelines.

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import subprocess


@dataclass
class DataVersion:
    """Metadata for a data version."""

    version_id: str
    timestamp: datetime
    source: str
    row_count: int
    hash: str
    schema_version: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DataVersionManager:
    """Manage data versioning with DVC.

    Features:
    - Track data file changes
    - Store version metadata
    - Enable reproducibility
    - Support rollback
    """

    def __init__(self, repo_path: Path, remote: str = "s3://price-stradamus/data"):
        self.repo_path = repo_path
        self.remote = remote
        self.versions_file = repo_path / ".data_versions.json"
        self._ensure_dvc_initialized()

    def _ensure_dvc_initialized(self) -> None:
        """Ensure DVC is initialized in the repository."""
        dvc_dir = self.repo_path / ".dvc"
        if not dvc_dir.exists():
            subprocess.run(["dvc", "init"], cwd=self.repo_path, check=True)
            subprocess.run(
                ["dvc", "remote", "add", "-d", "storage", self.remote],
                cwd=self.repo_path,
                check=True,
            )

    def track_data(self, data_path: Path, description: str = "") -> DataVersion:
        """Track a data file with DVC.

        Args:
            data_path: Path to data file
            description: Version description

        Returns:
            DataVersion with tracking metadata

        Example:
            >>> manager = DataVersionManager(Path("."))
            >>> version = manager.track_data(Path("data/btcusdt_1m.parquet"))
            >>> print(f"Tracked: {version.version_id}")
        """
        # Calculate hash
        file_hash = self._calculate_file_hash(data_path)

        # Add to DVC
        subprocess.run(["dvc", "add", str(data_path)], cwd=self.repo_path, check=True)

        # Create version entry
        version = DataVersion(
            version_id=file_hash[:12],
            timestamp=datetime.utcnow(),
            source=str(data_path),
            row_count=self._count_rows(data_path),
            hash=file_hash,
            schema_version=self._infer_schema_version(data_path),
            metadata={"description": description},
        )

        # Store version metadata
        self._save_version(version)

        return version

    def push_to_remote(self) -> None:
        """Push tracked data to remote storage."""
        subprocess.run(["dvc", "push"], cwd=self.repo_path, check=True)

    def pull_version(self, version_id: str) -> Path:
        """Pull a specific data version.

        Args:
            version_id: Version identifier

        Returns:
            Path to pulled data file
        """
        # Get version metadata
        version = self._get_version(version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found")

        # Checkout to specific version
        subprocess.run(
            ["git", "checkout", version.metadata.get("git_commit", "HEAD"), "--", f"{version.source}.dvc"],
            cwd=self.repo_path,
            check=True,
        )

        # Pull data
        subprocess.run(["dvc", "pull", str(version.source)], cwd=self.repo_path, check=True)

        return Path(version.source)

    def list_versions(self, source: str | None = None) -> list[DataVersion]:
        """List all tracked versions.

        Args:
            source: Filter by source file (optional)

        Returns:
            List of DataVersion objects
        """
        if not self.versions_file.exists():
            return []

        with open(self.versions_file) as f:
            versions_data = json.load(f)

        versions = [DataVersion(**v) for v in versions_data]

        if source:
            versions = [v for v in versions if v.source == source]

        return sorted(versions, key=lambda v: v.timestamp, reverse=True)

    def _calculate_file_hash(self, path: Path) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _count_rows(self, path: Path) -> int:
        """Count rows in data file."""
        import pandas as pd

        if path.suffix == ".parquet":
            return len(pd.read_parquet(path))
        elif path.suffix == ".csv":
            return sum(1 for _ in open(path)) - 1
        return 0

    def _infer_schema_version(self, path: Path) -> str:
        """Infer schema version from file."""
        import pandas as pd

        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, nrows=1)
        columns = sorted(df.columns.tolist())
        return hashlib.md5(",".join(columns).encode()).hexdigest()[:8]

    def _save_version(self, version: DataVersion) -> None:
        """Save version to versions file."""
        versions = []
        if self.versions_file.exists():
            with open(self.versions_file) as f:
                versions = json.load(f)

        versions.append({
            "version_id": version.version_id,
            "timestamp": version.timestamp.isoformat(),
            "source": version.source,
            "row_count": version.row_count,
            "hash": version.hash,
            "schema_version": version.schema_version,
            "metadata": version.metadata,
        })

        with open(self.versions_file, "w") as f:
            json.dump(versions, f, indent=2)

    def _get_version(self, version_id: str) -> DataVersion | None:
        """Get version by ID."""
        versions = self.list_versions()
        return next((v for v in versions if v.version_id == version_id), None)


# DVC pipeline definition (dvc.yaml)
DVC_PIPELINE_TEMPLATE = """
stages:
  fetch_data:
    cmd: python -m price_stradamus.cli.commands fetch --days 30
    deps:
      - src/price_stradamus/data/fetcher.py
    outs:
      - data/raw/btcusdt_1m.parquet

  preprocess:
    cmd: python scripts/preprocess.py
    deps:
      - data/raw/btcusdt_1m.parquet
      - src/price_stradamus/data/preprocessor.py
    outs:
      - data/processed/btcusdt_1m_clean.parquet

  features:
    cmd: python scripts/generate_features.py
    deps:
      - data/processed/btcusdt_1m_clean.parquet
      - src/price_stradamus/data/features.py
    outs:
      - data/features/btcusdt_1m_features.parquet
    metrics:
      - metrics/feature_stats.json:
          cache: false

  train:
    cmd: python -m price_stradamus.cli.commands train --model nbeats
    deps:
      - data/features/btcusdt_1m_features.parquet
      - src/price_stradamus/models/neural/nbeats.py
    outs:
      - models/nbeats_latest.pth
    metrics:
      - metrics/training_metrics.json:
          cache: false
"""
```

### Lakehouse Pattern

The lakehouse architecture combines data lake flexibility with data warehouse reliability.

```python
from enum import Enum
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from deltalake import DeltaTable, write_deltalake


class DataZone(str, Enum):
    """Lakehouse data zones."""

    BRONZE = "bronze"   # Raw ingested data
    SILVER = "silver"   # Cleaned and validated
    GOLD = "gold"       # Feature-ready for ML


@dataclass
class LakehouseConfig:
    """Configuration for lakehouse storage."""

    base_path: Path
    partition_cols: list[str] = field(default_factory=lambda: ["symbol", "year", "month"])
    retention_days: int = 365


class LakehouseManager:
    """Manage lakehouse data architecture.

    Medallion Architecture:
    - Bronze: Raw data as-is from source
    - Silver: Cleaned, validated, deduplicated
    - Gold: Aggregated, enriched, ML-ready
    """

    def __init__(self, config: LakehouseConfig):
        self.config = config
        self._ensure_zones_exist()

    def _ensure_zones_exist(self) -> None:
        """Create zone directories."""
        for zone in DataZone:
            zone_path = self.config.base_path / zone.value
            zone_path.mkdir(parents=True, exist_ok=True)

    def get_zone_path(self, zone: DataZone, table_name: str) -> Path:
        """Get path for a table in a zone."""
        return self.config.base_path / zone.value / table_name

    def ingest_to_bronze(
        self,
        df: pd.DataFrame,
        table_name: str,
        mode: str = "append",
    ) -> None:
        """Ingest raw data to bronze zone.

        Args:
            df: Raw DataFrame
            table_name: Target table name
            mode: Write mode (append, overwrite)
        """
        # Add ingestion metadata
        df["_ingested_at"] = datetime.utcnow()
        df["_source"] = "binance_api"

        path = self.get_zone_path(DataZone.BRONZE, table_name)

        write_deltalake(
            str(path),
            df,
            mode=mode,
            partition_by=["symbol"] if "symbol" in df.columns else None,
        )

        logger.info(f"Ingested {len(df)} rows to bronze/{table_name}")

    def promote_to_silver(
        self,
        source_table: str,
        target_table: str,
        transformations: list[Callable[[pd.DataFrame], pd.DataFrame]],
    ) -> None:
        """Promote data from bronze to silver with cleaning.

        Args:
            source_table: Bronze table name
            target_table: Silver table name
            transformations: List of transformation functions
        """
        # Read from bronze
        bronze_path = self.get_zone_path(DataZone.BRONZE, source_table)
        dt = DeltaTable(str(bronze_path))
        df = dt.to_pandas()

        # Apply transformations
        for transform in transformations:
            df = transform(df)

        # Add processing metadata
        df["_processed_at"] = datetime.utcnow()
        df["_quality_score"] = self._calculate_quality_score(df)

        # Write to silver
        silver_path = self.get_zone_path(DataZone.SILVER, target_table)
        write_deltalake(str(silver_path), df, mode="overwrite")

        logger.info(f"Promoted {len(df)} rows to silver/{target_table}")

    def create_gold_table(
        self,
        source_tables: list[str],
        target_table: str,
        aggregation: Callable[[list[pd.DataFrame]], pd.DataFrame],
    ) -> None:
        """Create gold table from silver sources.

        Args:
            source_tables: List of silver table names
            target_table: Gold table name
            aggregation: Aggregation function
        """
        # Read all source tables
        dfs = []
        for table in source_tables:
            path = self.get_zone_path(DataZone.SILVER, table)
            dt = DeltaTable(str(path))
            dfs.append(dt.to_pandas())

        # Apply aggregation
        gold_df = aggregation(dfs)

        # Add gold metadata
        gold_df["_created_at"] = datetime.utcnow()
        gold_df["_version"] = self._get_next_version(target_table)

        # Write to gold
        gold_path = self.get_zone_path(DataZone.GOLD, target_table)
        write_deltalake(str(gold_path), gold_df, mode="overwrite")

        logger.info(f"Created gold/{target_table} with {len(gold_df)} rows")

    def time_travel(
        self,
        zone: DataZone,
        table_name: str,
        version: int | None = None,
        timestamp: datetime | None = None,
    ) -> pd.DataFrame:
        """Read historical version of data.

        Args:
            zone: Data zone
            table_name: Table name
            version: Specific version number
            timestamp: Point-in-time query

        Returns:
            DataFrame at specified version
        """
        path = self.get_zone_path(zone, table_name)
        dt = DeltaTable(str(path))

        if version is not None:
            dt.load_version(version)
        elif timestamp is not None:
            dt.load_as_of_timestamp(timestamp)

        return dt.to_pandas()

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score (0-1)."""
        completeness = 1 - df.isnull().sum().sum() / df.size
        return completeness

    def _get_next_version(self, table_name: str) -> int:
        """Get next version number for gold table."""
        path = self.get_zone_path(DataZone.GOLD, table_name)
        if path.exists():
            dt = DeltaTable(str(path))
            return dt.version() + 1
        return 1
```

---

## Feature Store Architecture

### Feature Store Concepts

A Feature Store provides centralized feature management for ML, ensuring consistency between training and serving.

```python
from __future__ import annotations

import hashlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import redis


@dataclass
class FeatureDefinition:
    """Definition of a feature."""

    name: str
    description: str
    dtype: str
    entity_key: str  # e.g., "symbol"
    version: int = 1
    owner: str = ""
    tags: list[str] = field(default_factory=list)
    ttl_hours: int | None = None  # Time-to-live for online features

    # Computation metadata
    source_tables: list[str] = field(default_factory=list)
    transformation: str = ""  # SQL or Python expression

    @property
    def feature_id(self) -> str:
        """Unique feature identifier."""
        return f"{self.name}_v{self.version}"


@dataclass
class FeatureValue:
    """A single feature value with metadata."""

    feature_id: str
    entity_key: str
    entity_value: str
    value: Any
    timestamp: datetime
    created_at: datetime = field(default_factory=datetime.utcnow)


class FeatureStore:
    """Centralized feature store for ML.

    Components:
    - Feature Registry: Metadata about features
    - Offline Store: Historical feature values (for training)
    - Online Store: Low-latency feature serving (for inference)
    """

    def __init__(
        self,
        offline_store_path: Path,
        online_store_host: str = "localhost",
        online_store_port: int = 6379,
    ):
        self.offline_store_path = offline_store_path
        self.offline_store_path.mkdir(parents=True, exist_ok=True)

        # Online store (Redis)
        self.online_store = redis.Redis(
            host=online_store_host,
            port=online_store_port,
            decode_responses=False,
        )

        # Feature registry
        self.registry: dict[str, FeatureDefinition] = {}

    def register_feature(self, definition: FeatureDefinition) -> None:
        """Register a feature definition.

        Args:
            definition: Feature definition

        Example:
            >>> store.register_feature(FeatureDefinition(
            ...     name="rsi_14",
            ...     description="14-period Relative Strength Index",
            ...     dtype="float64",
            ...     entity_key="symbol",
            ...     source_tables=["silver.ohlcv"],
            ...     transformation="pandas_ta.rsi(close, 14)",
            ... ))
        """
        self.registry[definition.feature_id] = definition
        logger.info(f"Registered feature: {definition.feature_id}")

    def materialize_offline(
        self,
        feature_ids: list[str],
        df: pd.DataFrame,
        entity_col: str = "symbol",
        timestamp_col: str = "timestamp",
    ) -> Path:
        """Materialize features to offline store.

        Args:
            feature_ids: Features to materialize
            df: DataFrame with feature values
            entity_col: Entity column name
            timestamp_col: Timestamp column name

        Returns:
            Path to materialized feature file
        """
        # Validate features exist
        for fid in feature_ids:
            if fid not in self.registry:
                raise ValueError(f"Feature {fid} not registered")

        # Create feature DataFrame
        feature_df = df[[timestamp_col, entity_col] + [f.split("_v")[0] for f in feature_ids]].copy()
        feature_df["_materialized_at"] = datetime.utcnow()

        # Save to offline store
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = self.offline_store_path / f"features_{timestamp}.parquet"
        feature_df.to_parquet(output_path, index=False)

        logger.info(f"Materialized {len(feature_ids)} features to {output_path}")
        return output_path

    def push_to_online(
        self,
        feature_values: list[FeatureValue],
    ) -> int:
        """Push features to online store for serving.

        Args:
            feature_values: List of feature values

        Returns:
            Number of features pushed
        """
        pipe = self.online_store.pipeline()

        for fv in feature_values:
            key = f"feature:{fv.feature_id}:{fv.entity_value}"
            value = pickle.dumps({
                "value": fv.value,
                "timestamp": fv.timestamp.isoformat(),
            })

            # Get TTL from registry
            definition = self.registry.get(fv.feature_id)
            ttl = definition.ttl_hours * 3600 if definition and definition.ttl_hours else None

            if ttl:
                pipe.setex(key, ttl, value)
            else:
                pipe.set(key, value)

        pipe.execute()
        return len(feature_values)

    def get_online_features(
        self,
        feature_ids: list[str],
        entity_values: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Get features from online store for inference.

        Args:
            feature_ids: Features to retrieve
            entity_values: Entity values (e.g., ["BTCUSDT", "ETHUSDT"])

        Returns:
            Dictionary mapping entity to feature values

        Example:
            >>> features = store.get_online_features(
            ...     ["rsi_14_v1", "macd_v1"],
            ...     ["BTCUSDT"]
            ... )
            >>> print(features["BTCUSDT"]["rsi_14_v1"])
            65.5
        """
        result: dict[str, dict[str, Any]] = {ev: {} for ev in entity_values}

        for entity in entity_values:
            for fid in feature_ids:
                key = f"feature:{fid}:{entity}"
                raw_value = self.online_store.get(key)

                if raw_value:
                    data = pickle.loads(raw_value)
                    result[entity][fid] = data["value"]
                else:
                    result[entity][fid] = None
                    logger.warning(f"Feature {fid} not found for {entity}")

        return result

    def get_historical_features(
        self,
        feature_ids: list[str],
        entity_df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Point-in-time correct feature retrieval for training.

        Args:
            feature_ids: Features to retrieve
            entity_df: DataFrame with entity keys and timestamps
            timestamp_col: Timestamp column name

        Returns:
            DataFrame with features joined at correct timestamps
        """
        # Load offline features
        feature_files = sorted(self.offline_store_path.glob("features_*.parquet"))
        if not feature_files:
            raise ValueError("No offline features found")

        features_df = pd.concat([pd.read_parquet(f) for f in feature_files])

        # Point-in-time join (as-of join)
        result = pd.merge_asof(
            entity_df.sort_values(timestamp_col),
            features_df.sort_values(timestamp_col),
            on=timestamp_col,
            by="symbol",
            direction="backward",  # Use features available at that time
        )

        return result


class FeatureComputer:
    """Compute features from raw data."""

    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store

    def compute_features(
        self,
        df: pd.DataFrame,
        feature_ids: list[str],
    ) -> pd.DataFrame:
        """Compute features for a DataFrame.

        Args:
            df: Input DataFrame with OHLCV data
            feature_ids: Features to compute

        Returns:
            DataFrame with computed features
        """
        result = df.copy()

        for fid in feature_ids:
            definition = self.store.registry.get(fid)
            if definition is None:
                logger.warning(f"Feature {fid} not in registry, skipping")
                continue

            # Compute feature based on transformation
            feature_name = definition.name
            if "rsi" in feature_name:
                result[feature_name] = ta.rsi(df["close"], length=14)
            elif "macd" in feature_name:
                macd = ta.macd(df["close"])
                result[feature_name] = macd["MACD_12_26_9"]
            elif "sma" in feature_name:
                period = int(feature_name.split("_")[1])
                result[feature_name] = ta.sma(df["close"], length=period)
            # Add more feature computations...

        return result
```

---

## Data Quality Gates

### Automated Quality Validation

Quality gates ensure data meets standards before entering the next pipeline stage.

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import great_expectations as gx


class QualityStatus(str, Enum):
    """Quality check status."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class QualityCheckResult:
    """Result of a quality check."""

    check_name: str
    status: QualityStatus
    metric_value: float | None
    threshold: float | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityGateResult:
    """Result of all quality checks."""

    gate_name: str
    timestamp: datetime
    overall_status: QualityStatus
    checks: list[QualityCheckResult]
    blocking: bool = True

    @property
    def passed(self) -> bool:
        """Check if gate passed."""
        return self.overall_status != QualityStatus.FAILED

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "gate_name": self.gate_name,
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "passed": self.passed,
            "checks": [
                {
                    "name": c.check_name,
                    "status": c.status.value,
                    "metric": c.metric_value,
                    "threshold": c.threshold,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }


class QualityCheck(ABC):
    """Base class for quality checks."""

    def __init__(self, name: str, threshold: float | None = None):
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def check(self, df: pd.DataFrame) -> QualityCheckResult:
        """Run the quality check."""
        pass


class CompletenessCheck(QualityCheck):
    """Check for missing values."""

    def __init__(self, columns: list[str] | None = None, threshold: float = 0.95):
        super().__init__("completeness", threshold)
        self.columns = columns

    def check(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check completeness of specified columns."""
        cols = self.columns or df.columns.tolist()
        completeness = 1 - df[cols].isnull().sum().sum() / (len(df) * len(cols))

        status = QualityStatus.PASSED if completeness >= self.threshold else QualityStatus.FAILED

        return QualityCheckResult(
            check_name=self.name,
            status=status,
            metric_value=completeness,
            threshold=self.threshold,
            message=f"Completeness: {completeness:.2%} (threshold: {self.threshold:.2%})",
            details={"columns_checked": cols},
        )


class FreshnessCheck(QualityCheck):
    """Check data freshness."""

    def __init__(self, timestamp_col: str = "timestamp", max_age_hours: float = 1.0):
        super().__init__("freshness", max_age_hours)
        self.timestamp_col = timestamp_col

    def check(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check if data is fresh enough."""
        if self.timestamp_col not in df.columns:
            return QualityCheckResult(
                check_name=self.name,
                status=QualityStatus.FAILED,
                metric_value=None,
                threshold=self.threshold,
                message=f"Column {self.timestamp_col} not found",
            )

        latest = pd.to_datetime(df[self.timestamp_col]).max()
        age_hours = (datetime.utcnow() - latest.to_pydatetime()).total_seconds() / 3600

        status = QualityStatus.PASSED if age_hours <= self.threshold else QualityStatus.FAILED

        return QualityCheckResult(
            check_name=self.name,
            status=status,
            metric_value=age_hours,
            threshold=self.threshold,
            message=f"Data age: {age_hours:.1f} hours (max: {self.threshold} hours)",
            details={"latest_timestamp": latest.isoformat()},
        )


class OHLCValidityCheck(QualityCheck):
    """Check OHLC data validity."""

    def __init__(self, threshold: float = 0.99):
        super().__init__("ohlc_validity", threshold)

    def check(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check OHLC constraints are satisfied."""
        # High >= max(Open, Close, Low)
        high_valid = df["high"] >= df[["open", "close", "low"]].max(axis=1)
        # Low <= min(Open, Close, High)
        low_valid = df["low"] <= df[["open", "close", "high"]].min(axis=1)

        validity_rate = (high_valid & low_valid).mean()

        status = QualityStatus.PASSED if validity_rate >= self.threshold else QualityStatus.FAILED

        return QualityCheckResult(
            check_name=self.name,
            status=status,
            metric_value=validity_rate,
            threshold=self.threshold,
            message=f"OHLC validity: {validity_rate:.2%} (threshold: {self.threshold:.2%})",
            details={
                "invalid_high_count": (~high_valid).sum(),
                "invalid_low_count": (~low_valid).sum(),
            },
        )


class RowCountCheck(QualityCheck):
    """Check minimum row count."""

    def __init__(self, min_rows: int = 1000):
        super().__init__("row_count", float(min_rows))
        self.min_rows = min_rows

    def check(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check DataFrame has minimum rows."""
        row_count = len(df)
        status = QualityStatus.PASSED if row_count >= self.min_rows else QualityStatus.FAILED

        return QualityCheckResult(
            check_name=self.name,
            status=status,
            metric_value=float(row_count),
            threshold=float(self.min_rows),
            message=f"Row count: {row_count:,} (min: {self.min_rows:,})",
        )


class DuplicateCheck(QualityCheck):
    """Check for duplicate rows."""

    def __init__(self, columns: list[str], max_duplicate_rate: float = 0.01):
        super().__init__("duplicates", max_duplicate_rate)
        self.columns = columns

    def check(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check for duplicates based on key columns."""
        duplicate_count = df.duplicated(subset=self.columns).sum()
        duplicate_rate = duplicate_count / len(df) if len(df) > 0 else 0

        status = QualityStatus.PASSED if duplicate_rate <= self.threshold else QualityStatus.FAILED

        return QualityCheckResult(
            check_name=self.name,
            status=status,
            metric_value=duplicate_rate,
            threshold=self.threshold,
            message=f"Duplicate rate: {duplicate_rate:.2%} ({duplicate_count:,} rows)",
            details={"duplicate_count": duplicate_count, "key_columns": self.columns},
        )


class SchemaCheck(QualityCheck):
    """Check DataFrame schema."""

    def __init__(self, expected_columns: list[str], expected_dtypes: dict[str, str] | None = None):
        super().__init__("schema")
        self.expected_columns = expected_columns
        self.expected_dtypes = expected_dtypes or {}

    def check(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check DataFrame matches expected schema."""
        missing_cols = set(self.expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(self.expected_columns)

        dtype_mismatches = {}
        for col, expected_dtype in self.expected_dtypes.items():
            if col in df.columns and str(df[col].dtype) != expected_dtype:
                dtype_mismatches[col] = {
                    "expected": expected_dtype,
                    "actual": str(df[col].dtype),
                }

        status = QualityStatus.PASSED
        if missing_cols or dtype_mismatches:
            status = QualityStatus.FAILED
        elif extra_cols:
            status = QualityStatus.WARNING

        return QualityCheckResult(
            check_name=self.name,
            status=status,
            metric_value=None,
            threshold=None,
            message=f"Schema check: missing={len(missing_cols)}, extra={len(extra_cols)}, dtype_mismatch={len(dtype_mismatches)}",
            details={
                "missing_columns": list(missing_cols),
                "extra_columns": list(extra_cols),
                "dtype_mismatches": dtype_mismatches,
            },
        )


class QualityGate:
    """Quality gate with multiple checks."""

    def __init__(
        self,
        name: str,
        checks: list[QualityCheck],
        blocking: bool = True,
    ):
        self.name = name
        self.checks = checks
        self.blocking = blocking

    def validate(self, df: pd.DataFrame) -> QualityGateResult:
        """Run all quality checks.

        Args:
            df: DataFrame to validate

        Returns:
            QualityGateResult with all check results
        """
        results = [check.check(df) for check in self.checks]

        # Determine overall status
        statuses = [r.status for r in results]
        if QualityStatus.FAILED in statuses:
            overall = QualityStatus.FAILED
        elif QualityStatus.WARNING in statuses:
            overall = QualityStatus.WARNING
        else:
            overall = QualityStatus.PASSED

        gate_result = QualityGateResult(
            gate_name=self.name,
            timestamp=datetime.utcnow(),
            overall_status=overall,
            checks=results,
            blocking=self.blocking,
        )

        # Log result
        logger.info(f"Quality gate '{self.name}': {overall.value}")
        for check in results:
            log_fn = logger.info if check.status == QualityStatus.PASSED else logger.warning
            log_fn(f"  - {check.check_name}: {check.message}")

        return gate_result


# Pre-built quality gates
def create_bronze_gate() -> QualityGate:
    """Quality gate for bronze (raw) data."""
    return QualityGate(
        name="bronze_quality_gate",
        checks=[
            RowCountCheck(min_rows=100),
            CompletenessCheck(columns=["timestamp", "open", "high", "low", "close", "volume"]),
            DuplicateCheck(columns=["timestamp", "symbol"]),
        ],
        blocking=True,
    )


def create_silver_gate() -> QualityGate:
    """Quality gate for silver (cleaned) data."""
    return QualityGate(
        name="silver_quality_gate",
        checks=[
            RowCountCheck(min_rows=1000),
            CompletenessCheck(threshold=0.99),
            OHLCValidityCheck(threshold=0.999),
            DuplicateCheck(columns=["timestamp", "symbol"], max_duplicate_rate=0.0),
            FreshnessCheck(max_age_hours=24),
        ],
        blocking=True,
    )


def create_gold_gate() -> QualityGate:
    """Quality gate for gold (feature) data."""
    return QualityGate(
        name="gold_quality_gate",
        checks=[
            RowCountCheck(min_rows=5000),
            CompletenessCheck(threshold=0.999),
            FreshnessCheck(max_age_hours=1),
            SchemaCheck(
                expected_columns=["timestamp", "symbol", "close", "rsi_14", "macd", "sma_20"],
                expected_dtypes={"close": "float64", "rsi_14": "float64"},
            ),
        ],
        blocking=True,
    )
```

---

## Incremental Processing

### Incremental vs Full Refresh Patterns

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable


@dataclass
class ProcessingState:
    """State tracking for incremental processing."""

    last_processed_timestamp: datetime | None
    last_run_timestamp: datetime | None
    records_processed: int = 0
    checkpoints: dict[str, datetime] = field(default_factory=dict)


class IncrementalProcessor:
    """Handle incremental data processing.

    Patterns:
    - Timestamp-based: Process records since last timestamp
    - Offset-based: Process records since last offset
    - Change Data Capture: Process only changed records
    """

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state = self._load_state()

    def _load_state(self) -> ProcessingState:
        """Load processing state from disk."""
        if self.state_path.exists():
            with open(self.state_path) as f:
                data = json.load(f)
                return ProcessingState(
                    last_processed_timestamp=datetime.fromisoformat(data["last_processed_timestamp"]) if data.get("last_processed_timestamp") else None,
                    last_run_timestamp=datetime.fromisoformat(data["last_run_timestamp"]) if data.get("last_run_timestamp") else None,
                    records_processed=data.get("records_processed", 0),
                    checkpoints=data.get("checkpoints", {}),
                )
        return ProcessingState(None, None)

    def _save_state(self) -> None:
        """Save processing state to disk."""
        with open(self.state_path, "w") as f:
            json.dump({
                "last_processed_timestamp": self.state.last_processed_timestamp.isoformat() if self.state.last_processed_timestamp else None,
                "last_run_timestamp": self.state.last_run_timestamp.isoformat() if self.state.last_run_timestamp else None,
                "records_processed": self.state.records_processed,
                "checkpoints": self.state.checkpoints,
            }, f)

    def get_incremental_range(
        self,
        lookback_buffer: timedelta = timedelta(hours=1),
    ) -> tuple[datetime, datetime]:
        """Get time range for incremental processing.

        Args:
            lookback_buffer: Buffer to re-process for late data

        Returns:
            (start_time, end_time) tuple
        """
        end_time = datetime.utcnow()

        if self.state.last_processed_timestamp:
            # Start from last processed minus buffer
            start_time = self.state.last_processed_timestamp - lookback_buffer
        else:
            # First run - default to 30 days
            start_time = end_time - timedelta(days=30)

        return start_time, end_time

    def process_incremental(
        self,
        fetch_fn: Callable[[datetime, datetime], pd.DataFrame],
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
        load_fn: Callable[[pd.DataFrame], None],
        timestamp_col: str = "timestamp",
    ) -> int:
        """Run incremental ETL pipeline.

        Args:
            fetch_fn: Function to fetch data for time range
            transform_fn: Transformation function
            load_fn: Function to load data to destination
            timestamp_col: Timestamp column name

        Returns:
            Number of records processed
        """
        start_time, end_time = self.get_incremental_range()

        logger.info(f"Processing incremental: {start_time} to {end_time}")

        # Fetch
        df = fetch_fn(start_time, end_time)
        if df.empty:
            logger.info("No new data to process")
            return 0

        # Transform
        df = transform_fn(df)

        # Load
        load_fn(df)

        # Update state
        self.state.last_processed_timestamp = pd.to_datetime(df[timestamp_col]).max().to_pydatetime()
        self.state.last_run_timestamp = datetime.utcnow()
        self.state.records_processed += len(df)
        self._save_state()

        logger.info(f"Processed {len(df)} records (total: {self.state.records_processed})")
        return len(df)

    def needs_full_refresh(self, max_gap_hours: float = 48) -> bool:
        """Check if full refresh is needed due to large gap.

        Args:
            max_gap_hours: Maximum acceptable gap in hours

        Returns:
            True if full refresh recommended
        """
        if self.state.last_processed_timestamp is None:
            return True

        gap = datetime.utcnow() - self.state.last_processed_timestamp
        return gap.total_seconds() / 3600 > max_gap_hours


class MergeStrategy:
    """Strategies for merging incremental data."""

    @staticmethod
    def upsert(
        existing: pd.DataFrame,
        new: pd.DataFrame,
        key_columns: list[str],
    ) -> pd.DataFrame:
        """Merge with upsert (update existing, insert new).

        Args:
            existing: Existing data
            new: New data
            key_columns: Columns that define uniqueness

        Returns:
            Merged DataFrame
        """
        # Remove existing rows that will be updated
        merged = existing[~existing.set_index(key_columns).index.isin(new.set_index(key_columns).index)]
        # Append new/updated rows
        return pd.concat([merged, new], ignore_index=True)

    @staticmethod
    def append_only(
        existing: pd.DataFrame,
        new: pd.DataFrame,
        key_columns: list[str],
    ) -> pd.DataFrame:
        """Append only truly new records.

        Args:
            existing: Existing data
            new: New data
            key_columns: Columns that define uniqueness

        Returns:
            DataFrame with new records appended
        """
        # Only keep new rows
        new_only = new[~new.set_index(key_columns).index.isin(existing.set_index(key_columns).index)]
        return pd.concat([existing, new_only], ignore_index=True)

    @staticmethod
    def merge_with_priority(
        existing: pd.DataFrame,
        new: pd.DataFrame,
        key_columns: list[str],
        prefer_new: bool = True,
    ) -> pd.DataFrame:
        """Merge with configurable priority.

        Args:
            existing: Existing data
            new: New data
            key_columns: Key columns
            prefer_new: Whether to prefer new values on conflict

        Returns:
            Merged DataFrame
        """
        if prefer_new:
            return MergeStrategy.upsert(existing, new, key_columns)
        else:
            # Keep existing on conflict
            return MergeStrategy.append_only(new, existing, key_columns)
```

---

## Late-Arriving Data Handling

### Strategies for Late Data

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class LateDataStrategy(str, Enum):
    """Strategies for handling late-arriving data."""

    REJECT = "reject"          # Reject late data
    ACCEPT_ALWAYS = "accept"   # Always accept late data
    ACCEPT_WINDOW = "window"   # Accept within time window
    REPROCESS = "reprocess"    # Trigger reprocessing


@dataclass
class LateDataConfig:
    """Configuration for late data handling."""

    strategy: LateDataStrategy
    grace_period: timedelta = timedelta(hours=1)
    max_lateness: timedelta = timedelta(days=7)
    watermark_update_interval: timedelta = timedelta(minutes=5)


class LateDataHandler:
    """Handle late-arriving data in streaming context.

    Watermark Concept:
    - Watermark = timestamp up to which we consider data complete
    - Data arriving after watermark is considered "late"
    - Late data may trigger re-computation or be rejected
    """

    def __init__(self, config: LateDataConfig):
        self.config = config
        self.watermark: datetime | None = None
        self.last_watermark_update: datetime | None = None
        self.late_data_log: list[dict] = []

    def update_watermark(self, event_time: datetime) -> None:
        """Update watermark based on incoming data.

        Args:
            event_time: Timestamp of incoming event
        """
        now = datetime.utcnow()

        # Only update periodically to avoid watermark jitter
        if (
            self.last_watermark_update is None
            or now - self.last_watermark_update > self.config.watermark_update_interval
        ):
            # Watermark = current time - grace period
            new_watermark = now - self.config.grace_period

            if self.watermark is None or new_watermark > self.watermark:
                self.watermark = new_watermark
                self.last_watermark_update = now
                logger.debug(f"Watermark updated to {self.watermark}")

    def is_late(self, event_time: datetime) -> bool:
        """Check if an event is late based on watermark.

        Args:
            event_time: Event timestamp

        Returns:
            True if event is late
        """
        if self.watermark is None:
            return False
        return event_time < self.watermark

    def handle_late_data(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process DataFrame and separate late data.

        Args:
            df: Input DataFrame
            timestamp_col: Timestamp column name

        Returns:
            (on_time_data, late_data) tuple
        """
        if self.watermark is None:
            # No watermark yet - all data is on-time
            return df, pd.DataFrame()

        timestamps = pd.to_datetime(df[timestamp_col])

        # Separate late and on-time data
        is_late = timestamps < self.watermark
        late_df = df[is_late].copy()
        on_time_df = df[~is_late].copy()

        if len(late_df) > 0:
            self._process_late_data(late_df, timestamp_col)

        return on_time_df, late_df

    def _process_late_data(
        self,
        late_df: pd.DataFrame,
        timestamp_col: str,
    ) -> None:
        """Process late data based on configured strategy."""
        lateness = datetime.utcnow() - pd.to_datetime(late_df[timestamp_col]).max()

        match self.config.strategy:
            case LateDataStrategy.REJECT:
                logger.warning(f"Rejecting {len(late_df)} late records (lateness: {lateness})")
                self._log_late_data(late_df, "rejected")

            case LateDataStrategy.ACCEPT_ALWAYS:
                logger.info(f"Accepting {len(late_df)} late records")
                self._log_late_data(late_df, "accepted")

            case LateDataStrategy.ACCEPT_WINDOW:
                if lateness <= self.config.max_lateness:
                    logger.info(f"Accepting {len(late_df)} late records within window")
                    self._log_late_data(late_df, "accepted_window")
                else:
                    logger.warning(f"Rejecting {len(late_df)} records - too late ({lateness})")
                    self._log_late_data(late_df, "rejected_too_late")

            case LateDataStrategy.REPROCESS:
                logger.info(f"Scheduling reprocess for {len(late_df)} late records")
                self._log_late_data(late_df, "reprocess_scheduled")
                self._schedule_reprocess(late_df)

    def _log_late_data(self, df: pd.DataFrame, action: str) -> None:
        """Log late data for monitoring."""
        self.late_data_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "record_count": len(df),
            "action": action,
            "min_event_time": str(df["timestamp"].min()),
            "max_event_time": str(df["timestamp"].max()),
        })

    def _schedule_reprocess(self, df: pd.DataFrame) -> None:
        """Schedule reprocessing for late data."""
        # Implementation depends on your job scheduler
        # Could trigger Airflow DAG, push to queue, etc.
        pass

    def get_late_data_stats(self) -> dict:
        """Get statistics about late data."""
        if not self.late_data_log:
            return {"total_late_records": 0, "actions": {}}

        actions = {}
        total = 0
        for entry in self.late_data_log:
            action = entry["action"]
            count = entry["record_count"]
            actions[action] = actions.get(action, 0) + count
            total += count

        return {
            "total_late_records": total,
            "actions": actions,
            "current_watermark": self.watermark.isoformat() if self.watermark else None,
        }
```

---

## Schema Evolution

### Handling Schema Changes

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SchemaChangeType(str, Enum):
    """Types of schema changes."""

    ADD_COLUMN = "add_column"
    DROP_COLUMN = "drop_column"
    RENAME_COLUMN = "rename_column"
    CHANGE_TYPE = "change_type"
    ADD_CONSTRAINT = "add_constraint"
    DROP_CONSTRAINT = "drop_constraint"


@dataclass
class SchemaChange:
    """A single schema change."""

    change_type: SchemaChangeType
    column_name: str
    old_value: Any = None
    new_value: Any = None
    is_breaking: bool = False


@dataclass
class SchemaVersion:
    """A schema version with changes."""

    version: int
    created_at: datetime
    changes: list[SchemaChange]
    description: str = ""


class SchemaRegistry:
    """Track and manage schema evolution.

    Compatibility Modes:
    - BACKWARD: New schema can read old data
    - FORWARD: Old schema can read new data
    - FULL: Both backward and forward compatible
    - NONE: No compatibility required
    """

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.schemas: dict[str, list[SchemaVersion]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load schema registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
                for table, versions in data.items():
                    self.schemas[table] = [
                        SchemaVersion(
                            version=v["version"],
                            created_at=datetime.fromisoformat(v["created_at"]),
                            changes=[SchemaChange(**c) for c in v["changes"]],
                            description=v.get("description", ""),
                        )
                        for v in versions
                    ]

    def _save_registry(self) -> None:
        """Save schema registry to disk."""
        data = {}
        for table, versions in self.schemas.items():
            data[table] = [
                {
                    "version": v.version,
                    "created_at": v.created_at.isoformat(),
                    "changes": [
                        {
                            "change_type": c.change_type.value,
                            "column_name": c.column_name,
                            "old_value": c.old_value,
                            "new_value": c.new_value,
                            "is_breaking": c.is_breaking,
                        }
                        for c in v.changes
                    ],
                    "description": v.description,
                }
                for v in versions
            ]
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_schema(
        self,
        table_name: str,
        df: pd.DataFrame,
        description: str = "",
    ) -> SchemaVersion:
        """Register a new schema version.

        Args:
            table_name: Table name
            df: DataFrame with new schema
            description: Change description

        Returns:
            New SchemaVersion
        """
        current_version = self.get_latest_version(table_name)
        new_version_num = current_version + 1 if current_version else 1

        # Detect changes from previous version
        changes = []
        if current_version:
            changes = self._detect_changes(table_name, df)

        version = SchemaVersion(
            version=new_version_num,
            created_at=datetime.utcnow(),
            changes=changes,
            description=description,
        )

        if table_name not in self.schemas:
            self.schemas[table_name] = []
        self.schemas[table_name].append(version)
        self._save_registry()

        logger.info(f"Registered schema v{new_version_num} for {table_name}")
        return version

    def get_latest_version(self, table_name: str) -> int | None:
        """Get latest schema version for table."""
        versions = self.schemas.get(table_name, [])
        return versions[-1].version if versions else None

    def _detect_changes(
        self,
        table_name: str,
        new_df: pd.DataFrame,
    ) -> list[SchemaChange]:
        """Detect schema changes from current version.

        Args:
            table_name: Table name
            new_df: DataFrame with new schema

        Returns:
            List of detected changes
        """
        changes = []

        # Get current schema
        current_schema = self._get_current_schema(table_name)
        new_schema = {col: str(dtype) for col, dtype in new_df.dtypes.items()}

        # Detect added columns
        for col in set(new_schema.keys()) - set(current_schema.keys()):
            changes.append(SchemaChange(
                change_type=SchemaChangeType.ADD_COLUMN,
                column_name=col,
                new_value=new_schema[col],
                is_breaking=False,
            ))

        # Detect dropped columns
        for col in set(current_schema.keys()) - set(new_schema.keys()):
            changes.append(SchemaChange(
                change_type=SchemaChangeType.DROP_COLUMN,
                column_name=col,
                old_value=current_schema[col],
                is_breaking=True,
            ))

        # Detect type changes
        for col in set(current_schema.keys()) & set(new_schema.keys()):
            if current_schema[col] != new_schema[col]:
                changes.append(SchemaChange(
                    change_type=SchemaChangeType.CHANGE_TYPE,
                    column_name=col,
                    old_value=current_schema[col],
                    new_value=new_schema[col],
                    is_breaking=True,  # Type changes can be breaking
                ))

        return changes

    def _get_current_schema(self, table_name: str) -> dict[str, str]:
        """Get current schema for table."""
        # This would query your data store to get current schema
        # Placeholder implementation
        return {}

    def check_compatibility(
        self,
        table_name: str,
        new_df: pd.DataFrame,
        mode: str = "backward",
    ) -> tuple[bool, list[str]]:
        """Check if new schema is compatible.

        Args:
            table_name: Table name
            new_df: DataFrame with new schema
            mode: Compatibility mode (backward, forward, full)

        Returns:
            (is_compatible, list_of_issues)
        """
        changes = self._detect_changes(table_name, new_df)
        issues = []

        for change in changes:
            if change.is_breaking:
                if mode in ("backward", "full") and change.change_type == SchemaChangeType.DROP_COLUMN:
                    issues.append(f"Breaking change: dropped column '{change.column_name}'")
                if mode in ("backward", "full") and change.change_type == SchemaChangeType.CHANGE_TYPE:
                    issues.append(
                        f"Breaking change: type changed for '{change.column_name}' "
                        f"from {change.old_value} to {change.new_value}"
                    )

        return len(issues) == 0, issues


class SchemaMigrator:
    """Migrate data between schema versions."""

    def __init__(self, registry: SchemaRegistry):
        self.registry = registry

    def migrate(
        self,
        df: pd.DataFrame,
        from_version: int,
        to_version: int,
        table_name: str,
    ) -> pd.DataFrame:
        """Migrate data from one schema version to another.

        Args:
            df: Source DataFrame
            from_version: Source schema version
            to_version: Target schema version
            table_name: Table name

        Returns:
            Migrated DataFrame
        """
        if from_version == to_version:
            return df

        versions = self.registry.schemas.get(table_name, [])
        version_range = [v for v in versions if from_version < v.version <= to_version]

        result = df.copy()
        for version in version_range:
            for change in version.changes:
                result = self._apply_change(result, change)

        return result

    def _apply_change(self, df: pd.DataFrame, change: SchemaChange) -> pd.DataFrame:
        """Apply a single schema change to DataFrame."""
        match change.change_type:
            case SchemaChangeType.ADD_COLUMN:
                # Add column with default value
                df[change.column_name] = None

            case SchemaChangeType.DROP_COLUMN:
                if change.column_name in df.columns:
                    df = df.drop(columns=[change.column_name])

            case SchemaChangeType.RENAME_COLUMN:
                if change.old_value in df.columns:
                    df = df.rename(columns={change.old_value: change.new_value})

            case SchemaChangeType.CHANGE_TYPE:
                if change.column_name in df.columns:
                    try:
                        df[change.column_name] = df[change.column_name].astype(change.new_value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not convert {change.column_name}: {e}")

        return df
```

---

## Data Partitioning Strategies

### Beyond Time-Based Partitioning

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class PartitionStrategy(str, Enum):
    """Partitioning strategies."""

    TIME_BASED = "time"           # Partition by time period
    HASH_BASED = "hash"           # Partition by hash of key
    RANGE_BASED = "range"         # Partition by value range
    LIST_BASED = "list"           # Partition by specific values
    COMPOSITE = "composite"       # Multiple partition keys


@dataclass
class PartitionSpec:
    """Specification for a partition."""

    column: str
    strategy: PartitionStrategy
    params: dict[str, Any] = field(default_factory=dict)


class Partitioner(ABC):
    """Base class for data partitioners."""

    @abstractmethod
    def get_partition_key(self, row: dict) -> str:
        """Get partition key for a row."""
        pass

    @abstractmethod
    def get_partition_path(self, partition_key: str) -> Path:
        """Get file path for a partition."""
        pass


class TimePartitioner(Partitioner):
    """Partition by time.

    Granularities:
    - year: /year=2024/
    - month: /year=2024/month=01/
    - day: /year=2024/month=01/day=15/
    - hour: /year=2024/month=01/day=15/hour=14/
    """

    def __init__(
        self,
        base_path: Path,
        timestamp_col: str = "timestamp",
        granularity: str = "day",
    ):
        self.base_path = base_path
        self.timestamp_col = timestamp_col
        self.granularity = granularity

    def get_partition_key(self, row: dict) -> str:
        """Get partition key from timestamp."""
        ts = pd.to_datetime(row[self.timestamp_col])

        parts = [f"year={ts.year}"]

        if self.granularity in ("month", "day", "hour"):
            parts.append(f"month={ts.month:02d}")

        if self.granularity in ("day", "hour"):
            parts.append(f"day={ts.day:02d}")

        if self.granularity == "hour":
            parts.append(f"hour={ts.hour:02d}")

        return "/".join(parts)

    def get_partition_path(self, partition_key: str) -> Path:
        """Get file path for partition."""
        return self.base_path / partition_key


class HashPartitioner(Partitioner):
    """Partition by hash of a column value.

    Use cases:
    - Distribute data evenly across partitions
    - Enable parallel processing
    - Avoid hotspots from skewed data
    """

    def __init__(
        self,
        base_path: Path,
        key_col: str,
        num_partitions: int = 16,
    ):
        self.base_path = base_path
        self.key_col = key_col
        self.num_partitions = num_partitions

    def get_partition_key(self, row: dict) -> str:
        """Get partition key using hash."""
        key_value = str(row[self.key_col])
        partition_num = hash(key_value) % self.num_partitions
        return f"partition={partition_num:04d}"

    def get_partition_path(self, partition_key: str) -> Path:
        """Get file path for partition."""
        return self.base_path / partition_key


class RangePartitioner(Partitioner):
    """Partition by value ranges.

    Use cases:
    - Price ranges (0-100, 100-1000, 1000+)
    - Time ranges (trading hours, after hours)
    - Volume buckets
    """

    def __init__(
        self,
        base_path: Path,
        value_col: str,
        boundaries: list[float],
    ):
        self.base_path = base_path
        self.value_col = value_col
        self.boundaries = sorted(boundaries)

    def get_partition_key(self, row: dict) -> str:
        """Get partition key based on value range."""
        value = float(row[self.value_col])

        for i, boundary in enumerate(self.boundaries):
            if value < boundary:
                if i == 0:
                    return f"{self.value_col}_lt_{int(boundary)}"
                else:
                    return f"{self.value_col}_{int(self.boundaries[i-1])}_to_{int(boundary)}"

        return f"{self.value_col}_gte_{int(self.boundaries[-1])}"

    def get_partition_path(self, partition_key: str) -> Path:
        """Get file path for partition."""
        return self.base_path / partition_key


class CompositePartitioner(Partitioner):
    """Partition by multiple columns.

    Example: /symbol=BTCUSDT/year=2024/month=01/
    """

    def __init__(
        self,
        base_path: Path,
        partition_specs: list[PartitionSpec],
    ):
        self.base_path = base_path
        self.partition_specs = partition_specs

    def get_partition_key(self, row: dict) -> str:
        """Get composite partition key."""
        parts = []

        for spec in self.partition_specs:
            if spec.strategy == PartitionStrategy.LIST_BASED:
                parts.append(f"{spec.column}={row[spec.column]}")

            elif spec.strategy == PartitionStrategy.TIME_BASED:
                ts = pd.to_datetime(row[spec.column])
                granularity = spec.params.get("granularity", "day")

                if granularity in ("year", "month", "day"):
                    parts.append(f"year={ts.year}")
                if granularity in ("month", "day"):
                    parts.append(f"month={ts.month:02d}")
                if granularity == "day":
                    parts.append(f"day={ts.day:02d}")

        return "/".join(parts)

    def get_partition_path(self, partition_key: str) -> Path:
        """Get file path for partition."""
        return self.base_path / partition_key


class PartitionedWriter:
    """Write data to partitioned storage."""

    def __init__(self, partitioner: Partitioner):
        self.partitioner = partitioner

    def write(
        self,
        df: pd.DataFrame,
        file_format: str = "parquet",
    ) -> dict[str, int]:
        """Write DataFrame to partitioned files.

        Args:
            df: DataFrame to write
            file_format: Output format (parquet, csv)

        Returns:
            Dictionary of partition -> row count
        """
        # Group by partition
        partitions: dict[str, list[int]] = {}
        for idx, row in df.iterrows():
            partition_key = self.partitioner.get_partition_key(row.to_dict())
            if partition_key not in partitions:
                partitions[partition_key] = []
            partitions[partition_key].append(idx)

        # Write each partition
        stats = {}
        for partition_key, indices in partitions.items():
            partition_df = df.loc[indices]
            partition_path = self.partitioner.get_partition_path(partition_key)
            partition_path.mkdir(parents=True, exist_ok=True)

            file_name = f"data.{file_format}"
            file_path = partition_path / file_name

            if file_format == "parquet":
                partition_df.to_parquet(file_path, index=False)
            else:
                partition_df.to_csv(file_path, index=False)

            stats[partition_key] = len(partition_df)
            logger.debug(f"Wrote {len(partition_df)} rows to {file_path}")

        return stats


class PartitionPruner:
    """Prune partitions based on query predicates."""

    def __init__(self, partitioner: Partitioner, base_path: Path):
        self.partitioner = partitioner
        self.base_path = base_path

    def get_relevant_partitions(
        self,
        filters: dict[str, Any],
    ) -> list[Path]:
        """Get partitions relevant to query filters.

        Args:
            filters: Query filters (e.g., {"symbol": "BTCUSDT", "year": 2024})

        Returns:
            List of partition paths to read
        """
        # Get all existing partitions
        all_partitions = list(self.base_path.glob("**/*.parquet"))

        # Filter based on predicates
        relevant = []
        for partition_path in all_partitions:
            # Extract partition values from path
            partition_values = self._extract_partition_values(partition_path)

            # Check if partition matches all filters
            matches = all(
                partition_values.get(key) == str(value)
                for key, value in filters.items()
                if key in partition_values
            )

            if matches:
                relevant.append(partition_path)

        logger.info(f"Pruned {len(all_partitions)} partitions to {len(relevant)}")
        return relevant

    def _extract_partition_values(self, path: Path) -> dict[str, str]:
        """Extract partition values from path."""
        values = {}
        for part in path.parts:
            if "=" in part:
                key, value = part.split("=", 1)
                values[key] = value
        return values


# Example usage
def create_ohlcv_partitioner(base_path: Path) -> CompositePartitioner:
    """Create partitioner for OHLCV data."""
    return CompositePartitioner(
        base_path=base_path,
        partition_specs=[
            PartitionSpec(
                column="symbol",
                strategy=PartitionStrategy.LIST_BASED,
            ),
            PartitionSpec(
                column="timestamp",
                strategy=PartitionStrategy.TIME_BASED,
                params={"granularity": "month"},
            ),
        ],
    )
```

---

*Last Updated: 2025-12-03*
