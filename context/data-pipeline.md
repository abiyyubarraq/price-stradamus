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

---

## Advanced Topics

For advanced data pipeline topics, see [data-pipeline-advanced.md](data-pipeline-advanced.md):
- **Feature Store Architecture** (Phase 3-4)
- **Data Quality Gates** (Phase 3)
- **Incremental Processing** (Phase 3-4)
- **Stream Processing** (Phase 5)
- **Multi-Source Integration** (Phase 4-5)

---

*Last Updated: 2025-12-07*
*Version: 2.0 - Split into core (Phase 1) and advanced (Phase 3-5)*
