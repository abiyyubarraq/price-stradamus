# Data Pipeline - Advanced Topics (Phase 3-5)

---

## Overview

This guide contains **advanced data pipeline patterns** for production scale.

**For basic pipeline setup**, see [data-pipeline.md](data-pipeline.md).

### Contents

1. [Feature Store Architecture](#feature-store-architecture)
2. [Data Quality Gates](#data-quality-gates)
3. [Incremental Processing](#incremental-processing)
4. [Performance Optimization](#performance-optimization)

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
