# Module 05: Database & Data Pipeline

**Duration:** 3-4 hours | **Difficulty:** Intermediate | **Prerequisites:** Modules 01-04

## 🎯 Learning Objectives

After this module, you will:
- Set up and use PostgreSQL database
- Understand async database operations with asyncpg
- Fetch data from Binance API
- Validate and clean OHLCV data
- Understand the complete data pipeline

---

## PostgreSQL Setup

### Schema Structure

```sql
-- Schema for ML data
CREATE SCHEMA IF NOT EXISTS ml_data;

-- OHLCV table (candlestick data)
CREATE TABLE ml_data.ohlcv (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC(20, 8) NOT NULL,
    high NUMERIC(20, 8) NOT NULL,
    low NUMERIC(20, 8) NOT NULL,
    close NUMERIC(20, 8) NOT NULL,
    volume NUMERIC(30, 8) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timeframe, timestamp)  -- Prevents duplicates
);

-- Index for fast lookups
CREATE INDEX idx_ohlcv_lookup
    ON ml_data.ohlcv(symbol, timeframe, timestamp);
```

**Node.js MongoDB equivalent:**
```javascript
// MongoDB schema
{
  symbol: String,
  timeframe: String,
  timestamp: Date,
  open: Number,
  high: Number,
  low: Number,
  close: Number,
  volume: Number
}
```

---

## Async Database Operations

### Connection Pool (Like MongoDB Connection)

```python
# src/price_stradamus/data/database.py
import asyncpg
from price_stradamus.config import settings

class DatabaseManager:
    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Create connection pool."""
        self.pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=10,
            max_size=20,
        )

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
```

**Node.js equivalent:**
```typescript
// MongoDB connection
const client = new MongoClient(uri, {
  minPoolSize: 10,
  maxPoolSize: 20,
});
await client.connect();
```

### Saving Data

```python
async def save_ohlcv(
    self,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> int:
    """Save OHLCV data with automatic deduplication."""
    if self.pool is None:
        raise RuntimeError("Database not initialized")

    inserted = 0
    async with self.pool.acquire() as conn:
        async with conn.transaction():
            for _, row in df.iterrows():
                result = await conn.execute(
                    """
                    INSERT INTO ml_data.ohlcv (
                        symbol, timeframe, timestamp,
                        open, high, low, close, volume
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (symbol, timeframe, timestamp)
                    DO NOTHING
                    RETURNING id
                    """,
                    symbol, timeframe, row['timestamp'],
                    row['open'], row['high'], row['low'],
                    row['close'], row['volume']
                )
                if result == "INSERT 0 1":
                    inserted += 1

    return inserted
```

**Key points:**
- `ON CONFLICT DO NOTHING` prevents duplicates
- Transaction ensures all-or-nothing
- Returns count of actually inserted rows

### Loading Data

```python
async def get_ohlcv(
    self,
    symbol: str,
    timeframe: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Retrieve OHLCV data."""
    if self.pool is None:
        raise RuntimeError("Database not initialized")

    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM ml_data.ohlcv
        WHERE symbol = $1 AND timeframe = $2
        ORDER BY timestamp ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    async with self.pool.acquire() as conn:
        rows = await conn.fetch(query, symbol, timeframe)

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df
```

---

## Binance API Integration

### Fetching Historical Data

```python
# src/price_stradamus/data/fetcher.py
import aiohttp
from datetime import datetime, timedelta

class BinanceDataFetcher:
    def __init__(self):
        self.base_url = "https://api.binance.com"

    async def fetch_historical_range(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for date range.

        Binance limits: 1000 candles per request
        """
        # Calculate batches
        batches = self._calculate_batches(start_date, end_date, interval)

        # Fetch all batches in parallel
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_batch(session, symbol, interval, start, end)
                for start, end in batches
            ]
            results = await asyncio.gather(*tasks)

        # Combine all batches
        df = pd.concat(results, ignore_index=True)
        return df

    async def _fetch_batch(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
    ) -> pd.DataFrame:
        """Fetch one batch (max 1000 candles)."""
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000,
        }

        async with session.get(url, params=params) as response:
            data = await response.json()

        # Parse response
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Keep only needed columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        return df
```

**API Response Example:**
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
    ...
  ]
]
```

---

## Data Validation

### Preprocessing Pipeline

```python
# src/price_stradamus/data/preprocessor.py

class DataPreprocessor:
    def validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV data quality."""
        # 1. Check required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # 2. Check for negative values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        if (df[numeric_cols] < 0).any().any():
            raise ValueError("Negative values found")

        # 3. Validate OHLC relationships
        if not (df['high'] >= df['open']).all():
            raise ValueError("High must be >= open")
        if not (df['high'] >= df['close']).all():
            raise ValueError("High must be >= close")
        if not (df['low'] <= df['open']).all():
            raise ValueError("Low must be <= open")
        if not (df['low'] <= df['close']).all():
            raise ValueError("Low must be <= close")

        # 4. Check for NaN/Inf
        if df[numeric_cols].isnull().any().any():
            raise ValueError("NaN values found")
        if np.isinf(df[numeric_cols]).any().any():
            raise ValueError("Infinite values found")

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing candles."""
        # Detect missing timestamps
        df = df.sort_values('timestamp')
        df = df.set_index('timestamp')

        # Forward fill small gaps (<10 candles)
        df = df.fillna(method='ffill', limit=10)

        # Interpolate larger gaps
        df = df.interpolate(method='linear')

        return df.reset_index()
```

---

## Complete Data Pipeline

### End-to-End Flow

```python
async def fetch_and_store_data(
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    days: int = 30,
) -> dict:
    """Complete pipeline: Fetch → Validate → Store."""

    # 1. Initialize
    fetcher = BinanceDataFetcher()
    preprocessor = DataPreprocessor()
    db = DatabaseManager()
    await db.initialize()

    try:
        # 2. Fetch from Binance
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()

        logger.info(f"Fetching {symbol} {timeframe} data for {days} days")
        df = await fetcher.fetch_historical_range(
            symbol, timeframe, start_date, end_date
        )

        # 3. Validate
        logger.info("Validating data")
        df = preprocessor.validate_ohlcv(df)
        df = preprocessor.handle_missing_values(df)

        # 4. Store in database
        logger.info("Saving to database")
        inserted = await db.save_ohlcv(df, symbol, timeframe)

        return {
            "success": True,
            "total_candles": len(df),
            "inserted": inserted,
            "duplicates": len(df) - inserted,
            "date_range": (df['timestamp'].min(), df['timestamp'].max()),
        }

    finally:
        await db.close()
```

**Usage:**
```python
result = asyncio.run(fetch_and_store_data("BTCUSDT", "1m", days=30))
print(f"Inserted {result['inserted']} new candles")
```

---

## Quick Reference

### Database Operations

```python
# Initialize
db = DatabaseManager()
await db.initialize()

# Save data
inserted = await db.save_ohlcv(df, "BTCUSDT", "1m")

# Load data
df = await db.get_ohlcv("BTCUSDT", "1m", limit=1000)

# Close
await db.close()
```

### Binance API

```python
# Fetch historical data
fetcher = BinanceDataFetcher()
df = await fetcher.fetch_historical_range(
    symbol="BTCUSDT",
    interval="1m",
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now(),
)
```

### Data Validation

```python
# Validate
preprocessor = DataPreprocessor()
df = preprocessor.validate_ohlcv(df)
df = preprocessor.handle_missing_values(df)
```

---

## Practice Exercises

### Exercise 1: Fetch & Store (30 minutes)

Write a script to fetch 7 days of Bitcoin data and store it:

```python
import asyncio
from datetime import datetime, timedelta

async def main():
    # Your code here
    pass

asyncio.run(main())
```

<details>
<summary>Solution</summary>

```python
from price_stradamus.data.fetcher import BinanceDataFetcher
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.preprocessor import DataPreprocessor

async def main():
    fetcher = BinanceDataFetcher()
    preprocessor = DataPreprocessor()
    db = DatabaseManager()

    await db.initialize()

    try:
        # Fetch
        df = await fetcher.fetch_historical_range(
            "BTCUSDT", "1m",
            datetime.now() - timedelta(days=7),
            datetime.now()
        )

        # Validate
        df = preprocessor.validate_ohlcv(df)

        # Store
        inserted = await db.save_ohlcv(df, "BTCUSDT", "1m")
        print(f"Inserted {inserted} candles")

    finally:
        await db.close()
```
</details>

---

## Next Steps

**Next Module:** [06: Feature Engineering →](06-feature-engineering.md)

Learn how to generate technical indicators from OHLCV data.

---

## Summary Checklist

- [ ] I understand PostgreSQL schema
- [ ] I can use async database operations
- [ ] I can fetch data from Binance API
- [ ] I know how to validate OHLCV data
- [ ] I understand the complete data pipeline

**Continue to:** [Module 06: Feature Engineering](06-feature-engineering.md)
