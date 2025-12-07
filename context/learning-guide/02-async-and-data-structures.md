# Module 02: Async & Data Structures

**Duration:** 3-4 hours | **Difficulty:** Beginner-Intermediate | **Prerequisites:** Module 01

## 🎯 Learning Objectives

After this module, you will:
- Use async/await in Python (similar to Node.js)
- Work with pandas DataFrames (structured data tables)
- Understand numpy arrays (mathematical operations)
- Handle time series data
- Understand asyncio patterns

---

## 📋 Table of Contents

1. [Async/Await in Python](#asyncawait-in-python)
2. [Introduction to Pandas](#introduction-to-pandas)
3. [Introduction to Numpy](#introduction-to-numpy)
4. [Working with Time Series](#working-with-time-series)
5. [File I/O](#file-io)
6. [Quick Reference](#quick-reference)
7. [Practice Exercises](#practice-exercises)

---

## Async/Await in Python

### The Good News

**If you know async/await in Node.js, you already know 80% of Python async!**

### Basic Async Function

#### Node.js/TypeScript
```typescript
async function fetchData(url: string): Promise<string> {
  const response = await fetch(url);
  return await response.text();
}

const data = await fetchData("https://api.example.com");
```

#### Python
```python
import aiohttp

async def fetch_data(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

data = await fetch_data("https://api.example.com")
```

**Key Differences:**
1. Python uses `async with` for context managers (like try/finally in JS)
2. Uses `aiohttp` instead of `fetch` (need to install it)
3. Otherwise, almost identical!

### Running Async Code

#### Node.js/TypeScript
```typescript
// Top-level await (Node 14+)
await someAsyncFunction();

// Or with async IIFE
(async () => {
  await someAsyncFunction();
})();
```

#### Python
```python
import asyncio

# In scripts
asyncio.run(some_async_function())

# In async context (already running)
await some_async_function()
```

**Real Example from Price Stradamus:**
```python
# From src/price_stradamus/data/fetcher.py
async def fetch_historical_range(
    self,
    symbol: str,
    interval: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Fetch historical OHLCV data from Binance."""
    # Calculate batches
    batches = self._calculate_batches(start_date, end_date)

    # Fetch in parallel
    tasks = [self._fetch_batch(symbol, interval, start, end) for start, end in batches]
    results = await asyncio.gather(*tasks)

    # Combine results
    return pd.concat(results, ignore_index=True)
```

### Parallel Execution

#### Node.js/TypeScript
```typescript
// Run in parallel
const [users, posts, comments] = await Promise.all([
  fetchUsers(),
  fetchPosts(),
  fetchComments(),
]);
```

#### Python
```python
# Run in parallel
users, posts, comments = await asyncio.gather(
    fetch_users(),
    fetch_posts(),
    fetch_comments(),
)
```

### Context Managers (async with)

Python has a powerful pattern for resource management:

```python
# Automatically closes the connection when done
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.text()
# Connection automatically closed here, even if exception occurs
```

**Node.js equivalent (manual):**
```typescript
const session = createSession();
try {
  const response = await session.get(url);
  const data = await response.text();
} finally {
  session.close(); // Must manually close
}
```

### Database Operations

**Real example from Price Stradamus:**
```python
# From src/price_stradamus/data/database.py
import asyncpg

async def save_ohlcv(
    self,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> int:
    """Save OHLCV data to PostgreSQL."""
    async with self.pool.acquire() as conn:
        async with conn.transaction():
            for _, row in df.iterrows():
                await conn.execute(
                    """
                    INSERT INTO ml_data.ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                    """,
                    symbol, timeframe, row['timestamp'],
                    row['open'], row['high'], row['low'], row['close'], row['volume']
                )
```

**Node.js equivalent:**
```typescript
async function saveOHLCV(df: DataFrame, symbol: string, timeframe: string): Promise<number> {
  const connection = await pool.acquire();
  try {
    await connection.query('BEGIN');
    for (const row of df.rows) {
      await connection.query(
        'INSERT INTO ml_data.ohlcv (...) VALUES ($1, $2, ...) ON CONFLICT DO NOTHING',
        [symbol, timeframe, row.timestamp, ...]
      );
    }
    await connection.query('COMMIT');
  } catch (error) {
    await connection.query('ROLLBACK');
    throw error;
  } finally {
    connection.release();
  }
}
```

---

## Introduction to Pandas

**Think of pandas as:** Excel + SQL + Array operations in one library

### What is a DataFrame?

A DataFrame is a 2D table with labeled rows and columns.

```python
import pandas as pd

# Create a DataFrame (like an array of objects in JS)
data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

print(data)
```

Output:
```
      name  age  salary
0    Alice   25   50000
1      Bob   30   60000
2  Charlie   35   70000
```

**TypeScript equivalent concept:**
```typescript
const data = [
  { name: 'Alice', age: 25, salary: 50000 },
  { name: 'Bob', age: 30, salary: 60000 },
  { name: 'Charlie', age: 35, salary: 70000 },
];
```

### Basic Operations

```python
# Select a column (returns Series)
ages = data['age']

# Select multiple columns
subset = data[['name', 'salary']]

# Filter rows (boolean indexing)
high_earners = data[data['salary'] > 55000]

# Add a new column
data['bonus'] = data['salary'] * 0.1

# Sort
sorted_data = data.sort_values('age', ascending=False)
```

**TypeScript equivalent:**
```typescript
// Select column
const ages = data.map(d => d.age);

// Select multiple columns
const subset = data.map(d => ({ name: d.name, salary: d.salary }));

// Filter
const highEarners = data.filter(d => d.salary > 55000);

// Add column
const withBonus = data.map(d => ({ ...d, bonus: d.salary * 0.1 }));

// Sort
const sorted = [...data].sort((a, b) => b.age - a.age);
```

### Reading and Writing Files

```python
# Read CSV
df = pd.read_csv('data.csv')

# Read from database
df = pd.read_sql('SELECT * FROM users', connection)

# Write CSV
df.to_csv('output.csv', index=False)

# Write to database
df.to_sql('users', connection, if_exists='append')
```

### Aggregations

```python
# Calculate statistics
mean_salary = data['salary'].mean()
total_salary = data['salary'].sum()
max_age = data['age'].max()

# Group by and aggregate
grouped = data.groupby('department')['salary'].mean()
```

**Real Example from Price Stradamus:**
```python
# From src/price_stradamus/data/preprocessor.py
def validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLCV data quality."""
    # Check for required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check for negative values
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    if (df[numeric_cols] < 0).any().any():
        raise ValueError("Negative values found in OHLC data")

    # Validate OHLC relationships
    if not (df['high'] >= df['open']).all():
        raise ValueError("High price must be >= open price")

    return df
```

### Working with Time Series in Pandas

```python
# Parse datetime column
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set datetime as index
df = df.set_index('timestamp')

# Resample (like grouping by time buckets)
hourly_data = df.resample('1H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# Calculate rolling window (moving average)
df['sma_20'] = df['close'].rolling(window=20).mean()
```

---

## Introduction to Numpy

**Think of numpy as:** Efficient arrays for mathematical operations

### Why Numpy?

JavaScript arrays are slow for math. Numpy arrays are FAST.

```python
import numpy as np

# Create array
arr = np.array([1, 2, 3, 4, 5])

# Vectorized operations (fast!)
doubled = arr * 2            # [2, 4, 6, 8, 10]
squared = arr ** 2           # [1, 4, 9, 16, 25]
plus_ten = arr + 10          # [11, 12, 13, 14, 15]
```

**TypeScript equivalent (slower):**
```typescript
const arr = [1, 2, 3, 4, 5];
const doubled = arr.map(x => x * 2);
const squared = arr.map(x => x ** 2);
const plusTen = arr.map(x => x + 10);
```

### Array Creation

```python
# From list
arr = np.array([1, 2, 3])

# Range (like Array.from in JS)
arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Zeros/Ones
zeros = np.zeros(5)        # [0, 0, 0, 0, 0]
ones = np.ones(5)          # [1, 1, 1, 1, 1]

# Linspace (evenly spaced values)
arr = np.linspace(0, 1, 5) # [0.0, 0.25, 0.5, 0.75, 1.0]
```

### Array Operations

```python
# Element-wise operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = a + b       # [5, 7, 9]
c = a * b       # [4, 10, 18]
c = a ** 2      # [1, 4, 9]

# Aggregations
mean = a.mean()
std = a.std()
sum = a.sum()
max = a.max()
```

### Boolean Indexing

```python
arr = np.array([1, 2, 3, 4, 5])

# Filter (like Array.filter)
greater_than_3 = arr[arr > 3]  # [4, 5]

# Multiple conditions
between = arr[(arr > 2) & (arr < 5)]  # [3, 4]
```

**Real Example from Price Stradamus:**
```python
# From src/price_stradamus/evaluation/metrics.py
def calculate_all(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate all metrics."""
    # MAE
    mae = np.mean(np.abs(actual - predicted))

    # RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # MAPE
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    # Directional accuracy
    predicted_direction = np.sign(np.diff(predicted))
    actual_direction = np.sign(np.diff(actual))
    directional_accuracy = np.mean(predicted_direction == actual_direction) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
    }
```

---

## Working with Time Series

### Time Series in Price Stradamus

The project uses **Darts TimeSeries** objects (built on pandas):

```python
from darts import TimeSeries
import pandas as pd

# Create DataFrame with datetime index
df = pd.DataFrame({
    'close': [100, 102, 101, 103, 105],
}, index=pd.date_range('2024-01-01', periods=5, freq='1D'))

# Convert to Darts TimeSeries
ts = TimeSeries.from_dataframe(df)

# Split time series
train = ts[:70]   # First 70%
test = ts[70:]    # Last 30%

# Slice by date
recent = ts['2024-01-03':]
```

**Real Example from Price Stradamus:**
```python
# From src/price_stradamus/data/features.py
def to_darts_timeseries(
    self,
    df: pd.DataFrame,
    value_cols: list[str],
) -> TimeSeries:
    """Convert DataFrame to Darts TimeSeries.

    Args:
        df: DataFrame with datetime index
        value_cols: Columns to include in TimeSeries

    Returns:
        TimeSeries object
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('timestamp')

    # Select value columns
    df_values = df[value_cols]

    # Create TimeSeries
    ts = TimeSeries.from_dataframe(df_values)

    return ts
```

---

## File I/O

### Reading Files

```python
# Read text file
with open('data.txt', 'r') as f:
    content = f.read()
    lines = f.readlines()

# Read JSON
import json

with open('config.json', 'r') as f:
    data = json.load(f)

# Read CSV with pandas
df = pd.read_csv('data.csv')

# Read pickle (serialized Python objects)
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Writing Files

```python
# Write text file
with open('output.txt', 'w') as f:
    f.write('Hello, world!')

# Write JSON
with open('output.json', 'w') as f:
    json.dump({'key': 'value'}, f, indent=2)

# Write CSV with pandas
df.to_csv('output.csv', index=False)

# Write pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

**Real Example from Price Stradamus:**
```python
# From src/price_stradamus/models/neural/nbeats.py
def save(self, path: Path) -> None:
    """Save model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save Darts model (uses pickle internally)
    self.model.save(str(path))

    # Save metadata
    metadata = {
        'input_chunk_length': self.input_chunk_length,
        'output_chunk_length': self.output_chunk_length,
        'num_stacks': self.num_stacks,
    }

    with open(path.with_suffix('.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def load(self, path: Path) -> None:
    """Load model from disk."""
    # Load Darts model
    self.model = DartsNBEATS.load(str(path))

    # Load metadata
    with open(path.with_suffix('.json'), 'r') as f:
        metadata = json.load(f)
        self.input_chunk_length = metadata['input_chunk_length']
        self.output_chunk_length = metadata['output_chunk_length']
```

---

## Quick Reference

### Async/Await

```python
import asyncio
import aiohttp

# Define async function
async def fetch(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Run async function
result = asyncio.run(fetch('https://api.example.com'))

# Parallel execution
results = await asyncio.gather(
    fetch('url1'),
    fetch('url2'),
    fetch('url3'),
)
```

### Pandas Cheat Sheet

```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

# Read/Write
df = pd.read_csv('data.csv')
df.to_csv('output.csv', index=False)

# Select
df['col1']              # Single column
df[['col1', 'col2']]    # Multiple columns
df[df['col1'] > 1]      # Filter rows

# Operations
df['new_col'] = df['col1'] * 2
df.sort_values('col1')
df.groupby('col1')['col2'].mean()

# Time series
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df.resample('1H').mean()
```

### Numpy Cheat Sheet

```python
import numpy as np

# Create arrays
arr = np.array([1, 2, 3])
arr = np.arange(0, 10, 2)
arr = np.zeros(5)
arr = np.linspace(0, 1, 5)

# Operations
arr * 2
arr ** 2
arr + 10

# Aggregations
arr.mean()
arr.std()
arr.sum()
arr.max()

# Boolean indexing
arr[arr > 3]
arr[(arr > 2) & (arr < 5)]
```

---

## Practice Exercises

### Exercise 1: Async Fetching (20 minutes)

Write an async function to fetch Bitcoin price data from multiple exchanges in parallel:

```python
import asyncio
import aiohttp

async def fetch_price(session: aiohttp.ClientSession, exchange: str) -> dict:
    """Fetch Bitcoin price from an exchange API."""
    # Implement this
    pass

async def fetch_all_prices() -> list[dict]:
    """Fetch prices from multiple exchanges in parallel."""
    exchanges = ['binance', 'coinbase', 'kraken']
    # Implement parallel fetching
    pass

# Run
prices = asyncio.run(fetch_all_prices())
print(prices)
```

<details>
<summary>Solution</summary>

```python
import asyncio
import aiohttp

async def fetch_price(session: aiohttp.ClientSession, exchange: str) -> dict:
    """Fetch Bitcoin price from an exchange API."""
    urls = {
        'binance': 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',
        'coinbase': 'https://api.coinbase.com/v2/prices/BTC-USD/spot',
        'kraken': 'https://api.kraken.com/0/public/Ticker?pair=XBTUSD',
    }

    url = urls.get(exchange)
    if not url:
        return {'exchange': exchange, 'error': 'Unknown exchange'}

    try:
        async with session.get(url) as response:
            data = await response.json()
            return {'exchange': exchange, 'data': data}
    except Exception as e:
        return {'exchange': exchange, 'error': str(e)}

async def fetch_all_prices() -> list[dict]:
    """Fetch prices from multiple exchanges in parallel."""
    exchanges = ['binance', 'coinbase', 'kraken']

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_price(session, exchange) for exchange in exchanges]
        results = await asyncio.gather(*tasks)

    return results

# Run
prices = asyncio.run(fetch_all_prices())
for price in prices:
    print(f"{price['exchange']}: {price.get('data', price.get('error'))}")
```
</details>

---

### Exercise 2: Pandas DataFrames (30 minutes)

Load Bitcoin OHLCV data and calculate technical indicators:

```python
import pandas as pd

# Create sample data
data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
    'open': 50000 + (pd.Series(range(100)) * 10),
    'high': 50500 + (pd.Series(range(100)) * 10),
    'low': 49500 + (pd.Series(range(100)) * 10),
    'close': 50000 + (pd.Series(range(100)) * 10),
    'volume': 1000000,
})

# Tasks:
# 1. Set timestamp as index
# 2. Calculate 20-period SMA (Simple Moving Average)
# 3. Calculate daily returns (percentage change)
# 4. Filter rows where close > open (green candles)
# 5. Resample to daily data
```

<details>
<summary>Solution</summary>

```python
import pandas as pd

# Create sample data
data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
    'open': 50000 + (pd.Series(range(100)) * 10),
    'high': 50500 + (pd.Series(range(100)) * 10),
    'low': 49500 + (pd.Series(range(100)) * 10),
    'close': 50000 + (pd.Series(range(100)) * 10),
    'volume': 1000000,
})

# 1. Set timestamp as index
data = data.set_index('timestamp')

# 2. Calculate 20-period SMA
data['sma_20'] = data['close'].rolling(window=20).mean()

# 3. Calculate daily returns
data['returns'] = data['close'].pct_change() * 100

# 4. Filter green candles
green_candles = data[data['close'] > data['open']]
print(f"Green candles: {len(green_candles)}")

# 5. Resample to daily data
daily_data = data.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
})

print(daily_data.head())
```
</details>

---

### Exercise 3: Numpy Operations (20 minutes)

Calculate prediction metrics using numpy:

```python
import numpy as np

# Sample predictions vs actuals
actual = np.array([100, 102, 101, 103, 105, 104, 106, 108])
predicted = np.array([101, 103, 100, 104, 106, 103, 107, 109])

# Tasks:
# 1. Calculate MAE (Mean Absolute Error)
# 2. Calculate RMSE (Root Mean Squared Error)
# 3. Calculate directional accuracy (% correct up/down predictions)
```

<details>
<summary>Solution</summary>

```python
import numpy as np

actual = np.array([100, 102, 101, 103, 105, 104, 106, 108])
predicted = np.array([101, 103, 100, 104, 106, 103, 107, 109])

# 1. MAE
mae = np.mean(np.abs(actual - predicted))
print(f"MAE: {mae:.2f}")

# 2. RMSE
rmse = np.sqrt(np.mean((actual - predicted) ** 2))
print(f"RMSE: {rmse:.2f}")

# 3. Directional accuracy
actual_direction = np.sign(np.diff(actual))
predicted_direction = np.sign(np.diff(predicted))
directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
print(f"Directional Accuracy: {directional_accuracy:.1f}%")
```
</details>

---

## Next Steps

Great! You now understand async patterns and data manipulation in Python.

**Next Module:** [03: ML Fundamentals →](03-ml-fundamentals.md)

Learn the basics of machine learning and time series forecasting.

---

## Summary Checklist

- [ ] I understand async/await in Python
- [ ] I can use asyncio.gather for parallel operations
- [ ] I can create and manipulate pandas DataFrames
- [ ] I understand numpy arrays and vectorized operations
- [ ] I can work with time series data in pandas
- [ ] I know how to read/write files in Python

**Continue to:** [Module 03: ML Fundamentals](03-ml-fundamentals.md)
