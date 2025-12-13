# Module 04: Project Architecture

**Duration:** 2-3 hours | **Difficulty:** Intermediate | **Prerequisites:** Modules 01-03

## 🎯 Learning Objectives

After this module, you will:
- Understand the complete project structure
- Know where every file lives and why
- Understand the layer architecture
- Know where to add new code
- Trace data flow through the system

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Layer Architecture](#layer-architecture)
4. [Data Flow](#data-flow)
5. [Key Design Patterns](#key-design-patterns)
6. [Quick Reference](#quick-reference)
7. [Practice Exercises](#practice-exercises)

---

## Architecture Overview

### Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│                     ┌────────────────────┐                  │
│                     │    CLI LAYER       │                  │
│                     │ (User Interface)   │                  │
│                     └─────────┬──────────┘                  │
│                               │                              │
│                               ▼                              │
│                     ┌────────────────────┐                  │
│                     │   SERVICE LAYER    │                  │
│                     │ (Business Logic)   │                  │
│                     └────┬──────────┬────┘                  │
│                          │          │                        │
│             ┌────────────┘          └───────────┐           │
│             │                                   │           │
│             ▼                                   ▼           │
│   ┌───────────────────┐              ┌───────────────────┐ │
│   │   MODEL LAYER     │              │    DATA LAYER     │ │
│   │   (ML Models)     │─────────────▶│ (Database & API)  │ │
│   └───────────────────┘              └───────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Comparison to Node.js/Express

| Layer | Node.js/Express | Price Stradamus |
|-------|----------------|-----------------|
| **Routes** | Express routes | CLI commands (Typer) |
| **Controllers** | Controller functions | Service classes |
| **Models** | Database models | ML models + Data models |
| **Database** | MongoDB/PostgreSQL | PostgreSQL + asyncpg |

---

## Directory Structure

### Complete Structure

```
price-stradamus/
├── src/price_stradamus/           # Main package
│   ├── __init__.py
│   │
│   ├── config/                    # Configuration
│   │   ├── __init__.py
│   │   ├── settings.py            # Pydantic settings (like .env)
│   │   └── constants.py           # Global constants
│   │
│   ├── data/                      # Data pipeline
│   │   ├── __init__.py
│   │   ├── fetcher.py             # Binance API client
│   │   ├── database.py            # PostgreSQL operations
│   │   ├── preprocessor.py        # Data cleaning
│   │   └── features.py            # Feature engineering
│   │
│   ├── models/                    # ML models
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract base class
│   │   ├── registry.py            # Model factory
│   │   ├── exceptions.py          # Custom exceptions
│   │   │
│   │   ├── neural/                # Neural network models
│   │   │   ├── __init__.py
│   │   │   ├── nbeats.py          # N-BEATS
│   │   │   ├── lstm.py            # LSTM
│   │   │   ├── tcn.py             # TCN
│   │   │   └── tft.py             # TFT
│   │   │
│   │   ├── classical/             # Classical models
│   │   │   ├── __init__.py
│   │   │   ├── arima.py           # ARIMA
│   │   │   └── prophet.py         # Prophet
│   │   │
│   │   └── ml/                    # Machine learning models
│   │       ├── __init__.py
│   │       ├── xgboost.py         # XGBoost
│   │       └── .py   # 
│   │
│   ├── evaluation/                # Model evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py             # Metric calculations
│   │   └── backtester.py          # Walk-forward validation
│   │
│   ├── utils/                     # Utilities
│   │   ├── __init__.py
│   │   ├── logger.py              # Loguru setup
│   │   └── helpers.py             # Helper functions
│   │
│   └── cli/                       # CLI interface
│       ├── __init__.py
│       ├── data_commands.py       # Data commands (fetch)
│       ├── model_commands.py      # Model commands (train, predict)
│       └── info_commands.py       # Info commands (list-models)
│
├── tests/                         # Test suite
│   ├── test_data/
│   ├── test_models/
│   └── test_evaluation/
│
├── context/                       # Documentation
│   ├── learning-guide/            # This guide!
│   ├── architecture.md
│   ├── models.md
│   └── ...
│
├── docker/                        # Docker setup
│   └── docker-compose.yml         # PostgreSQL
│
├── pyproject.toml                 # Project config (like package.json)
├── .env                           # Environment variables
└── README.md
```

---

## Layer Architecture

### 1. CLI Layer (Entry Point)

**Purpose**: User interface via command line

**Node.js equivalent:**
```typescript
// Express routes
app.post('/api/train', trainController);
app.get('/api/predict', predictController);
```

**Price Stradamus:**
```python
# src/price_stradamus/cli/model_commands.py
import typer

app = typer.Typer()

@app.command()
def train(
    model: str = "nbeats",
    epochs: int = 100,
):
    """Train a model."""
    # Call service layer
    pass
```

**Key files:**
- `cli/__init__.py` - Main app setup
- `cli/data_commands.py` - fetch command
- `cli/model_commands.py` - train, predict, evaluate
- `cli/info_commands.py` - list-models, info

---

### 2. Service Layer (Business Logic)

**Purpose**: Core application logic

**Node.js equivalent:**
```typescript
// controllers/userController.ts
export class UserService {
  async getUsers() {
    const users = await db.users.find();
    return users;
  }
}
```

**Price Stradamus:**
```python
# src/price_stradamus/data/fetcher.py
class BinanceDataFetcher:
    async def fetch_historical_range(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance."""
        # Business logic here
        pass
```

**Key files:**
- `data/fetcher.py` - API fetching logic
- `data/preprocessor.py` - Data validation
- `data/features.py` - Feature generation
- `evaluation/backtester.py` - Backtesting logic

---

### 3. Model Layer

**Purpose**: Machine learning models

**Structure:**
```python
# All models inherit from BaseModel
class BaseModel(ABC):
    @abstractmethod
    def fit(self, train_series: TimeSeries) -> None:
        pass

    @abstractmethod
    def predict(self, n: int, series: TimeSeries) -> TimeSeries:
        pass
```

**Model Registry Pattern:**
```python
# Models self-register
@ModelRegistry.register("nbeats")
class NBEATSModel(BaseModel):
    def fit(self, train_series):
        # Training logic
        pass

# Usage
model = ModelRegistry.create_model("nbeats")
```

---

### 4. Data Layer

**Purpose**: Database and external APIs

**Node.js equivalent:**
```typescript
// models/User.ts
class User {
  static async findById(id: string) {
    return await db.collection('users').findOne({ _id: id });
  }
}
```

**Price Stradamus:**
```python
# src/price_stradamus/data/database.py
class DatabaseManager:
    async def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> int:
        """Save OHLCV data to PostgreSQL."""
        async with self.pool.acquire() as conn:
            # Database operations
            pass
```

---

## Data Flow

### Complete Flow: Fetch → Train → Predict

```
┌──────┐     ┌──────┐     ┌─────────┐    ┌───────┐    ┌────┐    ┌─────────┐
│ User │     │ CLI  │     │ Service │    │ Model │    │ DB │    │ Binance │
└───┬──┘     └───┬──┘     └────┬────┘    └───┬───┘    └─┬──┘    └────┬────┘
    │            │              │             │          │           │
    │ 1. FETCH DATA                                                  │
    │            │              │             │          │           │
    ├───────────▶│              │             │          │           │
    │ fetch      │              │             │          │           │
    │ --days 30  │              │             │          │           │
    │            │              │             │          │           │
    │            ├─────────────▶│             │          │           │
    │            │ BinanceData  │             │          │           │
    │            │ Fetcher.     │             │          │           │
    │            │ fetch()      │             │          │           │
    │            │              │             │          │           │
    │            │              ├────────────────────────────────────▶│
    │            │              │ GET /api/v3/klines                 │
    │            │              │                                     │
    │            │              │◀────────────────────────────────────┤
    │            │              │ OHLCV data                          │
    │            │              │             │          │           │
    │            │              │             │          │           │
    │            │              │ Validate &  │          │           │
    │            │              │ clean       │          │           │
    │            │              │             │          │           │
    │            │              ├────────────────────────▶│           │
    │            │              │ Save to PostgreSQL     │           │
    │            │              │                        │           │
    │            │◀─────────────┤◀───────────────────────┤           │
    │            │ Success      │                        │           │
    │            │              │             │          │           │
    │            │              │             │          │           │
    │ 2. TRAIN MODEL                                                 │
    │            │              │             │          │           │
    ├───────────▶│              │             │          │           │
    │ train      │              │             │          │           │
    │ --model    │              │             │          │           │
    │ nbeats     │              │             │          │           │
    │            │              │             │          │           │
    │            ├────────────────────────────────────────▶│           │
    │            │ Load OHLCV data                        │           │
    │            │                                        │           │
    │            │◀───────────────────────────────────────┤           │
    │            │ DataFrame                              │           │
    │            │              │             │          │           │
    │            ├─────────────▶│             │          │           │
    │            │ FeatureEng.  │             │          │           │
    │            │ generate()   │             │          │           │
    │            │              │             │          │           │
    │            │◀─────────────┤             │          │           │
    │            │ Features     │             │          │           │
    │            │ added        │             │          │           │
    │            │              │             │          │           │
    │            ├───────────────────────────▶│          │           │
    │            │ NBEATSModel.fit(data)     │          │           │
    │            │                            │          │           │
    │            │                            │ Train    │           │
    │            │                            │ neural   │           │
    │            │                            │ network  │           │
    │            │                            │          │           │
    │            │◀───────────────────────────┤          │           │
    │            │ Training complete          │          │           │
    │            │              │             │          │           │
    │            ├───────────────────────────▶│          │           │
    │            │ model.save()               │          │           │
    │            │              │             │          │           │
    │            │              │             │          │           │
    │ 3. MAKE PREDICTIONS                                            │
    │            │              │             │          │           │
    ├───────────▶│              │             │          │           │
    │ predict    │              │             │          │           │
    │ --steps 5  │              │             │          │           │
    │            │              │             │          │           │
    │            ├───────────────────────────▶│          │           │
    │            │ model.load()               │          │           │
    │            │              │             │          │           │
    │            ├───────────────────────────▶│          │           │
    │            │ model.predict(5)           │          │           │
    │            │              │             │          │           │
    │            │◀───────────────────────────┤          │           │
    │            │ Predictions                │          │           │
    │            │              │             │          │           │
    │◀───────────┤              │             │          │           │
    │ Display    │              │             │          │           │
    │ results    │              │             │          │           │
    │            │              │             │          │           │
```

### Step-by-Step Example

#### Step 1: Fetch Data
```python
# User runs: python -m price_stradamus.cli fetch --days 30

# 1. CLI receives command
# cli/data_commands.py
@app.command()
def fetch(days: int = 30):
    # 2. Call service layer
    fetcher = BinanceDataFetcher()
    data = asyncio.run(
        fetcher.fetch_historical_range(
            symbol="BTCUSDT",
            interval="1m",
            start_date=datetime.now() - timedelta(days=days),
            end_date=datetime.now(),
        )
    )

    # 3. Save to database
    db = DatabaseManager()
    asyncio.run(db.save_ohlcv(data, "BTCUSDT", "1m"))
```

#### Step 2: Train Model
```python
# User runs: python -m price_stradamus.cli train --model nbeats

# 1. Load data from database
db = DatabaseManager()
df = asyncio.run(db.get_ohlcv("BTCUSDT", "1m"))

# 2. Split raw data first (prevent data leakage)
train_raw, test_raw = train_test_split(df)

# 3. Generate features with fit-transform pattern
engineer = StatefulFeatureEngineer()
train_features = engineer.fit_transform(train_raw)  # Fit on training
test_features = engineer.transform(test_raw)        # Transform test

# 4. Convert to TimeSeries
train_ts = engineer.to_darts_timeseries(train_features, ["close"])
test_ts = engineer.to_darts_timeseries(test_features, ["close"])

# 4. Split data
train_ts = ts[:int(len(ts) * 0.7)]
val_ts = ts[int(len(ts) * 0.7):int(len(ts) * 0.85)]

# 5. Create and train model
model = ModelRegistry.create_model("nbeats")
model.fit(train_ts, val_ts)

# 6. Save model
model.save(Path("models/nbeats.pth"))
```

#### Step 3: Predict
```python
# User runs: python -m price_stradamus.cli predict --steps 5

# 1. Load model
model = ModelRegistry.create_model("nbeats")
model.load(Path("models/nbeats.pth"))

# 2. Load recent data
df = asyncio.run(db.get_ohlcv("BTCUSDT", "1m", limit=100))
df_with_features = feature_engineer.generate_all_features(df)
ts = feature_engineer.to_darts_timeseries(df_with_features, ["close"])

# 3. Make predictions
predictions = model.predict(n=5, series=ts)

# 4. Display
print(predictions.values())
```

---

## Key Design Patterns

### 1. Factory Pattern (ModelRegistry)

**Problem**: Need to create different model types dynamically

**Solution**:
```python
# Instead of:
if model_name == "nbeats":
    model = NBEATSModel()
elif model_name == "lstm":
    model = LSTMModel()
# ... many more

# Use:
model = ModelRegistry.create_model(model_name)
```

### 2. Strategy Pattern (BaseModel)

**Problem**: Multiple models with different algorithms but same interface

**Solution**:
```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, data): pass

    @abstractmethod
    def predict(self, n): pass

# All models implement same interface
def train_any_model(model: BaseModel, data):
    model.fit(data)
    predictions = model.predict(5)
```

### 3. Async Context Managers

**Problem**: Need to manage resources (connections, sessions)

**Solution**:
```python
async with aiohttp.ClientSession() as session:
    # Session auto-closes
    data = await session.get(url)
```

### 4. Dependency Injection

**Problem**: Hard to test with hard-coded dependencies

**Solution**:
```python
class ModelTrainer:
    def __init__(self, db: DatabaseManager):
        self.db = db  # Injected, easy to mock

# Testing
mock_db = MockDatabaseManager()
trainer = ModelTrainer(mock_db)
```

---

## Quick Reference

### Where to Add New Code

| What you're adding | Where it goes |
|-------------------|---------------|
| New data source | `src/price_stradamus/data/` |
| New model | `src/price_stradamus/models/neural/` or `ml/` |
| New metric | `src/price_stradamus/evaluation/metrics.py` |
| New CLI command | `src/price_stradamus/cli/` |
| New utility | `src/price_stradamus/utils/` |
| Configuration | `src/price_stradamus/config/settings.py` |

### Import Patterns

```python
# Standard library
from __future__ import annotations
import asyncio
from pathlib import Path

# Third-party
import pandas as pd
import torch

# Local (relative imports)
from price_stradamus.config import settings
from price_stradamus.data.fetcher import BinanceDataFetcher
```

---

## Practice Exercises

### Exercise 1: Trace Data Flow (20 minutes)

**Task**: Trace what happens when you run:
```bash
python -m price_stradamus.cli train --model nbeats
```

Write down:
1. Which file handles the CLI command?
2. Which services are called?
3. Where is data loaded from?
4. Where is the model saved?

<details>
<summary>Answer</summary>

1. **CLI**: `src/price_stradamus/cli/model_commands.py` - `train()` function
2. **Services**:
   - `DatabaseManager.get_ohlcv()` - Load data
   - `StatefulFeatureEngineer.fit_transform()` - Create features (training)
   - `StatefulFeatureEngineer.transform()` - Create features (test)
   - `ModelRegistry.create_model()` - Create model instance
3. **Data source**: PostgreSQL database (ml_data.ohlcv table)
4. **Model saved**: `models/nbeats.pth` (configurable path)
</details>

### Exercise 2: Add New Utility (30 minutes)

**Task**: Create a new utility function to format currency.

1. Create file: `src/price_stradamus/utils/currency.py`
2. Add function:
```python
def format_currency(amount: float, symbol: str = "USD") -> str:
    """Format currency with symbol and commas."""
    # Implement this
    pass
```
3. Import and use it in a CLI command

<details>
<summary>Solution</summary>

```python
# src/price_stradamus/utils/currency.py
def format_currency(amount: float, symbol: str = "USD") -> str:
    """Format currency with symbol and commas.

    Args:
        amount: The amount to format
        symbol: Currency symbol (default: USD)

    Returns:
        Formatted string (e.g., "$50,000.00")
    """
    if symbol == "USD":
        return f"${amount:,.2f}"
    elif symbol == "BTC":
        return f"₿{amount:,.8f}"
    else:
        return f"{symbol} {amount:,.2f}"

# Usage in CLI
from price_stradamus.utils.currency import format_currency

print(format_currency(50000.123))  # $50,000.12
```
</details>

---

## Next Steps

Excellent! You now understand the project architecture.

**Next Module:** [05: Database & Data Pipeline →](05-database-and-data-pipeline.md)

Learn how data flows from Binance API to PostgreSQL.

---

## Summary Checklist

- [ ] I understand the layer architecture
- [ ] I know the complete directory structure
- [ ] I can trace data flow through the system
- [ ] I know where to add new code
- [ ] I understand key design patterns

**Continue to:** [Module 05: Database & Data Pipeline](05-database-and-data-pipeline.md)
