# Price Stradamus - Complete Learning Guide (Phase 1)

> **Goal**: Understand every component so you can rebuild this system from scratch.

**Last Updated**: 2024-12-05
**Phase**: 1 - Core System
**Prerequisite Knowledge**: Python, basic ML concepts, SQL

---

## Table of Contents

1. [Big Picture: How Everything Works](#big-picture-how-everything-works)
2. [System Architecture](#system-architecture)
3. [Data Flow: From Binance to Predictions](#data-flow-from-binance-to-predictions)
4. [Component Deep Dive](#component-deep-dive)
5. [Code Structure Explained](#code-structure-explained)
6. [Key Design Patterns](#key-design-patterns)
7. [How to Rebuild from Scratch](#how-to-rebuild-from-scratch)
8. [Common Pitfalls & Solutions](#common-pitfalls--solutions)

---

## Big Picture: How Everything Works

### What Does Price Stradamus Do?

```
┌─────────────┐       ┌──────────────┐       ┌─────────────┐
│   Binance   │──────>│  PostgreSQL  │──────>│  ML Models  │
│     API     │ OHLCV │   Database   │ Time  │   (8 types) │
│  (BTC data) │ data  │   (storage)  │Series │             │
└─────────────┘       └──────────────┘       └─────────────┘
                                                      │
                                                      ▼
                                              ┌─────────────┐
                                              │  Future     │
                                              │  Prices     │
                                              │  (1-N steps)│
                                              └─────────────┘
```

### The 5-Step Process

1. **FETCH** - Download historical Bitcoin price data from Binance
2. **STORE** - Save OHLCV candles in PostgreSQL with deduplication
3. **ENGINEER** - Generate 42+ technical indicators (RSI, MACD, Bollinger, etc.)
4. **TRAIN** - Train ML models to learn price patterns
5. **PREDICT** - Forecast future prices (next 1-N candles)

---

## System Architecture

### Layer View

```
┌────────────────────────────────────────────────────────────────┐
│                      CLI Layer (Typer)                          │
│  Commands: fetch, train, predict, evaluate, list-models, info  │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                     Business Logic Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Data Pipeline│  │ Model Manager│  │  Evaluation  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    Persistence Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  PostgreSQL  │  │  Model Files │  │  Cache       │         │
│  │  (Database)  │  │  (.pkl/.pt)  │  │  (optional)  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                     External Services                           │
│  ┌──────────────┐                                               │
│  │  Binance API │  (REST API for historical data)              │
│  └──────────────┘                                               │
└────────────────────────────────────────────────────────────────┘
```

### Component Relationships

```
                    ┌─────────────────────────┐
                    │   CLI Commands          │
                    │   (User Interface)      │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴────────────┐
                    │                        │
                    ▼                        ▼
        ┌──────────────────┐      ┌──────────────────┐
        │  Data Pipeline   │      │  Model Pipeline  │
        └──────────────────┘      └──────────────────┘
                 │                          │
    ┌────────────┼────────────┐            │
    ▼            ▼            ▼            ▼
┌────────┐  ┌─────────┐  ┌────────┐  ┌─────────┐
│Fetcher │  │Features │  │Preproc.│  │ Models  │
└────────┘  └─────────┘  └────────┘  └─────────┘
    │            │            │            │
    └────────────┴────────────┴────────────┘
                    │
                    ▼
            ┌──────────────┐
            │  Database    │
            │  (PostgreSQL)│
            └──────────────┘
```

---

## Data Flow: From Binance to Predictions

### Complete Flow Diagram

```
START
  │
  ├─> [1. CLI Command: fetch]
  │         │
  │         ├─> BinanceDataFetcher.fetch_historical_range()
  │         │         │
  │         │         ├─> Calculate date range (start_date, end_date)
  │         │         ├─> Split into 1000-candle batches
  │         │         └─> For each batch:
  │         │               ├─> HTTP GET https://api.binance.com/api/v3/klines
  │         │               ├─> Parse JSON response
  │         │               └─> Convert to DataFrame
  │         │
  │         ├─> DatabaseManager.save_ohlcv()
  │         │         │
  │         │         ├─> BEGIN TRANSACTION
  │         │         ├─> For each row:
  │         │         │     ├─> INSERT INTO ml_data.ohlcv
  │         │         │     │   ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
  │         │         │     └─> (Automatic deduplication)
  │         │         └─> COMMIT
  │         │
  │         └─> Display summary (count, date range, price range)
  │
  ├─> [2. CLI Command: train]
  │         │
  │         ├─> DatabaseManager.get_ohlcv()
  │         │         │
  │         │         └─> SELECT * FROM ml_data.ohlcv
  │         │             WHERE symbol = ? AND timeframe = ?
  │         │             ORDER BY timestamp ASC
  │         │
  │         ├─> DataPreprocessor.validate_ohlcv()
  │         │         │
  │         │         ├─> Check required columns: [timestamp, open, high, low, close, volume]
  │         │         ├─> Validate data types
  │         │         ├─> Check for NaN/Inf values
  │         │         └─> Validate OHLC relationships (high >= open, high >= close, etc.)
  │         │
  │         ├─> DataPreprocessor.handle_missing_values()
  │         │         │
  │         │         ├─> Detect missing candles
  │         │         ├─> Forward fill for gaps <10 candles
  │         │         └─> Linear interpolation for larger gaps
  │         │
  │         ├─> StatefulFeatureEngineer.fit_transform()
  │         │         │
  │         │         ├─> Price Features:
  │         │         │     ├─> Returns (% change)
  │         │         │     ├─> Log returns
  │         │         │     └─> Price ratios
  │         │         │
  │         │         ├─> Trend Indicators:
  │         │         │     ├─> SMA (5, 10, 20, 50, 200)
  │         │         │     ├─> EMA (12, 26)
  │         │         │     └─> MACD (12-26-9)
  │         │         │
  │         │         ├─> Momentum Indicators:
  │         │         │     ├─> RSI (14)
  │         │         │     ├─> Stochastic (14)
  │         │         │     └─> ROC (Rate of Change)
  │         │         │
  │         │         ├─> Volatility Indicators:
  │         │         │     ├─> Bollinger Bands (20, 2σ)
  │         │         │     ├─> ATR (14)
  │         │         │     └─> Historical volatility
  │         │         │
  │         │         └─> Volume Indicators:
  │         │               ├─> OBV (On-Balance Volume)
  │         │               ├─> Volume SMA
  │         │               └─> VWAP
  │         │
  │         ├─> StatefulFeatureEngineer.to_darts_timeseries()
  │         │         │
  │         │         ├─> Set timestamp as index
  │         │         ├─> Convert to Darts TimeSeries object
  │         │         └─> Store metadata (frequency, columns)
  │         │
  │         ├─> Split data:
  │         │         │
  │         │         ├─> train_ts = ts[:int(len(ts) * 0.7)]      # 70%
  │         │         ├─> val_ts   = ts[train_size:train_size+val_size]  # 15%
  │         │         └─> test_ts  = ts[train_size+val_size:]     # 15%
  │         │
  │         ├─> ModelRegistry.create_model()
  │         │         │
  │         │         ├─> Look up model class by name
  │         │         ├─> Instantiate with hyperparameters
  │         │         └─> Return model instance
  │         │
  │         ├─> model.fit(train_ts, val_ts)
  │         │         │
  │         │         └─> Model-specific training:
  │         │               │
  │         │               ├─> [N-BEATS]: PyTorch neural network training
  │         │               │     ├─> Create dataset (windows of input_chunk_length)
  │         │               │     ├─> Initialize optimizer (Adam)
  │         │               │     ├─> For each epoch:
  │         │               │     │     ├─> Forward pass (predict)
  │         │               │     │     ├─> Calculate loss (MAE)
  │         │               │     │     ├─> Backward pass (gradients)
  │         │               │     │     └─> Update weights
  │         │               │     └─> Validate on val_ts
  │         │               │
  │         │               ├─> [LSTM/TCN/TFT]: Similar PyTorch training
  │         │               │
  │         │               ├─> [XGBoost]: Gradient boosting trees
  │         │               │     ├─> Create supervised dataset (X, y)
  │         │               │     ├─> Build trees iteratively
  │         │               │     └─> Optimize for MAE
  │         │               │
  │         │               ├─> []: Ensemble of decision trees
  │         │               │
  │         │               ├─> [ARIMA]: Statistical time series model
  │         │               │     ├─> Determine (p, d, q) orders
  │         │               │     └─> Maximum likelihood estimation
  │         │               │
  │         │               └─> [Prophet]: Additive decomposition model
  │         │                     ├─> Trend + Seasonality + Holidays
  │         │                     └─> Fit via Stan optimization
  │         │
  │         └─> model.save(path)
  │               │
  │               ├─> Serialize model state
  │               │     ├─> [Neural]: Save PyTorch state_dict + metadata
  │               │     ├─> [ML]: Save scikit-learn/XGBoost pickled model
  │               │     └─> [Classical]: Save model parameters
  │               └─> Write to disk
  │
  ├─> [3. CLI Command: predict]
  │         │
  │         ├─> Load recent data from database
  │         ├─> Preprocess + generate features (same as training)
  │         ├─> Convert to TimeSeries
  │         │
  │         ├─> model.load(path)
  │         │         │
  │         │         ├─> Read from disk
  │         │         ├─> Deserialize model state
  │         │         └─> Restore model architecture
  │         │
  │         ├─> model.predict(n=steps, series=ts)
  │         │         │
  │         │         └─> Model-specific prediction:
  │         │               │
  │         │               ├─> Take last `input_chunk_length` candles
  │         │               ├─> Pass through model
  │         │               └─> Output `output_chunk_length` future values
  │         │
  │         ├─> Format predictions as table
  │         └─> Optionally generate chart
  │
  └─> [4. CLI Command: evaluate]
            │
            ├─> Load data and model (same as predict)
            │
            ├─> Walk-forward validation:
            │         │
            │         ├─> For each window in test data:
            │         │     │
            │         │     ├─> Use data up to current point
            │         │     ├─> Predict next N steps
            │         │     ├─> Compare with actual values
            │         │     └─> Store predictions and actuals
            │         │
            │         └─> No lookahead bias!
            │
            ├─> MetricsCalculator.calculate_all()
            │         │
            │         ├─> MAE = mean(|actual - predicted|)
            │         ├─> RMSE = sqrt(mean((actual - predicted)²))
            │         ├─> MAPE = mean(|actual - predicted| / |actual|) * 100
            │         │
            │         └─> Directional Accuracy:
            │               ├─> predicted_direction = sign(predicted[t] - actual[t-1])
            │               ├─> actual_direction = sign(actual[t] - actual[t-1])
            │               └─> accuracy = mean(predicted_direction == actual_direction)
            │
            ├─> Display metrics table
            └─> Optionally generate chart
```

---

## Component Deep Dive

### 1. Data Pipeline (`src/price_stradamus/data/`)

#### 1.1 BinanceDataFetcher (`fetcher.py`)

**Purpose**: Download historical OHLCV data from Binance API.

**Key Methods**:
```python
class BinanceDataFetcher:
    async def fetch_historical_range(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for date range.

        Implementation:
        1. Calculate number of candles needed
        2. Split into batches of 1000 (Binance limit)
        3. For each batch:
           - Build API request URL
           - Make async HTTP GET request
           - Parse JSON response
           - Convert to DataFrame
        4. Concatenate all batches
        5. Return deduplicated DataFrame
        """
```

**API Endpoint Used**:
```
GET https://api.binance.com/api/v3/klines

Parameters:
  symbol: BTCUSDT
  interval: 1m, 5m, 15m, 1h, 4h, 1d
  startTime: Unix timestamp (milliseconds)
  endTime: Unix timestamp (milliseconds)
  limit: Max 1000

Response Format:
[
  [
    1499040000000,      // Open time
    "0.01634790",       // Open price
    "0.80000000",       // High price
    "0.01575800",       // Low price
    "0.01577100",       // Close price
    "148976.11427815",  // Volume
    1499644799999,      // Close time
    "2434.19055334",    // Quote asset volume
    308,                // Number of trades
    "1756.87402397",    // Taker buy base volume
    "28.46694368",      // Taker buy quote volume
    "0"                 // Ignore
  ]
]
```

**Error Handling**:
- Rate limiting: Retry with exponential backoff
- Network errors: Retry up to 3 times
- Invalid data: Log and skip

#### 1.2 DatabaseManager (`database.py`)

**Purpose**: Store and retrieve OHLCV data in PostgreSQL.

**Database Schema**:
```sql
-- Schema: ml_data
-- Table: ohlcv

CREATE SCHEMA IF NOT EXISTS ml_data;

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

CREATE INDEX idx_ohlcv_lookup
    ON ml_data.ohlcv(symbol, timeframe, timestamp);
```

**Key Methods**:
```python
class DatabaseManager:
    async def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> int:
        """Save OHLCV data with automatic deduplication.

        Implementation:
        1. Start database transaction
        2. For each row in DataFrame:
           - Try to INSERT
           - ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
           - (UNIQUE constraint handles deduplication)
        3. Count rows actually inserted
        4. Commit transaction
        5. Return count of new rows
        """

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Retrieve OHLCV data.

        Implementation:
        1. Build SELECT query
        2. Execute async query
        3. Fetch all rows
        4. Convert to pandas DataFrame
        5. Set timestamp as index
        6. Return DataFrame
        """
```

#### 1.3 DataPreprocessor (`preprocessor.py`)

**Purpose**: Validate and clean raw OHLCV data.

**Key Methods**:
```python
class DataPreprocessor:
    def validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV data quality.

        Checks:
        1. Required columns present
        2. No negative prices or volumes
        3. OHLC relationships: high >= open, high >= close, etc.
        4. No NaN or Inf values
        5. Timestamps in ascending order

        Raises ValueError if validation fails.
        """

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing candles.

        Strategy:
        1. Detect missing timestamps (gaps in sequence)
        2. For small gaps (<10 candles):
           - Forward fill (use last known value)
        3. For large gaps (>=10 candles):
           - Linear interpolation between boundaries
        4. Log all filled values for audit
        """
```

#### 1.4 StatefulFeatureEngineer (`stateful_features.py`)

**Purpose**: Generate technical indicators from OHLCV data with proper fit-transform pattern to prevent data leakage.

**Feature Categories**:

1. **Price Features** (4 features)
   - Returns: `(close[t] - close[t-1]) / close[t-1]`
   - Log returns: `log(close[t] / close[t-1])`
   - High-Low ratio: `high / low`
   - Close-Open ratio: `close / open`

2. **Trend Indicators** (10 features)
   - SMA: Simple Moving Average (5, 10, 20, 50, 200 periods)
   - EMA: Exponential Moving Average (12, 26 periods)
   - MACD: Moving Average Convergence Divergence
   - Signal line, histogram

3. **Momentum Indicators** (8 features)
   - RSI: Relative Strength Index (14)
   - Stochastic: %K and %D (14, 3)
   - ROC: Rate of Change (10)
   - Williams %R (14)
   - CCI: Commodity Channel Index (20)

4. **Volatility Indicators** (6 features)
   - Bollinger Bands (upper, middle, lower)
   - ATR: Average True Range (14)
   - Historical volatility (rolling std)
   - Keltner Channels

5. **Volume Indicators** (4 features)
   - OBV: On-Balance Volume
   - Volume SMA (20)
   - VWAP: Volume Weighted Average Price
   - Volume ratio

6. **Pattern Features** (10 features)
   - Higher highs / Lower lows
   - Support/Resistance levels
   - Pivot points
   - Fibonacci retracements

**Implementation**:
```python
class StatefulFeatureEngineer:
    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate all 58+ features using pandas-ta with fit-transform pattern.

        Implementation:
        1. Create copy of DataFrame
        2. Add each feature category:
           - Call pandas_ta methods (e.g., ta.rsi())
           - Handle NaN values from indicator calculations
           - Store normalization statistics (mean, std) from training data
        3. Normalize features using training statistics
        4. Drop warmup period rows
        3. Drop rows with remaining NaNs
        4. Return DataFrame with original + feature columns
        """

    def to_darts_timeseries(
        self,
        df: pd.DataFrame,
        value_cols: list[str],
    ) -> TimeSeries:
        """Convert pandas DataFrame to Darts TimeSeries.

        Darts Requirements:
        - DateTime index (must be sorted)
        - Numeric values only
        - No missing values
        - Regular frequency (e.g., 1min, 1H)

        Implementation:
        1. Ensure timestamp is index
        2. Select only value_cols
        3. Infer frequency from timestamps
        4. Create TimeSeries object
        5. Validate regularity
        """
```

---

### 2. Model Pipeline (`src/price_stradamus/models/`)

#### 2.1 Model Registry (`registry.py`)

**Purpose**: Central registry for all model classes.

**Design Pattern**: Factory Pattern

```python
class ModelRegistry:
    """Global registry for ML models.

    Models register themselves at import time using decorator:

    @ModelRegistry.register("nbeats")
    class NBEATSModel(BaseModel):
        ...
    """

    _models: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model."""
        def decorator(model_class: type[BaseModel]):
            cls._models[name] = model_class
            return model_class
        return decorator

    @classmethod
    def create_model(
        cls,
        name: str,
        **kwargs,
    ) -> BaseModel:
        """Factory method to create model instance."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}")

        model_class = cls._models[name]
        return model_class(**kwargs)
```

**Benefits**:
- Add new models without changing core code
- Type-safe model creation
- Easy to list available models
- Supports dependency injection

#### 2.2 Base Model Interface (`base.py`)

**Purpose**: Common interface all models must implement.

```python
class BaseModel(ABC):
    """Abstract base class for all prediction models."""

    def __init__(
        self,
        input_chunk_length: int = 60,
        output_chunk_length: int = 5,
        **kwargs,
    ):
        """Initialize model with hyperparameters."""
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def fit(
        self,
        train_series: TimeSeries,
        val_series: TimeSeries | None = None,
    ) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(
        self,
        n: int,
        series: TimeSeries,
    ) -> TimeSeries:
        """Make predictions."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        pass
```

**Why This Design?**:
- **Polymorphism**: Treat all models the same way
- **Type checking**: Catch errors at development time
- **Documentation**: Clear contract for new models
- **Testing**: Easy to mock models

#### 2.3 Model Types

##### Neural Networks (PyTorch + Darts)

**N-BEATS** (`models/neural/nbeats.py`):
```python
@ModelRegistry.register("nbeats")
class NBEATSModel(BaseModel):
    """Neural Basis Expansion Analysis for Time Series.

    Architecture:
    - Stack of "blocks" (trend + seasonality)
    - Each block:
      * Fully connected layers
      * Backward/forward pass
      * Basis function expansion (Fourier for seasonality, polynomial for trend)
    - No recurrence (faster than LSTM)
    - Interpretable components

    Paper: https://arxiv.org/abs/1905.10437
    """

    def __init__(
        self,
        input_chunk_length: int = 60,
        output_chunk_length: int = 5,
        num_stacks: int = 30,
        num_blocks: int = 1,
        layer_widths: int = 256,
        n_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        **kwargs,
    ):
        super().__init__(input_chunk_length, output_chunk_length)

        # Darts NBEATSModel wrapper around PyTorch implementation
        self.model = DartsNBEATS(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            generic_architecture=True,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=4,
            layer_widths=layer_widths,
            n_epochs=n_epochs,
            batch_size=batch_size,
            optimizer_kwargs={"lr": learning_rate},
            pl_trainer_kwargs={
                "accelerator": "auto",  # Use GPU if available
                "callbacks": [EarlyStopping(monitor="val_loss", patience=10)],
            },
        )
```

**LSTM** (`models/neural/lstm.py`):
```
Long Short-Term Memory network
- Recurrent architecture with memory cells
- Good for sequential patterns
- Slower than N-BEATS but handles longer dependencies
```

**TCN** (`models/neural/tcn.py`):
```
Temporal Convolutional Network
- 1D convolutions with dilations
- Parallel processing (faster than RNN)
- Large receptive field
```

**TFT** (`models/neural/tft.py`):
```
Temporal Fusion Transformer
- Attention mechanism
- Multi-horizon forecasting
- Interpretable attention weights
```

##### Classical Models

**ARIMA** (`models/classical/arima.py`):
```python
@ModelRegistry.register("arima")
class ARIMAModel(BaseModel):
    """AutoRegressive Integrated Moving Average.

    Model: ARIMA(p, d, q)
    - p: Autoregressive order (use past values)
    - d: Differencing order (make stationary)
    - q: Moving average order (use past errors)

    Automatic order selection via AIC/BIC.

    Best for: Stationary time series, short-term forecasts
    Limitations: Assumes linearity, struggles with complex patterns
    """
```

**Prophet** (`models/classical/prophet.py`):
```python
@ModelRegistry.register("prophet")
class ProphetModel(BaseModel):
    """Facebook Prophet model.

    Additive model:
    y(t) = trend(t) + seasonality(t) + holidays(t) + noise

    - Piecewise linear/logistic trend
    - Yearly/weekly/daily seasonality (Fourier series)
    - Holiday effects
    - Robust to missing data and outliers

    Best for: Data with strong seasonal patterns
    Limitations: Assumes additivity, not great for short-term predictions
    """
```

##### ML Models

**XGBoost** (`models/ml/xgboost.py`):
```python
@ModelRegistry.register("xgboost")
class XGBoostModel(BaseModel):
    """Gradient Boosted Trees (XGBoost).

    Ensemble of decision trees built iteratively:
    1. Fit tree to residuals of previous trees
    2. Add to ensemble with learning rate
    3. Repeat

    Features:
    - Handles non-linearity well
    - Feature importance
    - Fast training (C++ backend)
    - Regularization (L1/L2)

    Hyperparameters:
    - n_estimators: Number of trees
    - max_depth: Tree depth (prevent overfitting)
    - learning_rate: Step size
    - subsample: Row sampling
    """
```

**** (`models/ml/.py`):
```
Ensemble of decision trees with bagging
- Each tree sees random subset of data
- Voting/averaging for prediction
- Less prone to overfitting than single tree
```

---

### 3. Evaluation System (`src/price_stradamus/evaluation/`)

#### 3.1 MetricsCalculator (`metrics.py`)

```python
class MetricsCalculator:
    """Calculate prediction performance metrics."""

    @staticmethod
    def calculate_all(
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> dict[str, float]:
        """Calculate all metrics.

        Returns:
            {
                "mae": Mean Absolute Error,
                "rmse": Root Mean Squared Error,
                "mape": Mean Absolute Percentage Error,
                "directional_accuracy": % of correct direction predictions
            }
        """

        # MAE: Average absolute difference
        mae = np.mean(np.abs(actual - predicted))

        # RMSE: Square root of average squared difference
        # (Penalizes large errors more than MAE)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        # MAPE: Average percentage error
        # (Scale-independent, good for comparing across price levels)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        # Directional Accuracy: Did we predict up/down correctly?
        # This is crucial for trading!
        predicted_direction = np.sign(np.diff(predicted))
        actual_direction = np.sign(np.diff(actual))
        directional_accuracy = (
            np.mean(predicted_direction == actual_direction) * 100
        )

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "directional_accuracy": directional_accuracy,
        }
```

**Why These Metrics?**

1. **MAE**: Easy to interpret ($123 average error)
2. **RMSE**: Penalizes large errors (important for risk)
3. **MAPE**: Compare across different price levels
4. **Directional Accuracy**: Most important for trading! >52% is profitable

#### 3.2 Backtester (`backtester.py`)

**Purpose**: Test model on historical data without lookahead bias.

**Walk-Forward Validation**:
```
Timeline: [═════════════════════════════════════════]

Step 1:   [Train======][Val][Predict]
                              ↓
                              Compare with actual

Step 2:   [Train=============][Val][Predict]
                                      ↓
                                      Compare with actual

Step 3:   [Train====================][Val][Predict]
                                              ↓
                                              Compare with actual

Key Rule: NEVER use future data for prediction!
```

**Implementation**:
```python
class Backtester:
    """Walk-forward backtesting with no lookahead bias."""

    def run_expanding_window(
        self,
        series: TimeSeries,
    ) -> dict:
        """Expanding window backtest.

        Implementation:
        1. Split data: train (70%), val (15%), test (15%)
        2. Train model on train+val
        3. For each window in test set:
           a. Use only data up to current point
           b. Predict next N steps
           c. Store predictions and actuals
           d. Move window forward
        4. Calculate metrics on all predictions
        5. Return results

        This simulates real trading: you only know the past!
        """

        train_size = int(len(series) * 0.7)
        val_size = int(len(series) * 0.15)

        train_ts = series[:train_size]
        val_ts = series[train_size:train_size + val_size]
        test_ts = series[train_size + val_size:]

        # Train model
        self.model.fit(train_ts, val_ts)

        # Walk forward through test set
        predictions = []
        actuals = []

        current_idx = 0
        while current_idx + self.output_length <= len(test_ts):
            # Use data up to current point (no lookahead!)
            historical = series[:train_size + val_size + current_idx]

            # Predict next N steps
            pred = self.model.predict(n=self.output_length, series=historical)

            # Get actual values
            actual = test_ts[current_idx:current_idx + self.output_length]

            predictions.append(pred.values())
            actuals.append(actual.values())

            # Move window forward
            current_idx += self.output_length

        # Calculate metrics
        all_predictions = np.concatenate(predictions)
        all_actuals = np.concatenate(actuals)

        metrics = MetricsCalculator.calculate_all(all_actuals, all_predictions)

        return {
            "predictions": all_predictions,
            "actuals": all_actuals,
            "metrics": metrics,
        }
```

---

## Code Structure Explained

### File Organization

```
src/price_stradamus/
│
├── config/                 # Configuration & Settings
│   ├── __init__.py
│   ├── settings.py        # Pydantic settings (env vars, defaults)
│   └── constants.py       # Global constants (timeframes, limits)
│
├── data/                   # Data Pipeline
│   ├── __init__.py
│   ├── fetcher.py         # Binance API client
│   ├── database.py        # PostgreSQL operations
│   ├── preprocessor.py    # Data cleaning & validation
│   └── features.py        # Technical indicator generation
│
├── models/                 # ML Models
│   ├── __init__.py
│   ├── base.py            # BaseModel abstract class
│   ├── registry.py        # Model factory & registration
│   ├── exceptions.py      # Model-specific exceptions
│   │
│   ├── neural/            # Neural Network Models
│   │   ├── __init__.py
│   │   ├── nbeats.py      # N-BEATS
│   │   ├── lstm.py        # LSTM
│   │   ├── tcn.py         # TCN
│   │   └── tft.py         # Temporal Fusion Transformer
│   │
│   ├── classical/         # Classical Models
│   │   ├── __init__.py
│   │   ├── arima.py       # ARIMA
│   │   └── prophet.py     # Prophet
│   │
│   └── ml/                # Machine Learning Models
│       ├── __init__.py
│       ├── xgboost.py     # XGBoost
│       └── .py  # 
│
├── evaluation/             # Model Evaluation
│   ├── __init__.py
│   ├── metrics.py         # Metric calculations (MAE, RMSE, etc.)
│   ├── backtester.py      # Walk-forward validation
│   └── model_comparison.py  # Compare multiple models
│
├── visualization/          # Charting & Plotting
│   ├── __init__.py
│   └── charts.py          # Plotly interactive charts
│
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── logger.py          # Loguru configuration
│   └── helpers.py         # Helper functions
│
└── cli/                    # Command-Line Interface
    ├── __init__.py        # Main app setup
    ├── data_commands.py   # Data operations (fetch)
    ├── model_commands.py  # Model operations (train, predict, evaluate)
    ├── info_commands.py   # Info commands (list-models, info)
    └── CLI-COMMANDS.md    # Complete CLI reference
```

### Import Flow

```python
# Entry point: pyproject.toml
[project.scripts]
price-stradamus = "price_stradamus.cli:main"

# cli/__init__.py
from price_stradamus.cli.data_commands import register_data_commands
from price_stradamus.cli.model_commands import register_model_commands
from price_stradamus.cli.info_commands import register_info_commands

app = typer.Typer()
register_data_commands(app)
register_model_commands(app)
register_info_commands(app)

def main():
    app()

# cli/model_commands.py
from price_stradamus.models.registry import ModelRegistry
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.stateful_features import StatefulFeatureEngineer
# etc.

# models/neural/nbeats.py
from price_stradamus.models.base import BaseModel
from price_stradamus.models.registry import ModelRegistry

@ModelRegistry.register("nbeats")  # Self-registration!
class NBEATSModel(BaseModel):
    ...
```

**Key Design**: Models register themselves at import time, so no central list to maintain.

---

## Key Design Patterns

### 1. Factory Pattern (ModelRegistry)

**Problem**: Need to create different model types dynamically.

**Solution**: Registry + Factory method

```python
# Bad: Hard-coded model creation
if model_name == "nbeats":
    model = NBEATSModel(**kwargs)
elif model_name == "lstm":
    model = LSTMModel(**kwargs)
# ... 8 more elif statements

# Good: Factory pattern
model = ModelRegistry.create_model(model_name, **kwargs)
```

**Benefits**:
- Add new models without changing core code
- Type-safe
- Easy to test

### 2. Strategy Pattern (BaseModel)

**Problem**: Multiple models with different algorithms but same interface.

**Solution**: Abstract base class with polymorphism

```python
# All models implement the same interface
def train_any_model(model: BaseModel, data: TimeSeries):
    model.fit(data)
    predictions = model.predict(n=5)
    model.save(Path("model.pkl"))

# Works with any model!
train_any_model(NBEATSModel(), data)
train_any_model(XGBoostModel(), data)
```

### 3. Async/Await for I/O

**Problem**: Blocking I/O slows down data fetching.

**Solution**: Async operations with `asyncio`

```python
# Bad: Synchronous (blocks)
def fetch_data(symbol):
    response = requests.get(url)  # Blocks!
    return response.json()

# Fetch 3 symbols: 3 seconds total
for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    data = fetch_data(symbol)  # 1 second each

# Good: Asynchronous (concurrent)
async def fetch_data(symbol):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# Fetch 3 symbols: 1 second total!
tasks = [fetch_data(symbol) for symbol in symbols]
results = await asyncio.gather(*tasks)
```

### 4. Context Managers for Resources

**Problem**: Ensure database connections are properly closed.

**Solution**: Async context managers

```python
# Ensures connection is always closed
async with BinanceDataFetcher() as fetcher:
    data = await fetcher.fetch_historical_range(...)
# Connection automatically closed here, even if exception occurs
```

### 5. Dependency Injection

**Problem**: Hard-coded dependencies make testing difficult.

**Solution**: Inject dependencies via parameters

```python
# Bad: Hard-coded dependency
class ModelTrainer:
    def __init__(self):
        self.db = DatabaseManager("postgresql://...")  # Hard-coded!

# Good: Dependency injection
class ModelTrainer:
    def __init__(self, db: DatabaseManager):
        self.db = db  # Injected, easy to mock in tests

# Usage
db = DatabaseManager(settings.database_url)
trainer = ModelTrainer(db)

# Testing
mock_db = MockDatabaseManager()
trainer = ModelTrainer(mock_db)  # Easy to test!
```

---

## How to Rebuild from Scratch

If you deleted everything, here's how to rebuild the system step-by-step:

### Step 1: Project Setup (30 minutes)

```bash
# 1. Create directory structure
mkdir -p price-stradamus/{src/price_stradamus/{config,data,models/{neural,classical,ml},evaluation,utils,cli},tests,docker}

# 2. Initialize Python project
cd price-stradamus
uv init
uv venv

# 3. Create pyproject.toml
# Add dependencies: darts, pandas, numpy, torch, xgboost, scikit-learn,
#                   asyncpg, sqlalchemy, aiohttp, typer, rich, loguru

# 4. Setup Docker Compose for PostgreSQL
# docker/docker-compose.yml
```

### Step 2: Configuration System (1 hour)

```python
# src/price_stradamus/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/price_stradamus"
    binance_base_url: str = "https://api.binance.com"
    # ... other settings

    class Config:
        env_file = ".env"

settings = Settings()
```

### Step 3: Database Layer (2 hours)

```python
# 1. Create database schema (SQL in migrations/)
# 2. Implement DatabaseManager with asyncpg
#    - initialize()
#    - save_ohlcv()
#    - get_ohlcv()
#    - close()
# 3. Write tests
```

### Step 4: Data Fetcher (3 hours)

```python
# 1. Implement BinanceDataFetcher
#    - fetch_historical_range()
#    - Handle rate limiting
#    - Batch requests (1000 limit)
# 2. Implement retry logic
# 3. Write tests with mocked API
```

### Step 5: Preprocessing & Features (4 hours)

```python
# 1. Implement DataPreprocessor
#    - validate_ohlcv()
#    - handle_missing_values()
# 2. Implement StatefulFeatureEngineer
#    - fit_transform() using pandas-ta (fit on training)
#    - transform() for test data (no leakage)
#    - to_darts_timeseries()
# 3. Test on real data with proper train/test split
```

### Step 6: Model Infrastructure (2 hours)

```python
# 1. Create BaseModel abstract class
# 2. Implement ModelRegistry
# 3. Write tests for registry
```

### Step 7: Implement Models (8 hours, 1 hour per model)

```python
# For each model:
# 1. Create model file (e.g., nbeats.py)
# 2. Implement BaseModel interface
# 3. Add @ModelRegistry.register("name") decorator
# 4. Write tests

# Order: Start with XGBoost (simplest), then N-BEATS, then others
```

### Step 8: Evaluation System (3 hours)

```python
# 1. Implement MetricsCalculator
# 2. Implement Backtester with walk-forward validation
# 3. Write tests
```

### Step 9: CLI Commands (4 hours)

```python
# 1. Create Typer app
# 2. Implement commands:
#    - fetch
#    - train
#    - predict
#    - evaluate
#    - list-models
#    - info
# 3. Test manually
```

### Step 10: Polish & Documentation (4 hours)

```python
# 1. Add type hints to all functions
# 2. Write docstrings (Google style)
# 3. Setup Ruff, Pyright
# 4. Write README, QUICK-START
# 5. Create tests to reach 80% coverage
```

**Total Time**: ~30-35 hours of focused work

---

## Common Pitfalls & Solutions

### Pitfall 1: Lookahead Bias in Backtesting

**Problem**:
```python
# BAD: Uses future data!
def backtest(data):
    # Normalize using entire dataset's min/max
    scaler = MinMaxScaler()
    scaler.fit(data)  # This includes test data!

    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
```

**Solution**:
```python
# GOOD: Fit scaler only on training data
def backtest(data):
    train_data, test_data = split(data)

    # Fit only on training data
    scaler = MinMaxScaler()
    scaler.fit(train_data)

    # Apply same transformation to test
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
```

### Pitfall 2: Data Leakage Through Features

**Problem**:
```python
# BAD: Uses future information!
def add_target_encoding(df):
    # Group statistics include test data
    df["price_mean"] = df.groupby("hour")["close"].transform("mean")
```

**Solution**:
```python
# GOOD: Calculate statistics only on training data
def add_target_encoding(train_df, test_df):
    # Calculate stats only on training
    stats = train_df.groupby("hour")["close"].mean()

    # Apply to both
    train_df["price_mean"] = train_df["hour"].map(stats)
    test_df["price_mean"] = test_df["hour"].map(stats)
```

### Pitfall 3: Incorrect Train/Test Split for Time Series

**Problem**:
```python
# BAD: Random split destroys temporal order!
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, shuffle=True)
```

**Solution**:
```python
# GOOD: Temporal split (no shuffle!)
split_point = int(len(data) * 0.8)
train = data[:split_point]
test = data[split_point:]
```

### Pitfall 4: Not Handling Missing Candles

**Problem**:
```python
# BAD: Assumes all timestamps present
df = fetch_ohlcv()
# df has gaps, but model expects regular intervals!
```

**Solution**:
```python
# GOOD: Detect and fill gaps
df = fetch_ohlcv()
df = preprocessor.handle_missing_values(df)  # Fill gaps
df = preprocessor.validate_regular_intervals(df)  # Verify
```

### Pitfall 5: Ignoring GPU Availability

**Problem**:
```python
# BAD: Always uses CPU
model = NBEATSModel()
model.fit(data)  # Slow!
```

**Solution**:
```python
# GOOD: Use GPU if available
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = NBEATSModel(pl_trainer_kwargs={"accelerator": "auto"})
model.fit(data)  # Automatically uses GPU
```

### Pitfall 6: Not Saving Model Metadata

**Problem**:
```python
# BAD: Only save model weights
torch.save(model.state_dict(), "model.pth")
# Lost: input_length, output_length, feature names, normalization params!
```

**Solution**:
```python
# GOOD: Save complete model state
import pickle

state = {
    "model_state": model.state_dict(),
    "input_chunk_length": 60,
    "output_chunk_length": 5,
    "feature_names": ["close", "volume", ...],
    "scaler_params": scaler.get_params(),
}

with open("model.pkl", "wb") as f:
    pickle.dump(state, f)
```

### Pitfall 7: Overfitting on Test Set

**Problem**:
```python
# BAD: Tune hyperparameters on test set
for lr in [0.001, 0.01, 0.1]:
    model.learning_rate = lr
    model.fit(train)
    if model.score(test) > best_score:  # Looking at test!
        best_lr = lr
```

**Solution**:
```python
# GOOD: Use validation set for hyperparameter tuning
for lr in [0.001, 0.01, 0.1]:
    model.learning_rate = lr
    model.fit(train)
    if model.score(val) > best_score:  # Use validation!
        best_lr = lr

# Final evaluation on test (only once!)
final_score = model.score(test)
```

---

## Next Steps

Now that you understand the complete system:

1. **Read the actual code** - Start with simple files (constants.py, settings.py)
2. **Run the system** - Follow QUICK-START.md
3. **Modify something small** - Add a new feature, change a hyperparameter
4. **Implement a new model** - Use BaseModel as template
5. **Experiment** - Try different timeframes, symbols, model combinations

**Remember**: The best way to learn is by doing. Start small, make changes, and see what happens!

---

**Questions? Check**:
- [CLI-COMMANDS.md](../src/price_stradamus/cli/CLI-COMMANDS.md) - All CLI commands
- [models.md](models.md) - Detailed model documentation
- [evaluation.md](evaluation.md) - Metrics and backtesting
- [CLAUDE.md](../CLAUDE.md) - Complete development guide

**Happy Learning!** 🚀
