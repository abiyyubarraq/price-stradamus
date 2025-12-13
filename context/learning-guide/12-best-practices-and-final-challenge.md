# Module 12: Best Practices and Final Challenge

**Duration:** 4-6 hours | **Difficulty:** Advanced | **Prerequisites:** Modules 01-11

## 🎯 Learning Objectives

After this module, you will:
- Master code quality standards for ML projects
- Know how to test ML systems properly
- Understand ML-specific best practices
- Complete hands-on mini-projects
- **FINAL CHALLENGE**: Rebuild Price Stradamus from scratch!

---

## Code Quality Standards

### 1. Type Hints Everywhere

```python
# ❌ BAD: No type hints
def train_model(data, model_name, epochs):
    model = create_model(model_name)
    model.fit(data, epochs)
    return model

# ✅ GOOD: Full type hints
from __future__ import annotations
from darts import TimeSeries

def train_model(
    data: TimeSeries,
    model_name: str,
    epochs: int = 100,
) -> BaseModel:
    """Train a forecasting model.

    Args:
        data: Time series training data
        model_name: Name of model to train
        epochs: Number of training epochs

    Returns:
        Trained model instance
    """
    model = create_model(model_name)
    model.fit(data, n_epochs=epochs)
    return model
```

### 2. Docstrings (Google Style)

```python
class DataPreprocessor:
    """Preprocessor for OHLCV time series data.

    Handles validation, cleaning, and transformation of raw
    market data for model training.

    Attributes:
        config: Preprocessing configuration
        scaler: Fitted data scaler (None until fit() is called)

    Example:
        >>> preprocessor = DataPreprocessor()
        >>> df_clean = preprocessor.validate_ohlcv(df_raw)
        >>> df_normalized = preprocessor.normalize(df_clean)
    """

    def validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV data integrity.

        Checks for:
        - Required columns present
        - No negative values
        - Valid OHLC relationships (high >= low, etc.)
        - No NaN or Inf values

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            Validated DataFrame (same as input if valid)

        Raises:
            ValueError: If validation fails
        """
        pass
```

### 3. Error Handling

```python
# Custom exception hierarchy
class PriceStradamusError(Exception):
    """Base exception for Price Stradamus."""

class DataError(PriceStradamusError):
    """Data-related errors."""

class DataFetchError(DataError):
    """Failed to fetch data from API."""

class DataValidationError(DataError):
    """Data validation failed."""

class ModelError(PriceStradamusError):
    """Model-related errors."""

class ModelTrainingError(ModelError):
    """Model training failed."""

# Usage
try:
    data = await fetch_binance_data(symbol)
except aiohttp.ClientError as e:
    raise DataFetchError(f"Failed to fetch {symbol}: {e}") from e

try:
    validate_ohlcv(data)
except ValueError as e:
    raise DataValidationError(f"Invalid OHLCV data: {e}") from e
```

### 4. Pathlib for Paths

```python
# ❌ BAD: String concatenation
model_dir = "models/" + model_name
checkpoint = model_dir + "/checkpoint.pth"

# ✅ GOOD: pathlib
from pathlib import Path

model_dir = Path("models") / model_name
model_dir.mkdir(parents=True, exist_ok=True)
checkpoint = model_dir / "checkpoint.pth"

if checkpoint.exists():
    model.load(checkpoint)
```

### 5. Constants

```python
# src/price_stradamus/config/constants.py

# Use Enum for fixed choices
from enum import Enum

class Timeframe(str, Enum):
    """Supported timeframes."""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"

class ModelType(str, Enum):
    """Model categories."""
    NEURAL = "neural"
    CLASSICAL = "classical"
    ML = "ml"

# Named constants
DEFAULT_LOOKBACK_WINDOW = 60
DEFAULT_PREDICTION_STEPS = 5
MAX_EPOCHS = 1000
MIN_DATA_POINTS = 100

# Use constants
if len(data) < MIN_DATA_POINTS:
    raise ValueError(f"Need at least {MIN_DATA_POINTS} data points")
```

---

## Testing ML Systems

### Unit Tests

```python
# tests/test_preprocessor.py
import pytest
import pandas as pd
import numpy as np
from price_stradamus.data.preprocessor import DataPreprocessor

class TestDataPreprocessor:
    """Test DataPreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return DataPreprocessor()

    @pytest.fixture
    def valid_ohlcv(self):
        """Create valid OHLCV data."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=100, freq='1min'),
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(40000, 50000, 100),
            'low': np.random.uniform(40000, 50000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(1, 100, 100),
        })

    def test_validate_ohlcv_success(self, preprocessor, valid_ohlcv):
        """Test validation with valid data."""
        result = preprocessor.validate_ohlcv(valid_ohlcv)
        assert len(result) == len(valid_ohlcv)

    def test_validate_ohlcv_missing_columns(self, preprocessor):
        """Test validation fails with missing columns."""
        df = pd.DataFrame({'open': [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing columns"):
            preprocessor.validate_ohlcv(df)

    def test_validate_ohlcv_negative_values(self, preprocessor, valid_ohlcv):
        """Test validation fails with negative values."""
        valid_ohlcv.loc[0, 'close'] = -100
        with pytest.raises(ValueError, match="Negative values"):
            preprocessor.validate_ohlcv(valid_ohlcv)

    def test_handle_missing_values(self, preprocessor):
        """Test missing value handling."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=10, freq='1min'),
            'close': [100, 105, np.nan, np.nan, 110, 112, np.nan, 115, 118, 120],
        })
        result = preprocessor.handle_missing_values(df)
        assert result['close'].isnull().sum() == 0
```

### Integration Tests

```python
# tests/test_integration/test_data_pipeline.py
import pytest
from price_stradamus.data.fetcher import BinanceDataFetcher
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.preprocessor import DataPreprocessor

@pytest.mark.asyncio
async def test_full_data_pipeline():
    """Test complete data pipeline from fetch to storage."""
    # Setup
    fetcher = BinanceDataFetcher()
    preprocessor = DataPreprocessor()
    db = DatabaseManager()
    await db.initialize()

    try:
        # Fetch
        df = await fetcher.fetch_historical_range(
            "BTCUSDT", "1m",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 2),
        )
        assert len(df) > 0

        # Validate
        df = preprocessor.validate_ohlcv(df)

        # Store
        inserted = await db.save_ohlcv(df, "BTCUSDT", "1m")
        assert inserted > 0

        # Retrieve
        df_loaded = await db.get_ohlcv("BTCUSDT", "1m")
        assert len(df_loaded) >= inserted

    finally:
        await db.close()
```

### Model Tests

```python
# tests/test_models/test_nbeats.py
import pytest
from darts import TimeSeries
import numpy as np
from price_stradamus.models.neural.nbeats import NBEATSModel

class TestNBEATSModel:
    """Test N-BEATS model."""

    @pytest.fixture
    def model(self):
        """Create N-BEATS model with fast settings."""
        return NBEATSModel(
            input_chunk_length=10,
            output_chunk_length=2,
            n_epochs=1,  # Fast for testing
        )

    @pytest.fixture
    def dummy_data(self):
        """Create dummy time series data."""
        values = np.sin(np.linspace(0, 10, 100))
        return TimeSeries.from_values(values)

    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model.name == "nbeats"
        assert not model.is_fitted

    def test_model_fit(self, model, dummy_data):
        """Test model training."""
        train = dummy_data[:80]
        val = dummy_data[80:]

        model.fit(train, val)
        assert model.is_fitted

    def test_model_predict(self, model, dummy_data):
        """Test model predictions."""
        train = dummy_data[:80]
        model.fit(train)

        predictions = model.predict(n=5, series=train)
        assert len(predictions) == 5

    def test_model_save_load(self, model, dummy_data, tmp_path):
        """Test model persistence."""
        train = dummy_data[:80]
        model.fit(train)

        # Save
        save_path = tmp_path / "model.pth"
        model.save(save_path)
        assert save_path.exists()

        # Load
        new_model = NBEATSModel()
        new_model.load(save_path)
        assert new_model.is_fitted

        # Predictions should match
        pred1 = model.predict(n=5, series=train)
        pred2 = new_model.predict(n=5, series=train)
        np.testing.assert_array_almost_equal(pred1.values(), pred2.values())
```

### Property-Based Testing

```python
# tests/test_properties.py
from hypothesis import given, strategies as st
import numpy as np

@given(
    prices=st.lists(st.floats(min_value=1, max_value=100000), min_size=2, max_size=100),
)
def test_rsi_bounds(prices):
    """RSI should always be between 0 and 100."""
    rsi = calculate_rsi(np.array(prices))
    assert np.all((rsi >= 0) & (rsi <= 100))

@given(
    predicted=st.lists(st.floats(min_value=1, max_value=100000), min_size=10, max_size=100),
    actual=st.lists(st.floats(min_value=1, max_value=100000), min_size=10, max_size=100),
)
def test_mae_properties(predicted, actual):
    """MAE should have certain mathematical properties."""
    predicted = np.array(predicted[:min(len(predicted), len(actual))])
    actual = np.array(actual[:len(predicted)])

    mae = calculate_mae(predicted, actual)

    # MAE should be non-negative
    assert mae >= 0

    # MAE should be 0 for perfect predictions
    perfect_mae = calculate_mae(actual, actual)
    assert perfect_mae == 0
```

---

## ML-Specific Best Practices

### 1. Always Set Random Seeds

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Use at start of training
set_seed(42)
model.fit(train_data)
```

### 2. Log Everything

```python
from loguru import logger
import mlflow

def train_with_logging(model, train_data, val_data, config):
    """Train model with comprehensive logging."""

    # Log hyperparameters
    mlflow.log_params({
        "model_name": model.name,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    })

    # Log dataset info
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Val size: {len(val_data)}")

    # Train with progress tracking
    for epoch in range(config.epochs):
        train_loss = model.train_epoch(train_data)
        val_loss = model.validate(val_data)

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, step=epoch)

        logger.info(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

    # Log final model
    mlflow.pytorch.log_model(model, "model")
```

### 3. Version Your Data

```python
# Track data versions
DATA_VERSION = "v1.0"

def load_data(version: str = DATA_VERSION):
    """Load specific data version."""
    data_path = Path(f"data/{version}/btcusdt_1m.parquet")
    if not data_path.exists():
        raise ValueError(f"Data version {version} not found")

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded data version {version}: {len(df)} rows")
    return df
```

### 4. Save Model Metadata

```python
import json
from datetime import datetime

def save_model_with_metadata(model, path: Path, metadata: dict):
    """Save model with metadata."""
    # Save model
    model.save(path / "model.pth")

    # Save metadata
    metadata_full = {
        **metadata,
        "saved_at": datetime.now().isoformat(),
        "model_name": model.name,
        "is_fitted": model.is_fitted,
    }

    with open(path / "metadata.json", "w") as f:
        json.dump(metadata_full, f, indent=2)

# Usage
save_model_with_metadata(
    model,
    Path("models/nbeats_v1"),
    metadata={
        "train_size": len(train_data),
        "val_mae": 125.32,
        "hyperparameters": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
    },
)
```

### 5. Monitor Data Drift

```python
def check_data_drift(train_data, new_data):
    """Check if new data has drifted from training distribution."""
    from scipy import stats

    # Compare distributions
    train_mean = train_data['close'].mean()
    train_std = train_data['close'].std()
    new_mean = new_data['close'].mean()
    new_std = new_data['close'].std()

    # Statistical test
    t_stat, p_value = stats.ttest_ind(train_data['close'], new_data['close'])

    if p_value < 0.01:
        logger.warning(
            f"Data drift detected! Train: μ={train_mean:.2f}, σ={train_std:.2f} | "
            f"New: μ={new_mean:.2f}, σ={new_std:.2f} (p={p_value:.4f})"
        )
        return True

    return False
```

---

## Mini-Projects

### Project 1: Custom Feature Engineering (1-2 hours)

**Task**: Create 5 custom technical indicators

```python
class CustomFeatures:
    """Custom feature engineering."""

    @staticmethod
    def momentum_divergence(df: pd.DataFrame) -> pd.Series:
        """Calculate price-volume momentum divergence."""
        # TODO: Implement
        pass

    @staticmethod
    def volatility_regime(df: pd.DataFrame) -> pd.Series:
        """Classify into low/medium/high volatility regimes."""
        # TODO: Implement
        pass

    @staticmethod
    def support_resistance(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Identify support and resistance levels."""
        # TODO: Implement
        pass

# Test your features
df = load_data()
features = CustomFeatures()
df['momentum_div'] = features.momentum_divergence(df)
df['vol_regime'] = features.volatility_regime(df)

# Train model with new features
model = XGBoostModel()
model.fit(df)

# Did performance improve?
```

### Project 2: Ensemble Model (2-3 hours)

**Task**: Create an ensemble that combines multiple models

```python
class EnsembleModel(BaseModel):
    """Ensemble of multiple models."""

    def __init__(self, models: list[BaseModel], weights: list[float] | None = None):
        """Initialize ensemble.

        Args:
            models: List of models to ensemble
            weights: Optional weights for each model (default: equal weights)
        """
        super().__init__(name="ensemble")
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def fit(self, train_series, val_series=None):
        """Train all models."""
        for model in self.models:
            logger.info(f"Training {model.name}...")
            model.fit(train_series, val_series)
        self._is_fitted = True

    def predict(self, n: int, series: TimeSeries) -> TimeSeries:
        """Make ensemble predictions."""
        # TODO: Get predictions from all models
        # TODO: Combine using weights
        # TODO: Return weighted average
        pass

# Test ensemble
ensemble = EnsembleModel([
    NBEATSModel(),
    LSTMModel(),
    XGBoostModel(),
])
ensemble.fit(train_data)
predictions = ensemble.predict(5, train_data)
```

### Project 3: Real-Time Prediction Service (2-3 hours)

**Task**: Create a service that makes predictions every minute

```python
import asyncio
from datetime import datetime

class PredictionService:
    """Real-time prediction service."""

    def __init__(self, model: BaseModel, db: DatabaseManager):
        self.model = model
        self.db = db
        self.running = False

    async def run(self):
        """Run prediction loop."""
        self.running = True
        logger.info("Starting prediction service...")

        while self.running:
            try:
                # Wait until next minute
                await self._wait_until_next_minute()

                # Get latest data
                df = await self.db.get_ohlcv("BTCUSDT", "1m", limit=100)

                # Make prediction
                prediction = self.model.predict(n=5, series=df)

                # Log prediction
                logger.info(f"Prediction at {datetime.now()}: {prediction.values()}")

                # TODO: Store prediction
                # TODO: Send alert if prediction shows big move

            except Exception as e:
                logger.error(f"Prediction failed: {e}")

    async def _wait_until_next_minute(self):
        """Wait until the start of next minute."""
        now = datetime.now()
        seconds_until_next_minute = 60 - now.second
        await asyncio.sleep(seconds_until_next_minute)

    def stop(self):
        """Stop service."""
        self.running = False

# Run service
service = PredictionService(model, db)
asyncio.run(service.run())
```

---

## 🏆 FINAL CHALLENGE: Rebuild Price Stradamus

**Goal**: Delete the `src/` directory and rebuild Price Stradamus from scratch using only:
- This learning guide
- Official library documentation (Darts, pandas, PyTorch)
- Your notes

**No AI assistance allowed!**

### Phase 1: Foundation (2-4 hours)

1. **Project Setup**
   ```bash
   # Create directory structure
   mkdir -p src/price_stradamus/{config,data,models,evaluation,utils,cli}
   touch src/price_stradamus/__init__.py
   # ... create all __init__.py files
   ```

2. **Configuration**
   - Implement `config/settings.py` with Pydantic
   - Create `config/constants.py`

3. **Database Layer**
   - Implement `data/database.py` with asyncpg
   - Create schema
   - Implement save/load operations

**Checkpoint**: Can you store and retrieve data?

---

### Phase 2: Data Pipeline (2-3 hours)

4. **Data Fetching**
   - Implement `data/fetcher.py` with aiohttp
   - Handle Binance API rate limits
   - Implement batching for historical data

5. **Data Preprocessing**
   - Implement `data/preprocessor.py`
   - Validation functions
   - Missing value handling

6. **Feature Engineering**
   - Implement `data/features.py`
   - Add technical indicators with pandas-ta
   - Create lag features

**Checkpoint**: Can you fetch, validate, and engineer features?

---

### Phase 3: Models (4-6 hours)

7. **Base Model**
   - Implement `models/base.py` abstract class
   - Define interface: fit, predict, save, load

8. **Model Registry**
   - Implement `models/registry.py`
   - Factory pattern for model creation

9. **Implement Models**
   - Neural: N-BEATS, LSTM, TCN
   - Classical: ARIMA, Prophet
   - ML: XGBoost

**Checkpoint**: Can you train and save a model?

---

### Phase 4: Evaluation (2-3 hours)

10. **Metrics**
    - Implement `evaluation/metrics.py`
    - MAE, RMSE, MAPE, directional accuracy

11. **Backtesting**
    - Implement `evaluation/backtester.py`
    - Walk-forward validation
    - Result aggregation

**Checkpoint**: Can you evaluate model performance?

---

### Phase 5: CLI (2-3 hours)

12. **CLI Commands**
    - Implement `cli/data_commands.py` (fetch)
    - Implement `cli/model_commands.py` (train, predict, evaluate)
    - Implement `cli/info_commands.py` (list-models, stats)

13. **Utilities**
    - Implement `utils/logger.py` with Loguru
    - Implement `utils/helpers.py`

**Checkpoint**: Can you run all CLI commands?

---

### Phase 6: Testing & Polish (2-4 hours)

14. **Tests**
    - Write tests for data pipeline
    - Write tests for models
    - Write tests for evaluation

15. **Documentation**
    - Write README.md
    - Add docstrings everywhere
    - Create examples

16. **Final Verification**
    - Run full pipeline end-to-end
    - Train all models
    - Compare performance
    - Generate predictions

**Final Checkpoint**: Does everything work exactly like the original?

---

### Success Criteria

✅ All 8 models train and predict

✅ Walk-forward backtesting works
✅ All CLI commands functional
✅ Tests pass with >80% coverage
✅ No type errors (pyright passes)
✅ Code formatted (ruff passes)
✅ Documented (all functions have docstrings)

---

## What You've Mastered

After completing this learning guide, you now know:

### Python & Programming
- ✅ Python 3.13 syntax and features
- ✅ Type hints and type checking
- ✅ Async/await patterns
- ✅ Object-oriented programming
- ✅ Error handling and custom exceptions

### Data Science
- ✅ Pandas DataFrame operations
- ✅ Numpy numerical operations
- ✅ Data cleaning and validation
- ✅ Feature engineering
- ✅ Time series analysis

### Machine Learning
- ✅ Neural networks (N-BEATS, LSTM, TCN, TFT)
- ✅ Classical models (ARIMA, Prophet)
- ✅ ML models (XGBoost)
- ✅ Training and validation
- ✅ Hyperparameter tuning

### Engineering
- ✅ Project architecture
- ✅ Database operations (PostgreSQL + asyncpg)
- ✅ API integration (Binance)
- ✅ CLI development (Typer)
- ✅ Testing (pytest)
- ✅ Code quality (ruff, pyright)
- ✅ Logging (Loguru)

### ML Best Practices
- ✅ Evaluation metrics
- ✅ Walk-forward validation
- ✅ Avoiding overfitting
- ✅ Data versioning
- ✅ Model deployment

---

## Beyond This Guide

### Next Steps

1. **Experiment with new models**
   - Try Transformers
   - Implement Attention mechanisms
   - Explore AutoML (AutoGluon, FLAML)

2. **Add new features**
   - Sentiment analysis from Twitter/Reddit
   - Order book data
   - Market microstructure features

3. **Improve infrastructure**
   - Add monitoring (Grafana)
   - Implement alerts
   - Create web dashboard

4. **Deploy to production**
   - Dockerize application
   - Set up CI/CD
   - Deploy to cloud (AWS/GCP)

### Resources

- **Darts**: [unit8co.github.io/darts](https://unit8co.github.io/darts/)
- **PyTorch**: [pytorch.org/docs](https://pytorch.org/docs/)
- **Pandas**: [pandas.pydata.org](https://pandas.pydata.org/)
- **MLflow**: [mlflow.org](https://mlflow.org/)

---

## 🎉 Congratulations!

You've completed the Price Stradamus learning guide!

You went from TypeScript/React developer to Python ML engineer, capable of:
- Building production-grade ML systems
- Training and evaluating multiple model types
- Following best practices
- **Rebuilding entire systems from scratch**

**You're now ready to build your own ML projects!**

Good luck with the final challenge! 🚀📊🐍

---

*Module 12 Complete | Learning Guide Complete | Total: ~10,000 lines*

**Now go rebuild Price Stradamus and prove your mastery!**
