# Price Stradamus - Quick Start Guide

**Current Product Phase**: Phase 1 - Core System ✅
**Current Scale**: Personal use (just you, local laptop)
**Goal**: Build a Bitcoin price prediction system with great BTC predictions

---

## What's Actually Implemented

✅ **8 Forecasting Models**
- Neural: N-BEATS, LSTM, TCN, TFT
- Classical: ARIMA, Prophet
- ML: XGBoost, Random Forest

✅ **Data Pipeline**
- Fetch from Binance API (async)
- Store in PostgreSQL
- 50+ technical indicators (pandas-ta)
- Feature engineering

✅ **Evaluation System**
- Walk-forward validation
- Multiple metrics (MAE, RMSE, MAPE, directional accuracy)
- Backtesting framework

✅ **Developer Tools**
- CLI interface
- Ruff formatter & linter
- Pyright type checking
- Pytest with coverage
- Docker PostgreSQL

---

## What's NOT Implemented Yet

**Not in Phase 1 (Core System)**:
❌ AutoML (auto-sklearn) - Phase 2
❌ Bayesian Tournament - Phase 3
❌ Advanced ensembles - Later phases

**Not at Personal Scale**:
❌ REST API, authentication - When you have users
❌ Kubernetes, cloud deployment - When scaling to 100k+ users
❌ Load balancers, service mesh - Enterprise scale
❌ Distributed tracing - Production debugging
❌ CI/CD pipelines - Team collaboration
❌ Feature stores - When you have many models

**You don't need these yet.** Focus on getting great predictions first!

---

## Setup (5 Minutes)

```bash
# 1. Clone and setup environment
git clone https://github.com/yourusername/price-stradamus.git
cd price-stradamus
uv venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# 2. Start database
cd docker && docker-compose up -d postgres && cd ..

# 3. Configure (optional - defaults work)
cp .env.example .env
# Edit .env if needed

# 4. Fetch data
python -m price_stradamus.cli.commands fetch --days 30

# 5. Train a model
python -m price_stradamus.cli.commands train --model nbeats

# 6. Make predictions
python -m price_stradamus.cli.commands predict --steps 5

# Done! 🎉
```

---

## Daily Development Workflow

### 1. Run Code Quality Checks

```bash
# Format, lint, type check, test (runs in ~30 seconds)
uv run ruff format . && \
uv run ruff check --fix . && \
uv run pyright && \
uv run pytest --cov
```

**Goal**: Keep code clean and catch bugs early.

### 2. Experiment with Models

```bash
# Try different models
python -m price_stradamus.cli.commands train --model lstm --epochs 50
python -m price_stradamus.cli.commands train --model tcn --epochs 50

# Compare results
python -m price_stradamus.cli.commands compare --models nbeats lstm tcn

# Evaluate performance
python -m price_stradamus.cli.commands evaluate --model nbeats
```

**Goal**: Find the best model for BTC prediction.

### 3. Improve Features

```python
# Add new technical indicator
from price_stradamus.data.features import FeatureEngine

engine = FeatureEngine()
# Add your custom indicator
engine.add_feature("my_indicator", calculate_my_indicator)
```

**Goal**: Better features = better predictions.

### 4. Backtest Strategies

```python
from price_stradamus.evaluation.backtester import WalkForwardBacktester

backtester = WalkForwardBacktester(
    model=model,
    data=data,
    initial_train_size=0.7,
    step_size=100,
)
results = backtester.run()
```

**Goal**: Validate models work on unseen data.

---

## Project Structure (Simplified)

```
price-stradamus/
├── src/price_stradamus/
│   ├── config/              # Configuration (settings.py)
│   ├── data/                # Data fetching, features
│   ├── models/              # 8 ML models
│   │   ├── neural/          # N-BEATS, LSTM, TCN, TFT
│   │   ├── classical/       # ARIMA, Prophet
│   │   └── ml/              # XGBoost, Random Forest
│   ├── evaluation/          # Backtesting, metrics
│   ├── utils/               # Logging, helpers
│   └── cli/                 # Command-line interface
├── tests/                   # Test suite
├── docker/                  # PostgreSQL container
└── context/                 # Extended documentation
```

---

## Code Style (Quick Reference)

### Type Hints (Required)

```python
from __future__ import annotations

def predict_price(
    data: list[float],
    steps: int = 5,
) -> list[float]:
    """Predict future prices."""
    ...
```

### Docstrings (Google Style)

```python
def calculate_rsi(prices: list[float], period: int = 14) -> list[float]:
    """Calculate Relative Strength Index.

    Args:
        prices: List of closing prices
        period: RSI calculation period (default: 14)

    Returns:
        List of RSI values (0-100 scale)
    """
    ...
```

### Async for I/O

```python
async def fetch_data(symbol: str) -> pd.DataFrame:
    """Always use async for database/API calls."""
    async with fetcher:
        return await fetcher.fetch_ohlcv(symbol)
```

### Path Handling

```python
from pathlib import Path

# Always use Path, never string concatenation
model_path = Path("models") / f"{model_name}.pth"
model_path.parent.mkdir(parents=True, exist_ok=True)
```

---

## Performance Budget (Phase 1)

These are **reasonable targets** for personal use:

| Operation | Target | Current |
|-----------|--------|---------|
| Model inference | <100ms | ✅ 50-100ms |
| Data fetch (1000 candles) | <2s | ✅ ~1.2s |
| Feature generation | <5s | ✅ ~2.5s |
| Training (50 epochs) | <10min | ✅ 4-8min (GPU) |

**Don't optimize prematurely.** These targets handle personal use easily.

---

## Common Tasks

### Add a New Model

```python
from price_stradamus.models.base import BaseModel

class MyAwesomeModel(BaseModel):
    """My custom forecasting model."""

    def fit(self, data: TimeSeries) -> None:
        """Train the model."""
        # Your training logic
        pass

    def predict(self, steps: int) -> TimeSeries:
        """Make predictions."""
        # Your prediction logic
        pass

    def save(self, path: Path) -> None:
        """Save model to disk."""
        pass

    def load(self, path: Path) -> None:
        """Load model from disk."""
        pass
```

### Add a New Feature

```python
from price_stradamus.data.features import FeatureEngine

engine = FeatureEngine()

@engine.register_feature("my_feature")
def calculate_my_feature(df: pd.DataFrame) -> pd.Series:
    """Calculate custom feature."""
    return df["close"].rolling(20).mean()
```

### Run Specific Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_models/test_nbeats.py -v

# Specific test
pytest tests/test_models/test_nbeats.py::test_training -v

# With coverage
pytest --cov=price_stradamus --cov-report=html
```

---

## Troubleshooting

### Database Connection Failed

```bash
# Check if PostgreSQL is running
docker-compose ps

# Restart if needed
cd docker && docker-compose restart postgres
```

### CUDA Out of Memory

```python
# In .env, reduce batch size
BATCH_SIZE=16  # or 8

# Or use CPU
DEVICE=cpu
```

### Import Errors

```bash
# Reinstall in editable mode
uv pip install -e ".[dev]"
```

### Slow Training

```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
# Visit: https://pytorch.org/get-started/locally/
```

---

## Key Metrics to Watch

### Model Performance

- **MAE** (Mean Absolute Error): Lower is better
- **Directional Accuracy**: Target >52% (better than random)
- **RMSE**: Lower is better

### What "Good" Looks Like

| Metric | Poor | OK | Good | Great |
|--------|------|-----|------|-------|
| **MAE ($)** | >$100 | $50-100 | $30-50 | <$30 |
| **Dir. Acc.** | <50% | 50-52% | 52-55% | >55% |
| **Training Time** | >30min | 10-30min | 5-10min | <5min |

**Realistic Expectation**: Getting >55% directional accuracy is very challenging. 52-54% is already strong for trading.

---

## When to Add More Features (Phase 2+)

You're ready for **Phase 2 (AutoML)** when:

✅ You have a model with consistent >52% directional accuracy
✅ You've backtested over multiple market conditions
✅ Manual hyperparameter tuning is tedious
✅ You want automated model selection

You need **higher scale infrastructure** when:

✅ You have >100 actual users asking for access
✅ Local laptop can't handle the load
✅ You need API endpoints for other apps
✅ You're deploying to production

**For now**: Stay at Phase 1, Personal Scale. Perfect the predictions!

---

## Getting Help

### Documentation

- **This file**: Phase 1 essentials
- [context/LEARNING-GUIDE.md](context/LEARNING-GUIDE.md): **📚 Complete learning guide** - Understand how everything works from scratch
- [README.md](README.md): Full project overview
- [CLAUDE.md](CLAUDE.md): Comprehensive development guide (detailed but long)
- [src/price_stradamus/cli/CLI-COMMANDS.md](src/price_stradamus/cli/CLI-COMMANDS.md): **Complete CLI commands reference** with Bash/PowerShell examples
- [context/models.md](context/models.md): Model details
- [context/evaluation.md](context/evaluation.md): Evaluation methodology

### When Stuck

1. Check error logs in `logs/` directory
2. Search existing tests for examples
3. Review model documentation in `context/`
4. Check [CLAUDE.md](CLAUDE.md) for detailed patterns

---

## Next Steps

### Week 1: Get Comfortable

- [ ] Run all 8 models
- [ ] Compare their performance
- [ ] Understand evaluation metrics
- [ ] Read [context/models.md](context/models.md)

### Week 2-3: Experiment

- [ ] Try different hyperparameters
- [ ] Add custom features
- [ ] Run longer backtests
- [ ] Profile slow code

### Week 4+: Improve

- [ ] Focus on directional accuracy
- [ ] Optimize best-performing models
- [ ] Test on different timeframes
- [ ] Document your findings

---

## Remember

🎯 **Goal**: Great BTC price predictions
📊 **Target**: >52% directional accuracy
📦 **Product Phase**: Phase 1 - Core System
🖥️ **Scale**: Personal use (just you, local laptop)
🚫 **Not Goals**: AutoML (Phase 2), Cloud infrastructure, Microservices

**Two separate concepts:**
- **Product Phases** (1-7): What features to build (Core → AutoML → Bayesian → ...)
- **Scale Levels**: How many users (Personal → Team → Production → Cloud)

**Keep it simple. Make it work. Then make it better.**

---

*Last Updated: 2025-12-03*
*Product Phase: 1 - Core System*
*Scale: Personal Use*
