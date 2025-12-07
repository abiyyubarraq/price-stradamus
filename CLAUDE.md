# Price Stradamus - Claude Code Assistant Guide

---

## 📍 **CURRENT PHASE: Phase 1 - Core System**

**Scale**: Personal use → <1k users
**Focus**: Great BTC predictions
**Status**: ✅ 8 models implemented, backtesting ready

### How to Use This Guide

This guide is **Phase 1 focused** - everything you need for daily development NOW. For advanced topics (Phase 2-6), see [CLAUDE-ADVANCED.md](CLAUDE-ADVANCED.md).

#### For Daily Development (Phase 1) - **START HERE** 👈

1. **Quick Start**: [QUICK-START.md](QUICK-START.md) (5-minute setup)
2. **Code Standards**: See sections below
3. **Model Development**: [context/models.md](context/models.md)
4. **Evaluation**: [context/evaluation.md](context/evaluation.md)
5. **Agent Guides**: [.claude/agents/](.claude/agents/) (specialized AI assistants)

#### For Planning Future Phases

- **Advanced Guide**: [CLAUDE-ADVANCED.md](CLAUDE-ADVANCED.md) (Phase 2-6 content)
- **Phase Overview**: [CLAUDE-PHASE-GUIDE.md](CLAUDE-PHASE-GUIDE.md)
- **Roadmap**: [context/roadmap.md](context/roadmap.md)

---

### Phase Legend

Throughout this document, sections are marked with phase indicators:

- **[PHASE 1]** - ✅ Use these NOW (Personal use, <100 users)
- **[PHASE 2]** - ⏳ AutoML & Optimization (<1k users) - See [CLAUDE-ADVANCED.md](CLAUDE-ADVANCED.md)
- **[PHASE 3]** - ⏳ Production Hardening (<10k users) - See [CLAUDE-ADVANCED.md](CLAUDE-ADVANCED.md)
- **[PHASE 4]** - ⏳ API & Multi-User (<100k users) - See [CLAUDE-ADVANCED.md](CLAUDE-ADVANCED.md)
- **[PHASE 5]** - ⏳ Real-Time Streaming (<500k users) - See [CLAUDE-ADVANCED.md](CLAUDE-ADVANCED.md)
- **[PHASE 6]** - ⏳ Cloud Deployment (1M+ users) - See [CLAUDE-ADVANCED.md](CLAUDE-ADVANCED.md)

**If unsure, assume Phase 1. Focus on predictions first!**

---

## Project Overview

**Price Stradamus** is a professional-grade Bitcoin price prediction system that uses multiple machine learning models for multi-step time series forecasting. The system supports configurable timeframes starting with 1-minute candles and provides predictions for multiple future candles.

### Purpose
- Predict Bitcoin prices using state-of-the-art ML models
- Compare multiple model architectures (Neural, Classical, ML)
- Provide comprehensive backtesting and evaluation
- Support both manual and automated model selection (AutoML)

### Goals
- Achieve high directional accuracy (>55%)
- Minimize prediction error (MAE, RMSE)
- Build scalable, maintainable architecture
- Enable easy experimentation with new models
- Maintain professional code quality standards

---

## Quick Start [PHASE 1]

```bash
# Setup environment
uv venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e ".[dev]"

# Start database
cd docker && docker-compose up -d postgres

# Fetch data
python -m price_stradamus.cli.commands fetch --days 30

# Train model
python -m price_stradamus.cli.commands train --model nbeats

# Make predictions
python -m price_stradamus.cli.commands predict --steps 5

# Run tests
uv run pytest --cov
```

---

## Architecture Overview [PHASE 1]

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Binance   │─────▶│  PostgreSQL  │─────▶│  Features   │
│     API     │      │   Database   │      │  Engine     │
└─────────────┘      └──────────────┘      └─────────────┘
                                                   │
                                                   ▼
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│ Predictions │◀─────│    Models    │◀─────│   Training  │
│  Storage    │      │  (Multiple)  │      │     Data    │
└─────────────┘      └──────────────┘      └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Evaluation  │
                    │  Backtester  │
                    └──────────────┘
```

### Key Components
- **Data Layer**: Binance API client, PostgreSQL storage, feature engineering
- **Model Layer**: Neural (N-BEATS, LSTM, TCN, TFT), Classical (ARIMA, Prophet), ML (XGBoost, RF)
- **Evaluation Layer**: Metrics calculation, walk-forward backtesting
- **CLI Layer**: User interface for all operations

---

## Code Standards (State-of-the-Art 2025) [PHASE 1]

### Python Environment
- **Python Version**: 3.13.9 (strict requirement)
- **Package Manager**: `uv` (preferred) or `pip`
- **Virtual Environment**: Always use `.venv/`

### Code Quality Tools
- **Formatter**: Ruff (replaces Black + isort + flake8)
- **Type Checker**: Pyright in strict mode
- **Linter**: Ruff with comprehensive rule set
- **Testing**: pytest with pytest-cov, minimum 80% coverage
- **Pre-commit**: Automated checks before commits

### Code Style

#### Type Hints
- **Mandatory** for all function signatures
- Use `from __future__ import annotations` for forward references
- Prefer `list[str]` over `List[str]` (PEP 585)
- Use `| None` instead of `Optional[]` (PEP 604)

```python
from __future__ import annotations

def predict_prices(
    data: list[float],
    steps: int = 5,
    model_name: str | None = None,
) -> list[float]:
    """Predict future prices."""
    ...
```

#### Docstrings
- **Google Style** for all public functions and classes
- Include Args, Returns, Raises sections

```python
def calculate_rsi(prices: list[float], period: int = 14) -> list[float]:
    """Calculate Relative Strength Index.

    Args:
        prices: List of closing prices
        period: RSI calculation period (default: 14)

    Returns:
        List of RSI values (0-100 scale)

    Raises:
        ValueError: If period is less than 2
    """
    ...
```

#### Line Length
- **88 characters** (Ruff default)
- Let formatter handle line breaks

#### Imports
1. Standard library
2. Third-party packages
3. Local imports
4. Separate groups with blank lines

```python
from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import torch
from darts import TimeSeries

from price_stradamus.config import settings
from price_stradamus.utils import logger
```

### Coding Conventions

#### Data Structures
- Use **Pydantic v2** for data validation
- Use **dataclasses** for simple data containers
- Use **Enum** for fixed choices

```python
from enum import Enum
from pydantic import BaseModel, Field

class Timeframe(str, Enum):
    """Supported timeframes."""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"

class ModelConfig(BaseModel):
    """Model configuration."""
    name: str = Field(..., description="Model name")
    input_chunk_length: int = Field(60, gt=0)
    output_chunk_length: int = Field(5, gt=0)
    learning_rate: float = Field(0.001, gt=0)
```

#### Functions
- Prefer **pure functions** (no side effects)
- Keep functions **under 20 lines** when practical
- Use **meaningful names** (no abbreviations except common ones)
- One function = one responsibility

```python
# Good
def calculate_moving_average(prices: list[float], window: int) -> list[float]:
    """Calculate simple moving average."""
    return [sum(prices[i:i+window]) / window for i in range(len(prices) - window + 1)]

# Bad
def calc_ma_and_save(prices, w, file):  # Multiple responsibilities
    ma = [sum(prices[i:i+w]) / w for i in range(len(prices) - w + 1)]
    with open(file, 'w') as f:  # Side effect
        f.write(str(ma))
    return ma
```

#### Control Flow
- Use `match` statements for multiple conditions (Python 3.10+)
- Prefer early returns over nested if-else
- Use guard clauses

```python
# Good - match statement
match timeframe:
    case Timeframe.ONE_MINUTE:
        return fetch_minute_data()
    case Timeframe.ONE_HOUR:
        return fetch_hourly_data()
    case _:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

# Good - early return
def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Data is empty")
    if data.isnull().any().any():
        raise ValueError("Data contains null values")
    return data
```

#### Error Handling
- Create **custom exception hierarchy**
- Always catch specific exceptions
- Use context managers for resources

```python
class PriceStradamusError(Exception):
    """Base exception for Price Stradamus."""

class DataFetchError(PriceStradamusError):
    """Error fetching data from API."""

class ModelTrainingError(PriceStradamusError):
    """Error during model training."""

# Usage
try:
    data = await fetch_binance_data(symbol)
except aiohttp.ClientError as e:
    raise DataFetchError(f"Failed to fetch {symbol}") from e
```

#### Async/Await
- Use `async`/`await` for **all I/O operations**
- Database queries, API calls, file I/O
- Use `asyncio.gather()` for parallel operations

```python
async def fetch_multiple_symbols(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple symbols in parallel."""
    tasks = [fetch_binance_data(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    return dict(zip(symbols, results))
```

#### Path Handling
- **Always use `pathlib.Path`**
- Never use string concatenation for paths

```python
from pathlib import Path

# Good
model_path = Path("models") / f"{model_name}.pth"
model_path.parent.mkdir(parents=True, exist_ok=True)

# Bad
model_path = "models/" + model_name + ".pth"
```

### File Naming Conventions
- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Private**: `_leading_underscore`

```python
# constants.py
DEFAULT_LOOKBACK_WINDOW = 60
MAX_PREDICTION_STEPS = 100

# models/nbeats.py
class NBEATSModel:
    def __init__(self):
        self._is_trained = False

    def fit(self, data: TimeSeries) -> None:
        ...
```

### Logging
- Use **Loguru** for all logging
- Structured logging with context
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

```python
from loguru import logger

logger.info("Fetching data for {symbol}", symbol="BTCUSDT")
logger.warning("Model converged slowly, took {epochs} epochs", epochs=150)
logger.error("Training failed: {error}", error=str(e))
```

### Testing
- **Minimum 80% coverage**
- Test file naming: `test_*.py`
- Use fixtures for common setup
- Mock external dependencies (API calls, database)

```python
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
async def mock_binance_client():
    client = AsyncMock()
    client.fetch_klines.return_value = mock_ohlcv_data()
    return client

async def test_fetch_data(mock_binance_client):
    """Test data fetching."""
    data = await fetch_binance_data("BTCUSDT", client=mock_binance_client)
    assert len(data) > 0
    assert "close" in data.columns
```

---

## Git Conventions [PHASE 1]

### Commit Messages
Use **Conventional Commits**:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks
- `perf:` Performance improvement

```bash
feat: add N-BEATS model implementation
fix: handle missing data in feature engineering
docs: update model comparison in README
refactor: simplify data fetcher error handling
test: add backtester integration tests
chore: update dependencies to latest versions
```

### Branch Naming
- `feature/` - New features
- `bugfix/` - Bug fixes
- `hotfix/` - Urgent fixes for production
- `refactor/` - Code refactoring
- `docs/` - Documentation updates

```bash
feature/add-transformer-model
bugfix/fix-database-connection
hotfix/patch-api-rate-limit
refactor/simplify-evaluation-metrics
docs/add-model-comparison-guide
```

---

## Project-Specific Guidelines [PHASE 1]

### Model Development
1. **Always extend `BaseModel` interface** (src/price_stradamus/models/base.py)
2. Use **Darts library** for time series models when possible
3. Set **reproducible seeds** for experiments
4. **Log all hyperparameters** and results
5. Compare against **baseline models**

### Data Pipeline
1. **Validate data integrity** after fetching
2. Use **async for all I/O** operations
3. Implement **retry logic** with exponential backoff
4. **Cache computed features** when possible
5. Always use **walk-forward validation** (no lookahead bias)

### Feature Engineering
1. Use **pandas-ta** for technical indicators
2. **Document all features** with descriptions
3. Handle **missing values explicitly** (forward fill, interpolation)
4. **Normalize features** before training
5. Create **lag features** for sequential patterns

### Evaluation
1. Report **multiple metrics** (MAE, RMSE, MAPE, directional accuracy)
2. Use **walk-forward validation** for backtesting
3. Test **statistical significance** of results
4. Consider **transaction costs** in backtests
5. Be **skeptical of "too good" results** (likely overfitting)

### Performance
1. Use **GPU when available** (`torch.cuda.is_available()`)
2. **Batch operations** when possible
3. Use **connection pooling** for database
4. **Cache expensive computations**
5. **Profile slow code** (cProfile, line_profiler)

---

## Key Commands [PHASE 1]

### Development
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check --fix .

# Type check
uv run pyright

# Run tests
uv run pytest --cov

# Run specific test
uv run pytest tests/test_models/test_nbeats.py -v

# Run tests with markers
uv run pytest -m "not slow"
```

### Docker
```bash
# Start all services
docker-compose up -d

# Start only database
docker-compose up -d postgres

# View logs
docker-compose logs -f postgres

# Stop services
docker-compose down

# Remove volumes (caution: deletes data)
docker-compose down -v
```

### CLI Usage
```bash
# Fetch historical data
python -m price_stradamus.cli.commands fetch --symbol BTCUSDT --timeframe 1m --days 30

# Train a model
python -m price_stradamus.cli.commands train --model nbeats --epochs 100

# Make predictions
python -m price_stradamus.cli.commands predict --model nbeats --steps 5

# Run backtesting
python -m price_stradamus.cli.commands evaluate --model nbeats --test-size 0.2

# Compare models
python -m price_stradamus.cli.commands compare --models nbeats lstm tcn

# Run AutoML
python -m price_stradamus.cli.commands automl --time-budget 3600
```

### Database
```bash
# Create migration
alembic revision --autogenerate -m "Add predictions table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# View current version
alembic current
```

---

## Common Pitfalls to Avoid [PHASE 1]

### Time Series Specific
1. **Lookahead bias**: Never use future data for predictions
2. **Data leakage**: Keep train/test splits temporal
3. **Non-stationarity**: Check for trends and seasonality
4. **Overfitting**: Use walk-forward validation
5. **Ignoring transaction costs**: Include in backtests

### Code Quality
1. **Missing type hints**: All functions must be typed
2. **No docstrings**: Document all public APIs
3. **Magic numbers**: Use named constants
4. **Catching all exceptions**: Catch specific exceptions
5. **Not using async**: I/O operations should be async

### File Organization & Refactoring
1. **Premature splitting**: Don't split files <800 lines "just because"
2. **Over-optimization for tools**: Optimize for human understanding first, tools second
3. **Arbitrary rules**: No hard "500 line limit" - context matters
4. **Tiny file proliferation**: Many 50-line files is worse than one well-organized 600-line file
5. **Splitting without pain**: Only refactor when you feel actual navigation pain

**Rule of Thumb**: If you can easily find what you need, the file is fine. Don't refactor based on line count alone.

### Model Development
1. **Not setting seeds**: Results must be reproducible
2. **No baseline comparison**: Always compare to simple models
3. **Only optimizing one metric**: Report multiple metrics
4. **Not saving models**: Always save trained models
5. **No hyperparameter logging**: Track all experiments

---

## Security Basics [PHASE 1]

### Secrets Management
**Never commit secrets to version control.** Use environment variables.

```python
import os
from pydantic import SecretStr
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with secure secret handling."""

    # Use SecretStr to prevent logging
    database_password: SecretStr
    binance_api_key: SecretStr
    binance_api_secret: SecretStr

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

# Usage
settings = Settings()
# Password is masked in logs
print(settings.database_password)  # SecretStr('**********')
# Explicitly reveal when needed
password = settings.database_password.get_secret_value()
```

### Input Validation
Always validate and sanitize user input:

```python
class InputValidator:
    """Validate and sanitize user input."""

    VALID_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT"})
    VALID_TIMEFRAMES = frozenset({"1m", "5m", "15m", "1h", "4h", "1d"})

    @classmethod
    def validate_symbol(cls, symbol: str) -> str:
        """Validate trading symbol."""
        symbol_upper = symbol.upper().strip()
        if symbol_upper not in cls.VALID_SYMBOLS:
            raise ValueError(f"Invalid symbol: {symbol}")
        return symbol_upper

    @classmethod
    def validate_positive_int(cls, value: int, name: str, max_value: int = 10000) -> int:
        """Validate positive integer within bounds."""
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        if value > max_value:
            raise ValueError(f"{name} must not exceed {max_value}")
        return value
```

### SQL Injection Prevention
**Always use parameterized queries:**

```python
# ✅ GOOD: Parameterized query
async def get_predictions(symbol: str, model_name: str, limit: int = 100):
    query = """
        SELECT *
        FROM ml_data.predictions
        WHERE symbol = $1 AND model_name = $2
        ORDER BY timestamp DESC
        LIMIT $3
    """
    async with pool.acquire() as conn:
        return await conn.fetch(query, symbol, model_name, limit)

# ❌ BAD: String formatting (SQL injection vulnerability!)
# query = f"SELECT * FROM predictions WHERE symbol = '{symbol}'"  # NEVER DO THIS!
```

**For advanced security topics** (OWASP Top 10, authentication, encryption, etc.), see [CLAUDE-ADVANCED.md](CLAUDE-ADVANCED.md).

---

## Documentation Management [PHASE 1]

### How This Documentation is Organized

To keep files manageable, we've split documentation by usage phase:

#### Core Documentation (You're Here)
- **[CLAUDE.md](CLAUDE.md)** (this file) - Phase 1 essentials (~1400 lines)
  - Everything needed for daily development
  - Code standards, conventions, common pitfalls
  - Security basics

#### Advanced Topics (Phase 2-6)
- **[CLAUDE-ADVANCED.md](CLAUDE-ADVANCED.md)** (~1600 lines)
  - Dependency management, OWASP security
  - Performance testing, breaking changes
  - Team collaboration patterns

#### Specialized Agents
- **[.claude/agents/*.md](.claude/agents/)** - AI assistant guides
  - Each ~500-2000 lines
  - Specialized knowledge domains
  - See [Documentation Navigation](#documentation-navigation) below

#### Context Documentation
- **[context/*.md](context/)** - Domain-specific knowledge
  - Models, evaluation, data pipeline
  - Architecture, roadmap, research notes
  - Each ~1000-2000 lines

### Documentation Guidelines for Claude Code

**How I (Claude Code) Use These Files:**

#### Reading Strategy
1. **First mention**: Read entire file to understand structure
2. **Subsequent uses**: Grep/search for specific sections
3. **Large files** (>1500 lines): May need 2-3 reads for full context

#### File Size Guidelines for Documentation
- **<1500 lines**: ✅ Excellent - Single cohesive topic
- **1500-2500 lines**: ⚠️ Large but manageable if well-structured
- **>2500 lines**: 🔴 Consider splitting by natural boundaries

**Why we split CLAUDE.md**:
- Original was 2866 lines mixing Phase 1 + Phase 2-6 content
- Phase 1 users (NOW) don't need 1600 lines of future content
- Faster navigation for both humans and AI

#### When to Split Documentation

Split documentation files when:
- ✅ **>2000 lines** AND multiple distinct topics
- ✅ **Natural phase boundaries** exist (like Phase 1 vs Phase 2-6)
- ✅ **Users constantly search** for specific sections

Don't split when:
- ❌ File <1500 lines with single topic
- ❌ No clear boundaries (would create many cross-references)
- ❌ Just for "organization" without navigation pain

### Documentation Navigation

```
price-stradamus/
├── CLAUDE.md                    # You are here - Phase 1 essentials
├── CLAUDE-ADVANCED.md           # Phase 2-6 advanced topics
├── QUICK-START.md               # 5-minute setup guide
├── CLAUDE-PHASE-GUIDE.md        # Phase progression guide
│
├── .claude/agents/              # Specialized AI assistants
│   ├── senior-software-engineer.md  # Architecture decisions
│   ├── code-reviewer.md             # Code review checklists
│   ├── ml-researcher.md             # Model experimentation
│   ├── data-engineer.md             # Data pipeline expertise
│   ├── debugger.md                  # Debugging strategies
│   ├── devops.md                    # Deployment & ops
│   └── quant-analyst.md             # Trading & backtesting
│
└── context/                     # Domain knowledge
    ├── architecture.md          # System design
    ├── models.md                # Model documentation
    ├── data-pipeline.md         # Data flow details
    ├── evaluation.md            # Metrics & backtesting
    ├── roadmap.md               # Project phases
    ├── research-notes.md        # ML research
    ├── glossary.md              # Terminology
    └── LEARNING-GUIDE.md        # Learning resources
```

### Quick Reference

| Need to...                     | Check...                                |
|--------------------------------|-----------------------------------------|
| Setup project                  | [QUICK-START.md](QUICK-START.md)        |
| Write code (Phase 1)           | [CLAUDE.md](CLAUDE.md) (this file)      |
| Plan future phases             | [CLAUDE-ADVANCED.md](CLAUDE-ADVANCED.md)|
| Make architecture decision     | [.claude/agents/senior-software-engineer.md](.claude/agents/senior-software-engineer.md) |
| Review code quality            | [.claude/agents/code-reviewer.md](.claude/agents/code-reviewer.md) |
| Understand system design       | [context/architecture.md](context/architecture.md) |
| Learn about models             | [context/models.md](context/models.md)  |
| Debug issues                   | [.claude/agents/debugger.md](.claude/agents/debugger.md) |
| Understand terminology         | [context/glossary.md](context/glossary.md) |

---

## Resources [ALL PHASES]

### Documentation
- [Darts Documentation](https://unit8co.github.io/darts/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [pandas-ta Documentation](https://github.com/twopirllc/pandas-ta)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)

### Papers
- N-BEATS: [Neural Basis Expansion Analysis for Time Series](https://arxiv.org/abs/1905.10437)
- TFT: [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363)
- TCN: [Temporal Convolutional Networks](https://arxiv.org/abs/1803.01271)

### Context Files
- [context/architecture.md](context/architecture.md) - System architecture
- [context/models.md](context/models.md) - Model documentation
- [context/data-pipeline.md](context/data-pipeline.md) - Data flow
- [context/evaluation.md](context/evaluation.md) - Evaluation methodology
- [context/roadmap.md](context/roadmap.md) - Project roadmap

---

## Getting Help [PHASE 1]

1. Check [README.md](README.md) for quick start guide
2. Review context files for detailed documentation
3. Check [context/glossary.md](context/glossary.md) for terminology
4. Review test files for usage examples
5. Check logs in `logs/` directory

---

## Contributing [PHASE 1]

1. Create a feature branch
2. Write tests for new features
3. Ensure all tests pass (`pytest --cov`)
4. Run linting (`ruff check --fix .`)
5. Run type checking (`pyright`)
6. Format code (`ruff format .`)
7. Write conventional commit message
8. Create pull request

---

*Last Updated: 2025-12-07*
*Version: 2.0 - Restructured for Phase 1 Focus*
