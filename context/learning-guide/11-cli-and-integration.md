# Module 11: CLI and Integration

**Duration:** 2-3 hours | **Difficulty:** Intermediate | **Prerequisites:** Modules 01-10

## 🎯 Learning Objectives

After this module, you will:
- Master all Price Stradamus CLI commands
- Understand end-to-end workflows
- Know how to debug issues
- Integrate all components together
- Use logging effectively
- Automate common tasks

---

## CLI Framework: Typer

Price Stradamus uses **Typer** for the command-line interface.

### Why Typer?

```python
# Typer is like Express.js for CLI
# TypeScript Express:
app.get('/train', async (req, res) => {
  const model = req.query.model;
  // ...
});

# Python Typer:
@app.command()
def train(model: str = "nbeats"):
    """Train a model."""
    # ...
```

**Benefits:**
- Type hints = automatic validation
- Automatic help generation
- Beautiful error messages
- Easy to test

### Basic Typer Structure

```python
import typer
from typing import Optional

app = typer.Typer()

@app.command()
def hello(
    name: str = typer.Argument(..., help="Your name"),
    greeting: str = typer.Option("Hello", help="Greeting to use"),
    loud: bool = typer.Option(False, "--loud", "-l", help="Shout the greeting"),
):
    """Say hello to someone."""
    message = f"{greeting}, {name}!"
    if loud:
        message = message.upper()
    typer.echo(message)

if __name__ == "__main__":
    app()

# Usage:
# python script.py John
# python script.py John --greeting "Hi"
# python script.py John --loud
```

---

## Command Overview

Price Stradamus has 5 main command groups:

### 1. Data Commands

```bash
# Fetch historical data
python -m price_stradamus.cli fetch --days 30

# Fetch specific symbol and timeframe
python -m price_stradamus.cli fetch \
    --symbol BTCUSDT \
    --timeframe 1m \
    --days 7

# Fetch date range
python -m price_stradamus.cli fetch \
    --start-date 2025-01-01 \
    --end-date 2025-01-31
```

### 2. Model Commands

```bash
# Train a model
python -m price_stradamus.cli train --model nbeats

# Train with custom parameters
python -m price_stradamus.cli train \
    --model nbeats \
    --epochs 200 \
    --batch-size 64 \
    --learning-rate 0.001

# Make predictions
python -m price_stradamus.cli predict \
    --model nbeats \
    --steps 5

# Evaluate model
python -m price_stradamus.cli evaluate \
    --model nbeats \
    --test-size 0.2
```

### 3. Comparison Commands

```bash
# Compare multiple models
python -m price_stradamus.cli compare \
    --models nbeats lstm tcn

# Compare with walk-forward validation
python -m price_stradamus.cli compare \
    --models nbeats xgboost prophet \
    --walk-forward \
    --n-splits 5
```

### 4. Info Commands

```bash
# List available models
python -m price_stradamus.cli list-models

# Show model details
python -m price_stradamus.cli info --model nbeats

# Show database stats
python -m price_stradamus.cli db-stats
```

### 5. AutoML Commands (Future)

```bash
# Run AutoML optimization
python -m price_stradamus.cli automl --time-budget 3600
```

---

## Command Implementation Details

### Fetch Command

```python
# src/price_stradamus/cli/data_commands.py
import typer
import asyncio
from datetime import datetime, timedelta
from price_stradamus.data.fetcher import BinanceDataFetcher
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.preprocessor import DataPreprocessor

app = typer.Typer()

@app.command()
def fetch(
    symbol: str = typer.Option("BTCUSDT", help="Trading symbol"),
    timeframe: str = typer.Option("1m", help="Timeframe (1m, 5m, 1h, etc)"),
    days: int = typer.Option(30, help="Number of days to fetch"),
    start_date: str = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, help="End date (YYYY-MM-DD)"),
):
    """Fetch historical OHLCV data from Binance."""
    typer.echo(f"📊 Fetching {symbol} {timeframe} data...")

    # Parse dates
    if start_date and end_date:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    else:
        end = datetime.now()
        start = end - timedelta(days=days)

    # Run async function
    result = asyncio.run(_fetch_async(symbol, timeframe, start, end))

    # Display results
    typer.echo(f"✅ Fetched {result['total_candles']} candles")
    typer.echo(f"📝 Inserted {result['inserted']} new records")
    typer.echo(f"⏭️  Skipped {result['duplicates']} duplicates")

async def _fetch_async(symbol, timeframe, start_date, end_date):
    """Async helper for fetch command."""
    fetcher = BinanceDataFetcher()
    preprocessor = DataPreprocessor()
    db = DatabaseManager()

    await db.initialize()

    try:
        # Fetch data
        df = await fetcher.fetch_historical_range(
            symbol, timeframe, start_date, end_date
        )

        # Validate
        df = preprocessor.validate_ohlcv(df)
        df = preprocessor.handle_missing_values(df)

        # Store
        inserted = await db.save_ohlcv(df, symbol, timeframe)

        return {
            'total_candles': len(df),
            'inserted': inserted,
            'duplicates': len(df) - inserted,
        }
    finally:
        await db.close()
```

### Train Command

```python
# src/price_stradamus/cli/model_commands.py
import typer
from pathlib import Path
from price_stradamus.models.registry import ModelRegistry
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.stateful_features import StatefulFeatureEngineer

app = typer.Typer()

@app.command()
def train(
    model: str = typer.Option("nbeats", help="Model name"),
    symbol: str = typer.Option("BTCUSDT", help="Trading symbol"),
    timeframe: str = typer.Option("1m", help="Timeframe"),
    epochs: int = typer.Option(100, help="Training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    learning_rate: float = typer.Option(0.001, help="Learning rate"),
    input_length: int = typer.Option(60, help="Input chunk length"),
    output_length: int = typer.Option(5, help="Output chunk length"),
    save_path: str = typer.Option("models", help="Directory to save model"),
):
    """Train a forecasting model."""
    typer.echo(f"🔨 Training {model} model...")

    # Load data
    typer.echo("📊 Loading data...")
    db = DatabaseManager()
    df = asyncio.run(db.get_ohlcv(symbol, timeframe))
    typer.echo(f"Loaded {len(df)} candles")

    # Split raw data first (prevent data leakage)
    typer.echo("✂️ Splitting data...")
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)

    train_raw = df[:train_size]
    val_raw = df[train_size:train_size+val_size]
    test_raw = df[train_size+val_size:]

    # Generate features with fit-transform pattern
    typer.echo("🔧 Generating features...")
    engineer = StatefulFeatureEngineer()
    train_features = engineer.fit_transform(train_raw)  # Fit on training
    val_features = engineer.transform(val_raw)          # Transform with train stats
    test_features = engineer.transform(test_raw)        # Transform with train stats

    # Convert to TimeSeries
    train_ts = engineer.to_darts_timeseries(train_features, ['close'])
    val_ts = engineer.to_darts_timeseries(val_features, ['close'])
    test_ts = engineer.to_darts_timeseries(test_features, ['close'])

    typer.echo(f"Train: {len(train_ts)}, Val: {len(val_ts)}, Test: {len(test_ts)}")

    # Create model
    typer.echo(f"🧠 Creating {model} model...")
    model_instance = ModelRegistry.create_model(
        model,
        input_chunk_length=input_length,
        output_chunk_length=output_length,
        n_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Train
    typer.echo("🚀 Training...")
    with typer.progressbar(length=epochs) as progress:
        model_instance.fit(train_ts, val_ts)
        progress.update(epochs)

    # Evaluate on validation set
    typer.echo("📈 Evaluating...")
    predictions = model_instance.predict(n=len(val_ts), series=train_ts)
    mae = calculate_mae(predictions.values(), val_ts.values())
    typer.echo(f"Validation MAE: {mae:.2f}")

    # Save model
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{model}_{symbol}_{timeframe}.pth"

    model_instance.save(model_path)
    typer.echo(f"✅ Model saved to {model_path}")
```

---

## End-to-End Workflows

### Workflow 1: Train and Evaluate a Model

```bash
# Step 1: Fetch data (if not already done)
python -m price_stradamus.cli fetch --days 30

# Step 2: Train model
python -m price_stradamus.cli train \
    --model nbeats \
    --epochs 100

# Step 3: Evaluate on test set
python -m price_stradamus.cli evaluate \
    --model nbeats \
    --test-size 0.2

# Step 4: Make predictions
python -m price_stradamus.cli predict \
    --model nbeats \
    --steps 5
```

### Workflow 2: Compare Multiple Models

```bash
# Fetch more data for better comparison
python -m price_stradamus.cli fetch --days 60

# Train all models
for model in nbeats lstm tcn xgboost prophet; do
    python -m price_stradamus.cli train \
        --model $model \
        --epochs 100
done

# Compare with walk-forward validation
python -m price_stradamus.cli compare \
    --models nbeats lstm tcn xgboost prophet \
    --walk-forward \
    --n-splits 5

# Output:
# Model Performance Comparison
# ============================
# Model     | MAE    | RMSE   | MAPE  | Dir Acc
# ----------|--------|--------|-------|--------
# N-BEATS   | 125.32 | 180.45 | 1.85% | 58.3%
# LSTM      | 132.18 | 195.23 | 1.92% | 56.1%
# TCN       | 128.91 | 185.67 | 1.88% | 57.5%
# XGBoost   | 135.45 | 198.12 | 1.98% | 55.8%
# Prophet   | 142.67 | 205.34 | 2.15% | 54.2%
```

### Workflow 3: Hyperparameter Tuning

```bash
# Try different learning rates
for lr in 0.0001 0.001 0.01; do
    python -m price_stradamus.cli train \
        --model nbeats \
        --learning-rate $lr \
        --save-path "models/nbeats_lr_$lr"
done

# Compare results
python -m price_stradamus.cli compare-saved \
    --model-paths models/nbeats_lr_*
```

---

## Logging and Debugging

### Logging Setup

Price Stradamus uses **Loguru** for beautiful, structured logging.

```python
# src/price_stradamus/utils/logger.py
from loguru import logger
import sys

# Configure logger
logger.remove()  # Remove default handler

# Console output (colored, formatted)
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# File output (detailed, with rotation)
logger.add(
    "logs/price_stradamus_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="DEBUG",
)

# Usage in code
from price_stradamus.utils import logger

logger.info("Starting data fetch for {symbol}", symbol="BTCUSDT")
logger.debug("Fetched {count} candles", count=len(df))
logger.warning("Validation loss increased for {epochs} epochs", epochs=5)
logger.error("Failed to fetch data: {error}", error=str(e))
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python -m price_stradamus.cli train --model nbeats

# Or add --verbose flag
python -m price_stradamus.cli train \
    --model nbeats \
    --verbose
```

### Common Debugging Scenarios

#### 1. Data Not Found

```bash
# Check database contents
python -m price_stradamus.cli db-stats

# Output:
# Database Statistics
# ===================
# Symbol   | Timeframe | Count | Start Date | End Date
# ---------|-----------|-------|------------|----------
# BTCUSDT  | 1m        | 43200 | 2025-01-01 | 2025-01-30

# If empty, fetch data first
python -m price_stradamus.cli fetch --days 30
```

#### 2. Training Fails

```python
# Check logs
tail -f logs/price_stradamus_*.log

# Common issues:
# - CUDA out of memory → reduce batch size
# - Validation loss = NaN → lower learning rate
# - Model diverges → check data normalization
```

#### 3. Poor Performance

```bash
# Check data quality
python -m price_stradamus.cli check-data

# Check for data leakage
python -m price_stradamus.cli validate-features

# Try simpler model first
python -m price_stradamus.cli train --model prophet
```

---

## Configuration Management

### Settings File

```python
# src/price_stradamus/config/settings.py
from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    """Application settings."""

    # Database
    database_url: str = "postgresql://localhost:5432/price_stradamus"

    # Binance API
    binance_api_key: SecretStr | None = None
    binance_api_secret: SecretStr | None = None

    # Model defaults
    default_model: str = "nbeats"
    default_epochs: int = 100
    default_batch_size: int = 32

    # Paths
    models_dir: str = "models"
    logs_dir: str = "logs"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

# Usage
from price_stradamus.config import settings

print(settings.database_url)
print(settings.default_model)
```

### Environment Variables

```bash
# .env file
DATABASE_URL=postgresql://localhost:5432/price_stradamus
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
DEFAULT_MODEL=nbeats
DEFAULT_EPOCHS=100
LOG_LEVEL=INFO
```

---

## Testing CLI Commands

```python
# tests/test_cli/test_commands.py
from typer.testing import CliRunner
from price_stradamus.cli.commands import app

runner = CliRunner()

def test_fetch_command():
    """Test fetch command."""
    result = runner.invoke(app, ["fetch", "--days", "1"])
    assert result.exit_code == 0
    assert "Fetched" in result.stdout

def test_train_command():
    """Test train command."""
    result = runner.invoke(app, [
        "train",
        "--model", "nbeats",
        "--epochs", "1",  # Fast test
    ])
    assert result.exit_code == 0
    assert "Training" in result.stdout
```

---

## Automation Scripts

### Bash Script: Daily Data Update

```bash
#!/bin/bash
# scripts/update_data.sh

# Fetch latest data
python -m price_stradamus.cli fetch --days 1

# Retrain model if needed
if [ -f "models/nbeats.pth" ]; then
    model_age=$(find models/nbeats.pth -mtime +7)
    if [ -n "$model_age" ]; then
        echo "Model older than 7 days, retraining..."
        python -m price_stradamus.cli train --model nbeats
    fi
fi

# Make predictions
python -m price_stradamus.cli predict \
    --model nbeats \
    --steps 5 \
    > predictions/$(date +%Y%m%d).txt
```

### Python Script: Batch Training

```python
# scripts/train_all_models.py
import subprocess
from pathlib import Path

MODELS = ["nbeats", "lstm", "tcn", "xgboost", "prophet"]

def train_all():
    """Train all models."""
    for model in MODELS:
        print(f"Training {model}...")
        subprocess.run([
            "python", "-m", "price_stradamus.cli.commands",
            "train",
            "--model", model,
            "--epochs", "100",
        ])

if __name__ == "__main__":
    train_all()
```

---

## Practice Exercises

### Exercise 1: Create Custom Command (30 minutes)

Add a new command to export predictions to CSV:

```python
@app.command()
def export_predictions(
    model: str,
    output_file: str = "predictions.csv",
):
    """Export predictions to CSV."""
    # TODO: Implement
    pass
```

### Exercise 2: Build Complete Pipeline (45 minutes)

Create a script that:
1. Fetches latest data
2. Trains 3 models
3. Compares them
4. Saves best model
5. Makes predictions

---

## Next Steps

**Next Module:** [12: Best Practices and Final Challenge →](12-best-practices-and-final-challenge.md)

Complete your learning journey with best practices and the final rebuild challenge!

---

## Summary Checklist

- [ ] I know all CLI commands
- [ ] I can run end-to-end workflows
- [ ] I understand logging and debugging
- [ ] I can configure settings via environment variables
- [ ] I can test CLI commands
- [ ] I can automate common tasks

**Continue to:** [Module 12: Best Practices and Final Challenge](12-best-practices-and-final-challenge.md)
