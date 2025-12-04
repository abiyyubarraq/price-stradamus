# Price Stradamus CLI Commands Reference

Complete reference for all CLI commands with examples for both **Bash/Linux/macOS** and **PowerShell/Windows**.

---

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Data Commands](#data-commands)
  - [fetch](#fetch---fetch-historical-data)
- [Model Commands](#model-commands)
  - [train](#train---train-a-model)
  - [predict](#predict---make-predictions)
  - [evaluate](#evaluate---evaluate-model-performance)
  - [compare](#compare---compare-multiple-models)
- [Info Commands](#info-commands)
  - [list-models](#list-models---list-available-models)
  - [info](#info---show-system-information)
- [Common Workflows](#common-workflows)

---

## Installation & Setup

### Prerequisites

```bash
# Bash/Linux/macOS
python --version  # Should be 3.13+
uv --version      # Should be installed

# PowerShell/Windows
python --version  # Should be 3.13+
uv --version      # Should be installed
```

### Install Price Stradamus

```bash
# Bash/Linux/macOS
cd price-stradamus
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# PowerShell/Windows
cd price-stradamus
uv venv
.venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Running Commands

**Two ways to execute CLI commands:**

1. **Using installed command** (after `pip install -e .`):
   ```bash
   price-stradamus <command> [options]
   ```

2. **Using Python module** (works without installation):
   ```bash
   python -m price_stradamus.cli <command> [options]
   ```

Both methods work identically. Use whichever is more convenient!

### Start Database

```bash
# Bash/Linux/macOS
cd docker
docker-compose up -d postgres

# PowerShell/Windows
cd docker
docker-compose up -d postgres
```

---

## Data Commands

### `fetch` - Fetch Historical Data

Fetches OHLCV (Open, High, Low, Close, Volume) data from Binance and stores it in PostgreSQL.

#### Syntax

```
price-stradamus fetch [OPTIONS]
```

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--symbol` | `-s` | str | BTCUSDT | Trading pair symbol |
| `--timeframe` | `-t` | str | 1m | Candle timeframe |
| `--days` | `-d` | int | 30 | Number of days to fetch |

#### Supported Timeframes

- `1m` - 1 minute
- `5m` - 5 minutes
- `15m` - 15 minutes
- `1h` - 1 hour
- `4h` - 4 hours
- `1d` - 1 day

#### Examples

**Fetch 30 days of 1-minute BTC data (default)**
```bash
# Bash/PowerShell - Using installed command
price-stradamus fetch

# OR using Python module
python -m price_stradamus.cli fetch
```

**Fetch 7 days of 5-minute ETH data**
```bash
# Bash
price-stradamus fetch --symbol ETHUSDT --timeframe 5m --days 7

# PowerShell
price-stradamus fetch --symbol ETHUSDT --timeframe 5m --days 7
```

**Fetch 90 days of 1-hour BTC data (short form)**
```bash
# Bash
price-stradamus fetch -s BTCUSDT -t 1h -d 90

# PowerShell
price-stradamus fetch -s BTCUSDT -t 1h -d 90
```

**Fetch multiple symbols (run sequentially)**
```bash
# Bash
for symbol in BTCUSDT ETHUSDT SOLUSDT; do
    price-stradamus fetch -s $symbol -t 1h -d 30
done

# PowerShell
foreach ($symbol in @("BTCUSDT", "ETHUSDT", "SOLUSDT")) {
    price-stradamus fetch -s $symbol -t 1h -d 30
}
```

#### Output Example

```
Fetching BTCUSDT 1m data for 30 days...
Date range: 2024-11-05 to 2024-12-05
✓ Fetched 43200 candles from Binance
Saving to database...
✓ Saved 43200 new candles to database (skipped 0 duplicates)

Data Summary:
  First candle: 2024-11-05 00:00:00
  Last candle:  2024-12-05 23:59:00
  Price range:  $50123.45 - $98756.32
```

---

## Model Commands

### `train` - Train a Model

Train a machine learning model on historical data.

#### Syntax

```
price-stradamus train [OPTIONS]
```

#### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--model` | `-m` | str | Yes | - | Model name |
| `--symbol` | `-s` | str | No | BTCUSDT | Trading symbol |
| `--timeframe` | `-t` | str | No | 1m | Timeframe |
| `--input-length` | `-i` | int | No | 60 | Input sequence length |
| `--output-length` | `-o` | int | No | 5 | Output prediction steps |
| `--epochs` | `-e` | int | No | 100 | Training epochs |
| `--save` | - | str | No | None | Save path for model |

#### Available Models

- **Neural Networks**: `nbeats`, `lstm`, `tcn`, `tft`
- **Classical**: `arima`, `prophet`
- **ML**: `xgboost`, `random_forest`

#### Examples

**Train N-BEATS model with defaults**
```bash
# Using installed command
price-stradamus train --model nbeats

# OR using Python module
python -m price_stradamus.cli train --model nbeats
```

**Train XGBoost with custom parameters**
```bash
# Bash
price-stradamus train -m xgboost -i 120 -o 10 -e 200

# PowerShell
price-stradamus train -m xgboost -i 120 -o 10 -e 200
```

**Train and save model**
```bash
# Bash - Using installed command
mkdir -p models
price-stradamus train -m lstm --save models/lstm_model.pkl

# OR using Python module
python -m price_stradamus.cli train -m lstm --save models/lstm_model.pkl

# PowerShell - Using installed command
New-Item -ItemType Directory -Force -Path models
price-stradamus train -m lstm --save models/lstm_model.pkl

# OR using Python module
python -m price_stradamus.cli train -m lstm --save models/lstm_model.pkl
```

**Train on different symbol/timeframe**
```bash
# Bash
price-stradamus train -m nbeats -s ETHUSDT -t 5m -e 150

# PowerShell
price-stradamus train -m nbeats -s ETHUSDT -t 5m -e 150
```

**Train multiple models (batch training)**
```bash
# Bash
for model in nbeats lstm xgboost; do
    price-stradamus train -m $model -e 50 --save models/${model}_model.pkl
done

# PowerShell
foreach ($model in @("nbeats", "lstm", "xgboost")) {
    price-stradamus train -m $model -e 50 --save "models/$($model)_model.pkl"
}
```

#### Output Example

```
Training nbeats model...
Loading BTCUSDT 1m data from database...
✓ Loaded 43200 candles
Preprocessing data...
Generating technical indicators...
✓ Generated 42 features
Creating time series...
✓ Created time series with 43200 timesteps
Split: train=30240, val=6480

Creating nbeats model...
✓ Model created: nbeats

Training model (this may take a while)...
Press Ctrl+C to interrupt and save current progress
Epoch 1/100: loss=0.0234
...
Epoch 100/100: loss=0.0012
✓ Training complete!
✓ Model saved to models/nbeats_model.pkl
```

---

### `predict` - Make Predictions

Make future price predictions using a trained model.

#### Syntax

```
price-stradamus predict [OPTIONS]
```

#### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--model` | `-m` | str | Yes | - | Path to trained model file |
| `--steps` | `-s` | int | No | 5 | Number of steps to predict |
| `--symbol` | - | str | No | BTCUSDT | Trading symbol |
| `--timeframe` | - | str | No | 1m | Timeframe |
| `--plot` | - | bool | No | False | Show interactive chart |
| `--save-chart` | - | str | No | None | Save chart to HTML file |

#### Examples

**Make 5-step predictions**
```bash
# Using installed command
price-stradamus predict --model models/nbeats_model.pkl

# OR using Python module
python -m price_stradamus.cli predict --model models/nbeats_model.pkl
```

**Predict 10 steps ahead**
```bash
# Bash
price-stradamus predict -m models/xgboost_model.pkl -s 10

# PowerShell
price-stradamus predict -m models/xgboost_model.pkl -s 10
```

**Predict and show interactive chart**
```bash
# Bash
price-stradamus predict -m models/lstm_model.pkl -s 10 --plot

# PowerShell
price-stradamus predict -m models/lstm_model.pkl -s 10 --plot
```

**Predict and save chart to file**
```bash
# Bash
mkdir -p results
price-stradamus predict \
    -m models/nbeats_model.pkl \
    -s 20 \
    --save-chart results/predictions_$(date +%Y%m%d).html

# PowerShell
New-Item -ItemType Directory -Force -Path results
price-stradamus predict `
    -m models/nbeats_model.pkl `
    -s 20 `
    --save-chart "results/predictions_$(Get-Date -Format 'yyyyMMdd').html"
```

**Predict on different symbol**
```bash
# Bash
price-stradamus predict \
    -m models/nbeats_model.pkl \
    --symbol ETHUSDT \
    --timeframe 5m \
    -s 5

# PowerShell
price-stradamus predict `
    -m models/nbeats_model.pkl `
    --symbol ETHUSDT `
    --timeframe 5m `
    -s 5
```

#### Output Example

```
Making predictions with models/nbeats_model.pkl...
Loading recent BTCUSDT 1m data...
✓ Loaded 1000 candles
Preprocessing data...
Generating features...
Creating time series...
Loading model from models/nbeats_model.pkl...
✓ Model loaded: nbeats

Making 5-step predictions...

Predictions:
┌──────┬─────────────────┬──────────────────┐
│ Step │ Predicted Price │ Change from Last │
├──────┼─────────────────┼──────────────────┤
│ 1    │ $50,245.67      │ +12.34 (+0.02%)  │
│ 2    │ $50,289.12      │ +43.45 (+0.09%)  │
│ 3    │ $50,334.89      │ +45.77 (+0.09%)  │
│ 4    │ $50,378.45      │ +43.56 (+0.09%)  │
│ 5    │ $50,423.12      │ +44.67 (+0.09%)  │
└──────┴─────────────────┴──────────────────┘

Summary:
Current price: $50,233.33
First prediction (1-step): $50,245.67
Last prediction (5-step): $50,423.12
Total predicted change: +0.38%
```

---

### `evaluate` - Evaluate Model Performance

Run backtesting and calculate performance metrics.

#### Syntax

```
price-stradamus evaluate [OPTIONS]
```

#### Options

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--model` | `-m` | str | Model name (train from scratch) |
| `--model-path` | `-p` | str | Path to saved model (load existing) |
| `--symbol` | `-s` | str | Trading symbol (default: BTCUSDT) |
| `--timeframe` | `-t` | str | Timeframe (default: 1m) |
| `--method` | - | str | Backtest method (default: expanding) |
| `--plot` | - | bool | Show interactive chart |
| `--save-chart` | - | str | Save chart to HTML file |

**Note**: Use either `--model` OR `--model-path`, not both.

#### Examples

**Evaluate a saved model**
```bash
# Bash
price-stradamus evaluate --model-path models/xgboost_model.pkl

# PowerShell
price-stradamus evaluate --model-path models/xgboost_model.pkl
```

**Evaluate by training from scratch**
```bash
# Bash
price-stradamus evaluate --model xgboost

# PowerShell
price-stradamus evaluate --model xgboost
```

**Evaluate with visualization**
```bash
# Bash
price-stradamus evaluate -p models/nbeats_model.pkl --plot

# PowerShell
price-stradamus evaluate -p models/nbeats_model.pkl --plot
```

**Evaluate and save results**
```bash
# Bash
price-stradamus evaluate \
    -p models/lstm_model.pkl \
    --save-chart results/evaluation_lstm.html

# PowerShell
price-stradamus evaluate `
    -p models/lstm_model.pkl `
    --save-chart results/evaluation_lstm.html
```

**Batch evaluation of all models**
```bash
# Bash
for model_file in models/*.pkl; do
    model_name=$(basename "$model_file" .pkl)
    echo "Evaluating $model_name..."
    price-stradamus evaluate \
        -p "$model_file" \
        --save-chart "results/eval_${model_name}.html"
done

# PowerShell
Get-ChildItem models/*.pkl | ForEach-Object {
    $modelName = $_.BaseName
    Write-Host "Evaluating $modelName..."
    price-stradamus evaluate `
        -p $_.FullName `
        --save-chart "results/eval_$modelName.html"
}
```

#### Output Example

```
Evaluating saved model: models/nbeats_model.pkl
Loading BTCUSDT 1m data...
✓ Loaded 43200 candles
Preprocessing data...
Generating features...
Loading model from models/nbeats_model.pkl...
✓ Model loaded: nbeats

Evaluating pre-trained model on 43200 timesteps...
✓ Generated 862 prediction windows

Evaluation Results:
┌────────────────────────┬──────────┐
│ Metric                 │ Value    │
├────────────────────────┼──────────┤
│ MAE                    │ 123.45   │
│ RMSE                   │ 234.56   │
│ MAPE                   │ 0.42%    │
│ Directional Accuracy   │ 56.78%   │
│ Total Predictions      │ 4310     │
└────────────────────────┴──────────┘
```

---

### `compare` - Compare Multiple Models

Compare performance of multiple models side-by-side (Phase 2 feature).

#### Syntax

```
price-stradamus compare [OPTIONS]
```

#### Options

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--model` | `-m` | str | Model names (repeat for each) |

#### Examples

```bash
# Bash
price-stradamus compare -m nbeats -m lstm -m xgboost

# PowerShell
price-stradamus compare -m nbeats -m lstm -m xgboost
```

**Status**: Coming in Phase 2 with parallel training and statistical comparison.

---

## Info Commands

### `list-models` - List Available Models

Display all registered prediction models.

#### Syntax

```
price-stradamus list-models
```

#### Examples

```bash
# Bash & PowerShell (same)
price-stradamus list-models
```

#### Output Example

```
Available Models:

Registered Models (8 total)
┌───────────────┬────────────────────────┬─────────────────────────────────┐
│ Name          │ Class                  │ Description                     │
├───────────────┼────────────────────────┼─────────────────────────────────┤
│ nbeats        │ NBEATSModel            │ Neural Basis Expansion Analysis │
│ lstm          │ LSTMModel              │ Long Short-Term Memory network  │
│ tcn           │ TCNModel               │ Temporal Convolutional Network  │
│ tft           │ TFTModel               │ Temporal Fusion Transformer     │
│ arima         │ ARIMAModel             │ AutoRegressive Integrated MA    │
│ prophet       │ ProphetModel           │ Facebook Prophet                │
│ xgboost       │ XGBoostModel           │ Gradient Boosting Trees         │
│ random_forest │ RandomForestModel      │ Random Forest Regressor         │
└───────────────┴────────────────────────┴─────────────────────────────────┘
```

---

### `info` - Show System Information

Display configuration and system details.

#### Syntax

```
price-stradamus info
```

#### Examples

```bash
# Bash & PowerShell (same)
price-stradamus info
```

#### Output Example

```
Price Stradamus - System Information

Database:
  URL: postgresql://postgres:***@localhost:5432/price_stradamus

Binance API:
  URL: https://api.binance.com

Training Defaults:
  Lookback window: 60
  Prediction steps: 5
  Batch size: 32
  Learning rate: 0.001
  Epochs: 100

Registered Models:
  • nbeats
  • lstm
  • tcn
  • tft
  • arima
  • prophet
  • xgboost
  • random_forest
```

---

## Common Workflows

### Complete Workflow: Data → Train → Predict

```bash
# Bash
# 1. Fetch data
price-stradamus fetch -s BTCUSDT -t 1h -d 90

# 2. Train model
price-stradamus train -m nbeats -e 100 --save models/nbeats_model.pkl

# 3. Make predictions
price-stradamus predict -m models/nbeats_model.pkl -s 10 --plot

# PowerShell
# 1. Fetch data
price-stradamus fetch -s BTCUSDT -t 1h -d 90

# 2. Train model
price-stradamus train -m nbeats -e 100 --save models/nbeats_model.pkl

# 3. Make predictions
price-stradamus predict -m models/nbeats_model.pkl -s 10 --plot
```

### Model Evaluation Workflow

```bash
# Bash
# 1. Train and save model
price-stradamus train -m xgboost --save models/xgboost_model.pkl

# 2. Evaluate performance
price-stradamus evaluate -p models/xgboost_model.pkl --plot

# 3. Make predictions
price-stradamus predict -m models/xgboost_model.pkl -s 20

# PowerShell (same commands work)
```

### Train All Models and Compare

```bash
# Bash
models="nbeats lstm tcn xgboost random_forest"
for model in $models; do
    echo "Training $model..."
    price-stradamus train -m $model -e 50 --save models/${model}_model.pkl
    price-stradamus evaluate -p models/${model}_model.pkl --save-chart results/eval_${model}.html
done

# PowerShell
$models = @("nbeats", "lstm", "tcn", "xgboost", "random_forest")
foreach ($model in $models) {
    Write-Host "Training $model..."
    price-stradamus train -m $model -e 50 --save "models/$model`_model.pkl"
    price-stradamus evaluate -p "models/$model`_model.pkl" --save-chart "results/eval_$model.html"
}
```

### Daily Prediction Script

```bash
# Bash - save as daily_predict.sh
#!/bin/bash
set -e

# Fetch latest data
price-stradamus fetch -s BTCUSDT -t 1h -d 1

# Make predictions with each model
for model in models/*.pkl; do
    model_name=$(basename "$model" .pkl)
    price-stradamus predict \
        -m "$model" \
        -s 24 \
        --save-chart "predictions/${model_name}_$(date +%Y%m%d).html"
done

# PowerShell - save as daily_predict.ps1
# Fetch latest data
price-stradamus fetch -s BTCUSDT -t 1h -d 1

# Make predictions with each model
Get-ChildItem models/*.pkl | ForEach-Object {
    $modelName = $_.BaseName
    $date = Get-Date -Format 'yyyyMMdd'
    price-stradamus predict `
        -m $_.FullName `
        -s 24 `
        --save-chart "predictions/$modelName`_$date.html"
}
```

---

## Tips & Best Practices

### Performance

- **Use GPU**: Models train faster with CUDA-enabled GPUs
- **Batch operations**: Train multiple models in parallel on different terminals
- **Incremental fetching**: Fetch data incrementally rather than all at once

### Data Management

```bash
# Check database size
docker exec price-stradamus-postgres psql -U postgres -d price_stradamus -c "
    SELECT pg_size_pretty(pg_database_size('price_stradamus'));
"

# Backup database
docker exec price-stradamus-postgres pg_dump -U postgres price_stradamus > backup.sql

# Restore database
docker exec -i price-stradamus-postgres psql -U postgres price_stradamus < backup.sql
```

### Error Handling

```bash
# Bash - Retry on failure
until price-stradamus fetch -s BTCUSDT -t 1m -d 30; do
    echo "Fetch failed, retrying in 60 seconds..."
    sleep 60
done

# PowerShell - Retry on failure
while ($true) {
    try {
        price-stradamus fetch -s BTCUSDT -t 1m -d 30
        break
    } catch {
        Write-Host "Fetch failed, retrying in 60 seconds..."
        Start-Sleep -Seconds 60
    }
}
```

### Logging

```bash
# Bash - Log all output
price-stradamus train -m nbeats 2>&1 | tee logs/train_nbeats_$(date +%Y%m%d).log

# PowerShell - Log all output
price-stradamus train -m nbeats 2>&1 | Tee-Object -FilePath "logs/train_nbeats_$(Get-Date -Format 'yyyyMMdd').log"
```

---

## Troubleshooting

### Command not found

```bash
# Solution 1: Use Python module (always works)
python -m price_stradamus.cli <command>

# Solution 2: Ensure virtual environment is activated
source .venv/bin/activate  # Bash/Linux/macOS
.venv\Scripts\activate     # PowerShell/Windows

# Solution 3: Reinstall if needed
uv pip install -e ".[dev]"
```

### Database connection error

```bash
# Check if database is running
docker ps | grep postgres

# Start database if not running
cd docker && docker-compose up -d postgres
```

### Model file not found

```bash
# Check available models
ls models/

# Use absolute path if needed
price-stradamus predict -m /full/path/to/models/nbeats_model.pkl
```

### Out of memory during training

- Reduce batch size: Training will be slower but use less memory
- Use smaller input lengths: `--input-length 30` instead of 60
- Train on subset of data: Fetch fewer days

---

**Last Updated**: 2024-12-05
**Phase**: 1 (Core System)
**Supported OS**: Linux, macOS, Windows
