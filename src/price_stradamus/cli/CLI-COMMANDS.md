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
- [TimeWindow Commands (Data Leakage Prevention)](#timewindow-commands-data-leakage-prevention)
  - [train-window](#train-window---train-with-explicit-time-boundaries)
  - [eval-window](#eval-window---evaluate-on-explicit-time-period)
  - [Complete TimeWindow Workflow](#complete-timewindow-workflow)
  - [Common Use Cases](#common-use-cases)
  - [Troubleshooting TimeWindow Commands](#troubleshooting-timewindow-commands)
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

> **New**: Now supports explicit datetime windows for data leakage prevention! Use `--train-start`/`--train-end` and `--val-start`/`--val-end` parameters.

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
| `--train-start` | - | str | No | None | Training start datetime (e.g., '2025-08-01 00:00') |
| `--train-end` | - | str | No | None | Training end datetime (e.g., '2025-11-03 03:00') |
| `--val-start` | - | str | No | None | Validation start datetime (optional) |
| `--val-end` | - | str | No | None | Validation end datetime (optional) |
| `--save` | - | str | No | None | Save path for model |

#### Available Models

- **Neural Networks**: `nbeats`, `lstm`, `tcn`, `tft`
- **Classical**: `arima`, `prophet`
- **ML**: `xgboost`, ``

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

**✨ NEW: Train on specific datetime window (prevents data leakage)**
```bash
# Bash
price-stradamus train -m nbeats \
    --train-start "2025-08-01 00:00" \
    --train-end "2025-11-03 03:00"

# PowerShell
price-stradamus train -m nbeats `
    --train-start "2025-08-01 00:00" `
    --train-end "2025-11-03 03:00"
```

**✨ NEW: Train with separate train/val windows**
```bash
# Bash
price-stradamus train -m xgboost \
    --train-start "2025-08-01 00:00" \
    --train-end "2025-10-15 23:59" \
    --val-start "2025-10-16 00:00" \
    --val-end "2025-11-03 03:00" \
    --save models/xgboost_aug_oct.pkl

# PowerShell
price-stradamus train -m xgboost `
    --train-start "2025-08-01 00:00" `
    --train-end "2025-10-15 23:59" `
    --val-start "2025-10-16 00:00" `
    --val-end "2025-11-03 03:00" `
    --save models/xgboost_aug_oct.pkl
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

> **New**: Now supports explicit datetime windows for data leakage prevention! Use `--train-start`/`--train-end` and `--test-start`/`--test-end` parameters.

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
| `--train-start` | - | str | Training start datetime (e.g., '2025-08-01 00:00') |
| `--train-end` | - | str | Training end datetime (e.g., '2025-11-03 03:00') |
| `--test-start` | - | str | Test start datetime (e.g., '2025-11-03 03:01') |
| `--test-end` | - | str | Test end datetime (e.g., '2025-11-05 23:59') |
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

**✨ NEW: Evaluate on specific datetime test window**
```bash
# Bash
price-stradamus evaluate \
    --model-path models/nbeats_model.pkl \
    --test-start "2025-11-03 03:01" \
    --test-end "2025-11-05 23:59"

# PowerShell
price-stradamus evaluate `
    --model-path models/nbeats_model.pkl `
    --test-start "2025-11-03 03:01" `
    --test-end "2025-11-05 23:59"
```

**✨ NEW: Evaluate with explicit train/test windows (prevents data leakage)**
```bash
# Bash
price-stradamus evaluate \
    --model-path models/xgboost_model.pkl \
    --train-start "2025-08-01 00:00" \
    --train-end "2025-11-03 03:00" \
    --test-start "2025-11-03 03:01" \
    --test-end "2025-11-05 23:59" \
    --plot

# PowerShell
price-stradamus evaluate `
    --model-path models/xgboost_model.pkl `
    --train-start "2025-08-01 00:00" `
    --train-end "2025-11-03 03:00" `
    --test-start "2025-11-03 03:01" `
    --test-end "2025-11-05 23:59" `
    --plot
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

## TimeWindow Commands (Data Leakage Prevention)

### `train-window` - Train with Explicit Time Boundaries

Train a model on a specific time period to prevent data leakage.

**Why use this?** Explicitly specifying time boundaries ensures you never accidentally train on test data.

#### Syntax

```
price-stradamus train-window [OPTIONS]
```

#### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--model` | `-m` | str | Yes | - | Model name |
| `--train-start` | - | str | Yes | - | Training start date |
| `--train-end` | - | str | Yes | - | Training end date |
| `--symbol` | `-s` | str | No | BTCUSDT | Trading symbol |
| `--timeframe` | `-t` | str | No | 1m | Timeframe |
| `--input-length` | `-i` | int | No | 60 | Input sequence length |
| `--output-length` | `-o` | int | No | 5 | Output prediction steps |
| `--epochs` | `-e` | int | No | 100 | Training epochs |
| `--save` | - | str | No | Auto | Save path for model |

#### Supported Date Formats

```bash
# Just date (assumes 00:00:00)
--train-start "2025-08-01"

# Date with time
--train-start "2025-08-01 00:00"

# Date with seconds
--train-start "2025-08-01 00:00:00"

# DD/MM/YYYY format
--train-start "01/08/2025"
```

#### Examples

**Train on August-October data**
```bash
# Bash
price-stradamus train-window \
    --model nbeats \
    --train-start "2025-08-01" \
    --train-end "2025-10-31"

# PowerShell
price-stradamus train-window `
    --model nbeats `
    --train-start "2025-08-01" `
    --train-end "2025-10-31"
```

**Train and save with custom path**
```bash
# Bash
price-stradamus train-window \
    -m  \
    --train-start "2025-08-01" \
    --train-end "2025-10-31" \
    --epochs 100 \
    --save models/_aug_oct.pkl

# PowerShell
price-stradamus train-window `
    -m  `
    --train-start "2025-08-01" `
    --train-end "2025-10-31" `
    --epochs 100 `
    --save models/_aug_oct.pkl
```

**Train on specific hours**
```bash
# Bash
price-stradamus train-window \
    -m xgboost \
    --train-start "2025-08-01 00:00" \
    --train-end "2025-10-31 23:59" \
    -i 120 -o 10

# PowerShell
price-stradamus train-window `
    -m xgboost `
    --train-start "2025-08-01 00:00" `
    --train-end "2025-10-31 23:59" `
    -i 120 -o 10
```

#### What It Does

1. [OK] Loads data **ONLY** from [train-start, train-end]
2. [OK] Automatically checks for data leakage
3. [OK] Splits data 70/30 (train/val) within the window
4. [OK] Trains model on historical data
5. [OK] Saves with date-stamped filename (if --save not specified)

#### Default Save Path

If you don't specify `--save`, the model is automatically saved as:
```
models/{model_name}_{YYYYMMDD}_{YYYYMMDD}.pkl
```

Example: `models/nbeats_20250801_20251031.pkl`

#### Output Example

```
Training with Time Window (No Data Leakage)

Training Window:
  Start: 2025-08-01 00:00:00
  End:   2025-10-31 00:00:00

Loading BTCUSDT 1m data from database...
[OK] Loaded 130621 candles
Preprocessing data...
Generating technical indicators...
[OK] Generated 58 features
Creating time series...
[OK] Created time series with 130621 timesteps

Split within window: train=91434, val=39187

Verifying no data leakage...
[OK] No data leakage - safe to proceed

Creating  model...
[OK] Model created: Model(name=)

Training model (this may take a while)...
Press Ctrl+C to interrupt and save current progress
[OK] Training complete!
[OK] Model saved to models/_aug_oct.pkl

Training Summary:
  Model: 
  Trained on: 2025-08-01 00:00:00 to 2025-10-31 00:00:00
  Total samples: 130621
  Train samples: 91434
  Val samples: 39187
  Saved to: models/_aug_oct.pkl
```

---

### `eval-window` - Evaluate on Explicit Time Period

Evaluate a trained model on a specific time period to ensure no data leakage.

**Why use this?** Explicitly specifying test boundaries and training end date ensures your evaluation is valid.

#### Syntax

```
price-stradamus eval-window [OPTIONS]
```

#### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--model-path` | `-m` | str | Yes | - | Path to trained model |
| `--test-start` | - | str | Yes | - | Test start date |
| `--test-end` | - | str | Yes | - | Test end date |
| `--train-end` | - | str | No | None | Training end date (for leakage check) |
| `--symbol` | `-s` | str | No | BTCUSDT | Trading symbol |
| `--timeframe` | `-t` | str | No | 1m | Timeframe |
| `--check-leakage` | - | bool | No | True | Check for data leakage |
| `--plot` | - | bool | No | False | Show interactive chart |
| `--save-chart` | - | str | No | None | Save chart to HTML file |

#### Examples

**Evaluate on November-December data**
```bash
# Bash
price-stradamus eval-window \
    --model-path models/_aug_oct.pkl \
    --test-start "2025-11-01" \
    --test-end "2025-12-07"

# PowerShell
price-stradamus eval-window `
    --model-path models/_aug_oct.pkl `
    --test-start "2025-11-01" `
    --test-end "2025-12-07"
```

**Evaluate with leakage check and visualization**
```bash
# Bash
price-stradamus eval-window \
    --model-path models/nbeats_aug_oct.pkl \
    --test-start "2025-11-01" \
    --test-end "2025-12-07" \
    --train-end "2025-10-31" \
    --plot

# PowerShell
price-stradamus eval-window `
    --model-path models/nbeats_aug_oct.pkl `
    --test-start "2025-11-01" `
    --test-end "2025-12-07" `
    --train-end "2025-10-31" `
    --plot
```

**Evaluate and save results**
```bash
# Bash
price-stradamus eval-window \
    -m models/xgboost_model.pkl \
    --test-start "2025-11-01" \
    --test-end "2025-12-07" \
    --train-end "2025-10-31" \
    --save-chart results/evaluation_xgboost.html

# PowerShell
price-stradamus eval-window `
    -m models/xgboost_model.pkl `
    --test-start "2025-11-01" `
    --test-end "2025-12-07" `
    --train-end "2025-10-31" `
    --save-chart results/evaluation_xgboost.html
```

#### What It Does

1. [OK] Loads data **ONLY** from [test-start, test-end]
2. [OK] Optionally checks that test > train (if --train-end provided)
3. [OK] Makes predictions using sliding window
4. [OK] Calculates metrics (MAE, RMSE, MAPE, Directional Accuracy)
5. [OK] Optionally generates interactive chart

#### Leakage Detection

If you provide `--train-end`, the command will **ERROR** if test period overlaps:

```
Training ends:  2025-10-31 23:59:59
Testing starts: 2025-10-30 00:00:00  <- OVERLAPS!

[ERROR] DATA LEAKAGE DETECTED!
Test window overlaps with training window!
```

#### Output Example

```
Evaluating with Time Window (No Data Leakage)

Test Window:
  Start: 2025-11-01 00:00:00
  End:   2025-12-07 00:00:00

Training Window End:
  2025-10-31 00:00:00

[OK] No data leakage - 24.0 hour gap between train and test

Loading BTCUSDT 1m data from database...
[OK] Loaded 51841 candles
Preprocessing data...
Generating features...
[OK] Created time series with 51841 timesteps

Loading model from models/_aug_oct.pkl...
[OK] Model loaded: 

Making predictions on test window...
[OK] Generated 10367 prediction windows

Evaluation Results:
┌────────────────────────┬──────────┐
│ Metric                 │ Value    │
├────────────────────────┼──────────┤
│ MAE                    │ 45.23    │
│ RMSE                   │ 67.89    │
│ MAPE                   │ 1.2%     │
│ Directional Accuracy   │ 54.2%    │
│ Total Predictions      │ 51835    │
└────────────────────────┴──────────┘

Evaluation Summary:
  Model: 
  Tested on: 2025-11-01 00:00:00 to 2025-12-07 00:00:00
  Total samples: 51841
  Predictions: 51835
  [OK] No data leakage (train ended 2025-10-31 00:00:00, test started 2025-11-01 00:00:00)
```

---

### Complete TimeWindow Workflow

**Step 1: Fetch All Data Once**

```bash
# Bash
price-stradamus fetch \
    --symbol BTCUSDT \
    --timeframe 1m \
    --start-date "2025-08-01" \
    --end-date "2025-12-07"

# PowerShell
price-stradamus fetch `
    --symbol BTCUSDT `
    --timeframe 1m `
    --start-date "2025-08-01" `
    --end-date "2025-12-07"
```

**Step 2: Train on August-October**

```bash
# Bash
price-stradamus train-window \
    --model  \
    --train-start "2025-08-01" \
    --train-end "2025-10-31" \
    --save models/_aug_oct.pkl

# PowerShell
price-stradamus train-window `
    --model  `
    --train-start "2025-08-01" `
    --train-end "2025-10-31" `
    --save models/_aug_oct.pkl
```

**Step 3: Evaluate on November-December**

```bash
# Bash
price-stradamus eval-window \
    --model-path models/_aug_oct.pkl \
    --test-start "2025-11-01" \
    --test-end "2025-12-07" \
    --train-end "2025-10-31" \
    --plot

# PowerShell
price-stradamus eval-window `
    --model-path models/_aug_oct.pkl `
    --test-start "2025-11-01" `
    --test-end "2025-12-07" `
    --train-end "2025-10-31" `
    --plot
```

### Common Use Cases

**Monthly Evaluation**
```bash
# Train on November
price-stradamus train-window \
    -m nbeats \
    --train-start "2025-11-01" \
    --train-end "2025-11-30"

# Evaluate on December
price-stradamus eval-window \
    -m models/nbeats_20251101_20251130.pkl \
    --test-start "2025-12-01" \
    --test-end "2025-12-31" \
    --train-end "2025-11-30"
```

**Quarter-by-Quarter**
```bash
# Train on Q3 (Jul-Sep)
price-stradamus train-window \
    -m xgboost \
    --train-start "2025-07-01" \
    --train-end "2025-09-30"

# Evaluate on Q4 (Oct-Dec)
price-stradamus eval-window \
    -m models/xgboost_20250701_20250930.pkl \
    --test-start "2025-10-01" \
    --test-end "2025-12-31" \
    --train-end "2025-09-30"
```

**Rolling Windows (Weekly)**
```bash
# Week 1 training
price-stradamus train-window \
    -m lstm \
    --train-start "2025-11-01" \
    --train-end "2025-11-07"

# Week 2 testing
price-stradamus eval-window \
    -m models/lstm_20251101_20251107.pkl \
    --test-start "2025-11-08" \
    --test-end "2025-11-14" \
    --train-end "2025-11-07"
```

### Troubleshooting TimeWindow Commands

**Error: "No data found for specified window"**

Cause: Database doesn't have data for that period.

Solution:
```bash
price-stradamus fetch \
    --start-date "2025-08-01" \
    --end-date "2025-12-07"
```

**Error: "DATA LEAKAGE DETECTED!"**

Cause: Test period overlaps with training period.

Solution: Adjust dates to ensure test > train:
```bash
# Make sure test-start > train-end
--train-end "2025-10-31"
--test-start "2025-11-01"  # At least 1 second after train-end
```

**Error: "Unknown model type in filename"**

Cause: Model filename doesn't start with a registered model name.

Solution: Ensure filename starts with model name:
```bash
# Good filenames:
_aug_oct.pkl
nbeats_20250801_20251031.pkl
xgboost_model.pkl

# Bad filenames:
my_model.pkl  # Doesn't start with "", "nbeats", etc.
```

### Tips for TimeWindow Commands

1. **Always specify --train-end** when evaluating to enable automatic leakage detection

2. **Use descriptive save paths** that include the date range:
   ```bash
   --save models/nbeats_aug_oct.pkl
   ```

3. **Check your data range first**:
   ```bash
   # Make sure you have data for the entire period
   price-stradamus fetch --start-date "2025-08-01" --end-date "2025-12-07"
   ```

4. **Use --plot** to visually inspect predictions vs actuals

5. **Save charts** for documentation:
   ```bash
   --save-chart results/eval_nov_dec.html
   ```

### Comparison: Old vs New Approach

**Old Way (Risk of Leakage)**
```bash
# Train on ALL data
price-stradamus train --model nbeats

# Evaluate on SAME data (could overlap!)
price-stradamus evaluate --model-path models/nbeats.pkl
```
❌ Problem: No explicit boundaries, easy to accidentally test on training data.

**New Way (Leakage-Proof)**
```bash
# Train on SPECIFIC period
price-stradamus train-window \
    --model nbeats \
    --train-start "2025-08-01" \
    --train-end "2025-10-31"

# Evaluate on DIFFERENT period with automatic check
price-stradamus eval-window \
    --model-path models/nbeats_aug_oct.pkl \
    --test-start "2025-11-01" \
    --test-end "2025-12-07" \
    --train-end "2025-10-31"  # Automatically checks for overlap!
```
✅ Solution: Explicit time boundaries, automatic leakage detection.

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
│  │ Model      │  Regressor         │
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
  • 
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
models="nbeats lstm tcn xgboost "
for model in $models; do
    echo "Training $model..."
    price-stradamus train -m $model -e 50 --save models/${model}_model.pkl
    price-stradamus evaluate -p models/${model}_model.pkl --save-chart results/eval_${model}.html
done

# PowerShell
$models = @("nbeats", "lstm", "tcn", "xgboost", "")
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
