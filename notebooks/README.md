# Price Stradamus Notebooks

Interactive Jupyter notebooks for data exploration, model comparison, and backtesting visualization.

## Overview

These notebooks provide hands-on, interactive exploration of the Price Stradamus Bitcoin prediction system. Each notebook is self-contained and can be run independently.

## Prerequisites

1. **PostgreSQL database running**:
   ```bash
   cd docker && docker-compose up -d postgres
   ```

2. **Historical data fetched** (at least 30 days):
   ```bash
   python -m price_stradamus.cli.commands fetch --days 30
   ```

3. **Dependencies installed**:
   ```bash
   uv pip install -e ".[dev]"
   ```

## Notebook Structure

```
notebooks/
├── 01_exploratory/
│   └── data_exploration.ipynb          # Bitcoin data patterns & statistics
├── 02_experiments/
│   └── model_comparison.ipynb          # Train & compare 8 models
├── 03_results/
│   └── backtest_visualization.ipynb    # Walk-forward backtesting results
└── 04_examples/
    └── proper_train_test_split.ipynb   # Educational: prevent data leakage
```

## Getting Started

### Quick Start (Recommended Order)

1. **Start here** 👉 [04_examples/proper_train_test_split.ipynb](04_examples/proper_train_test_split.ipynb)
   - **Why first**: Understand data leakage prevention
   - **Runtime**: ~3-5 minutes
   - **Learn**: Proper temporal splits, walk-forward validation

2. **Then explore** 👉 [01_exploratory/data_exploration.ipynb](01_exploratory/data_exploration.ipynb)
   - **Purpose**: Understand Bitcoin price patterns
   - **Runtime**: ~2-3 minutes
   - **Learn**: Distributions, seasonality, feature correlations

3. **Compare models** 👉 [02_experiments/model_comparison.ipynb](02_experiments/model_comparison.ipynb)
   - **Purpose**: Train and compare all 8 models
   - **Runtime**: ~15-30 minutes
   - **Learn**: Which model performs best for your data

4. **Validate performance** 👉 [03_results/backtest_visualization.ipynb](03_results/backtest_visualization.ipynb)
   - **Purpose**: Robust walk-forward backtesting
   - **Runtime**: ~10-20 minutes
   - **Learn**: Production readiness assessment

## Notebook Details

### 1. Data Exploration (01_exploratory/)

**File**: `data_exploration.ipynb`

Comprehensive analysis of Bitcoin 1-minute data:
- Price distributions and outliers
- Temporal patterns (hourly/daily seasonality)
- Technical indicator generation (50+ features)
- Feature correlation analysis
- Modeling recommendations

**Visualizations**: matplotlib + seaborn (statistical focus)

**Output**: Insights to guide model selection

### 2. Model Comparison (02_experiments/)

**File**: `model_comparison.ipynb`

Train and compare all 8 available models:
- **Neural**: N-BEATS, LSTM, TCN, TFT
- **Classical**: ARIMA, Prophet
- **ML**: XGBoost

**Methodology**: Same data split (70/15/15), same hyperparameters

**Visualizations**: plotly (interactive) + Rich tables

**Output**: Best model recommendation based on RMSE, MAE, directional accuracy

### 3. Backtest Visualization (03_results/)

**File**: `backtest_visualization.ipynb`

Walk-forward validation for robust performance estimates:
- 30-day training window, 5-day test window
- Multiple folds evaluated
- Metrics stability over time
- Production readiness assessment

**Visualizations**: plotly (interactive charts)

**Output**: Unbiased performance metrics, stability analysis, bias check

### 4. Proper Train/Test Split (04_examples/)

**File**: `proper_train_test_split.ipynb`

Educational walkthrough on preventing data leakage:
- **Problem**: Why random splits fail for time series
- **Solution 1**: Simple 70/15/15 temporal split
- **Solution 2**: Walk-forward validation (gold standard)
- **Verification**: Data leakage checking

**Visualizations**: plotly (timeline) + Rich tables

**Output**: Understanding of proper time series validation

## Running the Notebooks

### Option 1: Jupyter Notebook (Classic)

```bash
# Start Jupyter
jupyter notebook notebooks/

# Opens browser at http://localhost:8888
```

### Option 2: Jupyter Lab (Recommended)

```bash
# Install if not already installed
uv pip install jupyterlab

# Start Jupyter Lab
jupyter lab notebooks/

# Opens browser at http://localhost:8888/lab
```

### Option 3: VS Code (Built-in Support)

1. Open VS Code in project directory
2. Install "Jupyter" extension
3. Open any `.ipynb` file
4. Select Python kernel (`.venv/Scripts/python.exe`)
5. Run cells with Shift+Enter

## Expected Results

### Data Exploration
- Identify top 3 predictive features
- Discover hourly volatility patterns
- Understand distribution properties (fat tails, skewness)

### Model Comparison
- Best model identified (likely N-BEATS or TFT for neural)
- Performance metrics for all 8 models
- Error distribution analysis

### Backtest Visualization
- Average RMSE across multiple folds
- Model stability assessment (CV < 20% is excellent)
- Production readiness recommendation

## Troubleshooting

### "Database not found"
```bash
# Start PostgreSQL
cd docker && docker-compose up -d postgres

# Check status
docker-compose ps
```

### "No data available"
```bash
# Fetch historical data
python -m price_stradamus.cli.commands fetch --days 30
```

### "Module not found"
```bash
# Reinstall with dev dependencies
uv pip install -e ".[dev]"
```

### "Kernel not found"
```bash
# Create Jupyter kernel for virtual environment
python -m ipykernel install --user --name=price-stradamus
```

### "GPU not available" (Optional)
- Notebooks work on CPU (just slower)
- For GPU: Ensure PyTorch with CUDA is installed
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

## Configuration

You can modify training parameters in each notebook:

```python
# Example configuration in model_comparison.ipynb
TRAINING_CONFIG = {
    "symbol": "BTCUSDT",        # Change symbol
    "timeframe": "1m",          # Change timeframe
    "days": 30,                 # More data = better models
    "n_epochs": 30,             # Increase for better training
    "train_split": 0.70,        # Adjust split ratios
}
```

## Performance Tips

### Faster Execution
- Reduce `n_epochs` for quick experiments (10-20)
- Use fewer models in model_comparison (comment out slow ones)
- Sample data in data_exploration (e.g., every 60th row for hourly)

### Better Results
- Increase `n_epochs` to 100-200 for production
- Use more historical data (60-90 days)
- Enable GPU if available
- Tune hyperparameters based on results

## Next Steps

After completing the notebooks:

1. **Choose best model** from model_comparison.ipynb
2. **Validate robustness** with backtest_visualization.ipynb
3. **Deploy for predictions** using CLI commands:
   ```bash
   python -m price_stradamus.cli.commands predict --model nbeats --steps 5
   ```
4. **Set up monitoring** for live predictions
5. **Paper trade** before live trading

## Additional Resources

- **Main Documentation**: [../../README.md](../../README.md)
- **Quick Start Guide**: [../../QUICK-START.md](../../QUICK-START.md)
- **Code Standards**: [../../CLAUDE.md](../../CLAUDE.md)
- **Model Documentation**: [../../context/models.md](../../context/models.md)
- **Evaluation Guide**: [../../context/evaluation.md](../../context/evaluation.md)

## Contributing

Found an issue or want to improve a notebook?

1. Test your changes locally
2. Ensure notebooks run end-to-end
3. Follow code standards (type hints, docstrings)
4. Submit a pull request

## Support

- **Issues**: [GitHub Issues](https://github.com/anthropics/price-stradamus/issues)
- **Documentation**: See `context/` directory
- **CLI Help**: `python -m price_stradamus.cli.commands --help`

---

**Happy Exploring!** 🚀📊

*Last Updated*: 2024-12-11
