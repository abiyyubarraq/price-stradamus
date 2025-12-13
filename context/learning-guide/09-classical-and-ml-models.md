# Module 09: Classical and ML Models

**Duration:** 3-4 hours | **Difficulty:** Intermediate | **Prerequisites:** Modules 07-08

## 🎯 Learning Objectives

After this module, you will:
- Understand ARIMA and when it works best
- Know how Prophet handles seasonality
- Master XGBoost for feature-based forecasting
- Understand  ensemble methods
- Know when to use classical/ML vs neural networks

---

## 1. ARIMA: AutoRegressive Integrated Moving Average

### What is ARIMA?

ARIMA is a **statistical forecasting method** that models time series based on:
- **AR (AutoRegressive)**: Past values predict future (like "price tends to continue its trend")
- **I (Integrated)**: Differencing to make series stationary
- **MA (Moving Average)**: Past errors predict future (like "if we overshot last time, correct this time")

### ARIMA Parameters: (p, d, q)

```python
# ARIMA(p, d, q)
# p = AutoRegressive order (how many past values to use)
# d = Differencing order (how many times to difference)
# q = Moving Average order (how many past errors to use)

# Examples:
ARIMA(1, 0, 0)  # AR(1): y_t = c + φ*y_{t-1}
ARIMA(0, 1, 1)  # IMA(1,1): Δy_t = c + θ*ε_{t-1}
ARIMA(1, 1, 1)  # Full ARIMA: Δy_t = c + φ*Δy_{t-1} + θ*ε_{t-1}
```

**TypeScript analogy:**
```typescript
// AR(1) is like a simple recursive function
let predict = (prevPrice: number, coef: number) => {
  return coef * prevPrice;
};

// ARIMA is like a more complex function with memory
let predict_arima = (
  pastPrices: number[],
  pastErrors: number[],
  coefs: {ar: number[], ma: number[]}
) => {
  let ar_part = sum(pastPrices.map((p, i) => p * coefs.ar[i]));
  let ma_part = sum(pastErrors.map((e, i) => e * coefs.ma[i]));
  return ar_part + ma_part;
};
```

### Stationarity: The Key Requirement

ARIMA requires **stationary** data (constant mean and variance over time).

```python
import pandas as pd
import numpy as np

# Check stationarity with Augmented Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series):
    """Check if time series is stationary."""
    result = adfuller(series, autolag='AIC')

    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")

    if result[1] <= 0.05:
        print("✅ Series is stationary")
    else:
        print("❌ Series is NOT stationary - need differencing")

# Example: Bitcoin prices are usually non-stationary
check_stationarity(btc_prices)  # Likely not stationary

# Make stationary by differencing
btc_returns = btc_prices.diff().dropna()
check_stationarity(btc_returns)  # More likely stationary
```

### ARIMA in Price Stradamus

```python
# src/price_stradamus/models/classical/arima.py
from darts.models import ARIMA as DartsARIMA
from darts import TimeSeries

class ARIMAModel(BaseModel):
    def __init__(
        self,
        p: int = 1,  # AR order
        d: int = 1,  # Differencing
        q: int = 1,  # MA order
    ):
        """Initialize ARIMA model.

        Args:
            p: AutoRegressive order (typically 0-5)
            d: Differencing order (typically 0-2)
            q: Moving Average order (typically 0-5)
        """
        super().__init__(name="arima")
        self.p = p
        self.d = d
        self.q = q

        self.model = DartsARIMA(p=p, d=d, q=q)

    def fit(self, train_series: TimeSeries, val_series: TimeSeries | None = None):
        """Train ARIMA model."""
        self.model.fit(train_series)
        self._is_fitted = True

    def predict(self, n: int, series: TimeSeries) -> TimeSeries:
        """Make predictions."""
        return self.model.predict(n=n)
```

### Finding Best ARIMA Parameters

```python
from itertools import product

def find_best_arima(train_series, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
    """Grid search for best ARIMA parameters."""
    best_aic = float('inf')
    best_params = None

    # Try all combinations
    for p, d, q in product(range(*p_range), range(*d_range), range(*q_range)):
        try:
            model = ARIMA(p=p, d=d, q=q)
            model.fit(train_series)

            # AIC = Akaike Information Criterion (lower is better)
            aic = model.model.fit.aic

            if aic < best_aic:
                best_aic = aic
                best_params = (p, d, q)

        except:
            continue  # Some combinations might not work

    print(f"Best ARIMA{best_params} with AIC = {best_aic:.2f}")
    return best_params
```

### When to Use ARIMA

✅ **Use ARIMA when:**
- Small dataset (<1000 points)
- Stationary or easily made stationary
- Need interpretable model
- Strong autocorrelation in data

❌ **Don't use ARIMA when:**
- Non-stationary data that can't be differenced
- Complex non-linear patterns
- Multiple input features (ARIMA is univariate)
- Large datasets (neural networks better)

---

## 2. Prophet: Facebook's Time Series Model

### What Makes Prophet Special?

Prophet is designed for **business forecasting** with:
- **Automatic seasonality detection** (daily, weekly, yearly)
- **Holiday effects** built-in
- **Trend changepoint detection** (when trends shift)
- **Robust to missing data**

### Prophet Components

```python
# Prophet decomposes time series into:
y(t) = trend(t) + seasonality(t) + holiday(t) + error(t)

# Example:
# Bitcoin price = long-term trend + weekly pattern + holiday effects + noise
```

### Prophet in Price Stradamus

```python
# src/price_stradamus/models/classical/prophet.py
from darts.models import Prophet as DartsProphet
from darts import TimeSeries

class ProphetModel(BaseModel):
    def __init__(
        self,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
    ):
        """Initialize Prophet model.

        Args:
            seasonality_mode: "additive" or "multiplicative"
            yearly_seasonality: Include yearly patterns
            weekly_seasonality: Include weekly patterns
            daily_seasonality: Include daily patterns
            changepoint_prior_scale: Trend flexibility (higher = more flexible)
        """
        super().__init__(name="prophet")

        self.model = DartsProphet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
        )

    def fit(self, train_series: TimeSeries, val_series: TimeSeries | None = None):
        """Train Prophet model."""
        self.model.fit(train_series)
        self._is_fitted = True

    def predict(self, n: int, series: TimeSeries) -> TimeSeries:
        """Make predictions."""
        return self.model.predict(n=n)
```

### Understanding Prophet Parameters

#### Seasonality Mode

```python
# Additive: seasonality is constant over time
# Bitcoin: $50k base + $2k weekly swing = $52k
seasonality_mode="additive"

# Multiplicative: seasonality scales with level
# Bitcoin: $50k base * 1.04 weekly multiplier = $52k
# When price doubles, seasonality doubles too
seasonality_mode="multiplicative"  # Better for prices!
```

#### Changepoint Prior Scale

```python
# Controls how flexible the trend can be

# Low (0.001): Smooth trend, won't adapt to changes
changepoint_prior_scale=0.001

# Medium (0.05): Balanced (default)
changepoint_prior_scale=0.05

# High (0.5): Very flexible, might overfit
changepoint_prior_scale=0.5
```

### When to Use Prophet

✅ **Use Prophet when:**
- Strong seasonal patterns (weekly, daily)
- Missing data points
- Trend changes over time
- Small to medium datasets
- Need interpretable seasonality

❌ **Don't use Prophet when:**
- No clear seasonality
- Need multi-step forecasting (Prophet is slow for this)
- Very short-term predictions (<1 hour)

---

## 3. XGBoost: Gradient Boosting Trees

### How XGBoost Works

XGBoost builds **many decision trees sequentially**, where each tree tries to correct the mistakes of previous trees.

```python
# Simplified concept
predictions = 0

# Build trees sequentially
for i in range(num_trees):
    # Calculate errors from current predictions
    errors = actual_values - predictions

    # Build new tree to predict errors
    tree = DecisionTree()
    tree.fit(features, errors)

    # Add tree's predictions (weighted)
    predictions += learning_rate * tree.predict(features)
```

**Analogy:** XGBoost is like a team of experts
- First expert makes a guess
- Second expert corrects the first's mistakes
- Third expert corrects the second's mistakes
- Combine all experts' opinions

### Decision Trees Explained

```python
# A decision tree for Bitcoin prediction
"""
                   RSI < 70?
                   /       \
                 Yes        No
                 /           \
         Volume high?     Price < $50k?
           /    \           /      \
         Yes    No        Yes      No
         /      |         |        \
     Predict  Predict  Predict  Predict
     $100     $50      -$200    -$100
"""

# XGBoost builds hundreds of these trees
```

### XGBoost in Price Stradamus

```python
# src/price_stradamus/models/ml/xgboost.py
from darts.models import XGBModel as DartsXGB

class XGBoostModel(BaseModel):
    def __init__(
        self,
        lags: int | list[int] = 60,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        lags_future_covariates: list[int] | None = None,
    ):
        """Initialize XGBoost model.

        Args:
            lags: Past values to use as features (60 = last 60 candles)
            n_estimators: Number of trees to build
            max_depth: Maximum tree depth (controls complexity)
            learning_rate: How much each tree contributes
            lags_future_covariates: Future features (e.g., time indicators)
        """
        super().__init__(name="xgboost")

        self.model = DartsXGB(
            lags=lags,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            lags_future_covariates=lags_future_covariates,
        )

    def fit(self, train_series: TimeSeries, val_series: TimeSeries | None = None):
        """Train XGBoost model."""
        self.model.fit(train_series, val_series=val_series)
        self._is_fitted = True
```

### XGBoost Hyperparameters

```python
# Conservative (less overfitting)
model = XGBoostModel(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.05,
)

# Balanced (recommended)
model = XGBoostModel(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
)

# Aggressive (more capacity)
model = XGBoostModel(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
)
```

### Feature Importance

```python
# XGBoost can tell you which features matter most
feature_importance = model.model.get_feature_importance()

# Example output:
# rsi_14: 0.35  (35% importance)
# volume: 0.25  (25% importance)
# sma_20: 0.20  (20% importance)
# lag_1:  0.15  (15% importance)
# others: 0.05  (5% importance)
```

### When to Use XGBoost

✅ **Use XGBoost when:**
- Have engineered features (RSI, MACD, etc.)
- Need feature importance
- Medium-sized dataset (1k-100k points)
- Fast predictions required
- Tabular/structured data

❌ **Don't use XGBoost when:**
- No feature engineering done (use N-BEATS)
- Very small dataset (<500 points)
- Raw sequential data (LSTM better)

---

## 4. : Ensemble of Trees

### How  Works

 builds **many decision trees in parallel** and averages their predictions.

```python
# Simplified concept
predictions = []

# Build trees in parallel (independent)
for i in range(num_trees):
    # Random subset of data
    sample = random_sample(data)

    # Random subset of features
    features = random_features(all_features)

    # Build tree
    tree = DecisionTree()
    tree.fit(sample, features)

    # Store prediction
    predictions.append(tree.predict(new_data))

# Average all predictions
final_prediction = mean(predictions)
```

**Key difference from XGBoost:**
- ****: Trees built independently in parallel
- **XGBoost**: Trees built sequentially, each correcting previous

###  in Price Stradamus

```python
# src/price_stradamus/models/ml/.py
from darts.models import  as DartsRF

class Model(BaseModel):
    def __init__(
        self,
        lags: int | list[int] = 60,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        lags_future_covariates: list[int] | None = None,
    ):
        """Initialize  model.

        Args:
            lags: Past values to use as features
            n_estimators: Number of trees
            max_depth: Maximum tree depth (None = unlimited)
            min_samples_split: Minimum samples to split node
            lags_future_covariates: Future features
        """
        super().__init__(name="")

        self.model = DartsRF(
            lags=lags,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            lags_future_covariates=lags_future_covariates,
        )
```

###  vs XGBoost

| Aspect |  | XGBoost |
|--------|--------------|---------|
| **Training** | Parallel (faster) | Sequential (slower) |
| **Overfitting** | Less prone | More prone (needs tuning) |
| **Accuracy** | Good | Excellent |
| **Interpretability** | High | Medium |
| **Best for** | Stability, less tuning | Maximum performance |

### When to Use 

✅ **Use  when:**
- Want stable predictions (less overfitting)
- Limited time for hyperparameter tuning
- Need feature importance
- Have engineered features

❌ **Don't use  when:**
- Need maximum accuracy (XGBoost better)
- No feature engineering (N-BEATS better)

---

## Model Comparison Summary

### Quick Decision Guide

```python
def choose_model(
    data_size: int,
    has_features: bool,
    need_interpretable: bool,
    has_seasonality: bool,
):
    """Choose best model based on characteristics."""

    if data_size < 500:
        return "Prophet" if has_seasonality else "ARIMA"

    if need_interpretable:
        if has_seasonality:
            return "Prophet"
        return "ARIMA" if data_size < 1000 else ""

    if has_features:
        return "XGBoost"  # Best with engineered features

    # Large dataset, raw data
    return "N-BEATS"  # Best neural network for raw time series
```

### Performance Characteristics

| Model | Training Speed | Prediction Speed | Interpretability | Data Needed |
|-------|----------------|------------------|------------------|-------------|
| **ARIMA** | Fast | Fast | High | <1k |
| **Prophet** | Fast | Medium | High | <5k |
| **XGBoost** | Medium | Fast | Medium | 1k-100k |
| **** | Fast | Fast | High | 1k-100k |
| **N-BEATS** | Slow | Medium | Medium | >1k |
| **LSTM** | Slow | Medium | Low | >500 |
| **TCN** | Medium | Fast | Low | >500 |
| **TFT** | Very Slow | Slow | Medium | >2k |

---

## Practice Exercises

### Exercise 1: ARIMA Parameter Tuning (30 minutes)

Find the best ARIMA parameters for Bitcoin data:

```python
from price_stradamus.models.classical.arima import ARIMAModel

# Load data
df = await db.get_ohlcv("BTCUSDT", "1m", limit=1000)

# TODO:
# 1. Check if data is stationary
# 2. If not, apply differencing
# 3. Try different (p,d,q) combinations
# 4. Compare AIC scores
# 5. Train best model and evaluate
```

### Exercise 2: Feature Importance (30 minutes)

Train XGBoost and analyze which features matter:

```python
from price_stradamus.models.ml.xgboost import XGBoostModel

# Train model with features
model = XGBoostModel()
model.fit(train_data_with_features)

# TODO:
# 1. Get feature importance
# 2. Plot top 10 features
# 3. Try removing least important features
# 4. Does performance improve?
```

### Exercise 3: Model Ensemble (45 minutes)

Combine predictions from multiple models:

```python
# TODO:
# 1. Train ARIMA, Prophet, XGBoost, 
# 2. Get predictions from each
# 3. Try different ensemble methods:
#    - Simple average
#    - Weighted average (by validation performance)
#    - Median
# 4. Does ensemble beat individual models?
```

---

## Next Steps

**Next Module:** [10: Evaluation and Backtesting →](10-evaluation-and-backtesting.md)

Learn how to properly evaluate models and avoid common pitfalls.

---

## Summary Checklist

- [ ] I understand how ARIMA works and its parameters
- [ ] I know when to use Prophet for seasonal data
- [ ] I understand XGBoost gradient boosting
- [ ] I can compare  vs XGBoost
- [ ] I know when to use classical/ML vs neural models
- [ ] I can choose the right model for my data

**Continue to:** [Module 10: Evaluation and Backtesting](10-evaluation-and-backtesting.md)
