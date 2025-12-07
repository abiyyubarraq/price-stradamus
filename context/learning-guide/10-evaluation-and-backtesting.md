# Module 10: Evaluation and Backtesting

**Duration:** 3-4 hours | **Difficulty:** Intermediate | **Prerequisites:** Modules 07-09

## 🎯 Learning Objectives

After this module, you will:
- Understand key evaluation metrics (MAE, RMSE, MAPE, directional accuracy)
- Master walk-forward validation
- Avoid lookahead bias and data leakage
- Interpret backtesting results correctly
- Assess statistical significance
- Understand common evaluation pitfalls

---

## Why Proper Evaluation Matters

**Bad evaluation leads to:**
- False confidence in models
- Overfit models that fail in production
- Wasted resources on poor strategies
- Financial losses

**Good evaluation ensures:**
- Realistic performance estimates
- Robust models that generalize
- Informed model selection
- Confidence in deployment

---

## Evaluation Metrics Explained

### 1. MAE: Mean Absolute Error

**What it measures:** Average absolute difference between predictions and actual values.

```python
# Formula
MAE = mean(|predicted - actual|)

# Example
predicted = [100, 105, 110]
actual    = [102, 103, 108]
errors    = [2, 2, 2]
MAE = (2 + 2 + 2) / 3 = 2.0

# Implementation
import numpy as np

def calculate_mae(predicted, actual):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(predicted - actual))
```

**Interpretation:**
- MAE = 100: On average, predictions are off by $100
- Lower is better
- Easy to interpret (same units as target)

**TypeScript analogy:**
```typescript
const mae = (predicted: number[], actual: number[]): number => {
  const errors = predicted.map((p, i) => Math.abs(p - actual[i]));
  return errors.reduce((sum, e) => sum + e, 0) / errors.length;
};
```

---

### 2. RMSE: Root Mean Squared Error

**What it measures:** Square root of average squared differences (penalizes large errors more).

```python
# Formula
RMSE = sqrt(mean((predicted - actual)²))

# Example
predicted = [100, 105, 110]
actual    = [102, 103, 108]
errors    = [2, 2, 2]
squared   = [4, 4, 4]
RMSE = sqrt((4 + 4 + 4) / 3) = sqrt(4) = 2.0

# With outlier
predicted = [100, 105, 110]
actual    = [102, 103, 120]  # Big error!
errors    = [2, 2, 10]
squared   = [4, 4, 100]
MAE  = (2 + 2 + 10) / 3 = 4.67
RMSE = sqrt((4 + 4 + 100) / 3) = sqrt(36) = 6.0  # Penalizes outlier more!

# Implementation
def calculate_rmse(predicted, actual):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((predicted - actual) ** 2))
```

**RMSE vs MAE:**
- RMSE penalizes large errors more heavily
- RMSE ≥ MAE always
- If RMSE >> MAE, you have outliers

---

### 3. MAPE: Mean Absolute Percentage Error

**What it measures:** Average absolute error as percentage of actual value.

```python
# Formula
MAPE = mean(|predicted - actual| / |actual|) × 100%

# Example
predicted = [100, 105, 110]
actual    = [102, 103, 108]
errors    = [2, 2, 2]
percent   = [2/102, 2/103, 2/108] = [1.96%, 1.94%, 1.85%]
MAPE = (1.96 + 1.94 + 1.85) / 3 = 1.92%

# Implementation
def calculate_mape(predicted, actual):
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs((predicted - actual) / actual)) * 100
```

**Interpretation:**
- MAPE = 5%: Predictions are off by 5% on average
- **Scale-independent** (compare across different price ranges)
- Problem: Undefined when actual = 0

**When to use:**
- Comparing models across different assets (BTC vs ETH)
- When percentage matters more than absolute error

---

### 4. Directional Accuracy

**What it measures:** How often the model predicts the correct direction (up/down).

```python
# Did we predict the direction correctly?
# If price goes up, did we predict up?
# If price goes down, did we predict down?

def calculate_directional_accuracy(predicted, actual):
    """Calculate percentage of correct direction predictions."""
    # Calculate actual direction
    actual_direction = np.sign(actual[1:] - actual[:-1])

    # Calculate predicted direction
    pred_direction = np.sign(predicted[1:] - predicted[:-1])

    # Compare directions
    correct = (actual_direction == pred_direction)

    return np.mean(correct) * 100

# Example
predicted = [100, 105, 103, 110]  # up, down, up
actual    = [102, 106, 104, 108]  # up, down, up
# All directions correct! Directional accuracy = 100%

predicted = [100, 95, 100, 105]   # down, up, up
actual    = [102, 106, 104, 108]  # up, down, up
# 0/3 correct! Directional accuracy = 0%
```

**Why it matters:**
- For trading, direction matters more than exact price
- A model with high MAE but high directional accuracy can still be profitable

---

### 5. R² Score: Coefficient of Determination

**What it measures:** How much variance in actual values is explained by predictions.

```python
# Formula
# R² = 1 - (sum of squared residuals / total sum of squares)
# R² = 1 - (SS_res / SS_tot)

def calculate_r2(predicted, actual):
    """Calculate R² score."""
    # Residual sum of squares
    ss_res = np.sum((actual - predicted) ** 2)

    # Total sum of squares
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    return 1 - (ss_res / ss_tot)

# Interpretation
# R² = 1.0:  Perfect predictions
# R² = 0.5:  Model explains 50% of variance
# R² = 0.0:  Model is as good as predicting the mean
# R² < 0.0:  Model is worse than predicting the mean!
```

---

## Walk-Forward Validation

### The Problem with Standard Train/Test Split

```python
# ❌ BAD: Random split (shuffles data)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# This breaks temporal order! You're training on future data!

# ✅ GOOD: Temporal split
split_point = int(len(data) * 0.8)
train_data = data[:split_point]
test_data = data[split_point:]
```

### Walk-Forward Validation Explained

```
┌──────────────────────────────────────────────────────────────────┐
│             WALK-FORWARD VALIDATION                               │
│                                                                   │
│                     ┌────────────────┐                           │
│                     │  Full Dataset  │                           │
│                     └────────┬───────┘                           │
│                              │                                    │
│                              ▼                                    │
│                     ┌────────────────┐                           │
│                     │    Split 1     │                           │
│                     └────────┬───────┘                           │
│                              │                                    │
│                              ▼                                    │
│                  ┌──────────────────┐                            │
│                  │ Train on 70%     │                            │
│                  │ [━━━━━━━━━━━━━━] │                            │
│                  └────────┬─────────┘                            │
│                           │                                       │
│                           ▼                                       │
│                  ┌──────────────────┐                            │
│                  │Test on next 10%  │                            │
│                  │      [━━]        │                            │
│                  └────────┬─────────┘                            │
│                           │                                       │
│                           ▼                                       │
│                     ┌────────────────┐                           │
│                     │    Split 2     │                           │
│                     └────────┬───────┘                           │
│                              │                                    │
│                              ▼                                    │
│                  ┌──────────────────┐                            │
│                  │Train on 70-80%   │                            │
│                  │  [━━━━━━━━━━━━━━]│                            │
│                  └────────┬─────────┘                            │
│                           │                                       │
│                           ▼                                       │
│                  ┌──────────────────┐                            │
│                  │Test on next 10%  │                            │
│                  │      [━━]        │                            │
│                  └────────┬─────────┘                            │
│                           │                                       │
│                           ▼                                       │
│                     ┌────────────────┐                           │
│                     │    Split 3     │                           │
│                     └────────┬───────┘                           │
│                              │                                    │
│                              ▼                                    │
│                  ┌──────────────────┐                            │
│                  │Train on 80-90%   │                            │
│                  │[━━━━━━━━━━━━━━━━]│                            │
│                  └────────┬─────────┘                            │
│                           │                                       │
│                           ▼                                       │
│                  ┌──────────────────┐                            │
│                  │Test on last 10%  │                            │
│                  │      [━━]        │                            │
│                  └────────┬─────────┘                            │
│                           │                                       │
│                           ▼                                       │
│                  ┌──────────────────┐                            │
│                  │ Average Results  │                            │
│                  │  (Final Score)   │                            │
│                  └──────────────────┘                            │
│                                                                   │
│  Training window moves forward, always predicting future data    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Why walk-forward?**
- Simulates real trading (only use past data)
- Tests model on multiple time periods
- More realistic performance estimate
- Detects overfitting to specific time periods

### Walk-Forward Implementation

```python
def walk_forward_validation(
    data: pd.DataFrame,
    model_class: type,
    train_size: int = 1000,
    test_size: int = 100,
    step_size: int = 100,
):
    """Perform walk-forward validation.

    Args:
        data: Time series data
        model_class: Model to test
        train_size: Size of training window
        test_size: Size of test window
        step_size: How far to move window each iteration
    """
    results = []

    # Slide window across data
    for i in range(0, len(data) - train_size - test_size, step_size):
        # Split data
        train_data = data[i:i+train_size]
        test_data = data[i+train_size:i+train_size+test_size]

        # Train model
        model = model_class()
        model.fit(train_data)

        # Predict
        predictions = model.predict(len(test_data), train_data)

        # Evaluate
        mae = calculate_mae(predictions, test_data['close'])
        results.append({
            'start_date': test_data.index[0],
            'mae': mae,
        })

    # Aggregate results
    avg_mae = np.mean([r['mae'] for r in results])
    std_mae = np.std([r['mae'] for r in results])

    return {
        'mean_mae': avg_mae,
        'std_mae': std_mae,
        'results': results,
    }
```

---

## Avoiding Common Pitfalls

### 1. Lookahead Bias

**Problem:** Using future information to make predictions

```python
# ❌ BAD: Using future data
def add_features(df):
    # This looks ahead! Uses future values to normalize
    df['normalized'] = (df['close'] - df['close'].mean()) / df['close'].std()
    return df

# ✅ GOOD: Only use past data
def add_features(df):
    # Rolling mean and std only use past values
    df['normalized'] = (
        (df['close'] - df['close'].rolling(100).mean()) /
        df['close'].rolling(100).std()
    )
    return df
```

**Other lookahead examples:**
- Using full dataset statistics (mean, std, min, max)
- Forward-filling missing values
- Using indicators that require future data

---

### 2. Data Leakage

**Problem:** Test data influences training

```python
# ❌ BAD: Normalize before split
X_normalized = (X - X.mean()) / X.std()  # Uses test data statistics!
X_train, X_test = X_normalized[:800], X_normalized[800:]

# ✅ GOOD: Normalize after split
X_train, X_test = X[:800], X[800:]
train_mean = X_train.mean()
train_std = X_train.std()
X_train_norm = (X_train - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std  # Use train statistics
```

---

### 3. Overfitting to Validation Set

**Problem:** Tuning hyperparameters until validation performance is great

```python
# ❌ BAD: Tune on validation set, report validation performance
best_mae = float('inf')
for learning_rate in [0.001, 0.01, 0.1]:
    model.fit(train_data, learning_rate=learning_rate)
    mae = evaluate(model, validation_data)
    if mae < best_mae:
        best_mae = mae

print(f"Performance: {best_mae}")  # This is optimistic!

# ✅ GOOD: Tune on validation set, report test performance
best_lr = None
best_val_mae = float('inf')

for learning_rate in [0.001, 0.01, 0.1]:
    model.fit(train_data, learning_rate=learning_rate)
    mae = evaluate(model, validation_data)
    if mae < best_val_mae:
        best_val_mae = mae
        best_lr = learning_rate

# Retrain with best hyperparameters
model.fit(train_data + validation_data, learning_rate=best_lr)
test_mae = evaluate(model, test_data)
print(f"Performance: {test_mae}")  # More realistic
```

---

### 4. Not Accounting for Transaction Costs

**Problem:** Ignoring costs makes strategies look better than they are

```python
# ❌ BAD: Ignoring costs
def backtest(predictions, prices):
    position = 0
    pnl = 0

    for i in range(len(predictions) - 1):
        signal = 1 if predictions[i+1] > prices[i] else -1

        # Buy or sell
        if signal != position:
            pnl += (prices[i+1] - prices[i]) * signal
            position = signal

    return pnl

# ✅ GOOD: Include transaction costs
def backtest_with_costs(predictions, prices, commission=0.001):
    position = 0
    pnl = 0

    for i in range(len(predictions) - 1):
        signal = 1 if predictions[i+1] > prices[i] else -1

        # Buy or sell
        if signal != position:
            # Pay commission on trade
            cost = prices[i] * abs(signal - position) * commission
            pnl -= cost

            pnl += (prices[i+1] - prices[i]) * signal
            position = signal

    return pnl
```

---

## Statistical Significance

### Is My Model Actually Better?

Just because Model A has MAE=100 and Model B has MAE=105 doesn't mean A is better!
We need to test if the difference is **statistically significant**.

### Paired t-test

```python
from scipy import stats

def compare_models(errors_model_a, errors_model_b, alpha=0.05):
    """Compare two models using paired t-test.

    Args:
        errors_model_a: Errors from model A
        errors_model_b: Errors from model B
        alpha: Significance level (typically 0.05)

    Returns:
        dict with results
    """
    # Calculate error differences
    error_diff = errors_model_a - errors_model_b

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(errors_model_a, errors_model_b)

    # Interpret
    is_significant = p_value < alpha

    return {
        'mean_diff': np.mean(error_diff),
        'std_diff': np.std(error_diff),
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'conclusion': (
            f"Model A is significantly better (p={p_value:.4f})"
            if is_significant and np.mean(error_diff) < 0
            else f"No significant difference (p={p_value:.4f})"
        )
    }

# Example
errors_a = np.array([2, 3, 2, 4, 3, 2, 3])
errors_b = np.array([3, 4, 3, 5, 4, 3, 4])

result = compare_models(errors_a, errors_b)
print(result['conclusion'])
```

---

## Backtesting Framework

### Complete Backtesting Implementation

```python
# src/price_stradamus/evaluation/backtester.py

class Backtester:
    """Walk-forward backtesting framework."""

    def __init__(
        self,
        model,
        train_size: int = 1000,
        test_size: int = 100,
        step_size: int = 100,
    ):
        self.model = model
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def run(self, data: TimeSeries) -> dict:
        """Run walk-forward backtest."""
        results = {
            'mae': [],
            'rmse': [],
            'mape': [],
            'directional_accuracy': [],
            'predictions': [],
            'actuals': [],
        }

        # Walk forward through data
        for i in range(0, len(data) - self.train_size - self.test_size, self.step_size):
            # Split
            train = data[i:i+self.train_size]
            test = data[i+self.train_size:i+self.train_size+self.test_size]

            # Train
            self.model.fit(train)

            # Predict
            predictions = self.model.predict(n=len(test), series=train)

            # Evaluate
            results['predictions'].append(predictions.values())
            results['actuals'].append(test.values())
            results['mae'].append(calculate_mae(predictions.values(), test.values()))
            results['rmse'].append(calculate_rmse(predictions.values(), test.values()))
            results['mape'].append(calculate_mape(predictions.values(), test.values()))
            results['directional_accuracy'].append(
                calculate_directional_accuracy(predictions.values(), test.values())
            )

        # Aggregate
        return {
            'mean_mae': np.mean(results['mae']),
            'std_mae': np.std(results['mae']),
            'mean_rmse': np.mean(results['rmse']),
            'mean_mape': np.mean(results['mape']),
            'mean_directional_accuracy': np.mean(results['directional_accuracy']),
            'detailed_results': results,
        }
```

---

## Interpreting Results

### What's a "Good" Score?

It depends on:
- Asset volatility
- Prediction horizon
- Baseline comparison

```python
# Always compare to baselines
baseline_scores = {
    'last_value': evaluate_last_value_model(data),
    'moving_average': evaluate_moving_average(data),
    'linear_trend': evaluate_linear_trend(data),
}

model_score = evaluate(your_model, data)

# Your model should beat all baselines!
for name, score in baseline_scores.items():
    improvement = (score - model_score) / score * 100
    print(f"vs {name}: {improvement:.1f}% better")
```

### Red Flags

- **Too good to be true**: MAE < 0.1% of price range → likely data leakage
- **Directional accuracy >> 60%**: Might be overfitting or lookahead bias
- **High train, low test accuracy**: Overfitting
- **Inconsistent across time periods**: Model not robust

---

## Practice Exercises

### Exercise 1: Implement Metrics (30 minutes)

Implement all metrics from scratch:

```python
def calculate_all_metrics(predicted, actual):
    """Calculate MAE, RMSE, MAPE, directional accuracy, R²."""
    # TODO: Implement
    pass
```

### Exercise 2: Walk-Forward Validation (45 minutes)

Implement walk-forward validation and compare N-BEATS vs ARIMA:

```python
results_nbeats = walk_forward_validation(data, NBEATSModel)
results_arima = walk_forward_validation(data, ARIMAModel)

# Which model is better?
# Is the difference statistically significant?
```

---

## Next Steps

**Next Module:** [11: CLI and Integration →](11-cli-and-integration.md)

Learn how to use all CLI commands and integrate everything together.

---

## Summary Checklist

- [ ] I understand MAE, RMSE, MAPE, and directional accuracy
- [ ] I know how to implement walk-forward validation
- [ ] I can avoid lookahead bias and data leakage
- [ ] I understand statistical significance testing
- [ ] I know how to interpret backtesting results
- [ ] I can identify evaluation red flags

**Continue to:** [Module 11: CLI and Integration](11-cli-and-integration.md)
