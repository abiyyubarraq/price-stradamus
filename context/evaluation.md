# Price Stradamus - Evaluation Methodology

## Overview

Proper evaluation is critical for time series forecasting. This document describes metrics, backtesting strategies, and best practices to avoid common pitfalls like lookahead bias and overfitting.

---

## Evaluation Metrics

### 1. Mean Absolute Error (MAE)

**Formula**:
```
MAE = (1/n) * Σ|y_true - y_pred|
```

**Interpretation**:
- Average absolute prediction error
- Same units as target variable (e.g., USD for Bitcoin price)
- Easy to interpret
- Less sensitive to outliers than MSE

**When to Use**:
- Need interpretable error in original units
- Outliers shouldn't dominate the metric
- Comparing models on same dataset

**Example**:
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: ${mae:.2f}")  # e.g., "MAE: $45.23"
```

---

### 2. Root Mean Squared Error (RMSE)

**Formula**:
```
RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)
```

**Interpretation**:
- Square root of average squared errors
- Same units as target variable
- Penalizes large errors more than MAE
- Always ≥ MAE

**When to Use**:
- Large errors are particularly bad
- Want to penalize outlier predictions
- Standard metric for comparisons

**Example**:
```python
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_true, y_pred, squared=False)
print(f"RMSE: ${rmse:.2f}")
```

---

### 3. Mean Absolute Percentage Error (MAPE)

**Formula**:
```
MAPE = (100/n) * Σ|((y_true - y_pred) / y_true)|
```

**Interpretation**:
- Percentage error (scale-independent)
- Easy to compare across different price ranges
- Undefined if y_true = 0
- Asymmetric (penalizes over-predictions more)

**When to Use**:
- Comparing models across different symbols
- Need scale-independent metric
- Business stakeholders prefer percentages

**Example**:
```python
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape*100:.2f}%")  # e.g., "MAPE: 1.23%"
```

---

### 4. Symmetric MAPE (sMAPE)

**Formula**:
```
sMAPE = (100/n) * Σ(2 * |y_true - y_pred| / (|y_true| + |y_pred|))
```

**Interpretation**:
- Symmetric version of MAPE
- Range: 0% to 200%
- Handles over/under-predictions equally
- Undefined if both y_true and y_pred = 0

**When to Use**:
- Need symmetric error metric
- MAPE is too asymmetric for use case

**Example**:
```python
def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return 100 * np.mean(numerator / denominator)
```

---

### 5. R² Score (Coefficient of Determination)

**Formula**:
```
R² = 1 - (SS_res / SS_tot)
where:
  SS_res = Σ(y_true - y_pred)²
  SS_tot = Σ(y_true - ȳ)²
```

**Interpretation**:
- Proportion of variance explained
- Range: (-∞, 1], where 1 = perfect fit
- Can be negative if model is worse than mean baseline
- Not suitable for time series (inflated by trend)

**When to Use**:
- Stationary time series
- Comparing to baseline models
- Understanding model fit quality

**Example**:
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")
```

---

### 6. Directional Accuracy

**Formula**:
```
DA = (1/n) * Σ[sign(y_true - y_true_lag1) == sign(y_pred - y_true_lag1)]
```

**Interpretation**:
- Percentage of correct up/down predictions
- Most important for trading strategies
- Ignores magnitude of error
- Random baseline = 50%

**When to Use**:
- Trading applications (long/short decisions)
- Model is used for direction, not exact price
- Complementary to error metrics

**Example**:
```python
def directional_accuracy(y_true, y_pred):
    """Calculate directional accuracy."""
    actual_direction = np.sign(y_true[1:] - y_true[:-1])
    pred_direction = np.sign(y_pred[1:] - y_true[:-1])  # Predict change from last true
    return np.mean(actual_direction == pred_direction)

da = directional_accuracy(y_true, y_pred)
print(f"Directional Accuracy: {da*100:.2f}%")  # e.g., "DA: 54.32%"
```

---

### 7. Max Error

**Formula**:
```
Max Error = max(|y_true - y_pred|)
```

**Interpretation**:
- Worst-case prediction error
- Identifies catastrophic failures
- Sensitive to single outlier

**When to Use**:
- Risk management
- Understanding worst-case scenarios
- Identifying model failures

---

### 8. Financial Metrics

**Sharpe Ratio** (if using for trading):
```python
def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
```

**Max Drawdown**:
```python
def max_drawdown(prices: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    return np.min(drawdown)
```

---

## Backtesting Strategy

### Walk-Forward Validation

**Goal**: Simulate realistic forecasting scenario

**Method**: Expanding or Sliding Window

```
Expanding Window:
├─────────────train─────────────┤test├──
├───────────────train───────────────┤test├──
├─────────────────train─────────────────┤test├──

Sliding Window (Fixed Size):
├───train───┤test├──
    ├───train───┤test├──
        ├───train───┤test├──
```

**Implementation**:
```python
class WalkForwardBacktester:
    def __init__(
        self,
        model: BaseModel,
        train_size: int = 1000,
        test_size: int = 100,
        step_size: int = 100,
        window_type: str = "expanding",
    ):
        self.model = model
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.window_type = window_type

    def run(self, data: TimeSeries) -> BacktestResults:
        """Run walk-forward backtesting."""
        results = []

        start = 0
        while start + self.train_size + self.test_size <= len(data):
            # Define train/test windows
            if self.window_type == "expanding":
                train = data[:start + self.train_size]
            else:  # sliding
                train = data[start:start + self.train_size]

            test = data[start + self.train_size:start + self.train_size + self.test_size]

            # Train model
            self.model.fit(train)

            # Predict
            predictions = self.model.predict(n=len(test))

            # Calculate metrics
            metrics = calculate_all_metrics(test, predictions)
            results.append({
                "fold": len(results),
                "train_start": train.start_time(),
                "train_end": train.end_time(),
                "test_start": test.start_time(),
                "test_end": test.end_time(),
                **metrics,
            })

            # Move forward
            start += self.step_size

        return BacktestResults(results)
```

### K-Fold Cross-Validation (Time Series Version)

**TimeSeriesSplit**:
```python
from sklearn.model_selection import TimeSeriesSplit

def timeseries_cv(data: TimeSeries, n_splits: int = 5) -> list[tuple]:
    """Time series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []

    for train_idx, test_idx in tscv.split(data.values()):
        train = data[train_idx]
        test = data[test_idx]
        splits.append((train, test))

    return splits
```

---

## Avoiding Common Pitfalls

### 1. Lookahead Bias

**Problem**: Using future information to make past predictions

**Examples**:
- Normalizing entire dataset before split
- Using statistics calculated on test data
- Feature engineering with future values

**Solution**:
```python
# BAD: Lookahead bias
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)  # Uses future data!
train, test = temporal_split(data_scaled)

# GOOD: Fit on train only
train, test = temporal_split(data)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)  # Use train statistics
```

### 2. Data Leakage

**Problem**: Information from test set leaks into training

**Examples**:
- Features that directly contain target
- Using same features for multiple time steps
- Not properly lagging features

**Solution**:
```python
# BAD: Target leakage
df["next_price"] = df["close"].shift(-1)  # Uses future!
df["feature"] = df["next_price"] * 2

# GOOD: Proper lagging
df["price_lag1"] = df["close"].shift(1)  # Uses past
df["feature"] = df["price_lag1"] * 2
```

### 3. Overfitting

**Problem**: Model memorizes training data, fails on new data

**Signs**:
- Train error << Validation error
- Excellent backtest, terrible live trading
- Too many parameters relative to data

**Solutions**:
1. **Regularization**: Add L1/L2 penalties
2. **Dropout**: Randomly drop neurons during training
3. **Early Stopping**: Stop when validation error increases
4. **Cross-Validation**: Ensure consistent performance across folds
5. **Simpler Models**: Reduce complexity

```python
# Early stopping example
early_stopper = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=0.001,
    mode="min",
)

model.fit(train, val_data=val, callbacks=[early_stopper])
```

### 4. Stationary Assumption

**Problem**: Many models assume stationarity (constant mean/variance)

**Test for Stationarity**:
```python
from statsmodels.tsa.stattools import adfuller

def test_stationarity(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test."""
    result = adfuller(series.dropna())
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < 0.05,  # p-value < 0.05
    }
```

**Make Data Stationary**:
```python
# Differencing
df["price_diff"] = df["close"].diff()

# Log returns
df["log_return"] = np.log(df["close"] / df["close"].shift(1))

# Detrending
from scipy.signal import detrend
df["detrended"] = detrend(df["close"])
```

---

## Statistical Significance Testing

### Diebold-Mariano Test

**Purpose**: Compare two forecasting models

**Null Hypothesis**: Both models have equal predictive accuracy

```python
from scipy.stats import ttest_rel

def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
) -> tuple[float, float]:
    """
    Test if two models have significantly different accuracy.

    Returns:
        (statistic, p_value)
    """
    # Calculate loss differential
    d = errors1**2 - errors2**2

    # T-test on differential
    statistic, p_value = ttest_rel(errors1**2, errors2**2)

    return statistic, p_value

# Usage
errors_model1 = y_true - pred_model1
errors_model2 = y_true - pred_model2
stat, p_val = diebold_mariano_test(errors_model1, errors_model2)

if p_val < 0.05:
    print(f"Models are significantly different (p={p_val:.4f})")
else:
    print(f"No significant difference (p={p_val:.4f})")
```

---

## Backtesting Report

**Generated Metrics**:
```python
@dataclass
class BacktestResults:
    """Backtesting results."""
    fold_results: list[dict]

    def summary(self) -> dict:
        """Aggregate metrics across folds."""
        df = pd.DataFrame(self.fold_results)

        return {
            "num_folds": len(df),
            "mae_mean": df["mae"].mean(),
            "mae_std": df["mae"].std(),
            "rmse_mean": df["rmse"].mean(),
            "rmse_std": df["rmse"].std(),
            "mape_mean": df["mape"].mean(),
            "mape_std": df["mape"].std(),
            "directional_accuracy_mean": df["directional_accuracy"].mean(),
            "directional_accuracy_std": df["directional_accuracy"].std(),
            "best_fold": df["mae"].idxmin(),
            "worst_fold": df["mae"].idxmax(),
        }

    def plot_results(self) -> None:
        """Visualize backtest results."""
        import matplotlib.pyplot as plt

        df = pd.DataFrame(self.fold_results)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # MAE over folds
        axes[0, 0].plot(df["fold"], df["mae"])
        axes[0, 0].set_title("MAE per Fold")
        axes[0, 0].set_xlabel("Fold")
        axes[0, 0].set_ylabel("MAE")

        # Directional Accuracy
        axes[0, 1].plot(df["fold"], df["directional_accuracy"])
        axes[0, 1].axhline(0.5, color="r", linestyle="--", label="Random")
        axes[0, 1].set_title("Directional Accuracy per Fold")
        axes[0, 1].legend()

        # Error distribution
        axes[1, 0].hist(df["mae"], bins=20)
        axes[1, 0].set_title("MAE Distribution")
        axes[1, 0].set_xlabel("MAE")

        # MAPE
        axes[1, 1].plot(df["fold"], df["mape"])
        axes[1, 1].set_title("MAPE per Fold")

        plt.tight_layout()
        plt.show()
```

---

## Model Comparison

```python
def compare_models(
    models: dict[str, BaseModel],
    data: TimeSeries,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compare multiple models using cross-validation."""
    results = []

    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        backtester = WalkForwardBacktester(model)
        backtest_results = backtester.run(data)
        summary = backtest_results.summary()

        results.append({
            "model": name,
            "mae": summary["mae_mean"],
            "mae_std": summary["mae_std"],
            "rmse": summary["rmse_mean"],
            "mape": summary["mape_mean"],
            "directional_accuracy": summary["directional_accuracy_mean"],
        })

    df = pd.DataFrame(results).sort_values("mae")
    return df
```

---

## Uncertainty Quantification

### Prediction Intervals (Quantile Forecasting)

**Why Point Predictions Aren't Enough**
- Point predictions give no information about confidence
- Risk management requires uncertainty bounds
- Decision making benefits from probabilistic forecasts

```python
import numpy as np
from dataclasses import dataclass
from typing import Literal

@dataclass
class PredictionInterval:
    """Prediction with uncertainty bounds."""
    point: float
    lower: float
    upper: float
    confidence: float

    @property
    def width(self) -> float:
        """Interval width."""
        return self.upper - self.lower

    @property
    def contains(self) -> callable:
        """Check if actual value falls within interval."""
        return lambda actual: self.lower <= actual <= self.upper


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Pinball loss for quantile regression.

    Used to train quantile prediction models.
    """
    errors = y_true - y_pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


class QuantileForecaster:
    """Wrapper for quantile forecasting from point predictions."""

    def __init__(
        self,
        model,
        quantiles: list[float] = [0.05, 0.5, 0.95],
        n_bootstrap: int = 100,
    ):
        self.model = model
        self.quantiles = quantiles
        self.n_bootstrap = n_bootstrap
        self.residual_distribution = None

    def fit(self, train_data, val_data):
        """Fit model and estimate residual distribution."""
        self.model.fit(train_data, val_data=val_data)

        # Get in-sample residuals for uncertainty estimation
        predictions = self.model.predict(n=len(val_data))
        self.residuals = val_data.values() - predictions.values()

        # Fit residual distribution
        self.residual_std = np.std(self.residuals)

    def predict_quantiles(self, n: int) -> list[PredictionInterval]:
        """Generate prediction intervals."""
        point_predictions = self.model.predict(n=n).values()
        intervals = []

        for i, point in enumerate(point_predictions):
            # Scale uncertainty with forecast horizon
            horizon_scaling = np.sqrt(i + 1)
            scaled_std = self.residual_std * horizon_scaling

            # Parametric quantiles (assuming normality)
            from scipy.stats import norm
            lower = point + norm.ppf(self.quantiles[0]) * scaled_std
            upper = point + norm.ppf(self.quantiles[-1]) * scaled_std

            intervals.append(PredictionInterval(
                point=point,
                lower=lower,
                upper=upper,
                confidence=self.quantiles[-1] - self.quantiles[0],
            ))

        return intervals
```

### Conformal Prediction for Time Series

**Distribution-Free Uncertainty Quantification**

Conformal prediction provides valid prediction intervals without assuming any distribution.

```python
import numpy as np
from typing import Callable

class EnbPI:
    """
    Ensemble Batch Prediction Intervals (EnbPI).

    First conformal prediction method for time series that:
    - Doesn't require data exchangeability
    - Handles non-stationary data
    - Provides valid coverage guarantees

    Reference: Xu & Xie (2021)
    """

    def __init__(
        self,
        model_factory: Callable,
        n_bootstrap: int = 100,
        alpha: float = 0.1,  # 1 - alpha = coverage level
    ):
        self.model_factory = model_factory
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.bootstrap_models = []
        self.residual_quantile = None

    def fit(self, data: np.ndarray, train_ratio: float = 0.8) -> "EnbPI":
        """Fit ensemble of models via bootstrap."""
        n = len(data)
        n_train = int(n * train_ratio)

        # Bootstrap aggregation
        residuals = []

        for b in range(self.n_bootstrap):
            # Bootstrap sample indices
            boot_idx = np.random.choice(n_train, size=n_train, replace=True)

            # Train model on bootstrap sample
            model = self.model_factory()
            boot_data = data[boot_idx]
            model.fit(boot_data)

            self.bootstrap_models.append(model)

            # Calculate out-of-bootstrap residuals
            oob_idx = list(set(range(n_train)) - set(boot_idx))
            if len(oob_idx) > 0:
                oob_pred = model.predict(n=len(oob_idx))
                oob_residuals = np.abs(data[oob_idx] - oob_pred)
                residuals.extend(oob_residuals)

        # Calculate residual quantile for conformal calibration
        self.residual_quantile = np.percentile(residuals, (1 - self.alpha) * 100)

        return self

    def predict(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate conformal prediction intervals.

        Returns:
            (point_predictions, lower_bounds, upper_bounds)
        """
        # Aggregate predictions from all bootstrap models
        all_predictions = np.array([
            model.predict(n=n) for model in self.bootstrap_models
        ])

        # Point prediction = mean of ensemble
        point = np.mean(all_predictions, axis=0)

        # Conformal intervals
        lower = point - self.residual_quantile
        upper = point + self.residual_quantile

        return point, lower, upper

    def update(self, new_observation: float, prediction: float) -> None:
        """
        Adaptive update after observing true value.

        EnbPI can be updated online to maintain valid coverage.
        """
        new_residual = abs(new_observation - prediction)

        # Update residual quantile (exponential moving average)
        self.residual_quantile = 0.95 * self.residual_quantile + 0.05 * new_residual


def coverage_test(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float = 0.9,
) -> dict:
    """
    Test if prediction intervals achieve nominal coverage.

    Returns:
        Dictionary with coverage statistics
    """
    n = len(actual)
    covered = np.sum((actual >= lower) & (actual <= upper))
    empirical_coverage = covered / n

    # Binomial test for coverage
    from scipy.stats import binom
    p_value = 2 * min(
        binom.cdf(covered, n, nominal_coverage),
        1 - binom.cdf(covered - 1, n, nominal_coverage),
    )

    return {
        "nominal_coverage": nominal_coverage,
        "empirical_coverage": empirical_coverage,
        "coverage_gap": empirical_coverage - nominal_coverage,
        "is_calibrated": p_value > 0.05,  # Not significantly different from nominal
        "p_value": p_value,
        "average_width": np.mean(upper - lower),
    }
```

### Calibration Analysis

**Reliability Diagrams and Calibration Metrics**

```python
import numpy as np
import matplotlib.pyplot as plt

def create_reliability_diagram(
    actual: np.ndarray,
    predicted_quantiles: dict[float, np.ndarray],
    n_bins: int = 10,
) -> plt.Figure:
    """
    Create reliability diagram for probabilistic forecasts.

    A well-calibrated model has points on the diagonal.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for quantile, predictions in predicted_quantiles.items():
        # Calculate observed frequency below each quantile
        observed_freq = np.mean(actual < predictions)
        ax.scatter(quantile, observed_freq, s=100, label=f"q={quantile}")

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def probability_integral_transform_test(
    actual: np.ndarray,
    cdf_predictions: Callable[[np.ndarray], np.ndarray],
) -> dict:
    """
    PIT test for probabilistic calibration.

    For calibrated forecasts, CDF(actual) should be uniform.
    """
    # Calculate PIT values
    pit_values = cdf_predictions(actual)

    # Kolmogorov-Smirnov test against uniform
    from scipy.stats import kstest
    stat, p_value = kstest(pit_values, "uniform")

    # Visual test: should be uniform histogram
    return {
        "ks_statistic": stat,
        "p_value": p_value,
        "is_calibrated": p_value > 0.05,
        "pit_values": pit_values,
    }


def interval_score(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Interval Score - proper scoring rule for prediction intervals.

    Lower is better. Penalizes:
    - Wide intervals (uncertainty)
    - Missed observations (miscalibration)
    """
    width = upper - lower

    # Penalty for observations below lower bound
    below_penalty = (2 / alpha) * (lower - actual) * (actual < lower)

    # Penalty for observations above upper bound
    above_penalty = (2 / alpha) * (actual - upper) * (actual > upper)

    return np.mean(width + below_penalty + above_penalty)
```

### Proper Scoring Rules

**CRPS and Other Scoring Rules for Probabilistic Forecasts**

```python
import numpy as np
from scipy.stats import norm

def crps_gaussian(actual: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score (CRPS) for Gaussian predictions.

    CRPS is a proper scoring rule that generalizes MAE to probabilistic forecasts.
    Lower is better.
    """
    z = (actual - mean) / std
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return np.mean(crps)


def crps_ensemble(actual: np.ndarray, ensemble: np.ndarray) -> float:
    """
    CRPS for ensemble predictions.

    Args:
        actual: True values
        ensemble: Shape (n_samples, n_ensemble_members)
    """
    n_samples, n_members = ensemble.shape
    crps_values = []

    for i in range(n_samples):
        y = actual[i]
        x = np.sort(ensemble[i])

        # CRPS decomposition
        mae = np.mean(np.abs(x - y))
        spread = np.mean(np.abs(np.subtract.outer(x, x))) / (2 * n_members)
        crps_values.append(mae - spread)

    return np.mean(crps_values)


def log_score(actual: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """
    Logarithmic Score (negative log-likelihood).

    Proper scoring rule that heavily penalizes confident wrong predictions.
    Lower is better.
    """
    log_likelihood = norm.logpdf(actual, loc=mean, scale=std)
    return -np.mean(log_likelihood)


def brier_score(actual_direction: np.ndarray, predicted_prob: np.ndarray) -> float:
    """
    Brier Score for directional predictions.

    Args:
        actual_direction: 1 if price went up, 0 if down
        predicted_prob: Predicted probability of up

    Returns:
        Brier score (lower is better, range [0, 1])
    """
    return np.mean((predicted_prob - actual_direction) ** 2)


def scoring_summary(
    actual: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    lower_pred: np.ndarray,
    upper_pred: np.ndarray,
) -> dict:
    """Comprehensive probabilistic forecast evaluation."""
    return {
        "crps": crps_gaussian(actual, mean_pred, std_pred),
        "log_score": log_score(actual, mean_pred, std_pred),
        "interval_score_90": interval_score(actual, lower_pred, upper_pred, alpha=0.1),
        "coverage_90": np.mean((actual >= lower_pred) & (actual <= upper_pred)),
        "avg_interval_width": np.mean(upper_pred - lower_pred),
        "sharpness": np.mean(std_pred),  # Lower = sharper predictions
    }
```

### Regime-Specific Evaluation

**Evaluate Performance Across Market Conditions**

```python
import numpy as np
import pandas as pd
from typing import Literal

def calculate_volatility_regimes(
    returns: np.ndarray,
    window: int = 100,
    low_threshold: float = 0.33,
    high_threshold: float = 0.67,
) -> np.ndarray:
    """
    Classify market into volatility regimes.

    Returns:
        Array of regime labels: 'low_vol', 'normal', 'high_vol'
    """
    rolling_vol = pd.Series(returns).rolling(window).std()

    # Percentile thresholds
    low_cutoff = rolling_vol.quantile(low_threshold)
    high_cutoff = rolling_vol.quantile(high_threshold)

    regimes = np.where(
        rolling_vol < low_cutoff, "low_vol",
        np.where(rolling_vol > high_cutoff, "high_vol", "normal")
    )

    return regimes


def calculate_trend_regimes(
    prices: np.ndarray,
    window: int = 100,
    threshold: float = 0.02,
) -> np.ndarray:
    """
    Classify market into trend regimes.

    Returns:
        Array of regime labels: 'bull', 'bear', 'sideways'
    """
    returns = np.diff(prices) / prices[:-1]
    cumulative = pd.Series(returns).rolling(window).sum()

    regimes = np.where(
        cumulative > threshold, "bull",
        np.where(cumulative < -threshold, "bear", "sideways")
    )

    return np.concatenate([[regimes[0]], regimes])  # Pad first value


def regime_specific_evaluation(
    actual: np.ndarray,
    predicted: np.ndarray,
    regimes: np.ndarray,
    metric_fn: callable,
) -> pd.DataFrame:
    """
    Calculate metrics broken down by market regime.

    This reveals if model performance varies across market conditions.
    """
    results = []

    for regime in np.unique(regimes):
        mask = regimes == regime
        if np.sum(mask) < 10:  # Skip if too few samples
            continue

        regime_actual = actual[mask]
        regime_pred = predicted[mask]

        results.append({
            "regime": regime,
            "n_samples": np.sum(mask),
            "pct_samples": np.mean(mask) * 100,
            "metric": metric_fn(regime_actual, regime_pred),
        })

    df = pd.DataFrame(results)

    # Add overall
    df = pd.concat([df, pd.DataFrame([{
        "regime": "ALL",
        "n_samples": len(actual),
        "pct_samples": 100.0,
        "metric": metric_fn(actual, predicted),
    }])])

    return df


def generate_regime_report(
    actual: np.ndarray,
    predicted: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
) -> str:
    """Generate comprehensive regime-specific evaluation report."""
    from sklearn.metrics import mean_absolute_error

    # Calculate regimes
    vol_regimes = calculate_volatility_regimes(returns)
    trend_regimes = calculate_trend_regimes(prices)

    # Evaluate by volatility regime
    vol_results = regime_specific_evaluation(
        actual, predicted, vol_regimes, mean_absolute_error
    )

    # Evaluate by trend regime
    trend_results = regime_specific_evaluation(
        actual, predicted, trend_regimes, mean_absolute_error
    )

    report = f"""
Regime-Specific Evaluation Report
=================================

## By Volatility Regime
{vol_results.to_markdown(index=False)}

## By Trend Regime
{trend_results.to_markdown(index=False)}

## Key Insights
- Best regime: {vol_results.loc[vol_results['metric'].idxmin(), 'regime']}
- Worst regime: {vol_results.loc[vol_results['metric'].idxmax(), 'regime']}
- Performance variation: {vol_results['metric'].std() / vol_results['metric'].mean() * 100:.1f}% CV
"""
    return report
```

### Model Drift Detection

**Detect When Model Performance Degrades**

```python
import numpy as np
from dataclasses import dataclass
from typing import Literal
from scipy.stats import ks_2samp, mannwhitneyu

@dataclass
class DriftDetectionResult:
    """Result from drift detection."""
    is_drift_detected: bool
    drift_type: Literal["data", "concept", "performance"] | None
    p_value: float
    metric_change: float
    recommendation: str


class ModelDriftDetector:
    """
    Detect model drift in production.

    Types of drift:
    - Data drift: Input distribution changes
    - Concept drift: Relationship between input and output changes
    - Performance drift: Model metrics degrade
    """

    def __init__(
        self,
        reference_predictions: np.ndarray,
        reference_actuals: np.ndarray,
        reference_errors: np.ndarray,
        significance_level: float = 0.05,
    ):
        self.reference_predictions = reference_predictions
        self.reference_actuals = reference_actuals
        self.reference_errors = reference_errors
        self.significance_level = significance_level

        # Calculate reference statistics
        self.reference_mae = np.mean(np.abs(reference_errors))
        self.reference_error_std = np.std(reference_errors)

    def detect_performance_drift(
        self,
        recent_predictions: np.ndarray,
        recent_actuals: np.ndarray,
    ) -> DriftDetectionResult:
        """Detect if model performance has degraded."""
        recent_errors = recent_actuals - recent_predictions
        recent_mae = np.mean(np.abs(recent_errors))

        # Compare error distributions
        stat, p_value = ks_2samp(
            np.abs(self.reference_errors),
            np.abs(recent_errors)
        )

        metric_change = (recent_mae - self.reference_mae) / self.reference_mae

        is_drift = p_value < self.significance_level and metric_change > 0.1

        return DriftDetectionResult(
            is_drift_detected=is_drift,
            drift_type="performance" if is_drift else None,
            p_value=p_value,
            metric_change=metric_change,
            recommendation=self._get_recommendation(is_drift, metric_change),
        )

    def detect_data_drift(
        self,
        recent_inputs: np.ndarray,
        reference_inputs: np.ndarray,
    ) -> DriftDetectionResult:
        """Detect if input data distribution has changed."""
        # Population Stability Index
        psi = self._calculate_psi(reference_inputs, recent_inputs)

        # KS test
        stat, p_value = ks_2samp(reference_inputs.flatten(), recent_inputs.flatten())

        is_drift = psi > 0.2 or p_value < self.significance_level

        return DriftDetectionResult(
            is_drift_detected=is_drift,
            drift_type="data" if is_drift else None,
            p_value=p_value,
            metric_change=psi,
            recommendation="Retrain model with recent data" if is_drift else "No action needed",
        )

    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10,
    ) -> float:
        """
        Population Stability Index (PSI).

        PSI < 0.1: No significant shift
        0.1 <= PSI < 0.2: Moderate shift
        PSI >= 0.2: Significant shift
        """
        # Bin the expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # Add small value to avoid log(0)
        expected_pct = (expected_counts + 0.001) / len(expected)
        actual_pct = (actual_counts + 0.001) / len(actual)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi

    def _get_recommendation(self, is_drift: bool, metric_change: float) -> str:
        """Get actionable recommendation based on drift detection."""
        if not is_drift:
            return "Model performance stable. Continue monitoring."

        if metric_change > 0.5:
            return "URGENT: Severe performance degradation. Immediate retraining required."
        elif metric_change > 0.2:
            return "WARNING: Significant degradation. Schedule retraining within 24 hours."
        else:
            return "NOTICE: Mild degradation detected. Monitor closely, consider retraining."


def continuous_monitoring_report(
    detector: ModelDriftDetector,
    recent_data: dict,
    window_size: int = 100,
) -> str:
    """Generate continuous monitoring report."""
    perf_result = detector.detect_performance_drift(
        recent_data["predictions"][-window_size:],
        recent_data["actuals"][-window_size:],
    )

    return f"""
Model Monitoring Report
======================
Window Size: {window_size} observations

Performance Drift:
  Detected: {'YES' if perf_result.is_drift_detected else 'NO'}
  P-Value: {perf_result.p_value:.4f}
  Metric Change: {perf_result.metric_change:+.1%}

Recommendation: {perf_result.recommendation}
"""
```

### A/B Testing Framework

**Compare Models in Production**

```python
import numpy as np
from dataclasses import dataclass
from scipy.stats import ttest_ind, mannwhitneyu
import hashlib

@dataclass
class ABTestResult:
    """Result from A/B test."""
    model_a_metric: float
    model_b_metric: float
    difference: float
    relative_improvement: float
    p_value: float
    is_significant: bool
    recommended_model: str
    sample_size_a: int
    sample_size_b: int


class ModelABTest:
    """
    A/B testing framework for comparing models in production.

    Uses hash-based assignment for reproducible traffic splitting.
    """

    def __init__(
        self,
        model_a,
        model_b,
        traffic_split: float = 0.5,  # Fraction of traffic to model B
        min_samples: int = 100,
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.min_samples = min_samples

        self.results_a: list[float] = []
        self.results_b: list[float] = []

    def assign_model(self, request_id: str) -> Literal["A", "B"]:
        """
        Deterministically assign request to model.

        Uses hash for reproducible assignment.
        """
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return "B" if (hash_value % 100) < (self.traffic_split * 100) else "A"

    def predict(self, request_id: str, input_data) -> tuple[np.ndarray, str]:
        """Make prediction with assigned model."""
        model_name = self.assign_model(request_id)

        if model_name == "A":
            return self.model_a.predict(input_data), "A"
        else:
            return self.model_b.predict(input_data), "B"

    def record_outcome(self, model_name: str, error: float) -> None:
        """Record prediction error for analysis."""
        if model_name == "A":
            self.results_a.append(error)
        else:
            self.results_b.append(error)

    def analyze(self, significance_level: float = 0.05) -> ABTestResult:
        """Analyze A/B test results."""
        if len(self.results_a) < self.min_samples or len(self.results_b) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} samples per model")

        errors_a = np.array(self.results_a)
        errors_b = np.array(self.results_b)

        metric_a = np.mean(errors_a)
        metric_b = np.mean(errors_b)

        # Mann-Whitney U test (non-parametric)
        stat, p_value = mannwhitneyu(errors_a, errors_b, alternative="two-sided")

        difference = metric_b - metric_a
        relative_improvement = -difference / metric_a if metric_a != 0 else 0

        is_significant = p_value < significance_level
        recommended = "B" if (is_significant and metric_b < metric_a) else "A"

        return ABTestResult(
            model_a_metric=metric_a,
            model_b_metric=metric_b,
            difference=difference,
            relative_improvement=relative_improvement,
            p_value=p_value,
            is_significant=is_significant,
            recommended_model=recommended,
            sample_size_a=len(errors_a),
            sample_size_b=len(errors_b),
        )

    def required_sample_size(
        self,
        minimum_detectable_effect: float = 0.05,
        power: float = 0.8,
        significance_level: float = 0.05,
    ) -> int:
        """
        Calculate required sample size per group.

        Based on current variance estimate.
        """
        from scipy.stats import norm

        # Use pooled variance estimate if we have some data
        if len(self.results_a) > 10 and len(self.results_b) > 10:
            pooled_std = np.sqrt(
                (np.var(self.results_a) + np.var(self.results_b)) / 2
            )
        else:
            pooled_std = 0.1  # Default assumption

        effect_size = minimum_detectable_effect / pooled_std

        z_alpha = norm.ppf(1 - significance_level / 2)
        z_beta = norm.ppf(power)

        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))
```

### Multi-Horizon Evaluation

**Evaluate Predictions at Different Forecast Horizons**

```python
import numpy as np
import pandas as pd
from typing import Callable

def multi_horizon_evaluation(
    actual: np.ndarray,
    predictions: np.ndarray,  # Shape: (n_samples, n_horizons)
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Evaluate model performance at each forecast horizon.

    Model performance typically degrades with longer horizons.
    """
    n_samples, n_horizons = predictions.shape

    if horizons is None:
        horizons = list(range(1, n_horizons + 1))

    results = []
    for h in range(n_horizons):
        horizon = horizons[h]

        # Get actual values at horizon h
        actual_h = actual[h:n_samples]
        pred_h = predictions[:n_samples - h, h]

        if len(actual_h) < 10:
            continue

        results.append({
            "horizon": horizon,
            "n_samples": len(actual_h),
            "metric": metric_fn(actual_h, pred_h),
            "directional_accuracy": np.mean(
                np.sign(actual_h[1:] - actual_h[:-1]) ==
                np.sign(pred_h[1:] - actual_h[:-1])
            ),
        })

    return pd.DataFrame(results)


def horizon_decay_analysis(results_df: pd.DataFrame) -> dict:
    """
    Analyze how performance decays with forecast horizon.

    Useful for determining optimal forecast horizon.
    """
    from scipy.stats import linregress

    horizons = results_df["horizon"].values
    metrics = results_df["metric"].values

    # Linear fit
    slope, intercept, r_value, p_value, std_err = linregress(horizons, metrics)

    # Find horizon where metric exceeds threshold
    threshold = metrics[0] * 1.5  # 50% degradation
    decay_horizon = None
    for h, m in zip(horizons, metrics):
        if m > threshold:
            decay_horizon = h
            break

    return {
        "decay_slope": slope,
        "decay_r_squared": r_value ** 2,
        "best_horizon": horizons[np.argmin(metrics)],
        "worst_horizon": horizons[np.argmax(metrics)],
        "usable_horizon": decay_horizon,
        "recommendation": f"Use horizons up to {decay_horizon or horizons[-1]}",
    }
```

---

## Evaluation Checklist

Before deploying a model, verify:

- [ ] Walk-forward validation used (not random split)
- [ ] No lookahead bias in features or normalization
- [ ] Directional accuracy > 52% (better than random)
- [ ] Consistent performance across multiple folds
- [ ] Compared to baseline models (ARIMA, naive)
- [ ] Statistical significance tested
- [ ] Results visualized and inspected
- [ ] Edge cases tested (volatility spikes, gaps)
- [ ] Computational time acceptable for use case

---

*Last Updated: 2025-12-03*
