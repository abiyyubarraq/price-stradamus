# Module 07: Model Fundamentals

**Duration:** 3-4 hours | **Difficulty:** Intermediate | **Prerequisites:** Modules 01-06

## 🎯 Learning Objectives

After this module, you will:
- Understand the three model categories (Neural, Classical, ML)
- Know when to use which model type
- Understand the training loop step-by-step
- Demystify hyperparameters and their effects
- Learn model selection strategies
- Understand the BaseModel interface

---

## Model Categories Overview

Price Stradamus implements 8 different models across 3 categories:

### 1. Neural Network Models (Deep Learning)

**Best for:** Complex patterns, large datasets, non-linear relationships

| Model | Strengths | Best Use Case |
|-------|-----------|---------------|
| **N-BEATS** | Interpretable, no feature engineering needed | General time series forecasting |
| **LSTM** | Captures long-term dependencies | Sequential patterns with memory |
| **TCN** | Fast training, parallel processing | Real-time predictions |
| **TFT** | Attention mechanisms, multi-horizon | Complex multi-variate forecasting |

**TypeScript analogy:**
```typescript
// Neural networks are like deep nested functions
const predict = (input) => {
  let hidden1 = relu(weights1 * input + bias1);
  let hidden2 = relu(weights2 * hidden1 + bias2);
  let output = weights3 * hidden2 + bias3;
  return output;
};
```

**Python example:**
```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(10, 50)
        self.hidden2 = nn.Linear(50, 50)
        self.output = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        return self.output(x)
```

---

### 2. Classical Statistical Models

**Best for:** Small datasets, interpretability, understanding trends

| Model | Strengths | Best Use Case |
|-------|-----------|---------------|
| **ARIMA** | Statistical foundations, interpretable | Stationary time series |
| **Prophet** | Handles seasonality automatically | Data with trends and holidays |

**When to use:**
- Limited training data (<1000 points)
- Need explainable predictions
- Strong seasonal patterns
- Regulatory requirements for interpretability

**Python example:**
```python
from statsmodels.tsa.arima.model import ARIMA

# ARIMA is explicit about what it models
model = ARIMA(
    data,
    order=(p, d, q),  # AR order, differencing, MA order
)
results = model.fit()
```

---

### 3. Machine Learning Models

**Best for:** Feature-based predictions, gradient boosting power

| Model | Strengths | Best Use Case |
|-------|-----------|---------------|
| **XGBoost** | Fast, handles missing data, feature importance | Structured data with many features |

**Node.js ML analogy:**
```typescript
// ML models are like decision trees
if (rsi > 70) {
  if (volume > avgVolume) {
    return "sell";  // Overbought with high volume
  }
}
```

**Python example:**
```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## The BaseModel Interface

All Price Stradamus models implement the same interface:

```python
# src/price_stradamus/models/base.py
from abc import ABC, abstractmethod
from darts import TimeSeries
from pathlib import Path

class BaseModel(ABC):
    """Base class for all forecasting models."""

    def __init__(self, name: str):
        self.name = name
        self._is_fitted = False

    @abstractmethod
    def fit(
        self,
        train_series: TimeSeries,
        val_series: TimeSeries | None = None,
    ) -> None:
        """Train the model.

        Args:
            train_series: Training data
            val_series: Validation data (optional)
        """
        pass

    @abstractmethod
    def predict(
        self,
        n: int,
        series: TimeSeries,
    ) -> TimeSeries:
        """Make predictions.

        Args:
            n: Number of steps to predict
            series: Historical data to predict from

        Returns:
            Predicted values as TimeSeries
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        pass

    @property
    def is_fitted(self) -> bool:
        """Check if model is trained."""
        return self._is_fitted
```

**Why this interface?**
1. **Consistency**: All models work the same way
2. **Swappable**: Easy to test different models
3. **Extensibility**: Add new models easily
4. **Type Safety**: IDEs can help with autocomplete

**TypeScript comparison:**
```typescript
// Similar to TypeScript interfaces
interface ForecastModel {
  fit(trainData: TimeSeries, valData?: TimeSeries): void;
  predict(steps: number, history: TimeSeries): TimeSeries;
  save(path: string): void;
  load(path: string): void;
  isFitted: boolean;
}

class NBEATSModel implements ForecastModel {
  // Must implement all interface methods
}
```

---

## Understanding the Training Loop

Let's break down what happens when you train a model:

### High-Level Flow

```
┌───────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                      │
│                                                           │
│              ┌──────────────────┐                         │
│              │   Load Data      │                         │
│              └────────┬─────────┘                         │
│                       │                                   │
│                       ▼                                   │
│              ┌──────────────────┐                         │
│              │   Preprocess     │                         │
│              └────────┬─────────┘                         │
│                       │                                   │
│                       ▼                                   │
│              ┌──────────────────┐                         │
│              │ Create Features  │                         │
│              └────────┬─────────┘                         │
│                       │                                   │
│                       ▼                                   │
│              ┌──────────────────┐                         │
│              │Split Train/Val/  │                         │
│              │      Test        │                         │
│              └────────┬─────────┘                         │
│                       │                                   │
│                       ▼                                   │
│              ┌──────────────────┐                         │
│              │Initialize Model  │                         │
│              └────────┬─────────┘                         │
│                       │                                   │
│                       ▼                                   │
│              ┌──────────────────┐                         │
│            ┌─┤  Training Loop   │◀──┐                     │
│            │ └──────────────────┘   │                     │
│            │          │             │                     │
│            │          ▼             │                     │
│            │    ┌──────────┐       │                     │
│            │    │Converged?│       │                     │
│            │    └────┬───┬─┘       │                     │
│            │         │   │         │                     │
│            │     No  │   │ Yes     │                     │
│            └─────────┘   │         │                     │
│                          │         │                     │
│                          ▼         │                     │
│              ┌──────────────────┐  │                     │
│              │ Evaluate on Val  │  │                     │
│              └────────┬─────────┘  │                     │
│                       │            │                     │
│                       ▼            │                     │
│              ┌──────────────────┐  │                     │
│              │ Save Best Model  │  │                     │
│              └──────────────────┘  │                     │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### Detailed Training Loop (Neural Networks)

```python
def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
):
    """Training loop explained step-by-step."""

    # 1. Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    # 2. Training loop
    for epoch in range(epochs):
        # 2a. Training phase
        model.train()  # Set to training mode
        train_losses = []

        for batch_x, batch_y in train_loader:
            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            # Backward pass
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()        # Compute new gradients
            optimizer.step()       # Update weights

            train_losses.append(loss.item())

        # 2b. Validation phase
        model.eval()  # Set to evaluation mode
        val_losses = []

        with torch.no_grad():  # Don't compute gradients
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_losses.append(loss.item())

        # 2c. Track progress
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # 2d. Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
```

**Key concepts explained:**

1. **Forward pass**: Input → Model → Predictions
2. **Loss calculation**: How wrong are we?
3. **Backward pass**: Calculate gradients (how to improve)
4. **Weight update**: Adjust model parameters
5. **Validation**: Check performance on unseen data

**TypeScript analogy:**
```typescript
// Training is like optimizing a function
let weights = initializeRandom();

for (let epoch = 0; epoch < 100; epoch++) {
  // Try prediction
  const predictions = model(data, weights);

  // Measure error
  const error = calculateError(predictions, actual);

  // Adjust weights to reduce error
  weights = updateWeights(weights, error);
}
```

---

## Hyperparameters Demystified

Hyperparameters are settings you choose **before** training. Think of them as configuration options.

### Common Hyperparameters

#### 1. Learning Rate

**What it does:** Controls how big steps the model takes during learning

```python
# Too high: Model jumps around, never converges
optimizer = Adam(model.parameters(), lr=0.1)  # ❌ Too high

# Too low: Training takes forever
optimizer = Adam(model.parameters(), lr=0.00001)  # ❌ Too low

# Just right: Steady improvement
optimizer = Adam(model.parameters(), lr=0.001)  # ✅ Good start
```

**Analogy:** Like GPS recalculating your route
- High learning rate = takes highway exits without checking
- Low learning rate = stops at every intersection
- Good learning rate = smooth navigation

**How to tune:**
1. Start with 0.001
2. If loss decreases slowly: increase to 0.01
3. If loss jumps around: decrease to 0.0001

---

#### 2. Batch Size

**What it does:** How many examples to process before updating weights

```python
# Small batch: More updates, noisier gradients
train_loader = DataLoader(dataset, batch_size=16)

# Large batch: Fewer updates, smoother gradients
train_loader = DataLoader(dataset, batch_size=128)
```

**Trade-offs:**
- **Small (16-32)**: Better generalization, more memory efficient, slower
- **Large (128-256)**: Faster training, more stable, needs more GPU memory

**TypeScript analogy:**
```typescript
// Like batching API requests
// Small batch
for (const item of items) {
  await api.post(item);  // One at a time
}

// Large batch
await api.postBatch(items);  // All at once
```

---

#### 3. Number of Epochs

**What it does:** How many times to see the entire dataset

```python
# Too few: Underfitting (didn't learn enough)
model.fit(data, epochs=5)

# Too many: Overfitting (memorized training data)
model.fit(data, epochs=1000)

# Monitor validation loss to find the right number
model.fit(data, epochs=100, early_stopping=True)
```

**Visual guide:**
```
Loss
 │
 │  Training Loss (keeps decreasing)
 │  ╲╲
 │    ╲╲
 │      ╲╲╲
 │         ╲╲___________
 │
 │  Validation Loss (starts increasing = overfitting!)
 │  ╲╲
 │    ╲╲
 │      ╲_____╱╱╱╱
 │              ↑
 │         Stop here!
 └────────────────────────> Epochs
```

---

#### 4. Model Architecture Parameters

**Input Chunk Length**: How much history to use

```python
# For 1-minute candles
model = NBEATSModel(
    input_chunk_length=60,   # Use last 60 minutes (1 hour)
    output_chunk_length=5,   # Predict next 5 minutes
)

# More history = captures longer patterns but slower
model = NBEATSModel(
    input_chunk_length=1440,  # Last 24 hours (1440 minutes)
    output_chunk_length=5,
)
```

**How to choose:**
- Start with 60 (1 hour for 1m candles)
- If patterns are longer term: increase to 240 (4 hours) or 1440 (1 day)
- If real-time speed matters: decrease to 30

---

**Hidden Units/Layers**: Model capacity

```python
# Simple model (might underfit)
model = LSTMModel(
    hidden_dim=32,
    n_layers=1,
)

# Complex model (might overfit)
model = LSTMModel(
    hidden_dim=256,
    n_layers=4,
)

# Balanced
model = LSTMModel(
    hidden_dim=128,
    n_layers=2,
)
```

**Rule of thumb:**
- Start with 2 layers, 128 hidden units
- If underfitting: increase to 256 units or 3 layers
- If overfitting: decrease to 64 units or 1 layer

---

### Hyperparameter Tuning Strategy

```python
# 1. Start with defaults
config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "input_chunk_length": 60,
    "hidden_dim": 128,
    "n_layers": 2,
}

# 2. Train baseline model
baseline_model = train_model(config)
baseline_mae = evaluate(baseline_model)

# 3. Tune one parameter at a time
for lr in [0.0001, 0.001, 0.01]:
    config["learning_rate"] = lr
    model = train_model(config)
    mae = evaluate(model)
    print(f"LR {lr}: MAE = {mae}")

# 4. Keep best value, tune next parameter
```

---

## Model Selection Strategies

### Strategy 1: Start Simple

```python
# 1. Baseline: Simple moving average
baseline_mae = evaluate_moving_average(data)

# 2. Classical: ARIMA
arima_mae = evaluate_arima(data)

# 3. ML: XGBoost with features
xgb_mae = evaluate_xgboost(data)

# 4. Neural: N-BEATS
nbeats_mae = evaluate_nbeats(data)

# Compare
print(f"Baseline: {baseline_mae:.2f}")
print(f"ARIMA: {arima_mae:.2f}")
print(f"XGBoost: {xgb_mae:.2f}")
print(f"N-BEATS: {nbeats_mae:.2f}")
```

### Strategy 2: Match Problem to Model

| Scenario | Recommended Model | Why |
|----------|-------------------|-----|
| <1000 data points | ARIMA, Prophet | Not enough data for neural networks |
| Clear seasonality | Prophet | Designed for seasonal patterns |
| Many engineered features | XGBoost | Leverages feature engineering |
| Raw price data only | N-BEATS, LSTM | Can learn features automatically |
| Need interpretability | ARIMA | Can explain decisions |
| Large dataset (>10k) | N-BEATS, TFT, TCN | Neural networks shine with data |
| Multi-step predictions | N-BEATS, TFT | Designed for multi-horizon |
| Real-time inference | TCN, XGBoost | Fast prediction time |

### Strategy 3: Ensemble Multiple Models

```python
# Combine predictions from multiple models
models = {
    "nbeats": NBEATSModel(),
    "lstm": LSTMModel(),
    "xgboost": XGBoostModel(),
}

# Train all models
for name, model in models.items():
    model.fit(train_data)

# Make predictions
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(steps=5)

# Ensemble: Average predictions
ensemble_prediction = np.mean([
    predictions["nbeats"],
    predictions["lstm"],
    predictions["xgboost"],
], axis=0)
```

**When ensembles help:**
- Reduces variance (more stable predictions)
- Combines different model strengths
- Better than any single model (usually)

---

## Quick Reference

### Choosing Your First Model

```python
# Quick decision tree
if dataset_size < 1000:
    model = ProphetModel()  # Works well with small data
elif need_interpretability:
    model = ARIMAModel()  # Statistical, explainable
elif have_many_features:
    model = XGBoostModel()  # Great with engineered features
else:
    model = NBEATSModel()  # Best all-around neural network
```

### Common Hyperparameter Ranges

```python
HYPERPARAMETER_RANGES = {
    "learning_rate": [0.0001, 0.001, 0.01],
    "batch_size": [16, 32, 64, 128],
    "input_chunk_length": [30, 60, 120, 240],
    "hidden_dim": [64, 128, 256],
    "n_layers": [1, 2, 3],
}
```

---

## Practice Exercises

### Exercise 1: Implement Training Loop (30 minutes)

Write a simple training loop for a linear model:

```python
import torch
import torch.nn as nn

# Simple linear model
model = nn.Linear(10, 1)

# Your task: Implement training loop
def train(model, X_train, y_train, epochs=100, lr=0.001):
    # TODO: Implement
    pass

# Test it
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)
train(model, X_train, y_train)
```

<details>
<summary>Solution</summary>

```python
def train(model, X_train, y_train, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        # Forward pass
        predictions = model(X_train)
        loss = criterion(predictions, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```
</details>

---

### Exercise 2: Hyperparameter Comparison (20 minutes)

Compare different learning rates:

```python
learning_rates = [0.0001, 0.001, 0.01, 0.1]

for lr in learning_rates:
    # Train model with this learning rate
    # Record final loss
    # Which learning rate works best?
    pass
```

---

## Next Steps

**Next Module:** [08: Neural Networks Deep Dive →](08-neural-networks-deep-dive.md)

Learn about N-BEATS, LSTM, TCN, and TFT in detail.

---

## Summary Checklist

- [ ] I understand the three model categories (Neural, Classical, ML)
- [ ] I know when to use which model type
- [ ] I understand the training loop components
- [ ] I can explain common hyperparameters and their effects
- [ ] I know strategies for model selection
- [ ] I understand the BaseModel interface

**Continue to:** [Module 08: Neural Networks Deep Dive](08-neural-networks-deep-dive.md)
