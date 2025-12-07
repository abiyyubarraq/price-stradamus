# Module 08: Neural Networks Deep Dive

**Duration:** 4-5 hours | **Difficulty:** Advanced | **Prerequisites:** Module 07

## 🎯 Learning Objectives

After this module, you will:
- Understand N-BEATS architecture and why it's special
- Know how LSTM captures long-term dependencies
- Understand TCN's convolutional approach
- Learn about TFT and attention mechanisms
- Master PyTorch basics for model development
- Know when to use each neural network type

---

## PyTorch Basics

Before diving into specific models, let's understand PyTorch fundamentals.

### Tensors: The Building Blocks

**Tensors are like multi-dimensional arrays** (similar to numpy arrays but GPU-compatible):

```python
import torch
import numpy as np

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.zeros(3, 4)  # 3x4 tensor of zeros
z = torch.randn(2, 3, 4)  # Random 2x3x4 tensor

# Convert from numpy
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)

# Move to GPU (if available)
if torch.cuda.is_available():
    x = x.cuda()  # or x.to('cuda')
```

**TypeScript analogy:**
```typescript
// Tensors are like typed multi-dimensional arrays
const vector: number[] = [1, 2, 3];  // 1D tensor
const matrix: number[][] = [[1, 2], [3, 4]];  // 2D tensor
const cube: number[][][] = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];  // 3D tensor
```

### Neural Network Layers

```python
import torch.nn as nn

# Linear layer: y = Wx + b
linear = nn.Linear(in_features=10, out_features=5)

# Input: batch_size x 10
x = torch.randn(32, 10)
# Output: batch_size x 5
y = linear(x)

# Common layers
conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
lstm = nn.LSTM(input_size=10, hidden_size=50, num_layers=2)
dropout = nn.Dropout(p=0.2)  # Randomly drops 20% of neurons
```

### Building a Simple Network

```python
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.fc1(x)  # First linear transformation
        x = self.relu(x)  # Non-linear activation
        x = self.fc2(x)  # Second linear transformation
        return x

# Usage
model = SimpleNet(input_size=10, hidden_size=50, output_size=1)
x = torch.randn(32, 10)  # Batch of 32 samples
predictions = model(x)  # Shape: (32, 1)
```

---

## 1. N-BEATS: Neural Basis Expansion Analysis

### Why N-BEATS is Special

N-BEATS won the M4 forecasting competition and has unique properties:
- **No feature engineering needed** - works with raw time series
- **Interpretable** - can decompose into trend and seasonality
- **State-of-the-art** - beats classical methods and other neural networks

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     N-BEATS ARCHITECTURE                     │
│                                                              │
│                 ┌───────────────────────┐                   │
│                 │  Input Time Series    │                   │
│                 └──────────┬────────────┘                   │
│                            │                                 │
│               ┌────────────┴────────────┐                   │
│               │                         │                   │
│               ▼                         ▼                   │
│     ┌──────────────────┐      ┌──────────────────┐         │
│     │  Stack 1: Trend  │      │ Stack 2: Season  │         │
│     └────────┬─────────┘      └────────┬─────────┘         │
│              │                         │                    │
│              ▼                         ▼                    │
│     ┌──────────────────┐      ┌──────────────────┐         │
│     │ Trend Forecast   │      │ Seasonal Forecast│         │
│     └────────┬─────────┘      └────────┬─────────┘         │
│              │                         │                    │
│              └────────────┬────────────┘                    │
│                           │                                 │
│                           ▼                                 │
│                 ┌──────────────────┐                        │
│                 │ Final Forecast   │                        │
│                 └──────────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### How N-BEATS Works

Think of N-BEATS as having **multiple "blocks"** that each make a prediction:

```python
# Simplified N-BEATS concept
def nbeats_forward(input_series):
    """
    N-BEATS uses blocks that:
    1. Make a prediction (forecast)
    2. Remove that prediction from input (backcast)
    3. Pass residual to next block
    """
    residual = input_series

    forecasts = []
    for block in blocks:
        # Block makes prediction
        backcast, forecast = block(residual)

        # Store forecast
        forecasts.append(forecast)

        # Remove backcast from residual
        residual = residual - backcast

    # Combine all forecasts
    final_forecast = sum(forecasts)
    return final_forecast
```

### N-BEATS in Price Stradamus

```python
# src/price_stradamus/models/neural/nbeats.py
from darts.models import NBEATSModel as DartsNBEATS

class NBEATSModel(BaseModel):
    def __init__(
        self,
        input_chunk_length: int = 60,
        output_chunk_length: int = 5,
        num_stacks: int = 30,
        num_blocks: int = 1,
        num_layers: int = 4,
        layer_widths: int = 256,
    ):
        """Initialize N-BEATS model.

        Args:
            input_chunk_length: How many past values to use (60 = 1 hour for 1m)
            output_chunk_length: How many future values to predict
            num_stacks: Number of stacks (more = more capacity)
            num_blocks: Blocks per stack
            num_layers: Layers per block
            layer_widths: Neurons per layer
        """
        super().__init__(name="nbeats")

        self.model = DartsNBEATS(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            n_epochs=100,
            batch_size=32,
            optimizer_kwargs={"lr": 0.001},
        )

    def fit(self, train_series: TimeSeries, val_series: TimeSeries | None = None):
        """Train N-BEATS model."""
        self.model.fit(
            series=train_series,
            val_series=val_series,
            verbose=True,
        )
        self._is_fitted = True

    def predict(self, n: int, series: TimeSeries) -> TimeSeries:
        """Make predictions."""
        return self.model.predict(n=n, series=series)
```

### When to Use N-BEATS

✅ **Use N-BEATS when:**
- You have >1000 data points
- You want minimal feature engineering
- You need multi-step predictions
- You care about interpretability (can see trend vs seasonality)

❌ **Don't use N-BEATS when:**
- Limited data (<500 points)
- Need real-time predictions (slower than TCN)
- Have strong domain features (use XGBoost instead)

### Hyperparameters for N-BEATS

```python
# Conservative (faster, less capacity)
model = NBEATSModel(
    input_chunk_length=30,
    num_stacks=10,
    layer_widths=128,
)

# Balanced (recommended)
model = NBEATSModel(
    input_chunk_length=60,
    num_stacks=30,
    layer_widths=256,
)

# Aggressive (slower, more capacity)
model = NBEATSModel(
    input_chunk_length=120,
    num_stacks=50,
    layer_widths=512,
)
```

---

## 2. LSTM: Long Short-Term Memory

### The Memory Problem

Regular neural networks have **short memory**:

```python
# Standard RNN forgets quickly
#  t=1     t=2     t=3     t=4
# "Buy" → "the" → "dip" → "?"
#         ↑ already forgot "Buy"
```

LSTM solves this with **memory cells** that can remember long-term patterns.

### LSTM Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      LSTM CELL STRUCTURE                        │
│                                                                 │
│                    ┌─────────────┐                             │
│                    │ Input at t  │                             │
│                    └──────┬──────┘                             │
│                           │                                     │
│          ┌────────────────┼────────────────┐                   │
│          │                │                │                   │
│          ▼                ▼                ▼                   │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│    │  Forget  │    │  Input   │    │  Output  │              │
│    │   Gate   │    │   Gate   │    │   Gate   │              │
│    │  (What   │    │  (What   │    │  (What   │              │
│    │   to     │    │   to     │    │   to     │              │
│    │ forget?) │    │remember?)│    │ output?) │              │
│    └────┬─────┘    └─────┬────┘    └────┬─────┘              │
│         │                │              │                      │
│         └────────┐       │              │                      │
│                  ▼       ▼              │                      │
│              ┌──────────────────┐       │                      │
│              │   Cell State     │       │                      │
│              │ (Long-term       │───────┘                      │
│              │  Memory)         │                              │
│              └─────────┬────────┘                              │
│                        │                                        │
│                        ▼                                        │
│                  ┌──────────┐                                  │
│                  │ Output   │                                  │
│                  │  at t    │                                  │
│                  └──────────┘                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### How LSTM Works (Simplified)

```python
class SimpleLSTM:
    """Simplified LSTM to understand the concept."""

    def forward(self, x, prev_hidden, prev_cell):
        """
        LSTM has 3 gates:
        1. Forget gate: What to forget from memory?
        2. Input gate: What new info to remember?
        3. Output gate: What to output?
        """

        # 1. Forget gate: Decide what to forget
        forget_gate = sigmoid(W_forget @ [prev_hidden, x])
        # forget_gate = [0.1, 0.9, 0.3, ...]  # Values close to 0 = forget

        # 2. Input gate: Decide what to remember
        input_gate = sigmoid(W_input @ [prev_hidden, x])
        candidate_memory = tanh(W_candidate @ [prev_hidden, x])
        # input_gate = [0.8, 0.2, 0.9, ...]  # Values close to 1 = remember

        # 3. Update memory
        cell = forget_gate * prev_cell + input_gate * candidate_memory

        # 4. Output gate: Decide what to output
        output_gate = sigmoid(W_output @ [prev_hidden, x])
        hidden = output_gate * tanh(cell)

        return hidden, cell
```

**Analogy:** LSTM is like taking notes
- **Forget gate**: Cross out old notes that aren't useful anymore
- **Input gate**: Write down new important information
- **Output gate**: Decide what to say based on your notes

### LSTM in Price Stradamus

```python
# src/price_stradamus/models/neural/lstm.py
from darts.models import RNNModel

class LSTMModel(BaseModel):
    def __init__(
        self,
        input_chunk_length: int = 60,
        output_chunk_length: int = 5,
        hidden_dim: int = 128,
        n_rnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize LSTM model.

        Args:
            hidden_dim: Size of hidden state (memory capacity)
            n_rnn_layers: Number of LSTM layers stacked
            dropout: Dropout rate (regularization)
        """
        super().__init__(name="lstm")

        self.model = RNNModel(
            model="LSTM",  # Can also be "RNN" or "GRU"
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            dropout=dropout,
            n_epochs=100,
            batch_size=32,
            optimizer_kwargs={"lr": 0.001},
        )
```

### When to Use LSTM

✅ **Use LSTM when:**
- Sequential patterns are important (e.g., "buy" followed by "the" followed by "dip")
- Need to remember long-term context
- Data has temporal dependencies
- You have time-ordered features

❌ **Don't use LSTM when:**
- Training is too slow (use TCN instead)
- No clear sequential patterns
- Limited data (<500 points)

### LSTM Hyperparameters

```python
# Small model (faster, less memory)
model = LSTMModel(
    hidden_dim=64,
    n_rnn_layers=1,
    dropout=0.1,
)

# Medium model (balanced)
model = LSTMModel(
    hidden_dim=128,
    n_rnn_layers=2,
    dropout=0.2,
)

# Large model (more capacity)
model = LSTMModel(
    hidden_dim=256,
    n_rnn_layers=3,
    dropout=0.3,
)
```

---

## 3. TCN: Temporal Convolutional Network

### Why TCN is Fast

Traditional RNNs (like LSTM) must process sequentially:
```
t1 → t2 → t3 → t4 → t5  (must wait for each step)
```

TCN processes in parallel using **convolutional filters**:
```
t1 ──┐
t2 ──┼─→ conv filter → output
t3 ──┘
```

### How Convolutions Work

```python
# 1D Convolution for time series
# Input: [10, 12, 15, 18, 20, 22, 25]
# Filter (kernel): [0.2, 0.5, 0.3]  (size 3)

# Slide filter across input:
# [10, 12, 15] * [0.2, 0.5, 0.3] = 10*0.2 + 12*0.5 + 15*0.3 = 12.5
# [12, 15, 18] * [0.2, 0.5, 0.3] = 12*0.2 + 15*0.5 + 18*0.3 = 15.3
# [15, 18, 20] * [0.2, 0.5, 0.3] = ...

# Output: [12.5, 15.3, 18.1, 20.9, 23.7]
```

### Dilated Convolutions (TCN Secret Sauce)

Regular convolution sees 3 consecutive points:
```
Kernel size 3:
[10, 12, 15]  ← sees 3 points
```

Dilated convolution sees 3 points spread out:
```
Dilation 1: [10, 12, 15]        ← consecutive
Dilation 2: [10, __, 15, __, 20]  ← skip 1
Dilation 4: [10, __, __, __, 20, __, __, __, 30]  ← skip 3
```

This lets TCN **see long-term patterns without many layers**.

### TCN Architecture

```
┌────────────────────────────────────────────────────────┐
│                 TCN ARCHITECTURE                        │
│                                                         │
│                  ┌────────────┐                        │
│                  │   Input    │                        │
│                  └──────┬─────┘                        │
│                         │                               │
│                         ▼                               │
│              ┌──────────────────┐                      │
│              │ Conv Block 1     │                      │
│              │ (dilation = 1)   │                      │
│              └─────────┬────────┘                      │
│                        │                                │
│                        ▼                                │
│              ┌──────────────────┐                      │
│              │ Conv Block 2     │                      │
│              │ (dilation = 2)   │                      │
│              └─────────┬────────┘                      │
│                        │                                │
│                        ▼                                │
│              ┌──────────────────┐                      │
│              │ Conv Block 3     │                      │
│              │ (dilation = 4)   │                      │
│              └─────────┬────────┘                      │
│                        │                                │
│                        ▼                                │
│              ┌──────────────────┐                      │
│              │ Conv Block 4     │                      │
│              │ (dilation = 8)   │                      │
│              └─────────┬────────┘                      │
│                        │                                │
│                        ▼                                │
│                  ┌────────────┐                        │
│                  │   Output   │                        │
│                  └────────────┘                        │
│                                                         │
│  (Dilations increase to capture longer patterns)       │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### TCN in Price Stradamus

```python
# src/price_stradamus/models/neural/tcn.py
from darts.models import TCNModel as DartsTCN

class TCNModel(BaseModel):
    def __init__(
        self,
        input_chunk_length: int = 60,
        output_chunk_length: int = 5,
        kernel_size: int = 3,
        num_filters: int = 32,
        num_layers: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize TCN model.

        Args:
            kernel_size: Convolution kernel size (usually 3 or 5)
            num_filters: Number of filters per layer
            num_layers: Number of convolutional layers
            dilation_base: Dilation growth rate (2 = doubles each layer)
            dropout: Dropout rate
        """
        super().__init__(name="tcn")

        self.model = DartsTCN(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            kernel_size=kernel_size,
            num_filters=num_filters,
            num_layers=num_layers,
            dilation_base=dilation_base,
            dropout=dropout,
            n_epochs=100,
            batch_size=32,
            optimizer_kwargs={"lr": 0.001},
        )
```

### When to Use TCN

✅ **Use TCN when:**
- Need fast training and inference
- Real-time predictions are important
- Parallel processing is available (GPU)
- Want something faster than LSTM but still powerful

❌ **Don't use TCN when:**
- Limited data (<500 points)
- LSTM already works well (no need to change)

### TCN Hyperparameters

```python
# Shallow (fast, less capacity)
model = TCNModel(
    num_layers=2,
    num_filters=16,
    kernel_size=3,
)

# Medium (balanced)
model = TCNModel(
    num_layers=3,
    num_filters=32,
    kernel_size=3,
)

# Deep (slower, more capacity)
model = TCNModel(
    num_layers=5,
    num_filters=64,
    kernel_size=5,
)
```

---

## 4. TFT: Temporal Fusion Transformer

### The Power of Attention

Transformers use **attention mechanisms** to focus on important parts of the input:

```python
# Example: Predicting next price
# Input: [p1, p2, p3, p4, p5]

# Attention learns what to focus on:
# "To predict next price, p5 is most important (80%),
#  p4 is somewhat important (15%),
#  p1-p3 barely matter (5%)"

attention_weights = [0.01, 0.01, 0.03, 0.15, 0.80]
prediction = sum(input * attention_weights)
```

### TFT Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TFT ARCHITECTURE                              │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Static    │  │ Historical  │  │    Known    │            │
│  │  Features   │  │    Data     │  │   Future    │            │
│  │ (e.g.,      │  │  (price,    │  │    (time    │            │
│  │  symbol)    │  │  volume)    │  │  features)  │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                       │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │  Variable Selection   │                          │
│              │ (Pick important ones) │                          │
│              └──────────┬────────────┘                          │
│                         │                                        │
│              ┌──────────┴──────────┐                            │
│              │                     │                            │
│              ▼                     ▼                            │
│     ┌─────────────────┐   ┌─────────────────┐                 │
│     │  LSTM Encoder   │   │  LSTM Decoder   │                 │
│     │ (Past context)  │   │(Future context) │                 │
│     └────────┬────────┘   └────────┬────────┘                 │
│              │                     │                            │
│              └──────────┬──────────┘                            │
│                         │                                        │
│                         ▼                                        │
│              ┌───────────────────────┐                          │
│              │ Multi-Head Attention  │                          │
│              │ (What's important?)   │                          │
│              └──────────┬────────────┘                          │
│                         │                                        │
│                         ▼                                        │
│              ┌───────────────────────┐                          │
│              │    Gating Layer       │                          │
│              └──────────┬────────────┘                          │
│                         │                                        │
│                         ▼                                        │
│                 ┌───────────────┐                               │
│                 │ Final Forecast│                               │
│                 └───────────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key TFT Features

1. **Variable Selection**: Automatically picks important features
2. **Attention**: Focuses on relevant time steps
3. **Multi-horizon**: Predicts multiple steps efficiently
4. **Interpretable**: Can see which features and times matter

### TFT in Price Stradamus

```python
# src/price_stradamus/models/neural/tft.py
from darts.models import TFTModel as DartsTFT

class TFTModel(BaseModel):
    def __init__(
        self,
        input_chunk_length: int = 60,
        output_chunk_length: int = 5,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize TFT model.

        Args:
            hidden_size: Size of hidden layers
            lstm_layers: Number of LSTM layers
            num_attention_heads: Number of attention heads (parallel attention)
            dropout: Dropout rate
        """
        super().__init__(name="tft")

        self.model = DartsTFT(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            n_epochs=100,
            batch_size=32,
            optimizer_kwargs={"lr": 0.001},
        )
```

### When to Use TFT

✅ **Use TFT when:**
- Have multiple input features (price, volume, indicators)
- Need multi-step forecasting
- Want to understand feature importance
- Have enough data (>2000 points)

❌ **Don't use TFT when:**
- Limited data (<1000 points)
- Only have raw price data (N-BEATS is simpler)
- Training time is a constraint (slowest model)

### TFT Hyperparameters

```python
# Small (faster)
model = TFTModel(
    hidden_size=64,
    lstm_layers=1,
    num_attention_heads=2,
)

# Medium (recommended)
model = TFTModel(
    hidden_size=128,
    lstm_layers=2,
    num_attention_heads=4,
)

# Large (best performance)
model = TFTModel(
    hidden_size=256,
    lstm_layers=3,
    num_attention_heads=8,
)
```

---

## Model Comparison

### Quick Reference Table

| Model | Speed | Data Needed | Best For | Complexity |
|-------|-------|-------------|----------|------------|
| **N-BEATS** | Medium | >1000 | Raw time series, interpretability | Medium |
| **LSTM** | Slow | >500 | Sequential patterns, context | Medium |
| **TCN** | Fast | >500 | Real-time, parallel processing | Low |
| **TFT** | Very Slow | >2000 | Multi-variate, feature importance | High |

### Performance Tips

```python
# 1. Start with N-BEATS (best all-around)
nbeats = NBEATSModel(input_chunk_length=60)
nbeats.fit(train_data)

# 2. If too slow → try TCN
tcn = TCNModel(input_chunk_length=60)
tcn.fit(train_data)

# 3. If have many features → try TFT
tft = TFTModel(input_chunk_length=60)
tft.fit(train_data_with_features)

# 4. Compare all on validation set
results = {
    "nbeats": evaluate(nbeats, val_data),
    "tcn": evaluate(tcn, val_data),
    "tft": evaluate(tft, val_data),
}
```

---

## Advanced PyTorch Techniques

### Custom Loss Functions

```python
class DirectionalLoss(nn.Module):
    """Custom loss that penalizes wrong direction."""

    def forward(self, predictions, targets):
        # Standard MSE
        mse = torch.mean((predictions - targets) ** 2)

        # Directional penalty
        pred_direction = torch.sign(predictions[1:] - predictions[:-1])
        true_direction = torch.sign(targets[1:] - targets[:-1])
        direction_penalty = torch.mean((pred_direction - true_direction) ** 2)

        # Combine
        return mse + 0.5 * direction_penalty
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Reduce learning rate when validation loss plateaus
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,  # Multiply lr by 0.5
    patience=10,  # After 10 epochs without improvement
)

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    # Update learning rate
    scheduler.step(val_loss)
```

### Early Stopping

```python
class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            return self.counter >= self.patience

# Usage
early_stopping = EarlyStopping(patience=10)

for epoch in range(epochs):
    val_loss = validate(model, val_loader)
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## Practice Exercises

### Exercise 1: Train N-BEATS (30 minutes)

Train an N-BEATS model on Bitcoin data:

```python
from price_stradamus.models.neural.nbeats import NBEATSModel
from price_stradamus.data.database import DatabaseManager

async def train_nbeats():
    # Load data
    db = DatabaseManager()
    await db.initialize()
    df = await db.get_ohlcv("BTCUSDT", "1m", limit=2000)

    # TODO: Create TimeSeries, split train/val, train model
    pass
```

### Exercise 2: Compare Models (45 minutes)

Compare N-BEATS, LSTM, and TCN:

```python
models = {
    "nbeats": NBEATSModel(),
    "lstm": LSTMModel(),
    "tcn": TCNModel(),
}

results = {}
for name, model in models.items():
    # Train each model
    # Evaluate on validation set
    # Store MAE, RMSE, training time
    pass

# Which model performed best?
# Which was fastest?
```

---

## Next Steps

**Next Module:** [09: Classical and ML Models →](09-classical-and-ml-models.md)

Learn about ARIMA, Prophet, XGBoost, and Random Forest.

---

## Summary Checklist

- [ ] I understand PyTorch tensor basics
- [ ] I know how N-BEATS works and when to use it
- [ ] I understand LSTM memory mechanisms
- [ ] I know why TCN is fast
- [ ] I understand TFT attention mechanisms
- [ ] I can compare and choose between neural models

**Continue to:** [Module 09: Classical and ML Models](09-classical-and-ml-models.md)
