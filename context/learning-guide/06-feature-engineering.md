# Module 06: Feature Engineering

**Duration:** 3-4 hours | **Difficulty:** Intermediate | **Prerequisites:** Modules 01-05

## 🎯 Learning Objectives

After this module, you will:
- Understand what technical indicators are
- Calculate common indicators (RSI, MACD, Bollinger Bands)
- Use pandas-ta library
- Create lag features for ML models
- Understand feature importance

---

## What Are Technical Indicators?

**Technical Indicators** = Mathematical calculations based on price/volume to identify patterns.

### Categories

1. **Trend Indicators** - Direction of price movement
   - SMA (Simple Moving Average)
   - EMA (Exponential Moving Average)
   - MACD (Moving Average Convergence Divergence)

2. **Momentum Indicators** - Speed of price movement
   - RSI (Relative Strength Index)
   - Stochastic Oscillator
   - ROC (Rate of Change)

3. **Volatility Indicators** - Price fluctuation range
   - Bollinger Bands
   - ATR (Average True Range)

4. **Volume Indicators** - Trading volume patterns
   - OBV (On-Balance Volume)
   - VWAP (Volume Weighted Average Price)

---

## Common Indicators Explained

### 1. SMA (Simple Moving Average)

**Formula:**
```
SMA = Average of last N prices

Example (N=3):
Prices: [100, 102, 101, 103, 105]
SMA[2] = (100 + 102 + 101) / 3 = 101
SMA[3] = (102 + 101 + 103) / 3 = 102
```

**Code:**
```python
df['sma_20'] = df['close'].rolling(window=20).mean()
```

**Use:** Identifies trend direction (price above SMA = uptrend)

### 2. RSI (Relative Strength Index)

**Concept:** Measures momentum (0-100 scale)
- RSI > 70: Overbought (might go down)
- RSI < 30: Oversold (might go up)

**Code:**
```python
import pandas_ta as ta

df['rsi'] = ta.rsi(df['close'], length=14)
```

### 3. MACD

**Concept:** Trend following momentum indicator

```python
macd = ta.macd(df['close'])
df['macd'] = macd['MACD_12_26_9']
df['macd_signal'] = macd['MACDs_12_26_9']
df['macd_hist'] = macd['MACDh_12_26_9']
```

**Signal:** When MACD crosses above signal line = bullish

### 4. Bollinger Bands

**Concept:** Volatility bands around moving average

```python
bb = ta.bbands(df['close'], length=20, std=2)
df['bb_upper'] = bb['BBU_20_2.0']
df['bb_middle'] = bb['BBM_20_2.0']
df['bb_lower'] = bb['BBL_20_2.0']
```

**Use:** Price touching lower band = potential buy signal

---

## Using pandas-ta

### Complete Feature Generation

```python
# src/price_stradamus/data/features.py
import pandas_ta as ta

class FeatureEngineer:
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all technical indicators."""

        # Trend indicators
        df['sma_5'] = ta.sma(df['close'], length=5)
        df['sma_10'] = ta.sma(df['close'], length=10)
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)

        # MACD
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']

        # Momentum indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['stoch_k'] = ta.stoch(df['high'], df['low'], df['close'])['STOCHk_14_3_3']

        # Volatility indicators
        bb = ta.bbands(df['close'], length=20)
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_lower'] = bb['BBL_20_2.0']
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        # Volume indicators
        df['obv'] = ta.obv(df['close'], df['volume'])
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

        # Drop NaN rows (from indicator calculations)
        df = df.dropna()

        return df
```

---

## Lag Features

### Creating Time-Shifted Features

```python
# Previous values as features
df['close_lag_1'] = df['close'].shift(1)  # Previous close
df['close_lag_2'] = df['close'].shift(2)  # 2 candles ago
df['volume_lag_1'] = df['volume'].shift(1)

# Returns (percentage change)
df['returns_1'] = df['close'].pct_change(1)  # 1-period return
df['returns_5'] = df['close'].pct_change(5)  # 5-period return
```

**Why useful:** ML models need historical context

---

## Quick Reference

```python
import pandas_ta as ta

# Trend
df['sma_20'] = ta.sma(df['close'], 20)
df['ema_12'] = ta.ema(df['close'], 12)

# Momentum
df['rsi'] = ta.rsi(df['close'], 14)

# Volatility
bb = ta.bbands(df['close'], 20)

# Volume
df['obv'] = ta.obv(df['close'], df['volume'])

# Lag features
df['close_lag_1'] = df['close'].shift(1)
df['returns'] = df['close'].pct_change()
```

---

## Next Steps

**Next Module:** [07: Model Fundamentals →](07-model-fundamentals.md)

---

## Summary Checklist

- [ ] I understand technical indicators
- [ ] I can calculate RSI, MACD, Bollinger Bands
- [ ] I know how to use pandas-ta
- [ ] I can create lag features
