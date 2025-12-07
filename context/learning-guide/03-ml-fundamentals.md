# Module 03: Machine Learning Fundamentals

**Duration:** 2-3 hours | **Difficulty:** Beginner | **Prerequisites:** Modules 01-02

## 🎯 Learning Objectives

After this module, you will:
- Understand what machine learning is (without complex math)
- Know the difference between supervised and unsupervised learning
- Understand time series forecasting
- Know how to split data properly (train/validation/test)
- Recognize overfitting and underfitting

---

## What is Machine Learning?

###The Simple Explanation

**Machine Learning** = Teaching computers to learn patterns from data instead of explicitly programming rules.

**Traditional Programming:**
```
Rules + Data → Output

Example:
if price > 50000:
    prediction = "will go up"
```

**Machine Learning:**
```
Data + Output → Rules (learned by model)

Historical prices + Actual movements → Model predicts future
```

### In Price Stradamus Context

```python
# We DON'T write thousands of rules
# Instead:
model = NBEATSModel()
model.fit(historical_data)  # Model learns patterns
prediction = model.predict(steps=5)  # Applies learned patterns
```

---

## Types of Machine Learning

### Supervised Learning (What Price Stradamus Uses)

**Definition**: Learning from labeled examples.

```
Input (Features) → Model → Output (Labels)

Bitcoin prediction:
Input: Past prices, volume, indicators
Output: Future prices
```

---

## Time Series Forecasting

### What is a Time Series?

**Time Series** = Data points indexed by time (ordered chronologically).

```
2024-01-01 09:00 → $42,000
2024-01-01 09:01 → $42,050
2024-01-01 09:02 → $42,025
```

**Key property**: Order matters!

### Multi-Step Forecasting (Price Stradamus)

```
Past 60 candles → Predict next 5 candles
[t-59, ..., t] → [t+1, t+2, t+3, t+4, t+5]
```

---

## Data Splitting

### The Three Splits

```
[==================][======][===========]
   Train (70%)    Val(15%)  Test(15%)

Train: Learn patterns
Validation: Tune hyperparameters
Test: Final evaluation (DON'T TOUCH until end!)
```

### Walk-Forward Validation (Time Series)

**CORRECT approach:**
```
Always use PAST data to predict FUTURE
Step 1: [Train====][Val][Test]
Step 2: [Train========][Val][Test]
Never use future data in training!
```

---

## Overfitting vs Underfitting

### Overfitting (Too Complex)

```
Training error: 0.5%  ← Great!
Test error: 25%       ← Terrible!
```

**Problem**: Model memorized training data instead of learning patterns.

**Solutions**: Simpler model, more data, early stopping

### Underfitting (Too Simple)

```
Training error: 30%   ← Bad
Test error: 32%       ← Consistently bad
```

**Problem**: Model too simple to capture patterns.

**Solutions**: More complex model, better features, train longer

---

## Model Evaluation

### Key Metrics

#### 1. MAE (Mean Absolute Error)
```python
MAE = Average of |predicted - actual|
```
**Interpretation**: Average prediction error in dollars

#### 2. RMSE (Root Mean Squared Error)
```python
RMSE = sqrt(average of (predicted - actual)²)
```
**Interpretation**: Penalizes large errors more

#### 3. Directional Accuracy (Most Important for Trading!)
```python
% of times we predicted direction correctly

50% = Random guessing
>52% = Potentially profitable
```

**Example from Price Stradamus:**
```python
def calculate_all(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # Directional accuracy
    pred_dir = np.sign(np.diff(predicted))
    actual_dir = np.sign(np.diff(actual))
    dir_acc = np.mean(pred_dir == actual_dir) * 100

    return {"mae": mae, "rmse": rmse, "directional_accuracy": dir_acc}
```

---

## Quick Reference

### ML Key Concepts

- **Supervised Learning**: Learning from labeled examples
- **Time Series**: Data ordered by time
- **Training**: Process of learning patterns
- **Validation**: Tuning hyperparameters
- **Testing**: Final evaluation (once!)
- **Overfitting**: Memorizing vs learning
- **Directional Accuracy**: Most important metric for trading

---

## Practice Exercises

### Exercise 1: Identifying Overfitting

Which shows overfitting?

A) Train: MAE=50, Test: MAE=55
B) Train: MAE=5, Test: MAE=80
C) Train: MAE=100, Test: MAE=105

<details>
<summary>Answer</summary>

**B)** - Great on training (5), terrible on test (80) = Overfitting
</details>

### Exercise 2: Time Series Splitting

You have 2020-2024 Bitcoin data. Correct split?

A) Random split
B) Train: 2020-2022, Val: 2023, Test: 2024
C) Train: 2024, Val: 2023, Test: 2020-2022

<details>
<summary>Answer</summary>

**B)** - Use past to predict future, respect chronological order
</details>

---

## Next Steps

**Next Module:** [04: Project Architecture →](04-project-architecture.md)

---

## Summary Checklist

- [ ] I understand what ML is
- [ ] I know supervised learning
- [ ] I understand time series forecasting
- [ ] I can split data properly
- [ ] I recognize overfitting/underfitting
- [ ] I understand key metrics

**Continue to:** [Module 04: Project Architecture](04-project-architecture.md)
