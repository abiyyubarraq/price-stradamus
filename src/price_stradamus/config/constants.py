"""Project constants."""

from __future__ import annotations

from enum import Enum


# Timeframes
class Timeframe(str, Enum):
    """Supported timeframes."""

    ONE_MINUTE = "1m"
    THREE_MINUTES = "3m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    TWO_HOURS = "2h"
    FOUR_HOURS = "4h"
    SIX_HOURS = "6h"
    EIGHT_HOURS = "8h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"
    THREE_DAYS = "3d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


# Timeframe to milliseconds mapping
TIMEFRAME_TO_MS = {
    Timeframe.ONE_MINUTE: 60_000,
    Timeframe.THREE_MINUTES: 180_000,
    Timeframe.FIVE_MINUTES: 300_000,
    Timeframe.FIFTEEN_MINUTES: 900_000,
    Timeframe.THIRTY_MINUTES: 1_800_000,
    Timeframe.ONE_HOUR: 3_600_000,
    Timeframe.TWO_HOURS: 7_200_000,
    Timeframe.FOUR_HOURS: 14_400_000,
    Timeframe.SIX_HOURS: 21_600_000,
    Timeframe.EIGHT_HOURS: 28_800_000,
    Timeframe.TWELVE_HOURS: 43_200_000,
    Timeframe.ONE_DAY: 86_400_000,
    Timeframe.THREE_DAYS: 259_200_000,
    Timeframe.ONE_WEEK: 604_800_000,
    Timeframe.ONE_MONTH: 2_592_000_000,  # 30 days
}


# Model types
class ModelType(str, Enum):
    """Model types."""

    NEURAL = "neural"
    CLASSICAL = "classical"
    ML = "ml"
    AUTOML = "automl"


# Normalization methods
class NormalizationMethod(str, Enum):
    """Normalization methods."""

    MINMAX = "minmax"
    STANDARD = "standard"
    ROBUST = "robust"


# OHLCV column names
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Default parameters
DEFAULT_LOOKBACK_WINDOW = 60  # 60 candles
DEFAULT_PREDICTION_HORIZON = 5  # 5 candles ahead
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 100

# Technical indicator periods
SMA_PERIODS = [7, 14, 30, 50, 200]
EMA_PERIODS = [7, 14, 30, 50, 200]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14
STOCHASTIC_K = 14
STOCHASTIC_D = 3
ADX_PERIOD = 14
AROON_PERIOD = 25
CCI_PERIOD = 20
ROC_PERIODS = [5, 10, 20]
MOMENTUM_PERIODS = [5, 10, 20]
WILLR_PERIOD = 14

# Lag features
LAG_PERIODS = [1, 2, 3, 5, 10, 15, 30]

# Technical indicators configuration (aggregated for feature engineering)
TECHNICAL_INDICATORS = {
    "sma_periods": SMA_PERIODS,
    "ema_periods": EMA_PERIODS,
    "rsi_period": RSI_PERIOD,
    "macd": {
        "fast": MACD_FAST,
        "slow": MACD_SLOW,
        "signal": MACD_SIGNAL,
    },
    "bollinger_bands": {
        "period": BOLLINGER_PERIOD,
        "std": BOLLINGER_STD,
    },
    "atr_period": ATR_PERIOD,
    "stochastic": {
        "k": STOCHASTIC_K,
        "d": STOCHASTIC_D,
    },
    "adx_period": ADX_PERIOD,
    "aroon_period": AROON_PERIOD,
    "cci_period": CCI_PERIOD,
    "roc_periods": ROC_PERIODS,
    "momentum_periods": MOMENTUM_PERIODS,
    "willr_period": WILLR_PERIOD,
    "lag_periods": LAG_PERIODS,
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Validation split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# API limits
BINANCE_MAX_CANDLES_PER_REQUEST = 1000
BINANCE_RATE_LIMIT_PER_MINUTE = 1200

# Evaluation thresholds
MIN_DIRECTIONAL_ACCURACY = 0.52  # Better than random (50%)
MAX_ACCEPTABLE_MAPE = 0.05  # 5%

# File paths (relative to project root)
MODELS_DIR = "models"
LOGS_DIR = "logs"
DATA_DIR = "data"

# Database tables
TABLE_OHLCV_RAW = "market_data.ohlcv_raw"
TABLE_FEATURES = "market_data.features"
TABLE_PREDICTIONS = "ml_data.predictions"
TABLE_MODEL_METADATA = "ml_data.model_metadata"


# =============================================================================
# TRADING COSTS - Realistic defaults for backtesting
# =============================================================================
# IMPORTANT: These costs are CRITICAL for realistic backtesting.
# Without accounting for costs, your results will be overly optimistic.

# Binance fee schedule (as of 2024)
DEFAULT_COMMISSION_PCT = 0.001  # 0.1% per trade (Binance taker fee)
DEFAULT_SLIPPAGE_PCT = 0.0005  # 0.05% estimated slippage for BTC/USDT
MIN_REALISTIC_ROUND_TRIP_COST = 0.002  # 0.2% minimum realistic cost

# Cost with BNB discount (25% off)
BNB_DISCOUNTED_COMMISSION_PCT = 0.00075  # 0.075% with BNB discount


# =============================================================================
# WALK-FORWARD VALIDATION - Defaults for realistic backtesting
# =============================================================================
# Walk-forward validation prevents overfitting by:
# 1. Retraining the model as new data arrives
# 2. Testing on out-of-sample data only
# 3. Simulating real-world model deployment

DEFAULT_WALK_FORWARD_FOLDS = 5  # Number of walk-forward folds
DEFAULT_TRAIN_WINDOW_DAYS = 60  # Training window size (days)
DEFAULT_TEST_WINDOW_DAYS = 10  # Test window size (days)
DEFAULT_GAP_DAYS = 0  # Gap between train/test (for processing delays)


# =============================================================================
# FEATURE ENGINEERING - Settings for StatefulFeatureEngineer
# =============================================================================
# Warmup period: Number of initial rows to drop after feature generation.
# Technical indicators (RSI, MACD, SMA_200) produce NaN for first N periods.
# This should be >= the longest indicator lookback period.

FEATURE_WARMUP_PERIOD = 200  # Covers SMA_200 and most indicators


# =============================================================================
# REALISTIC EXPECTATIONS - Benchmarks from quant-analyst.md
# =============================================================================
# Use these to validate your results. If metrics exceed these, check for bugs!

# Directional accuracy expectations for BTC 1-minute
REALISTIC_DIR_ACC_MIN = 0.51  # 51% - barely better than random
REALISTIC_DIR_ACC_MAX = 0.55  # 55% - exceptional
SUSPICIOUS_DIR_ACC = 0.60  # 60% - likely data leakage

# Sharpe ratio expectations
REALISTIC_SHARPE_MIN = 0.5  # Low but realistic
REALISTIC_SHARPE_MAX = 2.0  # Very good
SUSPICIOUS_SHARPE = 3.0  # Likely bugs or leakage

# Win rate expectations
REALISTIC_WIN_RATE_MIN = 0.48  # 48%
REALISTIC_WIN_RATE_MAX = 0.55  # 55%
SUSPICIOUS_WIN_RATE = 0.60  # 60% - check for problems
