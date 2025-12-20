"""Feature selection utilities for dimensionality reduction.

Provides methods to select important features and reduce redundancy.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from loguru import logger
from sklearn.feature_selection import mutual_info_regression

# Core feature set based on domain knowledge
# This list contains ONLY features that survived correlation analysis (threshold=0.95)
# and represent diverse aspects of price action without redundancy
#
# Selection criteria:
# 1. All features have <0.95 correlation with each other
# 2. Cover all aspects: trend, momentum, volatility, volume, price action
# 3. Proven effective in financial ML literature
CORE_FEATURES = [
    # === TREND (2 features) ===
    # Note: All SMAs and most EMAs dropped due to high correlation
    # Kept: ema_7 (short-term)
    "ema_7",  # Short-term trend (responsive)
    "ema_14",
    # === MOMENTUM (5 features) ===
    "rsi_14",  # Overbought/oversold
    "macd",  # Trend momentum
    "macd_signal",  # MACD crossover signal
    "macd_hist",  # MACD histogram (momentum strength)
    "roc_10",  # Rate of change (momentum_XX dropped - corr=1.0 with roc_XX)
    "stoch_k",
    "stoch_d",
    "momentum_5",
    "momentum_10",
    # === VOLATILITY (4 features) ===
    "atr_14",  # Average true range
    "std_20",  # Standard deviation (bb_width dropped - corr=1.0 with std_20)
    "bb_percent",  # Position within Bollinger Bands
    "hist_volatility_20",  # Historical volatility
    # === VOLUME (4 features) ===
    "obv",  # On-balance volume
    "volume_ratio",  # Current vs average volume
    "volume_price_trend",  # Volume-price trend
    "volume_change_pct",  # Volume momentum
    # === PRICE PATTERNS (2 features) ===
    # Note: typical_price, log_returns, price_change all dropped (corr ~1.0 with returns)
    "high_low_range",  # Daily range
    "returns",  # Simple returns
    "returns_1",
    # === ADVANCED (2 features) ===
    "adx_14",  # Trend strength
    "supertrend_direction",  # Supertrend signal
]


def select_features(
    df: pd.DataFrame,
    method: Literal["core", "correlation", "all"] = "core",
    correlation_threshold: float = 0.95,
) -> list[str]:
    """Select features using specified method.

    Args:
        df: DataFrame with all features
        method: Feature selection method:
            - 'core': Use predefined core features (~22)
            - 'correlation': Remove highly correlated features (~25-30)
            - 'all': Use all available features
        correlation_threshold: Correlation threshold for 'correlation' method

    Returns:
        List of selected feature names

    Raises:
        ValueError: If invalid method specified
    """
    # Get all numeric feature columns (exclude OHLCV)
    base_cols = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timestamp",
        "symbol",
        "timeframe",
    }
    all_features = [col for col in df.columns if col not in base_cols]

    match method:
        case "all":
            logger.info(f"Using all {len(all_features)} features")
            return all_features

        case "core":
            # Filter core features to only those available in df
            available_core = [f for f in CORE_FEATURES if f in df.columns]
            missing_core = [f for f in CORE_FEATURES if f not in df.columns]

            if missing_core:
                logger.warning(
                    f"Missing {len(missing_core)} core features: {missing_core}"
                )

            logger.info(f"Using {len(available_core)} core features")
            return available_core

        case "correlation":
            selected = _select_by_correlation(df, all_features, correlation_threshold)
            logger.info(
                f"Selected {len(selected)} features after correlation filtering"
            )
            return selected

        case _:
            raise ValueError(
                f"Invalid method: {method}. Must be 'core', 'correlation', or 'all'"
            )


def _select_by_correlation(
    df: pd.DataFrame, feature_cols: list[str], threshold: float
) -> list[str]:
    """Select features by removing highly correlated ones.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        threshold: Correlation threshold (e.g., 0.95)

    Returns:
        List of selected features
    """
    # Compute correlation matrix
    corr_matrix = df[feature_cols].corr().abs()

    # Track features to drop
    to_drop = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                # Drop the second feature
                feature_to_drop = corr_matrix.columns[j]
                to_drop.add(feature_to_drop)
                logger.debug(
                    f"Dropping {feature_to_drop} "
                    f"(corr={corr_matrix.iloc[i, j]:.3f} with {corr_matrix.columns[i]})"
                )

    selected = [col for col in feature_cols if col not in to_drop]

    if to_drop:
        logger.info(
            f"Dropped {len(to_drop)} highly correlated features (threshold={threshold})"
        )

    return selected


def get_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Calculate feature importance using mutual information.

    Args:
        X: Feature matrix
        y: Target variable
        top_k: If specified, return only top K features

    Returns:
        DataFrame with features and their importance scores, sorted by importance
    """
    logger.info("Calculating feature importance using mutual information...")

    # Calculate mutual information scores
    mi_scores = mutual_info_regression(X, y, random_state=42)

    # Create DataFrame
    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": mi_scores}
    ).sort_values("importance", ascending=False)

    if top_k is not None:
        importance_df = importance_df.head(top_k)
        logger.info(f"Selected top {top_k} features by importance")

    return importance_df


def select_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 20,
) -> list[str]:
    """Select top K features by mutual information importance.

    Args:
        X: Feature matrix
        y: Target variable
        top_k: Number of features to select

    Returns:
        List of top K feature names
    """
    importance_df = get_feature_importance(X, y, top_k=top_k)
    return importance_df["feature"].tolist()


def compare_feature_sets(
    df: pd.DataFrame,
) -> dict[str, list[str]]:
    """Get all available feature sets for comparison.

    Args:
        df: DataFrame with all features

    Returns:
        Dictionary mapping feature set names to feature lists
    """
    feature_sets = {
        "all": select_features(df, method="all"),
        "core": select_features(df, method="core"),
        "correlation_95": select_features(
            df, method="correlation", correlation_threshold=0.95
        ),
        "correlation_90": select_features(
            df, method="correlation", correlation_threshold=0.90
        ),
    }

    # Print summary
    logger.info("=== Available Feature Sets ===")
    for name, features in feature_sets.items():
        logger.info(f"{name:20s}: {len(features):3d} features")

    return feature_sets
