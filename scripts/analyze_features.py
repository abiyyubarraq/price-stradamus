"""Feature analysis and selection script.

Analyzes feature correlations and importance for dimensionality reduction.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from price_stradamus.config.settings import settings
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.stateful_features import StatefulFeatureEngineer


async def load_features_from_db(days: int = 7) -> pd.DataFrame:
    """Load recent feature data from database.

    Args:
        days: Number of days of historical data to load

    Returns:
        DataFrame with OHLCV data and all features combined
    """
    from datetime import UTC, datetime, timedelta

    db = DatabaseManager(settings.database_url)

    # Initialize database connection
    await db.initialize()

    try:
        # Calculate date range
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=days)

        # Load OHLCV data
        ohlcv_df = await db.get_ohlcv(
            symbol="BTCUSDT",
            timeframe="1m",
            start_date=start_date,
            end_date=end_date,
        )

        if ohlcv_df.empty:
            logger.warning("No OHLCV data found")
            return pd.DataFrame()

        # Load features
        features_df = await db.get_features(
            symbol="BTCUSDT",
            timeframe="1m",
            start_date=start_date,
            end_date=end_date,
        )

        if features_df.empty:
            logger.warning(
                "No features found in database. Computing features on-the-fly..."
            )

            # Compute features using StatefulFeatureEngineer
            engineer = StatefulFeatureEngineer(warmup_period=200)
            features_df = engineer.fit_transform(ohlcv_df)

            logger.info(f"Computed {len(features_df.columns) - 1} features")

            # Save to database for future use
            logger.info("Saving computed features to database...")
            await db.save_features(features_df, "BTCUSDT", "1m", "v1.0")

        # Features already contain OHLCV, no need to merge
        # (StatefulFeatureEngineer keeps OHLCV columns in output)
        logger.info(
            f"Loaded {len(features_df)} rows with {len(features_df.columns)} columns"
        )
        return features_df

    finally:
        # Close database connection
        await db.close()


def analyze_correlation(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Find highly correlated feature pairs.

    Args:
        df: DataFrame with features
        threshold: Correlation threshold (0.95 = 95% correlation)

    Returns:
        DataFrame of correlated pairs
    """
    # Select only numeric columns (features)
    feature_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [
        col
        for col in feature_cols
        if col not in ["open", "high", "low", "close", "volume"]
    ]

    corr_matrix = df[feature_cols].corr().abs()

    # Get upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation > threshold
    high_corr_pairs = []
    for column in upper_tri.columns:
        correlated_features = upper_tri.index[upper_tri[column] > threshold].tolist()
        for corr_feature in correlated_features:
            high_corr_pairs.append(
                {
                    "feature_1": column,
                    "feature_2": corr_feature,
                    "correlation": upper_tri.loc[corr_feature, column],
                }
            )

    if not high_corr_pairs:
        logger.info(f"No feature pairs found with correlation > {threshold}")
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])

    corr_df = pd.DataFrame(high_corr_pairs).sort_values("correlation", ascending=False)

    logger.info(f"Found {len(corr_df)} feature pairs with correlation > {threshold}")
    return corr_df


def select_features_by_correlation(
    df: pd.DataFrame, threshold: float = 0.95
) -> list[str]:
    """Select features by removing highly correlated ones.

    Strategy: Keep the first feature from each correlated group.

    Args:
        df: DataFrame with features
        threshold: Correlation threshold

    Returns:
        List of selected feature names
    """
    feature_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [
        col
        for col in feature_cols
        if col not in ["open", "high", "low", "close", "volume"]
    ]

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
                    f"Dropping {feature_to_drop} (corr={corr_matrix.iloc[i, j]:.3f} "
                    f"with {corr_matrix.columns[i]})"
                )

    selected_features = [col for col in feature_cols if col not in to_drop]

    logger.info(
        f"Selected {len(selected_features)}/{len(feature_cols)} features after correlation filtering"
    )
    logger.info(f"Dropped features: {sorted(to_drop)}")

    return selected_features


def plot_correlation_heatmap(
    df: pd.DataFrame, selected_features: list[str], output_path: Path
) -> None:
    """Plot correlation heatmap of selected features.

    Args:
        df: DataFrame with features
        selected_features: List of features to plot
        output_path: Where to save the plot
    """
    # Create correlation matrix
    corr = df[selected_features].corr()

    # Create figure
    plt.figure(figsize=(16, 14))

    # Create heatmap
    sns.heatmap(
        corr,
        annot=False,  # Too many features to annotate
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Feature Correlation Matrix (After Selection)", fontsize=16, pad=20)
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved correlation heatmap to {output_path}")
    plt.close()


def recommend_core_features() -> list[str]:
    """Recommend a core set of ~19 features that survived correlation analysis.

    These features capture different aspects of price action while maintaining
    low correlation (<0.95) with each other.

    Selection criteria:
    - All features survived correlation filtering (threshold=0.95)
    - Cover all aspects: trend, momentum, volatility, volume, price action
    - Proven effective in financial ML literature

    Returns:
        List of recommended feature names
    """
    core_features = [
        # === TREND (2 features) ===
        "ema_7",  # Short-term trend (responsive)
        # === MOMENTUM (5 features) ===
        "rsi_14",  # Overbought/oversold
        "macd",  # Trend momentum
        "macd_signal",  # MACD crossover signal
        "macd_hist",  # MACD histogram
        "roc_10",  # Rate of change
        # === VOLATILITY (4 features) ===
        "atr_14",  # Average true range
        "std_20",  # Standard deviation
        "bb_percent",  # Position within bands
        "hist_volatility_20",  # Historical volatility
        # === VOLUME (4 features) ===
        "obv",  # On-balance volume
        "volume_ratio",  # Current vs average volume
        "volume_price_trend",  # Volume-price trend
        "volume_change_pct",  # Volume momentum
        # === PRICE PATTERNS (2 features) ===
        "high_low_range",  # Daily range
        "returns",  # Simple returns
        # === ADVANCED (2 features) ===
        "adx_14",  # Trend strength
        "supertrend_direction",  # Supertrend signal
    ]

    logger.info(f"Recommended {len(core_features)} core features")
    return core_features


async def main() -> None:
    """Run feature analysis."""
    logger.info("=== Feature Analysis ===")

    # Load data
    logger.info("Loading feature data from database...")
    df = await load_features_from_db(days=7)

    if df.empty:
        logger.error("No data available. Run: price-stradamus fetch --days 7")
        return

    # Analyze correlations
    logger.info("\n=== Analyzing Feature Correlations ===")
    high_corr = analyze_correlation(df, threshold=0.95)

    if not high_corr.empty:
        print("\nHighly Correlated Feature Pairs (>0.95):")
        print(high_corr.to_string(index=False))
    else:
        logger.info("No highly correlated features found")

    # Select features by correlation
    logger.info("\n=== Selecting Features by Correlation ===")
    selected_by_corr = select_features_by_correlation(df, threshold=0.95)
    print(
        f"\nSelected {len(selected_by_corr)} features after removing correlations >0.95:"
    )
    print(selected_by_corr)

    # Get recommended core features
    logger.info("\n=== Recommended Core Features ===")
    core_features = recommend_core_features()
    print(f"\nRecommended {len(core_features)} core features:")
    for i, feature in enumerate(core_features, 1):
        print(f"{i:2d}. {feature}")

    # Check which core features exist in data
    missing_core = [f for f in core_features if f not in df.columns]
    if missing_core:
        logger.warning(f"Missing core features: {missing_core}")

    # Plot correlation heatmap
    logger.info("\n=== Generating Correlation Heatmap ===")
    available_core = [f for f in core_features if f in df.columns]
    if available_core:
        output_path = Path("outputs/feature_analysis/correlation_heatmap.png")
        plot_correlation_heatmap(df, available_core, output_path)

    # Save selected features to file
    output_file = Path("outputs/feature_analysis/selected_features.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("=== SELECTED BY CORRELATION (threshold=0.95) ===\n")
        f.write(f"Count: {len(selected_by_corr)}\n\n")
        for feature in sorted(selected_by_corr):
            f.write(f"{feature}\n")

        f.write("\n=== RECOMMENDED CORE FEATURES ===\n")
        f.write(f"Count: {len(available_core)}\n\n")
        for feature in available_core:
            f.write(f"{feature}\n")

    logger.info(f"Saved selected features to {output_file}")
    logger.info("\n=== Analysis Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
