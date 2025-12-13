"""Compare model performance with different feature sets.

Tests the impact of feature selection on model accuracy and training time.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb
from loguru import logger

from price_stradamus.config.settings import settings
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.stateful_features import StatefulFeatureEngineer
from price_stradamus.evaluation.metrics import MetricsCalculator
from price_stradamus.utils.feature_selection import (
    compare_feature_sets,
)


async def load_training_data(days: int = 7) -> pd.DataFrame:
    """Load historical data for training.

    Args:
        days: Number of days of data to load

    Returns:
        DataFrame with OHLCV data
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
        df = await db.get_ohlcv(
            symbol="BTCUSDT",
            timeframe="1m",
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(f"Loaded {len(df)} rows of training data")
        return df

    finally:
        # Close database connection
        await db.close()


async def train_and_evaluate_model(
    feature_set_name: str,
    selected_features: list[str],
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> dict:
    """Train model with specified features and evaluate.

    Args:
        feature_set_name: Name of feature set for logging
        selected_features: List of feature names to use
        df_train: Training data
        df_test: Test data

    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training with {feature_set_name} ({len(selected_features)} features)")
    logger.info(f"{'=' * 60}")

    # Initialize feature engineer
    engineer = StatefulFeatureEngineer(warmup_period=200)

    # Generate features
    logger.info("Generating features on training data...")
    train_features = engineer.fit_transform(df_train)

    logger.info("Generating features on test data...")
    test_features = engineer.transform(df_test)

    # Create target column (next step's close price)
    # For simplicity, predict the next 1-step close price
    train_features["target"] = train_features["close"].shift(-1)
    test_features["target"] = test_features["close"].shift(-1)

    # Drop rows with NaN target (last row after shift)
    train_features = train_features.dropna(subset=["target"])
    test_features = test_features.dropna(subset=["target"])

    logger.info(
        f"Created target column. Train: {len(train_features)} samples, Test: {len(test_features)} samples"
    )

    # Select only specified features + target
    available_features = [f for f in selected_features if f in train_features.columns]
    missing_features = [f for f in selected_features if f not in train_features.columns]

    if missing_features:
        logger.warning(f"Missing features: {missing_features}")

    # Prepare feature columns
    feature_cols = available_features + ["target"]

    train_subset = train_features[feature_cols].copy()
    test_subset = test_features[feature_cols].copy()

    logger.info(f"Using {len(available_features)} features for training")

    # Train model
    logger.info("Training XGBoost model...")
    start_time = time.time()

    # Use raw XGBoost (not Darts wrapper) for feature comparison
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.01,
        random_state=42,
        tree_method="auto",  # Will use GPU if available
    )

    # Prepare data for XGBoost (it needs numpy arrays)
    X_train = train_subset.drop("target", axis=1).values
    y_train = train_subset["target"].values

    X_test = test_subset.drop("target", axis=1).values
    y_test = test_subset["target"].values

    # Fit model
    model.fit(X_train, y_train, verbose=False)

    training_time = time.time() - start_time

    # Make predictions
    logger.info("Generating predictions...")
    predictions = model.predict(X_test)

    # Calculate metrics
    metrics = MetricsCalculator.calculate_all(y_test, predictions)

    # Get feature importance from XGBoost
    importance = model.feature_importances_
    feature_importance = pd.DataFrame(
        {"feature": available_features, "importance": importance}
    ).sort_values("importance", ascending=False)

    logger.info("\nTop 10 Features by Importance:")
    print(feature_importance.head(10).to_string(index=False))

    # Print results
    logger.info(f"\n=== Results for {feature_set_name} ===")
    logger.info(f"Training time: {training_time:.2f}s")
    logger.info(f"MAE: {metrics['mae']:.4f}")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"MAPE: {metrics['mape']:.4f}%")
    logger.info(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")

    return {
        "feature_set": feature_set_name,
        "n_features": len(available_features),
        "training_time": training_time,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "directional_accuracy": metrics["directional_accuracy"],
        "top_10_features": feature_importance.head(10)["feature"].tolist(),
    }


async def main() -> None:
    """Compare different feature sets."""
    logger.info("=== Feature Set Comparison Experiment ===")

    # Load data
    logger.info("Loading data...")
    df = await load_training_data(days=7)

    if df.empty:
        logger.error("No data available. Run: price-stradamus fetch --days 7")
        return

    # Split into train/test (80/20)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    logger.info(f"Training set: {len(df_train)} samples")
    logger.info(f"Test set: {len(df_test)} samples")

    # First, generate all features to analyze
    logger.info("\nGenerating all features for analysis...")
    engineer = StatefulFeatureEngineer(warmup_period=200)
    all_features_df = engineer.fit_transform(df_train)

    # Get different feature sets
    feature_sets = compare_feature_sets(all_features_df)

    # Compare each feature set
    results = []

    for set_name, features in feature_sets.items():
        if not features:
            logger.warning(f"Skipping {set_name} - no features available")
            continue

        result = await train_and_evaluate_model(
            set_name,
            features,
            df_train,
            df_test,
        )
        results.append(result)

    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("=== FINAL COMPARISON ===")
    logger.info("=" * 80)

    comparison_df = pd.DataFrame(results)

    print("\nPerformance Comparison:")
    print(
        comparison_df[
            [
                "feature_set",
                "n_features",
                "mae",
                "rmse",
                "directional_accuracy",
                "training_time",
            ]
        ].to_string(index=False)
    )

    # Find best by MAE
    best_mae = comparison_df.loc[comparison_df["mae"].idxmin()]
    logger.info(f"\n🏆 Best MAE: {best_mae['feature_set']} ({best_mae['mae']:.4f})")

    # Find best by directional accuracy
    best_dir_acc = comparison_df.loc[comparison_df["directional_accuracy"].idxmax()]
    logger.info(
        f"🏆 Best Directional Accuracy: {best_dir_acc['feature_set']} ({best_dir_acc['directional_accuracy']:.4f})"
    )

    # Find fastest
    fastest = comparison_df.loc[comparison_df["training_time"].idxmin()]
    logger.info(
        f"⚡ Fastest Training: {fastest['feature_set']} ({fastest['training_time']:.2f}s)"
    )

    # Save results
    output_dir = Path("outputs/feature_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_dir / "comparison_results.csv", index=False)
    logger.info(f"\n📊 Saved results to {output_dir / 'comparison_results.csv'}")

    # Recommendations
    logger.info("\n=== RECOMMENDATIONS ===")

    if len(results) >= 2:
        core_result = next((r for r in results if r["feature_set"] == "core"), None)
        all_result = next((r for r in results if r["feature_set"] == "all"), None)

        if core_result and all_result:
            mae_diff = (
                (core_result["mae"] - all_result["mae"]) / all_result["mae"]
            ) * 100
            time_diff = (
                (all_result["training_time"] - core_result["training_time"])
                / all_result["training_time"]
            ) * 100

            logger.info(
                f"Core features: {mae_diff:+.2f}% MAE change, {time_diff:.1f}% faster training"
            )

            if abs(mae_diff) < 5 and time_diff > 30:
                logger.info(
                    "✅ Recommendation: Use CORE features - minimal accuracy loss, much faster"
                )
            elif mae_diff < -5:
                logger.info(
                    "✅ Recommendation: Use ALL features - significantly better accuracy"
                )
            else:
                logger.info(
                    "✅ Recommendation: Use CORRELATION-selected features - balanced approach"
                )

    logger.info("\n=== Experiment Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
