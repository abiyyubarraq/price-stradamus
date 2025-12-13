"""Example: Training with Proper Temporal Split to Prevent Data Leakage.

This script demonstrates the CORRECT way to train and evaluate models
without data leakage, addressing the concern:

"If I train on 2025-11-04 to 2025-12-04, then evaluate on the same period,
isn't that cheating?"

YES! This script shows the proper approach.

Usage:
    python scripts/train_with_proper_split.py --model nbeats
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from price_stradamus.config.constants import FEATURE_WARMUP_PERIOD
from price_stradamus.config.settings import settings
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.preprocessor import DataPreprocessor
from price_stradamus.data.stateful_features import StatefulFeatureEngineer
from price_stradamus.evaluation.metrics import MetricsCalculator
from price_stradamus.evaluation.walk_forward import (
    DataLeakageChecker,
    WalkForwardSplitter,
)
from price_stradamus.models.registry import ModelRegistry

console = Console()
app = typer.Typer()


@app.command()
def main(
    model: str = typer.Option("nbeats", "--model", "-m", help="Model to train"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Trading symbol"),
    timeframe: str = typer.Option("1m", "--timeframe", help="Timeframe"),
    method: str = typer.Option(
        "simple",
        "--method",
        help="Split method: 'simple' or 'walk-forward'",
    ),
) -> None:
    """Train model with proper temporal split to prevent data leakage.

    Two approaches:
    1. Simple split (70/15/15) - Fast, good for Phase 1
    2. Walk-forward validation - More robust, gold standard
    """

    async def _run():
        console.print("[bold cyan]Training with Data Leakage Prevention[/bold cyan]\n")

        # 1. Load ALL data from database
        console.print(f"📊 Loading {symbol} {timeframe} data from database...")
        db = DatabaseManager(settings.database_url)
        await db.initialize()

        df = await db.get_ohlcv(symbol, timeframe)
        await db.close()

        if df.empty:
            console.print("[red]✗ No data found. Run 'fetch' command first.[/red]")
            raise typer.Exit(1)

        # Show data range
        import pandas as pd

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        data_start = df["timestamp"].min()
        data_end = df["timestamp"].max()

        console.print(f"[green]✓ Loaded {len(df)} candles[/green]")
        console.print(f"  Data range: {data_start} to {data_end}")
        console.print()

        # 2. Preprocess and create features
        console.print("⚙️  Preprocessing and generating features...")
        preprocessor = DataPreprocessor()
        df = preprocessor.validate_ohlcv(df)
        df = preprocessor.handle_missing_values(df)

        engineer = StatefulFeatureEngineer(warmup_period=FEATURE_WARMUP_PERIOD)
        df_features = engineer.fit_transform(df)
        ts = engineer.to_darts_timeseries(df_features, value_cols=["close"])

        console.print(
            f"[green]✓ Created time series with {len(ts)} timesteps[/green]\n"
        )

        # 3. Create model
        console.print(f"🤖 Creating {model} model...")
        model_instance = ModelRegistry.create_model(
            model,
            input_chunk_length=60,
            output_chunk_length=5,
            n_epochs=50,  # Reduced for demo
        )
        console.print(f"[green]✓ Model: {model_instance}[/green]\n")

        # 4. Choose split method
        if method == "simple":
            await _simple_split(
                model_instance,
                ts,
            )
        elif method == "walk-forward":
            await _walk_forward_split(model_instance, df_features)
        else:
            console.print(f"[red]Unknown method: {method}[/red]")
            raise typer.Exit(1)

    async def _simple_split(
        model_instance,
        ts,
    ):
        """Simple 70/15/15 temporal split."""
        console.print(
            "[bold yellow]📍 Method: Simple Temporal Split (70/15/15)[/bold yellow]"
        )
        console.print()

        # Calculate split indices
        train_size = int(len(ts) * 0.7)
        val_size = int(len(ts) * 0.15)

        train_ts = ts[:train_size]
        val_ts = ts[train_size : train_size + val_size]
        test_ts = ts[train_size + val_size :]

        # Show split details
        train_start = train_ts.time_index[0]
        train_end = train_ts.time_index[-1]
        val_start = val_ts.time_index[0]
        val_end = val_ts.time_index[-1]
        test_start = test_ts.time_index[0]
        test_end = test_ts.time_index[-1]

        console.print("📅 Data Split:")
        table = Table()
        table.add_column("Set", style="cyan")
        table.add_column("Size", style="yellow")
        table.add_column("Start", style="green")
        table.add_column("End", style="green")

        table.add_row("Training", str(len(train_ts)), str(train_start), str(train_end))
        table.add_row("Validation", str(len(val_ts)), str(val_start), str(val_end))
        table.add_row("Test", str(len(test_ts)), str(test_start), str(test_end))

        console.print(table)
        console.print()

        # ✅ CRITICAL: Check for data leakage
        console.print("🔍 Checking for data leakage...")
        checker = DataLeakageChecker()

        # Check train vs validation
        result_val = checker.check_temporal_leakage(train_ts, val_ts)
        if result_val["has_leakage"]:
            console.print("[red]✗ LEAKAGE DETECTED: Train/Val overlap![/red]")
            raise ValueError("Data leakage in train/val split")

        # Check validation vs test
        result_test = checker.check_temporal_leakage(val_ts, test_ts)
        if result_test["has_leakage"]:
            console.print("[red]✗ LEAKAGE DETECTED: Val/Test overlap![/red]")
            raise ValueError("Data leakage in val/test split")

        console.print("[green]✓ No data leakage detected - splits are valid![/green]")
        console.print(f"  Gap between train and val: {result_val['gap']:.1f} hours")
        console.print(f"  Gap between val and test: {result_test['gap']:.1f} hours")
        console.print()

        # Train model (only on training data!)
        console.print("🔥 Training model (only on training data)...")
        model_instance.fit(train_ts, val_ts)
        console.print("[green]✓ Training complete[/green]\n")

        # Evaluate on test set (model has NEVER seen this data!)
        console.print("📊 Evaluating on test set (unseen data)...")
        predictions = model_instance.predict(n=len(test_ts), series=train_ts)

        # Calculate metrics
        actuals = test_ts.values().flatten()
        preds = predictions.values().flatten()[: len(actuals)]

        metrics = MetricsCalculator.calculate_all(actuals, preds)

        # Display results
        console.print("\n[bold green]Test Set Results (No Data Leakage!):[/bold green]")
        results_table = Table()
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")

        results_table.add_row("MAE", f"{metrics['mae']:.2f}")
        results_table.add_row("RMSE", f"{metrics['rmse']:.2f}")
        results_table.add_row("MAPE", f"{metrics['mape']:.2f}%")
        results_table.add_row(
            "Directional Accuracy", f"{metrics['directional_accuracy']:.2f}%"
        )

        console.print(results_table)

        # Save model
        model_path = Path("models") / f"{model_instance.name}_no_leakage.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_instance.save(model_path)
        console.print(f"\n[green]✓ Model saved to {model_path}[/green]")

    async def _walk_forward_split(model_instance, df_features):
        """Walk-forward validation (gold standard)."""
        console.print("[bold yellow]📍 Method: Walk-Forward Validation[/bold yellow]")
        console.print()

        # Create splitter
        splitter = WalkForwardSplitter(
            train_window_days=30,  # Train on 30 days
            test_window_days=5,  # Test on next 5 days
            step_days=5,  # Move forward 5 days
            gap_days=0,  # No gap (can add for realism)
        )

        # Get all windows
        start_date = df_features["timestamp"].min()
        end_date = df_features["timestamp"].max()

        windows = list(splitter.split(start_date, end_date))

        console.print(f"📅 Generated {len(windows)} walk-forward windows:")
        console.print()

        # Show first few windows
        for i, window in enumerate(windows[:3], 1):
            console.print(f"[cyan]Fold {i}:[/cyan]")
            console.print(f"  Train: {window.train_start} → {window.train_end}")
            console.print(f"  Test:  {window.test_start} → {window.test_end}")
            console.print()

        if len(windows) > 3:
            console.print(f"[dim]... and {len(windows) - 3} more folds[/dim]\n")

        # Train and evaluate on each fold
        console.print("🔥 Training and evaluating on each fold...")
        all_metrics = []

        for i, (train_df, test_df) in enumerate(
            splitter.split_dataframe(df_features), 1
        ):
            console.print(f"\n[cyan]Processing Fold {i}/{len(windows)}...[/cyan]")

            # Create time series
            engineer = StatefulFeatureEngineer(warmup_period=FEATURE_WARMUP_PERIOD)
            train_ts = engineer.to_darts_timeseries(train_df, value_cols=["close"])
            test_ts = engineer.to_darts_timeseries(test_df, value_cols=["close"])

            # ✅ CRITICAL: Check for leakage
            checker = DataLeakageChecker()
            result = checker.check_temporal_leakage(train_ts, test_ts)
            if result["has_leakage"]:
                console.print(f"[red]✗ LEAKAGE in fold {i}![/red]")
                continue

            # Train
            model_instance.fit(train_ts)

            # Predict
            predictions = model_instance.predict(n=len(test_ts), series=train_ts)

            # Metrics
            actuals = test_ts.values().flatten()
            preds = predictions.values().flatten()[: len(actuals)]

            fold_metrics = MetricsCalculator.calculate_all(actuals, preds)
            fold_metrics["fold"] = i
            all_metrics.append(fold_metrics)

            console.print(
                f"  MAE: {fold_metrics['mae']:.2f}, "
                f"Dir Acc: {fold_metrics['directional_accuracy']:.2f}%"
            )

        # Aggregate results
        console.print(
            "\n[bold green]Walk-Forward Results (Average Across Folds):[/bold green]"
        )

        import numpy as np

        avg_metrics = {
            "MAE": np.mean([m["mae"] for m in all_metrics]),
            "RMSE": np.mean([m["rmse"] for m in all_metrics]),
            "MAPE": np.mean([m["mape"] for m in all_metrics]),
            "Dir Accuracy": np.mean([m["directional_accuracy"] for m in all_metrics]),
        }

        final_table = Table()
        final_table.add_column("Metric", style="cyan")
        final_table.add_column("Average", style="green")
        final_table.add_column("Std Dev", style="yellow")

        for metric_name, avg_value in avg_metrics.items():
            metric_key = {
                "MAE": "mae",
                "RMSE": "rmse",
                "MAPE": "mape",
                "Dir Accuracy": "directional_accuracy",
            }[metric_name]

            std_value = np.std([m[metric_key] for m in all_metrics])
            final_table.add_row(
                metric_name,
                f"{avg_value:.2f}",
                f"±{std_value:.2f}",
            )

        console.print(final_table)

        console.print(
            f"\n[green]✓ Evaluated on {len(all_metrics)} folds with NO DATA LEAKAGE[/green]"
        )

    asyncio.run(_run())


if __name__ == "__main__":
    app()
