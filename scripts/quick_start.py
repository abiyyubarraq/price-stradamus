#!/usr/bin/env python3
"""Quick start script for Price Stradamus.

This script runs a complete end-to-end workflow:
1. Fetch data from Binance
2. Train N-BEATS model
3. Evaluate with backtesting
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from price_stradamus.config.constants import FEATURE_WARMUP_PERIOD
from price_stradamus.config.settings import settings
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.stateful_features import StatefulFeatureEngineer
from price_stradamus.data.fetcher import BinanceDataFetcher
from price_stradamus.data.preprocessor import DataPreprocessor
from price_stradamus.evaluation.backtester import Backtester
from price_stradamus.models.neural.nbeats import NBEATSModel  # noqa: F401
from price_stradamus.models.registry import ModelRegistry

console = Console()


async def main():
    """Run quick start workflow."""
    console.print("\n[bold cyan]Price Stradamus - Quick Start[/bold cyan]\n")

    # Configuration
    symbol = "BTCUSDT"
    timeframe = "1m"
    days = 7
    model_name = "nbeats"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Fetch data
        task1 = progress.add_task(
            f"[cyan]Fetching {days} days of {symbol} data...", total=None
        )

        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=days)

        async with BinanceDataFetcher() as fetcher:
            df = await fetcher.fetch_historical_range(
                symbol=symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

        progress.update(task1, completed=True)
        console.print(f"[green]✓ Fetched {len(df)} candles[/green]")

        # Save to database
        task2 = progress.add_task("[cyan]Saving to database...", total=None)

        db = DatabaseManager(settings.database_url)
        await db.initialize()
        rows_saved = await db.save_ohlcv(df, symbol, timeframe)
        await db.close()

        progress.update(task2, completed=True)
        console.print(f"[green]✓ Saved {rows_saved} candles to database[/green]")

        # Step 2: Preprocess and generate features
        task3 = progress.add_task("[cyan]Generating features...", total=None)

        preprocessor = DataPreprocessor()
        df = preprocessor.validate_ohlcv(df)
        df = preprocessor.handle_missing_values(df)

        engineer = StatefulFeatureEngineer(warmup_period=FEATURE_WARMUP_PERIOD)
        df_features = engineer.fit_transform(df)

        progress.update(task3, completed=True)
        console.print(
            f"[green]✓ Generated {len(engineer.feature_list)} technical indicators[/green]"
        )

        # Create time series
        ts = engineer.to_darts_timeseries(df_features, value_cols=["close"])

        # Step 3: Train model
        task4 = progress.add_task(
            "[cyan]Training N-BEATS model (this may take a few minutes)...", total=None
        )

        model = ModelRegistry.create_model(
            model_name,
            input_chunk_length=60,
            output_chunk_length=5,
            n_epochs=50,  # Reduced for quick start
        )

        train_size = int(len(ts) * 0.7)
        val_size = int(len(ts) * 0.15)

        train_ts = ts[:train_size]
        val_ts = ts[train_size : train_size + val_size]

        model.fit(train_ts, val_ts)

        progress.update(task4, completed=True)
        console.print("[green]✓ Model training complete[/green]")

        # Save model
        model_path = Path("models") / f"{model_name}_quickstart.pth"
        model.save(model_path)
        console.print(f"[green]✓ Model saved to {model_path}[/green]")

        # Step 4: Evaluate
        task5 = progress.add_task("[cyan]Running backtest evaluation...", total=None)

        backtester = Backtester(model, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        results = backtester.run_expanding_window(ts)

        progress.update(task5, completed=True)
        console.print("[green]✓ Backtest complete[/green]")

    # Display results
    console.print("\n[bold]Backtest Results:[/bold]")
    metrics_df = backtester.get_summary(results)

    from rich.table import Table

    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for _, row in metrics_df.iterrows():
        table.add_row(str(row["Metric"]), str(row["Value"]))

    console.print(table)

    # Summary
    console.print("\n[bold green]Quick Start Complete![/bold green]")
    console.print(f"\nModel saved to: {model_path}")
    console.print(f"Database: {len(df)} candles stored")
    console.print(f"Features: {len(engineer.feature_list)} technical indicators")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("  • Try different models: price-stradamus list-models")
    console.print("  • Fetch more data: price-stradamus fetch --days 30")
    console.print("  • Train longer: price-stradamus train --model nbeats --epochs 200")
    console.print("  • Run tests: pytest --cov")


if __name__ == "__main__":
    asyncio.run(main())
