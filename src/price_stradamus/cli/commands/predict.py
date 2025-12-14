"""Predict command - Make predictions with trained models."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pandas as pd
import typer
from darts import TimeSeries
from loguru import logger
from rich.console import Console
from rich.table import Table

from price_stradamus.config.constants import FEATURE_WARMUP_PERIOD
from price_stradamus.config.settings import settings
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.preprocessor import DataPreprocessor
from price_stradamus.data.stateful_features import StatefulFeatureEngineer

# Import models to register them
from price_stradamus.models.classical.arima import ARIMAModel  # noqa: F401
from price_stradamus.models.classical.prophet import ProphetModel  # noqa: F401
from price_stradamus.models.ml.xgboost import XGBoostModel  # noqa: F401
from price_stradamus.models.neural.lstm import LSTMModel  # noqa: F401
from price_stradamus.models.neural.nbeats import NBEATSModel  # noqa: F401
from price_stradamus.models.neural.tcn import TCNModel  # noqa: F401
from price_stradamus.models.neural.tft import TFTModel  # noqa: F401
from price_stradamus.models.registry import ModelRegistry

# Create console for output
console = Console()


def predict(
    model_path: str = typer.Option(..., "--model", "-m", help="Path to trained model"),
    steps: int = typer.Option(5, "--steps", "-s", help="Number of steps to predict"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Trading symbol"),
    timeframe: str = typer.Option("1m", "--timeframe", help="Timeframe"),
    start_date: str | None = typer.Option(
        None,
        "--start-date",
        help="Start date for predictions (format: 'DD/MM/YYYY HH:MM' or 'YYYY-MM-DD HH:MM'). "
        "If provided, makes predictions from this historical date for backtesting.",
    ),
    lookback_days: int = typer.Option(
        30,
        "--lookback-days",
        help="Days of historical data to fetch before start-date (for technical indicators). "
        "Default: 30 days to match evaluation context.",
    ),
    plot: bool = typer.Option(
        False, "--plot", help="Show interactive chart (opens in browser)"
    ),
    save_chart: str | None = typer.Option(
        None, "--save-chart", help="Save chart to HTML file"
    ),
) -> None:
    """Make predictions with a trained model.

    This command:
    1. Loads a trained model from disk
    2. Fetches recent historical data
    3. Generates features
    4. Makes multi-step predictions
    5. Optionally visualizes results

    When --start-date is provided, the model will:
    - Only use data BEFORE the start date (no data leakage)
    - Make predictions starting from that date
    - Show actual prices (if available) for comparison/validation

    Examples:
        # Make 5-step predictions from latest data
        price-stradamus predict --model models/nbeats_model.pkl

        # Predict 10 steps and show chart
        price-stradamus predict -m models/xgboost_model.pkl -s 10 --plot

        # Backtest predictions from a historical date
        price-stradamus predict -m models/nbeats_model.pkl --start-date "05/11/2025 06:25" --plot

        # Save predictions chart to file
        price-stradamus predict -m models/lstm_model.pkl --save-chart results/predictions.html
    """

    async def _predict():
        console.print(f"[bold cyan]Making predictions with {model_path}...[/bold cyan]")

        # Parse start_date if provided
        prediction_start_dt: datetime | None = None
        if start_date:
            try:
                # Try multiple date formats
                for fmt in [
                    "%d/%m/%Y %H:%M",
                    "%Y-%m-%d %H:%M",
                    "%d/%m/%Y %H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                ]:
                    try:
                        prediction_start_dt = datetime.strptime(start_date, fmt)
                        break
                    except ValueError:
                        continue

                if prediction_start_dt is None:
                    raise ValueError("Invalid date format")

                console.print(
                    f"[yellow]🕐 Backtesting mode: Making predictions from {prediction_start_dt}[/yellow]"
                )
                console.print(
                    "[yellow]⚠️  Model will only see data BEFORE this date (no data leakage)[/yellow]"
                )
            except ValueError:
                console.print(
                    "[red]✗ Invalid date format. Use 'DD/MM/YYYY HH:MM' or 'YYYY-MM-DD HH:MM'[/red]"
                )
                console.print(
                    "[red]  Example: '05/11/2025 06:25' or '2025-11-05 06:25'[/red]"
                )
                raise typer.Exit(1) from None

        # Check if model file exists
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            console.print(f"[red]✗ Model file not found: {model_path}[/red]")
            raise typer.Exit(1) from None

        # Load data
        console.print(f"Loading {symbol} {timeframe} data...")
        db = DatabaseManager(settings.database_url)
        await db.initialize()

        # Get data based on whether we're backtesting or not
        if prediction_start_dt:
            # Backtesting mode: fetch sufficient historical context
            # Need enough history for technical indicators (SMA, RSI, etc.) to be accurate
            # Using configurable lookback (default 30 days) to match evaluation context
            start_fetch = prediction_start_dt - timedelta(days=lookback_days)
            end_fetch = prediction_start_dt + timedelta(days=1)
            console.print(
                f"[cyan]📊 Fetching data window: {start_fetch} to {end_fetch}[/cyan]"
            )
            console.print(
                f"[cyan]   (Using {lookback_days}-day lookback for technical indicators)[/cyan]"
            )
            df = await db.get_ohlcv(
                symbol, timeframe, start_date=start_fetch, end_date=end_fetch
            )
        else:
            # Normal mode: get last 1000 candles for context
            df = await db.get_ohlcv(symbol, timeframe)

        if df.empty:
            console.print("[red]✗ No data found. Run 'fetch' command first.[/red]")
            await db.close()
            raise typer.Exit(1) from None

        console.print(f"[green]✓ Loaded {len(df)} candles[/green]")

        # If backtesting, validate that start_date exists in data
        if prediction_start_dt:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            # Convert entire series to timezone-naive UTC if needed (for compatibility)
            if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)

            # Now get min/max (will be timezone-naive)
            min_timestamp = df["timestamp"].min()
            max_timestamp = df["timestamp"].max()

            if prediction_start_dt < min_timestamp:
                console.print(
                    f"[red]✗ Start date {prediction_start_dt} is before available data "
                    f"(earliest: {min_timestamp})[/red]"
                )
                await db.close()
                raise typer.Exit(1) from None

            if prediction_start_dt >= max_timestamp:
                console.print(
                    f"[red]✗ Start date {prediction_start_dt} is after available data "
                    f"(latest: {max_timestamp})[/red]"
                )
                await db.close()
                raise typer.Exit(1) from None

        # Preprocess and generate features
        console.print("Preprocessing data...")
        preprocessor = DataPreprocessor(normalization_method="minmax")
        df = preprocessor.validate_ohlcv(df)
        df = preprocessor.handle_missing_values(df)

        console.print("Generating features...")
        engineer = StatefulFeatureEngineer(warmup_period=FEATURE_WARMUP_PERIOD)
        df_features = engineer.fit_transform(df)

        # Convert to time series
        console.print("Creating time series...")
        ts = cast(TimeSeries, engineer.to_darts_timeseries(df_features, value_cols=["close"]))

        # If backtesting, slice data to prevent data leakage
        historical_cutoff_ts = ts  # For visualization
        if prediction_start_dt:
            # Find the index where prediction should start
            cutoff_idx = None
            prediction_start_ts = pd.Timestamp(prediction_start_dt)

            for idx, timestamp in enumerate(ts.time_index):
                ts_value = pd.Timestamp(timestamp)

                # Convert to timezone-naive if needed for comparison
                if ts_value.tz is not None:
                    ts_value = ts_value.tz_localize(None)

                # Check for valid timestamps (not NaT) before comparing
                # Type narrowing: pd.notna() ensures neither value is NaT before comparison
                if (
                    pd.notna(ts_value)
                    and pd.notna(prediction_start_ts)
                    and ts_value >= prediction_start_ts  # type: ignore[operator]
                ):
                    cutoff_idx = idx
                    break

            if cutoff_idx is None or cutoff_idx == 0:
                console.print(
                    "[red]✗ Could not find valid cutoff point in time series[/red]"
                )
                await db.close()
                raise typer.Exit(1) from None

            # IMPORTANT: Model only sees data BEFORE cutoff (no data leakage)
            ts = ts[:cutoff_idx]
            console.print(
                f"[green]✓ Using {len(ts)} candles before {prediction_start_dt} for prediction[/green]"
            )
            console.print(
                f"[yellow]  Model will predict {steps} steps starting from {prediction_start_dt}[/yellow]"
            )

        # Load the trained model
        console.print(f"Loading model from {model_path}...")

        # Extract model name from filename (e.g., "xgboost" from "xgboost_model.pkl")
        model_name = model_path_obj.stem.replace("_model", "")

        # Check if model is registered
        if not ModelRegistry.is_registered(model_name):
            console.print(f"[red]✗ Unknown model type: {model_name}[/red]")
            console.print(f"Available models: {', '.join(ModelRegistry.list_models())}")
            raise typer.Exit(1) from None

        # Create model instance and load weights
        model_instance = ModelRegistry.create_model(model_name)
        model_instance.load(model_path_obj)

        console.print(f"[green]✓ Model loaded: {model_instance.name}[/green]")

        # Check if model can predict requested steps
        output_chunk_length = getattr(model_instance, "output_chunk_length", None)
        if output_chunk_length and steps > output_chunk_length:
            console.print(
                f"[yellow]⚠️  Model output_chunk_length={output_chunk_length}, but {steps} steps requested[/yellow]"
            )
            console.print(
                "[yellow]  Will use model's multi-step prediction capability...[/yellow]"
            )

        # Make predictions
        console.print(f"\nMaking {steps}-step predictions...")
        try:
            predictions = cast(TimeSeries, model_instance.predict(n=steps, series=ts))
        except Exception as e:
            # If prediction fails, it might be due to steps > output_chunk_length
            if output_chunk_length and steps > output_chunk_length:
                console.print(
                    f"[red]✗ Model cannot predict {steps} steps (max: {output_chunk_length})[/red]"
                )
                console.print(
                    f"[yellow]Try reducing --steps to {output_chunk_length} or less[/yellow]"
                )
            raise e

        # Display predictions
        console.print("\n[bold green]Predictions:[/bold green]")
        pred_values = predictions.values().flatten()

        # Warn if we got fewer predictions than requested
        actual_steps = len(pred_values)  # Actual number of predictions received
        if actual_steps < steps:
            console.print(
                f"[yellow]⚠️  Model returned {actual_steps} predictions (requested {steps})[/yellow]"
            )
            console.print(
                f"[yellow]  This model's output_chunk_length is {output_chunk_length}[/yellow]"
            )

        # Extract actual values if backtesting (data already in DB)
        actual_values = None
        actual_candles_df: pd.DataFrame | None = None
        if prediction_start_dt and historical_cutoff_ts is not None:
            # Find cutoff index in original time series
            cutoff_idx = len(ts)  # Where predictions start

            # Check if we have enough actual data for comparison
            if cutoff_idx + actual_steps <= len(historical_cutoff_ts):
                actual_values = (
                    historical_cutoff_ts[cutoff_idx : cutoff_idx + actual_steps]
                    .values()
                    .flatten()
                )

                # Also extract the OHLCV candles for visualization
                # Find corresponding rows in df_features
                pred_start_time = historical_cutoff_ts.time_index[cutoff_idx]
                pred_end_time = historical_cutoff_ts.time_index[
                    min(cutoff_idx + actual_steps - 1, len(historical_cutoff_ts) - 1)
                ]
                candles_filtered = df_features[
                    (df_features["timestamp"] >= pred_start_time)
                    & (df_features["timestamp"] <= pred_end_time)
                ]
                if isinstance(candles_filtered, pd.DataFrame):
                    actual_candles_df = candles_filtered.copy()

                console.print(
                    f"[green]✓ Found {len(actual_values)} actual candles for comparison[/green]"
                )
            else:
                available_steps = len(historical_cutoff_ts) - cutoff_idx
                if available_steps > 0:
                    actual_values = (
                        historical_cutoff_ts[cutoff_idx : cutoff_idx + available_steps]
                        .values()
                        .flatten()
                    )

                    # Extract available OHLCV candles
                    pred_start_time = historical_cutoff_ts.time_index[cutoff_idx]
                    pred_end_time = historical_cutoff_ts.time_index[
                        min(
                            cutoff_idx + available_steps - 1,
                            len(historical_cutoff_ts) - 1,
                        )
                    ]
                    candles_filtered = df_features[
                        (df_features["timestamp"] >= pred_start_time)
                        & (df_features["timestamp"] <= pred_end_time)
                    ]
                    if isinstance(candles_filtered, pd.DataFrame):
                        actual_candles_df = candles_filtered.copy()

                    console.print(
                        f"[yellow]⚠️  Only {available_steps} actual candles available (requested {actual_steps})[/yellow]"
                    )

        # Build comparison table
        table = Table(title=f"{actual_steps}-Step Ahead Predictions")
        table.add_column("Step", style="cyan")
        table.add_column("Predicted Price", style="green")
        if actual_values is not None:
            table.add_column("Actual Price", style="yellow")
            table.add_column("Error", style="red")
        else:
            table.add_column("Change from Last", style="yellow")

        last_actual = float(ts.values()[-1][0])
        for i, pred_price in enumerate(pred_values[:actual_steps], 1):
            if actual_values is not None and i - 1 < len(actual_values):
                actual_price = actual_values[i - 1]
                error = pred_price - actual_price
                error_pct = (error / actual_price) * 100
                table.add_row(
                    str(i),
                    f"${pred_price:.2f}",
                    f"${actual_price:.2f}",
                    f"{error:+.2f} ({error_pct:+.2f}%)",
                )
            else:
                change = pred_price - last_actual
                change_pct = (change / last_actual) * 100
                table.add_row(
                    str(i), f"${pred_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)"
                )
            last_actual = pred_price

        console.print(table)

        # Summary
        first_pred = pred_values[0]
        last_pred = pred_values[-1]
        current_price = float(ts.values()[-1][0])
        total_change = ((last_pred - current_price) / current_price) * 100

        console.print("\n[bold]Summary:[/bold]")
        console.print(f"Current price: ${current_price:.2f}")
        console.print(f"First prediction (1-step): ${first_pred:.2f}")
        console.print(f"Last prediction ({actual_steps}-step): ${last_pred:.2f}")
        console.print(f"Total predicted change: {total_change:+.2f}%")

        # Show comparison metrics if backtesting
        if actual_values is not None:
            from price_stradamus.evaluation.metrics import MetricsCalculator

            # Only calculate metrics for steps where we have actuals
            min_len = min(len(pred_values), len(actual_values))
            metrics = MetricsCalculator.calculate_all(
                actual_values[:min_len], pred_values[:min_len]
            )

            console.print("\n[bold cyan]Backtesting Metrics:[/bold cyan]")
            console.print(f"MAE: {metrics['mae']:.2f}")
            console.print(f"RMSE: {metrics['rmse']:.2f}")
            console.print(f"MAPE: {metrics['mape']:.2f}%")
            console.print(
                f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%"
            )

        # Generate visualization if requested
        if plot or save_chart:
            console.print(
                "\n[bold yellow]Generating interactive chart...[/bold yellow]"
            )

            from price_stradamus.visualization.charts import PriceChartVisualizer

            # Get prediction timestamps
            pred_timestamps = predictions.time_index

            # Prepare chart title
            if prediction_start_dt:
                chart_title = (
                    f"{model_instance.name.upper()} - Backtesting from {prediction_start_dt.strftime('%Y-%m-%d %H:%M')} "
                    f"({actual_steps} steps) - {symbol} {timeframe}"
                )
            else:
                chart_title = f"{model_instance.name.upper()} - {actual_steps}-Step Predictions for {symbol} {timeframe}"

            # Create chart with actual values if available
            chart_path = Path(save_chart) if save_chart else None
            PriceChartVisualizer.plot_predictions(
                historical_df=df_features,
                predictions=pred_values,
                timestamps=pred_timestamps,
                actual_values=actual_values,  # Will show actuals line if backtesting
                actual_candles_df=actual_candles_df,  # Will show actual candlesticks if backtesting
                title=chart_title,
                save_path=chart_path,
                show=plot,
            )

            if save_chart:
                console.print(f"[green]✓ Chart saved to {save_chart}[/green]")
            if plot:
                console.print("[green]✓ Chart opened in browser[/green]")

        # Close database connection
        await db.close()

    try:
        asyncio.run(_predict())
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        logger.exception("Predict command failed")
        raise typer.Exit(1) from e
