"""Evaluate command - Evaluate model performance with backtesting."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from price_stradamus.config.constants import FEATURE_WARMUP_PERIOD
from price_stradamus.config.settings import settings
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.stateful_features import StatefulFeatureEngineer
from price_stradamus.data.preprocessor import DataPreprocessor
from price_stradamus.evaluation.backtester import Backtester
from price_stradamus.evaluation.metrics import MetricsCalculator

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


def evaluate(
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model name (for training from scratch)"
    ),
    model_path: str | None = typer.Option(
        None, "--model-path", "-p", help="Path to saved model file"
    ),
    symbol: str = typer.Option("BTCUSDT", "--symbol", "-s", help="Trading symbol"),
    timeframe: str = typer.Option("1m", "--timeframe", "-t", help="Timeframe"),
    method: str = typer.Option(
        "expanding", "--method", help="Backtest method (expanding, walk-forward)"
    ),
    train_start: str | None = typer.Option(
        None, "--train-start", help="Training start datetime (e.g., '2025-08-01 00:00')"
    ),
    train_end: str | None = typer.Option(
        None, "--train-end", help="Training end datetime (e.g., '2025-11-03 03:00')"
    ),
    test_start: str | None = typer.Option(
        None, "--test-start", help="Test start datetime (e.g., '2025-11-03 03:01')"
    ),
    test_end: str | None = typer.Option(
        None, "--test-end", help="Test end datetime (e.g., '2025-11-05 23:59')"
    ),
    plot: bool = typer.Option(
        False, "--plot", help="Show interactive chart (opens in browser)"
    ),
    save_chart: str | None = typer.Option(
        None, "--save-chart", help="Save chart to HTML file"
    ),
) -> None:
    """Evaluate model performance with backtesting.

    Use either --model (train from scratch) OR --model-path (load saved model).

    This command:
    1. Loads or trains a model
    2. Runs backtesting on historical data (optionally filtered by datetime windows)
    3. Calculates performance metrics (MAE, RMSE, MAPE, directional accuracy)
    4. Optionally visualizes results

    Examples:
        # Evaluate a saved model (percentage-based split)
        price-stradamus evaluate --model-path models/xgboost_model.pkl

        # Evaluate with specific datetime windows
        price-stradamus evaluate --model-path models/nbeats_model.pkl \
            --train-start "2025-08-01 00:00" --train-end "2025-11-03 03:00" \
            --test-start "2025-11-03 03:01" --test-end "2025-11-05 23:59"

        # Train and evaluate with datetime windows
        price-stradamus evaluate --model xgboost \
            --train-start "2025-08-01 00:00" --train-end "2025-11-03 00:00" \
            --test-start "2025-11-03 00:01" --test-end "2025-11-05 23:59" --plot
    """

    async def _evaluate():
        # Validate arguments
        if model is None and model_path is None:
            console.print(
                "[red]✗ Error: Must provide either --model or --model-path[/red]"
            )
            raise typer.Exit(1)

        if model is not None and model_path is not None:
            console.print(
                "[red]✗ Error: Cannot use both --model and --model-path[/red]"
            )
            console.print(
                "Use --model to train from scratch OR --model-path to load a saved model"
            )
            raise typer.Exit(1)

        try:
            # Parse datetime windows if provided
            train_start_dt: datetime | None = None
            train_end_dt: datetime | None = None
            test_start_dt: datetime | None = None
            test_end_dt: datetime | None = None

            if train_start or train_end or test_start or test_end:
                # Parse datetime strings
                date_formats = [
                    "%Y-%m-%d %H:%M",
                    "%Y-%m-%d %H:%M:%S",
                    "%d/%m/%Y %H:%M",
                    "%d/%m/%Y %H:%M:%S",
                ]

                def parse_datetime(date_str: str | None, param_name: str) -> datetime | None:
                    if not date_str:
                        return None
                    for fmt in date_formats:
                        try:
                            # Parse and make UTC-aware to match database timestamps
                            dt = datetime.strptime(date_str, fmt)
                            return dt.replace(tzinfo=timezone.utc)
                        except ValueError:
                            continue
                    console.print(
                        f"[red]✗ Invalid {param_name} format: {date_str}[/red]"
                    )
                    console.print(
                        "[red]  Use format: 'YYYY-MM-DD HH:MM' or 'DD/MM/YYYY HH:MM'[/red]"
                    )
                    raise typer.Exit(1)

                train_start_dt = parse_datetime(train_start, "--train-start")
                train_end_dt = parse_datetime(train_end, "--train-end")
                test_start_dt = parse_datetime(test_start, "--test-start")
                test_end_dt = parse_datetime(test_end, "--test-end")

                # Validate datetime windows
                if train_start_dt and train_end_dt and train_start_dt >= train_end_dt:
                    console.print(
                        "[red]✗ Training start must be before training end[/red]"
                    )
                    raise typer.Exit(1)

                if test_start_dt and test_end_dt and test_start_dt >= test_end_dt:
                    console.print(
                        "[red]✗ Test start must be before test end[/red]"
                    )
                    raise typer.Exit(1)

                if train_end_dt and test_start_dt and test_start_dt <= train_end_dt:
                    console.print(
                        "[red]✗ Data leakage! Test start must be after training end[/red]"
                    )
                    raise typer.Exit(1)

                console.print("\n[yellow]Using datetime-based windows:[/yellow]")
                if train_start_dt and train_end_dt:
                    console.print(f"  Train: {train_start_dt} to {train_end_dt}")
                if test_start_dt and test_end_dt:
                    console.print(f"  Test:  {test_start_dt} to {test_end_dt}")

            # Determine evaluation mode
            if model_path:
                # Load saved model mode
                model_path_obj = Path(model_path)
                if not model_path_obj.exists():
                    console.print(f"[red]✗ Model file not found: {model_path}[/red]")
                    raise typer.Exit(1)

                # Extract model name from filename
                model_name = model_path_obj.stem.replace("_model", "")
                console.print(
                    f"[bold cyan]Evaluating saved model: {model_path}[/bold cyan]"
                )
            else:
                # Train from scratch mode
                model_name = model
                console.print(
                    f"[bold cyan]Evaluating {model_name} model (training from scratch)...[/bold cyan]"
                )

            # Load data
            console.print(f"\nLoading {symbol} {timeframe} data...")
            db = DatabaseManager(settings.database_url)
            await db.initialize()

            # Determine data fetch window
            if train_start_dt or train_end_dt or test_start_dt or test_end_dt:
                # Fetch data covering all windows
                fetch_start = train_start_dt or test_start_dt
                fetch_end = test_end_dt or train_end_dt

                # Add buffer for technical indicators (need historical context)
                # If only test window specified, add buffer before test_start
                if fetch_start:
                    fetch_start = fetch_start - timedelta(days=30)
                    console.print(
                        f"[cyan]Fetching data from {fetch_start} to {fetch_end} (includes 30-day buffer)[/cyan]"
                    )
                else:
                    console.print(
                        f"[cyan]Fetching all data up to {fetch_end}[/cyan]"
                    )

                df = await db.get_ohlcv(
                    symbol, timeframe, start_date=fetch_start, end_date=fetch_end
                )
            else:
                # Use all available data
                df = await db.get_ohlcv(symbol, timeframe)

            await db.close()

            if df.empty:
                console.print("[red]✗ No data found. Run 'fetch' command first.[/red]")
                raise typer.Exit(1)

            console.print(f"[green]✓ Loaded {len(df)} candles[/green]")

            # Preprocess and create time series
            console.print("Preprocessing data...")
            preprocessor = DataPreprocessor(normalization_method="minmax")
            df = preprocessor.validate_ohlcv(df)
            df = preprocessor.handle_missing_values(df)

            console.print("Generating features...")
            engineer = StatefulFeatureEngineer(warmup_period=FEATURE_WARMUP_PERIOD)
            # Note: Using fit_transform on all data here for simplicity
            # TODO: For proper train/test split, should split raw data first
            df_features = engineer.fit_transform(df)

            # Ensure timestamp is datetime for filtering
            df_features["timestamp"] = pd.to_datetime(df_features["timestamp"], utc=True)

            ts = engineer.to_darts_timeseries(df_features, value_cols=["close"])

            # Create or load model
            if model_path:
                # Load saved model
                console.print(f"Loading model from {model_path}...")

                if not ModelRegistry.is_registered(model_name):
                    console.print(f"[red]✗ Unknown model type: {model_name}[/red]")
                    console.print(
                        f"Available models: {', '.join(ModelRegistry.list_models())}"
                    )
                    raise typer.Exit(1)

                model_instance = ModelRegistry.create_model(model_name)
                model_instance.load(model_path_obj)
                console.print(f"[green]✓ Model loaded: {model_instance.name}[/green]")

                # Get model dimensions
                output_length = model_instance.output_chunk_length
                input_length = model_instance.input_chunk_length

                # Filter data for test window if specified
                if test_start_dt and test_end_dt:
                    console.print(
                        f"\n[bold yellow]Evaluating on test window: {test_start_dt} to {test_end_dt}[/bold yellow]"
                    )

                    # Filter test data
                    test_mask = (df_features["timestamp"] >= test_start_dt) & (
                        df_features["timestamp"] <= test_end_dt
                    )
                    test_df = cast(pd.DataFrame, df_features[test_mask].copy())

                    if len(test_df) == 0:
                        console.print(
                            "[red]✗ No test data found in specified window[/red]"
                        )
                        raise typer.Exit(1)

                    # Use data up to test start for training context
                    train_mask = df_features["timestamp"] < test_start_dt
                    train_df = cast(pd.DataFrame, df_features[train_mask].copy())

                    if len(train_df) < input_length:
                        console.print(
                            f"[red]✗ Not enough training data ({len(train_df)} candles). "
                            f"Model needs at least {input_length} candles.[/red]"
                        )
                        raise typer.Exit(1)

                    # Create combined time series (train + test)
                    combined_df = pd.concat([train_df, test_df], ignore_index=True)
                    ts_eval = engineer.to_darts_timeseries(combined_df, value_cols=["close"])

                    console.print(
                        f"[green]✓ Using {len(train_df)} candles for context, {len(test_df)} for testing[/green]"
                    )

                    # Debug: Show the boundary between train and test
                    console.print("\n[cyan]Debug Information:[/cyan]")
                    console.print(f"[cyan]  Train data: {len(train_df)} candles[/cyan]")
                    console.print(f"[cyan]    First timestamp: {train_df['timestamp'].values[0]}[/cyan]")
                    console.print(f"[cyan]    Last timestamp:  {train_df['timestamp'].values[-1]}[/cyan]")
                    console.print(f"[cyan]  Test data: {len(test_df)} candles[/cyan]")
                    console.print(f"[cyan]    First timestamp: {test_df['timestamp'].values[0]}[/cyan]")
                    console.print(f"[cyan]    Last timestamp:  {test_df['timestamp'].values[-1]}[/cyan]")
                    console.print(f"[cyan]  Model input_chunk_length: {input_length}[/cyan]")
                    console.print(f"[cyan]  Combined TimeSeries length: {len(ts_eval)}[/cyan]")
                    console.print(f"[cyan]    First timestamp: {ts_eval.time_index[0]}[/cyan]")
                    console.print(f"[cyan]    Timestamp at index {len(train_df)}: {ts_eval.time_index[len(train_df)]}[/cyan]")
                    console.print(f"[cyan]    Last timestamp:  {ts_eval.time_index[-1]}[/cyan]")

                    # Check if model needs more historical context
                    if len(train_df) < input_length:
                        console.print(
                            f"\n[red]✗ Error: Model needs {input_length} historical candles, "
                            f"but only {len(train_df)} available before test period![/red]"
                        )
                        console.print(
                            "[yellow]Tip: Specify --train-start earlier or ensure database has enough historical data[/yellow]"
                        )
                        raise typer.Exit(1)

                    # Set start index to where test data begins
                    start_idx = len(train_df)
                    console.print(f"\n[bold cyan]Prediction will start at index {start_idx}[/bold cyan]")
                    console.print(f"[bold cyan]  This corresponds to timestamp: {ts_eval.time_index[start_idx]}[/bold cyan]")
                else:
                    # Use all available data
                    ts_eval = ts
                    console.print(
                        f"\n[bold yellow]Evaluating pre-trained model on {len(ts_eval)} timesteps...[/bold yellow]"
                    )

                    # Start predictions from input_length for full dataset evaluation
                    start_idx = input_length

                # Generate predictions iteratively
                predictions_list = []
                actuals_list = []
                timestamps_list = []

                # Start predicting from the appropriate index
                current_idx = start_idx
                while current_idx + output_length <= len(ts_eval):
                    # Use historical data up to current point
                    historical = ts_eval[:current_idx]

                    # Predict next N steps
                    pred = model_instance.predict(n=output_length, series=historical)

                    # Get actual values
                    actual = ts_eval[current_idx : current_idx + output_length]

                    predictions_list.append(pred.values().flatten())
                    actuals_list.append(actual.values().flatten())
                    timestamps_list.append(actual.time_index.to_numpy())

                    # Move forward by output_length
                    current_idx += output_length

                console.print(
                    f"[green]✓ Generated {len(predictions_list)} prediction windows[/green]"
                )

                # Check if we covered the full test range
                if test_start_dt and test_end_dt and len(predictions_list) > 0:
                    last_prediction_time = pd.Timestamp(timestamps_list[-1][-1])
                    if last_prediction_time < test_end_dt:
                        uncovered_minutes = int((test_end_dt - last_prediction_time).total_seconds() / 60)
                        if uncovered_minutes > 0:
                            console.print(
                                f"[yellow]Note: Last {uncovered_minutes} minutes not predicted "
                                f"(model predicts in {output_length}-step chunks)[/yellow]"
                            )

                # Flatten arrays
                all_predictions = np.concatenate(predictions_list)
                all_actuals = np.concatenate(actuals_list)

                # Calculate metrics
                metrics = MetricsCalculator.calculate_all(all_actuals, all_predictions)

                # Display results
                console.print("\n[bold green]Evaluation Results:[/bold green]")

                table = Table(title="Performance Metrics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("MAE", f"{metrics['mae']:.2f}")
                table.add_row("RMSE", f"{metrics['rmse']:.2f}")
                table.add_row("MAPE", f"{metrics['mape']:.2f}%")
                table.add_row(
                    "Directional Accuracy", f"{metrics['directional_accuracy']:.2f}%"
                )
                table.add_row("Total Predictions", str(len(all_predictions)))

                console.print(table)

                # Generate visualization if requested
                if plot or save_chart:
                    console.print(
                        "\n[bold yellow]Generating interactive chart...[/bold yellow]"
                    )

                    from price_stradamus.visualization.charts import (
                        PriceChartVisualizer,
                    )

                    # Flatten timestamps
                    all_timestamps = np.concatenate(timestamps_list)

                    # Create chart
                    chart_path = Path(save_chart) if save_chart else None
                    chart_title = f"{(model_name or 'Model').upper()} Evaluation - {symbol} {timeframe}"
                    PriceChartVisualizer.plot_evaluation_results(
                        df=df,
                        predictions=all_predictions,
                        actuals=all_actuals,
                        timestamps=all_timestamps,
                        metrics=metrics,
                        title=chart_title,
                        save_path=chart_path,
                        show=plot,
                    )

                    if save_chart:
                        console.print(f"[green]✓ Chart saved to {save_chart}[/green]")
                    if plot:
                        console.print("[green]✓ Chart opened in browser[/green]")

            else:
                # Create new model for training
                console.print(f"Creating {model_name} model...")
                model_instance = ModelRegistry.create_model(
                    model_name, input_chunk_length=60, output_chunk_length=5
                )

                # For new models, use train/val/test split
                backtester = Backtester(
                    model_instance, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
                )

                # Run backtest
                console.print(
                    f"\n[bold yellow]Running {method} backtest...[/bold yellow]"
                )

                if method == "expanding":
                    results = backtester.run_expanding_window(ts)
                else:
                    console.print("[red]Walk-forward method not yet implemented[/red]")
                    raise typer.Exit(1)

                # Display results
                console.print("\n[bold green]Backtest Results:[/bold green]")

                metrics_df = backtester.get_summary(results)

                # Create rich table
                table = Table(title="Performance Metrics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                for _, row in metrics_df.iterrows():
                    table.add_row(str(row["Metric"]), str(row["Value"]))

                console.print(table)

        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            logger.exception("Evaluate command failed")
            raise typer.Exit(1) from e

    asyncio.run(_evaluate())
