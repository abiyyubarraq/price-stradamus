"""CLI commands for TimeWindow-based training and evaluation.

These commands make it easy to specify training/evaluation boundaries
to prevent data leakage.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import cast

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
from price_stradamus.evaluation.metrics import MetricsCalculator
from price_stradamus.evaluation.walk_forward import DataLeakageChecker
from price_stradamus.models.registry import ModelRegistry

console = Console()


def register_window_commands(app: typer.Typer) -> None:
    """Register window-based commands to the main app.

    Args:
        app: Main Typer application instance
    """
    app.command(name="train-window")(train_window)
    app.command(name="eval-window")(eval_window)


def train_window(
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g., nbeats)"),
    train_start: str = typer.Option(
        ...,
        "--train-start",
        help="Training start date (YYYY-MM-DD or YYYY-MM-DD HH:MM)",
    ),
    train_end: str = typer.Option(
        ..., "--train-end", help="Training end date (YYYY-MM-DD or YYYY-MM-DD HH:MM)"
    ),
    symbol: str = typer.Option("BTCUSDT", "--symbol", "-s", help="Trading symbol"),
    timeframe: str = typer.Option("1m", "--timeframe", "-t", help="Timeframe"),
    input_length: int = typer.Option(
        60, "--input-length", "-i", help="Input sequence length"
    ),
    output_length: int = typer.Option(
        5, "--output-length", "-o", help="Output sequence length"
    ),
    epochs: int = typer.Option(100, "--epochs", "-e", help="Number of training epochs"),
    save_path: str | None = typer.Option(
        None, "--save", help="Path to save trained model"
    ),
) -> None:
    """Train a model using a specific time window.

    This command ensures NO DATA LEAKAGE by:
    1. Training ONLY on data within [train-start, train-end]
    2. Automatically validating temporal boundaries
    3. Warning if there's any overlap

    Examples:
        # Train on August-October data
        price-stradamus train-window \\
            --model nbeats \\
            --train-start "2025-08-01" \\
            --train-end "2025-10-31"

        # Train on specific time period with hours
        price-stradamus train-window \\
            --model xgboost \\
            --train-start "2025-08-01 00:00" \\
            --train-end "2025-10-31 23:59" \\
            --save models/xgboost_aug_oct.pkl
    """

    async def _train():
        console.print(
            "[bold cyan]Training with Time Window (No Data Leakage)[/bold cyan]\n"
        )

        try:
            # Parse dates
            train_start_dt = _parse_date(train_start)
            train_end_dt = _parse_date(train_end)

            console.print("[bold]Training Window:[/bold]")
            console.print(f"  Start: {train_start_dt}")
            console.print(f"  End:   {train_end_dt}")
            console.print()

            # Validate dates
            if train_start_dt >= train_end_dt:
                console.print(
                    "[red][ERROR] Error: train-start must be before train-end[/red]"
                )
                raise typer.Exit(1)

            # Load data from database
            console.print(f"Loading {symbol} {timeframe} data from database...")
            db = DatabaseManager(settings.database_url)
            await db.initialize()

            # Fetch data in the training window
            df = await db.get_ohlcv(
                symbol, timeframe, start_date=train_start_dt, end_date=train_end_dt
            )

            await db.close()

            if df.empty:
                console.print(
                    f"[red][ERROR] No data found for {symbol} {timeframe} in specified window.[/red]"
                )
                console.print(
                    "[yellow]Run 'price-stradamus fetch' to download data first.[/yellow]"
                )
                raise typer.Exit(1)

            console.print(f"[green][OK] Loaded {len(df)} candles[/green]")

            # Preprocess data
            console.print("Preprocessing data...")
            preprocessor = DataPreprocessor()
            df = preprocessor.validate_ohlcv(df)
            df = preprocessor.handle_missing_values(df)

            # Generate features
            console.print("Generating technical indicators...")
            engineer = StatefulFeatureEngineer(warmup_period=FEATURE_WARMUP_PERIOD)
            df_features = engineer.fit_transform(df)

            console.print(
                f"[green][OK] Generated {len(engineer.feature_list)} features[/green]"
            )

            # Convert to Darts TimeSeries
            console.print("Creating time series...")
            ts = cast(TimeSeries, engineer.to_darts_timeseries(df_features, value_cols=["close"]))

            console.print(
                f"[green][OK] Created time series with {len(ts)} timesteps[/green]"
            )

            # Split data (70% train, 30% val within the window)
            train_size = int(len(ts) * 0.7)

            train_ts = ts[:train_size]
            val_ts = ts[train_size:]

            console.print(
                f"\nSplit within window: train={len(train_ts)}, val={len(val_ts)}"
            )

            # ✅ Check for data leakage (should never happen with our window)
            console.print("\n🔍 Verifying no data leakage...")
            checker = DataLeakageChecker()
            result = checker.check_temporal_leakage(train_ts, val_ts)

            if result["has_leakage"]:
                console.print("[red][ERROR] UNEXPECTED: Data leakage detected![/red]")
                raise typer.Exit(1)

            console.print("[green][OK] No data leakage - safe to proceed[/green]\n")

            # Create and train model
            console.print(f"Creating {model} model...")
            model_instance = ModelRegistry.create_model(
                model,
                input_chunk_length=input_length,
                output_chunk_length=output_length,
                n_epochs=epochs,
            )

            console.print(f"[green][OK] Model created: {model_instance}[/green]")

            console.print(
                "\n[bold yellow]Training model (this may take a while)...[/bold yellow]"
            )
            console.print(
                "[dim]Press Ctrl+C to interrupt and save current progress[/dim]"
            )

            try:
                model_instance.fit(train_ts, val_ts)
                console.print("[green][OK] Training complete![/green]")
            except KeyboardInterrupt:
                console.print("\n[yellow]! Training interrupted by user[/yellow]")
                console.print("[yellow][OK] Saving partially trained model...[/yellow]")

            # Save model with default path if not provided
            final_save_path: str
            if save_path is None:
                final_save_path = (
                    f"models/{model_instance.name}_"
                    f"{train_start_dt.strftime('%Y%m%d')}_"
                    f"{train_end_dt.strftime('%Y%m%d')}.pkl"
                )
            else:
                final_save_path = save_path

            save_path_obj = Path(final_save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            model_instance.save(save_path_obj)
            console.print(f"[green][OK] Model saved to {final_save_path}[/green]")

            # Show summary
            console.print("\n[bold]Training Summary:[/bold]")
            console.print(f"  Model: {model_instance.name}")
            console.print(f"  Trained on: {train_start_dt} to {train_end_dt}")
            console.print(f"  Total samples: {len(ts)}")
            console.print(f"  Train samples: {len(train_ts)}")
            console.print(f"  Val samples: {len(val_ts)}")
            console.print(f"  Saved to: {final_save_path}")

        except Exception as e:
            console.print(f"[red][ERROR] Error: {e}[/red]")
            logger.exception("Train window command failed")
            raise typer.Exit(1) from e

    asyncio.run(_train())


def eval_window(
    model_path: str = typer.Option(
        ..., "--model-path", "-m", help="Path to trained model"
    ),
    test_start: str = typer.Option(
        ..., "--test-start", help="Test start date (YYYY-MM-DD or YYYY-MM-DD HH:MM)"
    ),
    test_end: str = typer.Option(
        ..., "--test-end", help="Test end date (YYYY-MM-DD or YYYY-MM-DD HH:MM)"
    ),
    symbol: str = typer.Option("BTCUSDT", "--symbol", "-s", help="Trading symbol"),
    timeframe: str = typer.Option("1m", "--timeframe", "-t", help="Timeframe"),
    check_leakage: bool = typer.Option(
        True,
        "--check-leakage/--no-check-leakage",
        help="Check for data leakage against training window",
    ),
    train_end: str | None = typer.Option(
        None,
        "--train-end",
        help="Training end date (for leakage check). If not provided, assumes test > train.",
    ),
    plot: bool = typer.Option(
        False, "--plot", help="Show interactive chart (opens in browser)"
    ),
    save_chart: str | None = typer.Option(
        None, "--save-chart", help="Save chart to HTML file"
    ),
) -> None:
    """Evaluate a model on a specific time window.

    This command ensures NO DATA LEAKAGE by:
    1. Testing ONLY on data within [test-start, test-end]
    2. Optionally checking that test window is AFTER training window
    3. Warning if there's any temporal overlap

    Examples:
        # Evaluate on November-December data
        price-stradamus eval-window \\
            --model-path models/nbeats_model.pkl \\
            --test-start "2025-11-01" \\
            --test-end "2025-12-04"

        # Evaluate with leakage check
        price-stradamus eval-window \\
            --model-path models/nbeats_aug_oct.pkl \\
            --test-start "2025-11-01" \\
            --test-end "2025-12-04" \\
            --train-end "2025-10-31" \\
            --plot
    """

    async def _evaluate():
        console.print(
            "[bold cyan]Evaluating with Time Window (No Data Leakage)[/bold cyan]\n"
        )

        try:
            # Parse dates
            test_start_dt = _parse_date(test_start)
            test_end_dt = _parse_date(test_end)

            console.print("[bold]Test Window:[/bold]")
            console.print(f"  Start: {test_start_dt}")
            console.print(f"  End:   {test_end_dt}")
            console.print()

            # Validate dates
            if test_start_dt >= test_end_dt:
                console.print(
                    "[red][ERROR] Error: test-start must be before test-end[/red]"
                )
                raise typer.Exit(1)

            # Check model file exists
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                console.print(f"[red][ERROR] Model file not found: {model_path}[/red]")
                raise typer.Exit(1)

            # Check for leakage if train_end is provided
            if check_leakage and train_end:
                train_end_dt = _parse_date(train_end)
                console.print("[bold]Training Window End:[/bold]")
                console.print(f"  {train_end_dt}")
                console.print()

                # ✅ Validate no overlap
                if test_start_dt <= train_end_dt:
                    console.print("[red][ERROR] DATA LEAKAGE DETECTED![/red]")
                    console.print(f"  Training ends:  {train_end_dt}")
                    console.print(f"  Testing starts: {test_start_dt}")
                    console.print(
                        "[red]  Test window overlaps with training window![/red]"
                    )
                    raise typer.Exit(1)

                gap_hours = (test_start_dt - train_end_dt).total_seconds() / 3600
                console.print(
                    f"[green][OK] No data leakage - {gap_hours:.1f} hour gap between train and test[/green]\n"
                )

            # Load data from database
            console.print(f"Loading {symbol} {timeframe} data from database...")
            db = DatabaseManager(settings.database_url)
            await db.initialize()

            # Fetch data in the test window
            df = await db.get_ohlcv(
                symbol, timeframe, start_date=test_start_dt, end_date=test_end_dt
            )

            await db.close()

            if df.empty:
                console.print(
                    f"[red][ERROR] No data found for {symbol} {timeframe} in specified window.[/red]"
                )
                raise typer.Exit(1)

            console.print(f"[green][OK] Loaded {len(df)} candles[/green]")

            # Preprocess and create features
            console.print("Preprocessing data...")
            preprocessor = DataPreprocessor(normalization_method="minmax")
            df = preprocessor.validate_ohlcv(df)
            df = preprocessor.handle_missing_values(df)

            console.print("Generating features...")
            engineer = StatefulFeatureEngineer(warmup_period=FEATURE_WARMUP_PERIOD)
            df_features = engineer.fit_transform(df)
            ts = cast(TimeSeries, engineer.to_darts_timeseries(df_features, value_cols=["close"]))

            console.print(
                f"[green][OK] Created time series with {len(ts)} timesteps[/green]\n"
            )

            # Load model
            console.print(f"Loading model from {model_path}...")

            # Extract model name from filename by matching against registered models
            filename = model_path_obj.stem.replace("_model", "")
            registered_models = ModelRegistry.list_models()

            # Try to find a registered model name in the filename
            model_name = None
            for registered in registered_models:
                if filename.startswith(registered):
                    model_name = registered
                    break

            if model_name is None or not ModelRegistry.is_registered(model_name):
                console.print(
                    f"[red][ERROR] Unknown model type in filename: {filename}[/red]"
                )
                console.print(f"Available models: {', '.join(registered_models)}")
                console.print(
                    "[yellow]Hint: Model filename should start with the model name (e.g., xgboost_..., nbeats_...)[/yellow]"
                )
                raise typer.Exit(1)

            model_instance = ModelRegistry.create_model(model_name)
            model_instance.load(model_path_obj)
            console.print(f"[green][OK] Model loaded: {model_instance.name}[/green]\n")

            # Make predictions
            console.print("Making predictions on test window...")
            output_length = model_instance.output_chunk_length
            input_length = model_instance.input_chunk_length

            predictions_list = []
            actuals_list = []

            # Sliding window predictions
            current_idx = input_length
            while current_idx + output_length <= len(ts):
                historical = ts[:current_idx]
                pred = cast(TimeSeries, model_instance.predict(n=output_length, series=historical))
                actual = ts[current_idx : current_idx + output_length]

                predictions_list.append(pred.values().flatten())
                actuals_list.append(actual.values().flatten())

                current_idx += output_length

            console.print(
                f"[green][OK] Generated {len(predictions_list)} prediction windows[/green]"
            )

            # Flatten and calculate metrics
            import numpy as np

            all_predictions = np.concatenate(predictions_list)
            all_actuals = np.concatenate(actuals_list)

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

            # Summary
            console.print("\n[bold]Evaluation Summary:[/bold]")
            console.print(f"  Model: {model_instance.name}")
            console.print(f"  Tested on: {test_start_dt} to {test_end_dt}")
            console.print(f"  Total samples: {len(ts)}")
            console.print(f"  Predictions: {len(all_predictions)}")

            if check_leakage and train_end:
                console.print(
                    f"  [OK] No data leakage (train ended {train_end_dt}, test started {test_start_dt})"
                )

            # Generate visualization if requested
            if plot or save_chart:
                console.print(
                    "\n[bold yellow]Generating interactive chart...[/bold yellow]"
                )

                from price_stradamus.visualization.charts import PriceChartVisualizer

                # Collect timestamps for each prediction window
                timestamp_arrays = []
                for i in range(input_length, len(ts), output_length):
                    end_idx = min(i + output_length, len(ts))
                    time_slice = ts.time_index[i:end_idx]

                    # Type guard: ensure time_slice is not a scalar
                    if isinstance(time_slice, (int, float)):
                        continue

                    if hasattr(time_slice, "to_numpy"):
                        timestamp_arrays.append(time_slice.to_numpy())  # type: ignore[attr-defined]
                    elif hasattr(time_slice, "values"):
                        timestamp_arrays.append(time_slice.values)  # type: ignore[attr-defined]
                    else:
                        # Last resort: convert to numpy array
                        timestamp_arrays.append(np.array(time_slice))

                timestamps = np.concatenate(timestamp_arrays)[: len(all_predictions)]

                chart_path = Path(save_chart) if save_chart else None
                chart_title = f"{model_instance.name.upper()} Evaluation - {test_start_dt.date()} to {test_end_dt.date()}"

                PriceChartVisualizer.plot_evaluation_results(
                    df=df,
                    predictions=all_predictions,
                    actuals=all_actuals,
                    timestamps=timestamps,
                    metrics=metrics,
                    title=chart_title,
                    save_path=chart_path,
                    show=plot,
                )

                if save_chart:
                    console.print(f"[green][OK] Chart saved to {save_chart}[/green]")
                if plot:
                    console.print("[green][OK] Chart opened in browser[/green]")

        except Exception as e:
            console.print(f"[red][ERROR] Error: {e}[/red]")
            logger.exception("Eval window command failed")
            raise typer.Exit(1) from e

    asyncio.run(_evaluate())


def _parse_date(date_str: str) -> datetime:
    """Parse date string in various formats.

    Args:
        date_str: Date string (YYYY-MM-DD or YYYY-MM-DD HH:MM)

    Returns:
        Parsed datetime object

    Raises:
        ValueError: If date format is invalid
    """
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"Invalid date format: {date_str}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM"
    )
