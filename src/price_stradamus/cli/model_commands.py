"""Model training and prediction CLI commands for Price Stradamus.

This module handles all ML model operations:
- Training models
- Making predictions
- Model evaluation and backtesting
- Model comparison
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from price_stradamus.config.settings import settings
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.features import FeatureEngineer
from price_stradamus.data.preprocessor import DataPreprocessor
from price_stradamus.evaluation.backtester import Backtester
from price_stradamus.evaluation.metrics import MetricsCalculator

# Import models to register them
from price_stradamus.models.classical.arima import ARIMAModel  # noqa: F401
from price_stradamus.models.classical.prophet import ProphetModel  # noqa: F401
from price_stradamus.models.ml.random_forest import RandomForestModel  # noqa: F401
from price_stradamus.models.ml.xgboost import XGBoostModel  # noqa: F401
from price_stradamus.models.neural.lstm import LSTMModel  # noqa: F401
from price_stradamus.models.neural.nbeats import NBEATSModel  # noqa: F401
from price_stradamus.models.neural.tcn import TCNModel  # noqa: F401
from price_stradamus.models.neural.tft import TFTModel  # noqa: F401
from price_stradamus.models.registry import ModelRegistry

# Create console for output
console = Console()


def register_model_commands(app: typer.Typer) -> None:
    """Register all model-related commands to the main app.

    Args:
        app: Main Typer application instance
    """
    app.command()(train)
    app.command()(predict)
    app.command()(evaluate)
    app.command()(compare)


def train(
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g., nbeats)"),
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
    """Train a prediction model.

    This command:
    1. Loads OHLCV data from database
    2. Preprocesses and generates technical indicators
    3. Splits data into train/validation sets (70%/15%)
    4. Trains the specified model
    5. Optionally saves the trained model

    Examples:
        # Train N-BEATS model (default settings)
        price-stradamus train --model nbeats

        # Train XGBoost with custom parameters
        price-stradamus train -m xgboost -i 120 -o 10 -e 200

        # Train and save model
        price-stradamus train -m lstm --save models/lstm_model.pkl
    """

    async def _train():
        console.print(f"[bold cyan]Training {model} model...[/bold cyan]")

        try:
            # Load data from database
            console.print(f"Loading {symbol} {timeframe} data from database...")
            db = DatabaseManager(settings.database_url)
            await db.initialize()

            df = await db.get_ohlcv(symbol, timeframe)

            await db.close()

            if df.empty:
                console.print(
                    f"[red]✗ No data found for {symbol} {timeframe}. "
                    f"Run 'price-stradamus fetch' first.[/red]"
                )
                raise typer.Exit(1)

            console.print(f"[green]✓ Loaded {len(df)} candles[/green]")

            # Preprocess data
            console.print("Preprocessing data...")
            preprocessor = DataPreprocessor()
            df = preprocessor.validate_ohlcv(df)
            df = preprocessor.handle_missing_values(df)

            # Generate features
            console.print("Generating technical indicators...")
            engineer = FeatureEngineer()
            df_features = engineer.generate_all_features(df)

            console.print(
                f"[green]✓ Generated {len(engineer.feature_list)} features[/green]"
            )

            # Convert to Darts TimeSeries (use only close price for now)
            console.print("Creating time series...")
            ts = engineer.to_darts_timeseries(df_features, value_cols=["close"])

            console.print(
                f"[green]✓ Created time series with {len(ts)} timesteps[/green]"
            )

            # Split data (70% train, 15% val, 15% test)
            train_size = int(len(ts) * 0.7)
            val_size = int(len(ts) * 0.15)

            train_ts = ts[:train_size]
            val_ts = ts[train_size : train_size + val_size]

            console.print(f"Split: train={len(train_ts)}, val={len(val_ts)}")

            # Create and train model
            console.print(f"\nCreating {model} model...")
            model_instance = ModelRegistry.create_model(
                model,
                input_chunk_length=input_length,
                output_chunk_length=output_length,
                n_epochs=epochs,
            )

            console.print(f"[green]✓ Model created: {model_instance}[/green]")

            console.print(
                "\n[bold yellow]Training model (this may take a while)...[/bold yellow]"
            )
            console.print(
                "[dim]Press Ctrl+C to interrupt and save current progress[/dim]"
            )

            try:
                model_instance.fit(train_ts, val_ts)
                console.print("[green]✓ Training complete![/green]")
            except KeyboardInterrupt:
                console.print("\n[yellow]! Training interrupted by user[/yellow]")
                console.print("[yellow]✓ Saving partially trained model...[/yellow]")
                # Model will still be saved below

            # Save model if path provided
            if save_path:
                save_path_obj = Path(save_path)
                model_instance.save(save_path_obj)
                console.print(f"[green]✓ Model saved to {save_path}[/green]")

            # Show model summary if available
            if hasattr(model_instance, "get_model_summary"):
                console.print("\n[bold]Model Summary:[/bold]")
                console.print(model_instance.get_model_summary())  # type: ignore[attr-defined]

        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            logger.exception("Train command failed")
            raise typer.Exit(1) from e

    asyncio.run(_train())


def predict(
    model_path: str = typer.Option(..., "--model", "-m", help="Path to trained model"),
    steps: int = typer.Option(5, "--steps", "-s", help="Number of steps to predict"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Trading symbol"),
    timeframe: str = typer.Option("1m", "--timeframe", help="Timeframe"),
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

    Examples:
        # Make 5-step predictions
        price-stradamus predict --model models/nbeats_model.pkl

        # Predict 10 steps and show chart
        price-stradamus predict -m models/xgboost_model.pkl -s 10 --plot

        # Save predictions chart to file
        price-stradamus predict -m models/lstm_model.pkl --save-chart results/predictions.html
    """

    async def _predict():
        console.print(f"[bold cyan]Making predictions with {model_path}...[/bold cyan]")

        # Check if model file exists
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            console.print(f"[red]✗ Model file not found: {model_path}[/red]")
            raise typer.Exit(1) from None

        # Load most recent data
        console.print(f"Loading recent {symbol} {timeframe} data...")
        db = DatabaseManager(settings.database_url)
        await db.initialize()

        # Get last 1000 candles for context
        df = await db.get_ohlcv(symbol, timeframe, limit=1000)
        await db.close()

        if df.empty:
            console.print("[red]✗ No data found. Run 'fetch' command first.[/red]")
            raise typer.Exit(1) from None

        console.print(f"[green]✓ Loaded {len(df)} candles[/green]")

        # Preprocess and generate features
        console.print("Preprocessing data...")
        preprocessor = DataPreprocessor(normalization_method="minmax")
        df = preprocessor.validate_ohlcv(df)
        df = preprocessor.handle_missing_values(df)

        console.print("Generating features...")
        engineer = FeatureEngineer()
        df_features = engineer.generate_all_features(df)

        # Convert to time series
        console.print("Creating time series...")
        ts = engineer.to_darts_timeseries(df_features, value_cols=["close"])

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

        # Make predictions
        console.print(f"\nMaking {steps}-step predictions...")
        predictions = model_instance.predict(n=steps, series=ts)

        # Display predictions
        console.print("\n[bold green]Predictions:[/bold green]")
        pred_values = predictions.values().flatten()

        table = Table(title=f"{steps}-Step Ahead Predictions")
        table.add_column("Step", style="cyan")
        table.add_column("Predicted Price", style="green")
        table.add_column("Change from Last", style="yellow")

        last_actual = float(ts.values()[-1][0])
        for i, pred_price in enumerate(pred_values[:steps], 1):
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
        console.print(f"Last prediction ({steps}-step): ${last_pred:.2f}")
        console.print(f"Total predicted change: {total_change:+.2f}%")

        # Generate visualization if requested
        if plot or save_chart:
            console.print(
                "\n[bold yellow]Generating interactive chart...[/bold yellow]"
            )

            from price_stradamus.visualization.charts import PriceChartVisualizer

            # Get prediction timestamps
            pred_timestamps = predictions.time_index

            # Create chart
            chart_path = Path(save_chart) if save_chart else None
            PriceChartVisualizer.plot_predictions(
                historical_df=df_features,
                predictions=pred_values,
                timestamps=pred_timestamps,
                actual_values=None,  # No actuals for future predictions
                title=f"{model_instance.name.upper()} - {steps}-Step Predictions for {symbol} {timeframe}",
                save_path=chart_path,
                show=plot,
            )

            if save_chart:
                console.print(f"[green]✓ Chart saved to {save_chart}[/green]")
            if plot:
                console.print("[green]✓ Chart opened in browser[/green]")

    try:
        asyncio.run(_predict())
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        logger.exception("Predict command failed")
        raise typer.Exit(1) from e


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
    2. Runs backtesting on historical data
    3. Calculates performance metrics (MAE, RMSE, MAPE, directional accuracy)
    4. Optionally visualizes results

    Examples:
        # Evaluate a saved model
        price-stradamus evaluate --model-path models/xgboost_model.pkl

        # Evaluate a model type (train from scratch)
        price-stradamus evaluate --model xgboost

        # Evaluate with visualization
        price-stradamus evaluate -p models/nbeats_model.pkl --plot
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
            console.print(f"Loading {symbol} {timeframe} data...")
            db = DatabaseManager(settings.database_url)
            await db.initialize()

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
            engineer = FeatureEngineer()
            df_features = engineer.generate_all_features(df)
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

                # For pre-trained models, evaluate directly without retraining
                console.print(
                    f"\n[bold yellow]Evaluating pre-trained model on {len(ts)} timesteps...[/bold yellow]"
                )

                # Generate predictions iteratively on the full dataset
                predictions_list = []
                actuals_list = []
                timestamps_list = []

                output_length = model_instance.output_chunk_length
                input_length = model_instance.input_chunk_length

                # Start predicting from input_length onwards
                current_idx = input_length
                while current_idx + output_length <= len(ts):
                    # Use historical data up to current point
                    historical = ts[:current_idx]

                    # Predict next N steps
                    pred = model_instance.predict(n=output_length, series=historical)

                    # Get actual values
                    actual = ts[current_idx : current_idx + output_length]

                    predictions_list.append(pred.values().flatten())
                    actuals_list.append(actual.values().flatten())
                    timestamps_list.append(actual.time_index.to_numpy())

                    # Move forward by output_length
                    current_idx += output_length

                console.print(
                    f"[green]✓ Generated {len(predictions_list)} prediction windows[/green]"
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


def compare(
    models: list[str] = typer.Option(
        ..., "--model", "-m", help="Model names to compare (repeat for multiple)"
    ),
) -> None:
    """Compare multiple models side-by-side.

    This command will be fully implemented in Phase 2 with:
    - Parallel model training
    - Side-by-side metrics comparison
    - Visual comparison charts
    - Statistical significance tests

    Examples:
        # Compare three models
        price-stradamus compare -m nbeats -m lstm -m xgboost

        # Compare with saved models
        price-stradamus compare --model-path models/nbeats.pkl --model-path models/lstm.pkl
    """
    console.print(f"[bold cyan]Comparing models: {', '.join(models)}...[/bold cyan]")
    console.print(
        "[yellow]Model comparison will be fully implemented in Phase 2[/yellow]"
    )
    console.print(
        "[dim]This feature will include parallel training, metrics comparison, and visualization[/dim]"
    )
