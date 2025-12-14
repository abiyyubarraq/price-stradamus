"""Train command - Train prediction models.

This command supports two validation modes:
1. Walk-forward validation (DEFAULT) - Prevents data leakage, realistic results
2. Simple split (--quick) - Faster but may overfit

Walk-forward validation retrains the model on each fold, testing on out-of-sample
data only. This prevents overfitting and produces realistic performance metrics.

Example:
    # Default: Walk-forward validation (5 folds)
    price-stradamus train --model nbeats

    # Quick experimentation (simple 70/15/15 split - shows warning)
    price-stradamus train --model nbeats --quick

    # Custom walk-forward settings
    price-stradamus train --model nbeats --folds 10 --train-window 90
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import typer
from darts import TimeSeries
from loguru import logger
from rich.console import Console
from rich.table import Table

from price_stradamus.config.constants import (
    DEFAULT_TEST_WINDOW_DAYS,
    DEFAULT_TRAIN_WINDOW_DAYS,
    DEFAULT_WALK_FORWARD_FOLDS,
    FEATURE_WARMUP_PERIOD,
    REALISTIC_DIR_ACC_MAX,
    SUSPICIOUS_DIR_ACC,
    SUSPICIOUS_SHARPE,
)
from price_stradamus.config.settings import settings
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.preprocessor import DataPreprocessor
from price_stradamus.data.stateful_features import StatefulFeatureEngineer
from price_stradamus.evaluation.financial_metrics import (
    TradingCosts,
)
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
    # Walk-forward validation options (NEW - default behavior)
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Use simple 70/15/15 split (faster, for experimentation only)",
    ),
    folds: int = typer.Option(
        DEFAULT_WALK_FORWARD_FOLDS,
        "--folds",
        help="Number of walk-forward folds",
    ),
    train_window: int = typer.Option(
        DEFAULT_TRAIN_WINDOW_DAYS,
        "--train-window",
        help="Training window size in days",
    ),
    test_window: int = typer.Option(
        DEFAULT_TEST_WINDOW_DAYS,
        "--test-window",
        help="Test window size in days",
    ),
    # Legacy datetime options (still supported)
    train_start: str | None = typer.Option(
        None, "--train-start", help="Training start datetime (e.g., '2025-08-01 00:00')"
    ),
    train_end: str | None = typer.Option(
        None, "--train-end", help="Training end datetime (e.g., '2025-11-03 03:00')"
    ),
    val_start: str | None = typer.Option(
        None, "--val-start", help="Validation start datetime (optional)"
    ),
    val_end: str | None = typer.Option(
        None, "--val-end", help="Validation end datetime (optional)"
    ),
    save_path: str | None = typer.Option(
        None, "--save", help="Path to save trained model"
    ),
    # Transaction costs for financial metrics
    commission: float = typer.Option(
        0.001, "--commission", help="Commission per trade (default 0.1%)"
    ),
    slippage: float = typer.Option(
        0.0005, "--slippage", help="Slippage per trade (default 0.05%)"
    ),
) -> None:
    """Train a prediction model with walk-forward validation.

    This command:
    1. Loads OHLCV data from database
    2. Splits data FIRST (prevents data leakage)
    3. Generates features separately for each fold (fit on train, transform test)
    4. Trains and evaluates on each fold
    5. Reports aggregated metrics with confidence intervals
    6. Optionally saves the final trained model

    VALIDATION MODES:
    - Walk-forward (DEFAULT): Realistic evaluation, prevents overfitting
    - Quick (--quick): Fast experimentation, may overfit

    Examples:
        # Default: Walk-forward validation (5 folds, 60-day train, 10-day test)
        price-stradamus train --model nbeats

        # Quick experimentation (simple 70/15/15 split - shows warning)
        price-stradamus train --model nbeats --quick

        # Custom walk-forward settings
        price-stradamus train --model nbeats --folds 10 --train-window 90 --test-window 15

        # With custom transaction costs
        price-stradamus train --model xgboost --commission 0.0005 --slippage 0.0002

        # Train and save model
        price-stradamus train -m lstm --save models/lstm_model.pkl
    """

    async def _train():
        console.print(f"[bold cyan]Training {model} model...[/bold cyan]")

        # Set up trading costs for financial metrics
        costs = TradingCosts(commission_pct=commission, slippage_pct=slippage)

        try:
            # Determine validation mode
            use_datetime_windows = train_start or train_end or val_start or val_end
            use_walk_forward = not quick and not use_datetime_windows

            if use_walk_forward:
                console.print(
                    "\n[bold green]Using walk-forward validation[/bold green]"
                )
                console.print(
                    f"  Folds: {folds}, Train window: {train_window} days, "
                    f"Test window: {test_window} days"
                )
            elif quick:
                console.print(
                    "\n[bold yellow]⚠️  Using quick mode (simple split)[/bold yellow]"
                )
                console.print(
                    "[yellow]  Warning: Results may be overly optimistic due to "
                    "potential data leakage.[/yellow]"
                )
                console.print(
                    "[yellow]  For realistic results, remove --quick flag.[/yellow]"
                )

            # Parse datetime windows if provided (legacy mode)
            train_start_dt: datetime | None = None
            train_end_dt: datetime | None = None
            val_start_dt: datetime | None = None
            val_end_dt: datetime | None = None

            if use_datetime_windows:
                date_formats = [
                    "%Y-%m-%d %H:%M",
                    "%Y-%m-%d %H:%M:%S",
                    "%d/%m/%Y %H:%M",
                    "%d/%m/%Y %H:%M:%S",
                ]

                def parse_datetime(
                    date_str: str | None, param_name: str
                ) -> datetime | None:
                    if not date_str:
                        return None
                    for fmt in date_formats:
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            return dt.replace(tzinfo=UTC)
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
                val_start_dt = parse_datetime(val_start, "--val-start")
                val_end_dt = parse_datetime(val_end, "--val-end")

                # Validate datetime windows
                if train_start_dt and train_end_dt and train_start_dt >= train_end_dt:
                    console.print(
                        "[red]✗ Training start must be before training end[/red]"
                    )
                    raise typer.Exit(1)
                if val_start_dt and val_end_dt and val_start_dt >= val_end_dt:
                    console.print(
                        "[red]✗ Validation start must be before validation end[/red]"
                    )
                    raise typer.Exit(1)
                if train_end_dt and val_start_dt and val_start_dt <= train_end_dt:
                    console.print(
                        "[red]✗ Data leakage! Validation start must be after training end[/red]"
                    )
                    raise typer.Exit(1)

                console.print("\n[yellow]Using datetime-based windows:[/yellow]")
                if train_start_dt and train_end_dt:
                    console.print(f"  Train: {train_start_dt} to {train_end_dt}")
                if val_start_dt and val_end_dt:
                    console.print(f"  Val:   {val_start_dt} to {val_end_dt}")

            # Load data from database
            console.print(f"\nLoading {symbol} {timeframe} data from database...")
            db = DatabaseManager(settings.database_url)
            await db.initialize()

            # Determine data fetch window
            if use_datetime_windows:
                fetch_start = train_start_dt
                fetch_end = val_end_dt or train_end_dt
                if fetch_start:
                    fetch_start = fetch_start - timedelta(days=30)
                console.print(
                    f"[cyan]Fetching data window: {fetch_start} to {fetch_end}[/cyan]"
                )
                df = await db.get_ohlcv(
                    symbol, timeframe, start_date=fetch_start, end_date=fetch_end
                )
            else:
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

            # Ensure timestamp is datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            # ==================================================================
            # WALK-FORWARD VALIDATION (DEFAULT)
            # ==================================================================
            if use_walk_forward:
                await _train_walk_forward(
                    model_name=model,
                    df=df,
                    num_folds=folds,
                    train_window_days=train_window,
                    test_window_days=test_window,
                    input_length=input_length,
                    output_length=output_length,
                    epochs=epochs,
                    costs=costs,
                    save_path=save_path,
                )
            # ==================================================================
            # QUICK MODE (Simple split - shows warning)
            # ==================================================================
            elif quick:
                await _train_quick(
                    model_name=model,
                    df=df,
                    input_length=input_length,
                    output_length=output_length,
                    epochs=epochs,
                    costs=costs,
                    save_path=save_path,
                )
            # ==================================================================
            # DATETIME WINDOWS MODE (Legacy)
            # ==================================================================
            else:
                await _train_datetime_windows(
                    model_name=model,
                    df=df,
                    train_start_dt=train_start_dt,
                    train_end_dt=train_end_dt,
                    val_start_dt=val_start_dt,
                    val_end_dt=val_end_dt,
                    input_length=input_length,
                    output_length=output_length,
                    epochs=epochs,
                    costs=costs,
                    save_path=save_path,
                )

        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            logger.exception("Train command failed")
            raise typer.Exit(1) from e

    asyncio.run(_train())


async def _train_walk_forward(
    model_name: str,
    df: pd.DataFrame,
    num_folds: int,
    train_window_days: int,
    test_window_days: int,
    input_length: int,
    output_length: int,
    epochs: int,
    costs: TradingCosts,
    save_path: str | None,
) -> None:
    """Train model using walk-forward validation.

    This is the CORRECT way to train - prevents data leakage by:
    1. Splitting data FIRST
    2. Fitting features on training data only
    3. Transforming test data using training statistics
    """
    # Calculate samples per day (assuming 1-minute data)
    samples_per_day = 24 * 60  # 1440 for 1-minute data
    train_samples = train_window_days * samples_per_day
    test_samples = test_window_days * samples_per_day

    total_needed = train_samples + (num_folds * test_samples)
    if len(df) < total_needed:
        console.print(
            f"[yellow]⚠️  Not enough data for {num_folds} folds. "
            f"Need {total_needed}, have {len(df)}[/yellow]"
        )
        # Adjust folds to what's possible
        available_test_samples = len(df) - train_samples
        num_folds = max(1, available_test_samples // test_samples)
        console.print(f"[yellow]  Adjusted to {num_folds} folds[/yellow]")

    all_fold_metrics: list[dict[str, float]] = []
    model_instance = None

    for fold in range(num_folds):
        console.print(f"\n[bold cyan]═══ Fold {fold + 1}/{num_folds} ═══[/bold cyan]")

        # Calculate fold boundaries
        test_end_idx = len(df) - (num_folds - fold - 1) * test_samples
        test_start_idx = test_end_idx - test_samples
        train_end_idx = test_start_idx
        train_start_idx = max(0, train_end_idx - train_samples)

        train_df = df.iloc[train_start_idx:train_end_idx].copy()
        test_df = df.iloc[test_start_idx:test_end_idx].copy()

        console.print(
            f"  Train: {len(train_df)} samples "
            f"({train_df['timestamp'].iloc[0]} to {train_df['timestamp'].iloc[-1]})"
        )
        console.print(
            f"  Test:  {len(test_df)} samples "
            f"({test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]})"
        )

        # CRITICAL: Generate features AFTER split (fit on train, transform test)
        console.print("  Generating features (fit on train, transform test)...")
        engineer = StatefulFeatureEngineer(warmup_period=FEATURE_WARMUP_PERIOD)

        train_features = engineer.fit_transform(train_df)
        test_features = engineer.transform(test_df, drop_warmup=False)

        console.print(
            f"  [green]✓ Train features: {len(train_features)} samples, "
            f"{len(engineer.feature_list)} features[/green]"
        )

        # Convert to Darts TimeSeries
        train_ts = cast(TimeSeries, engineer.to_darts_timeseries(train_features, value_cols=["close"]))
        test_ts = cast(TimeSeries, engineer.to_darts_timeseries(test_features, value_cols=["close"]))

        # Create and train model
        console.print(f"  Training {model_name}...")
        model_instance = ModelRegistry.create_model(
            model_name,
            input_chunk_length=input_length,
            output_chunk_length=output_length,
            n_epochs=epochs,
        )

        # Use last portion of train as validation
        val_split = int(len(train_ts) * 0.85)
        train_ts_fit = train_ts[:val_split]
        val_ts_fit = train_ts[val_split:]

        try:
            model_instance.fit(train_ts_fit, val_ts_fit)
        except KeyboardInterrupt:
            console.print("\n[yellow]! Training interrupted[/yellow]")
            break

        # Evaluate on test set
        console.print("  Evaluating on test set...")
        predictions = model_instance.predict(len(test_ts))

        # Extract values for metrics
        y_true = test_ts.values().flatten()
        y_pred = predictions.values().flatten()[: len(y_true)]

        # Calculate all metrics including financial
        fold_metrics = MetricsCalculator.calculate_all_with_financial(
            y_true, y_pred, include_financial=True, trading_costs=costs
        )
        all_fold_metrics.append(fold_metrics)

        # Display fold results
        console.print(f"  [green]✓ Fold {fold + 1} Results:[/green]")
        console.print(f"    MAE: {fold_metrics['mae']:.4f}")
        console.print(f"    RMSE: {fold_metrics['rmse']:.4f}")
        console.print(
            f"    Directional Accuracy: {fold_metrics['directional_accuracy']:.2f}%"
        )
        console.print(f"    Net P&L: ${fold_metrics['fin_net_pnl']:.2f}")
        console.print(f"    Sharpe Ratio: {fold_metrics['fin_sharpe_ratio']:.2f}")

    # Aggregate results across all folds
    if all_fold_metrics:
        _display_aggregated_results(all_fold_metrics, costs)

    # Save final model if requested
    if save_path and model_instance:
        save_path_obj = Path(save_path)
        model_instance.save(save_path_obj)
        console.print(f"\n[green]✓ Model saved to {save_path}[/green]")


async def _train_quick(
    model_name: str,
    df: pd.DataFrame,
    input_length: int,
    output_length: int,
    epochs: int,
    costs: TradingCosts,
    save_path: str | None,
) -> None:
    """Train with simple split (quick mode - may have data leakage)."""
    # Split data first (70% train, 15% val, 15% test)
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size : train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size :].copy()

    console.print(
        f"\n  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    # Generate features with fit-transform pattern
    console.print("  Generating features...")
    engineer = StatefulFeatureEngineer(warmup_period=FEATURE_WARMUP_PERIOD)

    train_features = engineer.fit_transform(train_df)
    val_features = engineer.transform(val_df, drop_warmup=False)
    test_features = engineer.transform(test_df, drop_warmup=False)

    console.print(f"  [green]✓ Generated {len(engineer.feature_list)} features[/green]")

    # Convert to time series
    train_ts = cast(TimeSeries, engineer.to_darts_timeseries(train_features, value_cols=["close"]))
    val_ts = cast(TimeSeries, engineer.to_darts_timeseries(val_features, value_cols=["close"]))
    test_ts = cast(TimeSeries, engineer.to_darts_timeseries(test_features, value_cols=["close"]))

    # Train model
    console.print(f"\n  Creating and training {model_name}...")
    model_instance = ModelRegistry.create_model(
        model_name,
        input_chunk_length=input_length,
        output_chunk_length=output_length,
        n_epochs=epochs,
    )

    console.print("[bold yellow]  Training (this may take a while)...[/bold yellow]")

    try:
        model_instance.fit(train_ts, val_ts)
        console.print("[green]  ✓ Training complete![/green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]  ! Training interrupted[/yellow]")

    # Evaluate on test set
    console.print("\n  Evaluating on test set...")
    predictions = model_instance.predict(len(test_ts))

    y_true = test_ts.values().flatten()
    y_pred = predictions.values().flatten()[: len(y_true)]

    metrics = MetricsCalculator.calculate_all_with_financial(
        y_true, y_pred, include_financial=True, trading_costs=costs
    )

    # Display results
    _display_single_results(metrics, costs)

    # Check for suspicious results
    _check_suspicious_results(metrics)

    # Save model
    if save_path:
        save_path_obj = Path(save_path)
        model_instance.save(save_path_obj)
        console.print(f"\n[green]✓ Model saved to {save_path}[/green]")


async def _train_datetime_windows(
    model_name: str,
    df: pd.DataFrame,
    train_start_dt: datetime | None,
    train_end_dt: datetime | None,
    val_start_dt: datetime | None,
    val_end_dt: datetime | None,
    input_length: int,
    output_length: int,
    epochs: int,
    costs: TradingCosts,
    save_path: str | None,
) -> None:
    """Train with explicit datetime windows."""
    # Filter train data
    train_mask = pd.Series(True, index=df.index)
    if train_start_dt:
        train_mask &= df["timestamp"] >= train_start_dt
    if train_end_dt:
        train_mask &= df["timestamp"] <= train_end_dt

    train_df = df[train_mask].copy()

    if len(train_df) == 0:
        console.print("[red]✗ No training data found in specified window[/red]")
        raise typer.Exit(1)

    # Filter validation data
    if val_start_dt and val_end_dt:
        val_mask = (df["timestamp"] >= val_start_dt) & (df["timestamp"] <= val_end_dt)
        val_df = df[val_mask].copy()
    else:
        # Use last 15% of training data as validation
        val_split = int(len(train_df) * 0.85)
        val_df = train_df.iloc[val_split:].copy()
        train_df = train_df.iloc[:val_split].copy()

    console.print(f"\n  Split: train={len(train_df)}, val={len(val_df)}")

    # Generate features with fit-transform pattern
    console.print("  Generating features...")
    engineer = StatefulFeatureEngineer(warmup_period=FEATURE_WARMUP_PERIOD)

    train_features = engineer.fit_transform(train_df)
    val_features = engineer.transform(val_df, drop_warmup=False)

    console.print(f"  [green]✓ Generated {len(engineer.feature_list)} features[/green]")

    # Convert to time series
    train_ts = cast(TimeSeries, engineer.to_darts_timeseries(train_features, value_cols=["close"]))
    val_ts = cast(TimeSeries, engineer.to_darts_timeseries(val_features, value_cols=["close"]))

    # Train model
    console.print(f"\n  Creating and training {model_name}...")
    model_instance = ModelRegistry.create_model(
        model_name,
        input_chunk_length=input_length,
        output_chunk_length=output_length,
        n_epochs=epochs,
    )

    console.print("[bold yellow]  Training (this may take a while)...[/bold yellow]")

    try:
        model_instance.fit(train_ts, val_ts)
        console.print("[green]  ✓ Training complete![/green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]  ! Training interrupted[/yellow]")

    # Evaluate on validation set
    console.print("\n  Evaluating...")
    predictions = model_instance.predict(len(val_ts))

    y_true = val_ts.values().flatten()
    y_pred = predictions.values().flatten()[: len(y_true)]

    metrics = MetricsCalculator.calculate_all_with_financial(
        y_true, y_pred, include_financial=True, trading_costs=costs
    )

    _display_single_results(metrics, costs)
    _check_suspicious_results(metrics)

    # Save model
    if save_path:
        save_path_obj = Path(save_path)
        model_instance.save(save_path_obj)
        console.print(f"\n[green]✓ Model saved to {save_path}[/green]")


def _display_aggregated_results(
    all_metrics: list[dict[str, float]], costs: TradingCosts
) -> None:
    """Display aggregated walk-forward results with confidence intervals."""
    console.print(
        "\n[bold green]═══ Walk-Forward Results (Aggregated) ═══[/bold green]"
    )

    # Calculate mean and std for key metrics
    key_metrics = [
        ("mae", "MAE", "{:.4f}"),
        ("rmse", "RMSE", "{:.4f}"),
        ("directional_accuracy", "Directional Accuracy", "{:.2f}%"),
        ("fin_net_pnl", "Net P&L", "${:.2f}"),
        ("fin_sharpe_ratio", "Sharpe Ratio", "{:.2f}"),
        ("fin_win_rate", "Win Rate", "{:.1f}%"),
        ("fin_max_drawdown_pct", "Max Drawdown", "{:.1f}%"),
    ]

    table = Table(title="Metrics Across Folds")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="yellow")
    table.add_column("Min", style="dim")
    table.add_column("Max", style="dim")

    for metric_key, metric_name, fmt in key_metrics:
        values = [m.get(metric_key, 0.0) for m in all_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        table.add_row(
            metric_name,
            fmt.format(mean_val),
            fmt.format(std_val),
            fmt.format(min_val),
            fmt.format(max_val),
        )

    console.print(table)

    # Transaction costs summary
    console.print(
        f"\n[dim]Transaction costs: {costs.commission_pct * 100:.2f}% commission + "
        f"{costs.slippage_pct * 100:.2f}% slippage = "
        f"{costs.round_trip_cost * 100:.2f}% per round trip[/dim]"
    )

    # Check for suspicious results
    avg_dir_acc = np.mean([m.get("directional_accuracy", 0) for m in all_metrics])
    avg_sharpe = np.mean([m.get("fin_sharpe_ratio", 0) for m in all_metrics])
    _check_suspicious_results(
        {"directional_accuracy": float(avg_dir_acc), "fin_sharpe_ratio": float(avg_sharpe)}
    )


def _display_single_results(metrics: dict[str, float], costs: TradingCosts) -> None:
    """Display results from a single evaluation."""
    console.print("\n[bold green]═══ Evaluation Results ═══[/bold green]")

    # Error metrics
    console.print("\n[bold]Error Metrics:[/bold]")
    console.print(f"  MAE:  {metrics['mae']:.4f}")
    console.print(f"  RMSE: {metrics['rmse']:.4f}")
    console.print(f"  MAPE: {metrics['mape']:.2f}%")
    console.print(f"  R²:   {metrics['r2']:.4f}")

    # Trading metrics
    console.print("\n[bold]Trading Metrics:[/bold]")
    console.print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")

    # Financial metrics
    console.print("\n[bold]Financial Metrics (with costs):[/bold]")
    console.print(f"  Gross P&L:    ${metrics['fin_gross_pnl']:.2f}")
    console.print(f"  Total Costs:  ${metrics['fin_total_costs']:.2f}")
    console.print(f"  Net P&L:      ${metrics['fin_net_pnl']:.2f}")
    console.print(f"  Return:       {metrics['fin_return_pct']:.2f}%")
    console.print(f"  Sharpe Ratio: {metrics['fin_sharpe_ratio']:.2f}")
    console.print(f"  Sortino Ratio:{metrics['fin_sortino_ratio']:.2f}")
    console.print(f"  Max Drawdown: {metrics['fin_max_drawdown_pct']:.1f}%")
    console.print(f"  Win Rate:     {metrics['fin_win_rate']:.1f}%")
    console.print(f"  Profit Factor:{metrics['fin_profit_factor']:.2f}")

    # Transaction costs summary
    console.print(
        f"\n[dim]Costs: {costs.commission_pct * 100:.2f}% commission + "
        f"{costs.slippage_pct * 100:.2f}% slippage[/dim]"
    )


def _check_suspicious_results(metrics: dict[str, float]) -> None:
    """Warn if results seem too good (likely data leakage)."""
    dir_acc = metrics.get("directional_accuracy", 0)
    sharpe = metrics.get("fin_sharpe_ratio", 0)

    if dir_acc > SUSPICIOUS_DIR_ACC:
        console.print(
            f"\n[bold red]⚠️  WARNING: Directional accuracy ({dir_acc:.1f}%) is suspiciously high![/bold red]"
        )
        console.print(
            "[red]  This may indicate data leakage. Realistic BTC 1-min accuracy is 51-55%.[/red]"
        )
        console.print(
            "[red]  Please verify your train/test split has no overlap.[/red]"
        )
    elif dir_acc > REALISTIC_DIR_ACC_MAX:
        console.print(
            f"\n[yellow]⚠️  Note: Directional accuracy ({dir_acc:.1f}%) is above typical range.[/yellow]"
        )
        console.print("[yellow]  Realistic range for BTC 1-min: 51-55%[/yellow]")

    if sharpe > SUSPICIOUS_SHARPE:
        console.print(
            f"\n[bold red]⚠️  WARNING: Sharpe ratio ({sharpe:.2f}) is suspiciously high![/bold red]"
        )
        console.print(
            "[red]  Realistic Sharpe for BTC is 0.5-2.0. Check for bugs.[/red]"
        )
