"""Interactive chart visualization for price predictions.

This module provides TradingView-style charts for comparing predictions
with actual prices using Plotly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Install with: pip install plotly")


class PriceChartVisualizer:
    """Create interactive price prediction charts."""

    @staticmethod
    def plot_predictions(
        historical_df: pd.DataFrame,
        predictions: np.ndarray,
        timestamps: pd.DatetimeIndex,
        actual_values: np.ndarray | None = None,
        actual_candles_df: pd.DataFrame | None = None,
        title: str = "Price Predictions",
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
        """Plot price predictions with candlestick chart.

        Args:
            historical_df: Historical OHLCV data with columns [timestamp, open, high, low, close, volume]
            predictions: Predicted close prices
            timestamps: Timestamps for predictions
            actual_values: Actual close prices from DB (if available, for comparison)
            actual_candles_df: Actual OHLCV candles for prediction period from DB (if available)
            title: Chart title
            save_path: Path to save HTML file (optional)
            show: Whether to open chart in browser
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not installed. Install with: pip install plotly")
            return

        # Validate inputs
        if historical_df is None or historical_df.empty:
            logger.error("Cannot create chart: historical_df is empty")
            return

        if len(predictions) == 0:
            logger.error("Cannot create chart: predictions array is empty")
            return

        if len(timestamps) == 0:
            logger.error("Cannot create chart: timestamps array is empty")
            return

        if len(predictions) != len(timestamps):
            logger.error(
                f"Data length mismatch: predictions={len(predictions)}, timestamps={len(timestamps)}"
            )
            return

        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.75, 0.25],
            subplot_titles=(title, "Volume"),
        )

        # 1. Add historical OHLC candlesticks from DB
        fig.add_trace(
            go.Candlestick(
                x=historical_df["timestamp"],
                open=historical_df["open"],
                high=historical_df["high"],
                low=historical_df["low"],
                close=historical_df["close"],
                name="Historical OHLC",
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            row=1,
            col=1,
        )

        # 2. Add actual OHLC candles for prediction period from DB (if available)
        if actual_candles_df is not None and not actual_candles_df.empty:
            fig.add_trace(
                go.Candlestick(
                    x=actual_candles_df["timestamp"],
                    open=actual_candles_df["open"],
                    high=actual_candles_df["high"],
                    low=actual_candles_df["low"],
                    close=actual_candles_df["close"],
                    name="Actual OHLC (Prediction Period)",
                    increasing_line_color="green",
                    decreasing_line_color="red",
                    opacity=0.7,
                ),
                row=1,
                col=1,
            )

        # 3. Add predicted close prices line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=predictions,
                mode="lines+markers",
                name="Predicted Close",
                line={"color": "blue", "width": 3},
                marker={"size": 8, "symbol": "diamond"},
            ),
            row=1,
            col=1,
        )

        # 4. Add actual close prices line from DB (if available)
        if actual_values is not None:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=actual_values,
                    mode="lines+markers",
                    name="Actual Close (DB)",
                    line={"color": "orange", "width": 3, "dash": "dot"},
                    marker={"size": 8, "symbol": "circle"},
                ),
                row=1,
                col=1,
            )

        # 5. Add prediction zone marker
        if len(timestamps) > 0:
            # Add vertical line at prediction start
            fig.add_vline(
                x=timestamps[0],
                line_width=2,
                line_dash="dash",
                line_color="yellow",
                annotation_text="Prediction Start",
                annotation_position="top",
            )

        # 6. Add volume bars (historical)
        colors = [
            "green" if c >= o else "red"
            for c, o in zip(historical_df["close"], historical_df["open"])
        ]

        fig.add_trace(
            go.Bar(
                x=historical_df["timestamp"],
                y=historical_df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.5,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20},
            },
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            yaxis2_title="Volume",
            hovermode="x unified",
            template="plotly_dark",
            height=800,
            showlegend=True,
            legend={
                "yanchor": "top",
                "y": 0.99,
                "xanchor": "left",
                "x": 0.01,
            },
        )

        # Update axes
        fig.update_xaxes(rangeslider_visible=False)

        # Save to file if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            logger.info(f"Chart saved to {save_path}")

        # Show in browser if requested
        if show:
            fig.show()

    @staticmethod
    def plot_evaluation_results(
        df: pd.DataFrame,
        predictions: np.ndarray,
        actuals: np.ndarray,
        timestamps: np.ndarray | pd.DatetimeIndex,
        metrics: dict[str, float],
        title: str = "Model Evaluation Results",
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
        """Plot comprehensive evaluation results.

        Shows 3 chart rows:
        1. OHLC candlesticks + Actual close line + Predicted close line (all overlaid)
        2. Prediction errors over time
        3. Volume bars
        Plus metrics summary overlay

        Args:
            df: Historical OHLCV data from DB
            predictions: All predicted close values
            actuals: All actual close values from DB
            timestamps: Timestamps for predictions
            metrics: Calculated metrics dictionary
            title: Chart title
            save_path: Path to save HTML file
            show: Whether to open in browser
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not installed. Cannot create charts.")
            return

        # Validate inputs
        if df is None or df.empty:
            logger.error("Cannot create chart: DataFrame is empty")
            return

        if len(predictions) == 0 or len(actuals) == 0 or len(timestamps) == 0:
            logger.error("Cannot create chart: predictions, actuals, or timestamps are empty")
            return

        if len(predictions) != len(actuals) or len(predictions) != len(timestamps):
            logger.error(
                f"Data length mismatch: predictions={len(predictions)}, "
                f"actuals={len(actuals)}, timestamps={len(timestamps)}"
            )
            return

        # Convert timestamps to pandas datetime if needed
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.to_datetime(timestamps, utc=True)

        # Create subplots with 3 rows
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.75, 0.125, 0.125],
            subplot_titles=(
                title,
                "Prediction Error",
                "Volume",
            ),
        )

        # Row 1: Historical OHLC candlesticks from DB
        fig.add_trace(
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Historical OHLC",
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            row=1,
            col=1,
        )

        # Row 1: Add actual close prices from DB (overlaid on candlesticks)
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=actuals,
                mode="lines+markers",
                name="Actual Close (DB)",
                line={"color": "orange", "width": 3},
                marker={"size": 6, "symbol": "circle"},
            ),
            row=1,
            col=1,
        )

        # Row 1: Add predicted close prices (overlaid on candlesticks)
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=predictions,
                mode="lines+markers",
                name="Predicted Close",
                line={"color": "blue", "width": 3, "dash": "dash"},
                marker={"size": 6, "symbol": "diamond"},
            ),
            row=1,
            col=1,
        )

        # Row 2: Error over time
        errors = predictions - actuals
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=errors,
                mode="lines",
                name="Prediction Error",
                line={"color": "red", "width": 2},
                fill="tozeroy",
            ),
            row=2,
            col=1,
        )

        # Add zero line for error
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        # Row 3: Volume
        colors = ["green" if c >= o else "red" for c, o in zip(df["close"], df["open"])]

        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.5,
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        # Create metrics annotation
        metrics_text = "<br>".join(
            [
                "<b>Performance Metrics:</b>",
                f"MAE: {metrics['mae']:.2f} USDT",
                f"RMSE: {metrics['rmse']:.2f} USDT",
                f"MAPE: {metrics['mape']:.2f}%",
                f"Dir. Accuracy: {metrics['directional_accuracy']:.2f}%",
            ]
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            text=metrics_text,
            showarrow=False,
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            borderwidth=1,
            font={"color": "white", "size": 12},
        )

        # Update layout
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20},
            },
            hovermode="x unified",
            template="plotly_dark",
            height=1200,
            showlegend=True,
            legend={
                "yanchor": "top",
                "y": 0.99,
                "xanchor": "left",
                "x": 0.01,
            },
        )

        # Update axes
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="Error (USDT)", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            logger.info(f"Evaluation chart saved to {save_path}")

        # Show if requested
        if show:
            fig.show()

    @staticmethod
    def plot_prediction_comparison(
        predictions_dict: dict[str, np.ndarray],
        actual: np.ndarray,
        timestamps: pd.DatetimeIndex,
        title: str = "Model Comparison",
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
        """Compare predictions from multiple models.

        Args:
            predictions_dict: Dictionary of model_name -> predictions
            actual: Actual close values from DB
            timestamps: Timestamps
            title: Chart title
            save_path: Path to save HTML
            show: Whether to show in browser
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not installed. Cannot create charts.")
            return

        fig = go.Figure()

        # Add actual close values from DB
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=actual,
                mode="lines+markers",
                name="Actual Close (DB)",
                line={"color": "white", "width": 4},
                marker={"size": 8, "symbol": "circle"},
            )
        )

        # Add each model's predictions
        colors = ["blue", "red", "green", "purple", "orange", "cyan", "magenta"]
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=predictions,
                    mode="lines+markers",
                    name=f"{model_name} Predicted",
                    line={"color": colors[i % len(colors)], "width": 2, "dash": "dash"},
                    marker={"size": 6, "symbol": "diamond"},
                )
            )

        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 18},
            },
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            hovermode="x unified",
            template="plotly_dark",
            height=600,
            showlegend=True,
            legend={
                "yanchor": "top",
                "y": 0.99,
                "xanchor": "left",
                "x": 0.01,
            },
        )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
            logger.info(f"Comparison chart saved to {save_path}")

        if show:
            fig.show()
