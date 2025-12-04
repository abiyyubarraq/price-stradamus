"""Interactive chart visualization for price predictions.

This module provides TradingView-style charts for comparing predictions
with actual prices.
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
        title: str = "Price Predictions",
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
        """Plot price predictions with interactive chart.

        Args:
            historical_df: Historical OHLCV data
            predictions: Predicted prices
            timestamps: Timestamps for predictions
            actual_values: Actual prices (if available, for comparison)
            title: Chart title
            save_path: Path to save HTML file (optional)
            show: Whether to open chart in browser
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not installed. Cannot create charts.")
            return

        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(title, "Volume"),
        )

        # Add candlestick chart for historical data
        fig.add_trace(
            go.Candlestick(
                x=historical_df["timestamp"],
                open=historical_df["open"],
                high=historical_df["high"],
                low=historical_df["low"],
                close=historical_df["close"],
                name="Historical Price",
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            row=1,
            col=1,
        )

        # Add predicted prices
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=predictions,
                mode="lines+markers",
                name="Predicted Price",
                line={"color": "blue", "width": 2},
                marker={"size": 8, "symbol": "diamond"},
            ),
            row=1,
            col=1,
        )

        # Add actual prices if available
        if actual_values is not None:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=actual_values,
                    mode="lines+markers",
                    name="Actual Price",
                    line={"color": "orange", "width": 2, "dash": "dot"},
                    marker={"size": 6, "symbol": "circle"},
                ),
                row=1,
                col=1,
            )

        # Add volume bars
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
        fig.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                # Hide gaps (weekends, etc.)
                {"bounds": ["sat", "mon"]},
            ],
        )

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
        timestamps: np.ndarray,
        metrics: dict[str, float],
        title: str = "Model Evaluation Results",
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
        """Plot comprehensive evaluation results.

        Shows:
        1. Historical prices (candlesticks)
        2. Predicted vs Actual comparison
        3. Prediction errors over time
        4. Metrics summary

        Args:
            df: Historical OHLCV data
            predictions: All predicted values
            actuals: All actual values
            timestamps: Timestamps for predictions
            metrics: Calculated metrics dictionary
            title: Chart title
            save_path: Path to save HTML file
            show: Whether to open in browser
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not installed. Cannot create charts.")
            return

        # Create subplots
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.3, 0.2, 0.1],
            subplot_titles=(
                title,
                "Predicted vs Actual",
                "Prediction Error",
                "Volume",
            ),
        )

        # Row 1: Historical candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Historical",
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            row=1,
            col=1,
        )

        # Row 2: Predicted vs Actual
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=actuals,
                mode="lines",
                name="Actual",
                line={"color": "orange", "width": 2},
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=predictions,
                mode="lines",
                name="Predicted",
                line={"color": "blue", "width": 2, "dash": "dash"},
            ),
            row=2,
            col=1,
        )

        # Row 3: Error over time
        errors = predictions - actuals
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=errors,
                mode="lines",
                name="Error",
                line={"color": "red", "width": 1},
                fill="tozeroy",
            ),
            row=3,
            col=1,
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

        # Row 4: Volume
        colors = ["green" if c >= o else "red" for c, o in zip(df["close"], df["open"])]

        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.5,
            ),
            row=4,
            col=1,
        )

        # Create metrics annotation
        metrics_text = "<br>".join(
            [
                "<b>Performance Metrics:</b>",
                f"MAE: {metrics['mae']:.2f}",
                f"RMSE: {metrics['rmse']:.2f}",
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
            height=1000,
            showlegend=True,
        )

        # Update axes
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="Price (USDT)", row=2, col=1)
        fig.update_yaxes(title_text="Error (USDT)", row=3, col=1)
        fig.update_yaxes(title_text="Volume", row=4, col=1)

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
            actual: Actual values
            timestamps: Timestamps
            title: Chart title
            save_path: Path to save HTML
            show: Whether to show in browser
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not installed. Cannot create charts.")
            return

        fig = go.Figure()

        # Add actual values
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=actual,
                mode="lines",
                name="Actual",
                line={"color": "black", "width": 3},
            )
        )

        # Add each model's predictions
        colors = ["blue", "red", "green", "purple", "orange"]
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=predictions,
                    mode="lines",
                    name=model_name,
                    line={"color": colors[i % len(colors)], "width": 2, "dash": "dash"},
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            hovermode="x unified",
            template="plotly_dark",
            height=600,
        )

        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Comparison chart saved to {save_path}")

        if show:
            fig.show()
