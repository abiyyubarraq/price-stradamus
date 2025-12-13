"""Financial metrics for trading strategy evaluation.

This module provides REALISTIC trading performance metrics that account for:
- Transaction costs (commission fees)
- Slippage estimation
- Net P&L calculation
- Risk-adjusted returns (Sharpe, Sortino)
- Win rate and profit factor

IMPORTANT: Without these metrics, your backtesting results are meaningless
for real trading. A 52% directional accuracy can be unprofitable after costs!

Example:
    costs = TradingCosts(commission_pct=0.001, slippage_pct=0.0005)
    calculator = FinancialMetricsCalculator(costs=costs)

    metrics = calculator.calculate_all(
        y_true=actual_prices,
        y_pred=predicted_prices,
        initial_capital=10000.0,
    )
    print(f"Net P&L: ${metrics['net_pnl']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class TradingCosts:
    """Trading cost configuration.

    Default values based on Binance fees for BTC/USDT:
    - Maker fee: 0.10% (limit orders)
    - Taker fee: 0.10% (market orders)
    - Slippage: 0.05% average for liquid pairs

    IMPORTANT: These are MINIMUM realistic values.
    Actual costs may be higher depending on:
    - Order size (larger orders = more slippage)
    - Market volatility (higher vol = more slippage)
    - Time of day (less liquidity = more slippage)

    Example:
        # Default Binance costs
        costs = TradingCosts()

        # Lower costs with BNB discount
        costs = TradingCosts(commission_pct=0.00075)

        # Higher costs for illiquid markets
        costs = TradingCosts(slippage_pct=0.002)
    """

    commission_pct: float = 0.001  # 0.1% per trade (Binance default)
    slippage_pct: float = 0.0005  # 0.05% estimated slippage

    @property
    def round_trip_cost(self) -> float:
        """Total cost for entering and exiting a position.

        Round trip = (commission + slippage) * 2 (entry + exit)
        """
        return 2 * (self.commission_pct + self.slippage_pct)

    def __post_init__(self):
        """Validate cost parameters."""
        if self.commission_pct < 0:
            raise ValueError("Commission cannot be negative")
        if self.slippage_pct < 0:
            raise ValueError("Slippage cannot be negative")

        # Warn if costs seem unrealistically low
        if self.round_trip_cost < 0.002:  # < 0.2%
            logger.warning(
                f"Round-trip costs ({self.round_trip_cost:.4%}) seem low. "
                f"Binance minimum is ~0.2% round-trip. "
                f"Consider if this is realistic for your trading scenario."
            )


class FinancialMetricsCalculator:
    """Calculate financial performance metrics for trading strategies.

    All metrics assume a simple strategy:
    - Go LONG when model predicts price UP
    - Go SHORT (or stay flat) when model predicts price DOWN
    - One position at a time (no pyramiding)

    CRITICAL: Always use this with MetricsCalculator to get a complete picture.
    Good error metrics (MAE, RMSE) don't guarantee profitability after costs!

    Example:
        costs = TradingCosts(commission_pct=0.001, slippage_pct=0.0005)
        calculator = FinancialMetricsCalculator(costs=costs)

        metrics = calculator.calculate_all(
            y_true=actual_prices,
            y_pred=predicted_prices,
            initial_capital=10000.0,
        )

        print(f"Net P&L: ${metrics['net_pnl']:.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")

        # Check if strategy is profitable after costs
        if metrics['net_pnl'] > 0:
            print("Strategy is profitable!")
        else:
            print("Strategy loses money after transaction costs")
    """

    def __init__(self, costs: TradingCosts | None = None):
        """Initialize financial metrics calculator.

        Args:
            costs: Trading cost configuration. If None, uses defaults
                  (0.1% commission + 0.05% slippage per trade).
        """
        self.costs = costs or TradingCosts()
        logger.info(
            f"FinancialMetricsCalculator initialized "
            f"(round-trip cost: {self.costs.round_trip_cost:.4%})"
        )

    def calculate_returns(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate trading returns from predictions.

        Strategy:
        - Predict price going UP: take LONG position (+1)
        - Predict price going DOWN: take SHORT position (-1)
        - Apply transaction costs when position changes

        Args:
            y_true: Actual prices
            y_pred: Predicted prices

        Returns:
            Tuple of (gross_returns, trade_costs, net_returns)
            All arrays have length (len(y_true) - 1)
        """
        if len(y_true) < 2 or len(y_pred) < 2:
            return np.array([]), np.array([]), np.array([])

        # Calculate actual price changes
        actual_returns = np.diff(y_true) / y_true[:-1]

        # Determine positions based on predictions
        # Compare predicted next price to current actual price
        # If pred[t+1] > true[t]: go long, else: go short
        predicted_directions = np.sign(y_pred[1:] - y_true[:-1])

        # Calculate gross returns (before costs)
        # Return = position * actual_return
        gross_returns = predicted_directions * actual_returns

        # Calculate costs (when position changes)
        # Position changes: from 0 to +/-1, from +1 to -1, from -1 to +1
        prev_positions = np.concatenate([[0], predicted_directions[:-1]])
        position_changes = np.abs(predicted_directions - prev_positions)

        # Cost is half round-trip per entry/exit
        trade_costs = position_changes * self.costs.round_trip_cost / 2

        # Net returns
        net_returns = gross_returns - trade_costs

        return gross_returns, trade_costs, net_returns

    def calculate_pnl(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        initial_capital: float = 10000.0,
    ) -> dict[str, float]:
        """Calculate P&L metrics from predictions.

        Args:
            y_true: Actual prices
            y_pred: Predicted prices
            initial_capital: Starting capital in USD

        Returns:
            Dictionary with P&L metrics:
            - gross_pnl: P&L before costs
            - net_pnl: P&L after costs
            - total_costs: Total trading costs paid
            - gross_return_pct: Gross return as percentage
            - net_return_pct: Net return as percentage
            - cost_drag_pct: How much costs reduced returns
            - final_capital: Final portfolio value
        """
        gross_returns, trade_costs, net_returns = self.calculate_returns(y_true, y_pred)

        if len(gross_returns) == 0:
            return {
                "gross_pnl": 0.0,
                "net_pnl": 0.0,
                "total_costs": 0.0,
                "gross_return_pct": 0.0,
                "net_return_pct": 0.0,
                "cost_drag_pct": 0.0,
                "final_capital": initial_capital,
            }

        # Calculate cumulative P&L
        gross_cumulative = initial_capital * np.cumprod(1 + gross_returns)
        net_cumulative = initial_capital * np.cumprod(1 + net_returns)

        total_costs = np.sum(trade_costs) * initial_capital
        final_gross = gross_cumulative[-1]
        final_net = net_cumulative[-1]

        return {
            "gross_pnl": float(final_gross - initial_capital),
            "net_pnl": float(final_net - initial_capital),
            "total_costs": float(total_costs),
            "gross_return_pct": float((final_gross / initial_capital - 1) * 100),
            "net_return_pct": float((final_net / initial_capital - 1) * 100),
            "cost_drag_pct": float((total_costs / initial_capital) * 100),
            "final_capital": float(final_net),
        }

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 525600,  # Minutes per year (1-min candles)
    ) -> float:
        """Calculate annualized Sharpe ratio.

        Sharpe Ratio = (mean_return - risk_free_rate) / std_return * sqrt(periods)

        Interpretation (from quant-analyst.md):
        - 0.5-1.0: Realistic for BTC 1-minute
        - 1.0-2.0: Exceptional
        - >3.0: Suspicious (check for bugs/leakage)

        Args:
            returns: Array of period returns
            risk_free_rate: Annual risk-free rate (default 0)
            periods_per_year: Trading periods per year for annualization
                             Default 525600 for 1-minute candles

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        # Convert annual risk-free to per-period
        rf_per_period = risk_free_rate / periods_per_year

        excess_returns = returns - rf_per_period

        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)

        if std_return == 0 or np.isnan(std_return):
            return 0.0

        # Annualize
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)

        return float(sharpe)

    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 525600,
    ) -> float:
        """Calculate annualized Sortino ratio (downside risk only).

        Sortino is like Sharpe but only penalizes downside volatility.
        Better for strategies that have asymmetric returns.

        Args:
            returns: Array of period returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        rf_per_period = risk_free_rate / periods_per_year
        excess_returns = returns - rf_per_period

        mean_return = np.mean(excess_returns)

        # Only consider negative returns for downside deviation
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            # No negative returns - infinite Sortino (or undefined)
            return float("inf") if mean_return > 0 else 0.0

        downside_std = np.std(negative_returns, ddof=1)

        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        sortino = (mean_return / downside_std) * np.sqrt(periods_per_year)

        return float(sortino)

    def calculate_max_drawdown(
        self,
        cumulative_values: np.ndarray,
    ) -> dict[str, float]:
        """Calculate maximum drawdown statistics.

        Max drawdown is the largest peak-to-trough decline.
        Important for risk assessment.

        Args:
            cumulative_values: Cumulative portfolio value over time

        Returns:
            Dictionary with:
            - max_drawdown_pct: Maximum drawdown as percentage (negative)
            - peak_value: Portfolio value at peak
            - trough_value: Portfolio value at trough
            - peak_idx: Index of peak
            - trough_idx: Index of trough
            - recovery_idx: Index where recovered (or -1 if not recovered)
        """
        if len(cumulative_values) < 2:
            return {
                "max_drawdown_pct": 0.0,
                "peak_value": 0.0,
                "trough_value": 0.0,
                "peak_idx": 0,
                "trough_idx": 0,
                "recovery_idx": -1,
            }

        # Running maximum
        running_max = np.maximum.accumulate(cumulative_values)

        # Drawdown at each point
        drawdowns = (cumulative_values - running_max) / running_max

        max_dd = np.min(drawdowns)
        max_dd_idx = np.argmin(drawdowns)

        # Find peak before max drawdown
        peak_idx = np.argmax(cumulative_values[: max_dd_idx + 1])

        # Find recovery (if any)
        recovery_idx = -1
        peak_value = cumulative_values[peak_idx]
        for i in range(max_dd_idx, len(cumulative_values)):
            if cumulative_values[i] >= peak_value:
                recovery_idx = i
                break

        return {
            "max_drawdown_pct": float(max_dd * 100),
            "peak_value": float(cumulative_values[peak_idx]),
            "trough_value": float(cumulative_values[max_dd_idx]),
            "peak_idx": int(peak_idx),
            "trough_idx": int(max_dd_idx),
            "recovery_idx": int(recovery_idx),
        }

    def calculate_win_rate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Calculate win rate and profit factor.

        Args:
            y_true: Actual prices
            y_pred: Predicted prices

        Returns:
            Dictionary with:
            - win_rate: Percentage of profitable trades
            - num_trades: Total number of trades
            - num_wins: Number of winning trades
            - num_losses: Number of losing trades
            - profit_factor: Gross profit / gross loss
            - avg_win_pct: Average winning trade return (%)
            - avg_loss_pct: Average losing trade return (%)
            - win_loss_ratio: Average win / average loss
        """
        _, _, net_returns = self.calculate_returns(y_true, y_pred)

        if len(net_returns) == 0:
            return {
                "win_rate": 0.0,
                "num_trades": 0,
                "num_wins": 0,
                "num_losses": 0,
                "profit_factor": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "win_loss_ratio": 0.0,
            }

        # Identify winning and losing trades
        wins = net_returns > 0
        losses = net_returns < 0

        num_trades = len(net_returns)
        num_wins = int(np.sum(wins))
        num_losses = int(np.sum(losses))

        win_rate = num_wins / num_trades if num_trades > 0 else 0

        # Profit factor = gross profit / gross loss
        total_wins = np.sum(net_returns[wins]) if num_wins > 0 else 0
        total_losses = np.abs(np.sum(net_returns[losses])) if num_losses > 0 else 0

        profit_factor = (
            total_wins / total_losses if total_losses > 0 else float("inf")
        )

        # Average win/loss
        avg_win = total_wins / num_wins if num_wins > 0 else 0
        avg_loss = total_losses / num_losses if num_losses > 0 else 0

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

        return {
            "win_rate": float(win_rate * 100),
            "num_trades": int(num_trades),
            "num_wins": int(num_wins),
            "num_losses": int(num_losses),
            "profit_factor": float(profit_factor),
            "avg_win_pct": float(avg_win * 100),
            "avg_loss_pct": float(avg_loss * 100),
            "win_loss_ratio": float(win_loss_ratio),
        }

    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        initial_capital: float = 10000.0,
    ) -> dict[str, float]:
        """Calculate all financial metrics.

        Args:
            y_true: Actual prices
            y_pred: Predicted prices
            initial_capital: Starting capital in USD

        Returns:
            Dictionary with all financial metrics:
            - P&L metrics (gross_pnl, net_pnl, total_costs, etc.)
            - Risk metrics (sharpe_ratio, sortino_ratio, max_drawdown_pct)
            - Trade statistics (win_rate, profit_factor, etc.)
        """
        logger.debug(f"Calculating financial metrics for {len(y_true)} predictions")

        # Get returns
        _, _, net_returns = self.calculate_returns(y_true, y_pred)

        # Calculate cumulative for drawdown
        if len(net_returns) > 0:
            net_cumulative = initial_capital * np.cumprod(1 + net_returns)
        else:
            net_cumulative = np.array([initial_capital])

        # Aggregate all metrics
        metrics: dict[str, float] = {}

        # P&L metrics
        metrics.update(self.calculate_pnl(y_true, y_pred, initial_capital))

        # Risk-adjusted returns
        metrics["sharpe_ratio"] = self.calculate_sharpe_ratio(net_returns)
        metrics["sortino_ratio"] = self.calculate_sortino_ratio(net_returns)

        # Drawdown
        dd_metrics = self.calculate_max_drawdown(net_cumulative)
        metrics["max_drawdown_pct"] = dd_metrics["max_drawdown_pct"]

        # Win rate
        win_metrics = self.calculate_win_rate(y_true, y_pred)
        metrics.update(win_metrics)

        logger.info(
            f"Financial metrics: Net P&L=${metrics['net_pnl']:.2f}, "
            f"Sharpe={metrics['sharpe_ratio']:.2f}, "
            f"Win Rate={metrics['win_rate']:.1f}%"
        )

        # Add warnings for suspicious results
        self._check_suspicious_results(metrics)

        return metrics

    def _check_suspicious_results(self, metrics: dict[str, float]) -> None:
        """Log warnings for results that seem too good to be true."""
        # Based on quant-analyst.md realistic expectations

        if metrics.get("win_rate", 0) > 60:
            logger.warning(
                f"Win rate ({metrics['win_rate']:.1f}%) is unusually high. "
                f"Realistic for BTC 1-min: 48-52%. Verify no data leakage."
            )

        if metrics.get("sharpe_ratio", 0) > 3.0:
            logger.warning(
                f"Sharpe ratio ({metrics['sharpe_ratio']:.2f}) is suspiciously high. "
                f"Realistic for BTC 1-min: 0.5-1.0. Check for bugs/leakage."
            )

        if metrics.get("max_drawdown_pct", -100) > -5:
            logger.warning(
                f"Max drawdown ({metrics['max_drawdown_pct']:.1f}%) is unusually small. "
                f"Ensure backtest covers volatile periods."
            )

        if metrics.get("profit_factor", 0) > 2.0:
            logger.warning(
                f"Profit factor ({metrics['profit_factor']:.2f}) is high. "
                f"Realistic: 1.0-1.2. Verify methodology."
            )

    @staticmethod
    def format_metrics(metrics: dict[str, float]) -> pd.DataFrame:
        """Format financial metrics for display.

        Args:
            metrics: Dictionary of metrics from calculate_all()

        Returns:
            Formatted DataFrame with Metric and Value columns
        """
        formatted = []

        # P&L section
        formatted.append(("Gross P&L", f"${metrics.get('gross_pnl', 0):,.2f}"))
        formatted.append(("Net P&L (after costs)", f"${metrics.get('net_pnl', 0):,.2f}"))
        formatted.append(("Total Trading Costs", f"${metrics.get('total_costs', 0):,.2f}"))
        formatted.append(("Net Return", f"{metrics.get('net_return_pct', 0):.2f}%"))
        formatted.append(("Cost Drag", f"{metrics.get('cost_drag_pct', 0):.2f}%"))

        # Risk metrics
        formatted.append(("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"))
        formatted.append(("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}"))
        formatted.append(("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%"))

        # Trade statistics
        formatted.append(("Win Rate", f"{metrics.get('win_rate', 0):.1f}%"))
        formatted.append(("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"))
        formatted.append(("Total Trades", f"{int(metrics.get('num_trades', 0))}"))
        formatted.append(("Winning Trades", f"{int(metrics.get('num_wins', 0))}"))
        formatted.append(("Losing Trades", f"{int(metrics.get('num_losses', 0))}"))
        formatted.append(("Avg Win", f"{metrics.get('avg_win_pct', 0):.4f}%"))
        formatted.append(("Avg Loss", f"{metrics.get('avg_loss_pct', 0):.4f}%"))

        return pd.DataFrame(formatted, columns=["Metric", "Value"])

    @staticmethod
    def validate_results(metrics: dict[str, float]) -> list[str]:
        """Validate backtest results against industry benchmarks.

        Returns list of warnings if results seem suspicious.
        Based on realistic expectations from quant-analyst.md.

        Args:
            metrics: Dictionary of financial metrics

        Returns:
            List of warning messages (empty if results look realistic)
        """
        warnings = []

        # Win rate
        if metrics.get("win_rate", 0) > 60:
            warnings.append(
                f"WARNING: Win rate {metrics['win_rate']:.1f}% is suspiciously high. "
                "Realistic for BTC 1-min: 48-55%. Check for data leakage."
            )

        # Sharpe ratio
        if metrics.get("sharpe_ratio", 0) > 3.0:
            warnings.append(
                f"WARNING: Sharpe ratio {metrics['sharpe_ratio']:.2f} is unrealistic. "
                "Expected: 0.5-2.0. Verify calculation and data."
            )

        # Max drawdown
        if metrics.get("max_drawdown_pct", -100) > -5:
            warnings.append(
                f"WARNING: Max drawdown {metrics['max_drawdown_pct']:.1f}% is too small. "
                "Ensure backtest covers volatile market periods."
            )

        # Profit factor
        if metrics.get("profit_factor", 0) > 2.0:
            warnings.append(
                f"WARNING: Profit factor {metrics['profit_factor']:.2f} is high. "
                "Expected: 1.0-1.5. Verify no overfitting."
            )

        return warnings
