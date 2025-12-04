"""Data management CLI commands for Price Stradamus.

This module handles all data-related operations:
- Fetching historical OHLCV data from Binance
- Saving data to PostgreSQL database
- Data validation and summaries
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import typer
from loguru import logger
from rich.console import Console

from price_stradamus.config.settings import settings
from price_stradamus.data.database import DatabaseManager
from price_stradamus.data.fetcher import BinanceDataFetcher

# Create console for output
console = Console()


def register_data_commands(app: typer.Typer) -> None:
    """Register all data-related commands to the main app.

    Args:
        app: Main Typer application instance
    """
    app.command()(fetch)


def fetch(
    symbol: str = typer.Option("BTCUSDT", "--symbol", "-s", help="Trading symbol"),
    timeframe: str = typer.Option(
        "1m", "--timeframe", "-t", help="Timeframe (1m, 5m, 1h, 1d)"
    ),
    days: int = typer.Option(30, "--days", "-d", help="Number of days to fetch"),
) -> None:
    """Fetch historical OHLCV data from Binance.

    This command:
    1. Fetches data from Binance API for the specified time range
    2. Validates the data
    3. Saves to PostgreSQL database (skips duplicates)
    4. Displays a summary of the fetched data

    Examples:
        # Fetch 30 days of 1-minute BTC data (default)
        price-stradamus fetch

        # Fetch 7 days of 5-minute ETH data
        price-stradamus fetch --symbol ETHUSDT --timeframe 5m --days 7

        # Short form
        price-stradamus fetch -s BTCUSDT -t 1h -d 90
    """

    async def _fetch():
        console.print(
            f"[bold cyan]Fetching {symbol} {timeframe} data for {days} days...[/bold cyan]"
        )

        # Calculate date range (timezone-aware)
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=days)

        console.print(
            f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        try:
            # Fetch data from Binance
            async with BinanceDataFetcher() as fetcher:
                df = await fetcher.fetch_historical_range(
                    symbol=symbol,
                    interval=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )

            console.print(f"[green]✓ Fetched {len(df)} candles from Binance[/green]")

            # Save to database
            console.print("Saving to database...")
            db = DatabaseManager(settings.database_url)
            await db.initialize()

            rows_saved = await db.save_ohlcv(df, symbol, timeframe)

            await db.close()

            console.print(
                f"[green]✓ Saved {rows_saved} new candles to database "
                f"(skipped {len(df) - rows_saved} duplicates)[/green]"
            )

            # Show data summary
            console.print("\n[bold]Data Summary:[/bold]")
            console.print(f"  First candle: {df['timestamp'].iloc[0]}")
            console.print(f"  Last candle:  {df['timestamp'].iloc[-1]}")
            console.print(
                f"  Price range:  ${df['close'].min():.2f} - ${df['close'].max():.2f}"
            )

        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            logger.exception("Fetch command failed")
            raise typer.Exit(1) from e

    asyncio.run(_fetch())
