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
    days: int | None = typer.Option(
        None, "--days", "-d", help="Number of days to fetch (relative)"
    ),
    start_date: str | None = typer.Option(
        None,
        "--start-date",
        help="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM). Mutually exclusive with --days.",
    ),
    end_date: str | None = typer.Option(
        None,
        "--end-date",
        help="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM). Defaults to now if not specified.",
    ),
) -> None:
    """Fetch historical OHLCV data from Binance.

    This command:
    1. Fetches data from Binance API for the specified time range
    2. Validates the data
    3. Saves to PostgreSQL database (skips duplicates)
    4. Displays a summary of the fetched data

    You can specify the date range in two ways:
    - Relative: Use --days (e.g., --days 30)
    - Absolute: Use --start-date and optionally --end-date

    Examples:
        # Fetch last 30 days (default)
        price-stradamus fetch

        # Fetch last 7 days
        price-stradamus fetch --days 7

        # Fetch from specific date to now
        price-stradamus fetch --start-date "2025-08-01"

        # Fetch between specific dates
        price-stradamus fetch --start-date "2025-08-01" --end-date "2025-12-04"

        # With symbol and timeframe
        price-stradamus fetch -s ETHUSDT -t 5m --start-date "2025-11-01"
    """

    async def _fetch():
        # Validate mutually exclusive options
        if days is not None and start_date is not None:
            console.print(
                "[red]✗ Error: Cannot use both --days and --start-date[/red]"
            )
            console.print(
                "[yellow]Use --days for relative time OR --start-date for absolute dates[/yellow]"
            )
            raise typer.Exit(1)

        # Parse dates
        start_dt: datetime
        end_dt: datetime

        if start_date is not None:
            # Absolute date mode
            try:
                start_dt = _parse_date(start_date)
                if end_date is not None:
                    end_dt = _parse_date(end_date)
                else:
                    end_dt = datetime.now(UTC)

                console.print(
                    f"[bold cyan]Fetching {symbol} {timeframe} data from {start_dt.strftime('%Y-%m-%d')} "
                    f"to {end_dt.strftime('%Y-%m-%d')}...[/bold cyan]"
                )
            except ValueError as e:
                console.print(f"[red]✗ Invalid date format: {e}[/red]")
                console.print(
                    "[yellow]Use format: YYYY-MM-DD or YYYY-MM-DD HH:MM[/yellow]"
                )
                raise typer.Exit(1) from e
        else:
            # Relative date mode (default)
            days_to_fetch = days if days is not None else 30
            end_dt = datetime.now(UTC)
            start_dt = end_dt - timedelta(days=days_to_fetch)

            console.print(
                f"[bold cyan]Fetching {symbol} {timeframe} data for {days_to_fetch} days...[/bold cyan]"
            )

        console.print(
            f"Date range: {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}"
        )

        try:
            # Fetch data from Binance
            async with BinanceDataFetcher() as fetcher:
                df = await fetcher.fetch_historical_range(
                    symbol=symbol,
                    interval=timeframe,
                    start_date=start_dt,
                    end_date=end_dt,
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


def _parse_date(date_str: str) -> datetime:
    """Parse date string in various formats.

    Args:
        date_str: Date string (YYYY-MM-DD or YYYY-MM-DD HH:MM)

    Returns:
        Parsed datetime object with UTC timezone

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
            # Parse as naive datetime
            dt = datetime.strptime(date_str, fmt)
            # Make timezone-aware (UTC)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue

    raise ValueError(
        f"Invalid date format: {date_str}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM"
    )
