"""Quick script to view fetched data from the database."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from rich.console import Console
from rich.table import Table

from price_stradamus.config import settings


async def view_klines_data(limit: int = 20) -> None:
    """View the most recent klines data.

    Args:
        limit: Number of records to display
    """
    console = Console()

    try:
        # Connect to database (convert SQLAlchemy URL to asyncpg format)
        db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(db_url)

        # Get total count
        count = await conn.fetchval("SELECT COUNT(*) FROM market_data.ohlcv_raw")
        console.print(f"\n[bold cyan]Total records in database:[/bold cyan] {count:,}")

        # Get recent data
        query = """
            SELECT
                timestamp,
                symbol,
                timeframe,
                open,
                high,
                low,
                close,
                volume
            FROM market_data.ohlcv_raw
            ORDER BY timestamp DESC
            LIMIT $1
        """

        rows = await conn.fetch(query, limit)

        if not rows:
            console.print("[yellow]No data found in database.[/yellow]")
            return

        # Create Rich table
        table = Table(title=f"Recent {limit} OHLCV Records", show_header=True)
        table.add_column("Timestamp", style="cyan")
        table.add_column("Symbol", style="magenta")
        table.add_column("Timeframe", style="blue")
        table.add_column("Open", justify="right", style="green")
        table.add_column("High", justify="right", style="green")
        table.add_column("Low", justify="right", style="red")
        table.add_column("Close", justify="right", style="yellow")
        table.add_column("Volume", justify="right", style="white")

        for row in rows:
            table.add_row(
                str(row["timestamp"]),
                row["symbol"],
                row["timeframe"],
                f"{row['open']:.2f}",
                f"{row['high']:.2f}",
                f"{row['low']:.2f}",
                f"{row['close']:.2f}",
                f"{row['volume']:.2f}",
            )

        console.print(table)

        # Show date range
        first_date = await conn.fetchval(
            "SELECT MIN(timestamp) FROM market_data.ohlcv_raw"
        )
        last_date = await conn.fetchval(
            "SELECT MAX(timestamp) FROM market_data.ohlcv_raw"
        )

        console.print("\n[bold]Data Range:[/bold]")
        console.print(f"  From: {first_date}")
        console.print(f"  To:   {last_date}")

        await conn.close()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


async def view_statistics() -> None:
    """View statistical summary of the data."""
    console = Console()

    try:
        # Connect to database (convert SQLAlchemy URL to asyncpg format)
        db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(db_url)

        query = """
            SELECT
                symbol,
                timeframe,
                COUNT(*) as record_count,
                MIN(timestamp) as first_timestamp,
                MAX(timestamp) as last_timestamp,
                AVG(close) as avg_price,
                MIN(close) as min_price,
                MAX(close) as max_price,
                SUM(volume) as total_volume
            FROM market_data.ohlcv_raw
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """

        rows = await conn.fetch(query)

        table = Table(title="Data Statistics by Symbol/Timeframe", show_header=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Timeframe", style="blue")
        table.add_column("Records", justify="right", style="yellow")
        table.add_column("First", style="green")
        table.add_column("Last", style="green")
        table.add_column("Avg Price", justify="right", style="white")
        table.add_column("Min Price", justify="right", style="red")
        table.add_column("Max Price", justify="right", style="green")
        table.add_column("Total Volume", justify="right", style="magenta")

        for row in rows:
            table.add_row(
                row["symbol"],
                row["timeframe"],
                f"{row['record_count']:,}",
                str(row["first_timestamp"]),
                str(row["last_timestamp"]),
                f"${row['avg_price']:.2f}",
                f"${row['min_price']:.2f}",
                f"${row['max_price']:.2f}",
                f"{row['total_volume']:.2f}",
            )

        console.print(table)

        await conn.close()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


async def main() -> None:
    """Main entry point."""
    console = Console()

    console.print("[bold green]Price Stradamus - Data Viewer[/bold green]\n")

    # View recent data
    await view_klines_data(limit=20)

    # View statistics
    console.print()
    await view_statistics()


if __name__ == "__main__":
    asyncio.run(main())
