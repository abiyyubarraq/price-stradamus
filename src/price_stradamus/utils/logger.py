"""Logging setup using Loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from price_stradamus.config import settings


def setup_logger() -> None:
    """Configure loguru logger with rotation and retention."""
    # Remove default handler
    logger.remove()

    # Console handler with custom format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.log_level,
        colorize=True,
    )

    # File handler for all logs
    logs_dir = Path(settings.model_save_path).parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        logs_dir / "price_stradamus.log",
        format=log_format,
        level="DEBUG",
        rotation="500 MB",
        retention="30 days",
        compression="zip",
    )

    # Separate file for errors
    logger.add(
        logs_dir / "errors.log",
        format=log_format,
        level="ERROR",
        rotation="100 MB",
        retention="90 days",
        compression="zip",
    )

    logger.info(f"Logger initialized (level={settings.log_level})")


# Initialize logger on import
setup_logger()


__all__ = ["logger", "setup_logger"]
