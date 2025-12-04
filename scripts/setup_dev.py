#!/usr/bin/env python3
"""Development environment setup script.

This script sets up the development environment for Price Stradamus.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command.

    Args:
        cmd: Command and arguments
        check: Whether to check return code

    Returns:
        CompletedProcess instance
    """
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def main() -> int:
    """Set up development environment."""
    print("Setting up Price Stradamus development environment...\n")

    print(f"✓ Python version: {sys.version}")

    # Install dependencies
    print("\n1. Installing dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])

    # Install pre-commit hooks
    print("\n2. Installing pre-commit hooks...")
    run_command([sys.executable, "-m", "pre_commit", "install"])

    # Create necessary directories
    print("\n3. Creating directories...")
    directories = [
        "logs",
        "models",
        "data",
    ]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  Created: {dir_name}/")

    # Copy .env.example to .env if it doesn't exist
    print("\n4. Setting up .env file...")
    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("  Created .env from .env.example")
        print("  ⚠️  Please update .env with your configuration")
    elif env_file.exists():
        print("  .env already exists")
    else:
        print("  ⚠️  .env.example not found")

    # Check Docker
    print("\n5. Checking Docker...")
    result = subprocess.run(
        ["docker", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        print(f"  ✓ {result.stdout.strip()}")
        print("  To start PostgreSQL: cd docker && docker-compose up -d postgres")
    else:
        print("  ⚠️  Docker not found. Please install Docker to use database features.")

    # Summary
    print("\n" + "=" * 60)
    print("Development environment setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Update .env with your configuration")
    print("  2. Start PostgreSQL: cd docker && docker-compose up -d postgres")
    print("  3. Run migrations: alembic upgrade head")
    print("  4. Fetch data: price-stradamus fetch --days 7")
    print("  5. Train model: price-stradamus train --model nbeats --epochs 50")
    print("\nRun tests: pytest --cov")
    print("Format code: ruff format .")
    print("Lint code: ruff check --fix .")

    return 0


if __name__ == "__main__":
    sys.exit(main())
