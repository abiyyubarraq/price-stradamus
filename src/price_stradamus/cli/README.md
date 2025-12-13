# Price Stradamus CLI

Command-line interface for the Bitcoin Price Prediction System.

---

## Quick Reference

### File Structure

```
cli/
├── __init__.py          # Main app entry point
├── __main__.py          # Module execution entry (python -m price_stradamus.cli)
├── data_commands.py     # Data operations (fetch)
├── model_commands.py    # ML command registration
├── info_commands.py     # System info (list-models, info)
├── window_commands.py   # Time window-based commands
├── commands/            # Model command implementations
│   ├── __init__.py      # Command exports
│   ├── train.py         # Train command implementation
│   ├── predict.py       # Predict command implementation
│   ├── evaluate.py      # Evaluate command implementation
│   └── compare.py       # Compare command implementation
├── CLI-COMMANDS.md      # Complete command reference
└── README.md            # This file
```

### Available Commands

| Command | Description | File |
|---------|-------------|------|
| `fetch` | Fetch historical data from Binance | [data_commands.py](data_commands.py) |
| `train` | Train a prediction model | [commands/train.py](commands/train.py) |
| `train-window` | Train with explicit time boundaries (prevents data leakage) | [window_commands.py](window_commands.py) |
| `predict` | Make predictions with trained model | [commands/predict.py](commands/predict.py) |
| `evaluate` | Evaluate model performance | [commands/evaluate.py](commands/evaluate.py) |
| `eval-window` | Evaluate on explicit time period (prevents data leakage) | [window_commands.py](window_commands.py) |
| `compare` | Compare multiple models | [commands/compare.py](commands/compare.py) |
| `list-models` | List available models | [info_commands.py](info_commands.py) |
| `info` | Show system information | [info_commands.py](info_commands.py) |

### Quick Examples

**Two ways to run commands:**

1. **Using installed CLI command** (after `pip install -e .`):
   ```bash
   price-stradamus fetch --days 30
   ```

2. **Using Python module** (works without installation):
   ```bash
   python -m price_stradamus.cli fetch --days 30
   ```

**Examples:**

```bash
# Fetch 30 days of data
price-stradamus fetch --days 30
# OR
python -m price_stradamus.cli fetch --days 30

# Train N-BEATS model
price-stradamus train --model nbeats
# OR
python -m price_stradamus.cli train --model nbeats

# Make predictions
price-stradamus predict --model models/nbeats_model.pkl --steps 10
# OR
python -m price_stradamus.cli predict --model models/nbeats_model.pkl --steps 10

# Evaluate model
price-stradamus evaluate --model-path models/nbeats_model.pkl
# OR
python -m price_stradamus.cli evaluate --model-path models/nbeats_model.pkl

# List all models
price-stradamus list-models
# OR
python -m price_stradamus.cli list-models

# Train with time window (prevents data leakage)
price-stradamus train-window \
    --model  \
    --train-start "2025-08-01" \
    --train-end "2025-10-31"
# OR
python -m price_stradamus.cli train-window \
    --model  \
    --train-start "2025-08-01" \
    --train-end "2025-10-31"

# Evaluate on time window (prevents data leakage)
price-stradamus eval-window \
    --model-path models/_aug_oct.pkl \
    --test-start "2025-11-01" \
    --test-end "2025-12-07" \
    --train-end "2025-10-31"
# OR
python -m price_stradamus.cli eval-window \
    --model-path models/_aug_oct.pkl \
    --test-start "2025-11-01" \
    --test-end "2025-12-07" \
    --train-end "2025-10-31"
```

---

## Documentation

- **[CLI-COMMANDS.md](CLI-COMMANDS.md)** - Complete CLI reference with Bash/PowerShell examples
- **[../../QUICK-START.md](../../QUICK-START.md)** - Quick start guide
- **[../../context/LEARNING-GUIDE.md](../../context/LEARNING-GUIDE.md)** - Complete system learning guide
- **[../../REFACTORING-SUMMARY.md](../../REFACTORING-SUMMARY.md)** - CLI refactoring details

---

## Architecture

### Registration Pattern

Each command module has a registration function:

```python
def register_data_commands(app: typer.Typer) -> None:
    """Register all data-related commands."""
    app.command()(fetch)
```

Called from `__init__.py`:

```python
app = typer.Typer()
register_data_commands(app)
register_model_commands(app)
register_info_commands(app)

def main() -> None:
    """Main entry point for the CLI."""
    app()
```

The `__main__.py` file enables module execution:

```python
from price_stradamus.cli import main

if __name__ == "__main__":
    main()
```

This allows running via `python -m price_stradamus.cli`.

**Benefits**:
- Automatic command discovery
- Type-safe registration
- Clear separation of concerns
- Easy to add new commands
- Two execution methods (installed CLI or Python module)

### Adding a New Command

1. **Choose the appropriate location** based on domain:
   - Data operations → `data_commands.py`
   - Model operations → Create new file in `commands/` directory
   - System info → `info_commands.py`

2. **For model commands, create a new file** in `commands/`:
   ```python
   # commands/my_command.py
   """My command - Brief description."""

   from __future__ import annotations

   import typer
   from rich.console import Console

   console = Console()

   def my_command(
       option1: str = typer.Option(..., "--option1", "-o1", help="Description"),
   ) -> None:
       """Command description.

       Detailed explanation of what the command does.
       """
       # Implementation
       pass
   ```

3. **Export from `commands/__init__.py`**:
   ```python
   from price_stradamus.cli.commands.my_command import my_command

   __all__ = ["train", "predict", "evaluate", "compare", "my_command"]
   ```

4. **Register in `model_commands.py`**:
   ```python
   from price_stradamus.cli.commands import compare, evaluate, predict, train, my_command

   def register_model_commands(app: typer.Typer) -> None:
       app.command()(train)
       app.command()(predict)
       app.command()(evaluate)
       app.command()(compare)
       app.command()(my_command)  # Add here
   ```

5. **Test**:
   ```bash
   # Using installed CLI
   price-stradamus --help  # Should show new command
   price-stradamus my-command --help  # Should show command help

   # OR using Python module
   python -m price_stradamus.cli --help
   python -m price_stradamus.cli my-command --help
   ```

---

## Code Quality Standards

All CLI code follows [CLAUDE.md](../../CLAUDE.md) standards:

- ✅ Type hints on all functions
- ✅ Google-style docstrings
- ✅ Async for I/O operations
- ✅ Error handling with specific exceptions
- ✅ Ruff formatting
- ✅ Pyright type checking

---

## Testing

Run tests:
```bash
# All CLI tests
pytest tests/test_cli/ -v

# Specific command test
pytest tests/test_cli/test_data_commands.py -v

# With coverage
pytest tests/test_cli/ --cov=price_stradamus.cli
```

Manual testing:
```bash
# Test command registration
python -c "from price_stradamus.cli import app; app(['--help'])"

# Test specific command
python -c "from price_stradamus.cli import app; app(['list-models'])"

# Test module execution
python -m price_stradamus.cli --help
python -m price_stradamus.cli list-models
```

---

## Troubleshooting

### Command not found

**Symptom**: `price-stradamus: command not found`

**Solution**:
```bash
# Option 1: Use Python module (always works)
python -m price_stradamus.cli <command>

# Option 2: Ensure virtual environment is activated
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Reinstall if needed
uv pip install -e ".[dev]"
```

### Import errors

**Symptom**: `ModuleNotFoundError: No module named 'price_stradamus'`

**Solution**:
```bash
# Reinstall in editable mode
uv pip install -e ".[dev]"
```

### Database connection errors

**Symptom**: `Connection refused` or `database does not exist`

**Solution**:
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Start if not running
cd docker && docker-compose up -d postgres
```

---

## Performance Considerations

### Lazy Imports

Models are imported at the module level to ensure registration:
```python
# These imports trigger @ModelRegistry.register() decorators
from price_stradamus.models.neural.nbeats import NBEATSModel  # noqa: F401
from price_stradamus.models.neural.lstm import LSTMModel  # noqa: F401
# ... etc
```

**Impact**: ~1-2 second startup time due to model imports.

**Why acceptable**:
- Only happens once per command invocation
- Ensures all models are registered
- Negligible compared to actual operation time (fetching, training, etc.)

### Async Operations

Data fetching and database operations use `asyncio.run()`:
```python
async def _fetch():
    async with BinanceDataFetcher() as fetcher:
        df = await fetcher.fetch_historical_range(...)

asyncio.run(_fetch())
```

**Benefits**:
- Concurrent API requests
- Non-blocking I/O
- Better performance for bulk operations

---

## Security Notes

### API Keys

Never hardcode API keys in CLI commands. Use environment variables:

```python
# ✅ Good
api_key = settings.binance_api_key

# ❌ Bad
api_key = "hardcoded_key_123"
```

### Path Traversal

Always validate user-provided paths:

```python
from pathlib import Path

def validate_path(user_path: str) -> Path:
    path = Path(user_path).resolve()
    # Validate path is within allowed directory
    # ... validation logic ...
    return path
```

---

## Maintenance

### Regular Tasks

**Weekly**:
- Check for outdated dependencies: `uv pip list --outdated`
- Review error logs: `tail -n 100 logs/app.log`

**Monthly**:
- Update CLI-COMMANDS.md if commands change
- Review and update examples
- Test all commands end-to-end

**On Feature Addition**:
- Update CLI-COMMANDS.md
- Add command to this README
- Update QUICK-START.md examples
- Write tests

---

## Resources

- **CLI Framework**: [Typer Documentation](https://typer.tiangolo.com/)
- **Rich Output**: [Rich Documentation](https://rich.readthedocs.io/)
- **Async Python**: [asyncio Documentation](https://docs.python.org/3/library/asyncio.html)

---

**Last Updated**: 2024-12-05
**Maintainer**: Price Stradamus Development Team
