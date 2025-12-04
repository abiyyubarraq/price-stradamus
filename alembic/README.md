# Alembic Database Migrations

This directory contains database migration scripts for Price Stradamus using Alembic with async SQLAlchemy support.

## Prerequisites

1. Ensure PostgreSQL is running:
   ```bash
   cd docker
   docker-compose up -d postgres
   ```

2. Ensure environment variables are set in `.env`:
   ```bash
   DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/price_stradamus
   ```

## Common Commands

### Apply Migrations (Upgrade to Latest)
```bash
alembic upgrade head
```

### Rollback Last Migration
```bash
alembic downgrade -1
```

### View Current Revision
```bash
alembic current
```

### View Migration History
```bash
alembic history
```

### Create New Migration (Auto-generate)
```bash
alembic revision --autogenerate -m "Description of changes"
```

### Create Empty Migration
```bash
alembic revision -m "Description of changes"
```

## Migration Files

- `001_initial_schema.py` - Initial database schema with 4 tables:
  - `market_data.ohlcv_raw` - Raw OHLCV candle data
  - `market_data.features` - Computed technical indicators (JSONB)
  - `ml_data.predictions` - Model predictions and actuals
  - `ml_data.model_metadata` - Model training metadata and hyperparameters

## Database Schema

### market_data.ohlcv_raw
Stores raw OHLCV (Open, High, Low, Close, Volume) data from Binance.

**Columns**:
- `id` (UUID, PK)
- `timestamp` (DateTime with TZ, indexed)
- `symbol` (String, indexed, e.g., "BTCUSDT")
- `timeframe` (String, indexed, e.g., "1m", "5m", "1h")
- `open`, `high`, `low`, `close`, `volume` (Numeric)
- `quote_volume`, `num_trades` (Optional)
- `created_at` (DateTime with TZ)

**Unique Constraint**: (symbol, timeframe, timestamp)

### market_data.features
Stores computed technical indicators using JSONB for flexibility.

**Columns**:
- `id` (UUID, PK)
- `timestamp`, `symbol`, `timeframe` (indexed)
- `features` (JSONB, GIN indexed) - All indicators in one column
- `feature_version` (String) - Track computation version
- `created_at` (DateTime with TZ)

**Unique Constraint**: (symbol, timeframe, timestamp)

### ml_data.predictions
Stores model predictions and actual values for evaluation.

**Columns**:
- `id` (UUID, PK)
- `model_name` (String, indexed)
- `prediction_timestamp` (DateTime, indexed) - When prediction was made
- `target_timestamp` (DateTime, indexed) - What time was predicted
- `symbol`, `timeframe`
- `steps_ahead` (Integer) - 1, 2, 3, 4, or 5 steps
- `predicted_close`, `actual_close` (Numeric)
- `predicted_direction`, `actual_direction` (String) - 'up', 'down', 'flat'
- `error` (Numeric) - Absolute error
- `created_at` (DateTime with TZ)

### ml_data.model_metadata
Stores model training metadata and hyperparameters.

**Columns**:
- `id` (UUID, PK)
- `model_name` (String, unique, indexed)
- `model_type` (String) - 'neural', 'classical', 'ml', 'automl'
- `model_class` (String) - e.g., 'NBEATSModel'
- `hyperparameters` (JSONB, GIN indexed)
- `trained_at` (DateTime, indexed)
- `training_duration_seconds` (Numeric)
- `metrics` (JSONB, GIN indexed) - MAE, RMSE, MAPE, etc.
- `data_start`, `data_end` (DateTime) - Training data range
- `num_samples` (Integer)
- `symbol`, `timeframe`
- `feature_list` (JSONB) - List of features used
- `version`, `file_path`, `notes`
- `created_at`, `updated_at` (DateTime with TZ)

## Troubleshooting

### Connection Issues
If you see connection errors:
1. Check PostgreSQL is running: `docker-compose ps`
2. Verify connection string in `.env`
3. Test connection: `psql -h localhost -U postgres -d price_stradamus`

### Migration Conflicts
If you see "revision not found" errors:
1. Check current revision: `alembic current`
2. View history: `alembic history`
3. Manually resolve by editing `alembic_version` table

### Reset Database (CAUTION: Deletes all data)
```bash
# Downgrade all migrations
alembic downgrade base

# Or drop and recreate database
docker-compose down -v
docker-compose up -d postgres

# Re-run migrations
alembic upgrade head
```

## Notes

- All tables use UUID primary keys for better distribution
- Timestamps are timezone-aware (UTC recommended)
- JSONB columns use GIN indexes for fast queries
- Composite indexes optimize common query patterns
- Migrations support both upgrade and downgrade

## Next Steps

After running migrations:
1. Verify tables: `psql -h localhost -U postgres -d price_stradamus -c "\dt market_data.*"`
2. Check schema: `psql -h localhost -U postgres -d price_stradamus -c "\d market_data.ohlcv_raw"`
3. Verify indexes: `psql -h localhost -U postgres -d price_stradamus -c "\di market_data.*"`
