"""Initial database schema

Revision ID: 001
Revises:
Create Date: 2025-12-03 00:00:00.000000

Creates the initial database schema for Price Stradamus:
- market_data.ohlcv_raw: Raw OHLCV candle data from Binance
- market_data.features: Computed technical indicators
- ml_data.predictions: Model predictions and actuals
- ml_data.model_metadata: Model training metadata and hyperparameters
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply migration changes."""

    # Create schemas (they should exist from init.sql, but ensure they exist)
    op.execute("CREATE SCHEMA IF NOT EXISTS market_data")
    op.execute("CREATE SCHEMA IF NOT EXISTS ml_data")

    # Enable extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm"')

    # Table 1: market_data.ohlcv_raw
    # Stores raw OHLCV (Open, High, Low, Close, Volume) candle data
    op.create_table(
        "ohlcv_raw",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("symbol", sa.String(20), nullable=False, index=True),
        sa.Column("timeframe", sa.String(10), nullable=False, index=True),
        sa.Column("open", sa.Numeric(20, 8), nullable=False),
        sa.Column("high", sa.Numeric(20, 8), nullable=False),
        sa.Column("low", sa.Numeric(20, 8), nullable=False),
        sa.Column("close", sa.Numeric(20, 8), nullable=False),
        sa.Column("volume", sa.Numeric(20, 8), nullable=False),
        sa.Column("quote_volume", sa.Numeric(20, 8), nullable=True),
        sa.Column("num_trades", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint(
            "symbol",
            "timeframe",
            "timestamp",
            name="uq_ohlcv_symbol_timeframe_timestamp",
        ),
        schema="market_data",
    )

    # Create composite index for efficient querying
    op.create_index(
        "ix_ohlcv_symbol_timeframe_timestamp",
        "ohlcv_raw",
        ["symbol", "timeframe", "timestamp"],
        schema="market_data",
    )

    # Table 2: market_data.features
    # Stores computed technical indicators
    # Using wide format with JSONB for flexibility
    op.create_table(
        "features",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("symbol", sa.String(20), nullable=False, index=True),
        sa.Column("timeframe", sa.String(10), nullable=False, index=True),
        sa.Column(
            "features", postgresql.JSONB, nullable=False
        ),  # All features in one JSONB column
        sa.Column(
            "feature_version", sa.String(50), nullable=True
        ),  # Track feature computation version
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint(
            "symbol",
            "timeframe",
            "timestamp",
            name="uq_features_symbol_timeframe_timestamp",
        ),
        schema="market_data",
    )

    # Create composite index
    op.create_index(
        "ix_features_symbol_timeframe_timestamp",
        "features",
        ["symbol", "timeframe", "timestamp"],
        schema="market_data",
    )

    # Create GIN index on JSONB for fast feature queries
    op.create_index(
        "ix_features_jsonb",
        "features",
        ["features"],
        postgresql_using="gin",
        schema="market_data",
    )

    # Table 3: ml_data.predictions
    # Stores model predictions and actual values for evaluation
    op.create_table(
        "predictions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column("model_name", sa.String(100), nullable=False, index=True),
        sa.Column(
            "prediction_timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            index=True,
        ),  # When prediction was made
        sa.Column(
            "target_timestamp", sa.DateTime(timezone=True), nullable=False, index=True
        ),  # What timestamp was predicted
        sa.Column("symbol", sa.String(20), nullable=False, index=True),
        sa.Column("timeframe", sa.String(10), nullable=False),
        sa.Column(
            "steps_ahead", sa.Integer, nullable=False
        ),  # How many steps ahead (1, 2, 3, 4, 5)
        sa.Column("predicted_close", sa.Numeric(20, 8), nullable=False),
        sa.Column(
            "actual_close", sa.Numeric(20, 8), nullable=True
        ),  # Filled in later when actual data available
        sa.Column(
            "predicted_direction", sa.String(10), nullable=True
        ),  # 'up', 'down', 'flat'
        sa.Column("actual_direction", sa.String(10), nullable=True),
        sa.Column(
            "error", sa.Numeric(20, 8), nullable=True
        ),  # Absolute error when actual is available
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        schema="ml_data",
    )

    # Create composite indexes for efficient querying
    op.create_index(
        "ix_predictions_model_timestamp",
        "predictions",
        ["model_name", "prediction_timestamp"],
        schema="ml_data",
    )

    op.create_index(
        "ix_predictions_target",
        "predictions",
        ["symbol", "timeframe", "target_timestamp"],
        schema="ml_data",
    )

    # Table 4: ml_data.model_metadata
    # Stores model training metadata, hyperparameters, and performance metrics
    op.create_table(
        "model_metadata",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "model_name", sa.String(100), nullable=False, unique=True, index=True
        ),
        sa.Column(
            "model_type", sa.String(50), nullable=False
        ),  # 'neural', 'classical', 'ml', 'automl'
        sa.Column(
            "model_class", sa.String(100), nullable=False
        ),  # e.g., 'NBEATSModel', 'LSTMModel'
        sa.Column(
            "hyperparameters", postgresql.JSONB, nullable=False
        ),  # All hyperparameters
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("training_duration_seconds", sa.Numeric(10, 2), nullable=True),
        sa.Column("metrics", postgresql.JSONB, nullable=False),  # MAE, RMSE, MAPE, etc.
        sa.Column(
            "data_start", sa.DateTime(timezone=True), nullable=True
        ),  # Training data start
        sa.Column(
            "data_end", sa.DateTime(timezone=True), nullable=True
        ),  # Training data end
        sa.Column("num_samples", sa.Integer, nullable=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timeframe", sa.String(10), nullable=False),
        sa.Column(
            "feature_list", postgresql.JSONB, nullable=True
        ),  # List of features used
        sa.Column(
            "version", sa.String(20), nullable=True
        ),  # Model version (e.g., "1.0.0")
        sa.Column(
            "file_path", sa.String(500), nullable=True
        ),  # Path to saved model file
        sa.Column("notes", sa.Text, nullable=True),  # Additional notes
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
            onupdate=sa.text("CURRENT_TIMESTAMP"),
        ),
        schema="ml_data",
    )

    # Create indexes
    op.create_index(
        "ix_model_metadata_trained_at",
        "model_metadata",
        ["trained_at"],
        schema="ml_data",
    )

    op.create_index(
        "ix_model_metadata_type",
        "model_metadata",
        ["model_type"],
        schema="ml_data",
    )

    # Create GIN indexes on JSONB columns
    op.create_index(
        "ix_model_metadata_hyperparameters",
        "model_metadata",
        ["hyperparameters"],
        postgresql_using="gin",
        schema="ml_data",
    )

    op.create_index(
        "ix_model_metadata_metrics",
        "model_metadata",
        ["metrics"],
        postgresql_using="gin",
        schema="ml_data",
    )


def downgrade() -> None:
    """Revert migration changes."""

    # Drop tables in reverse order
    op.drop_table("model_metadata", schema="ml_data")
    op.drop_table("predictions", schema="ml_data")
    op.drop_table("features", schema="market_data")
    op.drop_table("ohlcv_raw", schema="market_data")

    # Optionally drop schemas (commented out to be safe)
    # op.execute("DROP SCHEMA IF EXISTS ml_data CASCADE")
    # op.execute("DROP SCHEMA IF EXISTS market_data CASCADE")
