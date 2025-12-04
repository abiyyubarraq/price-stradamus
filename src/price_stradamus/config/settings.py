"""Configuration settings using Pydantic."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="price-stradamus", description="Application name")
    env: str = Field(
        default="development", description="Environment (development/production)"
    )
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Database
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="price_stradamus", description="Database name")
    db_user: str = Field(default="postgres", description="Database user")
    db_password: str = Field(default="postgres", description="Database password")
    database_url: str = Field(
        default="",
        description="Full database URL (overrides individual DB settings)",
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def assemble_database_url(cls, v: str | None, info) -> str:
        """Build database URL from components if not provided."""
        if v:
            return v

        data = info.data
        return (
            f"postgresql+asyncpg://{data['db_user']}:{data['db_password']}"
            f"@{data['db_host']}:{data['db_port']}/{data['db_name']}"
        )

    # Binance API
    binance_api_key: str | None = Field(default=None, description="Binance API key")
    binance_api_secret: str | None = Field(
        default=None, description="Binance API secret"
    )
    binance_base_url: str = Field(
        default="https://api.binance.com",
        description="Binance API base URL",
    )

    # Trading Configuration
    default_symbol: str = Field(default="BTCUSDT", description="Default trading symbol")
    default_timeframe: str = Field(default="1m", description="Default timeframe")
    default_lookback_candles: int = Field(
        default=60,
        ge=1,
        description="Default lookback window in candles",
    )
    default_prediction_steps: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Default number of prediction steps",
    )

    # Model Configuration
    model_save_path: Path = Field(
        default=Path("./models"),
        description="Directory for saving models",
    )
    default_model: str = Field(default="nbeats", description="Default model to use")
    device: str = Field(default="cuda", description="Device for training (cuda/cpu)")

    # Training Configuration
    batch_size: int = Field(default=32, ge=1, description="Training batch size")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate")
    epochs: int = Field(default=100, ge=1, description="Training epochs")
    early_stopping_patience: int = Field(
        default=10,
        ge=1,
        description="Early stopping patience",
    )

    # AutoML Configuration
    automl_time_budget: int = Field(
        default=3600,
        ge=60,
        description="AutoML time budget in seconds",
    )
    automl_ensemble_size: int = Field(
        default=10,
        ge=1,
        description="AutoML ensemble size",
    )

    # Feature Engineering
    use_technical_indicators: bool = Field(
        default=True,
        description="Use technical indicators as features",
    )
    normalization_method: str = Field(
        default="minmax",
        description="Normalization method (minmax/standard/robust)",
    )

    @field_validator("normalization_method")
    @classmethod
    def validate_normalization_method(cls, v: str) -> str:
        """Validate normalization method."""
        allowed = ["minmax", "standard", "robust"]
        if v.lower() not in allowed:
            msg = f"normalization_method must be one of {allowed}, got {v}"
            raise ValueError(msg)
        return v.lower()

    # Evaluation
    min_train_size: int = Field(
        default=1000,
        ge=100,
        description="Minimum training set size",
    )
    test_size: float = Field(default=0.2, gt=0, lt=1, description="Test set proportion")
    validation_size: float = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="Validation set proportion",
    )

    # API Rate Limiting
    binance_rate_limit: int = Field(
        default=1200,
        description="Binance API rate limit (requests/minute)",
    )
    request_timeout: int = Field(
        default=30, ge=1, description="Request timeout in seconds"
    )
    max_retries: int = Field(default=3, ge=1, description="Max retry attempts")

    def __repr__(self) -> str:
        """String representation (hide sensitive data)."""
        return f"Settings(app_name={self.app_name}, env={self.env})"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance (for dependency injection)."""
    return settings
