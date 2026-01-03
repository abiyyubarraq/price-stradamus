"""Stateful feature engineering with fit-transform pattern.

This module provides feature engineering that PREVENTS DATA LEAKAGE by:
1. Fitting statistics (mean, std, etc.) only on training data
2. Applying same statistics to transform validation/test data
3. Optional feature selection to reduce dimensionality

CRITICAL: Always use this class instead of FeatureEngineer for training/evaluation
to ensure your backtesting results match real-world performance.

Example:
    from price_stradamus.utils.feature_selection import CORE_FEATURES

    # Use only core features (recommended for dimensionality reduction)
    engineer = StatefulFeatureEngineer(warmup_period=200, feature_cols=CORE_FEATURES)

    # Fit on training data ONLY
    train_features = engineer.fit_transform(train_df)

    # Transform test data using training statistics (NO LEAKAGE)
    test_features = engineer.transform(test_df)

    # Save statistics for reproducible inference
    engineer.save_statistics("models/feature_stats.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta
from darts import TimeSeries
from loguru import logger

from price_stradamus.config.constants import TECHNICAL_INDICATORS


@dataclass
class FeatureStatistics:
    """Statistics learned from training data for feature transformations.

    These statistics are fitted on training data only and then applied
    to transform validation/test data without refitting.
    """

    # Normalization statistics per feature
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    mins: dict[str, float] = field(default_factory=dict)
    maxs: dict[str, float] = field(default_factory=dict)

    # For clipping extreme values
    percentile_1: dict[str, float] = field(default_factory=dict)
    percentile_99: dict[str, float] = field(default_factory=dict)

    is_fitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "means": self.means,
            "stds": self.stds,
            "mins": self.mins,
            "maxs": self.maxs,
            "percentile_1": self.percentile_1,
            "percentile_99": self.percentile_99,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureStatistics:
        """Create from dictionary."""
        return cls(
            means=data.get("means", {}),
            stds=data.get("stds", {}),
            mins=data.get("mins", {}),
            maxs=data.get("maxs", {}),
            percentile_1=data.get("percentile_1", {}),
            percentile_99=data.get("percentile_99", {}),
            is_fitted=data.get("is_fitted", False),
        )


class StatefulFeatureEngineer:
    """Feature engineering with fit-transform pattern to prevent data leakage.

    CRITICAL: This class ensures features are computed WITHOUT future data:
    1. fit(): Learn statistics from training data only
    2. transform(): Apply learned statistics to any data
    3. fit_transform(): Convenience method for training data

    The key difference from FeatureEngineer:
    - Technical indicators are generated on each split separately
    - Normalization statistics are fitted on training data only
    - Warmup period is dropped to avoid NaN contamination

    Example:
        # Use all features (default)
        engineer = StatefulFeatureEngineer(warmup_period=200)
        train_features = engineer.fit_transform(train_df)

        # Use only CORE_FEATURES (recommended for dimensionality reduction)
        from price_stradamus.utils.feature_selection import CORE_FEATURES
        engineer = StatefulFeatureEngineer(warmup_period=200, feature_cols=CORE_FEATURES)
        train_features = engineer.fit_transform(train_df)

        # For validation/test data (uses training statistics)
        val_features = engineer.transform(val_df)
        test_features = engineer.transform(test_df)

        # Save for reproducible inference
        engineer.save_statistics("models/nbeats_features.json")

    Attributes:
        warmup_period: Number of initial rows to drop after feature generation.
                      Technical indicators (RSI, MACD, etc.) produce NaN for
                      the first N periods. Default 200 covers most indicators.
        statistics: FeatureStatistics learned from training data.
        feature_list: List of generated feature names.
    """

    def __init__(
        self,
        warmup_period: int = 200,
        feature_cols: list[str] | None = None,
    ):
        """Initialize stateful feature engineer.

        Args:
            warmup_period: Number of initial rows to drop after feature
                          generation (technical indicators need history).
                          Default 200 covers most indicator lookback windows.
                          Set higher if using SMA_200 or similar long windows.
            feature_cols: Optional list of feature names to keep after generation.
                         If None, all generated features are kept.
                         Base OHLCV columns are always preserved.
                         Example: Pass CORE_FEATURES to use only core features.
        """
        self.warmup_period = warmup_period
        self.feature_cols = feature_cols
        self.statistics = FeatureStatistics()
        self.feature_list: list[str] = []
        self._original_columns: set[str] = set()

        mode = f"{len(feature_cols)} selected" if feature_cols else "all"
        logger.info(
            f"StatefulFeatureEngineer initialized (warmup_period={warmup_period}, "
            f"feature_cols={mode})"
        )

    def fit(self, df: pd.DataFrame) -> StatefulFeatureEngineer:
        """Fit feature statistics on training data only.

        This method:
        1. Generates raw technical indicators
        2. Drops warmup period (first N rows with NaN)
        3. Learns normalization statistics from remaining data

        Args:
            df: Training DataFrame with OHLCV data

        Returns:
            Self for method chaining

        Raises:
            ValueError: If required columns missing
        """
        self._validate_ohlcv(df)

        logger.info(f"Fitting feature statistics on {len(df)} training samples")

        # Store original columns
        self._original_columns = set(df.columns)

        # Generate raw features
        df_features = self._generate_raw_features(df)

        # Drop warmup period
        if len(df_features) <= self.warmup_period:
            raise ValueError(
                f"Training data ({len(df_features)} rows) must be larger than "
                f"warmup_period ({self.warmup_period}). Provide more data or "
                f"reduce warmup_period."
            )

        df_features = df_features.iloc[self.warmup_period :].copy()

        # Filter features if feature_cols specified
        df_features = self._filter_features(df_features)

        # Learn statistics from training data
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        ohlcv_cols = {"timestamp", "open", "high", "low", "close", "volume"}

        feature_cols = [col for col in numeric_cols if col not in ohlcv_cols]

        for col in feature_cols:
            values = df_features[col].dropna()
            if len(values) == 0:
                continue

            self.statistics.means[col] = float(values.mean())
            self.statistics.stds[col] = float(values.std())
            self.statistics.mins[col] = float(values.min())
            self.statistics.maxs[col] = float(values.max())
            self.statistics.percentile_1[col] = float(values.quantile(0.01))
            self.statistics.percentile_99[col] = float(values.quantile(0.99))

        self.statistics.is_fitted = True
        self.feature_list = feature_cols

        logger.info(f"Fitted statistics for {len(self.feature_list)} features")
        return self

    def transform(
        self,
        df: pd.DataFrame,
        normalize: bool = False,
        drop_warmup: bool = True,
    ) -> pd.DataFrame:
        """Transform data using fitted statistics.

        Args:
            df: DataFrame to transform (can be train, val, or test)
            normalize: Whether to normalize features using training stats.
                      Default False - most Darts models handle normalization.
            drop_warmup: Whether to drop warmup period. Set False if you need
                        all rows (e.g., for visualization).

        Returns:
            DataFrame with features (warmup period removed by default)

        Raises:
            ValueError: If fit() not called first
        """
        if not self.statistics.is_fitted:
            raise ValueError(
                "Must call fit() before transform(). "
                "Use fit_transform() for training data."
            )

        self._validate_ohlcv(df)

        logger.debug(f"Transforming {len(df)} samples")

        # Generate raw features
        df_features = self._generate_raw_features(df)

        # Filter features if feature_cols specified
        df_features = self._filter_features(df_features)

        # Drop warmup period if requested
        if drop_warmup:
            if len(df_features) <= self.warmup_period:
                logger.warning(
                    f"Data ({len(df_features)} rows) is smaller than warmup_period "
                    f"({self.warmup_period}). Returning empty DataFrame."
                )
                return df_features.iloc[0:0]
            df_features = df_features.iloc[self.warmup_period :].copy()

        # Normalize using training statistics if requested
        if normalize:
            df_features = self._normalize_with_training_stats(df_features)

        return df_features

    def fit_transform(
        self,
        df: pd.DataFrame,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Fit and transform training data in one step.

        Args:
            df: Training DataFrame
            normalize: Whether to normalize features

        Returns:
            Transformed DataFrame
        """
        self.fit(df)
        return self.transform(df, normalize=normalize)

    def _validate_ohlcv(self, df: pd.DataFrame) -> None:
        """Validate DataFrame has required OHLCV columns."""
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df.empty:
            raise ValueError("DataFrame is empty")

    def _should_generate(self, feature_name: str) -> bool:
        """Check if a feature should be generated based on feature_cols.

        Args:
            feature_name: Name of the feature to check

        Returns:
            True if feature should be generated, False otherwise
        """
        # If no filter specified, generate all features
        if self.feature_cols is None:
            return True

        # Check if this specific feature is requested
        return feature_name in self.feature_cols

    def _needs_any(self, *feature_names: str) -> bool:
        """Check if any of the given features are needed.

        Useful for generating intermediate features that multiple final features depend on.

        Args:
            *feature_names: Variable number of feature names to check

        Returns:
            True if any of the features should be generated
        """
        if self.feature_cols is None:
            return True

        return any(fname in self.feature_cols for fname in feature_names)

    def _generate_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate raw technical indicators without normalization.

        This mirrors FeatureEngineer but returns raw values.
        Features are computed on the provided data only - no future data.

        OPTIMIZATION: Only generates features that are in feature_cols (if specified).
        This significantly reduces computation time when using CORE_FEATURES.
        """
        df_features = df.copy()

        # === Price features (no leakage - uses only current/past) ===
        if self._should_generate("returns"):
            df_features["returns"] = (df["close"] - df["open"]) / df["open"]

        # log_returns needed for hist_volatility_20
        if self._needs_any("log_returns", "hist_volatility_20"):
            df_features["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        if self._should_generate("price_change"):
            df_features["price_change"] = df["close"].diff()

        if self._should_generate("price_change_pct"):
            df_features["price_change_pct"] = df["close"].pct_change()

        if self._should_generate("high_low_range"):
            df_features["high_low_range"] = df["high"] - df["low"]

        if self._should_generate("close_open_range"):
            df_features["close_open_range"] = df["close"] - df["open"]

        if self._should_generate("typical_price"):
            df_features["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

        # === Temporal features (seasonality) ===
        # Day of week (0=Monday, 6=Sunday) for capturing weekly patterns
        if self._should_generate("day_of_week"):
            if "timestamp" in df.columns:
                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df_timestamp = pd.to_datetime(df["timestamp"])
                else:
                    df_timestamp = df["timestamp"]
                df_features["day_of_week"] = df_timestamp.dt.dayofweek
            else:
                # If using DatetimeIndex
                df_features["day_of_week"] = df.index.dayofweek

        # === Forward returns for prediction target ===
        # returns_1 = (close[t+1] - close[t]) / close[t]
        # Shift -1 to get FUTURE return (will be dropped during dropna to avoid leakage)
        if self._should_generate("returns_1"):
            df_features["returns_1"] = (
                df["close"].pct_change().shift(-1)
            )  # Single-step return

        if self._should_generate("returns_5"):
            df_features["returns_5"] = (
                df["close"].shift(-5) / df["close"] - 1
            )  # 5-step cumulative return

        # === Moving averages (rolling - uses only past data) ===
        # Only generate requested SMAs
        for period in TECHNICAL_INDICATORS["sma_periods"]:
            if self._should_generate(f"sma_{period}"):
                df_features[f"sma_{period}"] = ta.sma(df["close"], length=period)

        # Only generate requested EMAs
        for period in TECHNICAL_INDICATORS["ema_periods"]:
            if self._should_generate(f"ema_{period}"):
                df_features[f"ema_{period}"] = ta.ema(df["close"], length=period)

        # VWAP
        if self._should_generate("vwap"):
            df_features["vwap"] = ta.vwap(
                df["high"], df["low"], df["close"], df["volume"]
            )

        # === Momentum indicators ===
        rsi_period = TECHNICAL_INDICATORS["rsi_period"]
        if self._should_generate(f"rsi_{rsi_period}"):
            df_features[f"rsi_{rsi_period}"] = ta.rsi(df["close"], length=rsi_period)

        # MACD - only compute if any MACD feature is needed
        if self._needs_any("macd", "macd_signal", "macd_hist"):
            macd_config = TECHNICAL_INDICATORS["macd"]
            macd = ta.macd(
                df["close"],
                fast=macd_config["fast"],
                slow=macd_config["slow"],
                signal=macd_config["signal"],
            )
            if macd is not None:
                macd_cols = macd.columns.tolist()
                for col in macd_cols:
                    if col.startswith("MACD_") and not col.startswith(
                        ("MACDs_", "MACDh_")
                    ):
                        if self._should_generate("macd"):
                            df_features["macd"] = macd[col]
                    elif col.startswith("MACDs_"):
                        if self._should_generate("macd_signal"):
                            df_features["macd_signal"] = macd[col]
                    elif col.startswith("MACDh_"):
                        if self._should_generate("macd_hist"):
                            df_features["macd_hist"] = macd[col]

        # Stochastic - only compute if any stochastic feature is needed
        if self._needs_any("stoch_k", "stoch_d"):
            stoch_config = TECHNICAL_INDICATORS["stochastic"]
            stoch = ta.stoch(
                df["high"],
                df["low"],
                df["close"],
                k=stoch_config["k"],
                d=stoch_config["d"],
            )
            if stoch is not None:
                stoch_cols = stoch.columns.tolist()
                for col in stoch_cols:
                    if col.startswith("STOCHk_"):
                        if self._should_generate("stoch_k"):
                            df_features["stoch_k"] = stoch[col]
                    elif col.startswith("STOCHd_"):
                        if self._should_generate("stoch_d"):
                            df_features["stoch_d"] = stoch[col]

        # ROC - only generate requested periods
        for period in TECHNICAL_INDICATORS["roc_periods"]:
            if self._should_generate(f"roc_{period}"):
                df_features[f"roc_{period}"] = ta.roc(df["close"], length=period)

        # Momentum - only generate requested periods
        for period in TECHNICAL_INDICATORS["momentum_periods"]:
            if self._should_generate(f"momentum_{period}"):
                df_features[f"momentum_{period}"] = ta.mom(df["close"], length=period)

        # CCI
        cci_period = TECHNICAL_INDICATORS["cci_period"]
        if self._should_generate(f"cci_{cci_period}"):
            df_features[f"cci_{cci_period}"] = ta.cci(
                df["high"], df["low"], df["close"], length=cci_period
            )

        # Williams %R
        willr_period = TECHNICAL_INDICATORS["willr_period"]
        if self._should_generate(f"willr_{willr_period}"):
            df_features[f"willr_{willr_period}"] = ta.willr(
                df["high"], df["low"], df["close"], length=willr_period
            )

        # === Volatility indicators ===
        atr_period = TECHNICAL_INDICATORS["atr_period"]
        if self._should_generate(f"atr_{atr_period}"):
            df_features[f"atr_{atr_period}"] = ta.atr(
                df["high"], df["low"], df["close"], length=atr_period
            )

        # Bollinger Bands - only compute if any BB feature is needed
        # Note: bb_percent depends on bb_upper and bb_lower
        if self._needs_any(
            "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent"
        ):
            bb_config = TECHNICAL_INDICATORS["bollinger_bands"]
            bbands = ta.bbands(
                df["close"], length=bb_config["period"], std=bb_config["std"]
            )
            if bbands is not None:
                bb_cols = bbands.columns.tolist()
                for col in bb_cols:
                    if col.startswith("BBU_"):
                        df_features["bb_upper"] = bbands[col]
                    elif col.startswith("BBM_"):
                        if self._should_generate("bb_middle"):
                            df_features["bb_middle"] = bbands[col]
                    elif col.startswith("BBL_"):
                        df_features["bb_lower"] = bbands[col]

                if (
                    "bb_upper" in df_features.columns
                    and "bb_lower" in df_features.columns
                ):
                    if self._should_generate("bb_width"):
                        df_features["bb_width"] = (
                            df_features["bb_upper"] - df_features["bb_lower"]
                        )
                    if self._should_generate("bb_percent"):
                        denom = df_features["bb_upper"] - df_features["bb_lower"]
                        df_features["bb_percent"] = (
                            df["close"] - df_features["bb_lower"]
                        ) / denom

        # Rolling std - only generate requested periods
        for period in [10, 20, 30]:
            if self._should_generate(f"std_{period}"):
                df_features[f"std_{period}"] = df["close"].rolling(window=period).std()

        # Historical volatility - requires log_returns
        if self._should_generate("hist_volatility_20"):
            # Ensure log_returns exists (already generated if needed above)
            if "log_returns" in df_features.columns:
                df_features["hist_volatility_20"] = df_features["log_returns"].rolling(
                    window=20
                ).std() * np.sqrt(252 * 24 * 60)

        # === Volume indicators ===
        if self._should_generate("obv"):
            df_features["obv"] = ta.obv(df["close"], df["volume"])

        # Volume SMAs - only generate if needed
        for period in [10, 20, 30]:
            if self._should_generate(f"volume_sma_{period}"):
                df_features[f"volume_sma_{period}"] = ta.sma(
                    df["volume"], length=period
                )

        if self._should_generate("volume_change"):
            df_features["volume_change"] = df["volume"].diff()

        if self._should_generate("volume_change_pct"):
            df_features["volume_change_pct"] = df["volume"].pct_change()

        if self._should_generate("volume_ratio"):
            df_features["volume_ratio"] = (
                df["volume"] / df["volume"].rolling(window=20).mean()
            )

        if self._should_generate("volume_price_trend"):
            df_features["volume_price_trend"] = ta.pvt(df["close"], df["volume"])

        # === Trend indicators ===
        adx_period = TECHNICAL_INDICATORS["adx_period"]
        # ADX - only compute if any ADX feature is needed
        if self._needs_any(
            f"adx_{adx_period}", f"dmp_{adx_period}", f"dmn_{adx_period}"
        ):
            adx_result = ta.adx(df["high"], df["low"], df["close"], length=adx_period)
            if adx_result is not None:
                adx_cols = adx_result.columns.tolist()
                for col in adx_cols:
                    if col.startswith("ADX_"):
                        if self._should_generate(f"adx_{adx_period}"):
                            df_features[f"adx_{adx_period}"] = adx_result[col]
                    elif col.startswith("DMP_"):
                        if self._should_generate(f"dmp_{adx_period}"):
                            df_features[f"dmp_{adx_period}"] = adx_result[col]
                    elif col.startswith("DMN_"):
                        if self._should_generate(f"dmn_{adx_period}"):
                            df_features[f"dmn_{adx_period}"] = adx_result[col]

        # Aroon - only compute if any Aroon feature is needed
        aroon_period = TECHNICAL_INDICATORS["aroon_period"]
        if self._needs_any(
            f"aroon_up_{aroon_period}",
            f"aroon_down_{aroon_period}",
            f"aroon_osc_{aroon_period}",
        ):
            aroon = ta.aroon(df["high"], df["low"], length=aroon_period)
            if aroon is not None:
                aroon_cols = aroon.columns.tolist()
                for col in aroon_cols:
                    if col.startswith("AROONU_"):
                        if self._should_generate(f"aroon_up_{aroon_period}"):
                            df_features[f"aroon_up_{aroon_period}"] = aroon[col]
                    elif col.startswith("AROOND_"):
                        if self._should_generate(f"aroon_down_{aroon_period}"):
                            df_features[f"aroon_down_{aroon_period}"] = aroon[col]
                    elif col.startswith("AROONOSC_"):
                        if self._should_generate(f"aroon_osc_{aroon_period}"):
                            df_features[f"aroon_osc_{aroon_period}"] = aroon[col]

        # Supertrend - only compute if any Supertrend feature is needed
        if self._needs_any("supertrend", "supertrend_direction"):
            supertrend = ta.supertrend(
                df["high"], df["low"], df["close"], length=10, multiplier=3
            )
            if supertrend is not None:
                st_cols = supertrend.columns.tolist()
                for col in st_cols:
                    if col.startswith("SUPERT_") and not col.startswith("SUPERTd_"):
                        if self._should_generate("supertrend"):
                            df_features["supertrend"] = supertrend[col]
                    elif col.startswith("SUPERTd_"):
                        if self._should_generate("supertrend_direction"):
                            df_features["supertrend_direction"] = supertrend[col]

        return df_features

    def _normalize_with_training_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using statistics from training data only.

        Uses z-score normalization: (x - mean) / std
        Clips extreme values using training percentiles to prevent outliers.
        """
        df_normalized = df.copy()

        for col in self.feature_list:
            if col not in df_normalized.columns:
                continue

            mean = self.statistics.means.get(col, 0)
            std = self.statistics.stds.get(col, 1)

            if std > 0:
                df_normalized[col] = (df_normalized[col] - mean) / std

            # Clip to training data range (prevents extreme outliers)
            p1 = self.statistics.percentile_1.get(col)
            p99 = self.statistics.percentile_99.get(col)
            if p1 is not None and p99 is not None:
                # Convert percentiles to z-scores for clipping
                z_p1 = (p1 - mean) / std if std > 0 else -3
                z_p99 = (p99 - mean) / std if std > 0 else 3
                df_normalized[col] = df_normalized[col].clip(z_p1, z_p99)

        return df_normalized

    def _filter_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to keep only specified features plus base OHLCV columns.

        Args:
            df: DataFrame with all generated features

        Returns:
            DataFrame with filtered features
        """
        if self.feature_cols is None:
            # No filtering - return all columns
            return df

        # Always keep base OHLCV columns
        base_cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "num_trades",
        ]
        cols_to_keep = [col for col in base_cols if col in df.columns]

        # Add requested feature columns
        missing_features = []
        for col in self.feature_cols:
            if col in df.columns:
                cols_to_keep.append(col)
            else:
                missing_features.append(col)

        if missing_features:
            logger.warning(
                f"Requested features not found: {missing_features}. "
                f"These features may not have been generated or may have been dropped."
            )

        # Log filtering results
        total_features = len([c for c in df.columns if c not in base_cols])
        kept_features = len([c for c in cols_to_keep if c not in base_cols])
        logger.info(
            f"Feature filtering: keeping {kept_features}/{total_features} features "
            f"(+ {len([c for c in cols_to_keep if c in base_cols])} base columns)"
        )

        # Return filtered DataFrame (cols_to_keep is always a list, so result is always DataFrame)
        result = df[cols_to_keep]
        assert isinstance(result, pd.DataFrame)  # Type narrowing for pyright
        return result

    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: list[str],
        lags: list[int] | None = None,
    ) -> pd.DataFrame:
        """Add lagged versions of columns.

        Creates features like close_lag_1, close_lag_2, etc.
        Useful for ML models that don't handle sequences natively.

        Args:
            df: DataFrame with data
            columns: Columns to create lags for
            lags: List of lag periods (default: [1, 2, 3, 5, 10])

        Returns:
            DataFrame with lag features added
        """
        if lags is None:
            lags = TECHNICAL_INDICATORS.get("lag_periods", [1, 2, 3, 5, 10])

        if lags is None:
            lags = [1, 2, 3, 5, 10]

        logger.debug(f"Adding lag features for {columns} with lags {lags}")

        df_lagged = df.copy()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping lag features")
                continue

            for lag in lags:
                df_lagged[f"{col}_lag_{lag}"] = df[col].shift(lag)

        return df_lagged

    def to_darts_timeseries(
        self,
        df: pd.DataFrame,
        value_cols: list[str] | None = None,
        fill_missing: bool = True,
        covariates: list[str] | None = None,
    ) -> TimeSeries | tuple[TimeSeries, TimeSeries | None]:
        """Convert DataFrame to Darts TimeSeries with optional covariates.

        Args:
            df: DataFrame with time series data
            value_cols: Columns to include as target values (default: ["returns_1"])
            fill_missing: Whether to forward fill missing values
            covariates: Optional list of covariate column names (e.g., CORE_FEATURES)

        Returns:
            If covariates provided: tuple of (target_ts, covariates_ts)
            Otherwise: target_ts only (backward compatible)

        Example:
            # Univariate (backward compatible)
            >>> ts = engineer.to_darts_timeseries(df, value_cols=["close"])

            # Multivariate with covariates
            >>> target_ts, cov_ts = engineer.to_darts_timeseries(
            ...     df, value_cols=["returns_1"], covariates=CORE_FEATURES
            ... )
        """
        logger.debug("Converting DataFrame to Darts TimeSeries")

        df_ts = df.copy()

        # Ensure datetime index
        if "timestamp" in df_ts.columns and not isinstance(
            df_ts.index, pd.DatetimeIndex
        ):
            df_ts = df_ts.set_index("timestamp")

        if not isinstance(df_ts.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

        # Default to returns instead of close (for ML models)
        if value_cols is None:
            value_cols = ["returns_1"]

        # Create target TimeSeries
        df_target = df_ts[value_cols].copy()
        if fill_missing:
            df_target = df_target.ffill()
        df_target = df_target.dropna()

        if df_target.empty:
            raise ValueError(
                "No valid target data remaining after handling missing values"
            )

        target_ts = TimeSeries.from_dataframe(
            df_target,
            value_cols=value_cols,
            fill_missing_dates=False,
        )

        logger.info(
            f"Created target TimeSeries with {len(target_ts)} timesteps "
            f"and {len(value_cols)} component(s)"
        )

        # Handle covariates if provided
        cov_ts = None
        if covariates is not None:
            # Filter to available covariates
            available_covs = [col for col in covariates if col in df_ts.columns]

            if len(available_covs) < len(covariates):
                missing = set(covariates) - set(available_covs)
                logger.warning(f"Missing {len(missing)} covariates: {missing}")

            if available_covs:
                df_cov = df_ts[available_covs].copy()

                # Remove columns that are entirely NaN (can't be used)
                cols_to_drop = df_cov.columns[df_cov.isna().all()].tolist()
                if cols_to_drop:
                    logger.warning(
                        f"Dropping {len(cols_to_drop)} all-NaN covariate columns: {cols_to_drop}"
                    )
                    df_cov = df_cov.drop(columns=cols_to_drop)
                    available_covs = [
                        c for c in available_covs if c not in cols_to_drop
                    ]

                if df_cov.empty or len(available_covs) == 0:
                    logger.warning(
                        "No valid covariate columns remaining after removing all-NaN features"
                    )
                else:
                    # Forward fill missing values
                    if fill_missing:
                        df_cov = df_cov.ffill()

                    # Only drop rows where ALL remaining covariates are NaN (not just ANY NaN)
                    df_cov = df_cov.dropna(how="all")

                    if df_cov.empty:
                        logger.warning(
                            "No valid covariate data after handling missing values"
                        )
                    else:
                        cov_ts = TimeSeries.from_dataframe(
                            df_cov,
                            value_cols=available_covs,
                            fill_missing_dates=False,
                        )
                        logger.info(
                            f"Created covariates TimeSeries with {len(cov_ts)} timesteps "
                            f"and {len(available_covs)} feature(s)"
                        )

                        # Verify alignment
                        if len(target_ts) != len(cov_ts):
                            logger.warning(
                                f"Length mismatch: target={len(target_ts)}, covariates={len(cov_ts)}"
                            )

        # Return tuple if covariates were requested, else just target (backward compatible)
        if covariates is not None:
            # Always return tuple when covariates requested, even if None
            return target_ts, cov_ts
        return target_ts

    def get_feature_names(self) -> list[str]:
        """Get list of generated feature names."""
        return self.feature_list.copy()

    def save_statistics(self, path: str | Path) -> None:
        """Save fitted statistics to file for reproducibility.

        Args:
            path: Path to save JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "statistics": self.statistics.to_dict(),
            "feature_list": self.feature_list,
            "warmup_period": self.warmup_period,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved feature statistics to {path}")

    def load_statistics(self, path: str | Path) -> None:
        """Load previously fitted statistics.

        Args:
            path: Path to JSON file
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        self.statistics = FeatureStatistics.from_dict(data["statistics"])
        self.feature_list = data.get("feature_list", [])
        self.warmup_period = data.get("warmup_period", 200)

        logger.info(
            f"Loaded feature statistics from {path} ({len(self.feature_list)} features)"
        )

    def check_data_leakage(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> bool:
        """Check for potential data leakage between train and test sets.

        Verifies that test data comes strictly after training data.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            timestamp_col: Name of timestamp column

        Returns:
            True if no leakage detected, False otherwise
        """
        if (
            timestamp_col not in train_df.columns
            or timestamp_col not in test_df.columns
        ):
            logger.warning(f"Cannot check leakage: '{timestamp_col}' column not found")
            return True

        train_max = train_df[timestamp_col].max()
        test_min = test_df[timestamp_col].min()

        if test_min <= train_max:
            logger.error(
                f"DATA LEAKAGE DETECTED! "
                f"Test min ({test_min}) <= Train max ({train_max}). "
                f"Test data must start AFTER training data ends."
            )
            return False

        logger.info(f"No data leakage: Train ends {train_max}, Test starts {test_min}")
        return True
