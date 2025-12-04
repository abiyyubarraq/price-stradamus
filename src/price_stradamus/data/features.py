"""Technical indicator feature engineering.

This module generates technical indicators for machine learning models using
pandas-ta library. Includes 50+ indicators across price, momentum, volatility,
volume, and trend categories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta
from darts import TimeSeries
from loguru import logger

from price_stradamus.config.constants import TECHNICAL_INDICATORS


class FeatureEngineer:
    """Generate technical indicators for ML models.

    Features include:
    - Price features (returns, log returns, price changes)
    - Moving averages (SMA, EMA, VWAP)
    - Momentum indicators (RSI, MACD, Stochastic, ROC, MOM, CCI)
    - Volatility indicators (ATR, Bollinger Bands, historical volatility)
    - Volume indicators (OBV, volume SMA, volume changes)
    - Trend indicators (ADX, Aroon, Supertrend)
    - Lag features

    Example:
        engineer = FeatureEngineer()
        df_with_features = engineer.generate_all_features(df)
        print(f"Generated {len(engineer.feature_list)} features")

        # Convert to Darts TimeSeries
        ts = engineer.to_darts_timeseries(df_with_features, value_cols=["close"])
    """

    def __init__(self):
        """Initialize feature engineer."""
        self.feature_list: list[str] = []
        logger.info("FeatureEngineer initialized")

    def generate_all_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate all technical indicators.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with original data plus all technical indicators

        Raises:
            ValueError: If required columns missing
        """
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df

        logger.info(f"Generating all technical indicators for {len(df)} candles")

        df_features = df.copy()

        # Generate each category
        df_features = self.add_price_features(df_features)
        df_features = self.add_moving_averages(df_features)
        df_features = self.add_momentum_indicators(df_features)
        df_features = self.add_volatility_indicators(df_features)
        df_features = self.add_volume_indicators(df_features)
        df_features = self.add_trend_indicators(df_features)

        # Store feature names (exclude original OHLCV columns)
        original_cols = set(df.columns)
        self.feature_list = [
            col for col in df_features.columns if col not in original_cols
        ]

        logger.info(f"Generated {len(self.feature_list)} features")
        return df_features

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features.

        Features:
        - Returns: (close - open) / open
        - Log returns: log(close / close.shift(1))
        - Price change: close - close.shift(1)
        - High-low range: high - low
        - Close-open range: close - open

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with price features added
        """
        logger.debug("Adding price features")

        # Returns
        df["returns"] = (df["close"] - df["open"]) / df["open"]

        # Log returns
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Price changes
        df["price_change"] = df["close"].diff()
        df["price_change_pct"] = df["close"].pct_change()

        # Ranges
        df["high_low_range"] = df["high"] - df["low"]
        df["close_open_range"] = df["close"] - df["open"]

        # Typical price (average of high, low, close)
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average indicators.

        Indicators:
        - SMA (Simple Moving Average): various periods
        - EMA (Exponential Moving Average): various periods
        - VWAP (Volume Weighted Average Price)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with moving averages added
        """
        logger.debug("Adding moving averages")

        # SMA for different periods
        for period in TECHNICAL_INDICATORS["sma_periods"]:
            df[f"sma_{period}"] = ta.sma(df["close"], length=period)

        # EMA for different periods
        for period in TECHNICAL_INDICATORS["ema_periods"]:
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)

        # VWAP (Volume Weighted Average Price)
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators using pandas-ta.

        Indicators:
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Stochastic Oscillator
        - ROC (Rate of Change)
        - MOM (Momentum)
        - CCI (Commodity Channel Index)
        - Williams %R
        - Ultimate Oscillator

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with momentum indicators added
        """
        logger.debug("Adding momentum indicators")

        # RSI (Relative Strength Index)
        rsi_period = TECHNICAL_INDICATORS["rsi_period"]
        df[f"rsi_{rsi_period}"] = ta.rsi(df["close"], length=rsi_period)

        # MACD (Moving Average Convergence Divergence)
        macd_config = TECHNICAL_INDICATORS["macd"]
        macd = ta.macd(
            df["close"],
            fast=macd_config["fast"],
            slow=macd_config["slow"],
            signal=macd_config["signal"],
        )
        if macd is not None:
            # Handle different pandas-ta versions with flexible column name matching
            macd_cols = macd.columns.tolist()
            macd_col = (
                [
                    c
                    for c in macd_cols
                    if c.startswith("MACD_") and not c.startswith(("MACDs_", "MACDh_"))
                ][0]
                if any(
                    c.startswith("MACD_") and not c.startswith(("MACDs_", "MACDh_"))
                    for c in macd_cols
                )
                else None
            )
            macd_signal_col = (
                [c for c in macd_cols if c.startswith("MACDs_")][0]
                if any(c.startswith("MACDs_") for c in macd_cols)
                else None
            )
            macd_hist_col = (
                [c for c in macd_cols if c.startswith("MACDh_")][0]
                if any(c.startswith("MACDh_") for c in macd_cols)
                else None
            )

            if macd_col:
                df["macd"] = macd[macd_col]
            if macd_signal_col:
                df["macd_signal"] = macd[macd_signal_col]
            if macd_hist_col:
                df["macd_hist"] = macd[macd_hist_col]

        # Stochastic Oscillator
        stoch_config = TECHNICAL_INDICATORS["stochastic"]
        stoch = ta.stoch(
            df["high"],
            df["low"],
            df["close"],
            k=stoch_config["k"],
            d=stoch_config["d"],
        )
        if stoch is not None:
            # Handle different pandas-ta versions with flexible column name matching
            stoch_cols = stoch.columns.tolist()
            stoch_k_col = (
                [c for c in stoch_cols if c.startswith("STOCHk_")][0]
                if any(c.startswith("STOCHk_") for c in stoch_cols)
                else None
            )
            stoch_d_col = (
                [c for c in stoch_cols if c.startswith("STOCHd_")][0]
                if any(c.startswith("STOCHd_") for c in stoch_cols)
                else None
            )

            if stoch_k_col:
                df["stoch_k"] = stoch[stoch_k_col]
            if stoch_d_col:
                df["stoch_d"] = stoch[stoch_d_col]

        # ROC (Rate of Change)
        for period in TECHNICAL_INDICATORS["roc_periods"]:
            df[f"roc_{period}"] = ta.roc(df["close"], length=period)

        # Momentum
        for period in TECHNICAL_INDICATORS["momentum_periods"]:
            df[f"momentum_{period}"] = ta.mom(df["close"], length=period)

        # CCI (Commodity Channel Index)
        cci_period = TECHNICAL_INDICATORS["cci_period"]
        df[f"cci_{cci_period}"] = ta.cci(
            df["high"], df["low"], df["close"], length=cci_period
        )

        # Williams %R
        willr_period = TECHNICAL_INDICATORS["willr_period"]
        df[f"willr_{willr_period}"] = ta.willr(
            df["high"], df["low"], df["close"], length=willr_period
        )

        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators.

        Indicators:
        - ATR (Average True Range)
        - Bollinger Bands (upper, middle, lower)
        - Standard deviation (rolling)
        - Historical volatility (std of log returns)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volatility indicators added
        """
        logger.debug("Adding volatility indicators")

        # ATR (Average True Range)
        atr_period = TECHNICAL_INDICATORS["atr_period"]
        df[f"atr_{atr_period}"] = ta.atr(
            df["high"], df["low"], df["close"], length=atr_period
        )

        # Bollinger Bands
        bb_config = TECHNICAL_INDICATORS["bollinger_bands"]
        bbands = ta.bbands(
            df["close"],
            length=bb_config["period"],
            std=bb_config["std"],
        )
        if bbands is not None:
            # Handle different pandas-ta versions with flexible column name matching
            bb_cols = bbands.columns.tolist()
            bb_upper_col = (
                [c for c in bb_cols if c.startswith("BBU_")][0]
                if any(c.startswith("BBU_") for c in bb_cols)
                else None
            )
            bb_middle_col = (
                [c for c in bb_cols if c.startswith("BBM_")][0]
                if any(c.startswith("BBM_") for c in bb_cols)
                else None
            )
            bb_lower_col = (
                [c for c in bb_cols if c.startswith("BBL_")][0]
                if any(c.startswith("BBL_") for c in bb_cols)
                else None
            )

            if bb_upper_col and bb_middle_col and bb_lower_col:
                df["bb_upper"] = bbands[bb_upper_col]
                df["bb_middle"] = bbands[bb_middle_col]
                df["bb_lower"] = bbands[bb_lower_col]
                df["bb_width"] = df["bb_upper"] - df["bb_lower"]
                df["bb_percent"] = (df["close"] - df["bb_lower"]) / (
                    df["bb_upper"] - df["bb_lower"]
                )

        # Standard deviation (rolling)
        for period in [10, 20, 30]:
            df[f"std_{period}"] = df["close"].rolling(window=period).std()

        # Historical volatility (annualized std of log returns)
        df["hist_volatility_20"] = df["log_returns"].rolling(window=20).std() * np.sqrt(
            252 * 24 * 60
        )  # For 1-min data

        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators.

        Indicators:
        - OBV (On-Balance Volume)
        - Volume SMA
        - Volume change
        - Volume ratio (current / average)
        - Volume price trend

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volume indicators added
        """
        logger.debug("Adding volume indicators")

        # OBV (On-Balance Volume)
        df["obv"] = ta.obv(df["close"], df["volume"])

        # Volume SMA
        for period in [10, 20, 30]:
            df[f"volume_sma_{period}"] = ta.sma(df["volume"], length=period)

        # Volume changes
        df["volume_change"] = df["volume"].diff()
        df["volume_change_pct"] = df["volume"].pct_change()

        # Volume ratio (current volume / 20-period average)
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

        # Volume-weighted indicators
        df["volume_price_trend"] = ta.pvt(df["close"], df["volume"])

        return df

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators.

        Indicators:
        - ADX (Average Directional Index)
        - Aroon (Aroon Up, Aroon Down)
        - Supertrend

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with trend indicators added
        """
        logger.debug("Adding trend indicators")

        # ADX (Average Directional Index)
        adx_period = TECHNICAL_INDICATORS["adx_period"]
        adx_result = ta.adx(df["high"], df["low"], df["close"], length=adx_period)
        if adx_result is not None:
            # Handle different pandas-ta versions with flexible column name matching
            adx_cols = adx_result.columns.tolist()
            adx_col = (
                [c for c in adx_cols if c.startswith("ADX_")][0]
                if any(c.startswith("ADX_") for c in adx_cols)
                else None
            )
            dmp_col = (
                [c for c in adx_cols if c.startswith("DMP_")][0]
                if any(c.startswith("DMP_") for c in adx_cols)
                else None
            )
            dmn_col = (
                [c for c in adx_cols if c.startswith("DMN_")][0]
                if any(c.startswith("DMN_") for c in adx_cols)
                else None
            )

            if adx_col:
                df[f"adx_{adx_period}"] = adx_result[adx_col]
            if dmp_col:
                df[f"dmp_{adx_period}"] = adx_result[dmp_col]
            if dmn_col:
                df[f"dmn_{adx_period}"] = adx_result[dmn_col]

        # Aroon
        aroon_period = TECHNICAL_INDICATORS["aroon_period"]
        aroon = ta.aroon(df["high"], df["low"], length=aroon_period)
        if aroon is not None:
            # Handle different pandas-ta versions with flexible column name matching
            aroon_cols = aroon.columns.tolist()
            aroon_up_col = (
                [c for c in aroon_cols if c.startswith("AROONU_")][0]
                if any(c.startswith("AROONU_") for c in aroon_cols)
                else None
            )
            aroon_down_col = (
                [c for c in aroon_cols if c.startswith("AROOND_")][0]
                if any(c.startswith("AROOND_") for c in aroon_cols)
                else None
            )
            aroon_osc_col = (
                [c for c in aroon_cols if c.startswith("AROONOSC_")][0]
                if any(c.startswith("AROONOSC_") for c in aroon_cols)
                else None
            )

            if aroon_up_col:
                df[f"aroon_up_{aroon_period}"] = aroon[aroon_up_col]
            if aroon_down_col:
                df[f"aroon_down_{aroon_period}"] = aroon[aroon_down_col]
            if aroon_osc_col:
                df[f"aroon_osc_{aroon_period}"] = aroon[aroon_osc_col]

        # Supertrend
        supertrend = ta.supertrend(
            df["high"], df["low"], df["close"], length=10, multiplier=3
        )
        if supertrend is not None:
            # Handle different pandas-ta versions with flexible column name matching
            st_cols = supertrend.columns.tolist()
            st_col = (
                [
                    c
                    for c in st_cols
                    if c.startswith("SUPERT_") and not c.startswith("SUPERTd_")
                ][0]
                if any(
                    c.startswith("SUPERT_") and not c.startswith("SUPERTd_")
                    for c in st_cols
                )
                else None
            )
            st_dir_col = (
                [c for c in st_cols if c.startswith("SUPERTd_")][0]
                if any(c.startswith("SUPERTd_") for c in st_cols)
                else None
            )

            if st_col:
                df["supertrend"] = supertrend[st_col]
            if st_dir_col:
                df["supertrend_direction"] = supertrend[st_dir_col]

        return df

    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: list[str],
        lags: list[int] | None = None,
    ) -> pd.DataFrame:
        """Add lagged versions of columns.

        Creates features like close_lag_1, close_lag_2, etc.
        Useful for classical ML models that don't handle sequences.

        Args:
            df: DataFrame with data
            columns: Columns to create lags for
            lags: List of lag periods (default: [1, 2, 3, 5, 10])

        Returns:
            DataFrame with lag features added
        """
        if lags is None:
            lags = TECHNICAL_INDICATORS.get("lag_periods", [1, 2, 3, 5, 10])

        # Ensure lags is not None
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

        num_new_features = len(columns) * len(lags)
        logger.info(f"Added {num_new_features} lag features")

        return df_lagged

    def select_features(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Select specific features from DataFrame.

        Args:
            df: DataFrame with all features
            feature_names: List of feature names to select

        Returns:
            DataFrame with only selected features

        Raises:
            ValueError: If requested features don't exist
        """
        missing = set(feature_names) - set(df.columns)
        if missing:
            raise ValueError(f"Requested features not found: {missing}")

        # Ensure we always return a DataFrame, not a Series
        result = df[feature_names]
        if isinstance(result, pd.Series):
            return result.to_frame()
        return result

    def to_darts_timeseries(
        self,
        df: pd.DataFrame,
        value_cols: list[str] | None = None,
        fill_missing: bool = True,
    ) -> TimeSeries:
        """Convert DataFrame to Darts TimeSeries.

        Args:
            df: DataFrame with time series data
            value_cols: Columns to include as values (default: all numeric columns)
            fill_missing: Whether to forward fill missing values (default: True)

        Returns:
            Darts TimeSeries object

        Raises:
            ValueError: If DataFrame doesn't have datetime index or timestamp column
        """
        logger.debug("Converting DataFrame to Darts TimeSeries")

        df_ts = df.copy()

        # Ensure datetime index
        if "timestamp" in df_ts.columns and not isinstance(
            df_ts.index, pd.DatetimeIndex
        ):
            df_ts = df_ts.set_index("timestamp")

        if not isinstance(df_ts.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame must have DatetimeIndex or 'timestamp' column for TimeSeries conversion"
            )

        # Select value columns
        if value_cols is None:
            # Use all numeric columns
            value_cols = df_ts.select_dtypes(include=[np.number]).columns.tolist()

        # Select only the columns we need
        df_ts = df_ts[value_cols]

        # Handle missing values
        if fill_missing:
            df_ts = df_ts.ffill()

        # Drop any remaining NaN rows (only in the selected columns)
        df_ts = df_ts.dropna()

        if df_ts.empty:
            raise ValueError("No valid data remaining after handling missing values")

        # Create Darts TimeSeries
        ts = TimeSeries.from_dataframe(
            df_ts,
            value_cols=value_cols,
            fill_missing_dates=False,  # Assume data is already complete
        )

        logger.info(
            f"Created Darts TimeSeries with {len(ts)} timesteps and {len(value_cols)} components"
        )
        return ts

    def get_feature_importance(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        method: str = "correlation",
    ) -> pd.DataFrame:
        """Calculate feature importance scores.

        Args:
            df: DataFrame with features and target
            target_col: Target column name
            method: Importance method ("correlation" only for now)

        Returns:
            DataFrame with feature names and importance scores, sorted descending

        Raises:
            ValueError: If target column not found
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        logger.debug(f"Calculating feature importance using method '{method}'")

        if method == "correlation":
            # Calculate absolute correlation with target
            correlations = df.corr()[target_col].abs()

            # Remove target itself and OHLCV columns
            exclude_cols = {
                target_col,
                "open",
                "high",
                "low",
                "close",
                "volume",
                "timestamp",
            }
            correlations = correlations.drop(labels=exclude_cols, errors="ignore")

            # Sort by importance
            importance_df = pd.DataFrame(
                {
                    "feature": correlations.index,
                    "importance": correlations.values,
                }
            ).sort_values("importance", ascending=False)

            logger.info(f"Calculated importance for {len(importance_df)} features")
            return importance_df

        else:
            raise ValueError(
                f"Unsupported method '{method}'. Choose from: 'correlation'"
            )
