"""Data preprocessing for time series.

This module provides data cleaning, validation, normalization, and sequence
generation for training machine learning models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from price_stradamus.config.constants import NormalizationMethod


class DataPreprocessor:
    """Data cleaning, validation, and normalization for time series.

    Features:
    - OHLCV data validation (high >= low, etc.)
    - Missing value handling (forward fill, interpolation)
    - Outlier removal (IQR, Z-score methods)
    - Normalization (MinMax, Standard, Robust scalers)
    - Sequence creation for time series models

    Example:
        preprocessor = DataPreprocessor(normalization_method=NormalizationMethod.MINMAX)
        df = preprocessor.validate_ohlcv(df)
        df = preprocessor.handle_missing_values(df)
        df_normalized = preprocessor.normalize(df, columns=["close"], fit=True)
        X, y = preprocessor.create_sequences(df_normalized, input_length=60, output_length=5)
    """

    def __init__(
        self,
        normalization_method: NormalizationMethod | str = NormalizationMethod.MINMAX,
    ):
        """Initialize preprocessor.

        Args:
            normalization_method: Normalization method to use (enum or string)
        """
        # Convert string to enum if needed
        if isinstance(normalization_method, str):
            try:
                self.normalization_method = NormalizationMethod(normalization_method)
            except ValueError:
                raise ValueError(
                    f"Invalid normalization method: {normalization_method}. "
                    f"Choose from: {[m.value for m in NormalizationMethod]}"
                )
        else:
            self.normalization_method = normalization_method

        self.scalers: dict[str, Any] = {}  # Column name -> fitted scaler

        logger.info(
            f"DataPreprocessor initialized (normalization={self.normalization_method.value})"
        )

    def validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV data integrity.

        Checks:
        - Required columns exist
        - No null values
        - high >= low
        - high >= open, close
        - low <= open, close
        - volume >= 0
        - Timestamps are monotonically increasing
        - No duplicate timestamps

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Validated DataFrame (same as input if valid)

        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for validation")
            return df

        # Check required columns exist
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check for null values
        if df[required_cols].isnull().any().any():  # type: ignore[attr-defined]
            null_counts = df[required_cols].isnull().sum()
            null_info = null_counts[null_counts > 0].to_dict()
            raise ValueError(f"Found null values in required columns: {null_info}")

        # Check OHLCV relationships
        violations = []

        # high should be >= all other prices
        high_violations = df[df["high"] < df[["open", "close", "low"]].max(axis=1)]
        if not high_violations.empty:
            violations.append(
                f"{len(high_violations)} rows where high < max(open, close, low)"
            )

        # low should be <= all other prices
        low_violations = df[df["low"] > df[["open", "close", "high"]].min(axis=1)]
        if not low_violations.empty:
            violations.append(
                f"{len(low_violations)} rows where low > min(open, close, high)"
            )

        # Volume should be non-negative
        negative_volume = df[df["volume"] < 0]
        if not negative_volume.empty:
            violations.append(f"{len(negative_volume)} rows with negative volume")

        if violations:
            error_msg = "OHLCV validation failed:\n  - " + "\n  - ".join(violations)
            raise ValueError(error_msg)

        # Check timestamp monotonicity (if timestamp is index or column)
        if "timestamp" in df.columns:
            if not df["timestamp"].is_monotonic_increasing:
                raise ValueError("Timestamps are not monotonically increasing")

            # Check for duplicate timestamps
            duplicates = df[df["timestamp"].duplicated()]
            if not duplicates.empty:
                raise ValueError(f"Found {len(duplicates)} duplicate timestamps")

        logger.debug(f"OHLCV validation passed for {len(df)} rows")
        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "ffill",
    ) -> pd.DataFrame:
        """Handle missing values in DataFrame.

        Args:
            df: DataFrame with potential missing values
            method: Method to use ("ffill", "interpolate", "drop")
                - "ffill": Forward fill
                - "interpolate": Linear interpolation
                - "drop": Drop rows with any null values

        Returns:
            DataFrame with missing values handled

        Raises:
            ValueError: If invalid method specified
        """
        if df.empty:
            return df

        null_count_before = df.isnull().sum().sum()

        if null_count_before == 0:
            logger.debug("No missing values found")
            return df

        logger.info(
            f"Handling {null_count_before} missing values using method '{method}'"
        )

        if method == "ffill":
            df = df.ffill()  # Forward fill
        elif method == "interpolate":
            df = df.interpolate(method="linear", limit_direction="forward")
        elif method == "drop":
            df = df.dropna()
        else:
            raise ValueError(
                f"Invalid method '{method}'. Choose from: 'ffill', 'interpolate', 'drop'"
            )

        null_count_after = df.isnull().sum().sum()
        logger.info(
            f"After handling: {null_count_after} missing values remain ({len(df)} rows)"
        )

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: list[str],
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """Remove or clip outliers from specified columns.

        Args:
            df: DataFrame with data
            columns: Columns to check for outliers
            method: Outlier detection method ("iqr", "zscore")
                - "iqr": Interquartile range (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
                - "zscore": Z-score (mean ± threshold * std)
            threshold: Threshold for Z-score method (default: 3.0 standard deviations)

        Returns:
            DataFrame with outliers removed

        Raises:
            ValueError: If invalid method or columns don't exist
        """
        if df.empty:
            return df

        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        logger.info(f"Removing outliers from {columns} using method '{method}'")

        df_clean = df.copy()
        outliers_removed = 0

        if method == "iqr":
            for col in columns:
                Q1 = df_clean[col].quantile(0.25)  # type: ignore[attr-defined]
                Q3 = df_clean[col].quantile(0.75)  # type: ignore[attr-defined]
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outliers_removed += outliers.sum()

                # Remove outlier rows
                df_clean = df_clean[~outliers]

        elif method == "zscore":
            for col in columns:
                mean = df_clean[col].mean()
                std = df_clean[col].std()

                z_scores = np.abs((df_clean[col] - mean) / std)
                outliers = z_scores > threshold
                outliers_removed += outliers.sum()

                # Remove outlier rows
                df_clean = df_clean[~outliers]

        else:
            raise ValueError(f"Invalid method '{method}'. Choose from: 'iqr', 'zscore'")

        logger.info(
            f"Removed {outliers_removed} outliers ({len(df_clean)} rows remaining)"
        )
        return df_clean  # type: ignore[return-value]

    def normalize(
        self,
        df: pd.DataFrame,
        columns: list[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """Normalize specified columns using configured scaler.

        Args:
            df: DataFrame to normalize
            columns: Columns to normalize
            fit: Whether to fit scaler (True) or use existing scaler (False)
                Set to True for training data, False for test data

        Returns:
            DataFrame with normalized columns

        Raises:
            ValueError: If columns don't exist or scaler not fitted (fit=False)
        """
        if df.empty:
            return df

        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        df_normalized = df.copy()

        # Create scaler instance based on method
        scaler_class = {
            NormalizationMethod.MINMAX: MinMaxScaler,
            NormalizationMethod.STANDARD: StandardScaler,
            NormalizationMethod.ROBUST: RobustScaler,
        }[self.normalization_method]

        for col in columns:
            if fit:
                # Fit new scaler
                scaler = scaler_class()
                df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
                self.scalers[col] = scaler
                logger.debug(f"Fitted and transformed column '{col}'")
            else:
                # Use existing scaler
                if col not in self.scalers:
                    raise ValueError(
                        f"No fitted scaler found for column '{col}'. "
                        "Call normalize() with fit=True first."
                    )
                scaler = self.scalers[col]
                df_normalized[col] = scaler.transform(df_normalized[[col]])
                logger.debug(f"Transformed column '{col}' using existing scaler")

        logger.info(
            f"Normalized {len(columns)} columns using {self.normalization_method.value}"
        )
        return df_normalized

    def denormalize(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        """Inverse transform (denormalize) specified columns.

        Args:
            df: DataFrame with normalized data
            columns: Columns to denormalize

        Returns:
            DataFrame with original scale restored

        Raises:
            ValueError: If scaler not fitted for column
        """
        if df.empty:
            return df

        df_denormalized = df.copy()

        for col in columns:
            if col not in self.scalers:
                raise ValueError(
                    f"No fitted scaler found for column '{col}'. "
                    "Cannot denormalize without fitted scaler."
                )

            scaler = self.scalers[col]
            df_denormalized[col] = scaler.inverse_transform(df_denormalized[[col]])
            logger.debug(f"Denormalized column '{col}'")

        logger.info(f"Denormalized {len(columns)} columns")
        return df_denormalized

    def create_sequences(
        self,
        df: pd.DataFrame,
        input_length: int,
        output_length: int,
        target_column: str = "close",
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for time series training.

        Args:
            df: DataFrame with features (all numeric columns used as features)
            input_length: Number of timesteps in input sequence (lookback window)
            output_length: Number of timesteps to predict (forecast horizon)
            target_column: Column to use as prediction target (default: "close")
            stride: Step size between sequences (default: 1 for overlapping windows)

        Returns:
            Tuple of (X, y):
            - X: Input sequences, shape (num_samples, input_length, num_features)
            - y: Target sequences, shape (num_samples, output_length)

        Raises:
            ValueError: If DataFrame is too small or parameters invalid
        """
        if df.empty:
            raise ValueError("Cannot create sequences from empty DataFrame")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        if input_length < 1 or output_length < 1:
            raise ValueError("input_length and output_length must be >= 1")

        if len(df) < input_length + output_length:
            raise ValueError(
                f"DataFrame too small: need at least {input_length + output_length} rows, "
                f"but have {len(df)}"
            )

        # Convert to numpy arrays
        # All columns are features for X, target column for y
        feature_cols = df.columns.tolist()
        X_data = df[feature_cols].values
        y_data = df[target_column].values

        X_sequences = []
        y_sequences = []

        # Create sliding windows
        for i in range(0, len(df) - input_length - output_length + 1, stride):
            # Input sequence: current position to current + input_length
            X_seq = X_data[i : i + input_length]

            # Output sequence: input_length ahead to input_length + output_length ahead
            y_seq = y_data[i + input_length : i + input_length + output_length]

            X_sequences.append(X_seq)
            y_sequences.append(y_seq)

        X = np.array(X_sequences)  # Shape: (num_samples, input_length, num_features)
        y = np.array(y_sequences)  # Shape: (num_samples, output_length)

        logger.info(
            f"Created sequences: X shape={X.shape}, y shape={y.shape} "
            f"(input_length={input_length}, output_length={output_length}, stride={stride})"
        )

        return X, y

    def get_scaler(self, column: str) -> Any:
        """Get fitted scaler for a specific column.

        Args:
            column: Column name

        Returns:
            Fitted sklearn scaler

        Raises:
            KeyError: If scaler not found
        """
        if column not in self.scalers:
            raise KeyError(f"No fitted scaler found for column '{column}'")

        return self.scalers[column]

    def reset_scalers(self) -> None:
        """Clear all fitted scalers."""
        self.scalers.clear()
        logger.info("All scalers reset")
