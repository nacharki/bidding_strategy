# External imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from category_encoders import TargetEncoder

# Internal imports
from logger.logging import getLogger

logger = getLogger()


class DataPreprocessor:
    """
    A class used to preprocess data for the bidding strategy.

    Parameters
    ----------
    high_cardinality_threshold : int, optional
        Threshold for high cardinality features (default is 100).
    missing_value_strategy : str, optional
        Strategy to handle missing values (default is "median").
    scaling_method : str, optional
        Method to scale numerical features (default is None).
    """

    def __init__(
        self,
        high_cardinality_threshold=100,
        missing_value_strategy="median",
        scaling_method=None,
    ):
        self.high_cardinality_threshold = high_cardinality_threshold
        self.missing_value_strategy = missing_value_strategy
        self.scaling_method = scaling_method
        self.target_encoders = {}
        self.label_encoders = {}
        self.cardinality_maps = {}
        self.scalers = {}
        self.dropped_columns = []

    def drop_unique_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns with only one unique value.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        pd.DataFrame
            The data with unique columns dropped.
        """
        # Make a copy of the data
        data_copy = data.copy()
        # Drop columns with only one unique value
        unique_cols = data_copy.columns[data_copy.nunique() == 1]
        self.dropped_columns.extend(unique_cols)
        data_copy.drop(unique_cols, axis=1, inplace=True)
        logger.info(f"Dropped columns with one unique value: {unique_cols}")
        return data_copy

    def drop_near_missing_columns(
        self, data: pd.DataFrame, missing_drop_threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Drop columns with a high percentage of missing values.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.
        missing_drop_threshold : float, optional
            Threshold for dropping columns with missing values (default is 0.6).

        Returns
        -------
        pd.DataFrame
            The data with columns having high missing values dropped.
        """
        # Make a copy of the data
        data_copy = data.copy()
        # Determine columns with high missing values
        missing_percentages = data_copy.isnull().mean() * 100
        time_columns = ["click_time", "transaction_timestamps"]
        columns_to_drop = list(
            set(missing_percentages[missing_percentages > missing_drop_threshold].index)
            - set(time_columns)
        )
        self.dropped_columns.extend(columns_to_drop)
        # Drop columns with high missing values
        data_copy.drop(columns=columns_to_drop, inplace=True)
        logger.info(f"Dropped columns with high missing values: {columns_to_drop}")
        return data_copy

    def drop_highly_correlated_columns(
        self, data: pd.DataFrame, correlation_threshold: float = 0.9
    ) -> pd.DataFrame:
        """
        Drop highly correlated numerical columns.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.
        correlation_threshold : float, optional
            Threshold for dropping highly correlated columns (default is 0.9).

        Returns
        -------
        pd.DataFrame
            The data with highly correlated columns dropped.
        """
        # Make a copy of the data
        data_copy = data.copy()
        # Determine highly correlated columns
        numerical_cols = data_copy.select_dtypes(include=[np.number]).columns
        corr_matrix = data_copy[numerical_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > correlation_threshold)
        ]
        # Drop highly correlated columns
        self.dropped_columns.extend(to_drop)
        data_copy.drop(to_drop, axis=1, inplace=True)
        logger.info(f"Dropped highly correlated columns: {to_drop}")
        return data_copy

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on the specified strategy.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        pd.DataFrame
            The data with missing values handled.
        """
        # Make a copy of the data
        data_copy = data.copy()
        numeric_cols = data_copy.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = data_copy.select_dtypes(include=["object"]).columns
        # Handle missing values based on the strategy
        if self.missing_value_strategy == "median":
            for col in numeric_cols:
                data_copy[col] = data_copy[col].fillna(data_copy[col].median())
        elif self.missing_value_strategy == "mean":
            for col in numeric_cols:
                data_copy[col] = data_copy[col].fillna(data_copy[col].mean())
        elif self.missing_value_strategy == "mode":
            for col in numeric_cols:
                data_copy[col] = data_copy[col].fillna(data_copy[col].mode()[0])
        # Handle missing values for categorical columns
        for col in categorical_cols:
            data_copy[col] = data_copy[col].fillna(data_copy[col].mode()[0])

        logger.info(
            f"Handled missing values using strategy: {self.missing_value_strategy}"
        )
        return data_copy

    def encode_categorical_features(
        self, data: pd.DataFrame, target_col=None
    ) -> pd.DataFrame:
        """
        Encode categorical features using appropriate encoding methods.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.
        target_col : str, optional
            The target column for target encoding (default is None).

        Returns
        -------
        pd.DataFrame
            The data with categorical features encoded.
        """
        # Make a copy of the data
        data_copy = data.copy()
        categorical_cols = data_copy.select_dtypes(include=["object"]).columns
        # Encode categorical features
        for col in categorical_cols:
            n_unique = data_copy[col].nunique()

            if n_unique > self.high_cardinality_threshold:
                if col in ["city", "device_model", "bundle_identifier"]:
                    # Target encoding for high-cardinality features
                    if target_col is not None:
                        encoder = TargetEncoder()
                        data_copy[f"{col}_encoded"] = encoder.fit_transform(
                            data_copy[col], data_copy[target_col]
                        )
                        self.target_encoders[col] = encoder
                elif col in ["creative_id", "supplier_id"]:
                    # Frequency encoding
                    freq_map = data_copy[col].value_counts(normalize=True)
                    data_copy[f"{col}_freq"] = data_copy[col].map(freq_map)
                    self.cardinality_maps[col] = freq_map
            else:
                # Label encoding for low-cardinality features
                encoder = LabelEncoder()
                data_copy[f"{col}_encoded"] = encoder.fit_transform(data_copy[col])
                self.label_encoders[col] = encoder

        logger.info("Encoded categorical features.")
        return data_copy

    def scale_numerical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using the specified scaling method.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        pd.DataFrame
            The data with numerical features scaled.
        """
        # Make a copy of the data
        data_copy = data.copy()
        numerical_cols = data_copy.select_dtypes(include=["int64", "float64"]).columns
        # Scale numerical features
        if self.scaling_method == "standard":
            scaler = StandardScaler()
            data_copy[numerical_cols] = scaler.fit_transform(data_copy[numerical_cols])
            self.scalers["standard"] = scaler
        elif self.scaling_method == "minmax":
            scaler = MinMaxScaler()
            data_copy[numerical_cols] = scaler.fit_transform(data_copy[numerical_cols])
            self.scalers["minmax"] = scaler

        logger.info(f"Scaled numerical features using {self.scaling_method} scaling.")
        return data_copy

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply saved encodings and scalings to new data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        # Make a copy of the data
        data_copy = data.copy()
        # Target encoding
        for col, encoder in self.target_encoders.items():
            data_copy[f"{col}_encoded"] = encoder.transform(data_copy[col])
        # Frequency encoding
        for col, freq_map in self.cardinality_maps.items():
            data_copy[f"{col}_freq"] = data_copy[col].map(freq_map)
        # Label encoding
        for col, encoder in self.label_encoders.items():
            data_copy[f"{col}_encoded"] = encoder.transform(data_copy[col])
        if self.scaling_method == "standard":
            data_copy[list(self.scalers["standard"].feature_names_in_)] = self.scalers[
                "standard"
            ].transform(data_copy[list(self.scalers["standard"].feature_names_in_)])
        elif self.scaling_method == "minmax":
            data_copy[list(self.scalers["minmax"].feature_names_in_)] = self.scalers[
                "minmax"
            ].transform(data_copy[list(self.scalers["minmax"].feature_names_in_)])

        return data_copy

    def run_all_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run all preprocessing steps on the data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        pd.DataFrame
            The preprocessed data.
        """
        data_copy = data.copy()
        data_copy = self.drop_unique_columns(data_copy)
        data_copy = self.drop_near_missing_columns(data_copy)
        data_copy = self.drop_highly_correlated_columns(data_copy)
        data_copy = self.handle_missing_values(data_copy)
        data_copy = self.encode_categorical_features(data_copy)
        data_copy = self.scale_numerical_features(data_copy)
        return data_copy


class FeatureEngineer:
    """
    A class used to perform feature engineering on the data.
    """

    def __init__(self):
        pass

    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from timestamp columns.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        pd.DataFrame
            The data with temporal features created.
        """
        data_copy = data.copy()
        # Convert imp_time to datetime if necessary
        if not pd.api.types.is_datetime64_any_dtype(data_copy["imp_time"]):
            data_copy["imp_time"] = pd.to_datetime(data_copy["imp_time"])
        # Extract time-based features
        data_copy["imp_hour"] = data_copy["imp_time"].dt.hour
        data_copy["imp_day_of_week"] = data_copy["imp_time"].dt.dayofweek
        data_copy["imp_month"] = data_copy["imp_time"].dt.month
        # Process transaction timestamps if available
        if "transaction_timestamps" in data_copy.columns:
            data_copy["last_transaction_time"] = pd.to_datetime(
                data_copy["transaction_timestamps"].str.split(",").str[-1]
            )
            data_copy["time_since_last_transaction"] = (
                data_copy["imp_time"] - data_copy["last_transaction_time"]
            ).dt.total_seconds() / 3600
        # Calculate time differences
        data_copy["click_latency"] = (data_copy["click_time"] - data_copy["imp_time"]).dt.total_seconds()
        data_copy["conversion_window"] = (
            data_copy["transaction_timestamps"] - data_copy["imp_time"]
        ).dt.total_seconds()

        logger.info("Created temporal features.")
        return data_copy

    def create_activity_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate activity ratios from impression and click data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        pd.DataFrame
            The data with activity ratio features created.
        """
        data_copy = data.copy()
        # Click-through rates for different windows
        for window in ["1d", "7d", "90d"]:
            clicks_col = f"imp_activity__click_{window}"
            imps_col = f"imp_activity__imp_{window}"

            # Avoid division by zero
            data_copy[f"ctr_{window}"] = np.where(
                data_copy[imps_col] > 0, data_copy[clicks_col] / data_copy[imps_col], 0
            )
        # App engagement density
        transaction_cols = [
            col for col in data_copy.columns if "transaction" in col and "60d" in col
        ]
        for col in transaction_cols:
            data_copy[f"{col}_daily_avg"] = data_copy[col] / 60

        logger.info("Created activity ratio features.")
        return data_copy

    def run_all_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run all feature engineering steps on the data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        pd.DataFrame
            The data with all feature engineering steps applied.
        """
        data_copy = data.copy()
        data_copy = self.create_temporal_features(data_copy)
        data_copy = self.create_activity_ratios(data_copy)
        return data_copy
