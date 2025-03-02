# External imports
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
import warnings

# Internal imports
from logger.logging import getLogger

logger = getLogger()


class TwoStageModel:
    """
    Two-stage model for click and conversion prediction in real-time bidding systems.

    This class implements a two-stage model approach where the first model predicts
    click probability and the second model predicts conversion probability. It includes
    feature selection, hyperparameter optimization, and various bid optimization strategies.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model parameters and settings.

    Attributes
    ----------
    click_model : object
        Trained model for click prediction.
    conversion_model : object
        Trained model for conversion prediction.
    click_feature_importance : dict
        Feature importance values for the click model.
    conversion_feature_importance : dict
        Feature importance values for the conversion model.
    selected_features : dict
        Dictionary of selected features for each model.
    feature_selector : dict
        Feature selectors for each model.
    inference_times : dict
        Dictionary tracking inference timing metrics.
    """
    def __init__(self, config):
        self.config = config
        self.click_model = None
        self.conversion_model = None
        self.click_feature_importance = None
        self.conversion_feature_importance = None
        self.selected_features = {"click": None, "conversion": None}
        self.feature_selector = {"click": None, "conversion": None}
        self.inference_times = {
            "preprocess": [],
            "click": [],
            "conversion": [],
            "total": [],
        }

    def _select_important_features(self, X: pd.DataFrame, y: pd.Series, model_type: str = "click") -> pd.DataFrame:
        """
        Select important features to reduce dimensionality and improve inference speed.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe.
        y : pd.Series or np.ndarray
            Target variable.
        model_type : str, optional
            Type of model ('click' or 'conversion'), by default "click".

        Returns
        -------
        pd.DataFrame
            Dataframe with selected features only.
        """
        # Initialize base model
        base_model = LGBMClassifier(
            objective="binary",
            class_weight={0: 1, 1: 10},
            n_estimators=100,
            random_state=self.config.get("random_state", 42),
            verbosity=-1,
        )

        # Fit on a sample of data for speed
        sample_size = min(50000, X.shape[0])
        if X.shape[0] > sample_size:
            X_sample, _, y_sample, _ = train_test_split(
                X,
                y,
                train_size=sample_size,
                random_state=self.config.get("random_state", 42),
            )
        else:
            X_sample, y_sample = X, y

        base_model.fit(X_sample, y_sample)

        # Select features using feature importance
        feature_selector = SelectFromModel(base_model, threshold="mean", prefit=True)

        # Get selected feature indices and names
        selected_indices = feature_selector.get_support()
        selected_feature_names = X.columns[selected_indices].tolist()

        # Store the selector and selected features
        self.feature_selector[model_type] = feature_selector
        self.selected_features[model_type] = selected_feature_names

        logger.info(
            f"Selected {len(selected_feature_names)} features for {model_type} model"
        )
        return X[selected_feature_names]

    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_type: str = "click") -> dict:
        """
        Perform hyperparameter optimization using RandomizedSearchCV.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe.
        y : pd.Series or np.ndarray
            Target variable.
        model_type : str, optional
            Type of model ('click' or 'conversion'), by default "click".

        Returns
        -------
        dict
            Dictionary of optimized hyperparameters.
        """
        # Define parameter grid
        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200, 300],
            "num_leaves": [31, 63, 127],
            "min_child_samples": [20, 50, 100],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }

        # Configure LightGBM
        base_model = LGBMClassifier(
            objective="binary",
            class_weight={0: 1, 1: self.config.get(f"{model_type}_class_weight", 10)},
            random_state=self.config.get("random_state", 42),
            verbosity=-1,
        )

        # Sample data for faster hyperparameter tuning
        sample_size = min(50000, X.shape[0])
        if X.shape[0] > sample_size:
            X_sample, _, y_sample, _ = train_test_split(
                X,
                y,
                train_size=sample_size,
                random_state=self.config.get("random_state", 42),
            )
        else:
            X_sample, y_sample = X, y

        # Run randomized search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=10,
                scoring="roc_auc",
                cv=3,
                n_jobs=-1,
                random_state=self.config.get("random_state", 42),
                verbose=0,
            )
            search.fit(X_sample, y_sample)

        logger.info(f"Best {model_type} model parameters: {search.best_params_}")
        return search.best_params_

    def train_click_model(
        self, X: pd.DataFrame, y: pd.Series, optimize_hyperparams: bool = False, feature_selection: bool = True
    ) -> None:
        """
        Train the click prediction model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe for training.
        y : pd.Series or np.ndarray
            Binary target variable (1 for click, 0 for no click).
        optimize_hyperparams : bool, optional
            Whether to perform hyperparameter optimization, by default False.
        feature_selection : bool, optional
            Whether to perform feature selection, by default True.

        Returns
        -------
        None
            The trained model is stored in the class instance.
        """
        start_time = time.time()

        # Apply feature selection if enabled
        if feature_selection:
            X_selected = self._select_important_features(X, y, "click")
        else:
            X_selected = X
            self.selected_features["click"] = X.columns.tolist()

        # Handle class imbalance
        smote = SMOTE(random_state=self.config.get("random_state", 42))
        X_resampled, y_resampled = smote.fit_resample(X_selected, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled,
            y_resampled,
            test_size=0.2,
            random_state=self.config.get("random_state", 42),
        )

        # Get best hyperparameters if enabled
        if optimize_hyperparams:
            best_params = self._optimize_hyperparameters(X_train, y_train, "click")
        else:
            best_params = {
                "learning_rate": 0.05,
                "n_estimators": self.config.get("click_n_estimators", 300),
                "num_leaves": self.config.get("click_num_leaves", 31),
                "min_child_samples": 50,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
            }

        # Train the model with optimized parameters
        self.click_model = LGBMClassifier(
            objective="binary",
            class_weight={0: 1, 1: self.config.get("click_class_weight", 10)},
            random_state=self.config.get("random_state", 42),
            verbosity=-1,
            **best_params,
        )

        self.click_model.fit(X_train, y_train)

        # Save feature importance
        self.click_feature_importance = dict(
            zip(X_selected.columns, self.click_model.feature_importances_)
        )

        # Calibrate probabilities if needed
        if self.config.get("calibrate_probabilities", True):
            self.click_model = CalibratedClassifierCV(
                self.click_model, cv="prefit", method="isotonic"
            )
            self.click_model.fit(X_test, y_test)

        # Evaluate the model
        click_pred = self.click_model.predict_proba(X_test)[:, 1]
        click_auc = roc_auc_score(y_test, click_pred)
        avg_precision = average_precision_score(y_test, click_pred)

        logger.info(
            f"Click Model AUC: {click_auc:.4f}, Avg Precision: {avg_precision:.4f}"
        )
        logger.info(
            f"Click model training completed in {time.time() - start_time:.2f} seconds"
        )

    def train_conversion_model(
        self, X: pd.DataFrame, y: pd.Series, optimize_hyperparams: bool = False, feature_selection: bool = True
    ) -> None:
        """
        Train the conversion prediction model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe for training.
        y : pd.Series or np.ndarray
            Binary target variable (1 for conversion, 0 for no conversion).
        optimize_hyperparams : bool, optional
            Whether to perform hyperparameter optimization, by default False.
        feature_selection : bool, optional
            Whether to perform feature selection, by default True.

        Returns
        -------
        None
            The trained model is stored in the class instance.
        """
        start_time = time.time()

        # Apply feature selection if enabled
        if feature_selection:
            X_selected = self._select_important_features(X, y, "conversion")
        else:
            X_selected = X
            self.selected_features["conversion"] = X.columns.tolist()

        # Handle class imbalance
        smote = SMOTE(random_state=self.config.get("random_state", 42))
        X_resampled, y_resampled = smote.fit_resample(X_selected, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled,
            y_resampled,
            test_size=0.2,
            random_state=self.config.get("random_state", 42),
        )

        # Get best hyperparameters if enabled
        if optimize_hyperparams:
            best_params = self._optimize_hyperparameters(X_train, y_train, "conversion")
        else:
            best_params = {
                "learning_rate": 0.05,
                "n_estimators": self.config.get("conversion_n_estimators", 300),
                "num_leaves": self.config.get("conversion_num_leaves", 31),
                "min_child_samples": 50,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
            }

        # Train the model with optimized parameters
        self.conversion_model = LGBMClassifier(
            objective="binary",
            class_weight={0: 1, 1: self.config.get("conversion_class_weight", 10)},
            random_state=self.config.get("random_state", 42),
            verbosity=-1,
            **best_params,
        )

        self.conversion_model.fit(X_train, y_train)

        # Save feature importance
        self.conversion_feature_importance = dict(
            zip(X_selected.columns, self.conversion_model.feature_importances_)
        )

        # Calibrate probabilities if needed
        if self.config.get("calibrate_probabilities", True):
            self.conversion_model = CalibratedClassifierCV(
                self.conversion_model, cv="prefit", method="isotonic"
            )
            self.conversion_model.fit(X_test, y_test)

        # Evaluate the model
        conversion_pred = self.conversion_model.predict_proba(X_test)[:, 1]
        conversion_auc = roc_auc_score(y_test, conversion_pred)
        avg_precision = average_precision_score(y_test, conversion_pred)

        logger.info(
            f"Conversion Model AUC: {conversion_auc:.4f}, Avg Precision: {avg_precision:.4f}"
        )
        logger.info(
            f"Conversion model training completed in {time.time() - start_time:.2f} seconds"
        )

    def predict(self, X: pd.DataFrame, benchmark: bool = False) -> dict:
        """
        Predict click and conversion probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe for prediction.
        benchmark : bool, optional
            Whether to benchmark prediction time, by default False.

        Returns
        -------
        dict
            Dictionary containing probability arrays:
            - 'click_prob': np.ndarray of click probabilities
            - 'conversion_prob': np.ndarray of conversion probabilities
            - 'combined_prob': np.ndarray of combined probabilities (click * conversion)
        """
        if self.click_model is None or self.conversion_model is None:
            raise ValueError("Models have not been trained yet.")
        total_start_time = time.time()

        # Select only the features needed for each model
        preprocess_start = time.time()

        X_click = (
            X[self.selected_features["click"]] if self.selected_features["click"] else X
        )
        X_conversion = (
            X[self.selected_features["conversion"]]
            if self.selected_features["conversion"]
            else X
        )

        preprocess_time = time.time() - preprocess_start
        self.inference_times["preprocess"].append(preprocess_time)
        # Predict click probabilities
        click_start = time.time()

        click_prob = self.click_model.predict_proba(X_click)[:, 1]
        click_time = time.time() - click_start
        self.inference_times["click"].append(click_time)

        # Predict conversion probabilities
        conversion_start = time.time()

        conversion_prob = self.conversion_model.predict_proba(X_conversion)[:, 1]
        conversion_time = time.time() - conversion_start
        self.inference_times["conversion"].append(conversion_time)

        # Calculate combined probability
        combined_prob = click_prob * conversion_prob

        total_time = time.time() - total_start_time
        self.inference_times["total"].append(total_time)

        logger.info(
            f"Prediction times (ms): "
            f"Preprocess={preprocess_time*1000:.2f}, "
            f"Click={click_time*1000:.2f}, "
            f"Conversion={conversion_time*1000:.2f}, "
            f"Total={total_time*1000:.2f}"
        )

        return {
            "click_prob": click_prob,
            "conversion_prob": conversion_prob,
            "combined_prob": combined_prob,
        }

    def evaluate(self, X: pd.DataFrame, y_click: pd.Series, y_conversion: pd.Series) -> dict:
        """
        Evaluate the performance of both models.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe for evaluation.
        y_click : pd.Series or np.ndarray
            Binary target variable for clicks.
        y_conversion : pd.Series or np.ndarray
            Binary target variable for conversions.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics:
            - 'click_auc': AUC-ROC score for click model
            - 'click_ap': Average precision score for click model
            - 'conversion_auc': AUC-ROC score for conversion model
            - 'conversion_ap': Average precision score for conversion model
        """
        if self.click_model is None or self.conversion_model is None:
            raise ValueError("Models have not been trained yet.")

        # Select only the features needed for each model
        X_click = (
            X[self.selected_features["click"]] if self.selected_features["click"] else X
        )
        X_conversion = (
            X[self.selected_features["conversion"]]
            if self.selected_features["conversion"]
            else X
        )

        # Evaluate click model
        click_pred = self.click_model.predict_proba(X_click)[:, 1]
        click_auc = roc_auc_score(y_click, click_pred)
        click_ap = average_precision_score(y_click, click_pred)
        logger.info(
            f"Click Model - AUC: {click_auc:.4f}, Avg Precision: {click_ap:.4f}"
        )

        # Evaluate conversion model
        conversion_pred = self.conversion_model.predict_proba(X_conversion)[:, 1]
        conversion_auc = roc_auc_score(y_conversion, conversion_pred)
        conversion_ap = average_precision_score(y_conversion, conversion_pred)
        logger.info(
            f"Conversion Model - AUC: {conversion_auc:.4f}, Avg Precision: {conversion_ap:.4f}"
        )

        # Get classification reports
        click_report = classification_report(y_click, (click_pred > 0.5).astype(int))
        conversion_report = classification_report(
            y_conversion, (conversion_pred > 0.5).astype(int)
        )

        logger.info("Click Model Classification Report:\n" + click_report)
        logger.info("Conversion Model Classification Report:\n" + conversion_report)

        return {
            "click_auc": click_auc,
            "click_ap": click_ap,
            "conversion_auc": conversion_auc,
            "conversion_ap": conversion_ap,
        }

    def optimize_bid(self, X: pd.Series, base_bid: float = 1.0, campaign_info: dict = None) -> np.ndarray:
        """
        Optimize bids based on predicted probabilities and campaign goals.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe for prediction.
        base_bid : float, optional
            Base bid value in currency units, by default 1.0.
        campaign_info : dict, optional
            Campaign information including KPIs and budget, by default None.
            Can include:
            - 'type': str - Campaign type ('cpa', 'cpc', 'roi')
            - 'target_cpa': float - Target cost per acquisition
            - 'target_cpc': float - Target cost per click
            - 'target_roi': float - Target return on investment
            - 'conversion_value': float - Value of a conversion
            - 'daily_budget': float - Daily budget limit

        Returns
        -------
        np.ndarray
            Array of optimized bid values for each row in X.
        """
        # Get predictions
        predictions = self.predict(X, benchmark=True)
        combined_prob = predictions["combined_prob"]
        click_prob = predictions["click_prob"]
        conversion_prob = predictions["conversion_prob"]

        # Default bid optimization strategy
        if campaign_info is None:
            # Simple probability-based bidding
            optimized_bids = base_bid * combined_prob
            return optimized_bids

        # Advanced bid optimization based on campaign goals
        campaign_type = campaign_info.get("type", "cpa")

        if campaign_type == "cpa":
            # Cost Per Acquisition optimization
            target_cpa = campaign_info.get("target_cpa", 10.0)
            expected_cpa = base_bid / (click_prob * conversion_prob)
            cpa_multiplier = np.clip(
                target_cpa / expected_cpa, 0.2, 5.0
            )  # Limit bid adjustment
            optimized_bids = base_bid * cpa_multiplier

        elif campaign_type == "cpc":
            # Cost Per Click optimization
            target_cpc = campaign_info.get("target_cpc", 0.5)
            expected_cpc = base_bid / click_prob
            cpc_multiplier = np.clip(target_cpc / expected_cpc, 0.2, 5.0)
            optimized_bids = base_bid * cpc_multiplier

        elif campaign_type == "roi":
            # Return on Investment optimization
            expected_value = (
                campaign_info.get("conversion_value", 20.0) * conversion_prob
            )
            target_roi = campaign_info.get("target_roi", 2.0)
            cost = base_bid * click_prob
            roi_multiplier = np.clip((expected_value / cost) / target_roi, 0.2, 5.0)
            optimized_bids = base_bid * roi_multiplier

        else:
            # Default to combined probability
            optimized_bids = base_bid * combined_prob

        # Apply budget constraints
        daily_budget = campaign_info.get("daily_budget")
        if daily_budget is not None:
            # Simple budget pacing - scale all bids to fit within budget
            expected_spend = np.sum(optimized_bids * click_prob)
            if expected_spend > daily_budget:
                budget_factor = daily_budget / expected_spend
                optimized_bids *= budget_factor

        return optimized_bids

    def save_models(self, path_prefix: str = "models/") -> None:
        """
        Save trained models to disk.

        Parameters
        ----------
        path_prefix : str, optional
            Directory path to save models, by default "models/".
        """
        if self.click_model is None or self.conversion_model is None:
            raise ValueError("Models have not been trained yet.")

        joblib.dump(self.click_model, f"{path_prefix}click_model.joblib")
        joblib.dump(self.conversion_model, f"{path_prefix}conversion_model.joblib")
        joblib.dump(
            {
                "click_features": self.selected_features["click"],
                "conversion_features": self.selected_features["conversion"],
                "click_importance": self.click_feature_importance,
                "conversion_importance": self.conversion_feature_importance,
            },
            f"{path_prefix}model_metadata.joblib",
        )

        logger.info(f"Models saved to {path_prefix}")

    def load_models(self, path_prefix: str = "models/") -> None:
        """
        Load trained models from disk.

        Parameters
        ----------
        path_prefix : str, optional
            Directory path to load models from, by default "models/".
        """
        self.click_model = joblib.load(f"{path_prefix}click_model.joblib")
        self.conversion_model = joblib.load(f"{path_prefix}conversion_model.joblib")

        metadata = joblib.load(f"{path_prefix}model_metadata.joblib")
        self.selected_features["click"] = metadata["click_features"]
        self.selected_features["conversion"] = metadata["conversion_features"]
        self.click_feature_importance = metadata["click_importance"]
        self.conversion_feature_importance = metadata["conversion_importance"]

        logger.info(f"Models loaded from {path_prefix}")

    def benchmark_inference(self, X: pd.Series, n_iterations: int = 100) -> tuple:
        """
        Benchmark inference speed.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe for benchmarking.
        n_iterations : int, optional
            Number of iterations for benchmarking, by default 100.

        Returns
        -------
        tuple
            (avg_times, p95_times) where each is a dictionary with timing metrics in milliseconds..

        Notes
        -----
        This method performs warm-up runs before benchmarking to avoid cold-start effects.
        It measures preprocessing time, click model inference time, conversion model inference time,
        and total inference time.
        """
        # Make sure we have trained models
        if self.click_model is None or self.conversion_model is None:
            raise ValueError("Models have not been trained yet.")

        # Select a small batch for benchmarking
        if X.shape[0] > 100:
            X_sample = X.sample(100)
        else:
            X_sample = X

        # Warm-up
        for _ in range(5):
            self.predict(X_sample)

        # Benchmark
        self.inference_times = {
            "preprocess": [],
            "click": [],
            "conversion": [],
            "total": [],
        }
        for _ in range(n_iterations):
            self.predict(X_sample, benchmark=True)

        # Calculate average times
        avg_times = {
            k: np.mean(v) * 1000 for k, v in self.inference_times.items()
        }
        p95_times = {
            k: np.percentile(v, 95) * 1000 for k, v in self.inference_times.items()
        }

        logger.info("Average inference times (ms):")
        logger.info(
            f"  Preprocessing: {avg_times['preprocess']:.2f} (p95: {p95_times['preprocess']:.2f})"
        )
        logger.info(
            f"  Click model: {avg_times['click']:.2f} (p95: {p95_times['click']:.2f})"
        )
        logger.info(
            f"  Conversion model: {avg_times['conversion']:.2f} (p95: {p95_times['conversion']:.2f})"
        )
        logger.info(
            f"  Total: {avg_times['total']:.2f} (p95: {p95_times['total']:.2f})"
        )

        return avg_times, p95_times
