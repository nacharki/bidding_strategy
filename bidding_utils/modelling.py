# External imports
from lazypredict.Supervised import LazyClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

# Internal imports
from logger.logging import getLogger

logger = getLogger()


class TwoStageModel:
    def __init__(self, config):
        self.config = config
        self.click_model = None
        self.conversion_model = None
        self.click_feature_importance = None
        self.conversion_feature_importance = None

    def train_click_model(self, X, y):
        """Train the click prediction model using LazyClassifier."""
        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=self.config.get("random_state", 42))
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=self.config.get("random_state", 42)
        )

        # Use LazyClassifier to evaluate multiple models
        lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=roc_auc_score)
        models, _ = lazy_clf.fit(X_train, X_test, y_train, y_test)

        # Display model performance
        logger.info("Click Model Performance:")
        print(models)

        # Select the best model based on AUC-ROC
        best_model_name = models.index[0]  # Best model is the first one in the sorted list
        logger.info("Best Click Model: %s", best_model_name)

        # Train the best model
        if "LightGBM" in best_model_name:
            self.click_model = LGBMClassifier(
                objective="binary",
                class_weight={0: 1, 1: self.config.get("click_class_weight", 10)},
                num_leaves=self.config.get("click_num_leaves", 31),
                n_estimators=self.config.get("click_n_estimators", 500),
                random_state=self.config.get("random_state", 42),
            )
        else:
            # Add other models as needed
            raise ValueError(f"Model {best_model_name} is not implemented yet.")

        self.click_model.fit(X_train, y_train)

        # Calibrate the model
        self.click_model = CalibratedClassifierCV(self.click_model, cv=3, method="isotonic")
        self.click_model.fit(X_train, y_train)

        # Save feature importance (if applicable)
        if hasattr(self.click_model, "feature_importances_"):
            self.click_feature_importance = self.click_model.feature_importances_

    def train_conversion_model(self, X, y):
        """Train the conversion prediction model using LazyClassifier."""
        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=self.config.get("random_state", 42))
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=self.config.get("random_state", 42)
        )

        # Use LazyClassifier to evaluate multiple models
        lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=roc_auc_score)
        models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)

        # Display model performance
        logger.info("Conversion Model Performance:")
        print(models)

        # Select the best model based on AUC-ROC
        best_model_name = models.index[0]  # Best model is the first one in the sorted list
        logger.info("Best Conversion Model: %s", best_model_name)

        # Train the best model
        if "LightGBM" in best_model_name:
            self.conversion_model = LGBMClassifier(
                objective="binary",
                class_weight={0: 1, 1: self.config.get("conversion_class_weight", 10)},
                num_leaves=self.config.get("conversion_num_leaves", 31),
                n_estimators=self.config.get("conversion_n_estimators", 500),
                random_state=self.config.get("random_state", 42),
            )
        else:
            # Add other models as needed
            raise ValueError(f"Model {best_model_name} is not implemented yet.")

        self.conversion_model.fit(X_train, y_train)

        # Calibrate the model
        self.conversion_model = CalibratedClassifierCV(self.conversion_model, cv=3, method="isotonic")
        self.conversion_model.fit(X_train, y_train)

        # Save feature importance (if applicable)
        if hasattr(self.conversion_model, "feature_importances_"):
            self.conversion_feature_importance = self.conversion_model.feature_importances_

    def predict(self, X):
        """Predict click and conversion probabilities."""
        if self.click_model is None or self.conversion_model is None:
            raise ValueError("Models have not been trained yet.")

        # Predict click probabilities
        click_prob = self.click_model.predict_proba(X)[:, 1]

        # Predict conversion probabilities (only for predicted clicks)
        conversion_prob = self.conversion_model.predict_proba(X)[:, 1]

        # Combined prediction
        combined_prob = click_prob * conversion_prob

        return {
            "click_prob": click_prob,
            "conversion_prob": conversion_prob,
            "combined_prob": combined_prob,
        }

    def evaluate(self, X, y_click, y_conversion):
        """Evaluate the performance of both models."""
        if self.click_model is None or self.conversion_model is None:
            raise ValueError("Models have not been trained yet.")

        # Evaluate click model
        click_pred = self.click_model.predict_proba(X)[:, 1]
        click_auc = roc_auc_score(y_click, click_pred)
        logger.info(f"Click Model AUC: {click_auc}")
        print(classification_report(y_click, self.click_model.predict(X)))

        # Evaluate conversion model
        conversion_pred = self.conversion_model.predict_proba(X)[:, 1]
        conversion_auc = roc_auc_score(y_conversion, conversion_pred)
        logger.info(f"Conversion Model AUC: {conversion_auc}")
        print(classification_report(y_conversion, self.conversion_model.predict(X)))

    def optimize_bid(self, X, base_bid=1.0, scale_factor=1.0):
        """Optimize bids based on predicted probabilities."""
        predictions = self.predict(X)
        combined_prob = predictions["combined_prob"]

        # Scale bids based on combined probabilities
        optimized_bids = base_bid * (combined_prob ** scale_factor)
        return optimized_bids
