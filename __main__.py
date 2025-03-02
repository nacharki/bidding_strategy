# main.py

# External imports
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# Internal imports
from bidding_utils.data_preprocessing import DataPreprocessor, FeatureEngineer
from bidding_utils.modelling import TwoStageModel
from logger.logging import getLogger

logger = getLogger()


def load_config():
    """
    Load configuration from the YAML file.

    Returns
    -------
    dict
        Configuration parameters.
    """
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Initialize components
    preprocessor = DataPreprocessor(
        high_cardinality_threshold=config.get("high_cardinality_threshold", 100),
        missing_value_strategy=config.get("missing_value_strategy", "median"),
        scaling_method=config.get("scaling_method", "standard")
    )
    feature_engineer = FeatureEngineer()
    model = TwoStageModel(config)

    # Load data
    original_data = pd.read_csv(config["data_path"])

    # Data preprocessing
    preprocessed_data = preprocessor.run_all_preprocessing(original_data)

    # Feature engineering
    preprocessed_data = feature_engineer.run_all_feature_engineering(preprocessed_data)

    # Prepare data for training
    X = preprocessed_data.drop(columns=["click", "conversion"])
    y_click = preprocessed_data["click"]
    y_conversion = preprocessed_data["conversion"]

    # Split data into training and testing sets
    X_train, X_test, y_click_train, y_click_test, y_conversion_train, y_conversion_test = train_test_split(
        X, y_click, y_conversion, test_size=0.2, random_state=config.get("random_state", 42)
    )

    # Train the click model
    model.train_click_model(X_train, y_click_train)

    # Train the conversion model
    model.train_conversion_model(X_train, y_conversion_train)

    # Evaluate the models
    model.evaluate(X_test, y_click_test, y_conversion_test)

    # Optimize bids (example)
    optimized_bids = model.optimize_bid(X_test, base_bid=1.0, scale_factor=1.0)
    logger.info("Optimized bids: ", optimized_bids)
