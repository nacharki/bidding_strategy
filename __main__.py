# main.py

# External imports
import yaml
import pandas as pd
import time
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Internal imports
from bidding_utils.data_preprocessing import DataPreprocessor, FeatureEngineer
from bidding_utils.modelling import TwoStageModel
from logger.logging import getLogger

logger = getLogger()


def load_config(config_path: str = "configs/config.yaml") -> dict | None:
    """
    Read configuration parameters from =specified YAML configuration file and returns
    its contents as a dictionary. If an error occurs during loading, the
    error is logged, and None is returned.

    Parameters
    ----------
    config_path : str, default="configs/config.yaml"
        Path to the configuration YAML file.

    Returns
    -------
    dict or None
        Dictionary containing configuration parameters if successful,
        None if an error occurred.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return None


def create_output_directories(base_dir: str = "output") -> tuple[str, str]:
    """
    Create necessary directories for storing model artifacts and results, under the specified base
    directory for organizing model outputs, including models, results, and logs.
    If the directories already exist, no action is taken.

    Parameters
    ----------
    base_dir : str, default="output"
        Base directory path where subdirectories will be created.

    Returns
    -------
    tuple of str
        A tuple containing paths to the models and results directories:
        (models_path, results_path).
    """
    directories = ["models", "results", "logs"]
    for directory in directories:
        path = os.path.join(base_dir, directory)
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory: {path}")
    return os.path.join(base_dir, "models"), os.path.join(base_dir, "results")


def save_bid_results(
    optimized_bids: np.ndarray,
    X_test: pd.DataFrame,
    y_click_test: pd.Series,
    y_conversion_test: pd.Series,
    path: str = "output/results/bid_results.csv"
) -> None:
    """
    Save bid optimization results to a CSV file for analysis. Create a DataFrame containing optimized bid values,
    ground truth labels for clicks and conversions, and a subset of feature
    values from the test dataset. The results are saved to the specified CSV path.

    Parameters
    ----------
    optimized_bids : array-like
        Array of optimized bid values for each test instance.
    X_test : pd.DataFrame
        Test feature dataset.
    y_click_test : pd.Series
        Ground truth click labels for test instances.
    y_conversion_test : pd.Series
        Ground truth conversion labels for test instances.
    path : str, default="output/results/bid_results.csv"
        Path where the CSV file will be saved.

    Returns
    -------
    None
        The function does not return any value but saves a CSV file to disk.
    """
    # Create results dataframe
    results_df = pd.DataFrame({
        'bid': optimized_bids,
        'click': y_click_test.values,
        'conversion': y_conversion_test.values
    })

    # Add a few sample features for analysis
    sample_features = X_test.columns[:5]
    for feature in sample_features:
        results_df[feature] = X_test[feature].values

    # Save to CSV
    results_df.to_csv(path, index=False)
    logger.info(f"Bid results saved to {path}")


if __name__ == "__main__":
    start_time = time.time()
    logger.info("Starting RTB pipeline execution")

    # Load configuration
    config = load_config()
    if config is None:
        logger.error("Failed to load configuration. Exiting.")
        exit(1)

    # Create output directories
    models_dir, results_dir = create_output_directories()

    # Initialize components
    logger.info("Initializing data preprocessing and modeling components")
    preprocessor = DataPreprocessor(
        high_cardinality_threshold=config.get("high_cardinality_threshold", 100),
        missing_value_strategy=config.get("missing_value_strategy", "median"),
        scaling_method=config.get("scaling_method", "standard")
    )
    feature_engineer = FeatureEngineer()
    model = TwoStageModel(config)

    # Load data
    logger.info(f"Loading data from {config['data_path']}")
    original_data = pd.read_csv(config["data_path"])
    logger.info(f"Loaded data with shape: {original_data.shape}")

    # Check for missing target columns
    if "click" not in original_data.columns or "conversion" not in original_data.columns:
        logger.error("Data missing required 'click' or 'conversion' columns. Exiting.")
        exit(1)

    # Data preprocessing
    logger.info("Starting data preprocessing")
    preprocessing_start = time.time()
    preprocessed_data = preprocessor.run_all_preprocessing(original_data)
    logger.info(f"Preprocessing completed in {time.time() - preprocessing_start:.2f} seconds")
    logger.info(f"Preprocessed data shape: {preprocessed_data.shape}")

    # Feature engineering
    logger.info("Starting feature engineering")
    feature_eng_start = time.time()
    preprocessed_data = feature_engineer.run_all_feature_engineering(preprocessed_data)
    logger.info(f"Feature engineering completed in {time.time() - feature_eng_start:.2f} seconds")
    logger.info(f"Final data shape: {preprocessed_data.shape}")

    # Prepare data for training
    X = preprocessed_data.drop(columns=["click", "conversion"])
    y_click = preprocessed_data["click"]
    y_conversion = preprocessed_data["conversion"]

    # Log class distribution
    logger.info(f"Click distribution: {y_click.value_counts().to_dict()}")
    logger.info(f"Conversion distribution: {y_conversion.value_counts().to_dict()}")

    # Split data into training and testing sets
    logger.info("Splitting data into training and testing sets")
    X_train, X_test, y_click_train, y_click_test, y_conversion_train, y_conversion_test = train_test_split(
        X, y_click, y_conversion, test_size=0.2, random_state=config.get("random_state", 42)
    )

    # Train models with optimizations based on configuration
    logger.info("Training click prediction model")
    use_hyperopt = config.get("optimize_hyperparameters", False)
    use_feature_selection = config.get("use_feature_selection", True)

    model.train_click_model(
        X_train,
        y_click_train,
        optimize_hyperparams=use_hyperopt,
        feature_selection=use_feature_selection
    )

    logger.info("Training conversion prediction model")
    model.train_conversion_model(
        X_train,
        y_conversion_train,
        optimize_hyperparams=use_hyperopt,
        feature_selection=use_feature_selection
    )

    # Save trained models
    logger.info("Saving trained models")
    model.save_models(path_prefix=os.path.join(models_dir, ""))

    # Evaluate model performance
    logger.info("Evaluating model performance")
    evaluation_results = model.evaluate(X_test, y_click_test, y_conversion_test)

    # Run inference benchmarking
    logger.info("Benchmarking inference performance")
    avg_times, p95_times = model.benchmark_inference(X_test, n_iterations=50)

    # Log inference performance summary
    logger.info(f"Average total inference time: {avg_times['total']:.2f} ms")
    logger.info(f"95th percentile inference time: {p95_times['total']:.2f} ms")

    # Optimize bids with campaign information if available
    logger.info("Running bid optimization")

    # Check if campaign info is defined in config
    campaign_info = config.get("campaign_info", None)
    if campaign_info is None:
        # Create default campaign info
        campaign_info = {
            "type": "cpa",
            "target_cpa": 10.0,
            "conversion_value": 15.0,
            "daily_budget": 5000.0
        }
        logger.info("Using default campaign settings for bid optimization")

    optimized_bids = model.optimize_bid(
        X_test,
        base_bid=config.get("base_bid", 1.0),
        campaign_info=campaign_info
    )

    # Save bid results for analysis
    save_bid_results(
        optimized_bids,
        X_test,
        y_click_test,
        y_conversion_test,
        path=os.path.join(results_dir, "bid_results.csv")
    )

    # Calculate expected campaign performance
    predicted_probs = model.predict(X_test)
    expected_clicks = np.sum(predicted_probs["click_prob"])
    expected_conversions = np.sum(predicted_probs["combined_prob"])
    total_bid_value = np.sum(optimized_bids)

    if expected_clicks > 0:
        expected_cpc = total_bid_value / expected_clicks
        expected_cpa = total_bid_value / max(expected_conversions, 1)
        expected_ctr = expected_clicks / len(X_test) * 100
        expected_cvr = expected_conversions / max(expected_clicks, 1) * 100

        logger.info("\nExpected Campaign Performance:")
        logger.info(f"Expected CTR: {expected_ctr:.2f}%")
        logger.info(f"Expected CVR: {expected_cvr:.2f}%")
        logger.info(f"Expected CPC: ${expected_cpc:.2f}")
        logger.info(f"Expected CPA: ${expected_cpa:.2f}")
        logger.info(f"Total estimated spend: ${total_bid_value:.2f}")

    # Log execution time
    total_time = time.time() - start_time
    logger.info(f"\nRTB pipeline execution completed in {total_time:.2f} seconds")
