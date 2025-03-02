# Real-Time Bidding Strategy Optimization

## Overview

This project implements a sophisticated two-stage prediction system for real-time bidding (RTB) in online advertising. The system predicts both click-through rate (CTR) and conversion rate (CVR) to optimize bid values based on campaign goals, enabling more effective ad spend allocation.

In the competitive RTB landscape, DSPs (Demand-Side Platforms) need to make millisecond-level bidding decisions. This system provides accurate predictions while maintaining the performance requirements necessary for production RTB environments.

## Features

- **Two-Stage Prediction Model**: Separate models for click and conversion prediction with probability calibration
- **Advanced Data Preprocessing**: Handles missing values, high-cardinality features, and feature scaling
- **Feature Engineering**: Temporal features, activity ratios, and other domain-specific features
- **Smart Bid Optimization**: Campaign-aware bidding strategies (CPA, CPC, ROI optimization)
- **Performance Optimization**: Feature selection and inference benchmarking for real-time requirements
- **Class Imbalance Handling**: SMOTE sampling to address the inherent imbalance in click and conversion data
- **Model Persistence**: Save and load trained models for production deployment

## Project Structure

```
bidding_strategy/
├── bidding_utils/
│   ├── __init__.py 
│   ├── data_preprocessing.py    # Data preprocessing and feature engineering
│   └── modelling.py             # Two-stage model implementation
├── configs/
│   └── config.yaml              # Configuration parameters
├── logger/
│   └── logging.py               # Logging utilities
├── output/
│   ├── models/                  # Saved model artifacts
│   ├── results/                 # Bid optimization results
│   └── logs/                    # Execution logs
├── __main__.py                  # Main execution script
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/bidding_strategy.git
   cd bidding_strategy
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Configuration

Edit the `configs/config.yaml` file to configure the system:

```yaml
# Data paths
data_path: "[interview_test_data.csv.zip](https://ak-interview-assets.s3.eu-west-1.amazonaws.com/ml-interview/interview_test_data.csv.zip)"

# Preprocessing settings
high_cardinality_threshold: 100
missing_value_strategy: "median"  # Options: median, mean, mode
scaling_method: "standard"        # Options: standard, minmax, None

# Model settings
random_state: 42
optimize_hyperparameters: false   # Set to true for hyperparameter tuning
use_feature_selection: true
calibrate_probabilities: true

# Click model settings
click_n_estimators: 300
click_num_leaves: 31
click_class_weight: 10

# Conversion model settings
conversion_n_estimators: 300
conversion_num_leaves: 31
conversion_class_weight: 20

# Bid optimization settings
base_bid: 1.0
campaign_info:
  type: "cpa"                     # Options: cpa, cpc, roi
  target_cpa: 10.0
  target_cpc: 0.5
  target_roi: 2.0
  conversion_value: 15.0
  daily_budget: 5000.0
```

### Running the Pipeline

Execute the main script to run the complete RTB pipeline:

```
python -m bidding_strategy
```

This will:
1. Load and preprocess the data
2. Apply feature engineering
3. Train the two-stage model
4. Evaluate model performance
5. Benchmark inference speed
6. Optimize bids based on campaign goals
7. Save results and models

### Input Data Format

The input data should be a CSV file with the following columns:
- Features for prediction (demographic, contextual, behavioral)
- `click` - Binary indicator if the impression was clicked (1/0)
- `conversion` - Binary indicator if the impression led to a conversion (1/0)

Temporal columns like `imp_time`, `click_time` and `transaction_timestamps` are used for feature engineering.

## Data Preprocessing Pipeline

The system implements a comprehensive preprocessing pipeline:

1. **Data Cleaning**:
   - Removal of columns with single values
   - Removal of columns with excessive missing values
   - Removal of highly correlated features

2. **Missing Value Handling**:
   - Configurable strategy (median, mean, mode)
   - Different handling for numerical and categorical features

3. **Feature Encoding**:
   - Target encoding for high-cardinality categorical features
   - Label encoding for low-cardinality categorical features
   - Special handling for city, device model, and bundle identifiers

## Model Details

### Two-Stage Prediction

The system uses a two-stage approach:
1. **Click Model**: Predicts probability of a user clicking on an ad
2. **Conversion Model**: Predicts probability of a conversion after a click

Both models use LightGBM classifiers with:
- Probability calibration for improved bid optimization
- Class weighting to handle imbalance
- Feature selection to improve inference speed

### Bid Optimization Strategies

Multiple campaign-aware bidding strategies are implemented:

- **CPA-based**: Optimizes bids to achieve target cost-per-acquisition
- **CPC-based**: Optimizes bids to achieve target cost-per-click
- **ROI-based**: Optimizes bids to achieve target return on investment
- **Budget Pacing**: Adjusts bids to stay within daily campaign budget

## Performance Considerations

Real-time bidding requires extremely fast inference. The system includes:

- Feature selection to reduce dimensionality
- Benchmarking tools to measure millisecond-level performance
- Separate preprocessing of click and conversion features
- Inference timing reports to identify bottlenecks

## Results and Evaluation

The system generates:
- Model performance metrics (AUC, Average Precision)
- Classification reports for click and conversion models
- Feature importance analysis
- Inference speed benchmarks
- Expected campaign performance (CTR, CVR, CPC, CPA, total spend)
- Bid results CSV with optimized bid values

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Original problem statement by adikteev for a case study.