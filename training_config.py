"""Central training configuration.

Edit this file only when you want to change input files, split settings,
model hyperparameters, or output locations.
"""

from __future__ import annotations

import os

# Input / output
TRAIN_FILE = 'battery_swapping_routing_data_train_time70.csv'  # Training CSV path
TEST_FILE = 'battery_swapping_routing_data_valid_time30.csv'  # Validation/Test CSV path
TRAINING_SCALE = [20000]  # Row scale list; None means full dataset
TRAINING_RESULTS_DIR = 'Training_Results_CatBoost'  # Root results directory
TRAINING_SUMMARY_CSV = os.path.join(TRAINING_RESULTS_DIR, 'training_summary.csv')  # Long-term summary CSV path
PREDICTION_OUTPUT_TEMPLATE = 'prediction_CB_scale_{scale}_{ts}.csv'  # Prediction filename template
PROGRESS_PLOT_TEMPLATE = 'training_progress_CB_{target}_{scale}_{ts}.png'  # Training-curve filename template

# Target transformation and evaluation space
USE_LOG_TARGET = True  # Whether to apply log1p on target y
REPORT_METRIC_SPACE = 'auto'  # Metric space: auto (follow USE_LOG_TARGET) | log | raw

# Data split settings
TRAIN_VALID_TEST_SIZE = 0.2  # Validation ratio for random split
TRAIN_VALID_RANDOM_STATE = 42  # Random seed for split reproducibility
SPLIT_MODE = 'time'  # Split mode: random | time
TIME_SPLIT_COLUMN = 'datetime'  # Time column name for time-based split
TIME_SPLIT_RATIO = 0.8  # Training ratio in time split (first 80% train, last 20% valid)
TIME_SPLIT_ASCENDING = True  # Time order, True means old -> new

# Model settings
CB_CATEGORICAL_FEATURES = ['h3']  # CatBoost categorical feature list
CB_PARAMS = {
    'loss_function': 'RMSE',  # Training objective
    'eval_metric': 'RMSE',  # Training monitoring metric
    'learning_rate': 0.03,  # Learning rate
    'depth': 9,  # Tree depth
    'l2_leaf_reg': 4.0,  # L2 regularization
    'random_seed': TRAIN_VALID_RANDOM_STATE,  # Model random seed
    'task_type': 'GPU',  # Training device type
    'devices': '0:1',  # GPU device IDs
    'thread_count': -1,  # CPU threads, -1 means auto
    'od_type': 'Iter',  # Early-stopping strategy type
    'od_wait': 10,  # Early-stopping wait rounds
}
CB_ITERATIONS = 1000  # Max training iterations
CB_LOG_EVAL_PERIOD = 5  # Log interval in rounds
