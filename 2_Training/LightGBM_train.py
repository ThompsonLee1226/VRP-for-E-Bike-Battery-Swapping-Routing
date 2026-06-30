import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Centralized input/output configuration
TRAIN_FILE = 'battery_swapping_routing_data.csv'
TEST_FILE = 'battery_swapping_routing_test_dataset.csv'
TRAINING_SCALE = [20000, 
                  None
                  ]
TRAINING_RESULTS_DIR = 'Training_Results_LightGBM'
PREDICTION_OUTPUT_TEMPLATE = 'prediction_scale_{scale}_{ts}.csv'
PROGRESS_PLOT_TEMPLATE = 'training_progress_{target}_{scale}_{ts}.png'

# Centralized hyperparameters
TRAIN_VALID_TEST_SIZE = 0.2      # Validation split ratio
TRAIN_VALID_RANDOM_STATE = 42    # Fixed random seed for reproducibility
LGB_CATEGORICAL_FEATURES = ['h3'] # Native categorical feature columns for LightGBM

LGB_PARAMS = {
    'objective': 'regression',   # Regression task
    'metric': 'rmse',            # Validation metric: RMSE
    'boosting_type': 'gbdt',     # Gradient boosting decision trees
    'learning_rate': 0.05,       # Learning rate
    'num_leaves': 63,            # Tree complexity
    'feature_fraction': 0.8,     # Feature subsampling ratio
    'n_jobs': -1,                # Parallel threads
    'verbose': -1                
}

LGB_NUM_BOOST_ROUND = 3000       # Max boosting rounds
LGB_EARLY_STOPPING_ROUNDS = 50   # Early stop patience rounds
LGB_LOG_EVAL_PERIOD = 50         # Logging period
LGB_PROGRESS_REFRESH_EVERY = 10  # Progress bar refresh frequency

# ==========================================
# 1. Data preprocessing pipeline
# ==========================================
def load_and_preprocess(file_path, scale=None):
    """
    Load and preprocess data.
    :param file_path: input data file path
    :param scale: sampled row count; None means full dataset
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading data...")
    if scale:
        print(f"Mode: small-scale run, reading first {scale} rows.")
        df = pd.read_csv(file_path, nrows=scale)
    else:
        print("Mode: full-data training (may require high memory).")
        df = pd.read_csv(file_path)
    
    print(f"Data loaded. Shape: {df.shape}")

    # A. Drop columns identified as low-information in EDA.
    cols_to_drop = ['region_code', 'Unnamed: 21', 'datetime']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # B. Convert categorical features.
    if 'h3' in df.columns:
        df['h3'] = df['h3'].astype('category')
        
    # C. Missing value handling.
    df = fill_missing_values(df)

    # D. Define feature set.
    features = [
        'h3', 'temperature', 'wind_level', 'rain_level', 
        'month', 'day_of_week', 'is_weekend', 'hour',
        'rent_mean_7d', 'return_mean_7d', 'lag_nb_rent', 'lag_nb_return',
        'normal_power_bike_count', 'soon_low_power_bike_count', 'low_power_bike_count',
        'latitude', 'longitude' # Spatial heterogeneity features
    ]
    
    return df, features


def fill_missing_values(df):
    """
    Unified missing-value strategy:
    - Fill categorical columns with "missing" (add category if needed)
    - Fill non-categorical columns with 0
    This prevents pandas category-type fill errors.
    """
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            if 'missing' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(['missing'])
            df[col] = df[col].fillna('missing')
        else:
            df[col] = df[col].fillna(0)
    return df


def validate_required_columns(df, required_cols, dataset_name):
    """
    Validate required columns before train/predict steps.
    Raise early if anything is missing.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} missing required columns: {missing}. "
            f"Current available column count: {len(df.columns)}"
        )


def get_run_output_dir(run_timestamp):
    """
    Create and return run-specific output directory by timestamp.
    """
    run_output_dir = os.path.join(TRAINING_RESULTS_DIR, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir


def plot_training_progress(eval_results, target_name, scale_tag, run_timestamp, run_output_dir):
    """
    Save train/valid RMSE curves for convergence/overfitting checks.
    """
    # LightGBM eval result structure:
    # eval_results['train']['rmse'] -> train RMSE by round
    # eval_results['valid']['rmse'] -> valid RMSE by round
    train_rmse = eval_results.get('train', {}).get('rmse', [])
    valid_rmse = eval_results.get('valid', {}).get('rmse', [])

    if not train_rmse or not valid_rmse:
        print(f"Warning: no training metrics captured for {target_name}, skip plotting.")
        return

    fig_path = os.path.join(
        run_output_dir,
        PROGRESS_PLOT_TEMPLATE.format(target=target_name, scale=scale_tag, ts=run_timestamp)
    )

    rounds = range(1, len(valid_rmse) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, train_rmse, label='Train RMSE', linewidth=1.8)
    plt.plot(rounds, valid_rmse, label='Valid RMSE', linewidth=1.8)
    plt.xlabel('Boosting Round')
    plt.ylabel('RMSE')
    plt.title(f'Training Progress - {target_name} ({scale_tag})')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()

    print(f"Training progress figure saved: {fig_path}")


def format_seconds(seconds):
    """
    Format seconds into HH:MM:SS for ETA display.
    """
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def create_progress_bar_callback(target_name, width=30, refresh_every=5):
    """
    LightGBM progress bar callback with iteration/elapsed/ETA.
    """
    start_time = time.time()

    def _callback(env):
        current_iter = env.iteration + 1
        total_iter = env.end_iteration

        # Control refresh frequency to avoid excessive terminal output.
        if (current_iter % refresh_every != 0) and (current_iter != total_iter):
            return

        progress = current_iter / total_iter if total_iter else 0
        filled = int(width * progress)
        bar = "#" * filled + "-" * (width - filled)

        elapsed = time.time() - start_time
        avg_per_iter = elapsed / current_iter if current_iter else 0
        eta = avg_per_iter * max(total_iter - current_iter, 0)

        msg = (
            f"\r[{target_name}] [{bar}] "
            f"{current_iter}/{total_iter} "
            f"elapsed {format_seconds(elapsed)} "
            f"ETA {format_seconds(eta)}"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

        if current_iter == total_iter:
            sys.stdout.write("\n")

    _callback.order = 15
    return _callback

# ==========================================
# 2. Core training function (Training Engine)
# ==========================================
def train_model(df, features, target_name, scale_tag='all', run_timestamp='unknown', run_output_dir='.'):
    """
    Train LightGBM for a specific target (rent or return).
    """
    print(f"\n{'='*40}")
    print(f"Start training target: [{target_name}]")
    print(f"{'='*40}")

    # Validate columns early for clearer failures.
    validate_required_columns(df, features + [target_name], 'train set')

    X = df[features]
    y = df[target_name]

    # Use random split with 20% validation.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TRAIN_VALID_TEST_SIZE,
        random_state=TRAIN_VALID_RANDOM_STATE
    )

    # Pack into LightGBM Dataset format.
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=LGB_CATEGORICAL_FEATURES)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    print(f"[{time.strftime('%H:%M:%S')}] Building trees; monitor error trend below:")

    # Collect per-round metrics for progress plots.
    eval_results = {}
    
    # Train model with callbacks.
    model = lgb.train(
        LGB_PARAMS,
        train_data,
        num_boost_round=LGB_NUM_BOOST_ROUND,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            create_progress_bar_callback(target_name=target_name, refresh_every=LGB_PROGRESS_REFRESH_EVERY),
            lgb.record_evaluation(eval_results),
            lgb.log_evaluation(period=LGB_LOG_EVAL_PERIOD),
            lgb.early_stopping(stopping_rounds=LGB_EARLY_STOPPING_ROUNDS)
        ]
    )

    # Print final stop status if early stopping is triggered.
    if model.best_iteration < LGB_NUM_BOOST_ROUND:
        print(f"[{target_name}] Early stopped at round {model.best_iteration} (early_stopping enabled).")

    # Final evaluation
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    final_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(f"\nTraining complete for [{target_name}]! Best iteration: {model.best_iteration}")
    print(f"Final validation RMSE: {final_rmse:.4f}")

    # Feature importance for reporting.
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain') # Use gain-based importance
    }).sort_values(by='importance', ascending=False)
    
    print("Top 5 feature importances:")
    print(importance.head(5).to_string(index=False))

    # Save training progress curves.
    plot_training_progress(
        eval_results,
        target_name=target_name,
        scale_tag=scale_tag,
        run_timestamp=run_timestamp,
        run_output_dir=run_output_dir
    )
    
    return model


def predict_on_test_data(models, feature_cols, test_file, output_file):
    
    """
    Predict on test set with trained rent/return models and export results.
    :param models: {'rent': rent_model, 'return': return_model}
    :param feature_cols: feature columns used in training
    :param test_file: test CSV path
    :param output_file: output prediction file path
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading test set: {test_file}")
    test_df = pd.read_csv(test_file)
    print(f"Test set loaded. Shape: {test_df.shape}")

    # Apply the same preprocessing as training.
    cols_to_drop = ['region_code', 'Unnamed: 21', 'datetime']
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')
    if 'h3' in test_df.columns:
        test_df['h3'] = test_df['h3'].astype('category')
    test_df = fill_missing_values(test_df)

    validate_required_columns(test_df, feature_cols, 'test set')
    X_test = test_df[feature_cols]

    # Predict both targets.
    result_df = pd.DataFrame(index=test_df.index)
    result_df['rent_pred'] = models['rent'].predict(X_test, num_iteration=models['rent'].best_iteration)
    result_df['return_pred'] = models['return'].predict(X_test, num_iteration=models['return'].best_iteration)

    # Keep one traceable ID column for downstream alignment.
    for id_col in ['id', 'station_id', 'h3']:
        if id_col in test_df.columns:
            result_df.insert(0, id_col, test_df[id_col].values)
            break

    result_df.to_csv(output_file, index=False)
    print(f"Test prediction complete. Saved to: {output_file}")

# ==========================================
# 3. Main pipeline
# ==========================================
if __name__ == "__main__":
    file_name = TRAIN_FILE
    os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_output_dir = get_run_output_dir(run_timestamp)
    print(f"Run timestamp: {run_timestamp}")
    print(f"Run output directory: {run_output_dir}")
    
    # Scale settings
    training_scales = TRAINING_SCALE 
    
    for scale in training_scales:
        if scale is None:
            input("\nAbout to start full-data training. Ensure enough memory, then press Enter to continue...")
            
        # 1. Load data.
        df, feature_cols = load_and_preprocess(file_name, scale=scale)
        scale_tag = str(scale) if scale is not None else 'all'
        
        # 2. Train outflow model (rent).
        rent_model = train_model(
            df,
            feature_cols,
            target_name='rent',
            scale_tag=scale_tag,
            run_timestamp=run_timestamp,
            run_output_dir=run_output_dir
        )
        
        # 3. Train inflow model (return).
        return_model = train_model(
            df,
            feature_cols,
            target_name='return',
            scale_tag=scale_tag,
            run_timestamp=run_timestamp,
            run_output_dir=run_output_dir
        )

        # 4. Predict both targets on test set and export result file.
        output_file = os.path.join(
            run_output_dir,
            PREDICTION_OUTPUT_TEMPLATE.format(
                scale=scale if scale is not None else 'all',
                ts=run_timestamp
            )
        )
        predict_on_test_data(
            models={'rent': rent_model, 'return': return_model},
            feature_cols=feature_cols,
            test_file=TEST_FILE,
            output_file=output_file
        )
        
        print("\n" + "="*50)
        print(f"Dual-target training and test prediction completed for scale {scale if scale else 'ALL'}.")
        print("="*50 + "\n")