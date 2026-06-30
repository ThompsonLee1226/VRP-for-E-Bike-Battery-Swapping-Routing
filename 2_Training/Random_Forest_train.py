import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
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
TRAINING_RESULTS_DIR = 'Training_Results_RF'
PREDICTION_OUTPUT_TEMPLATE = 'prediction_RF_scale_{scale}_{ts}.csv'
PROGRESS_PLOT_TEMPLATE = 'training_progress_RF_{target}_{scale}_{ts}.png'

# Centralized hyperparameters
TRAIN_VALID_TEST_SIZE = 0.2               # Validation split ratio
TRAIN_VALID_RANDOM_STATE = 42             # Fixed random seed for reproducibility
RF_STAGE_ESTIMATORS = list(range(50, 3001, 50)) # Stage-wise tree counts for convergence tracking

RF_WARM_START_PARAMS = {
    'n_estimators': RF_STAGE_ESTIMATORS[0], # Initial tree count; increased by stages
    'max_depth': 18,                        # Max tree depth
    'min_samples_leaf': 5,                  # Minimum samples per leaf
    'max_features': 'sqrt',                 # Features considered per split
    'n_jobs': -1,                           # Parallel threads, -1 uses all CPU
    'verbose': 0,                           # Disable low-level logs
    'warm_start': True,                     # Continue adding trees on existing forest
    'random_state': 42                      # Fixed randomness
}

RF_FINAL_MODEL_PARAMS = {
    'n_estimators': RF_STAGE_ESTIMATORS[0], # Replaced by best tree count after staging
    'max_depth': 18,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'n_jobs': -1,
    'verbose': 0,
    'random_state': 42
}

EARLY_STOPPING_PATIENCE = 3     # Stop after this many no-improvement stages
EARLY_STOPPING_MIN_DELTA = 1e-4 # Minimum RMSE drop counted as improvement

# ==========================================
# 1. Data preprocessing pipeline
# ==========================================
def load_and_preprocess(file_path, scale=None):
    """
    Load and preprocess training data.
    :param file_path: data file path
    :param scale: sampled row count; None means full dataset
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading data...")
    if scale:
        print(f"Mode: small-scale run, reading first {scale} rows.")
        df = pd.read_csv(file_path, nrows=scale)
    else:
        print("Mode: full-data training (Random Forest can be time-consuming).")
        df = pd.read_csv(file_path)
    
    print(f"Data loaded. Shape: {df.shape}")

    cols_to_drop = ['region_code', 'Unnamed: 21', 'datetime']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # RF does not directly support string categories; encode with stable mapping.
    h3_mapping = {}
    if 'h3' in df.columns:
        df['h3'], h3_mapping = encode_h3_with_mapping(df['h3'])

    df = fill_missing_values(df)

    features = [
        'h3', 'temperature', 'wind_level', 'rain_level', 
        'month', 'day_of_week', 'is_weekend', 'hour',
        'rent_mean_7d', 'return_mean_7d', 'lag_nb_rent', 'lag_nb_return',
        'normal_power_bike_count', 'soon_low_power_bike_count', 'low_power_bike_count',
        'latitude', 'longitude'
    ]
    
    return df, features, h3_mapping


def encode_h3_with_mapping(series):
    """
    Encode h3 strings to integers and return the mapping.
    """
    clean_series = series.fillna('missing').astype(str)
    categories = pd.Index(clean_series.unique())
    mapping = {cat: idx for idx, cat in enumerate(categories)}
    encoded = clean_series.map(mapping).astype(np.int32)
    return encoded, mapping

def fill_missing_values(df):
    """
    Unified missing-value strategy for RF: fill all with 0.
    """
    for col in df.columns:
        df[col] = df[col].fillna(0)
    return df

def validate_required_columns(df, required_cols, dataset_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")


def get_run_output_dir(run_timestamp):
    """
    Create and return run-specific output directory by timestamp.
    """
    run_output_dir = os.path.join(TRAINING_RESULTS_DIR, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir


def format_seconds(seconds):
    """
    Format seconds as HH:MM:SS for ETA display.
    """
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def plot_training_progress(progress_df, target_name, scale_tag, run_timestamp, run_output_dir):
    """
    Save stage-wise RF RMSE curves as a plot.
    """
    if progress_df.empty:
        print(f"Warning: no training metrics captured for {target_name}, skip plotting.")
        return

    fig_path = os.path.join(
        run_output_dir,
        PROGRESS_PLOT_TEMPLATE.format(target=target_name, scale=scale_tag, ts=run_timestamp)
    )

    plt.figure(figsize=(8, 5))
    plt.plot(progress_df['n_estimators'], progress_df['train_rmse'], label='Train RMSE', linewidth=1.8)
    plt.plot(progress_df['n_estimators'], progress_df['valid_rmse'], label='Valid RMSE', linewidth=1.8)
    plt.xlabel('Number of Trees')
    plt.ylabel('RMSE')
    plt.title(f'RF Training Progress - {target_name} ({scale_tag})')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()

    print(f"Random Forest training progress figure saved: {fig_path}")

# ==========================================
# 2. Core training function
# ==========================================
def train_model(df, features, target_name, scale_tag='all', run_timestamp='unknown', run_output_dir='.'):
    """
    Train Random Forest for target (rent or return).
    Increase trees by warm_start stages and record train/valid RMSE.
    Use early stopping when validation RMSE does not improve significantly.
    """
    print(f"\n{'='*40}")
    print(f"Start training target (RF): [{target_name}]")
    print(f"{'='*40}")

    validate_required_columns(df, features + [target_name], 'train set')

    X = df[features]
    y = df[target_name]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TRAIN_VALID_TEST_SIZE,
        random_state=TRAIN_VALID_RANDOM_STATE
    )

    stage_estimators = RF_STAGE_ESTIMATORS
    total_stages = len(stage_estimators)

    model = RandomForestRegressor(**RF_WARM_START_PARAMS)

    print(f"[{time.strftime('%H:%M:%S')}] Growing forest by stages; monitor RMSE changes:")
    start_time = time.time()

    progress_records = []
    best_valid_rmse = float('inf')
    best_n_estimators = stage_estimators[0]
    no_improve_rounds = 0
    stopped_early = False

    for idx, n_trees in enumerate(stage_estimators, start=1):
        model.set_params(n_estimators=n_trees)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

        progress_records.append(
            {
                'stage': idx,
                'n_estimators': n_trees,
                'train_rmse': train_rmse,
                'valid_rmse': valid_rmse
            }
        )

        if valid_rmse < (best_valid_rmse - EARLY_STOPPING_MIN_DELTA):
            best_valid_rmse = valid_rmse
            best_n_estimators = n_trees
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        elapsed = time.time() - start_time
        avg_stage = elapsed / idx
        eta = avg_stage * (total_stages - idx)
        msg = (
            f"\r[{target_name}] stage {idx}/{total_stages} "
            f"trees={n_trees} "
            f"train_rmse={train_rmse:.4f} "
            f"valid_rmse={valid_rmse:.4f} "
            f"elapsed {format_seconds(elapsed)} "
            f"ETA {format_seconds(eta)}"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

        if no_improve_rounds >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            break

    sys.stdout.write("\n")

    if stopped_early:
        print(
            f"[{target_name}] Early stopping triggered: validation RMSE did not significantly improve "
            f"for {EARLY_STOPPING_PATIENCE} stages (min_delta={EARLY_STOPPING_MIN_DELTA})."
        )

    # Retrain final model with best tree count.
    final_model_params = RF_FINAL_MODEL_PARAMS.copy()
    final_model_params['n_estimators'] = best_n_estimators
    final_model = RandomForestRegressor(**final_model_params)
    final_model.fit(X_train, y_train)

    # Final evaluation
    y_pred = final_model.predict(X_valid)
    final_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(f"\nTraining complete for [{target_name}]! Best tree count: {best_n_estimators}")
    print(f"Final validation RMSE: {final_rmse:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': final_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    print("Top 5 feature importances:")
    print(importance.head(5).to_string(index=False))

    progress_df = pd.DataFrame(progress_records)
    plot_training_progress(
        progress_df,
        target_name=target_name,
        scale_tag=scale_tag,
        run_timestamp=run_timestamp,
        run_output_dir=run_output_dir
    )
    
    return final_model


def preprocess_test_data(test_df, h3_mapping):
    """
    Apply the same preprocessing as training.
    h3 uses the training mapping; unknown categories map to -1.
    """
    cols_to_drop = ['region_code', 'Unnamed: 21', 'datetime']
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')

    if 'h3' in test_df.columns:
        test_df['h3'] = test_df['h3'].fillna('missing').astype(str).map(h3_mapping).fillna(-1).astype(np.int32)

    test_df = fill_missing_values(test_df)
    return test_df


def predict_on_test_data(models, feature_cols, test_file, output_file, h3_mapping):
    """
    Predict on test set using trained rent/return models and export results.
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading test set: {test_file}")
    test_df = pd.read_csv(test_file)

    print(f"Test set loaded. Shape: {test_df.shape}")
    test_df = preprocess_test_data(test_df, h3_mapping)
    validate_required_columns(test_df, feature_cols, 'test set')

    X_test = test_df[feature_cols]

    result_df = pd.DataFrame(index=test_df.index)
    result_df['rent_pred'] = models['rent'].predict(X_test)
    result_df['return_pred'] = models['return'].predict(X_test)

    # Keep one ID column for downstream merge-back.
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
    os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_output_dir = get_run_output_dir(run_timestamp)
    print(f"Run timestamp: {run_timestamp}")
    print(f"Run output directory: {run_output_dir}")
    
    training_scales = TRAINING_SCALE
    for scale in training_scales:
        if scale is None:
            input("\nAbout to start full-data Random Forest training (early stopping enabled). Press Enter to continue...")
            
        df, feature_cols, h3_mapping = load_and_preprocess(TRAIN_FILE, scale=scale)
        scale_tag = str(scale) if scale is not None else 'all'
        
        rent_model = train_model(df, feature_cols, 'rent', scale_tag, run_timestamp, run_output_dir)
        return_model = train_model(df, feature_cols, 'return', scale_tag, run_timestamp, run_output_dir)

        output_file = os.path.join(
            run_output_dir,
            PREDICTION_OUTPUT_TEMPLATE.format(scale=scale_tag, ts=run_timestamp)
        )
        predict_on_test_data(
            {'rent': rent_model, 'return': return_model},
            feature_cols,
            TEST_FILE,
            output_file,
            h3_mapping
        )
        
        print("\n" + "="*50)
        print(f"Random Forest dual-target training and test prediction completed for scale {scale_tag}.")
        print("="*50 + "\n")