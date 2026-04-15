import pandas as pd
import numpy as np
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import training_config as cfg   # Import training configuration
except ImportError:
    cfg = None

from training_summary_manager import append_row as append_summary_row
from training_summary_manager import build_run_summary_row, format_top_features


def cfg_value(name, default):
    """Read a config value from training_config.py with a fallback default."""
    if cfg is None:
        return default
    return getattr(cfg, name, default)

# Centralized input/output configuration
TRAIN_FILE = cfg_value('TRAIN_FILE', 'battery_swapping_routing_data_train_time70.csv')
TEST_FILE = cfg_value('TEST_FILE', 'battery_swapping_routing_test_dataset.csv')
TRAINING_SCALE = cfg_value('TRAINING_SCALE', [20000])
TRAINING_RESULTS_DIR = cfg_value('TRAINING_RESULTS_DIR', 'Training_Results_CatBoost')
TRAINING_SUMMARY_CSV = cfg_value('TRAINING_SUMMARY_CSV', os.path.join(TRAINING_RESULTS_DIR, 'training_summary.csv'))
PREDICTION_OUTPUT_TEMPLATE = cfg_value('PREDICTION_OUTPUT_TEMPLATE', 'prediction_CB_scale_{scale}_{ts}.csv')
PROGRESS_PLOT_TEMPLATE = cfg_value('PROGRESS_PLOT_TEMPLATE', 'training_progress_CB_{target}_{scale}_{ts}.png')
USE_LOG_TARGET = cfg_value('USE_LOG_TARGET', True)           # Apply log1p on target; affects training and evaluation space
REPORT_METRIC_SPACE = cfg_value('REPORT_METRIC_SPACE', 'auto')

# Centralized training hyperparameters
TRAIN_VALID_TEST_SIZE = cfg_value('TRAIN_VALID_TEST_SIZE', 0.2)
TRAIN_VALID_RANDOM_STATE = cfg_value('TRAIN_VALID_RANDOM_STATE', 42)
SPLIT_MODE = cfg_value('SPLIT_MODE', 'random')
TIME_SPLIT_COLUMN = cfg_value('TIME_SPLIT_COLUMN', 'datetime')
TIME_SPLIT_RATIO = cfg_value('TIME_SPLIT_RATIO', 0.7)
TIME_SPLIT_ASCENDING = cfg_value('TIME_SPLIT_ASCENDING', True)
CB_CATEGORICAL_FEATURES = cfg_value('CB_CATEGORICAL_FEATURES', ['h3']) # CatBoost native categorical features
CB_ALLOW_WRITING_FILES = cfg_value('CB_ALLOW_WRITING_FILES', False)


CB_PARAMS = cfg_value('CB_PARAMS', {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'learning_rate': 0.03,
    'depth': 9,                  # Symmetric tree depth
    'l2_leaf_reg': 4.0,          # L2 regularization
    'random_seed': TRAIN_VALID_RANDOM_STATE,
    'task_type': 'CPU',
    'devices': '0:1',         
    'thread_count': -1,
    'od_type': 'Iter',           # Early-stopping type
    'od_wait': 10                # Early-stopping patience rounds
})

CB_ITERATIONS = cfg_value('CB_ITERATIONS', 1000)             # Maximum iterations
CB_LOG_EVAL_PERIOD = cfg_value('CB_LOG_EVAL_PERIOD', 5)      # Console logging period


def resolve_report_metric_space():
    """Resolve final metric reporting space to either 'log' or 'raw'."""
    if REPORT_METRIC_SPACE == 'auto':
        return 'log' if USE_LOG_TARGET else 'raw'
    if REPORT_METRIC_SPACE not in {'log', 'raw'}:
        raise ValueError("REPORT_METRIC_SPACE only supports: 'auto' | 'log' | 'raw'")
    if REPORT_METRIC_SPACE == 'log' and not USE_LOG_TARGET:
        raise ValueError("Cannot use log report space when USE_LOG_TARGET=False")
    return REPORT_METRIC_SPACE


def rmse_value(y_true, y_pred):
    """Compute RMSE between ground truth and predictions."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def split_train_valid(df, features, y, y_raw):
    """Split data into train/valid sets using random or time-based strategy."""
    if SPLIT_MODE == 'time':
        if TIME_SPLIT_COLUMN not in df.columns:
            raise ValueError(f"Time split enabled but training data is missing column: {TIME_SPLIT_COLUMN}")

        dt = pd.to_datetime(df[TIME_SPLIT_COLUMN], errors='coerce')
        if dt.isna().all():
            raise ValueError(f"Column {TIME_SPLIT_COLUMN} cannot be parsed as datetime for time split")

        fallback = pd.Timestamp.max if TIME_SPLIT_ASCENDING else pd.Timestamp.min
        ordered_idx = dt.fillna(fallback).sort_values(ascending=TIME_SPLIT_ASCENDING).index
        split_pos = int(len(ordered_idx) * TIME_SPLIT_RATIO)
        if split_pos <= 0 or split_pos >= len(ordered_idx):
            raise ValueError("Time split failed: please check TIME_SPLIT_RATIO")

        train_idx = ordered_idx[:split_pos]
        valid_idx = ordered_idx[split_pos:]
        X = df[features]
        X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
        y_train_raw, y_valid_raw = y_raw.loc[train_idx], y_raw.loc[valid_idx]

        train_time = pd.to_datetime(df.loc[train_idx, TIME_SPLIT_COLUMN], errors='coerce').dropna()
        valid_time = pd.to_datetime(df.loc[valid_idx, TIME_SPLIT_COLUMN], errors='coerce').dropna()
        train_time_range = f"{train_time.min()} -> {train_time.max()}" if not train_time.empty else 'N/A'
        valid_time_range = f"{valid_time.min()} -> {valid_time.max()}" if not valid_time.empty else 'N/A'
        print(
            "Time split done: "
            f"train[{train_time_range}], valid[{valid_time_range}], "
            f"train={len(train_idx)}, valid={len(valid_idx)}"
        )
        return X_train, X_valid, y_train, y_valid, y_train_raw, y_valid_raw, train_time_range, valid_time_range

    X = df[features]
    X_train, X_valid, y_train, y_valid, y_train_raw, y_valid_raw = train_test_split(
        X, y, y_raw,
        test_size=TRAIN_VALID_TEST_SIZE,
        random_state=TRAIN_VALID_RANDOM_STATE
    )
    return X_train, X_valid, y_train, y_valid, y_train_raw, y_valid_raw, 'random_split', 'random_split'

# ==========================================
# 1. Data preprocessing pipeline
# ==========================================
def load_and_preprocess(file_path, scale=None):
    """Load CSV, clean basic columns, fill missing values, and build feature list."""
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading data...")
    if scale:
        print(f"Mode: small-scale run, loading first {scale} rows.")
        df = pd.read_csv(file_path, nrows=scale)
    else:
        print("Mode: full-data training.")
        df = pd.read_csv(file_path)
    
    print(f"Data loaded. Shape: {df.shape}")

    # Keep datetime for time split; it is not part of feature list.
    cols_to_drop = ['region_code', 'Unnamed: 21']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # CatBoost requires categorical features to be string/int and non-null.
    if 'h3' in df.columns:
        df['h3'] = df['h3'].astype(str) # h3 may include irregular values
        
    df = fill_missing_values(df)
    df = add_feature_engineering(df)

    features = [
        'h3', 'temperature', 'wind_level', 'rain_level', 
        'month', 'day_of_week', 'is_weekend', 'hour',
        'rent_mean_7d', 'return_mean_7d', 'lag_nb_rent', 'lag_nb_return',
        'normal_power_bike_count', 'soon_low_power_bike_count', 'low_power_bike_count',
        'latitude', 'longitude',
        # Tree models can learn discrete time variables directly.
        'temp_x_rain', 'available_power_bike_gap', 'is_rush_hour'
    ]
    
    return df, features


def add_feature_engineering(df):
    """Create additional handcrafted features used by CatBoost."""
    df = df.copy()

    if 'hour' in df.columns:
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    else:
        df['is_rush_hour'] = 0

    if 'temperature' in df.columns and 'rain_level' in df.columns:
        df['temp_x_rain'] = df['temperature'] * (1 + df['rain_level'])
    else:
        df['temp_x_rain'] = 0.0

    if 'normal_power_bike_count' in df.columns and 'low_power_bike_count' in df.columns:
        df['available_power_bike_gap'] = df['normal_power_bike_count'] - df['low_power_bike_count']
    else:
        df['available_power_bike_gap'] = 0.0

    return df

def fill_missing_values(df):
    """Fill missing values for both categorical and numeric columns."""
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].fillna('missing').astype(str)
        else:
            df[col] = df[col].fillna(0)
    return df

def validate_required_columns(df, required_cols, dataset_name):
    """Validate required columns exist before training/prediction."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")

def get_run_output_dir(run_timestamp):
    """Create and return per-run output directory by timestamp."""
    run_output_dir = os.path.join(TRAINING_RESULTS_DIR, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir


def get_catboost_train_dir(run_output_dir):
    """Create and return CatBoost internal logging directory for one run."""
    train_dir = os.path.join(run_output_dir, 'catboost_info')
    os.makedirs(train_dir, exist_ok=True)
    return train_dir

def plot_training_progress(train_rmse, valid_rmse, metric_name, metric_space, target_name, scale_tag, run_timestamp, run_output_dir):
    """Plot train/valid metric curves and save a progress image."""
    if not train_rmse or not valid_rmse:
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
    plt.ylabel(metric_name)
    plt.title(f'CatBoost Training Progress - {target_name} ({scale_tag}, {metric_space} space)')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()

    print(f"Training progress figure saved: {fig_path}")

# ==========================================
# 2. Core training function
# ==========================================
def train_model(df, features, target_name, scale_tag='all', run_timestamp='unknown', run_output_dir='.'):
    """Train a CatBoost model for one target and return model + summary dict."""
    print(f"\n{'='*40}")
    print(f"Start training target (CatBoost): [{target_name}]")
    print(f"{'='*40}")

    validate_required_columns(df, features + [target_name], 'train set')

    X = df[features]
    y_raw = df[target_name].astype(float)
    y = y_raw.copy()
    report_space = resolve_report_metric_space()
    target_space_desc = 'log1p(target)' if USE_LOG_TARGET else 'raw target'

    if USE_LOG_TARGET:
        y = np.log1p(y)

    train_start_time = time.time()
    X_train, X_valid, y_train, y_valid, y_train_raw, y_valid_raw, train_time_range, valid_time_range = split_train_valid(
        df=df,
        features=features,
        y=y,
        y_raw=y_raw,
    )

    # Extract categorical feature indices in the feature list.
    cat_features_indices = [features.index(f) for f in CB_CATEGORICAL_FEATURES if f in features]
    print(f"[{time.strftime('%H:%M:%S')}] Building trees; monitor error reduction below:")

    # Initialize CatBoost regressor
    model_kwargs = dict(
        iterations=CB_ITERATIONS,
        cat_features=cat_features_indices,
        allow_writing_files=CB_ALLOW_WRITING_FILES,
        **CB_PARAMS,
    )
    if CB_ALLOW_WRITING_FILES:
        model_kwargs['train_dir'] = get_catboost_train_dir(run_output_dir)

    model = cb.CatBoostRegressor(**model_kwargs)

    # Train model
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        verbose=CB_LOG_EVAL_PERIOD
    )

    best_iter = model.get_best_iteration()
    print(f"\nTraining complete for [{target_name}]! Best iteration: {best_iter}")
    print(f"Report metric space: {report_space}")
    print(f"Training objective (loss_function): {CB_PARAMS['loss_function']}, target space: {target_space_desc}")
    print(f"Training monitor metric (eval_metric): {CB_PARAMS['eval_metric']}, target space: {target_space_desc}")

    best_score = model.get_best_score()
    valid_rmse_objective = best_score.get('validation', {}).get('RMSE', None)

    if report_space == 'log':
        train_rmse_curve = model.evals_result_.get('learn', {}).get('RMSE', [])
        valid_rmse_curve = model.evals_result_.get('validation', {}).get('RMSE', [])
        y_pred_for_report = model.predict(X_valid)
        final_rmse = rmse_value(y_valid, y_pred_for_report)
        if valid_rmse_objective is not None:
            print(f"Best validation RMSE (log space): {valid_rmse_objective:.4f}")
    else:
        train_rmse_curve = []
        valid_rmse_curve = []
        for pred_train_step, pred_valid_step in zip(model.staged_predict(X_train), model.staged_predict(X_valid)):
            if USE_LOG_TARGET:
                pred_train_eval = np.clip(np.expm1(pred_train_step), 0, None)
                pred_valid_eval = np.clip(np.expm1(pred_valid_step), 0, None)
                train_rmse_curve.append(rmse_value(y_train_raw, pred_train_eval))
                valid_rmse_curve.append(rmse_value(y_valid_raw, pred_valid_eval))
            else:
                pred_train_eval = np.clip(pred_train_step, 0, None)
                pred_valid_eval = np.clip(pred_valid_step, 0, None)
                train_rmse_curve.append(rmse_value(y_train, pred_train_eval))
                valid_rmse_curve.append(rmse_value(y_valid, pred_valid_eval))

        y_pred_for_report = model.predict(X_valid)
        if USE_LOG_TARGET:
            y_pred_for_report = np.clip(np.expm1(y_pred_for_report), 0, None)
            final_rmse = rmse_value(y_valid_raw, y_pred_for_report)
        else:
            y_pred_for_report = np.clip(y_pred_for_report, 0, None)
            final_rmse = rmse_value(y_valid, y_pred_for_report)

        if valid_rmse_curve:
            best_iter_for_curve = best_iter if best_iter is not None and best_iter >= 0 else len(valid_rmse_curve) - 1
            best_iter_for_curve = min(best_iter_for_curve, len(valid_rmse_curve) - 1)
            print(f"Best validation RMSE (raw space): {valid_rmse_curve[best_iter_for_curve]:.4f}")
        if valid_rmse_objective is not None:
            print(f"Reference: native CatBoost log RMSE (objective space): {valid_rmse_objective:.4f}")
    
    print(f"Final validation RMSE ({report_space} space): {final_rmse:.4f}")

    training_seconds = time.time() - train_start_time

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.get_feature_importance()
    }).sort_values(by='importance', ascending=False)
    top_features = format_top_features(importance)
    
    print("Top 5 feature importances:")
    print(importance.head(5).to_string(index=False))

    plot_training_progress(
        train_rmse=train_rmse_curve,
        valid_rmse=valid_rmse_curve,
        metric_name='RMSE',
        metric_space=report_space,
        target_name=target_name,
        scale_tag=scale_tag,
        run_timestamp=run_timestamp,
        run_output_dir=run_output_dir
    )

    summary = {
        'target_name': target_name,
        'best_iteration': best_iter,
        'best_validation_rmse_objective': valid_rmse_objective,
        'final_validation_rmse': final_rmse,
        'train_size': len(X_train),
        'valid_size': len(X_valid),
        'train_time_range': train_time_range,
        'valid_time_range': valid_time_range,
        'training_seconds': round(training_seconds, 3),
        'top_features': top_features,
        'report_space': report_space,
        'target_space_desc': target_space_desc,
    }

    return model, summary

def predict_on_test_data(models, feature_cols, test_file, output_file):
    """Run inference on test set and save rent/return predictions to CSV."""
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading test set: {test_file}")
    test_df = pd.read_csv(test_file)
    print(f"Test set loaded. Shape: {test_df.shape}")

    cols_to_drop = ['region_code', 'Unnamed: 21', 'datetime']
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')
    
    if 'h3' in test_df.columns:
        test_df['h3'] = test_df['h3'].astype(str)
    test_df = fill_missing_values(test_df)
    test_df = add_feature_engineering(test_df)

    validate_required_columns(test_df, feature_cols, 'test set')
    X_test = test_df[feature_cols]

    result_df = pd.DataFrame(index=test_df.index)
    rent_pred = models['rent'].predict(X_test)
    return_pred = models['return'].predict(X_test)
    if USE_LOG_TARGET:
        rent_pred = np.expm1(rent_pred)
        return_pred = np.expm1(return_pred)
    result_df['rent_pred'] = np.clip(rent_pred, 0, None)
    result_df['return_pred'] = np.clip(return_pred, 0, None)

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
    
    for scale in TRAINING_SCALE:
        if scale is None:
            input("\nAbout to start full-data CatBoost training. Press Enter to continue...")
            
        df, feature_cols = load_and_preprocess(file_name, scale=scale)
        scale_tag = str(scale) if scale is not None else 'all'
        
        rent_model, rent_summary = train_model(
            df, feature_cols, target_name='rent', scale_tag=scale_tag,
            run_timestamp=run_timestamp, run_output_dir=run_output_dir
        )
        
        return_model, return_summary = train_model(
            df, feature_cols, target_name='return', scale_tag=scale_tag,
            run_timestamp=run_timestamp, run_output_dir=run_output_dir
        )

        output_file = os.path.join(
            run_output_dir,
            PREDICTION_OUTPUT_TEMPLATE.format(
                scale=scale if scale is not None else 'all', ts=run_timestamp
            )
        )
        
        predict_on_test_data(
            models={'rent': rent_model, 'return': return_model},
            feature_cols=feature_cols, test_file=TEST_FILE, output_file=output_file
        )

        run_summary_row = build_run_summary_row(
            run_timestamp=run_timestamp,
            scale_tag=scale_tag,
            train_file=file_name,
            test_file=TEST_FILE,
            output_file=output_file,
            report_space=rent_summary['report_space'],
            use_log_target=USE_LOG_TARGET,
            split_mode=SPLIT_MODE,
            train_size=rent_summary['train_size'],
            valid_size=rent_summary['valid_size'],
            train_time_range=rent_summary['train_time_range'],
            valid_time_range=rent_summary['valid_time_range'],
            rent_summary=rent_summary,
            return_summary=return_summary,
            shared_config={
                'cb_loss_function': CB_PARAMS['loss_function'],
                'cb_eval_metric': CB_PARAMS['eval_metric'],
                'cb_learning_rate': CB_PARAMS['learning_rate'],
                'cb_depth': CB_PARAMS['depth'],
                'cb_l2_leaf_reg': CB_PARAMS['l2_leaf_reg'],
                'cb_random_seed': CB_PARAMS['random_seed'],
                'cb_task_type': CB_PARAMS['task_type'],
                'cb_devices': CB_PARAMS['devices'],
                'cb_od_type': CB_PARAMS['od_type'],
                'cb_od_wait': CB_PARAMS['od_wait'],
                'cb_iterations': CB_ITERATIONS,
                'cb_log_eval_period': CB_LOG_EVAL_PERIOD,
                'train_valid_test_size': TRAIN_VALID_TEST_SIZE,
                'train_valid_random_state': TRAIN_VALID_RANDOM_STATE,
                'categorical_features': '|'.join(CB_CATEGORICAL_FEATURES),
            },
        )
        append_summary_row(TRAINING_SUMMARY_CSV, run_summary_row)
        print(f"Run summary appended to: {TRAINING_SUMMARY_CSV}")
        
        print("\n" + "="*50)
        print(f"CatBoost dual-target training finished for scale {scale if scale else 'ALL'}.")
        print("="*50 + "\n")