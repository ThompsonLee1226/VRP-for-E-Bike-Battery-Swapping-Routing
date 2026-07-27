import pandas as pd
import numpy as np
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_poisson_deviance
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import training_config as cfg
except ImportError:
    cfg = None

from training_summary_manager import append_row as append_summary_row
from training_summary_manager import build_run_summary_row


def cfg_value(name, default):
    if cfg is None:
        return default
    return getattr(cfg, name, default)

# Centralized input/output configuration
TRAIN_FILE = cfg_value('TRAIN_FILE', 'battery_swapping_routing_data_train_time70.csv')
TEST_FILE = cfg_value('TEST_FILE', 'battery_swapping_routing_data_valid_time30.csv')
TRAINING_SCALE = cfg_value('TRAINING_SCALE', [100000])
TRAINING_RESULTS_DIR = cfg_value('HURDLE_TRAINING_RESULTS_DIR', cfg_value('TRAINING_RESULTS_DIR', 'Training_Results'))
TRAINING_SUMMARY_CSV = cfg_value('TRAINING_SUMMARY_CSV', None)  # None → 运行时写入 run_output_dir
PREDICTION_OUTPUT_TEMPLATE = cfg_value('HURDLE_PREDICTION_OUTPUT_TEMPLATE', cfg_value('PREDICTION_OUTPUT_TEMPLATE', 'prediction_CB_Hurdle_scale_{scale}_{ts}.csv'))
PROGRESS_PLOT_TEMPLATE = cfg_value('HURDLE_PROGRESS_PLOT_TEMPLATE', cfg_value('PROGRESS_PLOT_TEMPLATE', 'training_progress_CB_hurdle_{target}_{context}_{scale}_{ts}.png'))
USE_LOG_TARGET = cfg_value('USE_LOG_TARGET', False)

# Centralized hyperparameters
TRAIN_VALID_TEST_SIZE = cfg_value('TRAIN_VALID_TEST_SIZE', 0.2)
TRAIN_VALID_RANDOM_STATE = cfg_value('TRAIN_VALID_RANDOM_STATE', 42)
SPLIT_MODE = cfg_value('SPLIT_MODE', 'time')
TIME_SPLIT_COLUMN = cfg_value('TIME_SPLIT_COLUMN', 'datetime')
TIME_SPLIT_RATIO = cfg_value('TIME_SPLIT_RATIO', 0.8)
TIME_SPLIT_ASCENDING = cfg_value('TIME_SPLIT_ASCENDING', True)
CB_CATEGORICAL_FEATURES = cfg_value('CB_CATEGORICAL_FEATURES', ['h3'])
CB_ALLOW_WRITING_FILES = cfg_value('CB_ALLOW_WRITING_FILES', True)

# Shared base parameters
CB_PARAMS = {
    'learning_rate': 0.03,
    'depth': 10,                  
    'l2_leaf_reg': 4.0,          
    'random_seed': TRAIN_VALID_RANDOM_STATE,
    'task_type': 'GPU',
    'devices': '0:1',
    'thread_count': -1,
    'od_type': 'Iter',           
    'od_wait': 30                
}

# Classifier and regressor use different objective settings
CB_CLASSIFIER_PARAMS = CB_PARAMS.copy()
CB_CLASSIFIER_PARAMS.update({
    'loss_function': 'Logloss',  # Logloss for classification
    'eval_metric': 'Logloss'         # AUC as classification metric
})

CB_REGRESSOR_PARAMS = CB_PARAMS.copy()
CB_REGRESSOR_PARAMS.update({
    'loss_function': 'Poisson',     # Poisson for regression
    'eval_metric': 'Poisson'
})

CB_ITERATIONS = 10000            
CB_LOG_EVAL_PERIOD = 50          

# ==========================================
# 1. Data preprocessing pipeline
# ==========================================
def load_and_preprocess(file_path, scale=None):
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading data...")
    if scale:
        print(f"Mode: small-scale run, reading first {scale} rows.")
        df = pd.read_csv(file_path, nrows=scale)
    else:
        print("Mode: full-data training.")
        df = pd.read_csv(file_path)
    
    print(f"Data loaded. Shape: {df.shape}")

    cols_to_drop = ['region_code', 'Unnamed: 21']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    if 'h3' in df.columns:
        df['h3'] = df['h3'].astype(str)
        
    df = fill_missing_values(df)
    df = add_feature_engineering(df)

    features = [
        'h3', 'temperature', 'wind_level', 'rain_level',
        'month', 'day_of_week', 'is_weekend', 'hour',
        'rent_mean_7d', 'return_mean_7d', 'lag_nb_rent', 'lag_nb_return',
        'normal_power_bike_count', 'soon_low_power_bike_count', 'low_power_bike_count',
        'latitude', 'longitude',
        'temp_x_rain', 'available_power_bike_gap', 'is_rush_hour',
        'lag_rent_is_zero', 'lag_return_is_zero',
        'rent_mean_7d_low', 'return_mean_7d_low'
    ]
    
    return df, features

def add_feature_engineering(df):
    df = df.copy()

    if 'hour' in df.columns:
        # Rush hour rule: 7-9 AM and 5-7 PM.
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

    # ── 零需求指示特征：帮助分类器识别零需求场景 ──
    # 上一时刻的需求是否为零（强零需求信号）
    if 'lag_nb_rent' in df.columns:
        df['lag_rent_is_zero'] = (df['lag_nb_rent'] == 0).astype(int)
    else:
        df['lag_rent_is_zero'] = 0

    if 'lag_nb_return' in df.columns:
        df['lag_return_is_zero'] = (df['lag_nb_return'] == 0).astype(int)
    else:
        df['lag_return_is_zero'] = 0

    # 7日滚动均值是否极低（<0.1 视为"几乎无需求"区域）
    if 'rent_mean_7d' in df.columns:
        df['rent_mean_7d_low'] = (df['rent_mean_7d'] < 0.1).astype(int)
    else:
        df['rent_mean_7d_low'] = 0

    if 'return_mean_7d' in df.columns:
        df['return_mean_7d_low'] = (df['return_mean_7d'] < 0.1).astype(int)
    else:
        df['return_mean_7d_low'] = 0

    return df

def fill_missing_values(df):
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].fillna('missing').astype(str)
        else:
            df[col] = df[col].fillna(0)
    return df

def validate_required_columns(df, required_cols, dataset_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")

def get_run_output_dir(run_timestamp):
    run_output_dir = os.path.join(TRAINING_RESULTS_DIR, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir

def get_catboost_train_dir(run_output_dir, target, context):
    train_dir = os.path.join(run_output_dir, f'catboost_info_{target}_{context}')
    os.makedirs(train_dir, exist_ok=True)
    return train_dir

def plot_training_progress(
    evals_result,
    target_name,
    context_name,
    scale_tag,
    run_timestamp,
    run_output_dir,
    metric_key,
    metric_label,
):
    if not evals_result:
        return

    train_curve = evals_result.get('learn', {}).get(metric_key, [])
    valid_curve = evals_result.get('validation', {}).get(metric_key, [])

    fig_path = os.path.join(
        run_output_dir,
        PROGRESS_PLOT_TEMPLATE.format(target=target_name, context=context_name, scale=scale_tag, ts=run_timestamp)
    )

    rounds = range(1, len(valid_curve) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, train_curve, label=f'Train {metric_label}', linewidth=1.8)
    plt.plot(rounds, valid_curve, label=f'Valid {metric_label}', linewidth=1.8)
    plt.xlabel('Boosting Round')
    plt.ylabel(metric_label)
    title_suffix = 'log1p target' if USE_LOG_TARGET else 'raw target'
    plt.title(f'CatBoost [{context_name}] Progress - {target_name} ({scale_tag}, {title_suffix})')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()
    print(f"[{context_name}] Training progress figure saved: {fig_path}")


def split_train_valid(df, features, y_raw):
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
        return X_train, X_valid, y_train_raw, y_valid_raw, train_time_range, valid_time_range

    X = df[features]
    X_train, X_valid, y_train_raw, y_valid_raw = train_test_split(
        X,
        y_raw,
        test_size=TRAIN_VALID_TEST_SIZE,
        random_state=TRAIN_VALID_RANDOM_STATE,
    )
    return X_train, X_valid, y_train_raw, y_valid_raw, 'random_split', 'random_split'

# ==========================================
# 2. Core training function
# ==========================================
def train_model(df, features, target_name, scale_tag='all', run_timestamp='unknown', run_output_dir='.'):
    print(f"\n{'='*50}")
    print(f"Start training zero-inflated Hurdle model -> target: [{target_name}]")
    print(f"{'='*50}")

    validate_required_columns(df, features + [target_name], 'train set')

    split_df = df.copy()
    X_full = split_df[features]
    y_full = split_df[target_name].astype(float) # Keep original ground truth

    X_train, X_valid, y_train_raw, y_valid_raw, train_time_range, valid_time_range = split_train_valid(
        split_df, features, y_full
    )

    cat_features_indices = [features.index(f) for f in CB_CATEGORICAL_FEATURES if f in features]

    # --- Stage 1: Train classifier (predict whether target > 0) ---
    print(f"[{time.strftime('%H:%M:%S')}] Stage 1: training classifier (whether demand exists)...")
    y_train_bin = (y_train_raw > 0).astype(int)
    y_valid_bin = (y_valid_raw > 0).astype(int)

    classifier = cb.CatBoostClassifier(
        iterations=CB_ITERATIONS,
        cat_features=cat_features_indices,
        allow_writing_files=CB_ALLOW_WRITING_FILES,
        **CB_CLASSIFIER_PARAMS
    )
    if CB_ALLOW_WRITING_FILES:
        classifier.set_params(train_dir=get_catboost_train_dir(run_output_dir, target_name, "Classifier"))
    classifier.fit(X_train, y_train_bin, eval_set=(X_valid, y_valid_bin),
                   use_best_model=True, verbose=CB_LOG_EVAL_PERIOD)
    
    plot_training_progress(
        classifier.evals_result_,
        target_name,
        "Classifier",
        scale_tag,
        run_timestamp,
        run_output_dir,
        metric_key="Logloss",
        metric_label="Logloss",
    )
   
    # --- Stage 2: Train regressor (only on samples with target > 0) ---
    print(f"\n[{time.strftime('%H:%M:%S')}] Stage 2: training regressor (fit positive demand only)...")
    mask_train_pos = y_train_raw > 0
    X_train_pos = X_train[mask_train_pos]
    y_train_pos = y_train_raw[mask_train_pos]
    
    mask_valid_pos = y_valid_raw > 0
    X_valid_pos = X_valid[mask_valid_pos]
    y_valid_pos = y_valid_raw[mask_valid_pos]

    # Note: No np.log1p() used here because Poisson loss handles the log-link internally.

    regressor = cb.CatBoostRegressor(
        iterations=CB_ITERATIONS,
        cat_features=cat_features_indices,
        allow_writing_files=CB_ALLOW_WRITING_FILES,
        **CB_REGRESSOR_PARAMS
    )
    if CB_ALLOW_WRITING_FILES:
        regressor.set_params(train_dir=get_catboost_train_dir(run_output_dir, target_name, "Regressor"))
    regressor.fit(X_train_pos, y_train_pos, eval_set=(X_valid_pos, y_valid_pos),
                  use_best_model=True, verbose=CB_LOG_EVAL_PERIOD)
                  
    plot_training_progress(
        regressor.evals_result_,
        target_name,
        "Regressor",
        scale_tag,
        run_timestamp,
        run_output_dir,
        metric_key="Poisson",
        metric_label="Poisson",
    )
    # --- Stage 3: Joint error evaluation (soft product, pre-threshold) ---
    print(f"\n--- Joint Evaluation: P(Demand) * E(Quantity) ---")
    eps = 1e-6  # 用于 Poisson deviance 的安全裁剪（y_pred 必须 > 0）
    prob_valid = classifier.predict_proba(X_valid)[:, 1]
    val_valid = regressor.predict(X_valid)

    # Note: No np.expm1() used here because Poisson output is already the expected value.
    val_valid = np.clip(val_valid, 0, None)

    soft_pred = prob_valid * val_valid
    soft_pred_safe = np.clip(soft_pred, eps, None)  # Poisson deviance 要求 y_pred > 0
    soft_poisson = mean_poisson_deviance(y_valid_raw, soft_pred_safe)

    classifier_logloss = log_loss(y_valid_bin, prob_valid)
    regressor_poisson_pos = mean_poisson_deviance(y_valid_pos, regressor.predict(X_valid_pos))

    # --- Stage 4: Optimize zero-decision threshold τ (constrained) ---
    # 在验证集上搜索最优阈值 τ，使用约束优化策略。
    #
    # 关键设计（v2 — 修复 Poisson 暴涨问题）：
    #   纯 Poisson 准则下 τ* 几乎总是 0（假阴性惩罚极端不对称）。
    #   纯 F1 准则下 τ* 过于激进（~0.59），导致 Poisson 暴涨 3-4×。
    #
    #   新策略：Poisson-约束的 F1 最大化 ——
    #   在 Poisson deviance 不超过 soft-product 的 10% 范围内，选择 F1 最高的 τ。
    print(f"\n--- Stage 4: Optimizing zero-decision threshold τ (constrained) ---")
    true_zero_ratio = (y_valid_raw <= eps).mean()
    n_true_zero = (y_valid_raw <= eps).sum()

    # ── 第一轮搜索：记录所有 τ 对应的指标 ──
    grid_tau = np.arange(0.01, 0.91, 0.02)
    grid_poisson = np.empty(len(grid_tau))
    grid_f1 = np.empty(len(grid_tau))
    grid_recall = np.empty(len(grid_tau))
    grid_precision = np.empty(len(grid_tau))
    grid_pred_zero_ratio = np.empty(len(grid_tau))

    for i, tau in enumerate(grid_tau):
        pred_zero_mask = prob_valid < tau
        thresh_pred = np.where(pred_zero_mask, 0.0, soft_pred)
        thresh_pred_safe = np.clip(thresh_pred, eps, None)
        grid_poisson[i] = mean_poisson_deviance(y_valid_raw, thresh_pred_safe)

        n_pred_zero = pred_zero_mask.sum()
        n_correct_zero = ((y_valid_raw <= eps) & pred_zero_mask).sum()
        rec = n_correct_zero / n_true_zero if n_true_zero > 0 else 0.0
        prec = n_correct_zero / n_pred_zero if n_pred_zero > 0 else 0.0
        f1 = (2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0.0)
        grid_recall[i] = rec
        grid_precision[i] = prec
        grid_f1[i] = f1
        grid_pred_zero_ratio[i] = n_pred_zero / len(y_valid_raw)

    # ── 无约束极值点 ──
    i_poisson_best = int(np.argmin(grid_poisson))
    i_f1_best = int(np.argmax(grid_f1))

    best_tau_poisson = float(grid_tau[i_poisson_best])
    best_tau_f1 = float(grid_tau[i_f1_best])
    best_poisson = float(grid_poisson[i_poisson_best])
    best_f1 = float(grid_f1[i_f1_best])

    # ── Poisson-约束的 F1 优化 ──
    POISSON_BUDGET_PCT = 0.10
    poisson_ceiling = soft_poisson * (1.0 + POISSON_BUDGET_PCT)

    feasible = grid_poisson <= poisson_ceiling
    if feasible.any():
        feasible_f1 = np.where(feasible, grid_f1, -1.0)
        i_constrained = int(np.argmax(feasible_f1))
        use_threshold = float(grid_tau[i_constrained])
        threshold_strategy = f'constrained (Poisson↑≤{POISSON_BUDGET_PCT:.0%})'
    else:
        poisson_increase = grid_poisson - soft_poisson
        i_constrained = int(np.argmin(poisson_increase))
        use_threshold = float(grid_tau[i_constrained])
        threshold_strategy = 'minimal-degradation fallback'

    # 应用最终阈值
    final_pred = np.where(prob_valid < use_threshold, 0.0, soft_pred)
    final_pred_safe = np.clip(final_pred, eps, None)
    final_poisson = mean_poisson_deviance(y_valid_raw, final_pred_safe)

    # 最终阈值下的零值指标
    final_pred_zero_mask = prob_valid < use_threshold
    n_final_pred_zero = final_pred_zero_mask.sum()
    n_final_correct_zero = ((y_valid_raw <= eps) & final_pred_zero_mask).sum()
    final_zero_recall = n_final_correct_zero / n_true_zero if n_true_zero > 0 else 0.0
    final_zero_precision = n_final_correct_zero / n_final_pred_zero if n_final_pred_zero > 0 else 0.0
    final_zero_f1 = (2 * final_zero_recall * final_zero_precision /
                     (final_zero_recall + final_zero_precision)
                     if (final_zero_recall + final_zero_precision) > 0 else 0.0)

    print(f"🎉 [{target_name}] Hurdle model training completed!")
    print(f"📌 Classifier Logloss (valid):        {classifier_logloss:.4f}")
    print(f"📌 Regressor Poisson on positives:    {regressor_poisson_pos:.4f}")
    print(f"📌 Soft-product Poisson (τ=0):        {soft_poisson:.4f}")
    print(f"─── Zero-Threshold Optimization [{threshold_strategy}] ───")
    print(f"🎯 Poisson-optimal τ*:                {best_tau_poisson:.2f}  "
          f"(Poisson={best_poisson:.4f})")
    print(f"🎯 F1-optimal τ*:                     {best_tau_f1:.2f}  "
          f"(F1={best_f1:.4f}, Poisson={grid_poisson[i_f1_best]:.4f})")
    print(f"🎯 Constrained-optimal τ*:            {use_threshold:.2f}  "
          f"(F1={grid_f1[i_constrained]:.4f}, Poisson={grid_poisson[i_constrained]:.4f})")
    print(f"📊 True zero ratio:                   {true_zero_ratio:.2%}")
    print(f"📊 Pred zero ratio (Poisson-τ):       {grid_pred_zero_ratio[i_poisson_best]:.2%}")
    print(f"📊 Pred zero ratio (F1-τ):            {grid_pred_zero_ratio[i_f1_best]:.2%}")
    print(f"📊 Pred zero ratio (constrained-τ):   {grid_pred_zero_ratio[i_constrained]:.2%}")
    print(f"📊 Zero F1 (Poisson-τ):               {grid_f1[i_poisson_best]:.4f}")
    print(f"📊 Zero F1 (F1-τ):                    {best_f1:.4f}")
    print(f"📊 Zero F1 (constrained-τ):           {grid_f1[i_constrained]:.4f}")
    # ----------------------------
    summary = {
        'target_name': target_name,
        'best_iteration': regressor.get_best_iteration(),
        'best_test': regressor.get_best_score().get('validation', {}).get('Poisson'),
        'final_metric': final_poisson,
        'soft_poisson': soft_poisson,
        'classifier_logloss': classifier_logloss,
        'regressor_poisson_pos': regressor_poisson_pos,
        'hurdle_threshold': use_threshold,
        'hurdle_threshold_poisson': best_tau_poisson,
        'hurdle_threshold_f1': best_tau_f1,
        'hurdle_threshold_strategy': threshold_strategy,
        'valid_zero_recall': final_zero_recall,
        'valid_zero_precision': final_zero_precision,
        'valid_zero_f1': final_zero_f1,
        'valid_pred_zero_ratio': n_final_pred_zero / len(y_valid_raw),
        'valid_true_zero_ratio': true_zero_ratio,
        'train_size': len(X_train),
        'valid_size': len(X_valid),
        'train_time_range': train_time_range,
        'valid_time_range': valid_time_range,
    }

    # Pack and return models (with the selected threshold)
    return {
        'classifier': classifier,
        'regressor': regressor,
        'threshold': use_threshold,
        'threshold_poisson': best_tau_poisson,
        'threshold_f1': best_tau_f1,
    }, summary


def predict_on_test_data(models_dict, feature_cols, test_file, output_file):
    """Run inference on test set with zero-inflated Hurdle models and save in Grid_Utility format.

    Output columns (matching 3_Optimization/Grid_Utility_Test.csv):
        h3, latitude, longitude, datetime, month, day_of_week, is_weekend, hour,
        rent_pred, return_pred,
        low_power_bike_count, soon_low_power_bike_count, normal_power_bike_count,
        rent, return
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading test set for zero-inflated prediction: {test_file}")
    test_df = pd.read_csv(test_file)
    print(f"Test set loaded. Shape: {test_df.shape}")

    # --- Preserve raw columns needed for Grid_Utility output BEFORE any dropping ---
    OUTPUT_META_COLS = ['h3', 'latitude', 'longitude', 'datetime',
                        'month', 'day_of_week', 'is_weekend', 'hour',
                        'low_power_bike_count', 'soon_low_power_bike_count', 'normal_power_bike_count']
    OUTPUT_GT_COLS = ['rent', 'return']

    preserved_meta = {}
    for col in OUTPUT_META_COLS:
        preserved_meta[col] = test_df[col].values if col in test_df.columns else None

    preserved_gt = {}
    for col in OUTPUT_GT_COLS:
        preserved_gt[col] = test_df[col].values if col in test_df.columns else None

    # --- Standard preprocessing ---
    cols_to_drop = ['region_code', 'Unnamed: 21']
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')

    if 'h3' in test_df.columns:
        test_df['h3'] = test_df['h3'].astype(str)
    test_df = fill_missing_values(test_df)
    test_df = add_feature_engineering(test_df)
    validate_required_columns(test_df, feature_cols, 'test set')

    X_test = test_df[feature_cols]

    # === Rent Prediction (Hurdle: 若 P(rent>0) < τ 则硬判为零) ===
    rent_prob = models_dict['rent']['classifier'].predict_proba(X_test)[:, 1]
    rent_val = models_dict['rent']['regressor'].predict(X_test)
    rent_soft = np.clip(rent_prob * rent_val, 0, None)
    rent_threshold = models_dict['rent'].get('threshold', 0.0)
    rent_pred = np.where(rent_prob < rent_threshold, 0.0, rent_soft)
    print(f"  rent 阈值 τ={rent_threshold:.2f}, "
          f"预测零值占比={float((rent_pred <= 1e-6).mean()):.2%}")

    # === Return Prediction (Hurdle: 若 P(return>0) < τ 则硬判为零) ===
    return_prob = models_dict['return']['classifier'].predict_proba(X_test)[:, 1]
    return_val = models_dict['return']['regressor'].predict(X_test)
    return_soft = np.clip(return_prob * return_val, 0, None)
    return_threshold = models_dict['return'].get('threshold', 0.0)
    return_pred = np.where(return_prob < return_threshold, 0.0, return_soft)
    print(f"  return 阈值 τ={return_threshold:.2f}, "
          f"预测零值占比={float((return_pred <= 1e-6).mean()):.2%}")

    # --- Assemble output in Grid_Utility format ---
    result_df = pd.DataFrame()

    # 1. Identifier + spatial columns
    for col in ['h3', 'latitude', 'longitude']:
        result_df[col] = preserved_meta[col] if preserved_meta.get(col) is not None else 0

    # 2. Temporal columns
    for col in ['datetime', 'month', 'day_of_week', 'is_weekend', 'hour']:
        result_df[col] = preserved_meta[col] if preserved_meta.get(col) is not None else 0

    # 3. Predictions
    result_df['rent_pred'] = rent_pred
    result_df['return_pred'] = return_pred

    # 4. Bike status columns
    for col in ['low_power_bike_count', 'soon_low_power_bike_count', 'normal_power_bike_count']:
        result_df[col] = preserved_meta[col] if preserved_meta.get(col) is not None else 0

    # 5. Ground truth (rent, return) — 0 if test set doesn't have them
    for col in ['rent', 'return']:
        result_df[col] = preserved_gt[col] if preserved_gt.get(col) is not None else 0

    result_df.to_csv(output_file, index=False)
    print(f"Test prediction complete. Saved to: {output_file}")
    print(f"Output columns: {list(result_df.columns)}")

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
            input("\nAbout to start full-data CatBoost Hurdle training. Press Enter to continue...")
            
        df, feature_cols = load_and_preprocess(file_name, scale=scale)
        scale_tag = str(scale) if scale is not None else 'all'
        
        models_dict = {}

        # Train zero-inflated models for rent and return sequentially
        models_dict['rent'], rent_summary = train_model(
            df, feature_cols, target_name='rent', 
            scale_tag=scale_tag, run_timestamp=run_timestamp, run_output_dir=run_output_dir
        )
        
        models_dict['return'], return_summary = train_model(
            df, feature_cols, target_name='return', 
            scale_tag=scale_tag, run_timestamp=run_timestamp, run_output_dir=run_output_dir
        )

        output_file = os.path.join(run_output_dir, PREDICTION_OUTPUT_TEMPLATE.format(scale=scale_tag, ts=run_timestamp))
        predict_on_test_data(models_dict, feature_cols, TEST_FILE, output_file)

        # 汇总CSV写入时间戳文件夹内
        summary_csv = TRAINING_SUMMARY_CSV if TRAINING_SUMMARY_CSV else os.path.join(run_output_dir, 'training_summary.csv')
        run_summary_row = build_run_summary_row(
            run_timestamp=run_timestamp,
            scale_tag=scale_tag,
            split_mode=SPLIT_MODE,
            train_size=rent_summary['train_size'],
            valid_size=rent_summary['valid_size'],
            train_time_range=rent_summary['train_time_range'],
            valid_time_range=rent_summary['valid_time_range'],
            rent_summary=rent_summary,
            return_summary=return_summary,
            model_type='CB_Hurdle',
            shared_config={
                'cb_loss_function': CB_REGRESSOR_PARAMS['loss_function'],
                'cb_eval_metric': CB_REGRESSOR_PARAMS['eval_metric'],
                'cb_learning_rate': CB_REGRESSOR_PARAMS['learning_rate'],
                'cb_depth': CB_REGRESSOR_PARAMS['depth'],
                'cb_l2_leaf_reg': CB_REGRESSOR_PARAMS['l2_leaf_reg'],
                'cb_od_wait': CB_REGRESSOR_PARAMS['od_wait'],
                'cb_iterations': CB_ITERATIONS,
            },
        )
        append_summary_row(summary_csv, run_summary_row)
        print(f"Run summary appended to: {summary_csv}")

        print("\n" + "="*50)
        print(f"Hurdle model pipeline completed for scale {scale_tag}.")
        print("="*50 + "\n")