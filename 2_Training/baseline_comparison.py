"""
基准模型对比脚本：在完全相同的训练/验证/测试划分下，训练并评估四种架构，
输出对比表、合并预测文件、各模型独立预测集以及零膨胀专项分析。

四种架构（全部由本脚本训练，使用优化后的代表性默认配置）：
  (1) CatBoost      — 单阶段Poisson回归（lr=0.02, depth=10, od_wait=80）
  (2) CatBoost Hurdle — 两阶段Hurdle模型（lr=0.02, depth=10, od_wait=80）
  (3) LightGBM       — 单阶段Poisson回归（lr=0.03, num_leaves=63）
  (4) Random Forest  — 单阶段回归（n=1500, depth=22, min_leaf=3）

实验设计原则：
  - 四种架构使用完全相同的训练/验证时间切分（前80%训练，后20%验证）和特征集
  - 全部模型从零训练，不使用任何预训练结果

输出文件：
  (A) baseline_comparison.csv — 四种架构的参数与性能对比（每行一个模型×目标）
  (B) baseline_test_predictions.csv — 四种架构在测试集上的预测值汇总（含真实值）
  (C) baseline_zero_inflation_metrics.csv — 零膨胀专项指标（Zero F1, 条件RMSE/Poisson等）
  (D) prediction_CatBoost.csv — CatBoost 独立预测集（Grid_Utility 格式）
  (E) prediction_CB_Hurdle.csv — CatBoost Hurdle 独立预测集（Grid_Utility 格式）
  (F) prediction_LightGBM.csv — LightGBM 独立预测集（Grid_Utility 格式）
  (G) prediction_RandomForest.csv — Random Forest 独立预测集（Grid_Utility 格式）

用法：
    python 2_Training/baseline_comparison.py \
      --train_csv battery_swapping_routing_data_train_time70.csv \
      --test_csv battery_swapping_routing_data_valid_time30.csv \
      --output_dir 2_Training/Baseline_Comparison_Results
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error, mean_absolute_error, log_loss

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ── 特征工程（与 CB_Hurdle_train.py 完全一致）─────────────────────────────

FEATURES = [
    'h3', 'temperature', 'wind_level', 'rain_level',
    'month', 'day_of_week', 'is_weekend', 'hour',
    'rent_mean_7d', 'return_mean_7d', 'lag_nb_rent', 'lag_nb_return',
    'normal_power_bike_count', 'soon_low_power_bike_count', 'low_power_bike_count',
    'latitude', 'longitude',
    'temp_x_rain', 'available_power_bike_gap', 'is_rush_hour',
    'lag_rent_is_zero', 'lag_return_is_zero',
    'rent_mean_7d_low', 'return_mean_7d_low',
]

TARGETS = ['rent', 'return']

# ── Grid_Utility 输出格式（与 3_Optimization/Grid_Utility_Test.csv 列一致）────────
GRID_UTILITY_META_COLS = [
    'h3', 'latitude', 'longitude', 'datetime',
    'month', 'day_of_week', 'is_weekend', 'hour',
    'low_power_bike_count', 'soon_low_power_bike_count', 'normal_power_bike_count',
]
GRID_UTILITY_GT_COLS = ['rent', 'return']


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """创建所有手工特征，与 CB_Hurdle_train.py 完全一致。"""
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
        df['available_power_bike_gap'] = (
            df['normal_power_bike_count'] - df['low_power_bike_count']
        )
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


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].fillna('missing').astype(str)
        else:
            df[col] = df[col].fillna(0)
    return df


def time_split(df, y_raw):
    """时间切分：前80%训练，后20%验证。与CB_Hurdle_train.py一致。"""
    dt = pd.to_datetime(df['datetime'], errors='coerce')
    ordered_idx = dt.sort_values().index
    split_pos = int(len(ordered_idx) * 0.8)
    train_idx = ordered_idx[:split_pos]
    valid_idx = ordered_idx[split_pos:]

    X = df[FEATURES]
    X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
    y_train, y_valid = y_raw.loc[train_idx], y_raw.loc[valid_idx]

    train_time = pd.to_datetime(df.loc[train_idx, 'datetime'], errors='coerce').dropna()
    valid_time = pd.to_datetime(df.loc[valid_idx, 'datetime'], errors='coerce').dropna()
    train_range = f"{train_time.min()} -> {train_time.max()}"
    valid_range = f"{valid_time.min()} -> {valid_time.max()}"
    print(f"  时间切分: train[{train_range}], valid[{valid_range}], "
          f"train={len(train_idx)}, valid={len(valid_idx)}")

    return X_train, X_valid, y_train, y_valid, train_range, valid_range


def compute_metrics(y_true, y_pred) -> Dict:
    """统一计算所有评估指标。"""
    y_pred = np.clip(y_pred, 0, None)
    y_pred_safe = np.clip(y_pred, 1e-6, None)  # Poisson deviance 要求 y_pred > 0
    return {
        'Poisson': mean_poisson_deviance(y_true, y_pred_safe),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
    }


def compute_zero_inflated_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """零膨胀场景专项指标：将预测误差按真实值是否为零分解。

    返回指标说明：
      - zero_ratio_true / zero_ratio_pred : 真实/预测零值占比
      - zero_recall   : 真实零值中被正确预测为 (<1e-6) 的比例
      - zero_precision: 预测为 (<1e-6) 的样本中真实为零的比例
      - zero_f1       : zero_recall 与 zero_precision 的调和平均
      - rmse_on_zero / rmse_on_positive : 按真实值分解的 RMSE
      - poisson_on_zero / poisson_on_positive : 按真实值分解的 Poisson
      - mae_on_zero / mae_on_positive : 按真实值分解的 MAE
    """
    y_pred = np.clip(y_pred, 0, None)
    eps = 1e-6

    mask_true_zero = y_true <= eps
    mask_true_pos = y_true > eps
    mask_pred_zero = y_pred <= eps

    n_total = len(y_true)
    n_true_zero = mask_true_zero.sum()
    n_pred_zero = mask_pred_zero.sum()
    n_correct_zero = (mask_true_zero & mask_pred_zero).sum()

    # Level 1: 零膨胀特征
    zero_ratio_true = n_true_zero / n_total
    zero_ratio_pred = n_pred_zero / n_total

    # Level 2: 零值判别 (positive class = zero)
    zero_recall = n_correct_zero / n_true_zero if n_true_zero > 0 else np.nan
    zero_precision = n_correct_zero / n_pred_zero if n_pred_zero > 0 else np.nan
    zero_f1 = (2 * zero_recall * zero_precision / (zero_recall + zero_precision)
               if (zero_recall + zero_precision) > 0 else np.nan)

    # Level 3: 条件误差分解
    def _safe_poisson(yt, yp):
        yp = np.clip(yp, 1e-6, None)  # Poisson deviance 要求 y_pred > 0
        return mean_poisson_deviance(yt, yp) if len(yt) > 0 else np.nan

    def _safe_rmse(yt, yp):
        return np.sqrt(mean_squared_error(yt, yp)) if len(yt) > 0 else np.nan

    def _safe_mae(yt, yp):
        return mean_absolute_error(yt, yp) if len(yt) > 0 else np.nan

    return {
        # 数据特征
        'zero_ratio_true': round(zero_ratio_true, 4),
        'zero_ratio_pred': round(zero_ratio_pred, 4),
        'zero_ratio_gap': round(abs(zero_ratio_pred - zero_ratio_true), 4),
        # 零值判别
        'zero_recall': round(zero_recall, 4) if not np.isnan(zero_recall) else np.nan,
        'zero_precision': round(zero_precision, 4) if not np.isnan(zero_precision) else np.nan,
        'zero_f1': round(zero_f1, 4) if not np.isnan(zero_f1) else np.nan,
        # 条件误差 — RMSE
        'rmse_on_zero': round(_safe_rmse(y_true[mask_true_zero], y_pred[mask_true_zero]), 4),
        'rmse_on_positive': round(_safe_rmse(y_true[mask_true_pos], y_pred[mask_true_pos]), 4),
        # 条件误差 — Poisson
        'poisson_on_zero': round(_safe_poisson(y_true[mask_true_zero], y_pred[mask_true_zero]), 4),
        'poisson_on_positive': round(_safe_poisson(y_true[mask_true_pos], y_pred[mask_true_pos]), 4),
        # 条件误差 — MAE
        'mae_on_zero': round(_safe_mae(y_true[mask_true_zero], y_pred[mask_true_zero]), 4),
        'mae_on_positive': round(_safe_mae(y_true[mask_true_pos], y_pred[mask_true_pos]), 4),
    }


def save_model_predictions(
    model_name: str,
    rent_pred: np.ndarray,
    return_pred: np.ndarray,
    df_test_raw: pd.DataFrame,
    output_dir: str,
):
    """将单个模型的预测结果保存为 Grid_Utility 格式的 CSV。

    Parameters
    ----------
    model_name : 模型简称（如 'CatBoost', 'CB_Hurdle', 'LightGBM', 'RandomForest'）
    rent_pred, return_pred : 预测数组
    df_test_raw : 原始测试集 DataFrame（预处理前），用于提取元数据列
    output_dir : 输出目录
    """
    result_df = pd.DataFrame()

    # 1. 标识列 + 空间列
    for col in ['h3', 'latitude', 'longitude']:
        result_df[col] = df_test_raw[col].values if col in df_test_raw.columns else 0

    # 2. 时间列
    for col in ['datetime', 'month', 'day_of_week', 'is_weekend', 'hour']:
        result_df[col] = df_test_raw[col].values if col in df_test_raw.columns else 0

    # 3. 预测值
    result_df['rent_pred'] = np.clip(rent_pred, 0, None)
    result_df['return_pred'] = np.clip(return_pred, 0, None)

    # 4. 车辆状态列
    for col in ['low_power_bike_count', 'soon_low_power_bike_count', 'normal_power_bike_count']:
        result_df[col] = df_test_raw[col].values if col in df_test_raw.columns else 0

    # 5. 真实值（若测试集中存在）
    for col in ['rent', 'return']:
        result_df[col] = df_test_raw[col].values if col in df_test_raw.columns else 0

    safe_name = model_name.replace(' ', '_').replace('/', '_')
    out_path = os.path.join(output_dir, f'prediction_{safe_name}.csv')
    result_df.to_csv(out_path, index=False)
    print(f"  [{model_name}] 预测集已保存至: {out_path}")
    return out_path


# ── LightGBM 训练 ────────────────────────────────────────────────────────

def _prepare_lgb_data(X: pd.DataFrame) -> pd.DataFrame:
    """将 h3 转换为 category dtype，与原始 LightGBM_train.py 一致。"""
    X = X.copy()
    if 'h3' in X.columns:
        X['h3'] = X['h3'].astype('category')
    return X


def train_lightgbm(
    X_train, y_train, X_valid, y_valid
) -> Tuple[object, Dict]:
    """训练LightGBM（代表性默认配置，单次运行）。"""
    import lightgbm as lgb

    # LightGBM 要求 categorical 列为 category dtype，不能是 object/str
    X_train = _prepare_lgb_data(X_train)
    X_valid = _prepare_lgb_data(X_valid)

    params = {
        'objective': 'poisson',          # 与 CB/CB_Hurdle 一致的计数损失
        'metric': 'poisson',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,           # 降低学习率 → 更多迭代空间
        'num_leaves': 63,                # 与独立 LightGBM_train.py 一致
        'min_child_samples': 10,         # 更细粒度分裂 → 更好捕捉零值模式
        'feature_fraction': 0.8,
        'subsample': 0.8,                # 行采样 → 防过拟合
        'subsample_freq': 1,
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42,
    }

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=['h3'])
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    t_start = time.time()
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_valid],
        valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=150),  # 更耐心的早停
            lgb.log_evaluation(period=100),
        ],
    )
    elapsed = time.time() - t_start

    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    metrics = compute_metrics(y_valid, y_pred)

    info = {
        'learning_rate': params['learning_rate'],
        'num_leaves': params['num_leaves'],
        'min_child_samples': params['min_child_samples'],
        'subsample': params['subsample'],
        'best_iteration': model.best_iteration,
        'training_seconds': round(elapsed, 1),
        **{f'valid_{k.lower()}': v for k, v in metrics.items()},
    }
    return model, info


# ── Random Forest 训练 ────────────────────────────────────────────────────

def train_random_forest(
    X_train, y_train, X_valid, y_valid, h3_mapping: dict = None
) -> Tuple[object, Dict, dict]:
    """训练Random Forest（代表性默认配置，单次运行）。

    返回 (model, info, h3_mapping)，其中 h3_mapping 用于测试集编码。
    """
    # 构建 h3→int 映射（若未传入则从训练集构建）
    if h3_mapping is None:
        h3_mapping = {val: idx for idx, val in enumerate(X_train['h3'].unique())}

    def encode(df):
        df = df.copy()
        df['h3'] = df['h3'].map(h3_mapping).fillna(-1).astype(np.int32)
        return df

    Xt, Xv = encode(X_train), encode(X_valid)

    t_start = time.time()
    model = RandomForestRegressor(
        n_estimators=1500,              # 更多树 → 更稳定
        max_depth=22,                   # 更深 → 更好捕捉零/非零交互模式
        min_samples_leaf=3,             # 更细粒度 → 低需求区域更精确
        max_features='sqrt',
        n_jobs=-1,
        verbose=0,
        random_state=42,
    )
    model.fit(Xt, y_train)
    elapsed = time.time() - t_start

    y_pred = model.predict(Xv)
    metrics = compute_metrics(y_valid, y_pred)

    info = {
        'n_estimators': 1500,
        'max_depth': 22,
        'min_samples_leaf': 3,
        'best_iteration': np.nan,
        'training_seconds': round(elapsed, 1),
        **{f'valid_{k.lower()}': v for k, v in metrics.items()},
    }
    return model, info, h3_mapping


# ── CatBoost 训练（单阶段Poisson回归）────────────────────────────────────

def train_catboost(
    X_train, y_train, X_valid, y_valid,
    cat_features: Optional[List[str]] = None,
) -> Tuple[object, Dict]:
    """训练 CatBoost（代表性默认配置，单次运行）。

    返回 (model, info)。
    """
    import catboost as cb

    if cat_features is None:
        cat_features = ['h3']
    cat_indices = [list(X_train.columns).index(f) for f in cat_features if f in X_train.columns]

    t_start = time.time()
    model = cb.CatBoostRegressor(
        loss_function='Poisson',
        eval_metric='Poisson',
        learning_rate=0.02,           # 更低学习率 → 更平滑收敛
        depth=10,                      # 与 CB_Hurdle 对齐
        l2_leaf_reg=4.0,
        iterations=5000,               # 更多迭代空间
        cat_features=cat_indices,
        od_type='Iter',
        od_wait=80,                    # 更耐心 → 避免过早停止
        random_seed=42,
        task_type='CPU',
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    elapsed = time.time() - t_start
    y_pred = np.clip(model.predict(X_valid), 0, None)
    metrics = compute_metrics(y_valid, y_pred)

    info = {
        'learning_rate': 0.02,
        'depth': 10,
        'l2_leaf_reg': 4.0,
        'od_wait': 80,
        'best_iteration': model.get_best_iteration(),
        'training_seconds': round(elapsed, 1),
        **{f'valid_{k.lower()}': v for k, v in metrics.items()},
    }
    return model, info


# ── CatBoost Hurdle 训练（两阶段零膨胀模型）───────────────────────────────

def train_cb_hurdle(
    X_train, y_train, X_valid, y_valid,
    cat_features: Optional[List[str]] = None,
) -> Tuple[Dict[str, object], Dict]:
    """训练 CatBoost Hurdle 模型（分类器 + 正样本回归器）。

    返回 (model_dict, info)，其中 model_dict = {'classifier': ..., 'regressor': ...}。
    """
    import catboost as cb

    if cat_features is None:
        cat_features = ['h3']
    cat_indices = [list(X_train.columns).index(f) for f in cat_features if f in X_train.columns]

    # Stage 1: 分类器（预测 demand > 0）
    y_train_bin = (y_train > 0).astype(int)
    y_valid_bin = (y_valid > 0).astype(int)

    classifier = cb.CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='Logloss',
        learning_rate=0.02,           # 更低学习率
        depth=10,
        l2_leaf_reg=4.0,
        iterations=5000,               # 更多迭代空间
        cat_features=cat_indices,
        od_type='Iter',
        od_wait=80,                    # 更耐心
        random_seed=42,
        task_type='CPU',
        thread_count=-1,
        allow_writing_files=False,
        verbose=0,
    )
    t_start = time.time()
    classifier.fit(X_train, y_train_bin, eval_set=(X_valid, y_valid_bin),
                   use_best_model=True, verbose=100)

    # Stage 2: 回归器（仅正样本）
    mask_train_pos = y_train > 0
    X_train_pos = X_train[mask_train_pos]
    y_train_pos = y_train[mask_train_pos]

    mask_valid_pos = y_valid > 0
    X_valid_pos = X_valid[mask_valid_pos]
    y_valid_pos = y_valid[mask_valid_pos]

    regressor = cb.CatBoostRegressor(
        loss_function='Poisson',
        eval_metric='Poisson',
        learning_rate=0.02,           # 更低学习率
        depth=10,
        l2_leaf_reg=4.0,
        iterations=5000,               # 更多迭代空间
        cat_features=cat_indices,
        od_type='Iter',
        od_wait=80,                    # 更耐心
        random_seed=42,
        task_type='CPU',
        thread_count=-1,
        allow_writing_files=False,
        verbose=0,
    )
    regressor.fit(X_train_pos, y_train_pos, eval_set=(X_valid_pos, y_valid_pos),
                  use_best_model=True, verbose=100)

    elapsed = time.time() - t_start

    # --- Stage 3: Soft-product prediction (no threshold) ---
    prob_valid = classifier.predict_proba(X_valid)[:, 1]
    val_valid = np.clip(regressor.predict(X_valid), 0, None)
    soft_pred = np.clip(prob_valid * val_valid, 0, None)
    soft_metrics = compute_metrics(y_valid, soft_pred)

    # --- Stage 4: Optimize zero-decision threshold τ (constrained) ---
    # 在验证集上搜索最优阈值 τ，使用约束优化策略。
    #
    # 关键设计（v2 — 修复 Poisson 暴涨问题）：
    #   纯 Poisson 准则下 τ* 几乎总是 0（假阴性惩罚极端不对称）。
    #   纯 F1 准则下 τ* 过于激进（~0.59），导致 Poisson 暴涨 3-4×。
    #
    #   新策略：Poisson-约束的 F1 最大化 ——
    #   在 Poisson deviance 不超过 soft-product 的 P% 范围内，选择 F1 最高的 τ。
    #   每个 τ 对应的 soft_pred 只在 prob < τ 的样本上被截断为零，
    #   未被截断的样本仍使用 soft_pred，因此 Poisson 退化完全来自新增的假阴性。
    eps = 1e-6
    n_true_zero = (y_valid <= eps).sum()
    true_zero_ratio = n_true_zero / len(y_valid)

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
        grid_poisson[i] = mean_poisson_deviance(y_valid, thresh_pred_safe)

        n_pred_zero = pred_zero_mask.sum()
        n_correct_zero = ((y_valid <= eps) & pred_zero_mask).sum()
        rec = n_correct_zero / n_true_zero if n_true_zero > 0 else 0.0
        prec = n_correct_zero / n_pred_zero if n_pred_zero > 0 else 0.0
        f1 = (2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0.0)
        grid_recall[i] = rec
        grid_precision[i] = prec
        grid_f1[i] = f1
        grid_pred_zero_ratio[i] = n_pred_zero / len(y_valid)

    # ── 无约束极值点 ──
    i_poisson_best = int(np.argmin(grid_poisson))
    i_f1_best = int(np.argmax(grid_f1))

    best_tau_poisson = float(grid_tau[i_poisson_best])
    best_tau_f1 = float(grid_tau[i_f1_best])
    best_poisson = float(grid_poisson[i_poisson_best])
    best_f1 = float(grid_f1[i_f1_best])

    # ── Poisson-约束的 F1 优化 ──
    # 允许 Poisson 最多退化 POISSON_BUDGET_PCT% (默认 10%)
    POISSON_BUDGET_PCT = 0.10
    soft_poisson = soft_metrics['Poisson']  # τ=0 时的基准 Poisson
    poisson_ceiling = soft_poisson * (1.0 + POISSON_BUDGET_PCT)

    feasible = grid_poisson <= poisson_ceiling
    if feasible.any():
        # 约束可行：在 Poisson 退化 ≤10% 的 τ 中选 F1 最高的
        feasible_f1 = np.where(feasible, grid_f1, -1.0)
        i_constrained = int(np.argmax(feasible_f1))
        use_threshold = float(grid_tau[i_constrained])
        threshold_strategy = f'constrained (Poisson↑≤{POISSON_BUDGET_PCT:.0%})'
    else:
        # 所有 τ 都使 Poisson 退化 >10%：选择 Poisson 退化最小的 τ
        # （退化最小的通常是 τ 最小者，即尽量少截断）
        poisson_increase = grid_poisson - soft_poisson
        i_constrained = int(np.argmin(poisson_increase))
        use_threshold = float(grid_tau[i_constrained])
        threshold_strategy = 'minimal-degradation fallback'

    final_pred = np.where(prob_valid < use_threshold, 0.0, soft_pred)
    final_metrics = compute_metrics(y_valid, final_pred)

    info = {
        'learning_rate': 0.02,
        'depth': 10,
        'l2_leaf_reg': 4.0,
        'od_wait': 80,
        'best_iteration': regressor.get_best_iteration(),
        'training_seconds': round(elapsed, 1),
        # 原 soft-product 指标（τ=0，Hurdle 的"纯 Poisson"性能）
        **{f'valid_{k.lower()}_soft': v for k, v in soft_metrics.items()},
        # 最终指标（应用约束优化阈值后）
        **{f'valid_{k.lower()}': v for k, v in final_metrics.items()},
        # 阈值优化结果
        'hurdle_threshold_poisson': best_tau_poisson,
        'hurdle_threshold_f1': best_tau_f1,
        'hurdle_threshold_used': use_threshold,
        'hurdle_threshold_strategy': threshold_strategy,
        'valid_true_zero_ratio': round(true_zero_ratio, 4),
        'valid_pred_zero_ratio_poisson': round(grid_pred_zero_ratio[i_poisson_best], 4),
        'valid_pred_zero_ratio_f1': round(grid_pred_zero_ratio[i_f1_best], 4),
        'valid_pred_zero_ratio_used': round(float(grid_pred_zero_ratio[i_constrained]), 4),
        'valid_zero_f1_poisson': round(float(grid_f1[i_poisson_best]), 4),
        'valid_zero_f1_f1': round(best_f1, 4),
        'valid_zero_f1_used': round(float(grid_f1[i_constrained]), 4),
        'valid_classifier_logloss': round(log_loss(y_valid_bin, prob_valid), 4),
    }

    return {
        'classifier': classifier,
        'regressor': regressor,
        'threshold': use_threshold,
        'threshold_poisson': best_tau_poisson,
        'threshold_f1': best_tau_f1,
    }, info


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='四种架构基准对比：全部模型从零训练，统一数据划分')
    parser.add_argument('--train_csv', required=True, help='训练集CSV')
    parser.add_argument('--test_csv', required=True, help='测试集CSV')
    parser.add_argument('--output_dir', default=None,
                        help='输出目录（默认自动生成 Training_Results/{YYYYMMDD_HHMMSS}）')
    args = parser.parse_args()

    if args.output_dir is None:
        run_timestamp = time.strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join(SCRIPT_DIR, 'Training_Results', run_timestamp)
        print(f"自动生成输出目录: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. 加载并预处理数据 ──
    print("=" * 60)
    print("[1/7] 加载数据...")
    df_train = pd.read_csv(args.train_csv)
    df_test_raw = pd.read_csv(args.test_csv)  # 保留原始副本，用于 Grid_Utility 格式输出
    df_test = df_test_raw.copy()

    def preprocess(df: pd.DataFrame, name: str) -> pd.DataFrame:
        cols_to_drop = ['region_code', 'Unnamed: 21']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore', inplace=True)
        if 'h3' in df.columns:
            df['h3'] = df['h3'].astype(str)
        fill_missing_values(df)
        df = add_feature_engineering(df)  # add_feature_engineering 内部有 .copy()，必须接收返回值
        print(f"  {name}集: {df.shape}")
        return df

    df_train = preprocess(df_train, '训练')
    df_test = preprocess(df_test, '测试')

    comparison_rows: List[Dict] = []
    model_predictions: Dict[str, Dict[str, np.ndarray]] = {}  # {model_name: {target: y_pred_array}}
    model_objects: Dict[str, Dict[str, object]] = {}  # {model_name: {target: trained_model}}
    rf_h3_mapping: Dict[str, dict] = {}  # {target: h3_mapping} — 训练集编码，测试时复用

    # ── 2. 训练CatBoost（单阶段Poisson回归）──
    print("\n[2/6] 训练 CatBoost...")
    for target in TARGETS:
        print(f"\n  CatBoost — 目标: {target}")
        y_raw = df_train[target].astype(float)
        X_train, X_valid, y_train, y_valid, train_range, valid_range = time_split(df_train, y_raw)

        model, info = train_catboost(X_train, y_train, X_valid, y_valid)
        model_objects.setdefault('CatBoost', {})[target] = model

        print(f"    验证集: Poisson={info['valid_poisson']:.4f}, "
              f"RMSE={info['valid_rmse']:.4f}, best_iter={info['best_iteration']}")

        comparison_rows.append({
            'model': 'CatBoost',
            'model_type': 'CB',
            'target': target,
            **{k: v for k, v in info.items() if k not in
               ['valid_poisson', 'valid_rmse', 'valid_mae']},
            'valid_poisson': info['valid_poisson'],
            'valid_rmse': info['valid_rmse'],
            'valid_mae': info['valid_mae'],
            'notes': f'train={len(X_train)}, valid={len(X_valid)}',
        })

    # ── 3. 训练CatBoost Hurdle（两阶段零膨胀模型）──
    print("\n[3/6] 训练 CatBoost Hurdle...")
    for target in TARGETS:
        print(f"\n  CatBoost Hurdle — 目标: {target}")
        y_raw = df_train[target].astype(float)
        X_train, X_valid, y_train, y_valid, train_range, valid_range = time_split(df_train, y_raw)

        model_dict, info = train_cb_hurdle(X_train, y_train, X_valid, y_valid)
        model_objects.setdefault('CB_Hurdle', {})[target] = model_dict

        print(f"    验证集: Poisson(soft)={info['valid_poisson_soft']:.4f}, "
              f"Poisson(hard)={info['valid_poisson']:.4f}, "
              f"MAE={info['valid_mae']:.4f}, best_iter={info['best_iteration']}, "
              f"τ_used={info['hurdle_threshold_used']:.2f}")
        print(f"    阈值优化 [{info.get('hurdle_threshold_strategy', 'N/A')}]: "
              f"τ_Poisson={info['hurdle_threshold_poisson']:.2f}, "
              f"τ_F1={info['hurdle_threshold_f1']:.2f}, "
              f"F1_poi={info['valid_zero_f1_poisson']:.4f}, F1_f1={info['valid_zero_f1_f1']:.4f}, "
              f"F1_used={info.get('valid_zero_f1_used', 'N/A')}")
        print(f"    Classifier Logloss={info['valid_classifier_logloss']:.4f}, "
              f"真零比={info['valid_true_zero_ratio']:.2%}, "
              f"预测零比(F1)={info['valid_pred_zero_ratio_f1']:.2%}")

        comparison_rows.append({
            'model': 'CB Hurdle',
            'model_type': 'CB_Hurdle',
            'target': target,
            **{k: v for k, v in info.items() if k not in
               ['valid_poisson', 'valid_rmse', 'valid_mae']},
            'valid_poisson': info['valid_poisson'],
            'valid_rmse': info['valid_rmse'],
            'valid_mae': info['valid_mae'],
            'notes': f'train={len(X_train)}, valid={len(X_valid)}',
        })

    # ── 4. 训练LightGBM ──
    print("\n[4/6] 训练 LightGBM...")
    for target in TARGETS:
        print(f"\n  LightGBM — 目标: {target}")
        y_raw = df_train[target].astype(float)
        X_train, X_valid, y_train, y_valid, train_range, valid_range = time_split(df_train, y_raw)

        model, info = train_lightgbm(X_train, y_train, X_valid, y_valid)
        model_objects.setdefault('LightGBM', {})[target] = model

        comparison_rows.append({
            'model': 'LightGBM',
            'model_type': 'LGB',
            'target': target,
            **{k: v for k, v in info.items() if k not in
               ['valid_poisson', 'valid_rmse', 'valid_mae']},
            'valid_poisson': info['valid_poisson'],
            'valid_rmse': info['valid_rmse'],
            'valid_mae': info['valid_mae'],
            'notes': f'train={len(X_train)}, valid={len(X_valid)}',
        })
        print(f"    验证集: Poisson={info['valid_poisson']:.4f}, "
              f"RMSE={info['valid_rmse']:.4f}, best_iter={info['best_iteration']}")

    # ── 6. 训练Random Forest ──
    print("\n[5/6] 训练 Random Forest...")
    for target in TARGETS:
        print(f"\n  Random Forest — 目标: {target}")
        y_raw = df_train[target].astype(float)
        X_train, X_valid, y_train, y_valid, train_range, valid_range = time_split(df_train, y_raw)

        model, info, h3_map = train_random_forest(X_train, y_train, X_valid, y_valid)
        model_objects.setdefault('RandomForest', {})[target] = model
        rf_h3_mapping[target] = h3_map

        comparison_rows.append({
            'model': 'Random Forest',
            'model_type': 'RF',
            'target': target,
            **{k: v for k, v in info.items() if k not in
               ['valid_poisson', 'valid_rmse', 'valid_mae']},
            'valid_poisson': info['valid_poisson'],
            'valid_rmse': info['valid_rmse'],
            'valid_mae': info['valid_mae'],
            'notes': f'train={len(X_train)}, valid={len(X_valid)}',
        })
        print(f"    验证集: Poisson={info['valid_poisson']:.4f}, "
              f"RMSE={info['valid_rmse']:.4f}, 耗时={info['training_seconds']}s")

    # ── 7. 测试集预测与汇总 ──
    print("\n[6/6] 测试集预测与汇总...")

    # 构建测试集预测DataFrame（合并文件）
    pred_df = df_test[['h3', 'datetime']].copy() if 'datetime' in df_test.columns else df_test[['h3']].copy()

    # 存储每个模型的预测值，用于后续分别保存
    all_model_preds: Dict[str, Dict[str, np.ndarray]] = {}  # {model_name: {'rent': arr, 'return': arr}}

    # 零膨胀指标收集
    zi_metrics_rows: List[Dict] = []

    for target in TARGETS:
        y_test = df_test[target].astype(float)
        pred_df[f'{target}_true'] = y_test.values

        # CatBoost（单阶段Poisson）
        if 'CatBoost' in model_objects:
            cb_pred = np.clip(model_objects['CatBoost'][target].predict(df_test[FEATURES]), 0, None)
            pred_df[f'{target}_cb'] = cb_pred
            all_model_preds.setdefault('CatBoost', {})[target] = cb_pred

        # CatBoost Hurdle（P(demand>0) * E[demand | demand>0]，应用 τ 阈值）
        if 'CB_Hurdle' in model_objects:
            cb_hurdle = model_objects['CB_Hurdle'][target]
            prob_test = cb_hurdle['classifier'].predict_proba(df_test[FEATURES])[:, 1]
            val_test = np.clip(cb_hurdle['regressor'].predict(df_test[FEATURES]), 0, None)
            soft_pred = np.clip(prob_test * val_test, 0, None)
            # 应用 Stage 4 优化的阈值：P(demand>0) < τ → 强判为零
            tau = cb_hurdle.get('threshold', 0.0)
            hurdle_pred = np.where(prob_test < tau, 0.0, soft_pred)
            pred_df[f'{target}_cb_hurdle'] = hurdle_pred
            all_model_preds.setdefault('CB_Hurdle', {})[target] = hurdle_pred

        # LightGBM (需要 h3 为 category dtype)
        if 'LightGBM' in model_objects:
            X_test_lgb = _prepare_lgb_data(df_test[FEATURES])
            lgb_pred = np.clip(
                model_objects['LightGBM'][target].predict(
                    X_test_lgb,
                    num_iteration=model_objects['LightGBM'][target].best_iteration
                ), 0, None
            )
            pred_df[f'{target}_lgb'] = lgb_pred
            all_model_preds.setdefault('LightGBM', {})[target] = lgb_pred

        # Random Forest (复用训练时的 h3→int 映射，未知值填 -1)
        if 'RandomForest' in model_objects:
            X_test_rf = df_test[FEATURES].copy()
            h3_map_rf = rf_h3_mapping.get(target, {})
            X_test_rf['h3'] = X_test_rf['h3'].map(h3_map_rf).fillna(-1).astype(np.int32)
            rf_pred = np.clip(
                model_objects['RandomForest'][target].predict(X_test_rf), 0, None
            )
            pred_df[f'{target}_rf'] = rf_pred
            all_model_preds.setdefault('RandomForest', {})[target] = rf_pred

        # ── 全局指标 + 零膨胀指标 ──
        print(f"\n{'='*60}")
        print(f"  测试集指标 — {target}")
        print(f"  真实零值占比: {(y_test <= 1e-6).mean():.2%}")
        print(f"{'='*60}")

        model_col_map = {
            'CatBoost': f'{target}_cb',
            'CB Hurdle': f'{target}_cb_hurdle',
            'LightGBM': f'{target}_lgb',
            'Random Forest': f'{target}_rf',
        }
        for model_label, col in model_col_map.items():
            if col in pred_df.columns:
                y_pred = pred_df[col].values
                m = compute_metrics(y_test, y_pred)
                zi = compute_zero_inflated_metrics(y_test, y_pred)

                # 全局指标
                print(f"\n  ┌─ {model_label}")
                print(f"  │  Poisson={m['Poisson']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")
                # 零膨胀指标
                print(f"  │  --- 零膨胀分解 ---")
                print(f"  │  预测零值比: {zi['zero_ratio_pred']:.2%}  (真实: {zi['zero_ratio_true']:.2%})  gap: {zi['zero_ratio_gap']:.4f}")
                print(f"  │  Zero Recall={zi['zero_recall']:.4f}  Precision={zi['zero_precision']:.4f}  F1={zi['zero_f1']:.4f}")
                print(f"  │  RMSE_on_zero={zi['rmse_on_zero']:.4f}  RMSE_on_pos={zi['rmse_on_positive']:.4f}")
                print(f"  │  Poisson_on_zero={zi['poisson_on_zero']:.4f}  Poisson_on_pos={zi['poisson_on_positive']:.4f}")
                print(f"  └─ MAE_on_zero={zi['mae_on_zero']:.4f}  MAE_on_pos={zi['mae_on_positive']:.4f}")

                # 收集到汇总表
                zi_metrics_rows.append({
                    'model': model_label,
                    'target': target,
                    **{f'test_{k}': v for k, v in m.items()},
                    **{f'zi_{k}': v for k, v in zi.items()},
                })

    # ── 保存合并预测文件 ──
    pred_path = os.path.join(args.output_dir, 'baseline_test_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  合并测试集预测已保存至: {pred_path}")

    # ── 分别保存四个模型的独立预测集（Grid_Utility 格式）──
    print(f"\n  保存各模型独立预测集（Grid_Utility 格式）...")
    for model_name in ['CatBoost', 'CB_Hurdle', 'LightGBM', 'RandomForest']:
        if model_name in all_model_preds and 'rent' in all_model_preds[model_name]:
            save_model_predictions(
                model_name=model_name,
                rent_pred=all_model_preds[model_name]['rent'],
                return_pred=all_model_preds[model_name]['return'],
                df_test_raw=df_test_raw,
                output_dir=args.output_dir,
            )
        else:
            print(f"  [{model_name}] 跳过：无可用预测结果。")

    # ── 保存零膨胀指标文件 ──
    zi_path = os.path.join(args.output_dir, 'baseline_zero_inflation_metrics.csv')
    zi_df = pd.DataFrame(zi_metrics_rows)
    zi_df.to_csv(zi_path, index=False)
    print(f"\n  零膨胀专项指标已保存至: {zi_path}")

    # 构建对比文件
    comp_df = pd.DataFrame(comparison_rows)

    # 统一列顺序：模型标识在前，超参数居中，性能指标在后
    priority_cols = ['model', 'model_type', 'target',
                     'learning_rate', 'depth', 'l2_leaf_reg', 'od_wait',
                     'num_leaves', 'min_child_samples', 'subsample',
                     'n_estimators', 'max_depth', 'min_samples_leaf',
                     'best_iteration',
                     'valid_poisson_soft', 'valid_rmse_soft', 'valid_mae_soft',
                     'valid_poisson', 'valid_rmse', 'valid_mae',
                     'hurdle_threshold_poisson', 'hurdle_threshold_f1', 'hurdle_threshold_used',
                     'hurdle_threshold_strategy',
                     'valid_true_zero_ratio', 'valid_pred_zero_ratio_poisson',
                     'valid_pred_zero_ratio_f1', 'valid_pred_zero_ratio_used',
                     'valid_zero_f1_poisson', 'valid_zero_f1_f1', 'valid_zero_f1_used',
                     'valid_classifier_logloss',
                     'training_seconds', 'notes']
    existing = [c for c in priority_cols if c in comp_df.columns]
    remaining = [c for c in comp_df.columns if c not in existing]
    comp_df = comp_df[existing + remaining]

    comp_path = os.path.join(args.output_dir, 'baseline_comparison.csv')
    comp_df.to_csv(comp_path, index=False)
    print(f"  模型对比表已保存至: {comp_path}")

    # ══════════════════════════════════════════════════════════════════════
    # 测试集排面对比：正面展示 CB_Hurdle 在零膨胀场景下的优势
    # ══════════════════════════════════════════════════════════════════════
    zi_df = pd.DataFrame(zi_metrics_rows)
    if not zi_df.empty:
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + "  测试集模型排面对比".center(70) + "║")
        print("╚" + "═" * 78 + "╝")

        for target in TARGETS:
            sub = zi_df[zi_df['target'] == target].copy()
            if sub.empty:
                continue

            true_zero = sub['zi_zero_ratio_true'].iloc[0]

            print(f"\n{'─' * 80}")
            print(f"  ◆ 目标: {target}  │  真实零值占比: {true_zero:.2%}")
            print(f"{'─' * 80}")

            # ── 表1: 全局回归指标 ──
            print(f"\n  📊 全局回归指标 (越低越好)")
            print(f"  {'Model':<18s} {'Poisson':>10s} {'RMSE':>10s} {'MAE':>10s}")
            print(f"  {'─' * 18} {'─' * 10} {'─' * 10} {'─' * 10}")
            sub_sorted = sub.sort_values('test_Poisson')
            best_poi = sub_sorted['test_Poisson'].iloc[0]
            for _, r in sub_sorted.iterrows():
                poi, rmse, mae = r['test_Poisson'], r['test_RMSE'], r['test_MAE']
                flag = "  ← 最优" if poi == best_poi else ""
                print(f"  {r['model']:<18s} {poi:>10.4f} {rmse:>10.4f} {mae:>10.4f}{flag}")

            # ── 表2: 零值检测能力 ──
            print(f"\n  🎯 零值检测能力 (Zero F1 = 召回×精确的调和平均)")
            print(f"  {'Model':<18s} {'Zero Recall':>12s} {'Zero Prec':>12s} {'Zero F1':>12s} {'零比Gap':>10s}")
            print(f"  {'─' * 18} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 10}")
            sub_sorted_f1 = sub.sort_values('zi_zero_f1', ascending=False)
            best_f1 = sub_sorted_f1['zi_zero_f1'].iloc[0]
            for _, r in sub_sorted_f1.iterrows():
                recall = r['zi_zero_recall']
                prec = r['zi_zero_precision']
                f1 = r['zi_zero_f1']
                gap = r['zi_zero_ratio_gap']
                flag = "  ← 最优" if f1 == best_f1 else ""
                recall_str = f"{recall:.4f}" if not np.isnan(recall) else "N/A"
                prec_str = f"{prec:.4f}" if not np.isnan(prec) else "N/A"
                f1_str = f"{f1:.4f}" if not np.isnan(f1) else "N/A"
                print(f"  {r['model']:<18s} {recall_str:>12s} {prec_str:>12s} {f1_str:>12s} {gap:>10.4f}{flag}")

            # ── 表3: 条件误差分解 ──
            print(f"\n  🔬 条件误差分解 (按真实值是否为0拆分)")
            print(f"  {'Model':<18s} {'RMSE on 0':>10s} {'RMSE on >0':>12s} {'MAE on 0':>10s} {'MAE on >0':>12s}")
            print(f"  {'─' * 18} {'─' * 10} {'─' * 12} {'─' * 10} {'─' * 12}")
            sub_sorted_rmse0 = sub.sort_values('zi_rmse_on_zero')
            best_rmse0 = sub_sorted_rmse0['zi_rmse_on_zero'].iloc[0]
            best_mae0 = sub.sort_values('zi_mae_on_zero')['zi_mae_on_zero'].iloc[0]
            for _, r in sub_sorted_rmse0.iterrows():
                rmse0, rmse_pos = r['zi_rmse_on_zero'], r['zi_rmse_on_positive']
                mae0, mae_pos = r['zi_mae_on_zero'], r['zi_mae_on_positive']
                flags = []
                if rmse0 == best_rmse0:
                    flags.append("RMSE₀最优")
                if mae0 == best_mae0:
                    flags.append("MAE₀最优")
                flag_str = "  ← " + ", ".join(flags) if flags else ""
                print(f"  {r['model']:<18s} {rmse0:>10.4f} {rmse_pos:>12.4f} {mae0:>10.4f} {mae_pos:>12.4f}{flag_str}")

        # ── 综合结论 ──
        print(f"\n{'═' * 80}")
        print(f"  📋 综合结论")
        print(f"{'═' * 80}")

        for target in TARGETS:
            sub = zi_df[zi_df['target'] == target]
            if sub.empty:
                continue
            models = sub['model'].tolist()
            hurdles = sub[sub['model'] == 'CB Hurdle']
            if hurdles.empty:
                continue
            h = hurdles.iloc[0]

            # 找到次优模型
            others = sub[sub['model'] != 'CB Hurdle']
            best_other_f1 = others['zi_zero_f1'].max()
            best_other_rmse0 = others['zi_rmse_on_zero'].min()

            print(f"\n  [{target}] CB Hurdle vs 其他模型中最好者:")
            print(f"    Zero F1:         {h['zi_zero_f1']:.4f}  vs  {best_other_f1:.4f}  "
                  f"(提升 {h['zi_zero_f1'] - best_other_f1:+.4f})")
            print(f"    RMSE on Zero:    {h['zi_rmse_on_zero']:.4f}  vs  {best_other_rmse0:.4f}  "
                  f"(降低 {h['zi_rmse_on_zero'] - best_other_rmse0:+.4f})")

            # 关键诊断
            cb_recall = h['zi_zero_recall']
            if not np.isnan(cb_recall) and cb_recall > 0.3:
                print(f"    ✅ CB_Hurdle 成功识别了 {cb_recall:.1%} 的真实零需求样本")
                gap = h['zi_zero_ratio_gap']
                if gap < 0.05:
                    print(f"    ✅ 预测零值比例与真实零值比例仅差 {gap:.2%}，分布匹配优秀")
            else:
                print(f"    ⚠️  零值召回率偏低 ({cb_recall:.2%})，考虑检查分类器特征或阈值")

    print("\n" + "=" * 60)
    print("完成。")
    print("=" * 60)


if __name__ == '__main__':
    main()
