"""
基准模型对比脚本：在完全相同的训练/验证/测试划分下，训练并评估四种架构，
输出两份可直接用于论文的CSV文件。

四种架构：
  (1) CatBoost      — 单阶段Poisson回归（已由 CatBoost_train.py 完成）
  (2) CatBoost Hurdle — 两阶段Hurdle模型   （已由 CB_Hurdle_train.py 完成）
  (3) LightGBM       — 单阶段RMSE回归      （本脚本训练）
  (4) Random Forest  — 单阶段RMSE回归      （本脚本训练）

实验设计原则：
  - 四种架构使用完全相同的训练/验证时间切分（前80%训练，后20%验证）
  - 使用完全相同的特征集
  - 统一以Poisson偏差作为核心对比指标
  - CatBoost和CB_Hurdle的结果通过读取已有的training_summary.csv获取最优行
  - LightGBM和Random Forest各使用一组有代表性的默认配置，不做参数扫描

输出文件：
  (A) baseline_comparison.csv — 四种架构的参数与性能对比（每行一个模型×目标）
  (B) baseline_test_predictions.csv — 四种架构在测试集上的预测值（含真实值）

用法：
    python 2_Training/baseline_comparison.py \
      --train_csv battery_swapping_routing_data_train_time70.csv \
      --test_csv battery_swapping_routing_data_valid_time30.csv \
      --cb_summary 2_Training/training_summary.csv \
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
from sklearn.metrics import mean_poisson_deviance, mean_squared_error, mean_absolute_error

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
]

TARGETS = ['rent', 'return']


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
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
    return {
        'Poisson': mean_poisson_deviance(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
    }


# ── CatBoost / CB_Hurdle 结果读取 ───────────────────────────────────────

def load_catboost_results(cb_summary_csv: str) -> List[Dict]:
    """
    从 training_summary.csv 中读取 CatBoost 和 CB_Hurdle 的最优结果。
    策略：对每种 model_type，取 od_wait=30 下 rent_final_metric 最小的行。
    """
    if not cb_summary_csv or not os.path.exists(cb_summary_csv):
        print(f"  注意: 未找到 {cb_summary_csv}，将跳过CatBoost结果。")
        return []

    df = pd.read_csv(cb_summary_csv)
    rows = []

    for model_type in ['CB', 'CB_Hurdle']:
        sub = df[df['model_type'] == model_type].copy()
        if sub.empty:
            print(f"  注意: CSV中无 model_type={model_type} 的记录。")
            continue

        # 优先选 od_wait=30
        sub30 = sub[sub.get('cb_od_wait', np.nan) == 30]
        if sub30.empty:
            sub30 = sub

        # 找 rent_final_metric 最小的行
        best = sub30.loc[sub30['rent_final_metric'].idxmin()]

        for target in TARGETS:
            rows.append({
                'model': model_type.replace('_', ' '),
                'model_type': model_type,
                'target': target,
                'learning_rate': best.get('cb_learning_rate', np.nan),
                'depth': best.get('cb_depth', np.nan),
                'l2_leaf_reg': best.get('cb_l2_leaf_reg', np.nan),
                'od_wait': best.get('cb_od_wait', np.nan),
                'best_iteration': best.get(f'{target}_best_iteration', np.nan),
                'valid_poisson': best.get(f'{target}_final_metric', np.nan),
                'valid_rmse': np.nan,  # CatBoost未记录RMSE
                'valid_mae': np.nan,
                'training_seconds': np.nan,
                'notes': f"来自 {cb_summary_csv}, run={best.get('run_timestamp', '?')}",
            })

    print(f"  从 {cb_summary_csv} 读取了 {len(rows)} 条CatBoost记录。")
    return rows


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
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
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
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=0),
        ],
    )
    elapsed = time.time() - t_start

    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    metrics = compute_metrics(y_valid, y_pred)

    info = {
        'learning_rate': params['learning_rate'],
        'num_leaves': params['num_leaves'],
        'min_child_samples': params['min_child_samples'],
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
        n_estimators=1000,
        max_depth=18,
        min_samples_leaf=5,
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
        'n_estimators': 1000,
        'max_depth': 18,
        'min_samples_leaf': 5,
        'best_iteration': np.nan,
        'training_seconds': round(elapsed, 1),
        **{f'valid_{k.lower()}': v for k, v in metrics.items()},
    }
    return model, info, h3_mapping


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='四种架构基准对比')
    parser.add_argument('--train_csv', required=True, help='训练集CSV')
    parser.add_argument('--test_csv', required=True, help='测试集CSV')
    parser.add_argument('--cb_summary', default=None, help='CatBoost training_summary.csv路径')
    parser.add_argument('--output_dir', default='Baseline_Comparison_Results', help='输出目录')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. 加载并预处理数据 ──
    print("=" * 60)
    print("[1/5] 加载数据...")
    df_train = pd.read_csv(args.train_csv)
    df_test = pd.read_csv(args.test_csv)

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

    # ── 2. 读取CatBoost结果 ──
    print("\n[2/5] 读取CatBoost / CB_Hurdle已有结果...")
    cb_rows = load_catboost_results(args.cb_summary)
    comparison_rows.extend(cb_rows)

    # ── 3. 训练LightGBM ──
    print("\n[3/5] 训练 LightGBM...")
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

    # ── 4. 训练Random Forest ──
    print("\n[4/5] 训练 Random Forest...")
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

    # ── 5. 测试集预测与汇总 ──
    print("\n[5/5] 测试集预测与汇总...")

    # 构建测试集预测DataFrame
    pred_df = df_test[['h3', 'datetime']].copy() if 'datetime' in df_test.columns else df_test[['h3']].copy()

    for target in TARGETS:
        y_test = df_test[target].astype(float)
        pred_df[f'{target}_true'] = y_test.values

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

        # Random Forest (复用训练时的 h3→int 映射，未知值填 -1)
        if 'RandomForest' in model_objects:
            X_test_rf = df_test[FEATURES].copy()
            h3_map_rf = rf_h3_mapping.get(target, {})
            X_test_rf['h3'] = X_test_rf['h3'].map(h3_map_rf).fillna(-1).astype(np.int32)
            rf_pred = np.clip(
                model_objects['RandomForest'][target].predict(X_test_rf), 0, None
            )
            pred_df[f'{target}_rf'] = rf_pred

        # CatBoost / CB_Hurdle测试集指标（从已有模型读取，若无则跳过）
        print(f"\n  测试集指标 — {target}:")
        for model_name in ['LightGBM', 'RandomForest']:
            col = f'{target}_{model_name[:3].lower()}'
            if col in pred_df.columns:
                m = compute_metrics(y_test, pred_df[col])
                print(f"    {model_name:15s}: Poisson={m['Poisson']:.4f}, "
                      f"RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}")

    # 保存预测文件
    pred_path = os.path.join(args.output_dir, 'baseline_test_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  测试集预测已保存至: {pred_path}")

    # 构建对比文件
    comp_df = pd.DataFrame(comparison_rows)

    # 统一列顺序：模型标识在前，超参数居中，性能指标在后
    priority_cols = ['model', 'model_type', 'target',
                     'learning_rate', 'depth', 'l2_leaf_reg', 'od_wait',
                     'num_leaves', 'min_child_samples',
                     'n_estimators', 'max_depth', 'min_samples_leaf',
                     'best_iteration',
                     'valid_poisson', 'valid_rmse', 'valid_mae',
                     'training_seconds', 'notes']
    existing = [c for c in priority_cols if c in comp_df.columns]
    remaining = [c for c in comp_df.columns if c not in existing]
    comp_df = comp_df[existing + remaining]

    comp_path = os.path.join(args.output_dir, 'baseline_comparison.csv')
    comp_df.to_csv(comp_path, index=False)
    print(f"  模型对比表已保存至: {comp_path}")

    # 打印核心对比结果
    print("\n" + "=" * 60)
    print("核心对比结果 (验证集Poisson偏差)")
    print("=" * 60)
    for target in TARGETS:
        print(f"\n  [{target}]")
        sub = comp_df[comp_df['target'] == target].sort_values('valid_poisson')
        for _, row in sub.iterrows():
            print(f"    {row['model']:20s}  Poisson={row['valid_poisson']:.4f}")

    print("\n" + "=" * 60)
    print("完成。")
    print("=" * 60)


if __name__ == '__main__':
    main()
