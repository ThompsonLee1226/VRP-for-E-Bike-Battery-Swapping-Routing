"""
轻量级参数扫描脚本：为 LightGBM 和 Random Forest 补充与 CatBoost 相同
超参数空间下的训练日志，以满足论文``需要补充的数据与训练''第(4)项：
与 LightGBM 和 Random Forest 的完整对比。

设计原则：
  - 不修改原有的 LightGBM_train.py / Random_Forest_train.py
  - 复用 CB_Hurdle_train.py 中的时间切分逻辑和特征工程
  - 使用与 CatBoost 扫描相同的训练/验证划分
  - 结果写入统一的 CSV 格式供 supplementary_analysis.py 读取

超参数空间（与 CatBoost 一致）：
  - LightGBM: num_leaves ∈ {63, 127, 255}, learning_rate ∈ {0.015, 0.02, 0.03, 0.05},
              min_child_samples ∈ {10, 20, 30}
  - Random Forest: n_estimators ∈ {500, 1000, 2000}, max_depth ∈ {12, 18, 24},
                   min_samples_leaf ∈ {3, 5, 10}

用法：
  python param_scan_lgb_rf.py --train_csv <训练集CSV> --test_csv <测试集CSV> \
      --model {lgb, rf, all} --output_dir Param_Scan_Results

注意：
  - LightGBM 需要较多样本，建议在 GPU 服务器上运行
  - Random Forest 的扫描耗时会显著高于 boosting 模型
  - 建议先运行 --model lgb，确认无误后再运行 --model rf
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance, mean_squared_error, log_loss
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 复用 CB_Hurdle_train.py 中的数据处理函数
try:
    from CB_Hurdle_train import (
        add_feature_engineering,
        fill_missing_values,
        split_train_valid,
    )
except ImportError:
    print("警告: 无法从 CB_Hurdle_train 导入函数，将使用内置版本")
    # Fallback: lightweight reimplementation
    def add_feature_engineering(df):
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
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].fillna('missing').astype(str)
            else:
                df[col] = df[col].fillna(0)
        return df

    def split_train_valid(df, features, y_raw):
        """时间切分：前80%训练，后20%验证"""
        time_col = 'datetime'
        if time_col not in df.columns:
            raise ValueError(f"缺少时间列: {time_col}")
        dt = pd.to_datetime(df[time_col], errors='coerce')
        ordered_idx = dt.sort_values().index
        split_pos = int(len(ordered_idx) * 0.8)
        train_idx = ordered_idx[:split_pos]
        valid_idx = ordered_idx[split_pos:]
        X = df[features]
        X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y_raw.loc[train_idx], y_raw.loc[valid_idx]
        train_time = pd.to_datetime(df.loc[train_idx, time_col], errors='coerce').dropna()
        valid_time = pd.to_datetime(df.loc[valid_idx, time_col], errors='coerce').dropna()
        train_time_range = f"{train_time.min()} -> {train_time.max()}"
        valid_time_range = f"{valid_time.min()} -> {valid_time.max()}"
        return X_train, X_valid, y_train, y_valid, train_time_range, valid_time_range


FEATURES = [
    'h3', 'temperature', 'wind_level', 'rain_level',
    'month', 'day_of_week', 'is_weekend', 'hour',
    'rent_mean_7d', 'return_mean_7d', 'lag_nb_rent', 'lag_nb_return',
    'normal_power_bike_count', 'soon_low_power_bike_count', 'low_power_bike_count',
    'latitude', 'longitude',
    'temp_x_rain', 'available_power_bike_gap', 'is_rush_hour',
]


# ======================================================================
# LightGBM 参数扫描
# ======================================================================

def scan_lightgbm(
    df_train: pd.DataFrame,
    features: List[str],
    targets: List[str],
    output_csv: str,
) -> pd.DataFrame:
    """对 LightGBM 进行网格扫描。"""
    try:
        import lightgbm as lgb
    except ImportError:
        print("错误: 需要安装 lightgbm。请运行: pip install lightgbm")
        return pd.DataFrame()

    lr_list = [0.015, 0.02, 0.03, 0.05]
    num_leaves_list = [63, 127, 255]
    min_child_samples_list = [10, 20, 30]
    early_stopping_rounds = 100
    num_boost_round = 5000

    results = []

    total_runs = len(lr_list) * len(num_leaves_list) * len(min_child_samples_list) * len(targets)
    run_idx = 0

    for target in targets:
        print(f"\n{'='*50}")
        print(f"LightGBM 扫描目标: {target}")
        print(f"{'='*50}")

        y_raw = df_train[target].astype(float)
        X_train, X_valid, y_train, y_valid, train_range, valid_range = split_train_valid(
            df_train, features, y_raw
        )

        # LightGBM需要单独处理类别特征
        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=['h3'])
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

        for lr in lr_list:
            for num_leaves in num_leaves_list:
                for min_child in min_child_samples_list:
                    run_idx += 1
                    print(f"\n[{run_idx}/{total_runs}] lr={lr}, num_leaves={num_leaves}, "
                          f"min_child_samples={min_child}, target={target}")

                    params = {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'learning_rate': lr,
                        'num_leaves': num_leaves,
                        'min_child_samples': min_child,
                        'feature_fraction': 0.8,
                        'n_jobs': -1,
                        'verbose': -1,
                        'random_state': 42,
                    }

                    t_start = time.time()
                    model = lgb.train(
                        params,
                        lgb_train,
                        num_boost_round=num_boost_round,
                        valid_sets=[lgb_train, lgb_valid],
                        valid_names=['train', 'valid'],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                            lgb.log_evaluation(period=0),
                        ],
                    )

                    y_pred = np.clip(model.predict(X_valid, num_iteration=model.best_iteration), 0, None)
                    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

                    # 计算Poisson偏差（与CatBoost对齐的评估指标）
                    try:
                        poisson_dev = mean_poisson_deviance(y_valid, y_pred)
                    except Exception:
                        poisson_dev = np.nan

                    result = {
                        'model': 'LightGBM',
                        'target': target,
                        'learning_rate': lr,
                        'num_leaves': num_leaves,
                        'min_child_samples': min_child,
                        'best_iteration': model.best_iteration,
                        'train_rmse': np.sqrt(mean_squared_error(
                            y_train, np.clip(model.predict(X_train, num_iteration=model.best_iteration), 0, None)
                        )),
                        'valid_rmse': rmse,
                        'valid_poisson_deviance': poisson_dev,
                        'training_seconds': round(time.time() - t_start, 1),
                        'train_time_range': train_range,
                        'valid_time_range': valid_range,
                    }
                    results.append(result)
                    print(f"    RMSE={rmse:.4f}, Poisson={poisson_dev:.4f}, "
                          f"best_iter={model.best_iteration}, time={result['training_seconds']}s")

        # 每完成一个target保存一次
        pd.DataFrame(results).to_csv(output_csv, index=False)

    return pd.DataFrame(results)


# ======================================================================
# Random Forest 参数扫描
# ======================================================================

def scan_random_forest(
    df_train: pd.DataFrame,
    features: List[str],
    targets: List[str],
    output_csv: str,
) -> pd.DataFrame:
    """对 Random Forest 进行网格扫描。"""
    n_estimators_list = [500, 1000, 2000]
    max_depth_list = [12, 18, 24]
    min_samples_leaf_list = [3, 5, 10]

    results = []
    total_runs = len(n_estimators_list) * len(max_depth_list) * len(min_samples_leaf_list) * len(targets)
    run_idx = 0

    for target in targets:
        print(f"\n{'='*50}")
        print(f"Random Forest 扫描目标: {target}")
        print(f"{'='*50}")

        y_raw = df_train[target].astype(float)
        X_train, X_valid, y_train, y_valid, train_range, valid_range = split_train_valid(
            df_train, features, y_raw
        )

        # RF需要编码h3
        h3_mapping = {val: idx for idx, val in enumerate(X_train['h3'].unique())}
        X_train_enc = X_train.copy()
        X_valid_enc = X_valid.copy()
        X_train_enc['h3'] = X_train_enc['h3'].map(h3_mapping).fillna(-1).astype(np.int32)
        X_valid_enc['h3'] = X_valid_enc['h3'].map(h3_mapping).fillna(-1).astype(np.int32)

        for n_est in n_estimators_list:
            for max_d in max_depth_list:
                for min_leaf in min_samples_leaf_list:
                    run_idx += 1
                    print(f"\n[{run_idx}/{total_runs}] n_estimators={n_est}, "
                          f"max_depth={max_d}, min_samples_leaf={min_leaf}, target={target}")

                    t_start = time.time()
                    model = RandomForestRegressor(
                        n_estimators=n_est,
                        max_depth=max_d,
                        min_samples_leaf=min_leaf,
                        max_features='sqrt',
                        n_jobs=-1,
                        verbose=0,
                        random_state=42,
                    )
                    model.fit(X_train_enc, y_train)

                    y_pred = np.clip(model.predict(X_valid_enc), 0, None)
                    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
                    try:
                        poisson_dev = mean_poisson_deviance(y_valid, y_pred)
                    except Exception:
                        poisson_dev = np.nan

                    result = {
                        'model': 'RandomForest',
                        'target': target,
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'min_samples_leaf': min_leaf,
                        'train_rmse': np.sqrt(mean_squared_error(
                            y_train, np.clip(model.predict(X_train_enc), 0, None)
                        )),
                        'valid_rmse': rmse,
                        'valid_poisson_deviance': poisson_dev,
                        'training_seconds': round(time.time() - t_start, 1),
                        'train_time_range': train_range,
                        'valid_time_range': valid_range,
                    }
                    results.append(result)
                    print(f"    RMSE={rmse:.4f}, Poisson={poisson_dev:.4f}, "
                          f"time={result['training_seconds']}s")

        pd.DataFrame(results).to_csv(output_csv, index=False)

    return pd.DataFrame(results)


# ======================================================================
# 主流程
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='LightGBM / Random Forest 参数扫描')
    parser.add_argument('--train_csv', type=str, required=True,
                        help='训练集CSV文件路径')
    parser.add_argument('--test_csv', type=str, default=None,
                        help='测试集CSV文件路径（预留，当前不使用）')
    parser.add_argument('--model', type=str, default='all',
                        choices=['lgb', 'rf', 'all'],
                        help='扫描哪个模型')
    parser.add_argument('--targets', type=str, nargs='+', default=['rent', 'return'],
                        help='目标变量')
    parser.add_argument('--output_dir', type=str, default='Param_Scan_Results',
                        help='结果输出目录')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("加载训练数据...")
    df = pd.read_csv(args.train_csv)
    cols_to_drop = ['region_code', 'Unnamed: 21']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    if 'h3' in df.columns:
        df['h3'] = df['h3'].astype(str)
    df = fill_missing_values(df)
    df = add_feature_engineering(df)
    print(f"数据形状: {df.shape}")

    # 验证所需列
    for col in FEATURES:
        if col not in df.columns:
            raise ValueError(f"缺少特征列: {col}")

    if args.model in ('lgb', 'all'):
        print("\n" + "="*60)
        print("开始 LightGBM 参数扫描")
        print("="*60)
        output_csv = os.path.join(args.output_dir, 'lgb_param_scan_results.csv')
        scan_lightgbm(df, FEATURES, args.targets, output_csv)
        print(f"\nLightGBM 扫描结果已保存至: {output_csv}")

    if args.model in ('rf', 'all'):
        print("\n" + "="*60)
        print("开始 Random Forest 参数扫描")
        print("="*60)
        output_csv = os.path.join(args.output_dir, 'rf_param_scan_results.csv')
        scan_random_forest(df, FEATURES, args.targets, output_csv)
        print(f"\nRandom Forest 扫描结果已保存至: {output_csv}")

    print("\n" + "="*60)
    print("参数扫描完成。")
    print("="*60)


if __name__ == '__main__':
    main()
