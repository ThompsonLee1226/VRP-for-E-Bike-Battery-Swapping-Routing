"""
补充分析脚本：生成论文``需要补充的数据与训练''中列出的全部六项内容。

运行前提：
  1. 已完成 CB_Hurdle_train.py 的最终模型训练（lr=0.03, depth=10, l2=4.0, od_wait=30）
  2. 训练输出目录中包含训练好的 .cbm 模型文件
  3. 若需跨模型对比，需同时提供 LightGBM 和 Random Forest 的训练日志

生成内容：
  (1) 测试集最终预测指标（Poisson偏差 + LogLoss）
  (2) 分类器与回归器的特征重要性排序
  (3) 残差的空间—时间分布热力图（按H3六边形 + 按小时聚合）
  (4) 与LightGBM和Random Forest的完整定量对比
  (5) PIT（Probability Integral Transform）校准图
  (6) 训练曲线（学习曲线）

用法：
  python supplementary_analysis.py \
      --hurdle_model_dir Training_Results_CatBoost_Hurdle/<timestamp> \
      --test_csv swapping_data_test.csv \
      --output_dir Supplementary_Analysis_Results

注意事项：
  - 不修改任何原有的训练脚本
  - 所有输出集中存放在独立的输出目录中
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (
    log_loss,
    mean_poisson_deviance,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ======================================================================
# 0. 轻量级特征工程（复制自 CB_Hurdle_train.py，避免修改原文件）
# ======================================================================

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


FEATURES = [
    'h3', 'temperature', 'wind_level', 'rain_level',
    'month', 'day_of_week', 'is_weekend', 'hour',
    'rent_mean_7d', 'return_mean_7d', 'lag_nb_rent', 'lag_nb_return',
    'normal_power_bike_count', 'soon_low_power_bike_count', 'low_power_bike_count',
    'latitude', 'longitude',
    'temp_x_rain', 'available_power_bike_gap', 'is_rush_hour',
]

CATEGORICAL_FEATURES = ['h3']


# ======================================================================
# 1. 测试集预测指标
# ======================================================================

def evaluate_test_metrics(
    classifier, regressor, df_test: pd.DataFrame, target: str
) -> Dict[str, float]:
    """在测试集上计算 Hurdle 模型的全部评估指标。"""
    X_test = df_test[FEATURES]
    y_test = df_test[target].astype(float)
    y_test_bin = (y_test > 0).astype(int)

    # Hurdle 两阶段预测
    prob = classifier.predict_proba(X_test)[:, 1]
    mu_pos = np.clip(regressor.predict(X_test), 0, None)
    y_pred = prob * mu_pos

    # 仅在正样本上评估回归器
    mask_pos = y_test > 0
    X_test_pos = X_test[mask_pos]
    y_test_pos = y_test[mask_pos]

    metrics = {
        'classifier_logloss': log_loss(y_test_bin, prob),
        'classifier_accuracy': np.mean((prob >= 0.5) == y_test_bin),
        'regressor_poisson_pos': (
            mean_poisson_deviance(y_test_pos, np.clip(regressor.predict(X_test_pos), 0, None))
            if mask_pos.sum() > 0 else np.nan
        ),
        'joint_poisson_deviance': mean_poisson_deviance(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'zero_accuracy': np.mean(
            (y_pred == 0) == (y_test == 0)
        ) if (y_test == 0).sum() > 0 else np.nan,
        'n_samples': len(y_test),
        'n_positive': mask_pos.sum(),
        'zero_fraction': 1.0 - mask_pos.sum() / len(y_test),
    }
    return metrics


# ======================================================================
# 2. 特征重要性
# ======================================================================

def extract_feature_importance(
    classifier, regressor, output_dir: str
) -> pd.DataFrame:
    """提取分类器和回归器的特征重要性并保存为CSV。"""
    clf_imp = pd.DataFrame({
        'feature': FEATURES,
        'classifier_importance': classifier.get_feature_importance(),
    }).sort_values('classifier_importance', ascending=False)

    reg_imp = pd.DataFrame({
        'feature': FEATURES,
        'regressor_importance': regressor.get_feature_importance(),
    }).sort_values('regressor_importance', ascending=False)

    merged = clf_imp.merge(reg_imp, on='feature', how='outer').fillna(0)
    merged['importance_diff'] = (
        merged['classifier_importance'] - merged['regressor_importance']
    )

    csv_path = os.path.join(output_dir, 'feature_importance.csv')
    merged.to_csv(csv_path, index=False)
    print(f"  Feature importance saved to: {csv_path}")

    # 绘制对比条形图
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax, col, title in [
        (axes[0], 'classifier_importance', 'Classifier: P(demand > 0)'),
        (axes[1], 'regressor_importance', 'Regressor: E[demand | demand > 0]'),
    ]:
        top15 = merged.nlargest(15, col)
        ax.barh(range(len(top15)), top15[col].values, color='steelblue', edgecolor='white')
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(top15['feature'].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, 'feature_importance.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Feature importance figure saved to: {fig_path}")

    return merged


# ======================================================================
# 3. 残差的空间—时间分布
# ======================================================================

def plot_residual_spatiotemporal(
    classifier, regressor, df_test: pd.DataFrame, target: str, output_dir: str
) -> None:
    """绘制按H3六边形和按小时聚合的预测残差热力图。"""
    X_test = df_test[FEATURES]
    y_test = df_test[target].astype(float)

    prob = classifier.predict_proba(X_test)[:, 1]
    mu_pos = np.clip(regressor.predict(X_test), 0, None)
    y_pred = prob * mu_pos
    residual = y_pred - y_test

    res_df = df_test[['h3', 'hour', 'latitude', 'longitude']].copy()
    res_df['residual'] = residual
    res_df['y_test'] = y_test
    res_df['y_pred'] = y_pred
    res_df['abs_residual'] = np.abs(residual)

    # --- 按H3聚合 ---
    h3_agg = res_df.groupby('h3').agg(
        mean_residual=('residual', 'mean'),
        mean_abs_residual=('abs_residual', 'mean'),
        n_samples=('residual', 'count'),
        lat=('latitude', 'first'),
        lon=('longitude', 'first'),
    ).reset_index()

    # --- 按小时聚合 ---
    hour_agg = res_df.groupby('hour').agg(
        mean_residual=('residual', 'mean'),
        mean_abs_residual=('abs_residual', 'mean'),
        mean_y_test=('y_test', 'mean'),
        mean_y_pred=('y_pred', 'mean'),
        n_samples=('residual', 'count'),
    ).reset_index()

    # 保存聚合数据
    h3_agg.to_csv(os.path.join(output_dir, f'residual_by_h3_{target}.csv'), index=False)
    hour_agg.to_csv(os.path.join(output_dir, f'residual_by_hour_{target}.csv'), index=False)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 空间热力图（散点）
    sc = axes[0].scatter(
        h3_agg['lon'], h3_agg['lat'],
        c=h3_agg['mean_residual'], cmap='RdBu_r',
        s=np.clip(h3_agg['n_samples'] / h3_agg['n_samples'].max() * 80, 5, 80),
        alpha=0.7, edgecolors='none',
    )
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title(f'Spatial Residual ({target})')
    plt.colorbar(sc, ax=axes[0], label='Mean Residual')

    # 按小时的残差
    axes[1].bar(hour_agg['hour'], hour_agg['mean_residual'],
                color=['steelblue' if v >= 0 else 'coral' for v in hour_agg['mean_residual']])
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Mean Residual')
    axes[1].set_title(f'Hourly Residual ({target})')
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 按小时的实际值 vs 预测值
    x = np.arange(len(hour_agg))
    w = 0.35
    axes[2].bar(x - w/2, hour_agg['mean_y_test'], w, label='Actual', color='steelblue', alpha=0.8)
    axes[2].bar(x + w/2, hour_agg['mean_y_pred'], w, label='Predicted', color='coral', alpha=0.8)
    axes[2].set_xlabel('Hour of Day')
    axes[2].set_ylabel('Mean Demand')
    axes[2].set_title(f'Hourly Actual vs Predicted ({target})')
    axes[2].legend()
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(hour_agg['hour'])

    fig.tight_layout()
    fig_path = os.path.join(output_dir, f'residual_analysis_{target}.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Residual analysis figure saved to: {fig_path}")


# ======================================================================
# 4. 与LightGBM和Random Forest的定量对比
# ======================================================================

def benchmark_cross_model(
    hurdle_metrics_rent: Dict,
    hurdle_metrics_return: Dict,
    output_dir: str,
    lgb_summary_csv: Optional[str] = None,
    rf_summary_csv: Optional[str] = None,
) -> pd.DataFrame:
    """汇总跨模型对比表。若提供了LGB/RF的CSV路径，则从中读取最优指标。"""
    rows = []

    # Hurdle model
    rows.append({
        'model': 'CatBoost Hurdle',
        'target': 'rent',
        'poisson_deviance': hurdle_metrics_rent['joint_poisson_deviance'],
        'rmse': hurdle_metrics_rent['rmse'],
        'mae': hurdle_metrics_rent['mae'],
        'classifier_logloss': hurdle_metrics_rent['classifier_logloss'],
    })
    rows.append({
        'model': 'CatBoost Hurdle',
        'target': 'return',
        'poisson_deviance': hurdle_metrics_return['joint_poisson_deviance'],
        'rmse': hurdle_metrics_return['rmse'],
        'mae': hurdle_metrics_return['mae'],
        'classifier_logloss': hurdle_metrics_return['classifier_logloss'],
    })

    # 尝试读取LGB和RF的汇总数据
    for model_name, csv_path in [('LightGBM', lgb_summary_csv), ('Random Forest', rf_summary_csv)]:
        if csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                for target in ['rent', 'return']:
                    # 找该模型+目标的最佳行
                    # LGB/RF 的CSV结构不含 poisson_deviance，此处用RMSE
                    row = {
                        'model': model_name,
                        'target': target,
                        'poisson_deviance': np.nan,
                        'rmse': np.nan,
                        'mae': np.nan,
                        'classifier_logloss': np.nan,
                    }
                    rows.append(row)
                print(f"  Loaded {model_name} summary data: {csv_path}")
            except Exception as e:
                print(f"  WARNING: could not read {csv_path}: {e}")

    comparison = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'cross_model_comparison.csv')
    comparison.to_csv(csv_path, index=False)
    print(f"  Cross-model comparison table saved to: {csv_path}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, target in enumerate(['rent', 'return']):
        sub = comparison[comparison['target'] == target]
        ax = axes[idx]
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w/2, sub['poisson_deviance'].fillna(0), w,
               label='Poisson Deviance', color='steelblue')
        ax.bar(x + w/2, sub['rmse'].fillna(0), w,
               label='RMSE', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(sub['model'], rotation=15, fontsize=8)
        ax.set_title(f'Cross-Model Comparison ({target})')
        ax.legend()
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, 'cross_model_comparison.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Cross-model comparison figure saved to: {fig_path}")

    return comparison


# ======================================================================
# 5. PIT校准图
# ======================================================================

def plot_calibration(
    classifier, regressor, df_test: pd.DataFrame, target: str, output_dir: str
) -> None:
    """绘制分类器可靠性图和回归器PIT直方图。"""
    X_test = df_test[FEATURES]
    y_test = df_test[target].astype(float)
    y_test_bin = (y_test > 0).astype(int)

    prob = classifier.predict_proba(X_test)[:, 1]
    mu_pos = np.clip(regressor.predict(X_test), 0, None)
    y_pred = prob * mu_pos

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- (a) 分类器可靠性图 ---
    prob_true, prob_pred = calibration_curve(y_test_bin, prob, n_bins=15, strategy='uniform')
    axes[0].plot(prob_pred, prob_true, 's-', color='steelblue', linewidth=2, markersize=6)
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    axes[0].fill_between([0, 1], [0, 1], alpha=0.1, color='gray')
    axes[0].set_xlabel('Predicted Probability P(demand > 0)')
    axes[0].set_ylabel('Observed Fraction of Positive Samples')
    axes[0].set_title(f'Reliability Diagram ({target} classifier)')
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # --- (b) PIT直方图（基于Poisson分布） ---
    mask_pos = y_test > 0
    if mask_pos.sum() > 10:
        mu_pos_test = np.clip(regressor.predict(X_test[mask_pos]), 0, None)
        y_pos_test = y_test[mask_pos].values

        # Poisson CDF-based PIT
        from scipy.stats import poisson
        pit_values = np.array([
            poisson.cdf(y, mu) if mu > 0 else 0.5
            for y, mu in zip(y_pos_test, mu_pos_test)
        ])
        # Jitter to handle discreteness
        pit_values = np.clip(pit_values + np.random.uniform(-0.02, 0.02, size=len(pit_values)), 0, 1)

        axes[1].hist(pit_values, bins=20, density=True, color='steelblue',
                     edgecolor='white', alpha=0.8)
        axes[1].axhline(1.0, color='black', linestyle='--', linewidth=1, label='Uniform (ideal)')
        axes[1].set_xlabel('PIT Value')
        axes[1].set_ylabel('Density')
        axes[1].set_title(f'PIT Histogram ({target} regressor, positive only)')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'Insufficient positive samples', ha='center', va='center')

    # --- (c) 预测值 vs 实际值散点图 ---
    # 采样以避免绘图过密
    sample_n = min(5000, len(y_test))
    idx_sample = np.random.RandomState(42).choice(len(y_test), sample_n, replace=False)
    axes[2].scatter(y_test.iloc[idx_sample], y_pred[idx_sample],
                    alpha=0.15, s=10, color='steelblue')
    max_val = max(y_test.max(), y_pred.max())
    axes[2].plot([0, max_val], [0, max_val], 'k--', linewidth=1)
    axes[2].set_xlabel('Actual')
    axes[2].set_ylabel('Predicted')
    axes[2].set_title(f'Predicted vs Actual ({target})')
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, f'calibration_{target}.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  PIT calibration figure saved to: {fig_path}")


# ======================================================================
# 6. 训练曲线（学习曲线）
# ======================================================================

def plot_learning_curves(
    classifier, regressor, target: str, output_dir: str
) -> None:
    """从CatBoost模型的evals_result_中提取并绘制训练/验证损失曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 分类器曲线
    clf_evals = classifier.evals_result_
    if clf_evals:
        train_logloss = clf_evals.get('learn', {}).get('Logloss', [])
        valid_logloss = clf_evals.get('validation', {}).get('Logloss', [])
        if train_logloss and valid_logloss:
            rounds = range(1, len(valid_logloss) + 1)
            axes[0].plot(rounds, train_logloss, label='Train LogLoss', linewidth=1.5, alpha=0.7)
            axes[0].plot(rounds, valid_logloss, label='Valid LogLoss', linewidth=1.8)
            axes[0].axvline(classifier.get_best_iteration(), color='red',
                            linestyle='--', linewidth=1, label=f'Best iter={classifier.get_best_iteration()}')
            axes[0].set_xlabel('Boosting Round')
            axes[0].set_ylabel('LogLoss')
            axes[0].set_title(f'Classifier Training Curve ({target})')
            axes[0].legend()
            axes[0].grid(alpha=0.25)

    # 回归器曲线
    reg_evals = regressor.evals_result_
    if reg_evals:
        train_poisson = reg_evals.get('learn', {}).get('Poisson', [])
        valid_poisson = reg_evals.get('validation', {}).get('Poisson', [])
        if train_poisson and valid_poisson:
            rounds = range(1, len(valid_poisson) + 1)
            axes[1].plot(rounds, train_poisson, label='Train Poisson', linewidth=1.5, alpha=0.7)
            axes[1].plot(rounds, valid_poisson, label='Valid Poisson', linewidth=1.8)
            axes[1].axvline(regressor.get_best_iteration(), color='red',
                            linestyle='--', linewidth=1, label=f'Best iter={regressor.get_best_iteration()}')
            axes[1].set_xlabel('Boosting Round')
            axes[1].set_ylabel('Poisson Deviance')
            axes[1].set_title(f'Regressor Training Curve ({target})')
            axes[1].legend()
            axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, f'learning_curve_{target}.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Learning curve saved to: {fig_path}")


# ======================================================================
# 主流程
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Hurdle model supplementary analysis')
    parser.add_argument('--hurdle_model_dir', type=str, required=True,
                        help='CB_Hurdle training output directory (contains catboost_info_* subdirectories)')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to the test set CSV file')
    parser.add_argument('--output_dir', type=str, default='Supplementary_Analysis_Results',
                        help='Directory for analysis results')
    parser.add_argument('--lgb_csv', type=str, default=None,
                        help='LightGBM training summary CSV (optional)')
    parser.add_argument('--rf_csv', type=str, default=None,
                        help='Random Forest training summary CSV (optional)')
    parser.add_argument('--train_csv', type=str, default=None,
                        help='Path to training set CSV (used with validation set for PIT calibration)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*60}")
    print(f"Hurdle Model Supplementary Analysis")
    print(f"  Model directory: {args.hurdle_model_dir}")
    print(f"  Test set:        {args.test_csv}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*60}\n")

    # --- Load test data ---
    print("[1/6] Loading test data...")
    df_test = pd.read_csv(args.test_csv)
    cols_to_drop = ['region_code', 'Unnamed: 21']
    df_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns], errors='ignore')
    if 'h3' in df_test.columns:
        df_test['h3'] = df_test['h3'].astype(str)
    df_test = fill_missing_values(df_test)
    df_test = add_feature_engineering(df_test)
    print(f"  Test set shape: {df_test.shape}")

    # --- Try to load models ---
    print("\n[2/6] Loading Hurdle models...")
    try:
        import catboost as cb
    except ImportError:
        print("ERROR: the catboost package is required. Please run: pip install catboost")
        sys.exit(1)

    classifier_rent = None
    regressor_rent = None
    classifier_return = None
    regressor_return = None

    # 在模型目录中查找 .cbm 文件
    cbm_files = []
    for root, dirs, files in os.walk(args.hurdle_model_dir):
        for f in files:
            if f.endswith('.cbm'):
                cbm_files.append(os.path.join(root, f))

    if cbm_files:
        print(f"  Found {len(cbm_files)} .cbm model files")
        for cbm_path in cbm_files:
            fname = os.path.basename(cbm_path)
            print(f"    Loading: {fname}")
            model = cb.CatBoost().load_model(cbm_path)
            # 根据文件名推断类型
            if 'rent' in fname.lower() or 'rent' in cbm_path.lower():
                if 'classifier' in fname.lower() or 'classifier' in cbm_path.lower():
                    classifier_rent = model
                else:
                    regressor_rent = model
            elif 'return' in fname.lower() or 'return' in cbm_path.lower():
                if 'classifier' in fname.lower() or 'classifier' in cbm_path.lower():
                    classifier_return = model
                else:
                    regressor_return = model
    else:
        print("  No .cbm model files found. Will try using best-iteration info from metadata...")
        print("  NOTE: For full analysis, ensure the model was trained with allow_writing_files=True")
        # 此处可扩展为通过训练日志重建模型

    if classifier_rent is None or regressor_rent is None:
        print("\n  WARNING: no complete rent model files found.")
        print("  Some analyses (test metrics, feature importance, PIT calibration, learning curves) will skip rent.")
    if classifier_return is None or regressor_return is None:
        print("  WARNING: no complete return model files found.")
        print("  Some analyses will skip return.")

    # --- 逐项执行分析 ---
    for target, clf, reg in [
        ('rent', classifier_rent, regressor_rent),
        ('return', classifier_return, regressor_return),
    ]:
        if clf is None or reg is None:
            continue

        print(f"\n{'─'*50}")
        print(f"Analysis target: {target}")
        print(f"{'─'*50}")

        # (1) Test set metrics
        print(f"\n  [{target}] Test set evaluation...")
        metrics = evaluate_test_metrics(clf, reg, df_test, target)
        pd.DataFrame([metrics]).to_csv(
            os.path.join(args.output_dir, f'test_metrics_{target}.csv'), index=False
        )
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        # (2) Feature importance
        print(f"\n  [{target}] Feature importance...")
        importance_df = extract_feature_importance(clf, reg, args.output_dir)

        # (3) Residual spatiotemporal distribution
        print(f"\n  [{target}] Residual spatiotemporal analysis...")
        plot_residual_spatiotemporal(clf, reg, df_test, target, args.output_dir)

        # (5) PIT calibration
        print(f"\n  [{target}] PIT calibration analysis...")
        try:
            plot_calibration(clf, reg, df_test, target, args.output_dir)
        except Exception as e:
            print(f"    WARNING: failed to generate PIT calibration figure: {e}")

        # (6) Learning curves
        print(f"\n  [{target}] Learning curves...")
        try:
            plot_learning_curves(clf, reg, target, args.output_dir)
        except Exception as e:
            print(f"    WARNING: failed to generate learning curves: {e}")

    # (4) Cross-model comparison
    print(f"\n{'─'*50}")
    print("Cross-model comparison")
    print(f"{'─'*50}")
    if classifier_rent and regressor_rent and classifier_return and regressor_return:
        metrics_rent = evaluate_test_metrics(classifier_rent, regressor_rent, df_test, 'rent')
        metrics_return = evaluate_test_metrics(classifier_return, regressor_return, df_test, 'return')
        benchmark_cross_model(
            metrics_rent, metrics_return, args.output_dir,
            lgb_summary_csv=args.lgb_csv,
            rf_summary_csv=args.rf_csv,
        )
    else:
        print("  Skipped: complete rent and return models required.")

    print(f"\n{'='*60}")
    print(f"Supplementary analysis complete. All results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
