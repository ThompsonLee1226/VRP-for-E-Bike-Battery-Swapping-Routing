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
    import training_config as cfg
except ImportError:
    cfg = None

from training_summary_manager import append_row as append_summary_row
from training_summary_manager import build_run_summary_row, format_top_features


def cfg_value(name, default):
    """Read a config value from training_config.py with a fallback default."""
    if cfg is None:
        return default
    return getattr(cfg, name, default)

# 统一管理输入输出文件名
TRAIN_FILE = cfg_value('TRAIN_FILE', 'battery_swapping_routing_data_train_time70.csv')
TEST_FILE = cfg_value('TEST_FILE', 'battery_swapping_routing_test_dataset.csv')
TRAINING_SCALE = cfg_value('TRAINING_SCALE', [20000])
TRAINING_RESULTS_DIR = cfg_value('TRAINING_RESULTS_DIR', 'Training_Results_CatBoost')
TRAINING_SUMMARY_CSV = cfg_value('TRAINING_SUMMARY_CSV', os.path.join(TRAINING_RESULTS_DIR, 'training_summary.csv'))
PREDICTION_OUTPUT_TEMPLATE = cfg_value('PREDICTION_OUTPUT_TEMPLATE', 'prediction_CB_scale_{scale}_{ts}.csv')
PROGRESS_PLOT_TEMPLATE = cfg_value('PROGRESS_PLOT_TEMPLATE', 'training_progress_CB_{target}_{scale}_{ts}.png')
USE_LOG_TARGET = cfg_value('USE_LOG_TARGET', True)           # 是否对目标变量进行 log1p 变换，影响训练目标空间和评测空间
REPORT_METRIC_SPACE = cfg_value('REPORT_METRIC_SPACE', 'auto')

# 统一管理训练超参数
TRAIN_VALID_TEST_SIZE = cfg_value('TRAIN_VALID_TEST_SIZE', 0.2)
TRAIN_VALID_RANDOM_STATE = cfg_value('TRAIN_VALID_RANDOM_STATE', 42)
SPLIT_MODE = cfg_value('SPLIT_MODE', 'random')
TIME_SPLIT_COLUMN = cfg_value('TIME_SPLIT_COLUMN', 'datetime')
TIME_SPLIT_RATIO = cfg_value('TIME_SPLIT_RATIO', 0.7)
TIME_SPLIT_ASCENDING = cfg_value('TIME_SPLIT_ASCENDING', True)
CB_CATEGORICAL_FEATURES = cfg_value('CB_CATEGORICAL_FEATURES', ['h3']) # CatBoost 原生强力支持类别特征


CB_PARAMS = cfg_value('CB_PARAMS', {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'learning_rate': 0.03,
    'depth': 9,                  # 对称树深度
    'l2_leaf_reg': 4.0,          # L2正则化
    'random_seed': TRAIN_VALID_RANDOM_STATE,
    'task_type': 'GPU',
    'devices': '0:1',         
    'thread_count': -1,
    'od_type': 'Iter',           # 早停类型
    'od_wait': 10                # 早停轮次
})

CB_ITERATIONS = cfg_value('CB_ITERATIONS', 1000)             # 最大迭代轮数
CB_LOG_EVAL_PERIOD = cfg_value('CB_LOG_EVAL_PERIOD', 5)           # 终端打印周期


def resolve_report_metric_space():
    """Resolve final metric reporting space to either 'log' or 'raw'."""
    if REPORT_METRIC_SPACE == 'auto':
        return 'log' if USE_LOG_TARGET else 'raw'
    if REPORT_METRIC_SPACE not in {'log', 'raw'}:
        raise ValueError("REPORT_METRIC_SPACE 仅支持: 'auto' | 'log' | 'raw'")
    if REPORT_METRIC_SPACE == 'log' and not USE_LOG_TARGET:
        raise ValueError("USE_LOG_TARGET=False 时不能使用 log 评测空间")
    return REPORT_METRIC_SPACE


def rmse_value(y_true, y_pred):
    """Compute RMSE between ground truth and predictions."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def split_train_valid(df, features, y, y_raw):
    """Split data into train/valid sets using random or time-based strategy."""
    if SPLIT_MODE == 'time':
        if TIME_SPLIT_COLUMN not in df.columns:
            raise ValueError(f"启用时间切分时，训练集缺少字段: {TIME_SPLIT_COLUMN}")

        dt = pd.to_datetime(df[TIME_SPLIT_COLUMN], errors='coerce')
        if dt.isna().all():
            raise ValueError(f"字段 {TIME_SPLIT_COLUMN} 无法解析为时间，无法按时间切分")

        fallback = pd.Timestamp.max if TIME_SPLIT_ASCENDING else pd.Timestamp.min
        ordered_idx = dt.fillna(fallback).sort_values(ascending=TIME_SPLIT_ASCENDING).index
        split_pos = int(len(ordered_idx) * TIME_SPLIT_RATIO)
        if split_pos <= 0 or split_pos >= len(ordered_idx):
            raise ValueError("时间切分失败: 请检查 TIME_SPLIT_RATIO 是否合理")

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
            "🕒 时间切分完成: "
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
# 1. 数据预处理管道
# ==========================================
def load_and_preprocess(file_path, scale=None):
    """Load CSV, clean basic columns, fill missing values, and build feature list."""
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始加载数据...")
    if scale:
        print(f"当前模式：小规模试跑，读取前 {scale} 行。")
        df = pd.read_csv(file_path, nrows=scale)
    else:
        print(f"当前模式：全量数据训练！")
        df = pd.read_csv(file_path)
    
    print(f"数据加载完成，形状: {df.shape}")

    # 保留 datetime 供时间切分使用；不会进入特征列表。
    cols_to_drop = ['region_code', 'Unnamed: 21']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # CatBoost 要求类别特征为 string 或 int，避免 null
    if 'h3' in df.columns:
        df['h3'] = df['h3'].astype(str) # h3有可能是无效信息
        
    df = fill_missing_values(df)
    df = add_feature_engineering(df)

    features = [
        'h3', 'temperature', 'wind_level', 'rain_level', 
        'month', 'day_of_week', 'is_weekend', 'hour',
        'rent_mean_7d', 'return_mean_7d', 'lag_nb_rent', 'lag_nb_return',
        'normal_power_bike_count', 'soon_low_power_bike_count', 'low_power_bike_count',
        'latitude', 'longitude',
        # 树模型直接学习时间离散变量，无需周期分解特征
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
        raise ValueError(f"{dataset_name} 缺少必要字段: {missing}")

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

    print(f"📈 训练进度图已保存: {fig_path}")

# ==========================================
# 2. 核心训练函数
# ==========================================
def train_model(df, features, target_name, scale_tag='all', run_timestamp='unknown', run_output_dir='.'):
    """Train a CatBoost model for one target and return model + summary dict."""
    print(f"\n{'='*40}")
    print(f"开始训练目标变量 (CatBoost): 【{target_name}】")
    print(f"{'='*40}")

    validate_required_columns(df, features + [target_name], '训练集')

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

    # 提取类别特征在特征列表中的索引，CatBoost 强制要求
    cat_features_indices = [features.index(f) for f in CB_CATEGORICAL_FEATURES if f in features]
    catboost_train_dir = get_catboost_train_dir(run_output_dir)

    print(f"[{time.strftime('%H:%M:%S')}] 正在建树，请关注误差下降情况：")

    # 初始化 CatBoost 回归器
    model = cb.CatBoostRegressor(
        iterations=CB_ITERATIONS,
        cat_features=cat_features_indices,
        train_dir=catboost_train_dir,
        allow_writing_files=True,
        **CB_PARAMS
    )

    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        verbose=CB_LOG_EVAL_PERIOD
    )

    best_iter = model.get_best_iteration()
    print(f"\n🎉 【{target_name}】CatBoost 模型训练完成！最优迭代次数: {best_iter}")
    print(f"🧭 统一评测空间: {report_space}")
    print(f"🧭 训练目标函数(loss_function): {CB_PARAMS['loss_function']}，目标空间: {target_space_desc}")
    print(f"🧭 训练监控指标(eval_metric): {CB_PARAMS['eval_metric']}，目标空间: {target_space_desc}")

    best_score = model.get_best_score()
    valid_rmse_objective = best_score.get('validation', {}).get('RMSE', None)

    if report_space == 'log':
        train_rmse_curve = model.evals_result_.get('learn', {}).get('RMSE', [])
        valid_rmse_curve = model.evals_result_.get('validation', {}).get('RMSE', [])
        y_pred_for_report = model.predict(X_valid)
        final_rmse = rmse_value(y_valid, y_pred_for_report)
        if valid_rmse_objective is not None:
            print(f"📏 验证集最优RMSE(log空间): {valid_rmse_objective:.4f}")
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
            print(f"📏 验证集最优RMSE(raw空间): {valid_rmse_curve[best_iter_for_curve]:.4f}")
        if valid_rmse_objective is not None:
            print(f"📎 参考: CatBoost原生日志RMSE(目标空间): {valid_rmse_objective:.4f}")
    
    print(f"🎯 最终验证集RMSE({report_space}空间): {final_rmse:.4f}")

    training_seconds = time.time() - train_start_time

    # 特征重要性
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.get_feature_importance()
    }).sort_values(by='importance', ascending=False)
    top_features = format_top_features(importance)
    
    print(f"📊 特征重要性 Top 5:")
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
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始加载测试集: {test_file}")
    test_df = pd.read_csv(test_file)
    print(f"测试集加载完成，形状: {test_df.shape}")

    cols_to_drop = ['region_code', 'Unnamed: 21', 'datetime']
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')
    
    if 'h3' in test_df.columns:
        test_df['h3'] = test_df['h3'].astype(str)
    test_df = fill_missing_values(test_df)
    test_df = add_feature_engineering(test_df)

    validate_required_columns(test_df, feature_cols, '测试集')
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
    print(f"✅ 测试集预测完成，结果已保存到: {output_file}")

# ==========================================
# 3. 任务执行流
# ==========================================
if __name__ == "__main__":
    file_name = TRAIN_FILE
    os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_output_dir = get_run_output_dir(run_timestamp)
    print(f"本次训练时间戳: {run_timestamp}")
    print(f"本次结果目录: {run_output_dir}")
    
    for scale in TRAINING_SCALE:
        if scale is None:
            input("\n⚠️ 准备进入全量 CatBoost 训练阶段！按回车键 (Enter) 继续...")
            
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
        print(f"🗂️ 本次完整运行摘要已追加到: {TRAINING_SUMMARY_CSV}")
        
        print("\n" + "="*50)
        print(f"✅ 规模 {scale if scale else 'ALL'} 的 CatBoost 双目标训练全部结束！")
        print("="*50 + "\n")