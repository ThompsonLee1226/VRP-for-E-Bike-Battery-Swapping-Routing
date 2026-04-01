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

# 统一管理输入输出文件名
TRAIN_FILE = 'battery_swapping_routing_data.csv'
TEST_FILE = 'battery_swapping_routing_test_dataset.csv'
TRAINING_SCALE = [20000, 
                  None
                  ]
TRAINING_RESULTS_DIR = 'Training_Results_RF'
PREDICTION_OUTPUT_TEMPLATE = 'prediction_RF_scale_{scale}_{ts}.csv'
PROGRESS_PLOT_TEMPLATE = 'training_progress_RF_{target}_{scale}_{ts}.png'

# 统一管理训练超参数，便于集中调参
TRAIN_VALID_TEST_SIZE = 0.2               # 验证集占比
TRAIN_VALID_RANDOM_STATE = 42             # 固定随机种子，保证可复现
RF_STAGE_ESTIMATORS = list(range(50, 2001, 50)) # 分阶段树数量，用于观察收敛曲线

RF_WARM_START_PARAMS = {
    'n_estimators': RF_STAGE_ESTIMATORS[0], # 初始树数，后续逐阶段增加
    'max_depth': 18,                        # 树深度上限，控制模型复杂度
    'min_samples_leaf': 5,                  # 叶子最小样本数，防止过拟合
    'max_features': 'sqrt',                 # 每次分裂考虑特征数
    'n_jobs': -1,                           # 并行线程，-1 使用全部 CPU
    'verbose': 0,                           # 关闭底层日志
    'warm_start': True,                     # 允许在已有森林上继续加树
    'random_state': 42                      # 固定随机性
}

RF_FINAL_MODEL_PARAMS = {
    'n_estimators': RF_STAGE_ESTIMATORS[0], # 会在训练后替换成最佳树数
    'max_depth': 18,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'n_jobs': -1,
    'verbose': 0,
    'random_state': 42
}

EARLY_STOPPING_PATIENCE = 3     # 连续多少个阶段无提升则停止
EARLY_STOPPING_MIN_DELTA = 1e-4 # 认为“有提升”的最小 RMSE 降幅

# ==========================================
# 1. 数据预处理管道 
# ==========================================
def load_and_preprocess(file_path, scale=None):
    """
    加载并处理训练数据。
    :param file_path: 数据文件路径
    :param scale: 抽样规模（整数），None 表示全量
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始加载数据...")
    if scale:
        print(f"当前模式：小规模试跑，读取前 {scale} 行。")
        df = pd.read_csv(file_path, nrows=scale)
    else:
        print("当前模式：全量数据训练！(随机森林耗时通常较高)")
        df = pd.read_csv(file_path)
    
    print(f"数据加载完成，形状: {df.shape}")

    cols_to_drop = ['region_code', 'Unnamed: 21', 'datetime']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # RF 不直接支持字符串类别，先统一编码并记录映射，确保训练/测试一致
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
    将 h3 字符串编码为整数，并返回映射字典。
    """
    clean_series = series.fillna('missing').astype(str)
    categories = pd.Index(clean_series.unique())
    mapping = {cat: idx for idx, cat in enumerate(categories)}
    encoded = clean_series.map(mapping).astype(np.int32)
    return encoded, mapping

def fill_missing_values(df):
    """
    RF 训练统一缺失值策略：全部填充为 0。
    """
    for col in df.columns:
        df[col] = df[col].fillna(0)
    return df

def validate_required_columns(df, required_cols, dataset_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} 缺少必要字段: {missing}")


def get_run_output_dir(run_timestamp):
    """
    基于运行时间戳创建并返回本次训练专属输出目录。
    """
    run_output_dir = os.path.join(TRAINING_RESULTS_DIR, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir


def format_seconds(seconds):
    """
    将秒数格式化为 HH:MM:SS，便于终端展示 ETA。
    """
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def plot_training_progress(progress_df, target_name, scale_tag, run_timestamp, run_output_dir):
    """
    将 RF 分阶段训练过程中的 RMSE 变化保存为可视化图。
    """
    if progress_df.empty:
        print(f"⚠️ 未捕获到 {target_name} 的训练过程指标，跳过可视化。")
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

    print(f"📈 随机森林训练进度图已保存: {fig_path}")

# ==========================================
# 2. 核心训练函数
# ==========================================
def train_model(df, features, target_name, scale_tag='all', run_timestamp='unknown', run_output_dir='.'):
    """
    针对指定目标（rent 或 return）训练随机森林。
    通过 warm_start 分阶段增加树数量，记录训练/验证 RMSE 作为可视化进度。
    同时加入早停策略：验证集 RMSE 连续若干阶段无显著提升则提前结束。
    """
    print(f"\n{'='*40}")
    print(f"开始训练目标变量 (RF): 【{target_name}】")
    print(f"{'='*40}")

    validate_required_columns(df, features + [target_name], '训练集')

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

    print(f"[{time.strftime('%H:%M:%S')}] 正在分阶段培育森林，请关注 RMSE 变化：")
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
            f"[{target_name}] 触发早停：验证集 RMSE 连续 {EARLY_STOPPING_PATIENCE} 个阶段"
            f" 无显著下降 (min_delta={EARLY_STOPPING_MIN_DELTA})。"
        )

    # 用最佳树数重训最终模型，避免后续阶段轻微过拟合拖累最终效果
    final_model_params = RF_FINAL_MODEL_PARAMS.copy()
    final_model_params['n_estimators'] = best_n_estimators
    final_model = RandomForestRegressor(**final_model_params)
    final_model.fit(X_train, y_train)

    # 最终评估
    y_pred = final_model.predict(X_valid)
    final_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(f"\n🎉 【{target_name}】模型训练完成！最佳树数: {best_n_estimators}")
    print(f"🎯 最终验证集 RMSE: {final_rmse:.4f}")

    # 特征重要性
    importance = pd.DataFrame({
        'feature': features,
        'importance': final_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    print(f"📊 特征重要性 Top 5:")
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
    对测试集执行与训练一致的预处理。
    h3 使用训练阶段映射，未知类别编码为 -1。
    """
    cols_to_drop = ['region_code', 'Unnamed: 21', 'datetime']
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')

    if 'h3' in test_df.columns:
        test_df['h3'] = test_df['h3'].fillna('missing').astype(str).map(h3_mapping).fillna(-1).astype(np.int32)

    test_df = fill_missing_values(test_df)
    return test_df


def predict_on_test_data(models, feature_cols, test_file, output_file, h3_mapping):
    """
    使用训练好的 rent/return 模型对测试集做预测，并导出结果。
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始加载测试集: {test_file}")
    test_df = pd.read_csv(test_file)

    print(f"测试集加载完成，形状: {test_df.shape}")
    test_df = preprocess_test_data(test_df, h3_mapping)
    validate_required_columns(test_df, feature_cols, '测试集')

    X_test = test_df[feature_cols]

    result_df = pd.DataFrame(index=test_df.index)
    result_df['rent_pred'] = models['rent'].predict(X_test)
    result_df['return_pred'] = models['return'].predict(X_test)

    # 保留 ID 方便回填
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
    os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_output_dir = get_run_output_dir(run_timestamp)
    print(f"本次训练时间戳: {run_timestamp}")
    print(f"本次结果目录: {run_output_dir}")
    
    training_scales = TRAINING_SCALE
    for scale in training_scales:
        if scale is None:
            input("\n⚠️ 准备进入随机森林全量训练阶段（已启用早停策略）。按回车开始...")
            
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
        print(f"✅ 规模 {scale_tag} 的随机森林双目标训练与测试集预测全部结束！")
        print("="*50 + "\n")