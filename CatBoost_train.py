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

# 统一管理输入输出文件名
TRAIN_FILE = 'battery_swapping_routing_data.csv'
TEST_FILE = 'battery_swapping_routing_test_dataset.csv'
TRAINING_SCALE = [20000, 
                  #None
                  ]
TRAINING_RESULTS_DIR = 'Training_Results_CatBoost'
PREDICTION_OUTPUT_TEMPLATE = 'prediction_CB_scale_{scale}_{ts}.csv'
PROGRESS_PLOT_TEMPLATE = 'training_progress_CB_{target}_{scale}_{ts}.png'
USE_LOG_TARGET = True

# 统一管理训练超参数
TRAIN_VALID_TEST_SIZE = 0.2
TRAIN_VALID_RANDOM_STATE = 42
CB_CATEGORICAL_FEATURES = ['h3'] # CatBoost 原生强力支持类别特征

# CatBoost 核心参数字典
CB_PARAMS = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'learning_rate': 0.02,
    'depth': 10,                 # 对称树深度，略加深以提升拟合能力
    'l2_leaf_reg': 4.0,          # L2正则化，控制过拟合
    'random_seed': TRAIN_VALID_RANDOM_STATE,
    'task_type': 'GPU',          
    'thread_count': -1,
    'od_type': 'Iter',           # 早停类型
    'od_wait': 200               # 早停轮次
}

CB_ITERATIONS = 10000            # 最大迭代轮数
CB_LOG_EVAL_PERIOD = 50          # 终端打印周期

# ==========================================
# 1. 数据预处理管道
# ==========================================
def load_and_preprocess(file_path, scale=None):
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始加载数据...")
    if scale:
        print(f"当前模式：小规模试跑，读取前 {scale} 行。")
        df = pd.read_csv(file_path, nrows=scale)
    else:
        print(f"当前模式：全量数据训练！")
        df = pd.read_csv(file_path)
    
    print(f"数据加载完成，形状: {df.shape}")

    cols_to_drop = ['region_code', 'Unnamed: 21', 'datetime']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # CatBoost 要求类别特征为 string 或 int，避免 null
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
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos', 'temp_x_rain', 'available_power_bike_gap', 'is_rush_hour'
    ]
    
    return df, features


def add_feature_engineering(df):
    df = df.copy()

    if 'hour' in df.columns:
        hour_angle = 2 * np.pi * (df['hour'] % 24) / 24.0
        df['hour_sin'] = np.sin(hour_angle)
        df['hour_cos'] = np.cos(hour_angle)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    else:
        df['hour_sin'] = 0.0
        df['hour_cos'] = 0.0
        df['is_rush_hour'] = 0

    if 'day_of_week' in df.columns:
        dow_angle = 2 * np.pi * ((df['day_of_week'] - 1) % 7) / 7.0
        df['day_of_week_sin'] = np.sin(dow_angle)
        df['day_of_week_cos'] = np.cos(dow_angle)
    else:
        df['day_of_week_sin'] = 0.0
        df['day_of_week_cos'] = 0.0

    if 'month' in df.columns:
        month_angle = 2 * np.pi * ((df['month'] - 1) % 12) / 12.0
        df['month_sin'] = np.sin(month_angle)
        df['month_cos'] = np.cos(month_angle)
    else:
        df['month_sin'] = 0.0
        df['month_cos'] = 0.0

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

def validate_required_columns(df, required_cols, dataset_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} 缺少必要字段: {missing}")

def get_run_output_dir(run_timestamp):
    run_output_dir = os.path.join(TRAINING_RESULTS_DIR, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir

def plot_training_progress(evals_result, target_name, scale_tag, run_timestamp, run_output_dir):
    # 解析 CatBoost 的 evals_result 字典
    if not evals_result:
        return

    train_rmse = evals_result.get('learn', {}).get('RMSE', [])
    valid_rmse = evals_result.get('validation', {}).get('RMSE', [])

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
    title_suffix = 'log1p target' if USE_LOG_TARGET else 'raw target'
    plt.title(f'CatBoost Training Progress - {target_name} ({scale_tag}, {title_suffix})')
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
    print(f"\n{'='*40}")
    print(f"开始训练目标变量 (CatBoost): 【{target_name}】")
    print(f"{'='*40}")

    validate_required_columns(df, features + [target_name], '训练集')

    X = df[features]
    y = df[target_name].astype(float)
    if USE_LOG_TARGET:
        y = np.log1p(y)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TRAIN_VALID_TEST_SIZE, random_state=TRAIN_VALID_RANDOM_STATE
    )

    # 提取类别特征在特征列表中的索引，CatBoost 强制要求
    cat_features_indices = [features.index(f) for f in CB_CATEGORICAL_FEATURES if f in features]

    print(f"[{time.strftime('%H:%M:%S')}] 正在建树，请关注误差下降情况：")

    # 初始化 CatBoost 回归器
    model = cb.CatBoostRegressor(
        iterations=CB_ITERATIONS,
        cat_features=cat_features_indices,
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
    
    # 最终评估
    y_pred = model.predict(X_valid)
    if USE_LOG_TARGET:
        y_pred = np.expm1(y_pred)
        y_valid_eval = np.expm1(y_valid)
    else:
        y_valid_eval = y_valid
    y_pred = np.clip(y_pred, 0, None)
    final_rmse = np.sqrt(mean_squared_error(y_valid_eval, y_pred))
    print(f"🎯 最终验证集 RMSE: {final_rmse:.4f}")

    # 特征重要性
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.get_feature_importance()
    }).sort_values(by='importance', ascending=False)
    
    print(f"📊 特征重要性 Top 5:")
    print(importance.head(5).to_string(index=False))

    plot_training_progress(
        model.evals_result_,
        target_name=target_name,
        scale_tag=scale_tag,
        run_timestamp=run_timestamp,
        run_output_dir=run_output_dir
    )
    
    return model

def predict_on_test_data(models, feature_cols, test_file, output_file):
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
        
        rent_model = train_model(
            df, feature_cols, target_name='rent', scale_tag=scale_tag,
            run_timestamp=run_timestamp, run_output_dir=run_output_dir
        )
        
        return_model = train_model(
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
        
        print("\n" + "="*50)
        print(f"✅ 规模 {scale if scale else 'ALL'} 的 CatBoost 双目标训练全部结束！")
        print("="*50 + "\n")