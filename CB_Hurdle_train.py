import pandas as pd
import numpy as np
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 统一管理输入输出文件名
TRAIN_FILE = 'battery_swapping_routing_data.csv'
TEST_FILE = 'battery_swapping_routing_test_dataset.csv'
TRAINING_SCALE = [100000, 
                  #None
                  ]
TRAINING_RESULTS_DIR = 'Training_Results_CatBoost_Hurdle'
PREDICTION_OUTPUT_TEMPLATE = 'prediction_CB_Hurdle_scale_{scale}_{ts}.csv'
PROGRESS_PLOT_TEMPLATE = 'training_progress_CB_hurdle_{target}_{context}_{scale}_{ts}.png' 
USE_LOG_TARGET = True

# 统一管理训练超参数
TRAIN_VALID_TEST_SIZE = 0.2
TRAIN_VALID_RANDOM_STATE = 42
CB_CATEGORICAL_FEATURES = ['h3'] 

# 通用参数基础
CB_PARAMS = {
    'learning_rate': 0.03,
    'depth': 9,                  
    'l2_leaf_reg': 4.0,          
    'random_seed': TRAIN_VALID_RANDOM_STATE,
    'task_type': 'CPU',
    'thread_count': -1,
    'od_type': 'Iter',           
    'od_wait': 50                
}

# 分类器和回归器指定不同的参数
CB_CLASSIFIER_PARAMS = CB_PARAMS.copy()
CB_CLASSIFIER_PARAMS.update({
    'loss_function': 'Logloss',  # 分类用对数损失
    'eval_metric': 'AUC'         # 评估指标用 AUC
})

CB_REGRESSOR_PARAMS = CB_PARAMS.copy()
CB_REGRESSOR_PARAMS.update({
    'loss_function': 'RMSE',     # 回归用 RMSE
    'eval_metric': 'RMSE'
})

CB_ITERATIONS = 10000            
CB_LOG_EVAL_PERIOD = 50          

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
        # 高峰期规则：早上7-9点，下午17-19点
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

def get_catboost_train_dir(run_output_dir, target, context):
    train_dir = os.path.join(run_output_dir, f'catboost_info_{target}_{context}')
    os.makedirs(train_dir, exist_ok=True)
    return train_dir

def plot_training_progress(evals_result, target_name, context_name, scale_tag, run_timestamp, run_output_dir):
    if not evals_result:
        return

    train_rmse = evals_result.get('learn', {}).get('RMSE', [])
    valid_rmse = evals_result.get('validation', {}).get('RMSE', [])

    fig_path = os.path.join(
        run_output_dir,
        PROGRESS_PLOT_TEMPLATE.format(target=target_name, context=context_name, scale=scale_tag, ts=run_timestamp)
    )

    rounds = range(1, len(valid_rmse) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, train_rmse, label='Train RMSE', linewidth=1.8)
    plt.plot(rounds, valid_rmse, label='Valid RMSE', linewidth=1.8)
    plt.xlabel('Boosting Round')
    plt.ylabel('RMSE')
    title_suffix = 'log1p target' if USE_LOG_TARGET else 'raw target'
    plt.title(f'CatBoost [{context_name}] Progress - {target_name} ({scale_tag}, {title_suffix})')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()
    print(f"📈 [{context_name}] 训练进度图已保存: {fig_path}")

# ==========================================
# 2. 核心训练函数
# ==========================================
def train_model(df, features, target_name, scale_tag='all', run_timestamp='unknown', run_output_dir='.'):
    print(f"\n{'='*50}")
    print(f"🟢 开始训练零膨胀架构 (Hurdle) -> 目标:【{target_name}】")
    print(f"{'='*50}")

    validate_required_columns(df, features + [target_name], '训练集')

    split_df = df.copy()
    if 'datetime' in split_df.columns:
        split_df['_datetime_sort_key'] = pd.to_datetime(split_df['datetime'], errors='coerce')
        split_df = split_df.sort_values(by=['_datetime_sort_key']).drop(columns=['_datetime_sort_key'])

    X = split_df[features]
    y_raw = split_df[target_name].astype(float) # 保留最原始的真实值
    
    # 划分数据集
    split_idx = int(len(split_df) * (1 - TRAIN_VALID_TEST_SIZE))
    split_idx = max(1, min(split_idx, len(split_df) - 1))
    X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_raw, y_valid_raw = y_raw.iloc[:split_idx], y_raw.iloc[split_idx:]

    cat_features_indices = [features.index(f) for f in CB_CATEGORICAL_FEATURES if f in features]

    # --- 阶段一：训练分类器 (预测是否 > 0) ---
    print(f"[{time.strftime('%H:%M:%S')}] 阶段1：训练分类器 (判断是否有需求)...")
    y_train_bin = (y_train_raw > 0).astype(int)
    y_valid_bin = (y_valid_raw > 0).astype(int)

    classifier = cb.CatBoostClassifier(
        iterations=CB_ITERATIONS, cat_features=cat_features_indices,
        train_dir=get_catboost_train_dir(run_output_dir, target_name, 'Classifier'),
        allow_writing_files=True, **CB_CLASSIFIER_PARAMS
    )
    classifier.fit(X_train, y_train_bin, eval_set=(X_valid, y_valid_bin), use_best_model=True, verbose=CB_LOG_EVAL_PERIOD)

    # --- 阶段二：训练回归器 (只用 > 0 的数据) ---
    print(f"\n[{time.strftime('%H:%M:%S')}] 阶段2：训练回归器 (剔除0后拟合数量)...")
    mask_train_pos = y_train_raw > 0
    X_train_pos, y_train_pos = X_train[mask_train_pos], y_train_raw[mask_train_pos]
    
    mask_valid_pos = y_valid_raw > 0
    X_valid_pos, y_valid_pos = X_valid[mask_valid_pos], y_valid_raw[mask_valid_pos]

    # 仅对回归任务的正样本进行 Log 转换
    if USE_LOG_TARGET:
        y_train_pos = np.log1p(y_train_pos)
        y_valid_pos = np.log1p(y_valid_pos)

    regressor = cb.CatBoostRegressor(
        iterations=CB_ITERATIONS, cat_features=cat_features_indices,
        train_dir=get_catboost_train_dir(run_output_dir, target_name, 'Regressor'),
        allow_writing_files=True, **CB_REGRESSOR_PARAMS
    )
    regressor.fit(X_train_pos, y_train_pos, eval_set=(X_valid_pos, y_valid_pos), use_best_model=True, verbose=CB_LOG_EVAL_PERIOD)

    # --- 阶段三：联合评估误差 ---
    prob_valid = classifier.predict_proba(X_valid)[:, 1]
    val_valid = regressor.predict(X_valid)
    if USE_LOG_TARGET: val_valid = np.expm1(val_valid)
    
    final_pred = np.clip(prob_valid * val_valid, 0, None)
    final_rmse = np.sqrt(mean_squared_error(y_valid_raw, final_pred))
    print(f"\n🎉 【{target_name}】零膨胀模型训练完成！最终联合 RMSE: {final_rmse:.4f}")

    # 将两个模型打包返回
    return {'classifier': classifier, 'regressor': regressor}


def predict_on_test_data(models_dict, feature_cols, test_file, output_file):
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始加载测试集进行零膨胀预测: {test_file}")
    test_df = pd.read_csv(test_file)

    cols_to_drop = ['region_code', 'Unnamed: 21', 'datetime']
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')
    
    if 'h3' in test_df.columns: test_df['h3'] = test_df['h3'].astype(str)
    test_df = fill_missing_values(test_df)
    test_df = add_feature_engineering(test_df)
    validate_required_columns(test_df, feature_cols, '测试集')

    X_test = test_df[feature_cols]
    result_df = pd.DataFrame(index=test_df.index)

    # 分别对 rent 和 return 进行 概率 × 数值 的预测
    for target in ['rent', 'return']:
        prob = models_dict[target]['classifier'].predict_proba(X_test)[:, 1]
        val = models_dict[target]['regressor'].predict(X_test)
        if USE_LOG_TARGET: 
            val = np.expm1(val)
        result_df[f'{target}_pred'] = np.clip(prob * val, 0, None)

    # 拼装主键并保存
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
            input("\n⚠️ 准备进入全量 CatBoost Hurdle训练！按回车键继续...")
            
        df, feature_cols = load_and_preprocess(file_name, scale=scale)
        scale_tag = str(scale) if scale is not None else 'all'
        
        models_dict = {}

        # 依次训练 rent 和 return 的零膨胀模型
        models_dict['rent'] = train_model(
            df, feature_cols, target_name='rent', 
            scale_tag=scale_tag, run_timestamp=run_timestamp, run_output_dir=run_output_dir
        )
        
        models_dict['return'] = train_model(
            df, feature_cols, target_name='return', 
            scale_tag=scale_tag, run_timestamp=run_timestamp, run_output_dir=run_output_dir
        )

        output_file = os.path.join(run_output_dir, PREDICTION_OUTPUT_TEMPLATE.format(scale=scale_tag, ts=run_timestamp))
        predict_on_test_data(models_dict, feature_cols, TEST_FILE, output_file)
        
        print("\n" + "="*50)
        print(f"✅ 规模 {scale_tag} 的 Hurdle Model 全流程结束！")
        print("="*50 + "\n")