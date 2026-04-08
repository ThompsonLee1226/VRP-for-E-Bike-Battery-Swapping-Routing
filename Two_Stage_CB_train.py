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
TRAINING_SCALE = [100000, 
                  #None
                  ]
TRAINING_RESULTS_DIR = 'Training_Results_CatBoost_TwoStage'
PREDICTION_OUTPUT_TEMPLATE = 'prediction_CB_TwoStage_scale_{scale}_{ts}.csv'
PROGRESS_PLOT_TEMPLATE = 'training_progress_CB_{target}_{context}_{scale}_{ts}.png' # 增加了 context 标识
USE_LOG_TARGET = True

# 统一管理训练超参数
TRAIN_VALID_TEST_SIZE = 0.2
TRAIN_VALID_RANDOM_STATE = 42
CB_CATEGORICAL_FEATURES = ['h3'] 

# CatBoost 核心参数字典
CB_PARAMS = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'learning_rate': 0.03,
    'depth': 9,                  
    'l2_leaf_reg': 4.0,          
    'random_seed': TRAIN_VALID_RANDOM_STATE,
    'task_type': 'CPU',
    #'devices': '0:1',        
    'thread_count': -1,
    'od_type': 'Iter',           
    'od_wait': 50                
}

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
# 2. 核心训练函数 (支持分场景)
# ==========================================
def train_model(df, features, target_name, context_name, scale_tag='all', run_timestamp='unknown', run_output_dir='.'):
    """
    context_name: 场景名称，例如 'Peak' (高峰期) 或 'OffPeak' (非高峰期)
    """
    print(f"\n{'-'*50}")
    print(f"🟢 开始训练两阶段模型 -> 场景:【{context_name}】| 目标:【{target_name}】")
    print(f"当前场景数据量: {len(df)} 行")
    print(f"{'-'*50}")

    if len(df) == 0:
        print(f"⚠️ 警告: 场景 {context_name} 没有数据，跳过训练！")
        return None

    validate_required_columns(df, features + [target_name], f'{context_name}训练集')

    split_df = df.copy()
    if 'datetime' in split_df.columns:
        split_df['_datetime_sort_key'] = pd.to_datetime(split_df['datetime'], errors='coerce')
        split_df = split_df.sort_values(by=['_datetime_sort_key']).drop(columns=['_datetime_sort_key'])

    X = split_df[features]
    y = split_df[target_name].astype(float)
    if USE_LOG_TARGET:
        y = np.log1p(y)

    split_idx = int(len(split_df) * (1 - TRAIN_VALID_TEST_SIZE))
    split_idx = max(1, min(split_idx, len(split_df) - 1))
    X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

    cat_features_indices = [features.index(f) for f in CB_CATEGORICAL_FEATURES if f in features]
    catboost_train_dir = get_catboost_train_dir(run_output_dir, target_name, context_name)

    print(f"[{time.strftime('%H:%M:%S')}] 正在建树，请关注误差下降情况：")

    model = cb.CatBoostRegressor(
        iterations=CB_ITERATIONS,
        cat_features=cat_features_indices,
        train_dir=catboost_train_dir,
        allow_writing_files=True,
        **CB_PARAMS
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        verbose=CB_LOG_EVAL_PERIOD
    )

    best_iter = model.get_best_iteration()
    print(f"\n🎉 【{context_name} - {target_name}】训练完成！最优迭代次数: {best_iter}")
    
    y_pred = model.predict(X_valid)
    if USE_LOG_TARGET:
        y_pred = np.expm1(y_pred)
        y_valid_eval = np.expm1(y_valid)
    else:
        y_valid_eval = y_valid
        
    y_pred = np.clip(y_pred, 0, None)
    final_rmse = np.sqrt(mean_squared_error(y_valid_eval, y_pred))
    print(f"🎯 [{context_name}] 最终验证集 RMSE: {final_rmse:.4f}")

    importance = pd.DataFrame({
        'feature': features,
        'importance': model.get_feature_importance()
    }).sort_values(by='importance', ascending=False)
    
    print(f"📊 特征重要性 Top 5:")
    print(importance.head(5).to_string(index=False))

    plot_training_progress(
        model.evals_result_,
        target_name=target_name,
        context_name=context_name,
        scale_tag=scale_tag,
        run_timestamp=run_timestamp,
        run_output_dir=run_output_dir
    )
    
    return model


def predict_on_test_data_two_stage(models_dict, feature_cols, test_file, output_file):
    """
    两阶段预测：自动拆分测试集，使用对应模型预测后，按原顺序拼接
    models_dict 格式: {
        'rent_peak': model1, 'rent_offpeak': model2,
        'return_peak': model3, 'return_offpeak': model4
    }
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始加载测试集进行两阶段预测: {test_file}")
    test_df = pd.read_csv(test_file)
    print(f"测试集加载完成，形状: {test_df.shape}")

    if test_df.empty:
        empty_result = pd.DataFrame(columns=['rent_pred', 'return_pred'])
        empty_result.to_csv(output_file, index=False)
        print(f"⚠️ 测试集为空，已输出空预测文件: {output_file}")
        return

    cols_to_drop = ['region_code', 'Unnamed: 21', 'datetime']
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')
    
    if 'h3' in test_df.columns:
        test_df['h3'] = test_df['h3'].astype(str)
    test_df = fill_missing_values(test_df)
    test_df = add_feature_engineering(test_df)
    validate_required_columns(test_df, feature_cols, '测试集')

    # 保留原始索引以便后续拼接
    test_df['original_index'] = test_df.index
    
    # 物理拆分测试集
    mask_peak = test_df['is_rush_hour'] == 1
    test_peak = test_df[mask_peak]
    test_offpeak = test_df[~mask_peak]

    result_pieces = []

    def predict_with_fallback(primary_key, fallback_key, X_part):
        primary_model = models_dict.get(primary_key)
        fallback_model = models_dict.get(fallback_key)

        if primary_model is not None:
            return primary_model.predict(X_part)

        if fallback_model is not None:
            print(f"⚠️ 模型 {primary_key} 不可用，回退使用 {fallback_key} 进行预测。")
            return fallback_model.predict(X_part)

        raise RuntimeError(f"主模型 {primary_key} 与回退模型 {fallback_key} 均不可用，无法完成预测。")

    # --- 高峰期预测 ---
    if len(test_peak) > 0:
        res_peak = pd.DataFrame(index=test_peak.index)
        res_peak['original_index'] = test_peak['original_index']
        
        pred_rent = predict_with_fallback('rent_peak', 'rent_offpeak', test_peak[feature_cols])
        pred_return = predict_with_fallback('return_peak', 'return_offpeak', test_peak[feature_cols])
        
        if USE_LOG_TARGET:
            pred_rent = np.expm1(pred_rent)
            pred_return = np.expm1(pred_return)
            
        res_peak['rent_pred'] = np.clip(pred_rent, 0, None)
        res_peak['return_pred'] = np.clip(pred_return, 0, None)
        
        # 保留 ID
        for id_col in ['id', 'station_id', 'h3']:
            if id_col in test_peak.columns:
                res_peak[id_col] = test_peak[id_col]
                break
        result_pieces.append(res_peak)

    # --- 非高峰期预测 ---
    if len(test_offpeak) > 0:
        res_offpeak = pd.DataFrame(index=test_offpeak.index)
        res_offpeak['original_index'] = test_offpeak['original_index']
        
        pred_rent = predict_with_fallback('rent_offpeak', 'rent_peak', test_offpeak[feature_cols])
        pred_return = predict_with_fallback('return_offpeak', 'return_peak', test_offpeak[feature_cols])
        
        if USE_LOG_TARGET:
            pred_rent = np.expm1(pred_rent)
            pred_return = np.expm1(pred_return)
            
        res_offpeak['rent_pred'] = np.clip(pred_rent, 0, None)
        res_offpeak['return_pred'] = np.clip(pred_return, 0, None)
        
        # 保留 ID
        for id_col in ['id', 'station_id', 'h3']:
            if id_col in test_offpeak.columns:
                res_offpeak[id_col] = test_offpeak[id_col]
                break
        result_pieces.append(res_offpeak)

    # 合并并恢复原顺序
    if not result_pieces:
        empty_result = pd.DataFrame(columns=['rent_pred', 'return_pred'])
        empty_result.to_csv(output_file, index=False)
        print(f"⚠️ 测试集拆分后无可预测样本，已输出空预测文件: {output_file}")
        return

    final_result = pd.concat(result_pieces).sort_values('original_index').drop(columns=['original_index'])
    
    # 调整列顺序，让 ID 列在最前
    cols = final_result.columns.tolist()
    id_cols_present = [c for c in ['id', 'station_id', 'h3'] if c in cols]
    other_cols = [c for c in cols if c not in id_cols_present]
    final_result = final_result[id_cols_present + other_cols]

    final_result.to_csv(output_file, index=False)
    print(f"✅ 两阶段测试集预测合并完成，结果已保存到: {output_file}")


# ==========================================
# 3. 任务执行流
# ==========================================
if __name__ == "__main__":
    file_name = TRAIN_FILE
    os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_output_dir = get_run_output_dir(run_timestamp)
    print(f"本次两阶段训练时间戳: {run_timestamp}")
    print(f"本次结果目录: {run_output_dir}")
    
    for scale in TRAINING_SCALE:
        if scale is None:
            input("\n⚠️ 准备进入全量 CatBoost 两阶段训练！将独立训练高峰/非高峰模型。按回车键继续...")
            
        df, feature_cols = load_and_preprocess(file_name, scale=scale)
        scale_tag = str(scale) if scale is not None else 'all'
        
        print("\n" + "="*50)
        print("🛠️ 正在执行物理拆分：高峰期 vs 非高峰期 (两阶段策略)")
        
        # 物理拆分数据集
        mask_rush = df['is_rush_hour'] == 1
        df_peak = df[mask_rush]
        df_offpeak = df[~mask_rush]
        print(f"✅ 拆分完毕 -> 高峰期样本数: {len(df_peak)} | 非高峰期样本数: {len(df_offpeak)}")
        print("="*50)

        models_dict = {}

        # 第一部分：训练高峰期模型 (Peak)
        models_dict['rent_peak'] = train_model(
            df_peak, feature_cols, target_name='rent', context_name='Peak', 
            scale_tag=scale_tag, run_timestamp=run_timestamp, run_output_dir=run_output_dir
        )
        models_dict['return_peak'] = train_model(
            df_peak, feature_cols, target_name='return', context_name='Peak', 
            scale_tag=scale_tag, run_timestamp=run_timestamp, run_output_dir=run_output_dir
        )

        # 第二部分：训练非高峰期模型 (OffPeak)
        models_dict['rent_offpeak'] = train_model(
            df_offpeak, feature_cols, target_name='rent', context_name='OffPeak', 
            scale_tag=scale_tag, run_timestamp=run_timestamp, run_output_dir=run_output_dir
        )
        models_dict['return_offpeak'] = train_model(
            df_offpeak, feature_cols, target_name='return', context_name='OffPeak', 
            scale_tag=scale_tag, run_timestamp=run_timestamp, run_output_dir=run_output_dir
        )

        # 执行分发合并预测
        output_file = os.path.join(
            run_output_dir,
            PREDICTION_OUTPUT_TEMPLATE.format(scale=scale_tag, ts=run_timestamp)
        )
        
        predict_on_test_data_two_stage(
            models_dict=models_dict, feature_cols=feature_cols, 
            test_file=TEST_FILE, output_file=output_file
        )
        
        print("\n" + "="*50)
        print(f"✅ 规模 {scale_tag} 的 CatBoost 场景感知(Two-Stage)全流程结束！")
        print("="*50 + "\n")