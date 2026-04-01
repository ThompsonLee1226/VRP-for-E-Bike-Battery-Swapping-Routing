import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore') # 忽略一些常规警告

# 统一管理输入输出文件名，方便后续修改与复用
TRAIN_FILE = 'battery_swapping_routing_data.csv'
TEST_FILE = 'battery_swapping_routing_test_dataset.csv'
TRAINING_SCALE = [20000, 
                  None
                  ]

TRAINING_RESULTS_DIR = 'Training_Results_LightGBM'
PREDICTION_OUTPUT_TEMPLATE = 'prediction_scale_{scale}_{ts}.csv'
PROGRESS_PLOT_DIR = TRAINING_RESULTS_DIR
PROGRESS_PLOT_TEMPLATE = 'training_progress_{target}_{scale}_{ts}.png'

# ==========================================
# 1. 工业级数据预处理管道 (Data Pipeline)
# ==========================================
def load_and_preprocess(file_path, scale=None):
    """
    加载并处理数据
    :param file_path: 数据文件路径
    :param scale: 抽样规模（整数），如果不填(None)则读取全量数据
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始加载数据...")
    if scale:
        print(f"当前模式：小规模试跑，读取前 {scale} 行。")
        df = pd.read_csv(file_path, nrows=scale)
    else:
        print(f"当前模式：全量数据训练！(这可能需要占用较多内存)")
        df = pd.read_csv(file_path)
    
    print(f"数据加载完成，形状: {df.shape}")

    # A. 剔除在 EDA 报告中被判定为无用/低信息的列
    cols_to_drop = ['region_code', 'Unnamed: 21', 'datetime']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # B. 类别特征转换（LightGBM 的绝技，不需要做 One-Hot 编码）
    if 'h3' in df.columns:
        df['h3'] = df['h3'].astype('category')
        
    # C. 缺失值处理：分类列与数值列分开填充，避免 category 列 fillna(0) 报错
    df = fill_missing_values(df)

    # D. 定义特征集合（根据你的 EDA 优先级报告）
    features = [
        'h3', 'temperature', 'wind_level', 'rain_level', 
        'month', 'day_of_week', 'is_weekend', 'hour',
        'rent_mean_7d', 'return_mean_7d', 'lag_nb_rent', 'lag_nb_return',
        'normal_power_bike_count', 'soon_low_power_bike_count', 'low_power_bike_count',
        'latitude', 'longitude' # 空间异质性特征
    ]
    
    return df, features


def fill_missing_values(df):
    """
    统一处理缺失值：
    - 类别列补 "missing"（若类别集中不存在，先新增该类别）
    - 非类别列补 0
    这样可避免 pandas 对 category 列直接填 0 导致的类型错误。
    """
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            if 'missing' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(['missing'])
            df[col] = df[col].fillna('missing')
        else:
            df[col] = df[col].fillna(0)
    return df


def validate_required_columns(df, required_cols, dataset_name):
    """
    检查数据集中是否包含必需列。
    如果有缺失列，直接抛出错误，避免训练阶段才报错。
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} 缺少必要字段: {missing}. "
            f"当前可用字段数: {len(df.columns)}"
        )


def plot_training_progress(eval_results, target_name, scale_tag, run_timestamp):
    """
    将训练过程中的 RMSE 变化可视化并保存为图片。
    目的：快速观察是否收敛、是否过拟合（训练集下降而验证集不降）。
    """
    # LightGBM 评估结果结构示例：
    # eval_results['训练集']['rmse'] -> 每一轮训练集 RMSE 列表
    # eval_results['验证集']['rmse'] -> 每一轮验证集 RMSE 列表
    train_rmse = eval_results.get('训练集', {}).get('rmse', [])
    valid_rmse = eval_results.get('验证集', {}).get('rmse', [])

    if not train_rmse or not valid_rmse:
        print(f"⚠️ 未捕获到 {target_name} 的训练过程指标，跳过可视化。")
        return

    os.makedirs(PROGRESS_PLOT_DIR, exist_ok=True)
    fig_path = os.path.join(
        PROGRESS_PLOT_DIR,
        PROGRESS_PLOT_TEMPLATE.format(target=target_name, scale=scale_tag, ts=run_timestamp)
    )

    rounds = range(1, len(valid_rmse) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, train_rmse, label='Train RMSE', linewidth=1.8)
    plt.plot(rounds, valid_rmse, label='Valid RMSE', linewidth=1.8)
    plt.xlabel('Boosting Round')
    plt.ylabel('RMSE')
    plt.title(f'Training Progress - {target_name} ({scale_tag})')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()

    print(f"📈 训练进度图已保存: {fig_path}")


def format_seconds(seconds):
    """
    将秒数格式化为 HH:MM:SS，便于在终端展示 ETA。
    """
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def create_progress_bar_callback(target_name, width=30, refresh_every=5):
    """
    LightGBM 训练进度条回调：在终端动态显示
    - 当前迭代进度
    - 已耗时
    - 预计剩余时间（ETA）
    """
    start_time = time.time()

    def _callback(env):
        current_iter = env.iteration + 1
        total_iter = env.end_iteration

        # 控制刷新频率，避免终端刷屏；最后一轮强制刷新
        if (current_iter % refresh_every != 0) and (current_iter != total_iter):
            return

        progress = current_iter / total_iter if total_iter else 0
        filled = int(width * progress)
        bar = "#" * filled + "-" * (width - filled)

        elapsed = time.time() - start_time
        avg_per_iter = elapsed / current_iter if current_iter else 0
        eta = avg_per_iter * max(total_iter - current_iter, 0)

        msg = (
            f"\r[{target_name}] [{bar}] "
            f"{current_iter}/{total_iter} "
            f"elapsed {format_seconds(elapsed)} "
            f"ETA {format_seconds(eta)}"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

        if current_iter == total_iter:
            sys.stdout.write("\n")

    _callback.order = 15
    return _callback

# ==========================================
# 2. 核心训练函数 (Training Engine)
# ==========================================
def train_model(df, features, target_name, scale_tag='all', run_timestamp='unknown'):
    """
    针对指定目标（rent 或 return）训练 LightGBM
    """
    print(f"\n{'='*40}")
    print(f"开始训练目标变量: 【{target_name}】")
    print(f"{'='*40}")

    # 训练前先做字段校验，给出更早、更清晰的报错信息
    validate_required_columns(df, features + [target_name], '训练集')

    X = df[features]
    y = df[target_name]

    # 按时间顺序或随机拆分（这里使用随机拆分，保留20%作为验证集）
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # 封装成 LightGBM 专用的 Dataset 格式（能极大降低内存占用，提升速度）
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=['h3'])
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    # 定义模型超参数
    params = {
        'objective': 'regression',      # 回归任务
        'metric': 'rmse',               # 评估指标：均方根误差
        'boosting_type': 'gbdt',        # 传统的梯度提升树
        'learning_rate': 0.05,          # 学习率：步子迈得小一点，学得更稳
        'num_leaves': 63,               # 叶子节点数：越大越容易过拟合，但学得越深 (默认31)
        'feature_fraction': 0.8,        # 每次建树随机抽取80%特征 (防过拟合)
        'n_jobs': -1,                   # 使用全部 CPU 核心
        'verbose': -1                   # 关闭框架层面的冗余日志
        # 'device': 'gpu'               # 如果你有配置好的GPU环境，取消这行注释可以起飞
    }

    print(f"[{time.strftime('%H:%M:%S')}] 正在建树，请关注下方误差下降情况：")

    # 用于保存每一轮评估指标，后续可视化训练进度
    eval_results = {}
    
    # 训练模型，并设置回调函数来观察进度
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000, # 最大迭代次数（最多建1000棵树）
        valid_sets=[train_data, valid_data],
        valid_names=['训练集', '验证集'],
        callbacks=[
            create_progress_bar_callback(target_name=target_name, refresh_every=10), # 每10轮刷新一次进度条
            lgb.record_evaluation(eval_results),
            lgb.log_evaluation(period=50), # 每 50 棵树打印一次进度！
            lgb.early_stopping(stopping_rounds=50) # 如果验证集误差 50 轮不下降，立刻提前停止！
        ]
    )

    # 若触发提前停止，补打一行最终状态，避免误解为“还没跑完”。
    if model.best_iteration < 1000:
        print(f"[{target_name}] 提前停止于第 {model.best_iteration} 轮（已启用 early_stopping）。")

    # 最终评估
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    final_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(f"\n🎉 【{target_name}】模型训练完成！最优迭代次数: {model.best_iteration}")
    print(f"🎯 最终验证集 RMSE: {final_rmse:.4f}")

    # 提取特征重要性，便于写报告
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain') # 使用 gain (信息增益) 衡量
    }).sort_values(by='importance', ascending=False)
    
    print(f"📊 特征重要性 Top 5:")
    print(importance.head(5).to_string(index=False))

    # 生成并保存训练过程曲线图
    plot_training_progress(
        eval_results,
        target_name=target_name,
        scale_tag=scale_tag,
        run_timestamp=run_timestamp
    )
    
    return model


def predict_on_test_data(models, feature_cols, test_file, output_file):
    
    """
    使用训练好的 rent/return 模型对测试集做预测，并导出结果。
    :param models: {'rent': rent_model, 'return': return_model}
    :param feature_cols: 训练时使用的特征列
    :param test_file: 测试集 CSV 路径
    :param output_file: 预测结果输出路径
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始加载测试集: {test_file}")
    test_df = pd.read_csv(test_file)
    print(f"测试集加载完成，形状: {test_df.shape}")

    # 对测试集做与训练一致的预处理，确保特征工程对齐
    cols_to_drop = ['region_code', 'Unnamed: 21', 'datetime']
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], errors='ignore')
    if 'h3' in test_df.columns:
        test_df['h3'] = test_df['h3'].astype('category')
    test_df = fill_missing_values(test_df)

    validate_required_columns(test_df, feature_cols, '测试集')
    X_test = test_df[feature_cols]

    # 同时输出两个目标的预测结果
    result_df = pd.DataFrame(index=test_df.index)
    result_df['rent_pred'] = models['rent'].predict(X_test, num_iteration=models['rent'].best_iteration)
    result_df['return_pred'] = models['return'].predict(X_test, num_iteration=models['return'].best_iteration)

    # 尽量保留一个可追踪主键，便于后续和原测试集做对齐
    for id_col in ['id', 'station_id', 'h3']:
        if id_col in test_df.columns:
            result_df.insert(0, id_col, test_df[id_col].values)
            break

    result_df.to_csv(output_file, index=False)
    print(f"✅ 测试集预测完成，结果已保存到: {output_file}")

# ==========================================
# 3. 任务执行流 (Main Pipeline)
# ==========================================
if __name__ == "__main__":
    file_name = TRAIN_FILE
    os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    print(f"本次训练时间戳: {run_timestamp}")
    
    # 规模设置
    training_scales = TRAINING_SCALE 
    
    for scale in training_scales:
        if scale is None:
            input("\n⚠️ 准备进入全量数据训练阶段。请确保服务器内存充足！按回车键 (Enter) 继续...")
            
        # 1. 获取数据
        df, feature_cols = load_and_preprocess(file_name, scale=scale)
        scale_tag = str(scale) if scale is not None else 'all'
        
        # 2. 训练“流出率”模型 (rent)
        rent_model = train_model(
            df,
            feature_cols,
            target_name='rent',
            scale_tag=scale_tag,
            run_timestamp=run_timestamp
        )
        
        # 3. 训练“流入率”模型 (return)
        return_model = train_model(
            df,
            feature_cols,
            target_name='return',
            scale_tag=scale_tag,
            run_timestamp=run_timestamp
        )

        # 4. 对测试集进行双目标预测，并导出结果文件
        output_file = os.path.join(
            TRAINING_RESULTS_DIR,
            PREDICTION_OUTPUT_TEMPLATE.format(
                scale=scale if scale is not None else 'all',
                ts=run_timestamp
            )
        )
        predict_on_test_data(
            models={'rent': rent_model, 'return': return_model},
            feature_cols=feature_cols,
            test_file=TEST_FILE,
            output_file=output_file
        )
        
        print("\n" + "="*50)
        print(f"✅ 规模 {scale if scale else 'ALL'} 的双目标训练与测试集预测全部结束！")
        print("="*50 + "\n")