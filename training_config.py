"""Central training configuration.

Edit this file only when you want to change input files, split settings,
model hyperparameters, or output locations.
"""

from __future__ import annotations

import os

# Input / output
TRAIN_FILE = 'battery_swapping_routing_data_train_time70.csv'  # 训练集 CSV 文件路径
TEST_FILE = 'battery_swapping_routing_data_valid_time30.csv'  # 测试集 CSV 文件路径
TRAINING_SCALE = [20000]  # 训练读取规模列表，None 表示全量读取
TRAINING_RESULTS_DIR = 'Training_Results_CatBoost'  # 训练结果总目录
TRAINING_SUMMARY_CSV = os.path.join(TRAINING_RESULTS_DIR, 'training_summary.csv')  # 长期汇总记录 CSV 路径
PREDICTION_OUTPUT_TEMPLATE = 'prediction_CB_scale_{scale}_{ts}.csv'  # 预测结果文件名模板
PROGRESS_PLOT_TEMPLATE = 'training_progress_CB_{target}_{scale}_{ts}.png'  # 训练曲线图文件名模板

# Target transformation and evaluation space
USE_LOG_TARGET = True  # 是否对目标 y 使用 log1p 变换
REPORT_METRIC_SPACE = 'auto'  # 指标空间: auto(随USE_LOG_TARGET) | log | raw

# Data split settings
TRAIN_VALID_TEST_SIZE = 0.2  # 随机切分时验证集比例
TRAIN_VALID_RANDOM_STATE = 42  # 随机切分种子
SPLIT_MODE = 'time'  # 切分方式: random | time
TIME_SPLIT_COLUMN = 'datetime'  # 时间切分时使用的时间列名
TIME_SPLIT_RATIO = 0.8  # 时间切分时训练集比例(前80%训练，后20%验证)
TIME_SPLIT_ASCENDING = True  # 时间排序方向，True=从早到晚

# Model settings
CB_CATEGORICAL_FEATURES = ['h3']  # CatBoost 类别特征列表
CB_PARAMS = {
    'loss_function': 'RMSE',  # 训练目标函数
    'eval_metric': 'RMSE',  # 训练过程监控指标
    'learning_rate': 0.03,  # 学习率
    'depth': 9,  # 树深度
    'l2_leaf_reg': 4.0,  # L2 正则
    'random_seed': TRAIN_VALID_RANDOM_STATE,  # 模型随机种子
    'task_type': 'GPU',  # 训练设备类型
    'devices': '0:1',  # 使用的 GPU 设备编号
    'thread_count': -1,  # CPU 线程数，-1 表示自动
    'od_type': 'Iter',  # 早停策略类型
    'od_wait': 10,  # 早停等待轮次
}
CB_ITERATIONS = 1000  # 最大训练轮次
CB_LOG_EVAL_PERIOD = 5  # 每多少轮打印一次日志
