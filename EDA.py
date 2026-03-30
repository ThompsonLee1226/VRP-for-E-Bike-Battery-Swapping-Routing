import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
jg
# 设置绘图风格（本科生进阶：使用更现代化的图表风格）
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)

def load_and_preprocess(file_path: str) -> pd.DataFrame:
    """加载数据并进行基础预处理"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 确保时间列格式正确
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 将分类变量转换为更有意义的字符，方便画图图例展示
    df['day_type'] = df['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
    
    print(f"Dataset shape: {df.shape}")
    return df

def plot_temporal_interaction(df: pd.DataFrame, plot_dir: Path):
    """分析时间潮汐效应与工作日/周末的交互作用"""
    print("Plotting temporal interaction effects...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 流出率 (Rent) 的时间分布
    sns.lineplot(data=df, x='hour', y='rent', hue='day_type', 
                 estimator='mean', errorbar=('ci', 95), marker='o', ax=axes[0])
    axes[0].set_title('Average Hourly Rent (Outflow) by Day Type')
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_ylabel('Rent Volume')

    # 流入率 (Return) 的时间分布
    sns.lineplot(data=df, x='hour', y='return', hue='day_type', 
                 estimator='mean', errorbar=('ci', 95), marker='o', ax=axes[1])
    axes[1].set_title('Average Hourly Return (Inflow) by Day Type')
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].set_ylabel('Return Volume')
    
    plt.tight_layout()
    plt.savefig(plot_dir / "temporal_interaction_effects.png", dpi=300)
    plt.close()

def plot_environmental_impact(df: pd.DataFrame, plot_dir: Path):
    """分析环境因素（天气、温度）对骑行需求的影响"""
    print("Plotting environmental impacts...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 气温与单车借出量的关系 (控制小时数，避免时间混淆变量)
    # 比如我们只看白天的核心时段 (8:00 - 20:00)
    daytime_df = df[(df['hour'] >= 8) & (df['hour'] <= 20)]
    
    sns.scatterplot(data=daytime_df, x='temperature', y='rent', 
                    alpha=0.3, edgecolor=None, ax=axes[0])
    sns.regplot(data=daytime_df, x='temperature', y='rent', 
                scatter=False, color='red', ax=axes[0])
    axes[0].set_title('Temperature vs. Rent Volume (8:00-20:00)')
    
    # 降雨量等级与借出量的关系
    sns.boxplot(data=df, x='rain_level', y='rent', palette="Blues", ax=axes[1])
    axes[1].set_title('Rain Level Impact on Rent Volume')
    
    plt.tight_layout()
    plt.savefig(plot_dir / "environmental_impact.png", dpi=300)
    plt.close()

def plot_feature_correlations(df: pd.DataFrame, plot_dir: Path):
    """筛选高相关性特征，特别是历史滞后特征与目标变量的关系"""
    print("Plotting feature correlations...")
    # 选择业务相关的数值型变量进行相关性分析
    target_cols = ['rent', 'return', 'temperature', 'wind_level', 'rain_level', 
                   'rent_mean_7d', 'return_mean_7d', 'lag_nb_rent', 'lag_nb_return',
                   'low_power_bike_count', 'normal_power_bike_count']
    
    # 过滤出数据集中实际存在的列
    existing_cols = [col for col in target_cols if col in df.columns]
    corr = df[existing_cols].corr(method='spearman') # 使用Spearman处理非线性关系更好
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Spearman Correlation Matrix of Key Features')
    
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_correlation.png", dpi=300)
    plt.close()

def main():
    # 设置路径
    script_dir = Path.cwd()
    data_file = script_dir / "swapping_data_test.csv" # 请确保文件与脚本在同级目录
    plot_dir = script_dir / "EDA_Results"
    plot_dir.mkdir(exist_ok=True)
    
    if not data_file.exists():
        print(f"Error: Could not find {data_file}")
        return

    # 1. 加载数据
    df = load_and_preprocess(str(data_file))
    
    # 2. 核心业务 EDA
    # 观察特征交互：工作日/周末 x 小时段 对潮汐的剧烈影响
    plot_temporal_interaction(df, plot_dir)
    
    # 观察外部噪音：天气、降水如何抑制需求
    plot_environmental_impact(df, plot_dir)
    
    # 观察特征共线性：历史均值(rent_mean_7d)和前序时刻滞后值(lag_nb_rent)谁起决定性作用？
    plot_feature_correlations(df, plot_dir)
    
    print(f"EDA is complete. Please check the '{plot_dir.name}' folder for your charts.")

if __name__ == "__main__":
    main()