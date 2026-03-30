import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)


def print_section(title: str):
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def infer_column_groups(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    categorical_cols = [
        col
        for col in df.columns
        if col not in numeric_cols and col not in datetime_cols
    ]
    return numeric_cols, categorical_cols, datetime_cols


def dataset_overview(df: pd.DataFrame):
    print_section("Dataset Overview")
    n_rows, n_cols = df.shape
    print(f"Rows: {n_rows:,}")
    print(f"Columns: {n_cols}")
    print(f"Total memory usage: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

    numeric_cols, categorical_cols, datetime_cols = infer_column_groups(df)
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Datetime columns ({len(datetime_cols)}): {datetime_cols}")

    duplicates = int(df.duplicated().sum())
    print(f"Duplicated rows: {duplicates} ({duplicates / max(n_rows, 1):.2%})")

    missing_count = df.isna().sum()
    missing_pct = (missing_count / max(n_rows, 1)) * 100
    missing_table = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "dtype": df.dtypes.astype(str)
    }).sort_values("missing_pct", ascending=False)

    print("Top missing columns:")
    print(missing_table.head(15).to_string())


def numeric_profile(df: pd.DataFrame):
    print_section("Numeric Profile")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found.")
        return

    desc = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc["skew"] = df[numeric_cols].skew(numeric_only=True)
    desc["kurtosis"] = df[numeric_cols].kurt(numeric_only=True)
    print(desc.head(15).to_string())

    # IQR-based outlier ratio
    outlier_records = []
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            outlier_records.append({"column": col, "outlier_count": 0, "outlier_pct": 0.0})
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_count = int(((s < q1) | (s > q3)).sum())
        else:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_count = int(((s < lower) | (s > upper)).sum())
        outlier_records.append({
            "column": col,
            "outlier_count": outlier_count,
            "outlier_pct": outlier_count / max(len(s), 1)
        })

    outlier_df = pd.DataFrame(outlier_records).sort_values("outlier_pct", ascending=False)
    print("Top outlier-heavy columns:")
    print(outlier_df.head(15).to_string(index=False))


def categorical_profile(df: pd.DataFrame):
    print_section("Categorical Profile")
    numeric_cols, categorical_cols, _ = infer_column_groups(df)
    _ = numeric_cols  # keep variable for readability symmetry
    if not categorical_cols:
        print("No categorical columns found.")
        return

    rows = []
    for col in categorical_cols:
        s = df[col]
        rows.append({
            "column": col,
            "n_unique": int(s.nunique(dropna=True)),
            "missing_pct": float(s.isna().mean())
        })

    cat_summary = pd.DataFrame(rows).sort_values(["n_unique", "missing_pct"], ascending=[False, False])
    print(cat_summary.head(15).to_string(index=False))

    # Export top categories for each categorical column
    topk_frames = []
    for col in categorical_cols:
        topk = df[col].value_counts(dropna=False).head(10).rename("count").reset_index()
        topk.columns = ["value", "count"]
        topk.insert(0, "column", col)
        topk["share"] = topk["count"] / max(len(df), 1)
        topk_frames.append(topk)
    if topk_frames:
        topk_all = pd.concat(topk_frames, ignore_index=True)
        print("Top categories snapshot (first 20 rows):")
        print(topk_all.head(20).to_string(index=False))


def temporal_profile(df: pd.DataFrame):
    print_section("Temporal Profile")
    if "datetime" not in df.columns:
        print("No 'datetime' column found, skipping temporal profiling.")
        return

    temp_df = df.copy()
    temp_df["date"] = temp_df["datetime"].dt.date
    temp_df["month"] = temp_df["datetime"].dt.to_period("M").astype(str)

    date_span = temp_df["datetime"].agg(["min", "max"])
    print(f"Date range: {date_span['min']} -> {date_span['max']}")
    print(f"Unique dates: {temp_df['date'].nunique()}")

    if "hour" in temp_df.columns:
        hourly_density = temp_df["hour"].value_counts().sort_index()
        print("Hourly record density (top hours by count):")
        print(hourly_density.sort_values(ascending=False).head(10).to_string())

    monthly_volume = temp_df.groupby("month").size().rename("record_count")
    print("Monthly record counts:")
    print(monthly_volume.to_string())


def target_diagnostics(df: pd.DataFrame):
    print_section("Target Diagnostics")
    targets = [col for col in ["rent", "return"] if col in df.columns]
    if not targets:
        print("No standard target columns (rent/return) found.")
        return

    diag_rows = []
    for t in targets:
        s = df[t].dropna()
        diag_rows.append({
            "target": t,
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "cv": s.std() / s.mean() if s.mean() != 0 else np.nan,
            "zero_ratio": (s == 0).mean(),
            "p95": s.quantile(0.95),
            "p99": s.quantile(0.99),
            "max": s.max(),
        })

    diag_df = pd.DataFrame(diag_rows)
    print(diag_df.to_string(index=False))

    # If both targets exist, inspect imbalance of net flow.
    if all(c in df.columns for c in ["rent", "return"]):
        net_flow = (df["rent"] - df["return"]).dropna()
        net_stats = {
            "mean_net_flow": net_flow.mean(),
            "std_net_flow": net_flow.std(),
            "p05_net_flow": net_flow.quantile(0.05),
            "p95_net_flow": net_flow.quantile(0.95),
            "positive_ratio": (net_flow > 0).mean(),
            "negative_ratio": (net_flow < 0).mean(),
        }
        print("Net flow summary:")
        print(pd.DataFrame([net_stats]).to_string(index=False))


def grouped_business_analysis(df: pd.DataFrame):
    print_section("Grouped Business Analysis")
    if "rent" not in df.columns and "return" not in df.columns:
        print("No rent/return columns found, skipping grouped analysis.")
        return

    metrics = [col for col in ["rent", "return"] if col in df.columns]
    group_candidates = [
        col for col in ["is_weekend", "day_type", "rain_level", "wind_level", "hour"] if col in df.columns
    ]
    if not group_candidates:
        print("No candidate grouping columns found.")
        return

    for g in group_candidates:
        summary = df.groupby(g)[metrics].agg(["count", "mean", "median", "std"])
        print(f"Group summary by '{g}' (first 10 rows):")
        print(summary.head(10).to_string())


def feature_target_ranking(df: pd.DataFrame):
    print_section("Feature-Target Correlation Ranking")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    targets = [col for col in ["rent", "return"] if col in numeric_cols]
    if not targets:
        print("No numeric target columns (rent/return) found.")
        return

    feature_cols = [c for c in numeric_cols if c not in targets]
    if not feature_cols:
        print("No candidate numeric features found.")
        return

    rank_frames = []
    for t in targets:
        corr_series = df[feature_cols + [t]].corr(method="spearman")[t].drop(labels=[t]).dropna()
        rank = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)
        temp = rank.reset_index()
        temp.columns = ["feature", "spearman_corr"]
        temp.insert(0, "target", t)
        rank_frames.append(temp)
        print(f"Top correlations for {t}:")
        print(temp.head(10).to_string(index=False))

    if rank_frames:
        rank_df = pd.concat(rank_frames, ignore_index=True)
        print("Combined feature-target ranking snapshot (first 20 rows):")
        print(rank_df.head(20).to_string(index=False))


def build_text_eda_report(df: pd.DataFrame):
    dataset_overview(df)
    numeric_profile(df)
    categorical_profile(df)
    temporal_profile(df)
    target_diagnostics(df)
    grouped_business_analysis(df)
    feature_target_ranking(df)


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
    plt.savefig(plot_dir / "official_temporal_interaction_effects.png", dpi=300)
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
    sns.boxplot(
        data=df,
        x='rain_level',
        y='rent',
        hue='rain_level',
        palette="Blues",
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title('Rain Level Impact on Rent Volume')
    
    plt.tight_layout()
    plt.savefig(plot_dir / "official_environmental_impact.png", dpi=300)
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
    plt.savefig(plot_dir / "official_feature_correlation.png", dpi=300)
    plt.close()

def main():
    # 设置路径
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir / "battery_swapping_routing_data.csv"
    plot_dir = script_dir / "EDA_Results"
    plot_dir.mkdir(exist_ok=True)
    
    if not data_file.exists():
        print(f"Error: Could not find {data_file}")
        return

    # 1. 加载数据
    df = load_and_preprocess(str(data_file))

    # 2. 文本型 EDA（不依赖图形，直接输出关键统计）
    build_text_eda_report(df)
    
    # 3. 核心业务可视化 EDA
    # 观察特征交互：工作日/周末 x 小时段 对潮汐的剧烈影响
    plot_temporal_interaction(df, plot_dir)
    
    # 观察外部噪音：天气、降水如何抑制需求
    plot_environmental_impact(df, plot_dir)
    
    # 观察特征共线性：历史均值(rent_mean_7d)和前序时刻滞后值(lag_nb_rent)谁起决定性作用？
    plot_feature_correlations(df, plot_dir)
    
    print(f"EDA is complete. Please check the '{plot_dir.name}' folder for charts.")

if __name__ == "__main__":
    main()