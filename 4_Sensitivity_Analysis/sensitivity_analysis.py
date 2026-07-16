#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
敏感性分析脚本 — ST-Graph 优化模型关键参数单因素扫描 (One-Factor Sweep)
=============================================================================

对以下三个关键变量分别进行单因素敏感性扫描:
  1. vehicle_speed_kmh (车速)       : [15, 20, 25, 30, 35, 40, 50] km/h
  2. C_max          (车载最大容量)   : [10, 15, 20, 25, 30] 块
  3. P_intervals    (时空离散段数)   : [6, 8, 10, 12, 16, 20] 段

每个变量扫描后自动:
  - 使用 tqdm + print 实时输出求解进度、状态与 MIP Gap
  - 提取关键指标并保存至 CSV 文件
  - 绘制并保存敏感性分析对比图表

运行方式:
  cd VRP-for-E-Bike-Battery-Swapping-Routing
  python 4_Sensitivity_Analysis/sensitivity_analysis.py

=============================================================================
"""

from __future__ import annotations

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径与导入配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OPTIM_DIR = os.path.join(PROJECT_ROOT, '3_Optimization')
sys.path.insert(0, OPTIM_DIR)

from main_STGraph import run_optimization_pipeline
from Pre_Process import DEFAULT_TARGET_DATETIME

# ---------------------------------------------------------------------------
# 可视化导入 (非交互后端, 适配服务器环境)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ---------------------------------------------------------------------------
# 进度条
# ---------------------------------------------------------------------------
from tqdm import tqdm

# =========================================================================
# 全局路径与输出配置
# =========================================================================
DATA_FILE = os.path.join(OPTIM_DIR, 'Grid_Utility_Test.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'Results')
PLOT_DIR = os.path.join(SCRIPT_DIR, 'Plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# =========================================================================
# 固定参数 (与 main_STGraph.py 默认值保持一致)
# =========================================================================
FIXED_SWAP_TIME_C = 0.02          # 单块换电服务时间 (h)
FIXED_T_TOTAL = 1.0               # 规划周期总时长 (h)
FIXED_MAX_TRAVEL_TIME = 0.2       # 弧段裁剪绝对阈值 (h)
FIXED_K_NEIGHBORS = 50            # KNN 近邻数
FIXED_TIME_LIMIT_S = 1200         # Gurobi 求解时限 (s)

# 各变量的默认基准值 (扫描其他变量时固定为基准值)
DEFAULT_SPEED_KMH = 30
DEFAULT_C_MAX = 20
DEFAULT_P_INTERVALS = 12
DEFAULT_Y_LEVELS = list(range(1, 11))  # 离散换电量 1~10

# =========================================================================
# 中文字体适配
# =========================================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
if HAS_SEABORN:
    sns.set_style("whitegrid")

# =========================================================================
# 辅助函数: 计算时序偏离度 Time_Gap
# =========================================================================
def compute_time_gap(result: dict, travel_time: dict, swap_time_c: float) -> float | None:
    """计算模型的时序偏离度 (Time_Gap)。

    定义为: 模型离散锚定到达时间 (arrival_time) 与物理真实模拟累加时间
    (cum_time, 由 haversine 距离÷速度 + 换电耗时累加得到) 之差的均值。

    该指标衡量时空图的离散化误差 — 值越小说明时序保真度越高。

    Parameters
    ----------
    result : dict
        run_optimization_pipeline 返回的结果字典。
    travel_time : dict
        旅行时间矩阵 travel_time[i][j], 其中 0 表示 depot。
    swap_time_c : float
        单块电池换电服务时间 (h)。

    Returns
    -------
    float or None
        所有非 depot 节点的 (arrival_time - cum_time) 均值;
        若路由为空则返回 None。
    """
    route = result.get("route", [])
    if not route:
        return None

    deviations = []
    cum_time = 0.0
    prev_spatial = 0  # depot 索引为 0

    for idx, step in enumerate(route):
        grid = step.get("grid")

        if grid == "DEPOT":
            # 返回 depot: 累加前一节点的换电耗时 + 行驶耗时
            if idx > 0:
                prev_y = route[idx - 1].get("y_swapped", 0)
                tt = travel_time.get(prev_spatial, {}).get(0, 0.0)
                cum_time += prev_y * swap_time_c + tt
        else:
            # 到达空间网格: 累加行驶耗时
            tt = travel_time.get(prev_spatial, {}).get(grid, 0.0)
            cum_time += tt

            # 记录模型时间与物理时间的偏差
            arrival_time = step.get("arrival_time", 0.0)
            deviations.append(arrival_time - cum_time)

            # 累加换电服务耗时, 准备前往下一节点
            y_swap = step.get("y_swapped", 0)
            cum_time += y_swap * swap_time_c
            prev_spatial = grid

    return float(np.mean(deviations)) if deviations else None


def extract_metrics(result: dict, swap_time_c: float = FIXED_SWAP_TIME_C) -> dict:
    """从 run_optimization_pipeline 的返回值中提取关键指标。

    Parameters
    ----------
    result : dict
        管线返回的完整结果字典。
    swap_time_c : float
        换电服务时间, 用于计算 Time_Gap。

    Returns
    -------
    dict
        扁平化的关键指标字典。
    """
    obj = result.get("objective")
    summary = result.get("summary", {})
    collector = result.get("collector")

    # 求解状态
    status = result.get("status", "UNKNOWN")
    elapsed = result.get("elapsed_seconds", 0.0)

    # MIP Gap (从 collector 获取)
    mip_gap = None
    if collector is not None:
        try:
            mip_gap = collector.mip_gap_pct
        except Exception:
            pass
    # 兜底: 如果 collector 未记录, 尝试从 status 推断
    if mip_gap is None or mip_gap == float('inf'):
        if "OPTIMAL" in str(status):
            mip_gap = 0.0

    # 总换电量
    total_swaps = summary.get("total_swaps", 0)

    # 时序偏离度
    travel_time = result.get("travel_time", {})
    time_gap = compute_time_gap(result, travel_time, swap_time_c)

    # 额外指标
    num_visited = len(result.get("visited_grids", []))
    makespan = summary.get("makespan_hrs", 0.0)

    return {
        "status": status,
        "elapsed_seconds": round(elapsed, 2),
        "MIPGap": round(mip_gap, 4) if mip_gap is not None else None,
        "objective": round(obj, 6) if obj is not None else None,
        "total_swaps": total_swaps,
        "Time_Gap": round(time_gap, 6) if time_gap is not None else None,
        "num_visited": num_visited,
        "makespan_hrs": round(makespan, 4),
    }


# =========================================================================
# 核心扫描函数
# =========================================================================
def _run_single_sweep(param_name: str, param_values: list, build_kwargs_fn) -> pd.DataFrame:
    """单变量扫描的通用执行引擎。

    Parameters
    ----------
    param_name : str
        被扫描的变量名 (用于打印和 CSV 命名)。
    param_values : list
        变量取值列表。
    build_kwargs_fn : callable
        接收单个参数值, 返回传给 run_optimization_pipeline 的 kwargs 字典。

    Returns
    -------
    pd.DataFrame
        包含所有扫描点关键指标的汇总表。
    """
    records = []
    n_total = len(param_values)

    print(f"\n{'='*70}")
    print(f"  敏感性扫描: {param_name}")
    print(f"  测试范围: {param_values}")
    print(f"  固定时间上限: {FIXED_TIME_LIMIT_S}s | 固定周期: {FIXED_T_TOTAL}h")
    print(f"{'='*70}")

    for idx, val in enumerate(tqdm(param_values, desc=f"扫描 {param_name}", unit="run")):
        print(f"\n--- [{idx+1}/{n_total}] {param_name} = {val} ---")

        kwargs = build_kwargs_fn(val)
        # 注入固定参数
        kwargs.setdefault("data_file", DATA_FILE)
        kwargs.setdefault("target_datetime", DEFAULT_TARGET_DATETIME)
        kwargs.setdefault("swap_time_c", FIXED_SWAP_TIME_C)
        kwargs.setdefault("T_total", FIXED_T_TOTAL)
        kwargs.setdefault("max_travel_time", FIXED_MAX_TRAVEL_TIME)
        kwargs.setdefault("K_neighbors", FIXED_K_NEIGHBORS)
        kwargs.setdefault("time_limit_s", FIXED_TIME_LIMIT_S)
        kwargs.setdefault("verbose", True)

        # 设置实验标签
        kwargs.setdefault("experiment_id", f"SENS_{param_name}")
        kwargs.setdefault("instance_name", f"{param_name}_{val}")

        t_start = time.perf_counter()
        try:
            result = run_optimization_pipeline(**kwargs)
        except Exception as exc:
            print(f"  [ERROR] 求解异常: {exc}")
            # 记录失败行
            records.append({
                param_name: val,
                "status": f"ERROR: {exc}",
                "elapsed_seconds": round(time.perf_counter() - t_start, 2),
                "MIPGap": None,
                "objective": None,
                "total_swaps": None,
                "Time_Gap": None,
                "num_visited": None,
                "makespan_hrs": None,
            })
            continue

        metrics = extract_metrics(result, swap_time_c=kwargs["swap_time_c"])

        # 实时打印关键结果
        print(f"  >> Status={metrics['status']} | "
              f"Obj={metrics['objective']} | "
              f"Swaps={metrics['total_swaps']} | "
              f"Gap={metrics['MIPGap']}% | "
              f"Time={metrics['elapsed_seconds']}s | "
              f"Time_Gap={metrics['Time_Gap']}")

        record = {param_name: val}
        record.update(metrics)
        records.append(record)

    # 汇总为 DataFrame
    df = pd.DataFrame(records)

    # 保存至 CSV
    csv_path = os.path.join(OUTPUT_DIR, f"sensitivity_results_{param_name}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  [CSV 已保存] {csv_path}")

    return df


# =========================================================================
# 三个变量的扫描接口
# =========================================================================

def sweep_vehicle_speed(speed_values: list = None) -> pd.DataFrame:
    """车速敏感性扫描。

    固定: C_max=20, P_intervals=12, y_levels=[1..10]
    扫描: vehicle_speed_kmh ∈ [15, 20, 25, 30, 35, 40, 50] km/h
    """
    if speed_values is None:
        speed_values = [15, 20, 25, 30, 35, 40, 50]

    def build_kwargs(speed):
        return {
            "vehicle_speed_kmh": speed,
            "C_max": DEFAULT_C_MAX,
            "P_intervals": DEFAULT_P_INTERVALS,
            "y_levels": DEFAULT_Y_LEVELS,
        }

    return _run_single_sweep("vehicle_speed_kmh", speed_values, build_kwargs)


def sweep_C_max(C_max_values: list = None) -> pd.DataFrame:
    """车载容量敏感性扫描。

    固定: vehicle_speed_kmh=30, P_intervals=12, y_levels 基准 = DEFAULT_Y_LEVELS
    扫描: C_max ∈ [10, 15, 20, 25, 30] 块

    重要说明:
      y_levels 使用固定基准值 (默认 [1..10])，仅按 C_max 裁剪上界
      (即过滤掉 y > C_max 的不可行换电量)。
      这与 main_STGraph.py 中 C_max=20 搭配 y_levels=[1..10] 的默认配置一致，
      确保模型规模在全部扫描点保持恒定，真正实现"仅车载容量"的单因素控制。
    """
    if C_max_values is None:
        C_max_values = [10, 15, 20, 25, 30]

    def build_kwargs(c_max):
        # 关键修复: 固定 y_levels 基准，仅过滤超出容量的值，避免模型规模膨胀
        y_levels = [y for y in DEFAULT_Y_LEVELS if y <= c_max]
        if not y_levels:
            y_levels = [c_max]  # 极端情况兜底: C_max 小于最小默认 y_level
        n_cut = len(DEFAULT_Y_LEVELS) - len(y_levels)
        print(f"  [参数适配] C_max={c_max} → y_levels={y_levels} "
              f"(固定基准{list(DEFAULT_Y_LEVELS)}, 裁剪{n_cut}个超容值)")
        return {
            "vehicle_speed_kmh": DEFAULT_SPEED_KMH,
            "C_max": c_max,
            "P_intervals": DEFAULT_P_INTERVALS,
            "y_levels": y_levels,
        }

    return _run_single_sweep("C_max", C_max_values, build_kwargs)


def sweep_P_intervals(P_values: list = None) -> pd.DataFrame:
    """时空离散段数敏感性扫描。

    固定: vehicle_speed_kmh=30, C_max=20, y_levels=[1..10]
    扫描: P_intervals ∈ [6, 8, 10, 12, 16, 20] 段
    """
    if P_values is None:
        P_values = [6, 8, 10, 12, 16, 20]

    def build_kwargs(p):
        return {
            "vehicle_speed_kmh": DEFAULT_SPEED_KMH,
            "C_max": DEFAULT_C_MAX,
            "P_intervals": p,
            "y_levels": DEFAULT_Y_LEVELS,
        }

    return _run_single_sweep("P_intervals", P_values, build_kwargs)


# =========================================================================
# 可视化函数
# =========================================================================

def plot_speed_sensitivity(df: pd.DataFrame):
    """绘制"车速 vs. 求解耗时与总变现效用"的双 Y 轴折线图。"""
    speeds = df["vehicle_speed_kmh"].values
    elapsed = df["elapsed_seconds"].values
    objective = df["objective"].values

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "#2B7BBD"
    color2 = "#E0533D"

    # 左 Y 轴: 求解耗时
    ax1.set_xlabel("车速 (km/h)", fontsize=13)
    ax1.set_ylabel("求解耗时 (s)", color=color1, fontsize=13)
    line1, = ax1.plot(speeds, elapsed, 'o-', color=color1, linewidth=2.2,
                       markersize=8, label="求解耗时 (elapsed_seconds)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # 右 Y 轴: 总变现效用
    ax2 = ax1.twinx()
    ax2.set_ylabel("总变现效用 (objective)", color=color2, fontsize=13)
    # 过滤掉 None 值
    valid_mask = objective != None  # noqa: E711
    line2, = ax2.plot(speeds[valid_mask], objective[valid_mask], 's--',
                       color=color2, linewidth=2.2, markersize=8,
                       label="总变现效用 (objective)")
    ax2.tick_params(axis='y', labelcolor=color2)

    # 图例合并
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=11, framealpha=0.9)

    ax1.set_title("车速敏感性分析: 求解耗时 vs. 总变现效用", fontsize=15, fontweight='bold')
    ax1.set_xticks(speeds)

    fig.tight_layout()
    save_path = os.path.join(PLOT_DIR, "sensitivity_vehicle_speed.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图表已保存] {save_path}")


def plot_C_max_sensitivity(df: pd.DataFrame):
    """绘制"车载容量 vs. 实际换电总量与总变现效用"的柱状 + 折线组合图。"""
    c_values = df["C_max"].values.astype(int)
    total_swaps = df["total_swaps"].values
    objective = df["objective"].values

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_bar = "#5B9BD5"
    color_line = "#E0533D"

    x_pos = np.arange(len(c_values))
    bar_width = 0.45

    # 左 Y 轴: 实际换电总量 (柱状图)
    ax1.set_xlabel("车载最大容量 C_max (块)", fontsize=13)
    ax1.set_ylabel("实际换电总量 (块)", color=color_bar, fontsize=13)
    bars = ax1.bar(x_pos, total_swaps, bar_width, color=color_bar, alpha=0.85,
                   edgecolor="white", linewidth=0.8, label="实际换电总量 (total_swaps)")
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(c_values)
    ax1.grid(True, axis='y', alpha=0.3)

    # 在柱状图上标注数值
    for bar, val in zip(bars, total_swaps):
        if val is not None:
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                     str(int(val)), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 右 Y 轴: 总变现效用 (折线)
    ax2 = ax1.twinx()
    ax2.set_ylabel("总变现效用 (objective)", color=color_line, fontsize=13)
    valid_mask = objective != None  # noqa: E711
    line, = ax2.plot(x_pos[valid_mask], objective[valid_mask], 'D-',
                      color=color_line, linewidth=2.2, markersize=9,
                      label="总变现效用 (objective)")
    ax2.tick_params(axis='y', labelcolor=color_line)

    # 在折线上标注数值
    for xi, obj in zip(x_pos[valid_mask], objective[valid_mask]):
        if obj is not None:
            ax2.annotate(f"{obj:.2f}", (xi, obj),
                         textcoords="offset points", xytext=(0, 12),
                         ha='center', fontsize=9, color=color_line, fontweight='bold')

    # 合并图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_bar, alpha=0.85, label="实际换电总量 (total_swaps)"),
        line,
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.9)

    ax1.set_title("车载容量敏感性分析: 换电总量 vs. 总变现效用", fontsize=15, fontweight='bold')

    fig.tight_layout()
    save_path = os.path.join(PLOT_DIR, "sensitivity_C_max.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图表已保存] {save_path}")


def plot_P_intervals_sensitivity(df: pd.DataFrame):
    """绘制"离散段数 vs. 求解耗时与 Time_Gap"的双 Y 轴折线图。"""
    p_values = df["P_intervals"].values.astype(int)
    elapsed = df["elapsed_seconds"].values
    time_gap = df["Time_Gap"].values

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "#2B7BBD"
    color2 = "#E0533D"

    # 左 Y 轴: 求解耗时
    ax1.set_xlabel("时空离散段数 P_intervals", fontsize=13)
    ax1.set_ylabel("求解耗时 (s)", color=color1, fontsize=13)
    line1, = ax1.plot(p_values, elapsed, 'o-', color=color1, linewidth=2.2,
                       markersize=8, label="求解耗时 (elapsed_seconds)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(p_values)

    # 右 Y 轴: Time_Gap (时序偏离度)
    ax2 = ax1.twinx()
    ax2.set_ylabel("时序偏离度 Time_Gap (h)", color=color2, fontsize=13)
    valid_mask = time_gap != None  # noqa: E711
    line2, = ax2.plot(p_values[valid_mask], time_gap[valid_mask], '^--',
                       color=color2, linewidth=2.2, markersize=9,
                       label="时序偏离度 (Time_Gap)")
    ax2.tick_params(axis='y', labelcolor=color2)

    # 标注零线, 便于观察正负偏离
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)

    # 合并图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=11, framealpha=0.9)

    ax1.set_title("时空离散段数敏感性分析: 求解耗时 vs. 时序偏离度", fontsize=15, fontweight='bold')

    fig.tight_layout()
    save_path = os.path.join(PLOT_DIR, "sensitivity_P_intervals.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图表已保存] {save_path}")


# =========================================================================
# 汇总对比表
# =========================================================================
def print_summary_table(all_dfs: dict):
    """打印三个变量扫描的汇总对比表。"""
    print(f"\n{'='*90}")
    print("  敏感性分析汇总")
    print(f"{'='*90}")

    for var_name, df in all_dfs.items():
        if df is None or df.empty:
            print(f"\n  [{var_name}] 无有效数据")
            continue
        print(f"\n--- {var_name} ---")
        cols = [var_name, "objective", "total_swaps", "elapsed_seconds",
                "MIPGap", "Time_Gap", "status"]
        display_cols = [c for c in cols if c in df.columns]
        print(df[display_cols].to_string(index=False))


# =========================================================================
# 主入口
# =========================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print("=" * 70)
    print("  ST-Graph 优化模型 — 敏感性分析脚本")
    print(f"  数据文件: {DATA_FILE}")
    print(f"  结果目录: {OUTPUT_DIR}")
    print(f"  图表目录: {PLOT_DIR}")
    print(f"  求解时限: {FIXED_TIME_LIMIT_S}s (单次)")
    print("=" * 70)

    all_results = {}

    # ------------------------------------------------------------------
    # 1. 车速敏感性扫描
    # ------------------------------------------------------------------
    """print("\n" + "█" * 70)
    print("  [阶段 1/3] 车速 (vehicle_speed_kmh) 敏感性扫描")
    print("█" * 70)
    df_speed = sweep_vehicle_speed()
    all_results["vehicle_speed_kmh"] = df_speed
    if df_speed is not None and not df_speed.empty:
        plot_speed_sensitivity(df_speed)
"""
    # ------------------------------------------------------------------
    # 2. 车载容量敏感性扫描
    # ------------------------------------------------------------------
    print("\n" + "█" * 70)
    print("  [阶段 2/3] 车载容量 (C_max) 敏感性扫描")
    print("█" * 70)
    df_cmax = sweep_C_max()
    all_results["C_max"] = df_cmax
    if df_cmax is not None and not df_cmax.empty:
        plot_C_max_sensitivity(df_cmax)

    # ------------------------------------------------------------------
    # 3. 时空离散段数敏感性扫描
    # ------------------------------------------------------------------
    print("\n" + "█" * 70)
    print("  [阶段 3/3] 时空离散段数 (P_intervals) 敏感性扫描")
    print("█" * 70)
    df_p = sweep_P_intervals()
    all_results["P_intervals"] = df_p
    if df_p is not None and not df_p.empty:
        plot_P_intervals_sensitivity(df_p)

    # ------------------------------------------------------------------
    # 汇总输出
    # ------------------------------------------------------------------
    print_summary_table(all_results)

    print(f"\n{'='*70}")
    print(f"  敏感性分析全部完成!")
    print(f"  CSV 结果目录: {OUTPUT_DIR}")
    print(f"  图表目录    : {PLOT_DIR}")
    print(f"{'='*70}")
