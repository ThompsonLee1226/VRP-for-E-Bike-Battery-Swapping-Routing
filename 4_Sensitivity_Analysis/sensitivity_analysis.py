#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
敏感性分析脚本 — ST-Graph 优化模型关键参数单因素扫描 (One-Factor Sweep)
=============================================================================

对以下四个关键变量分别进行单因素敏感性扫描:
  1. vehicle_speed_kmh (车速)       : [15, 20, 25, 30, 35, 40, 50] km/h
  2. swap_time_c      (单块换电服务时间) : [0.02, 0.03, 0.04, 0.05] h
  3. C_max          (车载最大容量)   : [10, 15, 20, 25, 30] 块
  4. P_intervals    (时空离散段数)   : [6, 8, 10, 12, 16, 20] 段

以及交叉项联合扫描:
  5. P_intervals × vehicle_speed : P ∈ {8, 12, 16} × speed ∈ {20, 30, 40, 50}

每个变量扫描后自动:
  - 使用 tqdm + print 实时输出求解进度、状态与 MIP Gap
  - 提取关键指标并保存至 CSV 文件
  - 绘制并保存敏感性分析对比图表

运行方式:
  python 4_Sensitivity_Analysis/sensitivity_analysis.py                                      # 使用默认时间
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --datetime "2025/10/28 14:00"        # 指定具体时间
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --random                             # 随机选取可用小时
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --random --seed 42                   # 随机选取 (固定种子)
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --list-hours                         # 列出可用小时
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --data path/to/other.csv             # 指定数据文件
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --skip-cross                         # 跳过交叉项扫描
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --only speed swap_time               # 仅运行指定扫描

=============================================================================
"""

from __future__ import annotations

import sys
import os
import time
import argparse
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
from Pre_Process import (
    DEFAULT_TARGET_DATETIME,
    DEFAULT_PREDICTION_FILE,
    DATETIME_RANGE_START,
    DATETIME_RANGE_END,
    select_random_datetime,
    list_available_hours,
)

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
# ★ 数据文件: 使用 CB_Hurdle 最优训练结果 (与 3_Optimization 共享)
#    修改 3_Optimization/Pre_Process.py 中的 TRAINING_RESULT_DIR 即可切换
#    也可通过 CLI --data 参数动态指定
DATA_FILE = DEFAULT_PREDICTION_FILE
TARGET_DATETIME = DEFAULT_TARGET_DATETIME   # 可通过 CLI --datetime / --random 覆盖

# ★ 输出根目录 (每次运行创建带时间戳的子目录, 对齐 batch_runner.py 模式)
OUTPUT_ROOT_DIR = os.path.join(SCRIPT_DIR, 'Results')
PLOT_ROOT_DIR = os.path.join(SCRIPT_DIR, 'Plots')
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
os.makedirs(PLOT_ROOT_DIR, exist_ok=True)

# ★ 运行时变量 (在 __main__ 中通过 CLI 解析后赋值)
RUN_TIMESTAMP: str = ""
RUN_OUTPUT_DIR: str = ""
RUN_PLOT_DIR: str = ""

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
def _run_single_sweep(param_name: str, param_values: list, build_kwargs_fn,
                      output_dir: str = "", target_datetime: str = "") -> pd.DataFrame:
    """单变量扫描的通用执行引擎。

    Parameters
    ----------
    param_name : str
        被扫描的变量名 (用于打印和 CSV 命名)。
    param_values : list
        变量取值列表。
    build_kwargs_fn : callable
        接收单个参数值, 返回传给 run_optimization_pipeline 的 kwargs 字典。
    output_dir : str
        本次运行的输出目录 (对齐 batch_runner.py 的 run_output_dir 模式)。
    target_datetime : str
        目标日期时间字符串 (对齐 3_Optimization 的 --datetime 模式)。

    Returns
    -------
    pd.DataFrame
        包含所有扫描点关键指标的汇总表。
    """
    records = []
    n_total = len(param_values)
    _out_dir = output_dir or RUN_OUTPUT_DIR
    _tgt_dt = target_datetime or TARGET_DATETIME

    print(f"\n{'='*70}")
    print(f"  敏感性扫描: {param_name}")
    print(f"  测试范围: {param_values}")
    print(f"  目标时间: {_tgt_dt}")
    print(f"  固定时间上限: {FIXED_TIME_LIMIT_S}s | 固定周期: {FIXED_T_TOTAL}h")
    print(f"{'='*70}")

    for idx, val in enumerate(tqdm(param_values, desc=f"扫描 {param_name}", unit="run")):
        print(f"\n--- [{idx+1}/{n_total}] {param_name} = {val} ---")

        kwargs = build_kwargs_fn(val)
        # 注入固定参数 (对齐 3_Optimization 的共享参数模式)
        kwargs.setdefault("data_file", DATA_FILE)
        kwargs.setdefault("target_datetime", _tgt_dt)
        kwargs.setdefault("swap_time_c", FIXED_SWAP_TIME_C)
        kwargs.setdefault("T_total", FIXED_T_TOTAL)
        kwargs.setdefault("max_travel_time", FIXED_MAX_TRAVEL_TIME)
        kwargs.setdefault("K_neighbors", FIXED_K_NEIGHBORS)
        kwargs.setdefault("time_limit_s", FIXED_TIME_LIMIT_S)
        kwargs.setdefault("verbose", True)
        # ★ 显式传入 output_dir, 让 run_optimization_pipeline 输出到本次运行目录
        kwargs.setdefault("output_dir", _out_dir)

        # 设置实验标签 (对齐 experiment_utils 的 {experiment_id}_{instance_name}_{timestamp} 模式)
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

    # ★ 保存至 CSV (对齐 export_experiment_result 的命名: {experiment_id}_{instance_name}_{timestamp})
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_name = f"SENS_{param_name}_sweep_summary_{timestamp}.csv"
    csv_path = os.path.join(_out_dir, csv_name)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  [CSV 已保存] {csv_path}")

    return df


# =========================================================================
# 单因素扫描接口
# =========================================================================

def sweep_vehicle_speed(speed_values: list = None,
                        output_dir: str = "", target_datetime: str = "") -> pd.DataFrame:
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

    return _run_single_sweep("vehicle_speed_kmh", speed_values, build_kwargs,
                             output_dir=output_dir, target_datetime=target_datetime)

def sweep_swap_time(swap_time_values: list = None,
                    output_dir: str = "", target_datetime: str = "") -> pd.DataFrame:
    """单块换电服务时间敏感性扫描。

    固定: vehicle_speed_kmh=30, C_max=20, P_intervals=12, y_levels=[1..10]
    扫描: swap_time_c ∈ [0.02, 0.03, 0.04, 0.05] h
    """
    if swap_time_values is None:
        swap_time_values = [0.02, 0.03, 0.04, 0.05]

    def build_kwargs(swap_time):
        return {
            "vehicle_speed_kmh": DEFAULT_SPEED_KMH,
            "C_max": DEFAULT_C_MAX,
            "P_intervals": DEFAULT_P_INTERVALS,
            "y_levels": DEFAULT_Y_LEVELS,
            "swap_time_c": swap_time,
        }

    return _run_single_sweep("swap_time_c", swap_time_values, build_kwargs,
                             output_dir=output_dir, target_datetime=target_datetime)

def sweep_C_max(C_max_values: list = None,
                output_dir: str = "", target_datetime: str = "") -> pd.DataFrame:
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

    return _run_single_sweep("C_max", C_max_values, build_kwargs,
                             output_dir=output_dir, target_datetime=target_datetime)


def sweep_P_intervals(P_values: list = None,
                      output_dir: str = "", target_datetime: str = "") -> pd.DataFrame:
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

    return _run_single_sweep("P_intervals", P_values, build_kwargs,
                             output_dir=output_dir, target_datetime=target_datetime)


# =========================================================================
# 交叉项联合扫描: (P_intervals × vehicle_speed)
# =========================================================================

def sweep_P_speed_cross(P_values: list = None, speed_values: list = None,
                        output_dir: str = "", target_datetime: str = "") -> pd.DataFrame:
    """P_intervals × vehicle_speed 交叉项联合扫描。

    固定: C_max=20, y_levels=[1..10]
    扫描: P_intervals ∈ {8, 12, 16} × vehicle_speed_kmh ∈ {20, 30, 40, 50}
    总计 3 × 4 = 12 个组合。

    Parameters
    ----------
    P_values : list, optional
        P_intervals 取值列表, 默认 [8, 12, 16]。
    speed_values : list, optional
        vehicle_speed_kmh 取值列表, 默认 [20, 30, 40, 50]。
    output_dir : str
        本次运行的输出目录 (对齐 batch_runner.py 的 run_output_dir 模式)。
    target_datetime : str
        目标日期时间字符串 (对齐 3_Optimization 的 --datetime 模式)。

    Returns
    -------
    pd.DataFrame
        包含所有扫描组合关键指标的汇总表。
    """
    if P_values is None:
        P_values = [8, 12, 16]
    if speed_values is None:
        speed_values = [20, 30, 40, 50]

    records = []
    n_total = len(P_values) * len(speed_values)
    _out_dir = output_dir or RUN_OUTPUT_DIR
    _tgt_dt = target_datetime or TARGET_DATETIME

    print(f"\n{'='*70}")
    print(f"  交叉项敏感性扫描: P_intervals × vehicle_speed")
    print(f"  P_intervals ∈ {P_values}")
    print(f"  vehicle_speed ∈ {speed_values}")
    print(f"  总计 {n_total} 个组合")
    print(f"  目标时间: {_tgt_dt}")
    print(f"  固定: C_max={DEFAULT_C_MAX}, y_levels={list(DEFAULT_Y_LEVELS)}")
    print(f"  固定时间上限: {FIXED_TIME_LIMIT_S}s | 固定周期: {FIXED_T_TOTAL}h")
    print(f"{'='*70}")

    idx = 0
    for p in P_values:
        for speed in speed_values:
            idx += 1
            print(f"\n--- [{idx}/{n_total}] P_intervals={p}, vehicle_speed={speed} km/h ---")

            kwargs = {
                "vehicle_speed_kmh": speed,
                "C_max": DEFAULT_C_MAX,
                "P_intervals": p,
                "y_levels": DEFAULT_Y_LEVELS,
                "data_file": DATA_FILE,
                "target_datetime": _tgt_dt,
                "swap_time_c": FIXED_SWAP_TIME_C,
                "T_total": FIXED_T_TOTAL,
                "max_travel_time": FIXED_MAX_TRAVEL_TIME,
                "K_neighbors": FIXED_K_NEIGHBORS,
                "time_limit_s": FIXED_TIME_LIMIT_S,
                "verbose": True,
                "experiment_id": "SENS_P_speed_cross",
                "instance_name": f"P{p}_spd{speed}",
                # ★ 显式传入 output_dir, 对齐 batch_runner 模式
                "output_dir": _out_dir,
            }

            t_start = time.perf_counter()
            try:
                result = run_optimization_pipeline(**kwargs)
            except Exception as exc:
                print(f"  [ERROR] 求解异常: {exc}")
                records.append({
                    "P_intervals": p,
                    "vehicle_speed_kmh": speed,
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

            print(f"  >> Status={metrics['status']} | "
                  f"Obj={metrics['objective']} | "
                  f"Swaps={metrics['total_swaps']} | "
                  f"Gap={metrics['MIPGap']}% | "
                  f"Time={metrics['elapsed_seconds']}s | "
                  f"Time_Gap={metrics['Time_Gap']}")

            record = {"P_intervals": p, "vehicle_speed_kmh": speed}
            record.update(metrics)
            records.append(record)

    df = pd.DataFrame(records)

    # ★ 保存至 CSV (对齐 export_experiment_result 命名模式)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_name = f"SENS_P_speed_cross_sweep_summary_{timestamp}.csv"
    csv_path = os.path.join(_out_dir, csv_name)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  [CSV 已保存] {csv_path}")

    return df


# =========================================================================
# 可视化函数
# =========================================================================

def plot_speed_sensitivity(df: pd.DataFrame, plot_dir: str = ""):
    """绘制"车速 vs. 求解耗时与总变现效用"的双 Y 轴折线图。"""
    _plot_dir = plot_dir or RUN_PLOT_DIR
    speeds = df["vehicle_speed_kmh"].values
    elapsed = df["elapsed_seconds"].values
    objective = df["objective"].values

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "#2B7BBD"
    color2 = "#E0533D"

    # Left Y axis: Solve Time
    ax1.set_xlabel("speed (km/h)", fontsize=13)
    ax1.set_ylabel("Solve Time (s)", color=color1, fontsize=13)
    line1, = ax1.plot(speeds, elapsed, 'o-', color=color1, linewidth=2.2,
                       markersize=8, label="Solve Time (elapsed_seconds)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Right Y axis: Total Utility
    ax2 = ax1.twinx()
    ax2.set_ylabel("Total Utility (objective)", color=color2, fontsize=13)
    # Filter out None values
    valid_mask = objective != None  # noqa: E711
    line2, = ax2.plot(speeds[valid_mask], objective[valid_mask], 's--',
                       color=color2, linewidth=2.2, markersize=8,
                       label="Total Utility (objective)")
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=11, framealpha=0.9)

    ax1.set_title("Sensitivity Analysis: Vehicle Speed — Solve Time vs. Total Utility", fontsize=15, fontweight='bold')
    ax1.set_xticks(speeds)

    fig.tight_layout()
    save_path = os.path.join(_plot_dir, "sensitivity_vehicle_speed.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot saved] {save_path}")

def plot_swap_time_sensitivity(df: pd.DataFrame, plot_dir: str = ""):
    """绘制"单块换电服务时间 vs. 求解耗时与总变现效用"的双 Y 轴折线图。"""
    _plot_dir = plot_dir or RUN_PLOT_DIR
    swap_times = df["swap_time_c"].values
    elapsed = df["elapsed_seconds"].values
    objective = df["objective"].values

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "#2B7BBD"
    color2 = "#E0533D"

    # Left Y axis: Solve Time
    ax1.set_xlabel("Per-Battery Swap Service Time (h)", fontsize=13)
    ax1.set_ylabel("Solve Time (s)", color=color1, fontsize=13)
    line1, = ax1.plot(swap_times, elapsed, 'o-', color=color1, linewidth=2.2,
                       markersize=8, label="Solve Time (elapsed_seconds)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Right Y axis: Total Utility
    ax2 = ax1.twinx()
    ax2.set_ylabel("Total Utility (objective)", color=color2, fontsize=13)
    valid_mask = objective != None  # noqa: E711
    line2, = ax2.plot(swap_times[valid_mask], objective[valid_mask], 's--',
                       color=color2, linewidth=2.2, markersize=8,
                       label="Total Utility (objective)")
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=11, framealpha=0.9)

    ax1.set_title("Sensitivity Analysis: Swap Service Time — Solve Time vs. Total Utility", fontsize=15, fontweight='bold')
    ax1.set_xticks(swap_times)

    fig.tight_layout()
    save_path = os.path.join(_plot_dir, "sensitivity_swap_time.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot saved] {save_path}")

def plot_C_max_sensitivity(df: pd.DataFrame, plot_dir: str = ""):
    """绘制"车载容量 vs. 实际换电总量与总变现效用"的柱状 + 折线组合图。"""
    _plot_dir = plot_dir or RUN_PLOT_DIR
    c_values = df["C_max"].values.astype(int)
    total_swaps = df["total_swaps"].values
    objective = df["objective"].values

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_bar = "#5B9BD5"
    color_line = "#E0533D"

    x_pos = np.arange(len(c_values))
    bar_width = 0.45

    # Left Y axis: Total Swapped Batteries (bar chart)
    ax1.set_xlabel("Vehicle Battery Capacity C_max (units)", fontsize=13)
    ax1.set_ylabel("Total Swapped Batteries (units)", color=color_bar, fontsize=13)
    bars = ax1.bar(x_pos, total_swaps, bar_width, color=color_bar, alpha=0.85,
                   edgecolor="white", linewidth=0.8, label="Total Swapped Batteries (total_swaps)")
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(c_values)
    ax1.grid(True, axis='y', alpha=0.3)

    # Annotate values on bars
    for bar, val in zip(bars, total_swaps):
        if val is not None:
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                     str(int(val)), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Right Y axis: Total Utility (line)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Total Utility (objective)", color=color_line, fontsize=13)
    valid_mask = objective != None  # noqa: E711
    line, = ax2.plot(x_pos[valid_mask], objective[valid_mask], 'D-',
                      color=color_line, linewidth=2.2, markersize=9,
                      label="Total Utility (objective)")
    ax2.tick_params(axis='y', labelcolor=color_line)

    # Annotate values on line
    for xi, obj in zip(x_pos[valid_mask], objective[valid_mask]):
        if obj is not None:
            ax2.annotate(f"{obj:.2f}", (xi, obj),
                         textcoords="offset points", xytext=(0, 12),
                         ha='center', fontsize=9, color=color_line, fontweight='bold')

    # Combined legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_bar, alpha=0.85, label="Total Swapped Batteries (total_swaps)"),
        line,
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.9)

    ax1.set_title("Sensitivity Analysis: Battery Capacity — Swaps vs. Total Utility", fontsize=15, fontweight='bold')

    fig.tight_layout()
    save_path = os.path.join(_plot_dir, "sensitivity_C_max.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot saved] {save_path}")


def plot_P_intervals_sensitivity(df: pd.DataFrame, plot_dir: str = ""):
    """绘制"离散段数 vs. 求解耗时与 Time_Gap"的双 Y 轴折线图。"""
    _plot_dir = plot_dir or RUN_PLOT_DIR
    p_values = df["P_intervals"].values.astype(int)
    elapsed = df["elapsed_seconds"].values
    time_gap = df["Time_Gap"].values

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "#2B7BBD"
    color2 = "#E0533D"

    # Left Y axis: Solve Time
    ax1.set_xlabel("Temporal Discretization Intervals P", fontsize=13)
    ax1.set_ylabel("Solve Time (s)", color=color1, fontsize=13)
    line1, = ax1.plot(p_values, elapsed, 'o-', color=color1, linewidth=2.2,
                       markersize=8, label="Solve Time (elapsed_seconds)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(p_values)

    # Right Y axis: Time_Gap (temporal deviation)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Temporal Deviation Time_Gap (h)", color=color2, fontsize=13)
    valid_mask = time_gap != None  # noqa: E711
    line2, = ax2.plot(p_values[valid_mask], time_gap[valid_mask], '^--',
                       color=color2, linewidth=2.2, markersize=9,
                       label="Temporal Deviation (Time_Gap)")
    ax2.tick_params(axis='y', labelcolor=color2)

    # Mark zero line to observe positive/negative deviation
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=11, framealpha=0.9)

    ax1.set_title("Sensitivity Analysis: P Intervals — Solve Time vs. Temporal Deviation", fontsize=15, fontweight='bold')

    fig.tight_layout()
    save_path = os.path.join(_plot_dir, "sensitivity_P_intervals.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot saved] {save_path}")


def plot_P_speed_cross(df: pd.DataFrame, plot_dir: str = ""):
    """绘制 P_intervals × vehicle_speed 交叉项敏感性分析图。

    生成四面板综合图:
      左上: 总变现效用热力图 (objective)
      右上: 求解耗时热力图 (elapsed_seconds)
      左下: 不同 P 下速度对目标函数影响的折线对比
      右下: 不同 P 下速度对时序偏离度影响的折线对比
    """
    if df is None or df.empty:
        print("  [跳过] 交叉项数据为空, 无法绘图")
        return

    _plot_dir = plot_dir or RUN_PLOT_DIR
    P_vals = sorted(df["P_intervals"].unique())
    speed_vals = sorted(df["vehicle_speed_kmh"].unique())

    # Pivot data
    pivot_obj = df.pivot_table(values="objective", index="P_intervals",
                               columns="vehicle_speed_kmh", aggfunc="first")
    pivot_time = df.pivot_table(values="elapsed_seconds", index="P_intervals",
                                columns="vehicle_speed_kmh", aggfunc="first")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors_line = ["#2B7BBD", "#E0533D", "#4DAF4A"]
    markers_line = ["o", "s", "D"]

    # ------------------------------------------------------------------
    # Top-left: Total Utility heatmap
    # ------------------------------------------------------------------
    ax1 = axes[0, 0]
    obj_data = pivot_obj.loc[P_vals, speed_vals].values
    im1 = ax1.imshow(obj_data, cmap="YlOrRd", aspect="auto", origin="lower")
    ax1.set_xticks(range(len(speed_vals)))
    ax1.set_xticklabels(speed_vals)
    ax1.set_yticks(range(len(P_vals)))
    ax1.set_yticklabels(P_vals)
    ax1.set_xlabel("speed (km/h)", fontsize=12)
    ax1.set_ylabel("P_intervals", fontsize=12)
    ax1.set_title("Total Utility (objective)", fontsize=13, fontweight="bold")
    vmax_obj = np.nanmax(obj_data)
    for i in range(len(P_vals)):
        for j in range(len(speed_vals)):
            val = obj_data[i, j]
            if not np.isnan(val):
                text_color = "white" if val < vmax_obj * 0.65 else "black"
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=10, fontweight="bold", color=text_color)
    fig.colorbar(im1, ax=ax1, shrink=0.85)

    # ------------------------------------------------------------------
    # Top-right: Solve Time heatmap
    # ------------------------------------------------------------------
    ax2 = axes[0, 1]
    time_data = pivot_time.loc[P_vals, speed_vals].values
    im2 = ax2.imshow(time_data, cmap="Blues", aspect="auto", origin="lower")
    ax2.set_xticks(range(len(speed_vals)))
    ax2.set_xticklabels(speed_vals)
    ax2.set_yticks(range(len(P_vals)))
    ax2.set_yticklabels(P_vals)
    ax2.set_xlabel("Vehicle Speed (km/h)", fontsize=12)
    ax2.set_ylabel("P_intervals", fontsize=12)
    ax2.set_title("Solve Time (s)", fontsize=13, fontweight="bold")
    for i in range(len(P_vals)):
        for j in range(len(speed_vals)):
            val = time_data[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f"{val:.1f}", ha="center", va="center",
                         fontsize=10, fontweight="bold")
    fig.colorbar(im2, ax=ax2, shrink=0.85)

    # ------------------------------------------------------------------
    # Bottom-left: Line comparison of speed vs. objective for different P
    # ------------------------------------------------------------------
    ax3 = axes[1, 0]
    for pi, p in enumerate(P_vals):
        sub = df[df["P_intervals"] == p].sort_values("vehicle_speed_kmh")
        ax3.plot(sub["vehicle_speed_kmh"], sub["objective"],
                 marker=markers_line[pi], color=colors_line[pi],
                 linewidth=2.2, markersize=8, label=f"P={p}")
    ax3.set_xlabel("Vehicle Speed (km/h)", fontsize=12)
    ax3.set_ylabel("Total Utility (objective)", fontsize=12)
    ax3.set_title("Impact of Speed on Objective for Different P", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=11, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(speed_vals)

    # ------------------------------------------------------------------
    # Bottom-right: Line comparison of speed vs. Time_Gap for different P
    # ------------------------------------------------------------------
    ax4 = axes[1, 1]
    for pi, p in enumerate(P_vals):
        sub = df[df["P_intervals"] == p].sort_values("vehicle_speed_kmh")
        valid = sub["Time_Gap"].notna()
        if valid.sum() > 0:
            ax4.plot(sub.loc[valid, "vehicle_speed_kmh"],
                     sub.loc[valid, "Time_Gap"],
                     marker=markers_line[pi], color=colors_line[pi],
                     linewidth=2.2, markersize=8, label=f"P={p}")
    ax4.set_xlabel("Vehicle Speed (km/h)", fontsize=12)
    ax4.set_ylabel("Temporal Deviation Time_Gap (h)", fontsize=12)
    ax4.set_title("Impact of Speed on Temporal Deviation for Different P", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=11, framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(speed_vals)
    ax4.axhline(y=0, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)

    # ------------------------------------------------------------------
    # Overall title and save
    # ------------------------------------------------------------------
    fig.suptitle("Cross Sensitivity Analysis: P_intervals × vehicle_speed",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    save_path = os.path.join(_plot_dir, "sensitivity_P_speed_cross.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot saved] {save_path}")


# =========================================================================
# 汇总对比表
# =========================================================================
def print_summary_table(all_dfs: dict, output_dir: str = ""):
    """打印所有扫描的汇总对比表，并保存为 CSV。"""
    print(f"\n{'='*90}")
    print("  敏感性分析汇总")
    print(f"{'='*90}")

    # 收集所有 DataFrame 合并为一个汇总表
    summary_rows = []
    for var_name, df in all_dfs.items():
        if df is None or df.empty:
            print(f"\n  [{var_name}] 无有效数据")
            continue
        print(f"\n--- {var_name} ---")
        cols = [var_name, "objective", "total_swaps", "elapsed_seconds",
                "MIPGap", "Time_Gap", "status"]
        display_cols = [c for c in cols if c in df.columns]
        print(df[display_cols].to_string(index=False))
        # 收集用于 CSV 导出的行
        for _, row in df.iterrows():
            row_data = {"variable": var_name}
            for c in display_cols:
                row_data[c] = row[c]
            summary_rows.append(row_data)

    # 保存为 CSV
    if output_dir and summary_rows:
        csv_path = os.path.join(output_dir, "sensitivity_analysis_summary.csv")
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n  [CSV saved] {csv_path}")


# =========================================================================
# 主入口
# =========================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # =========================================================================
    # CLI 参数解析 (对齐 3_Optimization/batch_runner.py 的接口模式)
    # =========================================================================
    parser = argparse.ArgumentParser(
        description="ST-Graph 优化模型 — 敏感性分析脚本 (One-Factor Sweep)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 4_Sensitivity_Analysis/sensitivity_analysis.py                                      # 使用默认时间
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --datetime "2025/10/28 14:00"        # 指定具体时间
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --random                             # 随机选取可用小时
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --random --seed 42                   # 随机选取 (固定种子)
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --list-hours                         # 列出可用小时
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --data path/to/other.csv             # 指定数据文件
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --skip-cross                         # 跳过交叉项扫描
  python 4_Sensitivity_Analysis/sensitivity_analysis.py --only speed swap_time               # 仅运行指定扫描
        """,
    )
    parser.add_argument(
        "--data", type=str, default=DEFAULT_PREDICTION_FILE,
        help=f"预测数据 CSV 文件路径 (默认: prediction_CB_Hurdle.csv)"
    )
    parser.add_argument(
        "--datetime", type=str, default=None,
        help="目标日期时间, 格式: 'YYYY/MM/DD HH:MM' (例如 '2025/10/28 14:00')"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="从可用小时中随机选取一个 (范围: %s ~ %s)" % (DATETIME_RANGE_START, DATETIME_RANGE_END)
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="配合 --random 使用, 固定随机种子"
    )
    parser.add_argument(
        "--start", type=str, default=DATETIME_RANGE_START,
        help=f"随机选取的起始时间 (默认: {DATETIME_RANGE_START})"
    )
    parser.add_argument(
        "--end", type=str, default=DATETIME_RANGE_END,
        help=f"随机选取的结束时间 (默认: {DATETIME_RANGE_END})"
    )
    parser.add_argument(
        "--list-hours", action="store_true",
        help="列出 CSV 中所有可用小时并退出"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出根目录 (默认: 4_Sensitivity_Analysis/Results/<timestamp>/)"
    )
    parser.add_argument(
        "--skip-cross", action="store_true",
        help="跳过交叉项 (P_intervals × vehicle_speed) 联合扫描"
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        choices=["speed", "swap_time", "C_max", "P_intervals", "cross"],
        help="仅运行指定的扫描项 (可多选)"
    )

    args = parser.parse_args()

    # =========================================================================
    # 模式 1: 列出所有可用小时 (对齐 Pre_Process 的 --list-hours 模式)
    # =========================================================================
    if args.list_hours:
        hours = list_available_hours(
            file_path=args.data,
            start=args.start,
            end=args.end,
        )
        print("=" * 60)
        print(f"  文件: {args.data}")
        print(f"  时间范围: {args.start} ~ {args.end}")
        print(f"  可用小时数: {len(hours)}")
        print("=" * 60)
        for h in hours:
            print(h.strftime("%Y/%m/%d %H:%M"))
        exit(0)

    # =========================================================================
    # 确定目标日期时间 (对齐 batch_runner.py 的 --datetime / --random 模式)
    # =========================================================================
    if args.random:
        TARGET_DATETIME = select_random_datetime(
            file_path=args.data,
            start=args.start,
            end=args.end,
            seed=args.seed,
        )
        print("=" * 60)
        print("  [随机] 敏感性分析 — 随机选取模式")
        if args.seed is not None:
            print(f"  随机种子: {args.seed}")
        print(f"  选择的目标时间: {TARGET_DATETIME}")
        print("=" * 60)
    elif args.datetime is not None:
        TARGET_DATETIME = args.datetime
        print("=" * 60)
        print("  [指定] 敏感性分析 — 用户指定日期时间模式")
        print(f"  选择的目标时间: {TARGET_DATETIME}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  [默认] 敏感性分析 — 默认日期时间模式")
        print(f"  选择的目标时间: {TARGET_DATETIME}")
        print("=" * 60)

    DATA_FILE = args.data

    # =========================================================================
    # 创建本次运行的时间戳输出目录 (对齐 batch_runner.py 的 run_output_dir 模式)
    # =========================================================================
    RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
    _output_root = args.output or OUTPUT_ROOT_DIR
    RUN_OUTPUT_DIR = os.path.join(_output_root, RUN_TIMESTAMP)
    RUN_PLOT_DIR = os.path.join(RUN_OUTPUT_DIR, "Plots")
    os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RUN_PLOT_DIR, exist_ok=True)

    # 确定运行哪些扫描
    if args.only:
        run_scans = set(args.only)
    else:
        run_scans = {"speed", "swap_time", "C_max", "P_intervals", "cross"}
    if args.skip_cross:
        run_scans.discard("cross")

    print("=" * 70)
    print("  ST-Graph 优化模型 — 敏感性分析脚本")
    print(f"  数据文件: {DATA_FILE}")
    print(f"  目标时间: {TARGET_DATETIME}")
    print(f"  运行时间戳: {RUN_TIMESTAMP}")
    print(f"  结果目录: {RUN_OUTPUT_DIR}")
    print(f"  图表目录: {RUN_PLOT_DIR}")
    print(f"  求解时限: {FIXED_TIME_LIMIT_S}s (单次)")
    print(f"  运行项目: {sorted(run_scans)}")
    print("=" * 70)

    all_results = {}

    # ------------------------------------------------------------------
    # 1. 车速敏感性扫描
    # ------------------------------------------------------------------
    if "speed" in run_scans:
        print("\n" + "█" * 70)
        print("  [阶段 1] 车速 (vehicle_speed_kmh) 敏感性扫描")
        print("█" * 70)
        df_speed = sweep_vehicle_speed(
            output_dir=RUN_OUTPUT_DIR, target_datetime=TARGET_DATETIME)
        all_results["vehicle_speed_kmh"] = df_speed
        if df_speed is not None and not df_speed.empty:
            plot_speed_sensitivity(df_speed, plot_dir=RUN_PLOT_DIR)

    # ------------------------------------------------------------------
    # 2. 换电速率敏感性扫描
    # ------------------------------------------------------------------
    if "swap_time" in run_scans:
        print("\n" + "█" * 70)
        print("  [阶段 2] 换电速率 (swap_rate) 敏感性扫描")
        print("█" * 70)
        df_swap = sweep_swap_time(
            output_dir=RUN_OUTPUT_DIR, target_datetime=TARGET_DATETIME)
        all_results["swap_rate"] = df_swap
        if df_swap is not None and not df_swap.empty:
            plot_swap_time_sensitivity(df_swap, plot_dir=RUN_PLOT_DIR)

    # ------------------------------------------------------------------
    # 3. 车载容量敏感性扫描
    # ------------------------------------------------------------------
    if "C_max" in run_scans:
        print("\n" + "█" * 70)
        print("  [阶段 3] 车载容量 (C_max) 敏感性扫描")
        print("█" * 70)
        df_cmax = sweep_C_max(
            output_dir=RUN_OUTPUT_DIR, target_datetime=TARGET_DATETIME)
        all_results["C_max"] = df_cmax
        if df_cmax is not None and not df_cmax.empty:
            plot_C_max_sensitivity(df_cmax, plot_dir=RUN_PLOT_DIR)

    # ------------------------------------------------------------------
    # 4. 时空离散段数敏感性扫描
    # ------------------------------------------------------------------
    if "P_intervals" in run_scans:
        print("\n" + "█" * 70)
        print("  [阶段 4] 时空离散段数 (P_intervals) 敏感性扫描")
        print("█" * 70)
        df_p = sweep_P_intervals(
            output_dir=RUN_OUTPUT_DIR, target_datetime=TARGET_DATETIME)
        all_results["P_intervals"] = df_p
        if df_p is not None and not df_p.empty:
            plot_P_intervals_sensitivity(df_p, plot_dir=RUN_PLOT_DIR)

    # ------------------------------------------------------------------
    # 5. 交叉项: P_intervals × vehicle_speed 联合扫描
    # ------------------------------------------------------------------
    if "cross" in run_scans:
        print("\n" + "█" * 70)
        print("  [阶段 5] 交叉项 (P_intervals × vehicle_speed) 联合扫描")
        print("█" * 70)
        df_cross = sweep_P_speed_cross(
            output_dir=RUN_OUTPUT_DIR, target_datetime=TARGET_DATETIME)
        all_results["P_speed_cross"] = df_cross
        if df_cross is not None and not df_cross.empty:
            plot_P_speed_cross(df_cross, plot_dir=RUN_PLOT_DIR)

    # ------------------------------------------------------------------
    # 汇总输出
    # ------------------------------------------------------------------
    print_summary_table(all_results, output_dir=RUN_OUTPUT_DIR)

    print(f"\n{'='*70}")
    print(f"  敏感性分析全部完成!")
    print(f"  CSV 结果目录: {RUN_OUTPUT_DIR}")
    print(f"  图表目录    : {RUN_PLOT_DIR}")
    print(f"{'='*70}")
