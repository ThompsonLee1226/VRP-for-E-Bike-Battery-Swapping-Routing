#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
时间敏感性分析脚本 — 随机采样 20 个时间戳评估 ST-Graph 模型的时间鲁棒性
=============================================================================

从数据可用范围内随机选取 20 个不同的小时级时间戳，分别运行
3_Optimization/main_STGraph.py 的默认 M1 管线，观察不同时间切片下
优化模型的求解效果波动。

运行方式:
  python 4_Sensitivity_Analysis/temporal_sensitivity.py
  python 4_Sensitivity_Analysis/temporal_sensitivity.py --seed 42
  python 4_Sensitivity_Analysis/temporal_sensitivity.py --n-samples 30
  python 4_Sensitivity_Analysis/temporal_sensitivity.py --data path/to/other.csv

输出结构:
  4_Sensitivity_Analysis/Results/
    ├── temporal_summary_20260728_XXXXXX.csv   # 20 个时间戳的汇总对比
    ├── 20251023_1200/                          # 以时间戳命名的子目录
    │   ├── M1_temporal_*_metrics.json
    │   ├── M1_temporal_*_summary.csv
    │   ├── M1_temporal_*_route.csv
    │   ├── M1_temporal_*_bb_progress.csv
    │   └── M1_temporal_*_route.png
    ├── 20251024_0800/
    │   └── ...
    └── ...

=============================================================================
"""

from __future__ import annotations

import sys
import os
import time
import random
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
    list_available_hours,
)

# =========================================================================
# 全局路径与输出配置
# =========================================================================
DATA_FILE = DEFAULT_PREDICTION_FILE
OUTPUT_ROOT_DIR = os.path.join(SCRIPT_DIR, 'Results')
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

# =========================================================================
# 固定参数 (与 main_STGraph.py M1 默认值保持一致)
# =========================================================================
FIXED_SPEED_KMH = 30.0
FIXED_C_MAX = 20
FIXED_T_TOTAL = 1.0
FIXED_P_INTERVALS = 12
FIXED_SWAP_TIME_C = 0.02
FIXED_MAX_TRAVEL_TIME = 0.2
FIXED_K_NEIGHBORS = 50
FIXED_Y_LEVELS = list(range(1, 11))
FIXED_TIME_LIMIT_S = 1200


def _ts_to_dirname(ts_str: str) -> str:
    """将 'YYYY/MM/DD HH:MM' 格式的时间戳转为安全的目录名 'YYYYMMDD_HHMM'。"""
    return ts_str.replace("/", "").replace(" ", "_").replace(":", "")


def run_temporal_sensitivity(
    n_samples: int = 20,
    seed: int | None = None,
    data_file: str = DATA_FILE,
) -> pd.DataFrame:
    """随机选取 n_samples 个时间戳, 逐一运行 M1 管线, 汇总对比。

    Parameters
    ----------
    n_samples : int
        随机采样的时间戳数量 (默认 20)。
    seed : int or None
        随机种子, 用于复现。
    data_file : str
        预测数据 CSV 文件路径。

    Returns
    -------
    pd.DataFrame
        所有采样点的关键指标汇总表。
    """
    # ---- 获取可用小时池 ----
    all_hours = list_available_hours(
        file_path=data_file,
        start=DATETIME_RANGE_START,
        end=DATETIME_RANGE_END,
    )
    n_available = len(all_hours)
    print(f"可用小时池: {n_available} 个 (范围 {DATETIME_RANGE_START} ~ {DATETIME_RANGE_END})")

    if n_samples > n_available:
        raise ValueError(
            f"请求采样 {n_samples} 个时间戳, 但可用小时仅 {n_available} 个"
        )

    # ---- 随机采样 ----
    rng = random.Random(seed)
    sampled_ts = rng.sample(all_hours, n_samples)
    # 按时间先后排序, 便于观察时间趋势
    sampled_ts = sorted(sampled_ts)

    print(f"随机采样 {n_samples} 个时间戳 (seed={seed}):")
    for i, ts in enumerate(sampled_ts):
        print(f"  [{i+1:2d}] {ts.strftime('%Y/%m/%d %H:%M')}")

    # ---- 创建本次运行的汇总输出目录 ----
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(OUTPUT_ROOT_DIR, f"temporal_batch_{run_timestamp}")
    os.makedirs(batch_dir, exist_ok=True)

    # ---- 逐时间戳运行 ----
    records = []
    n_total = len(sampled_ts)

    print(f"\n{'='*70}")
    print(f"  开始逐时间戳求解 (共 {n_total} 个, 单次时限 {FIXED_TIME_LIMIT_S}s)")
    print(f"  结果目录: {batch_dir}")
    print(f"{'='*70}")

    for idx, ts_dt in enumerate(sampled_ts):
        ts_str = ts_dt.strftime("%Y/%m/%d %H:%M")
        ts_dirname = _ts_to_dirname(ts_str)
        ts_output_dir = os.path.join(batch_dir, ts_dirname)
        os.makedirs(ts_output_dir, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"  [{idx+1}/{n_total}] 时间戳: {ts_str}")
        print(f"  输出子目录: {ts_dirname}")
        print(f"{'─'*60}")

        t_start = time.perf_counter()

        try:
            result = run_optimization_pipeline(
                data_file=data_file,
                target_datetime=ts_str,
                depot_lat=None,
                depot_lon=None,
                vehicle_speed_kmh=FIXED_SPEED_KMH,
                C_max=FIXED_C_MAX,
                T_total=FIXED_T_TOTAL,
                P_intervals=FIXED_P_INTERVALS,
                y_levels=FIXED_Y_LEVELS,
                swap_time_c=FIXED_SWAP_TIME_C,
                max_travel_time=FIXED_MAX_TRAVEL_TIME,
                K_neighbors=FIXED_K_NEIGHBORS,
                output_dir=ts_output_dir,
                verbose=True,
                experiment_id="M1",
                instance_name="temporal",
                geo_fencing=True,
                knn_enabled=True,
                time_limit_s=FIXED_TIME_LIMIT_S,
            )
        except Exception as exc:
            elapsed = round(time.perf_counter() - t_start, 2)
            print(f"  [ERROR] 求解异常: {exc}")
            records.append({
                "datetime": ts_str,
                "status": f"ERROR: {exc}",
                "elapsed_seconds": elapsed,
                "MIPGap": None,
                "objective": None,
                "total_swaps": None,
                "num_visited": None,
                "makespan_hrs": None,
                "utility_soon": None,
                "utility_normal": None,
                "utility_low": None,
            })
            continue

        elapsed = round(time.perf_counter() - t_start, 2)

        # 提取关键指标
        summary = result.get("summary", {})
        collector = result.get("collector")

        mip_gap = None
        if collector is not None:
            try:
                mip_gap = collector.mip_gap_pct
            except Exception:
                pass
        if mip_gap is None or mip_gap == float('inf'):
            if "OPTIMAL" in str(result.get("status", "")):
                mip_gap = 0.0

        record = {
            "datetime": ts_str,
            "status": result.get("status", "UNKNOWN"),
            "elapsed_seconds": elapsed,
            "MIPGap": round(mip_gap, 4) if mip_gap is not None else None,
            "objective": round(result.get("objective", 0) or 0, 6),
            "total_swaps": summary.get("total_swaps", 0),
            "num_visited": len(result.get("visited_grids", [])),
            "makespan_hrs": round(summary.get("makespan_hrs", 0), 4),
            "utility_soon": summary.get("utility_soon", 0),
            "utility_normal": summary.get("utility_normal", 0),
            "utility_low": summary.get("utility_low", 0),
        }
        records.append(record)

        print(f"  >> Status={record['status']} | "
              f"Obj={record['objective']:.4f} | "
              f"Swaps={record['total_swaps']} | "
              f"Visited={record['num_visited']} | "
              f"Gap={record['MIPGap']} | "
              f"Time={elapsed}s")

    # ---- 汇总 DataFrame ----
    df = pd.DataFrame(records)

    # 保存汇总 CSV
    summary_csv = os.path.join(batch_dir, f"temporal_summary_{run_timestamp}.csv")
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\n{'='*70}")
    print(f"  时间敏感性分析完成!")
    print(f"  汇总 CSV: {summary_csv}")
    print(f"{'='*70}")

    # 打印简要统计
    successful = df[~df["status"].str.contains("ERROR|SKIPPED|INFEASIBLE", na=False)]
    if not successful.empty:
        print(f"\n  成功求解: {len(successful)}/{n_total}")
        print(f"  Objective 均值: {successful['objective'].mean():.4f}")
        print(f"  Objective 标准差: {successful['objective'].std():.4f}")
        print(f"  Objective 范围: [{successful['objective'].min():.4f}, {successful['objective'].max():.4f}]")
        print(f"  平均求解耗时: {successful['elapsed_seconds'].mean():.1f}s")
        print(f"  平均访问节点数: {successful['num_visited'].mean():.1f}")
        print(f"  平均换电量: {successful['total_swaps'].mean():.1f}")

    return df


# =========================================================================
# 主入口
# =========================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = argparse.ArgumentParser(
        description="ST-Graph 时间敏感性分析 — 随机采样多时间戳评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 4_Sensitivity_Analysis/temporal_sensitivity.py                        # 默认: 20 个随机时间戳
  python 4_Sensitivity_Analysis/temporal_sensitivity.py --seed 42              # 固定种子
  python 4_Sensitivity_Analysis/temporal_sensitivity.py --n-samples 30         # 采样 30 个
  python 4_Sensitivity_Analysis/temporal_sensitivity.py --data other.csv       # 指定数据文件
        """,
    )
    parser.add_argument(
        "--data", type=str, default=DATA_FILE,
        help=f"预测数据 CSV 文件路径 (默认: prediction_CB_Hurdle.csv)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="随机采样的时间戳数量 (默认: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子, 用于结果复现"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  ST-Graph 优化模型 — 时间敏感性分析")
    print(f"  数据文件: {args.data}")
    print(f"  采样数量: {args.n_samples}")
    print(f"  随机种子: {args.seed}")
    print(f"  可用时间范围: {DATETIME_RANGE_START} ~ {DATETIME_RANGE_END}")
    print(f"  求解时限: {FIXED_TIME_LIMIT_S}s (单次)")
    print(f"  结果根目录: {OUTPUT_ROOT_DIR}")
    print("=" * 70)

    df = run_temporal_sensitivity(
        n_samples=args.n_samples,
        seed=args.seed,
        data_file=args.data,
    )
