"""
=============================================================================
批量实验执行器 (Batch Experiment Runner)
=============================================================================
一键运行全部 5 个实验组 (M1-M5) 的完整对比实验管线。

功能:
  - 按实验配置自动运行各模型变体
  - 统一采集全维度指标 (计算效率 / 解质量 / 时序保真度 / Soon-Normal分流)
  - 自动生成汇总对比 CSV 表格
  - 支持多实例 / 多时间段批量运行

用法:
  python batch_runner.py                          # 运行全部实验组 (默认实例)
  python batch_runner.py --groups M1 M3 M5        # 仅运行指定实验组
  python batch_runner.py --skip M2c               # 跳过可选消融组
  python batch_runner.py --instances small_01 small_02  # 指定实例列表
  python batch_runner.py --aligned                 # 使用对齐参数 (P=10, MIPGap=0.05)
  python batch_runner.py --random --seed 42       # 批量实验随机选取
  python 3_Optimization/batch_runner.py --datetime "2025/11/05 18:00"  # 批量实验指定时间( 可选时间范围：2025/10/24 00:00 ~ 2025/11/10 23:00，共 432 个可用小时。)
  =============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd

# 确保当前目录在 Python 路径中
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from experiment_config import (
    EXPERIMENT_GROUPS, SHARED_PARAMS, ALIGNED_PARAMS,
    get_experiment_config, list_experiments,
)
from experiment_utils import merge_batch_results
from Pre_Process import (
    DEFAULT_TARGET_DATETIME,
    DEFAULT_PREDICTION_FILE,
    DATETIME_RANGE_START,
    DATETIME_RANGE_END,
    select_random_datetime,
    list_available_hours,
)


# =============================================================================
# 默认运行配置
# =============================================================================
DEFAULT_DATA_FILE = DEFAULT_PREDICTION_FILE  # 默认使用 CB_Hurdle 预测数据
DEFAULT_OUTPUT_DIR = os.path.join(_current_dir, "Optimization_Result_Summary")
DEFAULT_OUTPUT_DIR = os.path.normpath(DEFAULT_OUTPUT_DIR)

# 默认运行全部实验组 (不包括可选的 M2c)
DEFAULT_GROUPS = ["M1", "M2a", "M2b", "M3", "M4", "M5", "M6"]

# M2c 是可选的极端消融组, 仅在显式指定时运行
OPTIONAL_GROUPS = ["M2c"]


# =============================================================================
# 实验运行器类
# =============================================================================
class BatchRunner:
    """管理批量实验的完整生命周期。"""

    def __init__(self, data_file: str = DEFAULT_DATA_FILE,
                 output_dir: str = DEFAULT_OUTPUT_DIR,
                 verbose: bool = True,
                 use_aligned_params: bool = False,
                 target_datetime: str = DEFAULT_TARGET_DATETIME):
        self.data_file = data_file
        self.output_dir = output_dir
        self.verbose = verbose
        self.use_aligned_params = use_aligned_params
        self.target_datetime = target_datetime
        self.results: list[dict] = []
        self.run_summary: list[dict] = []

    # -------------------------------------------------------------------------
    # 各实验组的运行方法
    # -------------------------------------------------------------------------

    def _run_stgraph_variant(self, experiment_id: str,
                             instance_name: str = "default",
                             **override_kwargs):
        """运行 STGraph 系列 (M1, M2a, M2b, M2c)。"""
        from main_STGraph import run_optimization_pipeline

        cfg = get_experiment_config(experiment_id)
        if self.use_aligned_params:
            cfg["P_intervals"] = ALIGNED_PARAMS["P_intervals"]
            cfg["mip_gap"] = ALIGNED_PARAMS["mip_gap"]
            cfg["time_limit_s"] = ALIGNED_PARAMS["time_limit_s"]

        params = {
            "data_file": self.data_file,
            "target_datetime": self.target_datetime,
            "vehicle_speed_kmh": cfg["vehicle_speed_kmh"],
            "C_max": cfg["C_max"],
            "T_total": cfg["T_total"],
            "P_intervals": cfg["P_intervals"],
            "y_levels": cfg["y_levels"],
            "swap_time_c": cfg["swap_time_c"],
            "max_travel_time": cfg["max_travel_time"],
            "K_neighbors": cfg["K_neighbors"],
            "output_dir": self.output_dir,
            "verbose": self.verbose,
            "experiment_id": experiment_id,
            "instance_name": instance_name,
            "geo_fencing": cfg["geo_fencing"],
            "knn_enabled": cfg["knn_enabled"],
            "time_limit_s": cfg["time_limit_s"],
        }
        params.update(override_kwargs)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  {cfg['name_en']} ({experiment_id})")
            print(f"  Role: {cfg['academic_role']}")
            print(f"  P={cfg['P_intervals']} | MIPGap={cfg['mip_gap']} | "
                  f"GeoFence={cfg['geo_fencing']} | KNN={cfg['knn_enabled']}")
            print(f"  Target Datetime: {self.target_datetime}")
            print(f"{'='*70}")

        t0 = time.perf_counter()
        try:
            result = run_optimization_pipeline(**params)
            elapsed = time.perf_counter() - t0
            self._record_result(experiment_id, instance_name, result, elapsed)
        except Exception as e:
            print(f"  [ERROR] {experiment_id} failed to run: {e}")
            import traceback
            traceback.print_exc()
            self._record_failure(experiment_id, instance_name, str(e))

    def _run_sos2(self, experiment_id: str = "M3",
                  instance_name: str = "default",
                  **override_kwargs):
        """运行 SOS2-PLA 模型 (M3)。"""
        from main_SOS2 import run_optimization_pipeline

        cfg = get_experiment_config(experiment_id)
        if self.use_aligned_params:
            cfg["P_intervals"] = ALIGNED_PARAMS["P_intervals"]
            cfg["mip_gap"] = ALIGNED_PARAMS["mip_gap"]
            cfg["time_limit_s"] = ALIGNED_PARAMS["time_limit_s"]

        params = {
            "data_file": self.data_file,
            "target_datetime": self.target_datetime,
            "vehicle_speed_kmh": cfg["vehicle_speed_kmh"],
            "C_max": cfg["C_max"],
            "T_total": cfg["T_total"],
            "P_intervals": cfg["P_intervals"],
            "y_levels": cfg["y_levels"],
            "swap_time_c": cfg["swap_time_c"],
            "BigM": 200.0,
            "max_travel_time": cfg["max_travel_time"],
            "output_dir": self.output_dir,
            "verbose": self.verbose,
            "experiment_id": experiment_id,
            "instance_name": instance_name,
            "geo_fencing": cfg["geo_fencing"],
            "time_limit_s": cfg["time_limit_s"],
        }
        params.update(override_kwargs)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  {cfg['name_en']} ({experiment_id})")
            print(f"  Role: {cfg['academic_role']}")
            print(f"  P={cfg['P_intervals']} | MIPGap={cfg['mip_gap']} | "
                  f"Method=Primal")
            print(f"  Target Datetime: {self.target_datetime}")
            print(f"{'='*70}")

        t0 = time.perf_counter()
        try:
            result = run_optimization_pipeline(**params)
            elapsed = time.perf_counter() - t0
            self._record_result(experiment_id, instance_name, result, elapsed)
        except Exception as e:
            print(f"  [ERROR] {experiment_id} failed to run: {e}")
            import traceback
            traceback.print_exc()
            self._record_failure(experiment_id, instance_name, str(e))

    def _run_delta(self, experiment_id: str = "M4",
                   instance_name: str = "default",
                   **override_kwargs):
        """运行 Delta-PLA 模型 (M4)。"""
        from main_delta import run_optimization_pipeline

        cfg = get_experiment_config(experiment_id)
        if self.use_aligned_params:
            cfg["P_intervals"] = ALIGNED_PARAMS["P_intervals"]
            cfg["mip_gap"] = ALIGNED_PARAMS["mip_gap"]
            cfg["time_limit_s"] = ALIGNED_PARAMS["time_limit_s"]

        params = {
            "data_file": self.data_file,
            "target_datetime": self.target_datetime,
            "vehicle_speed_kmh": cfg["vehicle_speed_kmh"],
            "C_max": cfg["C_max"],
            "T_total": cfg["T_total"],
            "P_intervals": cfg["P_intervals"],
            "y_levels": cfg["y_levels"],
            "swap_time_c": cfg["swap_time_c"],
            "BigM": 200.0,
            "max_travel_time": cfg["max_travel_time"],
            "output_dir": self.output_dir,
            "verbose": self.verbose,
            "experiment_id": experiment_id,
            "instance_name": instance_name,
            "geo_fencing": cfg["geo_fencing"],
            "time_limit_s": cfg["time_limit_s"],
        }
        params.update(override_kwargs)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  {cfg['name_en']} ({experiment_id})")
            print(f"  Role: {cfg['academic_role']}")
            print(f"  P={cfg['P_intervals']} | MIPGap={cfg['mip_gap']} | "
                  f"Cuts={cfg['cuts']}")
            print(f"  Target Datetime: {self.target_datetime}")
            print(f"{'='*70}")

        t0 = time.perf_counter()
        try:
            result = run_optimization_pipeline(**params)
            elapsed = time.perf_counter() - t0
            self._record_result(experiment_id, instance_name, result, elapsed)
        except Exception as e:
            print(f"  [ERROR] {experiment_id} failed to run: {e}")
            import traceback
            traceback.print_exc()
            self._record_failure(experiment_id, instance_name, str(e))

    def _run_delta_mcf(self, experiment_id: str = "M5",
                       instance_name: str = "default",
                       **override_kwargs):
        """运行 Delta-MCF 基准模型 (M5)。"""
        from main_delta_MCF import run_optimization_pipeline

        cfg = get_experiment_config(experiment_id)
        if self.use_aligned_params:
            cfg["P_intervals"] = ALIGNED_PARAMS["P_intervals"]
            cfg["mip_gap"] = ALIGNED_PARAMS["mip_gap"]
            cfg["time_limit_s"] = ALIGNED_PARAMS["time_limit_s"]

        params = {
            "data_file": self.data_file,
            "target_datetime": self.target_datetime,
            "vehicle_speed_kmh": cfg["vehicle_speed_kmh"],
            "C_max": cfg["C_max"],
            "T_total": cfg["T_total"],
            "P_intervals": cfg["P_intervals"],
            "y_levels": cfg["y_levels"],
            "swap_time_c": cfg["swap_time_c"],
            "BigM": 200.0,
            "max_travel_time": cfg["max_travel_time"],
            "output_dir": self.output_dir,
            "verbose": self.verbose,
            "experiment_id": experiment_id,
            "instance_name": instance_name,
            "geo_fencing": cfg["geo_fencing"],
            "time_limit_s": cfg["time_limit_s"],
        }
        params.update(override_kwargs)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  {cfg['name_en']} ({experiment_id})")
            print(f"  Role: {cfg['academic_role']}")
            print(f"  P={cfg['P_intervals']} | MIPGap={cfg['mip_gap']} | "
                  f"LazyMTZ={'Yes' if cfg.get('lazy_constraints') else 'No'}")
            print(f"  Target Datetime: {self.target_datetime}")
            print(f"{'='*70}")

        t0 = time.perf_counter()
        try:
            result = run_optimization_pipeline(**params)
            elapsed = time.perf_counter() - t0
            self._record_result(experiment_id, instance_name, result, elapsed)
        except Exception as e:
            print(f"  [ERROR] {experiment_id} failed to run: {e}")
            import traceback
            traceback.print_exc()
            self._record_failure(experiment_id, instance_name, str(e))

    def _run_heuristic(self, experiment_id: str = "M6",
                       instance_name: str = "default",
                       **override_kwargs):
        """运行元启发式基线 (M6)。"""
        from main_heuristic import run_heuristic_pipeline

        cfg = get_experiment_config(experiment_id)

        params = {
            "data_file": self.data_file,
            "target_datetime": self.target_datetime,
            "vehicle_speed_kmh": cfg["vehicle_speed_kmh"],
            "C_max": cfg["C_max"],
            "T_total": cfg["T_total"],
            "P_intervals": cfg["P_intervals"],
            "y_levels": cfg["y_levels"],
            "swap_time_c": cfg["swap_time_c"],
            "max_travel_time": cfg["max_travel_time"],
            "K_neighbors": cfg["K_neighbors"],
            "output_dir": self.output_dir,
            "verbose": self.verbose,
            "experiment_id": experiment_id,
            "instance_name": instance_name,
        }
        params.update(override_kwargs)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  {cfg['name_en']} ({experiment_id})")
            print(f"  Role: {cfg['academic_role']}")
            print(f"  P={cfg['P_intervals']} | KNN={cfg['knn_enabled']} (K={cfg['K_neighbors']})")
            print(f"  Target Datetime: {self.target_datetime}")
            print(f"{'='*70}")

        t0 = time.perf_counter()
        try:
            result = run_heuristic_pipeline(**params)
            elapsed = time.perf_counter() - t0
            self._record_result(experiment_id, instance_name, result, elapsed)
        except Exception as e:
            print(f"  [ERROR] {experiment_id} failed to run: {e}")
            import traceback
            traceback.print_exc()
            self._record_failure(experiment_id, instance_name, str(e))

    # -------------------------------------------------------------------------
    # 结果记录与汇总
    # -------------------------------------------------------------------------

    def _record_result(self, experiment_id: str, instance_name: str,
                       result: dict, total_elapsed_s: float):
        """从 result dict 中提取 MetricsCollector 数据并记录。"""
        collector = result.get("collector")
        if collector is not None:
            metrics = collector.to_dict()
            metrics["total_elapsed_s"] = round(total_elapsed_s, 2)
        else:
            # 降级: 手动提取关键字段
            metrics = {
                "experiment_id": experiment_id,
                "instance_name": instance_name,
                "solve_status": result.get("status", "UNKNOWN"),
                "objective_value": result.get("objective", 0.0) or 0.0,
                "elapsed_seconds": result.get("elapsed_seconds", 0.0),
                "total_elapsed_s": round(total_elapsed_s, 2),
            }

        self.results.append(metrics)

        # 打印一行关键摘要
        status = metrics.get("solve_status", "?")
        obj = metrics.get("objective_value", 0.0)
        cpu = metrics.get("cpu_time_s", 0.0)
        gap = metrics.get("mip_gap_pct", float('inf'))
        visited = metrics.get("num_visited_grids", 0)
        soon_r = metrics.get("soon_ratio", 0.0)
        print(f"  [✓] {experiment_id:4s} | Status={status:12s} | "
              f"Obj={obj:.4f} | CPU={cpu:.1f}s | Gap={gap:.2f}% | "
              f"Visited={visited} | SoonRatio={soon_r:.1%}")

    def _record_failure(self, experiment_id: str, instance_name: str,
                        error_msg: str):
        """记录失败的实验。"""
        self.results.append({
            "experiment_id": experiment_id,
            "instance_name": instance_name,
            "solve_status": "FAILED",
            "error": error_msg,
        })

    def run_experiment(self, experiment_id: str, instance_name: str = "default",
                       **override_kwargs):
        """根据 experiment_id 分发到对应的运行方法。"""
        cfg = get_experiment_config(experiment_id)
        optimizer = cfg.get("optimizer", "")

        if optimizer == "optimize_evrp_with_stgraph":
            self._run_stgraph_variant(experiment_id, instance_name,
                                      **override_kwargs)
        elif optimizer == "optimize_evrp_with_pla":
            self._run_sos2(experiment_id, instance_name, **override_kwargs)
        elif optimizer == "optimize_evrp_with_pla_delta":
            self._run_delta(experiment_id, instance_name, **override_kwargs)
        elif optimizer == "optimize_evrp_with_pla_delta_mcf":
            self._run_delta_mcf(experiment_id, instance_name, **override_kwargs)
        elif optimizer == "run_heuristic":
            self._run_heuristic(experiment_id, instance_name, **override_kwargs)
        else:
            raise ValueError(f"Unknown optimizer '{optimizer}' for {experiment_id}")

    def run_all(self, groups: list[str] | None = None,
                instance_name: str = "default",
                **override_kwargs):
        """依次运行所有指定实验组。"""
        if groups is None:
            groups = DEFAULT_GROUPS

        print(f"\n{'#'*70}")
        print(f"#  Batch experiment start")
        print(f"#  Experiment groups: {groups}")
        print(f"#  Data file: {self.data_file}")
        print(f"#  Target datetime: {self.target_datetime}")
        print(f"#  Output directory: {self.output_dir}")
        print(f"#  Alignment mode: {'ON' if self.use_aligned_params else 'OFF (each model uses its optimal config)'}")
        print(f"{'#'*70}")

        t_start = time.perf_counter()
        for i, eid in enumerate(groups, 1):
            print(f"\n--- [{i}/{len(groups)}] {eid} ---")
            self.run_experiment(eid, instance_name, **override_kwargs)

        total_time = time.perf_counter() - t_start
        print(f"\n{'#'*70}")
        print(f"#  Batch experiment complete! Total time: {total_time:.0f}s "
              f"({total_time/60:.1f}min)")
        print(f"{'#'*70}")

        # 生成汇总表
        self._export_summary(instance_name)

    def _export_summary(self, instance_name: str):
        """导出汇总 CSV 并打印对比表格。"""
        if not self.results:
            print("  [WARNING] No experiment records to export.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"batch_summary_{instance_name}_{timestamp}"

        # 完整指标 CSV
        full_path = os.path.join(self.output_dir, f"{prefix}_full.csv")
        merge_batch_results(self.results, full_path)

        # 精简对比表 CSV (仅核心列)
        core_cols = [
            "experiment_id", "solve_status",
            "objective_value", "cpu_time_s", "mip_gap_pct",
            "num_visited_grids", "total_swaps", "makespan_hrs",
            "utility_per_hour", "soon_ratio",
            "num_vars", "num_constrs", "bb_nodes",
            "original_grid_count", "active_grid_count",
            "feasible_arc_count", "num_st_arcs",
        ]
        available_cols = [c for c in core_cols if c in self.results[0]]
        core_df = pd.DataFrame(self.results)[available_cols]
        core_path = os.path.join(self.output_dir, f"{prefix}_core.csv")
        core_df.to_csv(core_path, index=False)

        # 打印对比表格
        print(f"\n{'='*100}")
        print(f"  Experiment Comparison Summary")
        print(f"{'='*100}")
        _print_comparison_table(self.results)
        print(f"\n  Full metrics: {full_path}")
        print(f"  Core metrics: {core_path}")


# =============================================================================
# 对比表格美化打印
# =============================================================================
def _print_comparison_table(results: list[dict]):
    """在终端打印格式化的对比表格。"""
    if not results:
        return

    # 表头
    header = (f"{'ID':5s} {'Status':12s} {'ObjVal':>10s} {'CPU(s)':>8s} "
              f"{'Gap%':>7s} {'Visit':>6s} {'Swaps':>6s} "
              f"{'Soon%':>7s} {'Vars':>7s} {'Constrs':>7s} {'B&B':>8s}")
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        eid = r.get("experiment_id", "?")
        status = r.get("solve_status", "?")[:12]
        obj = r.get("objective_value", 0) or 0
        cpu = r.get("cpu_time_s", 0) or 0
        gap = r.get("mip_gap_pct", float('inf'))
        visited = r.get("num_visited_grids", 0) or 0
        swaps = r.get("total_swaps", 0) or 0
        soon_r = r.get("soon_ratio", 0) or 0
        nvars = r.get("num_vars", 0) or 0
        nconstrs = r.get("num_constrs", 0) or 0
        bb = int(r.get("bb_nodes", 0) or 0)

        gap_str = f"{gap:.2f}" if gap != float('inf') else "N/A"
        print(f"{eid:5s} {status:12s} {obj:10.4f} {cpu:8.1f} "
              f"{gap_str:>7s} {visited:6d} {swaps:6d} "
              f"{soon_r:6.1%} {nvars:7d} {nconstrs:7d} {bb:8d}")
    print(sep)


# =============================================================================
# CLI 入口
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Batch experiment runner — E-Bike Battery Swapping Routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_runner.py                                                 # run all M1-M5 experiment groups (default time)
  python batch_runner.py --datetime "2025/11/02 12:00"                   # specify a concrete time
  python batch_runner.py --random                                        # randomly select an available hour
  python batch_runner.py --random --seed 42                              # random selection (fixed seed)
  python batch_runner.py --list-hours                                    # list available hours
  python batch_runner.py --groups M1 M3 M5                               # run only the specified experiment groups
  python batch_runner.py --skip M2c                                      # skip the optional ablation group
  python batch_runner.py --aligned                                       # use aligned parameters
  python batch_runner.py --data my_data.csv                              # specify a data file
        """,
    )
    parser.add_argument("--groups", nargs="+", default=None,
                        help="IDs of the experiment groups to run (default: M1 M2a M2b M3 M4 M5)")
    parser.add_argument("--skip", nargs="+", default=[],
                        help="IDs of experiment groups to skip (e.g., M2c)")
    parser.add_argument("--include-optional", action="store_true",
                        help="include the optional ablation group M2c")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_FILE,
                        help=f"data file path (default: prediction_CB_Hurdle.csv)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--instance", type=str, default="default",
                        help="instance name (used for naming output files)")
    parser.add_argument("--aligned", action="store_true",
                        help="use the aligned parameter set (P=10, MIPGap=0.05) for a fair comparison")
    parser.add_argument("--quiet", action="store_true",
                        help="reduce output")
    parser.add_argument("--list", action="store_true",
                        help="list all experiment group configurations and exit")
    # ---- 日期时间选择 ----
    parser.add_argument(
        "--datetime", type=str, default=None,
        help="target datetime, format: 'YYYY/MM/DD HH:MM' (e.g., '2025/11/02 12:00')"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="randomly select one from the available hours (range: %s ~ %s)" % (DATETIME_RANGE_START, DATETIME_RANGE_END)
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="use together with --random to fix the random seed"
    )
    parser.add_argument(
        "--start", type=str, default=DATETIME_RANGE_START,
        help=f"start time for the random selection (default: {DATETIME_RANGE_START})"
    )
    parser.add_argument(
        "--end", type=str, default=DATETIME_RANGE_END,
        help=f"end time for the random selection (default: {DATETIME_RANGE_END})"
    )
    parser.add_argument(
        "--list-hours", action="store_true",
        help="list all available hours in the CSV and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Experiment Groups:")
        print("=" * 80)
        for e in list_experiments():
            print(f"  {e['ID']:5s}  {e['Name']:35s}  P={e['P']}  "
                  f"Gap={e['MIPGap']}  GF={e['Geo-Fence']}  "
                  f"KNN={e['KNN']}  [{e['Role']}]")
        print()
        return

    # ---- 列出可用小时模式 ----
    if args.list_hours:
        hours = list_available_hours(
            file_path=args.data,
            start=args.start,
            end=args.end,
        )
        print("=" * 60)
        print(f"  File: {args.data}")
        print(f"  Time range: {args.start} ~ {args.end}")
        print(f"  Available hours: {len(hours)}")
        print("=" * 60)
        for h in hours:
            print(h.strftime("%Y/%m/%d %H:%M"))
        return

    # ---- 确定目标日期时间 ----
    if args.random:
        target_datetime = select_random_datetime(
            file_path=args.data,
            start=args.start,
            end=args.end,
            seed=args.seed,
        )
        print("=" * 60)
        print("  [Random] Batch experiment — random selection mode")
        if args.seed is not None:
            print(f"  Random seed: {args.seed}")
        print(f"  Selected target datetime: {target_datetime}")
        print("=" * 60)
    elif args.datetime is not None:
        target_datetime = args.datetime
        print("=" * 60)
        print("  [Specified] Batch experiment — user-specified datetime mode")
        print(f"  Selected target datetime: {target_datetime}")
        print("=" * 60)
    else:
        target_datetime = DEFAULT_TARGET_DATETIME
        print("=" * 60)
        print("  [Default] Batch experiment — default datetime mode")
        print(f"  Selected target datetime: {target_datetime}")
        print("=" * 60)

    # 确定实验组列表
    if args.groups:
        groups = args.groups
    else:
        groups = list(DEFAULT_GROUPS)

    # 添加可选的 M2c
    if args.include_optional:
        for g in OPTIONAL_GROUPS:
            if g not in groups:
                groups.append(g)

    # 移除跳过的组
    for g in args.skip:
        if g in groups:
            groups.remove(g)

    # 验证所有 group ID 有效
    for g in groups:
        if g not in EXPERIMENT_GROUPS:
            print(f"[ERROR] Unknown experiment group: {g}")
            print(f"  Available options: {list(EXPERIMENT_GROUPS.keys())}")
            sys.exit(1)

    # 每次运行创建时间戳子文件夹，本次运行全部结果保存于此
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)

    runner = BatchRunner(
        data_file=args.data,
        output_dir=run_output_dir,
        verbose=not args.quiet,
        use_aligned_params=args.aligned,
        target_datetime=target_datetime,
    )
    runner.run_all(groups=groups, instance_name=args.instance)


if __name__ == "__main__":
    main()
