"""
=============================================================================
共享实验工具模块 (Shared Experiment Utilities)
=============================================================================
为所有实验组 (M1-M5) 提供统一的:
  - GurobiProgressTracker (列表记录版, 支持导出 CSV)
  - MetricsCollector (全维度指标采集器)
  - 统一结果导出 (JSON + CSV)
  - LP 松弛值采集辅助
  - Soon/Normal 效用分流计算

作者: 本科研究项目 — E-Bike Battery Swapping Routing Optimization
=============================================================================
"""

from __future__ import annotations

import json
import os
import time
import numpy as np
import pandas as pd
from gurobipy import GRB


# =============================================================================
# 统一进度追踪器 (列表记录版 — 所有模型共用)
# =============================================================================
class GurobiProgressTracker:
    """实时捕获 B&B 搜索树收敛边界, 支持全量记录与导出。

    所有实验组 (M1-M5) 统一使用此类, 取代原先各文件中不一致的实现。
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.records: list = []          # [(runtime, best_obj, best_bound, node_count), ...]
        self.first_feasible_time: float | None = None
        self.root_lp_value: float | None = None

    def record(self, runtime: float, best_obj: float, best_bound: float,
               node_count: int):
        self.records.append((runtime, best_obj, best_bound, node_count))

    def record_first_feasible(self, runtime: float):
        if self.first_feasible_time is None:
            self.first_feasible_time = runtime

    def gap(self) -> float:
        if not self.records:
            return float('inf')
        _, obj, bnd, _ = self.records[-1]
        if obj is None or bnd is None:
            return float('inf')
        return abs(bnd - obj) / (abs(obj) + 1e-9) * 100

    def final_obj(self) -> float | None:
        if not self.records:
            return None
        return self.records[-1][1]

    def final_bound(self) -> float | None:
        if not self.records:
            return None
        return self.records[-1][2]

    def final_nodes(self) -> int:
        if not self.records:
            return 0
        return self.records[-1][3]

    def summary(self) -> str:
        if not self.records:
            return f"[{self.label}] No progress recorded."
        last = self.records[-1]
        return (
            f"[{self.label}] Runtime={last[0]:.1f}s | "
            f"Obj={last[1]:.4f} | Bound={last[2]:.4f} | "
            f"Gap={self.gap():.2f}% | Nodes={last[3]}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """将进度记录导出为 DataFrame, 方便批量分析。"""
        return pd.DataFrame(self.records, columns=[
            "runtime_s", "best_obj", "best_bound", "node_count"
        ])

    def to_dict(self) -> dict:
        """导出关键汇总指标为字典。"""
        return {
            "label": self.label,
            "num_records": len(self.records),
            "final_runtime_s": self.records[-1][0] if self.records else None,
            "final_obj": self.final_obj(),
            "final_bound": self.final_bound(),
            "final_gap_pct": round(self.gap(), 4),
            "final_nodes": self.final_nodes(),
            "first_feasible_time_s": self.first_feasible_time,
            "root_lp_value": self.root_lp_value,
        }


# =============================================================================
# 全维度指标采集器 (Metrics Collector)
# =============================================================================
class MetricsCollector:
    """汇总单次实验运行的全部关键指标, 生成可供论文直接使用的数据。

    使用方式:
        collector = MetricsCollector(experiment_id="M1", instance_name="small_01")
        # ... 运行优化 ...
        collector.record_solve(progress_tracker, elapsed_seconds, model)
        collector.record_solution(result)
        collector.record_temporal_fidelity(result)
        summary = collector.to_dict()
    """

    def __init__(self, experiment_id: str = "", instance_name: str = "",
                 config: dict | None = None):
        self.experiment_id = experiment_id
        self.instance_name = instance_name
        self.config = config or {}

        # 求解效率
        self.cpu_time_s: float = 0.0
        self.mip_gap_pct: float = float('inf')
        self.best_obj: float | None = None
        self.best_bound: float | None = None
        self.root_lp_value: float | None = None
        self.bb_nodes: int = 0
        self.first_feasible_time_s: float | None = None
        self.solve_status: str = "UNKNOWN"

        # 模型规模
        self.num_vars: int = 0
        self.num_constrs: int = 0
        self.num_binary: int = 0
        self.num_continuous: int = 0
        self.num_integer: int = 0

        # 解质量
        self.objective_value: float = 0.0
        self.num_visited_grids: int = 0
        self.total_swaps: int = 0
        self.total_travel_time_hrs: float = 0.0
        self.total_service_time_hrs: float = 0.0
        self.makespan_hrs: float = 0.0
        self.route_length: int = 0
        self.utility_per_hour: float = 0.0
        self.avg_utility_per_grid: float = 0.0

        # Soon/Normal 收益分流
        self.utility_soon: float = 0.0
        self.utility_normal: float = 0.0
        self.utility_low: float = 0.0
        self.soon_ratio: float = 0.0        # Soon 收益占总收益比例

        # 时序保真度
        self.mean_time_deviation_hrs: float = 0.0
        self.max_time_deviation_hrs: float = 0.0
        self.nodes_with_deviation: int = 0
        self.total_deviation_nodes: int = 0

        # 空间剪枝统计
        self.original_grid_count: int = 0
        self.active_grid_count: int = 0
        self.removed_zero_utility: int = 0
        self.feasible_arc_count: int = 0
        self.total_possible_arc_count: int = 0
        self.geo_fencing_enabled: bool = True
        self.knn_enabled: bool = True
        self.knn_k_value: int = 0

        # 模型特有
        self.num_st_arcs: int = 0           # STGraph 时空弧数
        self.lazy_mtz_count: int = 0         # Delta-MCF Lazy MTZ 回调次数

    def record_solve(self, progress: GurobiProgressTracker, elapsed_s: float,
                     model=None):
        """从进度追踪器和 Gurobi 模型采集求解效率指标。"""
        self.cpu_time_s = elapsed_s
        self.mip_gap_pct = progress.gap()
        self.best_obj = progress.final_obj()
        self.best_bound = progress.final_bound()
        self.bb_nodes = progress.final_nodes()
        self.first_feasible_time_s = progress.first_feasible_time
        self.root_lp_value = progress.root_lp_value

        if model is not None:
            try:
                self.solve_status = {
                    GRB.OPTIMAL: "OPTIMAL",
                    GRB.SUBOPTIMAL: "SUBOPTIMAL",
                    GRB.TIME_LIMIT: "TIME_LIMIT",
                    GRB.INFEASIBLE: "INFEASIBLE",
                    GRB.INF_OR_UNBD: "INF_OR_UNBD",
                    GRB.UNBOUNDED: "UNBOUNDED",
                }.get(model.Status, f"CODE_{model.Status}")
            except Exception:
                self.solve_status = "ERROR"

            try:
                self.num_vars = model.NumVars
                self.num_constrs = model.NumConstrs
                self.num_binary = model.NumBinVars
                self.num_continuous = model.NumVars - model.NumBinVars - model.NumIntVars
                self.num_integer = model.NumIntVars
            except Exception:
                pass

    def record_solution(self, result: dict):
        """从解析后的结果字典采集解质量指标。"""
        self.objective_value = result.get("objective", 0.0) or 0.0
        self.num_visited_grids = len(result.get("visited_grids", []))
        summary = result.get("summary", {})
        self.total_swaps = summary.get("total_swaps", 0)
        self.total_travel_time_hrs = summary.get("total_travel_time_hrs", 0.0)
        self.total_service_time_hrs = summary.get("total_service_time_hrs", 0.0)
        self.makespan_hrs = summary.get("makespan_hrs", 0.0)
        self.route_length = summary.get("route_length", 0)

        if self.makespan_hrs > 0:
            self.utility_per_hour = self.objective_value / self.makespan_hrs
        if self.num_visited_grids > 0:
            self.avg_utility_per_grid = self.objective_value / self.num_visited_grids

        # Soon/Normal 分流
        self.utility_soon = summary.get("utility_soon", 0.0)
        self.utility_normal = summary.get("utility_normal", 0.0)
        self.utility_low = summary.get("utility_low", 0.0)
        total_split = self.utility_soon + self.utility_normal + self.utility_low
        if total_split > 0:
            self.soon_ratio = self.utility_soon / total_split

        # 空间剪枝
        pruning = result.get("pruning_stats", {})
        self.original_grid_count = pruning.get("original_grids", 0)
        self.active_grid_count = pruning.get("active_grids", 0)
        self.removed_zero_utility = pruning.get("removed_zero_utility", 0)
        self.feasible_arc_count = pruning.get("feasible_arcs", 0)
        self.total_possible_arc_count = pruning.get("total_arcs_possible", 0)

        # STGraph 特有
        self.num_st_arcs = result.get("num_st_arcs", 0)

    def record_temporal_fidelity(self, deviations: list, total_deviation_nodes: int):
        """记录时序保真度指标。"""
        if deviations:
            self.mean_time_deviation_hrs = float(np.mean(deviations))
            self.max_time_deviation_hrs = float(np.max(deviations))
            self.nodes_with_deviation = sum(1 for d in deviations if d > 0.05)
        self.total_deviation_nodes = total_deviation_nodes

    def record_lazy_mtz(self, count: int):
        """记录 Lazy MTZ 回调统计 (Delta-MCF 特有)。"""
        self.lazy_mtz_count = count

    def to_dict(self) -> dict:
        """导出全部指标为扁平字典, 适合批量汇总。"""
        return {
            # 元信息
            "experiment_id": self.experiment_id,
            "instance_name": self.instance_name,
            # 求解效率
            "cpu_time_s": round(self.cpu_time_s, 2),
            "mip_gap_pct": round(self.mip_gap_pct, 4),
            "best_obj": self.best_obj,
            "best_bound": self.best_bound,
            "root_lp_value": self.root_lp_value,
            "bb_nodes": self.bb_nodes,
            "first_feasible_time_s": (round(self.first_feasible_time_s, 2)
                                       if self.first_feasible_time_s else None),
            "solve_status": self.solve_status,
            # 模型规模
            "num_vars": self.num_vars,
            "num_constrs": self.num_constrs,
            "num_binary": self.num_binary,
            "num_continuous": self.num_continuous,
            "num_integer": self.num_integer,
            # 解质量
            "objective_value": round(self.objective_value, 6),
            "num_visited_grids": self.num_visited_grids,
            "total_swaps": self.total_swaps,
            "total_travel_time_hrs": round(self.total_travel_time_hrs, 4),
            "total_service_time_hrs": round(self.total_service_time_hrs, 4),
            "makespan_hrs": round(self.makespan_hrs, 4),
            "route_length": self.route_length,
            "utility_per_hour": round(self.utility_per_hour, 4),
            "avg_utility_per_grid": round(self.avg_utility_per_grid, 4),
            # Soon/Normal 分流
            "utility_soon": round(self.utility_soon, 4),
            "utility_normal": round(self.utility_normal, 4),
            "utility_low": round(self.utility_low, 4),
            "soon_ratio": round(self.soon_ratio, 4),
            # 时序保真度
            "mean_time_deviation_hrs": round(self.mean_time_deviation_hrs, 6),
            "max_time_deviation_hrs": round(self.max_time_deviation_hrs, 6),
            "nodes_with_deviation": self.nodes_with_deviation,
            "total_deviation_nodes": self.total_deviation_nodes,
            # 空间剪枝
            "original_grid_count": self.original_grid_count,
            "active_grid_count": self.active_grid_count,
            "removed_zero_utility": self.removed_zero_utility,
            "feasible_arc_count": self.feasible_arc_count,
            "total_possible_arc_count": self.total_possible_arc_count,
            # 模型特有
            "num_st_arcs": self.num_st_arcs,
            "lazy_mtz_count": self.lazy_mtz_count,
            # 配置
            "config": self.config,
        }


# =============================================================================
# 统一结果导出
# =============================================================================
def export_experiment_result(collector: MetricsCollector,
                             progress: GurobiProgressTracker,
                             result: dict,
                             output_dir: str,
                             verbose: bool = True):
    """一站式导出: JSON 汇总 + CSV 进度记录 + CSV 路由明细。"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{collector.experiment_id}_{collector.instance_name}_{timestamp}"

    # 1) JSON 汇总
    json_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    export_dict = collector.to_dict()
    export_dict["timestamp"] = timestamp
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_dict, f, indent=2, ensure_ascii=False, default=str)

    # 2) B&B 进度 CSV
    if progress.records:
        csv_path = os.path.join(output_dir, f"{prefix}_bb_progress.csv")
        progress.to_dataframe().to_csv(csv_path, index=False)

    # 3) 路由明细 CSV
    route = result.get("route", [])
    if route:
        route_path = os.path.join(output_dir, f"{prefix}_route.csv")
        pd.DataFrame(route).to_csv(route_path, index=False)

    # 4) 汇总 CSV (单行, 方便批量合并)
    summary_path = os.path.join(output_dir, f"{prefix}_summary.csv")
    summary_dict = {k: v for k, v in export_dict.items()
                    if k not in ("config", "timestamp")}
    pd.DataFrame([summary_dict]).to_csv(summary_path, index=False)

    if verbose:
        print(f"  [实验导出] {prefix} → {output_dir}/")

    return json_path


def merge_batch_results(metrics_list: list[dict], output_path: str):
    """将多次实验的 metrics dict 列表合并为一张总表 CSV。"""
    if not metrics_list:
        return
    df = pd.DataFrame(metrics_list)
    df.to_csv(output_path, index=False)
    print(f"  [批量汇总] {len(df)} 条实验记录 → {output_path}")
    return df


# =============================================================================
# LP 松弛值采集辅助
# =============================================================================
def capture_root_lp_relaxation(model, progress: GurobiProgressTracker):
    """在 solve 完成后尝试从 Gurobi 日志/属性中提取根节点 LP 松弛值。

    注意: 需要在 optimize() 之后调用, 且仅当模型至少完成了根节点 LP 时有效。
    Gurobi 不直接暴露 root LP 值, 但可以通过 ObjBound 在根节点后的
    首次 bound 来近似。更精确的做法是使用 NodeLimit=0 单独跑一次 LP。
    """
    try:
        # 尝试从 model 属性获取
        if hasattr(model, 'ObjBound') and model.ObjBound is not None:
            progress.root_lp_value = model.ObjBound
    except Exception:
        pass


def solve_root_lp_only(model):
    """单独求解根节点 LP 松弛 (NodeLimit=0), 返回 LP 目标值。

    用法:
        lp_model = model.copy()
        lp_val = solve_root_lp_only(lp_model)
    """
    try:
        model.setParam('NodeLimit', 0)
        model.setParam('OutputFlag', 0)
        model.optimize()
        if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return model.ObjVal
    except Exception:
        pass
    return None


# =============================================================================
# Soon/Normal 效用分流计算
# =============================================================================
def compute_utility_split(u_j: float, y_j: int, grid_params: dict,
                          T_total: float) -> dict:
    """在路由解码后, 根据网格参数重新计算效用的 Soon/Normal/Low 分流。

    返回:
        {"total": ..., "soon": ..., "normal": ..., "low": ...}
    """
    from Grid_Utility import calculate_operational_utility_with_split
    try:
        result = calculate_operational_utility_with_split(
            u_j=u_j, y_j=y_j,
            n_low=grid_params.get("n_low", 0.0),
            n_soon=grid_params.get("n_soon", 0.0),
            n_normal=grid_params.get("n_normal", 0.0),
            theta_soon_global=grid_params.get("theta_soon", 0.5),
            theta_normal_global=grid_params.get("theta_normal", 0.5),
            rho_j=grid_params.get("rho", 0.0),
            lam_j=grid_params.get("lam", 0.0),
            T=T_total,
        )
        return result
    except Exception:
        return {"total": 0.0, "soon": 0.0, "normal": 0.0, "low": 0.0}
