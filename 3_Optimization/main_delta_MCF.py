from __future__ import annotations

import os
import time
import argparse
import numpy as np
import pandas as pd
from gurobipy import GRB

from Grid_Utility import calculate_operational_utility
from Optimize_PLA_Delta_MCF import optimize_evrp_with_pla_delta_mcf
from Pre_Process import (
    DEFAULT_TARGET_DATETIME,
    DEFAULT_PREDICTION_FILE,
    DATETIME_RANGE_START,
    DATETIME_RANGE_END,
    select_random_datetime,
    list_available_hours,
    generate_offline_utility_matrix,
    prepare_optimize_inputs,
    extract_grid_coordinates,
    haversine_distance,
    calculate_travel_time_matrix,
)

# 统一实验工具
from experiment_utils import (
    GurobiProgressTracker, MetricsCollector,
    export_experiment_result, compute_utility_split,
)
from experiment_config import get_experiment_config

# =========================================================================
# 默认参数配置
# =========================================================================
DEFAULT_DATA_FILE = DEFAULT_PREDICTION_FILE  # 默认使用 CB_Hurdle 预测数据
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Optimization_Result_Summary")
DEFAULT_SPEED_KMH = 30.0
DEFAULT_C_MAX = 20
DEFAULT_T_TOTAL = 1.0
DEFAULT_P_INTERVALS = 10                     # PLA时间分段数 (0.1h分辨率, 原5)
DEFAULT_SWAP_TIME_C = 0.02
DEFAULT_BIG_M = 1.5
DEFAULT_MAX_TRAVEL_TIME = 0.2
DEFAULT_Y_LEVELS = list(range(1, 11))               # 离散换电量: 1~10 (原 1~20, 减半)


# =========================================================================
# Gurobi 求解进度追踪器
# =========================================================================
# =========================================================================
# Geo-Fencing 优化: 零效用节点过滤
# =========================================================================
def filter_zero_utility_grids(grids, Omega, grid_params, grid_coords=None):
    """剔除在所有 (y, s) 组合下效用均为 0 的网格。"""
    active_grids = []
    for j in grids:
        has_utility = False
        for y_val in Omega[j]:
            for s_val in Omega[j][y_val]:
                if Omega[j][y_val][s_val] > 0:
                    has_utility = True
                    break
            if has_utility:
                break
        if has_utility:
            active_grids.append(j)

    removed = len(grids) - len(active_grids)

    active_Omega = {j: Omega[j] for j in active_grids}
    active_params = {j: grid_params[j] for j in active_grids if j in grid_params}

    result = [active_grids, active_Omega, active_params]
    if grid_coords is not None:
        active_coords = {j: grid_coords[j] for j in active_grids if j in grid_coords}
        result.append(active_coords)
    result.append(removed)
    return tuple(result)


# =========================================================================
# 核心管线
# =========================================================================
def build_full_travel_time_matrix(
    grids, grid_coords, depot_lat, depot_lon,
    vehicle_speed_kmh=DEFAULT_SPEED_KMH,
):
    """构建包含 depot (index=0) 的完整旅行时间矩阵。"""
    depot = 0
    travel_time = {depot: {}}

    for g in grids:
        lat_g, lon_g = grid_coords[g]
        dist = haversine_distance(depot_lat, depot_lon, lat_g, lon_g)
        t = dist / vehicle_speed_kmh
        travel_time[depot][g] = t
        if g not in travel_time:
            travel_time[g] = {}
        travel_time[g][depot] = t

    travel_time[depot][depot] = 0.0

    grid_matrix = calculate_travel_time_matrix(grids, grid_coords, vehicle_speed_kmh)
    for g1 in grids:
        for g2 in grids:
            travel_time[g1][g2] = grid_matrix[g1][g2]

    return travel_time


def run_optimization_pipeline(
    data_file=DEFAULT_DATA_FILE,
    target_datetime=DEFAULT_TARGET_DATETIME,
    depot_lat=None,
    depot_lon=None,
    vehicle_speed_kmh=DEFAULT_SPEED_KMH,
    C_max=DEFAULT_C_MAX,
    T_total=DEFAULT_T_TOTAL,
    P_intervals=DEFAULT_P_INTERVALS,
    y_levels=None,
    swap_time_c=DEFAULT_SWAP_TIME_C,
    BigM=DEFAULT_BIG_M,
    max_travel_time=DEFAULT_MAX_TRAVEL_TIME,
    output_dir=DEFAULT_OUTPUT_DIR,
    verbose=True,
    # === 实验元信息 ===
    experiment_id="M5",
    instance_name="default",
    geo_fencing=True,
    time_limit_s=1200,
):
    """一站式执行完整管线 (Delta + MCF 版本)。"""
    _y = y_levels if y_levels is not None else list(range(1, C_max + 1))

    # 初始化全维度指标采集器
    collector = MetricsCollector(
        experiment_id=experiment_id,
        instance_name=instance_name,
        config={
            "geo_fencing": geo_fencing,
            "P_intervals": P_intervals,
            "C_max": C_max,
            "T_total": T_total,
            "model_type": "Delta-MCF",
        }
    )

    if verbose:
        print(f"[{experiment_id}] Delta-MCF Pipeline | P={P_intervals} | Geo-Fence={'✓' if geo_fencing else '✗'}")
        print(f"  数据={data_file} | "
              f"C={C_max} T={T_total} P={P_intervals} "
              f"Y={{{min(_y)}..{max(_y)}}} "
              f"arc≤{max_travel_time}h | speed={vehicle_speed_kmh}km/h")

    # Step 1-2: 数据加载 + 坐标提取
    grids, grid_params, snapshot_df = prepare_optimize_inputs(
        data_file, target_datetime=target_datetime
    )
    original_grid_count = len(grids)
    grid_coords = extract_grid_coordinates(snapshot_df)

    if depot_lat is None or depot_lon is None:
        depot_lat = float(np.mean([c[0] for c in grid_coords.values()]))
        depot_lon = float(np.mean([c[1] for c in grid_coords.values()]))

    travel_time = build_full_travel_time_matrix(
        grids, grid_coords, depot_lat, depot_lon, vehicle_speed_kmh
    )

    # Step 3: PLA 效用矩阵
    Omega, tau_list = generate_offline_utility_matrix(
        grids=grids,
        C_max=C_max,
        T_total=T_total,
        P_intervals=P_intervals,
        grid_params=grid_params,
        calc_utility_func=calculate_operational_utility,
        y_levels=y_levels,
    )

    # Step 4: Geo-Fencing 空间剪枝 (条件执行)
    if geo_fencing:
        active_grids, Omega, active_params, active_coords, removed_zeros = \
            filter_zero_utility_grids(grids, Omega, grid_params, grid_coords)
    else:
        active_grids = list(grids)
        active_params = dict(grid_params)
        active_coords = dict(grid_coords)
        removed_zeros = 0

    travel_time = build_full_travel_time_matrix(
        active_grids, active_coords, depot_lat, depot_lon, vehicle_speed_kmh
    )

    feasible_arcs_count = sum(
        1 for i in travel_time for j in travel_time[i]
        if i != j and travel_time[i][j] <= max_travel_time
    )
    total_possible = (len(active_grids) + 1) * len(active_grids)

    if verbose:
        print(f"  网格: {original_grid_count}→{len(active_grids)} "
              f"(剔除{removed_zeros}零效用) | "
              f"弧段: {feasible_arcs_count}/{total_possible} "
              f"({100*feasible_arcs_count/max(1,total_possible):.0f}%)")

    collector.original_grid_count = original_grid_count
    collector.active_grid_count = len(active_grids)
    collector.removed_zero_utility = removed_zeros
    collector.feasible_arc_count = feasible_arcs_count
    collector.total_possible_arc_count = total_possible

    if not active_grids:
        if verbose:
            print("\n  [终止] 所有网格均无正效用, 无需调度。")
        collector.solve_status = "SKIPPED"
        return {
            "model": None, "grids": [], "travel_time": travel_time,
            "Omega": Omega, "tau_list": tau_list, "snapshot_df": snapshot_df,
            "grid_params": active_params, "elapsed_seconds": 0.0,
            "status": "SKIPPED — all grids zero-utility",
            "objective": 0.0, "route": [], "summary": {},
            "collector": collector,
        }

    # ===== Step 5: 求解 =====
    progress = GurobiProgressTracker(label=experiment_id) if verbose else None
    if verbose:
        print(f"[求解] 开始 Delta+MCF...")

    t_start = time.perf_counter()
    model = optimize_evrp_with_pla_delta_mcf(
        grids=active_grids,
        travel_time=travel_time,
        C_max=C_max,
        T_total=T_total,
        Omega=Omega,
        tau_list=tau_list,
        swap_time_c=swap_time_c,
        BigM=BigM,
        progress_tracker=progress,
        max_travel_time=max_travel_time,
        y_levels=y_levels,
        time_limit_s=time_limit_s,
    )
    t_elapsed = time.perf_counter() - t_start

    # 采集求解效率指标
    collector.record_solve(progress, t_elapsed, model)
    # 记录 Lazy MTZ 回调次数
    if hasattr(model, '_lazy_mtz_count'):
        collector.record_lazy_mtz(model._lazy_mtz_count[0]
                                  if isinstance(model._lazy_mtz_count, list)
                                  else model._lazy_mtz_count)

    # ===== Step 6: 结果解析 =====

    result = _parse_solution(model, active_grids, travel_time, swap_time_c,
                             grid_params=active_params, T_total=T_total, verbose=verbose)

    # 采集解质量指标
    collector.record_solution(result)

    # 统一实验指标导出 (始终执行, 确保结果落盘)
    export_experiment_result(collector, progress, result, output_dir, verbose=verbose)

    result["model"] = model
    result["grids"] = active_grids
    result["travel_time"] = travel_time
    result["Omega"] = Omega
    result["tau_list"] = tau_list
    result["snapshot_df"] = snapshot_df
    result["grid_params"] = active_params
    result["elapsed_seconds"] = t_elapsed
    result["collector"] = collector
    result["pruning_stats"] = {
        "original_grids": original_grid_count,
        "active_grids": len(active_grids),
        "removed_zero_utility": removed_zeros,
        "total_arcs_possible": total_possible,
        "feasible_arcs": feasible_arcs_count,
        "max_travel_time_h": max_travel_time,
        "geo_fencing_enabled": geo_fencing,
    }

    return result


# =========================================================================
# 结果解析与保存
# =========================================================================
def _parse_solution(model, grids, travel_time, swap_time_c,
                    grid_params=None, T_total=1.0, verbose=True):
    """从优化后的 Gurobi model 中提取可读的路由方案。

    新增:
      - grid_params: 用于计算 Soon/Normal 效用分流
      - T_total: 规划周期长度
    """
    status_map = {
        GRB.OPTIMAL: "OPTIMAL (全局最优)",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT (超时截断)",
        GRB.INFEASIBLE: "INFEASIBLE (不可行)",
        GRB.INF_OR_UNBD: "INFEASIBLE or UNBOUNDED",
        GRB.UNBOUNDED: "UNBOUNDED",
    }
    status = status_map.get(model.Status, f"Status code: {model.Status}")
    obj_val = model.ObjVal if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) else None

    result = {"status": status, "objective": obj_val, "route": [], "summary": {}}

    try:
        x = model._x
        v = model._v
        u = model._u
        y = model._y
        feasible_arcs = getattr(model, '_feasible_arcs', set())
    except AttributeError:
        if verbose:
            print("  [警告] 模型未返回可行解，跳过路由提取。")
        return result

    visited = [j for j in grids if v[j].X > 0.5]
    result["visited_grids"] = visited

    if verbose:
        obj_str = f"{obj_val:.4f}" if obj_val is not None else "N/A"
        print(f"  [{status}] obj={obj_str} visited={len(visited)}/{len(grids)}")

    if not visited:
        if verbose:
            print("  路由为空。")
        return result

    def _arc_val(i, j):
        if (i, j) in feasible_arcs:
            return x[i, j].X
        return 0.0

    depot = 0
    next_node = None
    for j in visited:
        if _arc_val(depot, j) > 0.5:
            next_node = j
            break

    if next_node is None:
        if verbose:
            print("  [警告] 未找到从 depot 出发的边。")
        return result

    route_seq = [depot]
    current = next_node
    while current != depot:
        route_seq.append(current)
        found = False
        for nxt in list(visited) + [depot]:
            if nxt != current and _arc_val(current, nxt) > 0.5:
                current = nxt
                found = True
                break
        if not found:
            break

    route_seq.append(depot)

    depot_indices = [idx for idx, n in enumerate(route_seq) if n == depot]

    for idx, node in enumerate(route_seq):
        if node == depot:
            if idx == depot_indices[0]:
                arrival_time = 0.0
            else:
                prev_node = route_seq[idx - 1]
                arrival_time = (
                    u[prev_node].X
                    + swap_time_c * int(round(y[prev_node].X))
                    + travel_time[prev_node][depot]
                )
            result["route"].append({
                "grid": "DEPOT",
                "arrival_time": arrival_time,
                "y_swapped": 0,
            })
        else:
            arr_time = u[node].X
            y_swap = int(round(y[node].X))
            result["route"].append({
                "grid": node,
                "arrival_time": arr_time,
                "y_swapped": y_swap,
            })

    total_swaps = sum(r["y_swapped"] for r in result["route"] if r["grid"] != "DEPOT")
    total_travel_time = 0.0
    for idx in range(len(route_seq) - 1):
        i, j = route_seq[idx], route_seq[idx + 1]
        total_travel_time += travel_time[i][j]
    total_service_time = total_swaps * swap_time_c
    makespan = total_travel_time + total_service_time

    # =========================================================================
    # Soon/Normal 效用分流计算
    # =========================================================================
    utility_soon_total = 0.0
    utility_normal_total = 0.0
    utility_low_total = 0.0

    if grid_params is not None:
        for r in result["route"]:
            g = r.get("grid")
            if g is None or g == "DEPOT":
                continue
            y_swap = r.get("y_swapped", 0)
            arr_time = r.get("arrival_time", 0.0)
            if y_swap > 0 and g in grid_params:
                split = compute_utility_split(
                    u_j=arr_time, y_j=y_swap,
                    grid_params=grid_params[g], T_total=T_total,
                )
                utility_soon_total += split["soon"]
                utility_normal_total += split["normal"]
                utility_low_total += split["low"]
                r["utility_soon"] = split["soon"]
                r["utility_normal"] = split["normal"]
                r["utility_low"] = split["low"]

    result["summary"] = {
        "num_visited": len(visited),
        "total_swaps": total_swaps,
        "total_travel_time_hrs": total_travel_time,
        "total_service_time_hrs": total_service_time,
        "makespan_hrs": makespan,
        "route_length": len(route_seq) - 1,
        "utility_soon": round(utility_soon_total, 6),
        "utility_normal": round(utility_normal_total, 6),
        "utility_low": round(utility_low_total, 6),
    }

    if verbose:
        print(f"\n  路由 ({len(route_seq)}节点, {total_swaps}块电池):")
        cum_time = 0.0
        timing_issues = 0
        prev_grid = depot
        for idx, step in enumerate(result["route"]):
            if step["grid"] == "DEPOT":
                if idx > 0:
                    cum_time += (swap_time_c * result["route"][idx - 1]["y_swapped"]
                                 + travel_time[prev_grid][depot])
                tag = "DEPOT"
                print(f"    [{tag}] @{cum_time:.3f}h")
            else:
                grid = step["grid"]
                cum_time += travel_time[prev_grid][grid]
                model_u = step["arrival_time"]
                y_swap = step["y_swapped"]
                diff = abs(model_u - cum_time)
                flag = " ⚠" if diff > 0.05 else ""
                if diff > 0.05:
                    timing_issues += 1
                print(f"    →{grid}  cum={cum_time:.3f}h u={model_u:.3f}h "
                      f"y={y_swap}{flag}")
                cum_time += swap_time_c * y_swap
                prev_grid = grid

        if timing_issues > 0:
            print(f"  ⚠ {timing_issues}处时序偏差>0.05h")
        else:
            print(f"  ✓ 时序一致 | 行驶{total_travel_time:.3f}h "
                  f"服务{total_service_time:.3f}h 完工{makespan:.3f}h")

    return result


# =========================================================================
# 主入口 (Production Simulation Entry Point)
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delta-MCF 多商品流优化引擎 —— E-Bike Battery Swapping VRP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main_delta_MCF.py                                                 # 使用默认时间
  python main_delta_MCF.py --datetime "2025/10/28 14:00"                   # 指定具体时间
  python main_delta_MCF.py --random                                        # 随机选取可用小时
  python main_delta_MCF.py --random --seed 42                              # 随机选取 (固定种子)
  python main_delta_MCF.py --list-hours                                    # 列出可用小时
  python main_delta_MCF.py --data path/to/other.csv                        # 指定数据文件
        """,
    )
    parser.add_argument(
        "--data", type=str, default=DEFAULT_DATA_FILE,
        help="预测数据 CSV 文件路径"
    )
    parser.add_argument(
        "--datetime", type=str, default=None,
        help="目标日期时间, 格式: 'YYYY/MM/DD HH:MM' (例如 '2025/10/28 14:00')"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="从可用小时中随机选取一个"
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

    args = parser.parse_args()

    # ---- 列出可用小时模式 ----
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

    # ---- 确定目标日期时间 ----
    if args.random:
        target_datetime = select_random_datetime(
            file_path=args.data,
            start=args.start,
            end=args.end,
            seed=args.seed,
        )
        print("=" * 60)
        print("  [随机] 随机选取模式 — Delta-MCF 多商品流优化引擎")
        if args.seed is not None:
            print(f"  随机种子: {args.seed}")
    elif args.datetime is not None:
        target_datetime = args.datetime
        print("=" * 60)
        print("  [指定] 用户指定日期时间 — Delta-MCF 多商品流优化引擎")
    else:
        target_datetime = DEFAULT_TARGET_DATETIME
        print("=" * 60)
        print("  [默认] 默认日期时间 — Delta-MCF 多商品流优化引擎")

    print(f"  选择的目标时间: {target_datetime}")
    print(f"  数据文件: {args.data}")
    print("=" * 60)

    result = run_optimization_pipeline(
        data_file=args.data,
        target_datetime=target_datetime,
        depot_lat=None,
        depot_lon=None,
        vehicle_speed_kmh=30.0,
        C_max=20,
        T_total=1.0,
        P_intervals=DEFAULT_P_INTERVALS,   # 10 (0.1h分辨率)
        y_levels=DEFAULT_Y_LEVELS,         # 1~10 (原 1~20)
        swap_time_c=0.02,
        BigM=200.0,
        max_travel_time=DEFAULT_MAX_TRAVEL_TIME,  # 0.2h
        verbose=True,
        experiment_id="M5",
        geo_fencing=True,
    )
