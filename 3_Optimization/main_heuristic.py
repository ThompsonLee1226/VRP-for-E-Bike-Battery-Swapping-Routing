#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
M6 元启发式基线 — 贪心最近邻构造 + 2-opt 局部搜索 (Greedy + Local Search)
=============================================================================

与 STGraph (M1) 使用完全相同的:
  - 数据快照与效用张量 Omega (右取整离散, 非连续效用)
  - KNN 空间稀疏化图 (K=50, max_travel_time=0.2h)
  - 换电量离散域 y ∈ {1,...,10} 与容量 C=20 / 周期 T=1h

仅依赖 numpy / pandas, 不调用 Gurobi 求解器。用于量化 STGraph 相对
元启发式的 "最优性溢价 (Optimality Premium)"。

输出与 M1-M5 走同一套 MetricsCollector + export_experiment_result 管线,
可直接由 batch_runner.py 统一调用, 实现 M1-M6 一键对比。

运行方式:
  python 3_Optimization/main_heuristic.py                                   # 默认时间
  python 3_Optimization/main_heuristic.py --datetime "2025/11/05 18:00"     # 指定时间 (工作日晚高峰)
  python 3_Optimization/main_heuristic.py --random --seed 42
=============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np

# 确保当前目录在 Python 路径中 (便于作为脚本直接运行)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from Pre_Process import (
    DEFAULT_TARGET_DATETIME,
    DEFAULT_PREDICTION_FILE,
    select_random_datetime,
    list_available_hours,
    prepare_optimize_inputs,
    extract_grid_coordinates,
    haversine_distance,
    calculate_travel_time_matrix,
    generate_offline_utility_matrix,
)
from Grid_Utility import calculate_operational_utility
from experiment_utils import (
    GurobiProgressTracker,
    MetricsCollector,
    export_experiment_result,
    compute_utility_split,
)

# =============================================================================
# 常量 (与 main_STGraph.py 保持一致)
# =============================================================================
SPEED_KMH = 30.0
C_MAX = 20
T_TOTAL = 1.0
P_INTERVALS = 12
SWAP_TIME_C = 0.02
MAX_TRAVEL_TIME = 0.2
K_NEIGHBORS = 50
Y_LEVELS = list(range(1, 11))          # 离散换电量 1~10
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "Optimization_Result_Summary")


# =============================================================================
# Geo-Fencing 零效用过滤 (与 main_STGraph.filter_zero_utility_grids 等价)
# =============================================================================
# ---------------------------------------------------------------------------
# 效用零判定的数值容差。
# U_j 由多次 exp()/pow() 组合而成，真实效用恰为 0 的网格可能算出 ~1e-16 的残差；
# 用严格 `> 0` 会把这类浮点噪声误判为真实效用（实测误留 15 个网格，均为 rho_j=0）。
# 依据命题 Z 的推论：U_j ≡ 0  ⟺  max_u (N_low(u) + N_soon(u)) = 0。
# ---------------------------------------------------------------------------
UTILITY_ZERO_TOL = 1e-9


def filter_zero_utility_grids(grids, Omega):
    """剔除在任意换电量及时间点下效用均无法变现的孤立网格。"""
    active_grids = []
    for j in grids:
        if any(Omega[j][y][s] > UTILITY_ZERO_TOL for y in Omega[j] for s in Omega[j][y]):
            active_grids.append(j)
    return active_grids


# =============================================================================
# 旅行时间矩阵 (与 main_STGraph.build_full_travel_time_matrix 等价)
# =============================================================================
def build_full_travel_time_matrix(grids, grid_coords, depot_lat, depot_lon,
                                  vehicle_speed_kmh=SPEED_KMH):
    depot = 0
    travel_time = {depot: {}}
    for g in grids:
        lat_g, lon_g = grid_coords[g]
        dist = haversine_distance(depot_lat, depot_lon, lat_g, lon_g)
        t = dist / vehicle_speed_kmh
        travel_time[depot][g] = t
        travel_time.setdefault(g, {})[depot] = t
    travel_time[depot][depot] = 0.0
    grid_matrix = calculate_travel_time_matrix(grids, grid_coords, vehicle_speed_kmh)
    for g1 in grids:
        for g2 in grids:
            travel_time[g1][g2] = grid_matrix[g1][g2]
    return travel_time


def build_spatial_neighbors(nodes, travel_time, K_neighbors=K_NEIGHBORS,
                            max_travel_time=MAX_TRAVEL_TIME, T_total=T_TOTAL):
    """构造 KNN 空间稀疏化图, 与 STGraph 引擎保持完全一致。"""
    depot = 0
    spatial_neighbors = {i: [] for i in nodes}
    for i in nodes:
        candidates = []
        for j in nodes:
            if i != j and travel_time[i][j] <= max_travel_time:
                candidates.append((travel_time[i][j], j))
        candidates.sort(key=lambda x: x[0])
        spatial_neighbors[i] = [tgt for _, tgt in candidates[:K_neighbors]]
        if i != depot and depot not in spatial_neighbors[i] and travel_time[i][depot] <= T_total:
            spatial_neighbors[i].append(depot)
    return spatial_neighbors


def right_round_step(t, tau_list):
    """寻找满足 tau_list[s] >= t 的最小离散步 index (右取整)。"""
    for s, ts in enumerate(tau_list):
        if ts >= t - 1e-7:
            return s
    return None


# =============================================================================
# 路由评估: 沿给定节点顺序, 贪心重优化每个节点的换电量 y
# =============================================================================
def evaluate_route(route, travel_time, Omega, tau_list, C_max, T_total, swap_time_c, Y_levels):
    """评估 [depot, g1, ..., gk, depot] 的总效用 (离散 Omega 张量 + 右取整)。

    返回 (objective, total_swaps, makespan, node_details);
    若不可行(无法按期返程)返回 None。
    node_details: [(grid, arrival_time, y_swapped), ...]
    """
    depot = 0
    if len(route) <= 2:
        return 0.0, 0, 0.0, []

    cum_time = 0.0
    cum_swaps = 0
    objective = 0.0
    total_swaps = 0
    details = []

    for idx in range(1, len(route) - 1):
        j = route[idx]
        prev = route[idx - 1]
        proj_arr = cum_time + travel_time[prev][j]
        s_idx = right_round_step(proj_arr, tau_list)
        if s_idx is None:
            return None
        best_y = None
        best_u = 0.0
        for y in Y_levels:
            if cum_swaps + y > C_max:
                break
            dep_time = proj_arr + swap_time_c * y
            if dep_time + travel_time[j][depot] > T_total:
                break   # y 递增, 更大的 y 只会更晚
            u = Omega[j][y][s_idx]
            if u > best_u:
                best_u = u
                best_y = y
        if best_y is None:
            best_y = 0
        objective += best_u
        total_swaps += best_y
        cum_swaps += best_y
        cum_time = proj_arr + swap_time_c * best_y
        details.append((j, round(proj_arr, 4), best_y))

    makespan = cum_time + travel_time[route[-2]][depot]
    if makespan > T_total + 1e-9:
        return None
    return objective, total_swaps, makespan, details


# =============================================================================
# 贪心最近邻构造 (与 STGraph Warm-Start 同源)
# =============================================================================
def greedy_construction(active_grids, travel_time, spatial_neighbors, Omega,
                        tau_list, C_max, T_total, swap_time_c, Y_levels):
    """贪心构造 [depot, ...], 返回 (route, greedy_obj)。"""
    depot = 0
    unvisited = set(active_grids)
    route = [depot]
    current = depot
    cum_time = 0.0
    cum_swaps = 0
    greedy_obj = 0.0

    max_iter = len(unvisited) + 5
    while unvisited and max_iter > 0:
        max_iter -= 1
        candidates = []
        for j in spatial_neighbors[current]:
            if j == depot or j not in unvisited:
                continue
            tt = travel_time[current][j]
            proj_arr = cum_time + tt
            if proj_arr >= T_total:
                continue
            s_idx = right_round_step(proj_arr, tau_list)
            if s_idx is None:
                continue
            max_u = max((Omega[j][yy][s_idx] for yy in Y_levels if cum_swaps + yy <= C_max), default=0.0)
            if max_u > 0:
                candidates.append((max_u / max(tt, 0.001), j, proj_arr, tt))

        if not candidates:
            break
        candidates.sort(reverse=True, key=lambda t: t[0])
        _, best_j, proj_arr, tt = candidates[0]

        s_idx = right_round_step(proj_arr, tau_list)
        if s_idx is None:
            unvisited.discard(best_j)
            continue
        best_y = max(((yy, Omega[best_j][yy][s_idx]) for yy in Y_levels if cum_swaps + yy <= C_max),
                     key=lambda p: p[1], default=(min(Y_levels), 0.0))[0]
        if Omega[best_j][best_y][s_idx] <= 0:
            unvisited.discard(best_j)
            continue

        svc = swap_time_c * best_y
        dep_time = proj_arr + svc
        if dep_time + travel_time[best_j][depot] > T_total:
            saved = False
            for yy in sorted(Y_levels, reverse=True):
                if cum_swaps + yy > C_max:
                    continue
                if proj_arr + swap_time_c * yy + travel_time[best_j][depot] <= T_total:
                    best_y = yy
                    svc = swap_time_c * yy
                    dep_time = proj_arr + svc
                    saved = True
                    break
            if not saved:
                unvisited.discard(best_j)
                continue

        route.append(best_j)
        greedy_obj += Omega[best_j][best_y][s_idx]
        cum_swaps += best_y
        cum_time = dep_time
        unvisited.remove(best_j)
        current = best_j

    route.append(depot)
    return route, greedy_obj


# =============================================================================
# 2-opt 局部搜索 (first-improvement)
# =============================================================================
def two_opt_local_search(route, travel_time, Omega, tau_list, C_max, T_total, swap_time_c, Y_levels):
    depot = 0
    best = evaluate_route(route, travel_time, Omega, tau_list, C_max, T_total, swap_time_c, Y_levels)
    if best is None:
        return route, (0.0, 0, 0.0, [])

    improved = True
    while improved:
        improved = False
        nodes = route[1:-1]           # 内部 grid 序列
        n = len(nodes)
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_nodes = nodes[:i] + nodes[i:j + 1][::-1] + nodes[j + 1:]
                new_route = [depot] + new_nodes + [depot]
                res = evaluate_route(new_route, travel_time, Omega, tau_list, C_max, T_total, swap_time_c, Y_levels)
                if res is not None and res[0] > best[0] + 1e-9:
                    route = new_route
                    best = res
                    improved = True
                    break
            if improved:
                break
    return route, best


# =============================================================================
# 统一输出管线 (与 M1-M5 走同一套 MetricsCollector + export_experiment_result)
# =============================================================================
def run_heuristic_pipeline(
    data_file=DEFAULT_PREDICTION_FILE,
    target_datetime=DEFAULT_TARGET_DATETIME,
    depot_lat=None,
    depot_lon=None,
    vehicle_speed_kmh=SPEED_KMH,
    C_max=C_MAX,
    T_total=T_TOTAL,
    P_intervals=P_INTERVALS,
    y_levels=None,
    swap_time_c=SWAP_TIME_C,
    max_travel_time=MAX_TRAVEL_TIME,
    K_neighbors=K_NEIGHBORS,
    output_dir=OUTPUT_DIR,
    verbose=True,
    experiment_id="M6",
    instance_name="default",
):
    """一站式执行贪心 + 2-opt 元启发式, 返回与 main_STGraph 同构的结果字典。"""
    _Y = list(y_levels) if y_levels is not None else Y_LEVELS

    collector = MetricsCollector(
        experiment_id=experiment_id,
        instance_name=instance_name,
        config={
            "C_max": C_max, "T_total": T_total, "P_intervals": P_intervals,
            "K_neighbors": K_neighbors, "vehicle_speed_kmh": vehicle_speed_kmh,
        },
    )

    if verbose:
        print(f"[{experiment_id}] data={os.path.basename(data_file)} | C={C_max} T={T_total} "
              f"P={P_intervals} Y={{{min(_Y)}..{max(_Y)}}} KNN(K={K_neighbors}) | "
              f"speed={vehicle_speed_kmh}km/h")

    # Step 1: 快照 + 参数
    grids, grid_params, snapshot_df = prepare_optimize_inputs(data_file, target_datetime=target_datetime)
    grid_coords = extract_grid_coordinates(snapshot_df)
    if depot_lat is None or depot_lon is None:
        depot_lat = float(np.mean([c[0] for c in grid_coords.values()]))
        depot_lon = float(np.mean([c[1] for c in grid_coords.values()]))

    # Step 2: 效用张量
    Omega, tau_list = generate_offline_utility_matrix(
        grids=grids, C_max=C_max, T_total=T_total, P_intervals=P_intervals,
        grid_params=grid_params, calc_utility_func=calculate_operational_utility,
        y_levels=_Y,
    )

    # Step 3: Geo-Fencing
    active_grids = filter_zero_utility_grids(grids, Omega)
    active_params = {j: grid_params[j] for j in active_grids if j in grid_params}

    # Step 4: 旅行时间矩阵
    travel_time = build_full_travel_time_matrix(active_grids, grid_coords, depot_lat, depot_lon, vehicle_speed_kmh)

    # Step 5: KNN 空间图
    nodes = [0] + active_grids
    spatial_neighbors = build_spatial_neighbors(nodes, travel_time, K_neighbors, max_travel_time, T_total)

    t_start = time.perf_counter()

    # Step 6: 贪心构造 + 2-opt 局部搜索
    route, greedy_obj = greedy_construction(
        active_grids, travel_time, spatial_neighbors, Omega, tau_list,
        C_max, T_total, swap_time_c, _Y,
    )
    route, (obj, total_swaps, makespan, details) = two_opt_local_search(
        route, travel_time, Omega, tau_list, C_max, T_total, swap_time_c, _Y,
    )

    elapsed = time.perf_counter() - t_start
    visited = [g for (g, _, _) in details]

    # Step 7: 效用分流 (soon/low/normal) + 路由记录
    util_soon = util_low = util_normal = 0.0
    route_records = []
    for (g, arr, y) in details:
        rec = {"grid": str(g), "arrival_time": round(arr, 4), "y_swapped": int(y)}
        if y > 0 and g in active_params:
            split = compute_utility_split(u_j=arr, y_j=y, grid_params=active_params[g], T_total=T_total)
            util_soon += split["soon"]
            util_low += split["low"]
            util_normal += split["normal"]
            rec["utility_soon"] = split["soon"]
            rec["utility_low"] = split["low"]
            rec["utility_normal"] = split["normal"]
        route_records.append(rec)

    total_travel = sum(travel_time[route[i]][route[i + 1]] for i in range(len(route) - 1))
    total_service = total_swaps * swap_time_c

    # 采集指标 (与 M1-M5 同构)
    collector.solve_status = "LOCAL_OPTIMAL"
    collector.objective_value = round(obj, 6)
    collector.cpu_time_s = round(elapsed, 4)
    collector.num_visited_grids = len(visited)
    collector.total_swaps = int(total_swaps)
    collector.total_travel_time_hrs = round(total_travel, 4)
    collector.total_service_time_hrs = round(total_service, 4)
    collector.makespan_hrs = round(makespan, 4)
    collector.route_length = len(route) - 1
    collector.utility_soon = round(util_soon, 4)
    collector.utility_low = round(util_low, 4)
    collector.utility_normal = round(util_normal, 4)
    total_split = util_soon + util_low + util_normal
    if total_split > 0:
        collector.soon_ratio = util_soon / total_split
    collector.num_vars = 0
    collector.num_constrs = 0
    collector.bb_nodes = 0

    result = {
        "model": None,
        "grids": active_grids,
        "travel_time": travel_time,
        "Omega": Omega,
        "tau_list": tau_list,
        "status": "LOCAL_OPTIMAL (Greedy+2-opt)",
        "objective": round(obj, 6),
        "greedy_objective": round(greedy_obj, 6),
        "route": route_records,
        "visited_grids": visited,
        "summary": {
            "num_visited": len(visited),
            "total_swaps": int(total_swaps),
            "total_travel_time_hrs": round(total_travel, 4),
            "total_service_time_hrs": round(total_service, 4),
            "makespan_hrs": round(makespan, 4),
            "route_length": len(route) - 1,
            "utility_soon": round(util_soon, 6),
            "utility_normal": round(util_normal, 6),
            "utility_low": round(util_low, 6),
        },
        "collector": collector,
        "elapsed_seconds": elapsed,
        "grid_params": active_params,
    }

    if verbose:
        print(f"  [M6] Greedy obj={greedy_obj:.4f} -> 2-opt obj={obj:.4f} | "
              f"visited={len(visited)} | swaps={int(total_swaps)} | "
              f"makespan={makespan:.3f}h | {elapsed:.3f}s")

    # 统一导出 (与 M1-M5 相同的 JSON/CSV/route 落盘)
    progress = GurobiProgressTracker(label=experiment_id)
    export_experiment_result(collector, progress, result, output_dir, verbose=verbose)

    return result


# =============================================================================
# CLI 入口
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="M6 Meta-heuristic baseline: Greedy + 2-opt")
    parser.add_argument("--data", type=str, default=DEFAULT_PREDICTION_FILE)
    parser.add_argument("--datetime", type=str, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--list-hours", action="store_true")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.list_hours:
        for h in list_available_hours(file_path=args.data):
            print(h.strftime("%Y/%m/%d %H:%M"))
        return

    if args.random:
        target_datetime = select_random_datetime(file_path=args.data, seed=args.seed)
    else:
        target_datetime = args.datetime or DEFAULT_TARGET_DATETIME

    run_heuristic_pipeline(
        data_file=args.data,
        target_datetime=target_datetime,
        output_dir=args.output,
        verbose=True,
        experiment_id="M6",
        instance_name="default",
    )


if __name__ == "__main__":
    main()
