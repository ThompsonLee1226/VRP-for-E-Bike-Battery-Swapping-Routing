from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from gurobipy import GRB

from Grid_Utility import calculate_operational_utility
from Optimize_PLA import optimize_evrp_with_pla
from Pre_Process import (
    DEFAULT_TARGET_DATETIME,
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
DEFAULT_DATA_FILE = "3_Optimization\\Grid_Utility_Test.csv"           # 输入数据文件
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Optimization_Result_Summary")  # 输出目录
DEFAULT_SPEED_KMH = 30.0                              # 车辆平均速度 (km/h)
DEFAULT_C_MAX = 20                                    # 车辆最大载电池数
DEFAULT_T_TOTAL = 1.0                                 # 规划周期总时长 (小时)
DEFAULT_P_INTERVALS = 5                               # PLA 时间分段数 (Tier-3: 10→5, 减半 SOS2 变量)
DEFAULT_SWAP_TIME_C = 0.02                            # 单块电池换电服务时间 (小时)
DEFAULT_BIG_M = 1.5                                   # Big-M 常数 (现由求解器内部自适应收紧, 此值仅作后备)
DEFAULT_MAX_TRAVEL_TIME = 0.2                         # 弧段裁剪阈值 (小时) — 收紧弧段
DEFAULT_Y_LEVELS = list(range(1, 11))               # 离散换电量: 1~10 (原 1~20, 减半)


# =========================================================================
# Geo-Fencing 优化 1: 零效用节点过滤
# =========================================================================
def filter_zero_utility_grids(grids, Omega, grid_params, grid_coords=None):
    """
    剔除在所有 (y, s) 组合下效用均为 0 的网格。

    若某个网格在任何换电量和到达时间下都无法产生正效用，
    则将其从优化问题中移除，不为其创建任何决策变量与约束。

    参数:
    - grids: 原始网格 ID 列表
    - Omega: 效用矩阵, Omega[j][y][s]
    - grid_params: 网格参数字典
    - grid_coords: 网格坐标字典 (可选, 传入则同步过滤)

    返回:
    - active_grids: 过滤后的网格列表
    - active_Omega: 过滤后的 Omega
    - active_params: 过滤后的 grid_params
    - active_coords: 过滤后的 grid_coords (若传入)
    - removed: 被剔除的网格数量
    """
    active_grids = []
    for j in grids:
        # 检查是否存在任意 (y, s) 使得 Omega[j][y][s] > 0
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
    grids,
    grid_coords,
    depot_lat,
    depot_lon,
    vehicle_speed_kmh=DEFAULT_SPEED_KMH,
):
    """
    构建包含 depot (index=0) 的完整旅行时间矩阵。

    参数:
    - grids: 格点 ID 列表
    - grid_coords: dict, grid_coords[grid_id] = (lat, lon)
    - depot_lat, depot_lon: depot (车场) 的经纬度
    - vehicle_speed_kmh: 恒定车速 (km/h)

    返回:
    - travel_time: dict of dict, travel_time[i][j] 其中 i,j ∈ {0} ∪ grids
    """
    depot = 0
    travel_time = {depot: {}}

    # depot → grid 与 grid → depot
    for g in grids:
        lat_g, lon_g = grid_coords[g]
        dist = haversine_distance(depot_lat, depot_lon, lat_g, lon_g)
        t = dist / vehicle_speed_kmh
        travel_time[depot][g] = t
        if g not in travel_time:
            travel_time[g] = {}
        travel_time[g][depot] = t

    # depot → depot
    travel_time[depot][depot] = 0.0

    # grid → grid
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
    experiment_id="M3",
    instance_name="default",
    geo_fencing=True,
):
    """
    一站式执行 "预测 → 预处理 → 效用矩阵 → Geo-Fencing 剪枝 → PLA 优化" 完整管线。

    返回:
    - result: dict, 包含 model, grids, travel_time, Omega, tau_list, snapshot_df, collector 等
    """
    # 初始化全维度指标采集器
    collector = MetricsCollector(
        experiment_id=experiment_id,
        instance_name=instance_name,
        config={
            "geo_fencing": geo_fencing,
            "P_intervals": P_intervals,
            "C_max": C_max,
            "T_total": T_total,
            "model_type": "SOS2-PLA",
        }
    )

    # -----------------------------------------------------------------
    # Step 1: 数据加载与预处理
    # -----------------------------------------------------------------
    if verbose:
        print(f"[{experiment_id}] SOS2-PLA Pipeline | P={P_intervals} | Geo-Fence={'✓' if geo_fencing else '✗'}")
        print("=" * 60)
        print("Step 1/6: 加载数据与预处理")
        print("=" * 60)

    grids, grid_params, snapshot_df = prepare_optimize_inputs(
        data_file, target_datetime=target_datetime
    )
    original_grid_count = len(grids)

    if verbose:
        print(f"  数据文件: {data_file}")
        print(f"  目标时间: {pd.Timestamp(target_datetime).floor('h')}")
        print(f"  原始网格数量: {original_grid_count}")

    # -----------------------------------------------------------------
    # Step 2: 提取坐标 & 构建旅行时间矩阵 (含 depot)
    # -----------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("Step 2/6: 提取坐标 & 构建旅行时间矩阵")
        print("=" * 60)

    grid_coords = extract_grid_coordinates(snapshot_df)

    # 若未指定 depot 坐标，使用所有网格的质心
    if depot_lat is None or depot_lon is None:
        depot_lat = float(np.mean([c[0] for c in grid_coords.values()]))
        depot_lon = float(np.mean([c[1] for c in grid_coords.values()]))
        if verbose:
            print(f"  Depot 坐标未指定，自动使用网格质心: "
                  f"({depot_lat:.6f}, {depot_lon:.6f})")
    else:
        if verbose:
            print(f"  Depot 坐标: ({depot_lat:.6f}, {depot_lon:.6f})")

    travel_time = build_full_travel_time_matrix(
        grids, grid_coords, depot_lat, depot_lon, vehicle_speed_kmh
    )

    # 统计弧段
    total_arcs = (len(grids) + 1) * len(grids)  # (|grids| + depot) × (|grids| + depot - 1) ≈ this
    feasible_arcs_pre = sum(
        1 for i in travel_time for j in travel_time[i]
        if i != j and travel_time[i][j] <= max_travel_time
    )

    if verbose:
        print(f"  车速设定: {vehicle_speed_kmh} km/h")
        print(f"  弧段裁剪阈值: ≤{max_travel_time}h")
        print(f"  可行弧段预览: ~{feasible_arcs_pre} / ~{total_arcs} "
              f"({100*feasible_arcs_pre/max(1,total_arcs):.1f}%)")

    # -----------------------------------------------------------------
    # Step 3: 构建 PLA 效用矩阵 Omega (在全量 grids 上)
    # -----------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("Step 3/6: 构建 PLA 效用矩阵 Omega")
        print("=" * 60)
        print(f"  C_max={C_max}, T_total={T_total}, P_intervals={P_intervals}")

    Omega, tau_list = generate_offline_utility_matrix(
        grids=grids,
        C_max=C_max,
        T_total=T_total,
        P_intervals=P_intervals,
        grid_params=grid_params,
        calc_utility_func=calculate_operational_utility,
        y_levels=y_levels,
    )

    # -----------------------------------------------------------------
    # Step 4: Geo-Fencing 空间剪枝
    # -----------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("Step 4/6: Geo-Fencing 空间剪枝")
        print("=" * 60)

    # 4a. 零效用节点过滤 (条件执行)
    if geo_fencing:
        active_grids, Omega, active_params, active_coords, removed_zeros = \
            filter_zero_utility_grids(grids, Omega, grid_params, grid_coords)
    else:
        active_grids = list(grids)
        active_Omega = Omega
        active_params = dict(grid_params)
        active_coords = dict(grid_coords)
        removed_zeros = 0

    if verbose:
        if geo_fencing:
            print(f"  [节点过滤] 零效用网格剔除: {removed_zeros} / {original_grid_count}")
            print(f"              活跃网格保留: {len(active_grids)}")
        else:
            print(f"  [节点过滤] 消融模式: 保留全部 {original_grid_count} 个网格 (零效用过滤已跳过)")

    # 4b. 为活跃网格重建旅行时间矩阵 (与 depot)
    travel_time = build_full_travel_time_matrix(
        active_grids, active_coords, depot_lat, depot_lon, vehicle_speed_kmh
    )

    # 统计剪枝后的弧段
    feasible_arcs_count = sum(
        1 for i in travel_time for j in travel_time[i]
        if i != j and travel_time[i][j] <= max_travel_time
    )
    total_possible = (len(active_grids) + 1) * len(active_grids)
    if verbose:
        print(f"  [边长裁剪] 活跃弧段: {feasible_arcs_count} / ~{total_possible} "
              f"({100*feasible_arcs_count/max(1,total_possible):.1f}%)")

    collector.original_grid_count = original_grid_count
    collector.active_grid_count = len(active_grids)
    collector.removed_zero_utility = removed_zeros
    collector.feasible_arc_count = feasible_arcs_count
    collector.total_possible_arc_count = total_possible

    # 若无活跃网格, 提前退出
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

    # -----------------------------------------------------------------
    # Step 5: 执行 PLA-MIP 求解
    # -----------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("Step 5/6: 执行 Gurobi PLA-MIP 求解")
        print("=" * 60)

    progress = GurobiProgressTracker(label=experiment_id) if verbose else None

    t_start = time.perf_counter()
    model = optimize_evrp_with_pla(
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
    )
    t_elapsed = time.perf_counter() - t_start

    # 采集求解效率指标
    collector.record_solve(progress, t_elapsed, model)

    if verbose:
        print(f"\n  求解耗时: {t_elapsed:.1f} 秒")
        print(f"  求解状态: {model.Status}")
        if progress and progress.records:
            print(f"  进度摘要: {progress.summary()}")

    # -----------------------------------------------------------------
    # Step 6: 结果解析与输出
    # -----------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("Step 6/6: 结果解析")
        print("=" * 60)

    result = _parse_solution(model, active_grids, travel_time, swap_time_c,
                             grid_params=active_params, T_total=T_total, verbose=verbose)

    # 采集解质量指标
    collector.record_solution(result)

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    _save_results(result, output_dir, verbose=verbose)

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
# 结果解析与保存 (适配稀疏 x)
# =========================================================================
def _parse_solution(model, grids, travel_time, swap_time_c,
                    grid_params=None, T_total=1.0, verbose=True):
    """从优化后的 Gurobi model 中提取可读的路由方案 (兼容稀疏 x)。

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

    # 被访问的 grid
    visited = [j for j in grids if v[j].X > 0.5]
    result["visited_grids"] = visited

    if verbose:
        print(f"  求解状态: {status}")
        print(f"  目标函数值: {obj_val:.6f}" if obj_val is not None else "  目标函数值: N/A")
        print(f"  访问网格数: {len(visited)} / {len(grids)}")

    if not visited:
        if verbose:
            print("  [信息] 未访问任何网格，路由为空。")
        return result

    # 辅助函数: 安全获取弧变量值 (稀疏 tupledict 不存在键时返回 0)
    def _arc_val(i, j):
        if (i, j) in feasible_arcs:
            return x[i, j].X
        return 0.0

    # 重建路由链: depot → g1 → g2 → ... → depot
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

    # 构建每站的详细信息
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

    # 汇总统计
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
        print(f"\n  路由序列 ({len(route_seq)} 个节点):")
        for step in result["route"]:
            if step["grid"] == "DEPOT":
                print(f"    [DEPOT]  到达时间: {step['arrival_time']:.4f}h")
            else:
                print(f"    → {step['grid']}  "
                      f"(到达 {step['arrival_time']:.4f}h, "
                      f"换电 {step['y_swapped']} 块)")
        print(f"\n  汇总:")
        print(f"    总换电量: {total_swaps}")
        print(f"    总行驶时间: {total_travel_time:.4f} h")
        print(f"    总服务时间: {total_service_time:.4f} h")
        print(f"    完工时间: {makespan:.4f} h")

    return result


def _save_results(result, output_dir, verbose=True):
    """将优化结果保存为 CSV 文件。"""
    route = result.get("route", [])
    if not route:
        return

    rows = []
    for step in route:
        rows.append({
            "grid_id": step["grid"],
            "arrival_time_hrs": step["arrival_time"],
            "batteries_swapped": step["y_swapped"],
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"optimization_route_{time.time()}.csv")
    df.to_csv(csv_path, index=False)

    # 保存汇总
    summary = result.get("summary", {})
    if summary:
        summary_path = os.path.join(output_dir, f"optimization_summary_{time.time()}.csv")
        pd.DataFrame([summary]).to_csv(summary_path, index=False)

    if verbose:
        print(f"\n  结果已保存至: {output_dir}/")


# =========================================================================
# 主入口
# =========================================================================
if __name__ == "__main__":
    result = run_optimization_pipeline(
        data_file=DEFAULT_DATA_FILE,
        target_datetime=DEFAULT_TARGET_DATETIME,
        depot_lat=None,
        depot_lon=None,
        vehicle_speed_kmh=30.0,
        C_max=20,
        T_total=1.0,
        P_intervals=DEFAULT_P_INTERVALS,   # 5 (原 10) — 减半 SOS2 变量
        y_levels=DEFAULT_Y_LEVELS,         # 1~10 (原 1~20) — 减半换电量选择
        swap_time_c=0.02,
        BigM=200.0,
        max_travel_time=DEFAULT_MAX_TRAVEL_TIME,  # 0.2h — 弧段裁剪阈值
        verbose=True,
        experiment_id="M3",
        geo_fencing=True,
    )
