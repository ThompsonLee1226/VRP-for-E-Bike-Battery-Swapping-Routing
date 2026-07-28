"""
运行命令：

# 默认时间 (2025/10/24 12:00)
  python 3_Optimization/main_STGraph.py
  # 指定具体小时
  python 3_Optimization/main_STGraph.py --datetime "2025/10/29 14:00"

  # 随机选取
  python 3_Optimization/main_STGraph.py --random

  # 随机选取 + 固定种子 (可复现)
  python 3_Optimization/main_STGraph.py --random --seed 42

  # 列出所有可用小时
  python 3_Optimization/main_STGraph.py --list-hours
"""

from __future__ import annotations

import os
import time
import argparse
import numpy as np
import pandas as pd
from gurobipy import GRB

# 导入时空图优化引擎与标准预处理管线
from Optimize_PLA_STGraph import optimize_evrp_with_stgraph
from Grid_Utility import calculate_operational_utility
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
from route_visualizer import visualize_route

# =========================================================================
# 默认统一参数配置 (与原有 MCF 骨架体系保持高度严谨对齐)
# =========================================================================
DEFAULT_DATA_FILE = DEFAULT_PREDICTION_FILE  # 默认使用 CB_Hurdle 预测数据
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Optimization_Result_Summary", time.strftime("%Y%m%d_%H%M%S"))
DEFAULT_SPEED_KMH = 30.0
DEFAULT_C_MAX = 20
DEFAULT_T_TOTAL = 1.0
DEFAULT_P_INTERVALS = 12                     # 时空图推荐：每 5 分钟一段，共 12 段 (原 10)
DEFAULT_SWAP_TIME_C = 0.02
DEFAULT_MAX_TRAVEL_TIME = 0.2
DEFAULT_K_NEIGHBORS = 50                     # KNN 稠密空间图裁剪决策空间界限
DEFAULT_Y_LEVELS = list(range(1, 11))        # 离散换电量: 1~10


# =========================================================================
# Geo-Fencing 优化: 零效用节点静态过滤
# =========================================================================
def filter_zero_utility_grids(grids, Omega, grid_params, grid_coords=None):
    """在建模前剔除在任意换电量及时间点下效用均无法变现的孤立网格。"""
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
# 旅行时间矩阵构建
# =========================================================================
def build_full_travel_time_matrix(
    grids, grid_coords, depot_lat, depot_lon,
    vehicle_speed_kmh=DEFAULT_SPEED_KMH,
):
    """全面构建涵盖 DEPOT (Index=0) 在内的完整空间转移耗时矩阵。"""
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


# =========================================================================
# 核心业务总控管线 (ST-Graph Integrated Execution Pipeline)
# =========================================================================
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
    max_travel_time=DEFAULT_MAX_TRAVEL_TIME,
    K_neighbors=DEFAULT_K_NEIGHBORS,
    output_dir=DEFAULT_OUTPUT_DIR,
    verbose=True,
    # === 消融实验控制参数 ===
    geo_fencing=True,          # False → M2a: 关闭零效用网格过滤
    knn_enabled=True,          # False → M2b: 关闭 KNN 稠密图裁剪
    # === 实验元信息 ===
    experiment_id="M1",        # 实验组代号, 用于 MetricsCollector 标注
    instance_name="default",   # 实例名, 用于输出文件命名
    time_limit_s=1200,         # Gurobi 求解时间上限 (秒)
):
    """一站式执行时空图全流程处理管线。

    消融控制:
      - geo_fencing=False: 跳过 filter_zero_utility_grids, 保留全部网格
      - knn_enabled=False:  将 K_neighbors 设为极大值, 保留所有满足
                             max_travel_time 的弧段
    """
    _y = y_levels if y_levels is not None else list(range(1, C_max + 1))

    # 初始化全维度指标采集器
    collector = MetricsCollector(
        experiment_id=experiment_id,
        instance_name=instance_name,
        config={
            "geo_fencing": geo_fencing,
            "knn_enabled": knn_enabled,
            "K_neighbors": K_neighbors if knn_enabled else "unlimited",
            "P_intervals": P_intervals,
            "C_max": C_max,
            "T_total": T_total,
        }
    )
    collector.geo_fencing_enabled = geo_fencing
    collector.knn_enabled = knn_enabled
    collector.knn_k_value = K_neighbors if knn_enabled else 0

    if verbose:
        pruning_info = []
        if geo_fencing:
            pruning_info.append("Geo-Fence")
        if knn_enabled:
            pruning_info.append(f"KNN(K={K_neighbors})")
        pruning_str = " + ".join(pruning_info) if pruning_info else "无剪枝(消融模式)"
        print(f"[{experiment_id}] 数据={data_file} | "
              f"C={C_max} T={T_total} P={P_intervals} "
              f"Y={{{min(_y)}..{max(_y)}}} "
              f"Pruning=[{pruning_str}] | speed={vehicle_speed_kmh}km/h")

    # Step 1-2: 原始快照加载与地理坐标解析
    grids, grid_params, snapshot_df = prepare_optimize_inputs(
        data_file, target_datetime=target_datetime
    )
    original_grid_count = len(grids)
    grid_coords = extract_grid_coordinates(snapshot_df)

    # 缺省状态下自动计算全网格质心作为 Depot
    if depot_lat is None or depot_lon is None:
        depot_lat = float(np.mean([c[0] for c in grid_coords.values()]))
        depot_lon = float(np.mean([c[1] for c in grid_coords.values()]))

    travel_time = build_full_travel_time_matrix(
        grids, grid_coords, depot_lat, depot_lon, vehicle_speed_kmh
    )

    # Step 3: 静态离散断点效用矩阵预计算
    Omega, tau_list = generate_offline_utility_matrix(
        grids=grids,
        C_max=C_max,
        T_total=T_total,
        P_intervals=P_intervals,
        grid_params=grid_params,
        calc_utility_func=calculate_operational_utility,
        y_levels=y_levels,
    )

    # Step 4: 空间零收益过滤 (条件执行 — 消融实验对照组跳过此步)
    if geo_fencing:
        active_grids, Omega, active_params, active_coords, removed_zeros = \
            filter_zero_utility_grids(grids, Omega, grid_params, grid_coords)
    else:
        active_grids = list(grids)
        active_Omega = Omega
        active_params = dict(grid_params)
        active_coords = dict(grid_coords)
        removed_zeros = 0

    # 针对活跃集二次构建高精度耗时图
    travel_time = build_full_travel_time_matrix(
        active_grids, active_coords, depot_lat, depot_lon, vehicle_speed_kmh
    )

    # 记录剪枝统计
    total_possible_arcs = (len(active_grids) + 1) * len(active_grids)
    feasible_arcs_count = sum(
        1 for i in travel_time for j in travel_time[i]
        if i != j and travel_time[i][j] <= max_travel_time
    )

    if verbose:
        if geo_fencing:
            print(f"  [Geo-Fencing ✓] 网格剪枝: {original_grid_count}→{len(active_grids)} "
                  f"(剔除 {removed_zeros} 个零收益网格)")
        else:
            print(f"  [Geo-Fencing ✗] 消融模式: 保留全部 {original_grid_count} 个网格 (零效用过滤已跳过)")
        print(f"  [弧段统计] 可行弧段: {feasible_arcs_count} / ~{total_possible_arcs} "
              f"({100*feasible_arcs_count/max(1,total_possible_arcs):.1f}%)")

    collector.original_grid_count = original_grid_count
    collector.active_grid_count = len(active_grids)
    collector.removed_zero_utility = removed_zeros
    collector.feasible_arc_count = feasible_arcs_count
    collector.total_possible_arc_count = total_possible_arcs

    if not active_grids:
        if verbose:
            print("\n  [终止] 全局网格在当前切片下均无正收益效用，取消外派调度。")
        collector.solve_status = "SKIPPED"
        return {
            "model": None, "grids": [], "travel_time": travel_time,
            "Omega": Omega, "tau_list": tau_list,
            "status": "SKIPPED — all grids zero-utility", "objective": 0.0, "route": [],
            "collector": collector,
        }

    # ===== Step 5: 调用时空图底座引擎求解 =====
    # 根据消融设置调整 K_neighbors
    _effective_K = K_neighbors if knn_enabled else 99999
    progress = GurobiProgressTracker(label=experiment_id) if verbose else None
    if verbose:
        knn_info = f"K={_effective_K}" if knn_enabled else "K=∞ (消融模式)"
        print(f"[求解] 启动 Space-Time Graph 优化引擎 | KNN={knn_info}...")

    t_start = time.perf_counter()
    model = optimize_evrp_with_stgraph(
        grids=active_grids,
        travel_time=travel_time,
        C_max=C_max,
        T_total=T_total,
        Omega=Omega,
        tau_list=tau_list,
        swap_time_c=swap_time_c,
        progress_tracker=progress,
        max_travel_time=max_travel_time,
        y_levels=y_levels,
        K_neighbors=_effective_K,
        time_limit_s=time_limit_s,
    )
    t_elapsed = time.perf_counter() - t_start

    # 采集求解效率指标
    collector.record_solve(progress, t_elapsed, model)

    # ===== Step 6: 高维时空有向流结果解析与落地 =====
    result = _parse_stgraph_solution(
        model, active_grids, travel_time, swap_time_c, tau_list,
        grid_params=active_params, T_total=T_total, verbose=verbose
    )

    # 采集解质量与剪枝指标
    collector.record_solution(result)

    # 统一实验指标导出 (始终执行, 确保结果落盘)
    export_experiment_result(collector, progress, result, output_dir, verbose=verbose)

    # 指针回传绑定
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
        "total_arcs_possible": total_possible_arcs,
        "feasible_arcs": feasible_arcs_count,
        "max_travel_time_h": max_travel_time,
        "geo_fencing_enabled": geo_fencing,
        "knn_enabled": knn_enabled,
        "K_neighbors": K_neighbors if knn_enabled else 0,
    }
    result["num_st_arcs"] = len(model._st_arcs) if hasattr(model, '_st_arcs') else 0

    # ── Route visualization ──────────────────────────────────────
    if result["route"]:
        try:
            vis_path = visualize_route(
                result, active_coords, snapshot_df, output_dir,
                depot_coords=(depot_lat, depot_lon),
                experiment_id=experiment_id, instance_name=instance_name,
                vehicle_speed_kmh=vehicle_speed_kmh, swap_time_c=swap_time_c,
                C_max=C_max,
            )
            if vis_path and verbose:
                print(f"  [Route viz] {vis_path}")
        except Exception as vis_err:
            if verbose:
                print(f"  [Viz warning] {vis_err}")

    return result


# =========================================================================
# 高维时空弧链路 (Arc-Chain) 解码核心算子
# =========================================================================
def _parse_stgraph_solution(model, grids, travel_time, swap_time_c, tau_list,
                            grid_params=None, T_total=1.0, verbose=True):
    """从 ST-Graph 模型的时空网络流向一维离散物理动作序列进行严格拓扑映射与时序保真校验。

    新增:
      - grid_params: 用于在解码后计算 Soon/Normal 效用分流
      - T_total: 规划周期长度, 传入效用分流函数
    """
    status_map = {
        GRB.OPTIMAL: "OPTIMAL (全局最优)",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT (超时截断)",
        GRB.INFEASIBLE: "INFEASIBLE (数学上不可行)",
    }
    status = status_map.get(model.Status, f"状态码: {model.Status}")
    obj_val = model.ObjVal if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) else None

    result = {"status": status, "objective": obj_val, "route": [], "summary": {}}

    if obj_val is None:
        if verbose:
            print("  [警告] 模型未寻获任何合法整数可行解，终止路由解码。")
        return result

    x = model._x
    st_arcs = model._st_arcs
    depot = 0

    # 过滤出所有流值为 1.0 的高维时空有向弧段
    activated_arcs = [st_arcs[e] for e in range(len(st_arcs)) if x[e].X > 0.5]
    
    # 提取实际被触达的空间活跃子集
    visited_spatial = [j for j in grids if model._v[j].X > 0.5]
    result["visited_grids"] = visited_spatial

    if verbose:
        print(f"  [{status}] 变现资产 obj={obj_val:.4f} | 物理空间触达节点={len(visited_spatial)}")

    if not activated_arcs:
        if verbose:
            print("  [提示] 车辆未出站。")
        return result

    # 核心：顺藤摸瓜。沿着时空流的 DAG 有向拓扑，以 (current_node, current_s) 为线索还原绝对唯一的行驶链
    route_chain = []
    curr_node = depot
    curr_s = 0
    max_steps_protection = max(len(visited_spatial) + 5, len(tau_list) + 2)  # 防死循环安全边界，随访问节点数自适应缩放

    while max_steps_protection > 0:
        # 检索当前时刻、当前网格点的下一个飞跃目标弧段
        next_arc = None
        for arc in activated_arcs:
            if arc[0] == curr_node and arc[1] == curr_s:
                next_arc = arc
                break
        
        if next_arc is None:
            break
        
        route_chain.append(next_arc)
        curr_node = next_arc[2]
        curr_s = next_arc[3]
        
        # 一旦时空汇流点收敛回 DEPOT，整条移动闭环链路完美还原完成
        if curr_node == depot:
            break
        max_steps_protection -= 1

    # 拼装符合下游标准数据看板要求的结构体
    for idx, arc in enumerate(route_chain):
        i, s1, j, s2, y_val = arc
        if idx == 0:
            # 录入始发 DEPOT 属性
            result["route"].append({
                "grid": "DEPOT",
                "arrival_time": 0.0,
                "y_swapped": 0,
            })
        
        if j == depot:
            # 映射模型对到达 DEPOT 的离散时序声明
            result["route"].append({
                "grid": "DEPOT",
                "arrival_time": tau_list[s2],
                "y_swapped": 0,
            })
        else:
            # 映射标准空间网格的触达属性，通过联动变量 model._u 读取高精度连续时间
            result["route"].append({
                "grid": j,
                "arrival_time": model._u[j].X,
                "y_swapped": int(round(model._y[j].X)),
            })

    # =========================================================================
    # 物理真实模拟器检验 (时序一致性双向交叉审计)
    # =========================================================================
    total_swaps = sum(r["y_swapped"] for r in result["route"])
    total_travel_time = 0.0
    
    # 根据真实的 haversine 地理速度进行累加校验
    spatial_seq = [depot] + [r["grid"] for r in result["route"] if r["grid"] != "DEPOT"] + [depot]
    for idx in range(len(spatial_seq) - 1):
        i_node, j_node = spatial_seq[idx], spatial_seq[idx + 1]
        total_travel_time += travel_time[i_node][j_node]
        
    total_service_time = total_swaps * swap_time_c
    makespan = total_travel_time + total_service_time

    # =========================================================================
    # Soon/Normal 效用分流计算 (逐节点调用分流函数)
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
                # 将分流值绑定到每条路由记录
                r["utility_soon"] = split["soon"]
                r["utility_normal"] = split["normal"]
                r["utility_low"] = split["low"]

    result["summary"] = {
        "num_visited": len(visited_spatial),
        "total_swaps": total_swaps,
        "total_travel_time_hrs": total_travel_time,
        "total_service_time_hrs": total_service_time,
        "makespan_hrs": makespan,
        "route_length": len(spatial_seq) - 1,
        # Soon/Normal 分流
        "utility_soon": round(utility_soon_total, 6),
        "utility_normal": round(utility_normal_total, 6),
        "utility_low": round(utility_low_total, 6),
    }

    if verbose and result["route"]:
        print(f"\n  时空扩展还原路由序列 ({len(result['route'])} 动作节点, 换电数量 {total_swaps} 块):")
        cum_time = 0.0
        timing_issues = 0
        prev_spatial = depot
        
        for idx, step in enumerate(result["route"]):
            if step["grid"] == "DEPOT":
                if idx > 0:
                    cum_time += (swap_time_c * result["route"][idx - 1]["y_swapped"] + travel_time[prev_spatial][depot])
                print(f"    [DEPOT] 站点  物理模拟时间={cum_time:.3f}h | 模型拓扑时间={step['arrival_time']:.3f}h")
            else:
                grid = step["grid"]
                cum_time += travel_time[prev_spatial][grid]
                model_u = step["arrival_time"]
                y_swap = step["y_swapped"]
                
                # 双向对齐校验：时空图由于右取整政策，模型时间 u 允许略微比物理时间 cum 快（即处于右对齐状态）
                # 这是时空扩展图的天然数学特质。但在物理上，只要模型 u 没超过 T 且 u >= cum，即为绝对可行。
                diff = model_u - cum_time
                flag = " ⚠ 物理超时" if diff < -1e-4 else ""
                if diff < -1e-4:
                    timing_issues += 1
                    
                print(f"    → 网格 {grid}  物理模拟累加={cum_time:.3f}h  模型联动锚定={model_u:.3f}h (换电 y={y_swap}){flag}")
                cum_time += swap_time_c * y_swap
                prev_spatial = grid

        if timing_issues > 0:
            print(f"  ⚠ 发现 {timing_issues} 处物理时序逆差突破可行边界！")
        else:
            print(f"  ✓ 时空图时序一阶偏序审计通过 | 行驶净耗时 {total_travel_time:.3f}h | 换电净耗时 {total_service_time:.3f}h | 周期总长 {makespan:.3f}h")

    return result


# =========================================================================
# 生产级仿真总入口 (Production Simulation Entry Point)
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ST-Graph 时空图优化引擎 —— E-Bike Battery Swapping VRP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main_STGraph.py                                                 # 使用默认时间
  python main_STGraph.py --datetime "2025/10/28 14:00"                   # 指定具体时间
  python main_STGraph.py --random                                        # 随机选取可用小时
  python main_STGraph.py --random --seed 42                              # 随机选取 (固定种子)
  python main_STGraph.py --list-hours                                    # 列出可用小时
  python main_STGraph.py --data path/to/other.csv                        # 指定数据文件
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
        print("  [随机] 随机选取模式 — ST-Graph 时空图优化引擎")
        if args.seed is not None:
            print(f"  随机种子: {args.seed}")
    elif args.datetime is not None:
        target_datetime = args.datetime
        print("=" * 60)
        print("  [指定] 用户指定日期时间 — ST-Graph 时空图优化引擎")
    else:
        target_datetime = DEFAULT_TARGET_DATETIME
        print("=" * 60)
        print("  [默认] 默认日期时间 — ST-Graph 时空图优化引擎")

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
        P_intervals=DEFAULT_P_INTERVALS,     # 固定 12 段 (每 5 分钟一个离散层级)
        y_levels=DEFAULT_Y_LEVELS,           # 换电深度控制：1~10
        swap_time_c=0.02,                    # 单块服务基准耗时：0.02h
        max_travel_time=DEFAULT_MAX_TRAVEL_TIME, # 空间长阻裁剪：0.2h
        K_neighbors=DEFAULT_K_NEIGHBORS,     # 强力空间稀疏化控制
        verbose=True,
        # 默认以 M1 完整模型运行
        experiment_id="M1",
        geo_fencing=True,
        knn_enabled=True,
    )