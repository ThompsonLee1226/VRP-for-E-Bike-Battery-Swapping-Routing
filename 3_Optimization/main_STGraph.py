from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from gurobipy import GRB

# 导入时空图优化引擎与标准预处理管线
from Optimize_PLA_STGraph import optimize_evrp_with_stgraph
from Grid_Utility import calculate_operational_utility
from Pre_Process import (
    DEFAULT_TARGET_DATETIME,
    generate_offline_utility_matrix,
    prepare_optimize_inputs,
    extract_grid_coordinates,
    haversine_distance,
    calculate_travel_time_matrix,
)

# =========================================================================
# 默认统一参数配置 (与原有 MCF 骨架体系保持高度严谨对齐)
# =========================================================================
DEFAULT_DATA_FILE = "3_Optimization\\Grid_Utility_Test.csv"
DEFAULT_OUTPUT_DIR = "Optimization Result"
DEFAULT_SPEED_KMH = 30.0
DEFAULT_C_MAX = 20
DEFAULT_T_TOTAL = 1.0
DEFAULT_P_INTERVALS = 12                     # 时空图推荐：每 5 分钟一段，共 12 段 (原 10)
DEFAULT_SWAP_TIME_C = 0.02
DEFAULT_MAX_TRAVEL_TIME = 0.2
DEFAULT_K_NEIGHBORS = 50                     # KNN 稠密空间图裁剪决策空间界限
DEFAULT_Y_LEVELS = list(range(1, 11))        # 离散换电量: 1~10


# =========================================================================
# Gurobi 求解进度追踪器 (精简健壮版)
# =========================================================================
class GurobiProgressTracker:
    """实时捕获、记录并精简输出 B&B 搜索树的收敛边界。"""
    def __init__(self):
        self.last = None

    def record(self, runtime, best_obj, best_bound, node_count):
        self.last = (runtime, best_obj, best_bound, node_count)

    def gap(self):
        if self.last is None:
            return float('inf')
        _, obj, bnd, _ = self.last
        return abs(bnd - obj) / (abs(obj) + 1e-9) * 100

    def summary(self):
        if self.last is None:
            return "无进度记录"
        runtime, obj, bnd, nodes = self.last
        return (f"{runtime:.0f}s obj={obj:.4f} bnd={bnd:.4f} "
                f"gap={self.gap():.1f}% nodes={nodes}")


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
):
    """一站式执行时空图全流程处理管线。"""
    _y = y_levels if y_levels is not None else list(range(1, C_max + 1))
    if verbose:
        print(f"[预处理] 数据={data_file} | "
              f"C={C_max} T={T_total} P={P_intervals} "
              f"Y={{{min(_y)}..{max(_y)}}} "
              f"KNN_K={K_neighbors} | speed={vehicle_speed_kmh}km/h")

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

    # Step 4: 空间零收益过滤
    active_grids, Omega, active_params, active_coords, removed_zeros = \
        filter_zero_utility_grids(grids, Omega, grid_params, grid_coords)

    # 针对活跃集二次构建高精度耗时图
    travel_time = build_full_travel_time_matrix(
        active_grids, active_coords, depot_lat, depot_lon, vehicle_speed_kmh
    )

    if verbose:
        print(f"  网格剪枝: {original_grid_count}→{len(active_grids)} "
              f"(成功剔除 {removed_zeros} 个零收益网格)")

    if not active_grids:
        if verbose:
            print("\n  [终止] 全局网格在当前切片下均无正收益效用，取消外派调度。")
        return {
            "model": None, "grids": [], "travel_time": travel_time,
            "status": "SKIPPED — all grids zero-utility", "objective": 0.0, "route": []
        }

    # ===== Step 5: 调用时空图底座引擎求解 =====
    progress = GurobiProgressTracker() if verbose else None
    if verbose:
        print(f"[求解] 启动一体化 Space-Time Graph 网络流图优化底座...")

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
        K_neighbors=K_neighbors
    )
    t_elapsed = time.perf_counter() - t_start

    # ===== Step 6: 高维时空有向流结果解析与落地 =====
    result = _parse_stgraph_solution(model, active_grids, travel_time, swap_time_c, tau_list, verbose=verbose)

    os.makedirs(output_dir, exist_ok=True)
    _save_results(result, output_dir, verbose=verbose)

    # 指针回传绑定
    result["model"] = model
    result["grids"] = active_grids
    result["travel_time"] = travel_time
    result["Omega"] = Omega
    result["tau_list"] = tau_list
    result["snapshot_df"] = snapshot_df
    result["grid_params"] = active_params
    result["elapsed_seconds"] = t_elapsed

    return result


# =========================================================================
# 高维时空弧链路 (Arc-Chain) 解码核心算子
# =========================================================================
def _parse_stgraph_solution(model, grids, travel_time, swap_time_c, tau_list, verbose=True):
    """从 ST-Graph 模型的时空网络流向一维离散物理动作序列进行严格拓扑映射与时序保真校验。"""
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

    result["summary"] = {
        "num_visited": len(visited_spatial),
        "total_swaps": total_swaps,
        "total_travel_time_hrs": total_travel_time,
        "total_service_time_hrs": total_service_time,
        "makespan_hrs": makespan,
        "route_length": len(spatial_seq) - 1,
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


def _save_results(result, output_dir, verbose=True):
    """将解码生成的一维标准化业务表格固化并写入 CSV 文件。"""
    route = result.get("route", [])
    if not route:
        return
    rows = [{"grid_id": s["grid"], "arrival_time_hrs": s["arrival_time"],
             "batteries_swapped": s["y_swapped"]} for s in route]
    csv_path = os.path.join(output_dir, f"optimization_route_stgraph_{time.time()}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    
    summary = result.get("summary", {})
    if summary:
        summary_path = os.path.join(output_dir, f"optimization_summary_stgraph_{time.time()}.csv")
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
    if verbose:
        print(f"  [数据中心] 业务报告及动作流表格已成功落盘至路径: {output_dir}/")


# =========================================================================
# 生产级仿真总入口 (Production Simulation Entry Point)
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
        P_intervals=DEFAULT_P_INTERVALS,     # 固定 12 段 (每 5 分钟一个离散层级)
        y_levels=DEFAULT_Y_LEVELS,           # 换电深度控制：1~10
        swap_time_c=0.02,                    # 单块服务基准耗时：0.02h
        max_travel_time=DEFAULT_MAX_TRAVEL_TIME, # 空间长阻裁剪：0.2h
        K_neighbors=DEFAULT_K_NEIGHBORS,     # 强力空间稀疏化控制
        verbose=True,
    )