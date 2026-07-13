import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time as _time

def optimize_evrp_with_stgraph(
    grids,
    travel_time,
    C_max,
    T_total,
    Omega,
    tau_list,
    swap_time_c,
    progress_tracker=None,
    max_travel_time=0.2,
    y_levels=None,
    K_neighbors=400
):
    """
    构建并求解基于时间扩展图 (Space-Time Graph, ST-Graph) 架构的电动汽车路径规划问题 (EVRP)。
    
    该架构通过将连续时间显式离散化并直接嵌入有向无环拓扑结构 (DAG)，从根本上消除了传统模型中
    极其沉重的 MTZ 子回路消除约束、大 M 时序同步约束以及多维增量式二进制张量。
    为了与现有的结果解析管线及启发式热启动生态完美兼容，本模型保留并联动了 v, u, y 变量。

    参数:
    - grids: 活跃网格 ID 列表 (已过滤零效用网格)
    - travel_time: 完整旅行时间矩阵 (含 depot=0), travel_time[i][j] 单位: 小时
    - C_max: 车载电池最大容量
    - T_total: 规划周期总时长 (小时)
    - Omega: 预计算效用矩阵, Omega[j][y][s]
    - tau_list: PLA 时间断点列表 (当前配置为 12 个时间段，13 个时间交界点)
    - swap_time_c: 单块电池换电服务时间 (小时)
    - progress_tracker: GurobiProgressTracker 实例或 None
    - max_travel_time: 弧段裁剪绝对物理阈值 (小时)
    - y_levels: 离散换电量列表。若为 None 则使用 range(1, C_max+1)
    - K_neighbors: 每个空间节点的最大近邻限制数 (KNN 稠密图稀疏化裁剪)
    """
    m = gp.Model("EVRP_SpaceTimeGraph_DeltaEngine")

    # ==========================================================================
    # Solver parameters (对齐 Delta-MCF 调优经验 — 无 LazyConstraints 故不需关闭 Presolve)
    # ==========================================================================
    # STGraph 与 Delta-MCF 的关键区别：无需 LazyConstraints=1，因此 Gurobi 可以正常
    # 进行预求解和割平面生成。参数配置借鉴 Delta-MCF 的调优经验，但保留割平面与启发式
    # 以充分利用完整的 MIP 求解能力。
    m.setParam('MIPGap', 0.01)
    m.setParam('TimeLimit', 1800)
    m.setParam('Method', 2)              # Barrier — 时空图LP松弛规模大，Barrier优势显著
    m.setParam('Crossover', 0)           # 禁用crossover → 省时间，直接进B&B
    m.setParam('MIPFocus', 0)            # 均衡策略（Warm-Start已提供初始可行解）
    m.setParam('Heuristics', 0.10)       # 轻量启发式（根节点LP较贵，不宜久留）
    m.setParam('Cuts', 1)                # 保留割平面（无LazyConstraints限制，可正常生成）
    m.setParam('Symmetry', 0)            # 关闭对称性处理（省时间）
    m.setParam('NoRelHeurTime', 15)      # 15s启发式（不再占用60s）
    m.setParam('Threads', 8)             # 八核全开
    m.setParam('PreDual', 0)             # 禁用对偶预求解（同Delta-MCF）
    m.setParam('PrePasses', 8)           # 尽力预求解
    m.setParam('NodeMethod', 1)          # 节点松弛采用对偶单纯形热启动
    m.setParam('ImproveStartTime', 0)    # 跳过根节点改善阶段，直接进入B&B分支
    m.setParam('OutputFlag', 1)          # 开启Gurobi原生日志
    m.setParam('BarHomogeneous', 1)      # Barrier同质模型优化

    # ==========================================================================
    # Parameter Notes (时空图特有的边界无大 M 说明)
    # ==========================================================================
    # 在时空图（ST-Graph）拓扑中，由于时间的物理推进直接嵌套于图的层级有向边中，
    # 传统模型必需的 BigM_mtz, BigM_sync, BigM_deadline 均在数学结构上被完全消灭。
    if progress_tracker is None:
        print(f"  [ST-Graph 架构提示] 连续时序已成功转换为拓扑网络层级结构。")
        print(f"  [ST-Graph 架构提示] 成功消灭所有大 M 约束，消除连续松弛严重退化的底层瓶颈。")

    # ==========================================================================
    # 0. 索引集合与空间拓扑剪枝 (KNN 邻域限制 + max_travel_time 裁剪)
    # ==========================================================================
    depot = 0
    nodes = [depot] + grids
    Y_domain = list(range(1, C_max + 1)) if y_levels is None else sorted(y_levels)
    P = len(tau_list) - 1  # 离散时间分段数 (当前为 12)

    # 右取整时间对齐辅助算子：寻找满足 tau_list[s] >= t_arr 的最小离散步 index
    def get_right_rounded_step(t_arr):
        for s, token_t in enumerate(tau_list):
            if token_t >= t_arr - 1e-7:  # 引入微小扰动规避浮点精度截断
                return s
        return None

    # 构建空间近邻网络
    spatial_neighbors = {i: [] for i in nodes}
    for i in nodes:
        candidates = []
        for j in nodes:
            if i != j and travel_time[i][j] <= max_travel_time:
                candidates.append((travel_time[i][j], j))
        
        # 严格按物理距离由近到远排序，保留最具吸引力的 K_neighbors 个目的地
        candidates.sort(key=lambda x: x[0])
        spatial_neighbors[i] = [tgt for _, tgt in candidates[:K_neighbors]]
        
        # 确保终点 DEPOT 始终在所有空间节点的生存圈内，防止强制返程被误剪枝
        if i != depot and depot not in spatial_neighbors[i] and travel_time[i][depot] <= T_total:
            spatial_neighbors[i].append(depot)

    # ==========================================================================
    # 0b. 时空图网络拓扑构建 (Space-Time Arc Precomputation)
    # ==========================================================================
    # 时空有向弧元组格式: (i, s1, j, s2, y_val) 
    # 代表含义：服务车在第 s1 时间步从节点 i 出发，在 i 点执行换电 y_val 块，并在第 s2 时间步准时到达 j 点
    st_arcs = []
    outflow_map = {(n, s): [] for n in nodes for s in range(P + 1)}
    inflow_map  = {(n, s): [] for n in nodes for s in range(P + 1)}
    arc_idx = 0

    # A. 构造从 Depot 出发的首发有向时空弧段 (固定起始步 s1=0，首站出发不涉及换电消耗 y=0)
    for j in spatial_neighbors[depot]:
        t_arr = travel_time[depot][j]
        s2 = get_right_rounded_step(t_arr)
        if s2 is not None and s2 <= P:
            arc_info = (depot, 0, j, s2, 0)
            st_arcs.append(arc_info)
            outflow_map[(depot, 0)].append(arc_idx)
            inflow_map[(j, s2)].append(arc_idx)
            arc_idx += 1

    # B. 遍历各个空间网格在任意合法出发时刻 s1 的转移时空有向弧段
    for i in grids:
        for s1 in range(P):  # 在期末时刻 P 出发已无物理意义
            for j in spatial_neighbors[i]:
                for y_val in Y_domain:
                    # 物理到达时间 = 当前起点离散物理时间 + 换电服务耗时 + 空间行驶耗时
                    t_arr = tau_list[s1] + (y_val * swap_time_c) + travel_time[i][j]
                    s2 = get_right_rounded_step(t_arr)
                    
                    # 只有当到达离散步未突破规划周期上限时，方可录入时空稠密网络
                    if s2 is not None and s2 <= P:
                        arc_info = (i, s1, j, s2, y_val)
                        st_arcs.append(arc_info)
                        outflow_map[(i, s1)].append(arc_idx)
                        inflow_map[(j, s2)].append(arc_idx)
                        arc_idx += 1

    total_possible_arcs = len(nodes) * (len(nodes) - 1) * len(Y_domain) * P
    if progress_tracker is None:
        print(f"  [空间拓扑稀疏化] 活跃网格数: {len(grids)} | 设定近邻数 K={K_neighbors}")
        print(f"  [时空图静态生成] 录入有效时空决策弧段数: {len(st_arcs)} / {total_possible_arcs} 条")

    # ==========================================================================
    # 1. 声明决策变量 (Decision Variables)
    # ==========================================================================
    # 核心网络流变量：x[e] = 1 代表车辆启用了第 e 条静态时空弧段
    x = m.addVars(len(st_arcs), vtype=GRB.BINARY, name="x")

    # 联动辅助变量：使数据层与原先的路径、时间、换电解析逻辑实现 100% 结构闭环
    v = m.addVars(grids, vtype=GRB.BINARY, name="v")
    u = m.addVars(nodes, vtype=GRB.CONTINUOUS, lb=0.0, name="u")
    y = m.addVars(nodes, vtype=GRB.INTEGER, lb=0, ub=C_max, name="y")

    # ---- 构建字符串键查找表 (免疫 numpy/pandas 类型漂移，供 Warm-Start 与解析器强健检索) ----
    _v_by_str = {str(k): var for k, var in v.items()}
    _u_by_str = {str(k): var for k, var in u.items()}
    _y_by_str = {str(k): var for k, var in y.items()}
    
    # 时空有向弧多维键精准对照表 (str(i), s1, str(j), s2, y) -> x_var
    _st_arc_lookup = {}
    for e, (i, s1, j, s2, y_val) in enumerate(st_arcs):
        _st_arc_lookup[(str(i), int(s1), str(j), int(s2), int(y_val))] = x[e]

    # ==========================================================================
    # 1b. 设定分支变量优先权 (Variable Branching Priorities)
    # ==========================================================================
    for j in grids:
        v[j].BranchPriority = 10

    # ==========================================================================
    # 2. 目标函数设定 (Objective Function)
    # ==========================================================================
    # 若某条激活的时空弧的出发站为网格 i，则直接无偏结算对应的离散分段效用 Omega[i][y][s1]
    obj_expr = gp.quicksum(
        Omega[st_arcs[e][0]][st_arcs[e][4]][st_arcs[e][1]] * x[e]
        for e in range(len(st_arcs)) if st_arcs[e][0] != depot
    )
    m.setObjective(obj_expr, GRB.MAXIMIZE)

    # ==========================================================================
    # 3. 网络拓扑约束与联动等式 (Topological & Linking Constraints)
    # ==========================================================================
    
    # 约束 3a: 全局网格时空流平衡守恒 (进入某一网格时空点 (j, s) 的弧流量必须等于离开该点的流量)
    m.addConstrs(
        (gp.quicksum(x[e] for e in inflow_map[(j, s)]) == gp.quicksum(x[e] for e in outflow_map[(j, s)])
         for j in grids for s in range(P + 1)),
        name="ST_FlowConservation"
    )

    # 约束 3b: Depot 出发边界控制 (外派车辆有且至多只有一辆)
    m.addConstr(
        gp.quicksum(x[e] for e in outflow_map[(depot, 0)]) <= 1,
        name="DepotDeparture"
    )

    # 约束 3c: Depot 返程闭环 (出发的车辆最终必须在时空周期结束前的任意时刻步 s 返回至 Depot)
    m.addConstr(
        gp.quicksum(x[e] for s in range(1, P + 1) for e in inflow_map[(depot, s)]) ==
        gp.quicksum(x[e] for e in outflow_map[(depot, 0)]),
        name="DepotReturn"
    )

    # 约束 3d: 空间物理唯一性访问控制 (杜绝服务车在相同的空间网格点通过不断兜圈反复刷收益效益)
    m.addConstrs(
        (gp.quicksum(x[e] for s1 in range(P) for e in outflow_map[(j, s1)]) <= 1 for j in grids),
        name="SpatialGridUniqueVisit"
    )

    # 约束 3e: 全局车载总容量控制边界 (整条时空移动链路所累积的换电总量不能突破最大车载限制)
    m.addConstr(
        gp.quicksum(st_arcs[e][4] * x[e] for e in range(len(st_arcs))) <= C_max,
        name="GlobalBatteryCapacity"
    )

    # --------------------------------------------------------------------------
    # 联动约束组: 建立高维空间拓扑流量向微观一维通用属性变量 (v, u, y) 的严格映射映射
    # --------------------------------------------------------------------------
    
    # 联动 3f: 映射空间激活标志 v[j]
    m.addConstrs(
        (gp.quicksum(x[e] for s1 in range(P) for e in outflow_map[(j, s1)]) == v[j] for j in grids),
        name="Link_Variable_V"
    )

    # 联动 3g: 映射物理到达时间 u[j] (通过进入时空点所绑定的离散步时刻直接解出，不再需要时序大 M 扩散)
    m.addConstrs(
        (u[j] == gp.quicksum(tau_list[st_arcs[e][3]] * x[e] for s2 in range(P + 1) for e in inflow_map[(j, s2)])
         for j in grids),
        name="Link_Variable_U"
    )

    # 联动 3h: 映射网格执行换电量 y[j]
    m.addConstrs(
        (y[j] == gp.quicksum(st_arcs[e][4] * x[e] for s1 in range(P) for e in outflow_map[(j, s1)])
         for j in grids),
        name="Link_Variable_Y"
    )

    # 联动 3i: 固化起点 Depot 基础物理初始值
    m.addConstr(u[depot] == 0, name="StartTimeDepot")
    m.addConstr(y[depot] == 0, name="StartBatteryDepot")

    # ==========================================================================
    # 4. 贪心 Warm-Start 启发式算子 (Greedy Warm-Start Heuristic)
    # ==========================================================================
    def _build_warm_start():
        """构建贪心启发式路径，并将离散移动时空链完美转换映射为 ST-Graph 的最优初值。"""
        unvisited = set(grids)
        route_seq = [depot]
        cum_time = 0.0
        cum_swaps = 0
        arrival = {depot: 0.0}
        swap_at = {depot: 0}

        current = depot
        while unvisited:
            candidates = []
            for j in spatial_neighbors[current]:
                if j == depot:
                    continue
                if j not in unvisited:
                    continue
                tt = travel_time[current][j]
                proj_arr = cum_time + tt
                if proj_arr >= T_total:
                    continue
                s_idx = get_right_rounded_step(proj_arr)
                if s_idx is None:
                    continue
                max_u = max((Omega[j][yy][s_idx] for yy in Y_domain if cum_swaps + yy <= C_max), default=0.0)
                if max_u > 0:
                    candidates.append((max_u / max(tt, 0.001), j, proj_arr, tt))

            if not candidates:
                break
            candidates.sort(reverse=True, key=lambda tup: tup[0])
            _, best_j, proj_arr, tt = candidates[0]

            s_idx = get_right_rounded_step(proj_arr)
            if s_idx is None:
                continue
            best_y = max(((yy, Omega[best_j][yy][s_idx]) for yy in Y_domain if cum_swaps + yy <= C_max),
                         key=lambda p: p[1], default=(min(Y_domain), 0))[0]

            svc = swap_time_c * best_y
            dep_time = proj_arr + svc

            if dep_time + travel_time[best_j][depot] > T_total:
                saved = False
                for yy in sorted(Y_domain, reverse=True):
                    if cum_swaps + yy > C_max:
                        continue
                    if proj_arr + swap_time_c * yy + travel_time[best_j][depot] <= T_total:
                        best_y = yy
                        svc = swap_time_c * yy
                        dep_time = proj_arr + svc
                        saved = True
                        break
                if not saved:
                    continue

            route_seq.append(best_j)
            arrival[best_j] = proj_arr
            swap_at[best_j] = best_y
            cum_swaps += best_y
            cum_time = dep_time
            unvisited.remove(best_j)
            current = best_j

        route_seq.append(depot)
        if len(route_seq) <= 2:
            return None, 0

        # ---- 将贪心链精准激活并转化为时空有向弧网络初值 ----
        st_arcs_activated = 0
        current_s = 0
        
        for idx in range(len(route_seq) - 1):
            i, j = route_seq[idx], route_seq[idx + 1]
            if i == depot:
                t_arr = travel_time[depot][j]
                s2 = get_right_rounded_step(t_arr)
                if s2 is not None and s2 <= P:
                    var = _st_arc_lookup.get((str(depot), 0, str(j), int(s2), 0))
                    if var is not None:
                        var.Start = 1.0
                        st_arcs_activated += 1
                    current_s = s2
            elif j == depot:
                y_val = swap_at.get(i, 0)
                t_arr = tau_list[current_s] + (y_val * swap_time_c) + travel_time[i][depot]
                s2 = get_right_rounded_step(t_arr)
                if s2 is not None and s2 <= P:
                    var = _st_arc_lookup.get((str(i), int(current_s), str(depot), int(s2), int(y_val)))
                    if var is not None:
                        var.Start = 1.0
                        st_arcs_activated += 1
            else:
                y_val = swap_at.get(i, 0)
                t_arr = tau_list[current_s] + (y_val * swap_time_c) + travel_time[i][j]
                s2 = get_right_rounded_step(t_arr)
                if s2 is not None and s2 <= P:
                    var = _st_arc_lookup.get((str(i), int(current_s), str(j), int(s2), int(y_val)))
                    if var is not None:
                        var.Start = 1.0
                        st_arcs_activated += 1
                    current_s = s2

        # 赋予辅助通用联动变量 MIP Start
        for j_node in route_seq[1:-1]:
            j_key = str(j_node)
            if j_key in _v_by_str: _v_by_str[j_key].Start = 1.0
            if j_key in _y_by_str: _y_by_str[j_key].Start = swap_at[j_node]
            
            # 时间变量 u 完美锚定离散步真实物理时间（右取整对齐ST-Graph语义）
            uj_proj = arrival[j_node]
            uj_s_idx = get_right_rounded_step(uj_proj)
            if uj_s_idx is not None and j_key in _u_by_str:
                _u_by_str[j_key].Start = tau_list[uj_s_idx]

        if progress_tracker is None:
            print(f"  [Warm-Start STGraph] 成功注入时空初始网络，联锁激活有向流弧段: {st_arcs_activated} 条")
        return True, len(route_seq) - 2

    warm_ok, warm_visited = _build_warm_start()
    if warm_ok:
        m.NumStart = 1
    else:
        if progress_tracker is None:
            print("  [Warm-Start] 贪心路径不符合时空扩展裁剪图界，跳过初值注入")

    # ==========================================================================
    # 5. 精简求解监控管线 (Progress Monitor Callback)
    # ==========================================================================
    _solve_start = _time.perf_counter()
    _last_log_time = [0.0]

    def _progress_callback(model, where):
        if where != GRB.Callback.MIP:
            return
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        if runtime - _last_log_time[0] < 30:
            return
        _last_log_time[0] = runtime
        obj = model.cbGet(GRB.Callback.MIP_OBJBST)
        bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        nodes = int(model.cbGet(GRB.Callback.MIP_NODCNT))
        gap = abs(bnd - obj) / (abs(obj) + 1e-9) * 100
        print(f"  [STGraph-Engine] obj={obj:.4f}  bnd={bnd:.4f}  gap={gap:.1f}%  nodes={nodes}  t={runtime:.0f}s")
        if progress_tracker is not None:
            progress_tracker.record(runtime, obj, bnd, nodes)

    # 开启一体化单阶段求解
    m.optimize(_progress_callback)

    # === 兜底: 确保最后一帧被记录 (回调有30s节流, 最终bound可能急剧收敛) ===
    if progress_tracker is not None:
        try:
            final_runtime = m.Runtime
            final_obj = m.ObjVal if m.SolCount > 0 else None
            final_bnd = m.ObjBound
            final_nodes = int(m.NodeCount)
            progress_tracker.record(final_runtime, final_obj, final_bnd, final_nodes)
        except Exception:
            pass

    # ==========================================================================
    # 6. 指针属性回传绑定 (Downstream Pipeline References Alignment)
    # ==========================================================================
    m._x = x
    m._v = v
    m._u = u
    m._y = y
    m._grids = grids
    m._nodes = nodes
    m._st_arcs = st_arcs
    m._feasible_arcs = set((st_arcs[e][0], st_arcs[e][2]) for e in range(len(st_arcs)))

    return m