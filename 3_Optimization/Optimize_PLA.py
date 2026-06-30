import gurobipy as gp
from gurobipy import GRB


def optimize_evrp_with_pla(
    grids,
    travel_time,
    C_max,
    T_total,
    Omega,
    tau_list,
    swap_time_c,
    BigM=1000,
    progress_tracker=None,
    max_travel_time=0.33,
):
    """
    构建并求解整合了双变量 PLA 的电动汽车路径规划问题 (EVRP)。

    参数:
    - grids: 活跃网格 ID 列表 (已过滤零效用网格)
    - travel_time: 完整旅行时间矩阵 (含 depot=0), travel_time[i][j] 单位: 小时
    - C_max: 车载电池最大容量
    - T_total: 规划周期总时长 (小时)
    - Omega: 预计算效用矩阵, Omega[j][y][s]
    - tau_list: PLA 时间断点列表
    - swap_time_c: 单块电池换电服务时间 (小时)
    - BigM: MTZ 大 M 常数
    - progress_tracker: GurobiProgressTracker 实例或 None
    - max_travel_time: 弧段裁剪阈值 (小时)。若 travel_time[i][j] > max_travel_time，
                       则不创建 x[i,j] 变量与 MTZ 约束。默认 0.33 小时 (≈20 分钟)
    """
    m = gp.Model("EVRP_PLA_GeoClipped")

    # ==========================================================================
    # 求解器参数
    # ==========================================================================
    m.setParam('MIPGap', 0.05)
    m.setParam('TimeLimit', 1800)
    m.setParam('Method', 2)              # Barrier
    m.setParam('Crossover', 0)           # 关闭交叉
    m.setParam('Heuristics', 0.2)

    # ==========================================================================
    # 0. 索引集合 & 弧段可行性预计算 (Geo-Fencing 边长裁剪)
    # ==========================================================================
    depot = 0
    nodes = [depot] + grids
    Y_domain = list(range(1, C_max + 1))
    S_domain = list(range(len(tau_list)))

    # 构建可行弧段集合: 只保留旅行时间 ≤ max_travel_time 的弧
    feasible_arcs = set()
    for i in nodes:
        for j in nodes:
            if i != j and travel_time[i][j] <= max_travel_time:
                feasible_arcs.add((i, j))

    # 为每个节点预计算可行入弧 / 出弧邻居 (加速后续约束生成)
    feasible_out = {i: [] for i in nodes}
    feasible_in  = {i: [] for i in nodes}
    for (i, j) in feasible_arcs:
        feasible_out[i].append(j)
        feasible_in[j].append(i)

    arc_list = sorted(feasible_arcs, key=lambda arc: (str(arc[0]), str(arc[1])))  # 确定性顺序

    total_possible = len(nodes) * (len(nodes) - 1)
    if progress_tracker is None:
        print(f"  [Geo-Fencing 边长裁剪] 弧段: {len(arc_list)} / {total_possible} "
              f"({100*len(arc_list)/max(1,total_possible):.1f}%) "
              f"| 阈值: ≤{max_travel_time}h")

    # ==========================================================================
    # 1. 声明决策变量
    # ==========================================================================
    # 路由变量 (稀疏 — 仅可行弧)
    x = m.addVars(arc_list, vtype=GRB.BINARY, name="x")
    v = m.addVars(grids, vtype=GRB.BINARY, name="v")

    # 时间与换电量
    u = m.addVars(nodes, vtype=GRB.CONTINUOUS, lb=0.0, name="u")
    y = m.addVars(nodes, vtype=GRB.INTEGER, lb=0, ub=C_max, name="y")

    # PLA 变量
    w = m.addVars(grids, Y_domain, vtype=GRB.BINARY, name="w")
    lam = m.addVars(grids, Y_domain, S_domain, vtype=GRB.CONTINUOUS, lb=0.0, name="lam")

    # ==========================================================================
    # 2. 目标函数
    # ==========================================================================
    m.setObjective(
        gp.quicksum(Omega[j][y_val][s] * lam[j, y_val, s]
                    for j in grids for y_val in Y_domain for s in S_domain),
        GRB.MAXIMIZE
    )

    # ==========================================================================
    # 3. 基础拓扑约束 (适配稀疏弧段)
    # ==========================================================================
    # 入度 / 出度守恒 — 仅对可行弧求和
    m.addConstrs(
        (gp.quicksum(x[i, j] for i in feasible_in[j]) == v[j] for j in grids),
        name="FlowIn"
    )
    m.addConstrs(
        (gp.quicksum(x[j, k] for k in feasible_out[j]) == v[j] for j in grids),
        name="FlowOut"
    )

    # Depot 出发 (至多 1 辆车)
    depot_out = feasible_out[depot]
    depot_in  = feasible_in[depot]
    m.addConstr(
        gp.quicksum(x[depot, j] for j in depot_out) <= 1,
        name="DepotDeparture"
    )
    m.addConstr(
        gp.quicksum(x[i, depot] for i in depot_in) ==
        gp.quicksum(x[depot, j] for j in depot_out),
        name="DepotReturn"
    )

    # 全局换电上限
    m.addConstr(
        gp.quicksum(y[j] for j in grids) <= C_max,
        name="GlobalBatteryCapacity"
    )

    # ==========================================================================
    # 4. PLA 联动约束 (不受弧段裁剪影响)
    # ==========================================================================
    for j in grids:
        m.addConstr(
            gp.quicksum(w[j, y_val] for y_val in Y_domain) == v[j],
            name=f"MC_Sum_{j}"
        )
        m.addConstr(
            y[j] == gp.quicksum(y_val * w[j, y_val] for y_val in Y_domain),
            name=f"Y_Mapping_{j}"
        )
        m.addConstr(
            y[j] <= C_max * v[j],
            name=f"ValidCapacity_{j}"
        )

        for y_val in Y_domain:
            m.addConstr(
                gp.quicksum(lam[j, y_val, s] for s in S_domain) == w[j, y_val],
                name=f"CC_{j}_{y_val}"
            )
            m.addSOS(GRB.SOS_TYPE2, [lam[j, y_val, s] for s in S_domain])

        # 时间同步: 物理时间 u_j ≈ PLA 插值时间
        pla_time_expr = gp.quicksum(
            tau_list[s] * lam[j, y_val, s] for y_val in Y_domain for s in S_domain
        )
        m.addConstr(
            u[j] - BigM * (1 - v[j]) <= pla_time_expr,
            name=f"Time_Sync_LB_{j}"
        )
        m.addConstr(
            pla_time_expr <= u[j] + BigM * (1 - v[j]),
            name=f"Time_Sync_UB_{j}"
        )

    # ==========================================================================
    # 5. MTZ 时序约束 (仅对可行弧生成)
    # ==========================================================================
    m.addConstr(u[depot] == 0, name="StartTimeDepot")
    m.addConstr(y[depot] == 0, name="StartBatteryDepot")

    # MTZ — 仅遍历可行弧, 且 j 必须为 grid (depot 不出现在 j 侧)
    for i in nodes:
        for j in feasible_out[i]:
            if j == depot:
                continue  # MTZ 弧 j 必须是 grid, depot 的 MTZ 由 CycleDeadline 处理
            m.addConstr(
                u[j] >= u[i] + (swap_time_c * y[i]) + travel_time[i][j]
                       - BigM * (1 - x[i, j]),
                name=f"MTZ_{i}_{j}"
            )

    # 周期截止 — 仅对存在 grid→depot 弧的 grid 生成
    for i in grids:
        if depot in feasible_out[i]:
            m.addConstr(
                u[i] + (swap_time_c * y[i]) + travel_time[i][depot]
                - BigM * (1 - x[i, depot]) <= T_total,
                name=f"CycleDeadline_{i}"
            )

    # ==========================================================================
    # 6. 求解执行
    # ==========================================================================
    def _gurobi_progress_callback(model, where):
        if progress_tracker is None:
            return
        if where == GRB.Callback.MIP:
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            best_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            best_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            node_count = model.cbGet(GRB.Callback.MIP_NODCNT)
            progress_tracker.record(runtime, best_obj, best_bound, node_count)

    if progress_tracker is None:
        m.optimize()
    else:
        m.optimize(_gurobi_progress_callback)

    # 保存引用
    m._x = x
    m._v = v
    m._u = u
    m._y = y
    m._w = w
    m._lam = lam
    m._grids = grids
    m._nodes = nodes
    m._feasible_arcs = feasible_arcs     # 供下游解析使用

    return m
