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
    # 求解器参数 (Tier-1 优化)
    # ==========================================================================
    m.setParam('MIPGap', 0.05)
    m.setParam('TimeLimit', 1800)
    # 移除 Barrier + 关闭 Crossover，改由 Gurobi 自动选择
    m.setParam('Method', -1)             # 自动选择算法
    m.setParam('Crossover', -1)          # 自动 (默认会在 Barrier 后执行交叉以提供顶点解)
    m.setParam('MIPFocus', 1)            # 优先寻找可行解 (快速获得 primal bound 用于剪枝)
    m.setParam('Heuristics', 0.5)        # 提高启发式搜索强度 (从 0.2 提升)
    m.setParam('Cuts', 2)                # 激进割平面生成
    m.setParam('PreDual', -1)            # 自动决定是否对偶预处理

    # ==========================================================================
    # 自适应 Big-M 计算 (Tier-1 优化)
    # ==========================================================================
    # 基于问题物理参数计算紧致 Big-M, 取代原先的硬编码 200 / 1000。
    # 过于松弛的 Big-M 是根节点 LP 退化的首要原因。
    _max_service = swap_time_c * C_max                        # 单节点最大服务耗时
    _max_arrival_diff = T_total + _max_service + max_travel_time  # 任意两点最大时间差

    # 保存用户原始传入值 (仅用于诊断日志)
    _BigM_original = BigM

    # 三类约束各自的最小有效 Big-M
    BigM_mtz      = _max_arrival_diff + 0.1                   # MTZ 子环消除
    BigM_sync     = T_total + 0.1                              # PLA 时间同步
    BigM_deadline = _max_service + max_travel_time + 0.1       # 周期截止约束

    # 若用户传入的 BigM 小于计算值则沿用 (尊重用户设置)
    if BigM > _max_arrival_diff:
        BigM = _max_arrival_diff + 0.1

    if progress_tracker is None:
        print(f"  [BigM 自适应] MTZ={BigM_mtz:.3f}  Sync={BigM_sync:.3f}  "
              f"Deadline={BigM_deadline:.3f}  "
              f"(原传入 {_BigM_original if _BigM_original <= 10 else '>1000'}, "
              f"已收紧至 ≤{BigM:.3f})")

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

    # ---- 构建字符串键查找表 (免疫 numpy/pandas 类型漂移, 供 Warm-Start 使用) ----
    _x_by_str = {}          # (str(i), str(j)) → Var
    for (i, j), var in x.items():
        _x_by_str[(str(i), str(j))] = var
    _v_by_str = {str(k): var for k, var in v.items()}
    _u_by_str = {str(k): var for k, var in u.items()}
    _y_by_str = {str(k): var for k, var in y.items()}
    _w_by_str = {}          # (str(j), str(yv)) → Var
    for (j, yv), var in w.items():
        _w_by_str[(str(j), str(yv))] = var
    _lam_by_str = {}        # (str(j), str(yv), str(s)) → Var
    for (j, yv, s_val), var in lam.items():
        _lam_by_str[(str(j), str(yv), str(s_val))] = var

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

        # 时间同步: 物理时间 u_j ≈ PLA 插值时间 (使用紧致 BigM_sync)
        pla_time_expr = gp.quicksum(
            tau_list[s] * lam[j, y_val, s] for y_val in Y_domain for s in S_domain
        )
        m.addConstr(
            u[j] - BigM_sync * (1 - v[j]) <= pla_time_expr,
            name=f"Time_Sync_LB_{j}"
        )
        m.addConstr(
            pla_time_expr <= u[j] + BigM_sync * (1 - v[j]),
            name=f"Time_Sync_UB_{j}"
        )

    # ==========================================================================
    # 5. MTZ 时序约束 (仅对可行弧生成)
    # ==========================================================================
    m.addConstr(u[depot] == 0, name="StartTimeDepot")
    m.addConstr(y[depot] == 0, name="StartBatteryDepot")

    # MTZ — 仅遍历可行弧, 且 j 必须为 grid (depot 不出现在 j 侧)
    # 使用紧致 BigM_mtz 替代原来的 BigM=200/1000
    for i in nodes:
        for j in feasible_out[i]:
            if j == depot:
                continue  # MTZ 弧 j 必须是 grid, depot 的 MTZ 由 CycleDeadline 处理
            m.addConstr(
                u[j] >= u[i] + (swap_time_c * y[i]) + travel_time[i][j]
                       - BigM_mtz * (1 - x[i, j]),
                name=f"MTZ_{i}_{j}"
            )

    # 周期截止 — 仅对存在 grid→depot 弧的 grid 生成
    # 使用紧致 BigM_deadline 替代原来的 BigM
    for i in grids:
        if depot in feasible_out[i]:
            m.addConstr(
                u[i] + (swap_time_c * y[i]) + travel_time[i][depot]
                - BigM_deadline * (1 - x[i, depot]) <= T_total,
                name=f"CycleDeadline_{i}"
            )

    # ==========================================================================
    # 5.5. 贪心 Warm-Start 启发式 (Tier-1 优化)
    # ==========================================================================
    # 最近邻贪心 + 效用/行程比排序, 构造一条初始可行路径。
    # 作为 Gurobi MIP Start 注入, 快速获得紧致 primal bound 用于剪枝。

    def _build_warm_start():
        """构建贪心路径并直接设置变量 Start 属性。返回 (ok: bool, num_visited: int)"""
        unvisited = set(grids)
        route_seq = [depot]
        cum_time = 0.0
        cum_swaps = 0
        arrival = {depot: 0.0}
        swap_at = {depot: 0}

        current = depot
        while unvisited:
            # 对每个候选节点评估效用/行程比
            candidates = []
            for j in unvisited:
                if (current, j) not in feasible_arcs:
                    continue
                tt = travel_time[current][j]
                proj_arr = cum_time + tt
                if proj_arr >= T_total:
                    continue
                s_idx = min(range(len(tau_list)),
                            key=lambda si: abs(tau_list[si] - proj_arr))
                max_u = max(
                    (Omega[j][yy][s_idx] for yy in Y_domain
                     if cum_swaps + yy <= C_max),
                    default=0.0,
                )
                if max_u > 0:
                    candidates.append((max_u / max(tt, 0.001), j, proj_arr, tt))

            if not candidates:
                break
            candidates.sort(reverse=True, key=lambda tup: tup[0])
            _, best_j, proj_arr, tt = candidates[0]

            # 决定换电量: 在当前到达时间下选效用最大的 y
            s_idx = min(range(len(tau_list)),
                        key=lambda si: abs(tau_list[si] - proj_arr))
            best_y = max(
                ((yy, Omega[best_j][yy][s_idx]) for yy in Y_domain
                 if cum_swaps + yy <= C_max),
                key=lambda p: p[1], default=(1, 0),
            )[0]

            svc = swap_time_c * best_y
            dep_time = proj_arr + svc

            # 可行性检查: 必须能返回 depot
            if ((best_j, depot) not in feasible_arcs
                    or dep_time + travel_time[best_j][depot] > T_total):
                # 尝试递减换电量以留出返程时间
                saved = False
                for yy in sorted(Y_domain, reverse=True):
                    if cum_swaps + yy > C_max:
                        continue
                    if (proj_arr + swap_time_c * yy
                            + travel_time[best_j][depot] <= T_total):
                        best_y = yy
                        svc = swap_time_c * yy
                        dep_time = proj_arr + svc
                        saved = True
                        break
                if not saved:
                    continue  # 此节点无法纳入, 跳过

            route_seq.append(best_j)
            arrival[best_j] = proj_arr
            swap_at[best_j] = best_y
            cum_swaps += best_y
            cum_time = dep_time
            unvisited.remove(best_j)
            current = best_j

        route_seq.append(depot)
        if len(route_seq) <= 2:                     # 仅 depot → depot
            return None, 0

        # ---- 通过字符串键查找表设置 MIP Start (免疫所有类型问题) ----
        tau_arr = tau_list
        warm_log_parts = []

        # 路由变量 x — 字符串键查找
        x_set = 0
        for idx in range(len(route_seq) - 1):
            i, j = route_seq[idx], route_seq[idx + 1]
            var = _x_by_str.get((str(i), str(j)))
            if var is not None:
                var.Start = 1
                x_set += 1
        warm_log_parts.append(f"x弧段={x_set}")

        # 访问 / 时间 / 换电量 + PLA 变量
        v_set = y_set = lam_set = 0
        for j in route_seq[1:-1]:                   # 跳过 depot
            j_key = str(j)

            # v[j]
            var = _v_by_str.get(j_key)
            if var is not None:
                var.Start = 1
                v_set += 1

            # u[j]
            var = _u_by_str.get(j_key)
            if var is not None:
                var.Start = arrival[j]

            # y[j]
            yj = swap_at[j]
            yj_key = str(yj)
            var = _y_by_str.get(j_key)
            if var is not None:
                var.Start = yj
                y_set += 1

            # w[j, y_val] — 仅所选 y_val 为 1
            for y_val in Y_domain:
                var = _w_by_str.get((j_key, str(y_val)))
                if var is not None:
                    var.Start = 1 if y_val == yj else 0

            # lam[j, y_val, s] — SOS2 线性插值
            uj = arrival[j]
            s_lo = max(si for si in range(len(tau_arr))
                       if tau_arr[si] <= uj + 1e-12)
            s_hi = min(si for si in range(len(tau_arr))
                       if tau_arr[si] >= uj - 1e-12)

            if s_lo == s_hi:
                for s in S_domain:
                    var = _lam_by_str.get((j_key, yj_key, str(s)))
                    if var is not None:
                        var.Start = 1.0 if s == s_lo else 0.0
                        lam_set += 1
            else:
                denom = tau_arr[s_hi] - tau_arr[s_lo]
                w_lo = ((tau_arr[s_hi] - uj) / denom) if denom > 0 else 1.0
                w_hi = ((uj - tau_arr[s_lo]) / denom) if denom > 0 else 0.0
                for s in S_domain:
                    var = _lam_by_str.get((j_key, yj_key, str(s)))
                    if var is not None:
                        if s == s_lo:
                            var.Start = w_lo
                        elif s == s_hi:
                            var.Start = w_hi
                        else:
                            var.Start = 0.0
                        lam_set += 1

            # 其他 y_val 的所有 lam 为 0
            for yy in Y_domain:
                if yy == yj:
                    continue
                for s in S_domain:
                    var = _lam_by_str.get((j_key, str(yy), str(s)))
                    if var is not None:
                        var.Start = 0.0

        warm_log_parts.append(f"v={v_set} y={y_set} lam>0={lam_set}")
        if progress_tracker is None:
            print(f"  [Warm-Start] 变量注入: {', '.join(warm_log_parts)}")

        return True, len(route_seq) - 2

    warm_ok, warm_visited = _build_warm_start()
    if warm_ok:
        m.NumStart = 1
        if progress_tracker is None:
            print(f"  [Warm-Start] 贪心启发式注入初始可行解, "
                  f"访问 {warm_visited} 个网格")
    else:
        if progress_tracker is None:
            print("  [Warm-Start] 贪心启发式未能构造可行解, 跳过")

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
