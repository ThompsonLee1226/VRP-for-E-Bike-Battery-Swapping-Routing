import gurobipy as gp
from gurobipy import GRB


def optimize_evrp_with_pla_delta_mcf(
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
    y_levels=None,
):
    """
    Build and solve EVRP with PLA using the Incremental (Delta) Model
    ** augmented with MCF flow constraints + Lazy MTZ callback **.

    Replaces the ~135K pairwise MTZ big-M constraints (O(N²)) with:
      - MCF flow conservation → valid tour structure + battery accounting
      - TotalRouteTime → single global time budget constraint
      - CycleDeadline → depot return feasibility (O(N))
      - Lazy MTZ callback → dynamic timing enforcement at MIPSOL:
        every integer solution is intercepted, its route is traced, and
        violated arrival-time constraints are injected via cbLazy() on
        the fly. This guarantees temporal correctness without the
        "escape to unprotected nodes" flaw of multi-phase approaches.
    This dramatically reduces per-node LP cost while MCF provides
    the structural tightness for the LP relaxation.

    Parameters (same as optimize_evrp_with_pla_delta):
    - grids: active grid ID list (zero-utility nodes already filtered)
    - travel_time: full travel time matrix including depot=0
    - C_max: vehicle battery capacity
    - T_total: planning horizon (hours)
    - Omega: precomputed utility matrix, Omega[j][y][s]
    - tau_list: PLA time breakpoints
    - swap_time_c: service time per battery swap (hours)
    - BigM: user-supplied Big-M (automatically tightened internally)
    - progress_tracker: GurobiProgressTracker instance or None
    - max_travel_time: arc pruning threshold (hours)
    - y_levels: optional discrete swap-quantity levels (default: 1..C_max)
    """
    m = gp.Model("EVRP_PLA_Delta_MCF_NoMTZ")

    # ==========================================================================
    # Solver parameters (tuned for LazyConstraints=1 — large raw LP at root)
    # ==========================================================================
    # With LazyConstraints enabled, Gurobi cannot apply aggressive presolve
    # reductions. The root LP is ~135K rows and must be solved efficiently.
    # Strategy: Barrier for root LP, no cut loop, short heuristics, then branch.
    m.setParam('MIPGap', 0.05)
    m.setParam('TimeLimit', 1800)
    m.setParam('Method', 2)              # Barrier — 大规模LP远超Simplex效率
    m.setParam('Crossover', 0)           # 禁用crossover → 省时间，直接进B&B
    m.setParam('MIPFocus', 0)            # 均衡策略（Warm-Start已提供初始可行解）
    m.setParam('Heuristics', 0.10)       # 轻量启发式（根节点LP太贵，不宜久留）
    m.setParam('Cuts', 0)                # ★ 关闭割平面！MCF已紧固LP，割平面循环
                                         #   每轮重解LP消耗全部时间，得不偿失
    m.setParam('Symmetry', 0)            # 关闭对称性处理（省时间）
    m.setParam('NoRelHeurTime', 15)      # 15s启发式（不再占用60s）
    m.setParam('Threads', 8)
    m.setParam('PreDual', 0)             # 禁用对偶预求解（LazyConstraints下效果有限）
    m.setParam('PrePasses', 8)           # 尽力预求解（补偿LazyConstraints的限制）
    m.setParam('NodeMethod', 1)          # Dual Simplex for nodes（子节点热启动）
    m.setParam('ImproveStartTime', 0)    # 跳过根节点改善阶段（直接进入B&B分支）
    m.setParam('OutputFlag', 1)          # 开启Gurobi默认日志，显示原生进度输出
    m.setParam('LazyConstraints', 1)     # 启用原生延迟约束回调 (MIPSOL cbLazy)
    m.setParam('BarHomogeneous', 1)      # Barrier同质模型优化（MCF约束高度同质）

    # ==========================================================================
    # Adaptive Big-M
    # ==========================================================================
    _max_service = swap_time_c * C_max
    _max_arrival_diff = T_total + _max_service + max_travel_time
    _BigM_original = BigM

    BigM_mtz      = _max_arrival_diff + 0.1
    BigM_sync     = T_total + 0.1
    BigM_deadline = _max_service + max_travel_time + 0.1

    if BigM > _max_arrival_diff:
        BigM = _max_arrival_diff + 0.1

    if progress_tracker is None:
        print(f"  [BigM 自适应] MTZ={BigM_mtz:.3f}  Sync={BigM_sync:.3f}  "
              f"Deadline={BigM_deadline:.3f}  "
              f"(原传入 {_BigM_original if _BigM_original <= 10 else '>1000'}, "
              f"已收紧至 ≤{BigM:.3f})")

    # ==========================================================================
    # 0. Index Sets & Feasible Arcs
    # ==========================================================================
    depot = 0
    nodes = [depot] + grids
    Y_domain = list(range(1, C_max + 1)) if y_levels is None else sorted(y_levels)
    P = len(tau_list) - 1                           # number of intervals
    K_domain = list(range(1, P + 1))                # delta indices: 1..P

    # Build feasible arc set
    feasible_arcs = set()
    for i in nodes:
        for j in nodes:
            if i != j and travel_time[i][j] <= max_travel_time:
                feasible_arcs.add((i, j))

    feasible_out = {i: [] for i in nodes}
    feasible_in  = {i: [] for i in nodes}
    for (i, j) in feasible_arcs:
        feasible_out[i].append(j)
        feasible_in[j].append(i)

    arc_list = sorted(feasible_arcs, key=lambda arc: (str(arc[0]), str(arc[1])))

    total_possible = len(nodes) * (len(nodes) - 1)
    if progress_tracker is None:
        print(f"  [Geo-Fencing] 弧段: {len(arc_list)} / {total_possible} "
              f"({100*len(arc_list)/max(1,total_possible):.1f}%) "
              f"| 阈值: ≤{max_travel_time}h")

    # ==========================================================================
    # 0b. Precompute Delta-Omega (increments) and Delta-Tau (interval widths)
    # ==========================================================================
    Delta_Omega = {}
    for j in grids:
        Delta_Omega[j] = {}
        for y_val in Y_domain:
            Delta_Omega[j][y_val] = {}
            Delta_Omega[j][y_val][0] = Omega[j][y_val][0]
            for k in K_domain:
                Delta_Omega[j][y_val][k] = Omega[j][y_val][k] - Omega[j][y_val][k-1]

    Delta_tau_list = [tau_list[k] - tau_list[k-1] for k in K_domain]

    # ==========================================================================
    # 1. Decision Variables
    # ==========================================================================
    # Routing (sparse — only feasible arcs)
    x = m.addVars(arc_list, vtype=GRB.BINARY, name="x")
    v = m.addVars(grids, vtype=GRB.BINARY, name="v")

    # Time and swap quantity
    u = m.addVars(nodes, vtype=GRB.CONTINUOUS, lb=0.0, name="u")
    y = m.addVars(nodes, vtype=GRB.INTEGER, lb=0, ub=C_max, name="y")

    # PLA variables — DELTA formulation
    w = m.addVars(grids, Y_domain, vtype=GRB.BINARY, name="w")
    delta = m.addVars(grids, Y_domain, K_domain, vtype=GRB.BINARY, name="delta")

    # ==========================================================================
    # 1a. MCF Flow Variables (LP strengthening via single-commodity flow)
    # ==========================================================================
    # f[i,j] = battery flow on arc (i,j), representing remaining battery
    # capacity carried forward on the route. Continuous → no branching cost.
    # Flow conservation coupled with routing x[i,j] provides a much tighter
    # LP relaxation than MTZ alone for the routing component.
    f = m.addVars(arc_list, vtype=GRB.CONTINUOUS, lb=0.0, ub=C_max, name="f")

    if progress_tracker is None:
        print(f"  [MCF] 流变量: {len(arc_list)} 个连续变量 (不增加分支维度)")

    # ---- String-key lookup tables (immune to numpy/pandas type drift) ----
    _x_by_str = {}
    for (i, j), var in x.items():
        _x_by_str[(str(i), str(j))] = var
    _v_by_str = {str(k): var for k, var in v.items()}
    _u_by_str = {str(k): var for k, var in u.items()}
    _y_by_str = {str(k): var for k, var in y.items()}
    _w_by_str = {}
    for (j_key, yv), var in w.items():
        _w_by_str[(str(j_key), str(yv))] = var
    _delta_by_str = {}
    for (j_key, y_val, kval), var in delta.items():
        _delta_by_str[(str(j_key), str(y_val), str(kval))] = var

    # ==========================================================================
    # 1b. Variable Branching Priorities
    # ==========================================================================
    for j in grids:
        v[j].BranchPriority = 10
        for y_val in Y_domain:
            w[j, y_val].BranchPriority = 5
            for k in K_domain:
                delta[j, y_val, k].BranchPriority = -1   # 时间离散应由传播确定, 避免分支

    # ==========================================================================
    # 2. Objective Function (Delta formulation)
    # ==========================================================================
    obj_expr = gp.quicksum(
        Delta_Omega[j][y_val][0] * w[j, y_val]
        for j in grids for y_val in Y_domain
    )
    obj_expr.add(gp.quicksum(
        Delta_Omega[j][y_val][k] * delta[j, y_val, k]
        for j in grids for y_val in Y_domain for k in K_domain
    ))
    m.setObjective(obj_expr, GRB.MAXIMIZE)

    # ==========================================================================
    # 3. Topological Constraints
    # ==========================================================================
    m.addConstrs(
        (gp.quicksum(x[i, j] for i in feasible_in[j]) == v[j] for j in grids),
        name="FlowIn"
    )
    m.addConstrs(
        (gp.quicksum(x[j, k_out] for k_out in feasible_out[j]) == v[j] for j in grids),
        name="FlowOut"
    )

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

    m.addConstr(
        gp.quicksum(y[j] for j in grids) <= C_max,
        name="GlobalBatteryCapacity"
    )

    # ==========================================================================
    # 3b. MCF Single-Commodity Flow Constraints
    # ==========================================================================
    # The MTZ big-M constraints produce a weak LP relaxation because
    # fractional x[i,j] in the LP allows the big-M term to decouple
    # the routing from the timing. MCF provides a tight lower bound
    # on the flow (battery capacity) through the network, which couples
    # strongly with the routing variables — all without adding binary vars.
    #
    # Flow balance at depot:
    #   Σ f[depot,j] - Σ f[i,depot] = Σ y[j]
    # Total outflow minus inflow equals total batteries swapped.
    m.addConstr(
        gp.quicksum(f[depot, j] for j in depot_out)
        - gp.quicksum(f[i, depot] for i in depot_in)
        == gp.quicksum(y[j] for j in grids),
        name="MCF_DepotBalance"
    )

    # Flow balance at each grid j:
    #   Σ f[i,j] - Σ f[j,k] = y[j]
    # Flow entering minus flow leaving = batteries swapped (consumed) at j.
    # This forces the vehicle's battery load to decrease by exactly y[j]
    # when visiting grid j — a tight linear coupling.
    for j in grids:
        m.addConstr(
            gp.quicksum(f[i, j] for i in feasible_in[j])
            - gp.quicksum(f[j, k] for k in feasible_out[j])
            == y[j],
            name=f"MCF_Balance_{j}"
        )

    # Arc coupling:
    #   f[i,j] ≤ C_max · x[i,j]
    # If arc is not used, flow is zero. If used, flow bounded by capacity.
    # This is a SIMPLE linear inequality — no big-M, no SOS — and
    # provides the key tightening: fractional x ⇒ proportional f.
    m.addConstrs(
        (f[i, j] <= C_max * x[i, j] for (i, j) in arc_list),
        name="MCF_Coupling"
    )

    if progress_tracker is None:
        n_mcf_constrs = 1 + len(grids) + len(arc_list)
        print(f"  [MCF] 约束: DepotBalance + {len(grids)} Balance + "
              f"{len(arc_list)} Coupling = {n_mcf_constrs} 行")

    # ==========================================================================
    # 4. PLA Constraints — DELTA FORMULATION
    # ==========================================================================
    for j in grids:
        # w-sum = v[j]  (exactly one y_val if grid is visited)
        m.addConstr(
            gp.quicksum(w[j, y_val] for y_val in Y_domain) == v[j],
            name=f"W_Sum_{j}"
        )

        # y[j] = Σ y_val * w[j, y_val]
        m.addConstr(
            y[j] == gp.quicksum(y_val * w[j, y_val] for y_val in Y_domain),
            name=f"Y_Mapping_{j}"
        )

        # y[j] upper bound via v[j]
        m.addConstr(
            y[j] <= C_max * v[j],
            name=f"ValidCapacity_{j}"
        )

        for y_val in Y_domain:
            # --- coupling: delta_jyk ≤ w_jy ---
            for k in K_domain:
                m.addConstr(
                    delta[j, y_val, k] <= w[j, y_val],
                    name=f"DeltaCp_{j}_{y_val}_{k}"
                )

            # --- monotonicity staircase: δ₁ ≥ δ₂ ≥ ... ≥ δ_P ---
            for idx_k in range(len(K_domain) - 1):
                k_curr = K_domain[idx_k]
                k_next = K_domain[idx_k + 1]
                m.addConstr(
                    delta[j, y_val, k_curr] >= delta[j, y_val, k_next],
                    name=f"DeltaMono_{j}_{y_val}_{k_curr}"
                )

        # --- time synchronisation: u[j] = Σ_{y,k} Δτ_k · δ_jyk ---
        delta_time = gp.quicksum(
            Delta_tau_list[k_idx] * delta[j, y_val, K_domain[k_idx]]
            for y_val in Y_domain for k_idx in range(len(K_domain))
        )
        m.addConstr(u[j] == delta_time, name=f"TimeSync_{j}")

    # ==========================================================================
    # 5. Timing Constraints — Native Lazy Constraint Callback
    # ==========================================================================
    # Pairwise MTZ constraints are NOT added to the initial model.
    # Instead, they are enforced via Gurobi's native lazy constraint callback
    # (MIPSOL), running within a SINGLE continuous branch-and-bound tree:
    #
    #   - LP relaxation: no MTZ → fast per-node simplex solves
    #   - MIPSOL callback: intercepts EVERY integer solution, traces the actual
    #     route, and validates timing along each active arc. If any arc violates
    #     the physical arrival-time propagation, the corresponding MTZ cut is
    #     injected via cbLazy() — instantly rejecting the invalid solution.
    #   - This guarantees: no feasible integer solution can ever escape with
    #     physically impossible timing (e.g., u[j]=0 for all nodes), because
    #     the callback sees and rejects EVERY candidate.
    #
    # Key advantages over the previous multi-phase approach:
    #   (a) Single B&B tree — no restart overhead, no fragmented time budget
    #   (b) No "escape" to unprotected nodes — every solution is validated
    #   (c) Minimal constraint addition — only violated arcs get MTZ, not all
    #       O(N²) pairs upfront
    #
    # Static constraints retained:
    #   (a) StartTimeDepot / StartBatteryDepot
    #   (b) TotalRouteTime → global time budget (1 constraint)
    #   (c) CycleDeadline → depot return feasibility (1 per grid)
    #   (d) MCF → routing structure (no subtours)
    m.addConstr(u[depot] == 0, name="StartTimeDepot")
    m.addConstr(y[depot] == 0, name="StartBatteryDepot")

    # ---- Total route time budget (single linear constraint) ----
    total_travel = gp.quicksum(
        travel_time[i][j] * x[i, j] for (i, j) in arc_list)
    total_service = gp.quicksum(swap_time_c * y[j] for j in grids)
    m.addConstr(total_travel + total_service <= T_total, name="TotalRouteTime")

    # ---- Depot return deadline ----
    for i in grids:
        if depot in feasible_out[i]:
            m.addConstr(
                u[i] + (swap_time_c * y[i]) + travel_time[i][depot]
                - BigM_deadline * (1 - x[i, depot]) <= T_total,
                name=f"CycleDeadline_{i}"
            )

    # ---- Lazy MTZ tracking set (populated dynamically inside MIPSOL callback) ----
    _mtz_added = set()

    if progress_tracker is None:
        print(f"  [Timing] MTZ → 原生延迟约束回调 (MIPSOL), "
              f"单一B&B树内动态拦截")
        print(f"  [Timing] 静态约束: TotalRouteTime (1行) + "
              f"CycleDeadline (~{len(grids)}行)")

    # ==========================================================================
    # 5.5. Greedy Warm-Start Heuristic (adapted for delta encoding)
    # ==========================================================================
    def _build_warm_start():
        """Construct a greedy route and set variable Start attributes."""
        unvisited = set(grids)
        route_seq = [depot]
        cum_time = 0.0
        cum_swaps = 0
        arrival = {depot: 0.0}
        swap_at = {depot: 0}

        current = depot
        while unvisited:
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

            s_idx = min(range(len(tau_list)),
                        key=lambda si: abs(tau_list[si] - proj_arr))
            best_y = max(
                ((yy, Omega[best_j][yy][s_idx]) for yy in Y_domain
                 if cum_swaps + yy <= C_max),
                key=lambda p: p[1], default=(min(Y_domain), 0),
            )[0]

            svc = swap_time_c * best_y
            dep_time = proj_arr + svc

            if ((best_j, depot) not in feasible_arcs
                    or dep_time + travel_time[best_j][depot] > T_total):
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

        warm_log_parts = []

        # --- x (arcs) ---
        x_set = 0
        for idx in range(len(route_seq) - 1):
            i, j = route_seq[idx], route_seq[idx + 1]
            var = _x_by_str.get((str(i), str(j)))
            if var is not None:
                var.Start = 1
                x_set += 1
        warm_log_parts.append(f"x={x_set}")

        # --- v, u, y, w, delta for each visited grid ---
        v_set = y_set = 0
        for j in route_seq[1:-1]:                         # skip depot
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
            var = _y_by_str.get(j_key)
            if var is not None:
                var.Start = yj
                y_set += 1

            # w[j, y_val] — exactly one is 1
            for y_val in Y_domain:
                var = _w_by_str.get((j_key, str(y_val)))
                if var is not None:
                    var.Start = 1 if y_val == yj else 0

            # -----------------------------------------------------------------
            # delta[j, y_val, k] — integer staircase (BINARY-safe warm-start)
            # -----------------------------------------------------------------
            uj = arrival[j]
            uj_s_idx = min(range(len(tau_list)),
                           key=lambda si: abs(tau_list[si] - uj))
            uj_rounded = tau_list[uj_s_idx]
            var_u = _u_by_str.get(j_key)
            if var_u is not None:
                var_u.Start = uj_rounded

            for y_val in Y_domain:
                for k_idx, k in enumerate(K_domain):
                    var = _delta_by_str.get((j_key, str(y_val), str(k)))
                    if var is None:
                        continue
                    if y_val != yj:
                        var.Start = 0.0
                    else:
                        var.Start = 1.0 if k <= uj_s_idx else 0.0

        warm_log_parts.append(f"v={v_set} y={y_set}")
        if progress_tracker is None:
            print(f"  [Warm-Start Delta] 变量注入: {', '.join(warm_log_parts)}")

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
    # 6. Lazy Constraint Callback — Dynamic MTZ Enforcement in Single B&B Tree
    # ==========================================================================
    #
    # Architecture: instead of multiple .optimize() calls with static MTZ
    # injection between them, we use Gurobi's native MIPSOL lazy constraint
    # callback. The solver runs a SINGLE continuous branch-and-bound tree.
    # Every time it discovers an integer-feasible solution, the callback:
    #
    #   1. Traces the actual vehicle route from the binary x[i,j] values
    #   2. Validates temporal consistency along every active arc
    #   3. If violated → injects the MTZ cut via cbLazy() → rejects solution
    #   4. If valid → accepts as incumbent → solver continues
    #
    # This guarantees NO integer solution can escape with physically impossible
    # timing (e.g., all u[j]=0), because every candidate is intercepted at
    # MIPSOL before it can become the final answer.

    import time as _time
    _solve_start = _time.perf_counter()
    _total_time_budget = 1800.0
    _last_log_time = [0.0]          # mutable for closure
    _lazy_callback_count = [0]      # mutable: total MIPSOL invocations
    _lazy_mtz_count = [0]           # mutable: cumulative MTZ cuts added via cbLazy

    # ==========================================================================
    # 6a. Store references on model for callback access
    # ==========================================================================
    # Gurobi callbacks receive only (model, where); all data must be reachable
    # via model attributes or closure variables. We attach the lookup tables
    # and parameters to the model object so the callback can access them.
    m._travel_time = travel_time
    m._swap_time_c = swap_time_c
    m._BigM_mtz = BigM_mtz
    m._mtz_added = _mtz_added
    m._grids = grids
    m._nodes = nodes
    m._feasible_arcs = feasible_arcs
    m._depot = depot
    m._lazy_callback_count = _lazy_callback_count
    m._lazy_mtz_count = _lazy_mtz_count
    m._progress_tracker = progress_tracker
    m._last_log_time = _last_log_time

    # ==========================================================================
    # 6b. Lazy MTZ enforcement — fired at every integer feasible solution
    # ==========================================================================
    def _lazy_mtz_enforce(model, where):
        """MIPSOL callback: validate timing on the current integer route.

        Traces the actual route from the depot through visited grids and back,
        checking the physical arrival-time propagation u[j] >= u[i] + service[i]
        + travel[i][j] along each active arc (x[i,j] ≈ 1).

        If any arc violates this, injects the corresponding MTZ constraint
        via cbLazy() — which permanently adds it to the model and rejects
        the current solution. The solver then continues the B&B tree.

        NOTE: All Gurobi variable objects (x, v, u, y) are accessed via
        Python closure from the outer optimize_evrp_with_pla_delta_mcf scope.
        Data lookup tables (travel_time, swap_time_c, BigM_mtz, etc.) are
        stored on model._* attributes.
        """
        _lazy_callback_count[0] += 1

        # --- Step 1: identify visited nodes from binary v[j] ---
        visited = []
        for j in grids:
            if model.cbGetSolution(v[j]) > 0.5:
                visited.append(j)

        if not visited:
            return  # empty route is valid (no swaps, zero utility)

        # --- Step 2: trace the deterministic route ---
        # At MIPSOL all x[i,j] are binary → route is uniquely defined.
        route = [depot]
        current = depot

        _feasible = model._feasible_arcs
        MAX_STEPS = len(visited) + 5
        for _ in range(MAX_STEPS):
            found = False
            candidates = list(visited) + [depot]
            for nxt in candidates:
                if nxt == current:
                    continue
                if (current, nxt) not in _feasible:
                    continue
                # x is the gurobipy tupledict captured via closure
                if model.cbGetSolution(x[current, nxt]) > 0.5:
                    route.append(nxt)
                    current = nxt
                    found = True
                    break
            if not found:
                break                   # chain broken — shouldn't happen at MIPSOL
            if current == depot:
                break                   # returned to depot

        # --- Step 3: validate timing along each arc of the traced route ---
        # For each arc (i→j) where j != depot, enforce:
        #   u[j] >= u[i] + swap_time_c * y[i] + travel_time[i][j]
        # The depot has u=0, y=0 by static constraints, so the formula
        # works uniformly for depot→first_grid arcs too.
        cuts_added = 0
        _tol = 1e-5                     # match Gurobi IntFeasTol default
        _mtz_set = model._mtz_added
        _swap = model._swap_time_c
        _tt = model._travel_time
        _bigM = model._BigM_mtz

        for idx in range(len(route) - 1):
            i = route[idx]
            j = route[idx + 1]

            if j == depot:
                continue                # i→depot: covered by CycleDeadline statically

            # Skip if MTZ already exists for this arc
            if (i, j) in _mtz_set:
                continue

            # u, y are gurobipy tupledicts captured via closure
            ui = model.cbGetSolution(u[i])
            uj = model.cbGetSolution(u[j])
            yi = model.cbGetSolution(y[i])

            rhs = ui + _swap * yi + _tt[i][j]
            if uj < rhs - _tol:
                # --- VIOLATION: inject MTZ lazy constraint via cbLazy ---
                # The constraint uses the big-M form so it's slack when
                # x[i,j]=0 and binding only when the arc is active.
                model.cbLazy(
                    u[j] >= u[i] + _swap * y[i] + _tt[i][j]
                    - _bigM * (1 - x[i, j])
                )
                _mtz_set.add((i, j))
                cuts_added += 1

        if cuts_added > 0:
            _lazy_mtz_count[0] += cuts_added

    # ==========================================================================
    # 6c. Merged callback — progress logging (MIP) + lazy MTZ (MIPSOL)
    # ==========================================================================
    def _combined_callback(model, where):
        if where == GRB.Callback.MIP:
            # ---- Progress logging (every 30s) ----
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            if runtime - _last_log_time[0] < 30:
                return
            _last_log_time[0] = runtime
            obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            nodes_cnt = int(model.cbGet(GRB.Callback.MIP_NODCNT))
            gap = abs(bnd - obj) / (abs(obj) + 1e-9) * 100
            lazy_info = (f" lazy={_lazy_mtz_count[0]}"
                         if _lazy_mtz_count[0] > 0 else "")
            if progress_tracker is None:
                print(f"  [B&B] obj={obj:.4f}  bnd={bnd:.4f}  "
                      f"gap={gap:.1f}%  nodes={nodes_cnt}{lazy_info}  "
                      f"t={runtime:.0f}s")
            if progress_tracker is not None:
                progress_tracker.record(runtime, obj, bnd, nodes_cnt)

        elif where == GRB.Callback.MIPSOL:
            _lazy_mtz_enforce(model, where)

    # ==========================================================================
    # 6d. Single-pass solve with lazy MTZ callback
    # ==========================================================================
    m.setParam('TimeLimit', _total_time_budget)
    if progress_tracker is None:
        print(f"  [Solve] 单一B&B树 + MIPSOL延迟约束回调, "
              f"时限{_total_time_budget:.0f}s "
              f"(NoRel=60s Improve=60s P={len(tau_list)-1})")

    m.optimize(_combined_callback)

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
    # 7. Final summary
    # ==========================================================================
    _elapsed = _time.perf_counter() - _solve_start
    if progress_tracker is None:
        try:
            _final_obj = m.ObjVal
            _final_bnd = m.ObjBound
            _final_gap = abs(_final_bnd - _final_obj) / (abs(_final_obj) + 1e-9) * 100
            print(f"  [完成] obj={_final_obj:.4f}  bnd={_final_bnd:.4f}  "
                  f"gap={_final_gap:.1f}%  "
                  f"lazy_mtz={_lazy_mtz_count[0]}  "
                  f"callback_invocations={_lazy_callback_count[0]}  "
                  f"耗时={_elapsed:.0f}s")
        except Exception:
            print(f"  [完成] 耗时={_elapsed:.0f}s  "
                  f"lazy_mtz={_lazy_mtz_count[0]}  "
                  f"callbacks={_lazy_callback_count[0]}")
    else:
        if progress_tracker is not None:
            progress_tracker.record(_elapsed, None, None, None)

    # Save references for downstream parsing
    m._x = x
    m._v = v
    m._u = u
    m._y = y
    m._w = w
    m._delta = delta
    m._f = f
    m._grids = grids
    m._nodes = nodes
    m._feasible_arcs = feasible_arcs

    return m
