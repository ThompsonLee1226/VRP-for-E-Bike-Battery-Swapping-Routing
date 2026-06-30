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
    ** augmented with MCF flow constraints, MTZ fully removed **.

    Replaces the ~135K pairwise MTZ big-M constraints (O(N²)) with:
      - MCF flow conservation → valid tour structure + battery accounting
      - TotalRouteTime → single global time budget constraint
      - CycleDeadline → depot return feasibility (O(N))
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
    # Solver parameters
    # ==========================================================================
    m.setParam('MIPGap', 0.05)
    m.setParam('TimeLimit', 1800)
    m.setParam('Method', 0)              # Primal Simplex
    m.setParam('Crossover', -1)          # auto
    m.setParam('MIPFocus', 1)            # find feasible solutions
    m.setParam('Heuristics', 0.15)       # default
    m.setParam('Cuts', 2)                # aggressive — MCF + cuts synergistic
    m.setParam('Symmetry', 2)            # aggressive symmetry breaking
    m.setParam('NoRelHeurTime', 120)     # doubled heuristic phase
    m.setParam('Threads', 8)             # full cores
    m.setParam('PreDual', -1)            # auto
    m.setParam('PrePasses', 5)           # extra presolve for MCF constraints

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
    # 5. Minimal Timing Constraints (MTZ fully removed)
    # ==========================================================================
    # MTZ constraints are completely removed. Instead, we rely on:
    #   (a) MCF flow conservation → valid tour structure (no subtours)
    #   (b) TotalRouteTime → global time budget
    #   (c) CycleDeadline → depot return feasibility
    # This eliminates ~135K big-M constraints, dramatically lightening the LP.
    m.addConstr(u[depot] == 0, name="StartTimeDepot")
    m.addConstr(y[depot] == 0, name="StartBatteryDepot")

    # ---- Total route time budget (single linear constraint) ----
    # Σ travel_time · x  +  Σ service_time · y  ≤  T_total
    total_travel = gp.quicksum(
        travel_time[i][j] * x[i, j] for (i, j) in arc_list)
    total_service = gp.quicksum(swap_time_c * y[j] for j in grids)
    m.addConstr(total_travel + total_service <= T_total, name="TotalRouteTime")

    # ---- Depot return deadline (1 per grid, only for arcs to depot) ----
    for i in grids:
        if depot in feasible_out[i]:
            m.addConstr(
                u[i] + (swap_time_c * y[i]) + travel_time[i][depot]
                - BigM_deadline * (1 - x[i, depot]) <= T_total,
                name=f"CycleDeadline_{i}"
            )

    if progress_tracker is None:
        print(f"  [Timing] MTZ 已移除, 替代为 TotalRouteTime (1行) + "
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
    # 6. Solve
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
