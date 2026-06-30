import gurobipy as gp
from gurobipy import GRB


def optimize_evrp_with_pla_delta(
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
    Build and solve EVRP with PLA using the Incremental (Delta) Model.

    Replaces the SOS2 constraints of the original SOS2-based PLA formulation
    with binary delta variables and monotonicity constraints. This yields a
    significantly tighter LP relaxation because the delta staircase pattern
    (delta_1 >= delta_2 >= ... >= delta_P) prevents the LP from dispersing
    weight across non-adjacent breakpoints — the root cause of the ~100% gap.

    Parameters (same as optimize_evrp_with_pla):
    - grids: active grid ID list (zero-utility nodes already filtered)
    - travel_time: full travel time matrix including depot=0, travel_time[i][j] in hours
    - C_max: vehicle battery capacity
    - T_total: planning horizon (hours)
    - Omega: precomputed utility matrix, Omega[j][y][s]
    - tau_list: PLA time breakpoints
    - swap_time_c: service time per battery swap (hours)
    - BigM: user-supplied Big-M (automatically tightened internally)
    - progress_tracker: GurobiProgressTracker instance or None
    - max_travel_time: arc pruning threshold (hours)
    """
    m = gp.Model("EVRP_PLA_Delta")

    # ==========================================================================
    # Solver parameters (Tier-3 — Delta model tuned)
    # ==========================================================================
    m.setParam('MIPGap', 0.05)
    m.setParam('TimeLimit', 1800)
    m.setParam('Method', 0)              # Primal Simplex (avoid Barrier deadlock)
    m.setParam('Crossover', -1)          # auto
    m.setParam('MIPFocus', 1)            # find feasible solutions
    m.setParam('Heuristics', 0.15)       # default — delta LP is already tight
    m.setParam('Cuts', 1)                # moderate — tighter LP needs fewer cuts
    m.setParam('Symmetry', 2)            # aggressive symmetry breaking
    m.setParam('NoRelHeurTime', 60)      # short heuristic phase
    m.setParam('Threads', 8)             # full cores for simplex speed
    m.setParam('PreDual', -1)            # auto

    # ==========================================================================
    # Adaptive Big-M (same logic as SOS2 model)
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
    Y_domain = list(range(1, C_max + 1))           # battery swap quantities: 1..C_max
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
    # Delta_Omega[j][y_val][0] = Omega[j][y_val][0]           (base utility)
    # Delta_Omega[j][y_val][k] = Omega[j][y_val][k] - Omega[j][y_val][k-1]  (increment)
    Delta_Omega = {}
    for j in grids:
        Delta_Omega[j] = {}
        for y_val in Y_domain:
            Delta_Omega[j][y_val] = {}
            Delta_Omega[j][y_val][0] = Omega[j][y_val][0]
            for k in K_domain:
                Delta_Omega[j][y_val][k] = Omega[j][y_val][k] - Omega[j][y_val][k-1]

    # Delta_tau[k_idx] = tau_list[k] - tau_list[k-1]  for k=1..P
    # (0-indexed list, Delta_tau[0] corresponds to k=1)
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

    # PLA variables — DELTA formulation (replaces w binary selection)
    # w[j, y_val] = 1 iff grid j swaps exactly y_val batteries
    w = m.addVars(grids, Y_domain, vtype=GRB.BINARY, name="w")

    # delta[j, y_val, k] ∈ {0,1}: incremental fill of interval k
    # Monotonicity  delta_1 >= delta_2 >= ... >= delta_P  ensures the
    # arrival time is a convex combination of at most two adjacent breakpoints.
    delta = m.addVars(grids, Y_domain, K_domain, vtype=GRB.BINARY, name="delta")

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
    # 1b. Variable Branching Priorities (Tier-4 fix)
    # ==========================================================================
    # Guide Gurobi to branch on visit/selection variables first.
    # Until v[j] and w[j,y] are integral, delta's staircase structure
    # cannot tighten the LP — so prioritize those decisions.
    for j in grids:
        v[j].BranchPriority = 10
        for y_val in Y_domain:
            w[j, y_val].BranchPriority = 5

    # ==========================================================================
    # 2. Objective Function (Delta formulation)
    # ==========================================================================
    # obj = Σ_j Σ_y [Omega[j][y][0] * w[j,y] + Σ_k ΔΩ[j][y][k] * delta[j,y,k]]
    # This is mathematically equivalent to the lam-based objective
    # under the delta encoding.
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
    # 3. Topological Constraints (unchanged from SOS2 model)
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
    # 4. PLA Constraints — DELTA FORMULATION
    # ==========================================================================
    # Replaces the original block:
    #   - SOS2 on lam[j, y_val, :]
    #   - CC constraint:  Σ_s lam[j,y,s] = w[j,y]
    #   - Time sync:      u[j] ≈ Σ τ_s * lam[j,y,s]   (with BigM)
    #
    # With the delta model:
    #   - delta[j,y,k] ≤ w[j,y]              (coupling)
    #   - delta[j,y,1] ≥ delta[j,y,2] ≥ ...  (monotonicity → staircase pattern)
    #   - u[j] = Σ_{y,k} Δτ_k * delta[j,y,k] (exact time equality)
    #   - No SOS2, no lam variables at all.

    for j in grids:
        # w-sum = v[j]  (exactly one y_val if grid is visited)
        m.addConstr(
            gp.quicksum(w[j, y_val] for y_val in Y_domain) == v[j],
            name=f"W_Sum_{j}"
        )

        # y[j] = Σ y_val * w[j, y_val]  (map w to integer swap count)
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

        # --- time synchronisation: u[j] = τ₀ + Σ_{y,k} Δτ_k · δ_jyk ---
        # Since τ₀ = 0, the constant term is omitted.
        delta_time = gp.quicksum(
            Delta_tau_list[k_idx] * delta[j, y_val, K_domain[k_idx]]
            for y_val in Y_domain for k_idx in range(len(K_domain))
        )
        m.addConstr(u[j] == delta_time, name=f"TimeSync_{j}")

    # ==========================================================================
    # 5. MTZ Temporal Constraints (unchanged)
    # ==========================================================================
    m.addConstr(u[depot] == 0, name="StartTimeDepot")
    m.addConstr(y[depot] == 0, name="StartBatteryDepot")

    for i in nodes:
        for j in feasible_out[i]:
            if j == depot:
                continue
            m.addConstr(
                u[j] >= u[i] + (swap_time_c * y[i]) + travel_time[i][j]
                       - BigM_mtz * (1 - x[i, j]),
                name=f"MTZ_{i}_{j}"
            )

    for i in grids:
        if depot in feasible_out[i]:
            m.addConstr(
                u[i] + (swap_time_c * y[i]) + travel_time[i][depot]
                - BigM_deadline * (1 - x[i, depot]) <= T_total,
                name=f"CycleDeadline_{i}"
            )

    # ==========================================================================
    # 5.5. Greedy Warm-Start Heuristic (adapted for delta encoding)
    # ==========================================================================
    def _build_warm_start():
        """Construct a greedy route and set variable Start attributes
        using the delta encoding instead of lam."""
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
                key=lambda p: p[1], default=(1, 0),
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
            # delta[j, y_val, k] — staircase fill based on arrival time u_j
            # -----------------------------------------------------------------
            # For the selected y_val:
            #   delta_k = 1        if tau_list[k] <= u_j   (full interval)
            #   delta_k = fraction  if tau_list[k-1] < u_j < tau_list[k]
            #   delta_k = 0        if tau_list[k-1] >= u_j
            # For all other y_val:  delta_k = 0
            # -----------------------------------------------------------------
            uj = arrival[j]
            for y_val in Y_domain:
                for k_idx, k in enumerate(K_domain):
                    var = _delta_by_str.get((j_key, str(y_val), str(k)))
                    if var is None:
                        continue
                    if y_val != yj:
                        var.Start = 0.0
                    else:
                        # Check position of u_j relative to breakpoints
                        if tau_list[k] <= uj + 1e-12:
                            # Interval k is fully before arrival → δ_k = 1
                            var.Start = 1.0
                        elif tau_list[k-1] < uj - 1e-12:
                            # Partial fill: u_j ∈ (τ_{k-1}, τ_k)
                            frac = (uj - tau_list[k-1]) / Delta_tau_list[k_idx]
                            var.Start = max(0.0, min(1.0, frac))
                        else:
                            var.Start = 0.0

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
    m._grids = grids
    m._nodes = nodes
    m._feasible_arcs = feasible_arcs

    return m
