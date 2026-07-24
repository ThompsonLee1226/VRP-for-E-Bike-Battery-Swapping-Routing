"""
=============================================================================
Validation — 10-Point VRP for E-Bike Battery Swapping
=============================================================================
Validates the utility function Ω_j(y, τ_s) and the MIP routing model
(equations 1–8 in the paper) on a 10-grid + 1-depot instance.

Core simplification vs. the original Validation.py:
  - Pre-compute Ω tensor via calculate_operational_utility()
  - No Gurobi general constraints (no addGenConstrPow / Min / Max)
  - Clean MIP ~150 lines, directly matching the paper's formulation
=============================================================================
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from Grid_Utility import calculate_operational_utility

# =============================================================================
# 1. Problem Data — 10 grids + 1 depot
# =============================================================================
THETA_SOON   = 0.3
THETA_NORMAL = 0.7
C_MAX        = 20        # vehicle battery capacity
T            = 1.0       # planning horizon (hours)
SWAP_TIME    = 0.02      # swap time per battery (hours)
SPEED        = 30.0      # vehicle speed (km/h)
P            = 10        # time discretization points

# grid_id: (lat, lon, n_low, n_soon, n_normal, rho, lam, description)
GRID_DEFS = {
    0:  (30.83156, 103.87241,  0,  0,  0,  0.0,  0.0, "Depot"),
    1:  (30.81960, 103.89837, 10,  7,  3,  6.0,  5.0, "High: lots of low+soon"),
    2:  (30.81232, 103.88553,  9,  6,  2,  7.0,  6.0, "High: lots of low+soon"),
    3:  (30.80654, 103.91715,  8,  8,  3,  6.5,  5.5, "High: lots of low+soon"),
    4:  (30.80105, 103.88134,  4,  5,  6,  6.0,  5.0, "Medium: moderate inventory"),
    5:  (30.79629, 103.93698,  3,  4,  7,  7.0,  6.0, "Medium: moderate inventory"),
    6:  (30.79175, 103.94205,  0,  0, 14,  2.0, 15.0, "Zero: no low/soon, U=0"),
    7:  (30.78713, 103.96397,  0,  1, 16,  3.0, 18.0, "Negative: 1 soon, high lam"),
    8:  (30.78220, 103.93175,  0,  0, 12,  1.0, 12.0, "Zero: no low/soon, U=0"),
    9:  (30.77671, 103.89594,  0,  1, 18,  1.5, 20.0, "Negative: 1 soon, high lam"),
    10: (30.77012, 103.85296,  1,  0, 12,  2.5, 14.0, "Negative: 1 low, high lam"),
}

GRIDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DEPOT = 0
ALL_NODES = [DEPOT] + GRIDS


# =============================================================================
# 2. Travel-time matrix (Haversine)
# =============================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1))
         * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = {gid: (lat, lon) for gid, (lat, lon, *_) in GRID_DEFS.items()}
travel_time = {}
for i in ALL_NODES:
    travel_time[i] = {}
    for j in ALL_NODES:
        travel_time[i][j] = (0.0 if i == j
                             else haversine_km(*coords[i], *coords[j]) / SPEED)


# =============================================================================
# 3. Pre-compute utility tensor  Ω_j(y, τ_s)
# =============================================================================
tau_list = np.linspace(0, T, P + 1)          # P+1 breakpoints: τ_0…τ_P
Y_domain = list(range(1, C_MAX + 1))          # 1…C_MAX
S_domain = list(range(P + 1))

sum_theta = THETA_SOON + THETA_NORMAL
theta_s_pure = THETA_SOON / sum_theta
theta_n_pure = THETA_NORMAL / sum_theta

Omega = {}   # Omega[j][y][s] = utility when arriving at grid j, swapping y, at time τ_s
grid_params = {}
for j in GRIDS:
    _, _, n_low, n_soon, n_normal, rho, lam, desc = GRID_DEFS[j]
    rho_pure = rho * sum_theta
    grid_params[j] = {'n_low': n_low, 'n_soon': n_soon, 'n_normal': n_normal,
                      'rho': rho_pure, 'lam': lam, 'desc': desc}
    Omega[j] = {}
    for y in Y_domain:
        Omega[j][y] = {}
        for s in S_domain:
            Omega[j][y][s] = calculate_operational_utility(
                u_j=tau_list[s], y_j=y,
                n_low=n_low, n_soon=n_soon, n_normal=n_normal,
                theta_soon_global=theta_s_pure,
                theta_normal_global=theta_n_pure,
                rho_j=rho_pure, lam_j=lam, T=T)

print("=" * 70)
print("Utility Tensor Summary (non-zero entries only)")
print("=" * 70)
for j in GRIDS:
    non_zero = [(y, s, round(Omega[j][y][s], 4))
                for y in Y_domain for s in S_domain
                if Omega[j][y][s] > 1e-6]
    if non_zero:
        print(f"  Grid {j} ({grid_params[j]['desc'][:40]}): "
              f"{len(non_zero)} non-zero entries, "
              f"max={max(v for _,_,v in non_zero):.4f}")
    else:
        print(f"  Grid {j}: all zero — skipped by solver")


# =============================================================================
# 4. MIP Model  (paper eqs. 1–8)
# =============================================================================
m = gp.Model("VRP_Validation")
m.setParam('MIPGap', 0.01)
m.setParam('TimeLimit', 300)
m.setParam('OutputFlag', 0)
m.setParam('MIPFocus', 1)     # prioritize feasible solutions
m.setParam('Heuristics', 0.2)

# --- 4a. Decision variables ---
x = {}  # x[i,j] ∈ {0,1}  routing arcs
for i in ALL_NODES:
    for j in GRIDS:
        if i != j:
            x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
# return-to-depot arcs
for i in GRIDS:
    x[i, DEPOT] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_0")

v = m.addVars(GRIDS, vtype=GRB.BINARY, name="v")                      # visit indicator
y = m.addVars(GRIDS, vtype=GRB.INTEGER, lb=0, ub=C_MAX, name="y")     # swap quantity
u = m.addVars(GRIDS, vtype=GRB.CONTINUOUS, lb=0.0, ub=T, name="u")    # arrival time

# SOS2 weights  λ[j,y,s]  (only create for non-zero utility to keep model small)
lam = {}
nonzero_pairs = {}  # j → [(y,s)]
for j in GRIDS:
    nonzero_pairs[j] = [(y, s) for y in Y_domain for s in S_domain
                        if Omega[j][y][s] > 1e-6]
    for (y_val, s_idx) in nonzero_pairs[j]:
        lam[j, y_val, s_idx] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                        name=f"lam_{j}_{y_val}_{s_idx}")

# --- 4b. Objective  (paper eq. 1) ---
m.setObjective(gp.quicksum(Omega[j][y_val][s_idx] * lam[j, y_val, s_idx]
                           for j in GRIDS
                           for (y_val, s_idx) in nonzero_pairs[j]),
               GRB.MAXIMIZE)

# --- 4c. Routing constraints ---
# Flow conservation  (paper eq. 2)
for j in GRIDS:
    in_arcs = [x[i, j] for i in ALL_NODES if i != j and (i, j) in x]
    out_arcs = [x[j, k] for k in ALL_NODES if k != j and (j, k) in x]
    m.addConstr(gp.quicksum(in_arcs) == v[j], name=f"flow_in_{j}")
    m.addConstr(gp.quicksum(out_arcs) == v[j], name=f"flow_out_{j}")

# Depot departure/return  (paper eq. 3)
depot_out = [x[DEPOT, j] for j in GRIDS if (DEPOT, j) in x]
depot_in  = [x[i, DEPOT] for i in GRIDS if (i, DEPOT) in x]
m.addConstr(gp.quicksum(depot_out) <= 1, name="depot_depart")
m.addConstr(gp.quicksum(depot_in) == gp.quicksum(depot_out), name="depot_return")

# Capacity  (paper eq. 4)
m.addConstr(gp.quicksum(y[j] for j in GRIDS) <= C_MAX, name="cap_total")
for j in GRIDS:
    m.addConstr(y[j] <= C_MAX * v[j], name=f"cap_visit_{j}")

# MTZ time propagation  (paper eqs. 5–6)
M = T + max(travel_time[i][j] for i in ALL_NODES for j in ALL_NODES
            if i != j) + SWAP_TIME * C_MAX + 0.1
for j in GRIDS:
    # from depot
    if (DEPOT, j) in x:
        m.addConstr(u[j] >= travel_time[DEPOT][j]
                    - M * (1 - x[DEPOT, j]), name=f"mtz_depot_{j}")
    # grid-to-grid
    for i in GRIDS:
        if i != j and (i, j) in x:
            m.addConstr(u[j] >= u[i] + SWAP_TIME * y[i] + travel_time[i][j]
                        - M * (1 - x[i, j]), name=f"mtz_{i}_{j}")
    # deadline to depot
    if (j, DEPOT) in x:
        m.addConstr(u[j] + SWAP_TIME * y[j] + travel_time[j][DEPOT] <= T
                    + M * (1 - x[j, DEPOT]), name=f"deadline_{j}")

# Time upper bound
for j in GRIDS:
    m.addConstr(u[j] <= T * v[j], name=f"time_ub_{j}")

# --- 4d. Time-sync linking  (paper eq. 7) ---
# u_j = Σ τ_s · λ_{j,y,s}   and   y_j = Σ y · λ_{j,y,s}
# Each visited grid selects exactly one (y,s) pair
for j in GRIDS:
    pairs_j = nonzero_pairs[j]
    if not pairs_j:
        m.addConstr(v[j] == 0, name=f"force_zero_{j}")
        continue
    lam_vars_j = [lam[j, y_val, s_idx] for (y_val, s_idx) in pairs_j]
    # convex combination: sum of λ = v_j (1 if visited, 0 otherwise)
    m.addConstr(gp.quicksum(lam_vars_j) == v[j], name=f"select_{j}")
    # arrival time sync
    m.addConstr(u[j] == gp.quicksum(
        tau_list[s_idx] * lam[j, y_val, s_idx]
        for (y_val, s_idx) in pairs_j), name=f"time_sync_{j}")
    # swap quantity sync
    m.addConstr(y[j] == gp.quicksum(
        y_val * lam[j, y_val, s_idx]
        for (y_val, s_idx) in pairs_j), name=f"swap_sync_{j}")

    # SOS2 in time dimension: for each (j,y), λ[j,y,:] are at most 2 adjacent
    y_used = sorted(set(y_val for (y_val, _) in pairs_j))
    for y_val in y_used:
        s_indices = sorted(s_idx for (yy, s_idx) in pairs_j if yy == y_val)
        if len(s_indices) >= 2:
            sos_vars = [lam[j, y_val, s_idx] for s_idx in s_indices]
            m.addSOS(GRB.SOS_TYPE2, sos_vars)  # weights sum to 1, at most 2 adjacent


# =============================================================================
# 5. Solve
# =============================================================================
m.update()  # flush lazy variable/constraint creation before querying counts
n_vars = len(m.getVars())
n_constrs = len(m.getConstrs())
n_bin = sum(1 for v in m.getVars() if v.VType == GRB.BINARY)
n_int = sum(1 for v in m.getVars() if v.VType == GRB.INTEGER)
print(f"\nModel stats: {n_vars} vars ({n_bin} binary, {n_int} integer), "
      f"{n_constrs} constraints")
m.optimize()

if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
    print(f"\nStatus: {'OPTIMAL' if m.Status == GRB.OPTIMAL else 'SUBOPTIMAL'}")
    print(f"Objective (total utility): {m.ObjVal:.6f}")
    print(f"Gap: {m.MIPGap * 100:.2f}%")
    print(f"Runtime: {m.Runtime:.2f}s")
    status_ok = True
elif m.Status == GRB.TIME_LIMIT and m.SolCount > 0:
    print(f"\nStatus: TIME_LIMIT (feasible solution found)")
    print(f"Objective (total utility): {m.ObjVal:.6f}")
    print(f"Gap: {m.MIPGap * 100:.2f}%")
    print(f"Runtime: {m.Runtime:.2f}s")
    status_ok = True
elif m.Status == GRB.INFEASIBLE:
    print("\nModel is INFEASIBLE — check constraints.")
    status_ok = False
else:
    print(f"\nSolver status: {m.Status} (SolCount={m.SolCount})")
    status_ok = m.SolCount > 0

if status_ok:

    # --- Route reconstruction ---
    route = []
    current = DEPOT
    visited = set()
    while True:
        nxt = None
        for j in ALL_NODES:
            if j != current and (current, j) in x and x[current, j].X > 0.5:
                nxt = j
                break
        if nxt is None or nxt == DEPOT or nxt in visited:
            break
        route.append(nxt)
        visited.add(nxt)
        current = nxt

    print(f"\nOptimal route: Depot → {' → '.join(str(g) for g in route)} → Depot")
    print(f"\n{'Grid':>5} {'Visit':>6} {'y_j':>5} {'u_j':>8}  "
          f"{'Utility':>10}  Description")
    print("-" * 70)

    total_y = 0
    for j in GRIDS:
        vis = v[j].X > 0.5
        y_val = int(round(y[j].X))
        u_val = u[j].X
        # Compute utility from selected λ
        util = 0.0
        for (y_sel, s_sel) in nonzero_pairs[j]:
            w = lam[j, y_sel, s_sel].X
            if w > 0.01:
                util += Omega[j][y_sel][s_sel] * w
        if vis:
            total_y += y_val
        marker = " ←" if vis else ""
        print(f"{j:>5} {str(vis):>6} {y_val:>5} {u_val:>8.4f}  "
              f"{util:>10.6f}  {grid_params[j]['desc'][:35]}{marker}")

    print(f"\nTotal batteries swapped: {total_y} / {C_MAX}")
    print(f"Total travel time: {sum(travel_time[route[i]][route[i+1]]
          for i in range(len(route)-1)):.4f}h "
          f"(+ depot out/back: {travel_time[0][route[0]]:.4f} / "
          f"{travel_time[route[-1]][0]:.4f}h)")

    # =========================================================================
    # 6. Manual verification — recompute utilities for the optimal route
    # =========================================================================
    print("\n" + "=" * 70)
    print("Manual Verification (hand-computed via calculate_operational_utility)")
    print("=" * 70)
    manual_sum = 0.0
    for j in route:
        y_val = int(round(y[j].X))
        u_val = u[j].X
        util = calculate_operational_utility(
            u_j=u_val, y_j=y_val,
            n_low=grid_params[j]['n_low'],
            n_soon=grid_params[j]['n_soon'],
            n_normal=grid_params[j]['n_normal'],
            theta_soon_global=theta_s_pure,
            theta_normal_global=theta_n_pure,
            rho_j=grid_params[j]['rho'],
            lam_j=grid_params[j]['lam'], T=T)
        manual_sum += util
        print(f"  Grid {j}: u={u_val:.4f}, y={y_val},  "
              f"U_j = {util:.6f}  ({grid_params[j]['desc'][:35]})")
    print(f"\n  MIP objective:        {m.ObjVal:.6f}")
    print(f"  Manual recomputation:  {manual_sum:.6f}")
    print(f"  Match: {'✓ PASS' if abs(m.ObjVal - manual_sum) < 1e-4 else '✗ MISMATCH'}")

elif m.Status == GRB.INFEASIBLE:
    print("\nModel is INFEASIBLE — check constraints.")
elif m.Status == GRB.TIME_LIMIT:
    print(f"\nTime limit reached. Best obj: {m.ObjVal:.6f}, gap: {m.MIPGap*100:.2f}%")
else:
    print(f"\nSolver status: {m.Status}")
