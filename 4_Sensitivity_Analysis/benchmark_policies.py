# -*- coding: utf-8 -*-
"""
Benchmark dispatching policies for e-bike battery-swapping routing.
==================================================================

THE TWO LAYERS
--------------
  layer 1  priority score  score(j)  -- "which grid should we head for?"
  layer 2  route construction         -- "in what order do we visit them?"

Score-driven policies use greedy nearest-neighbour routing (the standard
implementation of a dispatching rule).  P7 uses its own route construction: the
paper's routed path is part of the algorithm, so it is not forced onto
nearest-neighbour.

THE POLICIES
------------
  P1  nearest-neighbour only      no information at all -- the "no policy" floor
  P2  max low-battery count       n_low
  P3  max demand rate             lambda_hat
  P4  max demand / inventory      lambda_hat / max(soon+normal, 1)
  P5  max estimated unmet demand  (lambda-rho)*(T-t0)_+     Raviv, Tzur & Forma (2013, EJOR)
  P6  max target-inventory gap    |N - tau|, tau = lambda*T  Schuijbroek et al. (2017, C&OR)
  P7  Clarke & Wright savings     merge by s_ij = d(0,i)+d(0,j)-d(i,j)
                                  Clarke & Wright (1964, Operations Research)
  NONE  do nothing                the absolute floor


"""
import math

__all__ = ["all_policies", "list_policies", "POLICY_INFO",
           "route_nearest_neighbour", "route_clarke_wright", "no_route"]

SPEED_KMH = 30.0
SWAP_HOURS = 0.02          # service time per battery


# ---------------------------------------------------------------------------
# travel times
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (same formula the paper's model uses)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def travel_matrix(grids, coords, speed_kmh=SPEED_KMH):
    """{i: {j: hours}} including the depot, which is keyed by the integer 0."""
    dep_lat = sum(coords[g][0] for g in grids) / len(grids)
    dep_lon = sum(coords[g][1] for g in grids) / len(grids)
    nodes = [0] + list(grids)
    tt = {i: {} for i in nodes}
    pos = {0: (dep_lat, dep_lon)}
    for g in grids:
        pos[g] = (coords[g][0], coords[g][1])
    for i in nodes:
        for j in nodes:
            if i == j:
                tt[i][j] = 0.0
            else:
                tt[i][j] = haversine_km(pos[i][0], pos[i][1],
                                        pos[j][0], pos[j][1]) / speed_kmh
    return tt, (dep_lat, dep_lon)


# ---------------------------------------------------------------------------
# layer 1: priority scores
# ---------------------------------------------------------------------------
def _n_serv(p):
    return p["n_soon"] + p["n_normal"]


def score_nearest(p, T):
    """P1 - no information."""
    return 0.0


def score_low_count(p, T):
    """P2 - fix the grids holding the most unusable bikes."""
    return float(p["n_low"])


def score_demand(p, T):
    """P3 - chase demand."""
    return float(p["lam"])


def score_demand_per_inventory(p, T):
    """P4 - go where demand is high relative to the stock on hand."""
    return float(p["lam"]) / max(_n_serv(p), 1.0)


def score_lost_demand(p, T):
    """P5 - largest estimated unmet demand  (Raviv, Tzur & Forma 2013)."""
    n0 = _n_serv(p)
    if n0 <= 0 or p["lam"] <= p["rho"]:
        return 0.0
    t0 = n0 / (p["lam"] - p["rho"])
    return (p["lam"] - p["rho"]) * max(0.0, T - t0)


def score_target_deviation(p, T):
    """P6 - distance from a target inventory (Schuijbroek et al. 2017)."""
    return abs(_n_serv(p) - p["lam"] * T)


SCORE_FUNCTIONS = {
    "P1": score_nearest,
    "P2": score_low_count,
    "P3": score_demand,
    "P4": score_demand_per_inventory,
    "P5": score_lost_demand,
    "P6": score_target_deviation,
}

POLICY_INFO = {
    "P1": "nearest-neighbour only (no information floor)",
    "P2": "max low-battery count",
    "P3": "max demand rate lambda",
    "P4": "max demand / serviceable inventory",
    "P5": "max estimated unmet demand (Raviv, Tzur & Forma 2013, EJOR)",
    "P6": "max target-inventory deviation (Schuijbroek et al. 2017, C&OR)",
    "P7": "Clarke & Wright savings construction (1964, Operations Research)",
    "NONE": "do nothing (absolute floor)",
}


# ---------------------------------------------------------------------------
# layer 2a: greedy nearest-neighbour over a score
# ---------------------------------------------------------------------------
def route_nearest_neighbour(grid_params, coords, T, C_max,
                            score_fn=score_nearest, tt=None,
                            swap_hours=SWAP_HOURS, max_y=None):
    """Repeatedly head for the highest-scoring grid that is still reachable.

    Feasibility at each decision point: the vehicle must be able to travel there,
    swap, and still get back to the depot inside the horizon, and must have
    capacity left.  Swap quantity is min(capacity left, max_y, n_low + n_soon).

    `max_y` is the per-grid swap depth.  It MUST be set to the same value the
    paper's model is restricted to (`y_levels` = 1..10), otherwise the benchmarks
    have a strictly larger feasible set than the method they are being compared
    against -- they could place 11+ batteries at one grid, which the STGraph-MIP
    cannot represent.  None means "only the vehicle capacity binds" (the old,
    asymmetric behaviour).
    """
    grids = list(grid_params.keys())
    if tt is None:
        tt, _ = travel_matrix(grids, coords)
    if not grids:
        return []

    cap_y = C_max if max_y is None else max_y
    scores = {j: score_fn(grid_params[j], T) for j in grids}
    route, unvisited = [], set(grids)
    cur, now, cap = 0, 0.0, C_max

    while unvisited and cap > 0:
        best, best_score = None, None
        for j in unvisited:
            stock = grid_params[j]["n_low"] + grid_params[j]["n_soon"]
            y = int(min(cap, stock, cap_y))
            # A grid with nothing swappable is NOT a candidate.  Without this the
            # vehicle drives over, spends the travel time, swaps nothing and burns
            # the horizon -- which made the benchmarks look far weaker than they
            # are (P4 reached 11 grids to place 2 batteries).  The guard uses only
            # the observed snapshot counts, so it stays architecture-free and is
            # available to every policy equally.
            if y <= 0:
                continue
            t_arr = now + tt[cur][j]
            t_done = t_arr + swap_hours * y
            if t_done + tt[j][0] > T:          # cannot finish and return
                continue
            s = scores[j]
            if best is None or s > best_score:
                best, best_score = j, s
        if best is None:
            break
        t_arr = now + tt[cur][best]
        stock = grid_params[best]["n_low"] + grid_params[best]["n_soon"]
        y = int(min(cap, stock, cap_y))
        now = t_arr + swap_hours * y
        cap -= y
        route.append((best, round(t_arr, 4), y))
        unvisited.discard(best)
        cur = best
    return route


# ---------------------------------------------------------------------------
# layer 2b: Clarke & Wright savings (its own route construction)
# ---------------------------------------------------------------------------
def route_clarke_wright(grid_params, coords, T, C_max, tt=None,
                        swap_hours=SWAP_HOURS, max_y=None):
    """Classic savings construction: start with one loop per grid, merge the pair
    with the largest saving s_ij = d(0,i) + d(0,j) - d(i,j) whenever the merged
    route still fits capacity and the horizon.  No nearest-neighbour is used.

    `max_y` as in `route_nearest_neighbour`: the per-grid swap depth, capped to the
    paper's `y_levels` so the feasible set matches the method under test.
    """
    grids = list(grid_params.keys())
    if tt is None:
        tt, _ = travel_matrix(grids, coords)
    if not grids:
        return []

    cap_y = C_max if max_y is None else max_y
    stock = {j: grid_params[j]["n_low"] + grid_params[j]["n_soon"] for j in grids}
    savings = []
    for a in grids:
        for b in grids:
            if a < b:
                savings.append((tt[0][a] + tt[0][b] - tt[a][b], a, b))
    savings.sort(reverse=True)

    routes = {j: [j] for j in grids}
    of = {j: j for j in grids}                      # which route each grid is in

    def load(rt):
        return sum(int(min(stock[g], cap_y)) for g in rt)

    for _, a, b in savings:
        ra, rb = of[a], of[b]
        if ra == rb:
            continue
        la, lb = routes[ra], routes[rb]
        # merge only at the ends of the two routes
        if a not in (la[0], la[-1]) or b not in (lb[0], lb[-1]):
            continue
        merged = (la[::-1] if la[-1] == a else la) + (lb if lb[0] == b else lb[::-1])
        if load(merged) > C_max:
            continue
        routes[ra] = merged
        for g in lb:
            of[g] = ra
        del routes[rb]

    if not routes:
        return []
    best_route = max(routes.values(), key=lambda r: sum(stock[g] for g in r))

    # execute it from the depot, respecting time and capacity
    route, now, cap = [], 0.0, C_max
    cur = 0
    for j in best_route:
        if cap <= 0:
            break
        y = int(min(cap, stock[j], cap_y))
        if y <= 0:                                  # nothing to swap here either
            continue
        t_arr = now + tt[cur][j]
        t_done = t_arr + swap_hours * y
        if t_done + tt[j][0] > T:                   # cannot finish and return
            continue
        route.append((j, round(t_arr, 4), y))
        now, cap, cur = t_done, cap - y, j
    return route


def no_route(grid_params, coords, T, C_max, **kw):
    """The do-nothing floor."""
    return []


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def all_policies(max_y=None):
    """Return {policy_name: callable(grid_params, coords, T, C_max) -> route}.

    `max_y` is forwarded to every construction as the per-grid swap depth -- pass
    the largest value in the paper's `y_levels` so the benchmarks and the
    STGraph-MIP share one feasible set.
    """
    out = {}
    for name, fn in SCORE_FUNCTIONS.items():
        out[name] = (lambda f: (lambda gp, co, T, C, **kw:
                                route_nearest_neighbour(gp, co, T, C, score_fn=f,
                                                        max_y=max_y)))(fn)
    out["P7"] = (lambda gp, co, T, C, **kw:
                 route_clarke_wright(gp, co, T, C, max_y=max_y))
    out["NONE"] = no_route
    return out


def list_policies():
    """Print what is available."""
    print("Available benchmark policies:")
    for name in sorted(POLICY_INFO):
        print("   %-5s  %s" % (name, POLICY_INFO[name]))


if __name__ == "__main__":
    list_policies()
