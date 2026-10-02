# -*- coding: utf-8 -*-
"""Route-following multi-grid, three-state discrete-event simulation.

This is the *independent* evaluator for the objective-comparison experiment.  It is
deliberately NOT built on the fluid ODE: every count is sampled from a Poisson process
and every rental is resolved against the realised integer inventory, so the ranking it
produces cannot be an artefact of the approximation used to construct the objective.

State per grid:  (n_low, n_soon, n_normal)
  * returns   : Poisson(rho*dt); each returned bike is `soon` with prob theta_soon,
                otherwise `normal` (the low state never enters the return stream)
  * departures: Poisson(lam*dt); a user takes a serviceable bike, choosing `normal`
                over `soon` in proportion to the standing inventory; if neither is
                available the rental is LOST
  * swaps     : at the scheduled arrival time the vehicle converts
                min(y, n_low) low batteries first, then soon batteries, to normal
  * low stock is static between swaps (same convention as the fluid model)

Reported metrics are the ones an operator cares about:
    served_quality  = served_normal + 0.8 * served_soon   (quality-weighted service)
    lost            = unmet rental requests
    end_serviceable = n_soon + n_normal at T
    end_low         = n_low at T
"""
import numpy as np


def simulate_route(grid_params, route, T, theta_soon, theta_normal, rng,
                   dt=0.005, return_trace=False):
    """One replication.

    grid_params : {j: {'n_low','n_soon','n_normal','rho','lam'}}
    route       : [(grid, arrival_time, y), ...]  -- swaps executed at arrival_time
    """
    st = {}
    for j, p in grid_params.items():
        st[j] = [float(p["n_low"]), float(p["n_soon"]), float(p["n_normal"])]

    # per-grid swap schedule
    swaps = {}
    for g, t_a, y in route:
        if g in st and y and y > 0:
            swaps.setdefault(g, []).append((float(t_a), int(y)))
    for g in swaps:
        swaps[g].sort()

    sw_ptr = {g: 0 for g in swaps}
    served_n = served_s = lost = 0
    n_steps = max(1, int(round(T / dt)))

    for k in range(n_steps):
        t0 = k * dt
        t1 = t0 + dt

        # ---- swap events falling inside this step -------------------------
        for g, evs in swaps.items():
            while sw_ptr[g] < len(evs) and evs[sw_ptr[g]][0] < t1:
                y = evs[sw_ptr[g]][1]
                s = st[g]
                sw_low = min(y, int(s[0]))
                rem = y - sw_low
                sw_soon = min(rem, int(s[1]))
                s[0] -= sw_low
                s[1] -= sw_soon
                s[2] += sw_low + sw_soon          # swapped bikes come back full
                sw_ptr[g] += 1

        # ---- stochastic arrivals / departures -----------------------------
        for g, p in grid_params.items():
            s = st[g]
            rho, lam = p["rho"], p["lam"]
            if rho > 0:
                for _ in range(rng.poisson(rho * dt)):
                    if rng.random() < theta_soon:
                        s[1] += 1.0
                    else:
                        s[2] += 1.0
            if lam > 0:
                for _ in range(rng.poisson(lam * dt)):
                    avail_s, avail_n = s[1], s[2]
                    tot = avail_s + avail_n
                    if tot <= 0:
                        lost += 1
                        continue
                    # users pick proportional to the standing mix
                    if avail_n > 0 and (avail_s <= 0 or rng.random() < avail_n / tot):
                        s[2] -= 1.0
                        served_n += 1
                    else:
                        s[1] -= 1.0
                        served_s += 1

    end_serv = sum(st[j][1] + st[j][2] for j in st)
    end_low = sum(st[j][0] for j in st)
    out = dict(served_quality=served_n + 0.8 * served_s,
               served_normal=served_n, served_soon=served_s, lost=lost,
               end_serviceable=end_serv, end_low=end_low)
    if return_trace:
        out["final_state"] = {j: tuple(st[j]) for j in st}
    return out


def evaluate(grid_params, route, T, theta_soon, theta_normal, n_rep=500, seed=12345):
    """Run n_rep replications and return mean +/- 95% CI for each metric."""
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(n_rep):
        r = simulate_route(grid_params, route, T, theta_soon, theta_normal, rng)
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    res = {}
    for k, v in acc.items():
        v = np.asarray(v, dtype=float)
        m, sd = v.mean(), v.std(ddof=1)
        res[k] = dict(mean=float(m), sd=float(sd),
                      ci95=float(1.96 * sd / np.sqrt(len(v))))
    return res


def benchmark(grid_params, T, theta_soon, theta_normal,
              n_rep=500, seed=999):
    """Do-nothing baseline: no route at all. Gives the DES comparison a floor."""
    return evaluate(grid_params, [], T, theta_soon, theta_normal, n_rep, seed)
