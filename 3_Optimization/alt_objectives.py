# -*- coding: utf-8 -*-
"""Alternative objective-coefficient tensors for the objective-comparison experiment.

Motivation (advisor): show that the fluid-derived objective is not just *solvable* but
*right* — by holding the STGraph model fixed and swapping only the coefficient tensor
Omega[j][y][s], then judging every resulting route in an independent discrete-event
simulation.

The STGraph objective is  max  sum_e  Omega[i^e][y^e][s_1^e] * x_e
(Optimize_PLA_STGraph.py line 184), i.e. a pure lookup indexed by (departure grid,
swap quantity, departure step).  Every objective below is therefore node-local by
construction and plugs in with no change to the model, the constraints, or the solver.

Registered objectives
---------------------
  psiv      : the paper's closed-form U_j         (baseline; computed elsewhere)
  lost      : lost-demand reduction               Raviv, Tzur & Forma (2013, EJOR)
  lowrelief : relief of low/soon inventory        operational practice: "fix the worst batteries"
  target    : target-inventory deviation          Schuijbroek et al. (2017, C&OR);
                                                  Chemla et al. (2013)
  swap      : swap throughput                     operational practice
  dlswap    : demand-weighted swap volume         "go to the busiest grids, swap the most"

NOTE ON `lost`
--------------
Measured on the 2025-11-05 18:00 snapshot, `lost` is identically zero for every
(grid, y, u): at the evening peak returns exceed departures, so no grid drains and
the fluid lost demand (lam - rho)(T - t0)+ vanishes.  The objective is kept for
completeness and for off-peak snapshots, but it cannot be used as a competitor on
the peak snapshot -- a competitor that is constant cannot rank routes.
"""
import numpy as np

MODES = ("lost", "lowrelief", "target", "swap", "dlswap")


def _traj_full(u, n_soon, n_normal, theta_s, theta_n, rho_pure, lam):
    """Natural per-state inventory at time u (mirrors Grid_Utility's recursion).

    rho_pure == rho on this data, because theta_soon + theta_normal == 1 was verified
    empirically (the low state never enters the return stream).
    """
    N0 = n_soon + n_normal
    if N0 <= 1e-12:
        return 0.0, 0.0, 0.0
    if u <= 0:
        return n_soon, n_normal, N0
    tp = theta_s / (theta_s + theta_n)
    tn = theta_n / (theta_s + theta_n)
    if np.isclose(rho_pure, lam):
        e = np.exp(-lam / N0 * u)
        return (max(0.0, tp * N0 + (n_soon - tp * N0) * e),
                max(0.0, tn * N0 + (n_normal - tn * N0) * e), N0)
    lin = N0 + (rho_pure - lam) * u
    if lin <= 1e-12:
        return 0.0, 0.0, 0.0
    sc = (N0 / lin) ** (lam / (rho_pure - lam))
    return (max(0.0, (n_soon - tp * N0) * sc + tp * lin),
            max(0.0, (n_normal - tn * N0) * sc + tn * lin), lin)


def _lost(lam, rho_pure, N0, T, start):
    """Fluid lost demand on [start, T]: integral of (lam - rho_pure) while drained."""
    if lam <= rho_pure or N0 <= 1e-12:
        return 0.0
    t_stockout = start + N0 / (lam - rho_pure)
    return (lam - rho_pure) * max(0.0, T - t_stockout)


def value(mode, u, y, n_low, n_soon, n_normal, theta_s, theta_n, rho, lam, T,
          tau_target=None):
    """Node-local reward for one (arrival time, swap quantity) pair."""
    S = theta_s + theta_n
    rho_pure = rho * S
    N0 = n_soon + n_normal
    if N0 <= 1e-12 and mode != "swap":
        return 0.0

    n_s_u, n_n_u, _ = _traj_full(u, n_soon, n_normal, theta_s, theta_n, rho_pure, lam)
    sw_low = min(y, n_low)                      # static low pool (model convention)
    sw_soon = min(max(0, y - n_low), n_s_u)     # capped at available soon stock
    delta_in = sw_low + sw_soon

    if mode == "swap":
        return float(sw_low + sw_soon)

    if mode == "dlswap":
        return float(lam * (sw_low + sw_soon))

    if mode == "lost":
        # reduction in fluid lost demand attributable to this intervention
        before = _lost(lam, rho_pure, N0, T, 0.0)
        tN0 = n_s_u + n_n_u + sw_low           # post-swap serviceable stock
        after = _lost(lam, rho_pure, tN0, T, u)
        return max(0.0, before - after)

    if mode == "lowrelief":
        # poor-quality stock (low + soon) left at the end of the horizon,
        # natural vs. post-swap.  The post-swap trajectory is the same closed form
        # with the origin shifted to u (as in the paper's Section 4).
        n_s_T_nat, _, _ = _traj_full(T, n_soon, n_normal,
                                     theta_s, theta_n, rho_pure, lam)
        n_s_T_post, _, _ = _traj_full(T - u, n_s_u - sw_soon,
                                      n_n_u + sw_low + sw_soon,
                                      theta_s, theta_n, rho_pure, lam)
        nat_end = n_low + n_s_T_nat
        post_end = (n_low - sw_low) + n_s_T_post
        return max(0.0, nat_end - post_end)

    if mode == "target":
        # Minimise |final serviceable inventory - target|.
        # The target is the demand-implied cover, tau_j = lambda_j * T: the stock needed
        # to serve one horizon of rentals.  (Using tau_j = N_j0 instead makes the
        # objective vacuous, because any swap then overshoots the target and every
        # coefficient comes out negative.)
        tau = (lam * T) if tau_target is None else tau_target
        nat_T = N0 + (rho_pure - lam) * T
        post_T = (n_s_u + n_n_u + sw_low) + (rho_pure - lam) * (T - u)
        return abs(nat_T - tau) - abs(post_T - tau)

    raise ValueError("unknown mode %r" % mode)


def generate_alt_utility_matrix(grids, C_max, T_total, P_intervals, grid_params,
                                mode, y_levels=None, tau_targets=None):
    """Build Omega[j][y][s] for the requested objective, same shape as the PSIV tensor."""
    if mode not in MODES:
        raise ValueError("mode must be one of %s" % (MODES,))
    y_levels = list(y_levels) if y_levels else list(range(1, C_max + 1))
    tau = [T_total * s / P_intervals for s in range(P_intervals + 1)]
    theta_s = grid_params[grids[0]]["theta_soon"]
    theta_n = grid_params[grids[0]]["theta_normal"]

    Omega = {}
    for j in grids:
        p = grid_params[j]
        tau_j = None if tau_targets is None else tau_targets.get(j)
        Omega[j] = {}
        for y in y_levels:
            Omega[j][y] = {}
            for s, u in enumerate(tau):
                Omega[j][y][s] = value(
                    mode, u, y, p["n_low"], p["n_soon"], p["n_normal"],
                    theta_s, theta_n, p["rho"], p["lam"], T_total, tau_j)
    return Omega, tau


def scale_to_psiv(Alt, Psi, grids):
    """Scale an alternative tensor so its total magnitude matches the PSIV tensor.

    Objectives carry different units; without normalisation the Big-M bounds in the
    model would be calibrated for one and not the others.  A common scale factor is
    applied per instance, so the comparison is about *which nodes get visited*, not
    about which objective happens to produce larger numbers.
    """
    a = max((Alt[j][y][s] for j in grids for y in Alt[j] for s in Alt[j][y]), default=0.0)
    b = max((Psi[j][y][s] for j in grids for y in Psi[j] for s in Psi[j][y]), default=0.0)
    k = (b / a) if a > 0 else 1.0
    return {j: {y: {s: v * k for s, v in Alt[j][y].items()} for y in Alt[j]} for j in grids}, k
