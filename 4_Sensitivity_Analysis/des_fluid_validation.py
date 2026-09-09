#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Fluid-Approximation DES Validation — Birth-Death Process vs. ODE Trajectory
=============================================================================

Purpose: Quantify the systematic error of the fluid (mean-field) approximation
in the "low-inventory + net-outflow" regime.

  The fluid ODE replaces the stochastic Poisson jumps by their expectation,
  predicting a deterministic, nearly-flat inventory trajectory
  N(t) = N0 + (rho - lam)*t that hits zero at the MEAN stockout time
  t0 = N0/(lam-rho). The real birth-death process is a discrete random walk:
  even when t0 is LONG (>> the 1h planning horizon), low-inventory grids carry
  a significant probability of stocking out EARLY (within 1h) due to variance.
  This early-stockout tail is exactly what the ODE ignores.

Key findings to check (per grid):
  - Trajectory panel: the ODE is a flat deterministic line; the DES shows
    discrete "staircase" sample paths with spread (variance) around it.
  - Histogram panel: distribution of stockout times within the 1h horizon,
    with P(T0<1h) quantifying the early-stockout tail risk.

Grid selection rule:
  1) 0 < N0 = n_soon + n_normal <= N0_MAX(=5)   -- low inventory (variance matters)
  2) lam > rho_eff                                -- net outflow (eventual stockout)
  Stratified by N0 level: pick one grid per integer level, evenly spread across
  1..N0_MAX, to show the early-stockout tail risk decaying as N0 grows (O(1/N)).
  Within each level, pick the grid with the smallest t0 (largest net outflow).
  (Deliberately NO t0<T requirement — early stockout despite t0>>T is the point.)

Usage:
  python 4_Sensitivity_Analysis/des_fluid_validation.py
  python 4_Sensitivity_Analysis/des_fluid_validation.py --datetime "2025/11/05 18:00"
  python 4_Sensitivity_Analysis/des_fluid_validation.py --n0-max 5 --top-k 3 --reps 2000
=============================================================================
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OPTIM_DIR = os.path.join(PROJECT_ROOT, '3_Optimization')
if OPTIM_DIR not in sys.path:
    sys.path.insert(0, OPTIM_DIR)

from Pre_Process import (
    DEFAULT_TARGET_DATETIME,
    DEFAULT_PREDICTION_FILE,
    select_random_datetime,
    list_available_hours,
    prepare_optimize_inputs,
)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results", "des_validation")


def simulate_birth_death(N0, rho_eff, lam, T, rng):
    """One Gillespie birth-death replication over horizon T.

    State N(t) = serviceable inventory (soon + normal). Events:
      - arrival  (rate rho_eff): N += 1
      - departure (rate lam, only if N > 0): N -= 1
    Returns (t_events, N_events, t_stockout): t_stockout is the first time N
    hits 0 (or None if it never stocks out within T).
    """
    t = 0.0
    N = N0
    t_events = [0.0]
    N_events = [float(N)]
    t_stockout = None

    while t < T:
        rate_arr = rho_eff
        rate_dep = lam if N > 0 else 0.0
        total_rate = rate_arr + rate_dep
        if total_rate <= 1e-12:
            break
        dt = rng.exponential(1.0 / total_rate)
        t += dt
        if t > T:
            break
        if rng.random() < rate_arr / total_rate:
            N += 1
        else:
            N -= 1
        t_events.append(t)
        N_events.append(float(N))
        if N == 0 and t_stockout is None:
            t_stockout = t

    if t_events[-1] < T:
        t_events.append(T)
        N_events.append(float(N))
    return np.asarray(t_events), np.asarray(N_events), t_stockout


def select_candidate_grids(grid_params, n0_max=5.0, top_k=3):
    """Select low-inventory + net-outflow grids, stratified by N0 level.

    We deliberately do NOT require t0 < T: the point is that grids with a LONG
    mean stockout time t0 = N0/(lam-rho) >> T can still stock out EARLY within
    the horizon due to variance — the tail risk the ODE ignores.

    Picks ONE representative grid per integer N0 level (1..n0_max), evenly
    spread across the range, so the figure shows how the early-stockout tail
    risk decays as N0 grows (the O(1/N) error scaling). Within each level we
    pick the grid with the smallest t0 (largest net outflow lambda-rho), i.e.
    the most stockout-prone grid at that inventory level.
    """
    by_level = {}
    for j, p in grid_params.items():
        n0 = float(p["n_soon"]) + float(p["n_normal"])
        lam = float(p["lam"])
        rho_eff = float(p["rho"]) * (float(p["theta_soon"]) + float(p["theta_normal"]))
        if not (0.0 < n0 <= n0_max):
            continue
        if lam <= rho_eff:
            continue
        t0 = n0 / (lam - rho_eff)
        lvl = max(1, int(round(n0)))
        cand = {"grid": j, "n0": n0, "lam": lam, "rho_eff": rho_eff, "t0": t0}
        if lvl not in by_level or t0 < by_level[lvl]["t0"]:
            by_level[lvl] = cand

    levels = sorted(by_level.keys())
    if len(levels) <= top_k:
        chosen = levels
    else:
        idxs = np.unique(np.linspace(0, len(levels) - 1, top_k).round().astype(int))
        chosen = [levels[i] for i in idxs]

    return [by_level[lvl] for lvl in chosen]


def main():
    parser = argparse.ArgumentParser(description="DES fluid-approximation validation")
    parser.add_argument("--data", type=str, default=DEFAULT_PREDICTION_FILE)
    parser.add_argument("--datetime", type=str, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n0-max", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=3,
                        help="number of N0 strata to sample (evenly spread across 1..n0-max)")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--list-hours", action="store_true")
    args = parser.parse_args()

    if args.list_hours:
        for h in list_available_hours(file_path=args.data):
            print(h.strftime("%Y/%m/%d %H:%M"))
        return

    if args.random:
        target_datetime = select_random_datetime(file_path=args.data, seed=args.seed)
    else:
        target_datetime = args.datetime or DEFAULT_TARGET_DATETIME

    T = 1.0  # planning horizon (hours)
    rng = np.random.default_rng(args.seed)

    grids, grid_params, _ = prepare_optimize_inputs(args.data, target_datetime=target_datetime)
    cands = select_candidate_grids(grid_params, n0_max=args.n0_max, top_k=args.top_k)

    if not cands:
        print(f"[WARNING] No grid satisfies the rule at {target_datetime} "
              f"(0<N0<={args.n0_max} and lam>rho_eff).")
        print("          Hint: raise --n0-max (e.g. 8), or change --datetime.")
        return

    print("=" * 72)
    print(f"  DES fluid-approximation validation — {target_datetime}")
    print(f"  Selected grids (stratified by N0 level):")
    for c in cands:
        print(f"    grid={c['grid']}  N0={c['n0']:.1f}  lam={c['lam']:.3f}  "
              f"rho_eff={c['rho_eff']:.3f}  t0(ODE mean)={c['t0']:.1f}h")
    print("=" * 72)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    n = len(cands)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 8.0), squeeze=False)
    summary_rows = []

    for ax_idx, c in enumerate(cands):
        ax_traj = axes[0, ax_idx]
        ax_hist = axes[1, ax_idx]
        n0 = c["n0"]
        lam = c["lam"]
        rho_eff = c["rho_eff"]
        t0_ode = c["t0"]

        # ---- ODE smooth curve over [0, T] (nearly flat when net drift is small)
        t_ode = np.linspace(0.0, T, 200)
        n_ode = n0 + (rho_eff - lam) * t_ode

        # ---- Monte Carlo simulation over the 1h horizon
        t_grid = np.linspace(0.0, T, 500)
        stockout_times = []
        N_samples = []
        sample_paths = []
        for r in range(args.reps):
            te, ne, ts = simulate_birth_death(n0, rho_eff, lam, T, rng)
            stockout_times.append(ts)
            N_samples.append(np.interp(t_grid, te, ne))
            if r < 8:
                sample_paths.append((te, ne))

        N_samples = np.array(N_samples)
        N_mean = N_samples.mean(axis=0)
        N_low = np.percentile(N_samples, 5, axis=0)
        N_high = np.percentile(N_samples, 95, axis=0)

        # ---- Trajectory panel
        for te, ne in sample_paths:
            ax_traj.step(te, ne, where='post', alpha=0.22, lw=0.8, color='tab:blue')
        ax_traj.plot(t_ode, n_ode, color='tab:red', lw=2.0, label='ODE (fluid approx.)')
        ax_traj.plot(t_grid, N_mean, color='tab:green', lw=2.0, label='DES empirical mean')
        ax_traj.fill_between(t_grid, N_low, N_high, alpha=0.15, color='tab:green',
                             label='5%–95% band')
        ax_traj.set_xlabel('Time t (h)')
        ax_traj.set_ylabel('Serviceable inventory N(t)')
        ax_traj.set_title(f"grid {c['grid']}\nN0={n0:.1f}, lam={lam:.3f}, rho={rho_eff:.3f}",
                          fontsize=10)
        ax_traj.legend(fontsize=7)
        ax_traj.set_xlim(0, T)
        ax_traj.set_ylim(bottom=0)

        # ---- Statistics
        finite = np.array([ts for ts in stockout_times if ts is not None], dtype=float)
        n_censored = sum(1 for ts in stockout_times if ts is None)
        p_stockout_1h = 1.0 - n_censored / args.reps
        mean_early = float(np.mean(finite)) if finite.size else float('nan')
        median_early = float(np.median(finite)) if finite.size else float('nan')

        # ---- Stockout histogram panel (early stockouts within the 1h horizon)
        if finite.size:
            ax_hist.hist(finite, bins=30, range=(0.0, T), color='tab:blue',
                         alpha=0.6, edgecolor='white', linewidth=0.4)
        # mark the ODE mean t0 if on-scale, else annotate it
        if t0_ode <= T:
            ax_hist.axvline(t0_ode, color='tab:red', ls='--', lw=1.5,
                            label=f't0 (ODE mean) = {t0_ode:.2f}h')
        else:
            ax_hist.annotate(f't0 (ODE mean) = {t0_ode:.1f}h  (off-scale)',
                             xy=(0.98, 0.95), xycoords='axes fraction',
                             ha='right', va='top', fontsize=8, color='tab:red')
        ax_hist.axvspan(0.0, T, color='tab:red', alpha=0.04)
        ax_hist.set_xlabel('Stockout time T0 within horizon (h)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title(f"Early stockout within 1h (N0={n0:.1f})\n"
                          f"P(T0<1h)={p_stockout_1h:.2f}, median={median_early:.3f}h",
                          fontsize=10)
        if t0_ode <= T:
            ax_hist.legend(fontsize=7)
        ax_hist.set_xlim(0, T)

        summary_rows.append({
            "grid": c["grid"], "n0": n0, "lam": lam, "rho_eff": rho_eff,
            "t0_ode_h": round(t0_ode, 2),
            "p_stockout_1h": round(p_stockout_1h, 4),
            "mean_early_h": round(mean_early, 4),
            "median_early_h": round(median_early, 4),
        })

    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "des_vs_ode.png")
    fig.savefig(png_path, dpi=150)
    print(f"\n  Figure saved: {png_path}")

    # ---- Summary table
    print("\n  Summary statistics:")
    header = (f"{'grid':>18s} {'N0':>5s} {'lam':>7s} {'rho':>7s} {'t0(ODE)':>8s} "
              f"{'P(T0<1h)':>9s} {'medEarly':>9s} {'meanEarly':>9s}")
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(f"{r['grid']:>18s} {r['n0']:>5.1f} {r['lam']:>7.3f} {r['rho_eff']:>7.3f} "
              f"{r['t0_ode_h']:>8.2f} {r['p_stockout_1h']:>9.3f} "
              f"{r['median_early_h']:>9.3f} {r['mean_early_h']:>9.3f}")
    print("\n  Reading: 'P(T0<1h)' is the early-stockout tail risk that the deterministic")
    print("  fluid ODE (which uses the fixed mean t0) ignores. A large P(T0<1h) despite a")
    print("  long t0(ODE) means the grid can stock out well before its predicted mean time.")


if __name__ == "__main__":
    main()
