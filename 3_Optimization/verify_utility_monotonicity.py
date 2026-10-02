# -*- coding: utf-8 -*-
"""Proposition Z item 4: is U_j(.,y) non-increasing in the arrival time u?"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Pre_Process as PP
import Grid_Utility as GU

DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
        "..", "2_Training", "Training_Results", "20260727_115935",
        "prediction_CB_Hurdle.csv"))
TARGET, C_max, T = "2025-11-05 18:00:00", 20, 1.0
grids, gp, _ = PP.prepare_optimize_inputs(DATA, target_datetime=TARGET)
th_s = gp[grids[0]]["theta_soon"]; th_n = gp[grids[0]]["theta_normal"]

def U(u, y, n_low, n_soon, n_normal, rho, lam):
    return GU.calculate_operational_utility(
        u, y, n_low, n_soon, n_normal, th_s, th_n, rho, lam, T)

US = np.linspace(0.0, T, 101)
YS = [1, 3, 5, 10, 15, 20]

print("=== A. real grids (%d), u grid of %d points, y in %s ===" % (len(grids), len(US), YS))
viol = 0; checked = 0; worst = None
for j in grids:
    p = gp[j]
    for y in YS:
        v = np.array([U(u, y, p["n_low"], p["n_soon"], p["n_normal"], p["rho"], p["lam"]) for u in US])
        d = np.diff(v); checked += 1
        if (d > 1e-9).any():
            viol += 1
            i = int(np.argmax(d))
            if worst is None or d.max() > worst[0]:
                worst = (float(d.max()), j, y, float(US[i]), float(US[i+1]), float(v[i]), float(v[i+1]))
print("  (grid, y) pairs checked : %d" % checked)
print("  violations              : %d" % viol)
if worst:
    print("  worst: +%.3e  h3=%s y=%d  u %.3f->%.3f  U %.4f->%.4f" % worst)

print()
print("=== B. synthetic, both branches (600 cases each, %d u-points) ===" % len(US))
rng = np.random.default_rng(7)
tot = 0
for tag, tie in [("rho != lam", False), ("rho == lam", True)]:
    bad = n = 0
    for _ in range(600):
        if tie:
            rho = rng.uniform(0.3, 25); lam = rho
        else:
            rho = rng.uniform(0.3, 25); lam = rng.uniform(0.3, 25)
            if abs(rho - lam) < 0.05: lam = rho + 0.37
        n_low = int(rng.integers(0, 6)); n_soon = int(rng.integers(0, 10)); n_normal = int(rng.integers(0, 50))
        if n_soon + n_normal == 0: continue
        y = int(rng.integers(1, 21)); n += 1
        v = np.array([U(u, y, n_low, n_soon, n_normal, rho, lam) for u in US])
        d = np.diff(v)
        if (d > 1e-9).any():
            bad += 1
            if bad <= 3:
                i = int(np.argmax(d))
                print("    VIOL rho=%.3f lam=%.3f n=(%d,%d,%d) y=%d  +%.2e at u=%.3f->%.3f"
                      % (rho, lam, n_low, n_soon, n_normal, y, d.max(), US[i], US[i+1]))
    print("  %-11s cases=%d violations=%d" % (tag, n, bad))
    tot += bad

print()
print("=== C. U <= Delta ===")
exceed = 0
for j in grids[:100]:
    p = gp[j]; N0 = p["n_soon"] + p["n_normal"]
    if N0 <= 1e-12: continue
    tp = th_s / (th_s + th_n)
    for u in np.linspace(0, T, 31):
        if np.isclose(p["rho"], p["lam"]):
            ns = tp * N0 + (p["n_soon"] - tp * N0) * np.exp(-p["lam"] / N0 * u)
        else:
            lin = N0 + (p["rho"] - p["lam"]) * u
            if lin <= 1e-12: continue
            ns = (p["n_soon"] - tp * N0) * (N0 / lin) ** (p["lam"] / (p["rho"] - p["lam"])) + tp * lin
        ns = max(0.0, ns)
        for y in YS:
            delta = min(y, p["n_low"]) + 0.2 * min(max(0, y - p["n_low"]), ns)
            if U(u, y, p["n_low"], p["n_soon"], p["n_normal"], p["rho"], p["lam"]) > delta + 1e-9:
                exceed += 1
print("  violations of U <= Delta : %d" % exceed)

print()
print("VERDICT: monotonicity %s" % ("HOLDS on all tested cases" if viol == 0 and tot == 0 else "FAILS"))
