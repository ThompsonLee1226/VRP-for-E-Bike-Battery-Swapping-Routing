# -*- coding: utf-8 -*-
"""Is U_j(.,y) unimodal (single-peaked) rather than monotone?"""
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

def U(u, y, p):
    return GU.calculate_operational_utility(
        u, y, p["n_low"], p["n_soon"], p["n_normal"], th_s, th_n, p["rho"], p["lam"], T)

US = np.linspace(0.0, T, 201)
YS = [1, 3, 5, 10, 20]
TOL = 1e-9

def shape(v):
    d = np.diff(v)
    s = np.sign(np.where(np.abs(d) < TOL, 0, d))
    s = s[s != 0]
    return len(np.where(np.diff(s) < 0)[0]) + 1 if len(s) else 1   # number of rising runs

def count_local_max(v):
    d = np.diff(v)
    s = np.sign(np.where(np.abs(d) < TOL, 0, d))
    s = s[s != 0]
    return int((np.diff(s) < 0).sum())

print("=== A. real grids, u-grid of %d ===" % len(US))
nmax = {}; argmax_u = []; rises_from_zero = 0; flat = 0; monotone_dec = 0
for j in grids:
    p = gp[j]
    for y in YS:
        v = np.array([U(u, y, p) for u in US])
        nm = count_local_max(v)
        nmax[nm] = nmax.get(nm, 0) + 1
        if nm == 0:
            if v.max() <= TOL:
                flat += 1
            else:
                monotone_dec += 1
        if v.max() > TOL:
            argmax_u.append(float(US[int(np.argmax(v))]))
            if int(np.argmax(v)) > 0 and v[0] <= TOL:
                rises_from_zero += 1
print("  distribution of #local maxima:", dict(sorted(nmax.items())))
print("  flat (U == 0 everywhere)     :", flat)
print("  purely decreasing, no hump   :", monotone_dec)
print("  unimodal (exactly 1 max)     :", nmax.get(1, 0))
print("  multi-modal (>=2 maxima)     :", sum(v for k, v in nmax.items() if k >= 2))
if argmax_u:
    a = np.array(argmax_u)
    print("  argmax u: min=%.3f  median=%.3f  mean=%.3f  max=%.3f  (T=1.0)"
          % (a.min(), np.median(a), a.mean(), a.max()))
    print("  argmax at u=0 : %d   at u=T : %d   interior : %d"
          % (int((a <= TOL).sum()), int((a >= T - TOL).sum()),
             int(((a > TOL) & (a < T - TOL)).sum())))

print()
print("=== B. synthetic (700 cases) ===")
rng = np.random.default_rng(11)
cnt = {}
for _ in range(700):
    rho = rng.uniform(0.3, 25); lam = rng.uniform(0.3, 25)
    if abs(rho - lam) < 0.05: lam = rho
    p = dict(n_low=int(rng.integers(0, 6)), n_soon=int(rng.integers(0, 10)),
             n_normal=int(rng.integers(0, 50)), rho=rho, lam=lam)
    if p["n_soon"] + p["n_normal"] == 0: continue
    y = int(rng.integers(1, 21))
    v = np.array([U(u, y, p) for u in US])
    nm = count_local_max(v)
    cnt[nm] = cnt.get(nm, 0) + 1
print("  distribution of #local maxima:", dict(sorted(cnt.items())))
print("  unimodal (exactly 1 max)     :", cnt.get(1, 0), "/", sum(cnt.values()))
