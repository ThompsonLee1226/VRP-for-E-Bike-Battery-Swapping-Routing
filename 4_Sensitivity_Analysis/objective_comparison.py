# -*- coding: utf-8 -*-
"""Objective-function comparison across three snapshots.

Design
------
Hold the STGraph model fixed and swap ONLY the objective coefficient tensor
(that is the whole point: `Optimize_PLA_STGraph.py:184` reads Omega[j][y][s] as a
plain lookup, so the formulation, the constraints, the pruning and the solver
settings are identical for every arm).  Then judge every resulting route with the
independent discrete-event simulator in `des_route_eval.py`, which never touches the
fluid ODE.

If the fluid-derived objective carries information the alternatives do not, its route
should win on quality-weighted service / lost rentals in the DES even though those
quantities never appear in its coefficients.
"""
import os, sys, json, csv, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "3_Optimization"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import main_STGraph as M
import Pre_Process as PP
from alt_objectives import generate_alt_utility_matrix, MODES
import des_route_eval as DES

DATA = os.path.join(ROOT, "2_Training", "Training_Results",
                    "20260727_115935", "prediction_CB_Hurdle.csv")
SNAPSHOTS = ["2025/10/25 08:00",    # Saturday morning peak  (deficit-leaning)
             "2025/11/05 18:00",    # Wednesday evening peak  (supply-surplus)
             "2025/11/04 03:00"]    # Tuesday small hours     (low demand, worst objective)
OBJECTIVES = ["psiv"] + list(MODES)
N_REP = 400
T, C_MAX, P_INT = 1.0, 20, 12

BATCH = os.path.join(HERE, "Results", "objective_comparison_" + time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(BATCH, exist_ok=True)
ORIG_GEN = M.generate_offline_utility_matrix

def patch(mode):
    if mode == "psiv":
        M.generate_offline_utility_matrix = ORIG_GEN
        return
    def gen(**kw):
        return generate_alt_utility_matrix(
            kw["grids"], kw["C_max"], kw["T_total"], kw["P_intervals"],
            kw["grid_params"], mode, y_levels=kw.get("y_levels"))
    M.generate_offline_utility_matrix = gen

rows = []
for ts in SNAPSHOTS:
    print("\n" + "#" * 70, flush=True)
    print("#  SNAPSHOT %s" % ts, flush=True)
    print("#" * 70, flush=True)

    grids, gp, _ = PP.prepare_optimize_inputs(DATA, target_datetime=ts)
    th_s = gp[grids[0]]["theta_soon"]; th_n = gp[grids[0]]["theta_normal"]

    # DES floor: no intervention at all
    base = DES.benchmark(gp, T, th_s, th_n, n_rep=N_REP)
    print("  [DES] do-nothing floor: quality=%.2f lost=%.1f end_serv=%.0f"
          % (base["served_quality"]["mean"], base["lost"]["mean"],
             base["end_serviceable"]["mean"]), flush=True)

    routes = {}
    for mode in OBJECTIVES:
        d = os.path.join(BATCH, ts.replace("/", "").replace(" ", "_").replace(":", ""), mode)
        os.makedirs(d, exist_ok=True)
        print("\n  --- %s ---" % mode, flush=True)
        patch(mode)
        t0 = time.perf_counter()
        rec = dict(snapshot=ts, objective=mode, status=None, model_obj=None,
                   MIPGap=None, elapsed=None, visited=None, swaps=None,
                   active_grids=None)
        try:
            res = M.run_optimization_pipeline(
                data_file=DATA, target_datetime=ts, depot_lat=None, depot_lon=None,
                vehicle_speed_kmh=30.0, C_max=C_MAX, T_total=T, P_intervals=P_INT,
                y_levels=list(range(1, 11)), swap_time_c=0.02, max_travel_time=0.2,
                K_neighbors=50, output_dir=d, verbose=False,
                experiment_id=mode.upper(), instance_name="objcmp",
                geo_fencing=True, knn_enabled=True, time_limit_s=1200)
            rec["status"] = str(res.get("status"))
            rec["model_obj"] = res.get("objective")
            s, ps = res.get("summary", {}), res.get("pruning_stats", {})
            rec["visited"] = s.get("num_visited")
            rec["swaps"] = s.get("total_swaps")
            rec["active_grids"] = ps.get("active_grids")
            col = res.get("collector")
            if col is not None:
                rec["MIPGap"] = getattr(col, "mip_gap_pct", None)
            route = [(r["grid"], r["arrival_time"], r["y_swapped"])
                     for r in res.get("route", [])
                     if r["grid"] != "DEPOT" and r.get("y_swapped", 0) > 0]
            routes[mode] = route
        except Exception as exc:
            rec["status"] = "ERROR: %s" % str(exc)[:100]
            routes[mode] = []
            print("    [ERROR] %s" % exc, flush=True)
        rec["elapsed"] = round(time.perf_counter() - t0, 1)
        print("    %s obj=%s visited=%s swaps=%s t=%.0fs"
              % (rec["status"], rec["model_obj"], rec["visited"], rec["swaps"],
                 rec["elapsed"]), flush=True)

        # Guard: if the MIP failed we have no route.  Evaluating the empty route
        # would silently manufacture a "do-nothing" row that looks like a result.
        if not routes[mode]:
            rec["DES_note"] = "SKIPPED: no route (MIP failed)"
            rows.append(rec)
            print("    DES: skipped (no route produced)", flush=True)
            with open(os.path.join(BATCH, "objective_comparison.csv"), "w",
                      newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
                w.writeheader(); w.writerows(rows)
            continue

        # ---- independent DES judgement ------------------------------------
        ev = DES.evaluate(gp, routes[mode], T, th_s, th_n, n_rep=N_REP,
                          seed=4242 + hash(mode) % 1000)
        for k, v in ev.items():
            rec["DES_" + k] = round(v["mean"], 3)
            rec["DES_" + k + "_ci"] = round(v["ci95"], 3)
        rec["n_swaps_route"] = len(routes[mode])
        rows.append(rec)
        print("    DES: quality=%.3f+-%.3f  lost=%.2f  end_serv=%.1f  end_low=%.1f"
              % (ev["served_quality"]["mean"], ev["served_quality"]["ci95"],
                 ev["lost"]["mean"], ev["end_serviceable"]["mean"],
                 ev["end_low"]["mean"]), flush=True)

        with open(os.path.join(BATCH, "objective_comparison.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader(); w.writerows(rows)

    rows.append(dict(snapshot=ts, objective="NONE(do-nothing)", status="baseline",
                     DES_served_quality=round(base["served_quality"]["mean"], 3),
                     DES_lost=round(base["lost"]["mean"], 3),
                     DES_end_serviceable=round(base["end_serviceable"]["mean"], 1),
                     DES_end_low=round(base["end_low"]["mean"], 1),
                     n_swaps_route=0))
    with open(os.path.join(BATCH, "objective_comparison.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)

M.generate_offline_utility_matrix = ORIG_GEN
print("\nDONE ->", BATCH, flush=True)
