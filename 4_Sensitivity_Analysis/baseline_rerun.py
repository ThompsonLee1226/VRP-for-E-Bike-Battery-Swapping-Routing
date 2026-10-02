# -*- coding: utf-8 -*-
"""Re-run the three Big-M baselines (M3/M4/M5) plus the Greedy+2-opt heuristic (M6)
on the same machine and in the same session as the M1/M2a re-runs, so that every row
of the formulation-comparison table comes from one consistent environment.

Rationale: M2a's model is byte-identical before and after the Geo-Fencing tolerance
fix (it keeps all 661 grids), yet its solve time fell from 413s to 287s.  That proves
a large part of the observed speed-up is environmental.  Reporting M1 at 172s next to
M3/M4/M5 at ~1213s (measured in an earlier session) would be a cross-run comparison.
"""
import os, sys, time, csv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "3_Optimization"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from main_SOS2 import run_optimization_pipeline as run_sos2
from main_delta import run_optimization_pipeline as run_delta
from main_delta_MCF import run_optimization_pipeline as run_mcf

DATA = os.path.join(ROOT, "2_Training", "Training_Results",
                    "20260727_115935", "prediction_CB_Hurdle.csv")
TS = "2025/11/05 18:00"
BATCH = os.path.join(HERE, "Results", "baseline_rerun_" + time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(BATCH, exist_ok=True)

COMMON = dict(data_file=DATA, target_datetime=TS, depot_lat=None, depot_lon=None,
              vehicle_speed_kmh=30.0, C_max=20, T_total=1.0,
              y_levels=list(range(1, 11)), swap_time_c=0.02,
              max_travel_time=0.2, verbose=True,
              instance_name="baseline_rerun", time_limit_s=1200)

CASES = [("M3", run_sos2, dict(P_intervals=5),  "SOS2-PLA + MTZ"),
         ("M4", run_delta, dict(P_intervals=5), "Delta + MTZ"),
         ("M5", run_mcf,  dict(P_intervals=10), "Delta + MCF + lazy MTZ")]

rows = []
for eid, fn, extra, desc in CASES:
    d = os.path.join(BATCH, eid); os.makedirs(d, exist_ok=True)
    print("\n" + "=" * 62, flush=True)
    print("  %s  (%s)" % (eid, desc), flush=True)
    print("=" * 62, flush=True)
    t0 = time.perf_counter()
    rec = dict(experiment=eid, desc=desc, status=None, objective=None, MIPGap=None,
               elapsed=None, best_bound=None, dual_int_ratio=None,
               num_vars=None, num_constrs=None, bb_nodes=None,
               visited=None, swaps=None, active_grids=None)
    try:
        res = fn(experiment_id=eid, output_dir=d, **COMMON, **extra)
        col, s, ps = res.get("collector"), res.get("summary", {}), res.get("pruning_stats", {})
        rec.update(status=str(res.get("status")), objective=res.get("objective"),
                   visited=s.get("num_visited"), swaps=s.get("total_swaps"),
                   active_grids=ps.get("active_grids"))
        if col is not None:
            for a, k in [("mip_gap_pct", "MIPGap"), ("num_vars", "num_vars"),
                         ("num_constrs", "num_constrs"), ("bb_nodes", "bb_nodes"),
                         ("best_bound", "best_bound")]:
                rec[k] = getattr(col, a, None)
        # dual/integer ratio, the quantity reported in Table 5
        try:
            if rec["best_bound"] and rec["objective"]:
                rec["dual_int_ratio"] = round(rec["best_bound"] / rec["objective"], 3)
        except Exception:
            pass
    except Exception as exc:
        rec["status"] = "ERROR: %s" % str(exc)[:110]
        print("  [ERROR] %s" % exc, flush=True)
    rec["elapsed"] = round(time.perf_counter() - t0, 2)
    rows.append(rec)
    print("  -> %s obj=%s gap=%s t=%.0fs vars=%s cons=%s nodes=%s"
          % (rec["status"], rec["objective"], rec["MIPGap"], rec["elapsed"],
             rec["num_vars"], rec["num_constrs"], rec["bb_nodes"]), flush=True)
    with open(os.path.join(BATCH, "baseline_rerun_summary.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

print("\nDONE ->", BATCH, flush=True)
