# -*- coding: utf-8 -*-
"""Re-run the single-instance ablation (M2a, M2b) after the Geo-Fencing tolerance fix,
so that every number in the experiment section comes from one consistent run."""
import os, sys, time, csv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "3_Optimization"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from main_STGraph import run_optimization_pipeline

DATA = os.path.join(ROOT, "2_Training", "Training_Results",
                    "20260727_115935", "prediction_CB_Hurdle.csv")
TS = "2025/11/05 18:00"
BATCH = os.path.join(HERE, "Results", "ablation_rerun_" + time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(BATCH, exist_ok=True)

CASES = [("M2a", dict(geo_fencing=False, knn_enabled=True),  "no Geo-Fence"),
         ("M2b", dict(geo_fencing=True,  knn_enabled=False), "no KNN")]

rows = []
for eid, flags, desc in CASES:
    d = os.path.join(BATCH, eid)
    os.makedirs(d, exist_ok=True)
    print("\n" + "=" * 60, flush=True)
    print("  %s  (%s)" % (eid, desc), flush=True)
    print("=" * 60, flush=True)
    t0 = time.perf_counter()
    rec = dict(experiment=eid, desc=desc, status=None, objective=None,
               MIPGap=None, elapsed=None, num_vars=None, num_constrs=None,
               bb_nodes=None, visited=None, swaps=None, makespan=None,
               active_grids=None, removed=None, feasible_arcs=None)
    try:
        res = run_optimization_pipeline(
            data_file=DATA, target_datetime=TS, depot_lat=None, depot_lon=None,
            vehicle_speed_kmh=30.0, C_max=20, T_total=1.0, P_intervals=12,
            y_levels=list(range(1, 11)), swap_time_c=0.02, max_travel_time=0.2,
            K_neighbors=50, output_dir=d, verbose=True,
            experiment_id=eid, instance_name="ablation_rerun",
            time_limit_s=1200, **flags)
        col, s, ps = res.get("collector"), res.get("summary", {}), res.get("pruning_stats", {})
        rec.update(status=str(res.get("status")), objective=res.get("objective"),
                   visited=s.get("num_visited"), swaps=s.get("total_swaps"),
                   makespan=s.get("makespan_hrs"),
                   active_grids=ps.get("active_grids"), removed=ps.get("removed_zero_utility"),
                   feasible_arcs=ps.get("feasible_arcs"))
        if col is not None:
            for a, k in [("mip_gap_pct", "MIPGap"), ("num_vars", "num_vars"),
                         ("num_constrs", "num_constrs"), ("bb_nodes", "bb_nodes")]:
                rec[k] = getattr(col, a, None)
    except Exception as exc:
        rec["status"] = "ERROR: %s" % str(exc)[:100]
        print("  [ERROR] %s" % exc, flush=True)
    rec["elapsed"] = round(time.perf_counter() - t0, 2)
    rows.append(rec)
    print("  -> %s obj=%s gap=%s t=%.0fs active=%s removed=%s arcs=%s"
          % (rec["status"], rec["objective"], rec["MIPGap"], rec["elapsed"],
             rec["active_grids"], rec["removed"], rec["feasible_arcs"]), flush=True)
    with open(os.path.join(BATCH, "ablation_rerun_summary.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

print("\nDONE ->", BATCH, flush=True)
