# -*- coding: utf-8 -*-
"""Re-run the 20-snapshot cross-time-point study AFTER the Geo-Fencing tolerance fix.

The 20 timestamps are hard-coded to exactly match the earlier batch
(Results/temporal_batch_20260728_221205), so the two runs are directly comparable.
All solver parameters are identical to temporal_sensitivity.py.
"""
import os, sys, time, csv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "3_Optimization"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from main_STGraph import run_optimization_pipeline

DATA_FILE = os.path.join(ROOT, "2_Training", "Training_Results",
                         "20260727_115935", "prediction_CB_Hurdle.csv")

SPEED, C_MAX, T_TOTAL, P_INT = 30.0, 20, 1.0, 12
SWAP_C, MAX_TRAVEL, K_NB = 0.02, 0.2, 50
Y_LEVELS = list(range(1, 11))
TIME_LIMIT = 1200

TIMESTAMPS = [
    "2025/10/24 00:00", "2025/10/24 03:00", "2025/10/24 04:00",
    "2025/10/25 08:00", "2025/10/25 11:00", "2025/10/25 16:00", "2025/10/25 21:00",
    "2025/10/26 11:00",
    "2025/10/28 03:00", "2025/10/28 06:00", "2025/10/28 11:00", "2025/10/28 17:00",
    "2025/10/29 08:00",
    "2025/11/01 12:00",
    "2025/11/04 03:00",
    "2025/11/05 02:00",
    "2025/11/06 03:00", "2025/11/06 22:00",
    "2025/11/08 05:00", "2025/11/08 07:00",
]

BATCH = os.path.join(HERE, "Results", "temporal_rerun_" + time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(BATCH, exist_ok=True)
print("batch dir:", BATCH, flush=True)
print("data file:", DATA_FILE, flush=True)

FIELDS = ["datetime", "status", "elapsed_seconds", "MIPGap", "objective",
          "total_swaps", "num_visited", "makespan_hrs",
          "utility_soon", "utility_normal", "utility_low",
          "original_grids", "active_grids", "removed_grids", "feasible_arcs",
          "num_vars", "num_constrs", "bb_nodes"]

rows = []
for i, ts in enumerate(TIMESTAMPS, 1):
    ts_dir = os.path.join(BATCH, ts.replace("/", "").replace(" ", "_").replace(":", ""))
    os.makedirs(ts_dir, exist_ok=True)
    print("\n" + "=" * 66, flush=True)
    print("  [%d/%d] %s" % (i, len(TIMESTAMPS), ts), flush=True)
    print("=" * 66, flush=True)

    t0 = time.perf_counter()
    rec = {k: None for k in FIELDS}
    rec["datetime"] = ts
    try:
        res = run_optimization_pipeline(
            data_file=DATA_FILE, target_datetime=ts,
            depot_lat=None, depot_lon=None,
            vehicle_speed_kmh=SPEED, C_max=C_MAX, T_total=T_TOTAL,
            P_intervals=P_INT, y_levels=Y_LEVELS, swap_time_c=SWAP_C,
            max_travel_time=MAX_TRAVEL, K_neighbors=K_NB,
            output_dir=ts_dir, verbose=True,
            experiment_id="M1", instance_name="temporal_rerun",
            geo_fencing=True, knn_enabled=True, time_limit_s=TIME_LIMIT,
        )
        col = res.get("collector")
        s = res.get("summary", {})
        ps = res.get("pruning_stats", {})
        rec["status"] = str(res.get("status", "UNKNOWN"))
        # the objective lives at result["objective"], NOT in the summary dict
        rec["objective"] = res.get("objective")
        if rec["objective"] is None and col is not None:
            rec["objective"] = getattr(col, "objective_value", None)
        rec["total_swaps"] = s.get("total_swaps")
        rec["num_visited"] = s.get("num_visited")
        rec["makespan_hrs"] = s.get("makespan_hrs")
        rec["utility_soon"] = s.get("utility_soon")
        rec["utility_normal"] = s.get("utility_normal")
        rec["utility_low"] = s.get("utility_low")
        rec["original_grids"] = ps.get("original_grids")
        rec["active_grids"] = ps.get("active_grids")
        rec["removed_grids"] = ps.get("removed_zero_utility")
        rec["feasible_arcs"] = ps.get("feasible_arcs")
        if col is not None:
            for attr, key in [("mip_gap_pct", "MIPGap"), ("num_vars", "num_vars"),
                              ("num_constrs", "num_constrs"), ("bb_nodes", "bb_nodes")]:
                try:
                    rec[key] = getattr(col, attr)
                except Exception:
                    pass
        if rec["MIPGap"] in (None, float("inf")) and "OPTIMAL" in rec["status"]:
            rec["MIPGap"] = 0.0
    except Exception as exc:
        rec["status"] = "ERROR: %s" % str(exc)[:120]
        print("  [ERROR] %s" % exc, flush=True)

    rec["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    rows.append(rec)
    print("  -> status=%s  obj=%s  gap=%s  t=%.1fs"
          % (rec["status"], rec["objective"], rec["MIPGap"], rec["elapsed_seconds"]), flush=True)

    with open(os.path.join(BATCH, "temporal_rerun_summary.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

print("\nDONE. summary ->", os.path.join(BATCH, "temporal_rerun_summary.csv"), flush=True)
