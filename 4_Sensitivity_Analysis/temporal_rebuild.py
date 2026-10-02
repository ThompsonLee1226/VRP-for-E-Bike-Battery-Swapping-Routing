# -*- coding: utf-8 -*-
"""Rebuild the cross-time-point summary from the per-snapshot metrics JSONs.

The live run writes its own CSV, but this reconstructs the same table purely from
the artefacts each snapshot leaves on disk, so the numbers can be re-derived at any
time (and are unaffected by a mid-run restart).
"""
import os, sys, json, glob, csv, time

HERE = os.path.dirname(os.path.abspath(__file__))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

batch = sys.argv[1] if len(sys.argv) > 1 else None
if batch is None:
    cands = sorted(glob.glob(os.path.join(HERE, "Results", "temporal_rerun_*")))
    if not cands:
        sys.exit("no temporal_rerun_* batch found")
    batch = cands[-1]
batch = os.path.abspath(batch)
print("batch:", batch)

FIELDS = ["datetime", "status", "elapsed_seconds", "MIPGap", "objective",
          "total_swaps", "num_visited", "makespan_hrs",
          "utility_soon", "utility_normal", "utility_low",
          "original_grids", "active_grids", "removed_grids",
          "num_vars", "num_constrs", "bb_nodes"]

rows = []
for sub in sorted(os.listdir(batch)):
    d = os.path.join(batch, sub)
    if not os.path.isdir(d):
        continue
    js = glob.glob(os.path.join(d, "*metrics*.json"))
    rec = {k: None for k in FIELDS}
    rec["datetime"] = sub
    if not js:
        rec["status"] = "MISSING"
        rows.append(rec)
        continue
    m = json.load(open(js[0], encoding="utf-8"))
    rec["status"] = m.get("solve_status") or m.get("status")
    rec["MIPGap"] = m.get("mip_gap_pct")
    if rec["MIPGap"] is None:
        rec["MIPGap"] = m.get("mip_gap")
    rec["objective"] = m.get("objective_value", m.get("best_obj"))
    rec["total_swaps"] = m.get("total_swaps")
    rec["num_visited"] = m.get("num_visited_grids", m.get("num_visited"))
    rec["makespan_hrs"] = m.get("makespan_hrs")
    rec["utility_soon"] = m.get("utility_soon")
    rec["utility_normal"] = m.get("utility_normal")
    rec["utility_low"] = m.get("utility_low")
    rec["original_grids"] = m.get("original_grid_count")
    rec["active_grids"] = m.get("active_grid_count")
    rec["removed_grids"] = m.get("removed_zero_utility")
    rec["num_vars"] = m.get("num_vars")
    rec["num_constrs"] = m.get("num_constrs")
    rec["bb_nodes"] = m.get("bb_nodes")
    for k in ("cpu_time_s", "runtime_s", "elapsed_s"):
        if m.get(k):
            rec["elapsed_seconds"] = m[k]
            break
    rows.append(rec)

out = os.path.join(batch, "temporal_rebuilt_%s.csv" % time.strftime("%Y%m%d_%H%M%S"))
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

print("\n%-18s %-28s %8s %8s %8s %8s %6s %6s" %
      ("datetime", "status", "obj", "gap%", "t(s)", "swaps", "visit", "grids"))
print("-" * 100)
for r in rows:
    def f(x, p=3):
        return ("%." + str(p) + "f") % x if isinstance(x, (int, float)) else "-"
    print("%-18s %-28s %8s %8s %8s %8s %6s %6s" %
          (r["datetime"], str(r["status"])[:28], f(r["objective"]), f(r["MIPGap"], 4),
           f(r["elapsed_seconds"], 1), f(r["total_swaps"], 0),
           f(r["num_visited"], 0), f(r["active_grids"], 0)))

ok = [r for r in rows if isinstance(r["objective"], (int, float))]
print("\nsnapshots with an objective: %d / %d" % (len(ok), len(rows)))
if ok:
    import statistics as st
    objs = [r["objective"] for r in ok]
    ts = [r["elapsed_seconds"] for r in ok if isinstance(r["elapsed_seconds"], (int, float))]
    gaps = [r["MIPGap"] for r in ok if isinstance(r["MIPGap"], (int, float))]
    print("  objective  mean=%.3f  sd=%.3f  min=%.3f  max=%.3f" %
          (st.mean(objs), st.pstdev(objs), min(objs), max(objs)))
    if ts:
        print("  solve time mean=%.1f  sd=%.1f  min=%.1f  max=%.1f  CV=%.2f" %
              (st.mean(ts), st.pstdev(ts), min(ts), max(ts), st.pstdev(ts) / st.mean(ts)))
    if gaps:
        print("  MIP gap    mean=%.3f%%  max=%.3f%%" % (st.mean(gaps), max(gaps)))
print("\nwrote", out)
