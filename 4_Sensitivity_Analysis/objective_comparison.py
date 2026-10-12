# -*- coding: utf-8 -*-
"""Objective-function comparison across snapshots.

USAGE
-----
    python objective_comparison.py                      # 5 snapshots, all 6 objectives
    python objective_comparison.py --objectives psiv lost swap
    python objective_comparison.py --n-rep 1000
    python objective_comparison.py --out Results/objective_comparison_v2
    python objective_comparison.py --dry-run
"""
import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "3_Optimization"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import Pre_Process as PP
import des_route_eval as DES
from alt_objectives import generate_alt_utility_matrix, MODES
import main_STGraph as M

DATA = os.path.join(ROOT, "2_Training", "Training_Results",
                    "20260727_115935", "prediction_CB_Hurdle.csv")
# Same five FORMAL snapshots as policy_comparison.py -- ranks 1 / 2 / 50 / 200 / 400
# of the net-drift ranking, a pre-declared ladder from strong net outflow to no
# drain at all.  Keeping the two scripts on one instance set is the point: an
# objective comparison run on a different regime than the policy comparison cannot
# be read against it.
#   rank 1   2025/10/25 07:00  net_ds +0.1615  drain 485   morning peak
#   rank 2   2025/11/09 07:00  net_ds +0.1610  drain 494   morning peak
#   rank 50  2025/10/24 14:00  net_ds +0.0721  drain 220   moderate outflow
#   rank 200 2025/11/04 15:00  net_ds +0.0190  drain  58   marginal outflow
#   rank 400 2025/10/27 09:00  net_ds -0.0218  drain   0   CONTROL (nothing drains)
# net_ds = sum_j(lambda_j - rho_j) / sum_j(n_soon_j + n_normal_j).
# Regenerate with:  python select_snapshots.py --all-hours --ranks 1 2 50 200 400
# NOTE: on rank 400 every grid refills on its own, so a reduction-based objective
# (`lost`) and a deviation-based one (`target`) are both degenerate there by
# construction; that is a property of the regime, not a defect of the arm.
DEFAULT_SNAPSHOTS = ["2025/10/25 07:00",
                     "2025/11/09 07:00",
                     "2025/10/24 14:00",
                     "2025/11/04 15:00",
                     "2025/10/27 09:00"]

T, C_MAX, P_INT = 1.0, 20, 12
DEFAULT_N_REP = 500
TIME_LIMIT = 1200

FIELDS = ["snapshot", "objective", "kind", "status",
          "n_swaps_route", "active_grids", "removed_zero_utility",
          "model_obj", "MIPGap", "elapsed_s",
          # global scope
          "DES_served_quality", "DES_served_quality_ci",
          "DES_lost", "DES_lost_ci", "DES_stockout_events",
          "DES_end_serviceable", "DES_end_low",
          # visited scope
          "V_served_quality", "V_served_quality_ci",
          "V_lost", "V_lost_ci", "V_stockout_events", "V_stockout_events_ci",
          "V_end_serviceable", "V_end_low",
          "Vq_per_stop", "Vq_per_swap",
          # timing
          "late_visits", "late_visit_frac", "mean_stock_visit", "mean_stock_at_swap",
          # paired differences against the do-nothing floor
          "d_quality", "d_quality_ci", "d_lost", "d_lost_ci",
          "d_Vquality", "d_Vquality_ci", "d_Vlost", "d_Vlost_ci",
          "d_Vq_per_stop", "d_Vq_per_stop_ci"]

ORIG_GEN = M.generate_offline_utility_matrix


def patch(mode):
    """Install the coefficient tensor for `mode` (the model itself is untouched)."""
    if mode == "psiv":
        M.generate_offline_utility_matrix = ORIG_GEN
        return
    def gen(**kw):
        return generate_alt_utility_matrix(
            kw["grids"], kw["C_max"], kw["T_total"], kw["P_intervals"],
            kw["grid_params"], mode, y_levels=kw.get("y_levels"))
    M.generate_offline_utility_matrix = gen


def route_of(res):
    return [(r["grid"], r["arrival_time"], r["y_swapped"])
            for r in res.get("route", [])
            if r.get("grid") not in (None, "DEPOT") and r.get("y_swapped", 0) > 0]


def _stat(ev, key, name, field):
    per_method = ev.get(key)
    if not per_method:
        return None
    d = per_method.get(name)
    if d is None:
        return None
    v = d.get(field)
    return None if v is None else float(v)


def _r3(v):
    return None if v is None else round(v, 3)


def _delta(v, ci):
    """Paired difference with its CI, tagged REAL / ns (|d| > ci is the 5% test)."""
    if v is None:
        return "-"
    if ci is None:
        return "%+.3f" % v
    return "%+.3f+-%.3f %s" % (v, ci, "REAL" if abs(v) > ci else "ns")


def main():
    ap = argparse.ArgumentParser(
        description="Swap only the objective tensor; judge the routes in the DES.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--snapshots", nargs="+", default=DEFAULT_SNAPSHOTS)
    ap.add_argument("--objectives", nargs="+", default=["psiv"] + list(MODES),
                    choices=["psiv"] + list(MODES))
    ap.add_argument("--n-rep", type=int, default=DEFAULT_N_REP,
                    help="DES replications per arm  [%d]" % DEFAULT_N_REP)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--time-limit", type=int, default=TIME_LIMIT,
                    help="Gurobi time limit per arm, seconds  [%d]" % TIME_LIMIT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    batch = args.out or os.path.join(
        HERE, "Results", "objective_comparison_" + time.strftime("%Y%m%d_%H%M%S"))
    batch = os.path.abspath(batch)

    print(__doc__.split("USAGE")[0].strip()[:400])
    print("\n" + "=" * 72)
    print("  output    : %s" % batch)
    print("  snapshots : %s" % ", ".join(args.snapshots))
    print("  objectives: %s" % ", ".join(args.objectives))
    print("  DES reps  : %d  (common random numbers)" % args.n_rep)
    print("=" * 72, flush=True)
    if args.dry_run:
        print("\n[dry-run] nothing executed.")
        return 0

    os.makedirs(batch, exist_ok=True)
    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    prediction_file=DATA, snapshots=args.snapshots,
                    objectives=args.objectives, n_rep=args.n_rep, seed=args.seed,
                    time_limit_s=args.time_limit,
                    variance_reduction="DES.evaluate_paired (common random numbers)",
                    C_max=C_MAX, T_total=T, P_intervals=P_INT,
                    y_levels=list(range(1, 11)), runs=[])
    with open(os.path.join(batch, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    rows = []
    try:
        for index, ts in enumerate(args.snapshots):
            tag = ts.replace("/", "").replace(" ", "_").replace(":", "")
            print("\n" + "#" * 70)
            print("#  SNAPSHOT %s" % ts)
            print("#" * 70, flush=True)

            grids, gp, _ = PP.prepare_optimize_inputs(DATA, target_datetime=ts)
            th_s = gp[grids[0]]["theta_soon"]
            th_n = gp[grids[0]]["theta_normal"]
            run_seed = args.seed + 1000 * index
            print("  grids=%d  seed=%d" % (len(grids), run_seed), flush=True)

            # ---- phase 1: one MIP per objective, tensor swapped ----------------
            routes, status, scalars = {}, {}, {}
            for mode in args.objectives:
                d = os.path.join(batch, tag, mode)
                os.makedirs(d, exist_ok=True)
                print("\n  --- %s ---" % mode, flush=True)
                patch(mode)
                t0 = time.perf_counter()
                route, obj, gap, ps = [], None, None, {}
                try:
                    res = M.run_optimization_pipeline(
                        data_file=DATA, target_datetime=ts, depot_lat=None,
                        depot_lon=None, vehicle_speed_kmh=30.0, C_max=C_MAX,
                        T_total=T, P_intervals=P_INT, y_levels=list(range(1, 11)),
                        swap_time_c=0.02, max_travel_time=0.2, K_neighbors=50,
                        output_dir=d, verbose=False, experiment_id=mode.upper(),
                        instance_name="objcmp", geo_fencing=True, knn_enabled=True,
                        time_limit_s=args.time_limit)
                    col = res.get("collector")
                    route = route_of(res)
                    obj = res.get("objective")
                    gap = getattr(col, "mip_gap_pct", None)
                    ps = res.get("pruning_stats", {}) or {}
                    status[mode] = str(res.get("status"))
                    if not route:
                        # Not an error: the pruning step legitimately empties the
                        # candidate set for a degenerate tensor (`lost` at the peak,
                        # `target` off-peak).  Recorded, never DES-scored.
                        status[mode] = str(res.get("status")) + " [no route]"
                except Exception as exc:
                    status[mode] = "ERROR: %s" % str(exc)[:110]
                    print("    [ERROR] %s" % exc, flush=True)
                routes[mode] = route
                scalars[mode] = dict(
                    active_grids=ps.get("active_grids"),
                    removed_zero_utility=ps.get("removed_zero_utility"),
                    model_obj=obj, MIPGap=gap,
                    elapsed_s=round(time.perf_counter() - t0, 1))
                print("    %s | route=%d swaps=%d obj=%s | %.0fs"
                      % (status[mode], len(route), int(sum(y for _, _, y in route)),
                         obj, scalars[mode]["elapsed_s"]), flush=True)

            # ---- phase 2: ONE coupled DES pass --------------------------------
            evalable = {"NONE": []}
            evalable.update({m: routes[m] for m in args.objectives if routes[m]})
            print("\n  [DES] %d routes x %d replications, common random numbers"
                  % (len(evalable), args.n_rep), flush=True)
            ev = DES.evaluate_paired(gp, evalable, T, th_s, th_n,
                                     n_rep=args.n_rep, seed=run_seed, dt=args.dt)

            # ---- phase 3: rows ------------------------------------------------
            for mode in ["NONE"] + list(args.objectives):
                rec = {k: None for k in FIELDS}
                rec.update(snapshot=ts, objective=mode,
                           kind="floor" if mode == "NONE" else "objective")
                if mode == "NONE":
                    rec["status"] = "baseline (do nothing)"
                    rec["n_swaps_route"] = 0
                else:
                    rec.update(status=status[mode], **scalars[mode])
                    rec["n_swaps_route"] = len(routes[mode])
                if mode not in evalable:
                    rows.append(rec)
                    print("  %-9s DES skipped (no route)" % mode, flush=True)
                    continue

                def _v(metric, field="mean"):
                    return _stat(ev, metric, mode, field)

                rec["DES_served_quality"] = _r3(_v("served_quality"))
                rec["DES_served_quality_ci"] = _r3(_v("served_quality", "ci95"))
                rec["DES_lost"] = _r3(_v("lost"))
                rec["DES_lost_ci"] = _r3(_v("lost", "ci95"))
                rec["DES_stockout_events"] = _r3(_v("stockout_events"))
                rec["DES_end_serviceable"] = _r3(_v("end_serviceable"))
                rec["DES_end_low"] = _r3(_v("end_low"))
                rec["V_served_quality"] = _r3(_v("v_served_quality"))
                rec["V_served_quality_ci"] = _r3(_v("v_served_quality", "ci95"))
                rec["V_lost"] = _r3(_v("v_lost"))
                rec["V_lost_ci"] = _r3(_v("v_lost", "ci95"))
                rec["V_stockout_events"] = _r3(_v("v_stockout_events"))
                rec["V_stockout_events_ci"] = _r3(_v("v_stockout_events", "ci95"))
                rec["V_end_serviceable"] = _r3(_v("v_end_serviceable"))
                rec["V_end_low"] = _r3(_v("v_end_low"))
                rec["Vq_per_stop"] = _r3(_v("v_quality_per_stop"))
                rec["Vq_per_swap"] = _r3(_v("v_quality_per_swap"))
                nv = _v("n_visited") or 0.0
                lv = _v("late_visits") or 0.0
                rec["late_visits"] = _r3(lv)
                rec["late_visit_frac"] = _r3(lv / nv) if nv else None
                rec["mean_stock_visit"] = _r3(_v("mean_stock_visit"))
                rec["mean_stock_at_swap"] = _r3(_v("mean_stock_at_swap"))
                rec["d_quality"] = _r3(_v("served_quality", "paired_delta"))
                rec["d_quality_ci"] = _r3(_v("served_quality", "paired_ci95"))
                rec["d_lost"] = _r3(_v("lost", "paired_delta"))
                rec["d_lost_ci"] = _r3(_v("lost", "paired_ci95"))
                rec["d_Vquality"] = _r3(_v("v_served_quality", "paired_delta"))
                rec["d_Vquality_ci"] = _r3(_v("v_served_quality", "paired_ci95"))
                rec["d_Vlost"] = _r3(_v("v_lost", "paired_delta"))
                rec["d_Vlost_ci"] = _r3(_v("v_lost", "paired_ci95"))
                rec["d_Vq_per_stop"] = _r3(_v("v_quality_per_stop", "paired_delta"))
                rec["d_Vq_per_stop_ci"] = _r3(_v("v_quality_per_stop", "paired_ci95"))
                rows.append(rec)

                if mode == "NONE":
                    print("  %-9s G:q=%8.2f lost=%7.2f  (floor)"
                          % (mode, rec["DES_served_quality"], rec["DES_lost"]),
                          flush=True)
                else:
                    print("  %-9s G:q d=%16s lost d=%16s | V:q/stop=%8.3f d=%16s"
                          % (mode, _delta(rec["d_quality"], rec["d_quality_ci"]),
                             _delta(rec["d_lost"], rec["d_lost_ci"]),
                             rec["Vq_per_stop"],
                             _delta(rec["d_Vq_per_stop"], rec["d_Vq_per_stop_ci"])),
                          flush=True)

            manifest["runs"] += [{"snapshot": r["snapshot"], "objective": r["objective"],
                                  "status": r["status"], "elapsed_s": r["elapsed_s"]}
                                 for r in rows if r["snapshot"] == ts]
            _write(batch, rows, manifest)

        # ---- summary: which objective produces the best routes ---------------
        print("\n" + "=" * 96)
        print("  JUDGED IN THE DES (never sees Omega).  d:* = paired difference vs")
        print("  do-nothing; a difference counts only if |d| > its _ci.")
        print("=" * 96)
        for ts in args.snapshots:
            print("\n  %s" % ts)
            print("    %-9s %8s %16s %16s %10s %16s" %
                  ("objective", "route", "d:quality", "d:lost", "V:q/stop", "d:V:q/stop"))
            for r in [x for x in rows if x["snapshot"] == ts and x["objective"] != "NONE"]:
                if r["DES_served_quality"] is None:
                    print("    %-9s %8s %s" % (r["objective"], r["n_swaps_route"],
                                               r["status"]))
                    continue
                print("    %-9s %8d %16s %16s %10.3f %16s" %
                      (r["objective"], r["n_swaps_route"],
                       _delta(r["d_quality"], r["d_quality_ci"]),
                       _delta(r["d_lost"], r["d_lost_ci"]),
                       r["Vq_per_stop"] or 0.0,
                       _delta(r["d_Vq_per_stop"], r["d_Vq_per_stop_ci"])))
            floor = next((x for x in rows
                          if x["snapshot"] == ts and x["objective"] == "NONE"), None)
            if floor and floor["DES_served_quality"] is not None:
                print("    %-9s %8d %16.2f %16.2f %10s" %
                      ("(NONE)", 0, floor["DES_served_quality"], floor["DES_lost"], "-"))
    finally:
        M.generate_offline_utility_matrix = ORIG_GEN

    print("\narchived to: %s" % batch)
    return 0


def _write(batch, rows, manifest=None):
    with open(os.path.join(batch, "objective_comparison.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    if manifest is not None:
        with open(os.path.join(batch, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
