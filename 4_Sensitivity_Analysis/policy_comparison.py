# -*- coding: utf-8 -*-
"""
Benchmark the paper's pipeline against architecture-free dispatching policies.
=============================================================================

METHODS COMPARED
----------------
  Benchmarks (no part of the paper's method -- no fluid utility, no STGraph, no MIP)
     P1  nearest-neighbour only            P2  max low-battery count
     P3  max demand rate                   P4  max demand / inventory
     P5  max estimated unmet demand        P6  max target-inventory deviation
     P7  Clarke & Wright savings           NONE  do nothing
     (see benchmark_policies.py for the exact definitions and references)

  Reference of the paper's own method
     M1  full pipeline: fluid utility + STGraph + MIP
     M6  PSIV score + greedy nearest-neighbour + 2-opt
         -> M6 vs M1 isolates the value of the MIP routing
         -> M1 vs the P's isolates the value of the fluid utility


USAGE
-----
    # everything, five snapshots, 500 DES replications  (default)
    python policy_comparison.py

    # policies only -- no MIP is solved
    python policy_comparison.py --no-mip

    # one snapshot, fewer replications, for a quick look
    python policy_comparison.py 
    --snapshots "2025/10/25 07:00" "2025/11/09 07:00" "2025/10/24 14:00" "2025/11/04 15:00" "2025/10/27 09:00" 
    --n-rep 200 --out Results/policy_comparison_v2

    # list what would run, without running it
    python policy_comparison.py --dry-run

OUTPUT (always archived)
------------------------
    Results/policy_comparison_<YYYYmmdd_HHMMSS>/
        manifest.json                     parameters, snapshots, status, timing
        policy_comparison.csv             one row per (snapshot, method)
        routes/<snapshot>/<method>.json    the exact route each method produced

Nothing is written to stdout only: the CSV and the routes make every number in the
paper reproducible from the archive alone.
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
import benchmark_policies as BP

PRED_FILE = os.path.join(ROOT, "2_Training", "Training_Results",
                         "20260727_115935", "prediction_CB_Hurdle.csv")
# The five FORMAL snapshots: ranks 1 / 2 / 50 / 200 / 400 of the net-drift
# ranking over the all-hours pool, i.e. a pre-declared ladder from "the region is
# bleeding" to "nothing drains at all":
#
#   rank    snapshot          net_ds   drain   what it is for
#      1    2025/10/25 07:00   0.1615    485   morning peak, strongest net outflow
#      2    2025/11/09 07:00   0.1610    494   morning peak, strongest net outflow
#     50    2025/10/24 14:00   0.0721    220   moderate outflow
#    200    2025/11/04 15:00   0.0190     58   marginal outflow
#    400    2025/10/27 09:00  -0.0218      0   CONTROL: no grid drains, returns cover
#                                              demand; do-nothing is already at the
#                                              ceiling and no method can gain here
#
# net_ds = sum_j(lambda_j - rho_j) / sum_j(n_soon_j + n_normal_j): the fraction of
# the standing stock the region loses over the 1-hour horizon (negative = gains).
# The old criterion (d/s = sum lambda / stock) ignored rho and put 2025/11/08 18:00
# first -- a snapshot with 3293 rentals but 3257 returns, where nothing drains and
# every method ties.  A benchmark must be run where the treatment can matter, and
# must say which regime each instance is in.
# Regenerate with:  python select_snapshots.py --all-hours --ranks 1 2 50 200 400
DEFAULT_SNAPSHOTS = ["2025/10/25 07:00",
                     "2025/11/09 07:00",
                     "2025/10/24 14:00",
                     "2025/11/04 15:00",
                     "2025/10/27 09:00"]

C_MAX, T_TOTAL, P_INT = 20, 1.0, 12
SPEED, SWAP_C, MAX_TRAVEL, K_NB = 30.0, 0.02, 0.2, 50
Y_LEVELS = list(range(1, 11))
Y_MAX = max(Y_LEVELS)          # per-grid swap depth cap shared with the benchmarks
TIME_LIMIT = 1200

DEFAULT_N_REP = 500            # under CRN the paired CI is driven by the coupling,
                               # not by the replication count; 200 already resolves
                               # what 100 independent streams could not.

FIELDS = ["snapshot", "method", "kind", "status", "n_stops", "total_swaps",
          "route_span_h",
          # global scope (all grids)
          "DES_served_quality", "DES_served_quality_ci",
          "DES_lost", "DES_lost_ci", "DES_stockout_events",
          "DES_end_serviceable", "DES_end_low",
          # visited scope (only the grids the route stopped at) + its CIs
          "V_served_quality", "V_served_quality_ci",
          "V_lost", "V_lost_ci", "V_stockout_events", "V_stockout_events_ci",
          "V_end_serviceable", "V_end_low",
          # budget efficiency: raw visited totals reward spending the whole budget
          # on a handful of grids, so normalise by stops and by swaps actually placed
          "Vq_per_stop", "Vq_per_swap",
          # timing scope: the one axis only a time-indexed method competes on
          "late_visits", "late_visit_frac",
          "mean_stock_visit", "mean_stock_at_swap",
          # PAIRED differences against the do-nothing floor (common random numbers)
          "d_quality", "d_quality_ci", "d_lost", "d_lost_ci",
          "d_Vquality", "d_Vquality_ci", "d_Vlost", "d_Vlost_ci",
          "d_Vq_per_stop", "d_Vq_per_stop_ci",
          "model_objective", "MIPGap", "elapsed_s", "duplicate_of"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_snapshot(ts):
    """-> grids, grid_params (fluid rates NOT used by the benchmarks), coords."""
    grids, gp, snap = PP.prepare_optimize_inputs(PRED_FILE, target_datetime=ts)
    coords = PP.extract_grid_coordinates(snap)
    return grids, gp, coords


def _seed_for(args, index):
    """Deterministic per-snapshot DES seed.

    Deliberately NOT derived from `hash(tag)`: Python salts str hashes per process,
    so the same command produced different numbers on every run and the archive
    could not be reproduced even from itself.
    """
    return args.seed + 1000 * index


def _stat(ev, key, name, field):
    """Read mean / ci95 / paired_delta / paired_ci95 out of evaluate_paired's output."""
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
    """Format a paired difference with its CI, tagged REAL / ns.

    With common random numbers the CI attached to a difference is the CI *of that
    difference*, so `|d| > ci` is exactly the paired two-sided 5% test.  Printing
    the verdict beside the number is the difference between "M1 is 4.7 units better"
    and "M1 is 4.7 +- 3.5 units better, and that is outside the noise".
    """
    if v is None:
        return "-"
    if ci is None:
        return "%+.3f" % v
    return "%+.3f+-%.3f %s" % (v, ci, "REAL" if abs(v) > ci else "ns")


def _baseline_route(res):
    """Extract [(grid, arrival_time, y)] from a main_STGraph-style result dict."""
    return [(r["grid"], r["arrival_time"], r["y_swapped"])
            for r in res.get("route", [])
            if r.get("grid") not in (None, "DEPOT") and r.get("y_swapped", 0) > 0]


def run_m1(ts, out_dir):
    """Full pipeline.  Imported lazily so --no-mip never touches gurobipy."""
    from main_STGraph import run_optimization_pipeline
    os.makedirs(out_dir, exist_ok=True)
    res = run_optimization_pipeline(
        data_file=PRED_FILE, target_datetime=ts, depot_lat=None, depot_lon=None,
        vehicle_speed_kmh=SPEED, C_max=C_MAX, T_total=T_TOTAL, P_intervals=P_INT,
        y_levels=Y_LEVELS, swap_time_c=SWAP_C, max_travel_time=MAX_TRAVEL,
        K_neighbors=K_NB, output_dir=out_dir, verbose=False,
        experiment_id="M1", instance_name="policy_cmp",
        geo_fencing=True, knn_enabled=True, time_limit_s=TIME_LIMIT)
    col = res.get("collector")
    return _baseline_route(res), res.get("objective"), getattr(col, "mip_gap_pct", None)


def run_m6(ts, out_dir):
    """Greedy nearest-neighbour + 2-opt on the PSIV score."""
    from main_heuristic import run_heuristic_pipeline
    os.makedirs(out_dir, exist_ok=True)
    res = run_heuristic_pipeline(
        data_file=PRED_FILE, target_datetime=ts, depot_lat=None, depot_lon=None,
        vehicle_speed_kmh=SPEED, C_max=C_MAX, T_total=T_TOTAL, P_intervals=P_INT,
        y_levels=Y_LEVELS, swap_time_c=SWAP_C, max_travel_time=MAX_TRAVEL,
        K_neighbors=K_NB, output_dir=out_dir, verbose=False,
        experiment_id="M6", instance_name="policy_cmp")
    return _baseline_route(res), res.get("objective"), None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compare the paper's pipeline against architecture-free policies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--snapshots", nargs="+", default=DEFAULT_SNAPSHOTS,
                    help="datetime(s), format 'YYYY/MM/DD HH:MM'")
    ap.add_argument("--n-rep", type=int, default=DEFAULT_N_REP,
                    help="DES replications per snapshot  [%d]" % DEFAULT_N_REP)
    ap.add_argument("--dt", type=float, default=0.005,
                    help="DES time step in hours  [0.005].  Coarser steps are faster "
                         "but systematically depress lost demand: measured bias is "
                         "-0.76%% at 0.01 and -2.6%% at 0.02, so keep 0.005 unless "
                         "you are only after the quality metric")
    ap.add_argument("--seed", type=int, default=20261003, help="DES base seed")
    ap.add_argument("--y-max", type=int, default=Y_MAX,
                    help="per-grid swap depth for EVERY method, benchmarks included "
                         "[%d = max(y_levels)]" % Y_MAX)
    ap.add_argument("--no-mip", action="store_true",
                    help="skip M1/M6 -- policies only, finishes in seconds")
    ap.add_argument("--out", default=None, help="output directory (default: timestamped)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    policy_fns = BP.all_policies(max_y=args.y_max)
    methods = [("NONE", "benchmark")] + [(p, "benchmark")
                                         for p in ("P1", "P2", "P3", "P4",
                                                   "P5", "P6", "P7")]
    if not args.no_mip:
        methods += [("M1", "ours"), ("M6", "ours")]

    batch = args.out or os.path.join(
        HERE, "Results", "policy_comparison_" + time.strftime("%Y%m%d_%H%M%S"))
    batch = os.path.abspath(batch)

    print(__doc__.split("USAGE")[0].strip()[:400])
    print("\n" + "=" * 72)
    print("  output   : %s" % batch)
    print("  snapshots: %s" % ", ".join(args.snapshots))
    print("  methods  : %s" % ", ".join(m for m, _ in methods))
    print("  DES reps : %d  (common random numbers)" % args.n_rep)
    print("  y cap    : %d  (same for every method)" % args.y_max)
    print("=" * 72, flush=True)

    if args.dry_run:
        print("\n[dry-run] nothing executed.")
        return 0

    os.makedirs(batch, exist_ok=True)
    routes_dir = os.path.join(batch, "routes")
    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    prediction_file=PRED_FILE, snapshots=args.snapshots,
                    n_rep=args.n_rep, seed=args.seed, no_mip=args.no_mip,
                    y_max=args.y_max,
                    variance_reduction="DES.evaluate_paired (common random numbers)",
                    C_max=C_MAX, T_total=T_TOTAL, P_intervals=P_INT,
                    speed_kmh=SPEED, swap_time_h=SWAP_C,
                    max_travel_h=MAX_TRAVEL, K_neighbors=K_NB,
                    time_limit_s=TIME_LIMIT,
                    policy_definitions=BP.POLICY_INFO, runs=[])
    with open(os.path.join(batch, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    rows = []
    for index, ts in enumerate(args.snapshots):
        tag = ts.replace("/", "").replace(" ", "_").replace(":", "")
        os.makedirs(os.path.join(routes_dir, tag), exist_ok=True)
        print("\n" + "#" * 72)
        print("#  SNAPSHOT  %s" % ts)
        print("#" * 72, flush=True)
        t_snap = time.perf_counter()

        grids, gp, coords = _load_snapshot(ts)
        th_s = gp[grids[0]]["theta_soon"]
        th_n = gp[grids[0]]["theta_normal"]
        run_seed = _seed_for(args, index)
        print("  grids=%d  theta_soon=%.4f theta_normal=%.4f  seed=%d"
              % (len(grids), th_s, th_n, run_seed), flush=True)

        # ---- phase 1: build every route (the MIP runs here) ------------------
        routes, status, scalars = {}, {}, {}
        for name, kind in methods:
            t0 = time.perf_counter()
            print("\n  --- %-5s [%s] ---" % (name, kind), flush=True)
            route, obj, gap = [], None, None
            try:
                if name in policy_fns:
                    route = policy_fns[name](gp, coords, T_TOTAL, C_MAX)
                elif name == "M1":
                    route, obj, gap = run_m1(ts, os.path.join(batch, "m1_raw", tag))
                elif name == "M6":
                    route, obj, _ = run_m6(ts, os.path.join(batch, "m6_raw", tag))
                status[name] = "OK"
            except Exception as exc:                       # never kill the batch
                status[name] = "ERROR: %s" % str(exc)[:110]
                print("     [ERROR] %s" % exc, flush=True)
            routes[name] = route
            scalars[name] = dict(model_objective=obj, MIPGap=gap,
                                 elapsed_s=round(time.perf_counter() - t0, 2))
            print("     stops=%d swaps=%d in %.1fs"
                  % (len(route), int(sum(y for _, _, y in route)),
                     scalars[name]["elapsed_s"]), flush=True)

            with open(os.path.join(routes_dir, tag, name + ".json"), "w",
                      encoding="utf-8") as f:
                json.dump([{"grid": g, "arrival_time": t, "y": y} for g, t, y in route],
                          f, indent=1)

        # ---- phase 2: ONE coupled DES pass over every surviving route --------
        # NONE is always OK (constant route), so the paired deltas always have a
        # reference.  Arms that ERRORed are excluded: evaluating their empty route
        # would manufacture a fake "ties with do-nothing" result.
        evalable = {n: routes[n] for n, _ in methods if status[n] == "OK"}
        print("\n  [DES] %d routes x %d replications, common random numbers"
              % (len(evalable), args.n_rep), flush=True)
        t_des = time.perf_counter()
        ev = DES.evaluate_paired(gp, evalable, T_TOTAL, th_s, th_n,
                                 n_rep=args.n_rep, seed=run_seed, dt=args.dt)
        print("  [DES] done in %.1fs" % (time.perf_counter() - t_des), flush=True)

        # ---- phase 3: assemble rows -----------------------------------------
        seen_routes = {}
        for name, kind in methods:
            rec = {k: None for k in FIELDS}
            rec.update(snapshot=ts, method=name, kind=kind, status=status[name])
            route = routes[name]
            rec["n_stops"] = len(route)
            rec["total_swaps"] = int(sum(y for _, _, y in route))
            rec["route_span_h"] = round(max((t for _, t, _ in route), default=0.0), 4)
            rec.update({k: v for k, v in scalars[name].items()})

            key = tuple((g, round(float(t), 6), int(y)) for g, t, y in route)
            if key in seen_routes:
                rec["duplicate_of"] = seen_routes[key]
            else:
                seen_routes[key] = name

            if status[name] != "OK":
                print("  %-5s DES skipped (method failed)" % name, flush=True)
                rows.append(rec)
                continue

            def _v(metric, field="mean"):
                return _stat(ev, metric, name, field)

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

            # paired gains over the do-nothing floor.  paired_delta/paired_ci95 are
            # absent for NONE itself (it is the reference).
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

            print("  %-5s G:q=%8.2f d=%+7.2f+-%-6.2f lost=%7.2f d=%+6.2f | "
                  "V:q/st=%7.3f d=%+7.3f | late=%.2f"
                  % (name, rec["DES_served_quality"], rec["d_quality"] or 0.0,
                     rec["d_quality_ci"] or 0.0, rec["DES_lost"],
                     rec["d_lost"] or 0.0, rec["Vq_per_stop"] or 0.0,
                     rec["d_Vq_per_stop"] or 0.0, rec["late_visits"] or 0.0),
                  flush=True)
            rows.append(rec)

        manifest["runs"] += [{k: r[k] for k in
                              ("snapshot", "method", "status", "elapsed_s")}
                             for r in rows if r["snapshot"] == ts]
        with open(os.path.join(batch, "policy_comparison.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(batch, "manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print("\n  snapshot done in %.1fs" % (time.perf_counter() - t_snap), flush=True)

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 100)
    print("  SUMMARY   DES with COMMON RANDOM NUMBERS within each snapshot")
    print("  d:* = paired difference vs the do-nothing floor; a difference is real")
    print("        only if |d| > its _ci (the CI of the paired difference).")
    print("  GLOBAL = all grids   VISITED = only the grids the route stopped at")
    print("=" * 100)
    print("%-17s %-6s %5s %5s | %11s %16s | %11s %16s | %9s %16s" %
          ("snapshot", "method", "stops", "swap", "G:quality", "d:quality vs NONE",
           "G:lost", "d:lost vs NONE", "V:q/stop", "d:Vq/stop vs NONE"))
    print("-" * 116)
    for ts in args.snapshots:
        for r in [x for x in rows if x["snapshot"] == ts]:
            if r["DES_served_quality"] is None:
                print("%-17s %-6s %5d %5d | %s"
                      % (ts, r["method"], r["n_stops"] or 0, r["total_swaps"] or 0,
                         r["status"]))
                continue
            if r["method"] == "NONE":
                dq = dl = dv = "(floor)"
            else:
                dq, dl, dv = (_delta(r["d_quality"], r["d_quality_ci"]),
                              _delta(r["d_lost"], r["d_lost_ci"]),
                              _delta(r["d_Vq_per_stop"], r["d_Vq_per_stop_ci"]))
            print("%-17s %-6s %5d %5d | %11.2f %16s | %11.2f %16s | %9.3f %16s" %
                  (ts, r["method"], r["n_stops"], r["total_swaps"],
                   r["DES_served_quality"], dq, r["DES_lost"], dl,
                   r["Vq_per_stop"] or 0.0, dv))
        print("-" * 116)
    dup = [(r["snapshot"], r["method"], r["duplicate_of"]) for r in rows
           if r.get("duplicate_of")]
    if dup:
        print("\nIDENTICAL ROUTES (the second is not an independent benchmark):")
        for s, m, o in dup:
            print("   %-17s %-5s == %s" % (s, m, o))
    print("\narchived to: %s" % batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
