# -*- coding: utf-8 -*-
"""One-session rerun of EVERY row of the formulation-comparison table.

WHY THIS EXISTS
---------------
The rows of the formulation table were archived in three different batches:

    M1            3_Optimization/Optimization_Result_Summary/20261002_110051  (11:04)
    M2a, M2b      Results/ablation_rerun_20261002_181355                    (18:13)
    M3, M4, M5    Results/baseline_rerun_20261002_183500 / _213819          (18:35, 21:38)

Every number in that table is reproduced exactly by those runs, so nothing is
wrong with the reported values.  But the WALL-CLOCK column mixes sessions of the
same day, and the docstring of `baseline_rerun.py` records why that matters:
M2a's model is byte-identical before and after the Geo-Fencing tolerance fix, yet
its solve time moved 413s -> 287s between sessions.  Objective values, MIP gaps,
B&B node counts, dual bounds and model sizes are session-independent; only the
seconds are not.

This script re-measures all rows in ONE session so the whole table -- seconds
included -- is internally comparable.

USAGE
-----
    python single_instance_rerun.py                      # all six (~2 h)
    python single_instance_rerun.py --models M1           # M1 only   (~3 min)
    python single_instance_rerun.py --models M1 M2a M2b
    python single_instance_rerun.py --with-m6
    python single_instance_rerun.py --resume Results/single_instance_rerun_20261004_214740
    python single_instance_rerun.py --dry-run

    --resume DIR reuses the models already finished in DIR (read from their
    per-model `*_metrics.json`) and runs only what is missing, so an interrupted
    batch can be continued instead of restarted.

M1 runs first: it is the row with no same-session companion and the cheapest, so
an early Ctrl-C still leaves you with the number you came for.  Ctrl-C at any
point is safe -- each finished model is written to disk as it completes.

Outputs, into Results/single_instance_rerun_<timestamp>/:
    <MODEL>/..._metrics.json, ..._summary.csv, ..._route.csv   (per model)
    single_instance_rerun_summary.csv                           (one row per model)
    table_rows.tex                                             (paste-ready rows)

Budget: M1 ~185s, M2a ~290s, M2b ~960s, M3/M4/M5 up to the 1200s cap each
(M4's wall-clock additionally includes a long explicit model build).
"""
import argparse
import csv
import glob
import inspect
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "3_Optimization"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from main_STGraph import run_optimization_pipeline as run_stgraph
from main_SOS2 import run_optimization_pipeline as run_sos2
from main_delta import run_optimization_pipeline as run_delta
from main_delta_MCF import run_optimization_pipeline as run_mcf
from main_heuristic import run_heuristic_pipeline as run_heuristic

DATA = os.path.join(ROOT, "2_Training", "Training_Results",
                    "20260727_115935", "prediction_CB_Hurdle.csv")
TS = "2025/11/05 18:00"

# The parameters shared by every arm.  Not every runner declares all of them --
# main_SOS2 / main_delta / main_delta_MCF take BigM but no K_neighbors -- so the
# call is filtered against each runner's own signature in `call_model` below.
# (Passing K_neighbors to the Big-M arms is the exact failure recorded in
# Results/baseline_rerun_20261002_183352: "unexpected keyword argument".)
COMMON = dict(data_file=DATA, target_datetime=TS, depot_lat=None, depot_lon=None,
              vehicle_speed_kmh=30.0, C_max=20, T_total=1.0,
              y_levels=list(range(1, 11)), swap_time_c=0.02,
              max_travel_time=0.2, K_neighbors=50, verbose=True,
              instance_name="single_instance_rerun")

# (id, description, runner, extra kwargs) -- M1 first, on purpose.
CASES = [
    ("M1", "STGraph, P=12, Geo-Fence on, KNN K=50", run_stgraph,
     dict(geo_fencing=True, knn_enabled=True, P_intervals=12, time_limit_s=1200)),
    ("M2a", "no Geo-Fence, all 661 grids", run_stgraph,
     dict(geo_fencing=False, knn_enabled=True, P_intervals=12, time_limit_s=1200)),
    ("M2b", "no KNN, fully connected", run_stgraph,
     dict(geo_fencing=True, knn_enabled=False, P_intervals=12, time_limit_s=1200)),
    ("M3", "SOS2-PLA + explicit MTZ, P=5", run_sos2,
     dict(P_intervals=5, time_limit_s=1200)),
    ("M4", "Delta + explicit MTZ, P=5", run_delta,
     dict(P_intervals=5, time_limit_s=1200)),
    ("M5", "Delta + MCF + lazy MTZ, P=10", run_mcf,
     dict(P_intervals=10, time_limit_s=1200)),
]
M6_CASE = ("M6", "Greedy NN + 2-opt heuristic", run_heuristic,
           dict(P_intervals=12))

FIELDS = ["experiment", "desc", "status", "objective", "MIPGap", "best_bound",
          "dual_int_ratio", "elapsed_s", "cpu_time_s", "num_vars",
          "num_constrs", "bb_nodes", "visited", "swaps", "makespan_hrs",
          "reused_from"]


def call_model(target, eid, out_dir, common, extra):
    """Call `target` with only the keyword arguments it actually declares."""
    accepted = set(inspect.signature(target).parameters)
    kw = {k: v for k, v in common.items() if k in accepted}
    kw.update({k: v for k, v in extra.items() if k in accepted})
    dropped = sorted((set(common) | set(extra)) - accepted - set(kw))
    if dropped:
        print("    [kwargs] %s does not accept %s -- dropped"
              % (target.__module__, ", ".join(dropped)), flush=True)
    return target(experiment_id=eid, output_dir=out_dir, **kw)


def _num(v):
    """JSON-safe float: the M5 arm can carry 'inf' (no valid dual bound)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def run_one(eid, desc, target, extra, batch, common):
    d = os.path.join(batch, eid)
    os.makedirs(d, exist_ok=True)
    print("\n" + "=" * 68)
    print("  %s  (%s)" % (eid, desc))
    print("=" * 68, flush=True)
    rec = {k: None for k in FIELDS}
    rec.update(experiment=eid, desc=desc)
    t0 = time.perf_counter()
    try:
        res = call_model(target, eid, d, common, extra)
        col = res.get("collector")
        s = res.get("summary", {}) or {}
        rec.update(status=str(res.get("status")),
                   objective=_num(res.get("objective")),
                   visited=s.get("num_visited"),
                   swaps=s.get("total_swaps"),
                   makespan_hrs=s.get("makespan_hrs"))
        if col is not None:
            for attr, key in [("mip_gap_pct", "MIPGap"),
                              ("num_vars", "num_vars"),
                              ("num_constrs", "num_constrs"),
                              ("bb_nodes", "bb_nodes"),
                              ("best_bound", "best_bound"),
                              ("cpu_time_s", "cpu_time_s")]:
                rec[key] = _num(getattr(col, attr, None))
        if rec["best_bound"] and rec["objective"]:
            rec["dual_int_ratio"] = round(rec["best_bound"] / rec["objective"], 3)
    except Exception as exc:                        # one failing row must not
        rec["status"] = "ERROR: %s" % str(exc)[:110]  # take the others with it
        print("  [ERROR] %s" % exc, flush=True)
    rec["elapsed_s"] = round(time.perf_counter() - t0, 2)
    print("  -> %s | obj=%s gap=%s t=%.0fs vars=%s cons=%s nodes=%s bound=%s"
          % (rec["status"], rec["objective"], rec["MIPGap"], rec["elapsed_s"],
             rec["num_vars"], rec["num_constrs"], rec["bb_nodes"],
             rec["best_bound"]), flush=True)
    return rec


def _elapsed_from_summary(batch, eid):
    """Wall-clock of a model finished in an earlier invocation of this script.

    metrics.json stores only cpu_time_s, so a resumed row would otherwise lose
    the wall-clock number the table reports.
    """
    p = os.path.join(batch, "single_instance_rerun_summary.csv")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("experiment") == eid and r.get("elapsed_s"):
                    return _num(r["elapsed_s"])
    except Exception:
        pass
    return None


def load_finished(batch, eid):
    """Rebuild a record from a finished model's per-model metrics.json."""
    hits = sorted(glob.glob(os.path.join(batch, eid, "*_metrics.json")))
    if not hits:
        return None
    try:
        with open(hits[-1], encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return None
    obj = _num(j.get("best_obj"))
    if obj is None:
        obj = _num(j.get("objective_value"))
    if obj is None:
        return None
    rec = {k: None for k in FIELDS}
    bound = _num(j.get("best_bound"))
    rec.update(experiment=eid, desc="reused", status=j.get("solve_status"),
               objective=obj, best_bound=bound,
               MIPGap=_num(j.get("mip_gap_pct")),
               cpu_time_s=_num(j.get("cpu_time_s")),
               num_vars=_num(j.get("num_vars")),
               num_constrs=_num(j.get("num_constrs")),
               bb_nodes=_num(j.get("bb_nodes")),
               visited=j.get("num_visited_grids"),
               swaps=j.get("total_swaps"),
               makespan_hrs=j.get("makespan_hrs"),
               reused_from=os.path.basename(hits[-1]))
    rec["elapsed_s"] = _elapsed_from_summary(batch, eid)
    if bound and obj:
        rec["dual_int_ratio"] = round(bound / obj, 3)
    return rec


def _cell(v, fmt="%s"):
    return "--" if v is None else (fmt % v)


def _count(v):
    """Exact count with thousands separators (the paper's 'k' form is a manual
    choice: 2,628,762 -> 2,629k).  A zero is reported as '--': the heuristic arm
    sets num_vars/num_constrs/bb_nodes to 0 because it builds no MIP, and a MIP
    with no variables is not a reading."""
    if v is None or v == 0:
        return "--"
    return "{:,}".format(int(round(v)))


def _time_cell(r):
    """Prefer wall-clock; a resumed row may have kept only cpu_time_s."""
    if r.get("elapsed_s") is not None:
        return "%.0f" % r["elapsed_s"]
    if r.get("cpu_time_s") is not None:
        return "%.0f%s" % (r["cpu_time_s"], "\\textsuperscript{cpu}")
    return "--"


def write_tex(rows, path):
    """Paste-ready rows for the formulation-comparison table.

    Built by concatenation, not %-formatting: a bare '%' in a literal (e.g. the
    word 'generated' after a leading '%') is read as a conversion specifier and
    raises TypeError.
    """
    head = ("% generated by single_instance_rerun.py | instance " + TS
            + " | one session | " + time.strftime("%Y-%m-%d %H:%M") + "\n"
            + "% columns: group & status & obj & gap(%) & time(s) & vars & cons & nodes\n"
            + "% vars/cons are exact; collapse to 2,629k by hand if you want the "
              "narrow form\n")
    lines = []
    for r in rows:
        desc = str(r["desc"])
        label = r["experiment"] if desc.startswith("reused") \
            else r["experiment"] + " (" + desc[:20] + ")"
        cells = [label,
                 str(r["status"])[:26],
                 _cell(r["objective"], "%.3f"),
                 _cell(r["MIPGap"], "%.2f"),
                 _time_cell(r),
                 _count(r["num_vars"]),
                 _count(r["num_constrs"]),
                 _count(r["bb_nodes"])]
        lines.append(" & ".join(cells) + " \\\\")
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(lines) + "\n")


def _write(batch, rows):
    """Persist results.  A failure here is cosmetic and must never abort the batch."""
    if not rows:
        return
    try:
        with open(os.path.join(batch, "single_instance_rerun_summary.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    except Exception as exc:
        print("  [warn] could not write the summary CSV: %s" % exc, flush=True)
    try:
        write_tex(rows, os.path.join(batch, "table_rows.tex"))
    except Exception as exc:
        print("  [warn] could not write table_rows.tex: %s" % exc, flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Rerun every formulation-table row in one session.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--models", nargs="+", default=[c[0] for c in CASES],
                    help="subset of M1 M2a M2b M3 M4 M5   [all]")
    ap.add_argument("--with-m6", action="store_true",
                    help="also run the Greedy+2-opt heuristic")
    ap.add_argument("--resume", metavar="DIR", default=None,
                    help="reuse models already finished in DIR, run only the rest")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cases = [c for c in CASES if c[0] in args.models]
    if args.with_m6:
        cases.append(M6_CASE)
    if not cases:
        print("nothing selected; --models expects a subset of %s"
              % ", ".join(c[0] for c in CASES))
        return 2

    if args.resume:
        batch = os.path.abspath(args.resume)
        if not os.path.isdir(batch):
            print("--resume: %s is not a directory" % batch)
            return 2
    else:
        batch = os.path.join(HERE, "Results",
                             "single_instance_rerun_" + time.strftime("%Y%m%d_%H%M%S"))

    print("=" * 68)
    print("  instance : %s   (same machine, same session)" % TS)
    print("  models   : %s" % ", ".join(c[0] for c in cases))
    print("  output   : %s" % batch)
    print("=" * 68, flush=True)
    if args.dry_run:
        print("\n[dry-run] nothing executed.")
        return 0

    os.makedirs(batch, exist_ok=True)

    rows, todo = [], []
    for case in cases:
        done = load_finished(batch, case[0]) if args.resume else None
        if done is not None:
            print("\n  %s -- already finished, reusing %s"
                  % (case[0], done["reused_from"]))
            rows.append(done)
        else:
            todo.append(case)
    if rows:
        _write(batch, rows)
    if not todo:
        print("\nnothing left to run.")
    else:
        print("\n  %d to run: %s" % (len(todo), ", ".join(c[0] for c in todo)))
        try:
            for eid, desc, target, extra in todo:
                rows.append(run_one(eid, desc, target, extra, batch, COMMON))
                _write(batch, rows)          # safe after every model
        except KeyboardInterrupt:
            print("\n[interrupted] keeping what has finished.", flush=True)

    _write(batch, rows)
    print("\nDONE -> %s" % batch)
    print("summary : %s" % os.path.join(batch, "single_instance_rerun_summary.csv"))
    print("tex rows: %s" % os.path.join(batch, "table_rows.tex"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
