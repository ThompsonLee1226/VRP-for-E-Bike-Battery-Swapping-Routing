# -*- coding: utf-8 -*-
"""
Pick the snapshots the benchmark comparison is actually run on.
==============================================================

WHY THE CRITERION CHANGED (2026-10-04)
--------------------------------------
The old criterion was

        d/s = sum_j lambda_j  /  sum_j (n_soon_j + n_normal_j)

i.e. hourly RENTAL demand over the stock on the ground.  It ignores the return
rate rho entirely, and that is a defect with a measurable consequence: it ranks
"demand is high but returns cover it" above "the region is actually bleeding".

  snapshot            d/s (rank)      net outflow/stock (rank)
  2025/11/08 18:00    1.0759 (  1)    0.0118 (217)   <- old #1, drain only 36 of 3257 returns
  2025/10/25 07:00    0.6161 (187)    0.1615 (  1)   <- the real deficit regime

On the old top pick no grid drains, so do-nothing is already near the ceiling and
every dispatch method ties -- which is exactly what the 2026-10-03 benchmark run
showed.  A criterion that selects the regime where the treatment CANNOT matter
cannot be used to rank treatments.

THE CRITERION NOW
-----------------
Net inventory drift over the horizon, per unit of stock on the ground:

        net_ds = sum_j (lambda_j - rho_j)  /  sum_j (n_soon_j + n_normal_j)

Over a 1-hour horizon this is the fraction of the standing stock the region loses
(positive) or gains (negative) if nothing is dispatched.  It is signed on purpose:
clipping at zero would leave ~75 surplus snapshots tied at exactly 0.0000 and make
the low ranks non-reproducible.  Ranking is descending, so the top ranks mean "the
region is losing stock fast" -- the regime where a swap can still change the
outcome -- and the bottom ranks are the surplus/control regime.

`drain` (the clipped positive part) and the old `d/s` are still printed, so every
pick can be read both ways.

USAGE
-----
    python select_snapshots.py                        # top 3 by net drift
    python select_snapshots.py --all-hours            # candidate pool = all 445+
    python select_snapshots.py --ranks 1 2 50 200 400  # explicit 1-based ranks
    python select_snapshots.py --top 5
    python select_snapshots.py --span 5               # 5 evenly spaced by net drift
    python select_snapshots.py --criterion ds         # the old d/s behaviour
    python select_snapshots.py --hours 7 8 9 17 18 19
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRED = os.path.join(ROOT, "2_Training", "Training_Results",
                    "20260727_115935", "prediction_CB_Hurdle.csv")
PEAK_HOURS = [7, 8, 9, 17, 18, 19]

COL_LOW = "low_power_bike_count"
COL_SOON = "soon_low_power_bike_count"
COL_NORM = "normal_power_bike_count"

# The five snapshots used by the formal benchmark experiments, taken at ranks
# 1 / 2 / 50 / 200 / 400 of the net-drift ranking over the all-hours pool.
# Regenerate with:  python select_snapshots.py --all-hours --ranks 1 2 50 200 400
FORMAL_RANKS = [1, 2, 50, 200, 400]


def load():
    df = pd.read_csv(PRED)
    df["datetime"] = pd.to_datetime(df["datetime"])
    rent = "rent_pred" if "rent_pred" in df.columns else "rent"
    ret = "return_pred" if "return_pred" in df.columns else "return"
    df = df.rename(columns={rent: "lam", ret: "rho"})
    return df


def rank(df, hours=None, criterion="net"):
    """Snapshots sorted best-first.  Returns a DataFrame with all the diagnostics."""
    g = df.groupby("datetime").agg(
        demand=("lam", "sum"),
        returns=("rho", "sum"),
        soon=(COL_SOON, "sum"),
        normal=(COL_NORM, "sum"),
        low=(COL_LOW, "sum"),
        grids=("h3", "count")).reset_index()
    g["serviceable"] = g["soon"] + g["normal"]
    g["supply_demand"] = g["serviceable"] / g["demand"].replace(0, float("nan"))
    g["demand_supply"] = g["demand"] / g["serviceable"].replace(0, float("nan"))
    # net hourly drift of the region, signed, and its clipped positive part
    g["net_drift"] = g["demand"] - g["returns"]
    g["drain"] = g["net_drift"].clip(lower=0)
    g["net_ds"] = g["net_drift"] / g["serviceable"].replace(0, float("nan"))
    g["hour"] = g["datetime"].dt.hour
    g = g[g["serviceable"] > 0]
    if hours:
        g = g[g["hour"].isin(hours)]

    key = "net_ds" if criterion == "net" else "demand_supply"
    # datetime as the secondary key keeps the order total and reproducible even if
    # two snapshots ever tie (mergesort = stable, so the input order never leaks in).
    g = g.sort_values([key, "datetime"], ascending=[False, True], kind="mergesort")
    return g.reset_index(drop=True)


def _fmt_table(g, note=""):
    print("%-20s %9s %9s %8s %9s %8s %7s" %
          ("datetime", "net_ds", "d/s", "drain", "returns", "hour", "grids"))
    print("-" * 76)
    for _, r in g.iterrows():
        print("%-20s %9.4f %9.4f %8.0f %9.0f %8d %7d" %
              (str(r["datetime"])[:16], r["net_ds"], r["demand_supply"],
               r["drain"], r["returns"], r["hour"], r["grids"]))
    if note:
        print(note)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--hours", nargs="+", type=int, default=PEAK_HOURS)
    ap.add_argument("--all-hours", action="store_true")
    ap.add_argument("--criterion", choices=["net", "ds"], default="net",
                    help="net = sum(lam-rho)/stock (default); ds = the old lambda/stock")
    ap.add_argument("--ranks", nargs="+", type=int, default=None, metavar="R",
                    help="pick these explicit 1-based ranks of the ranking "
                         "(e.g. --ranks 1 2 50 200 400).  Ranks beyond the pool size "
                         "need --all-hours.")
    ap.add_argument("--span", type=int, default=0, metavar="N",
                    help="pick N snapshots evenly spaced across the ranking")
    args = ap.parse_args()

    df = load()
    hours = None if args.all_hours else args.hours
    g = rank(df, hours, args.criterion)

    print("criterion = %s"
          % ("sum(lam - rho) / stock   (net inventory drift, signed)"
             if args.criterion == "net" else
             "sum(lam) / stock   (the old demand/supply ratio)"))
    print("candidate pool: %d snapshots%s\n"
          % (len(g), "" if args.all_hours else " in hours %s" % (args.hours,)))
    _fmt_table(g.head(max(args.top, 8)))

    if args.span:
        n = len(g)
        k = min(args.span, n)
        idx = [0] if k == 1 else sorted({int(round(i * (n - 1) / (k - 1)))
                                        for i in range(k)})
        picks = list(g.loc[idx, "datetime"])
        print("\nSPAN selection: %d snapshots evenly spaced over the ranking" % len(idx))
        _fmt_table(g.loc[idx])
        print("\nCLI:\n    --snapshots %s"
              % " ".join('"%s"' % d.strftime("%Y/%m/%d %H:%M") for d in picks))
        return 0

    if args.ranks:
        bad = [r for r in args.ranks if r < 1 or r > len(g)]
        if bad:
            print("\n[error] rank(s) %s outside the %d-snapshot pool."
                  % (", ".join(str(b) for b in bad), len(g)))
            print("        The pool is limited to hours %s; pass --all-hours to rank "
                  "over every hour." % (args.hours,))
            return 1
        idx = [r - 1 for r in args.ranks]
        picks = list(g.loc[idx, "datetime"])
        print("\nRANK selection: %s" % ", ".join("rank %d" % r for r in args.ranks))
        _fmt_table(g.loc[idx])
        zero = [r for r, i in zip(args.ranks, idx) if g.loc[i, "drain"] <= 0]
        if zero:
            print("\n[note] rank(s) %s have NO draining grid (net drift <= 0): every "
                  "grid refills\n       on its own there, so do-nothing is already at "
                  "the ceiling.  They are a\n       CONTROL, not a test of skill -- "
                  "report them as such."
                  % ", ".join(str(z) for z in zero))
        print("\nCLI:")
        print("    --snapshots %s"
              % " ".join('"%s"' % d.strftime("%Y/%m/%d %H:%M") for d in picks))
        print("\nDEFAULT_SNAPSHOTS = [%s]"
              % ", ".join('"%s"' % d.strftime("%Y/%m/%d %H:%M") for d in picks))
        return 0

    picks = [d.strftime("%Y/%m/%d %H:%M") for d in g.head(args.top)["datetime"]]
    print("\nSELECTED (top %d by %s):" % (args.top, args.criterion))
    print("    DEFAULT_SNAPSHOTS = [%s]" % ", ".join('"%s"' % p for p in picks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
