# -*- coding: utf-8 -*-
"""Produce the paper-ready cross-time-point table (all / daytime / night)."""
import os, sys, glob, csv, statistics as st
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

batch = sorted(glob.glob(os.path.join(HERE, "Results", "temporal_rerun_*")))[-1]
csvs = sorted(glob.glob(os.path.join(batch, "temporal_rebuilt_*.csv")))
df = pd.read_csv(csvs[-1])
print("source:", os.path.basename(csvs[-1]))

df["dt"] = pd.to_datetime(df["datetime"], format="%Y%m%d_%H%M")
df["hour"] = df.dt.dt.hour
df["period"] = df.hour.apply(lambda h: "Daytime" if 7 <= h <= 22 else "Night")
df["ok"] = df.objective.notna()

print("\nsolved: %d / %d" % (df.ok.sum(), len(df)))
print(df.groupby("period").ok.agg(["sum", "count"]).to_string())

def blk(d):
    o = d.objective.dropna()
    t = d.elapsed_seconds.dropna()
    g = d.MIPGap.dropna()
    v = d.num_visited.dropna()
    s = d.total_swaps.dropna()
    m = d.makespan_hrs.dropna()
    def ms(x):
        return "%.3f $\\pm$ %.3f" % (x.mean(), x.std(ddof=1)) if len(x) > 1 else "%.3f" % x.mean()
    return dict(n=len(d), nobj=len(o),
                obj=ms(o), t=ms(t), gap="%.3f" % g.mean() if len(g) else "-",
                gaps="< %.1f" % (g.max() if len(g) else 0),
                visit=ms(v), swaps=ms(s), makespan=ms(m),
                cv=(t.std(ddof=1) / t.mean()) if len(t) > 1 else float("nan"))

rows = {"All": blk(df[df.ok]),
        "Daytime": blk(df[df.ok & (df.period == "Daytime")]),
        "Night": blk(df[df.ok & (df.period == "Night")])}

print("\n=== PAPER TABLE (LaTeX) ===")
hdr = ["Metric", "All (n=%d)" % rows["All"]["n"],
       "Daytime (n=%d)" % rows["Daytime"]["n"], "Night (n=%d)" % rows["Night"]["n"]]
print(" & ".join(hdr) + " \\\\")
for key, label in [("obj", "Objective"), ("t", "Solve time (s)"), ("gap", "MIP gap (\\%)"),
                   ("visit", "Visited grids"), ("swaps", "Swap quantity"),
                   ("makespan", "Makespan (h)")]:
    print("%s & %s & %s & %s \\\\" % (label, rows["All"][key],
                                      rows["Daytime"][key], rows["Night"][key]))
print("\nsuccess rate: %d/%d = %.0f%%" % (df.ok.sum(), len(df), 100 * df.ok.mean()))
print("solve-time CV (all) : %.3f" % rows["All"]["cv"])
print("solve-time range    : %.1f - %.1f s" % (df.elapsed_seconds.min(), df.elapsed_seconds.max()))
print("objective range     : %.3f - %.3f" % (df.objective.min(), df.objective.max()))
print("max MIP gap         : %.4f%%" % df.MIPGap.max())
print("time-limit hits     : %d" % int((df.elapsed_seconds >= 1199).sum()))

best = df.loc[df.objective.idxmax()]; worst = df.loc[df.objective.idxmin()]
print("best : %s  obj=%.3f" % (best.datetime, best.objective))
print("worst: %s  obj=%.3f" % (worst.datetime, worst.objective))

df.to_csv(os.path.join(batch, "temporal_paper_table.csv"), index=False, encoding="utf-8-sig")
print("\nwrote temporal_paper_table.csv")
