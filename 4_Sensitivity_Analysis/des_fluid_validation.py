#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
流体近似 DES 验证 — 生灭过程 vs ODE 平滑轨迹
=============================================================================

目的: 定量验证流体(平均场)近似在"低库存 + 净流出"区间的系统性误差。
  流体 ODE 以泊松过程期望值替换随机跳跃, 预测库存线性下降并在
  t0 = N0/(λ-ρ) 确定性地归零; 而真实生灭过程存在方差, 可能更早缺货,
  导致流体近似高估服务效用。本脚本通过 Monte Carlo 离散事件仿真(DES)
  复现"阶梯状真实库存曲线", 与 ODE 平滑曲线重叠对比。

grid 选取规则 (默认):
  1) 0 < N0 = n_soon + n_normal <= N0_MAX(=5)    —— 低库存区间(方差显著)
  2) λ_j > ρ_eff                                     —— 净流出(否则永不缺货)
  3) t0 = N0/(λ_j - ρ_eff) < T(=1h)                  —— 缺货发生在规划期内
  优先取 N0 最小(相对方差最大); 次优 t0 靠近中段使对比清晰。

运行方式:
  python 4_Sensitivity_Analysis/des_fluid_validation.py
  python 4_Sensitivity_Analysis/des_fluid_validation.py --datetime "2025/10/23 12:00"
  python 4_Sensitivity_Analysis/des_fluid_validation.py --n0-max 8 --top-k 3 --reps 1000
=============================================================================
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OPTIM_DIR = os.path.join(PROJECT_ROOT, '3_Optimization')
if OPTIM_DIR not in sys.path:
    sys.path.insert(0, OPTIM_DIR)

from Pre_Process import (
    DEFAULT_TARGET_DATETIME,
    DEFAULT_PREDICTION_FILE,
    select_random_datetime,
    list_available_hours,
    prepare_optimize_inputs,
)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results", "des_validation")


def simulate_birth_death(N0, rho_eff, lam, T, rng):
    """Gillespie 生灭过程仿真一次。

    状态 N(t) = 可服务库存(soon+normal)。事件:
      - 到达(率 rho_eff): N += 1
      - 离开(率 lam, 仅 N>0): N -= 1
    返回 (t_events, N_events, t_stockout): t_stockout 为首次 N 归零时刻(或 None)。
    """
    t = 0.0
    N = N0
    t_events = [0.0]
    N_events = [float(N)]
    t_stockout = None

    while t < T:
        rate_arr = rho_eff
        rate_dep = lam if N > 0 else 0.0
        total_rate = rate_arr + rate_dep
        if total_rate <= 1e-12:
            break
        dt = rng.exponential(1.0 / total_rate)
        t += dt
        if t > T:
            break
        if rng.random() < rate_arr / total_rate:
            N += 1
        else:
            N -= 1
        t_events.append(t)
        N_events.append(float(N))
        if N == 0 and t_stockout is None:
            t_stockout = t

    if t_events[-1] < T:
        t_events.append(T)
        N_events.append(float(N))
    return np.asarray(t_events), np.asarray(N_events), t_stockout


def select_candidate_grids(grid_params, T=1.0, n0_max=5.0, top_k=3):
    """按默认规则选取低库存候选 grid。

    到达率使用 rho_eff = ρ × (θ_soon + θ_normal), 与 Grid_Utility 中
    calculate_operational_utility 的 rho_j_pure 一致 (因 theta 已归一化,
    数值上等于 raw ρ)。"""
    candidates = []
    for j, p in grid_params.items():
        n0 = float(p["n_soon"]) + float(p["n_normal"])
        lam = float(p["lam"])
        rho_eff = float(p["rho"]) * (float(p["theta_soon"]) + float(p["theta_normal"]))
        if not (0.0 < n0 <= n0_max):
            continue
        if lam <= rho_eff:
            continue
        t0 = n0 / (lam - rho_eff)
        if t0 >= T:
            continue
        candidates.append({"grid": j, "n0": n0, "lam": lam, "rho_eff": rho_eff, "t0": t0})

    # 优先 N0 最小(相对方差最大); 次优 t0 靠近中段(0.5*T)使对比清晰
    candidates.sort(key=lambda c: (c["n0"], abs(c["t0"] - 0.5 * T)))
    return candidates[:top_k]


def main():
    parser = argparse.ArgumentParser(description="DES 流体近似验证")
    parser.add_argument("--data", type=str, default=DEFAULT_PREDICTION_FILE)
    parser.add_argument("--datetime", type=str, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n0-max", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--list-hours", action="store_true")
    args = parser.parse_args()

    if args.list_hours:
        for h in list_available_hours(file_path=args.data):
            print(h.strftime("%Y/%m/%d %H:%M"))
        return

    if args.random:
        target_datetime = select_random_datetime(file_path=args.data, seed=args.seed)
    else:
        target_datetime = args.datetime or DEFAULT_TARGET_DATETIME

    T = 1.0
    rng = np.random.default_rng(args.seed)

    grids, grid_params, _ = prepare_optimize_inputs(args.data, target_datetime=target_datetime)
    cands = select_candidate_grids(grid_params, T=T, n0_max=args.n0_max, top_k=args.top_k)

    if not cands:
        print(f"[警告] 快照 {target_datetime} 下无 grid 满足默认规则 "
              f"(N0<={args.n0_max} 且 λ>ρ_eff 且 t0<T)。")
        print("        建议: 增大 --n0-max (如 8), 或更换 --datetime。")
        return

    print("=" * 72)
    print(f"  DES 流体近似验证 — {target_datetime}")
    print(f"  入选 grid (按 N0 升序):")
    for c in cands:
        print(f"    grid={c['grid']}  N0={c['n0']:.1f}  λ={c['lam']:.3f}  "
              f"ρ_eff={c['rho_eff']:.3f}  t0(ODE)={c['t0']:.3f}h")
    print("=" * 72)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    n = len(cands)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), squeeze=False)
    summary_rows = []

    for ax_idx, c in enumerate(cands):
        ax = axes[0, ax_idx]
        n0 = c["n0"]
        lam = c["lam"]
        rho_eff = c["rho_eff"]
        t0_ode = c["t0"]

        # ODE 平滑曲线: N(t) = N0 + (rho_eff - lam)*t, 截断于 t0
        t_ode = np.linspace(0.0, t0_ode, 200)
        n_ode = n0 + (rho_eff - lam) * t_ode

        # Monte Carlo 仿真
        t_grid = np.linspace(0.0, T, 500)
        stockout_times = []
        horizons = []
        N_samples = []
        sample_paths = []
        for r in range(args.reps):
            te, ne, ts = simulate_birth_death(n0, rho_eff, lam, T, rng)
            stockout_times.append(ts)
            horizons.append(ts if ts is not None else T)
            N_samples.append(np.interp(t_grid, te, ne))
            if r < 8:
                sample_paths.append((te, ne))

        N_samples = np.array(N_samples)
        N_mean = N_samples.mean(axis=0)
        N_low = np.percentile(N_samples, 5, axis=0)
        N_high = np.percentile(N_samples, 95, axis=0)

        # 少数样本阶梯路径
        for te, ne in sample_paths:
            ax.step(te, ne, where='post', alpha=0.22, lw=0.8, color='tab:blue')

        ax.plot(t_ode, n_ode, color='tab:red', lw=2.0, label='ODE (流体近似)')
        ax.plot(t_grid, N_mean, color='tab:green', lw=2.0, label='DES 经验均值')
        ax.fill_between(t_grid, N_low, N_high, alpha=0.15, color='tab:green', label='5%–95% 分位带')
        ax.axvline(t0_ode, color='tab:red', ls='--', lw=1.0, alpha=0.7)
        ax.set_xlabel('时间 t (h)')
        ax.set_ylabel('可服务库存 N(t)')
        ax.set_title(f"grid {c['grid']}\nN0={n0:.1f}, λ={lam:.3f}, ρ={rho_eff:.3f}")
        ax.legend(fontsize=7)
        ax.set_xlim(0, T)
        ax.set_ylim(bottom=0)

        # 统计
        early_prob = float(np.mean(
            [1.0 if ts is not None and ts < t0_ode - 1e-9 else 0.0 for ts in stockout_times]
        ))
        stockout_cond = [ts for ts in stockout_times if ts is not None]
        mean_stockout = float(np.mean(stockout_cond)) if stockout_cond else float('nan')
        mean_horizon = float(np.mean(horizons))
        rel_gap = 1.0 - mean_horizon / t0_ode
        summary_rows.append({
            "grid": c["grid"], "n0": n0, "lam": lam, "rho_eff": rho_eff,
            "t0_ode_h": round(t0_ode, 4),
            "early_stockout_prob": round(early_prob, 4),
            "mean_stockout_h": round(mean_stockout, 4),
            "mean_service_horizon_h": round(mean_horizon, 4),
            "horizon_rel_gap": round(rel_gap, 4),
        })

    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "des_vs_ode.png")
    fig.savefig(png_path, dpi=150)
    print(f"\n  图已保存: {png_path}")

    # 汇总统计
    print("\n  汇总统计:")
    header = (f"{'grid':>18s} {'N0':>5s} {'λ':>7s} {'ρ_eff':>7s} {'t0(ODE)':>8s} "
              f"{'P(提前缺货)':>12s} {'均值缺货h':>10s} {'服务期收缩':>10s}")
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(f"{r['grid']:>18s} {r['n0']:>5.1f} {r['lam']:>7.3f} {r['rho_eff']:>7.3f} "
              f"{r['t0_ode_h']:>8.3f} {r['early_stockout_prob']:>12.3f} "
              f"{r['mean_stockout_h']:>10.3f} {r['horizon_rel_gap']:>10.3f}")
    print("\n  解读: '服务期收缩' > 0 表示流体近似高估了可服务时长(即高估服务效用)。")


if __name__ == "__main__":
    main()
