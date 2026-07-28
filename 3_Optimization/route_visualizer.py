#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Route Visualizer — H3 Grid Route Map
=============================================================================

Provides unified route visualization for four optimization architectures
(STGraph, SOS2-PLA, Delta-PLA, Delta-MCF).

Features:
  - Plots the optimized route on real H3 hexagon geographic coordinates
  - Dark-to-light color gradient indicates route direction (start dark, end light)
  - Direction arrows placed along path segments
  - Visited nodes numbered in visit order with swap quantities
  - Depot (charging station) shown as a distinctive marker
  - All candidate grids shown as light-gray background

Usage:
  from route_visualizer import visualize_route
  visualize_route(result, grid_coords, snapshot_df, output_dir, **kwargs)
=============================================================================
"""

from __future__ import annotations

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


# ─────────────────────────────────────────────────────────────────────
# Custom dark-to-light directional colormap
# ─────────────────────────────────────────────────────────────────────
# Deep purple -> blue -> teal -> bright orange -> pale gold
_ROUTE_COLORS = [
    (0.20, 0.10, 0.35),   # deep purple (start)
    (0.15, 0.35, 0.65),   # deep blue
    (0.10, 0.55, 0.70),   # teal
    (0.90, 0.55, 0.20),   # bright orange
    (0.98, 0.85, 0.45),   # pale gold (end)
]
_ROUTE_CMAP = LinearSegmentedColormap.from_list("route_direction", _ROUTE_COLORS, N=256)


def _get_route_segments(route, grid_coords, depot_coords):
    """Extract ordered line segments from a route dict.

    Returns
    -------
    segments : list of ((lon0, lat0), (lon1, lat1))
        Ordered path segments.
    visited_h3 : list of str
        H3 grid IDs in visit order (excluding depot).
    """
    segments = []
    visited_h3 = []

    # Build spatial node sequence: depot -> grid_1 -> grid_2 -> ... -> depot
    spatial_seq = [("DEPOT", depot_coords)]
    for step in route:
        gid = step.get("grid", step.get("grid_id", ""))
        if gid == "DEPOT":
            spatial_seq.append(("DEPOT", depot_coords))
        elif gid in grid_coords:
            spatial_seq.append((gid, grid_coords[gid]))
            visited_h3.append(gid)

    # Ensure final return to depot
    if spatial_seq[-1][0] != "DEPOT":
        spatial_seq.append(("DEPOT", depot_coords))

    # Build segments as (lon, lat) pairs for matplotlib (x=lon, y=lat)
    for k in range(len(spatial_seq) - 1):
        _, (lat0, lon0) = spatial_seq[k]
        _, (lat1, lon1) = spatial_seq[k + 1]
        segments.append(((lon0, lat0), (lon1, lat1)))

    return segments, visited_h3


def visualize_route(
    result: dict,
    grid_coords: dict,
    snapshot_df,
    output_dir: str,
    *,
    depot_coords: tuple = None,
    experiment_id: str = "",
    instance_name: str = "",
    vehicle_speed_kmh: float = 30.0,
    swap_time_c: float = 0.02,
    C_max: int = 20,
    dpi: int = 200,
    figsize: tuple = (14, 12),
) -> str | None:
    """Draw an H3 grid route visualization from optimization results.

    Parameters
    ----------
    result : dict
        Optimizer result dict. Must contain:
          - result["route"]: list[dict] with grid / arrival_time / y_swapped
          - result["objective"]: float
          - result["status"]: str
    grid_coords : dict
        h3_id -> (latitude, longitude).
    snapshot_df : pd.DataFrame
        Snapshot data for background grid distribution.
    output_dir : str
        Output directory for the saved image.
    depot_coords : tuple, optional
        (latitude, longitude) of the depot. Auto-computed if None.
    experiment_id : str
        Experiment group label.
    instance_name : str
        Instance name for the title and filename.
    vehicle_speed_kmh : float
        Vehicle speed for info annotation.
    swap_time_c : float
        Battery swap time for info annotation.
    C_max : int
        Vehicle battery capacity for info annotation.
    dpi : int
        Output image resolution.
    figsize : tuple
        Figure size (width, height) in inches.

    Returns
    -------
    str or None
        Saved image path, or None if no route to draw.
    """
    route = result.get("route", [])
    if not route:
        return None

    # ── Depot coordinates ──────────────────────────────────────────
    if depot_coords is None:
        if grid_coords:
            depot_coords = (
                float(np.mean([c[0] for c in grid_coords.values()])),
                float(np.mean([c[1] for c in grid_coords.values()])),
            )
        else:
            depot_coords = (0.0, 0.0)

    # ── Extract path segments ──────────────────────────────────────
    segments, visited_h3 = _get_route_segments(route, grid_coords, depot_coords)
    n_seg = len(segments)
    if n_seg == 0:
        return None

    # ── Per-node info ──────────────────────────────────────────────
    visited_info = {}
    for step in route:
        gid = step.get("grid", step.get("grid_id", ""))
        if gid != "DEPOT" and gid in grid_coords:
            visited_info[gid] = {
                "arrival_time": step.get("arrival_time", np.nan),
                "y_swapped": step.get("y_swapped", 0),
            }

    # ── Create figure ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # ── 1) Background: all candidate grids (light gray) ────────────
    all_lons = [c[1] for c in grid_coords.values()]
    all_lats = [c[0] for c in grid_coords.values()]
    ax.scatter(all_lons, all_lats, c="#e0e0e02a", s=3, alpha=0.5,
               zorder=0, label='Candidate grids (unvisited)')

    # ── 2) Path segments: dark-to-light gradient ───────────────────
    line_colors = np.linspace(0.0, 1.0, n_seg)
    lc = LineCollection(segments, cmap=_ROUTE_CMAP, array=line_colors,
                        linewidths=2.5, zorder=2, alpha=0.85)
    ax.add_collection(lc)

    # ── 3) Direction arrows (max 8) ────────────────────────────────
    arrow_spacing = max(1, n_seg // 8)
    for k in range(0, n_seg, arrow_spacing):
        (x0, y0), (x1, y1) = segments[k]
        dx, dy = x1 - x0, y1 - y0
        seg_len = np.hypot(dx, dy)
        if seg_len < 1e-6:
            continue
        mx, my = x0 + 0.5 * dx, y0 + 0.5 * dy
        frac = k / max(n_seg - 1, 1)
        color = _ROUTE_CMAP(frac)
        ax.annotate("", xy=(x0 + 0.52 * dx, y0 + 0.52 * dy),
                    xytext=(mx, my),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=1.8, alpha=0.9),
                    zorder=5)

    # ── 4) Visited nodes: gradient-colored markers ─────────────────
    for vi, h3_id in enumerate(visited_h3):
        if h3_id not in grid_coords:
            continue
        lat, lon = grid_coords[h3_id]
        frac = vi / max(len(visited_h3) - 1, 1)
        color = _ROUTE_CMAP(frac)

        ax.scatter(lon, lat, c=[color], s=80, edgecolors='#333333',
                   linewidths=1.2, zorder=4, marker='o')

        # Node sequence number
        ax.annotate(str(vi + 1), (lon, lat),
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=7, fontweight='bold', color='#333333',
                    zorder=6)

    # ── 5) Depot marker ────────────────────────────────────────────
    dep_lat, dep_lon = depot_coords
    ax.scatter(dep_lon, dep_lat, c='#2ecc71', s=250, marker='s',
               edgecolors='#1a7a3a', linewidths=2.0, zorder=5,
               label='Depot')

    # ── 6) Colorbar (direction indicator) ──────────────────────────
    sm = plt.cm.ScalarMappable(cmap=_ROUTE_CMAP,
                               norm=plt.Normalize(0, max(n_seg, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Route direction (dark -> light = start -> end)',
                   fontsize=11, labelpad=10)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Start', 'End'])

    # ── 7) Legend & title ──────────────────────────────────────────
    total_swaps = sum(step.get("y_swapped", 0) for step in route
                      if step.get("grid", "") != "DEPOT")
    makespan = result.get("summary", {}).get("makespan_hrs", np.nan)
    obj = result.get("objective", np.nan)
    status = result.get("status", "UNKNOWN")

    ax.legend(loc='upper right', fontsize=10, framealpha=0.9,
              markerscale=0.8)

    title_lines = [
        f"Route Visualization — {experiment_id} | {instance_name}",
        f"Obj={obj:.4f} | Nodes={len(visited_h3)} | "
        f"Swaps={total_swaps}/{C_max} | Makespan={makespan:.3f}h | "
        f"Status={status}",
    ]
    ax.set_title("\n".join(title_lines), fontsize=13, fontweight='bold',
                 linespacing=1.4)

    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')

    # ── Axis padding ───────────────────────────────────────────────
    if all_lons and all_lats:
        lon_range = max(all_lons) - min(all_lons)
        lat_range = max(all_lats) - min(all_lats)
        pad = 0.08
        ax.set_xlim(min(all_lons) - lon_range * pad,
                    max(all_lons) + lon_range * pad)
        ax.set_ylim(min(all_lats) - lat_range * pad,
                    max(all_lats) + lat_range * pad)

    # ── 8) Save ────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{experiment_id}_{instance_name}_route.png"
    save_path = os.path.join(output_dir, fname)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    return save_path


# =========================================================================
# CLI: standalone test entry
# =========================================================================
if __name__ == "__main__":
    print("route_visualizer module ready.")
    print("Call visualize_route(result, grid_coords, snapshot_df, "
          "output_dir, ...) from main_*.py to generate route maps.")
