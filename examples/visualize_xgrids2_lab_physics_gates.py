#!/usr/bin/env python3
"""Create deterministic 2-D audit views for XGRIDS2 physics gates 1 and 3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robolab.constants import PACKAGE_DIR
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
    COLLISION_SEGMENTS,
    DEFAULT_REQUIRED_LINE_CLEARANCE_M,
    PHYSICS_REGION,
    PROXY_WIDTH_M,
    RAW_STAGING_POSES,
    STAGING_POSES,
    build_navigation_grid,
    planned_route,
)


DEFAULT_OUTPUT = os.path.join(PACKAGE_DIR, "output/xgrids2_lab/physics_gates")


def _base_axis(title: str):
    figure, axis = plt.subplots(figsize=(12, 9), constrained_layout=True)
    axis.set_aspect("equal")
    axis.set_xlim(PHYSICS_REGION.xmin - 0.4, PHYSICS_REGION.xmax + 0.4)
    axis.set_ylim(PHYSICS_REGION.ymin - 0.4, PHYSICS_REGION.ymax + 0.4)
    axis.set_xlabel("XGRIDS X (m)")
    axis.set_ylabel("XGRIDS Y (m)")
    axis.set_title(title)
    axis.grid(alpha=0.15)
    return figure, axis


def _draw_pose(axis, name: str, pose, color: str, marker: str, label: str | None = None):
    x, y, _, yaw = pose
    axis.scatter([x], [y], s=75, marker=marker, color=color, zorder=8, label=label)
    axis.arrow(
        x,
        y,
        0.5 * np.cos(yaw),
        0.5 * np.sin(yaw),
        width=0.025,
        color=color,
        zorder=7,
        length_includes_head=True,
    )
    axis.text(x + 0.10, y + 0.10, name, fontsize=9, color=color, zorder=9)


def gate1(path: str) -> None:
    figure, axis = _base_axis("Gate 1 — route-first static collision overlay")
    axis.add_patch(
        Rectangle(
            (PHYSICS_REGION.xmin, PHYSICS_REGION.ymin),
            *PHYSICS_REGION.size,
            facecolor="#83c57c",
            edgecolor="#327a31",
            alpha=0.18,
            linewidth=2,
            label="clean floor slab",
        )
    )
    lines = [[segment.start_xy, segment.end_xy] for segment in COLLISION_SEGMENTS]
    collection = LineCollection(
        lines,
        colors="#e87527",
        linewidths=max(1.0, PROXY_WIDTH_M * 35.0),
        alpha=0.85,
        label=f"{len(lines)} static map proxies",
    )
    axis.add_collection(collection)
    for name, pose in RAW_STAGING_POSES.items():
        _draw_pose(axis, f"raw {name}", pose, "#9a348e", "x", "raw LabApt staging" if name == "dining_table" else None)
    for name, pose in STAGING_POSES.items():
        _draw_pose(axis, f"physics {name}", pose, "#087e8b", "o", "physics-safe staging" if name == "dining_table" else None)
    axis.legend(loc="upper right")
    axis.text(
        0.015,
        0.02,
        "Visual scan remains non-colliding; orange centerlines become thin vertical cuboids in PhysX.",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def gate2(path: str, results_path: str) -> None:
    with open(results_path, "r", encoding="utf-8") as stream:
        results = json.load(stream)
    figure, axis = _base_axis("Gate 2 — clean-floor dynamic drop validation")
    axis.add_patch(
        Rectangle(
            (PHYSICS_REGION.xmin, PHYSICS_REGION.ymin),
            *PHYSICS_REGION.size,
            facecolor="#83c57c",
            edgecolor="#327a31",
            alpha=0.22,
            linewidth=2,
            label="collision-backed floor slab, top Z=0",
        )
    )
    for record in results["probes"]:
        x, y, z = record["position_xyz_m"]
        color = "#0a9f42" if record["passed"] else "#cc2936"
        axis.scatter([x], [y], s=135, color=color, edgecolors="white", linewidths=1.4, zorder=5)
        axis.text(x + 0.07, y + 0.07, f"{record['index']}: z={z:.3f}", fontsize=8, color=color)
    status = "PASS" if results["passed"] else "FAIL"
    axis.legend(loc="upper right")
    axis.text(
        0.015,
        0.02,
        f"{status}: {sum(item['passed'] for item in results['probes'])}/{len(results['probes'])} probes settled; scan collision disabled; green probes use the clean slab only.",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def gate3(path: str) -> None:
    grid = build_navigation_grid()
    route = planned_route()
    figure, axis = _base_axis("Gate 3 — mobile-Kinova footprint inflation and connected route")
    extent = (
        grid.xs[0] - 0.5 * grid.resolution,
        grid.xs[-1] + 0.5 * grid.resolution,
        grid.ys[0] - 0.5 * grid.resolution,
        grid.ys[-1] + 0.5 * grid.resolution,
    )
    axis.imshow(
        np.where(grid.free, 1.0, 0.0),
        origin="lower",
        extent=extent,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        alpha=0.42,
        interpolation="nearest",
    )
    lines = [[segment.start_xy, segment.end_xy] for segment in COLLISION_SEGMENTS]
    axis.add_collection(LineCollection(lines, colors="#343a40", linewidths=1.0, alpha=0.55))
    axis.plot(route[:, 0], route[:, 1], color="#1769aa", linewidth=3.0, label="A* base-center path")
    for name, pose in STAGING_POSES.items():
        _draw_pose(axis, name, pose, "#087e8b", "o")
    axis.legend(loc="upper right")
    axis.text(
        0.015,
        0.02,
        f"Green = free after {DEFAULT_REQUIRED_LINE_CLEARANCE_M:.2f} m line clearance; red = blocked/boundary. Route length {np.linalg.norm(np.diff(route, axis=0), axis=1).sum():.2f} m.",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.gate in ("1", "all"):
        output = os.path.join(args.output_dir, "gate1_collision_overlay.png")
        gate1(output)
        print(output)
    if args.gate in ("2", "all"):
        output = os.path.join(args.output_dir, "gate2_floor_drop_summary.png")
        results = os.path.join(args.output_dir, "gate2_floor_drop_results.json")
        gate2(output, results)
        print(output)
    if args.gate in ("3", "all"):
        output = os.path.join(args.output_dir, "gate3_inflated_navigation.png")
        gate3(output)
        print(output)


if __name__ == "__main__":
    main()
