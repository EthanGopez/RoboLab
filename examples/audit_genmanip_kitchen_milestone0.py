# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure task surfaces and staging clearance in the authored EBench kitchen.

This is a deterministic Milestone-0 audit.  It reads the authored USD geometry,
extracts upward-facing horizontal mesh triangles, rasterizes their top heights,
and writes a machine-readable report plus a top-down diagnostic image.  It does
not add collision to the visual scene.
"""

import argparse
import json
import os
from collections import deque

import cv2
import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="output/genmanip_kitchen/milestone0_audit")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

from pxr import Gf, Usd, UsdGeom  # noqa: E402

from robolab.constants import PACKAGE_DIR  # noqa: E402


KITCHEN_USD = os.path.join(
    PACKAGE_DIR,
    "assets",
    "third_party",
    "ebench",
    "assets",
    "scene_usds",
    "ebench",
    "long_horizon",
    "scene_dishwasher",
    "scene.usd",
)

GRID_RESOLUTION = 0.02
XY_BOUNDS = (-1.0, 2.5, -2.5, 1.5)
WORK_SURFACE_Z = (0.70, 1.20)
UP_NORMAL_MIN = 0.985


def _triangles_from_stage(stage: Usd.Stage) -> list[dict]:
    """Return upward-facing triangles in world coordinates."""

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    triangles = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        transform = xform_cache.GetLocalToWorldTransform(prim)
        world_points = np.asarray(
            [tuple(transform.Transform(Gf.Vec3d(point))) for point in points], dtype=np.float64
        )
        cursor = 0
        for count in counts:
            face = indices[cursor : cursor + count]
            cursor += count
            if count < 3:
                continue
            anchor = world_points[face[0]]
            for offset in range(1, count - 1):
                tri = np.stack((anchor, world_points[face[offset]], world_points[face[offset + 1]]))
                normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                norm = np.linalg.norm(normal)
                if norm < 1.0e-10:
                    continue
                normal /= norm
                if normal[2] < UP_NORMAL_MIN:
                    continue
                z = float(np.mean(tri[:, 2]))
                if not WORK_SURFACE_Z[0] <= z <= WORK_SURFACE_Z[1]:
                    continue
                triangles.append({"xy": tri[:, :2], "z": z, "path": str(prim.GetPath())})
    return triangles


def _rasterize(triangles: list[dict]):
    xmin, xmax, ymin, ymax = XY_BOUNDS
    width = int(np.ceil((xmax - xmin) / GRID_RESOLUTION)) + 1
    height = int(np.ceil((ymax - ymin) / GRID_RESOLUTION)) + 1
    top = np.full((height, width), np.nan, dtype=np.float32)
    owner = np.full((height, width), -1, dtype=np.int32)
    for tri_index, item in enumerate(triangles):
        polygon = np.empty((3, 2), dtype=np.int32)
        polygon[:, 0] = np.rint((item["xy"][:, 0] - xmin) / GRID_RESOLUTION).astype(np.int32)
        polygon[:, 1] = np.rint((ymax - item["xy"][:, 1]) / GRID_RESOLUTION).astype(np.int32)
        mask = np.zeros_like(top, dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 1)
        update = (mask > 0) & (np.isnan(top) | (item["z"] > top))
        top[update] = item["z"]
        owner[update] = tri_index
    return top, owner


def _components(top: np.ndarray) -> list[dict]:
    valid = np.isfinite(top)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(valid.astype(np.uint8), 8)
    xmin, xmax, ymin, ymax = XY_BOUNDS
    output = []
    for label in range(1, count):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < 20:
            continue
        ys, xs = np.where(labels == label)
        x0 = xmin + xs.min() * GRID_RESOLUTION
        x1 = xmin + xs.max() * GRID_RESOLUTION
        y1 = ymax - ys.min() * GRID_RESOLUTION
        y0 = ymax - ys.max() * GRID_RESOLUTION
        output.append(
            {
                "label": int(label),
                "grid_cells": area_px,
                "area_m2": round(area_px * GRID_RESOLUTION**2, 4),
                "bounds_xy": [round(x0, 4), round(x1, 4), round(y0, 4), round(y1, 4)],
                "z_min": round(float(np.nanmin(top[labels == label])), 4),
                "z_max": round(float(np.nanmax(top[labels == label])), 4),
                "centroid_xy": [
                    round(xmin + centroids[label, 0] * GRID_RESOLUTION, 4),
                    round(ymax - centroids[label, 1] * GRID_RESOLUTION, 4),
                ],
            }
        )
    return sorted(output, key=lambda item: item["area_m2"], reverse=True)


def _write_heightmap(top: np.ndarray, components: list[dict], output_path: str) -> None:
    image = np.full((*top.shape, 3), 245, dtype=np.uint8)
    finite = np.isfinite(top)
    normalized = np.zeros_like(top, dtype=np.uint8)
    normalized[finite] = np.clip((top[finite] - WORK_SURFACE_Z[0]) * 255.0 / (WORK_SURFACE_Z[1] - WORK_SURFACE_Z[0]), 0, 255)
    image[finite] = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)[finite]
    xmin, xmax, ymin, ymax = XY_BOUNDS
    for index, component in enumerate(components[:12]):
        x, y = component["centroid_xy"]
        px = int(round((x - xmin) / GRID_RESOLUTION))
        py = int(round((ymax - y) / GRID_RESOLUTION))
        cv2.putText(image, str(index), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(image, str(index), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(output_path, image)


def main() -> None:
    stage = Usd.Stage.Open(KITCHEN_USD, load=Usd.Stage.LoadAll)
    triangles = _triangles_from_stage(stage)
    top, owner = _rasterize(triangles)
    components = _components(top)
    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "kitchen_usd": KITCHEN_USD,
        "grid_resolution_m": GRID_RESOLUTION,
        "xy_bounds": XY_BOUNDS,
        "upward_triangle_count": len(triangles),
        "surface_components": components,
    }
    report_path = os.path.join(output_dir, "surface_components.json")
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    image_path = os.path.join(output_dir, "work_surface_heightmap.png")
    _write_heightmap(top, components, image_path)
    print(json.dumps(report, indent=2))
    print(f"Saved report: {report_path}")
    print(f"Saved heightmap: {image_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
