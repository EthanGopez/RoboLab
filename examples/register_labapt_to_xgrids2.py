#!/usr/bin/env python3
"""Register the LabApt 2-D vector map to the XGRIDS2 reconstructed mesh.

The output transform maps LabApt navigation coordinates into the Z-up XGRIDS
wrapper coordinates:

    p_xgrids = R(theta) @ p_labapt + [tx, ty]
    yaw_xgrids = wrap(yaw_labapt + theta)

Registration uses horizontal triangle/plane intersections, so the Gaussian
appearance model is never mistaken for metric collision geometry.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, optimize, signal, spatial


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OBJ = REPO / "assets/scenes/xgrids2_lab/forward_compatible/mesh-files/UT_Robtics_lab_int.obj"
DEFAULT_VECTOR = REPO / "assets/scenes/xgrids2_lab/labapt_reference/LabApt.vectormap.txt"
DEFAULT_OUT = REPO / "output/xgrids2_lab/labapt_registration"
FLOOR_SHIFT_Z = 1.075
SEMANTIC_POSES = {
    "kitchen_sink": (6.40, -10.88, 1.70),
    "microwave": (7.52, -10.182, 1.58),
    "kitchen_shelf": (9.78, -10.81, 1.66),
    "living_room": (7.59, -22.69, 0.04),
    "snack_table": (10.30, -16.55, -1.57),
    "study_desk": (9.85, -13.44, 0.15),
    "dining_table": (7.35, -15.54, 1.57),
}


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only positions and triangular faces from the scan OBJ."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif line.startswith("f "):
                ids = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                for index in range(1, len(ids) - 1):
                    faces.append([ids[0], ids[index], ids[index + 1]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def horizontal_slice_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    z_world: float,
    spacing: float,
) -> np.ndarray:
    """Intersect all triangles with one Z plane and sample the resulting segments."""
    z_source = z_world - FLOOR_SHIFT_Z
    triangles = vertices[faces]
    low = triangles[:, :, 2].min(axis=1)
    high = triangles[:, :, 2].max(axis=1)
    triangles = triangles[(low <= z_source) & (high >= z_source)]
    segments: list[np.ndarray] = []
    for triangle in triangles:
        hits: list[np.ndarray] = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            a, b = triangle[first], triangle[second]
            da, db = a[2] - z_source, b[2] - z_source
            if abs(da) < 1e-7 and abs(db) < 1e-7:
                hits.extend((a[:2], b[:2]))
            elif da * db <= 0.0 and abs(a[2] - b[2]) > 1e-7:
                alpha = (z_source - a[2]) / (b[2] - a[2])
                if -1e-6 <= alpha <= 1.0 + 1e-6:
                    hits.append(a[:2] + alpha * (b[:2] - a[:2]))
        if len(hits) < 2:
            continue
        unique = np.unique(np.round(np.asarray(hits), 5), axis=0)
        if len(unique) >= 2:
            distances = np.linalg.norm(unique[None, :, :] - unique[:, None, :], axis=2)
            start, end = np.unravel_index(np.argmax(distances), distances.shape)
            segments.append(np.stack((unique[start], unique[end])))
    return sample_segments(np.asarray(segments), spacing)


def sample_segments(segments: np.ndarray, spacing: float) -> np.ndarray:
    """Convert line segments to a grid-deduplicated point cloud."""
    chunks: list[np.ndarray] = []
    for start, end in segments:
        count = max(2, int(math.ceil(np.linalg.norm(end - start) / spacing)) + 1)
        chunks.append(np.linspace(start, end, count))
    if not chunks:
        return np.empty((0, 2), dtype=np.float32)
    points = np.concatenate(chunks).astype(np.float32)
    cells = np.round(points / spacing).astype(np.int32)
    _, unique = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(unique)]


def load_vector_points(path: Path, spacing: float) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",").reshape(-1, 4)
    return sample_segments(raw.reshape(-1, 2, 2), spacing)


def rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray(((c, -s), (s, c)), dtype=np.float64)


def rasterize(points: np.ndarray, lower: np.ndarray, upper: np.ndarray, resolution: float) -> np.ndarray:
    shape_xy = np.ceil((upper - lower) / resolution).astype(int) + 1
    image = np.zeros((shape_xy[1], shape_xy[0]), dtype=np.float32)
    cells = np.round((points - lower) / resolution).astype(int)
    valid = (
        (cells[:, 0] >= 0) & (cells[:, 0] < shape_xy[0])
        & (cells[:, 1] >= 0) & (cells[:, 1] < shape_xy[1])
    )
    image[cells[valid, 1], cells[valid, 0]] = 1.0
    return image


def fft_translation(scan: np.ndarray, rotated_map: np.ndarray, resolution: float) -> tuple[np.ndarray, float]:
    """Find the best translation for a fixed rotation by raster correlation."""
    scan_min, scan_max = scan.min(axis=0), scan.max(axis=0)
    map_min, map_max = rotated_map.min(axis=0), rotated_map.max(axis=0)
    scan_image = rasterize(scan, scan_min, scan_max, resolution)
    map_image = rasterize(rotated_map, map_min, map_max, resolution)
    scan_image = ndimage.gaussian_filter(ndimage.binary_dilation(scan_image, iterations=2).astype(float), 1.0)
    map_image = ndimage.gaussian_filter(ndimage.binary_dilation(map_image, iterations=1).astype(float), 1.0)
    correlation = signal.fftconvolve(scan_image, map_image[::-1, ::-1], mode="full")
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift_cells = np.asarray((peak_x - map_image.shape[1] + 1, peak_y - map_image.shape[0] + 1))
    translation = scan_min + shift_cells * resolution - map_min
    score = float(correlation[peak_y, peak_x]) / max(1.0, float(map_image.sum()))
    return translation, score


def robust_score(transformed: np.ndarray, scan_tree: spatial.cKDTree) -> tuple[float, float, float]:
    distances, _ = scan_tree.query(transformed, k=1, workers=-1)
    return (
        float(np.mean(distances < 0.24)),
        float(np.median(distances)),
        float(np.quantile(distances, 0.75)),
    )


def refine_transform(
    map_points: np.ndarray,
    scan_points: np.ndarray,
    initial: tuple[float, float, float],
) -> tuple[float, float, float]:
    tree = spatial.cKDTree(scan_points)
    sample = map_points[:: max(1, len(map_points) // 12000)]

    def objective(parameters: np.ndarray) -> float:
        theta, tx, ty = parameters
        transformed = sample @ rotation(theta).T + (tx, ty)
        distances, _ = tree.query(transformed, k=1, workers=-1)
        return float(np.mean(np.minimum(distances, 0.50) ** 2))

    theta, tx, ty = initial
    bounds = ((theta - math.radians(4), theta + math.radians(4)), (tx - 0.8, tx + 0.8), (ty - 0.8, ty + 0.8))
    result = optimize.differential_evolution(objective, bounds, seed=7, popsize=10, maxiter=30, polish=True)
    return tuple(float(value) for value in result.x)


def transform_poses(theta: float, tx: float, ty: float) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    matrix = rotation(theta)
    for name, (x, y, yaw) in SEMANTIC_POSES.items():
        px, py = matrix @ np.asarray((x, y)) + np.asarray((tx, ty))
        new_yaw = math.atan2(math.sin(yaw + theta), math.cos(yaw + theta))
        result[name] = {"x": float(px), "y": float(py), "yaw": float(new_yaw)}
    return result


def plot_overlay(
    scan: np.ndarray,
    mapped: np.ndarray,
    poses: dict[str, dict[str, float]],
    path: Path,
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(13, 11), constrained_layout=True)
    axis.scatter(scan[:, 0], scan[:, 1], s=0.25, c="#38434f", alpha=0.38, label="XGRIDS mesh slices")
    axis.scatter(mapped[:, 0], mapped[:, 1], s=0.35, c="#e45826", alpha=0.50, label="LabApt vector map (aligned)")
    for name, pose in poses.items():
        x, y, yaw = pose["x"], pose["y"], pose["yaw"]
        axis.arrow(x, y, 0.65 * math.cos(yaw), 0.65 * math.sin(yaw), width=0.035, color="#087e8b")
        axis.text(x + 0.12, y + 0.12, name, fontsize=8, color="#07545c")
    axis.set_aspect("equal")
    axis.set_xlabel("XGRIDS X (m)")
    axis.set_ylabel("XGRIDS Y (m)")
    axis.set_title(title)
    axis.grid(alpha=0.16)
    axis.legend(loc="upper right")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=Path, default=DEFAULT_OBJ)
    parser.add_argument("--vector-map", type=Path, default=DEFAULT_VECTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--slice-heights", type=float, nargs="+", default=(0.25, 0.45, 0.75, 1.10, 1.55))
    parser.add_argument("--spacing", type=float, default=0.08)
    parser.add_argument("--angle-step-deg", type=float, default=5.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.obj}")
    vertices, faces = load_obj(args.obj)
    slices = []
    for height in args.slice_heights:
        points = horizontal_slice_points(vertices, faces, height, args.spacing)
        print(f"slice z={height:.2f}: {len(points)} points")
        slices.append(points)
    scan = np.concatenate(slices)
    scan_cells = np.round(scan / args.spacing).astype(np.int32)
    _, unique = np.unique(scan_cells, axis=0, return_index=True)
    scan = scan[np.sort(unique)]
    map_points = load_vector_points(args.vector_map, args.spacing)
    print(f"scan={len(scan)} unique points, LabApt={len(map_points)} points")

    candidates = []
    for degrees in np.arange(-180.0, 180.0, args.angle_step_deg):
        theta = math.radians(float(degrees))
        rotated = map_points @ rotation(theta).T
        translation, fft_score = fft_translation(scan, rotated, args.spacing)
        candidates.append((fft_score, theta, float(translation[0]), float(translation[1])))
    candidates.sort(reverse=True)
    tree = spatial.cKDTree(scan)
    checked = []
    for fft_score, theta, tx, ty in candidates[:8]:
        refined = refine_transform(map_points, scan, (theta, tx, ty))
        transformed = map_points @ rotation(refined[0]).T + refined[1:]
        inliers, median, q75 = robust_score(transformed, tree)
        checked.append((inliers, -median, -q75, fft_score, *refined))
        print(f"candidate theta={math.degrees(refined[0]):7.2f} tx={refined[1]:7.2f} ty={refined[2]:7.2f} inliers={inliers:.3f} median={median:.3f}")
    checked.sort(reverse=True)
    inliers, neg_median, neg_q75, fft_score, theta, tx, ty = checked[0]
    mapped = map_points @ rotation(theta).T + (tx, ty)
    poses = transform_poses(theta, tx, ty)

    report = {
        "convention": "p_xgrids = R(theta_rad) @ p_labapt + [tx_m, ty_m]; yaw_xgrids = wrap(yaw_labapt + theta_rad)",
        "theta_rad": theta,
        "theta_deg": math.degrees(theta),
        "tx_m": tx,
        "ty_m": ty,
        "floor_shift_z_m": FLOOR_SHIFT_Z,
        "slice_heights_z_m": list(args.slice_heights),
        "map_spacing_m": args.spacing,
        "quality": {"fraction_within_0.24_m": inliers, "median_nearest_m": -neg_median, "q75_nearest_m": -neg_q75, "fft_score": fft_score},
        "semantic_poses_xgrids": poses,
        "inputs": {"mesh_obj": str(args.obj), "labapt_vector_map": str(args.vector_map)},
    }
    report_path = args.output_dir / "labapt_to_xgrids2_transform.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    plot_overlay(scan, mapped, poses, args.output_dir / "labapt_xgrids_overlay.png", f"LabApt → XGRIDS2: theta={math.degrees(theta):.2f}°, t=({tx:.2f}, {ty:.2f}) m")
    np.savez_compressed(args.output_dir / "registration_geometry.npz", scan_points=scan, labapt_points=map_points, aligned_labapt_points=mapped)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
