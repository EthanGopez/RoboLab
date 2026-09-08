# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render reproducible numbered views of the floor-aligned XGRIDS2 lab scan."""

import argparse
import json
import math
import os
import sys

import cv2  # must be imported before isaaclab
from isaaclab.app import AppLauncher


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from robolab.core.utils.nurec_rendering import (
    configure_plain_nurec_volume,
    require_single_gpu,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir",
    default="output/xgrids2_lab/numbered_views",
    help="Directory for the numbered PNGs, montage, and camera manifest.",
)
parser.add_argument(
    "--settle-frames",
    type=int,
    default=12,
    help="Rendered simulation steps after every camera move.",
)
parser.add_argument(
    "--view-set",
    choices=("numbered", "labapt", "labapt_aligned", "sink_closeup"),
    default="numbered",
    help="Render audit views, raw LabApt poses, or geometrically aligned LabApt poses.",
)
parser.add_argument(
    "--labapt-transform",
    default="output/xgrids2_lab/labapt_registration/labapt_to_xgrids2_transform.json",
    help="Registration JSON produced by register_labapt_to_xgrids2.py.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
require_single_gpu(args_cli)
simulation_app = AppLauncher(args_cli).app
NUREC_RENDER_SETTINGS = configure_plain_nurec_volume()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402


WRAPPER_USD = os.path.join(
    REPO_ROOT, "assets", "scenes", "xgrids2_lab", "xgrids2_lab_zup.usda"
)
SOURCE_USDZ = os.path.join(
    REPO_ROOT,
    "assets",
    "scenes",
    "xgrids2_lab",
    "forward_compatible",
    "usdz-result",
    "mesh-model.usdz",
)
SOURCE_FLOOR_Z_M = -1.075
FLOOR_OFFSET_Z_M = 1.075
RESULT_FLOOR_Z_M = 0.0

# Positions are in the wrapper's corrected Z-up, floor-at-zero coordinate frame.
# The first-person height is 1.625 m: the original successful scan height of
# Z=0.55 m plus the wrapper's +1.075 m floor correction.
NUMBERED_VIEWS = (
    {
        "number": 1,
        "kind": "first_person",
        "slug": "positive_x_worktables",
        "label": "FIRST PERSON +X | DOOR AND WORKTABLES",
        "eye": (0.0, 0.0, 1.625),
        "target": (5.0, 0.0, 1.625),
    },
    {
        "number": 2,
        "kind": "first_person",
        "slug": "negative_y_cubicles",
        "label": "FIRST PERSON -Y | CUBICLES",
        "eye": (0.0, 0.0, 1.625),
        "target": (0.0, -5.0, 1.625),
    },
    {
        "number": 3,
        "kind": "first_person",
        "slug": "south_work_area",
        "label": "FIRST PERSON DIAGONAL | SOUTH WORK AREA",
        "eye": (10.0, -10.0, 1.625),
        "target": (5.0, -5.0, 1.625),
    },
    {
        "number": 4,
        "kind": "first_person",
        "slug": "central_reverse",
        "label": "FIRST PERSON -X | CENTRAL WORK AREA",
        "eye": (5.0, 0.0, 1.625),
        "target": (0.0, 0.0, 1.625),
    },
    {
        "number": 5,
        "kind": "overhead",
        "slug": "origin_zone",
        "label": "OVERHEAD | ORIGIN WORK ZONE",
        "eye": (0.0, 0.0, 3.20),
        # Small XY displacement avoids the look-at singularity of a perfectly
        # vertical camera while remaining visually top-down.
        "target": (0.20, 0.10, 0.0),
    },
    {
        "number": 6,
        "kind": "overhead",
        "slug": "south_zone",
        "label": "OVERHEAD | SOUTH WORK ZONE",
        "eye": (10.0, -10.0, 3.20),
        "target": (10.20, -9.90, 0.0),
    },
)


# Active semantic navigation goals from the cobot computer. Each entry is a
# robot staging pose [x, y, yaw], not an asset centroid.
LABAPT_LANDMARK_POSES = (
    ("kitchen_sink", "KITCHEN SINK", 6.40, -10.88, 1.70),
    ("microwave", "MICROWAVE", 7.52, -10.182, 1.58),
    ("kitchen_shelf", "KITCHEN SHELF", 9.78, -10.81, 1.66),
    ("living_room", "LIVING ROOM", 7.59, -22.69, 0.04),
    ("snack_table", "SNACK TABLE", 10.30, -16.55, -1.57),
    ("study_desk", "STUDY DESK", 9.85, -13.44, 0.15),
    ("dining_table", "DINING TABLE", 7.35, -15.54, 1.57),
)


def _labapt_views() -> tuple[dict, ...]:
    """Convert LabApt [x, y, yaw] navigation goals to forward camera views."""

    views = []
    look_distance_m = 4.0
    camera_z_m = 1.625
    for number, (slug, label, x, y, yaw) in enumerate(LABAPT_LANDMARK_POSES, start=1):
        views.append(
            {
                "number": number,
                "kind": "labapt_alignment",
                "slug": slug,
                "label": f"LABAPT POSE | EXPECTED {label}",
                "eye": (x, y, camera_z_m),
                "target": (
                    x + look_distance_m * math.cos(yaw),
                    y + look_distance_m * math.sin(yaw),
                    camera_z_m,
                ),
                "labapt_pose_xytheta": [x, y, yaw],
            }
        )
    return tuple(views)


def _aligned_labapt_views(transform_path: str) -> tuple[dict, ...]:
    """Transform LabApt staging poses into XGRIDS and aim along robot heading."""

    with open(transform_path, "r", encoding="utf-8") as stream:
        registration = json.load(stream)
    theta = float(registration["theta_rad"])
    tx = float(registration["tx_m"])
    ty = float(registration["ty_m"])
    cosine, sine = math.cos(theta), math.sin(theta)
    views = []
    look_distance_m = 4.0
    camera_z_m = 1.625
    for number, (slug, label, x, y, yaw) in enumerate(LABAPT_LANDMARK_POSES, start=1):
        x_aligned = cosine * x - sine * y + tx
        y_aligned = sine * x + cosine * y + ty
        yaw_aligned = math.atan2(math.sin(yaw + theta), math.cos(yaw + theta))
        views.append(
            {
                "number": number,
                "kind": "labapt_aligned",
                "slug": slug,
                "label": f"ALIGNED LABAPT | EXPECTED {label}",
                "eye": (x_aligned, y_aligned, camera_z_m),
                "target": (
                    x_aligned + look_distance_m * math.cos(yaw_aligned),
                    y_aligned + look_distance_m * math.sin(yaw_aligned),
                    camera_z_m,
                ),
                "labapt_pose_xytheta": [x, y, yaw],
                "xgrids_pose_xytheta": [x_aligned, y_aligned, yaw_aligned],
            }
        )
    return tuple(views)


def _sink_closeup_views(transform_path: str) -> tuple[dict, ...]:
    """Render pitched views around the aligned kitchen-sink staging pose."""

    aligned = next(
        view for view in _aligned_labapt_views(transform_path)
        if view["slug"] == "kitchen_sink"
    )
    x, y, yaw = aligned["xgrids_pose_xytheta"]
    forward = (math.cos(yaw), math.sin(yaw))
    lateral = (-forward[1], forward[0])
    basin_xy = (x + 1.05 * forward[0], y + 1.05 * forward[1])
    specs = (
        ("stage_down", "SINK STAGE PITCHED DOWN", (x, y, 1.45)),
        ("stage_high", "SINK STAGE HIGH", (x, y, 2.10)),
        (
            "stage_left",
            "SINK STAGE LEFT OFFSET",
            (x + 0.30 * lateral[0], y + 0.30 * lateral[1], 1.55),
        ),
        (
            "stage_right",
            "SINK STAGE RIGHT OFFSET",
            (x - 0.30 * lateral[0], y - 0.30 * lateral[1], 1.55),
        ),
    )
    return tuple(
        {
            "number": number,
            "kind": "sink_closeup",
            "slug": slug,
            "label": label,
            "eye": eye,
            "target": (*basin_xy, 0.78),
            "xgrids_pose_xytheta": [x, y, yaw],
            "candidate_basin_xy": list(basin_xy),
        }
        for number, (slug, label, eye) in enumerate(specs, start=1)
    )



class XGRIDS2LabAuditSceneCfg(InteractiveSceneCfg):
    """Floor-aligned XGRIDS2 lab and one movable audit camera."""

    lab = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/xgrids2_lab",
        spawn=sim_utils.UsdFileCfg(usd_path=WRAPPER_USD),
    )
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/audit_camera",
        height=540,
        width=960,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=4.0,
            horizontal_aperture=20.0,
            clipping_range=(0.05, 2000.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 1.625),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )


def _annotate(rgb: np.ndarray, view: dict, valid_depth_fraction: float) -> np.ndarray:
    """Add the stable view number, type, pose, and depth sanity check."""

    frame = np.ascontiguousarray(rgb[:, :, :3].astype(np.uint8))
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, 82), (18, 22, 28), -1)
    cv2.putText(
        frame,
        f"VIEW {view['number']:02d} | {view['label']}",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    pose_text = (
        f"eye={view['eye']}  target={view['target']}  "
        f"floor Z=0  valid depth={100.0 * valid_depth_fraction:.2f}%"
    )
    cv2.putText(
        frame,
        pose_text,
        (24, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (90, 215, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def _save_rgb(path: str, rgb: np.ndarray) -> None:
    if not cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to save image: {path}")


def _make_montage(images: list[np.ndarray], path: str) -> None:
    """Write a two-column montage without changing individual audit frames."""

    tiles = [cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA) for image in images]
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[index : index + 2]) for index in range(0, len(tiles), 2)]
    montage = np.vstack(rows)
    _save_rgb(path, montage)


def main() -> None:
    if not os.path.isfile(SOURCE_USDZ):
        raise FileNotFoundError(f"XGRIDS2 source package not found: {SOURCE_USDZ}")
    if not os.path.isfile(WRAPPER_USD):
        raise FileNotFoundError(f"XGRIDS2 Z-up wrapper not found: {WRAPPER_USD}")

    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1, use_fabric=False)
    )
    scene = InteractiveScene(
        XGRIDS2LabAuditSceneCfg(
            num_envs=1,
            env_spacing=60.0,
            replicate_physics=False,
            lazy_sensor_update=False,
        )
    )
    sim.reset()
    scene.reset()
    camera = scene["camera"]
    rendered_images = []
    records = []

    if args_cli.view_set in {"labapt_aligned", "sink_closeup"}:
        transform_path = os.path.abspath(args_cli.labapt_transform)
        views = (
            _sink_closeup_views(transform_path)
            if args_cli.view_set == "sink_closeup"
            else _aligned_labapt_views(transform_path)
        )
    elif args_cli.view_set == "labapt":
        views = _labapt_views()
    else:
        views = NUMBERED_VIEWS
    for view in views:
        eyes = torch.tensor([view["eye"]], dtype=torch.float32, device=camera.device)
        targets = torch.tensor([view["target"]], dtype=torch.float32, device=camera.device)
        camera.set_world_poses_from_view(eyes=eyes, targets=targets)

        for _ in range(max(1, args_cli.settle_frames)):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
        depth = camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy()
        valid = np.isfinite(depth) & (depth > 0.0)
        valid_fraction = float(np.mean(valid))
        rgb_range = int(np.ptp(rgb))
        if rgb_range < 8 or valid_fraction < 0.50:
            raise RuntimeError(
                f"View {view['number']:02d} failed render validation: "
                f"rgb_range={rgb_range}, valid_depth_fraction={valid_fraction:.6f}"
            )

        rendered = _annotate(rgb, view, valid_fraction)
        filename = f"{view['number']:02d}_{view['kind']}_{view['slug']}.png"
        output_path = os.path.join(output_dir, filename)
        _save_rgb(output_path, rendered)
        rendered_images.append(rendered)

        finite_depth = depth[valid]
        records.append(
            {
                **view,
                "eye": list(view["eye"]),
                "target": list(view["target"]),
                "image": filename,
                "rgb_range": rgb_range,
                "valid_depth_fraction": valid_fraction,
                "depth_min_m": float(np.min(finite_depth)),
                "depth_max_m": float(np.max(finite_depth)),
            }
        )
        print(
            f"Saved view {view['number']:02d}: {output_path} "
            f"(valid depth {100.0 * valid_fraction:.2f}%)",
            flush=True,
        )

    montage_path = os.path.join(output_dir, "00_numbered_views_montage.png")
    _make_montage(rendered_images, montage_path)
    manifest = {
        "coordinate_frame": {
            "up_axis": "Z",
            "meters_per_unit": 1.0,
            "source_dominant_floor_z_m": SOURCE_FLOOR_Z_M,
            "wrapper_translation_z_m": FLOOR_OFFSET_Z_M,
            "resulting_dominant_floor_z_m": RESULT_FLOOR_Z_M,
        },
        "wrapper_usd": WRAPPER_USD,
        "source_usdz": SOURCE_USDZ,
        "nurec_render_settings": NUREC_RENDER_SETTINGS,
        "camera": {
            "resolution": [960, 540],
            "focal_length_mm": 12.0,
            "horizontal_aperture_mm": 20.0,
        },
        "view_set": args_cli.view_set,
        "views": records,
    }
    manifest_path = os.path.join(output_dir, "numbered_views_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"Saved montage: {montage_path}", flush=True)
    print(f"Saved camera manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
