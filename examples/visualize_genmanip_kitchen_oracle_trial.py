# SPDX-License-Identifier: Apache-2.0

"""Render one annotated Milestone-2 mobile manipulation oracle trial."""

import argparse
import json
import math
import os

import cv2  # noqa: F401 -- import before Isaac Lab
from isaaclab.app import AppLauncher


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EBENCH_ASSETS = os.path.join(_REPO_ROOT, "assets", "third_party", "ebench", "assets")
_MDL_PATHS = [
    "/isaac-sim/materials/",
    os.path.join(_EBENCH_ASSETS, "miscs", "mdl", "ebench", "mdl"),
    os.path.join(
        _EBENCH_ASSETS,
        "scene_usds",
        "ebench",
        "long_horizon",
        "scene_dishwasher",
        "SubUSDs",
        "materials",
    ),
]
if os.environ.get("MDL_SYSTEM_PATH"):
    _MDL_PATHS.append(os.environ["MDL_SYSTEM_PATH"])
os.environ["MDL_SYSTEM_PATH"] = os.pathsep.join(_MDL_PATHS)
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--max-steps", type=int, default=6000)
parser.add_argument(
    "--scenario",
    choices=("baseline", "robot_other_side", "can_other_side"),
    default="baseline",
)
parser.add_argument("--pick-xy", nargs=2, type=float, metavar=("X", "Y"))
parser.add_argument("--place-xy", nargs=2, type=float, metavar=("X", "Y"))
parser.add_argument("--capture-stride", type=int, default=2)
parser.add_argument(
    "--sensor-views",
    action="store_true",
    help="Record synchronized wrist and base-mounted Kinect-proxy RGB views.",
)
parser.add_argument(
    "--output",
    default="output/genmanip_kitchen/milestone2/oracle_trial_visualized.mp4",
)
parser.add_argument(
    "--still-output",
    default="output/genmanip_kitchen/milestone2/oracle_trial_final.png",
)
parser.add_argument(
    "--montage-output",
    default="output/genmanip_kitchen/milestone2/oracle_trial_keyframes.png",
)
parser.add_argument(
    "--trace-output",
    default="output/genmanip_kitchen/milestone2/oracle_trial_visualized.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    convert_camera_frame_orientation_convention,
    quat_apply,
    quat_apply_inverse,
    quat_mul,
)

from robolab.core.environments.config import parse_env_cfg  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.registrations.kinova_mobile import (  # noqa: E402
    IK_ENV_NAME,
    register_genmanip_kitchen_mobile_ik_env,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (  # noqa: E402
    BASE_CLEARANCE_INFLATION,
    Bounds2D,
    CAN_HALF_HEIGHT,
    CAN_RADIUS,
    ORACLE_PLACE_REGION,
    PICK_REGION,
    PLACE_REGION,
    ROBOT_BASE_RADIUS,
    STAGING_POSES,
    navigation_obstacle_footprints,
)
from robolab.tasks.mobile_manipulation.kitchen_manipulation_oracle import (  # noqa: E402
    KitchenManipulationOracle,
)
from robolab.tasks.mobile_manipulation.kitchen_scenarios import (  # noqa: E402
    KitchenManipulationScenario,
    get_kitchen_scenario,
)


VIDEO_FPS = 15
CAMERA_HEIGHT = 540
CAMERA_WIDTH = 960
PANEL_WIDTH = 420
def _world_to_panel(
    point: tuple[float, float], panel_shape: tuple[int, int, int]
) -> tuple[int, int]:
    """Project kitchen-frame XY coordinates into the top-down inset."""

    height, width = panel_shape[:2]
    xmin, xmax = -1.55, 2.55
    ymin, ymax = -1.35, 1.75
    x = int(28 + (point[0] - xmin) / (xmax - xmin) * (width - 56))
    y = int(height - 28 - (point[1] - ymin) / (ymax - ymin) * (height - 58))
    return x, y


def _draw_bounds(panel: np.ndarray, bounds, color, thickness: int, *, inflate=0.0):
    bounds = bounds.inflated(inflate)
    cv2.rectangle(
        panel,
        _world_to_panel((bounds.xmin, bounds.ymax), panel.shape),
        _world_to_panel((bounds.xmax, bounds.ymin), panel.shape),
        color,
        thickness,
    )


def _phase_group(phase: str) -> str:
    groups = {
        "settle": "1  RESET + SETTLE",
        "navigate_to_pick": "2  BASE NAVIGATE TO PICK",
        "pregrasp": "2  APPROACH",
        "descend": "2  APPROACH",
        "close": "3  CONTACT + GRASP",
        "secure": "3  CONTACT + GRASP",
        "lift": "4  LIFT",
        "carry_settle": "4  LIFT",
        "traverse": "5  BASE TRAVERSE",
        "carry_reposition": "6  ARM REPOSITION",
        "carry_reorient": "6  ARM REPOSITION",
        "preplace": "7  PRE-PLACE",
        "place": "7  PRE-PLACE",
        "handoff_settle": "8  RELEASE + SETTLE",
        "open": "8  RELEASE + SETTLE",
        "retreat": "9  RETREAT",
        "success": "10 SUCCESS",
        "failure": "FAILURE",
    }
    return groups.get(phase, phase.upper())


def _top_down_panel(
    height: int,
    *,
    phase: str,
    robot_xy: tuple[float, float],
    can_xy: tuple[float, float],
    tcp_xy: tuple[float, float],
    target_xy: tuple[float, float],
    scenario: KitchenManipulationScenario,
) -> np.ndarray:
    """Draw collision contracts and live task state beside the RGB view."""

    panel = np.full((height, PANEL_WIDTH, 3), (24, 28, 34), dtype=np.uint8)
    cv2.putText(
        panel,
        "TOP-DOWN TASK CONTRACT",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        2,
    )
    map_view = panel[42:360]
    for obstacle in navigation_obstacle_footprints():
        _draw_bounds(map_view, obstacle, (70, 76, 84), -1)
        _draw_bounds(
            map_view,
            obstacle,
            (245, 150, 70),
            1,
            inflate=BASE_CLEARANCE_INFLATION,
        )
    _draw_bounds(map_view, PICK_REGION, (80, 185, 100), 1)
    _draw_bounds(map_view, PLACE_REGION, (75, 145, 220), 1)
    _draw_bounds(map_view, scenario.pick_region, (95, 245, 135), 2)
    _draw_bounds(map_view, ORACLE_PLACE_REGION, (90, 205, 255), 2)

    pick_xy = (
        scenario.pre_pick_waypoints[-1][:2]
        if scenario.pre_pick_waypoints
        else STAGING_POSES["pick"][:2]
    )
    pre_route = (scenario.initial_base_pose[:2],) + tuple(
        waypoint[:2] for waypoint in scenario.pre_pick_waypoints
    )
    for start, end in zip(pre_route, pre_route[1:]):
        cv2.line(
            map_view,
            _world_to_panel(start, map_view.shape),
            _world_to_panel(end, map_view.shape),
            (210, 120, 230),
            2,
        )
    carry_route = (pick_xy,) + tuple(
        waypoint[:2] for waypoint in scenario.carry_waypoints
    )
    for start, end in zip(carry_route, carry_route[1:]):
        cv2.line(
            map_view,
            _world_to_panel(start, map_view.shape),
            _world_to_panel(end, map_view.shape),
            (210, 210, 210),
            1,
        )
    pixels_per_metre = (PANEL_WIDTH - 56) / 4.10
    cv2.circle(
        map_view,
        _world_to_panel(robot_xy, map_view.shape),
        max(4, int(ROBOT_BASE_RADIUS * pixels_per_metre)),
        (255, 215, 70),
        -1,
    )
    cv2.circle(map_view, _world_to_panel(can_xy, map_view.shape), 6, (65, 80, 235), -1)
    cv2.circle(map_view, _world_to_panel(tcp_xy, map_view.shape), 5, (95, 245, 150), 2)
    target_px = _world_to_panel(target_xy, map_view.shape)
    cv2.drawMarker(
        map_view,
        target_px,
        (80, 215, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=2,
    )

    cv2.putText(
        panel,
        _phase_group(phase),
        (18, 397),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (80, 215, 255) if phase != "failure" else (245, 90, 90),
        2,
    )
    legend = (
        "yellow base  |  red can  |  green TCP",
        "cyan cross target  |  gray fixtures",
        "thin boxes physical support",
        "thick boxes oracle sampling domain",
        "orange boundary = base clearance inflation",
    )
    for index, line in enumerate(legend):
        cv2.putText(
            panel,
            line,
            (18, 430 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 205, 210),
            1,
        )
    return panel


def _annotate(
    rgb: np.ndarray,
    *,
    phase: str,
    step: int,
    robot_xy: tuple[float, float],
    can_xyz: tuple[float, float, float],
    tcp_xyz: tuple[float, float, float],
    target_xy: tuple[float, float],
    grasp_active: bool,
    scenario: KitchenManipulationScenario,
) -> np.ndarray:
    """Combine the scene image with live state and geometric context."""

    rgb = np.ascontiguousarray(rgb[:, :, :3])
    cv2.rectangle(rgb, (16, 16), (800, 106), (16, 20, 26), -1)
    cv2.putText(
        rgb,
        "ROBOLAB MILESTONE 2 | DETERMINISTIC MOBILE MANIPULATION ORACLE",
        (30, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        2,
    )
    cv2.putText(
        rgb,
        f"{phase.upper()}   step={step}   grasp_assist={'ON' if grasp_active else 'OFF'}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 215, 255),
        2,
    )
    cv2.putText(
        rgb,
        (
            f"base=({robot_xy[0]:.2f},{robot_xy[1]:.2f})  "
            f"can=({can_xyz[0]:.2f},{can_xyz[1]:.2f},{can_xyz[2]:.2f})  "
            f"target=({target_xy[0]:.2f},{target_xy[1]:.2f})"
        ),
        (30, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (215, 220, 225),
        1,
    )
    panel = _top_down_panel(
        rgb.shape[0],
        phase=phase,
        robot_xy=robot_xy,
        can_xy=can_xyz[:2],
        tcp_xy=tcp_xyz[:2],
        target_xy=target_xy,
        scenario=scenario,
    )
    return np.concatenate((rgb, panel), axis=1)


def _live_state(env, oracle: KitchenManipulationOracle) -> dict:
    origin = env.scene.env_origins[0]
    robot = env.scene["robot"]
    can = env.scene["can"]
    tcp_pos_w, _ = oracle.current_tcp_w()
    return {
        "phase": oracle.phase_names()[0],
        "robot_xy": tuple(
            float(value) for value in (robot.data.root_link_pos_w[0, :2] - origin[:2])
        ),
        "can_xyz": tuple(float(value) for value in (can.data.root_pos_w[0] - origin)),
        "tcp_xyz": tuple(float(value) for value in (tcp_pos_w[0] - origin)),
        "grasp_active": bool(oracle.grasp_constraint_active[0].item()),
    }


def _save_rgb(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write image: {path}")


def _can_visibility(
    camera,
    depth: np.ndarray,
    can_position_w: torch.Tensor,
    *,
    camera_position_w: torch.Tensor | None = None,
    camera_quat_w_ros: torch.Tensor | None = None,
) -> dict:
    """Project the can into a policy camera and check its rendered depth patch."""

    if camera_position_w is None:
        camera_position_w = camera.data.pos_w[0]
    if camera_quat_w_ros is None:
        camera_quat_w_ros = camera.data.quat_w_ros[0]
    relative_w = can_position_w - camera_position_w
    point_camera = quat_apply_inverse(
        camera_quat_w_ros.unsqueeze(0), relative_w.unsqueeze(0)
    )[0]
    x, y, z = (float(value) for value in point_camera)
    height, width = depth.shape[:2]
    intrinsic = camera.data.intrinsic_matrices[0]
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if z <= 0.0:
        return {
            "in_frame": False,
            "visible": False,
            "pixel": None,
            "camera_xyz": [x, y, z],
            "camera_depth_m": z,
            "matching_depth_pixels": 0,
        }

    u = fx * x / z + cx
    v = fy * y / z + cy
    in_frame = 0.0 <= u < width and 0.0 <= v < height
    matching_pixels = 0
    if in_frame:
        # The upright can projects to a small rectangle. A bounding-sphere depth
        # tolerance allows for its curved side and top without depending on its
        # texture, color, semantic label, or the robot model.
        radius_u = max(2, int(math.ceil(fx * CAN_RADIUS / z)))
        radius_v = max(2, int(math.ceil(fy * CAN_HALF_HEIGHT / z)))
        u0 = max(0, int(round(u)) - radius_u)
        u1 = min(width, int(round(u)) + radius_u + 1)
        v0 = max(0, int(round(v)) - radius_v)
        v1 = min(height, int(round(v)) + radius_v + 1)
        patch = depth[v0:v1, u0:u1]
        tolerance = math.hypot(CAN_RADIUS, CAN_HALF_HEIGHT) + 0.015
        depth_match = np.isfinite(patch) & (np.abs(patch - z) <= tolerance)
        matching_pixels = int(depth_match.sum())

    return {
        "in_frame": in_frame,
        "visible": bool(in_frame and matching_pixels >= 4),
        "pixel": [u, v],
        "camera_xyz": [x, y, z],
        "camera_depth_m": z,
        "matching_depth_pixels": matching_pixels,
    }


def _sensor_composite(
    overview_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    kinect_rgb: np.ndarray,
    *,
    phase: str,
    step: int,
    visibility: dict[str, dict],
    robot_xy: tuple[float, float],
) -> np.ndarray:
    """Build synchronized overview and robot-mounted camera panes."""

    views = []
    for name, label, rgb in (
        (None, "THIRD PERSON / OVERVIEW", overview_rgb),
        ("wrist_cam", "WRIST RGB", wrist_rgb),
        ("kinect_camera", "KINECT / BASE RGB", kinect_rgb),
    ):
        view = cv2.resize(np.ascontiguousarray(rgb[:, :, :3]), (640, 360))
        cv2.rectangle(view, (0, 0), (640, 42), (16, 20, 26), -1)
        cv2.putText(
            view,
            label,
            (16, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            2,
        )
        if name is None:
            cv2.putText(
                view,
                f"ROBOT xy=({robot_xy[0]:.2f}, {robot_xy[1]:.2f})",
                (16, 344),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (80, 215, 255),
                2,
            )
            views.append(view)
            continue
        can_view = visibility[name]
        if can_view["in_frame"]:
            u, v = can_view["pixel"]
            marker = (
                int(round(u * 640 / rgb.shape[1])),
                int(round(v * 360 / rgb.shape[0])),
            )
            color = (70, 235, 95) if can_view["visible"] else (235, 85, 70)
            cv2.drawMarker(view, marker, color, cv2.MARKER_CROSS, 20, 2)
            status = "CAN VISIBLE" if can_view["visible"] else "CAN OCCLUDED"
        else:
            color = (235, 85, 70)
            status = "CAN OUT OF VIEW"
        cv2.putText(
            view,
            status,
            (16, 344),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
        )
        views.append(view)
    composite = np.concatenate(views, axis=1)
    cv2.putText(
        composite,
        f"{phase.upper()}  step={step}",
        (1570, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 215, 255),
        2,
    )
    return composite


def _new_sensor_stats() -> dict:
    return {
        "frames": 0,
        "rgb_min": 255,
        "rgb_max": 0,
        "rgb_mean_sum": 0.0,
        "changed_frames": 0,
        "difference_sum": 0.0,
        "valid_depth_pixels": 0,
        "depth_pixels": 0,
        "depth_min_m": float("inf"),
        "depth_max_m": 0.0,
        "previous_rgb": None,
        "rgb_shape": None,
        "depth_shape": None,
        "projected_in_frame": 0,
        "can_visible_frames": 0,
        "can_visibility_by_phase": {},
    }


def _update_sensor_stats(
    stats: dict,
    rgb: np.ndarray,
    depth: np.ndarray,
    visibility: dict,
    phase: str,
) -> None:
    rgb = rgb[:, :, :3]
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    stats["frames"] += 1
    stats["rgb_shape"] = list(rgb.shape)
    stats["depth_shape"] = list(depth.shape)
    stats["rgb_min"] = min(stats["rgb_min"], int(rgb.min()))
    stats["rgb_max"] = max(stats["rgb_max"], int(rgb.max()))
    stats["rgb_mean_sum"] += float(rgb.mean())
    stats["valid_depth_pixels"] += int(valid_depth.sum())
    stats["depth_pixels"] += int(depth.size)
    stats["projected_in_frame"] += int(visibility["in_frame"])
    stats["can_visible_frames"] += int(visibility["visible"])
    phase_stats = stats["can_visibility_by_phase"].setdefault(
        phase, {"frames": 0, "in_frame": 0, "visible": 0}
    )
    phase_stats["frames"] += 1
    phase_stats["in_frame"] += int(visibility["in_frame"])
    phase_stats["visible"] += int(visibility["visible"])
    phase_stats["last_camera_xyz"] = visibility["camera_xyz"]
    phase_stats["last_pixel"] = visibility["pixel"]
    if np.any(valid_depth):
        stats["depth_min_m"] = min(
            stats["depth_min_m"], float(depth[valid_depth].min())
        )
        stats["depth_max_m"] = max(
            stats["depth_max_m"], float(depth[valid_depth].max())
        )
    previous = stats["previous_rgb"]
    sample = cv2.resize(rgb, (160, 90))
    if previous is not None:
        difference = float(
            np.mean(np.abs(sample.astype(np.float32) - previous.astype(np.float32)))
        )
        stats["difference_sum"] += difference
        if difference > 0.25:
            stats["changed_frames"] += 1
    stats["previous_rgb"] = sample


def _finalize_sensor_stats(stats: dict) -> dict:
    frames = stats["frames"]
    comparisons = max(frames - 1, 1)
    valid_fraction = stats["valid_depth_pixels"] / max(stats["depth_pixels"], 1)
    depth_min = stats["depth_min_m"]
    return {
        "frames": frames,
        "rgb_shape": stats["rgb_shape"],
        "depth_shape": stats["depth_shape"],
        "rgb_range": [stats["rgb_min"], stats["rgb_max"]],
        "rgb_mean": stats["rgb_mean_sum"] / max(frames, 1),
        "changed_frames": stats["changed_frames"],
        "mean_temporal_abs_difference": stats["difference_sum"] / comparisons,
        "valid_depth_fraction": valid_fraction,
        "depth_range_m": [
            None if not np.isfinite(depth_min) else depth_min,
            stats["depth_max_m"],
        ],
        "can_projected_in_frame": stats["projected_in_frame"],
        "can_visible_frames": stats["can_visible_frames"],
        "can_visible_fraction": stats["can_visible_frames"] / max(frames, 1),
        "can_visibility_by_phase": stats["can_visibility_by_phase"],
        "working": bool(
            frames > 1
            and stats["rgb_max"] > stats["rgb_min"]
            and stats["changed_frames"] > 0
            and valid_fraction > 0.0
        ),
    }


def _save_montage(path: str, phase_frames: dict[str, np.ndarray]) -> None:
    """Save eight representative hard-scenario frames as a 2x4 contact sheet."""

    preferred = (
        "settle",
        "navigate_to_pick",
        "pregrasp",
        "close",
        "lift",
        "traverse",
        "preplace",
        "success",
    )
    selected = [phase for phase in preferred if phase in phase_frames]
    if len(selected) < 8:
        selected.extend(phase for phase in phase_frames if phase not in selected)
    if not selected:
        raise RuntimeError("No montage phases were captured")
    while len(selected) < 8:
        selected.append(selected[-1])
    tiles = [cv2.resize(phase_frames[phase], (690, 270)) for phase in selected[:8]]
    montage = np.concatenate(
        [np.concatenate(tiles[row : row + 2], axis=1) for row in range(0, 8, 2)],
        axis=0,
    )
    _save_rgb(path, montage)


def main() -> int:
    if args_cli.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args_cli.capture_stride < 1:
        raise ValueError("--capture-stride must be positive")

    torch.manual_seed(args_cli.seed)
    scenario = get_kitchen_scenario(args_cli.scenario)
    place_xy = tuple(args_cli.place_xy or ORACLE_PLACE_REGION.center)
    register_genmanip_kitchen_mobile_ik_env()
    env_cfg = parse_env_cfg(
        IK_ENV_NAME,
        seed=args_cli.seed,
        num_envs=1,
        env_spacing=8.0,
        # RTX cameras render Fabric transforms. Without Fabric the tensor-side
        # articulation state advances, but Hydra can retain the robot's authored
        # USD pose, producing a video in which the base appears stationary.
        use_fabric=True,
    )
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s,
        args_cli.max_steps * env_cfg.sim.dt * env_cfg.decimation + 1.0,
    )
    base_x, base_y, base_z, base_yaw = scenario.initial_base_pose
    env_cfg.scene.robot.init_state.pos = (base_x, base_y, base_z)
    env_cfg.scene.robot.init_state.rot = (
        math.cos(0.5 * base_yaw), 0.0, 0.0, math.sin(0.5 * base_yaw)
    )
    pick_region = scenario.pick_region
    if args_cli.pick_xy is not None:
        pick_x, pick_y = args_cli.pick_xy
        if not (
            pick_region.xmin <= pick_x <= pick_region.xmax
            and pick_region.ymin <= pick_y <= pick_region.ymax
        ):
            raise ValueError(f"--pick-xy lies outside {pick_region}")
        pick_region = Bounds2D(pick_x, pick_x, pick_y, pick_y)
    env_cfg.events.sample_can.params = {"pick_region": pick_region}
    env_cfg.recorders = None
    if not args_cli.sensor_views:
        env_cfg.scene.wrist_cam = None
        env_cfg.scene.navigation_camera = None
    env_cfg.scene.trial_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/trial_camera",
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=26.0,
            focus_distance=3.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, 30.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            # Static high/wide view from the room's open south side. This keeps
            # the west -> north -> east navigation route and mobile base visible.
            pos=(0.60, -3.80, 6.00),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )

    env, _ = create_env(env_cfg)
    output_path = os.path.abspath(args_cli.output)
    still_path = os.path.abspath(args_cli.still_output)
    montage_path = os.path.abspath(args_cli.montage_output)
    trace_path = os.path.abspath(args_cli.trace_output)
    for path in (output_path, still_path, montage_path, trace_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    video = VideoWriter(output_path, fps=VIDEO_FPS)

    transitions = []
    phase_frames = {}
    final_frame = None
    sensor_stats = {
        "wrist_cam": _new_sensor_stats(),
        "kinect_camera": _new_sensor_stats(),
    }
    step = 0
    try:
        env.reset_eval_state()
        env.reset()
        oracle = KitchenManipulationOracle(
            env,
            place_xy=place_xy,
            use_grasp_assist=True,
            scenario=scenario,
        )
        if args_cli.sensor_views:
            wrist_camera = env.scene["wrist_cam"]
            kinect_camera = env.scene["navigation_camera"]
            kinect_parent_indices, _ = env.scene["robot"].find_bodies(
                "base_link"
            )
            if len(kinect_parent_indices) != 1:
                raise RuntimeError(
                    "Expected exactly one base_link for navigation camera, got "
                    f"{kinect_parent_indices}"
                )
            kinect_parent_index = kinect_parent_indices[0]
            kinect_offset_position = torch.tensor(
                kinect_camera.cfg.offset.pos,
                dtype=torch.float32,
                device=env.device,
            )
            kinect_offset_quat_gl = convert_camera_frame_orientation_convention(
                torch.tensor(
                    kinect_camera.cfg.offset.rot,
                    dtype=torch.float32,
                    device=env.device,
                ).unsqueeze(0),
                origin=kinect_camera.cfg.offset.convention,
                target="opengl",
            )[0]
            wrist_parent_indices, _ = env.scene["robot"].find_bodies(
                "bracelet_link"
            )
            if len(wrist_parent_indices) != 1:
                raise RuntimeError(
                    "Expected exactly one bracelet_link for wrist camera, got "
                    f"{wrist_parent_indices}"
                )
            wrist_parent_index = wrist_parent_indices[0]
            wrist_offset_position = torch.tensor(
                wrist_camera.cfg.offset.pos,
                dtype=torch.float32,
                device=env.device,
            )
            wrist_offset_quat_gl = convert_camera_frame_orientation_convention(
                torch.tensor(
                    wrist_camera.cfg.offset.rot,
                    dtype=torch.float32,
                    device=env.device,
                ).unsqueeze(0),
                origin=wrist_camera.cfg.offset.convention,
                target="opengl",
            )[0]
            camera = env.scene["trial_camera"]
        else:
            camera = env.scene["trial_camera"]
            wrist_camera = None
            kinect_camera = None
        camera.set_world_poses_from_view(
            eyes=torch.tensor([[0.60, -3.80, 6.00]], device=env.device),
            targets=torch.tensor([[0.60, 0.35, 0.35]], device=env.device),
        )
        previous_phase = None

        for step in range(1, args_cli.max_steps + 1):
            action = oracle.compute_action()
            _, _, terminated, truncated, _ = env.step(action)
            oracle.post_step()
            state = _live_state(env, oracle)
            phase = state["phase"]
            phase_changed = phase != previous_phase
            if phase_changed:
                transitions.append({"step": step, "phase": phase})
                previous_phase = phase

            if phase_changed or step % args_cli.capture_stride == 0 or oracle.finished[0]:
                if args_cli.sensor_views:
                    overview_rgb = (
                        camera.data.output["rgb"][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )
                    wrist_rgb = (
                        wrist_camera.data.output["rgb"][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )
                    kinect_rgb = (
                        kinect_camera.data.output["rgb"][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )
                    wrist_depth = (
                        wrist_camera.data.output["distance_to_image_plane"][0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    kinect_depth = (
                        kinect_camera.data.output["distance_to_image_plane"][0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    can_position_w = env.scene["can"].data.root_pos_w[0]
                    robot = env.scene["robot"]
                    wrist_parent_position_w = robot.data.body_pos_w[
                        0, wrist_parent_index
                    ]
                    wrist_parent_quat_w = robot.data.body_quat_w[
                        0, wrist_parent_index
                    ]
                    wrist_position_w = wrist_parent_position_w + quat_apply(
                        wrist_parent_quat_w.unsqueeze(0),
                        wrist_offset_position.unsqueeze(0),
                    )[0]
                    wrist_quat_w_gl = quat_mul(
                        wrist_parent_quat_w.unsqueeze(0),
                        wrist_offset_quat_gl.unsqueeze(0),
                    )[0]
                    wrist_quat_w_ros = convert_camera_frame_orientation_convention(
                        wrist_quat_w_gl.unsqueeze(0),
                        origin="opengl",
                        target="ros",
                    )[0]
                    kinect_parent_position_w = robot.data.body_pos_w[
                        0, kinect_parent_index
                    ]
                    kinect_parent_quat_w = robot.data.body_quat_w[
                        0, kinect_parent_index
                    ]
                    kinect_position_w = kinect_parent_position_w + quat_apply(
                        kinect_parent_quat_w.unsqueeze(0),
                        kinect_offset_position.unsqueeze(0),
                    )[0]
                    kinect_quat_w_gl = quat_mul(
                        kinect_parent_quat_w.unsqueeze(0),
                        kinect_offset_quat_gl.unsqueeze(0),
                    )[0]
                    kinect_quat_w_ros = convert_camera_frame_orientation_convention(
                        kinect_quat_w_gl.unsqueeze(0),
                        origin="opengl",
                        target="ros",
                    )[0]
                    visibility = {
                        "wrist_cam": _can_visibility(
                            wrist_camera,
                            wrist_depth,
                            can_position_w,
                            camera_position_w=wrist_position_w,
                            camera_quat_w_ros=wrist_quat_w_ros,
                        ),
                        "kinect_camera": _can_visibility(
                            kinect_camera,
                            kinect_depth,
                            can_position_w,
                            camera_position_w=kinect_position_w,
                            camera_quat_w_ros=kinect_quat_w_ros,
                        ),
                    }
                    _update_sensor_stats(
                        sensor_stats["wrist_cam"],
                        wrist_rgb,
                        wrist_depth,
                        visibility["wrist_cam"],
                        phase,
                    )
                    _update_sensor_stats(
                        sensor_stats["kinect_camera"],
                        kinect_rgb,
                        kinect_depth,
                        visibility["kinect_camera"],
                        phase,
                    )
                    final_frame = _sensor_composite(
                        overview_rgb,
                        wrist_rgb,
                        kinect_rgb,
                        phase=phase,
                        step=step,
                        visibility=visibility,
                        robot_xy=state["robot_xy"],
                    )
                else:
                    rgb = (
                        camera.data.output["rgb"][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )
                    final_frame = _annotate(
                        rgb,
                        phase=phase,
                        step=step,
                        robot_xy=state["robot_xy"],
                        can_xyz=state["can_xyz"],
                        tcp_xyz=state["tcp_xyz"],
                        target_xy=place_xy,
                        grasp_active=state["grasp_active"],
                        scenario=scenario,
                    )
                video.write(final_frame)
                phase_frames.setdefault(phase, final_frame.copy())

            unexpected_done = bool((terminated[0] or truncated[0]).item()) and not bool(
                oracle.finished[0].item()
            )
            if unexpected_done:
                raise RuntimeError("RoboLab environment terminated before oracle completion")
            if bool(oracle.finished[0].item()):
                break

        succeeded = bool(oracle.succeeded[0].item())
        if final_frame is None:
            raise RuntimeError("No camera frames were produced")

        _save_rgb(still_path, final_frame)
        _save_montage(montage_path, phase_frames)
        final_state = _live_state(env, oracle)
        result = {
            "success": succeeded,
            "failure_reason": None if succeeded else oracle.failure_reason[0],
            "seed": args_cli.seed,
            "scenario": scenario.name,
            "steps": step,
            "capture_stride": args_cli.capture_stride,
            "video_fps": VIDEO_FPS,
            "pick_xy": list(args_cli.pick_xy) if args_cli.pick_xy else None,
            "place_xy": list(place_xy),
            "grasp_mode": "strict_post_contact_constraint",
            "camera_mode": (
                "overview_wrist_and_kinect" if args_cli.sensor_views else "overview"
            ),
            "camera_validation": (
                {
                    name: _finalize_sensor_stats(stats)
                    for name, stats in sensor_stats.items()
                }
                if args_cli.sensor_views
                else None
            ),
            "transitions": transitions,
            "final": {
                "phase": final_state["phase"],
                "robot_xy": list(final_state["robot_xy"]),
                "can_xyz": list(final_state["can_xyz"]),
                "tcp_xyz": list(final_state["tcp_xyz"]),
                "grasp_constraint_active": final_state["grasp_active"],
            },
            "video": output_path,
            "still": still_path,
            "montage": montage_path,
        }
        with open(trace_path, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    finally:
        video.release()
        env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
