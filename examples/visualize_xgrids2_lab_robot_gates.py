# SPDX-License-Identifier: Apache-2.0

"""Validate and visualize XGRIDS2 robot spawning and base traversal gates."""

import argparse
import json
import math
import os
import sys

import cv2  # import before Isaac Lab
from isaaclab.app import AppLauncher


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from robolab.core.utils.nurec_rendering import (
    configure_xgrids_vendor_nurec_volume,
    configure_xgrids_vendor_render_products,
    require_single_gpu,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", default="output/xgrids2_lab/physics_gates"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
require_single_gpu(args_cli)
simulation_app = AppLauncher(args_cli).app
NUREC_RENDER_SETTINGS = configure_xgrids_vendor_nurec_volume()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.registrations.kinova_mobile import (  # noqa: E402
    register_xgrids2_lab_mobile_env,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (  # noqa: E402
    COLLISION_SEGMENTS,
    DEFAULT_REQUIRED_LINE_CLEARANCE_M,
    MOBILE_KINOVA_TRANSPORT_RADIUS_M,
    PHYSICS_REGION,
    STAGING_POSES,
    base_pose_is_clear,
    distance_to_segments,
    planned_route,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_navigation_oracle import (  # noqa: E402
    WaypointFollower,
    wrap_angle,
)


def _action_slices(env) -> dict[str, slice]:
    offset = 0
    result = {}
    for name, size in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
        result[name] = slice(offset, offset + size)
        offset += size
    return result


def _hold_action(env) -> torch.Tensor:
    action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        dtype=torch.float32,
        device=env.device,
    )
    robot = env.scene["robot"]
    indices = [robot.data.joint_names.index(f"joint_{index}") for index in range(1, 8)]
    action[:, _action_slices(env)["arm"]] = robot.data.default_joint_pos[:, indices]
    return action


def _set_stage(env, name: str) -> None:
    x, y, z, yaw = STAGING_POSES[name]
    pose = torch.tensor(
        [[x, y, z, math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]],
        device=env.device,
    )
    pose[:, :3] += env.scene.env_origins[:, :3]
    robot = env.scene["robot"]
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)


def _robot_state(env) -> tuple[np.ndarray, float]:
    robot = env.scene["robot"]
    xy = (robot.data.root_link_pos_w[0, :2] - env.scene.env_origins[0, :2]).detach().cpu().numpy()
    quaternion = robot.data.root_link_quat_w[0]
    yaw = math.atan2(
        2.0 * float(quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
        1.0 - 2.0 * float(quaternion[2] ** 2 + quaternion[3] ** 2),
    )
    return xy, yaw


def _rgb(camera) -> np.ndarray:
    return np.ascontiguousarray(
        camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy().astype(np.uint8)
    )


def _label(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 76), (15, 19, 25), -1)
    cv2.putText(image, title, (20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2)
    if subtitle:
        cv2.putText(image, subtitle, (20, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (90, 220, 255), 1)
    return image


def _map_panel(route: np.ndarray, robot_xy: np.ndarray, waypoint_index: int) -> np.ndarray:
    height, width = 270, 480
    panel = np.full((height, width, 3), (19, 23, 29), dtype=np.uint8)
    margin_x, margin_y = 32, 35

    def pixel(point):
        x, y = point
        px = margin_x + (x - PHYSICS_REGION.xmin) / PHYSICS_REGION.size[0] * (width - 2 * margin_x)
        py = height - margin_y - (y - PHYSICS_REGION.ymin) / PHYSICS_REGION.size[1] * (height - 2 * margin_y)
        return int(px), int(py)

    for segment in COLLISION_SEGMENTS:
        cv2.line(panel, pixel(segment.start_xy), pixel(segment.end_xy), (78, 85, 94), 1)
    route_pixels = np.asarray([pixel(point) for point in route], dtype=np.int32)
    cv2.polylines(panel, [route_pixels], False, (255, 190, 55), 2, cv2.LINE_AA)
    cv2.circle(panel, pixel(STAGING_POSES["dining_table"][:2]), 5, (80, 220, 120), -1)
    cv2.circle(panel, pixel(STAGING_POSES["snack_table"][:2]), 5, (70, 120, 245), -1)
    pixels_per_metre = (width - 2 * margin_x) / PHYSICS_REGION.size[0]
    cv2.circle(
        panel,
        pixel(robot_xy),
        max(4, int(MOBILE_KINOVA_TRANSPORT_RADIUS_M * pixels_per_metre)),
        (70, 220, 255),
        2,
    )
    cv2.putText(panel, "INFLATED ROUTE / LIVE BASE", (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1)
    cv2.putText(panel, f"waypoint {waypoint_index}/{len(route)-1}", (300, 253), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (205, 210, 215), 1)
    return panel


def _trial_frame(env, route: np.ndarray, follower: WaypointFollower, clearance: float, force_excess: float) -> np.ndarray:
    xy, yaw = _robot_state(env)
    third = cv2.resize(_rgb(env.scene["trial_camera"]), (960, 540), interpolation=cv2.INTER_AREA)
    nav = cv2.resize(_rgb(env.scene["navigation_camera"]), (480, 270), interpolation=cv2.INTER_AREA)
    third = _label(
        third,
        "GATE 5 | PHYSICS-VALID BASE TRAVERSAL",
        f"base=({xy[0]:.2f}, {xy[1]:.2f})  yaw={math.degrees(yaw):.1f} deg  clearance={clearance:.3f} m",
    )
    nav = _label(nav, "FIXED ROBOT CAMERA", "camera prim remains parented to base_link")
    map_image = _map_panel(route, xy, follower.waypoint_index)
    cv2.putText(map_image, f"max nonvertical force excess: {force_excess:.3f} N", (16, 253), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 220, 255), 1)
    return np.concatenate((third, np.concatenate((nav, map_image), axis=0)), axis=1)


def main() -> None:
    env_cfg_class, _ = register_xgrids2_lab_mobile_env()
    env_cfg = env_cfg_class()
    env_cfg.sim.use_fabric = True
    env_cfg.sim.render_interval = 8
    env_cfg.sim.render.antialiasing_mode = "OFF"
    # The registered task keeps both policy cameras.  This audit consumes only
    # the navigation view, so omit the wrist stream and lower audit-only render
    # resolution to coexist with other Isaac processes on the workstation.
    env_cfg.scene.wrist_cam = None
    env_cfg.scene.navigation_camera.height = 270
    env_cfg.scene.navigation_camera.width = 480
    env_cfg.scene.navigation_camera.update_period = 1.0 / 15.0
    env_cfg.scene.trial_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/trial_camera",
        height=360,
        width=640,
        update_period=1.0 / 15.0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=25.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, 30.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(convention="opengl"),
    )
    env, _ = create_env(env_cfg)
    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    montage_path = os.path.join(output_dir, "gate4_robot_spawn_montage.png")
    video_path = os.path.join(output_dir, "gate5_base_traversal.mp4")
    still_path = os.path.join(output_dir, "gate5_base_traversal_final.png")
    keyframes_path = os.path.join(output_dir, "gate5_base_traversal_keyframes.png")
    results_path = os.path.join(output_dir, "gate5_base_traversal_results.json")
    video = VideoWriter(video_path, fps=15)

    try:
        env.reset_eval_state()
        env.reset()
        nurec_render_products = configure_xgrids_vendor_render_products(
            env.scene["trial_camera"], env.scene["navigation_camera"]
        )
        hold = _hold_action(env)
        trial_camera = env.scene["trial_camera"]
        spawn_frames = []
        camera_views = {
            # Exact audited XGRIDS View 05 pose: a scan-supported overhead.
            "dining_table": ((0.0, 0.0, 3.2), (0.2, 0.1, 0.0)),
            "snack_table": ((6.0, -2.7, 2.25), (4.56, -0.76, 0.55)),
        }
        for name in ("dining_table", "snack_table"):
            _set_stage(env, name)
            eye, target = camera_views[name]
            trial_camera.set_world_poses_from_view(
                eyes=torch.tensor([eye], device=env.device),
                targets=torch.tensor([target], device=env.device),
            )
            for _ in range(45):
                env.step(hold)
            xy, yaw = _robot_state(env)
            if np.linalg.norm(xy - np.asarray(STAGING_POSES[name][:2])) >= 0.01:
                raise RuntimeError(f"{name} spawn moved during settle")
            third = cv2.resize(_rgb(trial_camera), (960, 540), interpolation=cv2.INTER_AREA)
            nav = cv2.resize(_rgb(env.scene["navigation_camera"]), (640, 360), interpolation=cv2.INTER_AREA)
            nav_canvas = np.full((540, 640, 3), (18, 22, 28), dtype=np.uint8)
            nav_canvas[90:450] = nav
            spawn_frames.append(
                np.concatenate(
                    (
                        _label(third, f"GATE 4 | {name.replace('_', ' ').upper()}", f"settled at ({xy[0]:.3f}, {xy[1]:.3f}), yaw={math.degrees(yaw):.1f} deg"),
                        _label(nav_canvas, "PARENTED NAVIGATION CAMERA", "same fixed mount at this staging pose"),
                    ),
                    axis=1,
                )
            )
        montage = np.concatenate(spawn_frames, axis=0)
        if not cv2.imwrite(montage_path, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write {montage_path}")

        _set_stage(env, "dining_table")
        trial_camera.set_world_poses_from_view(
            # Exact audited XGRIDS View 01 pose.  Keeping the third-person
            # audit on a captured camera trajectory avoids neural-view blur.
            eyes=torch.tensor([[0.0, 0.0, 1.625]], device=env.device),
            targets=torch.tensor([[5.0, 0.0, 1.625]], device=env.device),
        )
        for _ in range(12):
            env.step(hold)

        route = planned_route()
        target = STAGING_POSES["snack_table"]
        follower = WaypointFollower(route, final_yaw=target[3])
        slices = _action_slices(env)
        minimum_clearance = math.inf
        maximum_nonvertical_force_excess = 0.0
        reached = False
        step = -1
        final_frame = None
        keyframes = []
        keyframe_thresholds = [1, max(2, (len(route) - 1) // 2), len(route) - 1]
        for step in range(1000):
            xy, yaw = _robot_state(env)
            command, reached = follower.command(xy, yaw)
            action = hold.clone()
            action[0, slices["base"]] = torch.as_tensor(command, device=env.device)
            _, _, terminated, truncated, _ = env.step(action)
            if bool(torch.any(terminated | truncated)):
                raise RuntimeError("Traversal terminated before reaching the goal")
            xy, _ = _robot_state(env)
            line_clearance = float(distance_to_segments(xy[None, :])[0])
            minimum_clearance = min(minimum_clearance, line_clearance)
            if not base_pose_is_clear(tuple(float(value) for value in xy)):
                raise RuntimeError(f"Robot left inflated free space at {xy.tolist()}")
            net_force = env.scene["base_obstacle_contact"].data.net_forces_w[0, 0]
            horizontal = float(torch.linalg.vector_norm(net_force[:2]))
            vertical = abs(float(net_force[2]))
            maximum_nonvertical_force_excess = max(
                maximum_nonvertical_force_excess, horizontal - vertical
            )
            if step % 2 == 0:
                final_frame = _trial_frame(env, route, follower, line_clearance, maximum_nonvertical_force_excess)
                video.write(final_frame)
                if keyframe_thresholds and follower.waypoint_index >= keyframe_thresholds[0]:
                    keyframes.append(final_frame.copy())
                    keyframe_thresholds.pop(0)
            if reached:
                break

        if not reached:
            raise RuntimeError(f"Follower stopped at waypoint {follower.waypoint_index}")
        for hold_step in range(24):
            env.step(hold)
            if hold_step % 2 == 0:
                final_frame = _trial_frame(env, route, follower, minimum_clearance, maximum_nonvertical_force_excess)
                video.write(final_frame)

        xy, yaw = _robot_state(env)
        final_position_error = float(np.linalg.norm(xy - np.asarray(target[:2])))
        final_yaw_error = abs(wrap_angle(yaw - target[3]))
        if final_position_error >= 0.07 or final_yaw_error >= 0.07:
            raise RuntimeError(
                f"Final pose error: position={final_position_error:.4f}, yaw={final_yaw_error:.4f}"
            )
        if maximum_nonvertical_force_excess >= 0.5:
            raise RuntimeError(
                f"Base contact left the vertical support cone by {maximum_nonvertical_force_excess:.3f} N"
            )
        if final_frame is None or not cv2.imwrite(still_path, cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write {still_path}")
        if len(keyframes) != 3:
            raise RuntimeError(f"Expected three trajectory keyframes, received {len(keyframes)}")
        keyframe_montage = np.concatenate(keyframes, axis=0)
        if not cv2.imwrite(keyframes_path, cv2.cvtColor(keyframe_montage, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write {keyframes_path}")
        results = {
            "passed": True,
            "route_points": int(len(route)),
            "steps": step + 1,
            "required_line_clearance_m": DEFAULT_REQUIRED_LINE_CLEARANCE_M,
            "minimum_line_clearance_m": minimum_clearance,
            "maximum_nonvertical_contact_force_excess_n": maximum_nonvertical_force_excess,
            "final_position_error_m": final_position_error,
            "final_yaw_error_rad": final_yaw_error,
            "nurec_render_settings": NUREC_RENDER_SETTINGS,
            "nurec_render_products": nurec_render_products,
        }
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(json.dumps(results, indent=2))
        print(f"Saved Gate 4 montage: {montage_path}")
        print(f"Saved Gate 5 video: {video_path}")
        print(f"Saved Gate 5 still: {still_path}")
        print(f"Saved Gate 5 keyframes: {keyframes_path}")
    finally:
        video.release()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
