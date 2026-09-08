# SPDX-License-Identifier: Apache-2.0

"""Render one Milestone-1 can-settle and mobile-base traversal trial."""

import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    default="output/genmanip_kitchen/milestone1/physics_trial.mp4",
)
parser.add_argument(
    "--still-output",
    default="output/genmanip_kitchen/milestone1/physics_trial_final.png",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.registrations.kinova_mobile import (  # noqa: E402
    register_genmanip_kitchen_mobile_env,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (  # noqa: E402
    BASE_CLEARANCE_INFLATION,
    CAN_HALF_HEIGHT,
    ISLAND_SURFACE_Z,
    PICK_REGION,
    PLACE_REGION,
    ROBOT_BASE_RADIUS,
    STAGING_POSES,
    base_pose_is_clear,
    navigation_obstacle_footprints,
)


SETTLE_STEPS = 90
MAX_TRAVERSAL_STEPS = 300
FINAL_HOLD_STEPS = 45
VIDEO_FPS = 30


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
    arm_indices = [robot.data.joint_names.index(f"joint_{index}") for index in range(1, 8)]
    action[:, _action_slices(env)["arm"]] = robot.data.default_joint_pos[:, arm_indices]
    return action


def _world_to_panel(
    point: tuple[float, float],
    panel_shape: tuple[int, int, int],
) -> tuple[int, int]:
    height, width = panel_shape[:2]
    xmin, xmax = -1.15, 2.55
    ymin, ymax = -2.65, 1.35
    x = int(35 + (point[0] - xmin) / (xmax - xmin) * (width - 70))
    y = int(height - 40 - (point[1] - ymin) / (ymax - ymin) * (height - 105))
    return x, y


def _draw_bounds(panel: np.ndarray, bounds, color, thickness: int, *, inflate: float = 0.0):
    bounds = bounds.inflated(inflate)
    p0 = _world_to_panel((bounds.xmin, bounds.ymax), panel.shape)
    p1 = _world_to_panel((bounds.xmax, bounds.ymin), panel.shape)
    cv2.rectangle(panel, p0, p1, color, thickness)


def _top_down_panel(
    height: int,
    robot_xy: tuple[float, float],
    can_xy: tuple[float, float],
) -> np.ndarray:
    panel = np.full((height, 430, 3), (24, 28, 34), dtype=np.uint8)
    cv2.putText(panel, "COLLISION / CLEARANCE MAP", (22, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (245, 245, 245), 2)

    for obstacle in navigation_obstacle_footprints():
        _draw_bounds(panel, obstacle, (72, 82, 92), -1)
        _draw_bounds(panel, obstacle, (88, 170, 255), 2, inflate=BASE_CLEARANCE_INFLATION)

    _draw_bounds(panel, PICK_REGION, (75, 210, 120), 2)
    _draw_bounds(panel, PLACE_REGION, (205, 150, 65), 2)
    pick_xy = STAGING_POSES["pick"][:2]
    place_xy = STAGING_POSES["place"][:2]
    cv2.line(
        panel,
        _world_to_panel(pick_xy, panel.shape),
        _world_to_panel(place_xy, panel.shape),
        (230, 230, 230),
        2,
    )

    pixels_per_metre = (panel.shape[1] - 70) / 3.70
    robot_radius_px = max(4, int(ROBOT_BASE_RADIUS * pixels_per_metre))
    cv2.circle(panel, _world_to_panel(robot_xy, panel.shape), robot_radius_px, (255, 210, 60), -1)
    cv2.circle(panel, _world_to_panel(can_xy, panel.shape), 5, (60, 75, 235), -1)

    legend_y = height - 72
    cv2.putText(panel, "gray: collision proxy", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 195, 200), 1)
    cv2.putText(panel, "orange: 0.35 m inflated boundary", (20, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (88, 170, 255), 1)
    cv2.putText(panel, "yellow: robot   red: can", (20, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1)
    return panel


def _annotate(
    rgb: np.ndarray,
    phase: str,
    robot_xy: tuple[float, float],
    can_xyz: tuple[float, float, float],
) -> np.ndarray:
    rgb = np.ascontiguousarray(rgb[:, :, :3])
    cv2.rectangle(rgb, (18, 18), (760, 105), (16, 20, 26), -1)
    cv2.putText(rgb, "ROBOLAB MILESTONE 1 | PHYSICS TRIAL", (36, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2)
    cv2.putText(rgb, phase, (36, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (80, 215, 255), 2)
    cv2.putText(
        rgb,
        f"base=({robot_xy[0]:.2f}, {robot_xy[1]:.2f}) m   can z={can_xyz[2]:.3f} m",
        (36, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (215, 220, 225),
        1,
    )
    return np.concatenate((rgb, _top_down_panel(rgb.shape[0], robot_xy, can_xyz[:2])), axis=1)


def main() -> None:
    env_cfg_class, _ = register_genmanip_kitchen_mobile_env()
    env_cfg = env_cfg_class()
    # Cameras render through Fabric in Isaac Sim 5.0.  The physics tests keep
    # Fabric disabled so they can inspect authored USD APIs, but recordings
    # need it enabled to synchronize moving articulation transforms to RTX.
    env_cfg.sim.use_fabric = True
    env_cfg.scene.trial_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/trial_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            focus_distance=3.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, 30.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.75, 1.65, 1.55),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )

    env, _ = create_env(env_cfg)
    output_path = os.path.abspath(args_cli.output)
    still_path = os.path.abspath(args_cli.still_output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(still_path), exist_ok=True)
    video = VideoWriter(output_path, fps=VIDEO_FPS)

    try:
        env.reset_eval_state()
        env.reset()
        camera = env.scene["trial_camera"]
        camera.set_world_poses_from_view(
            eyes=torch.tensor([[0.75, 1.65, 1.55]], device=env.device),
            targets=torch.tensor([[0.80, 0.05, 0.60]], device=env.device),
        )
        action = _hold_action(env)
        slices = _action_slices(env)
        goal = torch.tensor(STAGING_POSES["place"][:2], device=env.device)
        expected_can_z = ISLAND_SURFACE_Z + CAN_HALF_HEIGHT
        reached = False
        arrival_step = None
        final_frame = None

        total_steps = SETTLE_STEPS + MAX_TRAVERSAL_STEPS + FINAL_HOLD_STEPS
        for step in range(total_steps):
            phase = "RESET + GRAVITY SETTLE"
            command = action.clone()
            if step >= SETTLE_STEPS and not reached:
                phase = "BASE TRAVERSAL: PICK STAGE -> COUNTER STAGE"
                root_xy = env.scene["robot"].data.root_link_pos_w[0, :2] - env.scene.env_origins[0, :2]
                delta = goal - root_xy
                distance = torch.linalg.vector_norm(delta)
                if distance <= 0.02:
                    reached = True
                    arrival_step = step
                else:
                    # Use a brisk but still conservative demo speed.  The
                    # registered action term remains bounded at 0.6 m/s.
                    speed = min(0.40, 1.8 * float(distance))
                    command[0, slices["base"].start : slices["base"].start + 2] = delta / distance * speed
            if reached:
                phase = "ARRIVED AT COUNTER STAGING POSE"

            _, _, terminated, truncated, _ = env.step(command)
            if bool(torch.any(terminated | truncated)):
                raise RuntimeError("Trial terminated before traversal completed")

            robot_xy_tensor = env.scene["robot"].data.root_link_pos_w[0, :2] - env.scene.env_origins[0, :2]
            can_xyz_tensor = env.scene["can"].data.root_pos_w[0] - env.scene.env_origins[0]
            robot_xy = tuple(float(value) for value in robot_xy_tensor)
            can_xyz = tuple(float(value) for value in can_xyz_tensor)
            if not base_pose_is_clear(robot_xy):
                raise RuntimeError(f"Base entered inflated obstacle at {robot_xy}")
            if step >= SETTLE_STEPS and abs(can_xyz[2] - expected_can_z) > 0.008:
                raise RuntimeError(f"Can lost support during traversal: z={can_xyz[2]:.4f}")

            rgb = camera.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
            final_frame = _annotate(rgb, phase, robot_xy, can_xyz)
            video.write(final_frame)
            if arrival_step is not None and step - arrival_step >= FINAL_HOLD_STEPS:
                break

        final_distance = torch.linalg.vector_norm(robot_xy_tensor - goal).item()
        if not reached or final_distance >= 0.03:
            raise RuntimeError(f"Base did not reach counter stage; remaining={final_distance:.4f} m")
        if final_frame is None or not cv2.imwrite(still_path, cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not save final frame: {still_path}")
        print(f"Saved physics trial: {output_path}")
        print(f"Saved final frame: {still_path}")
        print(f"Final base distance to stage: {final_distance:.4f} m")
    finally:
        video.release()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
