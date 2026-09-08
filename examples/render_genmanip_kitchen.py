# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the authored EBench/GenManip dishwasher kitchen through Isaac Lab."""

import argparse
import copy
import os
import traceback

import cv2  # noqa: F401  must be imported before isaaclab
from isaaclab.app import AppLauncher

# GenManip registers both its shared material library and the scene-local
# generated modules before Kit starts. Mirror that contract here so Isaac's
# MDL compiler resolves KooPbr and KooPbr_maps instead of using fallbacks.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EBENCH_ASSETS = os.path.join(_REPO_ROOT, "assets", "third_party", "ebench", "assets")
_SHARED_MDL_DIR = os.path.join(_EBENCH_ASSETS, "miscs", "mdl", "ebench", "mdl")
_SCENE_MDL_DIR = os.path.join(
    _EBENCH_ASSETS,
    "scene_usds",
    "ebench",
    "long_horizon",
    "scene_dishwasher",
    "SubUSDs",
    "materials",
)
_mdl_search_paths = ["/isaac-sim/materials/", _SHARED_MDL_DIR, _SCENE_MDL_DIR]
if os.environ.get("MDL_SYSTEM_PATH"):
    _mdl_search_paths.append(os.environ["MDL_SYSTEM_PATH"])
os.environ["MDL_SYSTEM_PATH"] = os.pathsep.join(_mdl_search_paths)


parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", type=int, default=240)
parser.add_argument(
    "--output",
    type=str,
    default="output/genmanip_kitchen/genmanip_kitchen_robolab.mp4",
)
parser.add_argument("--can-pose", choices=("none", "pick", "place"), default="none")
parser.add_argument("--still-output", type=str, default=None)
parser.add_argument("--show-cobot", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from robolab.constants import PACKAGE_DIR  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.robots.kinova_mobile_cobot import KinovaMobileCobotCfg  # noqa: E402


KITCHEN_DIR = os.path.join(
    PACKAGE_DIR,
    "assets",
    "third_party",
    "ebench",
    "assets",
    "scene_usds",
    "ebench",
    "long_horizon",
    "scene_dishwasher",
)
KITCHEN_USD = os.path.join(KITCHEN_DIR, "scene.usd")
CAN_USD = os.path.join(PACKAGE_DIR, "assets", "objects", "ycb", "tomato_soup_can.usd")
CAN_POSES = {
    # Region centers from audit_genmanip_kitchen_milestone0.py, with the
    # measured YCB can half-height added to the corresponding work surface.
    "pick": (-0.01, 0.175, 1.0960185),
    "place": (1.81, 0.10, 0.8963185),
}
BASE_POSES = {
    "pick": (0.50, 0.175, 0.468),
    "place": (1.12, 0.10, 0.468),
}
CAMERA_TARGETS = {
    "none": (1.25, 0.20, 0.90),
    "pick": (0.15, 0.175, 0.95),
    "place": (1.55, 0.10, 0.85),
}


@configclass
class GenManipKitchenSceneCfg(InteractiveSceneCfg):
    """The complete authored scene, with its original transforms and lighting."""

    kitchen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/kitchen",
        spawn=sim_utils.UsdFileCfg(usd_path=KITCHEN_USD),
    )
    # The published dishwasher USD has the fixtures and floor, but no room
    # shell. Leave the camera side open and enclose the other three sides.
    back_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/back_wall",
        spawn=sim_utils.CuboidCfg(
            size=(5.6, 0.12, 3.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.46, 0.43, 0.39), roughness=0.85
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 2.55, 1.5)),
    )
    left_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/left_wall",
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 5.4, 3.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.46, 0.43, 0.39), roughness=0.85
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.8, -0.15, 1.5)),
    )
    right_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/right_wall",
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 5.4, 3.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.46, 0.43, 0.39), roughness=0.85
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(3.8, -0.15, 1.5)),
    )
    if args_cli.can_pose != "none":
        can = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/tomato_soup_can",
            spawn=sim_utils.UsdFileCfg(
                usd_path=CAN_USD,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True, kinematic_enabled=True
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=CAN_POSES[args_cli.can_pose]),
        )
    if args_cli.show_cobot:
        robot = copy.deepcopy(KinovaMobileCobotCfg().robot)
        # Use the audited staging pose for task views. Keep the earlier room
        # overview pose only when no task pose was requested.
        robot.init_state.pos = BASE_POSES.get(args_cli.can_pose, (1.0, -0.45, 0.468))
        robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        robot.spawn.articulation_props.fix_root_link = True
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/render_camera",
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=3.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, 30.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.2, -2.7, 1.75),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )


def camera_eye(step: int, total_steps: int, device: str) -> torch.Tensor:
    """Follow a short interior-safe arc facing the authored task workspace."""

    progress = 0.5 if args_cli.still_output else step / max(1, total_steps - 1)
    angle = np.deg2rad(-98.0 + 16.0 * progress)
    radius = 2.70 - 0.10 * np.sin(np.pi * progress)
    eye = (
        0.35 + radius * np.cos(angle),
        radius * np.sin(angle),
        1.85 + 0.10 * np.sin(np.pi * progress),
    )
    return torch.tensor([eye], dtype=torch.float32, device=device)


def annotate(frame: np.ndarray, step: int, total_steps: int) -> np.ndarray:
    frame = np.ascontiguousarray(frame[:, :, :3])
    cv2.rectangle(frame, (20, 20), (625, 100), (16, 20, 26), -1)
    cv2.putText(
        frame,
        "GENMANIP KITCHEN  |  ROBOLAB ISAAC LAB",
        (38, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
    )
    pose_label = {
        "pick": "PICK POSE - CAN ON ISLAND",
        "place": "PLACE POSE - CAN ON OPPOSING COUNTER",
    }.get(args_cli.can_pose, "Authored 3D room geometry - original scene scale")
    if args_cli.show_cobot:
        pose_label = f"{pose_label} | KINOVA STAGED AT AUTHORED SCALE"
    cv2.putText(
        frame,
        pose_label,
        (38, 83),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (80, 215, 255),
        2,
    )
    progress = int((frame.shape[1] - 80) * (step + 1) / total_steps)
    cv2.rectangle(
        frame,
        (40, frame.shape[0] - 28),
        (frame.shape[1] - 40, frame.shape[0] - 18),
        (55, 60, 66),
        -1,
    )
    cv2.rectangle(
        frame,
        (40, frame.shape[0] - 28),
        (40 + progress, frame.shape[0] - 18),
        (80, 215, 255),
        -1,
    )
    return frame


def main() -> None:
    if not os.path.isfile(KITCHEN_USD):
        raise FileNotFoundError(f"GenManip kitchen scene not found: {KITCHEN_USD}")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1, use_fabric=False)
    )
    scene = InteractiveScene(
        GenManipKitchenSceneCfg(
            num_envs=1,
            env_spacing=8.0,
            replicate_physics=False,
            lazy_sensor_update=False,
        )
    )
    sim.reset()
    scene.reset()
    camera = scene["camera"]
    target = torch.tensor(
        [CAMERA_TARGETS[args_cli.can_pose]], dtype=torch.float32, device=camera.device
    )
    output_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video = VideoWriter(output_path, fps=30)

    try:
        for _ in range(8):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        for step in range(args_cli.num_steps):
            camera.set_world_poses_from_view(
                eyes=camera_eye(step, args_cli.num_steps, camera.device),
                targets=target,
            )
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

            rgb = camera.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
            if step == 0:
                depth = camera.data.output["distance_to_image_plane"][0]
                rgb_range = int(np.ptp(rgb[:, :, :3]))
                valid_depth = int(torch.count_nonzero(torch.isfinite(depth) & (depth > 0.0)))
                if rgb_range < 8 or valid_depth == 0:
                    raise RuntimeError(
                        f"Invalid kitchen render: rgb_range={rgb_range}, valid_depth={valid_depth}"
                    )
                print(
                    f"Kitchen validation: rgb_range={rgb_range}, valid_depth={valid_depth}",
                    flush=True,
                )
            rendered = annotate(rgb, step, args_cli.num_steps)
            video.write(rendered)
            if args_cli.still_output and step == args_cli.num_steps - 1:
                still_path = os.path.abspath(args_cli.still_output)
                os.makedirs(os.path.dirname(still_path), exist_ok=True)
                if not cv2.imwrite(still_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"Failed to save still image: {still_path}")
                print(f"Saved kitchen still: {still_path}", flush=True)

        print(f"Kitchen asset: {KITCHEN_USD}", flush=True)
        print(f"Saved kitchen video: {output_path}", flush=True)
    finally:
        video.release()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Terminated with error: {error}", flush=True)
        traceback.print_exc()
        os._exit(1)
    # The imported scene leaves USD/MDL resource work active, which makes
    # Isaac 5.0's plugin shutdown stall after the MP4 has already been flushed.
    os._exit(0)
