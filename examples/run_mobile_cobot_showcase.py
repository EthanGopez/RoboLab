# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a short CoBot MoMa omnidirectional + arm/gripper showcase."""

import argparse
import copy
import os
import sys
import traceback

import cv2  # noqa: F401  must be imported before isaaclab
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", type=int, default=240)
parser.add_argument("--output", type=str, default="output/mobile_cobot_showcase/omnidirectional_cobot_moma.mp4")
parser.add_argument("--debug-state", action="store_true", help="Print root/body poses while rendering.")
parser.add_argument(
    "--room",
    choices=("empty", "kitchen"),
    default="empty",
    help="Render on an empty floor or in RoboLab's existing U-shaped kitchen.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.envs.mdp as mdp  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.constants import FIXTURE_DIR  # noqa: E402
from robolab.robots.kinova_mobile_cobot import (  # noqa: E402
    KinovaMobileCobotCfg,
    KinovaMobileJointPositionActionCfg,
)


@configclass
class ShowcaseSceneCfg(KinovaMobileCobotCfg, InteractiveSceneCfg):
    if args_cli.room == "kitchen":
        robot = copy.deepcopy(KinovaMobileCobotCfg().robot)
        # The fixture occupies x=[-0.36, 2.19], y=[-3.02, 0.41]. Start on
        # the open floor just outside the counter footprint.
        robot.init_state.pos = (2.55, 0.75, 0.468)

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        debug_vis=False,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.82, 0.86), intensity=1100.0),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.92, 0.82), intensity=1800.0, angle=0.45),
    )
    if args_cli.room == "kitchen":
        room = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/room",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(FIXTURE_DIR, "kitchen_u_shaped.usd"),
            ),
        )
    showcase_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/showcase_camera",
        height=540,
        width=960,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=4.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, 50.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            # Three-quarter view keeps the full arm and both wheel pairs legible
            # while the platform demonstrates forward, lateral, and yaw motion.
            pos=(5.0, 4.0, 3.0) if args_cli.room == "kitchen" else (2.6, 2.6, 2.1),
            rot=(0.2737301, 0.1837697, 0.5262260, 0.7838282)
            if args_cli.room == "kitchen"
            else (0.3145670, 0.2179317, 0.5261337, 0.7594319),
            convention="opengl",
        ),
    )


@configclass
class EmptyObservationsCfg:
    pass


@configclass
class ShowcaseEventCfg:
    reset = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class ShowcaseEnvCfg(ManagerBasedEnvCfg):
    scene: ShowcaseSceneCfg = ShowcaseSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=True)
    observations: EmptyObservationsCfg = EmptyObservationsCfg()
    actions: KinovaMobileJointPositionActionCfg = KinovaMobileJointPositionActionCfg()
    events: ShowcaseEventCfg = ShowcaseEventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.use_fabric = True
        self.viewer.eye = (3.2, 3.2, 2.3)
        self.viewer.lookat = (0.0, 0.0, 0.65)


PHASES = (
    "FORWARD",
    "STRAFE LEFT",
    "ROTATE IN PLACE",
    "DIAGONAL",
    "ARM + GRIPPER",
)


def base_command(step: int, total_steps: int) -> tuple[torch.Tensor, str]:
    section = min(4, (5 * step) // max(1, total_steps))
    commands = (
        (0.42, 0.0, 0.0),
        (0.0, 0.42, 0.0),
        (0.0, 0.0, 0.75),
        (-0.30, -0.30, 0.0),
        (0.0, 0.0, 0.0),
    )
    return torch.tensor(commands[section]), PHASES[section]


def annotate(frame: np.ndarray, label: str, step: int, total_steps: int) -> np.ndarray:
    frame = np.ascontiguousarray(frame[:, :, :3])
    cv2.rectangle(frame, (18, 18), (430, 92), (18, 24, 30), -1)
    cv2.putText(frame, "ROBOLAB  |  COBOT MOMA", (34, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2)
    cv2.putText(frame, label, (34, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 215, 255), 2)
    progress = int(900 * (step + 1) / total_steps)
    cv2.rectangle(frame, (30, frame.shape[0] - 28), (930, frame.shape[0] - 18), (55, 60, 66), -1)
    cv2.rectangle(frame, (30, frame.shape[0] - 28), (30 + progress, frame.shape[0] - 18), (80, 215, 255), -1)
    return frame


def main() -> None:
    env = ManagerBasedEnv(cfg=ShowcaseEnvCfg())
    output_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video = VideoWriter(output_path, fps=30)

    try:
        env.reset()
        camera = env.scene["showcase_camera"]
        robot = env.scene["robot"]
        if args_cli.debug_state:
            print(f"Body names: {robot.data.body_names}", flush=True)
            print(f"Initial root pose: {robot.data.root_link_pose_w[0].tolist()}", flush=True)
            print(f"Initial body quaternions: {robot.data.body_quat_w[0].tolist()}", flush=True)
        arm_indices = [robot.data.joint_names.index(f"joint_{index}") for index in range(1, 8)]
        home = robot.data.default_joint_pos[0, arm_indices].clone()

        for step in range(args_cli.num_steps):
            phase = 2.0 * torch.pi * step / max(1, args_cli.num_steps)
            base, label = base_command(step, args_cli.num_steps)
            arm = home.clone()
            # The arm moves throughout, making it clear the full articulation is mobile.
            arm[0] += 0.28 * torch.sin(torch.as_tensor(phase, device=env.device))
            arm[5] += 0.20 * torch.sin(torch.as_tensor(2.0 * phase, device=env.device))
            gripper = 1.0 if (step // 30) % 2 else 0.0
            action = torch.cat(
                (
                    base.to(env.device),
                    arm,
                    torch.tensor([gripper], device=env.device),
                )
            ).unsqueeze(0)
            env.step(action)

            if args_cli.debug_state and step in (0, 1, args_cli.num_steps - 1):
                print(
                    f"Step {step} root pose: {robot.data.root_link_pose_w[0].tolist()}",
                    flush=True,
                )

            rgb = camera.data.output.get("rgb")
            if rgb is not None:
                frame = rgb[0].detach().cpu().numpy().astype(np.uint8)
                video.write(annotate(frame, label, step, args_cli.num_steps))

        final_xy = (robot.data.root_pos_w[0, :2] - env.scene.env_origins[0, :2]).tolist()
        print(f"Final base xy: {final_xy}")
        print(f"Saved showcase video: {output_path}")
    finally:
        video.release()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Terminated with error: {error}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
