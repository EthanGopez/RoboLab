# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify positional Kinova reach for the Milestone-0 kitchen task regions."""

import argparse
import copy
import json
import os

import cv2  # noqa: F401 -- import before Isaac Lab
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--output", default="output/genmanip_kitchen/milestone0_audit/reachability.json")
parser.add_argument("--steps-per-target", type=int, default=240)
parser.add_argument(
    "--eef-clearance",
    type=float,
    default=0.05,
    help="Controlled-flange clearance above the can center in metres.",
)
parser.add_argument("--region", choices=("pick", "place", "both"), default="both")
parser.add_argument("--center-only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402
import isaaclab.envs.mdp as mdp  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg  # noqa: E402
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from robolab.robots.kinova_mobile_cobot import KinovaMobileCobotCfg  # noqa: E402


BASE_POSES = {
    "pick": (0.50, 0.175, 0.468),
    "place": (1.12, 0.10, 0.468),
}

REGIONS = {
    "pick": {
        "bounds_xy": (-0.06, 0.04, 0.10, 0.25),
        "surface_z": 1.0450,
    },
    "place": {
        "bounds_xy": (1.76, 1.86, -0.02, 0.22),
        "surface_z": 0.8453,
    },
}

CAN_HALF_HEIGHT = 0.0510185
POSITION_TOLERANCE = 0.015


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    robot = copy.deepcopy(KinovaMobileCobotCfg().robot)
    robot.spawn.articulation_props.fix_root_link = True
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)


@configclass
class EmptyObservationsCfg:
    pass


@configclass
class ReachActionsCfg:
    arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint_[1-7]"],
        body_name="robotiq_85_base_link",
        controller=DifferentialIKControllerCfg(
            command_type="position", use_relative_mode=False, ik_method="dls"
        ),
        scale=1.0,
    )


@configclass
class ReachEventsCfg:
    reset = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class ReachEnvCfg(ManagerBasedEnvCfg):
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
    observations: EmptyObservationsCfg = EmptyObservationsCfg()
    actions: ReachActionsCfg = ReachActionsCfg()
    events: ReachEventsCfg = ReachEventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.use_fabric = False


def _targets(region: dict) -> list[tuple[str, tuple[float, float, float]]]:
    xmin, xmax, ymin, ymax = region["bounds_xy"]
    z = region["surface_z"] + CAN_HALF_HEIGHT + args_cli.eef_clearance
    targets = [
        ("center", ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, z)),
        ("corner_xmin_ymin", (xmin, ymin, z)),
        ("corner_xmin_ymax", (xmin, ymax, z)),
        ("corner_xmax_ymin", (xmax, ymin, z)),
        ("corner_xmax_ymax", (xmax, ymax, z)),
    ]
    return targets[:1] if args_cli.center_only else targets


def _run_region(name: str) -> dict:
    cfg = ReachEnvCfg()
    cfg.scene.robot.init_state.pos = BASE_POSES[name]
    cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    env = ManagerBasedEnv(cfg=cfg)
    env.reset()
    robot = env.scene["robot"]
    body_index = robot.data.body_names.index("robotiq_85_base_link")
    arm_indices = [robot.data.joint_names.index(f"joint_{index}") for index in range(1, 8)]
    results = []
    try:
        for target_name, target in _targets(REGIONS[name]):
            world_target = torch.tensor(target, dtype=torch.float32, device=env.device)
            robot_root_position = torch.tensor(BASE_POSES[name], dtype=torch.float32, device=env.device)
            command = (world_target - robot_root_position).unsqueeze(0)
            for _ in range(args_cli.steps_per_target):
                env.step(command)
            actual = robot.data.body_pos_w[0, body_index] - env.scene.env_origins[0]
            error = torch.linalg.vector_norm(actual - world_target).item()
            joint_pos = robot.data.joint_pos[0, arm_indices]
            limits = robot.data.soft_joint_pos_limits[0, arm_indices]
            lower_margin = torch.min(joint_pos - limits[:, 0]).item()
            upper_margin = torch.min(limits[:, 1] - joint_pos).item()
            finite = bool(torch.isfinite(joint_pos).all() and torch.isfinite(actual).all())
            results.append(
                {
                    "name": target_name,
                    "target_xyz": list(target),
                    "command_xyz_robot_root": command[0].detach().cpu().tolist(),
                    "actual_xyz": actual.tolist(),
                    "arm_joint_positions_rad": joint_pos.detach().cpu().tolist(),
                    "position_error_m": error,
                    "finite": finite,
                    "min_lower_joint_margin_rad": lower_margin,
                    "min_upper_joint_margin_rad": upper_margin,
                    "success": finite and error <= POSITION_TOLERANCE,
                }
            )
    finally:
        env.close()
    return {
        "base_pose_xyz": list(BASE_POSES[name]),
        "base_yaw_rad": 0.0,
        "region": REGIONS[name],
        "position_tolerance_m": POSITION_TOLERANCE,
        "targets": results,
        "all_targets_reached": all(item["success"] for item in results),
    }


def main() -> None:
    region_names = ("pick", "place") if args_cli.region == "both" else (args_cli.region,)
    report = {
        "method": "Isaac Lab DifferentialInverseKinematicsAction, absolute position, DLS",
        "controlled_body": "robotiq_85_base_link",
        "can_half_height_m": CAN_HALF_HEIGHT,
        "eef_clearance_above_can_center_m": args_cli.eef_clearance,
        "regions": {name: _run_region(name) for name in region_names},
    }
    output = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved reachability report: {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
