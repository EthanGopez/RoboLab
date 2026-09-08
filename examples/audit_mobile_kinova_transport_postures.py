#!/usr/bin/env python3
"""Measure planar link envelopes for candidate mobile-Kinova transport postures."""

import argparse
import copy
import math
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from robolab.robots.kinova_mobile_cobot import KinovaMobileCobotCfg  # noqa: E402


CANDIDATES = {
    "fixed_home": (0.0, 0.26, math.pi, -2.27, 0.0, 0.96, math.pi / 2.0),
    "shoulder_back": (0.0, -0.70, math.pi, -2.50, 0.0, 0.70, math.pi / 2.0),
    "shoulder_forward": (0.0, 0.70, math.pi, -2.50, 0.0, 0.70, math.pi / 2.0),
    "limit_safe_fold": (0.0, -0.35, math.pi, -2.50, 0.0, 0.35, math.pi / 2.0),
}


@configclass
class PostureAuditSceneCfg(InteractiveSceneCfg):
    robot = copy.deepcopy(KinovaMobileCobotCfg().robot)


def main() -> None:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=10, use_fabric=False)
    )
    scene = InteractiveScene(
        PostureAuditSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
    )
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    arm_indices = [robot.data.joint_names.index(f"joint_{index}") for index in range(1, 8)]
    root_xy = robot.data.root_link_pos_w[0, :2].clone()
    for name, values in CANDIDATES.items():
        position = robot.data.default_joint_pos.clone()
        velocity = torch.zeros_like(position)
        position[0, arm_indices] = torch.tensor(values, device=robot.device)
        robot.write_joint_state_to_sim(position, velocity)
        for _ in range(3):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
        relative_xy = robot.data.body_pos_w[0, :, :2] - root_xy
        radius = torch.linalg.vector_norm(relative_xy, dim=1)
        maximum, index = torch.max(radius, dim=0)
        flange_index = robot.data.body_names.index("robotiq_85_base_link")
        print(
            name,
            "max_radius_m=", round(float(maximum), 4),
            "outer_body=", robot.data.body_names[int(index)],
            "flange_xyz_m=", [round(float(value), 4) for value in (robot.data.body_pos_w[0, flange_index] - robot.data.root_link_pos_w[0])],
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
