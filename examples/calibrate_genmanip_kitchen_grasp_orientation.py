# SPDX-License-Identifier: Apache-2.0

"""Batch-search the free wrist yaw for a top-down kitchen pre-grasp."""

import argparse
import json
import math
import os

import cv2  # noqa: F401 -- import before Isaac Lab

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300)
parser.add_argument(
    "--output",
    default="output/genmanip_kitchen/milestone2/grasp_orientation_calibration.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402

from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.kinova_mobile import (  # noqa: E402
    IK_ENV_NAME,
    register_genmanip_kitchen_mobile_ik_env,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (  # noqa: E402
    PREGRASP_CLEARANCE,
)
from robolab.tasks.mobile_manipulation.kitchen_manipulation_oracle import (  # noqa: E402
    KitchenManipulationOracle,
)


def main() -> int:
    axis_degrees = tuple(range(0, 180, 22))[:8]
    register_genmanip_kitchen_mobile_ik_env()
    env, _ = create_env(
        IK_ENV_NAME,
        num_envs=len(axis_degrees),
        env_spacing=8.0,
        seed=0,
        use_fabric=False,
    )
    try:
        env.reset_eval_state()
        env.reset()
        oracle = KitchenManipulationOracle(env)
        target_tcp_w = env.scene["can"].data.root_pos_w.clone()
        target_tcp_w[:, 2] += PREGRASP_CLEARANCE
        _, approach_quat_w = oracle.current_tcp_w()
        target_quat_w = torch.zeros((env.num_envs, 4), device=env.device)
        for env_id, degrees in enumerate(axis_degrees):
            angle = math.radians(degrees)
            target_quat_w[env_id] = torch.tensor(
                (0.0, math.cos(angle), math.sin(angle), 0.0),
                device=env.device,
            )

        action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim), device=env.device
        )
        # First enter the known position-reachable basin without asking IK to
        # perform the wrist flip at the same time.
        for _ in range(args_cli.steps):
            action[:, oracle.action_slices["arm"]] = (
                oracle._tcp_world_to_flange_root(target_tcp_w, approach_quat_w)
            )
            env.step(action)
        seed_tcp_pos_w, seed_tcp_quat_w = oracle.current_tcp_w()
        seed_position_error = torch.linalg.vector_norm(
            seed_tcp_pos_w - target_tcp_w, dim=1
        )
        seed_orientation_error = math_utils.quat_error_magnitude(
            seed_tcp_quat_w, approach_quat_w
        )
        for _ in range(args_cli.steps):
            action[:, oracle.action_slices["arm"]] = (
                oracle._tcp_world_to_flange_root(target_tcp_w, target_quat_w)
            )
            env.step(action)

        tcp_pos_w, tcp_quat_w = oracle.current_tcp_w()
        position_error = torch.linalg.vector_norm(tcp_pos_w - target_tcp_w, dim=1)
        orientation_error = math_utils.quat_error_magnitude(
            tcp_quat_w, target_quat_w
        )
        base_pos = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
        results = []
        for env_id, degrees in enumerate(axis_degrees):
            results.append(
                {
                    "axis_degrees": degrees,
                    "target_quat_wxyz": target_quat_w[env_id].cpu().tolist(),
                    "seed_position_error_m": float(
                        seed_position_error[env_id].item()
                    ),
                    "seed_orientation_error_rad": float(
                        seed_orientation_error[env_id].item()
                    ),
                    "position_error_m": float(position_error[env_id].item()),
                    "orientation_error_rad": float(orientation_error[env_id].item()),
                    "tcp_xyz": (
                        tcp_pos_w[env_id] - env.scene.env_origins[env_id]
                    ).cpu().tolist(),
                    "base_xyz": base_pos[env_id].cpu().tolist(),
                }
            )
        results.sort(
            key=lambda item: item["position_error_m"]
            + 0.1 * item["orientation_error_rad"]
        )
        report = {"steps": args_cli.steps, "results_best_first": results}
        output = os.path.abspath(args_cli.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2), flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
