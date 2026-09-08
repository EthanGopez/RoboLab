# SPDX-License-Identifier: Apache-2.0

"""Run the Milestone-1 reset-and-settle gate for the GenManip kitchen."""

import argparse
import json
import os

import cv2  # noqa: F401 -- import before Isaac Lab
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=100)
parser.add_argument("--num-envs", type=int, default=10)
parser.add_argument("--settle-steps", type=int, default=90)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--output",
    default="output/genmanip_kitchen/milestone1/reset_and_settle.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.kinova_mobile import (  # noqa: E402
    ENV_NAME,
    register_genmanip_kitchen_mobile_env,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (  # noqa: E402
    CAN_HALF_HEIGHT,
    ISLAND_SURFACE_Z,
    PICK_REGION,
    ROBOT_BASE_Z,
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
    slices = _action_slices(env)
    robot = env.scene["robot"]
    arm_indices = [robot.data.joint_names.index(f"joint_{index}") for index in range(1, 8)]
    action[:, slices["arm"]] = robot.data.default_joint_pos[:, arm_indices]
    return action


def _failure(trial: int, check: str, value) -> dict:
    return {"trial": trial, "check": check, "value": value}


def main() -> int:
    if args_cli.trials < 1 or args_cli.num_envs < 1 or args_cli.settle_steps < 1:
        raise ValueError("--trials, --num-envs, and --settle-steps must be positive")

    register_genmanip_kitchen_mobile_env()
    batch_size = min(args_cli.trials, args_cli.num_envs)
    env, _ = create_env(
        ENV_NAME,
        num_envs=batch_size,
        env_spacing=8.0,
        seed=args_cli.seed,
        use_fabric=False,
    )
    expected_can_z = ISLAND_SURFACE_Z + CAN_HALF_HEIGHT
    failures = []
    completed = 0
    batch_index = 0

    try:
        action = _hold_action(env)
        while completed < args_cli.trials:
            active = min(batch_size, args_cli.trials - completed)
            torch.manual_seed(args_cli.seed + batch_index)
            env.reset_eval_state()
            env.reset()

            min_z = torch.full((batch_size,), float("inf"), device=env.device)
            max_z = torch.full((batch_size,), -float("inf"), device=env.device)
            finite_throughout = torch.ones(batch_size, dtype=torch.bool, device=env.device)
            terminated_early = torch.zeros(batch_size, dtype=torch.bool, device=env.device)

            for _ in range(args_cli.settle_steps):
                _, _, terminated, truncated, _ = env.step(action)
                can_pos = env.scene["can"].data.root_pos_w - env.scene.env_origins
                min_z = torch.minimum(min_z, can_pos[:, 2])
                max_z = torch.maximum(max_z, can_pos[:, 2])
                finite_throughout &= torch.isfinite(can_pos).all(dim=1)
                terminated_early |= terminated | truncated

            can = env.scene["can"]
            can_pos = can.data.root_pos_w - env.scene.env_origins
            can_linear_speed = torch.linalg.vector_norm(can.data.root_lin_vel_w, dim=1)
            can_angular_speed = torch.linalg.vector_norm(can.data.root_ang_vel_w, dim=1)
            robot_pos = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins

            for env_index in range(active):
                trial = completed + env_index
                xyz = can_pos[env_index].detach().cpu().tolist()
                xy = (float(can_pos[env_index, 0]), float(can_pos[env_index, 1]))
                if not bool(finite_throughout[env_index]) or not bool(torch.isfinite(robot_pos[env_index]).all()):
                    failures.append(_failure(trial, "finite_state", {"can": xyz}))
                if bool(terminated_early[env_index]):
                    failures.append(_failure(trial, "premature_termination", True))
                if not PICK_REGION.contains(xy, tolerance=0.005):
                    failures.append(_failure(trial, "can_xy_in_pick_region", xyz[:2]))
                if abs(float(can_pos[env_index, 2]) - expected_can_z) > 0.006:
                    failures.append(_failure(trial, "can_rest_height", xyz[2]))
                if float(min_z[env_index]) < expected_can_z - 0.003:
                    failures.append(_failure(trial, "can_penetration", float(min_z[env_index])))
                if float(max_z[env_index]) > expected_can_z + 0.03:
                    failures.append(_failure(trial, "can_flying", float(max_z[env_index])))
                if float(can_linear_speed[env_index]) >= 0.02:
                    failures.append(_failure(trial, "can_linear_speed", float(can_linear_speed[env_index])))
                if float(can_angular_speed[env_index]) >= 0.15:
                    failures.append(_failure(trial, "can_angular_speed", float(can_angular_speed[env_index])))
                if abs(float(robot_pos[env_index, 2]) - ROBOT_BASE_Z) >= 0.003:
                    failures.append(_failure(trial, "robot_spawn_height", float(robot_pos[env_index, 2])))

            completed += active
            batch_index += 1
            print(f"validated {completed}/{args_cli.trials} trials")
    finally:
        env.close()

    report = {
        "environment": ENV_NAME,
        "trials": args_cli.trials,
        "parallel_envs": batch_size,
        "settle_steps": args_cli.settle_steps,
        "seed": args_cli.seed,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }
    output_dir = os.path.dirname(args_cli.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args_cli.output, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
