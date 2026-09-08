#!/usr/bin/env python3
"""Run the fixed-base CuRobo grasp and local-place fidelity gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: F401 -- load before Isaac Lab/Kit

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=40)
parser.add_argument(
    "--trial-indices",
    type=str,
    default=None,
    help="Comma-separated schedule indices for targeted diagnostics (0--39).",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--num-seeds", type=int, default=8)
parser.add_argument(
    "--output",
    default="output/genmanip_kitchen/fixed_base_curobo/gate_40.json",
)
parser.add_argument(
    "--record-data",
    action="store_true",
    help="Enable RoboLab's HDF5 recorder (off for the gate).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

from robolab.core.environments.config import parse_env_cfg  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.kinova_mobile import (  # noqa: E402
    ENV_NAME,
    register_genmanip_kitchen_mobile_env,
)
from robolab.tasks.mobile_manipulation.curobo_fixed_base_planner import (  # noqa: E402
    CuroboFixedBasePlanner,
)
from robolab.tasks.mobile_manipulation.fixed_base_curobo_oracle import (  # noqa: E402
    FixedBaseCuroboPrimitive,
)
from robolab.tasks.mobile_manipulation.fixed_base_manipulation import (  # noqa: E402
    ARM_JOINT_NAMES,
    FIXED_BASE_ISLAND_STAGE,
    fixed_base_trial_schedule,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (  # noqa: E402
    CAN_INITIAL_Z,
)


def _action_slices(env) -> dict[str, slice]:
    output = {}
    offset = 0
    for name, size in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
        output[name] = slice(offset, offset + size)
        offset += size
    return output


def _hold_action(env, *, closed: bool = False) -> torch.Tensor:
    action = torch.zeros(
        (1, env.action_manager.total_action_dim),
        dtype=torch.float32,
        device=env.device,
    )
    slices = _action_slices(env)
    robot = env.scene["robot"]
    indices = [robot.data.joint_names.index(name) for name in ARM_JOINT_NAMES]
    action[0, slices["arm"]] = robot.data.joint_pos[0, indices]
    action[0, slices["gripper"]] = 1.0 if closed else 0.0
    return action


def _reset_trial(env, pick_xy: tuple[float, float], settle_steps: int) -> None:
    env.reset_eval_state()
    env.reset()
    robot = env.scene["robot"]
    can = env.scene["can"]
    x, y, z, yaw = FIXED_BASE_ISLAND_STAGE
    robot_pose = torch.tensor(
        [[x, y, z, math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]],
        dtype=torch.float32,
        device=env.device,
    )
    robot_pose[:, :3] += env.scene.env_origins
    robot.write_root_pose_to_sim(robot_pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)

    can_pose = torch.tensor(
        [[pick_xy[0], pick_xy[1], CAN_INITIAL_Z, 1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=env.device,
    )
    can_pose[:, :3] += env.scene.env_origins
    can.write_root_pose_to_sim(can_pose)
    can.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))

    action = _hold_action(env)
    for _ in range(settle_steps):
        _, _, terminated, truncated, _ = env.step(action)
        if bool(terminated[0].item() or truncated[0].item()):
            raise RuntimeError("environment terminated while settling reset")


def _summary(results: list[dict], requested: int) -> dict:
    grasps = sum(bool(item["grasp_success"]) for item in results)
    placements = sum(bool(item["place_success"]) for item in results)
    full_gate = requested == 40 and len(results) == 40
    return {
        "backend": "curobo_v1_ebb7170_via_planner_neutral_api",
        "grasp_mode": "bilateral_contact_then_post_contact_constraint",
        "requested_trials": requested,
        "completed_trials": len(results),
        "grasp_successes": grasps,
        "placement_successes": placements,
        "exit_threshold": 35,
        "full_40_trial_gate": full_gate,
        "passed": full_gate and grasps >= 35 and placements >= 35,
        "results": results,
    }


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    if args_cli.trials < 1 or args_cli.trials > 40:
        raise ValueError("--trials must be between 1 and 40")
    if args_cli.settle_steps < 1:
        raise ValueError("--settle-steps must be positive")
    torch.manual_seed(args_cli.seed)
    register_genmanip_kitchen_mobile_env()
    env_cfg = parse_env_cfg(
        ENV_NAME,
        seed=args_cli.seed,
        num_envs=1,
        env_spacing=8.0,
        use_fabric=False,
    )
    if not args_cli.enable_cameras:
        env_cfg.scene.wrist_cam = None
        env_cfg.scene.navigation_camera = None
    if not args_cli.record_data:
        env_cfg.recorders = None
    env, _ = create_env(env_cfg)
    output_path = Path(args_cli.output)
    try:
        planner = CuroboFixedBasePlanner(
            FIXED_BASE_ISLAND_STAGE, num_seeds=args_cli.num_seeds
        )
        primitive = FixedBaseCuroboPrimitive(env, planner)
        results = []
        full_schedule = fixed_base_trial_schedule(40)
        if args_cli.trial_indices:
            selected_indices = tuple(
                int(value.strip()) for value in args_cli.trial_indices.split(",")
            )
            if not selected_indices or any(index < 0 or index >= 40 for index in selected_indices):
                raise ValueError("--trial-indices values must be between 0 and 39")
            if len(set(selected_indices)) != len(selected_indices):
                raise ValueError("--trial-indices must not contain duplicates")
            selected_trials = tuple(full_schedule[index] for index in selected_indices)
        elif args_cli.trials == 1:
            selected_trials = full_schedule[:1]
        else:
            selected_trials = tuple(
                full_schedule[round(index * 39 / (args_cli.trials - 1))]
                for index in range(args_cli.trials)
            )
        requested_trials = len(selected_trials)
        for trial in selected_trials:
            print(
                json.dumps(
                    {
                        "trial": trial.index,
                        "scale": trial.scale,
                        "pick_xy": trial.pick_xy,
                        "place_xy": trial.place_xy,
                    }
                ),
                flush=True,
            )
            try:
                _reset_trial(env, trial.pick_xy, args_cli.settle_steps)
                item = primitive.run(pick_xy=trial.pick_xy, place_xy=trial.place_xy).as_dict()
            except Exception as error:
                item = {
                    "grasp_success": False,
                    "place_success": False,
                    "failure_reason": str(error),
                    "phase_history": ["reset", "failure"],
                    "pick_xy": list(trial.pick_xy),
                    "place_xy": list(trial.place_xy),
                }
            item.update({"trial": trial.index, "scale": trial.scale})
            results.append(item)
            report = _summary(results, requested_trials)
            _write_report(output_path, report)
            print(json.dumps(item), flush=True)
        report = _summary(results, requested_trials)
        summary = {key: value for key, value in report.items() if key != "results"}
        print(json.dumps(summary, indent=2))
        if args_cli.trial_indices is None and args_cli.trials == 40:
            return 0 if report["passed"] else 1
        return 0 if all(item["place_success"] for item in results) else 1
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
