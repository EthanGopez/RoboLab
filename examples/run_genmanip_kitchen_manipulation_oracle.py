# SPDX-License-Identifier: Apache-2.0

"""Run deterministic mobile-Kinova kitchen pick-carry-place trials."""

import argparse
import json
import os
import random

import cv2  # noqa: F401 -- import before Isaac Lab

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--max-steps", type=int, default=1800)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--trials", type=int, default=1)
parser.add_argument(
    "--randomize-place",
    action="store_true",
    help="Uniformly sample each trial's target inside the audited place region.",
)
grasp_mode = parser.add_mutually_exclusive_group()
grasp_mode.add_argument(
    "--grasp-assist",
    dest="grasp_assist",
    action="store_true",
    help="Use the supported post-contact constraint transport mode (default).",
)
grasp_mode.add_argument(
    "--unassisted-friction",
    dest="grasp_assist",
    action="store_false",
    help="Experimental: transport using fingertip friction only.",
)
parser.set_defaults(grasp_assist=True)
parser.add_argument(
    "--record-data",
    action="store_true",
    help="Enable RoboLab's full HDF5 state/action recorder (off for gates).",
)
parser.add_argument(
    "--output",
    default="output/genmanip_kitchen/milestone2/oracle_trial.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

from robolab.core.environments.config import parse_env_cfg  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.kinova_mobile import (  # noqa: E402
    IK_ENV_NAME,
    register_genmanip_kitchen_mobile_ik_env,
)
from robolab.core.world.world_state import get_world  # noqa: E402
from robolab.robots.kinova_gen3 import (  # noqa: E402
    GRIPPER_JOINT_COMMANDS,
    GRIPPER_JOINT_NAME,
)
from robolab.robots.kinova_mobile_cobot import (  # noqa: E402
    MOBILE_KINOVA_LEFT_FINGERTIP_BODY,
    MOBILE_KINOVA_RIGHT_FINGERTIP_BODY,
)
from robolab.tasks.mobile_manipulation.kitchen_manipulation_oracle import (  # noqa: E402
    KitchenManipulationOracle,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (  # noqa: E402
    ORACLE_PLACE_REGION,
)


def _snapshot(env, oracle, step: int) -> dict:
    world = get_world(env)
    robot = env.scene["robot"]
    can = env.scene["can"]
    can_pos = can.data.root_pos_w - env.scene.env_origins
    base_pos = robot.data.root_link_pos_w - env.scene.env_origins
    tcp_pos_w, tcp_quat_w = oracle.current_tcp_w()
    tcp_pos = tcp_pos_w - env.scene.env_origins
    left_index = robot.data.body_names.index(MOBILE_KINOVA_LEFT_FINGERTIP_BODY)
    right_index = robot.data.body_names.index(MOBILE_KINOVA_RIGHT_FINGERTIP_BODY)
    gripper_index = robot.data.joint_names.index(GRIPPER_JOINT_NAME)
    gripper_indices = [
        robot.data.joint_names.index(name) for name in GRIPPER_JOINT_COMMANDS
    ]
    left_pos = robot.data.body_pos_w[:, left_index] - env.scene.env_origins
    right_pos = robot.data.body_pos_w[:, right_index] - env.scene.env_origins
    left_quat = robot.data.body_quat_w[:, left_index]
    right_quat = robot.data.body_quat_w[:, right_index]
    left_force = world.get_contact_force("can", "left_fingertip")
    right_force = world.get_contact_force("can", "right_fingertip")
    return {
        "step": step,
        "phase": oracle.phase_names()[0],
        "can_xyz": can_pos[0].detach().cpu().tolist(),
        "can_quat_wxyz": can.data.root_quat_w[0].detach().cpu().tolist(),
        "can_linear_speed": float(
            torch.linalg.vector_norm(can.data.root_lin_vel_w[0]).item()
        ),
        "can_angular_speed": float(
            torch.linalg.vector_norm(can.data.root_ang_vel_w[0]).item()
        ),
        "base_xyz": base_pos[0].detach().cpu().tolist(),
        "tcp_xyz": tcp_pos[0].detach().cpu().tolist(),
        "tcp_quat_wxyz": tcp_quat_w[0].detach().cpu().tolist(),
        "left_fingertip_xyz": left_pos[0].detach().cpu().tolist(),
        "left_fingertip_quat_wxyz": left_quat[0].detach().cpu().tolist(),
        "right_fingertip_xyz": right_pos[0].detach().cpu().tolist(),
        "right_fingertip_quat_wxyz": right_quat[0].detach().cpu().tolist(),
        "left_contact_force_xyz": left_force[0].detach().cpu().tolist(),
        "right_contact_force_xyz": right_force[0].detach().cpu().tolist(),
        "left_contact": bool(
            world.in_contact("can", "left_fingertip", force_threshold=0.1)[0].item()
        ),
        "right_contact": bool(
            world.in_contact("can", "right_fingertip", force_threshold=0.1)[0].item()
        ),
        "grasp_constraint_active": bool(
            oracle.grasp_constraint_active[0].item()
        ),
        "grasp_quality": oracle.grasp_quality_snapshot(0),
        "secure_grasp_quality": oracle.secure_grasp_quality[0],
        "gripper_joint_rad": float(robot.data.joint_pos[0, gripper_index].item()),
        "gripper_joint_positions": robot.data.joint_pos[
            0, gripper_indices
        ].detach().cpu().tolist(),
    }


def _run_trial(env, trial_index: int, place_xy: tuple[float, float]) -> dict:
    trace = []
    env.reset_eval_state()
    env.reset()
    oracle = KitchenManipulationOracle(
        env,
        place_xy=place_xy,
        use_grasp_assist=args_cli.grasp_assist,
    )
    trace.append(_snapshot(env, oracle, 0))
    last_phase = trace[-1]["phase"]
    terminated_early = False
    for step in range(1, args_cli.max_steps + 1):
        action = oracle.compute_action()
        _, _, terminated, truncated, _ = env.step(action)
        oracle.post_step()
        phase = oracle.phase_names()[0]
        if phase != last_phase:
            trace.append(_snapshot(env, oracle, step))
            if args_cli.trials == 1:
                print(json.dumps(trace[-1]), flush=True)
            else:
                print(json.dumps(trace[-1]), flush=True)
            last_phase = phase
        elif args_cli.trials == 1 and phase in {
            "close",
            "secure",
            "lift",
            "carry_settle",
            "traverse",
        } and step % 10 == 0:
            trace.append(_snapshot(env, oracle, step))
            print(json.dumps(trace[-1]), flush=True)
        if bool(terminated[0] or truncated[0]):
            terminated_early = True
            break
        if bool(oracle.finished[0].item()):
            break
    return {
        "trial": trial_index,
        "success": bool(oracle.succeeded[0].item()),
        "grasp_assist": args_cli.grasp_assist,
        "place_xy": list(place_xy),
        "failure_reason": oracle.failure_reason[0],
        "terminated_early": terminated_early,
        "steps": step,
        "trace": trace,
        "final": _snapshot(env, oracle, step),
    }


def main() -> int:
    if args_cli.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args_cli.trials < 1:
        raise ValueError("--trials must be positive")
    torch.manual_seed(args_cli.seed)
    place_rng = random.Random(args_cli.seed)
    register_genmanip_kitchen_mobile_ik_env()
    env_cfg = parse_env_cfg(
        IK_ENV_NAME,
        seed=args_cli.seed,
        num_envs=1,
        env_spacing=8.0,
        use_fabric=False,
    )
    # The scripted oracle consumes proprioception and contact state only.
    # Avoid paying RTX and HDF5 costs during large deterministic physics gates;
    # both remain explicitly available for visual inspection/data collection.
    if not args_cli.enable_cameras:
        env_cfg.scene.wrist_cam = None
        env_cfg.scene.navigation_camera = None
    if not args_cli.record_data:
        env_cfg.recorders = None
    env, _ = create_env(
        env_cfg,
    )
    try:
        results = []
        for trial_index in range(args_cli.trials):
            if args_cli.randomize_place:
                place_xy = (
                    place_rng.uniform(
                        ORACLE_PLACE_REGION.xmin, ORACLE_PLACE_REGION.xmax
                    ),
                    place_rng.uniform(
                        ORACLE_PLACE_REGION.ymin, ORACLE_PLACE_REGION.ymax
                    ),
                )
            else:
                place_xy = ORACLE_PLACE_REGION.center
            result = _run_trial(env, trial_index, place_xy)
            results.append(result)
            print(
                json.dumps(
                    {
                        "trial": trial_index,
                        "success": result["success"],
                        "steps": result["steps"],
                        "failure_reason": result["failure_reason"],
                    }
                ),
                flush=True,
            )

        successes = sum(result["success"] for result in results)
        result = results[0] if args_cli.trials == 1 else {
            "all_success": successes == args_cli.trials,
            "requested_trials": args_cli.trials,
            "completed_trials": len(results),
            "successes": successes,
            "grasp_assist": args_cli.grasp_assist,
            "randomize_place": args_cli.randomize_place,
            "seed": args_cli.seed,
            "trials": results,
        }
        output = os.path.abspath(args_cli.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if successes == args_cli.trials else 1
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
