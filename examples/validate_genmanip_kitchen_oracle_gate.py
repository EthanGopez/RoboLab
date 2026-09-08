# SPDX-License-Identifier: Apache-2.0

"""Run the randomized Milestone 2 oracle gate in vectorized Isaac Lab envs."""

import argparse
import json
import math
import os
import random
import traceback
from collections import Counter

import cv2  # noqa: F401 -- import before Isaac Lab

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=10)
parser.add_argument("--max-steps", type=int, default=2600)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--scenario",
    choices=("baseline", "robot_other_side", "can_other_side"),
    default="baseline",
)
parser.add_argument(
    "--place-xy",
    action="append",
    nargs=2,
    type=float,
    metavar=("X", "Y"),
    help="Use an explicit target; repeat once per trial for reproductions.",
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
    help="Enable RoboLab's full HDF5 recorder (off by default for gates).",
)
parser.add_argument(
    "--fail-fast",
    action="store_true",
    help="Stop after the first batch containing a failed trial.",
)
parser.add_argument(
    "--grasp-trace-stride",
    type=int,
    default=0,
    help="Record physical-grasp diagnostics every N steps; zero disables it.",
)
parser.add_argument(
    "--output",
    default="output/genmanip_kitchen/milestone2/oracle_gate_100.json",
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
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (  # noqa: E402
    ORACLE_PICK_REGION,
    ORACLE_PLACE_REGION,
)
from robolab.tasks.mobile_manipulation.kitchen_manipulation_oracle import (  # noqa: E402
    KitchenManipulationOracle,
    OraclePhase,
)
from robolab.tasks.mobile_manipulation.kitchen_scenarios import (  # noqa: E402
    KitchenManipulationScenario,
    get_kitchen_scenario,
)


def _sample_place_targets(count: int, seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [
        (
            rng.uniform(ORACLE_PLACE_REGION.xmin, ORACLE_PLACE_REGION.xmax),
            rng.uniform(ORACLE_PLACE_REGION.ymin, ORACLE_PLACE_REGION.ymax),
        )
        for _ in range(count)
    ]


def _final_state(env, oracle: KitchenManipulationOracle, env_id: int) -> dict:
    origin = env.scene.env_origins[env_id]
    can = env.scene["can"]
    robot = env.scene["robot"]
    can_xyz = can.data.root_pos_w[env_id] - origin
    base_xyz = robot.data.root_link_pos_w[env_id] - origin
    target_xy = oracle.place_xy[env_id]
    return {
        "phase": oracle.phase_names()[env_id],
        "can_xyz": can_xyz.detach().cpu().tolist(),
        "can_quat_wxyz": can.data.root_quat_w[env_id].detach().cpu().tolist(),
        "can_linear_speed": float(
            torch.linalg.vector_norm(can.data.root_lin_vel_w[env_id]).item()
        ),
        "can_angular_speed": float(
            torch.linalg.vector_norm(can.data.root_ang_vel_w[env_id]).item()
        ),
        "base_xyz": base_xyz.detach().cpu().tolist(),
        "placement_error_xy": (can_xyz[:2] - target_xy).detach().cpu().tolist(),
        "grasp_constraint_active": bool(
            oracle.grasp_constraint_active[env_id].item()
        ),
        "grasp_quality": oracle.grasp_quality_snapshot(env_id),
        "secure_grasp_quality": oracle.secure_grasp_quality[env_id],
    }


def _run_batch(
    env,
    *,
    batch_index: int,
    first_trial: int,
    place_targets: list[tuple[float, float]],
    scenario: KitchenManipulationScenario,
) -> list[dict]:
    active_count = len(place_targets)
    padded_targets = place_targets + [ORACLE_PLACE_REGION.center] * (
        env.num_envs - active_count
    )

    env.reset_eval_state()
    env.reset()
    initial_can_xyz = (
        env.scene["can"].data.root_pos_w - env.scene.env_origins
    ).detach().cpu().tolist()
    oracle = KitchenManipulationOracle(
        env,
        place_xy=padded_targets,
        use_grasp_assist=args_cli.grasp_assist,
        scenario=scenario,
    )
    if active_count < env.num_envs:
        oracle.phase[active_count:] = int(OraclePhase.SUCCESS)

    histories = [[{"step": 0, "phase": "settle"}] for _ in range(active_count)]
    previous_phases = oracle.phase[:active_count].clone()
    finished_steps = [-1] * active_count
    terminal_states = [None] * active_count
    grasp_traces = [[] for _ in range(active_count)]

    for step in range(1, args_cli.max_steps + 1):
        action = oracle.compute_action()
        _, _, terminated, truncated, _ = env.step(action)
        oracle.post_step()

        if (
            args_cli.grasp_trace_stride > 0
            and step % args_cli.grasp_trace_stride == 0
        ):
            phase_names = oracle.phase_names()
            for env_id in range(active_count):
                snapshot = oracle.grasp_quality_snapshot(env_id)
                if snapshot is not None:
                    grasp_traces[env_id].append(
                        {"step": step, "phase": phase_names[env_id], **snapshot}
                    )

        current_phases = oracle.phase[:active_count]
        changed = current_phases != previous_phases
        for env_id in torch.nonzero(changed, as_tuple=False).squeeze(-1).tolist():
            histories[env_id].append(
                {"step": step, "phase": oracle.phase_names()[env_id]}
            )
        previous_phases = current_phases.clone()

        unexpected_done = (terminated[:active_count] | truncated[:active_count]) & (
            ~oracle.finished[:active_count]
        )
        for env_id in torch.nonzero(
            unexpected_done, as_tuple=False
        ).squeeze(-1).tolist():
            oracle._fail(env_id, "RoboLab environment terminated before oracle completion")

        for env_id in range(active_count):
            if finished_steps[env_id] < 0 and bool(oracle.finished[env_id].item()):
                finished_steps[env_id] = step
                terminal_states[env_id] = _final_state(env, oracle, env_id)

        if step % 100 == 0 or bool(torch.all(oracle.finished[:active_count]).item()):
            phase_counts = Counter(oracle.phase_names()[:active_count])
            print(
                json.dumps(
                    {
                        "batch": batch_index,
                        "step": step,
                        "phase_counts": dict(sorted(phase_counts.items())),
                    }
                ),
                flush=True,
            )
        if bool(torch.all(oracle.finished[:active_count]).item()):
            break

    results = []
    for env_id in range(active_count):
        results.append(
            {
                "trial": first_trial + env_id,
                "batch": batch_index,
                "env_id": env_id,
                "success": bool(oracle.succeeded[env_id].item()),
                "failure_reason": oracle.failure_reason[env_id],
                "failure_diagnostics": oracle.failure_diagnostics[env_id],
                "steps": finished_steps[env_id]
                if finished_steps[env_id] >= 0
                else args_cli.max_steps,
                "initial_can_xyz": initial_can_xyz[env_id],
                "place_xy": list(place_targets[env_id]),
                "phase_history": histories[env_id],
                "grasp_trace": grasp_traces[env_id],
                "terminal": terminal_states[env_id],
                "final": _final_state(env, oracle, env_id),
            }
        )
    return results


def _build_summary(
    results: list[dict],
    *,
    num_envs: int,
    scenario: KitchenManipulationScenario,
) -> dict:
    """Build a checkpoint-safe summary from all completed trial results."""

    successes = sum(result["success"] for result in results)
    return {
        "all_success": (
            len(results) == args_cli.trials and successes == args_cli.trials
        ),
        "requested_trials": args_cli.trials,
        "completed_trials": len(results),
        "successes": successes,
        "batch_size": num_envs,
        "seed": args_cli.seed,
        "scenario": scenario.name,
        "grasp_mode": (
            "strict_post_contact_constraint"
            if args_cli.grasp_assist
            else "unassisted_friction"
        ),
        "oracle_place_region": {
            "xmin": ORACLE_PLACE_REGION.xmin,
            "xmax": ORACLE_PLACE_REGION.xmax,
            "ymin": ORACLE_PLACE_REGION.ymin,
            "ymax": ORACLE_PLACE_REGION.ymax,
        },
        "oracle_pick_region": {
            "xmin": scenario.pick_region.xmin,
            "xmax": scenario.pick_region.xmax,
            "ymin": scenario.pick_region.ymin,
            "ymax": scenario.pick_region.ymax,
        },
        "swept_collision_checks": True,
        "results": results,
    }


def _write_summary(summary: dict) -> str:
    """Atomically replace the gate JSON so interruptions keep valid evidence."""

    output = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary_output = f"{output}.tmp"
    with open(temporary_output, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_output, output)
    return output


def main() -> int:
    if args_cli.trials < 1:
        raise ValueError("--trials must be positive")
    if args_cli.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args_cli.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args_cli.grasp_trace_stride < 0:
        raise ValueError("--grasp-trace-stride must be non-negative")

    torch.manual_seed(args_cli.seed)
    scenario = get_kitchen_scenario(args_cli.scenario)
    if args_cli.place_xy is not None:
        place_targets = [tuple(target) for target in args_cli.place_xy]
        if len(place_targets) != args_cli.trials:
            raise ValueError(
                "When --place-xy is used, provide exactly one target per "
                f"trial ({args_cli.trials} expected, {len(place_targets)} given)"
            )
    else:
        place_targets = _sample_place_targets(args_cli.trials, args_cli.seed)
    num_envs = min(args_cli.batch_size, args_cli.trials)

    register_genmanip_kitchen_mobile_ik_env()
    env_cfg = parse_env_cfg(
        IK_ENV_NAME,
        seed=args_cli.seed,
        num_envs=num_envs,
        env_spacing=8.0,
        use_fabric=False,
    )
    # Keep the manager's episode timeout outside the explicit oracle gate.
    # Hard scenarios add long navigation legs, so --max-steps must govern the
    # reproduction instead of the baseline task's shorter default horizon.
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s,
        args_cli.max_steps * env_cfg.sim.dt * env_cfg.decimation + 1.0,
    )
    base_x, base_y, base_z, base_yaw = scenario.initial_base_pose
    env_cfg.scene.robot.init_state.pos = (base_x, base_y, base_z)
    env_cfg.scene.robot.init_state.rot = (
        math.cos(0.5 * base_yaw), 0.0, 0.0, math.sin(0.5 * base_yaw)
    )
    env_cfg.events.sample_can.params = {"pick_region": scenario.pick_region}
    if scenario.initial_arm_joint_pos is not None:
        env_cfg.scene.robot.init_state.joint_pos.update(
            {
                f"joint_{index}": value
                for index, value in enumerate(
                    scenario.initial_arm_joint_pos, start=1
                )
            }
        )
    if not args_cli.enable_cameras:
        env_cfg.scene.wrist_cam = None
        env_cfg.scene.navigation_camera = None
    if not args_cli.record_data:
        env_cfg.recorders = None
    env, _ = create_env(env_cfg)

    results = []
    try:
        for batch_index, first_trial in enumerate(
            range(0, args_cli.trials, num_envs)
        ):
            batch_targets = place_targets[
                first_trial : first_trial + num_envs
            ]
            batch_results = _run_batch(
                env,
                batch_index=batch_index,
                first_trial=first_trial,
                place_targets=batch_targets,
                scenario=scenario,
            )
            results.extend(batch_results)
            checkpoint = _build_summary(results, num_envs=num_envs, scenario=scenario)
            _write_summary(checkpoint)
            print(
                json.dumps(
                    {
                        "batch": batch_index,
                        "completed_trials": len(results),
                        "successes": sum(result["success"] for result in results),
                    }
                ),
                flush=True,
            )
            if args_cli.fail_fast and not all(
                result["success"] for result in batch_results
            ):
                break
    finally:
        env.close()

    summary = _build_summary(results, num_envs=num_envs, scenario=scenario)
    _write_summary(summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    return 0 if summary["all_success"] else 1


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
