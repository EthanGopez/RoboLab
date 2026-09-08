#!/usr/bin/env python3
"""Run the randomized XGRIDS2 CuRobo pick-place repeatability gate."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2  # noqa: F401 -- load before Isaac Lab/Kit
from isaaclab.app import AppLauncher


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from robolab.core.utils.nurec_rendering import (
    configure_xgrids_vendor_nurec_volume,
    configure_xgrids_vendor_render_products,
    require_single_gpu,
)


parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=20)
parser.add_argument(
    "--trial-indices",
    type=str,
    default=None,
    help="Comma-separated indices from the generated schedule for targeted reruns.",
)
parser.add_argument("--seed", type=int, default=20260829)
parser.add_argument(
    "--resume-from",
    type=int,
    default=None,
    help="Keep checkpoint records before this index and rerun from it onward.",
)
parser.add_argument("--num-seeds", type=int, default=8)
parser.add_argument("--settle-steps", type=int, default=24)
parser.add_argument("--minimum-success-rate", type=float, default=0.90)
parser.add_argument(
    "--output",
    default="output/xgrids2_lab/robustness/gate_20.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
require_single_gpu(args_cli)
simulation_app = AppLauncher(args_cli).app
NUREC_SETTINGS = configure_xgrids_vendor_nurec_volume()

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.kinova_mobile import (  # noqa: E402
    register_xgrids2_lab_manipulation_env,
)
from robolab.tasks.mobile_manipulation.fixed_base_manipulation import (  # noqa: E402
    ARM_JOINT_NAMES,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (  # noqa: E402
    STAGING_POSES,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_curobo_oracle import (  # noqa: E402
    XGRIDS2CuroboOracle,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_manipulation_geometry import (  # noqa: E402
    CAN_INITIAL_Z,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_robustness import (  # noqa: E402
    evaluate_trial,
    robustness_trial_schedule,
    summarize_robustness,
)


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _selected_schedule():
    full = robustness_trial_schedule(args_cli.trials, seed=args_cli.seed)
    if args_cli.trial_indices is None:
        return full
    values = tuple(int(value.strip()) for value in args_cli.trial_indices.split(","))
    if not values or len(values) != len(set(values)):
        raise ValueError("--trial-indices must contain unique comma-separated values")
    if any(index < 0 or index >= len(full) for index in values):
        raise ValueError(
            f"--trial-indices must be between 0 and {len(full) - 1}"
        )
    return tuple(full[index] for index in values)


def _reset_trial(env, trial) -> tuple[float, ...]:
    torch.manual_seed(trial.seed)
    env.reset_eval_state()
    env.reset()
    robot = env.scene["robot"]
    can = env.scene["can"]
    origin = env.scene.env_origins[0]

    robot_pose = torch.tensor(
        [
            [
                trial.base_xy[0],
                trial.base_xy[1],
                float(robot.data.default_root_state[0, 2]),
                math.cos(0.5 * trial.base_yaw),
                0.0,
                0.0,
                math.sin(0.5 * trial.base_yaw),
            ]
        ],
        dtype=torch.float32,
        device=env.device,
    )
    robot_pose[0, :3] += origin
    baseline = STAGING_POSES["dining_table"]
    perturbed_base = (
        math.dist(trial.base_xy, baseline[:2]) > 1.0e-9
        or abs(trial.base_yaw - baseline[3]) > 1.0e-9
    )
    if perturbed_base:
        # Vary only the floating root.  Joint perturbations are commanded
        # smoothly below; writing articulation joint state after reset creates
        # a drive/state discontinuity and is not a physical robustness test.
        robot.write_root_pose_to_sim(robot_pose)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
        planar_xy = robot_pose[:, :2]
    else:
        planar_xy = robot.data.root_link_pos_w[:, :2]
    env.action_manager.get_term("base").set_planar_pose(
        planar_xy,
        torch.tensor([trial.base_yaw], dtype=torch.float32, device=env.device),
    )

    can_pose = torch.tensor(
        [
            [
                trial.pick_xy[0],
                trial.pick_xy[1],
                CAN_INITIAL_Z,
                math.cos(0.5 * trial.can_yaw),
                0.0,
                0.0,
                math.sin(0.5 * trial.can_yaw),
            ]
        ],
        dtype=torch.float32,
        device=env.device,
    )
    can_pose[0, :3] += origin
    can.write_root_pose_to_sim(can_pose)
    can.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))

    arm_indices = [
        robot.data.joint_names.index(name) for name in ARM_JOINT_NAMES
    ]
    baseline = robot.data.default_joint_pos[0, arm_indices]
    target = baseline + torch.tensor(
        trial.arm_offsets_rad, dtype=baseline.dtype, device=env.device
    )
    return tuple(float(value) for value in target.detach().cpu().tolist())


def main() -> int:
    if args_cli.trials < 1 or args_cli.trials > 40:
        raise ValueError("--trials must be between 1 and 40")
    if args_cli.num_seeds < 1 or args_cli.settle_steps < 1:
        raise ValueError("--num-seeds and --settle-steps must be positive")
    if not 0.0 < args_cli.minimum_success_rate <= 1.0:
        raise ValueError("--minimum-success-rate must be in (0, 1]")

    selected_schedule = _selected_schedule()
    full_schedule = robustness_trial_schedule(args_cli.trials, seed=args_cli.seed)
    if args_cli.resume_from is not None and args_cli.trial_indices is not None:
        raise ValueError("--resume-from cannot be combined with --trial-indices")
    if args_cli.resume_from is not None and not (
        0 <= args_cli.resume_from < len(full_schedule)
    ):
        raise ValueError(
            f"--resume-from must be between 0 and {len(full_schedule) - 1}"
        )
    report_schedule = (
        full_schedule if args_cli.resume_from is not None else selected_schedule
    )
    execution_schedule = (
        tuple(trial for trial in full_schedule if trial.index >= args_cli.resume_from)
        if args_cli.resume_from is not None
        else selected_schedule
    )
    env_cfg_class, env_name = register_xgrids2_lab_manipulation_env()
    env_cfg = env_cfg_class()
    env_cfg.seed = args_cli.seed
    env_cfg.recorders = None
    # Perturbed trials include smooth arm normalization and base alignment
    # before the original full task.  Preserve the task predicates while
    # allowing those recovery phases to finish before the time-limit term.
    env_cfg.episode_length_s = 300
    env_cfg.sim.use_fabric = True
    # Keep tiny render products alive.  In this Isaac/Fabric stack, removing all
    # camera sensors changes articulation update behavior and destabilizes the
    # otherwise identical arm controller; 64x36 preserves the validated update
    # path at a fraction of evidence-video rendering cost.
    env_cfg.sim.render_interval = 8
    env_cfg.sim.render.antialiasing_mode = "OFF"
    env_cfg.scene.navigation_camera.width = 64
    env_cfg.scene.navigation_camera.height = 36
    env_cfg.scene.wrist_cam.width = 64
    env_cfg.scene.wrist_cam.height = 36
    env_cfg.scene.audit_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robustness_audit_camera",
        height=36,
        width=64,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, 30.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(convention="opengl"),
    )
    env, _ = create_env(env_cfg)
    render_products = configure_xgrids_vendor_render_products(
        env.scene["navigation_camera"],
        env.scene["wrist_cam"],
        env.scene["audit_camera"],
    )
    env.scene["audit_camera"].set_world_poses_from_view(
        eyes=torch.tensor([[2.1, -2.25, 1.75]], device=env.device),
        targets=torch.tensor([[1.20, -1.13, 0.90]], device=env.device),
    )
    output_path = Path(args_cli.output).resolve()
    records = []
    if args_cli.resume_from is not None:
        if not output_path.is_file():
            raise FileNotFoundError(f"resume report does not exist: {output_path}")
        checkpoint = json.loads(output_path.read_text(encoding="utf-8"))
        if checkpoint.get("seed") != args_cli.seed:
            raise ValueError("resume report seed does not match --seed")
        expected_schedule = [trial.as_dict() for trial in full_schedule]
        if checkpoint.get("schedule") != expected_schedule:
            raise ValueError(
                "resume report schedule does not match the generated schedule"
            )
        records = [
            record
            for record in checkpoint.get("records", [])
            if int(record["trial"]["index"]) < args_cli.resume_from
        ]
        expected_indices = list(range(args_cli.resume_from))
        observed_indices = [int(record["trial"]["index"]) for record in records]
        if observed_indices != expected_indices:
            raise ValueError(
                "resume report must contain contiguous records "
                f"{expected_indices}; got {observed_indices}"
            )
    started = time.perf_counter()

    try:
        for trial in execution_schedule:
            trial_started = time.perf_counter()
            print(json.dumps({"starting_trial": trial.as_dict()}), flush=True)
            initial_arm_positions = _reset_trial(env, trial)
            oracle = XGRIDS2CuroboOracle(env, num_seeds=args_cli.num_seeds)
            oracle._hold(args_cli.settle_steps, closed=False, default=True)
            result = oracle.run(
                stage="full",
                place_xy=trial.place_xy,
                initial_arm_positions=(
                    None
                    if all(abs(value) < 1.0e-12 for value in trial.arm_offsets_rad)
                    else initial_arm_positions
                ),
            ).as_dict()
            record = evaluate_trial(trial, result)
            record["runtime_s"] = time.perf_counter() - trial_started
            records.append(record)
            report = summarize_robustness(
                report_schedule,
                records,
                minimum_success_rate=args_cli.minimum_success_rate,
            )
            report.update(
                {
                    "environment": env_name,
                    "seed": args_cli.seed,
                    "num_curobo_seeds": args_cli.num_seeds,
                    "settle_steps": args_cli.settle_steps,
                    "minimal_camera_render_products": [
                        "navigation_camera",
                        "wrist_cam",
                        "audit_camera",
                    ],
                    "nurec_render_products": render_products,
                    "nurec_render_settings": NUREC_SETTINGS,
                    "runtime_s": time.perf_counter() - started,
                    "resumed_from": args_cli.resume_from,
                }
            )
            _write_report(output_path, report)
            print(
                json.dumps(
                    {
                        "completed_trial": trial.index,
                        "passed": record["passed"],
                        "failure_category": record["failure_category"],
                        "runtime_s": record["runtime_s"],
                        "progress": f"{len(records)}/{len(report_schedule)}",
                    }
                ),
                flush=True,
            )
            # A post-grasp failure can leave the USD joint-disable edit
            # pending until PhysX advances. Process it before the next reset.
            oracle.grasp_joint.GetJointEnabledAttr().Set(False)
            try:
                oracle._hold(8, closed=False)
            except RuntimeError:
                pass
            del oracle
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        env.close()

    report = summarize_robustness(
        report_schedule,
        records,
        minimum_success_rate=args_cli.minimum_success_rate,
    )
    report.update(
        {
            "environment": env_name,
            "seed": args_cli.seed,
            "num_curobo_seeds": args_cli.num_seeds,
            "settle_steps": args_cli.settle_steps,
            "minimal_camera_render_products": [
                "navigation_camera",
                "wrist_cam",
                "audit_camera",
            ],
            "nurec_render_products": render_products,
            "nurec_render_settings": NUREC_SETTINGS,
            "runtime_s": time.perf_counter() - started,
            "resumed_from": args_cli.resume_from,
        }
    )
    _write_report(output_path, report)
    summary = {key: value for key, value in report.items() if key not in {"schedule", "records"}}
    print(json.dumps(summary, indent=2))
    print(f"Saved robustness report: {output_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
