# SPDX-License-Identifier: Apache-2.0

"""Validate the proxy-free-style XGRIDS2 task-space reach/retract gate."""

import argparse
import json
import os
import sys

import cv2
from isaaclab.app import AppLauncher


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from robolab.core.utils.nurec_rendering import (
    configure_xgrids_vendor_nurec_volume,
    configure_xgrids_vendor_render_products,
    require_single_gpu,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", default="output/xgrids2_lab/ik_reach_gate"
)
parser.add_argument(
    "--stage", choices=("hold", "local", "reach", "pick"), default="reach"
)
parser.add_argument(
    "--collision-profile",
    choices=("task", "tables-only", "map-only", "floor-only", "robot-only"),
    default="task",
    help="Diagnostic collision ablation; production validation uses task.",
)
parser.add_argument("--record-video", action="store_true")
parser.add_argument("--video-fps", type=int, default=10)
parser.add_argument("--capture-every", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
require_single_gpu(args_cli)
simulation_app = AppLauncher(args_cli).app
NUREC_SETTINGS = configure_xgrids_vendor_nurec_volume()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.registrations.kinova_mobile import (  # noqa: E402
    register_xgrids2_lab_manipulation_ik_env,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_ik_oracle import (  # noqa: E402
    XGRIDS2IKReachOracle,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (  # noqa: E402
    COLLISION_SEGMENTS,
)


def _rgb(camera) -> np.ndarray:
    return np.ascontiguousarray(
        camera.data.output["rgb"][0, :, :, :3]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


def _panel(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    size: tuple[int, int] = (480, 270),
) -> np.ndarray:
    width, height = size
    panel = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (width, 50), (15, 19, 25), -1)
    cv2.putText(
        panel,
        title,
        (14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
    )
    cv2.putText(
        panel,
        subtitle,
        (14, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (90, 220, 255),
        1,
    )
    return panel


def main() -> None:
    env_cfg_class, _ = register_xgrids2_lab_manipulation_ik_env()
    env_cfg = env_cfg_class()
    if args_cli.collision_profile in {"tables-only", "floor-only", "robot-only"}:
        for segment in COLLISION_SEGMENTS:
            setattr(env_cfg.scene, segment.name, None)
    if args_cli.collision_profile in {"map-only", "floor-only", "robot-only"}:
        env_cfg.scene.dining_table_top = None
        env_cfg.scene.snack_table_top = None
    if args_cli.collision_profile == "robot-only":
        env_cfg.scene.floor = None
    env_cfg.sim.use_fabric = True
    env_cfg.sim.render_interval = (
        args_cli.capture_every if args_cli.record_video else 8
    )
    env_cfg.sim.render.antialiasing_mode = "OFF"
    # Interface and reach gates contain no grasp constraint. The pick gate
    # keeps a disabled joint available, enabling it only after measured
    # bilateral fingertip contact.
    if args_cli.stage != "pick":
        env_cfg.events.author_oracle_grasp_joints = None
    env_cfg.scene.navigation_camera.height = 270
    env_cfg.scene.navigation_camera.width = 480
    env_cfg.scene.wrist_cam.height = 270
    env_cfg.scene.wrist_cam.width = 480
    env_cfg.scene.audit_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/ik_reach_audit_camera",
        height=540,
        width=960,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, 30.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(convention="opengl"),
    )
    env, _ = create_env(env_cfg)
    output_dir = os.path.abspath(os.path.join(args_cli.output_dir, args_cli.stage))
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")
    montage_path = os.path.join(output_dir, f"{args_cli.stage}_montage.png")
    video_path = os.path.join(output_dir, f"{args_cli.stage}.mp4")
    video_writer = None
    try:
        env.reset_eval_state()
        env.reset()
        products = configure_xgrids_vendor_render_products(
            env.scene["navigation_camera"],
            env.scene["wrist_cam"],
            env.scene["audit_camera"],
        )
        audit_camera = env.scene["audit_camera"]
        audit_camera.set_world_poses_from_view(
            eyes=torch.tensor([[2.1, -2.25, 1.75]], device=env.device),
            targets=torch.tensor([[1.20, -1.13, 0.90]], device=env.device),
        )
        oracle = XGRIDS2IKReachOracle(env)
        initial_tcp_pos_w, initial_tcp_quat_w = oracle.current_tcp_w()
        initial_arm_positions = (
            oracle.robot.data.joint_pos[0, oracle.arm_indices]
            .detach()
            .cpu()
            .tolist()
        )
        initial_tcp_position = (
            initial_tcp_pos_w[0] - env.scene.env_origins[0]
        ).detach().cpu().tolist()
        initial_tcp_quaternion_wxyz = (
            initial_tcp_quat_w[0].detach().cpu().tolist()
        )
        original_step = oracle._step
        first_frame = None
        last_frame = None
        if args_cli.record_video:
            video_writer = VideoWriter(video_path, fps=args_cli.video_fps)
            recorded_steps = 0

            def recording_step(action):
                nonlocal recorded_steps, first_frame, last_frame
                original_step(action)
                recorded_steps += 1
                if recorded_steps % args_cli.capture_every != 0:
                    return
                root = (
                    oracle.robot.data.root_link_pos_w[0]
                    - oracle.env.scene.env_origins[0]
                )
                subtitle = (
                    f"{oracle.phase.upper()} | base "
                    f"({float(root[0]):.2f}, {float(root[1]):.2f})"
                )
                third_person = _panel(
                    _rgb(audit_camera),
                    "THIRD-PERSON",
                    subtitle,
                    size=(960, 540),
                )
                robot_views = np.concatenate(
                    (
                        _panel(
                            _rgb(env.scene["navigation_camera"]),
                            "NAVIGATION CAMERA",
                            "fixed base mount",
                        ),
                        _panel(
                            _rgb(env.scene["wrist_cam"]),
                            "WRIST CAMERA",
                            "bracelet mount",
                        ),
                    ),
                    axis=0,
                )
                frame = np.concatenate((third_person, robot_views), axis=1)
                if first_frame is None:
                    first_frame = frame.copy()
                last_frame = frame.copy()
                video_writer.write(frame)

            oracle._step = recording_step

        if args_cli.stage == "hold":
            result = oracle.run_interface_gate(local_move=False)
        elif args_cli.stage == "local":
            result = oracle.run_interface_gate(local_move=True)
        elif args_cli.stage == "reach":
            result = oracle.run()
        else:
            result = oracle.run_pick()
        if args_cli.record_video:
            oracle._hold(args_cli.video_fps * args_cli.capture_every)
            oracle._step = original_step
            video_writer.release()
            video_writer = None
        if first_frame is None:
            third = _panel(
                _rgb(audit_camera),
                "THIRD-PERSON",
                oracle.phase.upper(),
                size=(960, 540),
            )
            robot_views = np.concatenate(
                (
                    _panel(
                        _rgb(env.scene["navigation_camera"]),
                        "NAVIGATION CAMERA",
                        "fixed base mount",
                    ),
                    _panel(
                        _rgb(env.scene["wrist_cam"]),
                        "WRIST CAMERA",
                        "bracelet mount",
                    ),
                ),
                axis=0,
            )
            first_frame = np.concatenate((third, robot_views), axis=1)
            last_frame = first_frame.copy()
        montage = np.concatenate((first_frame, last_frame), axis=0)
        if not cv2.imwrite(
            montage_path, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
        ):
            raise RuntimeError(f"could not write {montage_path}")
        results = {
            **result.as_dict(),
            "validation_stage": args_cli.stage,
            "collision_profile": args_cli.collision_profile,
            "initial_arm_positions": initial_arm_positions,
            "initial_tcp_position": initial_tcp_position,
            "initial_tcp_quaternion_wxyz": initial_tcp_quaternion_wxyz,
            "nurec_render_settings": NUREC_SETTINGS,
            "nurec_render_products": products,
            "grasp_joint_authored": args_cli.stage == "pick",
            "grasp_joint_activation_policy": (
                "after_measured_bilateral_contact"
                if args_cli.stage == "pick"
                else "not_authored"
            ),
            "state_injection": False,
        }
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(json.dumps(results, indent=2), flush=True)
        print(f"Saved IK reach montage: {montage_path}")
        if args_cli.record_video:
            print(f"Saved IK reach video: {video_path}")
        if not result.passed:
            raise RuntimeError(result.failure_reason or "IK reach gate failed")
    finally:
        if video_writer is not None:
            video_writer.release()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
