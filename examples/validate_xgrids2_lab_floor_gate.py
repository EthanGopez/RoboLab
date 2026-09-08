#!/usr/bin/env python3
"""Gate 2: drop a grid of rigid probes onto the clean XGRIDS2 floor slab."""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="output/xgrids2_lab/physics_gates")
parser.add_argument("--steps", type=int, default=180)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from robolab.constants import PACKAGE_DIR  # noqa: E402
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (  # noqa: E402
    XGRIDS_VISUAL_USD_RELATIVE,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_navigation_task import (  # noqa: E402
    make_floor_cfg,
)


PROBE_RADIUS_M = 0.06
PROBE_POSITIONS = (
    (1.20, -2.40),
    (1.80, -1.76),
    (2.60, -1.50),
    (3.40, -1.25),
    (4.10, -1.00),
    (4.64, -0.76),
    (2.80, -0.60),
    (3.50, -0.20),
    (4.20, 0.20),
)


def _probe_cfg(index: int, xy: tuple[float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/floor_probe_{index:02d}",
        spawn=sim_utils.SphereCfg(
            radius=PROBE_RADIUS_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                linear_damping=0.04,
                angular_damping=0.04,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.85, 0.18), roughness=0.55
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(xy[0], xy[1], 0.40 + 0.025 * index),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class FloorGateSceneCfg(InteractiveSceneCfg):
    lab_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/xgrids2_lab_visual",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(PACKAGE_DIR, XGRIDS_VISUAL_USD_RELATIVE),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
        ),
    )
    floor = make_floor_cfg()
    dome_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dome_light",
        spawn=sim_utils.DomeLightCfg(intensity=1400.0),
    )
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/gate_camera",
        height=720,
        width=960,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            horizontal_aperture=20.0,
            clipping_range=(0.05, 2000.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(3.0, -1.0, 7.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )


for _index, _xy in enumerate(PROBE_POSITIONS):
    setattr(FloorGateSceneCfg, f"probe_{_index:02d}", _probe_cfg(_index, _xy))


def main() -> None:
    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=4, use_fabric=False)
    )
    scene = InteractiveScene(
        FloorGateSceneCfg(
            num_envs=1,
            env_spacing=12.0,
            replicate_physics=False,
            lazy_sensor_update=False,
        )
    )
    sim.reset()
    scene.reset()
    camera = scene["camera"]
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[3.0, -1.0, 7.0]], device=camera.device),
        targets=torch.tensor([[3.15, -0.92, 0.0]], device=camera.device),
    )
    for _ in range(args_cli.steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    expected_z = PROBE_RADIUS_M
    records = []
    failures = []
    for index, _ in enumerate(PROBE_POSITIONS):
        probe = scene[f"probe_{index:02d}"]
        position = probe.data.root_pos_w[0] - scene.env_origins[0]
        linear_speed = torch.linalg.vector_norm(probe.data.root_lin_vel_w[0]).item()
        angular_speed = torch.linalg.vector_norm(probe.data.root_ang_vel_w[0]).item()
        passed = (
            torch.isfinite(position).all().item()
            and abs(float(position[2]) - expected_z) <= 0.004
            and linear_speed < 0.02
            and angular_speed < 0.20
        )
        record = {
            "index": index,
            "position_xyz_m": [float(value) for value in position],
            "linear_speed_m_s": linear_speed,
            "angular_speed_rad_s": angular_speed,
            "passed": bool(passed),
        }
        records.append(record)
        if not passed:
            failures.append(record)

    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
    frame = np.ascontiguousarray(rgb)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (18, 22, 28), -1)
    status = "PASS" if not failures else "FAIL"
    cv2.putText(
        frame,
        f"GATE 2 {status} | {len(records) - len(failures)}/{len(records)} FLOOR PROBES SETTLED AT Z={expected_z:.3f} m",
        (22, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (90, 240, 120) if not failures else (80, 80, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Green spheres are dynamic; reconstructed scan is visual-only; clean slab supplies contact.",
        (22, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    image_path = os.path.join(output_dir, "gate2_floor_drop.png")
    cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    result = {
        "gate": 2,
        "passed": not failures,
        "steps": args_cli.steps,
        "expected_center_z_m": expected_z,
        "probes": records,
        "image": image_path,
    }
    result_path = os.path.join(output_dir, "gate2_floor_drop_results.json")
    with open(result_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"Gate 2 failed for {len(failures)} probes")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
