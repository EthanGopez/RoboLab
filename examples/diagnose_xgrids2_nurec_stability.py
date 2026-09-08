# SPDX-License-Identifier: Apache-2.0

"""Quantitatively verify that a fixed XGRIDS2 NuRec view does not flicker."""

import argparse
import json
import os
import sys

import cv2  # import before Isaac Lab
from isaaclab.app import AppLauncher


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from robolab.core.utils.nurec_rendering import (  # noqa: E402
    configure_nurec_render_products,
    configure_plain_nurec_volume,
    require_single_gpu,
)


parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=120)
parser.add_argument(
    "--output-dir", default="output/xgrids2_lab/nurec_stability"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
require_single_gpu(args_cli)
simulation_app = AppLauncher(args_cli).app
render_settings = configure_plain_nurec_volume()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402


WRAPPER_USD = os.path.join(
    REPO_ROOT, "assets", "scenes", "xgrids2_lab", "xgrids2_lab_zup.usda"
)


@configclass
class StabilitySceneCfg(InteractiveSceneCfg):
    lab = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/xgrids2_lab",
        spawn=sim_utils.UsdFileCfg(usd_path=WRAPPER_USD),
    )
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",
        height=360,
        width=640,
        update_period=0.0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            horizontal_aperture=20.0,
            clipping_range=(0.05, 2000.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(convention="opengl"),
    )
    camera_b = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/camera_b",
        height=270,
        width=480,
        update_period=0.0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            horizontal_aperture=20.0,
            clipping_range=(0.05, 2000.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(convention="opengl"),
    )


def main() -> None:
    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0, render_interval=1, use_fabric=False
    )
    sim_cfg.render.antialiasing_mode = "OFF"
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(
        StabilitySceneCfg(
            num_envs=1,
            env_spacing=60.0,
            replicate_physics=False,
            lazy_sensor_update=False,
        )
    )
    sim.reset()
    scene.reset()
    camera = scene["camera"]
    camera_b = scene["camera_b"]
    render_products = configure_nurec_render_products(camera, camera_b)
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[0.0, 0.0, 1.625]], device=camera.device),
        targets=torch.tensor([[5.0, 0.0, 1.625]], device=camera.device),
    )
    camera_b.set_world_poses_from_view(
        eyes=torch.tensor([[5.0, 0.0, 1.625]], device=camera_b.device),
        targets=torch.tensor([[0.0, 0.0, 1.625]], device=camera_b.device),
    )

    for _ in range(30):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    frames = []
    moving_frames = []
    means = []
    differences = []
    moving_means = []
    moving_differences = []
    previous = None
    moving_previous = None
    for frame_index in range(args_cli.frames):
        phase = frame_index / max(1, args_cli.frames - 1)
        moving_eye = (4.56 * phase, -0.76 * phase, 1.625)
        camera_b.set_world_poses_from_view(
            eyes=torch.tensor([moving_eye], device=camera_b.device),
            targets=torch.tensor(
                [[moving_eye[0] + 5.0, moving_eye[1], moving_eye[2]]],
                device=camera_b.device,
            ),
        )
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        rgb = (
            camera.data.output["rgb"][0, :, :, :3]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        moving_rgb = (
            camera_b.data.output["rgb"][0, :, :, :3]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        frames.append(rgb.copy())
        moving_frames.append(moving_rgb.copy())
        means.append(float(np.mean(rgb)))
        moving_means.append(float(np.mean(moving_rgb)))
        if previous is not None:
            differences.append(float(np.mean(np.abs(rgb.astype(np.int16) - previous))))
            moving_differences.append(
                float(np.mean(np.abs(moving_rgb.astype(np.int16) - moving_previous)))
            )
        previous = rgb.astype(np.int16)
        moving_previous = moving_rgb.astype(np.int16)

    video_path = os.path.join(output_dir, "fixed_camera.mp4")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (2 * frames[0].shape[1], frames[0].shape[0]),
    )
    for frame, moving_frame in zip(frames, moving_frames):
        moving_frame = cv2.resize(
            moving_frame,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        writer.write(cv2.cvtColor(np.concatenate((frame, moving_frame), axis=1), cv2.COLOR_RGB2BGR))
    writer.release()

    result = {
        "frames": len(frames),
        "mean_luminance_min": min(means),
        "mean_luminance_max": max(means),
        "mean_luminance_range": max(means) - min(means),
        "maximum_mean_absolute_frame_difference": max(differences, default=0.0),
        "moving_camera_luminance_min": min(moving_means),
        "moving_camera_luminance_max": max(moving_means),
        "moving_camera_maximum_mean_absolute_frame_difference": max(
            moving_differences, default=0.0
        ),
        "render_settings": render_settings,
        "render_products": render_products,
        "video": video_path,
    }
    result["passed"] = (
        result["mean_luminance_range"] < 2.0
        and result["maximum_mean_absolute_frame_difference"] < 2.0
    )
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    if not cv2.imwrite(
        os.path.join(output_dir, "fixed_camera.png"),
        cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR),
    ):
        raise RuntimeError("Could not save fixed-camera still")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError("Fixed NuRec camera still flickers")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
