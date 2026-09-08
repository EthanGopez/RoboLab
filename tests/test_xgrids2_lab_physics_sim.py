# SPDX-License-Identifier: Apache-2.0

"""Live Isaac gates for the physics-enabled XGRIDS2 lab environment."""

import math

import numpy as np
import omni.usd
import pytest
import torch
from pxr import Usd, UsdPhysics

from robolab.core.environments.runtime import create_env
from robolab.core.utils.nurec_rendering import configure_xgrids_vendor_nurec_volume
from robolab.registrations.kinova_mobile import (
    XGRIDS2_LAB_ENV_NAME,
    register_xgrids2_lab_mobile_env,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
    COLLISION_SEGMENTS,
    ROBOT_BASE_Z_M,
    STAGING_POSES,
    base_pose_is_clear,
    planned_route,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_navigation_oracle import (
    WaypointFollower,
    wrap_angle,
)
from robolab.tasks.mobile_manipulation.xgrids2_motion_diagnostics import (
    ArmMotionDiagnostics,
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
    indices = [robot.data.joint_names.index(f"joint_{index}") for index in range(1, 8)]
    action[:, slices["arm"]] = robot.data.default_joint_pos[:, indices]
    return action


def _step(env, action: torch.Tensor, count: int) -> None:
    for _ in range(count):
        _, _, terminated, truncated, _ = env.step(action)
        assert not bool(torch.any(terminated))
        assert not bool(torch.any(truncated))


def _set_stage(env, name: str) -> None:
    x, y, z, yaw = STAGING_POSES[name]
    pose = torch.tensor(
        [[x, y, z, math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]],
        device=env.device,
    )
    pose[:, :3] += env.scene.env_origins[:, :3]
    robot = env.scene["robot"]
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)


@pytest.fixture(scope="module")
def lab_env():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for XGRIDS2 physics tests")
    configure_xgrids_vendor_nurec_volume()
    register_xgrids2_lab_mobile_env()
    env, _ = create_env(
        XGRIDS2_LAB_ENV_NAME,
        num_envs=1,
        env_spacing=12.0,
        use_fabric=True,
    )
    try:
        yield env
    finally:
        env.close()


def test_gate1_live_overlay_is_static_and_scan_collision_is_disabled(lab_env):
    env = lab_env
    env.reset_eval_state()
    env.reset()
    stage = omni.usd.get_context().get_stage()
    floor = stage.GetPrimAtPath("/World/envs/env_0/collision_floor")
    assert floor.IsValid()
    assert any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(floor))
    for segment in COLLISION_SEGMENTS:
        path = f"/World/envs/env_0/collision_{segment.name}"
        prim = stage.GetPrimAtPath(path)
        assert prim.IsValid(), path
        subtree = list(Usd.PrimRange(prim))
        assert any(child.HasAPI(UsdPhysics.CollisionAPI) for child in subtree)
        assert not any(child.HasAPI(UsdPhysics.RigidBodyAPI) for child in subtree)

    visual = stage.GetPrimAtPath("/World/envs/env_0/xgrids2_lab_visual")
    assert visual.IsValid()
    visual_colliders = [
        prim for prim in Usd.PrimRange(visual) if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    for prim in visual_colliders:
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False


@pytest.mark.parametrize("stage_name", ("dining_table", "snack_table"))
def test_gate4_robot_spawns_and_settles_at_each_stage(lab_env, stage_name):
    env = lab_env
    env.reset_eval_state()
    env.reset()
    _set_stage(env, stage_name)
    _step(env, _hold_action(env), 45)
    root = env.scene["robot"].data.root_link_pos_w[0] - env.scene.env_origins[0]
    target = STAGING_POSES[stage_name]
    assert torch.isfinite(root).all()
    assert base_pose_is_clear(target[:2])
    position_error = torch.linalg.vector_norm(root[:2] - torch.tensor(target[:2], device=env.device)).item()
    base_action = env.action_manager.get_term("base")
    diagnostics = (
        f"root={root.detach().cpu().tolist()} "
        f"base_mass={env.scene['robot'].data.default_mass[0, base_action._base_body_id].item()} "
        f"base_inertia={env.scene['robot'].data.default_inertia[0, base_action._base_body_id].detach().cpu().tolist()} "
        f"quat={env.scene['robot'].data.root_link_quat_w[0].detach().cpu().tolist()} "
        f"velocity={env.scene['robot'].data.root_link_vel_w[0].detach().cpu().tolist()} "
        f"force={base_action._last_force_w[0].detach().cpu().tolist()} "
        f"torque={base_action._last_torque_w[0].detach().cpu().tolist()}"
    )
    assert position_error < 0.01, diagnostics
    assert abs(float(root[2]) - ROBOT_BASE_Z_M) < 0.004
    assert torch.linalg.vector_norm(env.scene["robot"].data.root_link_lin_vel_w[0]).item() < 0.02


def test_gate4_parented_robot_cameras_follow_rgb_only_vendor_contract(lab_env):
    env = lab_env
    env.reset_eval_state()
    env.reset()
    _step(env, _hold_action(env), 8)
    for name in ("navigation_camera", "wrist_cam"):
        camera = env.scene[name]
        assert tuple(camera.cfg.data_types) == ("rgb",)
        rgb = camera.data.output["rgb"]
        assert rgb.shape[-1] >= 3
        assert torch.isfinite(rgb.float()).all(), name


def test_gate5_closed_loop_base_traverses_inflated_route(lab_env):
    env = lab_env
    env.reset_eval_state()
    env.reset()
    _set_stage(env, "dining_table")
    hold = _hold_action(env)
    _step(env, hold, 12)

    target = STAGING_POSES["snack_table"]
    follower = WaypointFollower(planned_route(), final_yaw=target[3])
    slices = _action_slices(env)
    robot = env.scene["robot"]
    arm_indices = [
        robot.data.joint_names.index(f"joint_{index}") for index in range(1, 8)
    ]
    arm_diagnostics = ArmMotionDiagnostics(
        env.step_dt, robot.data.joint_pos[0, arm_indices].detach().cpu().numpy()
    )
    reached = False
    minimum_clearance_ok = True
    maximum_nonvertical_force_excess = 0.0
    maximum_tilt_rad = 0.0

    for _ in range(1000):
        root_xy = (
            env.scene["robot"].data.root_link_pos_w[0, :2]
            - env.scene.env_origins[0, :2]
        )
        quaternion = env.scene["robot"].data.root_link_quat_w[0]
        yaw = math.atan2(
            2.0 * float(quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
            1.0 - 2.0 * float(quaternion[2] ** 2 + quaternion[3] ** 2),
        )
        command, reached = follower.command(root_xy.detach().cpu().numpy(), yaw)
        action = hold.clone()
        action[0, slices["base"]] = torch.as_tensor(command, device=env.device)
        _step(env, action, 1)
        arm_diagnostics.update(
            target=hold[0, slices["arm"]].detach().cpu().numpy(),
            position=robot.data.joint_pos[0, arm_indices].detach().cpu().numpy(),
            raw_velocity=robot.data.joint_vel[0, arm_indices].detach().cpu().numpy(),
            retention_active=False,
            phase="base_traverse_arm_hold",
        )
        root_xy_after = (
            env.scene["robot"].data.root_link_pos_w[0, :2]
            - env.scene.env_origins[0, :2]
        )
        minimum_clearance_ok &= base_pose_is_clear(
            tuple(float(value) for value in root_xy_after)
        )
        net_force = env.scene["base_obstacle_contact"].data.net_forces_w[0, 0]
        horizontal = float(torch.linalg.vector_norm(net_force[:2]))
        vertical = abs(float(net_force[2]))
        quaternion_after = robot.data.root_link_quat_w[0]
        body_z_component = 1.0 - 2.0 * (
            quaternion_after[1] ** 2 + quaternion_after[2] ** 2
        )
        maximum_tilt_rad = max(
            maximum_tilt_rad,
            math.acos(float(torch.clamp(body_z_component, min=-1.0, max=1.0))),
        )
        maximum_nonvertical_force_excess = max(
            maximum_nonvertical_force_excess, horizontal - vertical
        )
        if reached:
            break

    assert reached, f"route follower stopped at waypoint {follower.waypoint_index}"
    _step(env, hold, 24)
    root = env.scene["robot"].data.root_link_pos_w[0] - env.scene.env_origins[0]
    quaternion = env.scene["robot"].data.root_link_quat_w[0]
    final_yaw = math.atan2(
        2.0 * float(quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
        1.0 - 2.0 * float(quaternion[2] ** 2 + quaternion[3] ** 2),
    )
    assert minimum_clearance_ok
    assert maximum_nonvertical_force_excess < 0.5
    assert np.linalg.norm(root[:2].detach().cpu().numpy() - np.asarray(target[:2])) < 0.07
    assert abs(wrap_angle(final_yaw - target[3])) < 0.07
    assert abs(float(root[2]) - ROBOT_BASE_Z_M) < 0.004
    diagnostics = arm_diagnostics.summary()
    assert maximum_tilt_rad < math.radians(2.0), maximum_tilt_rad
    assert not diagnostics["jerk_violation"], diagnostics
    assert not diagnostics["oscillation_violation"], diagnostics
