# SPDX-License-Identifier: Apache-2.0

"""Isaac simulation tests for the registered GenManip kitchen environment."""

from __future__ import annotations

import math

import pytest
import torch
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

from robolab.core.environments.runtime import create_env
from robolab.core.world.world_state import get_world
from robolab.registrations.kinova_mobile import (
    ENV_NAME,
    register_genmanip_kitchen_mobile_env,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
    CAN_HALF_HEIGHT,
    COUNTER_SURFACE_Z,
    ISLAND_SURFACE_Z,
    PICK_REGION,
    PLACE_REGION,
    ROBOT_BASE_Z,
    STATIC_COLLISION_BOXES,
    STAGING_POSES,
    base_path_is_clear,
    base_pose_is_clear,
)
from robolab.tasks.mobile_manipulation.kitchen_manipulation_predicates import (
    can_supported_by_island,
)


def _matrix_values(matrix) -> tuple[float, ...]:
    return tuple(float(value) for row in matrix for value in row)


def _action_slices(env) -> dict[str, slice]:
    offset = 0
    output = {}
    for name, size in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
        output[name] = slice(offset, offset + size)
        offset += size
    return output


def _hold_action(env) -> torch.Tensor:
    """Hold the arm at its default joint state while commanding a stopped base."""

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


def _step(env, action: torch.Tensor, count: int) -> None:
    for _ in range(count):
        _, _, terminated, truncated, _ = env.step(action)
        assert not bool(torch.any(terminated))
        assert not bool(torch.any(truncated))


def _set_robot_stage(env, stage: str) -> None:
    robot = env.scene["robot"]
    x, y, z, yaw = STAGING_POSES[stage]
    half_yaw = yaw / 2.0
    pose = torch.tensor(
        [[x, y, z, math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]],
        dtype=torch.float32,
        device=env.device,
    )
    pose[:, :3] += env.scene.env_origins[:, :3]
    velocity = torch.zeros((env.num_envs, 6), device=env.device)
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(velocity)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)


@pytest.fixture(scope="module")
def kitchen_env():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for Isaac kitchen physics tests")
    register_genmanip_kitchen_mobile_env()
    env, _ = create_env(
        ENV_NAME,
        num_envs=1,
        env_spacing=8.0,
        use_fabric=False,
    )
    try:
        yield env
    finally:
        env.close()


def test_can_drops_and_settles_on_island(kitchen_env):
    env = kitchen_env
    env.reset_eval_state()
    env.reset()
    action = _hold_action(env)

    max_height = -float("inf")
    min_height = float("inf")
    for _ in range(90):  # 3 seconds at the 30 Hz environment step.
        _step(env, action, 1)
        z = float((env.scene["can"].data.root_pos_w[0, 2] - env.scene.env_origins[0, 2]).item())
        max_height = max(max_height, z)
        min_height = min(min_height, z)

    can = env.scene["can"]
    position = can.data.root_pos_w[0] - env.scene.env_origins[0]
    linear_speed = torch.linalg.vector_norm(can.data.root_lin_vel_w[0]).item()
    angular_speed = torch.linalg.vector_norm(can.data.root_ang_vel_w[0]).item()
    expected_z = ISLAND_SURFACE_Z + CAN_HALF_HEIGHT

    assert torch.isfinite(position).all()
    assert PICK_REGION.contains((float(position[0]), float(position[1])), tolerance=0.005)
    assert abs(float(position[2]) - expected_z) <= 0.006
    assert min_height >= expected_z - 0.003
    assert max_height <= expected_z + 0.03
    assert linear_speed < 0.02
    assert angular_speed < 0.15
    world = get_world(env)
    support_force = world.get_contact_force("can", "island_top_east", env_id=0)
    assert can_supported_by_island(env, env_id=0), {
        "force_w": support_force.detach().cpu().tolist(),
        "in_contact": world.in_contact("can", "island_top_east", env_id=0),
        "force_supported": world.is_supported_on_surface(
            "can", "island_top_east", env_id=0
        ),
    }


def test_live_static_overlay_has_collision_and_cannot_move(kitchen_env):
    """Verify the authored configs produced actual static USD colliders."""

    env = kitchen_env
    env.reset_eval_state()
    env.reset()
    stage = omni.usd.get_context().get_stage()
    before = {}

    for box in STATIC_COLLISION_BOXES:
        prim_path = f"/World/envs/env_0/collision_{box.name}"
        prim = stage.GetPrimAtPath(prim_path)
        assert prim.IsValid(), prim_path
        subtree = list(Usd.PrimRange(prim))
        collision_prims = [
            child for child in subtree if child.HasAPI(UsdPhysics.CollisionAPI)
        ]
        assert collision_prims, prim_path
        for collision_prim in collision_prims:
            assert UsdPhysics.CollisionAPI(collision_prim).GetCollisionEnabledAttr().Get() is True
        assert not any(child.HasAPI(UsdPhysics.RigidBodyAPI) for child in subtree), prim_path
        before[prim_path] = _matrix_values(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )

    _step(env, _hold_action(env), 30)

    for prim_path, initial_transform in before.items():
        prim = stage.GetPrimAtPath(prim_path)
        final_transform = _matrix_values(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )
        assert final_transform == pytest.approx(initial_transform, abs=1.0e-9)


@pytest.mark.parametrize("stage", ["pick", "place"])
def test_robot_spawns_and_settles_at_staging_pose(kitchen_env, stage):
    env = kitchen_env
    env.reset_eval_state()
    env.reset()
    _set_robot_stage(env, stage)
    action = _hold_action(env)
    _step(env, action, 30)

    root = env.scene["robot"].data.root_link_pos_w[0] - env.scene.env_origins[0]
    target = STAGING_POSES[stage]
    assert torch.isfinite(root).all()
    assert base_pose_is_clear(target[:2])
    assert torch.linalg.vector_norm(root[:2] - torch.tensor(target[:2], device=env.device)).item() < 0.01
    assert abs(float(root[2]) - ROBOT_BASE_Z) < 0.003


def test_mobile_base_traverses_between_staging_poses(kitchen_env):
    env = kitchen_env
    env.reset_eval_state()
    env.reset()
    _set_robot_stage(env, "pick")
    hold = _hold_action(env)
    slices = _action_slices(env)
    goal = torch.tensor(STAGING_POSES["place"][:2], device=env.device)
    assert base_path_is_clear(STAGING_POSES["pick"][:2], STAGING_POSES["place"][:2])

    reached = False
    # The base controller ramps velocity using the physics timestep.  Ten
    # seconds keeps this a conservative, acceleration-limited traversal test.
    for _ in range(300):
        root_xy = env.scene["robot"].data.root_link_pos_w[0, :2] - env.scene.env_origins[0, :2]
        delta = goal - root_xy
        distance = torch.linalg.vector_norm(delta)
        action = hold.clone()
        if distance > 0.02:
            speed = min(0.25, 1.2 * float(distance))
            action[0, slices["base"].start : slices["base"].start + 2] = delta / distance * speed
        else:
            reached = True
        _step(env, action, 1)
        root = env.scene["robot"].data.root_link_pos_w[0] - env.scene.env_origins[0]
        assert torch.isfinite(root).all()
        assert abs(float(root[2]) - ROBOT_BASE_Z) < 0.005
        assert base_pose_is_clear((float(root[0]), float(root[1])))
        if reached:
            break

    final_xy = env.scene["robot"].data.root_link_pos_w[0, :2] - env.scene.env_origins[0, :2]
    assert reached, {
        "final_xy": final_xy.detach().cpu().tolist(),
        "goal_xy": goal.detach().cpu().tolist(),
        "remaining_m": torch.linalg.vector_norm(final_xy - goal).item(),
    }
    assert torch.linalg.vector_norm(final_xy - goal).item() < 0.03
    # The can must remain supported while the base crosses the aisle.
    can_z = float((env.scene["can"].data.root_pos_w[0, 2] - env.scene.env_origins[0, 2]).item())
    assert abs(can_z - (ISLAND_SURFACE_Z + CAN_HALF_HEIGHT)) < 0.008
    assert COUNTER_SURFACE_Z < ISLAND_SURFACE_Z
    assert PLACE_REGION.contains(PLACE_REGION.center)
