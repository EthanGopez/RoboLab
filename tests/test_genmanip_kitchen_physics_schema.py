# SPDX-License-Identifier: Apache-2.0

"""Static contract tests for the physics-valid GenManip kitchen."""

import math

import gymnasium as gym
import pytest

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from robolab.registrations.kinova_mobile import (
    ENV_NAME,
    IK_ENV_NAME,
    register_genmanip_kitchen_mobile_env,
    register_genmanip_kitchen_mobile_ik_env,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
    BASE_CLEARANCE_INFLATION,
    CAN_DYNAMIC_FRICTION,
    CAN_HALF_HEIGHT,
    CAN_MASS_KG,
    CAN_PHYSICS_USD_RELATIVE,
    CAN_RADIUS,
    CAN_STATIC_FRICTION,
    ORACLE_PICK_REGION,
    PICK_REGION,
    PLACE_REGION,
    ORACLE_PLACE_REGION,
    ROBOT_BASE_RADIUS,
    SINK_OPENING,
    STAGING_POSES,
    STATIC_COLLISION_BOXES,
    base_path_is_clear,
    base_pose_is_clear,
    swept_center_intersects_box,
    swept_proxy_is_clear,
)
from robolab.tasks.mobile_manipulation.kitchen_can_transfer_task import (
    GenManipKitchenCanTransferTask,
    GenManipKitchenPhysicsSceneCfg,
)
from robolab.tasks.mobile_manipulation.kitchen_scenarios import (
    CAN_OTHER_SIDE_SCENARIO,
    SCENARIOS,
    WEST_ORACLE_PICK_REGION,
)


def test_all_overlay_fixtures_are_static_colliders():
    scene_cfg = GenManipKitchenPhysicsSceneCfg()
    for box in STATIC_COLLISION_BOXES:
        cfg = getattr(scene_cfg, box.name)
        assert isinstance(cfg, AssetBaseCfg)
        assert cfg.spawn.collision_props.collision_enabled is True
        assert cfg.spawn.rigid_props is None, f"{box.name} must not be a dynamic/kinematic rigid body"
        assert cfg.spawn.mass_props is None, f"{box.name} must not have mass"
        assert all(size > 0.0 for size in box.size_xyz)


def test_can_wrapper_has_explicit_dynamic_physics():
    can_cfg = GenManipKitchenPhysicsSceneCfg().can
    assert isinstance(can_cfg, RigidObjectCfg)
    assert can_cfg.spawn.rigid_props.rigid_body_enabled is True
    assert can_cfg.spawn.rigid_props.kinematic_enabled is False
    assert can_cfg.spawn.rigid_props.disable_gravity is False
    assert can_cfg.spawn.mass_props.mass == CAN_MASS_KG
    assert can_cfg.spawn.collision_props.collision_enabled is True
    assert can_cfg.spawn.activate_contact_sensors is True
    assert can_cfg.spawn.usd_path.endswith(CAN_PHYSICS_USD_RELATIVE)
    wrapper_text = open(can_cfg.spawn.usd_path, encoding="utf-8").read()
    assert 'def Cylinder "collision"' in wrapper_text
    assert f"double radius = {CAN_RADIUS}" in wrapper_text
    assert f"double height = {2.0 * CAN_HALF_HEIGHT}" in wrapper_text
    assert 'token visibility = "invisible"' in wrapper_text
    assert 'def Xform "visual"' in wrapper_text
    assert "bool physics:collisionEnabled = 0" in wrapper_text
    assert f"physics:mass = {CAN_MASS_KG}" in wrapper_text
    assert f"physics:staticFriction = {CAN_STATIC_FRICTION}" in wrapper_text
    assert f"physics:dynamicFriction = {CAN_DYNAMIC_FRICTION}" in wrapper_text


def test_sink_opening_is_not_filled_by_a_worktop_slab():
    sink_center = SINK_OPENING.center
    top_boxes = [
        box
        for box in STATIC_COLLISION_BOXES
        if box.name.startswith("island_top_")
    ]
    assert top_boxes
    assert not any(box.footprint.contains(sink_center) for box in top_boxes)


def test_pick_and_place_centers_have_physical_support():
    pick_support = next(box for box in STATIC_COLLISION_BOXES if box.name == "island_top_east")
    place_support = next(box for box in STATIC_COLLISION_BOXES if box.name == "counter_right_top")
    assert pick_support.footprint.contains(PICK_REGION.center)
    assert place_support.footprint.contains(PLACE_REGION.center)
    assert PLACE_REGION.contains(ORACLE_PLACE_REGION.center)
    assert PLACE_REGION.contains(
        (ORACLE_PLACE_REGION.xmin, ORACLE_PLACE_REGION.ymin)
    )
    assert PLACE_REGION.contains(
        (ORACLE_PLACE_REGION.xmax, ORACLE_PLACE_REGION.ymax)
    )
    assert PICK_REGION.contains(ORACLE_PICK_REGION.center)
    assert PICK_REGION.contains(
        (ORACLE_PICK_REGION.xmin, ORACLE_PICK_REGION.ymin)
    )
    assert PICK_REGION.contains(
        (ORACLE_PICK_REGION.xmax, ORACLE_PICK_REGION.ymax)
    )


def test_staging_poses_and_full_path_clear_inflated_fixtures():
    assert BASE_CLEARANCE_INFLATION > ROBOT_BASE_RADIUS
    assert all(base_pose_is_clear(pose[:2]) for pose in STAGING_POSES.values())
    assert base_path_is_clear(STAGING_POSES["pick"][:2], STAGING_POSES["place"][:2])
    assert base_path_is_clear(
        STAGING_POSES["place"][:2], STAGING_POSES["retreat"][:2]
    )


def test_swept_proxy_detects_a_between_frame_wall_crossing():
    back_wall = next(box for box in STATIC_COLLISION_BOXES if box.name == "back_wall")
    assert swept_center_intersects_box(
        (1.0, 2.40, 1.50),
        (1.0, 2.70, 1.50),
        back_wall,
        (0.04, 0.04, 0.04),
    )


def test_audited_raised_can_corridor_is_swept_clear():
    assert swept_proxy_is_clear(
        (0.20, 0.10, 1.13),
        (1.30, 0.10, 1.13),
        (CAN_RADIUS, CAN_RADIUS, CAN_HALF_HEIGHT),
    )
    assert not swept_proxy_is_clear(
        (0.00, 0.10, 1.07),
        (0.30, 0.10, 1.07),
        (CAN_RADIUS, CAN_RADIUS, CAN_HALF_HEIGHT),
    )


def test_environment_registers_as_robolab_env():
    env_cfg, env_name = register_genmanip_kitchen_mobile_env()
    assert env_name == ENV_NAME
    spec = gym.spec(env_name)
    assert spec.entry_point == "robolab.core.environments.env:RobolabEnv"
    assert spec.kwargs["env_cfg_entry_point"] is env_cfg


def test_ik_environment_registers_with_mobile_base_flange_pose_and_gripper():
    env_cfg_type, env_name = register_genmanip_kitchen_mobile_ik_env()
    assert env_name == IK_ENV_NAME
    spec = gym.spec(env_name)
    assert spec.entry_point == "robolab.core.environments.env:RobolabEnv"
    assert spec.kwargs["env_cfg_entry_point"] is env_cfg_type
    env_cfg = env_cfg_type()
    assert env_cfg.actions.base.max_linear_speed == 0.6
    assert env_cfg.actions.arm.body_name == "robotiq_85_base_link"
    assert env_cfg.actions.arm.controller.command_type == "pose"
    assert env_cfg.actions.arm.controller.use_relative_mode is False
    assert env_cfg.sim.physx.max_velocity_iteration_count == 8
    assert set(env_cfg.contact_gripper) == {
        "gripper",
        "left_fingertip",
        "right_fingertip",
    }


def test_west_oracle_region_is_inside_registered_static_support():
    scene_cfg = GenManipKitchenPhysicsSceneCfg()
    support = next(
        box for box in STATIC_COLLISION_BOXES if box.name == "island_top_west"
    )
    for point in (
        WEST_ORACLE_PICK_REGION.center,
        (WEST_ORACLE_PICK_REGION.xmin, WEST_ORACLE_PICK_REGION.ymin),
        (WEST_ORACLE_PICK_REGION.xmax, WEST_ORACLE_PICK_REGION.ymax),
    ):
        assert support.footprint.contains(point)
    assert isinstance(scene_cfg.island_top_west, AssetBaseCfg)
    assert "island_top_west" in GenManipKitchenCanTransferTask.contact_object_list
    assert CAN_OTHER_SIDE_SCENARIO.pick_support_name == "island_top_west"


def test_all_hard_scenario_base_routes_clear_inflated_fixtures():
    for scenario in SCENARIOS.values():
        previous = scenario.initial_base_pose[:2]
        assert base_pose_is_clear(previous), scenario.name
        for waypoint in scenario.pre_pick_waypoints:
            current = waypoint[:2]
            assert base_pose_is_clear(current), (scenario.name, current)
            assert base_path_is_clear(previous, current), (scenario.name, previous, current)
            previous = current

        if not scenario.pre_pick_waypoints:
            previous = STAGING_POSES["pick"][:2]
        for waypoint in scenario.carry_waypoints:
            current = waypoint[:2]
            assert base_pose_is_clear(current), (scenario.name, current)
            assert base_path_is_clear(previous, current), (scenario.name, previous, current)
            previous = current


def test_place_route_rotates_base_to_face_counter_with_fixed_camera():
    assert STAGING_POSES["pick"][3] == 0.0
    assert STAGING_POSES["place"][3] == pytest.approx(math.pi)
    assert STAGING_POSES["retreat"][3] == pytest.approx(math.pi)
    for scenario in SCENARIOS.values():
        approach, facing = scenario.carry_waypoints[-2:]
        assert approach[:2] == facing[:2] == STAGING_POSES["place"][:2]
        assert approach[2] == 0.0
        assert facing[2] == pytest.approx(math.pi)
