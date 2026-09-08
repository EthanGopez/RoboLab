# SPDX-License-Identifier: Apache-2.0

"""Static gates for the route-first XGRIDS2 lab physics contract."""

import math

import numpy as np
import pytest

from robolab.core.task.task import verify_task_valid
from robolab.registrations.kinova_mobile.auto_env_registrations_xgrids2_lab import (
    XGRIDS2_LAB_ENV_NAME,
    register_xgrids2_lab_mobile_env,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_navigation_task import (
    XGRIDS2LabKinovaMobileCobotCfg,
    XGRIDS2LabNavigationTask,
    XGRIDS2LabPhysicsSceneCfg,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_navigation_oracle import (
    WaypointFollower,
    world_velocity_to_body,
)

from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
    COLLISION_SEGMENTS,
    DEFAULT_REQUIRED_LINE_CLEARANCE_M,
    FLOOR_CENTER_Z_M,
    FLOOR_THICKNESS_M,
    LABAPT_TO_XGRIDS_THETA_RAD,
    MOBILE_KINOVA_TRANSPORT_RADIUS_M,
    PHYSICS_REGION,
    PROXY_HEIGHT_M,
    PROXY_WIDTH_M,
    RAW_STAGING_POSES,
    ROBOT_BASE_Z_M,
    STAGING_POSES,
    XGRIDS_FLOOR_Z,
    XGRIDS_VISUAL_USD_RELATIVE,
    XGRIDS_VISUAL_Z_OFFSET_M,
    base_pose_is_clear,
    build_navigation_grid,
    distance_to_segments,
    planned_route,
    route_minimum_clearance,
)


def test_gate1_collision_contract_is_finite_and_route_scoped():
    assert 100 <= len(COLLISION_SEGMENTS) <= 300
    assert 0.02 <= PROXY_WIDTH_M <= 0.10
    assert PROXY_HEIGHT_M >= 1.0
    for segment in COLLISION_SEGMENTS:
        values = (*segment.start_xy, *segment.end_xy, segment.length, segment.yaw)
        assert np.isfinite(values).all(), segment.name
        assert segment.length >= 0.15
        assert -math.pi <= segment.yaw <= math.pi
        assert (
            PHYSICS_REGION.contains(segment.start_xy)
            or PHYSICS_REGION.contains(segment.end_xy)
            or segment.length > 0.0
        )


def test_gate1_registered_map_heading_is_a_rigid_rotation():
    assert math.degrees(LABAPT_TO_XGRIDS_THETA_RAD) == pytest.approx(
        35.61771832058649, abs=1.0e-9
    )


def test_gate1_scene_uses_static_collision_only_overlay():
    scene = XGRIDS2LabPhysicsSceneCfg()
    floor = scene.floor
    assert floor.spawn.collision_props.collision_enabled is True
    assert floor.spawn.rigid_props is None
    assert floor.spawn.mass_props is None
    for segment in COLLISION_SEGMENTS:
        cfg = getattr(scene, segment.name)
        assert cfg.spawn.collision_props.collision_enabled is True
        assert cfg.spawn.rigid_props is None
        assert cfg.spawn.mass_props is None
        assert cfg.spawn.size[1] == PROXY_WIDTH_M
        assert cfg.spawn.size[2] == PROXY_HEIGHT_M


def test_gate2_clean_floor_top_is_exactly_z_zero():
    assert FLOOR_CENTER_Z_M + 0.5 * FLOOR_THICKNESS_M == XGRIDS_FLOOR_Z
    assert PHYSICS_REGION.size == (8.0, 6.0)


def test_gate2_vendor_visual_is_registered_to_the_physics_floor():
    assert XGRIDS_VISUAL_USD_RELATIVE.endswith("/0/ut_lab_1.usdz")
    assert XGRIDS_VISUAL_Z_OFFSET_M == pytest.approx(1.075)
    scene = XGRIDS2LabPhysicsSceneCfg()
    assert scene.lab_visual.init_state.pos == (0.0, 0.0, XGRIDS_VISUAL_Z_OFFSET_M)


def test_gate4_xgrids_robot_uses_stable_rgb_only_contract():
    cfg = XGRIDS2LabKinovaMobileCobotCfg()
    assert cfg.navigation_camera.data_types == ["rgb"]
    assert cfg.wrist_cam.data_types == ["rgb"]
    assert cfg.robot.actuators["arm_1_to_4"].stiffness == 800.0
    assert cfg.robot.actuators["arm_1_to_4"].damping == 40.0
    assert cfg.robot.actuators["arm_5_to_7"].stiffness == 800.0
    assert cfg.robot.actuators["arm_5_to_7"].damping == 40.0


def test_gate3_unsafe_raw_snack_pose_is_explicitly_corrected():
    raw = RAW_STAGING_POSES["snack_table"]
    corrected = STAGING_POSES["snack_table"]
    assert not base_pose_is_clear(raw[:2])
    assert base_pose_is_clear(corrected[:2])
    assert math.dist(raw[:2], corrected[:2]) < 0.23
    assert corrected[2] == ROBOT_BASE_Z_M
    assert corrected[3] == raw[3]


def test_gate3_staging_poses_and_inflated_route_are_clear():
    grid = build_navigation_grid()
    for pose in STAGING_POSES.values():
        assert base_pose_is_clear(pose[:2])
        cell = grid.nearest_cell(pose[:2], require_free=True)
        assert grid.free[cell]
    path = planned_route()
    assert np.isfinite(path).all()
    assert np.linalg.norm(path[0] - STAGING_POSES["dining_table"][:2]) < 1.0e-9
    assert np.linalg.norm(path[-1] - STAGING_POSES["snack_table"][:2]) < 1.0e-9
    assert route_minimum_clearance(path) >= DEFAULT_REQUIRED_LINE_CLEARANCE_M


def test_gate3_robot_radius_parameter_changes_free_space():
    small = build_navigation_grid(robot_radius=0.20)
    mobile_kinova = build_navigation_grid(robot_radius=MOBILE_KINOVA_TRANSPORT_RADIUS_M)
    assert np.count_nonzero(small.free) > np.count_nonzero(mobile_kinova.free)


def test_gate4_task_is_valid_and_registered_as_robolab_env():
    valid, error = verify_task_valid(XGRIDS2LabNavigationTask)
    assert valid, error
    _, name = register_xgrids2_lab_mobile_env()
    assert name == XGRIDS2_LAB_ENV_NAME
    import gymnasium as gym

    spec = gym.spec(name)
    assert spec.entry_point == "robolab.core.environments.env:RobolabEnv"


def test_gate5_world_velocity_is_converted_to_robot_body_frame():
    body = world_velocity_to_body(np.asarray((1.0, 0.0)), math.pi / 2.0)
    assert body == pytest.approx((0.0, -1.0), abs=1.0e-9)


def test_gate5_waypoint_follower_faces_fixed_sensor_without_moving_it():
    follower = WaypointFollower(
        np.asarray(((0.0, 0.0), (1.0, 0.0))),
        final_yaw=0.25,
    )
    command, done = follower.command(np.asarray((0.0, 0.0)), yaw_world=0.0)
    assert not done
    assert command[0] > 0.0
    assert command[2] == pytest.approx(0.0)
    command, done = follower.command(np.asarray((1.0, 0.0)), yaw_world=0.25)
    assert done
    assert command == pytest.approx((0.0, 0.0, 0.0))
