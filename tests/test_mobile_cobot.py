# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch

from robolab.robots.mobile_base_math import (
    body_twist_to_world,
    clamp_vector_norm,
    jerk_limited_velocity_step,
    planar_base_wrench,
    quaternion_z_axis_world,
    slew_rate_limit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET = REPO_ROOT / "assets" / "robots" / "kinova_mobile_cobot" / "kinova_mobile_cobot.usda"


def test_mobile_asset_composes_existing_kinova_and_handoff_fixtures():
    contents = ASSET.read_text()
    assert "kinova_gen3_7dof_robotiq_2f85.usd" in contents
    base_link_declaration = contents.split('over Xform "base_link"', 1)[1].split("{", 1)[0]
    assert "PhysicsArticulationRootAPI" in base_link_declaration
    assert "PhysicsMassAPI" in base_link_declaration
    assert "float physics:mass = 65" in contents
    assert "float3 physics:diagonalInertia = (4, 4, 6)" in contents
    wrapper_declaration = contents.split('def Xform "KinovaMobileCobot"', 1)[1].split(")", 1)[0]
    assert "PhysicsArticulationRootAPI" not in wrapper_declaration
    assert 'def Cylinder "drive_base"' in contents
    assert "double radius = 0.3" in contents
    assert "double height = 0.162" in contents
    assert 'def Xform "wheels"' not in contents
    assert 'def Camera "navigation_camera"' not in contents
    assert 'def Xform "navigation_camera_link"' not in contents
    assert 'def PhysicsFixedJoint "navigation_camera_joint"' not in contents
    assert "double3 xformOp:translate = (0.0656, -0.3507, 0.60)" in contents


def test_gripper_has_symmetric_flat_rubber_collision_pads():
    contents = ASSET.read_text()
    assert contents.count('def Cube "flat_contact_pad_collision"') == 2
    assert contents.count(
        'prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]'
    ) == 2
    assert "double3 xformOp:scale = (0.003175, 0.011, 0.01875)" in contents
    assert "double3 xformOp:translate = (0.008, 0, 0.030)" in contents
    assert "double3 xformOp:translate = (-0.008, 0, 0.030)" in contents
    assert 'float physics:staticFriction = 1.5' in contents
    assert 'float physics:dynamicFriction = 1.2' in contents
    assert contents.count("float physxCollision:contactOffset = 0.001") == 2
    assert contents.count("float physxCollision:restOffset = 0") == 2
    assert contents.count("bool physics:collisionEnabled = false") == 2
    assert contents.count("material:binding:physics = </KinovaMobileCobot/gripper_pad_physics>") >= 2
    assert contents.count("PhysxMimicJointAPI:rotY") == 4
    assert contents.count(
        "physxMimicJoint:rotY:referenceJoint = "
        "</KinovaMobileCobot/joints/robotiq_85_left_knuckle_joint>"
    ) == 2
    assert contents.count(
        "physxMimicJoint:rotY:referenceJoint = "
        "</KinovaMobileCobot/joints/robotiq_85_right_knuckle_joint>"
    ) == 2


def test_body_twist_is_rotated_by_root_yaw():
    half = torch.tensor(torch.pi / 4)
    quat = torch.tensor([[torch.cos(half), 0.0, 0.0, torch.sin(half)]])
    twist = body_twist_to_world(torch.tensor([[1.0, 0.0, 0.5]]), quat)
    torch.testing.assert_close(twist, torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.5]]), atol=1e-6, rtol=0)


def test_slew_rate_limit_bounds_each_control_axis():
    result = slew_rate_limit(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, -2.0, 0.2]]),
        torch.tensor([0.1, 0.1, 0.1]),
    )
    torch.testing.assert_close(result, torch.tensor([[0.1, -0.1, 0.1]]))



def test_jerk_limited_velocity_bounds_acceleration_change():
    velocity, acceleration = jerk_limited_velocity_step(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        max_acceleration=torch.tensor([0.5, 0.5, 0.5]),
        max_jerk=torch.tensor([1.0, 1.0, 1.0]),
        dt=0.1,
    )
    torch.testing.assert_close(
        acceleration, torch.tensor([[0.1, 0.0, 0.0]]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        velocity, torch.tensor([[0.01, 0.0, 0.0]]), atol=1e-6, rtol=0
    )



def test_navigation_camera_uses_raised_forward_facing_real_mount():
    from robolab.robots.kinova_mobile_cobot import (
        HolonomicBaseVelocityActionCfg,
        KinovaMobileCobotCfg,
    )
    from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
        ISLAND_SURFACE_Z,
        ROBOT_BASE_Z,
    )

    camera = KinovaMobileCobotCfg().navigation_camera
    assert camera.update_latest_camera_pose
    # The fixed local offset is authored once in the robot sensor config. The
    # lens sits 6.3 cm above the island at nominal base Z.
    assert ROBOT_BASE_Z + camera.offset.pos[2] > ISLAND_SURFACE_Z
    assert camera.spawn is not None
    assert camera.offset.pos == (0.0656, -0.3507, 0.64)
    assert camera.offset.rot == (0.9961947, 0.0, 0.0871557, 0.0)
    assert camera.offset.convention == "world"
    assert camera.prim_path.endswith(
        "/robot/base_link/navigation_camera"
    )
    assert not hasattr(HolonomicBaseVelocityActionCfg(), "attached_camera_name")
    source = (REPO_ROOT / "robolab" / "robots" / "kinova_mobile_cobot.py").read_text()
    assert "set_world_poses(" not in source
    assert "attached_camera_name" not in source


def test_policy_cameras_expose_rgb_and_depth():
    from robolab.robots.kinova_gen3 import KinovaGen3Cfg
    from robolab.robots.kinova_mobile_cobot import KinovaMobileCobotCfg

    assert KinovaGen3Cfg().wrist_cam.data_types == [
        "rgb",
        "distance_to_image_plane",
    ]
    assert KinovaGen3Cfg().wrist_cam.update_latest_camera_pose
    assert KinovaMobileCobotCfg().navigation_camera.data_types == [
        "rgb",
        "distance_to_image_plane",
    ]


def test_mobile_gripper_has_force_controlled_drive_contract():
    from robolab.robots.kinova_mobile_cobot import KinovaMobileCobotCfg

    robot_cfg = KinovaMobileCobotCfg().robot
    assert robot_cfg.spawn.articulation_props.solver_position_iteration_count == 124
    actuator = robot_cfg.actuators["gripper"]
    assert actuator.stiffness == 100.0
    assert actuator.damping == 12.0
    assert actuator.effort_limit == 50.0
    assert actuator.velocity_limit == 0.5
    assert actuator.joint_names_expr == [
        "robotiq_85_left_knuckle_joint",
        "robotiq_85_right_knuckle_joint",
    ]
    passive = KinovaMobileCobotCfg().robot.actuators["gripper_mimics"]
    assert len(passive.joint_names_expr) == 4
    assert passive.stiffness == 0.0
    assert passive.effort_limit == 0.0


def test_vector_norm_clamp_preserves_direction():
    result = clamp_vector_norm(torch.tensor([[3.0, 4.0]]), 2.5)
    torch.testing.assert_close(result, torch.tensor([[1.5, 2.0]]))


def test_quaternion_z_axis_reports_tilt_in_world_frame():
    half_roll = torch.tensor(torch.pi / 12.0)
    quaternion = torch.tensor(
        [[torch.cos(half_roll), torch.sin(half_roll), 0.0, 0.0]]
    )
    axis = quaternion_z_axis_world(quaternion)
    roll = torch.tensor(torch.pi / 6.0)
    expected = torch.tensor(
        [[0.0, -torch.sin(roll), torch.cos(roll)]]
    )
    torch.testing.assert_close(axis, expected, atol=1.0e-6, rtol=0.0)


def test_planar_wrench_is_bounded_and_restores_height_and_tilt():
    force, torque = planar_base_wrench(
        target_world_twist=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.5]]),
        measured_world_twist=torch.zeros((1, 6)),
        root_height=torch.tensor([0.40]),
        target_height=torch.tensor([0.50]),
        body_z_axis_world=torch.tensor([[0.0, -0.5, 0.8660254]]),
        linear_velocity_gain=10.0,
        angular_velocity_gain=10.0,
        height_stiffness=20.0,
        height_damping=2.0,
        tilt_stiffness=8.0,
        tilt_damping=1.0,
        maximum_planar_force=5.0,
        maximum_vertical_force=1.0,
        maximum_yaw_torque=2.0,
        maximum_tilt_torque=3.0,
    )
    torch.testing.assert_close(force, torch.tensor([[5.0, 0.0, 1.0]]))
    torch.testing.assert_close(torque, torch.tensor([[-3.0, 0.0, 2.0]]))


def test_mobile_base_controller_does_not_overwrite_runtime_root_state():
    source = (REPO_ROOT / "robolab" / "robots" / "kinova_mobile_cobot.py").read_text()
    assert "write_root_link_pose_to_sim" not in source
    assert "write_root_velocity_to_sim" not in source
    assert "set_external_force_and_torque" in source
