# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Omnidirectional CoBot MoMa using RoboLab's Kinova Gen3 + Robotiq asset."""

import copy
import os
from collections.abc import Sequence

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from robolab.constants import ROBOTS_DIR
from robolab.robots.droid import BinaryJointPositionZeroToOneActionCfg
from robolab.robots.kinova_gen3 import (
    GRIPPER_JOINT_COMMANDS,
    GRIPPER_JOINT_NAME,
    KinovaGen3Cfg,
)
from robolab.robots.mobile_base_math import (
    body_twist_to_world,
    jerk_limited_velocity_step,
    planar_base_wrench,
    quaternion_z_axis_world,
)


_FIXED_KINOVA_CFG = KinovaGen3Cfg()

MOBILE_KINOVA_FLANGE_BODY = "robotiq_85_base_link"
MOBILE_KINOVA_LEFT_FINGERTIP_BODY = "robotiq_85_left_finger_tip_link"
MOBILE_KINOVA_RIGHT_FINGERTIP_BODY = "robotiq_85_right_finger_tip_link"
MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS = (
    GRIPPER_JOINT_NAME,
    "robotiq_85_right_knuckle_joint",
)

# Centers of the simplified rubber pad colliders in their fingertip-link
# frames.  Keeping this beside the USD contract lets grasp diagnostics compute
# contact-point velocity instead of treating the fingertip link origin as the
# contact patch.
MOBILE_KINOVA_FINGERTIP_PAD_OFFSETS = {
    MOBILE_KINOVA_LEFT_FINGERTIP_BODY: (-0.008, 0.0, 0.030),
    MOBILE_KINOVA_RIGHT_FINGERTIP_BODY: (0.008, 0.0, 0.030),
}

# Keep the aggregate name for existing tasks, while exposing each fingertip so
# a grasp can require independent left/right contacts instead of "any contact".
MOBILE_KINOVA_CONTACT_GRIPPER = {
    "gripper": "{ENV_REGEX_NS}/robot/robotiq_85_.*_finger_tip_link",
    "left_fingertip": (
        "{ENV_REGEX_NS}/robot/robotiq_85_left_finger_tip_link"
    ),
    "right_fingertip": (
        "{ENV_REGEX_NS}/robot/robotiq_85_right_finger_tip_link"
    ),
}


class HolonomicBaseVelocityAction(ActionTerm):
    """Dynamically drive a floating root from bounded body-frame velocity."""

    cfg: "HolonomicBaseVelocityActionCfg"

    def __init__(self, cfg: "HolonomicBaseVelocityActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros((self.num_envs, 3), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._applied_actions = torch.zeros_like(self._raw_actions)
        self._limits = torch.tensor(
            [cfg.max_linear_speed, cfg.max_linear_speed, cfg.max_angular_speed],
            device=self.device,
        )
        self._max_acceleration = torch.tensor(
            [cfg.max_linear_acceleration, cfg.max_linear_acceleration, cfg.max_angular_acceleration],
            device=self.device,
        )
        self._max_jerk = torch.tensor(
            [cfg.max_linear_jerk, cfg.max_linear_jerk, cfg.max_angular_jerk],
            device=self.device,
        )
        self._applied_acceleration = torch.zeros_like(self._applied_actions)
        self._base_body_id = self._asset.data.body_names.index(cfg.base_body_name)
        self._last_force_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_torque_w = torch.zeros_like(self._last_force_w)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, min=-self._limits, max=self._limits)

    def apply_actions(self) -> None:
        self._applied_actions[:], self._applied_acceleration[:] = (
            jerk_limited_velocity_step(
                self._applied_actions,
                self._applied_acceleration,
                self._processed_actions,
                max_acceleration=self._max_acceleration,
                max_jerk=self._max_jerk,
                dt=float(self._env.physics_dt),
            )
        )
        root_pos = self._asset.data.root_link_pos_w
        root_quat = self._asset.data.root_link_quat_w
        measured_twist = self._asset.data.root_link_vel_w
        world_twist = body_twist_to_world(self._applied_actions, root_quat)
        target_height = self._env.scene.env_origins[:, 2] + self.cfg.planar_height
        force_w, torque_w = planar_base_wrench(
            target_world_twist=world_twist,
            measured_world_twist=measured_twist,
            root_height=root_pos[:, 2],
            target_height=target_height,
            body_z_axis_world=quaternion_z_axis_world(root_quat),
            linear_velocity_gain=self.cfg.linear_velocity_gain,
            angular_velocity_gain=self.cfg.angular_velocity_gain,
            height_stiffness=self.cfg.height_stiffness,
            height_damping=self.cfg.height_damping,
            tilt_stiffness=self.cfg.tilt_stiffness,
            tilt_damping=self.cfg.tilt_damping,
            maximum_planar_force=self.cfg.maximum_planar_force,
            maximum_vertical_force=self.cfg.maximum_vertical_force,
            maximum_yaw_torque=self.cfg.maximum_yaw_torque,
            maximum_tilt_torque=self.cfg.maximum_tilt_torque,
        )
        self._last_force_w[:] = force_w
        self._last_torque_w[:] = torque_w
        self._asset.set_external_force_and_torque(
            force_w.unsqueeze(1),
            torque_w.unsqueeze(1),
            body_ids=[self._base_body_id],
            is_global=True,
        )

    def set_planar_pose(self, xy_w: torch.Tensor, yaw: torch.Tensor) -> None:
        """Reset command dynamics after an externally authored reset pose."""

        del xy_w, yaw
        self._applied_actions.zero_()
        self._applied_acceleration.zero_()
        self._last_force_w.zero_()
        self._last_torque_w.zero_()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._applied_actions[env_ids] = 0.0
        self._applied_acceleration[env_ids] = 0.0
        self._last_force_w[env_ids] = 0.0
        self._last_torque_w[env_ids] = 0.0


@configclass
class HolonomicBaseVelocityActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = HolonomicBaseVelocityAction
    asset_name: str = "robot"
    max_linear_speed: float = 0.6
    max_angular_speed: float = 1.0
    max_linear_acceleration: float = 1.2
    max_angular_acceleration: float = 2.0
    max_linear_jerk: float = 2.4
    max_angular_jerk: float = 4.0
    planar_height: float = 0.468
    base_body_name: str = "base_link"
    linear_velocity_gain: float = 400.0
    angular_velocity_gain: float = 160.0
    height_stiffness: float = 4000.0
    height_damping: float = 400.0
    tilt_stiffness: float = 1000.0
    tilt_damping: float = 100.0
    maximum_planar_force: float = 250.0
    maximum_vertical_force: float = 500.0
    maximum_yaw_torque: float = 150.0
    maximum_tilt_torque: float = 300.0


@configclass
class KinovaMobileCobotCfg(KinovaGen3Cfg):
    """Floating Kinova articulation with handoff-aligned CoBot fixtures."""

    # Configclass fields live on instances, so clone the fixed-base config
    # instance and override only the spawn/root properties needed for mobility.
    robot = copy.deepcopy(_FIXED_KINOVA_CFG.robot)
    robot.spawn.usd_path = os.path.join(ROBOTS_DIR, "kinova_mobile_cobot", "kinova_mobile_cobot.usda")
    robot.spawn.articulation_props.fix_root_link = False
    robot.spawn.articulation_props.solver_position_iteration_count = 124
    # The hardware description limits each driven outer knuckle to 0.5 rad/s and
    # 50 N m but contains no measured position-to-pinch-force map. The packaged
    # 100/12 position gains are paired with the oracle's slow contact-force
    # feedback and bounded preload; this avoids an impact-like full-close command.
    # Replacing these spec-bounded values with identified gains requires real
    # gripper force/displacement traces and belongs to embodiment calibration.
    robot.actuators["gripper"].joint_names_expr = list(MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS)
    robot.actuators["gripper"].stiffness = 100.0
    robot.actuators["gripper"].damping = 12.0
    robot.actuators["gripper_mimics"] = ImplicitActuatorCfg(
        joint_names_expr=[
            name for name in GRIPPER_JOINT_COMMANDS if name not in MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS
        ],
        effort_limit=0.0,
        velocity_limit=0.5,
        stiffness=0.0,
        damping=0.0,
    )
    # The handoff drive-base bottom is 0.468 m below the Kinova base_link.
    robot.init_state.pos = (0.0, 0.0, 0.468)
    # Measured position-IK seed for the pick-region center with the flange at
    # z=1.316 m. It reached the target with 0.61 mm error and remains well away
    # from all joint limits (see milestone2/position_ik_seed.json).
    robot.init_state.joint_pos.update(
        {
            "joint_1": 2.4244625568,
            "joint_2": 0.1725206971,
            "joint_3": 3.9796526432,
            "joint_4": -1.0094391108,
            "joint_5": 0.3149960935,
            "joint_6": -0.3811610639,
            "joint_7": 1.5699596405,
        }
    )

    navigation_camera = TiledCameraCfg(
        # Standard Isaac Lab mount: the camera prim is a child of the existing
        # rigid base link, so its one authored local transform follows base
        # translation and yaw without runtime camera pose writes.
        prim_path="{ENV_REGEX_NS}/robot/base_link/navigation_camera",
        update_latest_camera_pose=True,
        height=360,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.0,
            horizontal_aperture=4.0,
            vertical_aperture=2.25,
            clipping_range=(0.1, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0656, -0.3507, 0.64),
            # Face robot +X (the repository-wide forward convention) with a
            # mild ten-degree downward pitch.  The previous quaternion faced
            # -X, making the fixed mount and chassis look 180 degrees reversed.
            rot=(0.9961947, 0.0, 0.0871557, 0.0),
            convention="world",
        ),
    )


@configclass
class KinovaMobileJointPositionActionCfg:
    """Base velocity followed by the existing arm and gripper policy actions."""

    base = HolonomicBaseVelocityActionCfg()
    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_[1-7]"],
        preserve_order=True,
        use_default_offset=False,
    )
    gripper = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=list(MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS),
        open_command_expr={name: 0.0 for name in MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS},
        close_command_expr={
            name: GRIPPER_JOINT_COMMANDS[name]
            for name in MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS
        },
    )


@configclass
class KinovaMobileContinuousGripperJointPositionActionCfg(
    KinovaMobileJointPositionActionCfg
):
    """Joint-position arm with signed continuous outer-knuckle commands."""

    gripper = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS),
        preserve_order=True,
        use_default_offset=False,
    )





@configclass
class KinovaMobileIKActionCfg:
    """Planar base velocity, absolute flange-pose IK, and binary gripper.

    The seven arm values are ``(x, y, z, qw, qx, qy, qz)`` in the robot-root
    frame.  The controlled body is the Robotiq mounting flange.  Task code
    converts a desired fingertip-center TCP pose to a flange pose explicitly;
    keeping ``body_offset`` at zero avoids mixing offset-frame and root-frame
    rotational Jacobians in Isaac Lab's differential-IK action term.
    """

    base = HolonomicBaseVelocityActionCfg()
    arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint_[1-7]"],
        body_name=MOBILE_KINOVA_FLANGE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        ),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0)
        ),
    )
    gripper = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=list(MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS),
        open_command_expr={name: 0.0 for name in MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS},
        close_command_expr={
            name: GRIPPER_JOINT_COMMANDS[name]
            for name in MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS
        },
    )


@configclass
class KinovaMobileContinuousGripperIKActionCfg(KinovaMobileIKActionCfg):
    """Absolute arm IK with direct signed positions for all gripper joints.

    This keeps the robot/action interface policy agnostic while allowing a
    scripted controller to stop at the actual contact width instead of always
    driving the Robotiq mechanism to its generic fully-closed target.
    """

    gripper = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS),
        preserve_order=True,
        use_default_offset=False,
    )
