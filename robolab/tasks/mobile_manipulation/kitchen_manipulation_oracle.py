"""Deterministic pick-carry-place oracle for the GenManip kitchen task."""

from __future__ import annotations

from enum import IntEnum
import math

import torch
from isaaclab.utils import math as math_utils
from pxr import Gf, UsdPhysics

from robolab.robots.kinova_gen3 import GRIPPER_JOINT_COMMANDS
from robolab.robots.kinova_mobile_cobot import (
    MOBILE_KINOVA_FINGERTIP_PAD_OFFSETS,
    MOBILE_KINOVA_FLANGE_BODY,
    MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS,
    MOBILE_KINOVA_LEFT_FINGERTIP_BODY,
    MOBILE_KINOVA_RIGHT_FINGERTIP_BODY,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
    BASE_POSITION_TOLERANCE,
    CARRY_BASE_SPEED,
    CARRY_REPOSITION_TCP_SPEED,
    CAN_HALF_HEIGHT,
    CAN_RADIUS,
    CARRY_ANGULAR_SETTLE_SPEED,
    CARRY_LINEAR_SETTLE_SPEED,
    CARRY_SETTLE_FRAMES,
    COUNTER_SURFACE_Z,
    EEF_ORIENTATION_TOLERANCE_RAD,
    EEF_POSITION_TOLERANCE,
    GRASP_TCP_OFFSET_FLANGE,
    GRASP_CONTACT_FORCE_N,
    GRASP_LOST_CONTACT_FRAMES,
    GRIPPER_CLOSE_SEARCH_SPEED_RAD_S,
    GRIPPER_FORCE_CONTROL_GAIN_RAD_PER_N_S,
    GRIPPER_FORCE_CONTROL_SPEED_RAD_S,
    GRIPPER_FORCE_DEADBAND_N,
    GRIPPER_MAX_PRELOAD_RAD,
    GRIPPER_TARGET_NORMAL_FORCE_N,
    GRIPPER_OPEN_SPEED_RAD_S,
    HANDOFF_ANGULAR_SETTLE_SPEED,
    HANDOFF_LINEAR_SETTLE_SPEED,
    HANDOFF_SETTLE_FRAMES,
    BASE_YAW_TOLERANCE_RAD,
    ISLAND_SURFACE_Z,
    LIFT_CLEARANCE,
    LIFT_TCP_SPEED,
    MIN_LIFT_CLEARANCE,
    ORACLE_PLACE_REGION,
    PLACE_REGION,
    PLACE_APPROACH_CLEARANCE,
    PLACE_RELEASE_CLEARANCE,
    PLACE_SIDE_GRASP_QUAT_WXYZ,
    PLACE_TARGET_TOLERANCE,
    PLACE_TCP_SPEED,
    STAGING_POSES,
    swept_proxy_is_clear,
)
from robolab.tasks.mobile_manipulation.kitchen_manipulation_predicates import (
    base_at_place_stage,
    can_bilaterally_grasped,
    can_lifted_clear,
    can_supported_by_island,
    can_supported_in_place_region,
    PhysicalGraspMonitor,
    gripper_released,
    robot_collision_free,
)
from robolab.tasks.mobile_manipulation.kitchen_can_transfer_task import (
    ORACLE_GRASP_JOINT_NAME,
)
from robolab.tasks.mobile_manipulation.kitchen_scenarios import (
    BASELINE_SCENARIO,
    KitchenManipulationScenario,
)

class OraclePhase(IntEnum):
    SETTLE = 0
    ALIGN = 1
    PREGRASP = 2
    DESCEND = 3
    CLOSE = 4
    SECURE = 5
    LIFT = 6
    CARRY_SETTLE = 7
    TRAVERSE = 8
    CARRY_REPOSITION = 9
    CARRY_REORIENT = 10
    PREPLACE = 11
    PLACE = 12
    HANDOFF_SETTLE = 13
    OPEN = 14
    RETREAT = 15
    SUCCESS = 16
    FAILURE = 17
    NAVIGATE_TO_PICK = 18


_PHASE_TIMEOUT_STEPS = {
    OraclePhase.SETTLE: 180,
    OraclePhase.ALIGN: 360,
    OraclePhase.PREGRASP: 240,
    OraclePhase.DESCEND: 180,
    OraclePhase.CLOSE: 120,
    OraclePhase.SECURE: 120,
    OraclePhase.LIFT: 480,
    OraclePhase.CARRY_SETTLE: 180,
    OraclePhase.TRAVERSE: 2400,
    OraclePhase.CARRY_REPOSITION: 480,
    OraclePhase.CARRY_REORIENT: 240,
    OraclePhase.PREPLACE: 480,
    OraclePhase.PLACE: 360,
    OraclePhase.HANDOFF_SETTLE: 240,
    OraclePhase.OPEN: 180,
    OraclePhase.RETREAT: 180,
    OraclePhase.NAVIGATE_TO_PICK: 2400,
}


class KitchenManipulationOracle:
    """Vectorized finite-state controller for one or many kitchen envs.

    The oracle commands the fingertip-center task frame (TCP), but the Isaac
    action term controls ``robotiq_85_base_link``. Every target therefore goes
    through two explicit transforms: TCP-to-flange in world coordinates, then
    world-to-mobile-root for absolute differential IK.
    """

    def __init__(
        self,
        env,
        *,
        place_xy: (
            tuple[float, float]
            | list[tuple[float, float]]
            | torch.Tensor
            | None
        ) = None,
        use_grasp_assist: bool = True,
        scenario: KitchenManipulationScenario = BASELINE_SCENARIO,
    ):
        self.env = env
        self.use_grasp_assist = use_grasp_assist
        self.scenario = scenario
        self.device = env.device
        self.num_envs = env.num_envs
        self.robot = env.scene["robot"]
        self.can = env.scene["can"]
        stage = env.scene.stage
        self.grasp_joints = [
            UsdPhysics.FixedJoint(
                stage.GetPrimAtPath(f"{env_path}/{ORACLE_GRASP_JOINT_NAME}")
            )
            for env_path in env.scene.env_prim_paths
        ]
        if any(not joint.GetPrim().IsValid() for joint in self.grasp_joints):
            raise RuntimeError("The kitchen prestartup grasp joints were not authored")
        self.flange_body_index = self.robot.data.body_names.index(
            MOBILE_KINOVA_FLANGE_BODY
        )
        arm_name_tokens = ("shoulder", "arm", "forearm", "wrist", "bracelet", "robotiq")
        self.swept_arm_body_indices = [
            index
            for index, name in enumerate(self.robot.data.body_names)
            if any(token in name for token in arm_name_tokens)
        ]
        if self.flange_body_index not in self.swept_arm_body_indices:
            self.swept_arm_body_indices.append(self.flange_body_index)
        self.action_slices = self._resolve_action_slices()
        required = {"base", "arm", "gripper"}
        if set(self.action_slices) != required:
            raise ValueError(
                f"Oracle requires action terms {sorted(required)}, got "
                f"{sorted(self.action_slices)}"
            )
        if self.action_slices["arm"].stop - self.action_slices["arm"].start != 7:
            raise ValueError("Oracle requires seven-value absolute pose IK actions")
        if (
            self.action_slices["gripper"].stop
            - self.action_slices["gripper"].start
            != len(MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS)
        ):
            raise ValueError("Oracle requires two Robotiq outer-knuckle actions")

        self.gripper_drive_joint_indices = [
            self.robot.data.joint_names.index(name)
            for name in MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS
        ]
        self.gripper_joint_indices = [
            self.robot.data.joint_names.index(name)
            for name in GRIPPER_JOINT_COMMANDS
        ]
        self.gripper_close_command = torch.tensor(
            [[GRIPPER_JOINT_COMMANDS[name] for name in MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS]],
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.gripper_signs = torch.sign(self.gripper_close_command)
        self.gripper_mimic_signs = torch.sign(
            torch.tensor(list(GRIPPER_JOINT_COMMANDS.values()), device=self.device)
        ).unsqueeze(0)
        self.gripper_mimic_drive_indices = torch.tensor(
            [0 if "left" in name else 1 for name in GRIPPER_JOINT_COMMANDS],
            dtype=torch.long,
            device=self.device,
        )
        self.grasp_monitor = PhysicalGraspMonitor(
            env,
            flange_body_name=MOBILE_KINOVA_FLANGE_BODY,
            left_fingertip_body_name=MOBILE_KINOVA_LEFT_FINGERTIP_BODY,
            right_fingertip_body_name=MOBILE_KINOVA_RIGHT_FINGERTIP_BODY,
            fingertip_pad_offsets=MOBILE_KINOVA_FINGERTIP_PAD_OFFSETS,
            gripper_joint_names=tuple(GRIPPER_JOINT_COMMANDS),
        )

        self.tcp_offset_f = torch.tensor(
            GRASP_TCP_OFFSET_FLANGE, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self.side_grasp_quat_w = torch.tensor(
            scenario.pick_grasp_quat_wxyz, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self.pick_pregrasp_offset_w = torch.tensor(
            scenario.pick_pregrasp_offset_xy, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self.place_side_grasp_quat_w = torch.tensor(
            PLACE_SIDE_GRASP_QUAT_WXYZ, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        place_xy = ORACLE_PLACE_REGION.center if place_xy is None else place_xy
        place_targets = torch.as_tensor(
            place_xy, dtype=torch.float32, device=self.device
        )
        if place_targets.shape == (2,):
            place_targets = place_targets.repeat(self.num_envs, 1)
        elif place_targets.shape != (self.num_envs, 2):
            raise ValueError(
                "place_xy must be one (x, y) pair or have shape "
                f"({self.num_envs}, 2); got {tuple(place_targets.shape)}"
            )
        inside = (
            (place_targets[:, 0] >= ORACLE_PLACE_REGION.xmin)
            & (place_targets[:, 0] <= ORACLE_PLACE_REGION.xmax)
            & (place_targets[:, 1] >= ORACLE_PLACE_REGION.ymin)
            & (place_targets[:, 1] <= ORACLE_PLACE_REGION.ymax)
        )
        if not bool(torch.all(inside).item()):
            invalid_ids = torch.nonzero(~inside, as_tuple=False).squeeze(-1)
            invalid_targets = place_targets[invalid_ids].detach().cpu().tolist()
            raise ValueError(
                f"place_xy targets {invalid_targets} for envs "
                f"{invalid_ids.detach().cpu().tolist()} lie outside "
                f"{ORACLE_PLACE_REGION}"
            )
        self.place_xy = place_targets.clone()

        self.phase = torch.full(
            (self.num_envs,), int(OraclePhase.SETTLE), dtype=torch.long, device=self.device
        )
        self.phase_steps = torch.zeros_like(self.phase)
        self.pose_stable_steps = torch.zeros_like(self.phase)
        self.carry_settle_steps = torch.zeros_like(self.phase)
        self.handoff_settle_steps = torch.zeros_like(self.phase)
        self.pick_tcp_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.secure_tcp_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.lift_tcp_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.lift_command_tcp_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.motion_command_tcp_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.approach_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.carry_flange_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.navigation_flange_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.pre_pick_waypoint_index = torch.zeros_like(self.phase)
        self.carry_waypoint_index = torch.zeros_like(self.phase)
        self.grasp_joint_positions = torch.zeros_like(self.gripper_close_command)
        self.grasp_contact_joint_positions = torch.zeros_like(
            self.gripper_close_command
        )
        self.grasp_lost_steps = torch.zeros_like(self.phase)
        self.secure_grasp_quality: list[dict | None] = [None] * self.num_envs
        self.open_joint_positions = torch.zeros_like(self.gripper_close_command)
        self.failure_reason = [""] * self.num_envs
        self.failure_diagnostics: list[dict | None] = [None] * self.num_envs
        self.grasp_constraint_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.grasp_can_pose_f = torch.zeros((self.num_envs, 7), device=self.device)
        self.control_dt = float(env.step_dt)
        self.previous_can_position = (
            self.can.data.root_pos_w - self.env.scene.env_origins
        ).clone()
        self.previous_arm_positions = (
            self.robot.data.body_pos_w[:, self.swept_arm_body_indices]
            - self.env.scene.env_origins[:, None, :]
        ).clone()

    def _resolve_action_slices(self) -> dict[str, slice]:
        offset = 0
        result = {}
        for name, size in zip(
            self.env.action_manager.active_terms,
            self.env.action_manager.action_term_dim,
        ):
            result[name] = slice(offset, offset + size)
            offset += size
        return result

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._detach_grasp_constraints(env_ids)
        self.phase[env_ids] = int(OraclePhase.SETTLE)
        self.phase_steps[env_ids] = 0
        self.pose_stable_steps[env_ids] = 0
        self.carry_settle_steps[env_ids] = 0
        self.handoff_settle_steps[env_ids] = 0
        self.pick_tcp_w[env_ids] = 0.0
        self.secure_tcp_w[env_ids] = 0.0
        self.lift_tcp_w[env_ids] = 0.0
        self.lift_command_tcp_w[env_ids] = 0.0
        self.motion_command_tcp_w[env_ids] = 0.0
        self.approach_quat_w[env_ids] = 0.0
        self.carry_flange_b[env_ids] = 0.0
        self.navigation_flange_b[env_ids] = 0.0
        self.pre_pick_waypoint_index[env_ids] = 0
        self.carry_waypoint_index[env_ids] = 0
        self.grasp_joint_positions[env_ids] = 0.0
        self.grasp_contact_joint_positions[env_ids] = 0.0
        self.grasp_lost_steps[env_ids] = 0
        self.grasp_monitor.reset(env_ids)
        self.open_joint_positions[env_ids] = 0.0
        self.grasp_can_pose_f[env_ids] = 0.0
        self.previous_can_position[env_ids] = (
            self.can.data.root_pos_w[env_ids] - self.env.scene.env_origins[env_ids]
        )
        self.previous_arm_positions[env_ids] = (
            self.robot.data.body_pos_w[env_ids][:, self.swept_arm_body_indices]
            - self.env.scene.env_origins[env_ids, None, :]
        )
        for env_id in env_ids.detach().cpu().tolist():
            self.failure_reason[env_id] = ""
            self.failure_diagnostics[env_id] = None
            self.secure_grasp_quality[env_id] = None

    def _attach_grasp_constraint(self, env_id: int) -> None:
        """Capture the can pose in the flange frame after verified contact."""

        if bool(self.grasp_constraint_active[env_id]):
            return
        flange_pos_w, flange_quat_w = self._current_flange_w()
        can_pos_f, can_quat_f = math_utils.subtract_frame_transforms(
            flange_pos_w[env_id : env_id + 1],
            flange_quat_w[env_id : env_id + 1],
            self.can.data.root_pos_w[env_id : env_id + 1],
            self.can.data.root_quat_w[env_id : env_id + 1],
        )
        self.grasp_can_pose_f[env_id, :3] = can_pos_f[0]
        self.grasp_can_pose_f[env_id, 3:] = can_quat_f[0]
        if not self.use_grasp_assist:
            return
        position = self.grasp_can_pose_f[env_id, :3].detach().cpu().tolist()
        quaternion = self.grasp_can_pose_f[env_id, 3:].detach().cpu().tolist()
        joint = self.grasp_joints[env_id]
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*position))
        joint.GetLocalRot0Attr().Set(
            Gf.Quatf(quaternion[0], Gf.Vec3f(*quaternion[1:]))
        )
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0))
        joint.GetJointEnabledAttr().Set(True)
        self.grasp_constraint_active[env_id] = True

    def _detach_grasp_constraints(self, env_ids: torch.Tensor) -> None:
        if self.use_grasp_assist:
            for env_id in env_ids.detach().cpu().tolist():
                self.grasp_joints[env_id].GetJointEnabledAttr().Set(False)
        self.grasp_constraint_active[env_ids] = False

    def _apply_grasp_assist(self) -> None:
        """The pre-authored PhysX joint advances the assisted can during physics."""

    def post_step(self) -> None:
        """Retained as the runner hook; the pre-authored joint needs no pose writes."""

        self._apply_grasp_assist()

    @property
    def finished(self) -> torch.Tensor:
        return (self.phase == int(OraclePhase.SUCCESS)) | (
            self.phase == int(OraclePhase.FAILURE)
        )

    @property
    def succeeded(self) -> torch.Tensor:
        return self.phase == int(OraclePhase.SUCCESS)

    def phase_names(self) -> list[str]:
        return [OraclePhase(int(value)).name.lower() for value in self.phase]


    def grasp_quality_snapshot(self, env_id: int) -> dict | None:
        """Return the latest measured physical-grasp state for evidence files."""

        if self.grasp_monitor.last is None:
            return None
        return self.grasp_monitor.last.env_dict(env_id)
    def _current_flange_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.robot.data.body_pos_w[:, self.flange_body_index],
            self.robot.data.body_quat_w[:, self.flange_body_index],
        )

    def current_tcp_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        flange_pos_w, flange_quat_w = self._current_flange_w()
        tcp_pos_w = flange_pos_w + math_utils.quat_apply(
            flange_quat_w, self.tcp_offset_f
        )
        return tcp_pos_w, flange_quat_w

    def _tcp_world_to_flange_root(
        self, tcp_pos_w: torch.Tensor, tcp_quat_w: torch.Tensor
    ) -> torch.Tensor:
        flange_pos_w = tcp_pos_w - math_utils.quat_apply(
            tcp_quat_w, self.tcp_offset_f
        )
        flange_pos_b, flange_quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            flange_pos_w,
            tcp_quat_w,
        )
        return torch.cat((flange_pos_b, flange_quat_b), dim=-1)

    def _pose_reached(
        self, target_tcp_w: torch.Tensor, target_quat_w: torch.Tensor
    ) -> torch.Tensor:
        tcp_pos_w, tcp_quat_w = self.current_tcp_w()
        position_error = torch.linalg.vector_norm(tcp_pos_w - target_tcp_w, dim=1)
        orientation_error = math_utils.quat_error_magnitude(
            tcp_quat_w, target_quat_w
        )
        position_tolerance = torch.full_like(position_error, EEF_POSITION_TOLERANCE)
        pregrasp = self.phase == int(OraclePhase.PREGRASP)
        position_tolerance[pregrasp] = self.scenario.pregrasp_position_tolerance
        raw = (position_error <= position_tolerance) & (
            orientation_error <= EEF_ORIENTATION_TOLERANCE_RAD
        )
        self.pose_stable_steps = torch.where(
            raw, self.pose_stable_steps + 1, torch.zeros_like(self.pose_stable_steps)
        )
        return self.pose_stable_steps >= 5

    def _set_phase(self, env_id: int, phase: OraclePhase) -> None:
        self.phase[env_id] = int(phase)
        self.phase_steps[env_id] = 0
        self.pose_stable_steps[env_id] = 0

    def _swept_motion_clear(self) -> tuple[torch.Tensor, list[str]]:
        """Continuously check carried-can and sampled arm-link motion."""

        clear = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        reasons = [""] * self.num_envs
        current_can = self.can.data.root_pos_w - self.env.scene.env_origins
        current_arm = (
            self.robot.data.body_pos_w[:, self.swept_arm_body_indices]
            - self.env.scene.env_origins[:, None, :]
        )
        checked_phases = {
            OraclePhase.TRAVERSE,
            OraclePhase.CARRY_REPOSITION,
            OraclePhase.CARRY_REORIENT,
            OraclePhase.PREPLACE,
        }
        for env_id in range(self.num_envs):
            phase = OraclePhase(int(self.phase[env_id].item()))
            if phase not in checked_phases:
                continue
            can_start = tuple(self.previous_can_position[env_id].detach().cpu().tolist())
            can_end = tuple(current_can[env_id].detach().cpu().tolist())
            if not swept_proxy_is_clear(
                can_start,
                can_end,
                (CAN_RADIUS, CAN_RADIUS, CAN_HALF_HEIGHT),
            ):
                clear[env_id] = False
                reasons[env_id] = "swept carried-can proxy intersected a fixture"
                continue
            for body_offset, body_index in enumerate(self.swept_arm_body_indices):
                start = tuple(
                    self.previous_arm_positions[env_id, body_offset]
                    .detach()
                    .cpu()
                    .tolist()
                )
                end = tuple(current_arm[env_id, body_offset].detach().cpu().tolist())
                if not swept_proxy_is_clear(start, end, (0.035, 0.035, 0.035)):
                    clear[env_id] = False
                    reasons[env_id] = (
                        "swept arm proxy intersected a fixture: "
                        f"{self.robot.data.body_names[body_index]}"
                    )
                    break
        self.previous_can_position[:] = current_can
        self.previous_arm_positions[:] = current_arm
        return clear, reasons

    def _fail(self, env_id: int, reason: str) -> None:
        current_tcp_w, _ = self.current_tcp_w()
        target_tcp_w, _ = self._target_tcp_w()
        origin = self.env.scene.env_origins[env_id]
        can_position = self.can.data.root_pos_w[env_id] - origin
        self.failure_diagnostics[env_id] = {
            "phase": OraclePhase(int(self.phase[env_id].item())).name.lower(),
            "phase_steps": int(self.phase_steps[env_id].item()),
            "can_xyz": can_position.detach().cpu().tolist(),
            "can_linear_speed": float(
                torch.linalg.vector_norm(
                    self.can.data.root_lin_vel_w[env_id]
                ).item()
            ),
            "can_angular_speed": float(
                torch.linalg.vector_norm(
                    self.can.data.root_ang_vel_w[env_id]
                ).item()
            ),
            "tcp_xyz": (current_tcp_w[env_id] - origin).detach().cpu().tolist(),
            "target_tcp_xyz": (target_tcp_w[env_id] - origin)
            .detach()
            .cpu()
            .tolist(),
            "tcp_position_error": float(
                torch.linalg.vector_norm(
                    target_tcp_w[env_id] - current_tcp_w[env_id]
                ).item()
            ),
            "placement_error_xy": float(
                torch.linalg.vector_norm(
                    can_position[:2] - self.place_xy[env_id]
                ).item()
            ),
            "handoff_settle_steps": int(
                self.handoff_settle_steps[env_id].item()
            ),
            "grasp_constraint_active": bool(
                self.grasp_constraint_active[env_id].item()
            ),
            "physical_grasp_quality": self.grasp_quality_snapshot(env_id),
            "secure_grasp_quality": self.secure_grasp_quality[env_id],
        }
        self.failure_reason[env_id] = reason
        self._detach_grasp_constraints(
            torch.tensor([env_id], dtype=torch.long, device=self.device)
        )
        self._set_phase(env_id, OraclePhase.FAILURE)

    def _capture_pick_targets(self, mask: torch.Tensor) -> None:
        can_pos_w = self.can.data.root_pos_w
        self.pick_tcp_w[mask] = can_pos_w[mask]

    def _target_tcp_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        tcp_pos_w, tcp_quat_w = self.current_tcp_w()
        target = tcp_pos_w.clone()
        q = tcp_quat_w.clone()
        pick_motion = (
            (self.phase >= int(OraclePhase.PREGRASP))
            & (self.phase <= int(OraclePhase.CARRY_SETTLE))
        )
        q[pick_motion] = self.approach_quat_w[pick_motion]
        # TRAVERSE may rotate the entire base. During CARRY_REPOSITION preserve
        # the resulting physical gripper orientation while moving the load to
        # the high, central reorientation pose. Forcing the old world-frame
        # pick quaternion here makes the wrist sweep the can through the
        # counter after a pi-radian base turn.
        place_motion = (
            (self.phase >= int(OraclePhase.PREPLACE))
            & (self.phase <= int(OraclePhase.RETREAT))
        )
        q[place_motion] = self.place_side_grasp_quat_w[place_motion]
        reorient = self.phase == int(OraclePhase.CARRY_REORIENT)
        q[reorient] = self.place_side_grasp_quat_w[reorient]
        origins = self.env.scene.env_origins
        desired_can_w = torch.zeros_like(target)
        desired_can_w[:, :2] = self.place_xy + origins[:, :2]
        desired_can_w[:, 2] = (
            COUNTER_SURFACE_Z + CAN_HALF_HEIGHT + origins[:, 2]
        )
        # ``place_xy`` specifies the desired can center, not the task TCP.
        # Convert through the grasp captured at SECURE so edge samples do not
        # inherit the 14--15 mm side-grasp offset seen at the fingertips.
        place_center_w = desired_can_w + math_utils.quat_apply(
            self.place_side_grasp_quat_w,
            self.tcp_offset_f - self.grasp_can_pose_f[:, :3],
        )
        carry_reposition_w = self.robot.data.root_pos_w.clone()
        carry_reposition_w[:, 2] = 1.25 + origins[:, 2]

        pregrasp = self.pick_tcp_w.clone()
        pregrasp[:, :2] += self.pick_pregrasp_offset_w
        lift = self.lift_tcp_w.clone()
        preplace = place_center_w.clone()
        preplace[:, 2] += PLACE_APPROACH_CLEARANCE

        target[self.phase == int(OraclePhase.PREGRASP)] = pregrasp[
            self.phase == int(OraclePhase.PREGRASP)
        ]
        target[self.phase == int(OraclePhase.DESCEND)] = self.pick_tcp_w[
            self.phase == int(OraclePhase.DESCEND)
        ]
        target[self.phase == int(OraclePhase.CLOSE)] = self.pick_tcp_w[
            self.phase == int(OraclePhase.CLOSE)
        ]
        target[self.phase == int(OraclePhase.SECURE)] = self.secure_tcp_w[
            self.phase == int(OraclePhase.SECURE)
        ]
        target[self.phase == int(OraclePhase.LIFT)] = lift[
            self.phase == int(OraclePhase.LIFT)
        ]
        target[self.phase == int(OraclePhase.CARRY_REPOSITION)] = carry_reposition_w[
            self.phase == int(OraclePhase.CARRY_REPOSITION)
        ]
        target[self.phase == int(OraclePhase.CARRY_REORIENT)] = carry_reposition_w[
            self.phase == int(OraclePhase.CARRY_REORIENT)
        ]
        target[self.phase == int(OraclePhase.PREPLACE)] = preplace[
            self.phase == int(OraclePhase.PREPLACE)
        ]
        target[self.phase == int(OraclePhase.PLACE)] = place_center_w[
            self.phase == int(OraclePhase.PLACE)
        ]
        target[self.phase == int(OraclePhase.HANDOFF_SETTLE)] = place_center_w[
            self.phase == int(OraclePhase.HANDOFF_SETTLE)
        ]
        target[self.phase == int(OraclePhase.OPEN)] = place_center_w[
            self.phase == int(OraclePhase.OPEN)
        ]
        target[self.phase == int(OraclePhase.RETREAT)] = place_center_w[
            self.phase == int(OraclePhase.RETREAT)
        ]
        return target, q

    def _route_complete(
        self,
        route: tuple[tuple[float, float, float], ...],
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return which environments have settled at the final route waypoint."""

        if not route:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        root_pos = self.robot.data.root_link_pos_w - self.env.scene.env_origins
        root_quat = self.robot.data.root_link_quat_w
        w, x, y, z = root_quat.unbind(dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        final = torch.tensor(
            route[-1], dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        distance = torch.linalg.vector_norm(root_pos[:, :2] - final[:, :2], dim=1)
        yaw_error = torch.remainder(yaw - final[:, 2] + math.pi, 2.0 * math.pi) - math.pi
        planar_speed = torch.linalg.vector_norm(
            self.robot.data.root_link_lin_vel_w[:, :2], dim=1
        )
        angular_speed = torch.abs(self.robot.data.root_link_ang_vel_w[:, 2])
        return (
            (indices >= len(route) - 1)
            & (distance <= self.scenario.pick_base_position_tolerance)
            & (torch.abs(yaw_error) <= self.scenario.pick_base_yaw_tolerance)
            & (planar_speed < 0.02)
            & (angular_speed < 0.03)
        )

    def _base_command(self) -> torch.Tensor:
        """Follow the scenario route with bounded body-frame holonomic commands."""

        command = torch.zeros((self.num_envs, 3), device=self.device)
        root_pos = self.robot.data.root_link_pos_w - self.env.scene.env_origins
        root_quat = self.robot.data.root_link_quat_w
        w, x, y, z = root_quat.unbind(dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        for env_id in range(self.num_envs):
            phase = OraclePhase(int(self.phase[env_id].item()))
            if phase == OraclePhase.NAVIGATE_TO_PICK:
                route = self.scenario.pre_pick_waypoints
                indices = self.pre_pick_waypoint_index
                max_speed = self.scenario.pre_pick_base_speed
            elif phase == OraclePhase.TRAVERSE:
                route = self.scenario.carry_waypoints
                indices = self.carry_waypoint_index
                max_speed = self.scenario.carry_base_speed
            elif phase == OraclePhase.RETREAT:
                route = ((*STAGING_POSES["retreat"][:2], STAGING_POSES["retreat"][3]),)
                indices = None
                max_speed = CARRY_BASE_SPEED
            else:
                continue
            if not route:
                continue

            route_index = 0 if indices is None else int(indices[env_id].item())
            route_index = min(route_index, len(route) - 1)
            goal = torch.tensor(
                route[route_index], dtype=torch.float32, device=self.device
            )
            goal_position_tolerance = BASE_POSITION_TOLERANCE
            if phase == OraclePhase.NAVIGATE_TO_PICK and route_index == len(route) - 1:
                goal_position_tolerance = self.scenario.pick_base_position_tolerance
            delta_w = goal[:2] - root_pos[env_id, :2]
            distance = torch.linalg.vector_norm(delta_w)
            yaw_error = torch.remainder(
                goal[2] - yaw[env_id] + math.pi, 2.0 * math.pi
            ) - math.pi

            if (
                indices is not None
                and route_index < len(route) - 1
                and float(distance.item()) <= BASE_POSITION_TOLERANCE
                and abs(float(yaw_error.item())) <= BASE_YAW_TOLERANCE_RAD
            ):
                indices[env_id] += 1
                route_index += 1
                goal = torch.tensor(
                    route[route_index], dtype=torch.float32, device=self.device
                )
                goal_position_tolerance = BASE_POSITION_TOLERANCE
                if phase == OraclePhase.NAVIGATE_TO_PICK and route_index == len(route) - 1:
                    goal_position_tolerance = self.scenario.pick_base_position_tolerance
                delta_w = goal[:2] - root_pos[env_id, :2]
                distance = torch.linalg.vector_norm(delta_w)
                yaw_error = torch.remainder(
                    goal[2] - yaw[env_id] + math.pi, 2.0 * math.pi
                ) - math.pi

            c = torch.cos(yaw[env_id])
            s = torch.sin(yaw[env_id])
            delta_b = torch.stack(
                (
                    c * delta_w[0] + s * delta_w[1],
                    -s * delta_w[0] + c * delta_w[1],
                )
            )
            if float(distance.item()) > goal_position_tolerance * 0.7:
                speed = min(max_speed, 0.5 * float(distance.item()))
                command[env_id, :2] = (
                    delta_b / torch.clamp(distance, min=1.0e-6) * speed
                )
            command[env_id, 2] = torch.clamp(yaw_error, min=-0.35, max=0.35)
        return command

    def _base_at_retreat_stage(self) -> torch.Tensor:
        root_pos = self.robot.data.root_link_pos_w - self.env.scene.env_origins
        goal_xy = torch.tensor(
            STAGING_POSES["retreat"][:2], dtype=torch.float32, device=self.device
        )
        distance = torch.linalg.vector_norm(root_pos[:, :2] - goal_xy, dim=1)
        planar_speed = torch.linalg.vector_norm(
            self.robot.data.root_link_lin_vel_w[:, :2], dim=1
        )
        return (distance <= BASE_POSITION_TOLERANCE) & (planar_speed < 0.02)

    def _advance(self, target_tcp_w: torch.Tensor, target_quat_w: torch.Tensor) -> None:
        pose_reached = self._pose_reached(target_tcp_w, target_quat_w)
        supported_pick = can_supported_by_island(
            self.env,
            region=self.scenario.pick_region,
            support_name=self.scenario.pick_support_name,
        )
        # Latch the six-joint mechanism at first credible bilateral contact;
        # SECURE performs the consecutive-frame confirmation after preload.
        bilateral = can_bilaterally_grasped(self.env, stable_frames=1)
        grasp_quality = self.grasp_monitor.update(
            torch.abs(
                self.grasp_joint_positions[:, self.gripper_mimic_drive_indices]
            ) * self.gripper_mimic_signs
        )
        secure_physical = grasp_quality.secure
        transport_contact = (
            grasp_quality.bilateral_contact
            & (grasp_quality.left_normal_force >= GRASP_CONTACT_FORCE_N)
            & (grasp_quality.right_normal_force >= GRASP_CONTACT_FORCE_N)
        )
        lifted = can_lifted_clear(
            self.env, support_name=self.scenario.pick_support_name
        )
        if self.use_grasp_assist:
            can_position = self.can.data.root_pos_w - self.env.scene.env_origins
            assisted_clear = self.grasp_constraint_active & (
                can_position[:, 2] - CAN_HALF_HEIGHT
                >= ISLAND_SURFACE_Z + MIN_LIFT_CLEARANCE
            )
            lifted |= assisted_clear
        pick_navigation_complete = self._route_complete(
            self.scenario.pre_pick_waypoints,
            self.pre_pick_waypoint_index,
        )
        at_place_base = base_at_place_stage(self.env)
        at_retreat_base = self._base_at_retreat_stage()
        supported_place = can_supported_in_place_region(self.env)
        released = gripper_released(self.env)
        collision_free = robot_collision_free(self.env)
        swept_clear, swept_reasons = self._swept_motion_clear()
        can_linear_speed = torch.linalg.vector_norm(
            self.can.data.root_lin_vel_w, dim=1
        )
        can_angular_speed = torch.linalg.vector_norm(
            self.can.data.root_ang_vel_w, dim=1
        )
        can_position = self.can.data.root_pos_w - self.env.scene.env_origins
        target_place_error = torch.linalg.vector_norm(
            can_position[:, :2] - self.place_xy, dim=1
        )
        release_ready = (
            (can_position[:, 0] >= PLACE_REGION.xmin)
            & (can_position[:, 0] <= PLACE_REGION.xmax)
            & (can_position[:, 1] >= PLACE_REGION.ymin)
            & (can_position[:, 1] <= PLACE_REGION.ymax)
            & (
                can_position[:, 2]
                <= COUNTER_SURFACE_Z + CAN_HALF_HEIGHT + PLACE_RELEASE_CLEARANCE
            )
            & (target_place_error <= PLACE_TARGET_TOLERANCE)
        )
        current_tcp_w, _ = self.current_tcp_w()
        flange_pos_b, flange_quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            *self._current_flange_w(),
        )
        current_flange_b = torch.cat((flange_pos_b, flange_quat_b), dim=-1)
        for env_id in range(self.num_envs):
            phase = OraclePhase(int(self.phase[env_id].item()))
            if phase in (OraclePhase.SUCCESS, OraclePhase.FAILURE):
                continue
            if not bool(collision_free[env_id].item()):
                self._fail(env_id, "robot or can left the finite collision-free workspace")
                continue
            if not bool(swept_clear[env_id].item()):
                self._fail(env_id, swept_reasons[env_id])
                continue
            timeout = _PHASE_TIMEOUT_STEPS[phase]
            if int(self.phase_steps[env_id].item()) > timeout:
                self._fail(env_id, f"timeout in {phase.name.lower()}")
                continue
            loaded_unassisted = (
                not self.use_grasp_assist
                and OraclePhase.LIFT <= phase <= OraclePhase.PLACE
            )
            if loaded_unassisted:
                self.grasp_lost_steps[env_id] = (
                    0
                    if bool(transport_contact[env_id].item())
                    else self.grasp_lost_steps[env_id] + 1
                )
                if int(self.grasp_lost_steps[env_id].item()) >= GRASP_LOST_CONTACT_FRAMES:
                    self._fail(
                        env_id,
                        "unassisted grasp lost bilateral opposing contact",
                    )
                    continue
            if phase == OraclePhase.SETTLE and bool(supported_pick[env_id].item()):
                self._capture_pick_targets(self.phase == int(OraclePhase.SETTLE))
                self.approach_quat_w[env_id] = self.side_grasp_quat_w[env_id]
                if self.scenario.pre_pick_waypoints:
                    self.navigation_flange_b[env_id] = current_flange_b[env_id]
                    self.pre_pick_waypoint_index[env_id] = 0
                    self._set_phase(env_id, OraclePhase.NAVIGATE_TO_PICK)
                else:
                    self._set_phase(env_id, OraclePhase.PREGRASP)
            elif (
                phase == OraclePhase.NAVIGATE_TO_PICK
                and bool(pick_navigation_complete[env_id].item())
            ):
                self._capture_pick_targets(
                    self.phase == int(OraclePhase.NAVIGATE_TO_PICK)
                )
                self._set_phase(env_id, OraclePhase.PREGRASP)
            elif phase == OraclePhase.PREGRASP and bool(pose_reached[env_id].item()):
                self._set_phase(env_id, OraclePhase.DESCEND)
            elif phase == OraclePhase.DESCEND and bool(pose_reached[env_id].item()):
                self.grasp_joint_positions[env_id] = self.robot.data.joint_pos[
                    env_id, self.gripper_drive_joint_indices
                ]
                self._set_phase(env_id, OraclePhase.CLOSE)
            elif phase == OraclePhase.CLOSE and bool(bilateral[env_id].item()):
                actual = self.robot.data.joint_pos[
                    env_id, self.gripper_drive_joint_indices
                ]
                self.grasp_contact_joint_positions[env_id] = actual
                self.grasp_joint_positions[env_id] = actual
                self.grasp_monitor.reset(
                    torch.tensor([env_id], dtype=torch.long, device=self.device)
                )
                self.secure_tcp_w[env_id] = current_tcp_w[env_id]
                self._set_phase(env_id, OraclePhase.SECURE)
            elif phase == OraclePhase.SECURE and (
                bool(pose_reached[env_id].item())
                and bool(secure_physical[env_id].item())
            ):
                self.lift_tcp_w[env_id] = current_tcp_w[env_id]
                self.lift_command_tcp_w[env_id] = current_tcp_w[env_id]
                self.lift_tcp_w[env_id, 2] += LIFT_CLEARANCE
                self.secure_grasp_quality[env_id] = grasp_quality.env_dict(env_id)
                if self.use_grasp_assist:
                    accepted_joint_positions = self.robot.data.joint_pos[
                        env_id, self.gripper_drive_joint_indices
                    ]
                    self.grasp_joint_positions[env_id] = accepted_joint_positions
                    self.grasp_contact_joint_positions[env_id] = (
                        accepted_joint_positions
                    )
                self._attach_grasp_constraint(env_id)
                self._set_phase(env_id, OraclePhase.LIFT)
            elif (
                phase == OraclePhase.LIFT
                and bool(lifted[env_id].item())
                and float(
                    torch.linalg.vector_norm(
                        self.lift_tcp_w[env_id] - self.lift_command_tcp_w[env_id]
                    ).item()
                ) <= 0.002
            ):
                # In assisted mode the object-clearance predicate is the
                # physical transition gate.  Continue commanding the authored
                # lift pose rather than freezing a contact-deflected sample.
                self.carry_flange_b[env_id] = self._tcp_world_to_flange_root(
                    self.lift_tcp_w, self.approach_quat_w
                )[env_id]
                self.carry_settle_steps[env_id] = 0
                self._set_phase(env_id, OraclePhase.CARRY_SETTLE)
            elif phase == OraclePhase.CARRY_SETTLE:
                if not bool(lifted[env_id].item()):
                    self.carry_settle_steps[env_id] = 0
                else:
                    if self.use_grasp_assist:
                        carry_stable = (
                            float(can_linear_speed[env_id].item())
                            <= CARRY_LINEAR_SETTLE_SPEED
                            and float(can_angular_speed[env_id].item())
                            <= CARRY_ANGULAR_SETTLE_SPEED
                        )
                    else:
                        # During a hard physical pinch, PhysX can expose
                        # pre-constraint rigid-body velocities even when the
                        # post-solver can-to-gripper transform is stationary.
                        # For an unassisted grasp, sustained bilateral force,
                        # low tangential slip, and bounded pose drift are the
                        # physically relevant loaded-carry gate.
                        carry_stable = bool(
                            grasp_quality.instantaneous_valid[env_id].item()
                        )
                    if carry_stable:
                        self.carry_settle_steps[env_id] += 1
                        if (
                            int(self.carry_settle_steps[env_id].item())
                            >= CARRY_SETTLE_FRAMES
                        ):
                            self.carry_flange_b[env_id] = current_flange_b[env_id]
                            self._set_phase(env_id, OraclePhase.TRAVERSE)
                    else:
                        self.carry_settle_steps[env_id] = 0
            elif phase == OraclePhase.TRAVERSE and bool(at_place_base[env_id].item()):
                self.motion_command_tcp_w[env_id] = current_tcp_w[env_id]
                self._set_phase(env_id, OraclePhase.CARRY_REPOSITION)
            elif phase == OraclePhase.CARRY_REPOSITION and bool(pose_reached[env_id].item()):
                self._set_phase(env_id, OraclePhase.CARRY_REORIENT)
            elif phase == OraclePhase.CARRY_REORIENT and bool(pose_reached[env_id].item()):
                self.motion_command_tcp_w[env_id] = current_tcp_w[env_id]
                self._set_phase(env_id, OraclePhase.PREPLACE)
            elif phase == OraclePhase.PREPLACE and bool(pose_reached[env_id].item()):
                self.motion_command_tcp_w[env_id] = current_tcp_w[env_id]
                self._set_phase(env_id, OraclePhase.PLACE)
            elif (
                phase == OraclePhase.PLACE
                and bool(release_ready[env_id].item())
            ):
                self.carry_flange_b[env_id] = current_flange_b[env_id]
                self.handoff_settle_steps[env_id] = 0
                ids = torch.tensor([env_id], dtype=torch.long, device=self.device)
                # The release pose is already collision-checked and inside the
                # target tolerance.  Detach here so the dynamic can and static
                # counter form an ordinary PhysX contact pair while settling;
                # keeping the can fixed to a driven flange at contact would
                # overconstrain the solver and can inject large velocities.
                self.can.write_root_velocity_to_sim(
                    torch.zeros((1, 6), device=self.device), env_ids=ids
                )
                self._detach_grasp_constraints(ids)
                if self.use_grasp_assist:
                    open_position = torch.zeros(
                        (1, len(self.gripper_joint_indices)), device=self.device
                    )
                    self.robot.write_joint_state_to_sim(
                        open_position,
                        torch.zeros_like(open_position),
                        joint_ids=self.gripper_joint_indices,
                        env_ids=ids,
                    )
                    self.open_joint_positions[env_id] = 0.0
                else:
                    self.open_joint_positions[env_id] = self.robot.data.joint_pos[
                        env_id, self.gripper_drive_joint_indices
                    ]
                self._set_phase(env_id, OraclePhase.HANDOFF_SETTLE)
            elif phase == OraclePhase.HANDOFF_SETTLE:
                settled = (
                    float(can_linear_speed[env_id].item())
                    <= HANDOFF_LINEAR_SETTLE_SPEED
                    and float(can_angular_speed[env_id].item())
                    <= HANDOFF_ANGULAR_SETTLE_SPEED
                    and bool(supported_place[env_id].item())
                    and bool(released[env_id].item())
                    and float(target_place_error[env_id].item())
                    <= PLACE_TARGET_TOLERANCE
                )
                if not settled:
                    self.handoff_settle_steps[env_id] = 0
                    continue
                self.handoff_settle_steps[env_id] += 1
                if (
                    int(self.handoff_settle_steps[env_id].item())
                    < HANDOFF_SETTLE_FRAMES
                ):
                    continue
                self._set_phase(env_id, OraclePhase.OPEN)
            elif (
                phase == OraclePhase.OPEN
                and bool(supported_place[env_id].item())
                and bool(released[env_id].item())
                and float(target_place_error[env_id].item())
                <= PLACE_TARGET_TOLERANCE
            ):
                self._set_phase(env_id, OraclePhase.RETREAT)
            elif (
                phase == OraclePhase.RETREAT
                and bool(at_retreat_base[env_id].item())
                and bool(supported_place[env_id].item())
                and bool(released[env_id].item())
                and float(target_place_error[env_id].item())
                <= PLACE_TARGET_TOLERANCE
            ):
                self._set_phase(env_id, OraclePhase.SUCCESS)

    def compute_action(self) -> torch.Tensor:
        """Evaluate transitions and return the next complete environment action."""

        target_tcp_w, target_quat_w = self._target_tcp_w()
        self._advance(target_tcp_w, target_quat_w)
        # Recompute after transitions so each new phase receives its target
        # immediately rather than one environment step later.
        target_tcp_w, target_quat_w = self._target_tcp_w()
        command_tcp_w = target_tcp_w.clone()
        # A large absolute IK jump can generate enough transient acceleration
        # to throw a friction grasp. Rate-limit only the loaded vertical lift;
        # reach tests and free-space motions retain the normal IK interface.
        lift = self.phase == int(OraclePhase.LIFT)
        if torch.any(lift):
            delta = target_tcp_w - self.lift_command_tcp_w
            distance = torch.linalg.vector_norm(delta, dim=1)
            lift_speed = 0.05 if self.use_grasp_assist else LIFT_TCP_SPEED
            max_step = lift_speed * self.control_dt
            scale = torch.clamp(max_step / torch.clamp(distance, min=1.0e-6), max=1.0)
            self.lift_command_tcp_w[lift] = (
                self.lift_command_tcp_w[lift]
                + delta[lift] * scale[lift].unsqueeze(-1)
            )
            command_tcp_w[lift] = self.lift_command_tcp_w[lift]
        reposition = (
            (self.phase == int(OraclePhase.CARRY_REPOSITION))
            | (self.phase == int(OraclePhase.PREPLACE))
        )
        if torch.any(reposition):
            delta = target_tcp_w - self.motion_command_tcp_w
            distance = torch.linalg.vector_norm(delta, dim=1)
            max_step = CARRY_REPOSITION_TCP_SPEED * self.control_dt
            scale = torch.clamp(max_step / torch.clamp(distance, min=1.0e-6), max=1.0)
            self.motion_command_tcp_w[reposition] = (
                self.motion_command_tcp_w[reposition]
                + delta[reposition] * scale[reposition].unsqueeze(-1)
            )
            command_tcp_w[reposition] = self.motion_command_tcp_w[reposition]
        place = self.phase == int(OraclePhase.PLACE)
        if torch.any(place):
            delta = target_tcp_w - self.motion_command_tcp_w
            distance = torch.linalg.vector_norm(delta, dim=1)
            max_step = PLACE_TCP_SPEED * self.control_dt
            scale = torch.clamp(max_step / torch.clamp(distance, min=1.0e-6), max=1.0)
            self.motion_command_tcp_w[place] = (
                self.motion_command_tcp_w[place]
                + delta[place] * scale[place].unsqueeze(-1)
            )
            command_tcp_w[place] = self.motion_command_tcp_w[place]
        flange_b = self._tcp_world_to_flange_root(command_tcp_w, target_quat_w)
        navigate = self.phase == int(OraclePhase.NAVIGATE_TO_PICK)
        flange_b[navigate] = self.navigation_flange_b[navigate]
        carry_hold = (
            (self.phase == int(OraclePhase.CARRY_SETTLE))
            | (self.phase == int(OraclePhase.TRAVERSE))
            | (self.phase == int(OraclePhase.HANDOFF_SETTLE))
            | (self.phase == int(OraclePhase.OPEN))
            | (self.phase == int(OraclePhase.RETREAT))
        )
        flange_b[carry_hold] = self.carry_flange_b[carry_hold]

        action = torch.zeros(
            (self.num_envs, self.env.action_manager.total_action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        action[:, self.action_slices["base"]] = self._base_command()
        action[:, self.action_slices["arm"]] = flange_b
        gripper_command = torch.zeros_like(self.gripper_close_command)
        search = self.phase == int(OraclePhase.CLOSE)
        if torch.any(search):
            search_magnitude = (
                self.grasp_joint_positions[search] * self.gripper_signs[search]
            )
            close_limit = torch.abs(self.gripper_close_command[search])
            search_magnitude = torch.minimum(
                search_magnitude
                + GRIPPER_CLOSE_SEARCH_SPEED_RAD_S * self.control_dt,
                close_limit,
            )
            self.grasp_joint_positions[search] = (
                search_magnitude * self.gripper_signs[search]
            )
            gripper_command[search] = self.grasp_joint_positions[search]

        hold = (
            (self.phase >= int(OraclePhase.SECURE))
            & (self.phase <= int(OraclePhase.PLACE))
        )
        if torch.any(hold):
            force_hold = hold & ~self.grasp_constraint_active
            quality = self.grasp_monitor.last
            if quality is not None and torch.any(force_hold):
                normal_force = torch.minimum(
                    quality.left_normal_force, quality.right_normal_force
                )
                force_error = GRIPPER_TARGET_NORMAL_FORCE_N - normal_force
                force_error = torch.where(
                    torch.abs(force_error) <= GRIPPER_FORCE_DEADBAND_N,
                    torch.zeros_like(force_error),
                    force_error,
                )
                preload_step = torch.clamp(
                    GRIPPER_FORCE_CONTROL_GAIN_RAD_PER_N_S
                    * force_error
                    * self.control_dt,
                    min=-GRIPPER_FORCE_CONTROL_SPEED_RAD_S * self.control_dt,
                    max=GRIPPER_FORCE_CONTROL_SPEED_RAD_S * self.control_dt,
                )
                magnitude = self.grasp_joint_positions * self.gripper_signs
                contact_magnitude = (
                    self.grasp_contact_joint_positions * self.gripper_signs
                )
                # The reference may move past the physical joint limit: its
                # bounded spring deflection is how an implicit position drive
                # applies contact force. Actual joints remain constrained by
                # their authored limits and the actuator by its effort limit.
                upper_magnitude = (
                    contact_magnitude + GRIPPER_MAX_PRELOAD_RAD
                )
                magnitude[force_hold] = torch.minimum(
                    torch.maximum(
                        magnitude[force_hold] + preload_step[force_hold, None],
                        contact_magnitude[force_hold],
                    ),
                    upper_magnitude[force_hold],
                )
                self.grasp_joint_positions[force_hold] = (
                    magnitude[force_hold] * self.gripper_signs[force_hold]
                )
            gripper_command[hold] = self.grasp_joint_positions[hold]
        opening = (self.phase == int(OraclePhase.HANDOFF_SETTLE)) | (
            self.phase == int(OraclePhase.OPEN)
        )
        if torch.any(opening):
            max_step = GRIPPER_OPEN_SPEED_RAD_S * self.control_dt
            self.open_joint_positions[opening] += torch.clamp(
                -self.open_joint_positions[opening], min=-max_step, max=max_step
            )
            gripper_command[opening] = self.open_joint_positions[opening]
        action[:, self.action_slices["gripper"]] = gripper_command
        self.phase_steps[~self.finished] += 1
        return action
