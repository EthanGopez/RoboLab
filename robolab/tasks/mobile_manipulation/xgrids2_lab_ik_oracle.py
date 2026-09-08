"""Closed-loop task-space IK gates for XGRIDS2 manipulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import torch
from isaaclab.utils import math as math_utils
from pxr import Gf, UsdPhysics

from robolab.robots.kinova_gen3 import GRIPPER_JOINT_COMMANDS
from robolab.robots.kinova_mobile_cobot import (
    MOBILE_KINOVA_FLANGE_BODY,
    MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS,
)
from robolab.tasks.mobile_manipulation.kitchen_can_transfer_task import (
    ORACLE_GRASP_JOINT_NAME,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
    CAN_HALF_HEIGHT,
    GRIPPER_CLOSE_SEARCH_SPEED_RAD_S,
    GRASP_TCP_OFFSET_FLANGE,
)
from robolab.tasks.mobile_manipulation.kitchen_manipulation_predicates import (
    can_bilaterally_grasped,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_manipulation_geometry import (
    DINING_TABLE,
)
PICK_GRASP_QUAT_WXYZ = (-0.2520023, -0.2520023, 0.6606776, 0.6606776)
PICK_APPROACH_DIRECTION = (-0.6658, 0.7461)


REACH_POSITION_TOLERANCE_M = 0.012
REACH_ORIENTATION_TOLERANCE_RAD = 0.08
MAX_REACH_ARM_STEP_RAD = 0.08
MAX_REACH_BASE_DRIFT_M = 0.015


@dataclass
class XGRIDS2IKReachResult:
    passed: bool = False
    failure_reason: str = ""
    phase_history: list[str] = field(default_factory=list)
    can_position: list[float] = field(default_factory=list)
    closest_tcp_position: list[float] = field(default_factory=list)
    closest_target_position: list[float] = field(default_factory=list)
    closest_position_error_m: float = math.inf
    maximum_arm_step_rad: float = 0.0
    maximum_arm_step_phase: str = ""
    maximum_arm_speed_rad_s: float = 0.0
    maximum_base_drift_m: float = 0.0
    bilateral_contact_verified: bool = False
    grasp_constraint_active: bool = False
    gripper_contact_joint_positions: list[float] = field(default_factory=list)
    initial_can_z_m: float = math.nan
    lifted_can_z_m: float = math.nan
    lift_height_m: float = 0.0
    final_can_position: list[float] = field(default_factory=list)
    execution_mode: str = "rate_limited_absolute_pose_ik"
    grasp_coupling: str = "none"
    planner: str = "none"

    def as_dict(self) -> dict:
        return asdict(self)


class XGRIDS2IKReachOracle:
    """Prove reach and retract before adding grasp or planning layers."""

    def __init__(self, env) -> None:
        if env.num_envs != 1:
            raise ValueError("XGRIDS2 IK reach gate requires num_envs=1")
        self.env = env
        self.robot = env.scene["robot"]
        self.can = env.scene["can"]
        self.device = env.device
        self.control_dt = float(env.step_dt)
        self.action_slices = self._action_slices()
        if self.action_slices["arm"].stop - self.action_slices["arm"].start != 7:
            raise ValueError("XGRIDS2 IK reach gate requires seven-value pose actions")
        self.arm_indices = [
            self.robot.data.joint_names.index(f"joint_{index}")
            for index in range(1, 8)
        ]
        self.gripper_drive_indices = [
            self.robot.data.joint_names.index(name)
            for name in MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS
        ]
        self.gripper_close_command = torch.tensor(
            [[GRIPPER_JOINT_COMMANDS[name] for name in MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS]],
            device=self.device,
            dtype=torch.float32,
        )
        self.gripper_signs = torch.sign(self.gripper_close_command)
        self.flange_index = self.robot.data.body_names.index(MOBILE_KINOVA_FLANGE_BODY)
        self.tcp_offset_f = torch.tensor(
            [GRASP_TCP_OFFSET_FLANGE], device=self.device, dtype=torch.float32
        )
        self.phase = "idle"
        self.maximum_arm_step_rad = 0.0
        self.maximum_arm_step_phase = ""
        self.maximum_arm_speed_rad_s = 0.0
        self.maximum_base_drift_m = 0.0
        self._initial_root_xy = self.robot.data.root_link_pos_w[0, :2].clone()
        self._synchronize_base_controller()

    def _action_slices(self) -> dict[str, slice]:
        offset = 0
        result = {}
        for name, size in zip(
            self.env.action_manager.active_terms,
            self.env.action_manager.action_term_dim,
        ):
            result[name] = slice(offset, offset + size)
            offset += size
        return result

    def _root_yaw(self) -> float:
        quaternion = self.robot.data.root_link_quat_w[0]
        return math.atan2(
            2.0
            * float(
                quaternion[0] * quaternion[3]
                + quaternion[1] * quaternion[2]
            ),
            1.0 - 2.0 * float(quaternion[2] ** 2 + quaternion[3] ** 2),
        )

    def _synchronize_base_controller(self) -> None:
        self.env.action_manager.get_term("base").set_planar_pose(
            self.robot.data.root_link_pos_w[:, :2],
            torch.tensor([self._root_yaw()], device=self.device),
        )

    def current_tcp_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        flange_pos = self.robot.data.body_pos_w[:, self.flange_index]
        flange_quat = self.robot.data.body_quat_w[:, self.flange_index]
        tcp_pos = flange_pos + math_utils.quat_apply(flange_quat, self.tcp_offset_f)
        return tcp_pos, flange_quat

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

    def _action(
        self,
        tcp_pos_w: torch.Tensor,
        tcp_quat_w: torch.Tensor,
        gripper_command: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action = torch.zeros(
            (1, self.env.action_manager.total_action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        action[:, self.action_slices["arm"]] = self._tcp_world_to_flange_root(
            tcp_pos_w, tcp_quat_w
        )
        if gripper_command is not None:
            action[:, self.action_slices["gripper"]] = gripper_command
        return action

    def _step(self, action: torch.Tensor) -> None:
        arm_before = self.robot.data.joint_pos[0, self.arm_indices].clone()
        _, _, terminated, truncated, _ = self.env.step(action)
        arm_after = self.robot.data.joint_pos[0, self.arm_indices]
        arm_delta = torch.remainder(
            arm_after - arm_before + math.pi, 2.0 * math.pi
        ) - math.pi
        arm_step = float(torch.max(torch.abs(arm_delta)).item())
        if arm_step > self.maximum_arm_step_rad:
            self.maximum_arm_step_rad = arm_step
            self.maximum_arm_step_phase = self.phase
        self.maximum_arm_speed_rad_s = max(
            self.maximum_arm_speed_rad_s,
            float(
                torch.max(
                    torch.abs(self.robot.data.joint_vel[0, self.arm_indices])
                ).item()
            ),
        )
        root_drift = torch.linalg.vector_norm(
            self.robot.data.root_link_pos_w[0, :2] - self._initial_root_xy
        )
        self.maximum_base_drift_m = max(
            self.maximum_base_drift_m, float(root_drift.item())
        )
        if bool(terminated[0].item() or truncated[0].item()):
            raise RuntimeError(f"environment terminated during {self.phase}")

    def _hold(
        self, steps: int, gripper_command: torch.Tensor | None = None
    ) -> None:
        tcp_pos, tcp_quat = self.current_tcp_w()
        target_pos = tcp_pos.clone()
        target_quat = tcp_quat.clone()
        for _ in range(steps):
            self._step(self._action(target_pos, target_quat, gripper_command))

    def _move_tcp(
        self,
        target_position_w: tuple[float, float, float],
        target_quaternion_wxyz: tuple[float, float, float, float],
        *,
        linear_speed_m_s: float,
        angular_speed_rad_s: float = 0.35,
        max_steps: int = 600,
        gripper_command: torch.Tensor | None = None,
    ) -> None:
        target_pos = torch.tensor(
            [target_position_w], device=self.device, dtype=torch.float32
        )
        target_quat = torch.tensor(
            [target_quaternion_wxyz], device=self.device, dtype=torch.float32
        )
        command_pos, command_quat = self.current_tcp_w()
        command_pos = command_pos.clone()
        command_quat = command_quat.clone()
        stable_steps = 0
        for _ in range(max_steps):
            delta = target_pos - command_pos
            distance = torch.linalg.vector_norm(delta, dim=1)
            max_position_step = linear_speed_m_s * self.control_dt
            position_scale = torch.clamp(
                max_position_step / torch.clamp(distance, min=1.0e-6), max=1.0
            )
            command_pos += delta * position_scale.unsqueeze(-1)

            orientation_error = math_utils.quat_error_magnitude(
                command_quat, target_quat
            )
            max_orientation_step = angular_speed_rad_s * self.control_dt
            tau = min(
                1.0,
                max_orientation_step
                / max(float(orientation_error[0].item()), 1.0e-6),
            )
            command_quat = math_utils.quat_slerp(
                command_quat[0], target_quat[0], tau=tau
            ).unsqueeze(0)
            self._step(self._action(command_pos, command_quat, gripper_command))

            measured_pos, measured_quat = self.current_tcp_w()
            measured_position_error = torch.linalg.vector_norm(
                measured_pos - target_pos, dim=1
            )
            measured_orientation_error = math_utils.quat_error_magnitude(
                measured_quat, target_quat
            )
            reached = (
                float(measured_position_error[0].item())
                <= REACH_POSITION_TOLERANCE_M
                and float(measured_orientation_error[0].item())
                <= REACH_ORIENTATION_TOLERANCE_RAD
            )
            stable_steps = stable_steps + 1 if reached else 0
            if stable_steps >= 5:
                return
        raise RuntimeError(
            f"{self.phase} did not converge to the task-space target; "
            f"position_error={float(measured_position_error[0]):.4f} m, "
            f"orientation_error={float(measured_orientation_error[0]):.4f} rad"
        )

    def _grasp_joint(self) -> UsdPhysics.FixedJoint:
        joint = UsdPhysics.FixedJoint(
            self.env.scene.stage.GetPrimAtPath(
                f"{self.env.scene.env_prim_paths[0]}/{ORACLE_GRASP_JOINT_NAME}"
            )
        )
        if not joint.GetPrim().IsValid():
            raise RuntimeError("pick gate requires the disabled post-contact grasp joint")
        return joint

    def _attach_after_contact(self, joint: UsdPhysics.FixedJoint) -> None:
        """Enable retention at the measured contact pose without moving either body."""

        flange_pos = self.robot.data.body_pos_w[:, self.flange_index]
        flange_quat = self.robot.data.body_quat_w[:, self.flange_index]
        can_pos_f, can_quat_f = math_utils.subtract_frame_transforms(
            flange_pos,
            flange_quat,
            self.can.data.root_pos_w,
            self.can.data.root_quat_w,
        )
        position = can_pos_f[0].detach().cpu().tolist()
        quaternion = can_quat_f[0].detach().cpu().tolist()
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*position))
        joint.GetLocalRot0Attr().Set(
            Gf.Quatf(quaternion[0], Gf.Vec3f(*quaternion[1:]))
        )
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0))
        joint.GetJointEnabledAttr().Set(True)

    def _close_to_bilateral_contact(self) -> torch.Tensor:
        """Rate-limit the two physical gripper drives until both pads contact."""

        tcp_pos, tcp_quat = self.current_tcp_w()
        command = self.robot.data.joint_pos[
            0:1, self.gripper_drive_indices
        ].clone()
        close_limit = torch.abs(self.gripper_close_command)
        for _ in range(180):
            magnitude = torch.abs(command)
            magnitude = torch.minimum(
                magnitude + GRIPPER_CLOSE_SEARCH_SPEED_RAD_S * self.control_dt,
                close_limit,
            )
            command = magnitude * self.gripper_signs
            self._step(self._action(tcp_pos, tcp_quat, command))
            if bool(can_bilaterally_grasped(self.env, stable_frames=4)[0].item()):
                return self.robot.data.joint_pos[
                    0:1, self.gripper_drive_indices
                ].clone()
        raise RuntimeError("bilateral fingertip contact was not established")

    def _record_quality(self, result: XGRIDS2IKReachResult) -> None:
        result.maximum_arm_step_rad = self.maximum_arm_step_rad
        result.maximum_arm_step_phase = self.maximum_arm_step_phase
        result.maximum_arm_speed_rad_s = self.maximum_arm_speed_rad_s
        result.maximum_base_drift_m = self.maximum_base_drift_m

    def run_interface_gate(self, *, local_move: bool) -> XGRIDS2IKReachResult:
        """Prove zero-motion hold, optionally followed by a 1 cm TCP move."""

        result = XGRIDS2IKReachResult()
        try:
            self.phase = "settle"
            result.phase_history.append(self.phase)
            self._hold(30)
            if local_move:
                start_pos, start_quat = self.current_tcp_w()
                target_pos = start_pos.clone()
                target_pos[:, 2] += 0.01
                self.phase = "local_up_1cm"
                result.phase_history.append(self.phase)
                self._move_tcp(
                    tuple(float(value) for value in target_pos[0]),
                    tuple(float(value) for value in start_quat[0]),
                    linear_speed_m_s=0.01,
                    angular_speed_rad_s=0.10,
                    max_steps=240,
                )
                self.phase = "local_return"
                result.phase_history.append(self.phase)
                self._move_tcp(
                    tuple(float(value) for value in start_pos[0]),
                    tuple(float(value) for value in start_quat[0]),
                    linear_speed_m_s=0.01,
                    angular_speed_rad_s=0.10,
                    max_steps=240,
                )
            else:
                self.phase = "hold"
                result.phase_history.append(self.phase)
                self._hold(120)
            self._record_quality(result)
            result.passed = (
                result.maximum_arm_step_rad <= 0.03
                and result.maximum_base_drift_m <= 0.01
            )
            if not result.passed:
                raise RuntimeError("IK controller interface gate failed")
            result.phase_history.append("success")
        except Exception as error:
            self._record_quality(result)
            result.failure_reason = str(error)
            result.phase_history.append("failure")
        return result

    def run(self) -> XGRIDS2IKReachResult:
        result = XGRIDS2IKReachResult()
        try:
            self.phase = "settle"
            result.phase_history.append(self.phase)
            self._hold(30)
            can_position = self.can.data.root_pos_w[0].clone()
            result.can_position = (
                can_position - self.env.scene.env_origins[0]
            ).detach().cpu().tolist()
            dx, dy = PICK_APPROACH_DIRECTION
            pregrasp = (
                float(can_position[0]) - 0.12 * dx,
                float(can_position[1]) - 0.12 * dy,
                float(can_position[2]) + 0.06,
            )
            closest = (
                float(can_position[0]) - 0.045 * dx,
                float(can_position[1]) - 0.045 * dy,
                float(can_position[2]),
            )

            self.phase = "pregrasp"
            result.phase_history.append(self.phase)
            self._move_tcp(
                pregrasp,
                PICK_GRASP_QUAT_WXYZ,
                linear_speed_m_s=0.04,
            )
            self.phase = "near_grasp"
            result.phase_history.append(self.phase)
            self._move_tcp(
                closest,
                PICK_GRASP_QUAT_WXYZ,
                linear_speed_m_s=0.025,
            )
            self._hold(20)
            measured_pos, _ = self.current_tcp_w()
            closest_target = torch.tensor([closest], device=self.device)
            result.closest_tcp_position = (
                measured_pos[0] - self.env.scene.env_origins[0]
            ).detach().cpu().tolist()
            result.closest_target_position = (
                closest_target[0] - self.env.scene.env_origins[0]
            ).detach().cpu().tolist()
            result.closest_position_error_m = float(
                torch.linalg.vector_norm(measured_pos - closest_target).item()
            )

            self.phase = "retract"
            result.phase_history.append(self.phase)
            self._move_tcp(
                pregrasp,
                PICK_GRASP_QUAT_WXYZ,
                linear_speed_m_s=0.035,
            )
            self._hold(30)
            result.maximum_arm_step_rad = self.maximum_arm_step_rad
            result.maximum_arm_step_phase = self.maximum_arm_step_phase
            result.maximum_arm_speed_rad_s = self.maximum_arm_speed_rad_s
            result.maximum_base_drift_m = self.maximum_base_drift_m
            result.passed = (
                result.closest_position_error_m <= REACH_POSITION_TOLERANCE_M
                and result.maximum_arm_step_rad <= MAX_REACH_ARM_STEP_RAD
                and result.maximum_base_drift_m <= MAX_REACH_BASE_DRIFT_M
            )
            if not result.passed:
                raise RuntimeError("reach/retract motion-quality gate failed")
            result.phase_history.append("success")
        except Exception as error:
            result.maximum_arm_step_rad = self.maximum_arm_step_rad
            result.maximum_arm_step_phase = self.maximum_arm_step_phase
            result.maximum_arm_speed_rad_s = self.maximum_arm_speed_rad_s
            result.maximum_base_drift_m = self.maximum_base_drift_m
            result.failure_reason = str(error)
            result.phase_history.append("failure")
        return result

    def run_pick(self) -> XGRIDS2IKReachResult:
        """Prove one physical pick before adding carry or place behavior."""

        result = XGRIDS2IKReachResult(
            grasp_coupling="bilateral_contact_then_fixed_joint_retention"
        )
        grasp_joint = None
        try:
            grasp_joint = self._grasp_joint()
            grasp_joint.GetJointEnabledAttr().Set(False)

            self.phase = "settle"
            result.phase_history.append(self.phase)
            self._hold(30)
            can_position = self.can.data.root_pos_w[0].clone()
            result.can_position = (
                can_position - self.env.scene.env_origins[0]
            ).detach().cpu().tolist()
            result.initial_can_z_m = float(result.can_position[2])
            dx, dy = PICK_APPROACH_DIRECTION
            pregrasp = (
                float(can_position[0]) - 0.12 * dx,
                float(can_position[1]) - 0.12 * dy,
                float(can_position[2]) + 0.06,
            )
            near_grasp = (
                float(can_position[0]) - 0.045 * dx,
                float(can_position[1]) - 0.045 * dy,
                float(can_position[2]),
            )
            grasp = tuple(float(value) for value in can_position)

            self.phase = "pregrasp"
            result.phase_history.append(self.phase)
            self._move_tcp(
                pregrasp,
                PICK_GRASP_QUAT_WXYZ,
                linear_speed_m_s=0.04,
            )
            self.phase = "near_grasp"
            result.phase_history.append(self.phase)
            self._move_tcp(
                near_grasp,
                PICK_GRASP_QUAT_WXYZ,
                linear_speed_m_s=0.025,
            )
            self.phase = "grasp_approach"
            result.phase_history.append(self.phase)
            self._move_tcp(
                grasp,
                PICK_GRASP_QUAT_WXYZ,
                linear_speed_m_s=0.010,
                max_steps=360,
            )

            self.phase = "close"
            result.phase_history.append(self.phase)
            contact_command = self._close_to_bilateral_contact()
            result.bilateral_contact_verified = True
            result.gripper_contact_joint_positions = (
                contact_command[0].detach().cpu().tolist()
            )

            self.phase = "post_contact_retention"
            result.phase_history.append(self.phase)
            self._attach_after_contact(grasp_joint)
            result.grasp_constraint_active = True
            self._hold(12, contact_command)

            lift_start, lift_quat = self.current_tcp_w()
            lift_target = lift_start.clone()
            lift_target[:, 2] += 0.12
            self.phase = "lift"
            result.phase_history.append(self.phase)
            self._move_tcp(
                tuple(float(value) for value in lift_target[0]),
                tuple(float(value) for value in lift_quat[0]),
                linear_speed_m_s=0.010,
                angular_speed_rad_s=0.10,
                max_steps=600,
                gripper_command=contact_command,
            )
            self._hold(30, contact_command)

            final_can = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
            result.final_can_position = final_can.detach().cpu().tolist()
            result.lifted_can_z_m = float(final_can[2].item())
            result.lift_height_m = result.lifted_can_z_m - result.initial_can_z_m
            self._record_quality(result)
            minimum_lift_z = DINING_TABLE.surface_z + CAN_HALF_HEIGHT + 0.08
            result.passed = (
                result.bilateral_contact_verified
                and result.grasp_constraint_active
                and result.lifted_can_z_m >= minimum_lift_z
                and result.maximum_arm_step_rad <= MAX_REACH_ARM_STEP_RAD
                and result.maximum_base_drift_m <= MAX_REACH_BASE_DRIFT_M
            )
            if not result.passed:
                raise RuntimeError(
                    "pick gate failed: can was not retained at the required lift height"
                )
            result.phase_history.append("success")
        except Exception as error:
            self._record_quality(result)
            result.failure_reason = str(error)
            result.phase_history.append("failure")
            if grasp_joint is not None:
                grasp_joint.GetJointEnabledAttr().Set(False)
                result.grasp_constraint_active = False
        return result
