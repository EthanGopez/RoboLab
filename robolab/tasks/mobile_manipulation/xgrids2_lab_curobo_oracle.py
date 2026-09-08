"""CuRobo pick-carry-place oracle for the XGRIDS2 lab."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch
from isaaclab.utils import math as math_utils
from pxr import Gf, UsdPhysics

from robolab.core.world.world_state import get_world
from robolab.robots.kinova_gen3 import GRIPPER_JOINT_COMMANDS
from robolab.robots.kinova_mobile_cobot import (
    MOBILE_KINOVA_FLANGE_BODY,
    MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS,
    MOBILE_KINOVA_LEFT_FINGERTIP_BODY,
    MOBILE_KINOVA_RIGHT_FINGERTIP_BODY,
)
from robolab.tasks.mobile_manipulation.curobo_fixed_base_planner import (
    CuroboFixedBasePlanner,
    world_pose_to_base,
)
from robolab.tasks.mobile_manipulation.fixed_base_manipulation import (
    ARM_JOINT_NAMES,
    JointTrajectory,
    Pose,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
    CAN_HALF_HEIGHT,
    CAN_RADIUS,
    GRASP_TCP_OFFSET_FLANGE,
)
from robolab.tasks.mobile_manipulation.kitchen_can_transfer_task import (
    ORACLE_GRASP_JOINT_NAME,
)
from robolab.tasks.mobile_manipulation.kitchen_manipulation_predicates import (
    can_bilaterally_grasped,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
    STAGING_POSES,
    planned_route_between,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_can_transfer_task import (
    manipulation_base_pose_is_clear,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_navigation_oracle import (
    WaypointFollower,
    world_velocity_to_body,
    wrap_angle,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_manipulation_geometry import (
    DINING_TABLE,
    PLACE_CENTER,
    PLACE_REGION,
    SNACK_MANIPULATION_BASE_POSE,
    SNACK_TABLE,
    TABLETOPS,
    snack_table_contains_xy,
)
from robolab.tasks.mobile_manipulation.xgrids2_motion_diagnostics import (
    ArmMotionDiagnostics,
    minimum_jerk_joint_targets,
    time_parameterized_joint_targets,
)


PICK_GRASP_QUAT_WXYZ = (-0.2520023, -0.2520023, 0.6606776, 0.6606776)
PLACE_GRASP_QUAT_WXYZ = (0.6532815, 0.6532815, 0.2705981, 0.2705981)
PICK_APPROACH_DIRECTION = (-0.6658, 0.7461)
PLACE_APPROACH_DIRECTION = (0.7071068, -0.7071068)
SIDE_PREGRASP_DISTANCE = 0.12
PICK_PREGRASP_HEIGHT_M = 0.12
PICK_LIFT_HEIGHT_M = 0.22
CARRY_ROUTE_RADIUS_M = 0.40
CARRY_PLANNING_RADIUS_M = 0.42
CARRY_ROUTE_APPROACH_XY = (4.46, -1.14)
CARRY_BASE_SPEED_M_S = 0.06
CARRY_MAX_LINK_CENTER_RADIUS_M = 0.350
CONTINUOUS_ARM_INDICES = (0, 2, 4, 6)
ARM_POSITION_LIMITS_RAD = (
    (-2.0 * math.pi, 2.0 * math.pi),
    (-2.24, 2.24),
    (-2.0 * math.pi, 2.0 * math.pi),
    (-2.57, 2.57),
    (-2.0 * math.pi, 2.0 * math.pi),
    (-2.09, 2.09),
    (-2.0 * math.pi, 2.0 * math.pi),
)
ARM_LIMIT_EPSILON_RAD = 1.0e-4
CONTINUOUS_LIMIT_GUARD_BAND_RAD = 0.40
# Compact the held object near the final transport TCP while preserving the
# side-grasp orientation. This prevents the subsequent wrist/arm fold from
# choosing a low joint-space sweep beside the dining table.
CARRY_TUCK_TCP_BASE = (0.32, 0.03, 0.68)
# Free-space arm tracking remains tight. Loaded pick/place segments use their
# own tolerances in ``_plan_execute`` because the physical drives exhibit a
# measurable payload-dependent steady-state error.
ARM_WAYPOINT_TRACKING_TOLERANCE_RAD = 0.050
LOADED_ARM_TARGET_STEP_RAD = 0.004
PREPLACE_ARM_TARGET_STEP_RAD = 0.004
PLACE_RELEASE_CLEARANCE_M = 0.040
PREPLACE_WAYPOINT_TRACKING_TOLERANCE_RAD = 0.250
PLACE_WAYPOINT_TRACKING_TOLERANCE_RAD = 0.075
PICK_BRANCH_ANCHOR_JOINTS = (
    -0.1202665002,
    0.2163686709,
    3.1756161042,
    -1.9628153204,
    0.0365685478,
    0.6087319635,
    1.5334912904,
)
# The loaded can introduces a small steady-state error in the physical PD
# drives. Forty milliradians (2.3 degrees) remains tighter than the compliant
# waypoint gate while allowing the arm to settle under that load.
ARM_ENDPOINT_TOLERANCE_RAD = 0.040
MAX_ARM_TARGET_STEP_RAD = 0.020
CONTINUOUS_ARM_TARGET_STEP_RAD = 0.010
MAX_ARM_STEP_QUALITY_RAD = 0.05
MAX_ARM_SPEED_QUALITY_RAD_S = 3.5
BASE_POSITION_TOLERANCE_M = 0.055
BASE_YAW_TOLERANCE_RAD = 0.06


def _slerp_quaternion(
    start: tuple[float, float, float, float],
    end: tuple[float, float, float, float],
    fraction: float,
) -> tuple[float, float, float, float]:
    """Interpolate unit quaternions along their shortest rotation."""

    left = np.asarray(start, dtype=np.float64)
    right = np.asarray(end, dtype=np.float64)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        value = left + fraction * (right - left)
        value /= np.linalg.norm(value)
    else:
        angle = math.acos(dot)
        value = (
            math.sin((1.0 - fraction) * angle) * left
            + math.sin(fraction * angle) * right
        ) / math.sin(angle)
    return tuple(float(component) for component in value)


@dataclass
class XGRIDS2CuroboResult:
    passed: bool = False
    requested_stage: str = "full"
    completed_stage: str = ""
    failure_reason: str = ""
    phase_history: list[str] = field(default_factory=list)
    planning_times_s: dict[str, float] = field(default_factory=dict)
    trajectory_waypoints: dict[str, int] = field(default_factory=dict)
    arm_endpoints_rad: dict[str, list[float]] = field(default_factory=dict)
    minimum_route_clearance_valid: bool = True
    pick_position: list[float] = field(default_factory=list)
    final_position: list[float] = field(default_factory=list)
    maximum_arm_step_rad: float = 0.0
    maximum_arm_speed_rad_s: float = 0.0
    maximum_raw_physx_arm_speed_rad_s: float = 0.0
    motion_diagnostics: dict = field(default_factory=dict)
    final_base_position_error_m: float = 0.0
    final_base_yaw_error_rad: float = 0.0
    carry_posture_radius_m: float = 0.0
    place_target: list[float] = field(default_factory=list)
    place_target_error_m: float = 0.0

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "requested_stage": self.requested_stage,
            "completed_stage": self.completed_stage,
            "execution_mode": "curobo_time_parameterized_continuous_position_tracking",
            "grasp_mode": "bilateral_contact_then_fixed_joint_retention",
            "transport_mode": "closed_loop_holonomic_inflated_astar_route",
            "release_mode": "constraint_release_then_planned_retreat",
            "carry_route_radius_m": CARRY_ROUTE_RADIUS_M,
            "carry_planning_radius_m": CARRY_PLANNING_RADIUS_M,
            "failure_reason": self.failure_reason,
            "phase_history": self.phase_history,
            "planning_times_s": self.planning_times_s,
            "trajectory_waypoints": self.trajectory_waypoints,
            "arm_endpoints_rad": self.arm_endpoints_rad,
            "minimum_route_clearance_valid": self.minimum_route_clearance_valid,
            "pick_position": self.pick_position,
            "final_position": self.final_position,
            "maximum_arm_step_rad": self.maximum_arm_step_rad,
            "maximum_arm_speed_rad_s": self.maximum_arm_speed_rad_s,
            "maximum_raw_physx_arm_speed_rad_s": (
                self.maximum_raw_physx_arm_speed_rad_s
            ),
            "motion_diagnostics": self.motion_diagnostics,
            "final_base_position_error_m": self.final_base_position_error_m,
            "final_base_yaw_error_rad": self.final_base_yaw_error_rad,
            "carry_posture_radius_m": self.carry_posture_radius_m,
            "place_target": self.place_target,
            "place_target_error_m": self.place_target_error_m,
        }


class XGRIDS2CuroboOracle:
    """Plan arm segments with CuRobo and traverse with the verified base route."""

    def __init__(
        self,
        env,
        *,
        num_seeds: int = 8,
        object_name: str = "can",
        grasp_joint_name: str = ORACLE_GRASP_JOINT_NAME,
        diagnostic_profile=None,
    ) -> None:
        if env.num_envs != 1:
            raise ValueError("XGRIDS2 CuRobo oracle currently requires num_envs=1")
        self.env = env
        self.diagnostic_profile = diagnostic_profile
        self.robot = env.scene["robot"]
        # ``can`` remains the default so the validated dining-to-snack task is
        # byte-for-byte equivalent at runtime.  The alias lets object-specific
        # subclasses reuse the physical execution and retention helpers.
        self.object_name = object_name
        self.can = env.scene[object_name]
        self.device = env.device
        self.num_seeds = num_seeds
        self._stage_label = "idle"
        self.arm_indices = [
            self.robot.data.joint_names.index(name) for name in ARM_JOINT_NAMES
        ]
        self.flange_index = self.robot.data.body_names.index(MOBILE_KINOVA_FLANGE_BODY)
        self.slices = self._action_slices()
        self.gripper_dim = self.slices["gripper"].stop - self.slices["gripper"].start
        self.gripper_drive_indices = [
            self.robot.data.joint_names.index(name)
            for name in MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS
        ]
        self.gripper_close_command = torch.tensor(
            [
                GRIPPER_JOINT_COMMANDS[name]
                for name in MOBILE_KINOVA_GRIPPER_DRIVE_JOINTS
            ],
            device=self.device,
        )
        self._gripper_target = self.robot.data.joint_pos[
            0, self.gripper_drive_indices
        ].clone()
        if self.slices["arm"].stop - self.slices["arm"].start != 7:
            raise ValueError("XGRIDS2 CuRobo oracle requires seven joint actions")
        self.tcp_offset = torch.tensor(
            [GRASP_TCP_OFFSET_FLANGE], device=self.device, dtype=torch.float32
        )
        self.grasp_joint = UsdPhysics.FixedJoint(
            env.scene.stage.GetPrimAtPath(
                f"{env.scene.env_prim_paths[0]}/{grasp_joint_name}"
            )
        )
        if not self.grasp_joint.GetPrim().IsValid():
            raise RuntimeError("XGRIDS2 task did not author its grasp-assist joint")
        self.grasp_joint.GetJointEnabledAttr().Set(False)
        self._maximum_arm_step_rad = 0.0
        self._maximum_arm_speed_rad_s = 0.0
        self._maximum_raw_physx_arm_speed_rad_s = 0.0
        self._control_dt = float(env.step_dt)
        initial_arm_position = self.robot.data.joint_pos[0, self.arm_indices].clone()
        self._motion_diagnostics = ArmMotionDiagnostics(
            self._control_dt, initial_arm_position.detach().cpu().tolist()
        )
        self._latest_motion_sample: dict = {}
        self._last_arm_position = initial_arm_position
        self._last_arm_target = self._last_arm_position.clone()
        self.defer_extended_motion_quality_gate = False
        self._maximum_arm_step_event = {}
        self._empty_return_segments: tuple[JointTrajectory, ...] = ()
        self._base_hold_xy, self._base_hold_yaw = self._robot_planar_state()
        self._base_hold_xy = self._base_hold_xy.copy()

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

    def _arm_positions(self) -> tuple[float, ...]:
        positions = self.robot.data.joint_pos[0, self.arm_indices].clone()
        for index in CONTINUOUS_ARM_INDICES:
            if float(torch.abs(positions[index]).item()) < (
                2.0 * math.pi - CONTINUOUS_LIMIT_GUARD_BAND_RAD
            ):
                positions[index] = torch.remainder(
                    positions[index] + math.pi, 2.0 * math.pi
                ) - math.pi
        return tuple(float(value) for value in positions.detach().cpu().tolist())

    def _arm_target_positions(self) -> tuple[float, ...]:
        """Return the last command, preserving it across chassis motion."""

        return tuple(
            float(value)
            for value in self._last_arm_target.detach().cpu().tolist()
        )

    def _default_arm_positions(self) -> tuple[float, ...]:
        return tuple(
            float(value)
            for value in self.robot.data.default_joint_pos[0, self.arm_indices]
            .detach()
            .cpu()
            .tolist()
        )

    def _action(
        self,
        arm_positions: tuple[float, ...],
        *,
        closed: bool,
        base_command=None,
    ) -> torch.Tensor:
        action = torch.zeros(
            (1, self.env.action_manager.total_action_dim), device=self.device
        )
        if base_command is None:
            base_command = self._base_hold_command()
        action[0, self.slices["base"]] = torch.tensor(
            base_command, device=self.device
        )
        desired_arm = torch.tensor(arm_positions, device=self.device)
        # Only joints 1/3/5/7 are continuous. Rebase their equivalent CuRobo
        # angles onto the physical articulation branch, but never outside the
        # finite +/-2*pi hardware/URDF range. Treating joints 2/4/6 as circular
        # previously generated out-of-range targets and a violent PhysX snap.
        actual_arm = self.robot.data.joint_pos[0, self.arm_indices]
        for index, (lower, upper) in enumerate(ARM_POSITION_LIMITS_RAD):
            if index in CONTINUOUS_ARM_INDICES:
                candidates = desired_arm[index] + 2.0 * math.pi * torch.arange(
                    -2, 3, device=self.device, dtype=desired_arm.dtype
                )
                valid = (candidates >= lower + ARM_LIMIT_EPSILON_RAD) & (
                    candidates <= upper - ARM_LIMIT_EPSILON_RAD
                )
                candidates = candidates[valid]
                if candidates.numel() == 0:
                    raise RuntimeError(f"joint_{index + 1} has no limit-valid circular target")
                distances = torch.abs(candidates - actual_arm[index])
                chosen = candidates[torch.argmin(distances)]
                if float(torch.abs(chosen - actual_arm[index]).item()) > math.pi:
                    raise RuntimeError(
                        f"joint_{index + 1} circular branch would cross its physical limit"
                    )
                desired_arm[index] = chosen
            elif not lower <= float(desired_arm[index].item()) <= upper:
                raise RuntimeError(
                    f"joint_{index + 1} target exceeds its physical limit: "
                    f"{float(desired_arm[index].item()):.4f} rad"
                )
        self._last_arm_target = desired_arm.clone()
        action[0, self.slices["arm"]] = desired_arm
        if self.gripper_dim == 2:
            desired = (
                self.gripper_close_command
                if closed
                else torch.zeros_like(self.gripper_close_command)
            )
            self._gripper_target += torch.clamp(
                desired - self._gripper_target, min=-0.008, max=0.008
            )
            action[0, self.slices["gripper"]] = self._gripper_target
        else:
            action[0, self.slices["gripper"]] = 1.0 if closed else 0.0
        return action

    def _circular_arm_error(self, desired: torch.Tensor) -> torch.Tensor:
        actual = self.robot.data.joint_pos[0, self.arm_indices]
        error = actual - desired
        error[list(CONTINUOUS_ARM_INDICES)] = torch.remainder(
            error[list(CONTINUOUS_ARM_INDICES)] + math.pi, 2.0 * math.pi
        ) - math.pi
        return error

    def _step(self, action: torch.Tensor) -> None:
        """Advance one physical control tick without overwriting articulation state."""

        before = self.robot.data.joint_pos[0, self.arm_indices].clone()
        _, _, terminated, truncated, _ = self.env.step(action)
        after = self.robot.data.joint_pos[0, self.arm_indices].clone()
        delta = after - before
        delta[list(CONTINUOUS_ARM_INDICES)] = torch.remainder(
            delta[list(CONTINUOUS_ARM_INDICES)] + math.pi, 2.0 * math.pi
        ) - math.pi
        step_magnitude = float(torch.max(torch.abs(delta)).item())
        if step_magnitude > self._maximum_arm_step_rad:
            self._maximum_arm_step_rad = step_magnitude
            self._maximum_arm_step_event = {
                "phase": self._stage_label,
                "before": before.detach().cpu().tolist(),
                "after": after.detach().cpu().tolist(),
                "target": self._last_arm_target.detach().cpu().tolist(),
            }
        raw_arm_speed = self.robot.data.joint_vel[0, self.arm_indices]
        raw_speed_magnitude = float(torch.max(torch.abs(raw_arm_speed)).item())
        self._maximum_raw_physx_arm_speed_rad_s = max(
            self._maximum_raw_physx_arm_speed_rad_s,
            raw_speed_magnitude,
        )
        self._latest_motion_sample = self._motion_diagnostics.update(
            target=self._last_arm_target.detach().cpu().tolist(),
            position=after.detach().cpu().tolist(),
            raw_velocity=raw_arm_speed.detach().cpu().tolist(),
            retention_active=bool(
                self.grasp_joint.GetJointEnabledAttr().Get()
            ),
            phase=self._stage_label,
        )
        # Use the finite difference over the complete control tick for the
        # quality gate. PhysX's final-substep velocity can spike while the
        # mobile base constraint is solved even when the observed joint pose
        # changes only a few milliradians over the rendered/control interval.
        speed_magnitude = step_magnitude / self._control_dt
        self._maximum_arm_speed_rad_s = max(
            self._maximum_arm_speed_rad_s,
            speed_magnitude,
        )
        step_limit = getattr(
            self, "max_arm_step_quality_rad", MAX_ARM_STEP_QUALITY_RAD
        )
        quality_exceeded = (
            step_magnitude > step_limit
            or speed_magnitude > MAX_ARM_SPEED_QUALITY_RAD_S
        )
        diagnostic_deferred = (
            getattr(self, "defer_extended_motion_quality_gate", False)
            and step_magnitude <= 2.0 * step_limit
            and speed_magnitude <= 2.0 * MAX_ARM_SPEED_QUALITY_RAD_S
        )
        if quality_exceeded and not diagnostic_deferred:
            raise RuntimeError(
                "physical arm motion-quality limit exceeded; "
                f"step={step_magnitude:.4f} rad; "
                f"speed={speed_magnitude:.4f} rad/s"
            )
        self._last_arm_position = after
        if bool(terminated[0].item() or truncated[0].item()):
            raise RuntimeError("environment terminated during XGRIDS2 CuRobo oracle")

    def _hold(self, steps: int, *, closed: bool, default: bool = False) -> None:
        if default:
            target = self._default_arm_positions()
        else:
            target = tuple(float(value) for value in self._last_arm_target.tolist())
        for _ in range(steps):
            self._step(self._action(target, closed=closed))

    def reset_robot_to_default(
        self,
        settle_steps: int = 24,
        *,
        target_arm_positions: tuple[float, ...] | None = None,
    ) -> None:
        """Let the damped drives converge to the baseline or a perturbed reset posture."""

        self._maximum_arm_step_rad = 0.0
        self._maximum_arm_speed_rad_s = 0.0
        self._maximum_raw_physx_arm_speed_rad_s = 0.0
        self._last_arm_position = self.robot.data.joint_pos[0, self.arm_indices].clone()
        self._last_arm_target = self._last_arm_position.clone()
        self._gripper_target = self.robot.data.joint_pos[
            0, self.gripper_drive_indices
        ].clone()
        root_xy_w = self.robot.data.root_link_pos_w[:, :2]
        _, yaw = self._robot_planar_state()
        self.env.action_manager.get_term("base").set_planar_pose(
            root_xy_w, torch.tensor([yaw], device=self.device)
        )
        if target_arm_positions is None:
            target = self._default_arm_positions()
            self._interpolate_arm_to(target, closed=False, max_step_rad=0.025)
            self._hold(settle_steps, closed=False, default=True)
            return
        target = tuple(float(value) for value in target_arm_positions)
        if len(target) != len(ARM_JOINT_NAMES):
            raise ValueError("target_arm_positions must contain seven arm joints")
        self._interpolate_arm_to(target, closed=False, max_step_rad=0.025)
        for _ in range(settle_steps):
            self._step(self._action(target, closed=False))

    def _interpolate_arm_to(
        self,
        target: tuple[float, ...],
        *,
        closed: bool,
        max_step_rad: float = MAX_ARM_TARGET_STEP_RAD,
        tracking_steps: int = 45,
        tracking_tolerance_rad: float = ARM_WAYPOINT_TRACKING_TOLERANCE_RAD,
        wrap_continuous_delta: bool = True,
        minimum_tracking_steps: int = 1,
    ) -> JointTrajectory:
        """Stream a minimum-jerk transition and settle only at its endpoint."""

        start = tuple(
            float(value) for value in self._last_arm_target.detach().cpu().tolist()
        )
        targets = minimum_jerk_joint_targets(
            start,
            target,
            maximum_target_step_rad=min(
                max_step_rad, CONTINUOUS_ARM_TARGET_STEP_RAD
            ),
            wrap_continuous_delta=wrap_continuous_delta,
        )
        for waypoint in targets:
            self._step(self._action(waypoint, closed=closed))

        endpoint = targets[-1]
        desired_tensor = torch.tensor(endpoint, device=self.device)
        for tracking_index in range(tracking_steps):
            self._step(self._action(endpoint, closed=closed))
            error = torch.max(torch.abs(self._circular_arm_error(desired_tensor)))
            if (
                tracking_index + 1 >= minimum_tracking_steps
                and float(error.item()) <= tracking_tolerance_rad
            ):
                return JointTrajectory(
                    (), targets, 0.0, sample_dt_s=self._control_dt
                )
        raise RuntimeError(
            "arm tracking stalled at a minimum-jerk endpoint; "
            f"max circular joint error={float(error.item()):.4f} rad"
        )

    def _execute(
        self,
        trajectory,
        *,
        closed: bool,
        max_step_rad: float = MAX_ARM_TARGET_STEP_RAD,
        tracking_steps: int = 45,
        tracking_tolerance_rad: float = ARM_WAYPOINT_TRACKING_TOLERANCE_RAD,
        endpoint_steps: int = 40,
        endpoint_tolerance_rad: float = ARM_ENDPOINT_TOLERANCE_RAD,
        wrap_continuous_delta: bool = True,
        minimum_tracking_steps: int = 1,
    ) -> None:
        if trajectory.sample_dt_s is not None and len(trajectory.positions) > 1:
            previous_target = tuple(
                float(value)
                for value in self._last_arm_target.detach().cpu().tolist()
            )
            sampler = time_parameterized_joint_targets
            options = {}
            step_limit = min(max_step_rad, CONTINUOUS_ARM_TARGET_STEP_RAD)
            if self.diagnostic_profile is not None:
                from robolab.tasks.mobile_manipulation.xgrids2_can_diagnostic import eased_joint_targets
                sampler = eased_joint_targets
                options["speed_factor"] = self.diagnostic_profile.arm_speed_factor
                step_limit = min(max_step_rad*self.diagnostic_profile.arm_speed_factor,
                                 self.diagnostic_profile.arm_target_step_ceiling_rad)
            targets = sampler(
                previous_target,
                trajectory.positions,
                source_dt_s=trajectory.sample_dt_s,
                control_dt_s=self._control_dt,
                maximum_target_step_rad=step_limit,
                **options,
            )
            for target in targets:
                self._step(self._action(target, closed=closed))
        else:
            for waypoint in trajectory.positions:
                self._interpolate_arm_to(
                    waypoint,
                    closed=closed,
                    max_step_rad=max_step_rad,
                    tracking_steps=tracking_steps,
                    tracking_tolerance_rad=tracking_tolerance_rad,
                    wrap_continuous_delta=wrap_continuous_delta,
                    minimum_tracking_steps=minimum_tracking_steps,
                )
        final = trajectory.positions[-1]
        desired = torch.tensor(final, device=self.device)
        for _ in range(endpoint_steps):
            self._step(self._action(final, closed=closed))
            error = torch.max(torch.abs(self._circular_arm_error(desired)))
            if float(error.item()) <= endpoint_tolerance_rad:
                return
        raise RuntimeError(
            "arm did not converge to the CuRobo segment endpoint; "
            f"max circular joint error={float(error.item()):.4f} rad"
        )

    def _plan_execute(
        self,
        planner: CuroboFixedBasePlanner,
        name: str,
        pose_w: Pose,
        result: XGRIDS2CuroboResult,
        *,
        closed: bool,
        allow_hand_contact: bool = False,
    ) -> None:
        result.phase_history.append(name)
        self._stage_label = name
        trajectory = planner.plan(
            self._arm_positions(),
            world_pose_to_base(pose_w, planner.base_pose),
            allow_hand_contact=allow_hand_contact,
        )
        result.planning_times_s[name] = trajectory.planning_time_s
        result.trajectory_waypoints[name] = len(trajectory.positions)
        is_place_reorientation = name.startswith("place_reorient_")
        loaded_pick = name in {"lift", "retract", "carry_tuck"}
        slow_loaded_place = is_place_reorientation or name in {
            "preplace",
            "place_approach",
            "place_lower",
            "place",
            "place_continuous",
        }
        try:
            self._execute(
                trajectory,
                closed=closed,
                max_step_rad=(
                    PREPLACE_ARM_TARGET_STEP_RAD
                    if name == "preplace"
                    else LOADED_ARM_TARGET_STEP_RAD
                    if slow_loaded_place or loaded_pick
                    else MAX_ARM_TARGET_STEP_RAD
                ),
                tracking_steps=(
                    120
                    if name == "preplace"
                    else 60
                    if slow_loaded_place or loaded_pick
                    else 45
                ),
                tracking_tolerance_rad=(
                    PREPLACE_WAYPOINT_TRACKING_TOLERANCE_RAD
                    if name == "preplace"
                    else PLACE_WAYPOINT_TRACKING_TOLERANCE_RAD
                    if slow_loaded_place
                    else 0.100
                    if loaded_pick
                    else ARM_WAYPOINT_TRACKING_TOLERANCE_RAD
                ),
                endpoint_steps=(
                    160
                    if name in {"preplace", "place", "place_continuous"}
                    else 100
                    if slow_loaded_place or loaded_pick
                    else 40
                ),
                endpoint_tolerance_rad=(
                    0.075
                    if is_place_reorientation
                    else 0.075
                    if name == "carry_tuck"
                    else 0.100
                    if name in {"lift", "retract"}
                    else 0.110
                    if name == "preplace"
                    else 0.075
                    if name in {"place_approach", "place_lower"}
                    else 0.050
                    if name in {"place", "place_continuous"}
                    else ARM_ENDPOINT_TOLERANCE_RAD
                ),
            )
        except RuntimeError as error:
            raise RuntimeError(f"{name}: {error}") from error
        result.arm_endpoints_rad[name] = list(self._arm_positions())
        return trajectory

    def _plan_execute_joint(
        self,
        planner: CuroboFixedBasePlanner,
        name: str,
        target: tuple[float, ...],
        result: XGRIDS2CuroboResult,
        *,
        closed: bool,
    ) -> None:
        result.phase_history.append(name)
        self._stage_label = name
        start = self._arm_positions()
        last_error = None
        trajectory = None
        for fraction in (1.0, 0.75, 0.5, 0.25):
            candidate_values = []
            for index, (current, goal) in enumerate(zip(start, target)):
                delta = goal - current
                if index in CONTINUOUS_ARM_INDICES:
                    delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
                candidate_values.append(current + fraction * delta)
            candidate = tuple(candidate_values)
            try:
                trajectory = planner.plan_joint(start, candidate)
                break
            except RuntimeError as error:
                last_error = error
        if trajectory is None:
            raise RuntimeError(f"{name}: {last_error}") from last_error
        result.planning_times_s[name] = trajectory.planning_time_s
        result.trajectory_waypoints[name] = len(trajectory.positions)
        try:
            self._execute(trajectory, closed=closed)
        except RuntimeError as error:
            raise RuntimeError(f"{name}: {error}") from error
        result.arm_endpoints_rad[name] = list(self._arm_positions())

    def _current_tcp_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        flange_pos = self.robot.data.body_pos_w[:, self.flange_index]
        flange_quat = self.robot.data.body_quat_w[:, self.flange_index]
        return (
            flange_pos + math_utils.quat_apply(flange_quat, self.tcp_offset),
            flange_quat,
        )

    def _attach(self) -> None:
        flange_pos = self.robot.data.body_pos_w[:, self.flange_index]
        flange_quat = self.robot.data.body_quat_w[:, self.flange_index]
        can_pos_f, can_quat_f = math_utils.subtract_frame_transforms(
            flange_pos, flange_quat, self.can.data.root_pos_w, self.can.data.root_quat_w
        )
        position = can_pos_f[0].detach().cpu().tolist()
        quaternion = can_quat_f[0].detach().cpu().tolist()
        self.grasp_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*position))
        self.grasp_joint.GetLocalRot0Attr().Set(
            Gf.Quatf(quaternion[0], Gf.Vec3f(*quaternion[1:]))
        )
        self.grasp_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0))
        self.grasp_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0))
        self.grasp_joint.GetJointEnabledAttr().Set(True)

    def _held_pose(self) -> Pose:
        tcp_pos, tcp_quat = self._current_tcp_pose()
        can_pos_t, can_quat_t = math_utils.subtract_frame_transforms(
            tcp_pos,
            tcp_quat,
            self.can.data.root_pos_w,
            self.can.data.root_quat_w,
        )
        return Pose(
            tuple(float(v) for v in can_pos_t[0].detach().cpu().tolist()),
            tuple(float(v) for v in can_quat_t[0].detach().cpu().tolist()),
        )

    def _tool_pose_for_can(
        self,
        can_position_w: tuple[float, float, float],
        held_pose: Pose,
        *,
        tool_quaternion_wxyz: tuple[float, float, float, float] | None = None,
    ) -> Pose:
        """Solve the tool pose that puts the measured held can at a world target."""

        can_position = torch.tensor([can_position_w], device=self.device)
        held_position = torch.tensor([held_pose.position], device=self.device)
        if tool_quaternion_wxyz is None:
            can_quaternion = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]], device=self.device
            )
            held_quaternion = torch.tensor(
                [held_pose.quaternion_wxyz], device=self.device
            )
            tool_quaternion = math_utils.quat_mul(
                can_quaternion, math_utils.quat_inv(held_quaternion)
            )
        else:
            tool_quaternion = torch.tensor(
                [tool_quaternion_wxyz], device=self.device
            )
        tool_position = can_position - math_utils.quat_apply(
            tool_quaternion, held_position
        )
        return Pose(
            tuple(float(value) for value in tool_position[0].detach().cpu().tolist()),
            tuple(float(value) for value in tool_quaternion[0].detach().cpu().tolist()),
        )

    def _tool_pose_for_object(
        self,
        object_position_w: tuple[float, float, float],
        object_quaternion_wxyz: tuple[float, float, float, float],
        held_pose: Pose,
    ) -> Pose:
        """Solve a tool pose for a full desired held-object world pose."""

        object_position = torch.tensor([object_position_w], device=self.device)
        object_quaternion = torch.tensor(
            [object_quaternion_wxyz], device=self.device
        )
        held_position = torch.tensor([held_pose.position], device=self.device)
        held_quaternion = torch.tensor(
            [held_pose.quaternion_wxyz], device=self.device
        )
        tool_quaternion = math_utils.quat_mul(
            object_quaternion, math_utils.quat_inv(held_quaternion)
        )
        tool_position = object_position - math_utils.quat_apply(
            tool_quaternion, held_position
        )
        return Pose(
            tuple(float(value) for value in tool_position[0].detach().cpu().tolist()),
            tuple(
                float(value)
                for value in tool_quaternion[0].detach().cpu().tolist()
            ),
        )

    def _close_and_attach(self, result: XGRIDS2CuroboResult) -> Pose:
        """Close to bilateral contact, then enable retention at coincident frames."""

        result.phase_history.append("close")
        self._stage_label = "close"
        target = self._arm_target_positions() if self.diagnostic_profile else self._arm_positions()
        for _ in range(150):
            self._step(self._action(target, closed=True))
            if bool(can_bilaterally_grasped(self.env, stable_frames=4)[0].item()):
                # Retain the measured bilateral-contact preload. Continuing
                # toward the full +/-0.8-rad close command over-constrains a
                # can that is now fixed to the flange and injects squeeze
                # impulses into the arm during loaded motion.
                self.gripper_close_command = self._gripper_target.clone()
                if self.diagnostic_profile:
                    self._hold(self.diagnostic_profile.attach_settle_ticks, closed=True)
                    if not bool(can_bilaterally_grasped(self.env, stable_frames=4)[0].item()):
                        raise RuntimeError("bilateral contact lost before retention activation")
                self._attach()
                self._hold(12, closed=True)
                return self._held_pose()
        world = get_world(self.env)
        left_index = self.robot.data.body_names.index(
            MOBILE_KINOVA_LEFT_FINGERTIP_BODY
        )
        right_index = self.robot.data.body_names.index(
            MOBILE_KINOVA_RIGHT_FINGERTIP_BODY
        )
        tcp_pos, tcp_quat = self._current_tcp_pose()
        diagnostics = {
            "tcp": tcp_pos[0].detach().cpu().tolist(),
            "tcp_quat_wxyz": tcp_quat[0].detach().cpu().tolist(),
            "can": self.can.data.root_pos_w[0].detach().cpu().tolist(),
            "left_fingertip": self.robot.data.body_pos_w[
                0, left_index
            ].detach().cpu().tolist(),
            "right_fingertip": self.robot.data.body_pos_w[
                0, right_index
            ].detach().cpu().tolist(),
            "gripper_drive_positions": self.robot.data.joint_pos[
                0, self.gripper_drive_indices
            ].detach().cpu().tolist(),
            "left_force": world.get_contact_force(
                "can", "left_fingertip"
            )[0].detach().cpu().tolist(),
            "right_force": world.get_contact_force(
                "can", "right_fingertip"
            )[0].detach().cpu().tolist(),
        }
        raise RuntimeError(
            "bilateral fingertip contact was not established; "
            f"contact_diagnostics={diagnostics}"
        )

    def _robot_planar_state(self) -> tuple[np.ndarray, float]:
        root = self.robot.data.root_link_pos_w[0] - self.env.scene.env_origins[0]
        quaternion = self.robot.data.root_link_quat_w[0]
        yaw = math.atan2(
            2.0
            * float(
                quaternion[0] * quaternion[3]
                + quaternion[1] * quaternion[2]
            ),
            1.0 - 2.0 * float(quaternion[2] ** 2 + quaternion[3] ** 2),
        )
        return root[:2].detach().cpu().numpy(), yaw

    def _set_base_hold(self, pose_xyzyaw) -> None:
        self._base_hold_xy = np.asarray(pose_xyzyaw[:2], dtype=np.float64)
        self._base_hold_yaw = float(pose_xyzyaw[3])

    def _base_hold_command(self) -> tuple[float, float, float]:
        """Reject arm-reaction drift without fixing or teleporting the base."""

        xy, yaw = self._robot_planar_state()
        delta = self._base_hold_xy - xy
        distance = float(np.linalg.norm(delta))
        speed = min(0.04, 1.25 * distance)
        velocity_world = delta / max(distance, 1.0e-9) * speed
        velocity_body = world_velocity_to_body(velocity_world, yaw)
        angular = float(
            np.clip(1.8 * wrap_angle(self._base_hold_yaw - yaw), -0.20, 0.20)
        )
        return float(velocity_body[0]), float(velocity_body[1]), angular

    def _carry_posture_radius(self) -> float:
        """Measure the robot-plus-can link-center radius around the mobile base."""

        root_xy = self.robot.data.root_link_pos_w[0, :2]
        robot_radius = torch.max(
            torch.linalg.vector_norm(
                self.robot.data.body_pos_w[0, :, :2] - root_xy, dim=1
            )
        )
        can_radius = torch.linalg.vector_norm(
            self.can.data.root_pos_w[0, :2] - root_xy
        )
        return float(torch.maximum(robot_radius, can_radius).item())

    def _align_pick_base(self, result: XGRIDS2CuroboResult) -> None:
        """Recover small randomized root offsets before fixed-base planning."""

        target = STAGING_POSES["dining_table"]
        self._set_base_hold(target)
        xy, yaw = self._robot_planar_state()
        if (
            float(np.linalg.norm(xy - np.asarray(target[:2]))) <= 0.003
            and abs(wrap_angle(yaw - target[3])) <= 0.008
        ):
            return
        result.phase_history.append("align_pick_base")
        self._stage_label = "align_pick_base"
        route = np.asarray((xy, target[:2]), dtype=np.float64)
        follower = WaypointFollower(
            route,
            final_yaw=target[3],
            max_linear_speed=0.04,
            max_angular_speed=0.20,
            waypoint_tolerance=0.005,
            final_position_tolerance=0.006,
            final_yaw_tolerance=0.012,
        )
        arm = self._arm_target_positions()
        reached = False
        for _ in range(600):
            xy, yaw = self._robot_planar_state()
            command, reached = follower.command(xy, yaw)
            self._step(
                self._action(
                    arm, closed=False, base_command=tuple(command.tolist())
                )
            )
            if reached:
                break
        if not reached:
            raise RuntimeError("base did not recover the dining manipulation pose")
        self._hold(15, closed=False)

    def _traverse(self, result: XGRIDS2CuroboResult) -> None:
        result.phase_history.append("traverse")
        self._stage_label = "traverse"
        target = SNACK_MANIPULATION_BASE_POSE
        self._set_base_hold(target)
        start_xy, _ = self._robot_planar_state()
        route = planned_route_between(
            tuple(float(value) for value in start_xy),
            CARRY_ROUTE_APPROACH_XY,
            robot_radius=CARRY_PLANNING_RADIUS_M,
        )
        route = np.vstack((route, np.asarray(target[:2], dtype=np.float64)))
        follower = WaypointFollower(
            route,
            final_yaw=target[3],
            max_linear_speed=self.diagnostic_profile.base_speed_m_s if self.diagnostic_profile else CARRY_BASE_SPEED_M_S,
            max_angular_speed=self.diagnostic_profile.base_yaw_speed_rad_s if self.diagnostic_profile else 0.35,
            smooth_heading=self.diagnostic_profile is not None,
            waypoint_tolerance=0.040,
            final_position_tolerance=0.020,
        )
        arm = self._arm_target_positions()
        reached = False
        for _ in range(3000):
            xy, yaw = self._robot_planar_state()
            command, reached = follower.command(xy, yaw)
            self._step(
                self._action(arm, closed=True, base_command=tuple(command.tolist()))
            )
            xy_after, _ = self._robot_planar_state()
            result.minimum_route_clearance_valid &= manipulation_base_pose_is_clear(
                tuple(float(value) for value in xy_after),
                robot_radius=CARRY_ROUTE_RADIUS_M,
            )
            if reached:
                break
        if not reached:
            raise RuntimeError(
                f"base follower stopped at route waypoint {follower.waypoint_index}"
            )
        self._hold(30, closed=True)
        xy, yaw = self._robot_planar_state()
        result.final_base_position_error_m = float(
            np.linalg.norm(xy - np.asarray(target[:2]))
        )
        result.final_base_yaw_error_rad = abs(wrap_angle(yaw - target[3]))
        if (
            result.final_base_position_error_m > BASE_POSITION_TOLERANCE_M
            or result.final_base_yaw_error_rad > BASE_YAW_TOLERANCE_RAD
        ):
            raise RuntimeError(
                "base did not settle at the snack manipulation stage; "
                f"position_error={result.final_base_position_error_m:.4f} m, "
                f"yaw_error={result.final_base_yaw_error_rad:.4f} rad"
            )

    def _capture_result_state(self, result: XGRIDS2CuroboResult) -> None:
        can_local = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
        result.final_position = can_local.detach().cpu().tolist()
        result.maximum_arm_step_rad = self._maximum_arm_step_rad
        result.maximum_arm_speed_rad_s = self._maximum_arm_speed_rad_s
        result.maximum_raw_physx_arm_speed_rad_s = (
            self._maximum_raw_physx_arm_speed_rad_s
        )
        result.motion_diagnostics = self._motion_diagnostics.summary()
        xy, yaw = self._robot_planar_state()
        result.final_base_position_error_m = float(
            np.linalg.norm(xy - self._base_hold_xy)
        )
        result.final_base_yaw_error_rad = abs(
            wrap_angle(yaw - self._base_hold_yaw)
        )

    def _complete_stage(
        self, result: XGRIDS2CuroboResult, completed_stage: str
    ) -> XGRIDS2CuroboResult:
        self._capture_result_state(result)
        result.completed_stage = completed_stage
        step_limit = getattr(
            self, "max_arm_step_quality_rad", MAX_ARM_STEP_QUALITY_RAD
        )
        if result.maximum_arm_step_rad > step_limit:
            raise RuntimeError(
                "arm motion-quality gate failed; "
                f"maximum one-tick joint step={result.maximum_arm_step_rad:.4f} rad"
            )
        if (
            not self.defer_extended_motion_quality_gate
            and result.motion_diagnostics.get("jerk_violation")
        ):
            raise RuntimeError(
                "arm jerk-quality gate failed; "
                f"diagnostics={result.motion_diagnostics}"
            )
        if (
            not self.defer_extended_motion_quality_gate
            and result.motion_diagnostics.get("oscillation_violation")
        ):
            raise RuntimeError(
                "arm oscillation-quality gate failed; "
                f"diagnostics={result.motion_diagnostics}"
            )
        result.passed = True
        result.phase_history.append(f"{completed_stage}_success")
        self._stage_label = f"{completed_stage}_success"
        return result

    def run(
        self,
        *,
        stage: str = "full",
        place_xy: tuple[float, float] | None = None,
        initial_arm_positions: tuple[float, ...] | None = None,
    ) -> XGRIDS2CuroboResult:
        if stage not in {"pick", "carry", "place", "full"}:
            raise ValueError("stage must be one of: pick, carry, place, full")
        place_xy = PLACE_CENTER if place_xy is None else tuple(float(v) for v in place_xy)
        result = XGRIDS2CuroboResult(requested_stage=stage)
        result.place_target = list(place_xy)
        pick_planner = None
        place_planner = None
        try:
            xmin, xmax, ymin, ymax = PLACE_REGION
            if not (
                xmin <= place_xy[0] <= xmax
                and ymin <= place_xy[1] <= ymax
                and snack_table_contains_xy(place_xy, margin=CAN_RADIUS)
            ):
                raise ValueError(f"place target is outside the validated region: {place_xy}")
            self.grasp_joint.GetPrim().SetActive(True)
            self.grasp_joint.GetJointEnabledAttr().Set(False)
            self.reset_robot_to_default(
                settle_steps=30 if initial_arm_positions is not None else 60,
                target_arm_positions=initial_arm_positions,
            )
            if initial_arm_positions is not None:
                result.phase_history.append("normalize_start")
                self._stage_label = "normalize_start"
                self._interpolate_arm_to(
                    self._default_arm_positions(),
                    closed=False,
                    max_step_rad=0.015,
                    tracking_steps=60,
                )
                self._hold(30, closed=False, default=True)
            self._align_pick_base(result)
            can_local = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
            result.pick_position = can_local.detach().cpu().tolist()
            pick = tuple(float(v) for v in self.can.data.root_pos_w[0].tolist())
            pick_base_xy, pick_base_yaw = self._robot_planar_state()
            pick_base_z = float(
                (
                    self.robot.data.root_link_pos_w[0, 2]
                    - self.env.scene.env_origins[0, 2]
                ).item()
            )
            pick_base_pose = (
                float(pick_base_xy[0]),
                float(pick_base_xy[1]),
                pick_base_z,
                pick_base_yaw,
            )
            pick_planner = CuroboFixedBasePlanner(
                pick_base_pose,
                num_seeds=self.num_seeds,
                collision_boxes=TABLETOPS,
            )
            self._plan_execute_joint(
                pick_planner,
                "pick_branch_anchor",
                PICK_BRANCH_ANCHOR_JOINTS,
                result,
                closed=False,
            )
            dx, dy = PICK_APPROACH_DIRECTION
            self._plan_execute(
                pick_planner,
                "pregrasp",
                Pose(
                    (
                        pick[0] - SIDE_PREGRASP_DISTANCE * dx,
                        pick[1] - SIDE_PREGRASP_DISTANCE * dy,
                        pick[2] + PICK_PREGRASP_HEIGHT_M,
                    ),
                    PICK_GRASP_QUAT_WXYZ,
                ),
                result,
                closed=False,
            )
            self._plan_execute(
                pick_planner,
                "approach",
                Pose(pick, PICK_GRASP_QUAT_WXYZ),
                result,
                closed=False,
                allow_hand_contact=True,
            )
            tcp_pos, _ = self._current_tcp_pose()
            reach_error = float(
                torch.linalg.vector_norm(tcp_pos[0] - self.can.data.root_pos_w[0])
            )
            if reach_error > 0.035:
                delta = tcp_pos[0] - self.can.data.root_pos_w[0]
                raise RuntimeError(
                    f"CuRobo approach stopped {reach_error:.4f} m from the can; "
                    f"tcp_minus_can={delta.detach().cpu().tolist()}"
                )
            held_pose = self._close_and_attach(result)
            self._plan_execute(
                pick_planner,
                "lift",
                Pose(
                    (pick[0], pick[1], pick[2] + PICK_LIFT_HEIGHT_M),
                    PICK_GRASP_QUAT_WXYZ,
                ),
                result,
                closed=True,
                allow_hand_contact=True,
            )
            self._hold(20, closed=True)
            lifted = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
            minimum_lift_z = DINING_TABLE.surface_z + CAN_HALF_HEIGHT + 0.08
            if float(lifted[2].item()) < minimum_lift_z:
                # The retention joint is intentionally compliant enough to avoid
                # violent post-contact motion. On a cold PhysX run that can leave
                # the object slightly below the nominal tool-space lift. Correct
                # from the measured pose through another planned, physically
                # executed segment instead of relaxing the clearance gate or
                # changing the object's simulation state.
                correction_z = minimum_lift_z - float(lifted[2].item()) + 0.025
                tcp_position, tcp_quaternion = self._current_tcp_pose()
                corrected_tcp_position = tcp_position[0].clone()
                corrected_tcp_position[2] += correction_z
                self._plan_execute(
                    pick_planner,
                    "lift_correction",
                    Pose(
                        tuple(
                            float(value)
                            for value in corrected_tcp_position.detach().cpu().tolist()
                        ),
                        tuple(
                            float(value)
                            for value in tcp_quaternion[0].detach().cpu().tolist()
                        ),
                    ),
                    result,
                    closed=True,
                    allow_hand_contact=True,
                )
                self._hold(20, closed=True)
                lifted = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
                if float(lifted[2].item()) < minimum_lift_z:
                    raise RuntimeError(
                        "pick did not lift the can clear of the dining table after "
                        f"physical correction; z={float(lifted[2]):.4f}"
                    )
            if stage == "pick":
                return self._complete_stage(result, "pick")

            pick_planner.set_held_object(held_pose)
            # Clear the dining-table edge at the lifted height before changing
            # arm configuration. Folding directly from above the pick point
            # lets small physical tracking errors drag the loaded gripper
            # through the tabletop even when the nominal CuRobo path is clear.
            self._plan_execute(
                pick_planner,
                "retract",
                Pose(
                    (
                        pick[0] - SIDE_PREGRASP_DISTANCE * dx,
                        pick[1] - SIDE_PREGRASP_DISTANCE * dy,
                        pick[2] + PICK_LIFT_HEIGHT_M,
                    ),
                    PICK_GRASP_QUAT_WXYZ,
                ),
                result,
                closed=True,
            )
            bx, by, bz, base_yaw = pick_base_pose
            tuck_x, tuck_y, tuck_z = CARRY_TUCK_TCP_BASE
            cosine, sine = math.cos(base_yaw), math.sin(base_yaw)
            self._plan_execute(
                pick_planner,
                "carry_tuck",
                Pose(
                    (
                        bx + cosine * tuck_x - sine * tuck_y,
                        by + sine * tuck_x + cosine * tuck_y,
                        bz + tuck_z,
                    ),
                    PICK_GRASP_QUAT_WXYZ,
                ),
                result,
                closed=True,
            )
            result.carry_posture_radius_m = self._carry_posture_radius()
            if result.carry_posture_radius_m > CARRY_MAX_LINK_CENTER_RADIUS_M:
                raise RuntimeError(
                    "CuRobo carry tuck exceeds the validated navigation envelope; "
                    f"link_center_radius={result.carry_posture_radius_m:.4f} m"
                )
            self._traverse(result)
            carried = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
            if float(carried[2].item()) < SNACK_TABLE.surface_z + CAN_HALF_HEIGHT + 0.05:
                raise RuntimeError("can was not retained above the snack table after traversal")
            if not result.minimum_route_clearance_valid:
                raise RuntimeError("base left the carried-object inflated free-space route")
            if stage == "carry":
                return self._complete_stage(result, "carry")

            place_planner = CuroboFixedBasePlanner(
                SNACK_MANIPULATION_BASE_POSE,
                num_seeds=self.num_seeds,
                collision_boxes=TABLETOPS,
            )
            place_planner.set_held_object(held_pose)
            place_path_segments: list[JointTrajectory] = []
            bx, by, bz, base_yaw = SNACK_MANIPULATION_BASE_POSE
            tuck_x, tuck_y, tuck_z = CARRY_TUCK_TCP_BASE
            cosine, sine = math.cos(base_yaw), math.sin(base_yaw)
            place_tuck_position = (
                bx + cosine * tuck_x - sine * tuck_y,
                by + sine * tuck_x + cosine * tuck_y,
                bz + tuck_z,
            )
            # Reorient the side grasp in settled increments. A single 177-degree
            # request can cross a continuous-joint branch while the can is loaded.
            for index, fraction in enumerate((1.0 / 3.0, 2.0 / 3.0, 1.0), start=1):
                place_path_segments.append(self._plan_execute(
                    place_planner,
                    f"place_reorient_{index}",
                    Pose(
                        place_tuck_position,
                        _slerp_quaternion(
                            PICK_GRASP_QUAT_WXYZ,
                            PLACE_GRASP_QUAT_WXYZ,
                            fraction,
                        ),
                    ),
                    result,
                    closed=True,
                ))
            place_z = SNACK_TABLE.surface_z + CAN_HALF_HEIGHT
            dx, dy = PLACE_APPROACH_DIRECTION
            preplace_can = (
                place_xy[0] - SIDE_PREGRASP_DISTANCE * dx,
                place_xy[1] - SIDE_PREGRASP_DISTANCE * dy,
                place_z + 0.14,
            )
            preplace_tool = self._tool_pose_for_can(
                preplace_can,
                held_pose,
                tool_quaternion_wxyz=PLACE_GRASP_QUAT_WXYZ,
            )
            place_path_segments.append(self._plan_execute(
                place_planner,
                "preplace",
                preplace_tool,
                result,
                closed=True,
            ))
            continuous_place = bool(self.diagnostic_profile and self.diagnostic_profile.continuous_place)
            # A single planned descent gets one start/end ramp. Separate
            # intermediate goals previously imposed two unnecessary stops.
            intermediate_goals = () if continuous_place else (
                ("place_approach", 0.14), ("place_lower", 0.09),
            )
            for name, clearance in intermediate_goals:
                transition_tool = self._tool_pose_for_can(
                    (place_xy[0], place_xy[1], place_z + clearance),
                    held_pose,
                    tool_quaternion_wxyz=PLACE_GRASP_QUAT_WXYZ,
                )
                place_path_segments.append(self._plan_execute(
                    place_planner,
                    name,
                    transition_tool,
                    result,
                    closed=True,
                ))
            place_tool = self._tool_pose_for_can(
                (place_xy[0], place_xy[1], place_z + PLACE_RELEASE_CLEARANCE_M),
                held_pose,
                tool_quaternion_wxyz=PLACE_GRASP_QUAT_WXYZ,
            )
            place_trajectory = self._plan_execute(
                place_planner,
                "place_continuous" if continuous_place else "place",
                place_tool,
                result,
                closed=True,
                allow_hand_contact=not continuous_place,
            )
            place_path_segments.append(place_trajectory)
            # After the explicit reverse of `place` below, these segments
            # retrace the rest of the already collision-checked loaded path
            # back to the navigation tuck. The return arm is empty, so its
            # collision envelope is strictly smaller than on the forward path.
            self._empty_return_segments = tuple(
                JointTrajectory(
                    segment.joint_names,
                    tuple(reversed(segment.positions)),
                    0.0,
                    sample_dt_s=segment.sample_dt_s,
                )
                for segment in reversed(place_path_segments[:-1])
            )
            self._hold(30, closed=True)
            if self.diagnostic_profile:
                from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import get_xgrids2_scenario
                from robolab.tasks.mobile_manipulation.xgrids2_passive_evaluation import placement_geometry
                spec = get_xgrids2_scenario("can_dining_to_snack")
                leg = spec.legs[0]
                support = spec.support(leg.place.support_name)
                geometry = placement_geometry(spec.object("can"), leg, support, {
                    "position_xyz": (self.can.data.root_pos_w[0]-self.env.scene.env_origins[0]).detach().cpu().tolist(),
                    "quaternion_wxyz": self.can.data.root_quat_w[0].detach().cpu().tolist(),
                })
                clearance = max(0., geometry["bottom_z"]-support.surface_z)
                speed = float(torch.linalg.vector_norm(self.can.data.root_lin_vel_w[0]).item())
                if not geometry["contained"] or clearance > self.diagnostic_profile.release_max_clearance_m or speed > .05:
                    raise RuntimeError(f"unsafe release: contained={geometry['contained']}, clearance={clearance:.4f}m, speed={speed:.4f}m/s")
            result.phase_history.append("release")
            self._stage_label = "release"
            self.grasp_joint.GetJointEnabledAttr().Set(False)
            place_planner.set_held_object(None)
            self._hold(90, closed=False)
            result.phase_history.append("retreat")
            self._stage_label = "retreat"
            retreat_trajectory = JointTrajectory(
                joint_names=place_trajectory.joint_names,
                positions=tuple(reversed(place_trajectory.positions)),
                planning_time_s=0.0,
                sample_dt_s=place_trajectory.sample_dt_s,
            )
            result.planning_times_s["retreat"] = 0.0
            result.trajectory_waypoints["retreat"] = len(
                retreat_trajectory.positions
            )
            self._execute(retreat_trajectory, closed=False)
            self._hold(120, closed=False)
            final = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
            result.place_target_error_m = math.dist(
                (float(final[0]), float(final[1])), place_xy
            )
            settled = (
                snack_table_contains_xy(
                    (float(final[0]), float(final[1])), margin=CAN_RADIUS
                )
                and result.place_target_error_m <= 0.08
                and abs(float(final[2]) - place_z) < 0.025
                and float(torch.linalg.vector_norm(self.can.data.root_lin_vel_w[0]))
                < 0.05
            )
            if not settled:
                raise RuntimeError(
                    "released can did not settle at the sampled snack-table target: "
                    f"final={final.detach().cpu().tolist()}, target={place_xy}, "
                    f"xy_error={result.place_target_error_m:.4f} m"
                )
            return self._complete_stage(result, "place" if stage == "place" else "full")
        except Exception as error:
            self._capture_result_state(result)
            root_xy, root_yaw = self._robot_planar_state()
            tcp_pos, _ = self._current_tcp_pose()
            tcp_local = tcp_pos[0] - self.env.scene.env_origins[0]
            result.failure_reason = (
                f"{error}; arm={self._arm_positions()}; "
                f"base_xy={root_xy.tolist()}; base_yaw={root_yaw:.4f}; "
                f"tcp={tcp_local.detach().cpu().tolist()}; "
                f"max_step_event={self._maximum_arm_step_event}"
            )
            result.phase_history.append("failure")
            self._stage_label = "failure"
            self.grasp_joint.GetJointEnabledAttr().Set(False)
            if pick_planner is not None:
                pick_planner.set_held_object(None)
            if place_planner is not None:
                place_planner.set_held_object(None)
            return result
