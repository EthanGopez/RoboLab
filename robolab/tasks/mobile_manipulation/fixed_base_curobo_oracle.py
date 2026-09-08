"""Single-environment fixed-base pick/place primitive driven by CuRobo plans."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from isaaclab.utils import math as math_utils
from pxr import Gf, UsdPhysics

from robolab.robots.kinova_mobile_cobot import MOBILE_KINOVA_FLANGE_BODY
from robolab.tasks.mobile_manipulation.curobo_fixed_base_planner import (
    CuroboFixedBasePlanner,
    world_pose_to_base,
)
from robolab.tasks.mobile_manipulation.fixed_base_manipulation import (
    ARM_JOINT_NAMES,
    FIXED_BASE_LOCAL_PLACE_REGION,
    FIXED_BASE_PREGRASP_HEIGHT,
    FixedBasePhase,
    Pose,
    can_center_height,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
    GRASP_TCP_OFFSET_FLANGE,
    ISLAND_SURFACE_Z,
    MIN_LIFT_CLEARANCE,
    CAN_HALF_HEIGHT,
    PLACE_SIDE_GRASP_QUAT_WXYZ,
    SIDE_PREGRASP_DISTANCE,
)
from robolab.tasks.mobile_manipulation.kitchen_can_transfer_task import (
    ORACLE_GRASP_JOINT_NAME,
)
from robolab.tasks.mobile_manipulation.kitchen_manipulation_predicates import (
    can_bilaterally_grasped,
    can_supported_by_island,
    gripper_released,
    robot_collision_free,
)


@dataclass
class FixedBasePrimitiveResult:
    """JSON-friendly evidence from one local pick/place trial."""

    grasp_success: bool = False
    place_success: bool = False
    failure_reason: str = ""
    phase_history: list[str] = field(default_factory=list)
    planning_times_s: dict[str, float] = field(default_factory=dict)
    trajectory_waypoints: dict[str, int] = field(default_factory=dict)
    pick_xy: tuple[float, float] = (0.0, 0.0)
    place_xy: tuple[float, float] = (0.0, 0.0)
    pre_release_can_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pre_release_error_xy: float = math.inf
    final_can_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    placement_error_xy: float = math.inf

    def as_dict(self) -> dict:
        return {
            "grasp_success": self.grasp_success,
            "place_success": self.place_success,
            "failure_reason": self.failure_reason,
            "phase_history": self.phase_history,
            "planning_times_s": self.planning_times_s,
            "trajectory_waypoints": self.trajectory_waypoints,
            "pick_xy": list(self.pick_xy),
            "place_xy": list(self.place_xy),
            "pre_release_can_xyz": list(self.pre_release_can_xyz),
            "pre_release_error_xy": self.pre_release_error_xy,
            "final_can_xyz": list(self.final_can_xyz),
            "placement_error_xy": self.placement_error_xy,
        }


class FixedBaseCuroboPrimitive:
    """Execute open, pregrasp, grasp, lift, retract, and local place."""

    def __init__(self, env, planner: CuroboFixedBasePlanner) -> None:
        if env.num_envs != 1:
            raise ValueError("fixed-base CuRobo primitive currently requires num_envs=1")
        self.env = env
        self.planner = planner
        self.robot = env.scene["robot"]
        self.can = env.scene["can"]
        self.arm_indices = [
            self.robot.data.joint_names.index(name) for name in ARM_JOINT_NAMES
        ]
        self.flange_index = self.robot.data.body_names.index(MOBILE_KINOVA_FLANGE_BODY)
        self.action_slices = self._resolve_action_slices()
        if self.action_slices["arm"].stop - self.action_slices["arm"].start != 7:
            raise ValueError("joint-position environment must expose seven arm actions")
        stage = env.scene.stage
        self.grasp_joint = UsdPhysics.FixedJoint(
            stage.GetPrimAtPath(
                f"{env.scene.env_prim_paths[0]}/{ORACLE_GRASP_JOINT_NAME}"
            )
        )
        if not self.grasp_joint.GetPrim().IsValid():
            raise RuntimeError("the task did not author its post-contact grasp joint")
        self.grasp_joint.GetJointEnabledAttr().Set(False)
        self.tcp_offset = torch.tensor(
            [GRASP_TCP_OFFSET_FLANGE], device=env.device, dtype=torch.float32
        )

    def _resolve_action_slices(self) -> dict[str, slice]:
        result: dict[str, slice] = {}
        offset = 0
        for name, size in zip(
            self.env.action_manager.active_terms,
            self.env.action_manager.action_term_dim,
        ):
            result[name] = slice(offset, offset + size)
            offset += size
        return result

    def _arm_positions(self) -> tuple[float, ...]:
        return tuple(
            float(value)
            for value in self.robot.data.joint_pos[0, self.arm_indices]
            .detach()
            .cpu()
            .tolist()
        )

    def _command(self, arm_positions: tuple[float, ...], *, closed: bool) -> torch.Tensor:
        action = torch.zeros(
            (1, self.env.action_manager.total_action_dim), device=self.env.device
        )
        action[0, self.action_slices["arm"]] = torch.tensor(
            arm_positions, device=self.env.device
        )
        action[0, self.action_slices["gripper"]] = 1.0 if closed else 0.0
        return action

    def _step(self, arm_positions: tuple[float, ...], *, closed: bool) -> None:
        _, _, terminated, truncated, _ = self.env.step(
            self._command(arm_positions, closed=closed)
        )
        if bool(terminated[0].item() or truncated[0].item()):
            raise RuntimeError("RoboLab environment terminated during primitive")

    def _hold(self, steps: int, *, closed: bool) -> None:
        target = self._arm_positions()
        for _ in range(steps):
            self._step(target, closed=closed)

    def _execute_trajectory(
        self,
        positions: tuple[tuple[float, ...], ...],
        *,
        closed: bool,
        final_tolerance: float = 0.018,
        final_steps: int = 20,
    ) -> None:
        for waypoint in positions:
            # One control tick follows the 50 ms CuRobo interpolation closely;
            # repeat only when the position servo falls materially behind.
            self._step(waypoint, closed=closed)
            for _ in range(1):
                actual = self.robot.data.joint_pos[0, self.arm_indices]
                error = torch.max(
                    torch.abs(actual - torch.tensor(waypoint, device=self.env.device))
                )
                if float(error.item()) <= 0.035:
                    break
                self._step(waypoint, closed=closed)
        final = positions[-1]
        for _ in range(final_steps):
            self._step(final, closed=closed)
            actual = self.robot.data.joint_pos[0, self.arm_indices]
            error = torch.max(
                torch.abs(actual - torch.tensor(final, device=self.env.device))
            )
            if float(error.item()) <= final_tolerance:
                break

    def _plan_and_execute(
        self,
        phase: FixedBasePhase,
        target_world: Pose,
        result: FixedBasePrimitiveResult,
        *,
        closed: bool,
        allow_hand_contact: bool = False,
        final_tolerance: float = 0.018,
        final_steps: int = 20,
    ) -> None:
        result.phase_history.append(phase.value)
        trajectory = self.planner.plan(
            self._arm_positions(),
            world_pose_to_base(target_world, self.planner.base_pose),
            allow_hand_contact=allow_hand_contact,
        )
        result.planning_times_s[phase.value] = trajectory.planning_time_s
        result.trajectory_waypoints[phase.value] = len(trajectory.positions)
        self._execute_trajectory(
            trajectory.positions,
            closed=closed,
            final_tolerance=final_tolerance,
            final_steps=final_steps,
        )

    def _current_tcp_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        flange_pos = self.robot.data.body_pos_w[:, self.flange_index]
        flange_quat = self.robot.data.body_quat_w[:, self.flange_index]
        tcp_pos = flange_pos + math_utils.quat_apply(flange_quat, self.tcp_offset)
        return tcp_pos, flange_quat

    def _tool_pose_for_can(
        self, can_position_w: tuple[float, float, float], held_pose: Pose
    ) -> Pose:
        """Solve T_world_tool = T_world_can times inverse(T_tool_can)."""

        can_position = torch.tensor([can_position_w], device=self.env.device)
        can_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.env.device)
        held_position = torch.tensor([held_pose.position], device=self.env.device)
        held_quaternion = torch.tensor(
            [held_pose.quaternion_wxyz], device=self.env.device
        )
        tool_quaternion = math_utils.quat_mul(
            can_quaternion, math_utils.quat_inv(held_quaternion)
        )
        tool_position = can_position - math_utils.quat_apply(
            tool_quaternion, held_position
        )
        return Pose(
            tuple(float(value) for value in tool_position[0].detach().cpu().tolist()),
            tuple(float(value) for value in tool_quaternion[0].detach().cpu().tolist()),
        )

    def _attach_verified_can(self) -> None:
        """Enable the existing PhysX assist at the measured contact transform."""

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
        self.grasp_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*position))
        self.grasp_joint.GetLocalRot0Attr().Set(
            Gf.Quatf(quaternion[0], Gf.Vec3f(*quaternion[1:]))
        )
        self.grasp_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0))
        self.grasp_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0))
        self.grasp_joint.GetJointEnabledAttr().Set(True)

    def _current_can_in_tcp_pose(self) -> Pose:
        """Measure the carried-object transform after PhysX has settled."""

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

    def _close_and_verify(self, result: FixedBasePrimitiveResult) -> Pose:
        result.phase_history.append(FixedBasePhase.CLOSE.value)
        target = self._arm_positions()
        for _ in range(120):
            self._step(target, closed=True)
            if bool(can_bilaterally_grasped(self.env, stable_frames=4)[0].item()):
                self._attach_verified_can()
                self._hold(12, closed=True)
                return self._current_can_in_tcp_pose()
        raise RuntimeError("bilateral fingertip contact was not established")

    def run(
        self,
        *,
        pick_xy: tuple[float, float],
        place_xy: tuple[float, float],
    ) -> FixedBasePrimitiveResult:
        result = FixedBasePrimitiveResult(pick_xy=pick_xy, place_xy=place_xy)
        grasp_quat = PLACE_SIDE_GRASP_QUAT_WXYZ
        pick_z = can_center_height()
        try:
            self.grasp_joint.GetJointEnabledAttr().Set(False)
            self.planner.set_held_object(None)
            result.phase_history.append(FixedBasePhase.OPEN.value)
            self._hold(30, closed=False)
            actual_pick = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
            pick_xy = (float(actual_pick[0].item()), float(actual_pick[1].item()))
            result.pick_xy = pick_xy

            self._plan_and_execute(
                FixedBasePhase.PREGRASP,
                Pose(
                    (
                        pick_xy[0] - SIDE_PREGRASP_DISTANCE,
                        pick_xy[1],
                        pick_z + FIXED_BASE_PREGRASP_HEIGHT,
                    ),
                    grasp_quat,
                ),
                result,
                closed=False,
            )
            self._plan_and_execute(
                FixedBasePhase.APPROACH,
                Pose((pick_xy[0], pick_xy[1], pick_z), grasp_quat),
                result,
                closed=False,
                allow_hand_contact=True,
            )
            held_pose = self._close_and_verify(result)
            self._plan_and_execute(
                FixedBasePhase.LIFT,
                Pose((pick_xy[0], pick_xy[1], pick_z + 0.15), grasp_quat),
                result,
                closed=True,
                allow_hand_contact=True,
            )
            # The support contact is now cleared, so the can becomes an active
            # CuRobo collision body for every subsequent carried segment.
            self.planner.set_held_object(held_pose)
            self._plan_and_execute(
                FixedBasePhase.RETRACT,
                Pose(
                    (pick_xy[0] - SIDE_PREGRASP_DISTANCE, pick_xy[1], pick_z + 0.15),
                    grasp_quat,
                ),
                result,
                closed=True,
            )
            can_local = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
            lifted = float(can_local[2].item() - CAN_HALF_HEIGHT) >= (
                ISLAND_SURFACE_Z + MIN_LIFT_CLEARANCE
            )
            result.grasp_success = lifted and bool(robot_collision_free(self.env)[0].item())
            if not result.grasp_success:
                raise RuntimeError("can was not retained through lift and retract")

            preplace_tool = self._tool_pose_for_can(
                (
                    place_xy[0] - SIDE_PREGRASP_DISTANCE,
                    place_xy[1],
                    pick_z + 0.15,
                ),
                held_pose,
            )
            self._plan_and_execute(
                FixedBasePhase.PREPLACE,
                preplace_tool,
                result,
                closed=True,
            )
            place_tool = self._tool_pose_for_can(
                (place_xy[0], place_xy[1], can_center_height(hover=0.015)),
                held_pose,
            )
            self._plan_and_execute(
                FixedBasePhase.PLACE,
                place_tool,
                result,
                closed=True,
                allow_hand_contact=True,
                # Placement is the one phase where a small residual joint
                # error turns into a centimetre-scale object-target error.
                # Let the existing position servo settle before releasing;
                # this changes no controller or robot-specific planning logic.
                final_tolerance=0.005,
                final_steps=80,
            )
            pre_release_can = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
            result.pre_release_can_xyz = tuple(
                float(value) for value in pre_release_can.detach().cpu().tolist()
            )
            result.pre_release_error_xy = math.hypot(
                result.pre_release_can_xyz[0] - place_xy[0],
                result.pre_release_can_xyz[1] - place_xy[1],
            )
            result.phase_history.append(FixedBasePhase.RELEASE.value)
            self.grasp_joint.GetJointEnabledAttr().Set(False)
            self.planner.set_held_object(None)
            self._hold(150, closed=False)

            self._plan_and_execute(
                FixedBasePhase.RETREAT,
                preplace_tool,
                result,
                closed=False,
            )
            self._hold(60, closed=False)
            final_can = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
            result.final_can_xyz = tuple(
                float(value) for value in final_can.detach().cpu().tolist()
            )
            result.placement_error_xy = math.hypot(
                result.final_can_xyz[0] - place_xy[0],
                result.final_can_xyz[1] - place_xy[1],
            )
            supported = bool(
                can_supported_by_island(
                    self.env,
                    region=FIXED_BASE_LOCAL_PLACE_REGION,
                    support_name="island_top_west",
                )[0].item()
            )
            released = bool(gripper_released(self.env, stable_frames=1)[0].item())
            collision_free = bool(robot_collision_free(self.env)[0].item())
            result.place_success = (
                supported
                and released
                and collision_free
            )
            if not result.place_success:
                raise RuntimeError("released can did not settle in the local place region")
            result.phase_history.append(FixedBasePhase.SUCCESS.value)
        except Exception as error:
            result.failure_reason = str(error)
            result.phase_history.append(FixedBasePhase.FAILURE.value)
            self.grasp_joint.GetJointEnabledAttr().Set(False)
            self.planner.set_held_object(None)
        return result
