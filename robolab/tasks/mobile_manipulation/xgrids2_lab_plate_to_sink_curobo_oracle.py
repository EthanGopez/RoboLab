"""CuRobo oracle for moving a plate from the dining table into the sink."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from isaaclab.utils import math as math_utils
from pxr import Gf, Sdf

from robolab.core.world.world_state import get_world
from robolab.robots.kinova_mobile_cobot import (
    MOBILE_KINOVA_LEFT_FINGERTIP_BODY,
    MOBILE_KINOVA_RIGHT_FINGERTIP_BODY,
)
from robolab.tasks.mobile_manipulation.curobo_fixed_base_planner import (
    CuroboFixedBasePlanner,
    world_pose_to_base,
)
from robolab.tasks.mobile_manipulation.fixed_base_manipulation import (
    JointTrajectory,
    Pose,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_curobo_oracle import (
    BASE_POSITION_TOLERANCE_M,
    BASE_YAW_TOLERANCE_RAD,
    CARRY_BASE_SPEED_M_S,
    PICK_BRANCH_ANCHOR_JOINTS,
    XGRIDS2CuroboOracle,
    XGRIDS2CuroboResult,
    _slerp_quaternion,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_manipulation_geometry import (
    DINING_TABLE,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_navigation_oracle import (
    WaypointFollower,
    wrap_angle,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_plate_sink_geometry import (
    PICK_APPROACH_DIRECTION,
    PLANNER_COLLISION_BOXES,
    PLATE_HELD_SPHERES,
    PLATE_RADIUS_M,
    SINK_CENTER_XY,
    SINK_MANIPULATION_BASE_POSE,
    SINK_PLACE_SETTLED_Z,
    SINK_RELEASE_CLEARANCE_M,
    planned_sink_route,
    sink_base_pose_is_clear,
    sink_contains_plate,
)


# The original can side grasp has its finger-closing plane horizontal.  Rolling
# it 90 degrees about the approach axis makes the fingers straddle the thin
# plate rim: one pad above the lip and one below its intentional overhang.
PLATE_RIM_GRASP_QUAT_WXYZ = (
    -0.6453621139,
    0.2889770614,
    0.6453621139,
    0.2889770614,
)
PLATE_PREGRASP_DISTANCE_M = 0.11
PLATE_LIFT_M = 0.24
PLATE_RETRACT_M = 0.11
PLATE_CARRY_OBJECT_BASE = (0.38, 0.0, 0.62)
PLATE_ROUTE_RADIUS_M = 0.47
PLATE_ROUTE_PLANNING_RADIUS_M = 0.47
PLATE_MAX_CARRY_RADIUS_M = 0.47
PLATE_CARRY_RADIUS_EPSILON_M = 0.001
# The TCP-to-pad offset puts both fingertip centers about 30 mm farther out
# than the TCP.  Targeting a 75 mm TCP radius puts the pads near 105 mm: on the
# plate's flatter annulus, yet 18 mm beyond the dining-table edge so the lower
# fingertip does not collide with the tabletop.
PLATE_PICK_CONTACT_INSET_M = 0.074
PLATE_GRASP_TCP_Z_OFFSET_M = 0.055
# Collision-checked CuRobo joint anchors from the stable 70-degree carry roll.
# Pose-only IK can choose an equivalent joint-3 solution beside +2*pi; these
# anchors keep the same physical plate motion on the interior branch.
PLATE_CARRY_REORIENT_3_JOINTS = (
    -1.1387212, -0.4628392, -1.8586628, -0.9247512,
    2.4632547, 1.5619117, -1.1065943,
)
PLATE_CARRY_REORIENT_4_JOINTS = (
    -1.5833946, -0.4968171, -1.7015154, -0.5413312,
    2.6900036, 1.8006258, -1.0914924,
)
PLATE_CARRY_TUCK_JOINTS = (
    -0.9043367, -0.6051639, -1.9961073, -1.4252756,
    2.5577042, 1.4717948, -0.6289308,
)


@dataclass
class XGRIDS2PlateSinkResult(XGRIDS2CuroboResult):
    plate_upright_cosine: float = 0.0

    def as_dict(self) -> dict:
        data = super().as_dict()
        data.update(
            {
                "task": "dining_plate_to_sink",
                "object": "plate",
                "destination": "sink_basin",
                "grasp_mode": "measured_fingertip_contact_then_fixed_joint_retention",
                "transport_mode": (
                    "base_anchored_plate_with_unloaded_arm_stow_"
                    "inflated_astar_route"
                ),
                "release_mode": (
                    "base_retention_guided_lowering_then_constraint_release"
                ),
                "carry_route_radius_m": PLATE_ROUTE_RADIUS_M,
                "carry_planning_radius_m": PLATE_ROUTE_PLANNING_RADIUS_M,
                "plate_upright_cosine": self.plate_upright_cosine,
            }
        )
        return data


def _vertical_plate_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Quaternion whose plate normal points along the base longitudinal axis.

    The disk is symmetric under a normal flip.  Use the backward-facing normal
    because the forward-facing equivalent winds joint 3 into its +2*pi guard.
    """

    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    root_half = math.sqrt(0.5)
    return (
        cy * root_half,
        sy * root_half,
        -cy * root_half,
        sy * root_half,
    )


class XGRIDS2PlateToSinkCuroboOracle(XGRIDS2CuroboOracle):
    """Contact-verified plate pick, vertical carry, and basin placement."""

    def __init__(self, env, *, num_seeds: int = 16) -> None:
        super().__init__(env, num_seeds=num_seeds, object_name="plate")
        self.max_arm_step_quality_rad = 0.065
        targets = self.grasp_joint.GetBody0Rel().GetTargets()
        if not targets:
            raise RuntimeError("plate retention joint has no flange body target")
        self._flange_body_path = targets[0]

    def _retarget_retention(
        self,
        body_path: Sdf.Path,
        body_position_w: torch.Tensor,
        body_quaternion_w: torch.Tensor,
    ) -> None:
        """Move the enabled retention joint to a new body at coincident frames."""

        object_position_b, object_quaternion_b = math_utils.subtract_frame_transforms(
            body_position_w,
            body_quaternion_w,
            self.can.data.root_pos_w,
            self.can.data.root_quat_w,
        )
        position = object_position_b[0].detach().cpu().tolist()
        quaternion = object_quaternion_b[0].detach().cpu().tolist()
        self.grasp_joint.GetBody0Rel().SetTargets([body_path])
        self.grasp_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*position))
        self.grasp_joint.GetLocalRot0Attr().Set(
            Gf.Quatf(quaternion[0], Gf.Vec3f(*quaternion[1:]))
        )
        self.grasp_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0))
        self.grasp_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0))

    def _transfer_retention_to_base(self) -> None:
        base_path = Sdf.Path(f"{self.env.scene.env_prim_paths[0]}/robot/base_link")
        self._retarget_retention(
            base_path,
            self.robot.data.root_link_pos_w,
            self.robot.data.root_link_quat_w,
        )

    def _transfer_retention_to_flange(self) -> None:
        flange_position = self.robot.data.body_pos_w[:, self.flange_index]
        flange_quaternion = self.robot.data.body_quat_w[:, self.flange_index]
        self._retarget_retention(
            self._flange_body_path,
            flange_position,
            flange_quaternion,
        )

    def _guided_lower_retained_plate(
        self,
        target_center_z: float,
        result: XGRIDS2CuroboResult,
    ) -> None:
        """Lower a base-retained plate vertically in bounded joint-frame steps."""

        result.phase_history.append("basin_guided_lowering")
        self._stage_label = "basin_guided_lowering"
        current = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
        vertical_delta = target_center_z - float(current[2].item())
        if vertical_delta >= -0.005:
            raise RuntimeError(
                "guided lowering requires the plate to begin above its release pose"
            )
        local_frame = list(self.grasp_joint.GetLocalPos0Attr().Get())
        steps = max(1, int(math.ceil(abs(vertical_delta) / 0.002)))
        arm_target = tuple(float(value) for value in self._last_arm_target.tolist())
        for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
            self.grasp_joint.GetLocalPos0Attr().Set(
                Gf.Vec3f(
                    float(local_frame[0]),
                    float(local_frame[1]),
                    float(local_frame[2] + fraction * vertical_delta),
                )
            )
            self._locked_transport_step(
                self._action(arm_target, closed=False)
            )
        result.planning_times_s["basin_guided_lowering"] = 0.0
        result.trajectory_waypoints["basin_guided_lowering"] = steps
        self._hold(30, closed=False)
        lowered = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
        if (
            not sink_contains_plate(
                (float(lowered[0]), float(lowered[1])), margin=0.005
            )
            or abs(float(lowered[2]) - target_center_z) > 0.02
        ):
            raise RuntimeError(
                "guided plate did not reach its contained release pose; "
                f"position={lowered.detach().cpu().tolist()}"
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
    ):
        """Use a payload-appropriate convergence gate on loaded plate moves.

        The earlier 0.22 rad tracking allowance let the physical arm trail a
        CuRobo waypoint by roughly 0.3 rad across successive interpolations.
        With proper payload-holding gains that lag becomes a fast catch-up
        transient.  Keep each loaded interpolation close to its commanded
        branch instead of accepting the lag and policing only the resulting
        speed spike.
        """

        if not closed:
            return super()._plan_execute(
                planner,
                name,
                pose_w,
                result,
                closed=closed,
                allow_hand_contact=allow_hand_contact,
            )
        result.phase_history.append(name)
        self._stage_label = name
        trajectory = planner.plan(
            self._arm_positions(),
            world_pose_to_base(pose_w, planner.base_pose),
            allow_hand_contact=allow_hand_contact,
        )
        result.planning_times_s[name] = trajectory.planning_time_s
        result.trajectory_waypoints[name] = len(trajectory.positions)
        contact_loaded_pick = name in {"lift", "retract"}
        tracking_tolerance = (
            0.100 if name == "lift" else 0.050 if contact_loaded_pick else 0.040
        )
        endpoint_tolerance = (
            0.065 if name == "lift" else 0.055 if contact_loaded_pick else 0.045
        )
        try:
            self._execute(
                trajectory,
                closed=True,
                max_step_rad=0.005,
                tracking_steps=140,
                tracking_tolerance_rad=tracking_tolerance,
                endpoint_steps=180,
                endpoint_tolerance_rad=endpoint_tolerance,
            )
        except RuntimeError as error:
            raise RuntimeError(f"{name}: {error}") from error
        result.arm_endpoints_rad[name] = list(self._arm_positions())
        return trajectory

    def _plan_execute_loaded_joint_anchor(
        self,
        planner: CuroboFixedBasePlanner,
        name: str,
        target: tuple[float, ...],
        result: XGRIDS2CuroboResult,
    ) -> None:
        """Plan to a calibrated carry branch, then track it conservatively."""

        result.phase_history.append(name)
        self._stage_label = name
        trajectory = planner.plan_joint(self._arm_positions(), target)
        result.planning_times_s[name] = trajectory.planning_time_s
        result.trajectory_waypoints[name] = len(trajectory.positions)
        try:
            self._execute(
                trajectory,
                closed=True,
                max_step_rad=0.005,
                tracking_steps=140,
                tracking_tolerance_rad=0.040,
                endpoint_steps=180,
                endpoint_tolerance_rad=0.045,
            )
        except RuntimeError as error:
            raise RuntimeError(f"{name}: {error}") from error
        result.arm_endpoints_rad[name] = list(self._arm_positions())

    def _close_and_attach(self, result: XGRIDS2CuroboResult) -> Pose:
        result.phase_history.append("close")
        self._stage_label = "close"
        target = self._arm_positions()
        world = get_world(self.env)
        for _ in range(180):
            self._step(self._action(target, closed=True))
            left_force = world.get_contact_force("plate", "left_fingertip")[0]
            right_force = world.get_contact_force("plate", "right_fingertip")[0]
            contact_force = torch.maximum(
                torch.linalg.vector_norm(left_force),
                torch.linalg.vector_norm(right_force),
            )
            if float(contact_force.item()) >= 1.0:
                # Thin curved plates can slip before both pads simultaneously
                # load.  The user-approved assist therefore latches on the
                # first *measured* fingertip contact, never on proximity alone.
                result.phase_history.append("physical_contact_verified")
                self._attach()
                result.phase_history.append("retention_enabled")
                self.gripper_close_command = self._gripper_target.clone()
                self._hold(12, closed=True)
                return self._held_pose()
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
            "plate": self.can.data.root_pos_w[0].detach().cpu().tolist(),
            "left_fingertip": self.robot.data.body_pos_w[0, left_index]
            .detach()
            .cpu()
            .tolist(),
            "right_fingertip": self.robot.data.body_pos_w[0, right_index]
            .detach()
            .cpu()
            .tolist(),
            "left_force": world.get_contact_force("plate", "left_fingertip")[0]
            .detach()
            .cpu()
            .tolist(),
            "right_force": world.get_contact_force("plate", "right_fingertip")[0]
            .detach()
            .cpu()
            .tolist(),
        }
        raise RuntimeError(
            "physical fingertip-to-plate contact was not established; "
            f"contact_diagnostics={diagnostics}"
        )

    def _carry_posture_radius(self) -> float:
        root_xy = self.robot.data.root_link_pos_w[0, :2]
        robot_radius = torch.max(
            torch.linalg.vector_norm(
                self.robot.data.body_pos_w[0, :, :2] - root_xy, dim=1
            )
        )
        angles = torch.arange(8, device=self.device) * (math.pi / 4.0)
        local_rim = torch.stack(
            (
                PLATE_RADIUS_M * torch.cos(angles),
                PLATE_RADIUS_M * torch.sin(angles),
                torch.zeros_like(angles),
            ),
            dim=1,
        )
        quaternion = self.can.data.root_quat_w[0].expand(8, -1)
        rim_world = self.can.data.root_pos_w[0] + math_utils.quat_apply(
            quaternion, local_rim
        )
        plate_radius = torch.max(
            torch.linalg.vector_norm(rim_world[:, :2] - root_xy, dim=1)
        )
        return float(torch.maximum(robot_radius, plate_radius).item())

    def _locked_transport_step(self, action: torch.Tensor) -> None:
        """Advance the base while explicitly locking the unloaded stowed arm.

        This state write is deliberately confined to base-retained traversal.
        It is not used for pick, CuRobo manipulation, regrasp, or placement.
        """

        _, _, terminated, truncated, _ = self.env.step(action)
        # _action() has already rebased continuous targets onto the articulation's
        # physical branch.  Write that exact branch with zero velocity so direct
        # floating-root motion cannot make PhysX re-solve the unloaded arm.
        position = self._last_arm_target.unsqueeze(0)
        velocity = torch.zeros_like(position)
        self.robot.write_joint_state_to_sim(
            position,
            velocity,
            joint_ids=self.arm_indices,
        )
        self._last_arm_position = position[0].clone()
        if bool(terminated[0].item() or truncated[0].item()):
            raise RuntimeError("environment terminated during locked plate transport")

    def _scripted_transport_arm_transition(
        self,
        target: tuple[float, ...],
        *,
        closed: bool,
        max_step_rad: float = 0.005,
    ) -> None:
        """Smoothly move the unloaded transport arm under the explicit lock."""

        current = np.asarray(self._arm_positions(), dtype=np.float64)
        desired = np.asarray(target, dtype=np.float64)
        delta = desired - current
        continuous = np.asarray((0, 2, 4, 6), dtype=np.int64)
        delta[continuous] = (delta[continuous] + math.pi) % (2.0 * math.pi) - math.pi
        count = max(1, int(math.ceil(float(np.max(np.abs(delta))) / max_step_rad)))
        for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
            waypoint = tuple(float(value) for value in current + fraction * delta)
            action = self._action(waypoint, closed=closed)
            self._locked_transport_step(action)

    def _traverse(self, result: XGRIDS2CuroboResult) -> None:
        result.phase_history.append("traverse")
        self._stage_label = "traverse"
        target = SINK_MANIPULATION_BASE_POSE
        self._set_base_hold(target)
        _, transport_yaw = self._robot_planar_state()
        route = planned_sink_route(robot_radius=PLATE_ROUTE_PLANNING_RADIUS_M)
        follower = WaypointFollower(
            route,
            final_yaw=target[3],
            max_linear_speed=min(CARRY_BASE_SPEED_M_S, 0.06),
            max_angular_speed=0.10,
            waypoint_tolerance=0.040,
            final_position_tolerance=0.020,
        )
        arm = self._arm_positions()
        reached = False
        for _ in range(3600):
            xy, yaw = self._robot_planar_state()
            command, reached = follower.command(xy, yaw)
            # The base is holonomic.  Keep the loaded arm in one world
            # orientation during translation instead of turning to face every
            # A* segment; only the small sink alignment is needed at the end.
            if float(np.linalg.norm(xy - np.asarray(target[:2]))) > 0.025:
                command[2] = float(
                    np.clip(
                        1.8 * wrap_angle(transport_yaw - yaw),
                        -0.10,
                        0.10,
                    )
                )
            action = self._action(
                arm, closed=False, base_command=tuple(command.tolist())
            )
            self._locked_transport_step(action)
            xy_after, _ = self._robot_planar_state()
            result.minimum_route_clearance_valid &= sink_base_pose_is_clear(
                tuple(float(value) for value in xy_after),
                robot_radius=PLATE_ROUTE_RADIUS_M,
            )
            if reached:
                break
        if not reached:
            raise RuntimeError(
                f"base follower stopped at route waypoint {follower.waypoint_index}"
            )
        self._hold(30, closed=False)
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
                "base did not settle at the sink manipulation stage; "
                f"position_error={result.final_base_position_error_m:.4f} m, "
                f"yaw_error={result.final_base_yaw_error_rad:.4f} rad"
            )

    def _capture_result_state(self, result: XGRIDS2PlateSinkResult) -> None:
        super()._capture_result_state(result)
        normal = math_utils.quat_apply(
            self.can.data.root_quat_w,
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device),
        )
        result.plate_upright_cosine = float(normal[0, 2].item())

    def run(self, *, stage: str = "full") -> XGRIDS2PlateSinkResult:
        if stage not in {"pick", "carry", "place", "full"}:
            raise ValueError("stage must be one of: pick, carry, place, full")
        result = XGRIDS2PlateSinkResult(requested_stage=stage)
        result.place_target = list(SINK_CENTER_XY)
        pick_planner = None
        place_planner = None
        try:
            self.grasp_joint.GetPrim().SetActive(True)
            self.grasp_joint.GetJointEnabledAttr().Set(False)
            self.reset_robot_to_default(settle_steps=60)
            self._align_pick_base(result)

            plate = self.can.data.root_pos_w[0]
            plate_local = plate - self.env.scene.env_origins[0]
            result.pick_position = plate_local.detach().cpu().tolist()
            pick = tuple(float(value) for value in plate.tolist())
            dx, dy = PICK_APPROACH_DIRECTION
            rim_radius = PLATE_RADIUS_M - PLATE_PICK_CONTACT_INSET_M
            rim = (
                pick[0] - rim_radius * dx,
                pick[1] - rim_radius * dy,
                # Physical tracking settled about 15 mm below the nominal
                # endpoint in the first contact trials.  Compensate here so
                # the converged fingertip pair straddles the plate mid-plane.
                pick[2] + PLATE_GRASP_TCP_Z_OFFSET_M,
            )
            pregrasp = (
                rim[0] - PLATE_PREGRASP_DISTANCE_M * dx,
                rim[1] - PLATE_PREGRASP_DISTANCE_M * dy,
                rim[2] + 0.05,
            )
            base_xy, base_yaw = self._robot_planar_state()
            base_z = float(
                (
                    self.robot.data.root_link_pos_w[0, 2]
                    - self.env.scene.env_origins[0, 2]
                ).item()
            )
            pick_base_pose = (
                float(base_xy[0]),
                float(base_xy[1]),
                base_z,
                base_yaw,
            )
            pick_planner = CuroboFixedBasePlanner(
                pick_base_pose,
                num_seeds=self.num_seeds,
                collision_boxes=PLANNER_COLLISION_BOXES,
            )
            self._plan_execute_joint(
                pick_planner,
                "pick_branch_anchor",
                PICK_BRANCH_ANCHOR_JOINTS,
                result,
                closed=False,
            )
            self._plan_execute(
                pick_planner,
                "pregrasp",
                Pose(pregrasp, PLATE_RIM_GRASP_QUAT_WXYZ),
                result,
                closed=False,
            )
            self._plan_execute(
                pick_planner,
                "approach",
                Pose(rim, PLATE_RIM_GRASP_QUAT_WXYZ),
                result,
                closed=False,
                allow_hand_contact=True,
            )
            tcp_pos, _ = self._current_tcp_pose()
            reach_error = math.dist(
                tcp_pos[0].detach().cpu().tolist(), rim
            )
            if reach_error > 0.035:
                raise RuntimeError(
                    f"CuRobo approach stopped {reach_error:.4f} m from the plate rim"
                )
            held_pose = self._close_and_attach(result)
            self._plan_execute(
                pick_planner,
                "lift",
                Pose(
                    (rim[0], rim[1], rim[2] + PLATE_LIFT_M),
                    PLATE_RIM_GRASP_QUAT_WXYZ,
                ),
                result,
                closed=True,
                allow_hand_contact=True,
            )
            self._hold(20, closed=True)
            lifted = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
            if float(lifted[2].item()) < DINING_TABLE.surface_z + 0.10:
                raise RuntimeError("plate was not lifted clear of the dining table")
            if stage == "pick":
                return self._complete_stage(result, "pick")

            pick_planner.set_held_object(
                held_pose, object_spheres=PLATE_HELD_SPHERES
            )
            self._plan_execute(
                pick_planner,
                "retract",
                Pose(
                    (
                        rim[0] - PLATE_RETRACT_M * dx,
                        rim[1] - PLATE_RETRACT_M * dy,
                        rim[2] + PLATE_LIFT_M,
                    ),
                    PLATE_RIM_GRASP_QUAT_WXYZ,
                ),
                result,
                closed=True,
            )
            plate_position = tuple(
                float(value) for value in self.can.data.root_pos_w[0].tolist()
            )
            plate_quaternion = tuple(
                float(value) for value in self.can.data.root_quat_w[0].tolist()
            )
            carry_quaternion = _vertical_plate_quaternion(base_yaw)
            for index, fraction in enumerate((0.2, 0.4), start=1):
                object_quaternion = _slerp_quaternion(
                    plate_quaternion, carry_quaternion, fraction
                )
                self._plan_execute(
                    pick_planner,
                    f"carry_reorient_{index}",
                    self._tool_pose_for_object(
                        plate_position, object_quaternion, held_pose
                    ),
                    result,
                    closed=True,
                )
            bx, by, bz, yaw = pick_base_pose
            cosine, sine = math.cos(yaw), math.sin(yaw)

            # At full reach, rotation beyond roughly 36 degrees has no IK
            # solution.  Bring the partially tilted disk inward while its
            # projected radius still clears the base, then finish the roll.
            intermediate_center = (
                bx + cosine * 0.42,
                by + sine * 0.42,
                bz + 0.78,
            )
            intermediate_quaternion = _slerp_quaternion(
                plate_quaternion, carry_quaternion, 0.4
            )
            self._plan_execute(
                pick_planner,
                "carry_mid_tuck",
                self._tool_pose_for_object(
                    intermediate_center, intermediate_quaternion, held_pose
                ),
                result,
                closed=True,
            )
            # Finish the roll on measured interior-branch joint anchors.  The
            # held-object collision model remains active in CuRobo's joint
            # planner, while avoiding a nondeterministic equivalent solution
            # beside joint 3's +2*pi articulation boundary.
            self._plan_execute_loaded_joint_anchor(
                pick_planner,
                "carry_reorient_3",
                PLATE_CARRY_REORIENT_3_JOINTS,
                result,
            )
            self._plan_execute_loaded_joint_anchor(
                pick_planner,
                "carry_reorient_4",
                PLATE_CARRY_REORIENT_4_JOINTS,
                result,
            )
            self._plan_execute_loaded_joint_anchor(
                pick_planner,
                "carry_tuck",
                PLATE_CARRY_TUCK_JOINTS,
                result,
            )
            result.carry_posture_radius_m = self._carry_posture_radius()
            if result.carry_posture_radius_m > (
                PLATE_MAX_CARRY_RADIUS_M + PLATE_CARRY_RADIUS_EPSILON_M
            ):
                raise RuntimeError(
                    "plate carry posture exceeds navigation envelope; "
                    f"radius={result.carry_posture_radius_m:.4f} m"
                )
            # Establish that the loaded arm can physically hold the planned
            # posture before coupling it to base motion.  Besides preventing
            # an unsettled trajectory endpoint from contaminating traversal,
            # this makes any payload-drive problem fail in the correct phase.
            result.phase_history.append("carry_loaded_settle")
            self._stage_label = "carry_loaded_settle"
            self._hold(90, closed=True)
            result.phase_history.append("retention_transferred_to_base")
            self._stage_label = "retention_transferred_to_base"
            self._transfer_retention_to_base()
            self._hold(90, closed=False)
            pick_planner.set_held_object(None)
            result.phase_history.append("transport_arm_stow")
            self._stage_label = "transport_arm_stow"
            carry_arm = self._arm_positions()
            result.planning_times_s["transport_arm_stow"] = 0.0
            result.trajectory_waypoints["transport_arm_stow"] = 1
            self._scripted_transport_arm_transition(
                self._default_arm_positions(),
                closed=False,
                max_step_rad=0.005,
            )
            result.arm_endpoints_rad["transport_arm_stow"] = list(
                self._arm_positions()
            )
            result.phase_history.append("transport_arm_state_lock_enabled")
            self._traverse(result)
            if not result.minimum_route_clearance_valid:
                raise RuntimeError("base left the plate-inflated free-space route")
            if stage == "carry":
                return self._complete_stage(result, "carry")

            self._stage_label = "transport_arm_restore"
            result.phase_history.append("transport_arm_restore")
            result.planning_times_s["transport_arm_restore"] = 0.0
            result.trajectory_waypoints["transport_arm_restore"] = 1
            self._scripted_transport_arm_transition(
                carry_arm,
                closed=False,
                max_step_rad=0.005,
            )
            result.arm_endpoints_rad["transport_arm_restore"] = list(
                self._arm_positions()
            )
            result.phase_history.append("retention_returned_to_gripper")
            self._stage_label = "retention_returned_to_gripper"
            self._hold(90, closed=True)
            self._transfer_retention_to_flange()
            self._hold(30, closed=True)
            held_pose = self._held_pose()

            place_planner = CuroboFixedBasePlanner(
                SINK_MANIPULATION_BASE_POSE,
                num_seeds=self.num_seeds,
                collision_boxes=PLANNER_COLLISION_BOXES,
            )
            place_planner.set_held_object(
                held_pose, object_spheres=PLATE_HELD_SPHERES
            )
            current_position = tuple(
                float(value) for value in self.can.data.root_pos_w[0].tolist()
            )
            current_quaternion = tuple(
                float(value) for value in self.can.data.root_quat_w[0].tolist()
            )
            horizontal_quaternion = (1.0, 0.0, 0.0, 0.0)
            for index, fraction in enumerate((1.0 / 3.0, 2.0 / 3.0, 1.0), start=1):
                self._plan_execute(
                    place_planner,
                    f"place_reorient_{index}",
                    self._tool_pose_for_object(
                        current_position,
                        _slerp_quaternion(
                            current_quaternion, horizontal_quaternion, fraction
                        ),
                        held_pose,
                    ),
                    result,
                    closed=True,
                )
            preplace_center = (
                SINK_CENTER_XY[0],
                SINK_CENTER_XY[1],
                SINK_PLACE_SETTLED_Z + 0.24,
            )
            self._plan_execute(
                place_planner,
                "preplace",
                self._tool_pose_for_object(
                    preplace_center, horizontal_quaternion, held_pose
                ),
                result,
                closed=True,
            )
            approach_trajectory = self._plan_execute(
                place_planner,
                "place_approach",
                self._tool_pose_for_object(
                    (
                        SINK_CENTER_XY[0],
                        SINK_CENTER_XY[1],
                        SINK_PLACE_SETTLED_Z + 0.16,
                    ),
                    horizontal_quaternion,
                    held_pose,
                ),
                result,
                closed=True,
            )
            place_planner.set_held_object(None)
            result.phase_history.append("placement_retention_transferred_to_base")
            self._stage_label = "placement_retention_transferred_to_base"
            self._transfer_retention_to_base()
            self._hold(60, closed=False)
            self._guided_lower_retained_plate(
                SINK_PLACE_SETTLED_Z + SINK_RELEASE_CLEARANCE_M,
                result,
            )
            result.phase_history.append("release")
            self._stage_label = "release"
            self.grasp_joint.GetJointEnabledAttr().Set(False)
            self._hold(90, closed=False)
            result.phase_history.append("retreat")
            self._stage_label = "retreat"
            retreat = JointTrajectory(
                joint_names=approach_trajectory.joint_names,
                positions=tuple(reversed(approach_trajectory.positions)),
                planning_time_s=0.0,
            )
            result.planning_times_s["retreat"] = 0.0
            result.trajectory_waypoints["retreat"] = len(retreat.positions)
            self._execute(retreat, closed=False)
            self._hold(120, closed=False)

            final = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
            result.place_target_error_m = math.dist(
                (float(final[0]), float(final[1])), SINK_CENTER_XY
            )
            velocity_ok = (
                float(torch.linalg.vector_norm(self.can.data.root_lin_vel_w[0]))
                < 0.05
                and float(torch.linalg.vector_norm(self.can.data.root_ang_vel_w[0]))
                < 0.20
            )
            self._capture_result_state(result)
            settled = (
                sink_contains_plate(
                    (float(final[0]), float(final[1])), margin=0.005
                )
                and abs(float(final[2]) - SINK_PLACE_SETTLED_Z) < 0.03
                and result.plate_upright_cosine >= math.cos(math.radians(20.0))
                and velocity_ok
            )
            if not settled:
                raise RuntimeError(
                    "released plate did not settle inside the sink; "
                    f"final={final.detach().cpu().tolist()}, "
                    f"xy_error={result.place_target_error_m:.4f} m, "
                    f"upright_cosine={result.plate_upright_cosine:.4f}"
                )
            return self._complete_stage(
                result, "place" if stage == "place" else "full"
            )
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
