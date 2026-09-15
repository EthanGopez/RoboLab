"""Common execution layer for the configurable XGRIDS2 benchmark pack."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch
from isaaclab.utils import math as math_utils

from robolab.core.world.world_state import get_world
from robolab.tasks.mobile_manipulation.curobo_fixed_base_planner import (
    CuroboFixedBasePlanner,
    world_pose_to_base,
)
from robolab.tasks.mobile_manipulation.fixed_base_manipulation import (
    JointTrajectory,
    Pose,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_can_transfer_task import (
    MANIPULATION_COLLISION_SEGMENTS,
    manipulation_base_pose_is_clear,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_curobo_oracle import (
    ARM_LIMIT_EPSILON_RAD,
    ARM_POSITION_LIMITS_RAD,
    BASE_POSITION_TOLERANCE_M,
    BASE_YAW_TOLERANCE_RAD,
    CARRY_TUCK_TCP_BASE,
    CONTINUOUS_ARM_INDICES,
    MAX_ARM_SPEED_QUALITY_RAD_S,
    MAX_ARM_STEP_QUALITY_RAD,
    PICK_BRANCH_ANCHOR_JOINTS,
    XGRIDS2CuroboOracle,
    XGRIDS2CuroboResult,
    _slerp_quaternion,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
    planned_route_between,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_navigation_oracle import (
    WaypointFollower,
    wrap_angle,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenario_task import (
    scenario_grasp_joint_name,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import (
    XGRIDS2ScenarioSpec,
    XGRIDS2TransferLegSpec,
    get_xgrids2_scenario,
)


@dataclass(frozen=True)
class _PlannerBox:
    name: str
    center_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    yaw: float = 0.0


@dataclass
class XGRIDS2ScenarioResult:
    scenario: str
    requested_stage: str = "full"
    completed_stage: str = ""
    passed: bool = False
    failure_reason: str = ""
    phase_history: list[str] = field(default_factory=list)
    leg_results: list[dict] = field(default_factory=list)
    maximum_arm_step_rad: float = 0.0
    maximum_arm_speed_rad_s: float = 0.0
    maximum_raw_physx_arm_speed_rad_s: float = 0.0
    motion_diagnostics: dict = field(default_factory=dict)
    minimum_route_clearance_valid: bool = True
    distractor_max_displacement_m: float = 0.0
    pouring_metrics: dict[str, float | bool | str] = field(default_factory=dict)
    released_objects: list[str] = field(default_factory=list)
    retention_active_end: bool = False

    def as_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "passed": self.passed,
            "requested_stage": self.requested_stage,
            "completed_stage": self.completed_stage,
            "failure_reason": self.failure_reason,
            "phase_history": self.phase_history,
            "leg_results": self.leg_results,
            "maximum_arm_step_rad": self.maximum_arm_step_rad,
            "maximum_arm_speed_rad_s": self.maximum_arm_speed_rad_s,
            "maximum_raw_physx_arm_speed_rad_s": (
                self.maximum_raw_physx_arm_speed_rad_s
            ),
            "motion_diagnostics": self.motion_diagnostics,
            "minimum_route_clearance_valid": self.minimum_route_clearance_valid,
            "distractor_max_displacement_m": self.distractor_max_displacement_m,
            "pouring_metrics": self.pouring_metrics,
            "released_objects": self.released_objects,
            "retention_active_end": self.retention_active_end,
            "execution_mode": "curobo_time_parameterized_continuous_position_tracking",
            "retention_mode": "measured_fingertip_contact_then_temporary_fixed_joint",
        }


class XGRIDS2ConfigurableTransferOracle(XGRIDS2CuroboOracle):
    """Profile-driven single-object transfer or pose-level pouring executor."""

    def __init__(
        self,
        env,
        scenario: XGRIDS2ScenarioSpec,
        leg: XGRIDS2TransferLegSpec,
        *,
        num_seeds: int = 12,
    ) -> None:
        super().__init__(
            env,
            num_seeds=num_seeds,
            object_name=leg.object_name,
            grasp_joint_name=scenario_grasp_joint_name(leg.object_name),
        )
        self.scenario = scenario
        self.leg = leg
        self.object_spec = scenario.object(leg.object_name)
        self._distractor_initial_positions = {
            item.name: env.scene[item.name].data.root_pos_w[0].clone()
            for item in scenario.objects
            if item.distractor
        }

    def _align_pick_base(self, result: XGRIDS2CuroboResult) -> None:
        """Align to the leg-owned source staging pose before fixed-base IK."""

        target = (
            self.leg.route.source_base_pose_xyzyaw
            if self.leg.route is not None
            else (1.7656, -1.7639, 0.13335, 2.29876)
        )
        self._set_base_hold(target)
        xy, yaw = self._robot_planar_state()
        if (
            float(np.linalg.norm(xy - np.asarray(target[:2]))) <= 0.003
            and abs(wrap_angle(yaw - target[3])) <= 0.008
        ):
            return
        result.phase_history.append("align_pick_base")
        self._stage_label = "align_pick_base"
        follower = WaypointFollower(
            np.asarray((xy, target[:2]), dtype=np.float64),
            final_yaw=target[3],
            max_linear_speed=0.04,
            max_angular_speed=0.20,
            waypoint_tolerance=0.005,
            final_position_tolerance=0.006,
            final_yaw_tolerance=0.012,
        )
        arm = self._arm_target_positions()
        for _ in range(900):
            xy, yaw = self._robot_planar_state()
            command, reached = follower.command(xy, yaw)
            self._step(
                self._action(
                    arm, closed=False, base_command=tuple(command.tolist())
                )
            )
            if reached:
                self._hold(15, closed=False)
                return
        raise RuntimeError("base did not reach the configured source staging pose")

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
        """Track loaded configurable objects without accumulating drive lag."""

        if not closed:
            return super()._plan_execute(
                planner,
                name,
                pose_w,
                result,
                closed=False,
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
        try:
            self._execute(
                trajectory,
                closed=True,
                max_step_rad=0.004,
                tracking_steps=120,
                tracking_tolerance_rad=0.050,
                endpoint_steps=160,
                endpoint_tolerance_rad=0.060,
            )
        except RuntimeError as error:
            raise RuntimeError(f"{name}: {error}") from error
        result.arm_endpoints_rad[name] = list(self._arm_positions())
        return trajectory

    def _planner_boxes(self, *, include_destination: bool) -> tuple[_PlannerBox, ...]:
        boxes: list[_PlannerBox] = []
        for support in self.scenario.supports:
            if support.kind in {"bowl", "sink"}:
                # Receivers are approached through their opening; representing
                # the entire bounds as a solid CuRobo cuboid would make valid
                # insertion impossible. Runtime physics still owns the walls.
                continue
            boxes.append(
                _PlannerBox(
                    support.name,
                    support.center_xyz,
                    support.size_xyz,
                    support.yaw_rad,
                )
            )
        for item in self.scenario.objects:
            if not item.distractor:
                continue
            position = self.env.scene[item.name].data.root_pos_w[0]
            local = position - self.env.scene.env_origins[0]
            boxes.append(
                _PlannerBox(
                    item.name,
                    tuple(float(value) for value in local.tolist()),
                    tuple(value + 0.012 for value in item.dimensions_xyz),
                )
            )
        return tuple(boxes)

    def _close_and_attach(self, result: XGRIDS2CuroboResult) -> Pose:
        result.phase_history.append("close")
        self._stage_label = "close"
        target = self._arm_positions()
        world = get_world(self.env)
        mode = self.leg.grasp.contact_mode
        threshold = self.leg.grasp.contact_force_n
        for _ in range(180):
            self._step(self._action(target, closed=True))
            left = torch.linalg.vector_norm(
                world.get_contact_force(self.object_name, "left_fingertip")[0]
            )
            right = torch.linalg.vector_norm(
                world.get_contact_force(self.object_name, "right_fingertip")[0]
            )
            contacted = (
                min(float(left.item()), float(right.item())) >= threshold
                if mode == "bilateral"
                else max(float(left.item()), float(right.item())) >= threshold
            )
            if contacted:
                result.phase_history.append("physical_contact_verified")
                self._attach()
                result.phase_history.append("retention_enabled")
                self.gripper_close_command = self._gripper_target.clone()
                self._hold(12, closed=True)
                return self._held_pose()
        raise RuntimeError(
            f"{mode} fingertip contact was not established for {self.object_name}"
        )

    def _pick_targets(
        self,
        quaternion: tuple[float, float, float, float] | None = None,
    ) -> tuple[Pose, Pose, tuple[float, float, float]]:
        position = tuple(float(value) for value in self.can.data.root_pos_w[0].tolist())
        grasp = self.leg.grasp
        if grasp.kind == "top":
            pregrasp = (
                position[0],
                position[1],
                position[2] + grasp.pregrasp_height_m,
            )
            approach = (position[0], position[1], position[2] + 0.040)
        else:
            dx, dy = grasp.approach_direction_xy
            pregrasp = (
                position[0] - grasp.pregrasp_distance_m * dx,
                position[1] - grasp.pregrasp_distance_m * dy,
                position[2] + grasp.pregrasp_height_m,
            )
            approach = position
        grasp_quaternion = quaternion or grasp.grasp_quaternion_wxyz
        return Pose(pregrasp, grasp_quaternion), Pose(approach, grasp_quaternion), position

    def _top_grasp_pregrasp(
        self,
        planner: CuroboFixedBasePlanner,
        result: XGRIDS2CuroboResult,
    ) -> tuple[
        Pose,
        Pose,
        tuple[float, float, float],
        tuple[float, float, float, float],
        JointTrajectory,
    ]:
        """Choose the first reachable free yaw while preserving vertical approach."""

        configured = self.leg.grasp.grasp_quaternion_wxyz
        candidates = [configured]
        exact_top_candidates = [configured]
        for degrees in (0.0, 22.5, 67.5, 90.0, 112.5, 135.0, 157.5):
            angle = math.radians(degrees)
            candidate = (0.0, math.cos(angle), math.sin(angle), 0.0)
            if candidate not in candidates:
                candidates.append(candidate)
                exact_top_candidates.append(candidate)
        side_reference = (-0.2520023, -0.2520023, 0.6606776, 0.6606776)
        for top_candidate in exact_top_candidates:
            for downward_fraction in (0.50, 0.65, 0.80):
                candidates.append(
                    _slerp_quaternion(
                        side_reference, top_candidate, downward_fraction
                    )
                )
        result.phase_history.append("pregrasp")
        self._stage_label = "pregrasp"
        last_error = None
        for quaternion in candidates:
            pregrasp, approach, position = self._pick_targets(quaternion)
            try:
                trajectory = planner.plan(
                    self._arm_positions(),
                    world_pose_to_base(pregrasp, planner.base_pose),
                )
                approach_trajectory = planner.plan(
                    trajectory.positions[-1],
                    world_pose_to_base(approach, planner.base_pose),
                    allow_hand_contact=True,
                )
            except RuntimeError as error:
                last_error = error
                continue
            if not self._trajectories_stay_on_physical_branch(
                trajectory, approach_trajectory
            ):
                last_error = RuntimeError(
                    "candidate trajectory crosses a continuous-joint hardware limit"
                )
                continue
            result.planning_times_s["pregrasp"] = trajectory.planning_time_s
            result.trajectory_waypoints["pregrasp"] = len(trajectory.positions)
            self._execute(trajectory, closed=False)
            result.arm_endpoints_rad["pregrasp"] = list(self._arm_positions())
            return pregrasp, approach, position, quaternion, approach_trajectory
        raise RuntimeError(f"no reachable top-down grasp yaw: {last_error}")

    def _trajectories_stay_on_physical_branch(self, *trajectories) -> bool:
        current = np.asarray(self._arm_positions(), dtype=np.float64)
        for trajectory in trajectories:
            for waypoint in trajectory.positions:
                desired = np.asarray(waypoint, dtype=np.float64)
                for index, (lower, upper) in enumerate(ARM_POSITION_LIMITS_RAD):
                    if index in CONTINUOUS_ARM_INDICES:
                        candidates = desired[index] + 2.0 * math.pi * np.arange(-2, 3)
                        candidates = candidates[
                            (candidates >= lower + ARM_LIMIT_EPSILON_RAD)
                            & (candidates <= upper - ARM_LIMIT_EPSILON_RAD)
                        ]
                        if not len(candidates):
                            return False
                        chosen = candidates[np.argmin(np.abs(candidates - current[index]))]
                        # Preserve enough branch margin for the measured
                        # endpoint tracking tolerance. A nominal change just
                        # under pi can become non-executable when the physical
                        # joint trails the planned endpoint by a few hundredths
                        # of a radian.
                        if (
                            abs(chosen - current[index])
                            > math.pi - self.leg.grasp.continuous_branch_margin_rad
                        ):
                            return False
                        desired[index] = chosen
                    elif not lower <= desired[index] <= upper:
                        return False
                current = desired
        return True

    def _rebase_trajectory_to_physical_branch(
        self,
        trajectory: JointTrajectory,
        start: tuple[float, ...] | np.ndarray,
    ) -> tuple[JointTrajectory, tuple[float, ...]]:
        """Materialize CuRobo's circular angles on one executable Kinova branch.

        CuRobo may express the same continuous-joint pose with an angle that
        differs by 2*pi. Merely validating an equivalent angle is insufficient:
        the raw value can make the rate-limited executor choose the other
        finite URDF branch after normal tracking lag. Store the chosen
        equivalent value in every waypoint so preflight and execution consume
        identical joint targets.
        """

        current = np.asarray(start, dtype=np.float64).copy()
        rebased: list[tuple[float, ...]] = []
        for waypoint in trajectory.positions:
            desired = np.asarray(waypoint, dtype=np.float64).copy()
            for index, (lower, upper) in enumerate(ARM_POSITION_LIMITS_RAD):
                if index in CONTINUOUS_ARM_INDICES:
                    candidates = desired[index] + 2.0 * math.pi * np.arange(-2, 3)
                    candidates = candidates[
                        (candidates >= lower + ARM_LIMIT_EPSILON_RAD)
                        & (candidates <= upper - ARM_LIMIT_EPSILON_RAD)
                    ]
                    if not len(candidates):
                        raise RuntimeError(
                            f"joint_{index + 1} has no limit-valid circular target"
                        )
                    chosen = candidates[np.argmin(np.abs(candidates - current[index]))]
                    if abs(chosen - current[index]) > math.pi:
                        raise RuntimeError(
                            f"joint_{index + 1} circular branch would cross its physical limit"
                        )
                    desired[index] = chosen
                else:
                    limit_overflow = max(
                        lower - desired[index], desired[index] - upper, 0.0
                    )
                    if limit_overflow > 0.001:
                        raise RuntimeError(
                            f"joint_{index + 1} target exceeds its physical limit"
                        )
                    if limit_overflow > 0.0:
                        # CuRobo can return a finite-joint value a few 1e-5 rad
                        # outside its URDF limit. Treat only sub-milliradian
                        # overflow as optimizer noise and command the existing
                        # interior safety margin; material violations still fail.
                        desired[index] = float(
                            np.clip(
                                desired[index],
                                lower + ARM_LIMIT_EPSILON_RAD,
                                upper - ARM_LIMIT_EPSILON_RAD,
                            )
                        )
            rebased.append(tuple(float(value) for value in desired))
            current = desired
        result = JointTrajectory(
            trajectory.joint_names,
            tuple(rebased),
            trajectory.planning_time_s,
            sample_dt_s=trajectory.sample_dt_s,
        )
        return result, rebased[-1]

    def _traverse_configured(self, result: XGRIDS2CuroboResult) -> None:
        route_spec = self.leg.route
        if route_spec is None:
            return
        target = route_spec.destination_base_pose_xyzyaw
        arm = self._arm_target_positions()
        for stage_index, staging_pose in enumerate(self.leg.staging_poses):
            result.phase_history.append(f"departure_stage_{stage_index + 1}")
            self._stage_label = f"departure_stage_{stage_index + 1}"
            self._set_base_hold(staging_pose)
            start_xy, _ = self._robot_planar_state()
            follower = WaypointFollower(
                np.asarray((start_xy, staging_pose[:2]), dtype=np.float64),
                final_yaw=staging_pose[3],
                max_linear_speed=min(route_spec.max_linear_speed_m_s, 0.04),
                max_angular_speed=min(route_spec.max_angular_speed_rad_s, 0.20),
                waypoint_tolerance=0.012,
                final_position_tolerance=0.012,
                final_yaw_tolerance=0.020,
            )
            reached_stage = False
            for _ in range(900):
                xy, yaw = self._robot_planar_state()
                command, reached_stage = follower.command(xy, yaw)
                self._step(
                    self._action(
                        arm,
                        closed=True,
                        base_command=tuple(command.tolist()),
                    )
                )
                result.minimum_route_clearance_valid &= (
                    manipulation_base_pose_is_clear(
                        tuple(float(value) for value in xy),
                        robot_radius=route_spec.runtime_radius_m,
                    )
                )
                if reached_stage:
                    break
            if not reached_stage:
                raise RuntimeError(
                    f"base did not reach departure stage {stage_index + 1}"
                )
            # Let the loaded arm and temporary retention joint dissipate the
            # staging-stop reaction before changing the chassis direction.
            # The cube trial's only quality violation occurred on the first
            # post-stage route tick when this settle was too short.
            self._hold(120, closed=True)
        result.phase_history.append("traverse")
        self._stage_label = "traverse"
        self._set_base_hold(target)
        start_xy, _ = self._robot_planar_state()
        route = planned_route_between(
            tuple(float(value) for value in start_xy),
            route_spec.approach_xy,
            robot_radius=route_spec.planning_radius_m,
            collision_segments=MANIPULATION_COLLISION_SEGMENTS,
        )
        route = np.vstack((route, np.asarray(target[:2], dtype=np.float64)))
        follower = WaypointFollower(
            route,
            final_yaw=target[3],
            max_linear_speed=route_spec.max_linear_speed_m_s,
            max_angular_speed=route_spec.max_angular_speed_rad_s,
            waypoint_tolerance=0.020,
            final_position_tolerance=0.012,
        )
        reached = False
        previous_command = np.zeros(3, dtype=np.float64)
        for _ in range(4800):
            xy, yaw = self._robot_planar_state()
            desired_command, reached = follower.command(xy, yaw)
            # Avoid injecting an instantaneous chassis acceleration into the
            # arm/object retention chain at route departure.  The motion gate
            # remains unchanged; this makes the commanded base motion satisfy
            # it instead of tolerating a reaction spike.
            command = previous_command + np.clip(
                desired_command - previous_command,
                (-0.0005, -0.0005, -0.0025),
                (0.0005, 0.0005, 0.0025),
            )
            previous_command = command
            self._step(
                self._action(arm, closed=True, base_command=tuple(command.tolist()))
            )
            xy_after, _ = self._robot_planar_state()
            result.minimum_route_clearance_valid &= manipulation_base_pose_is_clear(
                tuple(float(value) for value in xy_after),
                robot_radius=route_spec.runtime_radius_m,
            )
            if reached:
                break
        if not reached:
            raise RuntimeError(
                f"base follower stopped at route waypoint {follower.waypoint_index}"
            )
        # A loaded omnidirectional base can acquire a small heading transient
        # when the route follower stops.  Close that residual with a strict,
        # low-speed physical alignment instead of assuming a fixed-duration
        # zero-command hold will settle at the manipulation pose.
        result.phase_history.append("align_destination_base")
        self._stage_label = "align_destination_base"
        xy, _ = self._robot_planar_state()
        aligner = WaypointFollower(
            np.asarray((xy, target[:2]), dtype=np.float64),
            final_yaw=target[3],
            max_linear_speed=0.04,
            max_angular_speed=0.20,
            waypoint_tolerance=0.005,
            final_position_tolerance=0.006,
            final_yaw_tolerance=0.012,
        )
        aligned = False
        stable_frames = 0
        for _ in range(1200):
            xy, yaw = self._robot_planar_state()
            command, aligned = aligner.command(xy, yaw)
            self._step(
                self._action(
                    arm,
                    closed=True,
                    base_command=tuple(command.tolist()),
                )
            )
            position_error = float(
                np.linalg.norm(xy - np.asarray(target[:2]))
            )
            yaw_error = abs(wrap_angle(yaw - target[3]))
            if (
                aligned
                and position_error <= 0.006
                and yaw_error <= 0.012
            ):
                stable_frames += 1
            else:
                stable_frames = 0
            if stable_frames >= 20:
                break
        if stable_frames < 20:
            raise RuntimeError("base did not align to the destination manipulation pose")
        xy, yaw = self._robot_planar_state()
        position_error = float(np.linalg.norm(xy - np.asarray(target[:2])))
        yaw_error = abs(wrap_angle(yaw - target[3]))
        result.final_base_position_error_m = position_error
        result.final_base_yaw_error_rad = yaw_error
        if position_error > BASE_POSITION_TOLERANCE_M or yaw_error > BASE_YAW_TOLERANCE_RAD:
            raise RuntimeError(
                "base did not settle at the configured manipulation stage; "
                f"position_error={position_error:.4f}, yaw_error={yaw_error:.4f}"
            )

    def navigate_empty_to_source(
        self,
        validated_tuck_joints: tuple[float, ...] | None = None,
        validated_return_segments: tuple[JointTrajectory, ...] = (),
    ) -> None:
        route_spec = self.leg.route
        if route_spec is None:
            return
        self._stage_label = "return_to_source"
        destination_base = route_spec.destination_base_pose_xyzyaw
        empty_planner = CuroboFixedBasePlanner(
            destination_base,
            num_seeds=self.num_seeds,
            collision_boxes=self._planner_boxes(include_destination=True),
        )
        start_joints = self._arm_positions()
        if validated_return_segments:
            for index, segment in enumerate(validated_return_segments, start=1):
                self._stage_label = f"return_tuck_replay_{index}"
                self._execute(
                    segment,
                    closed=False,
                    max_step_rad=0.004,
                    tracking_steps=160,
                    tracking_tolerance_rad=0.050,
                    endpoint_steps=160,
                    endpoint_tolerance_rad=0.060,
                    minimum_tracking_steps=2,
                )
            tuck_trajectory = None
        elif validated_tuck_joints is not None:
            # Reuse the exact compact joint posture that already transported
            # the first object safely. This avoids solving a new Cartesian IK
            # branch from the post-place wrist orientation.
            tuck_trajectory = empty_planner.plan_joint(
                start_joints, validated_tuck_joints
            )
            tuck_trajectory, _ = self._rebase_trajectory_to_physical_branch(
                tuck_trajectory, start_joints
            )
        else:
            tuck_x, tuck_y, tuck_z = CARRY_TUCK_TCP_BASE
            bx, by, bz, base_yaw = destination_base
            cosine, sine = math.cos(base_yaw), math.sin(base_yaw)
            tcp_position = (
                bx + cosine * tuck_x - sine * tuck_y,
                by + sine * tuck_x + cosine * tuck_y,
                bz + tuck_z,
            )
            _, tcp_quaternion = self._current_tcp_pose()
            tuck_trajectory = empty_planner.plan(
                start_joints,
                world_pose_to_base(
                    Pose(
                        tcp_position,
                        tuple(float(value) for value in tcp_quaternion[0].tolist()),
                    ),
                    empty_planner.base_pose,
                ),
            )
        if tuck_trajectory is not None:
            self._stage_label = "return_tuck"
            self._execute(
                tuck_trajectory,
                closed=False,
                max_step_rad=0.004,
                tracking_steps=160,
                tracking_tolerance_rad=0.050,
                endpoint_steps=160,
                endpoint_tolerance_rad=0.060,
                minimum_tracking_steps=2,
            )
        target = route_spec.source_base_pose_xyzyaw
        self._stage_label = "return_traverse"
        self._set_base_hold(target)
        start_xy, _ = self._robot_planar_state()
        route = planned_route_between(
            tuple(float(value) for value in start_xy),
            target[:2],
            robot_radius=0.34,
            collision_segments=MANIPULATION_COLLISION_SEGMENTS,
        )
        follower = WaypointFollower(
            route,
            final_yaw=target[3],
            max_linear_speed=0.08,
            max_angular_speed=0.35,
            waypoint_tolerance=0.04,
            final_position_tolerance=0.02,
        )
        arm = self._arm_target_positions()
        for _ in range(3200):
            xy, yaw = self._robot_planar_state()
            command, reached = follower.command(xy, yaw)
            self._step(
                self._action(arm, closed=False, base_command=tuple(command.tolist()))
            )
            if reached:
                self._stage_label = "return_settle"
                # Keep the collision-safe empty tuck reached above.  `default=True`
                # would issue a discontinuous reset-pose command at the end of the
                # route and was the source of the sequential task's arm-step spike.
                self._hold(30, closed=False)
                return
        raise RuntimeError("empty return traversal did not reach the dining stage")

    def _place_and_validate(
        self,
        planner: CuroboFixedBasePlanner,
        held_pose: Pose,
        result: XGRIDS2CuroboResult,
    ) -> None:
        place = self.leg.place
        destination = self.scenario.support(place.support_name)
        current_quaternion = tuple(
            float(value) for value in self.can.data.root_quat_w[0].tolist()
        )
        # Surface placement preserves the verified held orientation.  A
        # near-spherical container target may additionally reorient in the
        # collision-checked preflight below because its task geometry is
        # invariant to object attitude.
        if place.tool_quaternion_wxyz is not None:
            tool_quaternion = torch.tensor(
                [place.tool_quaternion_wxyz], device=self.device
            )
            held_quaternion = torch.tensor(
                [held_pose.quaternion_wxyz], device=self.device
            )
            object_quaternion = math_utils.quat_mul(
                tool_quaternion, held_quaternion
            )
            desired_quaternion = tuple(
                float(value) for value in object_quaternion[0].tolist()
            )
        else:
            desired_quaternion = (
                current_quaternion
                if place.kind in {"surface", "container"}
                else place.target_quaternion_wxyz
            )
        target = (*place.center_xy, place.settled_z)
        selected_target = target
        if (
            place.tool_quaternion_wxyz is not None
            and place.reorientation_steps > 0
        ):
            route = self.leg.route
            assert route is not None
            bx, by, bz, base_yaw = route.destination_base_pose_xyzyaw
            tuck_x, tuck_y, tuck_z = CARRY_TUCK_TCP_BASE
            cosine, sine = math.cos(base_yaw), math.sin(base_yaw)
            place_tuck_position = (
                bx + cosine * tuck_x - sine * tuck_y,
                by + sine * tuck_x + cosine * tuck_y,
                bz + tuck_z,
            )
            for index in range(1, place.reorientation_steps + 1):
                self._plan_execute(
                    planner,
                    f"place_reorient_{index}",
                    Pose(
                        place_tuck_position,
                        _slerp_quaternion(
                            self.leg.grasp.grasp_quaternion_wxyz,
                            place.tool_quaternion_wxyz,
                            index / place.reorientation_steps,
                        ),
                    ),
                    result,
                    closed=True,
                )
        if place.kind == "container" or place.preferred_orientation_mode is not None:
            # Preflight the whole placement chain before moving. Rotations around
            # world Z can lie on very different Kinova wrist branches even when
            # they are equivalent for a lime or cube placement. Small center
            # offsets stay inside the configured target and give fixed-base IK a
            # deterministic alternative near a workspace boundary.
            yaw_offsets = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)
            center_offsets = ((0.0, 0.0), (-0.015, 0.0), (0.015, 0.0), (0.0, -0.015), (0.0, 0.015))
            def multiply_quaternions(left, right):
                lw, lx, ly, lz = left
                rw, rx, ry, rz = right
                return (
                    lw * rw - lx * rx - ly * ry - lz * rz,
                    lw * rx + lx * rw + ly * rz - lz * ry,
                    lw * ry - lx * rz + ly * rw + lz * rx,
                    lw * rz + lx * ry - ly * rx + lz * rw,
                )

            orientation_candidates = []
            for degrees in yaw_offsets:
                half = 0.5 * math.radians(degrees)
                yaw_quaternion = (math.cos(half), 0.0, 0.0, math.sin(half))
                orientation_candidates.append(
                    (
                        "preserve_carried_tilt",
                        degrees,
                        multiply_quaternions(yaw_quaternion, desired_quaternion),
                    )
                )
            side_reference = (-0.2520023, -0.2520023, 0.6606776, 0.6606776)
            configured_top = self.leg.grasp.grasp_quaternion_wxyz
            tool_orientation_bases = (
                ("side_release", side_reference),
                (
                    "oblique_top_50",
                    _slerp_quaternion(side_reference, configured_top, 0.50),
                ),
                (
                    "oblique_top_65",
                    _slerp_quaternion(side_reference, configured_top, 0.65),
                ),
                (
                    "oblique_top_80",
                    _slerp_quaternion(side_reference, configured_top, 0.80),
                ),
            )
            for mode, tool_base in tool_orientation_bases:
                for degrees in yaw_offsets:
                    half = 0.5 * math.radians(degrees)
                    yaw_quaternion = (math.cos(half), 0.0, 0.0, math.sin(half))
                    tool_quaternion = multiply_quaternions(
                        yaw_quaternion, tool_base
                    )
                    orientation_candidates.append(
                        (
                            mode,
                            degrees,
                            multiply_quaternions(
                                tool_quaternion, held_pose.quaternion_wxyz
                            ),
                        )
                    )
            if place.preferred_orientation_mode is not None:
                orientation_candidates.sort(
                    key=lambda candidate: (
                        candidate[0] != place.preferred_orientation_mode,
                        candidate[1] != place.preferred_yaw_offset_degrees,
                    )
                )
            if place.preferred_center_offset_xy is not None:
                center_offsets = (
                    place.preferred_center_offset_xy,
                    *(
                        offset
                        for offset in center_offsets
                        if offset != place.preferred_center_offset_xy
                    ),
                )
            selected_chain = None
            candidate_chains = []
            last_error = None
            preflight_depth_counts = {
                "preplace": 0,
                "place_approach": 0,
                "place": 0,
            }
            for x_offset, y_offset in center_offsets:
                candidate_target = (
                    target[0] + x_offset,
                    target[1] + y_offset,
                    target[2],
                )
                for orientation_mode, degrees, candidate_quaternion in orientation_candidates:
                    chain = []
                    start = self._arm_positions()
                    try:
                        for name, clearance in (
                            ("preplace", place.preplace_clearance_m),
                            ("place_approach", place.approach_clearance_m),
                            ("place", place.release_clearance_m),
                        ):
                            if (
                                name == "place"
                                and place.release_from_approach
                            ):
                                trajectory = JointTrajectory(
                                    chain[-1][1].joint_names,
                                    (tuple(start),),
                                    0.0,
                                )
                            else:
                                trajectory = planner.plan(
                                    start,
                                    world_pose_to_base(
                                        self._tool_pose_for_object(
                                            (
                                                candidate_target[0],
                                                candidate_target[1],
                                                candidate_target[2] + clearance,
                                            ),
                                            candidate_quaternion,
                                            held_pose,
                                        ),
                                        planner.base_pose,
                                    ),
                                    allow_hand_contact=name == "place",
                                )
                                trajectory, start = (
                                    self._rebase_trajectory_to_physical_branch(
                                        trajectory, start
                                    )
                                )
                            chain.append((name, trajectory))
                            preflight_depth_counts[name] += 1
                    except RuntimeError as error:
                        last_error = error
                        continue
                    if not self._trajectories_stay_on_physical_branch(
                        *(trajectory for _, trajectory in chain)
                    ):
                        last_error = RuntimeError(
                            "placement candidate crosses a continuous-joint hardware limit"
                        )
                        continue
                    continuous = np.asarray(
                        CONTINUOUS_ARM_INDICES, dtype=np.int64
                    )
                    branch_start = np.asarray(
                        self._arm_positions(), dtype=np.float64
                    )
                    maximum_segment_branch_delta = 0.0
                    for _, candidate_trajectory in chain:
                        endpoint = np.asarray(
                            candidate_trajectory.positions[-1], dtype=np.float64
                        )
                        maximum_segment_branch_delta = max(
                            maximum_segment_branch_delta,
                            float(
                                np.max(
                                    np.abs(
                                        endpoint[continuous]
                                        - branch_start[continuous]
                                    )
                                )
                            ),
                        )
                        branch_start = endpoint
                    candidate_chains.append(
                        (
                            maximum_segment_branch_delta,
                            chain,
                            candidate_target,
                            orientation_mode,
                            degrees,
                            (x_offset, y_offset),
                        )
                    )
                    if len(candidate_chains) >= place.placement_preflight_candidate_count:
                        break
                if len(candidate_chains) >= place.placement_preflight_candidate_count:
                    break
            if candidate_chains:
                (
                    self._selected_place_branch_cost_rad,
                    selected_chain,
                    selected_target,
                    self._selected_place_orientation_mode,
                    self._selected_place_yaw_offset_degrees,
                    self._selected_place_center_offset_xy,
                ) = min(
                    candidate_chains,
                    key=lambda candidate: (
                        math.dist(
                            candidate[5],
                            place.preferred_center_offset_xy,
                        )
                        if place.preferred_center_offset_xy is not None
                        else 0.0,
                        candidate[0],
                    ),
                )
            if selected_chain is None:
                raise RuntimeError(
                    "no reachable full placement chain; "
                    f"successful_depth_counts={preflight_depth_counts}; "
                    f"last_error={last_error}"
                )
        else:
            selected_chain = []
            start = self._arm_positions()
            dx, dy = place.approach_direction_xy
            steps = [
                (
                    "preplace",
                    place.preplace_clearance_m,
                    (
                        -place.preplace_offset_m * dx,
                        -place.preplace_offset_m * dy,
                    ),
                ),
                ("place_approach", place.approach_clearance_m, (0.0, 0.0)),
            ]
            steps.extend(
                (f"place_lower_{index}", clearance, (0.0, 0.0))
                for index, clearance in enumerate(
                    place.intermediate_clearances_m, start=1
                )
            )
            steps.append(("place", place.release_clearance_m, (0.0, 0.0)))
            for name, clearance, xy_offset in steps:
                trajectory = planner.plan(
                    start,
                    world_pose_to_base(
                        self._tool_pose_for_object(
                            (
                                selected_target[0] + xy_offset[0],
                                selected_target[1] + xy_offset[1],
                                selected_target[2] + clearance,
                            ),
                            desired_quaternion,
                            held_pose,
                        ),
                        planner.base_pose,
                    ),
                    allow_hand_contact=name == "place",
                )
                trajectory, start = self._rebase_trajectory_to_physical_branch(
                    trajectory, start
                )
                selected_chain.append((name, trajectory))

        place_trajectory = None
        for name, trajectory in selected_chain:
            result.phase_history.append(name)
            self._stage_label = name
            result.planning_times_s[name] = trajectory.planning_time_s
            result.trajectory_waypoints[name] = len(trajectory.positions)
            try:
                self._execute(
                    trajectory,
                    closed=True,
                    max_step_rad=(
                        place.preplace_target_step_rad
                        if name == "preplace"
                        else place.placement_target_step_rad
                    ),
                    tracking_steps=360 if name == "preplace" else 240,
                    tracking_tolerance_rad=(
                        place.preplace_tracking_tolerance_rad
                        if name == "preplace"
                        else place.placement_tracking_tolerance_rad
                    ),
                    endpoint_steps=300 if name == "preplace" else 160,
                    endpoint_tolerance_rad=(
                        place.preplace_endpoint_tolerance_rad
                        if name == "preplace"
                        else place.placement_endpoint_tolerance_rad
                    ),
                    wrap_continuous_delta=False,
                    minimum_tracking_steps=(
                        1
                        if name == "preplace"
                        else place.placement_minimum_tracking_steps
                    ),
                )
            except RuntimeError as error:
                raise RuntimeError(f"{name}: {error}") from error
            result.arm_endpoints_rad[name] = list(self._arm_positions())
            if name == "place":
                if place.release_from_approach:
                    place_trajectory = selected_chain[-2][1]
                else:
                    place_trajectory = trajectory
        assert place_trajectory is not None
        self._hold(30, closed=True)
        result.phase_history.append("release")
        self._stage_label = "release"
        self.grasp_joint.GetJointEnabledAttr().Set(False)
        planner.set_held_object(None)
        self._hold(90, closed=False)
        result.phase_history.append("retreat")
        self._stage_label = "retreat"
        retreat = JointTrajectory(
            place_trajectory.joint_names,
            tuple(reversed(place_trajectory.positions)),
            0.0,
            sample_dt_s=place_trajectory.sample_dt_s,
        )
        self._execute(retreat, closed=False)
        self._hold(90, closed=False)
        final = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
        final_xy = (float(final[0]), float(final[1]))
        result.place_target_error_m = math.dist(final_xy, place.center_xy)
        speed_ok = float(torch.linalg.vector_norm(self.can.data.root_lin_vel_w[0])) < 0.05
        radius = 0.5 * max(self.object_spec.dimensions_xyz[:2])
        if place.kind == "container":
            opening = destination.opening_size_xy
            assert opening is not None
            contained = (
                abs(final_xy[0] - destination.center_xyz[0]) + radius
                <= 0.5 * opening[0] - place.containment_margin_m
                and abs(final_xy[1] - destination.center_xyz[1]) + radius
                <= 0.5 * opening[1] - place.containment_margin_m
            )
            settled = contained and speed_ok
        else:
            distance = math.dist(final_xy, destination.center_xyz[:2])
            settled = (
                distance + radius <= 0.5 * min(destination.size_xyz[:2])
                and result.place_target_error_m <= place.xy_tolerance_m
                and abs(float(final[2]) - place.settled_z) <= place.z_tolerance_m
                and speed_ok
            )
        if not settled:
            raise RuntimeError(
                f"{self.object_name} did not satisfy {place.kind} placement; "
                f"final={final.tolist()}, target={selected_target}"
            )

    def _pour_and_validate(
        self,
        planner: CuroboFixedBasePlanner,
        held_pose: Pose,
        pick_position: tuple[float, float, float],
        result: XGRIDS2CuroboResult,
    ) -> dict[str, float | bool | str]:
        pour = self.scenario.pour
        assert pour is not None
        receiver = self.scenario.support(pour.receiver_name)
        current_quaternion = tuple(
            float(value) for value in self.can.data.root_quat_w[0].tolist()
        )
        rim_z = receiver.center_xyz[2] + 0.5 * receiver.size_xyz[2]
        target_spout = (
            receiver.center_xyz[0],
            receiver.center_xyz[1],
            rim_z + 0.5 * sum(pour.spout_height_range_m),
        )
        # The benchmark constrains bottle tilt and spout placement, not the
        # azimuth of the horizontal tilt axis.  Select the first CuRobo-
        # reachable azimuth deterministically.  This avoids forcing the wrist
        # through a joint limit in this particular base/table geometry.
        tilt_radians = math.radians(pour.tilt_degrees)
        tilt_candidates: list[
            tuple[float, tuple[float, float, float, float], tuple[float, float, float], Pose]
        ] = []
        for azimuth_degrees in (-90.0, 0.0, 90.0, 180.0, -45.0, 45.0):
            azimuth = math.radians(azimuth_degrees)
            half_sine = math.sin(0.5 * tilt_radians)
            tilted_quaternion = (
                math.cos(0.5 * tilt_radians),
                half_sine * math.cos(azimuth),
                half_sine * math.sin(azimuth),
                0.0,
            )
            q = torch.tensor([tilted_quaternion], device=self.device)
            spout_offset = math_utils.quat_apply(
                q, torch.tensor([pour.spout_local_xyz], device=self.device)
            )[0]
            tilted_center = tuple(
                target_spout[index] - float(spout_offset[index].item())
                for index in range(3)
            )
            target_pose = self._tool_pose_for_object(
                tilted_center, tilted_quaternion, held_pose
            )
            tilt_candidates.append(
                (azimuth_degrees, tilted_quaternion, tilted_center, target_pose)
            )

        selected = None
        pour_trajectory = None
        result.phase_history.append("pour_tilt")
        self._stage_label = "pour_tilt"
        for azimuth_degrees, quaternion, center, target_pose in tilt_candidates:
            try:
                candidate = planner.plan(
                    self._arm_positions(),
                    world_pose_to_base(target_pose, planner.base_pose),
                )
            except RuntimeError:
                continue
            selected = (azimuth_degrees, quaternion, center)
            pour_trajectory = candidate
            break
        if selected is None or pour_trajectory is None:
            raise RuntimeError(
                "CuRobo found no reachable 110-degree spout-aligned pour azimuth"
            )
        result.planning_times_s["pour_tilt"] = pour_trajectory.planning_time_s
        result.trajectory_waypoints["pour_tilt"] = len(pour_trajectory.positions)
        self._execute(
            pour_trajectory,
            closed=True,
            max_step_rad=0.002,
            tracking_steps=240,
            tracking_tolerance_rad=0.085,
            endpoint_steps=300,
            endpoint_tolerance_rad=0.040,
        )
        result.arm_endpoints_rad["pour_tilt"] = list(self._arm_positions())
        selected_azimuth, tilted_quaternion, tilted_center = selected
        result.phase_history.append("pour_hold")
        self._stage_label = "pour_hold"
        self._hold(pour.hold_control_steps, closed=True)
        object_q = self.can.data.root_quat_w[0:1]
        spout_world = self.can.data.root_pos_w[0] + math_utils.quat_apply(
            object_q, torch.tensor([pour.spout_local_xyz], device=self.device)
        )[0]
        spout_local = spout_world - self.env.scene.env_origins[0]
        opening = receiver.opening_size_xy
        assert opening is not None
        inside = (
            abs(float(spout_local[0]) - receiver.center_xyz[0])
            <= 0.5 * opening[0] - pour.opening_margin_m
            and abs(float(spout_local[1]) - receiver.center_xyz[1])
            <= 0.5 * opening[1] - pour.opening_margin_m
        )
        spout_height = float(spout_local[2]) - rim_z
        height_ok = pour.spout_height_range_m[0] <= spout_height <= pour.spout_height_range_m[1]
        bottle_axis = math_utils.quat_apply(
            object_q, torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        )[0]
        measured_tilt = math.degrees(
            math.acos(float(torch.clamp(bottle_axis[2], -1.0, 1.0).item()))
        )
        tilt_ok = abs(measured_tilt - pour.tilt_degrees) <= pour.tilt_tolerance_degrees
        if not (inside and height_ok and tilt_ok):
            raise RuntimeError(
                "pose-level pour geometry failed; "
                f"inside={inside}, spout_height={spout_height:.4f}, tilt={measured_tilt:.2f}"
            )
        # Reverse the verified collision-free tilt path before moving back to
        # the original upright placement pose.  Reusing the same path is both
        # more reliable and easier to audit than solving a second large wrist
        # reorientation through a different branch.
        result.phase_history.append("pour_return")
        self._stage_label = "pour_return"
        reverse_trajectory = JointTrajectory(
            pour_trajectory.joint_names,
            tuple(reversed(pour_trajectory.positions)),
            0.0,
            sample_dt_s=pour_trajectory.sample_dt_s,
        )
        self._execute(
            reverse_trajectory,
            closed=True,
            max_step_rad=0.002,
            tracking_steps=240,
            tracking_tolerance_rad=0.085,
            endpoint_steps=300,
            endpoint_tolerance_rad=0.040,
        )
        self._plan_execute(
            planner,
            "return_to_place",
            self._tool_pose_for_object(
                pick_position, (1.0, 0.0, 0.0, 0.0), held_pose
            ),
            result,
            closed=True,
            allow_hand_contact=True,
        )
        result.phase_history.append("release")
        self._stage_label = "release"
        self.grasp_joint.GetJointEnabledAttr().Set(False)
        planner.set_held_object(None)
        self._hold(120, closed=False)
        final_q = self.can.data.root_quat_w[0:1]
        final_axis = math_utils.quat_apply(
            final_q, torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        )[0]
        upright_error = math.degrees(
            math.acos(float(torch.clamp(final_axis[2], -1.0, 1.0).item()))
        )
        if upright_error > pour.return_upright_tolerance_degrees:
            raise RuntimeError(
                f"bottle did not return upright; error={upright_error:.2f} deg"
            )
        return {
            "fidelity": "pose_level_pour_no_fluid",
            "spout_inside_receiver": inside,
            "spout_height_above_rim_m": spout_height,
            "measured_tilt_degrees": measured_tilt,
            "selected_tilt_azimuth_degrees": selected_azimuth,
            "hold_control_steps": pour.hold_control_steps,
            "return_upright_error_degrees": upright_error,
        }

    def run(self, *, stage: str = "full") -> tuple[XGRIDS2CuroboResult, dict]:
        if stage not in {"pick", "carry", "place", "full"}:
            raise ValueError("stage must be one of: pick, carry, place, full")
        result = XGRIDS2CuroboResult(requested_stage=stage)
        result.place_target = list(self.leg.place.center_xy)
        planner = None
        extra: dict = {}
        try:
            self.grasp_joint.GetPrim().SetActive(True)
            self.grasp_joint.GetJointEnabledAttr().Set(False)
            self.reset_robot_to_default(settle_steps=60)
            self._align_pick_base(result)
            local = self.can.data.root_pos_w[0] - self.env.scene.env_origins[0]
            result.pick_position = local.detach().cpu().tolist()
            pregrasp, approach, pick_position = self._pick_targets()
            base_xy, base_yaw = self._robot_planar_state()
            base_z = float(
                (
                    self.robot.data.root_link_pos_w[0, 2]
                    - self.env.scene.env_origins[0, 2]
                ).item()
            )
            base_pose = (float(base_xy[0]), float(base_xy[1]), base_z, base_yaw)
            planner = CuroboFixedBasePlanner(
                base_pose,
                num_seeds=self.num_seeds,
                collision_boxes=self._planner_boxes(include_destination=False),
            )
            self._plan_execute_joint(
                planner,
                "pick_branch_anchor",
                PICK_BRANCH_ANCHOR_JOINTS,
                result,
                closed=False,
            )
            grasp_quaternion = self.leg.grasp.grasp_quaternion_wxyz
            top_approach_trajectory = None
            if self.leg.grasp.kind == "top":
                (
                    pregrasp,
                    approach,
                    pick_position,
                    grasp_quaternion,
                    top_approach_trajectory,
                ) = (
                    self._top_grasp_pregrasp(planner, result)
                )
                extra["selected_grasp_quaternion_wxyz"] = list(grasp_quaternion)
            else:
                self._plan_execute(
                    planner, "pregrasp", pregrasp, result, closed=False
                )
            if top_approach_trajectory is None:
                self._plan_execute(
                    planner,
                    "approach",
                    approach,
                    result,
                    closed=False,
                    allow_hand_contact=True,
                )
            else:
                result.phase_history.append("approach")
                self._stage_label = "approach"
                result.planning_times_s["approach"] = (
                    top_approach_trajectory.planning_time_s
                )
                result.trajectory_waypoints["approach"] = len(
                    top_approach_trajectory.positions
                )
                self._execute(top_approach_trajectory, closed=False)
                result.arm_endpoints_rad["approach"] = list(
                    self._arm_positions()
                )
            held_pose = self._close_and_attach(result)
            grasp = self.leg.grasp
            lift_pose = Pose(
                (
                    pick_position[0],
                    pick_position[1],
                    pick_position[2] + grasp.lift_height_m,
                ),
                grasp_quaternion,
            )
            if top_approach_trajectory is None:
                self._plan_execute(
                    planner,
                    "lift",
                    lift_pose,
                    result,
                    closed=True,
                    allow_hand_contact=True,
                )
            else:
                result.phase_history.append("lift")
                self._stage_label = "lift"
                measured_state_lift = JointTrajectory(
                    top_approach_trajectory.joint_names,
                    (top_approach_trajectory.positions[0],),
                    0.0,
                )
                result.planning_times_s["lift"] = 0.0
                result.trajectory_waypoints["lift"] = len(
                    measured_state_lift.positions
                )
                self._execute(
                    measured_state_lift,
                    closed=True,
                    max_step_rad=0.002,
                    tracking_steps=180,
                    tracking_tolerance_rad=0.085,
                    endpoint_steps=300,
                    endpoint_tolerance_rad=0.060,
                )
                result.arm_endpoints_rad["lift"] = list(self._arm_positions())
            self._hold(20, closed=True)
            if stage == "pick":
                return self._complete_stage(result, "pick"), extra

            planner.set_held_object(
                held_pose,
                object_spheres=self.object_spec.held_spheres or None,
            )
            dx, dy = grasp.approach_direction_xy
            retract_position = (
                pick_position[0] - grasp.retract_distance_m * dx,
                pick_position[1] - grasp.retract_distance_m * dy,
                pick_position[2] + grasp.lift_height_m,
            )
            self._plan_execute(
                planner,
                "retract",
                Pose(retract_position, grasp_quaternion),
                result,
                closed=True,
            )
            if self.scenario.strategy == "pour":
                extra["pouring_metrics"] = self._pour_and_validate(
                    planner, held_pose, pick_position, result
                )
                return self._complete_stage(
                    result, "place" if stage == "place" else "full"
                ), extra

            bx, by, bz, yaw = base_pose
            tx, ty, tz = CARRY_TUCK_TCP_BASE
            cosine, sine = math.cos(yaw), math.sin(yaw)
            self._plan_execute(
                planner,
                "carry_tuck",
                Pose(
                    (
                        bx + cosine * tx - sine * ty,
                        by + sine * tx + cosine * ty,
                        bz + tz,
                    ),
                    grasp_quaternion,
                ),
                result,
                closed=True,
            )
            result.carry_posture_radius_m = self._carry_posture_radius()
            if result.carry_posture_radius_m > grasp.carry_radius_m:
                raise RuntimeError(
                    "configured carry posture exceeds its navigation envelope; "
                    f"radius={result.carry_posture_radius_m:.4f}"
                )
            self._traverse_configured(result)
            if stage == "carry":
                return self._complete_stage(result, "carry"), extra
            destination_pose = self.leg.route.destination_base_pose_xyzyaw
            place_planner = CuroboFixedBasePlanner(
                destination_pose,
                num_seeds=self.num_seeds,
                collision_boxes=self._planner_boxes(include_destination=True),
            )
            held_pose = self._held_pose()
            place_planner.set_held_object(
                held_pose,
                object_spheres=self.object_spec.held_spheres or None,
            )
            self._place_and_validate(place_planner, held_pose, result)
            if hasattr(self, "_selected_place_yaw_offset_degrees"):
                extra["selected_place_yaw_offset_degrees"] = (
                    self._selected_place_yaw_offset_degrees
                )
                extra["selected_place_orientation_mode"] = (
                    self._selected_place_orientation_mode
                )
                extra["selected_place_center_offset_xy"] = list(
                    self._selected_place_center_offset_xy
                )
            return self._complete_stage(
                result, "place" if stage == "place" else "full"
            ), extra
        except Exception as error:
            self._capture_result_state(result)
            result.failure_reason = str(error)
            result.phase_history.append("failure")
            self._stage_label = "failure"
            if hasattr(self, "_selected_place_yaw_offset_degrees"):
                extra["selected_place_yaw_offset_degrees"] = (
                    self._selected_place_yaw_offset_degrees
                )
                extra["selected_place_orientation_mode"] = (
                    self._selected_place_orientation_mode
                )
                extra["selected_place_center_offset_xy"] = list(
                    self._selected_place_center_offset_xy
                )
            self.grasp_joint.GetJointEnabledAttr().Set(False)
            if planner is not None:
                planner.set_held_object(None)
            return result, extra


class XGRIDS2ScenarioOracle:
    """Public facade selecting the validated strategy for one scenario."""

    def __init__(
        self,
        env,
        scenario: str | XGRIDS2ScenarioSpec,
        *,
        num_seeds: int = 12,
        execution_profile: str = "legacy",
    ) -> None:
        self.env = env
        self.scenario = (
            get_xgrids2_scenario(scenario) if isinstance(scenario, str) else scenario
        )
        self.num_seeds = num_seeds
        if execution_profile not in {"legacy", "can_diagnostic"}:
            raise ValueError(f"unknown execution profile: {execution_profile}")
        if execution_profile == "can_diagnostic" and self.scenario.name != "can_dining_to_snack":
            raise ValueError("can_diagnostic is only calibrated for can_dining_to_snack")
        self.execution_profile = execution_profile
        self.backend = None
        self._step_observer = None

    @property
    def _stage_label(self) -> str:
        return "idle" if self.backend is None else self.backend._stage_label

    @property
    def robot(self):
        return self.env.scene["robot"]

    @property
    def device(self):
        return self.env.device

    def set_step_observer(self, observer) -> None:
        """Observe actual physical control steps across every strategy/leg."""

        self._step_observer = observer

    def _install_observer(self, backend) -> None:
        if self._step_observer is None:
            return
        original = backend._step

        def observed_step(action):
            try:
                original(action)
            finally:
                self._step_observer(self, backend)

        backend._step = observed_step
        # The accepted plate strategy has a deliberately separate physical
        # stepping path while its base-retention workaround is active.  Wrap
        # that path too so evidence never skips the carry portion.
        if hasattr(backend, "_locked_transport_step"):
            original_locked = backend._locked_transport_step

            def observed_locked_step(action):
                try:
                    original_locked(action)
                finally:
                    self._step_observer(self, backend)

            backend._locked_transport_step = observed_locked_step

    @staticmethod
    def _merge_motion_diagnostics(aggregate: dict, current: dict) -> dict:
        if not current:
            return aggregate
        if not aggregate:
            return dict(current)
        merged = dict(aggregate)
        merged["control_ticks"] = int(aggregate.get("control_ticks", 0)) + int(
            current.get("control_ticks", 0)
        )
        for key in (
            "maximum_acceleration_rad_s2",
            "maximum_jerk_rad_s3",
            "maximum_target_jerk_rad_s3",
            "maximum_oscillation_reversals_per_window",
        ):
            merged[key] = max(float(aggregate.get(key, 0.0)), float(current.get(key, 0.0)))
        for key in ("jerk_violation", "oscillation_violation"):
            merged[key] = bool(aggregate.get(key, False) or current.get(key, False))
        return merged

    def _aggregate_leg(
        self,
        aggregate: XGRIDS2ScenarioResult,
        leg_name: str,
        result: XGRIDS2CuroboResult,
        extra: dict | None = None,
    ) -> None:
        payload = result.as_dict()
        payload["leg"] = leg_name
        leg = next(item for item in self.scenario.legs if item.name == leg_name)
        payload["grasp_mode"] = (
            f"{leg.grasp.contact_mode}_fingertip_contact_then_"
            "temporary_fixed_joint_retention"
        )
        payload["transport_mode"] = (
            "same_surface_pose_level_reorientation"
            if leg.route is None
            else "closed_loop_holonomic_configured_clearance_route"
        )
        payload["carry_route_radius_m"] = (
            None if leg.route is None else leg.route.runtime_radius_m
        )
        payload["carry_planning_radius_m"] = (
            None if leg.route is None else leg.route.planning_radius_m
        )
        if extra:
            payload["strategy_metrics"] = {
                key: value
                for key, value in extra.items()
                if key != "pouring_metrics"
            }
        aggregate.leg_results.append(payload)
        aggregate.phase_history.extend(
            f"{leg_name}/{phase}" for phase in result.phase_history
        )
        aggregate.maximum_arm_step_rad = max(
            aggregate.maximum_arm_step_rad, result.maximum_arm_step_rad
        )
        aggregate.maximum_arm_speed_rad_s = max(
            aggregate.maximum_arm_speed_rad_s, result.maximum_arm_speed_rad_s
        )
        aggregate.maximum_raw_physx_arm_speed_rad_s = max(
            aggregate.maximum_raw_physx_arm_speed_rad_s,
            result.maximum_raw_physx_arm_speed_rad_s,
        )
        aggregate.motion_diagnostics = self._merge_motion_diagnostics(
            aggregate.motion_diagnostics, result.motion_diagnostics
        )
        aggregate.minimum_route_clearance_valid &= result.minimum_route_clearance_valid
        if extra and "pouring_metrics" in extra:
            aggregate.pouring_metrics = extra["pouring_metrics"]

    @staticmethod
    def _finalize_aggregate(
        aggregate: XGRIDS2ScenarioResult, stage: str
    ) -> XGRIDS2ScenarioResult:
        aggregate.passed = True
        aggregate.completed_stage = stage
        aggregate.phase_history.append(f"{stage}_success")
        if (
            aggregate.maximum_arm_step_rad > MAX_ARM_STEP_QUALITY_RAD
            or aggregate.maximum_arm_speed_rad_s > MAX_ARM_SPEED_QUALITY_RAD_S
        ):
            aggregate.passed = False
            aggregate.failure_reason = "aggregate arm motion-quality gate failed"
        elif aggregate.motion_diagnostics.get("jerk_violation"):
            aggregate.passed = False
            aggregate.failure_reason = "aggregate arm jerk-quality gate failed"
        elif aggregate.motion_diagnostics.get("oscillation_violation"):
            aggregate.passed = False
            aggregate.failure_reason = "aggregate arm oscillation-quality gate failed"
        elif not aggregate.minimum_route_clearance_valid:
            aggregate.passed = False
            aggregate.failure_reason = "aggregate route-clearance gate failed"
        return aggregate

    def _distractor_displacement(self) -> float:
        if self.backend is None or not hasattr(
            self.backend, "_distractor_initial_positions"
        ):
            return 0.0
        maximum = 0.0
        for name, initial in self.backend._distractor_initial_positions.items():
            current = self.env.scene[name].data.root_pos_w[0]
            maximum = max(
                maximum,
                float(torch.linalg.vector_norm(current - initial).item()),
            )
        return maximum

    def run(self, *, stage: str = "full"):
        if self.scenario.name == "can_dining_to_snack":
            from robolab.tasks.mobile_manipulation.xgrids2_can_diagnostic import CanDiagnosticProfile
            from robolab.tasks.mobile_manipulation.xgrids2_unassisted_can_oracle import (
                XGRIDS2UnassistedCanOracle,
            )
            self.backend = XGRIDS2UnassistedCanOracle(
                self.env, num_seeds=min(self.num_seeds, 8),
                diagnostic_profile=CanDiagnosticProfile() if self.execution_profile == "can_diagnostic" else None,
            )
            self._install_observer(self.backend)
            return self.backend.run(stage=stage)
        if self.scenario.name == "plate_dining_to_sink":
            from robolab.tasks.mobile_manipulation.xgrids2_lab_plate_to_sink_curobo_oracle import (
                XGRIDS2PlateToSinkCuroboOracle,
            )

            self.backend = XGRIDS2PlateToSinkCuroboOracle(
                self.env, num_seeds=max(self.num_seeds, 16)
            )
            self._install_observer(self.backend)
            return self.backend.run(stage=stage)

        aggregate = XGRIDS2ScenarioResult(
            self.scenario.name, requested_stage=stage
        )
        for index, leg in enumerate(self.scenario.legs):
            use_validated_can_strategy = (
                self.scenario.strategy == "sequential"
                and index == 0
                and leg.object_name == "can"
            )
            if use_validated_can_strategy:
                self.backend = XGRIDS2CuroboOracle(
                    self.env,
                    num_seeds=min(self.num_seeds, 8),
                    object_name=leg.object_name,
                    grasp_joint_name=scenario_grasp_joint_name(
                        leg.object_name
                    ),
                )
            else:
                self.backend = XGRIDS2ConfigurableTransferOracle(
                    self.env, self.scenario, leg, num_seeds=self.num_seeds
                )
            self.backend.defer_extended_motion_quality_gate = (
                len(self.scenario.legs) > 1
            )
            self._install_observer(self.backend)
            if use_validated_can_strategy:
                leg_result = self.backend.run(
                    stage=stage, place_xy=leg.place.center_xy
                )
                extra = {"strategy": "validated_reference_can_transfer"}
            else:
                leg_result, extra = self.backend.run(stage=stage)
            self._aggregate_leg(aggregate, leg.name, leg_result, extra)
            aggregate.retention_active_end = bool(
                self.backend.grasp_joint.GetJointEnabledAttr().Get()
            )
            if (
                not aggregate.retention_active_end
                and leg_result.completed_stage in {"place", "full"}
            ):
                aggregate.released_objects.append(leg.object_name)
            if not leg_result.passed:
                aggregate.failure_reason = (
                    f"{leg.name}: {leg_result.failure_reason}"
                )
                return aggregate
            if stage != "full" or not leg.return_to_source_after:
                break
            return_backend = XGRIDS2ConfigurableTransferOracle(
                self.env, self.scenario, leg, num_seeds=self.num_seeds
            )
            return_backend.defer_extended_motion_quality_gate = True
            self._install_observer(return_backend)
            try:
                carry_tuck = leg_result.arm_endpoints_rad.get("carry_tuck")
                return_segments = tuple(
                    getattr(self.backend, "_empty_return_segments", ())
                )
                return_backend.navigate_empty_to_source(
                    tuple(carry_tuck) if carry_tuck else None,
                    validated_return_segments=return_segments,
                )
            except Exception as error:
                aggregate.maximum_arm_step_rad = max(
                    aggregate.maximum_arm_step_rad,
                    return_backend._maximum_arm_step_rad,
                )
                aggregate.maximum_arm_speed_rad_s = max(
                    aggregate.maximum_arm_speed_rad_s,
                    return_backend._maximum_arm_speed_rad_s,
                )
                aggregate.maximum_raw_physx_arm_speed_rad_s = max(
                    aggregate.maximum_raw_physx_arm_speed_rad_s,
                    return_backend._maximum_raw_physx_arm_speed_rad_s,
                )
                aggregate.motion_diagnostics = self._merge_motion_diagnostics(
                    aggregate.motion_diagnostics,
                    return_backend._motion_diagnostics.summary(),
                )
                aggregate.failure_reason = (
                    f"{leg.name}/return_to_source[{return_backend._stage_label}]: "
                    f"{error}"
                )
                aggregate.phase_history.append(
                    f"{leg.name}/return_to_source_failure"
                )
                aggregate.retention_active_end = bool(
                    return_backend.grasp_joint.GetJointEnabledAttr().Get()
                )
                self.backend = return_backend
                return aggregate
            aggregate.maximum_arm_step_rad = max(
                aggregate.maximum_arm_step_rad,
                return_backend._maximum_arm_step_rad,
            )
            aggregate.maximum_arm_speed_rad_s = max(
                aggregate.maximum_arm_speed_rad_s,
                return_backend._maximum_arm_speed_rad_s,
            )
            aggregate.maximum_raw_physx_arm_speed_rad_s = max(
                aggregate.maximum_raw_physx_arm_speed_rad_s,
                return_backend._maximum_raw_physx_arm_speed_rad_s,
            )
            aggregate.motion_diagnostics = self._merge_motion_diagnostics(
                aggregate.motion_diagnostics,
                return_backend._motion_diagnostics.summary(),
            )
            self.backend = return_backend
            aggregate.phase_history.append(f"{leg.name}/return_to_source")
        aggregate.distractor_max_displacement_m = self._distractor_displacement()
        if (
            self.scenario.name == "lime_clutter_to_bowl"
            and aggregate.distractor_max_displacement_m > 0.03
        ):
            aggregate.failure_reason = (
                "clutter distractor displacement exceeded 0.03 m: "
                f"{aggregate.distractor_max_displacement_m:.4f} m"
            )
            return aggregate
        return self._finalize_aggregate(aggregate, stage)
