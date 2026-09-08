"""Authoritative geometry and physics constants for the GenManip kitchen task.

The detailed EBench USD remains the rendering source.  The primitives below are
the deliberately simple collision contract used by PhysX, reset validation, and
base-clearance tests.  All dimensions are metres in the authored kitchen frame.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Bounds2D:
    """Closed axis-aligned XY bounds."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def size(self) -> tuple[float, float]:
        return (self.xmax - self.xmin, self.ymax - self.ymin)

    def inflated(self, radius: float) -> "Bounds2D":
        return Bounds2D(
            self.xmin - radius,
            self.xmax + radius,
            self.ymin - radius,
            self.ymax + radius,
        )

    def contains(self, point: tuple[float, float], *, tolerance: float = 0.0) -> bool:
        x, y = point
        return (
            self.xmin - tolerance <= x <= self.xmax + tolerance
            and self.ymin - tolerance <= y <= self.ymax + tolerance
        )

    def distance_to(self, point: tuple[float, float]) -> float:
        """Euclidean distance to the box (zero for points inside it)."""

        x, y = point
        dx = max(self.xmin - x, 0.0, x - self.xmax)
        dy = max(self.ymin - y, 0.0, y - self.ymax)
        return hypot(dx, dy)


@dataclass(frozen=True)
class CollisionBox:
    """One static cuboid in the task collision overlay."""

    name: str
    center_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    category: str
    visible: bool = False

    @property
    def footprint(self) -> Bounds2D:
        cx, cy, _ = self.center_xyz
        sx, sy, _ = self.size_xyz
        return Bounds2D(cx - sx / 2.0, cx + sx / 2.0, cy - sy / 2.0, cy + sy / 2.0)

    @property
    def z_bounds(self) -> tuple[float, float]:
        _, _, cz = self.center_xyz
        _, _, sz = self.size_xyz
        return (cz - sz / 2.0, cz + sz / 2.0)


KITCHEN_USD_RELATIVE = (
    "assets/third_party/ebench/assets/scene_usds/ebench/long_horizon/"
    "scene_dishwasher/scene.usd"
)
CAN_PHYSICS_USD_RELATIVE = "assets/objects/ycb/tomato_soup_can_physics.usda"

# Milestone-0 measured support geometry.
ISLAND_SURFACE_Z = 1.0450
COUNTER_SURFACE_Z = 0.8453
PICK_REGION = Bounds2D(-0.06, 0.04, 0.10, 0.25)
PLACE_REGION = Bounds2D(1.76, 1.86, -0.02, 0.22)
# Reach-conditioned domains for the current scripted side-grasp oracle.  The
# broader PICK_REGION/PLACE_REGION remain the physical support contracts.
ORACLE_PICK_REGION = Bounds2D(-0.03, 0.04, 0.12, 0.23)
# The full PLACE_REGION is collision-backed support geometry.  The current
# side-grasp IK oracle uses this conservative interior domain, established by
# the randomized center-target and edge-stress trials, so policy comparisons
# do not conflate controller singularities with physical support validity.
ORACLE_PLACE_REGION = Bounds2D(1.79, 1.83, 0.09, 0.11)
SINK_OPENING = Bounds2D(-0.48, -0.10, -0.88, -0.52)

# Robot footprint and verified staging poses.
ROBOT_BASE_RADIUS = 0.30
BASE_SAFETY_MARGIN = 0.05
BASE_CLEARANCE_INFLATION = ROBOT_BASE_RADIUS + BASE_SAFETY_MARGIN
ROBOT_BASE_Z = 0.468
STAGING_POSES = {
    "pick": (0.52, 0.175, ROBOT_BASE_Z, 0.0),
    # The navigation camera looks along robot -X. Face +X at the opposing
    # counter so the fixed sensor observes the placement workspace.
    "place": (1.12, 0.10, ROBOT_BASE_Z, math.pi),
    "retreat": (1.00, 0.10, ROBOT_BASE_Z, math.pi),
}

# Measured YCB can envelope and initial physical parameters.
CAN_RADIUS = 0.033927
CAN_HEIGHT = 0.102037
CAN_HALF_HEIGHT = CAN_HEIGHT / 2.0
CAN_MASS_KG = 0.349
CAN_STATIC_FRICTION = 0.80
CAN_DYNAMIC_FRICTION = 0.70
CAN_RESTITUTION = 0.02
CAN_CONTACT_OFFSET = 0.002
CAN_REST_OFFSET = 0.0
CAN_DROP_HEIGHT = 0.02
CAN_INITIAL_Z = ISLAND_SURFACE_Z + CAN_HALF_HEIGHT + CAN_DROP_HEIGHT

# Manipulation geometry measured from the live Kinova/Robotiq articulation.
# The task TCP is the center of the two flat fingertip pads, not the front-most
# tangent at which the source rounded collision meshes happened to touch the
# cylinder. The flange-to-link origin is 0.1129 m and each pad is centered
# another 0.030 m along the flange's local +Z axis.
GRASP_TCP_OFFSET_FLANGE = (0.0, 0.0, 0.1429)
# Top-down grasp: local gripper X spans the jaws along world +Y and local Z
# points down along world -Z. Quaternion convention is (w, x, y, z).
TOP_DOWN_GRASP_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)
# Island side grasp: local Z approaches along world -X, local X spans the
# jaws along world -Y, and local Y points up. This makes both fingertips meet
# the cylindrical can at the same X plane instead of one finger leading.
SIDE_GRASP_QUAT_WXYZ = (0.5, 0.5, -0.5, -0.5)
# Yaw-mirrored side grasp for approaching the opposing counter from -X.  The
# tool and can remain upright; only the approach direction changes.
PLACE_SIDE_GRASP_QUAT_WXYZ = (0.5, 0.5, 0.5, 0.5)
PREGRASP_CLEARANCE = 0.20
SIDE_PREGRASP_DISTANCE = 0.12
LIFT_CLEARANCE = 0.04
MIN_LIFT_CLEARANCE = 0.02
LIFT_TCP_SPEED = 0.003
CARRY_BASE_SPEED = 0.05
CARRY_BASE_LINEAR_ACCELERATION = 0.12
CARRY_BASE_LINEAR_JERK = 0.40
CARRY_REPOSITION_TCP_SPEED = 0.05
PLACE_TCP_SPEED = 0.015
PLACE_APPROACH_CLEARANCE = 0.10
PLACE_RELEASE_CLEARANCE = 0.020
PLACE_TARGET_TOLERANCE = 0.025
HANDOFF_SETTLE_FRAMES = 5
HANDOFF_LINEAR_SETTLE_SPEED = 0.10
HANDOFF_ANGULAR_SETTLE_SPEED = 0.50
# Gate-A gripper control and physical grasp qualification.  Search speed is
# half the 0.5 rad/s limit in the packaged Robotiq URDF.  Force feedback then
# varies a bounded virtual spring preload instead of commanding full closure.
GRIPPER_CLOSE_SEARCH_SPEED_RAD_S = 0.25
GRIPPER_FORCE_CONTROL_SPEED_RAD_S = 0.04
GRIPPER_FORCE_CONTROL_GAIN_RAD_PER_N_S = 0.04
GRIPPER_MAX_PRELOAD_RAD = 0.500
# Keep the commanded 50 N total pinch inside the Robotiq 2F-85 published
# 20--235 N force range. Each pad therefore regulates to 25 N; this remains a
# simulation calibration point until force/displacement bench traces are added.
GRIPPER_TARGET_NORMAL_FORCE_N = 25.0
GRIPPER_FORCE_DEADBAND_N = 0.50
GRIPPER_OPEN_SPEED_RAD_S = 0.20
GRIPPER_POSITION_TOLERANCE_RAD = 0.03
CARRY_SETTLE_FRAMES = 5
GRASP_CONTACT_FORCE_N = 0.5
GRASP_STABLE_FRAMES = 8
GRASP_SECURE_FORCE_N = 2.0
GRASP_SECURE_STABLE_FRAMES = 8
GRASP_FORCE_NORMAL_CONE_RAD = math.radians(35.0)
GRASP_NORMAL_OPPOSITION_TOLERANCE_RAD = math.radians(25.0)
GRASP_MAX_RELATIVE_POSITION_DRIFT_M = 0.003
GRASP_MAX_RELATIVE_ROTATION_DRIFT_RAD = math.radians(5.0)
GRASP_MAX_PAD_SLIP_SPEED_M_S = 0.025
GRASP_MAX_JOINT_TRACKING_ERROR_RAD = 0.40
GRASP_LOST_CONTACT_FRAMES = 3
PLACEMENT_STABLE_FRAMES = 15
EEF_POSITION_TOLERANCE = 0.012
EEF_ORIENTATION_TOLERANCE_RAD = 0.10
BASE_POSITION_TOLERANCE = 0.025
BASE_YAW_TOLERANCE_RAD = 0.06
CAN_LINEAR_SETTLE_SPEED = 0.02
CAN_ANGULAR_SETTLE_SPEED = 0.15
# A can constrained to a driven articulation retains small PhysX solver
# motion even when its grasp transform is stable.  Keep this distinct from the
# strict free-body thresholds used for support and placement predicates.
CARRY_LINEAR_SETTLE_SPEED = 0.12
CARRY_ANGULAR_SETTLE_SPEED = 0.25

# Geometry used to build the collision-only layer.  The aisle-facing island
# edge (x=0.14) and opposing-counter edge (x=1.48) come from the Milestone-0
# cross-section measurement, giving a 1.34 m aisle.
STATIC_COLLISION_BOXES = (
    CollisionBox("floor", (1.00, -0.15, -0.05), (5.60, 5.40, 0.10), "floor"),
    CollisionBox("back_wall", (1.00, 2.55, 1.50), (5.60, 0.12, 3.00), "wall", True),
    CollisionBox("left_wall", (-1.80, -0.15, 1.50), (0.12, 5.40, 3.00), "wall", True),
    CollisionBox("right_wall", (3.80, -0.15, 1.50), (0.12, 5.40, 3.00), "wall", True),
    # Island cabinet below the sink basin.
    CollisionBox("island_lower_cabinet", (-0.28, -0.07, 0.41), (0.84, 2.18, 0.82), "island"),
    # Four worktop slabs retain the measured sink opening.
    CollisionBox("island_top_west", (-0.59, -0.07, 1.025), (0.22, 2.18, 0.04), "island"),
    CollisionBox("island_top_east", (0.02, -0.07, 1.025), (0.24, 2.18, 0.04), "island"),
    CollisionBox("island_top_south", (-0.29, -1.02, 1.025), (0.38, 0.28, 0.04), "island"),
    CollisionBox("island_top_north", (-0.29, 0.25, 1.025), (0.38, 1.54, 0.04), "island"),
    # Basin bottom and four walls, based on the measured 0.838--0.843 m recess.
    CollisionBox("sink_bottom", (-0.29, -0.70, 0.83), (0.36, 0.34, 0.02), "sink"),
    CollisionBox("sink_wall_west", (-0.47, -0.70, 0.9325), (0.02, 0.36, 0.205), "sink"),
    CollisionBox("sink_wall_east", (-0.11, -0.70, 0.9325), (0.02, 0.36, 0.205), "sink"),
    CollisionBox("sink_wall_south", (-0.29, -0.87, 0.9325), (0.36, 0.02, 0.205), "sink"),
    CollisionBox("sink_wall_north", (-0.29, -0.53, 0.9325), (0.36, 0.02, 0.205), "sink"),
    # L-shaped opposing counter: the task placement region is on right_run.
    CollisionBox("counter_right_cabinet", (1.82, -0.98, 0.41265), (0.68, 3.04, 0.8253), "counter"),
    CollisionBox("counter_right_top", (1.82, -0.98, 0.8253), (0.68, 3.04, 0.04), "counter"),
    CollisionBox("counter_back_cabinet", (0.66, -2.16, 0.41265), (1.64, 0.68, 0.8253), "counter"),
    CollisionBox("counter_back_top", (0.66, -2.16, 0.8253), (1.64, 0.68, 0.04), "counter"),
)

STATIC_COLLIDER_NAMES = tuple(box.name for box in STATIC_COLLISION_BOXES)
NAVIGATION_OBSTACLE_CATEGORIES = frozenset({"wall", "island", "sink", "counter"})


def _segment_intersects_box_interior(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: Bounds2D,
    *,
    tolerance: float = 1.0e-9,
) -> bool:
    """Return whether a line segment enters the interior of an AABB.

    Touching a boundary is not an intersection.  The implementation is the
    slab-form line-clipping test and is deterministic without geometry-library
    dependencies.
    """

    x0, y0 = start
    dx = end[0] - x0
    dy = end[1] - y0
    lower = (bounds.xmin + tolerance, bounds.ymin + tolerance)
    upper = (bounds.xmax - tolerance, bounds.ymax - tolerance)
    t_min, t_max = 0.0, 1.0
    for origin, delta, lo, hi in ((x0, dx, lower[0], upper[0]), (y0, dy, lower[1], upper[1])):
        if lo >= hi:
            return False
        if abs(delta) < tolerance:
            if not lo < origin < hi:
                return False
            continue
        t0 = (lo - origin) / delta
        t1 = (hi - origin) / delta
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_min > t_max:
            return False
    return t_max >= 0.0 and t_min <= 1.0


def navigation_obstacle_footprints() -> tuple[Bounds2D, ...]:
    """Return unique task-relevant fixture footprints."""

    # Use the outer furniture footprints rather than every constituent slab.
    return (
        Bounds2D(-1.86, -1.74, -2.85, 2.55),  # left wall
        Bounds2D(3.74, 3.86, -2.85, 2.55),  # right wall
        Bounds2D(-1.80, 3.80, 2.49, 2.61),  # back wall
        Bounds2D(-0.70, 0.14, -1.16, 1.02),  # island and sink unit
        Bounds2D(1.48, 2.16, -2.50, 0.54),  # counter right run
        Bounds2D(-0.16, 1.48, -2.50, -1.82),  # counter back run
    )


def base_pose_is_clear(pose_xy: tuple[float, float]) -> bool:
    """Check one base center against safety-inflated fixture footprints."""

    return not any(
        obstacle.inflated(BASE_CLEARANCE_INFLATION).contains(pose_xy)
        for obstacle in navigation_obstacle_footprints()
    )


def base_path_is_clear(
    start_xy: tuple[float, float], end_xy: tuple[float, float]
) -> bool:
    """Check the entire straight base-center path against inflated fixtures."""

    return not any(
        _segment_intersects_box_interior(
            start_xy,
            end_xy,
            obstacle.inflated(BASE_CLEARANCE_INFLATION),
        )
        for obstacle in navigation_obstacle_footprints()
    )


def swept_center_intersects_box(
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    box: CollisionBox,
    inflation_xyz: tuple[float, float, float],
    *,
    tolerance: float = 1.0e-9,
) -> bool:
    """Test a swept axis-aligned proxy against one collision-box interior.

    Inflating the fixture by the proxy half extents turns the continuous
    moving-volume test into a segment-versus-AABB slab test.  This catches
    between-frame tunnelling that checking only the two endpoint poses misses.
    """

    half = tuple(0.5 * size for size in box.size_xyz)
    lower = tuple(
        center - extent - inflation
        for center, extent, inflation in zip(box.center_xyz, half, inflation_xyz)
    )
    upper = tuple(
        center + extent + inflation
        for center, extent, inflation in zip(box.center_xyz, half, inflation_xyz)
    )
    t_min, t_max = 0.0, 1.0
    for origin, end, lo, hi in zip(start_xyz, end_xyz, lower, upper):
        delta = end - origin
        if abs(delta) < tolerance:
            if origin < lo or origin > hi:
                return False
            continue
        t0 = (lo - origin) / delta
        t1 = (hi - origin) / delta
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_min > t_max:
            return False
    return t_max >= 0.0 and t_min <= 1.0


def swept_proxy_is_clear(
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    inflation_xyz: tuple[float, float, float],
    *,
    excluded_boxes: frozenset[str] = frozenset(),
) -> bool:
    """Check one swept proxy against the task's complete static overlay."""

    return not any(
        box.name not in excluded_boxes
        and swept_center_intersects_box(start_xyz, end_xyz, box, inflation_xyz)
        for box in STATIC_COLLISION_BOXES
    )
