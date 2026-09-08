"""Audited geometry for the XGRIDS2 dining-table-to-sink plate task."""

from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass

import numpy as np

from robolab.constants import PACKAGE_DIR
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
    Bounds2D,
    CollisionSegment,
    DEFAULT_SAFETY_MARGIN_M,
    LABAPT_VECTOR_MAP_RELATIVE,
    NAVIGATION_RESOLUTION_M,
    PROXY_WIDTH_M,
    ROBOT_BASE_Z_M,
    STAGING_POSES,
    transform_labapt_xy,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_manipulation_geometry import (
    DINING_TABLE,
)


PLATE_USD_RELATIVE = "assets/objects/vomp/plate_small/plate_small.usd"
PLATE_DIAMETER_M = 0.2976294
PLATE_RADIUS_M = 0.5 * PLATE_DIAMETER_M
PLATE_THICKNESS_M = 0.0211513
PLATE_HALF_THICKNESS_M = 0.5 * PLATE_THICKNESS_M
PLATE_MASS_KG = 0.25

# Reuse the proven near-edge can center.  A 29.8 cm plate overhangs the
# robot-facing edge by about 6.2 cm while its center of mass remains supported.
# That overhang is intentional: the lower Robotiq fingertip can reach beneath
# the rim without intersecting the clean tabletop collider.
# Shift the center 3 cm toward the robot-facing edge relative to the can pick.
# The center of mass remains 5.7 cm inside the tabletop, while the lower
# fingertip at the 10.5 cm grasp radius clears the table edge by about 4.8 cm.
PLATE_PICK_CENTER = (1.2659409754, -1.2017122341)
PLATE_PICK_REGION = (1.260, 1.272, -1.208, -1.196)
PLATE_INITIAL_DROP_M = 0.012
PLATE_SETTLED_Z = DINING_TABLE.surface_z + PLATE_HALF_THICKNESS_M
PLATE_INITIAL_Z = PLATE_SETTLED_Z + PLATE_INITIAL_DROP_M

PICK_APPROACH_DIRECTION = (-0.66596903, 0.74597939)
PLATE_RIM_GRASP_XY = (
    PLATE_PICK_CENTER[0] - (PLATE_RADIUS_M - 0.018) * PICK_APPROACH_DIRECTION[0],
    PLATE_PICK_CENTER[1] - (PLATE_RADIUS_M - 0.018) * PICK_APPROACH_DIRECTION[1],
)

# Registered semantic navigation pose and visually audited basin center.
SINK_RAW_STAGE_POSE = (
    -1.7205498006,
    1.4710070578,
    ROBOT_BASE_Z_M,
    2.3216464567,
)
_SINK_FORWARD = (
    math.cos(SINK_RAW_STAGE_POSE[3]),
    math.sin(SINK_RAW_STAGE_POSE[3]),
)
SINK_MANIPULATION_BASE_POSE = (
    SINK_RAW_STAGE_POSE[0] + 0.138 * _SINK_FORWARD[0],
    SINK_RAW_STAGE_POSE[1] + 0.138 * _SINK_FORWARD[1],
    ROBOT_BASE_Z_M,
    SINK_RAW_STAGE_POSE[3],
)
SINK_CENTER_XY = (-2.4369233721, 2.2386716370)
SINK_YAW_RAD = SINK_RAW_STAGE_POSE[3] - 0.5 * math.pi
SINK_OPENING_SIZE_M = (0.56, 0.42)
SINK_COUNTER_SIZE_M = (1.80, 0.72)
SINK_COUNTER_SURFACE_Z_M = 0.90
SINK_BOTTOM_SURFACE_Z_M = 0.736
SINK_PLACE_CENTER = SINK_CENTER_XY
SINK_PLACE_SETTLED_Z = SINK_BOTTOM_SURFACE_Z_M + PLATE_HALF_THICKNESS_M
# The user-approved retention aid guides the already centered, horizontal plate
# to this height before release, avoiding an unreliable free drop.
SINK_RELEASE_CLEARANCE_M = 0.025

# This task reaches the kitchen fixture, which is outside the intentionally
# cropped dining-to-snack can-task overlay.  Keep the expansion isolated here.
SINK_PHYSICS_REGION = Bounds2D(-4.0, 7.0, -4.0, 4.0)
SINK_MAP_MASK_RADIUS_M = 1.10
DINING_TABLE_MAP_MASK_RADIUS_M = 0.95


@dataclass(frozen=True)
class CollisionBox:
    """One oriented cuboid shared by PhysX and CuRobo."""

    name: str
    center_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    yaw: float = 0.0


def offset_local(
    center_xy: tuple[float, float],
    yaw: float,
    local_x: float,
    local_y: float,
) -> tuple[float, float]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return (
        center_xy[0] + cosine * local_x - sine * local_y,
        center_xy[1] + sine * local_x + cosine * local_y,
    )


def _box(
    name: str,
    local_xy: tuple[float, float],
    size_xyz: tuple[float, float, float],
    center_z: float,
) -> CollisionBox:
    return CollisionBox(
        name,
        (*offset_local(SINK_CENTER_XY, SINK_YAW_RAD, *local_xy), center_z),
        size_xyz,
        SINK_YAW_RAD,
    )


_opening_x, _opening_y = SINK_OPENING_SIZE_M
_counter_x, _counter_y = SINK_COUNTER_SIZE_M
_counter_thickness = 0.06
_counter_center_z = SINK_COUNTER_SURFACE_Z_M - 0.5 * _counter_thickness
_side_width = 0.5 * (_counter_x - _opening_x)
_front_depth = 0.5 * (_counter_y - _opening_y)
_wall_thickness = 0.025
_wall_height = SINK_COUNTER_SURFACE_Z_M - SINK_BOTTOM_SURFACE_Z_M
_wall_center_z = SINK_BOTTOM_SURFACE_Z_M + 0.5 * _wall_height

SINK_COLLISION_BOXES = (
    _box(
        "sink_counter_left",
        (-0.5 * (_opening_x + _side_width), 0.0),
        (_side_width, _counter_y, _counter_thickness),
        _counter_center_z,
    ),
    _box(
        "sink_counter_right",
        (0.5 * (_opening_x + _side_width), 0.0),
        (_side_width, _counter_y, _counter_thickness),
        _counter_center_z,
    ),
    _box(
        "sink_counter_front",
        (0.0, -0.5 * (_opening_y + _front_depth)),
        (_opening_x, _front_depth, _counter_thickness),
        _counter_center_z,
    ),
    _box(
        "sink_counter_back",
        (0.0, 0.5 * (_opening_y + _front_depth)),
        (_opening_x, _front_depth, _counter_thickness),
        _counter_center_z,
    ),
    _box(
        "sink_bottom",
        (0.0, 0.0),
        (_opening_x, _opening_y, 0.02),
        SINK_BOTTOM_SURFACE_Z_M - 0.01,
    ),
    _box(
        "sink_wall_left",
        (-0.5 * _opening_x, 0.0),
        (_wall_thickness, _opening_y, _wall_height),
        _wall_center_z,
    ),
    _box(
        "sink_wall_right",
        (0.5 * _opening_x, 0.0),
        (_wall_thickness, _opening_y, _wall_height),
        _wall_center_z,
    ),
    _box(
        "sink_wall_front",
        (0.0, -0.5 * _opening_y),
        (_opening_x, _wall_thickness, _wall_height),
        _wall_center_z,
    ),
    _box(
        "sink_wall_back",
        (0.0, 0.5 * _opening_y),
        (_opening_x, _wall_thickness, _wall_height),
        _wall_center_z,
    ),
)

PLANNER_COLLISION_BOXES = (
    CollisionBox(
        DINING_TABLE.name,
        DINING_TABLE.center_xyz,
        DINING_TABLE.size_xyz,
        DINING_TABLE.yaw,
    ),
    *SINK_COLLISION_BOXES,
)


def _load_expanded_segments() -> tuple[CollisionSegment, ...]:
    path = os.path.join(PACKAGE_DIR, LABAPT_VECTOR_MAP_RELATIVE)
    raw = np.loadtxt(path, delimiter=",").reshape(-1, 2, 2)
    transformed = transform_labapt_xy(raw)
    keep = (
        (transformed[:, :, 0].max(axis=1) >= SINK_PHYSICS_REGION.xmin)
        & (transformed[:, :, 0].min(axis=1) <= SINK_PHYSICS_REGION.xmax)
        & (transformed[:, :, 1].max(axis=1) >= SINK_PHYSICS_REGION.ymin)
        & (transformed[:, :, 1].min(axis=1) <= SINK_PHYSICS_REGION.ymax)
    )
    return tuple(
        CollisionSegment(
            name=f"sink_map_{source_index:04d}",
            start_xy=tuple(float(value) for value in transformed[source_index, 0]),
            end_xy=tuple(float(value) for value in transformed[source_index, 1]),
        )
        for source_index in np.flatnonzero(keep)
    )


EXPANDED_COLLISION_SEGMENTS = _load_expanded_segments()


def distance_to_segment_xy(
    point: tuple[float, float], segment: CollisionSegment
) -> float:
    query = np.asarray(point, dtype=np.float64)
    start = np.asarray(segment.start_xy, dtype=np.float64)
    end = np.asarray(segment.end_xy, dtype=np.float64)
    delta = end - start
    length_squared = float(delta @ delta)
    if length_squared <= 1.0e-12:
        return float(np.linalg.norm(query - start))
    fraction = float(np.clip(((query - start) @ delta) / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(query - (start + fraction * delta)))


def is_manipulation_fixture_segment(segment: CollisionSegment) -> bool:
    return (
        distance_to_segment_xy(DINING_TABLE.center_xyz[:2], segment)
        <= DINING_TABLE_MAP_MASK_RADIUS_M
        or distance_to_segment_xy(SINK_CENTER_XY, segment)
        <= SINK_MAP_MASK_RADIUS_M
    )


ACTIVE_COLLISION_SEGMENTS = tuple(
    segment
    for segment in EXPANDED_COLLISION_SEGMENTS
    if not is_manipulation_fixture_segment(segment)
)


def _counter_boundary_segments() -> tuple[CollisionSegment, ...]:
    hx, hy = 0.5 * _counter_x, 0.5 * _counter_y
    corners = (
        offset_local(SINK_CENTER_XY, SINK_YAW_RAD, -hx, -hy),
        offset_local(SINK_CENTER_XY, SINK_YAW_RAD, hx, -hy),
        offset_local(SINK_CENTER_XY, SINK_YAW_RAD, hx, hy),
        offset_local(SINK_CENTER_XY, SINK_YAW_RAD, -hx, hy),
    )
    return tuple(
        CollisionSegment(
            name=f"sink_counter_edge_{index}",
            start_xy=corners[index],
            end_xy=corners[(index + 1) % len(corners)],
        )
        for index in range(len(corners))
    )


NAVIGATION_COLLISION_SEGMENTS = (
    *ACTIVE_COLLISION_SEGMENTS,
    *_counter_boundary_segments(),
)


def distance_to_navigation_segments(points: np.ndarray) -> np.ndarray:
    query = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    minimum = np.full(len(query), np.inf, dtype=np.float64)
    for segment in NAVIGATION_COLLISION_SEGMENTS:
        start = np.asarray(segment.start_xy)
        end = np.asarray(segment.end_xy)
        delta = end - start
        length_squared = float(delta @ delta)
        if length_squared <= 1.0e-12:
            distances = np.linalg.norm(query - start, axis=1)
        else:
            alpha = np.clip(((query - start) @ delta) / length_squared, 0.0, 1.0)
            closest = start + alpha[:, None] * delta
            distances = np.linalg.norm(query - closest, axis=1)
        minimum = np.minimum(minimum, distances)
    return minimum


def required_line_clearance(
    robot_radius: float, safety_margin: float = DEFAULT_SAFETY_MARGIN_M
) -> float:
    return robot_radius + safety_margin + 0.5 * PROXY_WIDTH_M


def sink_base_pose_is_clear(
    point: tuple[float, float],
    *,
    robot_radius: float,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
) -> bool:
    boundary_margin = robot_radius + safety_margin
    return SINK_PHYSICS_REGION.contains(point, margin=boundary_margin) and bool(
        distance_to_navigation_segments(np.asarray([point]))[0]
        >= required_line_clearance(robot_radius, safety_margin)
    )


@dataclass(frozen=True)
class NavigationGrid:
    resolution: float
    xs: np.ndarray
    ys: np.ndarray
    free: np.ndarray

    def nearest_cell(
        self, point: tuple[float, float], *, require_free: bool
    ) -> tuple[int, int]:
        x, y = point
        distances = (self.xs[None, :] - x) ** 2 + (self.ys[:, None] - y) ** 2
        if require_free:
            distances = np.where(self.free, distances, np.inf)
        flat = int(np.argmin(distances))
        if not np.isfinite(distances.flat[flat]):
            raise RuntimeError(f"No free navigation cell for {point}")
        return np.unravel_index(flat, distances.shape)

    def cell_xy(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, column = cell
        return float(self.xs[column]), float(self.ys[row])


def build_sink_navigation_grid(
    robot_radius: float,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
    resolution: float = NAVIGATION_RESOLUTION_M,
) -> NavigationGrid:
    xs = np.arange(
        SINK_PHYSICS_REGION.xmin + 0.5 * resolution,
        SINK_PHYSICS_REGION.xmax,
        resolution,
    )
    ys = np.arange(
        SINK_PHYSICS_REGION.ymin + 0.5 * resolution,
        SINK_PHYSICS_REGION.ymax,
        resolution,
    )
    x_grid, y_grid = np.meshgrid(xs, ys)
    points = np.stack((x_grid.ravel(), y_grid.ravel()), axis=1)
    clearance = distance_to_navigation_segments(points).reshape(len(ys), len(xs))
    boundary_margin = robot_radius + safety_margin
    inside = (
        (x_grid >= SINK_PHYSICS_REGION.xmin + boundary_margin)
        & (x_grid <= SINK_PHYSICS_REGION.xmax - boundary_margin)
        & (y_grid >= SINK_PHYSICS_REGION.ymin + boundary_margin)
        & (y_grid <= SINK_PHYSICS_REGION.ymax - boundary_margin)
    )
    free = inside & (
        clearance >= required_line_clearance(robot_radius, safety_margin)
    )
    return NavigationGrid(resolution, xs, ys, free)


def _astar_cells(
    grid: NavigationGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    neighbors = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    cost = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for row_delta, column_delta in neighbors:
            candidate = (
                current[0] + row_delta,
                current[1] + column_delta,
            )
            if not (
                0 <= candidate[0] < len(grid.ys)
                and 0 <= candidate[1] < len(grid.xs)
                and grid.free[candidate]
            ):
                continue
            step = grid.resolution * math.hypot(row_delta, column_delta)
            candidate_cost = cost[current] + step
            if candidate_cost >= cost.get(candidate, math.inf):
                continue
            cost[candidate] = candidate_cost
            parent[candidate] = current
            heuristic = grid.resolution * math.dist(candidate, goal)
            heapq.heappush(frontier, (candidate_cost + heuristic, candidate))
    if goal not in cost:
        raise RuntimeError(f"No path between navigation cells {start} and {goal}")
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def planned_sink_route(
    *,
    robot_radius: float,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
) -> np.ndarray:
    start_xy = STAGING_POSES["dining_table"][:2]
    goal_xy = SINK_MANIPULATION_BASE_POSE[:2]
    grid = build_sink_navigation_grid(robot_radius, safety_margin)
    if not sink_base_pose_is_clear(
        start_xy, robot_radius=robot_radius, safety_margin=safety_margin
    ):
        raise RuntimeError(f"Unsafe dining start pose: {start_xy}")
    if not sink_base_pose_is_clear(
        goal_xy, robot_radius=robot_radius, safety_margin=safety_margin
    ):
        raise RuntimeError(f"Unsafe sink goal pose: {goal_xy}")
    start_cell = grid.nearest_cell(start_xy, require_free=True)
    goal_cell = grid.nearest_cell(goal_xy, require_free=True)
    cells = _astar_cells(grid, start_cell, goal_cell)
    points = np.asarray([grid.cell_xy(cell) for cell in cells], dtype=np.float64)
    raw_route = np.vstack((np.asarray(start_xy), points, np.asarray(goal_xy)))
    return simplify_route(
        raw_route,
        required_clearance=required_line_clearance(robot_radius, safety_margin),
    )


def simplify_route(
    path: np.ndarray,
    *,
    required_clearance: float,
    sample_spacing: float = 0.02,
) -> np.ndarray:
    """Greedily remove grid corners while preserving line clearance.

    Driving every eight-connected A* cell made the base change direction every
    few centimetres.  Those commands remain collision-safe geometrically but
    excite a retained payload.  Farthest-visible shortcutting produces long,
    smooth follower segments and checks the complete shortcut against the
    exact same vector-map clearance field.
    """

    route = np.asarray(path, dtype=np.float64)
    if len(route) <= 2:
        return route.copy()
    simplified = [route[0]]
    start_index = 0
    while start_index < len(route) - 1:
        end_index = len(route) - 1
        while end_index > start_index + 1:
            distance = float(np.linalg.norm(route[end_index] - route[start_index]))
            count = max(2, int(math.ceil(distance / sample_spacing)) + 1)
            samples = np.linspace(route[start_index], route[end_index], count)
            if float(distance_to_navigation_segments(samples).min()) >= required_clearance:
                break
            end_index -= 1
        simplified.append(route[end_index])
        start_index = end_index
    return np.asarray(simplified, dtype=np.float64)


def route_minimum_clearance(path: np.ndarray) -> float:
    samples = []
    for start, end in zip(path[:-1], path[1:]):
        count = max(2, int(math.ceil(math.dist(start, end) / 0.02)) + 1)
        samples.append(np.linspace(start, end, count))
    return float(
        distance_to_navigation_segments(np.concatenate(samples, axis=0)).min()
    )


def plate_overhang_m(point: tuple[float, float] = PLATE_PICK_CENTER) -> float:
    dx = point[0] - DINING_TABLE.center_xyz[0]
    dy = point[1] - DINING_TABLE.center_xyz[1]
    cosine, sine = math.cos(DINING_TABLE.yaw), math.sin(DINING_TABLE.yaw)
    local_x = cosine * dx + sine * dy
    edge_margin = 0.5 * DINING_TABLE.size_xyz[0] - abs(local_x)
    return PLATE_RADIUS_M - edge_margin


def sink_contains_plate(point: tuple[float, float], margin: float = 0.0) -> bool:
    dx = point[0] - SINK_CENTER_XY[0]
    dy = point[1] - SINK_CENTER_XY[1]
    cosine, sine = math.cos(SINK_YAW_RAD), math.sin(SINK_YAW_RAD)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    required = PLATE_RADIUS_M + margin
    return (
        abs(local_x) <= 0.5 * SINK_OPENING_SIZE_M[0] - required
        and abs(local_y) <= 0.5 * SINK_OPENING_SIZE_M[1] - required
    )


_PLATE_SPHERE_RADIUS_M = 0.053
PLATE_HELD_SPHERES = (
    (0.0, 0.0, 0.0, _PLATE_SPHERE_RADIUS_M),
    *tuple(
        (
            0.10 * math.cos(index * math.pi / 4.0),
            0.10 * math.sin(index * math.pi / 4.0),
            0.0,
            _PLATE_SPHERE_RADIUS_M,
        )
        for index in range(8)
    ),
)
