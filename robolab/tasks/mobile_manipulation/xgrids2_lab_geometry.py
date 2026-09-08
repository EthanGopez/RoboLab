"""Physics and navigation contract for the route-first XGRIDS2 lab.

The reconstructed USD remains appearance-only.  This module derives a small,
deterministic collision/navigation overlay from the registered LabApt vector
map and exposes robot-independent obstacle segments.  Robot footprint and
safety margin are parameters of the navigation functions.
"""

from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass

import numpy as np

from robolab.constants import PACKAGE_DIR


XGRIDS_VISUAL_USD_RELATIVE = (
    "assets/scenes/xgrids2_lab/new_data_0827/isaacsim-import-package/"
    "ut_lab_1_scene_pack/0/ut_lab_1.usdz"
)
# Register the vendor visual floor (approximately z=-1.075 m) to physics z=0.
XGRIDS_VISUAL_Z_OFFSET_M = 1.075
LABAPT_VECTOR_MAP_RELATIVE = (
    "assets/scenes/xgrids2_lab/labapt_reference/LabApt.vectormap.txt"
)

# Audited map registration.  See docs/XGRIDS2_LABAPT_ALIGNMENT.md.
LABAPT_TO_XGRIDS_THETA_RAD = 0.6216464567421395
LABAPT_TO_XGRIDS_TRANSLATION_M = (-13.259475652851433, 6.588388241325163)
XGRIDS_FLOOR_Z = 0.0


@dataclass(frozen=True)
class Bounds2D:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) * 0.5, (self.ymin + self.ymax) * 0.5)

    @property
    def size(self) -> tuple[float, float]:
        return (self.xmax - self.xmin, self.ymax - self.ymin)

    def contains(self, point: tuple[float, float], margin: float = 0.0) -> bool:
        x, y = point
        return (
            self.xmin + margin <= x <= self.xmax - margin
            and self.ymin + margin <= y <= self.ymax - margin
        )


@dataclass(frozen=True)
class CollisionSegment:
    """One thin, static, vertical collision proxy centered on a map segment."""

    name: str
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]

    @property
    def center_xy(self) -> tuple[float, float]:
        return (
            0.5 * (self.start_xy[0] + self.end_xy[0]),
            0.5 * (self.start_xy[1] + self.end_xy[1]),
        )

    @property
    def length(self) -> float:
        return math.dist(self.start_xy, self.end_xy)

    @property
    def yaw(self) -> float:
        return math.atan2(
            self.end_xy[1] - self.start_xy[1],
            self.end_xy[0] - self.start_xy[0],
        )


@dataclass(frozen=True)
class NavigationGrid:
    resolution: float
    xs: np.ndarray
    ys: np.ndarray
    clearance_m: np.ndarray
    free: np.ndarray

    def nearest_cell(self, point: tuple[float, float], *, require_free: bool) -> tuple[int, int]:
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
        return (float(self.xs[column]), float(self.ys[row]))


# Initial physics scope: the dining/snack route plus enough surrounding context
# to prevent paths from escaping around un-authored obstacles.
PHYSICS_REGION = Bounds2D(-1.0, 7.0, -4.0, 2.0)
FLOOR_THICKNESS_M = 0.10
FLOOR_CENTER_Z_M = -0.5 * FLOOR_THICKNESS_M
PROXY_WIDTH_M = 0.06
PROXY_HEIGHT_M = 1.50
PROXY_CENTER_Z_M = 0.5 * PROXY_HEIGHT_M
NAVIGATION_RESOLUTION_M = 0.08

# The drive-base collision cylinder is radius 0.30 m.  Navigation uses the
# measured *whole-robot transport envelope*: the limit-safe folded Kinova has
# a 0.305 m maximum link-center radius, rounded up to 0.36 m to include link
# thickness and modeling tolerance.  Environment geometry remains independent
# of this choice; callers may provide another embodiment's radius.
MOBILE_KINOVA_BASE_RADIUS_M = 0.30
MOBILE_KINOVA_TRANSPORT_RADIUS_M = 0.36
DEFAULT_SAFETY_MARGIN_M = 0.05
DEFAULT_REQUIRED_LINE_CLEARANCE_M = (
    MOBILE_KINOVA_TRANSPORT_RADIUS_M + DEFAULT_SAFETY_MARGIN_M + 0.5 * PROXY_WIDTH_M
)
ROBOT_BASE_Z_M = 0.468

# Directly transformed real navigation goals, retained as provenance.
RAW_STAGING_POSES = {
    "dining_table": (1.765589517190083, -1.7639477574242441, ROBOT_BASE_Z_M, 2.1916464567421396),
    "snack_table": (4.751903727995076, -0.8669932243824894, ROBOT_BASE_Z_M, -0.9483535432578606),
}

# The snack goal is only 0.269 m from the closest mapped obstacle.  Shift it
# 0.220 m to a nearby cell satisfying the full 0.44 m whole-robot clearance
# contract.  Heading is unchanged.
STAGING_POSES = {
    "dining_table": RAW_STAGING_POSES["dining_table"],
    "snack_table": (4.56, -0.76, ROBOT_BASE_Z_M, RAW_STAGING_POSES["snack_table"][3]),
}


def transform_labapt_xy(points: np.ndarray) -> np.ndarray:
    theta = LABAPT_TO_XGRIDS_THETA_RAD
    cosine, sine = math.cos(theta), math.sin(theta)
    matrix = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ matrix.T + np.asarray(
        LABAPT_TO_XGRIDS_TRANSLATION_M
    )


def _load_route_collision_segments() -> tuple[CollisionSegment, ...]:
    path = os.path.join(PACKAGE_DIR, LABAPT_VECTOR_MAP_RELATIVE)
    raw = np.loadtxt(path, delimiter=",").reshape(-1, 2, 2)
    transformed = transform_labapt_xy(raw)
    keep = (
        (transformed[:, :, 0].max(axis=1) >= PHYSICS_REGION.xmin)
        & (transformed[:, :, 0].min(axis=1) <= PHYSICS_REGION.xmax)
        & (transformed[:, :, 1].max(axis=1) >= PHYSICS_REGION.ymin)
        & (transformed[:, :, 1].min(axis=1) <= PHYSICS_REGION.ymax)
    )
    selected = transformed[keep]
    return tuple(
        CollisionSegment(
            name=f"map_{index:03d}",
            start_xy=tuple(float(value) for value in segment[0]),
            end_xy=tuple(float(value) for value in segment[1]),
        )
        for index, segment in enumerate(selected)
    )


COLLISION_SEGMENTS = _load_route_collision_segments()


def distance_to_segments(
    points: np.ndarray,
    collision_segments: tuple[CollisionSegment, ...] | None = None,
) -> np.ndarray:
    """Return minimum XY distance from each point to the collision centerlines."""

    query = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    minimum = np.full(len(query), np.inf, dtype=np.float64)
    segments = COLLISION_SEGMENTS if collision_segments is None else collision_segments
    for segment in segments:
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
    robot_radius: float = MOBILE_KINOVA_TRANSPORT_RADIUS_M,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
) -> float:
    return robot_radius + safety_margin + 0.5 * PROXY_WIDTH_M


def base_pose_is_clear(
    point: tuple[float, float],
    robot_radius: float = MOBILE_KINOVA_TRANSPORT_RADIUS_M,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
    collision_segments: tuple[CollisionSegment, ...] | None = None,
) -> bool:
    required = required_line_clearance(robot_radius, safety_margin)
    return PHYSICS_REGION.contains(point, margin=robot_radius + safety_margin) and bool(
        distance_to_segments(np.asarray([point]), collision_segments)[0] >= required
    )


def build_navigation_grid(
    robot_radius: float = MOBILE_KINOVA_TRANSPORT_RADIUS_M,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
    resolution: float = NAVIGATION_RESOLUTION_M,
    collision_segments: tuple[CollisionSegment, ...] | None = None,
) -> NavigationGrid:
    """Build free space from the same proxy centerlines consumed by PhysX."""

    xs = np.arange(PHYSICS_REGION.xmin + 0.5 * resolution, PHYSICS_REGION.xmax, resolution)
    ys = np.arange(PHYSICS_REGION.ymin + 0.5 * resolution, PHYSICS_REGION.ymax, resolution)
    x_grid, y_grid = np.meshgrid(xs, ys)
    points = np.stack((x_grid.ravel(), y_grid.ravel()), axis=1)
    clearance = distance_to_segments(points, collision_segments).reshape(
        len(ys), len(xs)
    )
    boundary_margin = robot_radius + safety_margin
    inside = (
        (x_grid >= PHYSICS_REGION.xmin + boundary_margin)
        & (x_grid <= PHYSICS_REGION.xmax - boundary_margin)
        & (y_grid >= PHYSICS_REGION.ymin + boundary_margin)
        & (y_grid <= PHYSICS_REGION.ymax - boundary_margin)
    )
    free = inside & (clearance >= required_line_clearance(robot_radius, safety_margin))
    return NavigationGrid(resolution, xs, ys, clearance, free)


def _astar_cells(
    grid: NavigationGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    neighbors = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    cost = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for d_row, d_column in neighbors:
            candidate = (current[0] + d_row, current[1] + d_column)
            if not (
                0 <= candidate[0] < len(grid.ys)
                and 0 <= candidate[1] < len(grid.xs)
                and grid.free[candidate]
            ):
                continue
            step = grid.resolution * math.hypot(d_row, d_column)
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


def planned_route(
    start_name: str = "dining_table",
    goal_name: str = "snack_table",
    robot_radius: float = MOBILE_KINOVA_TRANSPORT_RADIUS_M,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
) -> np.ndarray:
    """Return a footprint-inflated A* path with exact staging endpoints."""

    return planned_route_between(
        STAGING_POSES[start_name][:2],
        STAGING_POSES[goal_name][:2],
        robot_radius=robot_radius,
        safety_margin=safety_margin,
    )


def planned_route_between(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    robot_radius: float = MOBILE_KINOVA_TRANSPORT_RADIUS_M,
    safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
    collision_segments: tuple[CollisionSegment, ...] | None = None,
) -> np.ndarray:
    """Return a footprint-inflated A* path between explicit safe endpoints."""

    grid = build_navigation_grid(
        robot_radius, safety_margin, collision_segments=collision_segments
    )
    if not base_pose_is_clear(
        start_xy, robot_radius, safety_margin, collision_segments
    ):
        raise RuntimeError(f"Unsafe start pose: {start_xy}")
    if not base_pose_is_clear(
        goal_xy, robot_radius, safety_margin, collision_segments
    ):
        raise RuntimeError(f"Unsafe goal pose: {goal_xy}")
    start_cell = grid.nearest_cell(start_xy, require_free=True)
    goal_cell = grid.nearest_cell(goal_xy, require_free=True)
    cells = _astar_cells(grid, start_cell, goal_cell)
    points = np.asarray([grid.cell_xy(cell) for cell in cells], dtype=np.float64)
    return np.vstack((np.asarray(start_xy), points, np.asarray(goal_xy)))


def route_minimum_clearance(path: np.ndarray) -> float:
    """Conservatively sample every path segment at 2 cm spacing."""

    samples = []
    for start, end in zip(path[:-1], path[1:]):
        count = max(2, int(math.ceil(math.dist(start, end) / 0.02)) + 1)
        samples.append(np.linspace(start, end, count))
    return float(distance_to_segments(np.concatenate(samples)).min())
