"""Deterministic, embodiment-configurable waypoint follower for XGRIDS2."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_velocity_to_body(
    velocity_xy_world: np.ndarray, yaw_world: float
) -> np.ndarray:
    """Rotate an XY velocity from world coordinates into the base frame."""

    cosine, sine = math.cos(yaw_world), math.sin(yaw_world)
    vx, vy = np.asarray(velocity_xy_world, dtype=np.float64)
    return np.asarray((cosine * vx + sine * vy, -sine * vx + cosine * vy))


@dataclass
class WaypointFollower:
    """Generate body-frame ``[vx, vy, wz]`` commands along a collision-free path.

    ``sensor_forward_yaw_offset`` expresses the fixed camera optical axis in
    the robot base frame.  It is configuration, not a camera pose command; the
    controller rotates the base until that fixed sensor faces along the route.
    """

    route_xy: np.ndarray
    final_yaw: float
    sensor_forward_yaw_offset: float = 0.0
    max_linear_speed: float = 0.25
    max_angular_speed: float = 0.65
    linear_gain: float = 1.25
    angular_gain: float = 1.8
    waypoint_tolerance: float = 0.11
    final_position_tolerance: float = 0.045
    final_yaw_tolerance: float = 0.05
    waypoint_index: int = 1
    smooth_heading: bool = False
    initial_yaw: float | None = None
    _progress: float = 0.0
    _final_alignment: bool = False

    def __post_init__(self) -> None:
        self.route_xy = np.asarray(self.route_xy, dtype=np.float64)
        if self.route_xy.ndim != 2 or self.route_xy.shape[1] != 2:
            raise ValueError("route_xy must have shape (N, 2)")
        if len(self.route_xy) < 2:
            raise ValueError("route_xy must contain at least two points")

    def command(self, xy_world: np.ndarray, yaw_world: float) -> tuple[np.ndarray, bool]:
        xy = np.asarray(xy_world, dtype=np.float64)
        last = len(self.route_xy) - 1
        while self.waypoint_index < last:
            distance = float(np.linalg.norm(self.route_xy[self.waypoint_index] - xy))
            if distance > self.waypoint_tolerance:
                break
            self.waypoint_index += 1

        delta = self.route_xy[self.waypoint_index] - xy
        distance = float(np.linalg.norm(delta))
        at_final_position = self.waypoint_index == last and distance <= self.final_position_tolerance

        if self.smooth_heading:
            # Holonomic translation does not require chasing every short path
            # segment's bearing. Blend the initial and final chassis headings
            # by monotone route progress, then retain final-yaw alignment.
            if self.initial_yaw is None:
                self.initial_yaw = yaw_world
            lengths = np.linalg.norm(np.diff(self.route_xy, axis=0), axis=1)
            start = self.route_xy[self.waypoint_index-1]
            segment = self.route_xy[self.waypoint_index]-start
            fraction = np.clip(np.dot(xy-start, segment)/max(np.dot(segment, segment), 1e-12), 0, 1)
            progress = (sum(lengths[:self.waypoint_index-1])+fraction*lengths[self.waypoint_index-1])/max(sum(lengths), 1e-9)
            self._progress = max(self._progress, float(progress))
            self._final_alignment |= at_final_position
            t = 1.0 if self._final_alignment else self._progress
            blend = 10*t**3-15*t**4+6*t**5
            desired_yaw = self.initial_yaw+blend*wrap_angle(self.final_yaw-self.initial_yaw)
            angular = float(np.clip(self.angular_gain*wrap_angle(desired_yaw-yaw_world),
                                    -self.max_angular_speed, self.max_angular_speed))
            done = at_final_position and abs(wrap_angle(self.final_yaw-yaw_world)) <= self.final_yaw_tolerance
            # Keep positional feedback while aligning; do not alternate between
            # incompatible route-bearing and final-yaw controllers at a radius.
            speed = min(self.max_linear_speed, self.linear_gain*distance)
            velocity = world_velocity_to_body(delta/max(distance, 1e-9)*speed, yaw_world)
            return np.array([*velocity, angular]), done

        if at_final_position:
            yaw_error = wrap_angle(self.final_yaw - yaw_world)
            angular = float(
                np.clip(self.angular_gain * yaw_error, -self.max_angular_speed, self.max_angular_speed)
            )
            done = abs(yaw_error) <= self.final_yaw_tolerance
            return np.asarray((0.0, 0.0, 0.0 if done else angular)), done

        speed = min(self.max_linear_speed, self.linear_gain * distance)
        velocity_world = delta / max(distance, 1.0e-9) * speed
        velocity_body = world_velocity_to_body(velocity_world, yaw_world)

        route_heading = math.atan2(float(delta[1]), float(delta[0]))
        desired_base_yaw = wrap_angle(route_heading - self.sensor_forward_yaw_offset)
        yaw_error = wrap_angle(desired_base_yaw - yaw_world)
        angular = float(
            np.clip(self.angular_gain * yaw_error, -self.max_angular_speed, self.max_angular_speed)
        )
        return np.asarray((*velocity_body, angular)), False
