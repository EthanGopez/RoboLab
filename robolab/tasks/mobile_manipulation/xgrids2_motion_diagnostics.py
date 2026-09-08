"""Pure control-rate motion diagnostics for XGrids2 manipulation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np


CONTINUOUS_JOINT_INDICES = (0, 2, 4, 6)
MAX_ARM_JERK_RAD_S3 = 1500.0
MAX_TARGET_JERK_RAD_S3 = 900.0
OSCILLATION_SPEED_THRESHOLD_RAD_S = 0.08
OSCILLATION_WINDOW_TICKS = 30
MAX_OSCILLATION_REVERSALS_PER_WINDOW = 8


def _branch_delta(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    delta = current - previous
    continuous = np.asarray(CONTINUOUS_JOINT_INDICES, dtype=np.int64)
    delta[continuous] = (delta[continuous] + math.pi) % (2.0 * math.pi) - math.pi
    return delta


def time_parameterized_joint_targets(
    start: Sequence[float],
    positions: Sequence[Sequence[float]],
    *,
    source_dt_s: float,
    control_dt_s: float,
    maximum_target_step_rad: float,
) -> tuple[tuple[float, ...], ...]:
    """Resample a timed CuRobo path onto the physical control clock.

    The whole path is streamed continuously. It is uniformly slowed when
    needed to respect the configured per-control-tick target displacement;
    intermediate planner samples are never used as stop-and-wait waypoints.
    """

    if source_dt_s <= 0.0 or control_dt_s <= 0.0:
        raise ValueError("trajectory and control periods must be positive")
    if maximum_target_step_rad <= 0.0:
        raise ValueError("maximum target step must be positive")
    if not positions:
        raise ValueError("trajectory must contain at least one position")

    start_array = np.asarray(start, dtype=np.float64)
    path = [start_array]
    for position in positions:
        point = np.asarray(position, dtype=np.float64)
        if point.shape != start_array.shape:
            raise ValueError("trajectory positions must match the start shape")
        previous = path[-1]
        point = previous + _branch_delta(point.copy(), previous)
        if float(np.max(np.abs(point - previous))) <= 1.0e-12:
            continue
        path.append(point)
    if len(path) == 1:
        return (tuple(float(value) for value in path[0]),)
    points = np.stack(path)
    source_times = np.arange(len(points), dtype=np.float64) * source_dt_s
    maximum_source_step = float(np.max(np.abs(np.diff(points, axis=0))))
    allowed_source_step = maximum_target_step_rad * source_dt_s / control_dt_s
    time_scale = max(1.0, maximum_source_step / allowed_source_step)
    duration_s = float(source_times[-1] * time_scale)
    tick_count = max(1, int(math.ceil(duration_s / control_dt_s)))
    output_times = np.linspace(0.0, duration_s, tick_count + 1)[1:]
    query_times = output_times / time_scale
    resampled = np.stack(
        [np.interp(query_times, source_times, points[:, joint]) for joint in range(points.shape[1])],
        axis=1,
    )
    resampled[-1] = points[-1]
    return tuple(tuple(float(value) for value in row) for row in resampled)


def minimum_jerk_joint_targets(
    start: Sequence[float],
    target: Sequence[float],
    *,
    maximum_target_step_rad: float,
    wrap_continuous_delta: bool = True,
) -> tuple[tuple[float, ...], ...]:
    """Build a zero-velocity/acceleration point-to-point target stream."""

    if maximum_target_step_rad <= 0.0:
        raise ValueError("maximum target step must be positive")
    start_array = np.asarray(start, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if start_array.shape != target_array.shape:
        raise ValueError("target must match the start shape")
    delta = target_array - start_array
    if wrap_continuous_delta:
        delta = _branch_delta(target_array.copy(), start_array)
    desired = start_array + delta
    excursion = float(np.max(np.abs(desired - start_array)))
    if excursion <= 1.0e-12:
        return (tuple(float(value) for value in desired),)
    # The quintic blend has a peak normalized slope of 1.875. Account for it
    # so even the middle of the transition obeys the per-tick target bound.
    tick_count = max(1, int(math.ceil(1.875 * excursion / maximum_target_step_rad)))
    normalized_time = np.linspace(0.0, 1.0, tick_count + 1)[1:]
    blend = (
        10.0 * normalized_time**3
        - 15.0 * normalized_time**4
        + 6.0 * normalized_time**5
    )
    positions = start_array + blend[:, None] * (desired - start_array)
    positions[-1] = desired
    return tuple(tuple(float(value) for value in row) for row in positions)


@dataclass
class ArmMotionDiagnostics:
    """Incremental, bounded-memory control trace and smoothness gates."""

    control_dt_s: float
    initial_position: Sequence[float]
    previous_position: np.ndarray = field(init=False)
    previous_target: np.ndarray = field(init=False)
    previous_velocity: np.ndarray = field(init=False)
    previous_acceleration: np.ndarray = field(init=False)
    previous_target_velocity: np.ndarray = field(init=False)
    previous_target_acceleration: np.ndarray = field(init=False)
    velocity_window: deque[np.ndarray] = field(init=False)
    tick: int = 0
    maximum_acceleration_rad_s2: float = 0.0
    maximum_jerk_rad_s3: float = 0.0
    maximum_target_jerk_rad_s3: float = 0.0
    maximum_oscillation_reversals: int = 0
    jerk_violation: bool = False
    oscillation_violation: bool = False
    latest_sample: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_position, dtype=np.float64)
        self.previous_position = initial.copy()
        self.previous_target = initial.copy()
        self.previous_velocity = np.zeros_like(initial)
        self.previous_acceleration = np.zeros_like(initial)
        self.previous_target_velocity = np.zeros_like(initial)
        self.previous_target_acceleration = np.zeros_like(initial)
        self.velocity_window = deque(maxlen=OSCILLATION_WINDOW_TICKS)

    @staticmethod
    def _reversal_count(window: Sequence[np.ndarray]) -> int:
        if len(window) < 2:
            return 0
        values = np.stack(window)
        maximum = 0
        for joint in range(values.shape[1]):
            series = values[:, joint]
            active = series[np.abs(series) >= OSCILLATION_SPEED_THRESHOLD_RAD_S]
            if len(active) < 2:
                continue
            reversals = int(np.sum(active[1:] * active[:-1] < 0.0))
            maximum = max(maximum, reversals)
        return maximum

    def update(
        self,
        *,
        target: Sequence[float],
        position: Sequence[float],
        raw_velocity: Sequence[float],
        retention_active: bool,
        phase: str,
    ) -> dict:
        target_array = np.asarray(target, dtype=np.float64)
        position_array = np.asarray(position, dtype=np.float64)
        velocity = _branch_delta(position_array.copy(), self.previous_position) / self.control_dt_s
        target_velocity = _branch_delta(target_array.copy(), self.previous_target) / self.control_dt_s
        acceleration = (velocity - self.previous_velocity) / self.control_dt_s
        jerk = (acceleration - self.previous_acceleration) / self.control_dt_s
        target_acceleration = (
            target_velocity - self.previous_target_velocity
        ) / self.control_dt_s
        target_jerk = (
            target_acceleration - self.previous_target_acceleration
        ) / self.control_dt_s
        self.velocity_window.append(velocity.copy())
        reversals = self._reversal_count(tuple(self.velocity_window))

        # The first two updates establish finite-difference history rather than
        # representing physical jerk. All samples are still written to trace.
        if self.tick >= 2:
            acceleration_max = float(np.max(np.abs(acceleration)))
            jerk_max = float(np.max(np.abs(jerk)))
            target_jerk_max = float(np.max(np.abs(target_jerk)))
            self.maximum_acceleration_rad_s2 = max(
                self.maximum_acceleration_rad_s2, acceleration_max
            )
            self.maximum_jerk_rad_s3 = max(self.maximum_jerk_rad_s3, jerk_max)
            self.maximum_target_jerk_rad_s3 = max(
                self.maximum_target_jerk_rad_s3, target_jerk_max
            )
            self.maximum_oscillation_reversals = max(
                self.maximum_oscillation_reversals, reversals
            )
            self.jerk_violation |= (
                jerk_max > MAX_ARM_JERK_RAD_S3
                or target_jerk_max > MAX_TARGET_JERK_RAD_S3
            )
            self.oscillation_violation |= (
                reversals > MAX_OSCILLATION_REVERSALS_PER_WINDOW
            )

        self.tick += 1
        self.latest_sample = {
            "tick": self.tick,
            "time_s": self.tick * self.control_dt_s,
            "phase": phase,
            "target_position_rad": target_array.tolist(),
            "position_rad": position_array.tolist(),
            "velocity_rad_s": velocity.tolist(),
            "raw_physx_velocity_rad_s": list(raw_velocity),
            "acceleration_rad_s2": acceleration.tolist(),
            "jerk_rad_s3": jerk.tolist(),
            "target_velocity_rad_s": target_velocity.tolist(),
            "target_acceleration_rad_s2": target_acceleration.tolist(),
            "target_jerk_rad_s3": target_jerk.tolist(),
            "retention_active": bool(retention_active),
            "rolling_oscillation_reversals": reversals,
        }
        self.previous_position = position_array.copy()
        self.previous_target = target_array.copy()
        self.previous_velocity = velocity
        self.previous_acceleration = acceleration
        self.previous_target_velocity = target_velocity
        self.previous_target_acceleration = target_acceleration
        return self.latest_sample

    def summary(self) -> dict:
        return {
            "control_ticks": self.tick,
            "maximum_acceleration_rad_s2": self.maximum_acceleration_rad_s2,
            "maximum_jerk_rad_s3": self.maximum_jerk_rad_s3,
            "maximum_target_jerk_rad_s3": self.maximum_target_jerk_rad_s3,
            "maximum_oscillation_reversals_per_window": self.maximum_oscillation_reversals,
            "jerk_limit_rad_s3": MAX_ARM_JERK_RAD_S3,
            "target_jerk_limit_rad_s3": MAX_TARGET_JERK_RAD_S3,
            "oscillation_window_ticks": OSCILLATION_WINDOW_TICKS,
            "oscillation_reversal_limit": MAX_OSCILLATION_REVERSALS_PER_WINDOW,
            "jerk_violation": self.jerk_violation,
            "oscillation_violation": self.oscillation_violation,
        }
