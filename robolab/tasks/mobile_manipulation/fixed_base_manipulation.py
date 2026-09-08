"""Planner-neutral contracts for fixed-base kitchen pick and place.

This module deliberately has no CuRobo or Isaac Sim imports. Task geometry,
phase semantics, and gate sampling remain usable by another planner or policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
    Bounds2D,
    CAN_HALF_HEIGHT,
    ISLAND_SURFACE_Z,
    ROBOT_BASE_Z,
)


ARM_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))
FIXED_BASE_PICK_REGION = Bounds2D(-0.63, -0.55, 0.12, 0.23)
FIXED_BASE_LOCAL_PLACE_REGION = Bounds2D(-0.63, -0.55, 0.38, 0.50)
FIXED_BASE_PREGRASP_HEIGHT = 0.06
FIXED_BASE_ISLAND_STAGE = (-1.06, 0.14, ROBOT_BASE_Z, math.pi)
# The island cabinet begins at x=-0.70.  This leaves 0.36 m between its
# face and the base center: 0.30 m footprint radius plus the authored 0.05 m
# safety margin, with 0.01 m numerical clearance.


class FixedBasePhase(str, Enum):
    """Observable phases shared by planner backends and gate reports."""

    OPEN = "open"
    PREGRASP = "pregrasp"
    APPROACH = "approach"
    CLOSE = "close"
    LIFT = "lift"
    RETRACT = "retract"
    PREPLACE = "preplace"
    PLACE = "place"
    RELEASE = "release"
    RETREAT = "retreat"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class Pose:
    """Rigid pose with a scalar-first quaternion."""

    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class JointTrajectory:
    """Planner result expressed only in the seven task arm joints."""

    joint_names: tuple[str, ...]
    positions: tuple[tuple[float, ...], ...]
    planning_time_s: float
    sample_dt_s: float | None = None


class FixedBaseMotionPlanner(Protocol):
    """Minimal backend boundary; no Kinova controller semantics leak through."""

    def plan(
        self,
        start_joint_positions: tuple[float, ...],
        target_pose_base: Pose,
        *,
        allow_hand_contact: bool = False,
    ) -> JointTrajectory: ...

    def set_held_object(self, pose_in_tool: Pose | None) -> None: ...


@dataclass(frozen=True)
class FixedBaseTrial:
    """One deterministic pick/place perturbation pair."""

    index: int
    scale: float
    pick_xy: tuple[float, float]
    place_xy: tuple[float, float]


def _elliptic_sample(region: Bounds2D, scale: float, angle: float) -> tuple[float, float]:
    """Sample an ellipse inscribed in ``region`` at a normalized radius."""

    cx, cy = region.center
    rx, ry = region.size[0] / 2.0, region.size[1] / 2.0
    return (cx + scale * rx * math.cos(angle), cy + scale * ry * math.sin(angle))


def fixed_base_trial_schedule(count: int = 40) -> tuple[FixedBaseTrial, ...]:
    """Return deterministic center-to-edge perturbations for the fidelity gate."""

    if count < 1:
        raise ValueError("count must be positive")
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    trials = []
    for index in range(count):
        scale = 0.0 if count == 1 else 0.95 * math.sqrt(index / (count - 1))
        angle = index * golden_angle
        trials.append(
            FixedBaseTrial(
                index=index,
                scale=scale,
                pick_xy=_elliptic_sample(FIXED_BASE_PICK_REGION, scale, angle),
                # The gate perturbs the initial object pose.  A fixed local
                # destination isolates grasp/replan robustness from a separate
                # placement-goal randomization benchmark.
                place_xy=FIXED_BASE_LOCAL_PLACE_REGION.center,
            )
        )
    return tuple(trials)


def can_center_height(*, hover: float = 0.0) -> float:
    """World Z for an upright can above the island worktop."""

    return ISLAND_SURFACE_Z + CAN_HALF_HEIGHT + hover
