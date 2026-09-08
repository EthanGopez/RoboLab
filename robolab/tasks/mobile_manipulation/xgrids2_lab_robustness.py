"""Deterministic trial generation and quality gates for XGRIDS2 manipulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Mapping, Sequence

from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import CAN_RADIUS
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import STAGING_POSES
from robolab.tasks.mobile_manipulation.xgrids2_lab_manipulation_geometry import (
    PICK_CENTER,
    PICK_REGION,
    PLACE_CENTER,
    PLACE_REGION,
    snack_table_contains_xy,
)


DEFAULT_TRIAL_COUNT = 20
DEFAULT_MINIMUM_SUCCESS_RATE = 0.90
MAX_QUALITY_ARM_STEP_RAD = 0.05
# Instantaneous PhysX velocity can briefly spike inside one control tick under
# the retained payload/contact constraint. The independent 0.05-rad measured
# control-tick displacement gate rejects visible discontinuities; 3.5 rad/s
# remains a diagnostic ceiling without misclassifying a substep solver impulse.
MAX_QUALITY_ARM_SPEED_RAD_S = 3.50
MAX_PLACE_TARGET_ERROR_M = 0.08
MAX_BASE_POSITION_ERROR_M = 0.055
MAX_BASE_YAW_ERROR_RAD = 0.06
BASE_XY_JITTER_M = 0.025
BASE_YAW_JITTER_RAD = math.radians(2.0)
ARM_JOINT_JITTER_RAD = 0.025
GOLDEN_PICK_XY = (1.2459619045257568, -1.1793328523635864)


@dataclass(frozen=True)
class XGRIDS2RobustnessTrial:
    """One reproducible perturbation of the working XGRIDS2 task."""

    index: int
    seed: int
    pick_xy: tuple[float, float]
    can_yaw: float
    base_xy: tuple[float, float]
    base_yaw: float
    arm_offsets_rad: tuple[float, ...]
    place_xy: tuple[float, float]

    def as_dict(self) -> dict:
        value = asdict(self)
        for key in ("pick_xy", "base_xy", "arm_offsets_rad", "place_xy"):
            value[key] = list(value[key])
        return value


def robustness_trial_schedule(
    count: int = DEFAULT_TRIAL_COUNT,
    *,
    seed: int = 20260829,
) -> tuple[XGRIDS2RobustnessTrial, ...]:
    """Return a stable schedule with one baseline and seeded perturbations."""

    if count < 1:
        raise ValueError("count must be positive")
    bx, by, _, base_yaw = STAGING_POSES["dining_table"]
    trials = [
        XGRIDS2RobustnessTrial(
            index=0,
            seed=0,
            pick_xy=GOLDEN_PICK_XY,
            can_yaw=0.0,
            base_xy=(bx, by),
            base_yaw=base_yaw,
            arm_offsets_rad=(0.0,) * 7,
            place_xy=PLACE_CENTER,
        )
    ]
    rng = random.Random(seed)
    for index in range(1, count):
        trial_seed = rng.randrange(0, 2**31)
        trial_rng = random.Random(trial_seed)
        trials.append(
            XGRIDS2RobustnessTrial(
                index=index,
                seed=trial_seed,
                pick_xy=(
                    trial_rng.uniform(PICK_REGION[0], PICK_REGION[1]),
                    trial_rng.uniform(PICK_REGION[2], PICK_REGION[3]),
                ),
                can_yaw=trial_rng.uniform(-math.pi, math.pi),
                base_xy=(
                    bx + trial_rng.uniform(-BASE_XY_JITTER_M, BASE_XY_JITTER_M),
                    by + trial_rng.uniform(-BASE_XY_JITTER_M, BASE_XY_JITTER_M),
                ),
                base_yaw=base_yaw
                + trial_rng.uniform(-BASE_YAW_JITTER_RAD, BASE_YAW_JITTER_RAD),
                arm_offsets_rad=tuple(
                    trial_rng.uniform(-ARM_JOINT_JITTER_RAD, ARM_JOINT_JITTER_RAD)
                    for _ in range(7)
                ),
                place_xy=(
                    trial_rng.uniform(PLACE_REGION[0], PLACE_REGION[1]),
                    trial_rng.uniform(PLACE_REGION[2], PLACE_REGION[3]),
                ),
            )
        )
    return tuple(trials)


def classify_failure(result: Mapping) -> str:
    """Map a detailed oracle failure to a stable benchmark category."""

    if bool(result.get("passed")):
        return ""
    reason = str(result.get("failure_reason", "")).lower()
    phases = list(result.get("phase_history", []))
    phase = phases[-2] if len(phases) >= 2 and phases[-1] == "failure" else (
        phases[-1] if phases else ""
    )
    if "bilateral" in reason or phase.startswith("approach") or phase == "close":
        return "grasp"
    if "tracking" in reason or "joint step" in reason:
        return "arm_tracking"
    if phase == "traverse" or "base follower" in reason or "free-space route" in reason:
        return "transport"
    if phase in {"release", "retreat", "place", "preplace"} or "settle" in reason:
        return "placement"
    if "planning failed" in reason or phase in {
        "pregrasp",
        "pick_branch_anchor",
        "lift",
        "retract",
        "carry_tuck",
        "place_reorient_1",
        "place_reorient_2",
        "place_reorient_3",
    }:
        return "planning"
    return "other"


def evaluate_trial(
    trial: XGRIDS2RobustnessTrial,
    oracle_result: Mapping,
) -> dict:
    """Apply benchmark-level quality gates to one oracle result."""

    final = list(oracle_result.get("final_position", []))
    final_on_table = len(final) >= 2 and snack_table_contains_xy(
        (float(final[0]), float(final[1])), margin=CAN_RADIUS
    )
    checks = {
        "oracle_completed_full_task": bool(oracle_result.get("passed"))
        and oracle_result.get("completed_stage") == "full",
        "bilateral_contact_phase_completed": "close"
        in oracle_result.get("phase_history", []),
        "minimum_route_clearance_valid": bool(
            oracle_result.get("minimum_route_clearance_valid", False)
        ),
        "final_can_entirely_on_snack_table": final_on_table,
        "place_target_error_within_limit": float(
            oracle_result.get("place_target_error_m", math.inf)
        )
        <= MAX_PLACE_TARGET_ERROR_M,
        "maximum_arm_step_within_limit": float(
            oracle_result.get("maximum_arm_step_rad", math.inf)
        )
        <= MAX_QUALITY_ARM_STEP_RAD,
        "maximum_arm_speed_within_limit": float(
            oracle_result.get("maximum_arm_speed_rad_s", math.inf)
        )
        <= MAX_QUALITY_ARM_SPEED_RAD_S,
        "base_position_error_within_limit": float(
            oracle_result.get("final_base_position_error_m", math.inf)
        )
        <= MAX_BASE_POSITION_ERROR_M,
        "base_yaw_error_within_limit": float(
            oracle_result.get("final_base_yaw_error_rad", math.inf)
        )
        <= MAX_BASE_YAW_ERROR_RAD,
    }
    passed = all(checks.values())
    failure_category = "" if passed else (
        classify_failure(oracle_result)
        if not bool(oracle_result.get("passed"))
        else "motion_quality"
    )
    return {
        "trial": trial.as_dict(),
        "passed": passed,
        "failure_category": failure_category,
        "quality_checks": checks,
        "oracle_result": dict(oracle_result),
    }


def summarize_robustness(
    schedule: Sequence[XGRIDS2RobustnessTrial],
    records: Sequence[Mapping],
    *,
    minimum_success_rate: float = DEFAULT_MINIMUM_SUCCESS_RATE,
) -> dict:
    """Aggregate completed records and enforce the requested success-rate gate."""

    if not 0.0 < minimum_success_rate <= 1.0:
        raise ValueError("minimum_success_rate must be in (0, 1]")
    successes = sum(bool(record.get("passed")) for record in records)
    completed = len(records)
    requested = len(schedule)
    success_rate = successes / completed if completed else 0.0
    failure_categories: dict[str, int] = {}
    for record in records:
        category = str(record.get("failure_category", ""))
        if category:
            failure_categories[category] = failure_categories.get(category, 0) + 1

    def maximum(metric: str) -> float:
        values = [
            float(record.get("oracle_result", {}).get(metric, 0.0))
            for record in records
        ]
        return max(values, default=0.0)

    return {
        "benchmark": "xgrids2_lab_curobo_pick_place_robustness",
        "requested_trials": requested,
        "completed_trials": completed,
        "successes": successes,
        "failures": completed - successes,
        "success_rate": success_rate,
        "minimum_success_rate": minimum_success_rate,
        "complete": completed == requested,
        "passed": completed == requested and success_rate >= minimum_success_rate,
        "failure_categories": failure_categories,
        "quality_thresholds": {
            "maximum_arm_step_rad": MAX_QUALITY_ARM_STEP_RAD,
            "maximum_arm_speed_rad_s": MAX_QUALITY_ARM_SPEED_RAD_S,
            "maximum_place_target_error_m": MAX_PLACE_TARGET_ERROR_M,
            "maximum_base_position_error_m": MAX_BASE_POSITION_ERROR_M,
            "maximum_base_yaw_error_rad": MAX_BASE_YAW_ERROR_RAD,
            "minimum_route_clearance_valid": True,
            "final_can_entirely_on_snack_table": True,
        },
        "observed_maxima": {
            "maximum_arm_step_rad": maximum("maximum_arm_step_rad"),
            "maximum_arm_speed_rad_s": maximum("maximum_arm_speed_rad_s"),
            "place_target_error_m": maximum("place_target_error_m"),
            "final_base_position_error_m": maximum("final_base_position_error_m"),
            "final_base_yaw_error_rad": maximum("final_base_yaw_error_rad"),
        },
        "schedule": [trial.as_dict() for trial in schedule],
        "records": list(records),
    }
