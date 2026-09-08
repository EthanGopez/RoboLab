"""Static tests for the XGRIDS2 randomized CuRobo robustness gate."""

from __future__ import annotations

import math

import pytest

from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import STAGING_POSES
from robolab.tasks.mobile_manipulation.xgrids2_lab_manipulation_geometry import (
    PICK_REGION,
    PLACE_CENTER,
    PLACE_REGION,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_robustness import (
    GOLDEN_PICK_XY,
    MAX_QUALITY_ARM_SPEED_RAD_S,
    MAX_QUALITY_ARM_STEP_RAD,
    classify_failure,
    evaluate_trial,
    robustness_trial_schedule,
    summarize_robustness,
)


def _successful_result(trial) -> dict:
    return {
        "passed": True,
        "completed_stage": "full",
        "failure_reason": "",
        "phase_history": [
            "pregrasp",
            "approach",
            "close",
            "lift",
            "retract",
            "carry_tuck",
            "traverse",
            "place_reorient_1",
            "place_reorient_2",
            "place_reorient_3",
            "preplace",
            "place",
            "release",
            "retreat",
            "full_success",
        ],
        "minimum_route_clearance_valid": True,
        "final_position": [trial.place_xy[0], trial.place_xy[1], 0.763],
        "maximum_arm_step_rad": 0.03,
        "maximum_arm_speed_rad_s": 1.40,
        "final_base_position_error_m": 0.015,
        "final_base_yaw_error_rad": 0.035,
        "place_target_error_m": 0.0,
    }


def test_schedule_is_reproducible_and_keeps_a_golden_baseline():
    left = robustness_trial_schedule(20, seed=17)
    right = robustness_trial_schedule(20, seed=17)
    assert left == right
    assert len(left) == 20
    baseline = left[0]
    assert baseline.pick_xy == GOLDEN_PICK_XY
    assert baseline.seed == 0
    assert baseline.place_xy == PLACE_CENTER
    assert baseline.base_xy == STAGING_POSES["dining_table"][:2]
    assert baseline.base_yaw == STAGING_POSES["dining_table"][3]
    assert baseline.arm_offsets_rad == (0.0,) * 7


def test_randomized_schedule_stays_inside_valid_bounds():
    schedule = robustness_trial_schedule(40, seed=20260829)
    assert len({trial.seed for trial in schedule}) == len(schedule)
    assert any(trial.arm_offsets_rad != (0.0,) * 7 for trial in schedule[1:])
    for trial in schedule:
        assert PICK_REGION[0] <= trial.pick_xy[0] <= PICK_REGION[1]
        assert PICK_REGION[2] <= trial.pick_xy[1] <= PICK_REGION[3]
        assert PLACE_REGION[0] <= trial.place_xy[0] <= PLACE_REGION[1]
        assert PLACE_REGION[2] <= trial.place_xy[1] <= PLACE_REGION[3]
        assert -math.pi <= trial.can_yaw <= math.pi
        assert len(trial.arm_offsets_rad) == 7


def test_trial_quality_gate_accepts_a_complete_smooth_result():
    trial = robustness_trial_schedule(1)[0]
    record = evaluate_trial(trial, _successful_result(trial))
    assert record["passed"]
    assert not record["failure_category"]
    assert all(record["quality_checks"].values())


def test_trial_quality_gate_rejects_motion_that_oracle_would_otherwise_accept():
    trial = robustness_trial_schedule(1)[0]
    result = _successful_result(trial)
    result["maximum_arm_step_rad"] = MAX_QUALITY_ARM_STEP_RAD + 0.001
    record = evaluate_trial(trial, result)
    assert not record["passed"]
    assert record["failure_category"] == "motion_quality"
    assert not record["quality_checks"]["maximum_arm_step_within_limit"]


def test_trial_quality_gate_rejects_excessive_instantaneous_speed():
    trial = robustness_trial_schedule(1)[0]
    result = _successful_result(trial)
    result["maximum_arm_speed_rad_s"] = MAX_QUALITY_ARM_SPEED_RAD_S + 0.001
    record = evaluate_trial(trial, result)
    assert not record["passed"]
    assert record["failure_category"] == "motion_quality"
    assert not record["quality_checks"]["maximum_arm_speed_within_limit"]


@pytest.mark.parametrize(
    ("phase", "reason", "expected"),
    [
        ("close", "bilateral fingertip contact was not established", "grasp"),
        ("approach_retry", "approach stopped away from can", "grasp"),
        ("carry_tuck", "CuRobo joint planning failed", "planning"),
        ("traverse", "base follower stopped", "transport"),
        ("release", "released can did not settle", "placement"),
        ("place", "arm tracking stalled", "arm_tracking"),
    ],
)
def test_failure_taxonomy_is_stable(phase, reason, expected):
    assert (
        classify_failure(
            {
                "passed": False,
                "phase_history": [phase, "failure"],
                "failure_reason": reason,
            }
        )
        == expected
    )


def test_summary_requires_completion_and_ninety_percent_success():
    schedule = robustness_trial_schedule(20)
    records = [
        evaluate_trial(trial, _successful_result(trial)) for trial in schedule
    ]
    records[-1]["passed"] = False
    records[-1]["failure_category"] = "grasp"
    records[-2]["passed"] = False
    records[-2]["failure_category"] = "placement"
    partial = summarize_robustness(schedule, records[:-1])
    assert not partial["passed"]
    assert not partial["complete"]
    report = summarize_robustness(schedule, records)
    assert report["passed"]
    assert report["successes"] == 18
    assert report["success_rate"] == pytest.approx(0.90)
    assert report["failure_categories"] == {"placement": 1, "grasp": 1}
