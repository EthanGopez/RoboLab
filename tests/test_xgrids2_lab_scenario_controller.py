"""Focused controller contracts for the configurable XGRIDS2 oracle."""

from types import SimpleNamespace

import pytest
import torch

from robolab.tasks.mobile_manipulation.xgrids2_motion_diagnostics import (
    ArmMotionDiagnostics,
    MAX_OSCILLATION_REVERSALS_PER_WINDOW,
    minimum_jerk_joint_targets,
    time_parameterized_joint_targets,
)

from robolab.tasks.mobile_manipulation.xgrids2_lab_curobo_oracle import (
    XGRIDS2CuroboResult,
)
from robolab.tasks.mobile_manipulation.fixed_base_manipulation import (
    JointTrajectory,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenario_oracle import (
    XGRIDS2ConfigurableTransferOracle,
    XGRIDS2ScenarioOracle,
    XGRIDS2ScenarioResult,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import (
    get_xgrids2_scenario,
)


class _Scene(dict):
    env_origins = torch.zeros((1, 3))


def test_top_grasp_constructs_a_vertical_cartesian_descent():
    scenario = get_xgrids2_scenario("lime_clutter_to_bowl")
    backend = XGRIDS2ConfigurableTransferOracle.__new__(
        XGRIDS2ConfigurableTransferOracle
    )
    backend.leg = scenario.legs[0]
    backend.can = SimpleNamespace(
        data=SimpleNamespace(root_pos_w=torch.tensor([[1.24, -1.18, 0.92]]))
    )
    pregrasp, approach, position = backend._pick_targets()
    assert pregrasp.position[:2] == approach.position[:2] == position[:2]
    assert pregrasp.position[2] > approach.position[2] > position[2]
    assert pregrasp.quaternion_wxyz == approach.quaternion_wxyz


def test_live_clutter_is_converted_to_namespaced_planner_obstacles():
    scenario = get_xgrids2_scenario("lime_clutter_to_bowl")
    scene = _Scene()
    for item in scenario.objects:
        scene[item.name] = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[*item.center_xy, item.spawn_z]])
            )
        )
    backend = XGRIDS2ConfigurableTransferOracle.__new__(
        XGRIDS2ConfigurableTransferOracle
    )
    backend.scenario = scenario
    backend.env = SimpleNamespace(scene=scene)
    boxes = backend._planner_boxes(include_destination=True)
    names = {box.name for box in boxes}
    assert {item.name for item in scenario.objects if item.distractor} <= names
    assert "target_bowl" not in names
    assert "dining_table_top" in names
    assert "snack_table_top" in names


def test_leg_aggregation_namespaces_phases_and_uses_profile_metadata():
    scenario = get_xgrids2_scenario("can_and_cube_dining_to_snack")
    facade = XGRIDS2ScenarioOracle.__new__(XGRIDS2ScenarioOracle)
    facade.scenario = scenario
    aggregate = XGRIDS2ScenarioResult(scenario.name)
    leg_result = XGRIDS2CuroboResult(
        passed=True,
        completed_stage="full",
        phase_history=["close", "release", "full_success"],
        maximum_arm_step_rad=0.02,
        maximum_arm_speed_rad_s=1.0,
    )
    facade._aggregate_leg(
        aggregate,
        "cube_transfer",
        leg_result,
        {"selected_grasp_quaternion_wxyz": [0.0, 1.0, 0.0, 0.0]},
    )
    payload = aggregate.leg_results[0]
    assert aggregate.phase_history == [
        "cube_transfer/close",
        "cube_transfer/release",
        "cube_transfer/full_success",
    ]
    assert payload["grasp_mode"].startswith("either_fingertip_contact")
    assert payload["carry_route_radius_m"] == 0.36
    assert payload["strategy_metrics"]["selected_grasp_quaternion_wxyz"]


def test_sequential_retention_and_reverse_route_contracts_are_independent():
    scenario = get_xgrids2_scenario("can_and_cube_dining_to_snack")
    can_leg, cube_leg = scenario.legs
    assert can_leg.object_name != cube_leg.object_name
    assert can_leg.return_to_source_after is True
    assert cube_leg.staging_poses
    assert cube_leg.route.source_base_pose_xyzyaw != can_leg.route.source_base_pose_xyzyaw
    assert can_leg.name == "can_transfer"
    assert can_leg.return_to_source_after is True


def test_preflight_rejects_a_near_pi_continuous_branch_change():
    backend = XGRIDS2ConfigurableTransferOracle.__new__(
        XGRIDS2ConfigurableTransferOracle
    )
    backend.leg = get_xgrids2_scenario("can_and_cube_dining_to_snack").legs[1]
    current = (0.0, 0.0, -3.0, -1.0, 0.0, 0.0, 1.5)
    backend._arm_positions = lambda: current
    brittle = list(current)
    brittle[6] += 3.141592653589793 - 0.05
    safe = list(current)
    safe[6] += 3.141592653589793 - 0.35
    assert not backend._trajectories_stay_on_physical_branch(
        JointTrajectory((), (tuple(brittle),), 0.0)
    )
    assert backend._trajectories_stay_on_physical_branch(
        JointTrajectory((), (tuple(safe),), 0.0)
    )


def test_finite_joint_rebase_clamps_only_numerical_limit_overflow():
    backend = XGRIDS2ConfigurableTransferOracle.__new__(
        XGRIDS2ConfigurableTransferOracle
    )
    start = (0.0, 0.0, -3.0, -1.0, 0.0, -2.0, 1.5)
    numerical = list(start)
    numerical[5] = -2.090023
    rebased, endpoint = backend._rebase_trajectory_to_physical_branch(
        JointTrajectory((), (tuple(numerical),), 0.0), start
    )
    assert rebased.positions[0][5] == pytest.approx(-2.0899)
    assert endpoint[5] == pytest.approx(-2.0899)

    material = list(start)
    material[5] = -2.092
    with pytest.raises(
        RuntimeError,
        match="joint_6 target exceeds its physical limit",
    ):
        backend._rebase_trajectory_to_physical_branch(
            JointTrajectory((), (tuple(material),), 0.0), start
        )


def test_pour_contract_and_result_disclose_pose_only_fidelity():
    scenario = get_xgrids2_scenario("mustard_pour_into_bowl")
    assert scenario.pour is not None
    assert scenario.pour.tilt_degrees == 110.0
    assert scenario.pour.hold_control_steps == 30
    result = XGRIDS2ScenarioResult(
        scenario.name,
        passed=True,
        completed_stage="full",
        pouring_metrics={"fidelity": "pose_level_pour_no_fluid"},
        released_objects=["mustard_bottle"],
    ).as_dict()
    assert result["pouring_metrics"]["fidelity"] == "pose_level_pour_no_fluid"
    assert result["retention_active_end"] is False


def test_aggregate_route_clearance_is_a_hard_success_gate():
    result = XGRIDS2ScenarioResult(
        "can_and_cube_dining_to_snack",
        minimum_route_clearance_valid=False,
    )
    finalized = XGRIDS2ScenarioOracle._finalize_aggregate(result, "full")
    assert finalized.passed is False
    assert finalized.failure_reason == "aggregate route-clearance gate failed"
    assert finalized.completed_stage == "full"


def test_aggregate_motion_quality_precedes_route_failure_reporting():
    result = XGRIDS2ScenarioResult(
        "can_and_cube_dining_to_snack",
        maximum_arm_step_rad=0.051,
        minimum_route_clearance_valid=False,
    )
    finalized = XGRIDS2ScenarioOracle._finalize_aggregate(result, "full")
    assert finalized.passed is False
    assert finalized.failure_reason == "aggregate arm motion-quality gate failed"


def test_time_parameterized_targets_stream_without_stop_and_wait():
    start = (0.0,) * 7
    positions = ((0.0,) * 7, (0.4,) * 7, (0.8,) * 7)
    targets = time_parameterized_joint_targets(
        start,
        positions,
        source_dt_s=0.05,
        control_dt_s=1.0 / 30.0,
        maximum_target_step_rad=0.02,
    )
    assert targets[-1] == pytest.approx(positions[-1])
    maximum_step = max(
        max(abs(right - left) for left, right in zip(previous, current))
        for previous, current in zip((start, *targets[:-1]), targets)
    )
    assert maximum_step <= 0.020001
    assert len(set(targets)) == len(targets)


def test_time_parameterization_unwraps_continuous_joint_branches():
    start = (3.10, 0.0, 3.10, 0.0, 3.10, 0.0, 3.10)
    wrapped = (-3.10, 0.0, -3.10, 0.0, -3.10, 0.0, -3.10)
    targets = time_parameterized_joint_targets(
        start,
        (wrapped,),
        source_dt_s=0.05,
        control_dt_s=1.0 / 30.0,
        maximum_target_step_rad=0.02,
    )
    assert all(targets[-1][index] > 3.10 for index in (0, 2, 4, 6))
    assert all(abs(targets[-1][index]) < 1e-12 for index in (1, 3, 5))


def test_motion_diagnostics_logs_required_fields_without_false_oscillation():
    tracker = ArmMotionDiagnostics(0.1, (0.0,) * 7)
    for tick in range(12):
        position = (0.01 * tick,) * 7
        sample = tracker.update(
            target=position,
            position=position,
            raw_velocity=(0.1,) * 7,
            retention_active=tick >= 5,
            phase="carry",
        )
    assert {
        "target_position_rad",
        "position_rad",
        "velocity_rad_s",
        "acceleration_rad_s2",
        "jerk_rad_s3",
        "retention_active",
    } <= set(sample)
    assert sample["retention_active"] is True
    assert tracker.summary()["oscillation_violation"] is False


def test_motion_diagnostics_flags_sustained_direction_reversals():
    tracker = ArmMotionDiagnostics(0.1, (0.0,) * 7)
    for tick in range(MAX_OSCILLATION_REVERSALS_PER_WINDOW + 4):
        value = 0.02 if tick % 2 else 0.0
        tracker.update(
            target=(value,) * 7,
            position=(value,) * 7,
            raw_velocity=(0.0,) * 7,
            retention_active=True,
            phase="carry",
        )
    assert tracker.summary()["oscillation_violation"] is True


def test_aggregate_jerk_gate_is_part_of_motion_quality():
    result = XGRIDS2ScenarioResult(
        "can_and_cube_dining_to_snack",
        motion_diagnostics={"jerk_violation": True},
    )
    finalized = XGRIDS2ScenarioOracle._finalize_aggregate(result, "full")
    assert finalized.passed is False
    assert finalized.failure_reason == "aggregate arm jerk-quality gate failed"


def test_step_observer_records_a_tick_that_triggers_a_gate():
    facade = XGRIDS2ScenarioOracle.__new__(XGRIDS2ScenarioOracle)
    observed = []
    facade._step_observer = lambda _facade, backend: observed.append(backend.sample)

    def fail_after_step(_action):
        backend.sample = {"tick": 1}
        raise RuntimeError("quality gate")

    backend = SimpleNamespace(
        _step=fail_after_step,
        sample={},
    )
    facade._install_observer(backend)
    with pytest.raises(RuntimeError, match="quality gate"):
        backend._step(None)
    assert observed == [{"tick": 1}]


def test_minimum_jerk_transition_is_continuous_and_step_bounded():
    start = (0.0,) * 7
    target = (1.0,) * 7
    targets = minimum_jerk_joint_targets(
        start,
        target,
        maximum_target_step_rad=0.02,
    )
    steps = [
        max(abs(right - left) for left, right in zip(previous, current))
        for previous, current in zip((start, *targets[:-1]), targets)
    ]
    midpoint = steps[len(steps) // 2]
    assert targets[-1] == pytest.approx(target)
    assert max(steps) <= 0.020001
    assert steps[0] < midpoint
    assert steps[-1] < midpoint
