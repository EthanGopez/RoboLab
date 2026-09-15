"""CPU regression tests for the single-can diagnostic profile."""
import numpy as np
import pytest
import torch

from robolab.robots.mobile_base_math import jerk_limited_velocity_step
from robolab.tasks.mobile_manipulation.xgrids2_can_diagnostic import (
    eased_joint_targets, CanQualityMonitor, can_diagnostic_lines,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_navigation_oracle import WaypointFollower
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import get_xgrids2_scenario
from robolab.tasks.mobile_manipulation.xgrids2_passive_evaluation import PassiveTransferEvaluator


def test_actual_base_jerk_bounded_at_arrival_and_reversal():
    dt = 1/120
    v = torch.zeros(3, dtype=torch.float64)
    a = v.clone()
    limit_a = torch.tensor([.2, .2, 2.], dtype=torch.float64)
    limit_j = torch.tensor([.5, .5, 4.], dtype=torch.float64)
    for goal in ([.15, -.1, .35], [-.15, .1, -.35], [0., 0., 0.]):
        target = torch.tensor(goal, dtype=torch.float64)
        for _ in range(1500):
            new_v, new_a = jerk_limited_velocity_step(v, a, target,
                max_acceleration=limit_a, max_jerk=limit_j, dt=dt)
            actual_a = (new_v-v)/dt
            assert torch.all(actual_a.abs() <= limit_a+1e-10)
            assert torch.all(((actual_a-a)/dt).abs() <= limit_j+1e-9)
            torch.testing.assert_close(new_a, actual_a)
            v, a = new_v, new_a
        torch.testing.assert_close(v, target, atol=1e-6, rtol=0)


def test_eased_targets_limits_endpoint_and_original_polyline():
    start = [0.]*7
    middle, end = start.copy(), start.copy()
    middle[1], end[1], end[3] = .4, .4, .4
    targets = np.asarray(eased_joint_targets(start, [middle, end], source_dt_s=1.,
        control_dt_s=1/30, maximum_target_step_rad=.008))
    steps = np.diff(np.vstack((start, targets)), axis=0)
    assert np.max(np.abs(steps)) <= .008+1e-10
    assert np.max(np.abs(steps[0])) < .001
    assert np.max(np.abs(steps[-1])) < .001
    np.testing.assert_allclose(targets[-1], end)
    assert np.all((targets[:, 3] == 0) | np.isclose(targets[:, 1], .4))
    with pytest.raises(ValueError):
        eased_joint_targets(start, [end], source_dt_s=1, control_dt_s=.03,
            maximum_target_step_rad=.01, speed_factor=0)


def test_heading_does_not_flip_at_goal_tolerance_boundary():
    follower = WaypointFollower(np.array([[0, 0], [1, 0]]), final_yaw=1.,
        initial_yaw=0., smooth_heading=True, final_position_tolerance=.02)
    follower.command(np.array([.99, 0]), .8)
    before, _ = follower.command(np.array([1.01, 0]), .8)
    after, _ = follower.command(np.array([1.03, 0]), .8)
    assert before[2] == after[2] > 0


def test_can_initial_source_not_navigation_success_and_warning_is_not_failure():
    spec = get_xgrids2_scenario('can_dining_to_snack')
    leg = spec.legs[0]
    obj = spec.object('can')
    sample = {'tick': 1, 'base': {
        'position_xyz': list(leg.route.source_base_pose_xyzyaw[:3]),
        'yaw_rad': leg.route.source_base_pose_xyzyaw[3], 'linear_velocity_xyz': [0,0,0]},
        'objects': {'can': {'position_xyz': [*obj.center_xy, obj.spawn_z],
            'quaternion_wxyz': [1,0,0,0], 'linear_velocity_xyz': [0,0,0],
            'angular_velocity_xyz': [0,0,0], 'retention_active': False,
            'fingertip_forces_n': [0,0]}},
        'motion': {key: [0.]*7 for key in ('target_position_rad', 'position_rad', 'jerk_rad_s3')}}
    evaluator = PassiveTransferEvaluator(spec, dt=1/30)
    for tick in range(1,6):
        sample['tick'] = tick
        sample['motion']['jerk_rad_s3'] = [70.]*7
        row = evaluator.observe(sample.copy())
    summary = evaluator.summary()
    assert summary['handoffs']['navigation_to_manipulation']['eligible'] == 0
    assert 'already_at_source' in summary['handoffs']['navigation_to_manipulation']['legs'][leg.name]['status']
    assert row['quality_diagnostics']['warning_episodes']['arm_jerk_rad_s3'] == 1
    assert 'already at source; not tested' in ' '.join(can_diagnostic_lines(row))
    assert summary['progress'] == 0
    # A diagnostic warning must not break a later predicate achievement.
    sample['objects']['can']['fingertip_forces_n'] = [5,5]
    for _ in range(8):
        sample['tick'] = evaluator.tick + 1
        evaluator.observe(sample)
    assert evaluator.summary()['predicates'][f'{leg.name}/grasped']
