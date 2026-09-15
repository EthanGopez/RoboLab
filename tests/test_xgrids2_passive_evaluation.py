"""CPU-only evaluation, failure injection, observer and dashboard export tests."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import get_xgrids2_scenario
from robolab.tasks.mobile_manipulation.xgrids2_passive_evaluation import PassiveTransferEvaluator
from robolab.tasks.mobile_manipulation.xgrids2_evaluation_observer import PostStepObserver
from robolab.tasks.mobile_manipulation.xgrids2_evaluation_export import export_dashboard_episode


SCENARIO = get_xgrids2_scenario("can_and_cube_dining_to_snack")


def snapshot(tick=1):
    return {"tick": tick,
            "base": {"position_xyz": list(SCENARIO.legs[0].route.source_base_pose_xyzyaw[:3]),
                     "yaw_rad": SCENARIO.legs[0].route.source_base_pose_xyzyaw[3],
                     "linear_velocity_xyz": [0, 0, 0], "command_velocity_body": [0, 0, 0]},
            "motion": {k: [0]*7 for k in ("target_position_rad", "position_rad", "velocity_rad_s",
                                          "raw_physx_velocity_rad_s", "acceleration_rad_s2", "jerk_rad_s3")},
            "objects": {obj.name: {"position_xyz": [*obj.center_xy, obj.spawn_z],
                                   "quaternion_wxyz": [1, 0, 0, 0],
                                   "linear_velocity_xyz": [0, 0, 0], "angular_velocity_xyz": [0, 0, 0],
                                   "retention_active": False, "fingertip_forces_n": [0, 0]}
                        for obj in SCENARIO.objects}}


def advance(evaluator, sample, count=1):
    for _ in range(count):
        sample["tick"] = evaluator.tick+1
        result = evaluator.observe(deepcopy(sample))
    return result


def complete_leg(evaluator, sample, leg):
    obj = sample["objects"][leg.object_name]
    obj["fingertip_forces_n"] = [2, 2]
    advance(evaluator, sample, 3)
    obj["position_xyz"][2] += .3
    advance(evaluator, sample)
    obj["position_xyz"] = [*leg.place.center_xy, 1.1]
    advance(evaluator, sample)
    obj["retention_active"] = False
    obj["fingertip_forces_n"] = [0, 0]
    obj["position_xyz"][2] = leg.place.settled_z
    return advance(evaluator, sample, 10)


def test_complete_episode_and_maintained_placement_regression():
    ev, sample = PassiveTransferEvaluator(SCENARIO, dt=.1), snapshot()
    for leg in SCENARIO.legs:
        complete_leg(ev, sample, leg)
    assert ev.summary()["task_success"]
    assert ev.summary()["progress"] == 1
    assert ev.summary()["handoffs"]["transport_to_place"]["rate"] == 1
    sample["objects"]["can"]["position_xyz"][0] += 1
    advance(ev, sample)
    assert not ev.summary()["task_success"]
    assert ev.summary()["progress"] < 1
    assert ev.summary()["failure_localization"][0]["first_unsatisfied_capability"] == "placed"
    sample["objects"]["can"]["position_xyz"][0] -= 1
    advance(ev, sample, 10)
    assert ev.summary()["task_success"]
    assert ev.summary()["metrics"]["predicate_recoveries"] == 1


def test_no_contact_no_grasped_and_unknown_is_not_zero():
    ev, sample = PassiveTransferEvaluator(SCENARIO, dt=.1), snapshot()
    sample["objects"]["can"]["retention_active"] = True
    advance(ev, sample)
    result = ev.summary()
    assert not result["predicates"]["can_transfer/grasped"]
    assert result["handoffs"]["grasp_to_transport"]["rate"] is None
    assert result["metrics"]["collision_count"] is None
    assert result["metrics"]["unassisted_drop_rate"] is None


def test_navigation_does_not_gate_success_and_diagnostics_do_not_change_progress():
    ev, sample = PassiveTransferEvaluator(SCENARIO, dt=.1), snapshot()
    sample["base"]["position_xyz"] = [9, 9, .13]
    for leg in SCENARIO.legs:
        complete_leg(ev, sample, leg)
    assert ev.summary()["task_success"]
    assert ev.summary()["handoffs"]["navigation_to_manipulation"]["eligible"] == 0
    assert ev.summary()["progress"] == 1


def test_all_objects_observed_no_oracle_phases_and_no_snapshot_mutation():
    ev, sample = PassiveTransferEvaluator(SCENARIO, dt=.1), snapshot()
    for obj in sample["objects"].values():
        obj["fingertip_forces_n"] = [3, 3]
    before = deepcopy(sample)
    ev.observe(sample)
    assert sample == before
    assert ev.states["can_transfer/contact_verified"]
    assert ev.states["cube_transfer/contact_verified"]
    assert ev.latest["video_time_s"] == 0
    assert ev.latest["sim_time_s"] == .1


def test_consecutive_ticks_enforced_and_stationary_path_zero():
    ev = PassiveTransferEvaluator(SCENARIO, dt=.1)
    with pytest.raises(ValueError, match="consecutive"):
        ev.observe(snapshot(3))
    sample = snapshot()
    advance(ev, sample, 20)
    assert ev.summary()["metrics"]["base_path_length_m"] == 0


def test_observer_is_passive_restores_step_and_never_records_failed_step():
    observed = []
    env = SimpleNamespace(step=lambda action: (action, "done"))
    original = env.step
    with PostStepObserver(env, lambda action, result: observed.append((action, result))):
        assert env.step(12) == (12, "done")
    assert env.step is original
    assert observed == [(12, (12, "done"))]
    def broken(action):
        raise RuntimeError("physics error")
    env.step = broken
    with pytest.raises(RuntimeError):
        with PostStepObserver(env, lambda *_: observed.append("bad")):
            env.step(1)
    assert env.step is broken
    assert len(observed) == 1


def test_dashboard_export_loader_and_time_alignment(tmp_path, monkeypatch):
    from dashboard.loaders.hdf5 import episode_timeseries
    from dashboard.loaders.local import LocalLoader
    ev, sample = PassiveTransferEvaluator(SCENARIO, dt=.1), snapshot()
    with (tmp_path / "evaluation_trace.jsonl").open("w") as stream:
        for _ in range(3):
            row = advance(ev, sample)
            stream.write(json.dumps(row)+"\n")
    source = export_dashboard_episode(tmp_path, SCENARIO, ev.summary(), ev.events, dt=.1)
    task = tmp_path / "dashboard/passive_transfer" / SCENARIO.task_name
    series = episode_timeseries(task / "data.hdf5", "demo_0")
    assert series["series"]["state.arm.position_rad"]["data"][0][0] == 0
    assert series["series"]["state.arm.position_rad"]["data"][-1][0] == .2
    runs = LocalLoader(source).list_runs()
    assert len(runs) == 1
    assert runs[0].num_episodes == 1
    assert not runs[0].num_success
    assert runs[0].policy == "privileged_scripted_curobo_oracle"
    # Exercise the real endpoint/schema with an isolated source registry.
    from fastapi.testclient import TestClient
    from dashboard.app import create_app, _event_severity
    assert _event_severity("can_transfer/placed_REGRESSION") == "failure"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    client = TestClient(create_app(initial_dir=tmp_path / "dashboard"))
    url = f"/api/runs/passive_transfer/tasks/{SCENARIO.task_name}/episodes/0/run/0"
    response = client.get(url+"/events")
    assert response.status_code == 200
    data = response.json()
    assert data["evaluation"]["schema"] == "xgrids2_passive_transfer_v1"
    assert data["events"][0]["time_s"] == 0
    assert data["events"][0]["value"] is True
    response = client.get(url+"/timeseries")
    assert response.status_code == 200
    assert "state.arm.target_position_rad" in response.json()["series"]
    assert "passive-evaluation.js" in client.get("/").text


def test_settling_rejects_fast_motion_and_retention_activation():
    ev, sample = PassiveTransferEvaluator(SCENARIO, dt=.1), snapshot()
    leg = SCENARIO.legs[0]
    complete_leg(ev, sample, leg)
    obj = sample["objects"]["can"]
    obj["linear_velocity_xyz"] = [.2, 0, 0]
    advance(ev, sample, 15)
    assert not ev.states["can_transfer/placed"]
    obj["linear_velocity_xyz"] = [0, 0, 0]
    obj["retention_active"] = True
    advance(ev, sample, 15)
    assert not ev.states["can_transfer/placed"]


def test_stale_contact_cannot_certify_grasp():
    ev, sample = PassiveTransferEvaluator(SCENARIO, dt=.1), snapshot()
    sample["objects"]["can"]["fingertip_forces_n"] = [3, 3]
    advance(ev, sample)
    sample["objects"]["can"]["fingertip_forces_n"] = [0, 0]
    advance(ev, sample, 5)
    advance(ev, sample)
    assert ev.states["can_transfer/contact_verified"]
    assert not ev.states["can_transfer/grasped"]


def test_dashboard_playback_does_not_reveal_future_achievements():
    import shutil
    import subprocess
    from pathlib import Path
    if shutil.which("node") is None:
        pytest.skip("Node unavailable")
    script = Path(__file__).resolve().parents[1] / "dashboard/static/passive-evaluation.js"
    js = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const element = () => ({children: [], style: {}, appendChild(x) { this.children.push(x); }});
const context = { document: { createElement: element } };
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const host = element(), callbacks = {};
const video = {currentTime: 0, addEventListener(k, f) {callbacks[k] = f;}};
const evaluation = {predicates: {'can/grasped': true}, metrics: {}, validity: {}};
const events = [{time_s: 5, predicate: 'can/grasped', value: true}];
context.renderPassiveEvaluation(host, evaluation, events, [video]);
const panel = host.children[0].children[2];
assert(panel.textContent.includes('pending: Physical grasp established'));
video.currentTime = 6; callbacks.seeked();
assert(panel.textContent.includes('✓ Physical grasp established'));
video.currentTime = 1; callbacks.seeked();
assert(panel.textContent.includes('pending: Physical grasp established'));
'''
    subprocess.run(["node", "-e", js, str(script)], check=True)


def test_replay_audit_detects_score_tampering(tmp_path):
    from examples.audit_xgrids2_evaluation import audit
    ev, sample = PassiveTransferEvaluator(SCENARIO, dt=.1), snapshot()
    row = advance(ev, sample)
    trace = tmp_path / "evaluation_trace.jsonl"
    trace.write_text(json.dumps(row)+"\n")
    (tmp_path / "evaluation.json").write_text(json.dumps(ev.summary()))
    (tmp_path / "evaluation_events.json").write_text(json.dumps(ev.events))
    (tmp_path / "video_frame_map.json").write_text("[]")
    (tmp_path / "result.json").write_text(json.dumps({"recording_diagnostics": {"control_dt_s": .1},
                                                     "artifacts": {"video": None}}))
    assert audit(tmp_path)["scoring_replay_exact"]
    row["progress"] = .99
    trace.write_text(json.dumps(row)+"\n")
    with pytest.raises(AssertionError, match="differs at tick"):
        audit(tmp_path)


def test_tracking_error_respects_continuous_joint_branches():
    import math
    ev, sample = PassiveTransferEvaluator(SCENARIO, dt=.1), snapshot()
    sample["motion"]["target_position_rad"][0] = 2*math.pi
    advance(ev, sample)
    assert ev.summary()["metrics"]["arm_tracking_rmse_rad"] == pytest.approx(0)
    # Joint 2 is not continuous: it must not be wrapped.
    sample["motion"]["target_position_rad"][1] = 2*math.pi
    advance(ev, sample)
    assert ev.summary()["metrics"]["arm_tracking_rmse_rad"] > 1


def test_oriented_rectangle_containment_uses_support_yaw():
    import math
    from dataclasses import replace
    from robolab.tasks.mobile_manipulation.xgrids2_passive_evaluation import placement_geometry
    spec = SCENARIO.object("cube")
    leg = SCENARIO.legs[1]
    support = replace(SCENARIO.support(leg.place.support_name), kind="rectangle",
                      center_xyz=(0, 0, .7), size_xyz=(1, .2, .02), yaw_rad=math.pi/2)
    obj = snapshot()["objects"]["cube"]
    obj["position_xyz"] = [0, .3, .74]
    assert placement_geometry(spec, leg, support, obj)["contained"]
    obj["position_xyz"] = [.3, 0, .74]
    assert not placement_geometry(spec, leg, support, obj)["contained"]
