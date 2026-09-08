"""Export passive evidence to RoboLab's existing local dashboard schema."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def export_dashboard_episode(output_dir, scenario, evaluation, events, *, dt, video_path=None, seed=0):
    """Stream JSONL into chunked HDF5; preserve original evidence in place.

    Source for dashboard: <output_dir>/dashboard. No Isaac dependency.
    """
    output_dir = Path(output_dir)
    run = output_dir / "dashboard" / "passive_transfer"
    task = run / scenario.task_name
    task.mkdir(parents=True, exist_ok=True)
    trace = output_dir / "evaluation_trace.jsonl"
    with h5py.File(task / "data.hdf5", "w") as h5, trace.open() as source:
        ep = h5.create_group("data/demo_0")
        ep.attrs["dt"] = dt
        ep.attrs["arm_joint_names"] = json.dumps([f"joint_{i}" for i in range(1, 8)])
        ep.attrs["timebase"] = "video time=(global control tick-1)*dt; post-step sim time=video time+dt"
        datasets = {}
        batches = {}

        def flush():
            for name, values in batches.items():
                array = np.asarray(values, dtype=np.float64)
                if name not in datasets:
                    datasets[name] = ep.create_dataset(name, shape=(0, array.shape[1]), maxshape=(None, array.shape[1]),
                                                       chunks=True, compression="gzip")
                ds = datasets[name]
                n = len(ds)
                ds.resize(n+len(array), axis=0)
                ds[n:] = array
            batches.clear()

        for index, line in enumerate(source):
            row = json.loads(line)
            obs, motion = row["observation"], row["observation"]["motion"]
            signals = {"states/base/position_xyz": obs["base"]["position_xyz"],
                       "states/base/command_velocity_body": obs["base"]["command_velocity_body"],
                       "states/base/linear_velocity_world": obs["base"]["linear_velocity_xyz"],
                       "states/evaluation/progress": [row["progress"]],
                       "states/evaluation/base_path_length_m": [row["base_path_length_m"]]}
            for name in ("target_position_rad", "position_rad", "velocity_rad_s", "raw_physx_velocity_rad_s", "acceleration_rad_s2", "jerk_rad_s3"):
                signals[f"states/arm/{name}"] = motion[name]
            for name in ("applied_velocity_body", "applied_acceleration_body", "applied_force_world_n", "applied_torque_world_nm", "angular_velocity_xyz"):
                if name in obs["base"]:
                    signals[f"states/base/{name}"] = obs["base"][name]
            if "quality_diagnostics" in row:
                for name, value in row["quality_diagnostics"]["current"].items():
                    signals[f"states/quality/{name}"] = [value]
                for name, value in row["quality_diagnostics"]["warning_episodes"].items():
                    signals[f"states/quality/warning_episodes/{name}"] = [value]
            for name, value in obs.get("tool_flange", {}).items():
                signals[f"states/tool_flange/{name}"] = value
            for group, value in row["groups"].items():
                signals[f"states/capability/{group}"] = [value]
            for name, obj in obs["objects"].items():
                signals[f"states/objects/{name}/position_xyz"] = obj["position_xyz"]
                signals[f"states/objects/{name}/retention_active"] = [obj["retention_active"]]
            for name, values in signals.items():
                batches.setdefault(name, []).append(values)
            if (index+1) % 512 == 0:
                flush()
        flush()
    (task / "env_cfg.json").write_text(json.dumps({"sim": {"dt": dt}, "decimation": 1,
                                                  "note": "export control clock, not physics substep configuration"}, indent=2)+"\n")
    (task / "log_0_env0.json").write_text(json.dumps({"dt": dt, "events": events, "evaluation": evaluation}, indent=2)+"\n")
    if video_path is not None and Path(video_path).is_file():
        link = task / "video_0_env0.mp4"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(Path(video_path).resolve())
    metrics = {k: v for k, v in evaluation["metrics"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    episode = {"env_name": scenario.task_name, "env_id": 0, "episode": 0, "run": 0,
               "success": evaluation["task_success"], "score": evaluation["progress"],
               "duration": evaluation["metrics"]["elapsed_sim_time_s"], "episode_step": round(evaluation["metrics"]["elapsed_sim_time_s"]/dt),
               "reason": None if evaluation["task_success"] else json.dumps(evaluation["failure_localization"]),
               "instruction": scenario.instruction, "attributes": list(scenario.capability_tags),
               "policy": "privileged_scripted_curobo_oracle", "seed": seed, "metrics": metrics}
    (run / "episode_results.jsonl").write_text(json.dumps(episode)+"\n")
    return str(output_dir / "dashboard")
