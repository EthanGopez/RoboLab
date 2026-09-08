#!/usr/bin/env python3
"""Audit a completed 1-frame/control-tick recording without starting Isaac."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def audit(directory: Path) -> dict:
    result = json.loads((directory / "result.json").read_text())
    samples = [json.loads(line) for line in (directory / "control_trace.jsonl").open()]
    recording = result["recording_diagnostics"]
    if recording["capture_every_control_ticks"] != 1:
        raise ValueError("Frame-to-tick audit requires capture-every=1")
    fps = recording["video_fps"]
    phases = []
    metrics = defaultdict(lambda: {"ticks": 0, "max_jerk": 0., "max_raw_speed": 0., "max_reversals": 0})
    events = []
    previous_retention = False
    for index, sample in enumerate(samples):
        label = f"{sample['object_name']}/{sample['phase']}"
        if not phases or phases[-1]["phase"] != label:
            phases.append({"phase": label, "first_frame": index, "start_s": index / fps})
        metric = metrics[label]
        metric["ticks"] += 1
        if sample["tick"] >= 3:
            metric["max_jerk"] = max(metric["max_jerk"], max(map(abs, sample["jerk_rad_s3"])))
            metric["max_raw_speed"] = max(metric["max_raw_speed"], max(map(abs, sample["raw_physx_velocity_rad_s"])))
            metric["max_reversals"] = max(metric["max_reversals"], sample["rolling_oscillation_reversals"])
        active = sample["retention_active"]
        if active != previous_retention:
            events.append({"name": f"{sample['object_name']}_{'attach' if active else 'release'}", "frame": index})
        previous_retention = active
    peak = max((i for i, s in enumerate(samples) if s["tick"] >= 3),
               key=lambda i: max(map(abs, samples[i]["jerk_rad_s3"])))
    events.append({"name": "peak_measured_jerk", "frame": peak})
    for object_name in ("can", "cube"):
        ids = [i for i, s in enumerate(samples) if s["object_name"] == object_name and s["phase"] == "traverse"]
        if ids:
            events.append({"name": f"{object_name}_loaded_traverse", "frame": ids[len(ids) // 2]})
    audit_dir = directory / "audit"
    audit_dir.mkdir(exist_ok=True)
    selected_frames = {}
    for event in events:
        event["time_s"] = event["frame"] / fps
        event["sample_frames"] = [max(0, min(len(samples) - 1, event["frame"] + offset)) for offset in (-15, -5, 0, 5, 15, 30)]
        for index in event["sample_frames"]:
            selected_frames[index] = None
    capture = cv2.VideoCapture(str(directory / "task.mp4"))
    encoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    encoded_fps = capture.get(cv2.CAP_PROP_FPS)
    panel_changes = defaultdict(list)
    motion_changes = defaultdict(list)
    previous = None
    decoded_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if decoded_count in selected_frames:
            selected_frames[decoded_count] = frame.copy()
        # Exclude text overlays: count only changes in the rendered view.
        third_bottom = 454 if result.get("passive_evaluation") and recording.get("metrics_panel_layout") != "appended_footer" else 540
        panels = {"third_person": frame[52:third_bottom, :960],
                  "navigation": frame[52:270, 960:1440],
                  "wrist": frame[322:540, 960:1440]}
        if previous is not None:
            for name, panel in panels.items():
                change = float(np.mean(cv2.absdiff(panel, previous[name]) > 8))
                panel_changes[name].append(change)
                if decoded_count < len(samples) and max(map(abs, samples[decoded_count]["velocity_rad_s"])) > 0.02:
                    motion_changes[name].append(change)
        previous = panels
        decoded_count += 1
    capture.release()
    for event in events:
        cells = []
        for index in event["sample_frames"]:
            frame = selected_frames[index]
            if frame is None:
                continue
            cell = cv2.resize(frame, (720, round(frame.shape[0]/2)), interpolation=cv2.INTER_AREA)
            cv2.putText(cell, f"frame {index} | {index / fps:.3f}s", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cells.append(cell)
        if len(cells) == 6:
            sheet = np.concatenate([np.concatenate(cells[:3], axis=1), np.concatenate(cells[3:], axis=1)], axis=0)
            sheet_path = audit_dir / f"{event['name']}.jpg"
            if not cv2.imwrite(str(sheet_path), sheet):
                raise RuntimeError(f"Cannot write {sheet_path}")
            event["frame_sheet"] = str(sheet_path)
    summary = {
        "task_passed": result["passed"],
        "motion_quality_violation": result["motion_quality_violation"],
        "decoded_frames": decoded_count,
        "encoded_frames": encoded_count,
        "trace_ticks": len(samples),
        "frames_match_ticks": decoded_count == len(samples) == recording["written_video_frames"],
        "global_ticks_contiguous": [s["global_tick"] for s in samples] == list(range(1, len(samples) + 1)),
        "encoded_fps": encoded_fps,
        "real_time_playback": abs(encoded_fps * recording["control_dt_s"] - 1.) < 1e-8,
        "duration_s": decoded_count / encoded_fps,
        "panel_changes_excluding_overlays": {
            name: {"maximum_fraction": max(values), "mean_fraction": float(np.mean(values)),
                   "mean_fraction_during_arm_motion": float(np.mean(motion_changes[name])) if motion_changes[name] else None}
            for name, values in panel_changes.items()
        },
        "phase_intervals": phases,
        "phase_motion_metrics": dict(metrics),
        "inspection_events": events,
        "limitation": "Pixel changes detect frozen panels; smoothness is judged from joint traces plus visual frame/clip inspection.",
    }
    (audit_dir / "recording_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = audit(args.directory.resolve())
    print(json.dumps(summary, indent=2))
    if not (summary["frames_match_ticks"] and summary["global_ticks_contiguous"] and summary["real_time_playback"]):
        raise SystemExit("Recording synchronization audit failed")
