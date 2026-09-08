#!/usr/bin/env python3
"""Replay passive scores from measured snapshots and check frame/tick alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import get_xgrids2_scenario
from robolab.tasks.mobile_manipulation.xgrids2_passive_evaluation import PassiveTransferEvaluator


def audit(directory):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    expected = json.loads((directory / "evaluation.json").read_text())
    dt = result["recording_diagnostics"]["control_dt_s"]
    evaluator = PassiveTransferEvaluator(get_xgrids2_scenario(expected["scenario"]), dt=dt)
    trace_rows = 0
    with (directory / "evaluation_trace.jsonl").open() as stream:
        for line in stream:
            saved = json.loads(line)
            replayed = evaluator.observe(saved["observation"])
            if saved != replayed:
                raise AssertionError(f"evaluation replay differs at tick {saved['tick']}")
            trace_rows += 1
    replay_summary = evaluator.summary()
    for key, value in replay_summary.items():
        if expected[key] != value:
            raise AssertionError(f"summary replay differs: {key}")
    expected_events = json.loads((directory / "evaluation_events.json").read_text())
    if expected_events != evaluator.events:
        raise AssertionError("event timeline differs on replay")
    frame_map = json.loads((directory / "video_frame_map.json").read_text())
    for index, frame in enumerate(frame_map):
        assert frame["frame"] == index
        assert frame["global_tick"] == frame["evaluation_tick"] == index+1
        assert abs(frame["video_time_s"]-index*dt) < 1e-6
        assert abs(frame["sim_time_s"]-(index+1)*dt) < 1e-6
    if result["artifacts"]["video"]:
        import cv2
        video = cv2.VideoCapture(result["artifacts"]["video"])
        encoded_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(video.get(cv2.CAP_PROP_FPS))
        video.release()
        assert len(frame_map) == trace_rows == encoded_frames
        assert abs(fps*dt-1) < 1e-6
    else:
        encoded_frames = None
        assert not frame_map
    summary = {"passed": True, "replayed_ticks": trace_rows, "replayed_events": len(evaluator.events),
               "encoded_frames": encoded_frames, "task_success": expected["task_success"],
               "progress": expected["progress"], "scoring_replay_exact": True,
               "frame_evaluation_timebase_matches": True,
               "note": "Use audit_xgrids2_recording.py for full MP4 decode and non-overlay pixel-motion checks."}
    (directory / "evaluation_audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))
