#!/usr/bin/env python3
"""Build an offline, simulator-free video/trace viewer for the single-can trial."""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


ASSETS = Path(__file__).with_name("xgrids2_diagnostic_viewer")


def context_for(*, held, lifted, transported, placed, destination, contact, moving):
    """Display selection only: observations, never oracle phase advancement."""
    if placed:
        return "settled"
    if transported and not held:
        return "settling"
    if held and (transported or destination) and not moving:
        return "placement"
    if moving:
        return "navigation"
    if contact and not lifted:
        return "grasp"
    return "arm"


def build_payload(directory):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    spec = result["scenario_contract"]
    if spec["name"] != "can_dining_to_snack":
        raise ValueError("This prototype is scoped to the single-can diagnostic")
    rec = result["recording_diagnostics"]
    if rec["capture_every_control_ticks"] != 1 or rec["playback_speed_ratio"] != 1:
        raise ValueError("Expected one video frame per control tick at 1x playback")
    dt = rec["control_dt_s"]
    leg = spec["legs"][0]
    supports = {s["name"]: s for s in spec["supports"]}
    heights = {k: supports[v]["center_xyz"][2]+supports[v]["size_xyz"][2]/2
               for k, v in (("source", leg["source_support_name"]),
                            ("destination", leg["place"]["support_name"]))}
    columns = {k: [] for k in ("time", "sim_time", "context", "held", "lifted", "transported", "placed",
        "base_command", "base_applied", "base_measured", "yaw_command", "yaw_applied", "yaw_measured",
        "joint_target_velocity", "joint_velocity", "joint_error", "max_joint_error",
        "arm_jerk", "target_jerk", "left_contact", "right_contact", "source_clearance",
        "destination_clearance", "object_speed", "placement_error", "path", "release_clearance")}
    moving = False
    events = []
    previous_held = False
    with (directory / "evaluation_trace.jsonl").open() as stream:
        for i, line in enumerate(stream):
            row = json.loads(line)
            if row["tick"] != i+1 or abs(row["video_time_s"]-i*dt) > 1e-7:
                raise ValueError(f"Non-contiguous or misaligned trace at row {i}")
            obs = row["observation"]
            base, motion, obj = obs["base"], obs["motion"], obs["objects"]["can"]
            pred = lambda name: bool(row["predicates"].get(f"{leg['name']}/{name}", False))
            held = pred("grasped") and not pred("placed")
            speed = math.hypot(*base["linear_velocity_xyz"][:2])
            commanded_speed = math.hypot(*base["command_velocity_body"][:2])
            moving = (max(speed, commanded_speed) > (.005 if moving else .015)
                      or max(abs(base['angular_velocity_xyz'][2]), abs(base['command_velocity_body'][2]))
                      > (.015 if moving else .04))
            mode = context_for(held=held, lifted=pred("lifted"), transported=pred("transported"),
                placed=pred("placed"), destination=pred("navigation_destination_reached"),
                contact=pred("contact_verified"), moving=moving)
            error = [t-q for t, q in zip(motion["target_position_rad"], motion["position_rad"])]
            for j in (0, 2, 4, 6):
                error[j] = (error[j]+math.pi) % (2*math.pi)-math.pi
            geo = row["placement"][leg["name"]]
            values = dict(time=i*dt, sim_time=row["sim_time_s"], context=mode, held=held,
                lifted=pred("lifted"), transported=pred("transported"), placed=pred("placed"),
                base_command=commanded_speed, base_applied=math.hypot(*base["applied_velocity_body"][:2]),
                base_measured=speed, yaw_command=base["command_velocity_body"][2],
                yaw_applied=base["applied_velocity_body"][2], yaw_measured=base["angular_velocity_xyz"][2],
                joint_target_velocity=motion["target_velocity_rad_s"], joint_velocity=motion["velocity_rad_s"],
                joint_error=error, max_joint_error=max(map(abs, error)),
                arm_jerk=max(map(abs, motion["jerk_rad_s3"])) if i >= 3 else None,
                target_jerk=max(map(abs, motion["target_jerk_rad_s3"])) if i >= 3 else None,
                left_contact=obj["fingertip_forces_n"][0], right_contact=obj["fingertip_forces_n"][1],
                source_clearance=100*(geo["bottom_z"]-heights["source"]),
                destination_clearance=100*(geo["bottom_z"]-heights["destination"]) if pred("transported") else None,
                object_speed=math.sqrt(sum(v*v for v in obj["linear_velocity_xyz"])),
                placement_error=100*geo["xy_error_m"] if pred("transported") else None,
                path=row["base_path_length_m"], release_clearance=row.get("release_clearance_m"))
            for key, value in values.items():
                columns[key].append(value)
            if obj["retention_active"]:
                raise ValueError("benchmark trace contains forbidden fixed-joint retention")
            if held != previous_held:
                events.append({"time": i*dt, "label": "Physical grasp observed" if held else "Physical release observed",
                               "kind": "grasp", "detail": "Derived from passive contact and task predicates"})
            previous_held = held
    n = len(columns["time"])
    if n == 0 or n != rec["written_video_frames"]:
        raise ValueError("Trace and encoded frame counts differ")
    frame_map = json.loads((directory / "video_frame_map.json").read_text())
    if len(frame_map) != n:
        raise ValueError("Frame map length differs from trace")
    for i, frame in enumerate(frame_map):
        if frame["frame"] != i or frame["evaluation_tick"] != i+1 or abs(frame["video_time_s"]-i*dt) > 1e-7:
            raise ValueError("Frame map does not match post-step trace")
    labels = {"contact_verified": "Finger contact observed", "grasped": "Physical grasp established",
              "lifted": "Can cleared dining table", "navigation_destination_reached": "Base reached snack-table goal",
              "transported": "Can over snack table", "placed": "Released and settled at target"}
    for event in json.loads((directory / "evaluation_events.json").read_text()):
        key = event.get("predicate", "").split("/")[-1]
        if key in labels:
            events.append({"time": event["step"]*dt, "label": labels[key] if event["value"] else labels[key]+" — lost",
                           "kind": "outcome" if event["value"] else "warning", "detail": event.get("info", "")})
        elif event["name"] == "MOTION_WARNING":
            events.append({"time": event["step"]*dt, "label": "Motion advisory", "kind": "warning", "detail": event["info"]})
    summary = json.loads((directory / "evaluation.json").read_text())
    return {"schema": "xgrids2_standalone_viewer_v1", "dt": dt, "duration": n*dt, "frames": n,
            "video_width": result["video_diagnostics"]["width"], "video_height": result["video_diagnostics"]["height"],
            "visible_video_height": 540 if rec.get("metrics_panel_layout") == "appended_footer" else result["video_diagnostics"]["height"],
            "columns": columns, "events": sorted(events, key=lambda e: e["time"]), "summary": summary,
            "contact_threshold": leg["grasp"]["contact_force_n"],
            "settled_speed_threshold": summary["thresholds"]["settled_linear_speed_m_s"],
            "placement_tolerance_cm": 100*leg["place"]["xy_tolerance_m"]}


def build(directory, output=None):
    directory = Path(directory).resolve()
    output = Path(output).resolve() if output else directory / "viewer" / "index.html"
    data = build_payload(directory)
    video = directory / "task.mp4"
    if not video.is_file():
        raise ValueError("Missing task.mp4")
    data["video"] = quote(os.path.relpath(video, output.parent).replace(os.sep, "/"), safe="/.")
    template = (ASSETS / "viewer.html").read_text()
    template = template.replace("__STYLE__", (ASSETS / "viewer.css").read_text())
    template = template.replace("__VIDEO__", html.escape(data["video"], quote=True))
    # Data is embedded rather than fetched, so file:// viewing works offline.
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = template.replace("__DATA__", encoded)
    template = template.replace("__SCRIPT__", (ASSETS / "viewer.js").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build(args.directory, args.output))
