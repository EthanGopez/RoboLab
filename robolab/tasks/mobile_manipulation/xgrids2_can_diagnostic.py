"""Pure execution profile and path-preserving endpoint easing for can audits."""

from dataclasses import dataclass

import numpy as np

from robolab.tasks.mobile_manipulation.xgrids2_motion_diagnostics import time_parameterized_joint_targets


@dataclass(frozen=True)
class CanDiagnosticProfile:
    name: str = "can_diagnostic_v2_continuous_place"
    base_speed_m_s: float = 0.15
    base_yaw_speed_rad_s: float = 0.30
    arm_speed_factor: float = 2.0
    arm_target_step_ceiling_rad: float = 0.018
    release_max_clearance_m: float = 0.015
    attach_settle_ticks: int = 12
    continuous_place: bool = True


def eased_joint_targets(start, positions, *, source_dt_s, control_dt_s,
                        maximum_target_step_rad, speed_factor=2.0):
    """Ease the clock, not the collision-checked joint-space path.

    Ten-percent start/end ramps have zero endpoint velocity and acceleration;
    the middle runs at constant clock rate. Every returned position remains on
    an original polyline segment, unlike smoothing joint positions spatially.
    This is endpoint-continuous execution, not a claim of globally C2 geometry
    at every internal planner waypoint.
    """
    if speed_factor <= 0:
        raise ValueError("speed_factor must be positive")
    ramp = .10
    dense = time_parameterized_joint_targets(
        start, positions, source_dt_s=source_dt_s/speed_factor,
        control_dt_s=control_dt_s,
        maximum_target_step_rad=maximum_target_step_rad*(1-ramp),
    )
    points = np.asarray([start, *positions], dtype=float)
    # Unwrap the start-to-first-sample branch consistently with the source.
    for j in (0, 2, 4, 6):
        points[:, j] = np.unwrap(points[:, j])
    keep = np.r_[True, np.max(np.abs(np.diff(points, axis=0)), axis=1) > 1e-12]
    points = points[keep]
    count = max(len(dense), 30)
    t = np.arange(1, count+1)/count
    def start_ramp(u):
        x = u/ramp
        return ramp/(1-ramp)*(x**3-.5*x**4)
    clock = np.where(t < ramp, start_ramp(t),
                     np.where(t > 1-ramp, 1-start_ramp(1-t), (t-ramp/2)/(1-ramp)))
    out = np.stack([np.interp(clock, np.linspace(0, 1, len(points)), points[:, j])
                    for j in range(points.shape[1])], axis=1)
    out[-1] = points[-1]
    return tuple(tuple(float(v) for v in q) for q in out)


class CanQualityMonitor:
    """Advisory transition warnings, separate from correctness/safety gates.

    Thresholds are disclosed development heuristics, not hardware limits or
    validated human-perceptual thresholds. Count threshold-entry episodes.
    """
    THRESHOLDS = {"arm_jerk_rad_s3": 50., "arm_target_jerk_rad_s3": 100.,
                  "base_yaw_acceleration_rad_s2": 1.0}

    def __init__(self, dt):
        self.dt = dt
        self.previous = None
        self.active = set()
        self.counts = {key: 0 for key in self.THRESHOLDS}
        self.peaks = {key: 0. for key in self.THRESHOLDS}

    def observe(self, sample):
        motion = sample["motion"]
        values = {
            "arm_jerk_rad_s3": max(map(abs, motion["jerk_rad_s3"])),
            "arm_target_jerk_rad_s3": max(map(abs, motion.get("target_jerk_rad_s3", [0]))),
            "base_yaw_acceleration_rad_s2": 0.,
        }
        if self.previous:
            values["base_yaw_acceleration_rad_s2"] = abs(
                sample["base"].get("angular_velocity_xyz", [0, 0, 0])[2]
                - self.previous["base"].get("angular_velocity_xyz", [0, 0, 0])[2])/self.dt
        active = {key for key, value in values.items()
                  if sample["tick"] > 3 and value > self.THRESHOLDS[key]}
        entered = sorted(active-self.active)
        for key in entered:
            self.counts[key] += 1
        if sample["tick"] > 3:
            self.peaks = {key: max(self.peaks[key], value) for key, value in values.items()}
        self.active, self.previous = active, sample
        return {"current": values, "active_warnings": sorted(active),
                "new_warnings": entered, "warning_episodes": dict(self.counts),
                "peaks": dict(self.peaks), "advisory_thresholds": dict(self.THRESHOLDS)}


def can_diagnostic_lines(row, *, camera_cut=False):
    """Plain-language live evidence, with absent capabilities explicitly marked."""
    state = row["predicates"]
    leg = next(iter(row["placement"]))
    has = lambda key: state.get(f"{leg}/{key}", False)
    obj = row["observation"]["objects"]["can"]
    geo = row["placement"][leg]
    quality = row.get("quality_diagnostics", {})
    outcomes = [("Contact", "contact_verified"), ("Physical grasp", "grasped"),
                ("Lift clearance", "lifted"), ("Over snack table", "transported"),
                ("Released + settled", "placed")]
    outcome_line = " | ".join(f"{label}: {'YES' if has(key) else 'pending'}" for label, key in outcomes)
    handoff = lambda a, b: "success" if has(a) and has(b) else "pending" if has(a) else "not reached"
    nav = "already at source; not tested" if row.get("source_initially_reached") else "goal not reached" if not has("navigation_source_reached") else handoff("navigation_source_reached", "grasped")
    status = "PLACED SUCCESSFULLY" if row["task_success"] else "IN PROGRESS"
    release = row.get("release_clearance_m")
    release_text = "not released yet" if release is None else f"{100*release:.1f} cm above table"
    error_text = f"{100*geo['xy_error_m']:.1f} cm" if has('transported') else "not at destination yet"
    counts = quality.get("warning_episodes", {})
    labels = {"arm_jerk_rad_s3": "sharp arm motion", "arm_target_jerk_rad_s3": "sharp arm command",
              "base_yaw_acceleration_rad_s2": "rapid base turning"}
    active_text = ', '.join(labels[key] for key in quality.get('active_warnings', [])) or 'none'
    return [
        f"CAN: dining table -> snack table | {status} | sim {row['sim_time_s']:.1f}s | base path {row['base_path_length_m']:.2f}m | 1x playback",
        outcome_line,
        f"Nav -> grasp: {nav} | Grasp -> transport: {handoff('grasped', 'transported')} | Transport -> place: {handoff('transported', 'placed')}",
        f"Release: {release_text} | Placement XY error: {error_text} | Fixed-joint retention: {'INVALID' if obj['retention_active'] else 'OFF'}",
        f"Advisory alerts: arm motion {counts.get('arm_jerk_rad_s3', 0)}, arm command {counts.get('arm_target_jerk_rad_s3', 0)}, base turning {counts.get('base_yaw_acceleration_rad_s2', 0)} | Now: {active_text}" + (" | CAMERA CUT" if camera_cut else ""),
        "Not scored: perception (oracle), whole-body collisions (uninstrumented), path efficiency (no reference).",
    ]
