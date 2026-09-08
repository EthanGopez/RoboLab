"""Policy-independent, control-rate evaluation of the two-object transfer MVP.

The predicate graph is observational, never an execution state machine. Inputs
are plain measured snapshots; this module cannot command or advance a robot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EvaluationThresholds:
    lift_clearance_m: float = 0.05
    settled_duration_s: float = 1.0
    settled_linear_speed_m_s: float = 0.05
    settled_angular_speed_rad_s: float = 0.3
    contact_memory_s: float = 0.25
    navigation_xy_tolerance_m: float = 0.12
    navigation_yaw_tolerance_rad: float = 0.25
    navigation_speed_m_s: float = 0.025


def rotation_matrix(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def placement_geometry(spec, leg, support, obj):
    """Conservative oriented-box footprint plus configured settled root height."""
    position = np.asarray(obj["position_xyz"])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                        for z in (-1, 1)]) * np.asarray(spec.dimensions_xyz) / 2
    corners = corners @ rotation_matrix(obj["quaternion_wxyz"]).T + position
    offsets = corners[:, :2] - np.asarray(support.center_xyz[:2])
    c, s = math.cos(support.yaw_rad), math.sin(support.yaw_rad)
    local = offsets @ np.array([[c, -s], [s, c]])
    margin = leg.place.containment_margin_m
    if support.kind == "circle":
        contained = bool(np.max(np.linalg.norm(local, axis=1)) <= min(support.size_xyz[:2])/2-margin)
    else:
        size = support.opening_size_xy or support.size_xyz[:2]
        contained = bool(np.all(np.abs(local) <= np.asarray(size)/2-margin))
    xy_error = math.dist(position[:2], leg.place.center_xy)
    z_error = abs(float(position[2]) - leg.place.settled_z)
    return {
        "contained": contained,
        "xy_error_m": xy_error,
        "z_error_m": z_error,
        "at_target": bool(contained and xy_error <= leg.place.xy_tolerance_m
                          and z_error <= leg.place.z_tolerance_m),
        "bottom_z": float(corners[:, 2].min()),
    }


class PassiveTransferEvaluator:
    """Evaluate all legs on every snapshot, with explicit macro-group weights.

    Grasp/lift/transport achievements are historical. Placement is maintained:
    knocking a previously placed object away revokes final success/progress.
    Feasible interaction poses and perception are unknown without instrumentation.
    """

    GROUPS = {"Grasp": ("contact_verified", "assisted_grasp"),
              "Lift": ("lifted",), "Transport": ("transported",),
              "Place": ("placed",)}

    def __init__(self, scenario, *, dt: float, thresholds=EvaluationThresholds()):
        if scenario.name not in {"can_and_cube_dining_to_snack", "can_dining_to_snack"}:
            raise ValueError("Passive evaluation supports the single-can and can-and-cube transfers")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.scenario, self.dt, self.thresholds = scenario, dt, thresholds
        self.tick = 0
        self.events = []
        self.states = {}
        self.first = {}
        self.last_contact = {}
        self.settling = {}
        self.previous = None
        self.latest = None
        self.distance = self.yaw_distance = 0.0
        self.arm_squared_error = self.arm_isj = 0.0
        self.arm_samples = 0
        self.max_jerk = 0.0
        self.recovery_count = 0
        self.release_heights = {leg.name: [] for leg in scenario.legs}
        self.repositionings = {leg.name: 0 for leg in scenario.legs}
        self.last_nav_inside = {}
        self.completion_first_s = None
        self.initial_source_goals = set()
        self.quality_monitor = None
        if scenario.name == "can_dining_to_snack":
            from robolab.tasks.mobile_manipulation.xgrids2_can_diagnostic import CanQualityMonitor
            self.quality_monitor = CanQualityMonitor(dt)

    def _set(self, key, value, *, historical=False):
        old = self.states.get(key, False)
        value = bool(value or (historical and old))
        self.states[key] = value
        if value and key not in self.first:
            self.first[key] = self.tick * self.dt
        if value != old:
            recovered = value and any(e.get("predicate") == key and not e.get("value", False) for e in self.events)
            self.recovery_count += int(recovered)
            self.events.append({"step": self.tick-1, "sim_time_s": self.tick*self.dt,
                                "predicate": key, "value": value,
                                "name": key + ("_SUCCESS" if value else "_REGRESSION"),
                                "info": "recovered" if recovered else "measured predicate transition"})

    def _achieved(self, leg, name):
        return self.states.get(f"{leg.name}/{name}", False)

    def observe(self, snapshot):
        """Read one post-control-step sample; return a JSON-serializable row."""
        if snapshot["tick"] != self.tick + 1:
            raise ValueError("snapshots must contain consecutive global control ticks")
        self.tick += 1
        start_events = len(self.events)
        t = self.tick*self.dt
        base = snapshot["base"]
        if self.previous is not None:
            prev_base = self.previous["base"]
            self.distance += math.dist(base["position_xyz"][:2], prev_base["position_xyz"][:2])
            yaw_delta = (base["yaw_rad"]-prev_base["yaw_rad"]+math.pi) % (2*math.pi)-math.pi
            self.yaw_distance += abs(yaw_delta)
        motion = snapshot["motion"]
        error = np.asarray(motion["target_position_rad"]) - np.asarray(motion["position_rad"])
        error[[0, 2, 4, 6]] = (error[[0, 2, 4, 6]] + math.pi) % (2*math.pi)-math.pi
        self.arm_squared_error += float(np.sum(error**2))
        self.arm_samples += len(error)
        if self.tick > 2:
            jerk = np.asarray(motion["jerk_rad_s3"])
            self.arm_isj += float(np.sum(jerk**2))*self.dt
            self.max_jerk = max(self.max_jerk, float(np.max(np.abs(jerk))))
        geometry = {}
        for leg in self.scenario.legs:
            name = leg.object_name
            obj = snapshot["objects"][name]
            spec = self.scenario.object(name)
            source = self.scenario.support(leg.source_support_name)
            support = self.scenario.support(leg.place.support_name)
            geo = placement_geometry(spec, leg, support, obj)
            geometry[leg.name] = geo
            forces = obj.get("fingertip_forces_n")
            if forces is not None:
                contact = (min(forces) if leg.grasp.contact_mode == "bilateral" else max(forces)) >= leg.grasp.contact_force_n
                if contact:
                    self.last_contact[name] = t
                    self._set(f"{leg.name}/contact_verified", True, historical=True)
            retained = obj["retention_active"]
            recent_contact = t-self.last_contact.get(name, -1e10) <= self.thresholds.contact_memory_s
            self._set(f"{leg.name}/assisted_grasp", retained and recent_contact, historical=True)
            lifted = retained and self._achieved(leg, "assisted_grasp") and geo["bottom_z"] >= source.surface_z+self.thresholds.lift_clearance_m
            self._set(f"{leg.name}/lifted", lifted, historical=True)
            # Navigation is a diagnostic goal-region observation, not a
            # mandatory approach pose nor a claim of planner/IK feasibility.
            route = leg.route
            if route is not None:
                for which, pose in (("source", route.source_base_pose_xyzyaw), ("destination", route.destination_base_pose_xyzyaw)):
                    yaw_error = abs((base["yaw_rad"]-pose[3]+math.pi) % (2*math.pi)-math.pi)
                    inside = (math.dist(base["position_xyz"][:2], pose[:2]) <= self.thresholds.navigation_xy_tolerance_m
                              and yaw_error <= self.thresholds.navigation_yaw_tolerance_rad
                              and np.linalg.norm(base["linear_velocity_xyz"][:2]) <= self.thresholds.navigation_speed_m_s)
                    self._set(f"{leg.name}/navigation_{which}_reached", inside, historical=True)
                    if self.tick == 1 and which == "source" and inside:
                        self.initial_source_goals.add(leg.name)
                    if which == "source":
                        prior = self.last_nav_inside.get(leg.name, False)
                        if inside and not prior and f"{leg.name}/source_visit" in self.first and not self._achieved(leg, "assisted_grasp"):
                            self.repositionings[leg.name] += 1
                        if inside:
                            self.first.setdefault(f"{leg.name}/source_visit", t)
                        self.last_nav_inside[leg.name] = inside
            # Object arriving over the receiving surface is admissible from
            # any base pose, not just the baseline's configured staging pose.
            transported = (self._achieved(leg, "lifted") and retained and geo["contained"]
                           and obj["position_xyz"][2] > support.surface_z)
            self._set(f"{leg.name}/transported", transported, historical=True)
            stable = (np.linalg.norm(obj["linear_velocity_xyz"]) <= self.thresholds.settled_linear_speed_m_s
                      and np.linalg.norm(obj["angular_velocity_xyz"]) <= self.thresholds.settled_angular_speed_rad_s)
            candidate = geo["at_target"] and stable and not retained and self._achieved(leg, "transported")
            self.settling[leg.name] = self.settling.get(leg.name, 0.0)+self.dt if candidate else 0.0
            self._set(f"{leg.name}/placed", self.settling[leg.name]+1e-9 >= self.thresholds.settled_duration_s)
            if self.previous is not None and self.previous["objects"][name]["retention_active"] and not retained:
                height = max(0.0, geo["bottom_z"]-support.surface_z)
                self.release_heights[leg.name].append(height)
                self.events.append({"step": self.tick-1, "sim_time_s": t,
                                    "predicate": f"{leg.name}/retention", "value": False,
                                    "name": f"{leg.name}/RETENTION_RELEASED",
                                    "info": f"clearance above destination plane: {height:.3f} m"})
        groups = {group: sum(float(self._achieved(leg, p)) for leg in self.scenario.legs for p in predicates)
                  / (len(self.scenario.legs)*len(predicates)) for group, predicates in self.GROUPS.items()}
        progress = sum(groups.values())/len(groups)
        success = all(self._achieved(leg, "placed") for leg in self.scenario.legs)
        if success and self.completion_first_s is None:
            self.completion_first_s = t
        for event in self.events[start_events:]:
            event["score"] = progress
        self.latest = {"tick": self.tick, "sim_time_s": t, "video_time_s": (self.tick-1)*self.dt,
                       "progress": progress, "groups": groups, "predicates": dict(self.states),
                       "task_success": success, "base_path_length_m": self.distance,
                       "placement": geometry, "observation": snapshot}
        if self.quality_monitor is not None:
            quality = self.quality_monitor.observe(snapshot)
            self.latest["quality_diagnostics"] = quality
            self.latest["source_initially_reached"] = bool(self.initial_source_goals)
            heights = self.release_heights[self.scenario.legs[0].name]
            self.latest["release_clearance_m"] = heights[-1] if heights else None
            for warning in quality["new_warnings"]:
                self.events.append({"step": self.tick-1, "sim_time_s": t,
                    "name": "MOTION_WARNING", "info": f"{warning}: {quality['current'][warning]:.3f} > {quality['advisory_thresholds'][warning]}",
                    "diagnostic": warning, "score": progress})
        self.previous = snapshot
        return self.latest

    def summary(self):
        handoffs = {}
        for label, upstream, downstream in (
            ("navigation_to_manipulation", "navigation_source_reached", "assisted_grasp"),
            ("grasp_to_transport", "assisted_grasp", "transported"),
            ("transport_to_place", "transported", "placed"),
        ):
            legs = {}
            for leg in self.scenario.legs:
                start = self.first.get(f"{leg.name}/{upstream}")
                end = self.first.get(f"{leg.name}/{downstream}")
                eligible = start is not None and (end is None or end >= start)
                initially_ready = (self.scenario.name == "can_dining_to_snack"
                                   and label == "navigation_to_manipulation"
                                   and leg.name in self.initial_source_goals)
                eligible = eligible and not initially_ready
                succeeded = eligible and end is not None
                legs[leg.name] = {"eligible": eligible, "succeeded": succeeded,
                                  "status": "already_at_source_not_tested" if initially_ready else "success" if succeeded else "not_completed" if eligible else "not_observed",
                                  "latency_s": end-start if succeeded else None}
            denominator = sum(x["eligible"] for x in legs.values())
            numerator = sum(x["succeeded"] for x in legs.values())
            handoffs[label] = {"successes": numerator, "eligible": denominator,
                               "rate": numerator/denominator if denominator else None, "legs": legs}
        failures = []
        for leg in self.scenario.legs:
            missing = next((p for p in ("assisted_grasp", "lifted", "transported", "placed") if not self._achieved(leg, p)), None)
            if missing:
                failures.append({"leg": leg.name, "first_unsatisfied_capability": missing,
                                 "interpretation": "observed missing outcome; physical root cause not inferred"})
        result = {
            "schema": "xgrids2_passive_transfer_v1", "scenario": self.scenario.name,
            "task_success": bool(self.latest and self.latest["task_success"]),
            "progress": self.latest["progress"] if self.latest else 0.0,
            "groups": self.latest["groups"] if self.latest else {},
            "group_weights": {g: 1.0 for g in self.GROUPS},
            "predicates": dict(self.states), "first_achievement_time_s": self.first,
            "handoffs": handoffs, "failure_localization": failures,
            "thresholds": asdict(self.thresholds),
            "metrics": {"elapsed_sim_time_s": self.tick*self.dt,
                        "completion_time_s": self.completion_first_s if self.latest and self.latest["task_success"] else None,
                        "base_path_length_m": self.distance, "base_yaw_travel_rad": self.yaw_distance,
                        "arm_tracking_rmse_rad": math.sqrt(self.arm_squared_error/self.arm_samples) if self.arm_samples else None,
                        "arm_integrated_squared_jerk_rad2_s5": self.arm_isj,
                        "maximum_arm_jerk_rad_s3": self.max_jerk,
                        "source_goal_reentries_before_grasp": self.repositionings,
                        "predicate_recoveries": self.recovery_count,
                        "release_clearances_above_destination_m": self.release_heights,
                        "final_placement": self.latest["placement"] if self.latest else {},
                        "collision_count": None, "unassisted_drop_rate": None,
                        "reference_normalized_path_efficiency": None},
            "validity": {
                "policy": "privileged_scripted_curobo_oracle", "perception": "not_evaluated_privileged_object_state",
                "interaction_pose_feasibility": "not_instrumented; successful grasp is observed, not an IK certificate",
                "retention_mode": "contact_verified_temporary_fixed_joint",
                "grasp_stability_slip_drop_resistance": "not_valid_under_retention_assist",
                "lift": "geometric clearance only; not grasp load-bearing quality",
                "collision_count": "not_instrumented_for_whole_body; unknown, not zero",
                "path_reference": "none_available", "comparison": "same physics/retention configuration only",
                "diagnosis": "structured failure localization, not causal root-cause inference",
                "timebase": "post-step control samples; sim time=tick*dt, video time=(tick-1)*dt",
            },
        }
        if self.quality_monitor is not None:
            result["quality_diagnostics"] = self.latest["quality_diagnostics"] if self.latest else None
            result["initial_source_goals"] = sorted(self.initial_source_goals)
        return result
