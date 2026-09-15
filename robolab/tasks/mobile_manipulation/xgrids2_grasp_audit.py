# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Pure, retention-independent measurements for staged grasp validation."""

import math


class GraspAudit:
    """Observe physical outcomes; never command or attach an object.

    Contact is evidence, not proof of a grasp. A lift is scored only after
    8 cm clearance for one second. Relative translation drift is a disclosed
    slip proxy; it does not measure contact-surface slip directly.
    """

    def __init__(self, dt, source_center_z, mode="unassisted"):
        self.dt = dt
        self.source_center_z = source_center_z
        self.mode = mode
        self.ticks = 0
        self.lift_ticks = 0
        self.lift_achieved = False
        self.retention_ever = False
        self.reference_relative = None
        self.max_relative_drift_m = 0.0
        self.drop_events = 0
        self.dropped = False
        self.released = False
        self.events = []

    def observe(self, *, center_z, relative_xyz, retention, left_force, right_force,
                release_commanded=False):
        self.ticks += 1
        self.retention_ever |= retention
        bilateral = min(left_force, right_force) >= 0.1
        lifted_now = center_z >= self.source_center_z + 0.08
        self.lift_ticks = self.lift_ticks + 1 if lifted_now and bilateral else 0
        if self.lift_ticks * self.dt >= 1.0 and not self.lift_achieved:
            self.lift_achieved = True
            self.reference_relative = tuple(relative_xyz)
            self.events.append({"tick": self.ticks, "event": "sustained_lift"})
        self.released |= release_commanded
        drift = 0.0
        if self.reference_relative is not None and not self.released:
            drift = math.dist(relative_xyz, self.reference_relative)
            self.max_relative_drift_m = max(self.max_relative_drift_m, drift)
            dropped = drift > 0.08
            if dropped and not self.dropped:
                self.drop_events += 1
                self.events.append({"tick": self.ticks, "event": "object_separated_from_gripper"})
            self.dropped = dropped
        return {"tick": self.ticks, "sim_time_s": self.ticks * self.dt,
                "bilateral_contact": bilateral, "sustained_lift": self.lift_achieved,
                "lift_clearance_m": center_z - self.source_center_z,
                "relative_drift_m": drift, "retention_active": retention,
                "release_commanded": self.released}

    def summary(self):
        return {"mode": self.mode, "control_ticks": self.ticks,
                "sustained_lift": self.lift_achieved,
                "retention_ever_active": self.retention_ever,
                "unassisted_integrity_passed": not self.retention_ever if self.mode == "unassisted" else None,
                "maximum_relative_translation_drift_m": self.max_relative_drift_m,
                "separation_events": self.drop_events, "events": self.events,
                "limitation": "30 Hz observations; drift is an object-to-tool translation proxy, not contact-surface slip or a full collision audit"}
