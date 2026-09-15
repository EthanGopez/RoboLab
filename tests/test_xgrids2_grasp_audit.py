# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("grasp_audit", Path(__file__).resolve().parents[1] / "robolab/tasks/mobile_manipulation/xgrids2_grasp_audit.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GraspAudit = module.GraspAudit


def sample(a, **kw):
    row = dict(center_z=1.1, relative_xyz=(0,0,0), retention=False,
               left_force=1, right_force=1)
    row.update(kw)
    return a.observe(**row)


def test_contact_alone_is_not_lift():
    a = GraspAudit(1/30, 1)
    for _ in range(60):
        sample(a, center_z=1)
    assert not a.lift_achieved


def test_airborne_without_contact_is_not_a_grasp():
    a = GraspAudit(1/30, 1)
    for _ in range(60):
        sample(a, left_force=0, right_force=0)
    assert not a.lift_achieved


def test_unassisted_lift_does_not_require_retention():
    a = GraspAudit(1/30, 1)
    for _ in range(30):
        sample(a)
    assert a.lift_achieved
    assert a.summary()["unassisted_integrity_passed"]


def test_any_retention_invalidates_unassisted_integrity():
    a = GraspAudit(1/30, 1)
    sample(a, retention=True)
    sample(a)
    assert not a.summary()["unassisted_integrity_passed"]


def test_separation_and_deliberate_release_distinct():
    a = GraspAudit(1/30, 1)
    for _ in range(30):
        sample(a)
    sample(a, relative_xyz=(.1,0,0))
    sample(a, relative_xyz=(.2,0,0))
    assert a.drop_events == 1
    b = GraspAudit(1/30, 1)
    for _ in range(30):
        sample(b)
    sample(b, relative_xyz=(.2,0,0), release_commanded=True)
    assert b.drop_events == 0
