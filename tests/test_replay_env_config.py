# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from types import SimpleNamespace

from robolab.core.replay.env_config import apply_recorded_env_cfg


def _predicate(*, object):
    return object


def test_overlay_preserves_callable_subtask_conditions():
    condition = partial(_predicate, object="banana")
    cfg = SimpleNamespace(
        subtasks=[
            SimpleNamespace(
                conditions={"banana": [(condition, 0.25)]},
                score=1.0,
            )
        ]
    )
    recorded = {
        "subtasks": [
            {
                "conditions": {
                    "banana": [
                        [
                            "functools.partial(<function _predicate at 0x1234>, object='banana')",
                            0.5,
                        ]
                    ]
                },
                "score": 2.0,
            }
        ]
    }

    skipped = apply_recorded_env_cfg(cfg, recorded)

    restored_condition, restored_weight = cfg.subtasks[0].conditions["banana"][0]
    assert restored_condition is condition
    assert restored_weight == 0.25
    assert cfg.subtasks[0].score == 2.0
    assert any("/subtasks[0]/conditions/banana" in item for item in skipped)


def test_overlay_still_replaces_plain_lists():
    cfg = SimpleNamespace(values=[1, 2])

    skipped = apply_recorded_env_cfg(cfg, {"values": [3, 4]})

    assert skipped == []
    assert cfg.values == [3, 4]
