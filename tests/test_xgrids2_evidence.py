import json
from types import SimpleNamespace

import cv2
import numpy as np

from robolab.tasks.mobile_manipulation.xgrids2_evidence import (
    choose_camera_shot,
    inspect_video,
    make_contact_sheet,
    render_camera_panel,
    update_benchmark_manifest,
)


def test_camera_shot_selection_uses_anchor_then_context():
    context = SimpleNamespace(name="context", base_anchor_xy=None)
    close = SimpleNamespace(
        name="close", base_anchor_xy=(1.0, 2.0), activation_radius_m=0.5
    )
    scenario = SimpleNamespace(camera_shots=(context, close))

    assert choose_camera_shot(scenario, (1.1, 2.1)) is close
    assert choose_camera_shot(scenario, (5.0, 5.0)) is context


def test_panel_and_contact_sheet_have_stable_shapes():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    assert render_camera_panel(frame, "camera", "state").shape == (270, 480, 3)
    assert make_contact_sheet([frame] * 5, columns=4).shape == (540, 1920, 3)


def test_manifest_merge_preserves_other_scenarios(tmp_path):
    names = ("can", "plate")
    update_benchmark_manifest(tmp_path, "can", {"result": {"passed": True}}, names)
    path = update_benchmark_manifest(
        tmp_path, "plate", {"result": {"passed": False}}, names
    )

    payload = json.loads(path.read_text())
    assert set(payload["scenarios"]) == set(names)
    assert payload["complete_pack"] is True
    assert payload["all_recorded_scenarios_passed"] is False


def test_video_inspection_decodes_and_detects_change(tmp_path):
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24)
    )
    assert writer.isOpened()
    for value in (0, 64, 128):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()

    report = inspect_video(path)
    assert report["readable"] is True
    assert report["frame_count"] == 3
    assert report["maximum_adjacent_changed_fraction"] > 0.9
