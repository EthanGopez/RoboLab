"""Small, simulator-independent helpers for XGrids2 evidence artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def camera_rgb(camera: Any) -> np.ndarray:
    """Copy one Isaac camera tensor into an RGB uint8 array."""
    return np.ascontiguousarray(
        camera.data.output["rgb"][0, :, :, :3]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


def render_camera_panel(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    size: tuple[int, int] = (480, 270),
) -> np.ndarray:
    """Resize a camera frame and add its fixed two-line header."""
    width, height = size
    panel = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (width, 52), (15, 19, 25), -1)
    cv2.putText(
        panel, title, (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1
    )
    cv2.putText(
        panel, subtitle, (14, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 220, 255), 1
    )
    return panel


def choose_camera_shot(scenario: Any, xy: tuple[float, float]) -> Any:
    """Select the closest active anchored shot, then fall back to context."""
    anchored = [shot for shot in scenario.camera_shots if shot.base_anchor_xy]
    if anchored:
        distances = [math.dist(xy, shot.base_anchor_xy) for shot in anchored]
        best = anchored[int(np.argmin(distances))]
        if min(distances) <= best.activation_radius_m:
            return best
    return next(
        (shot for shot in scenario.camera_shots if shot.base_anchor_xy is None),
        scenario.camera_shots[0],
    )


def set_camera_shot(camera: Any, *, device: Any, shot: Any) -> None:
    """Apply a configured third-person shot to an Isaac camera."""
    # Imported lazily so artifact-only tests do not need to initialize Isaac.
    import torch

    camera.set_world_poses_from_view(
        eyes=torch.tensor([shot.eye], device=device, dtype=torch.float32),
        targets=torch.tensor([shot.target], device=device, dtype=torch.float32),
    )


def make_contact_sheet(frames: list[np.ndarray], *, columns: int = 4) -> np.ndarray:
    """Build a bounded 12-frame overview from a potentially long recording."""
    if not frames:
        raise ValueError("contact sheet requires at least one frame")
    selected = frames
    if len(frames) > 12:
        indices = np.linspace(0, len(frames) - 1, 12).round().astype(int)
        selected = [frames[index] for index in indices]
    cells = [
        cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
        for frame in selected
    ]
    rows = int(np.ceil(len(cells) / columns))
    cells.extend(np.zeros_like(cells[0]) for _ in range(rows * columns - len(cells)))
    return np.concatenate(
        [
            np.concatenate(cells[start : start + columns], axis=1)
            for start in range(0, len(cells), columns)
        ],
        axis=0,
    )


def inspect_video(path: Path) -> dict[str, Any]:
    """Fully decode a video and report simple readability/motion evidence."""
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = last = previous = None
    maximum_change = change_sum = 0.0
    adjacent_pairs = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        first = frame if first is None else first
        if previous is not None:
            changed = float(np.mean(cv2.absdiff(previous, frame) > 8))
            maximum_change = max(maximum_change, changed)
            change_sum += changed
            adjacent_pairs += 1
        previous = last = frame
    capture.release()
    first_last_change = 0.0
    if first is not None and last is not None:
        first_last_change = float(np.mean(cv2.absdiff(first, last) > 8))
    return {
        "readable": first is not None and last is not None,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "first_last_changed_fraction": first_last_change,
        "maximum_adjacent_changed_fraction": maximum_change,
        "mean_adjacent_changed_fraction": change_sum / adjacent_pairs if adjacent_pairs else 0.0,
    }


def update_benchmark_manifest(
    pack_root: Path,
    scenario_name: str,
    entry: dict[str, Any],
    scenario_names: Iterable[str],
) -> Path:
    """Merge one run into the pack manifest without changing other entries."""
    ordered_names = tuple(scenario_names)
    path = pack_root / "benchmark_manifest.json"
    payload = {
        "benchmark": "xgrids2_configurable_manipulation_pack_v1",
        "scenario_order": list(ordered_names),
        "scenarios": {},
    }
    if path.is_file():
        payload = json.loads(path.read_text())
    payload.setdefault("scenarios", {})[scenario_name] = entry
    payload["all_recorded_scenarios_passed"] = all(
        item.get("result", {}).get("passed", False)
        for item in payload["scenarios"].values()
    )
    payload["complete_pack"] = set(payload["scenarios"]) == set(ordered_names)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
