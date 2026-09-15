# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Check external XGrids2 assets and the pinned CuRobo runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / (
    "assets/scenes/xgrids2_lab/new_data_0827/isaacsim-import-package/"
    "ut_lab_1_scene_pack/0/ut_lab_1.usdz"
)
SCENE_SHA256 = "f2dea7d23fe0c277051d944d254c0d1eeeb95e801f14d8aa72d6cd4d5a777115"
CUROBO_REVISION = "ebb71702f3f70e767f40fd8e050674af0288abe8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def curobo_status() -> dict:
    spec = importlib.util.find_spec("curobo")
    result = {"installed": spec is not None, "expected_revision": CUROBO_REVISION}
    if spec is None or spec.origin is None:
        return result
    source = Path(spec.origin).resolve()
    result["module"] = str(source)
    for parent in source.parents:
        if (parent / ".git").exists():
            revision = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                text=True, capture_output=True, check=False,
            )
            if revision.returncode == 0:
                result["revision"] = revision.stdout.strip()
                result["revision_matches"] = result["revision"] == CUROBO_REVISION
            break
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-checksum", action="store_true")
    args = parser.parse_args()
    scene = {"path": str(SCENE), "exists": SCENE.is_file()}
    if scene["exists"]:
        scene["size_bytes"] = SCENE.stat().st_size
        if args.verify_checksum:
            scene["sha256"] = sha256(SCENE)
            scene["checksum_matches"] = scene["sha256"] == SCENE_SHA256
    report = {
        "scene": scene,
        "curobo": curobo_status(),
        "ffmpeg": shutil.which("ffmpeg"),
    }
    # RoboLab's OpenCV writer can record without a system ffmpeg executable;
    # report ffmpeg for richer codec tooling, but do not make it a run blocker.
    checks = [scene["exists"], report["curobo"]["installed"]]
    if args.verify_checksum and scene["exists"]:
        checks.append(scene["checksum_matches"])
    if "revision_matches" in report["curobo"]:
        checks.append(report["curobo"]["revision_matches"])
    report["passed"] = all(checks)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
