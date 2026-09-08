#!/usr/bin/env python3
"""Compatibility launcher for the XGrids2 plate-to-sink scenario."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    runner = Path(__file__).with_name("validate_xgrids2_lab_scenario.py")
    defaults = [
        "--scenario",
        "plate_dining_to_sink",
        "--output-dir",
        "output/xgrids2_lab/plate_to_sink",
        "--num-seeds",
        "8",
        "--video-fps",
        "8",
        "--capture-every",
        "8",
    ]
    # User-provided options come last and therefore override these defaults.
    os.execv(sys.executable, [sys.executable, str(runner), *defaults, *sys.argv[1:]])


if __name__ == "__main__":
    main()
