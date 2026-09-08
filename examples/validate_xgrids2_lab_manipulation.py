#!/usr/bin/env python3
"""Compatibility launcher for the XGrids2 can-transfer scenario."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    runner = Path(__file__).with_name("validate_xgrids2_lab_scenario.py")
    defaults = [
        "--scenario",
        "can_dining_to_snack",
        "--execution-profile",
        "legacy",
        "--output-dir",
        "output/xgrids2_lab/manipulation_gate",
        "--video-fps",
        "30",
        "--capture-every",
        "1",
    ]
    # Replace this process so Isaac is initialized exactly once by the runner.
    os.execv(sys.executable, [sys.executable, str(runner), *defaults, *sys.argv[1:]])


if __name__ == "__main__":
    main()
