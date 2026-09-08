#!/usr/bin/env python3
"""Validate and record one configurable XGrids2 benchmark scenario."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from robolab.core.utils.nurec_rendering import (  # noqa: E402
    configure_xgrids_vendor_nurec_volume,
    require_single_gpu,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import (  # noqa: E402
    list_xgrids2_scenarios,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", choices=list_xgrids2_scenarios(), required=True)
parser.add_argument("--stage", choices=("pick", "carry", "place", "full"), default="full")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--execution-profile",
    choices=("auto", "legacy", "can_diagnostic"),
    default="auto",
    help="auto uses the diagnostic profile only for the single-can task",
)
parser.add_argument("--num-seeds", type=int, default=12)
parser.add_argument("--record-video", action="store_true")
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--capture-every", type=int, default=1)
parser.add_argument(
    "--record-control-trace",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="stream per-control-tick arm and retention diagnostics",
)
parser.add_argument("--output-dir")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
require_single_gpu(args_cli)

simulation_app = AppLauncher(args_cli).app
nurec_settings = configure_xgrids_vendor_nurec_volume()

# Isaac-dependent modules must be imported after SimulationApp starts.
from robolab.tasks.mobile_manipulation.xgrids2_runner import (  # noqa: E402
    run_xgrids2_scenario,
)


try:
    run_xgrids2_scenario(
        args_cli, repo_root=REPO_ROOT, nurec_settings=nurec_settings
    )
finally:
    simulation_app.close()
