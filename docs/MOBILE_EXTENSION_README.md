# RoboLab mobile-manipulation extension

This branch extends RoboLab from fixed-base manipulation to room-scale mobile
manipulation. It provides a simplified planar CoBot carrying a Kinova Gen3 arm,
reference CuRobo controllers, configuration-driven XGrids2 tasks, passive task
evaluation, synchronized traces/videos, and dashboard playback.

## What is included

- A physics-native planar base commanded by `(vx, vy, vtheta)`.
- Joint-position control for the Kinova arm and position control for the gripper.
- One common scenario runner for five XGrids2 manipulation demonstrations.
- A passive evaluator for correctness, progress, capability handoffs, structured
  failure localization, and focused trajectory-quality diagnostics.
- Control-rate traces and synchronized third-person, navigation, and wrist video.
- Fixed-base and mobile kitchen reference controllers used during development.

The current controllers are deterministic reference baselines. They do not
define the only valid solution to a benchmark task.

## Installation

The extension was validated with Python 3.11, Isaac Sim 5.1, and Isaac Lab
2.3.2.post1:

```bash
git clone https://github.com/SetasAditya/RoboLab.git
cd RoboLab
git switch mobile-extension
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra isaac51 --extra xgrids2
```

The upstream `isaac50` environment remains supported by RoboLab, but the
XGrids2 evidence in this branch was calibrated on `isaac51`; compare benchmark
results only under the same simulator and task configuration.

## External scene assets

Large third-party scene files are intentionally not stored in Git. Obtain them
from their respective owners and preserve their licenses.

For XGrids2, place the Nucleus4D Isaac package at:

```text
assets/scenes/xgrids2_lab/new_data_0827/isaacsim-import-package/
  ut_lab_1_scene_pack/0/ut_lab_1.usdz
```

The lightweight registered LabApt vector map is included for collision overlays
and route planning. The vendor USDZ is loaded strictly as collision-free visual
appearance; clean fixtures and task objects provide simulation physics.

The optional kitchen demonstrations expect the EBench scene at:

```text
assets/third_party/ebench/assets/scene_usds/ebench/long_horizon/
  scene_dishwasher/scene.usd
```

Validate the external setup before launching Isaac:

```bash
python examples/check_xgrids2_setup.py --verify-checksum
```

## Run an XGrids2 scenario

The only benchmark-eligible scenario currently is the unassisted can transfer:

```bash
python examples/validate_xgrids2_lab_scenario.py --help
```

Run the current single-can diagnostic:

```bash
python examples/validate_xgrids2_lab_scenario.py \
  --scenario can_dining_to_snack \
  --stage full --seed 0 --headless
```

Record synchronized real-time evidence:

```bash
python examples/validate_xgrids2_lab_scenario.py \
  --scenario can_dining_to_snack \
  --stage full --seed 0 --headless \
  --record-video --video-fps 30 --capture-every 1
```

By default, artifacts are written under:

```text
output/xgrids2_lab/benchmark_pack/<scenario>/
```

Use `--output-dir` for diagnostic trials so accepted evidence is not replaced.
The other four scenario records are retained for development but are blocked
from benchmark execution until their fixed-joint retention paths are replaced
and physically validated.

## Evaluation outputs

Depending on the scenario and recording options, a run writes:

```text
result.json
evaluation.json
evaluation_trace.jsonl
evaluation_events.json
control_trace.jsonl
video_frame_map.json
task.mp4
montage.png
contact_sheet.jpg
```

The predicate evaluator observes execution continuously and never advances or
gates the policy. Privileged simulator state is evaluator-only and is not added
to the policy observation interface.

## Validation

Pure evaluator and evidence tests can run without initializing Isaac:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest \
  tests/test_xgrids2_evidence.py \
  tests/test_xgrids2_can_diagnostic.py \
  tests/test_xgrids2_continuous_place.py \
  tests/test_xgrids2_passive_evaluation.py \
  tests/test_xgrids2_video_variants.py \
  tests/test_xgrids2_diagnostic_viewer.py -q
```

Simulator-backed registration and controller tests require the Isaac environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_mobile_cobot.py \
  tests/test_xgrids2_lab_scenario_controller.py \
  tests/test_xgrids2_lab_scenarios.py \
  tests/test_xgrids2_lab_scenario_factory.py -q
```

See [XGRIDS2_CODEBASE_GUIDE.md](XGRIDS2_CODEBASE_GUIDE.md) for code ownership and
[XGRIDS2_PASSIVE_EVALUATION.md](XGRIDS2_PASSIVE_EVALUATION.md) for metric semantics.

## Current limitations

- The base abstracts wheels and drivetrain mechanics; it is not a mecanum model.
- Benchmark execution forbids fixed-joint retention. The current can task uses
  physical gripper contact for grasp, lift, transport, and release.
- Plate, multi-object, clutter, and pouring configurations are development-only
  until each has an unassisted manipulation implementation.
- The pouring scenario is a pose-level baseline with no simulated liquid.
- Whole-body collision counting and perception success are not yet fully
  instrumented; unavailable metrics are reported as such rather than as zero.
- No claims are made about wheel slip, battery use, motor effort, or real-robot
  odometry drift.
- Nucleus4D scene data and EBench assets are not redistributed by this branch.
