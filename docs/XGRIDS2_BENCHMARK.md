# XGrids2 mobile-manipulation benchmark

## Current benchmark contract

The benchmark currently exposes one validated task:

```text
can_dining_to_snack
```

It uses the stock RoboLab YCB tomato-soup can and physical Robotiq fingertip
contact for grasp, lift, transport, unloading, and release. Fixed-joint
retention is forbidden: observing it at any control tick invalidates the run.

Four additional typed scenario records remain in the registry for development:

- `plate_dining_to_sink`
- `can_and_cube_dining_to_snack`
- `lime_clutter_to_bowl`
- `mustard_pour_into_bowl`

They are not benchmark-eligible until their retention-dependent controllers
are replaced and independently validated with unassisted manipulation. The
pour configuration is pose-level only and contains no simulated liquid.

## Architecture

```text
scenario record -> generated Isaac scene -> controller/policy
                                      |
                                      +-> passive evaluator -> trace/evidence
```

The evaluator observes every physical control step. It never gates the policy
or requires the reference controller's phase order. Privileged state used for
scoring is not added to policy observations.

The Nucleus NuRec USDZ supplies collision-disabled RGB appearance. A registered
LabApt vector map supplies coarse navigation geometry, while explicit colliders
model the floor and task-relevant supports. Interactive objects are separate
RoboLab rigid-body assets.

## Run

```bash
uv sync --extra isaac51 --extra xgrids2
python examples/check_xgrids2_setup.py --verify-checksum
python examples/validate_xgrids2_lab_scenario.py \
  --scenario can_dining_to_snack --stage full --seed 0 --headless
```

Add `--record-video --video-fps 30 --capture-every 1` for synchronized
third-person, navigation-camera, wrist-camera, and control-trace evidence.

## Evaluation

The initial evaluator reports:

- physical contact, sustained grasp, lift, transport, release, and placement;
- capability progress and navigation-to-grasp, grasp-to-transport, and
  transport-to-place handoffs;
- structured failure localization;
- completion time, base path length/yaw travel, arm tracking error and jerk;
- placement geometry and motion-quality diagnostics.

Perception remains unscored for the privileged reference controller. Complete
whole-body collision counting and reference-normalized path efficiency remain
unavailable and are reported as unavailable rather than zero.

## External assets

The vendor scene is intentionally not redistributed in Git. Its required path
and checksum are checked by `examples/check_xgrids2_setup.py`. Coordinate
registration details are in `XGRIDS2_LABAPT_ALIGNMENT.md`; physics construction
is documented in `XGRIDS2_LAB_PHYSICS_WALKTHROUGH.md`.
