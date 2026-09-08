# Continuous can placement

The earlier single-can video visibly slowed at two intermediate placement
targets. The control trace confirmed that the requested joint velocities went
near zero at those boundaries: these were commanded stops, not an arm that kept
receiving moving targets while failing to follow them.

The can diagnostic profile now requests a single CuRobo trajectory from the
pre-place pose to the final release pose. It no longer executes independent
`place_approach` and `place_lower` trajectories. The new segment is called
`place_continuous`; start/end easing is applied once to that whole segment.
The pre-place staging motion and earlier carry reorientations remain separate.
This change does not claim the entire pick-and-place episode is one trajectory.

## Scope and safety

- `CanDiagnosticProfile` identifies the change as
  `can_diagnostic_v2_continuous_place`, with `continuous_place=True`.
- `XGRIDS2CuroboOracle.run` selects one placement goal for this profile; the
  legacy/non-diagnostic branch retains its existing intermediate goals.
- `_plan_execute` applies the existing loaded-object target-step bound and final
  placement tracking tolerance to the new segment.
- The continuous plan keeps hand and attached-object collision
  spheres enabled, unlike the previous short contact-enabled final segment.
  A failed plan is not silently replaced with an unchecked direct motion.
- The one-second pre-release hold, receiving-surface containment check,
  maximum 1.5 cm bottom clearance, low object-speed guard, retention release,
  and passive post-release settling requirements remain unchanged.
- The empty-hand retreat reverses the new planned placement path. No object or
  robot teleportation is added. Physics, base control, grasp logic, and scenario
  geometry are unchanged.

The runner saves the full diagnostic-profile dataclass in
`result.json:execution_profile_config`, not just its CLI selector.

## Verification

`examples/audit_xgrids2_continuous_place.py` compares commanded and measured arm
near-stops with the previous recording. A near-stop means all seven absolute
joint speeds are below 0.005 rad/s for at least 0.25 seconds. The initial 0.5
seconds and final ease-out/hold are excluded from the intermediate-stop test.
This is a disclosed operational pause detector, not a perceptual smoothness
standard or proof of physical root cause.

The can is validated with a full non-video physical trial, then an independent
full recorded trial. `export_xgrids2_can_video_variants.py` produces the two
requested videos from the new recording: camera-only and instant-relevant
metrics. These are not time edits to the previous run.

New evidence is kept separately under
`output/xgrids2_lab/can_continuous_place_20260905/`. Earlier accepted videos and
traces remain untouched.

### Full non-video calibration

The full seed-0 trial passed, including the passive task evaluator, route
clearance, final release, and settling. It completed in 120.13 simulated seconds;
stable placement was first observed at 113.00 seconds. Final XY error was 4.91 mm,
and measured release clearance was 8.30 mm. Retention was inactive at the end.

The new `place_continuous` segment, including its final safety hold, lasted
6.30 seconds versus 18.73 seconds for the earlier approach/lower/place sequence.
The same near-stop detector found **two intermediate command pauses and two
measured pauses in the old recording, and zero of either in the new trial**.
Peak measured arm jerk for the full new run was 21.75 rad/s³; this is descriptive
single-run evidence, not a general guarantee of jerk-free motion.

All 3,604 control samples and nine evaluation events replayed exactly. Twenty
focused CPU tests and 47 simulator-backed controller/scenario compatibility tests
passed. The successful single placement plan used the stricter collision setting;
no contact-disabled or unchecked fallback was needed.

### Independent recorded trial

The independent seed-0 recording also passed the full task, route-clearance,
release, and settling checks. It contains 3,628 frames at 30 FPS, covering
120.93 simulated seconds at 1× playback. Stable placement was first observed
at 113.97 seconds. Final XY error was 4.93 mm and release clearance was 8.87 mm;
retention was inactive at completion.

The continuous placement segment runs from 106.27 to 112.47 seconds, including
the final safety hold: **6.20 seconds versus 18.73 seconds previously**. The
same trace audit found **zero intermediate commanded or measured near-stops**,
versus two of each in the old recording. The intentional final stop before
release remains. Peak measured arm jerk over the full recorded episode was
30.49 rad/s³; earlier pickup/retraction command alerts are outside this fix.
This result does not claim globally jerk-free motion.

All 3,628 control samples and nine evaluator events replayed exactly. The
recording audit passed, and both exports decoded all 3,628 frames. Sampled
camera-image comparisons verified correspondence with the source recording;
camera pixels changed during motion. Placement/descent and post-release
frames were also visually inspected.

Shareable videos, with no cuts or speed changes:

- [Camera-only video](../output/xgrids2_lab/can_continuous_place_20260905/video/video_versions/can_pick_place_clean.mp4)
- [Current-relevant metrics video](../output/xgrids2_lab/can_continuous_place_20260905/video/video_versions/can_pick_place_current_metrics.mp4)

Machine-readable checks are saved beside the recording as
`continuous_place_audit.json`, `audit/recording_audit.json`, and
`video_versions/video_variants.json`. The original recording and control trace
are retained, alongside the independent non-video calibration.

## Reproduce

```bash
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/validate_xgrids2_lab_scenario.py \
  --scenario can_dining_to_snack --execution-profile can_diagnostic \
  --stage full --seed 0 --headless --record-video --video-fps 30 --capture-every 1 \
  --output-dir output/xgrids2_lab/can_continuous_place_20260905/video

/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/audit_xgrids2_continuous_place.py \
  output/xgrids2_lab/can_continuous_place_20260905/video \
  --reference output/xgrids2_lab/can_diagnostic_20260905/video

/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/export_xgrids2_can_video_variants.py \
  output/xgrids2_lab/can_continuous_place_20260905/video
```

Choose a new output directory for another trial to preserve these recordings.
The exported variants retain every frame and the 30 Hz physical timebase;
they do not hide pauses with cuts or accelerated playback.
