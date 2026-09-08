# Single-can motion and task diagnosis

The subsequent [continuous placement change](XGRIDS2_CONTINUOUS_PLACEMENT.md)
removes the two intermediate descent stops. Evidence below describes the earlier
diagnostic profile and remains available for comparison.

This diagnostic uses `can_dining_to_snack`, seed 0. It separates **task
correctness** from **trajectory quality**, with synchronized three-camera
evidence at one frame per control step and 30 FPS (1x simulated time).
It does not establish hardware fidelity or unassisted grasp robustness.

## What changed

### 1. Remove the base controller's arrival snap

`robolab/robots/mobile_base_math.py:jerk_limited_velocity_step` previously
limited acceleration changes, but snapped velocity to the requested value and
reset acceleration when crossing that value. That reset violated the actual
finite-difference jerk bound. A simple yaw-command test reproduced approximately
140 rad/s³ despite a configured 4 rad/s³ limit.

The replacement uses a damped second-order velocity servo:

```text
frequency = max_jerk / max_acceleration
jerk = clamp(frequency² * (target_velocity - velocity)
             - 2 * frequency * acceleration, ±max_jerk)
acceleration_next = clamp(acceleration + jerk * dt, ±max_acceleration)
velocity_next = velocity + acceleration_next * dt
```

There is no crossing snap or acceleration reset. The controller settles
asymptotically; saturated reversals can overshoot. Limits constrain commanded
motion, not measured motion under contact. PhysX still integrates the dynamically
driven base, arm, and payload. No robot/object state teleportation was added.

This shared fix affects all users of this base backend. Only the single-can
task is being physically recalibrated here; other scenarios need fresh trials.

### 2. Scoped faster, smoother can execution

`xgrids2_can_diagnostic.py:CanDiagnosticProfile` holds the can-only parameters.
The runner's `--execution-profile auto` selects `can_diagnostic` for the single
can and leaves other scenario strategies on `legacy`. Direct oracle construction
still defaults to `legacy` for compatibility. Note that `legacy` does **not** undo
the shared base limiter repair.

- Loaded base speed cap: 0.06 → 0.15 m/s; yaw cap: 0.30 rad/s.
- The optional `WaypointFollower.smooth_heading` blends initial to final chassis
  yaw with a quintic function of monotone route progress. It does not chase every
  short waypoint bearing or switch back and forth at the final-position radius.
  A holonomic base can translate sideways; its navigation camera remains rigidly
  mounted, not artificially aimed along the route.
- Timed arm trajectories request a 2x speed factor, subject to per-tick target
  displacement limits (twice the old segment limit, ceiling 0.018 rad).
  Endpoint time ramps reduce start/stop discontinuities. Samples remain on the
  original unwrapped joint-space polyline. This is **not** a guarantee of globally
  C2 motion at internal planner corners, nor a hard measured-jerk guarantee.
- The close transition holds the last commanded arm target instead of replacing
  it with the lagging measured position. Finger commands freeze at measured
  bilateral contact, contact is rechecked after a short dwell, then the existing
  coincident-frame retention joint is activated.
- Before release, the oriented can footprint must be inside the receiving
  support, bottom clearance must be at most 1.5 cm, and linear speed must be at
  most 0.05 m/s. A failed check stops the trial rather than releasing from height.
  No special downward teleport or unplanned lowering was introduced.

Existing route-clearance and arm-motion safety gates remain. Existing settling
holds remain, including the evaluator's one-second post-release settling test.
This is a calibrated scripted baseline, not a claim of optimal execution.

### 3. Readable passive diagnosis

The evaluator observes every completed environment control step independently
of the oracle's phase labels. It never advances or gates execution. Release
safety checks above belong to the controller, not this passive evaluator.

The video appends a diagnostic footer below the original three camera panels:

- Contact established; grasp established **with assistance**; object cleared
  source; object transported over destination; released and settled at target.
- Navigation → grasp, grasp → transport, transport → place handoffs. If the
  robot began at the source goal, navigation is explicitly **not tested** and is
  excluded from the navigation-handoff rate denominator.
- Simulated elapsed time, measured base path length, target XY error, measured
  release clearance, and retention state.
- Motion-quality warnings, independently of task correctness; camera cuts are
  labeled briefly instead of being confused with physical robot jolts.

Advisory thresholds are developmental heuristics, not validated perceptual or
hardware limits: measured arm jerk 50 rad/s³, commanded arm jerk 100 rad/s³,
measured base yaw acceleration 1 rad/s². A warning episode begins on each
below-to-above-threshold transition; counts across signals can describe the
same physical event. First three samples are excluded from derivative warnings.
Raw values, thresholds, peaks, and event timestamps are retained. No warning is
silently converted into task failure or a claim that the simulator is broken.

Perception is privileged/not evaluated; interaction-pose feasibility is not
instrumented. Whole-body collision count is unknown, not zero. Assisted grasp
slip/drop robustness is not valid. Reference-normalized path efficiency is
unavailable until a reference exists. Geometric release/settling remain observed
outcomes. Progress percentages are retained in data, but are not the main video
explanation.

## Code map and evidence

| Component | Responsibility |
|---|---|
| `xgrids2_can_diagnostic.py` | Immutable execution profile, path-preserving time easing, advisory monitor, plain-language footer |
| `xgrids2_lab_curobo_oracle.py` | Profile selection during execution, close handoff, release guard |
| `xgrids2_lab_navigation_oracle.py` | Optional continuous route-progress heading |
| `xgrids2_passive_evaluation.py` | Single-can observations, initial-goal eligibility, warning timeline |
| `xgrids2_evaluation_observer.py` | Measured physics snapshots, world-frame flange motion, and applied base velocity/acceleration/wrench |
| `xgrids2_evaluation_export.py` | Dashboard-compatible HDF5 signals and event timeline |
| `dashboard/static/passive-evaluation.js` | Readable outcome and handoff labels during playback |
| `validate_xgrids2_lab_scenario.py` | Profile CLI, synchronized capture, appended diagnostic footer |
| `audit_xgrids2_evaluation.py` | Exact scoring replay and frame/tick alignment |
| `audit_xgrids2_recording.py` | Full MP4 decode and non-overlay camera-motion checks |

Run a gate, then a full trial, then an independent recorded trial:

```bash
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/validate_xgrids2_lab_scenario.py \
  --scenario can_dining_to_snack --stage full --seed 0 \
  --execution-profile can_diagnostic --headless \
  --record-video --video-fps 30 --capture-every 1 \
  --output-dir output/xgrids2_lab/can_diagnostic_20260905/video
```

Use `--stage pick` and omit `--record-video` for the pick gate. Use separate
output directories for each trial. Custom-output trials write their manifest
beside those directories, not over the accepted benchmark-pack manifest.

Each trial includes `result.json`, `evaluation.json`, `control_trace.jsonl`,
`evaluation_trace.jsonl`, `evaluation_events.json`, and dashboard exports.
Recorded trials also include `task.mp4`, `video_frame_map.json`, montage/contact
sheet, and audit results. Every trace has physical timestamps and retention state.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  -m pytest --noconftest tests/test_xgrids2_can_diagnostic.py \
  tests/test_xgrids2_passive_evaluation.py -q
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/audit_xgrids2_evaluation.py output/xgrids2_lab/can_diagnostic_20260905/video
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/audit_xgrids2_recording.py output/xgrids2_lab/can_diagnostic_20260905/video
```

## Calibration log

The first pick attempt was interrupted by an evaluator bug: historical-predicate
recovery assumed every event had a `predicate` field, but motion warnings are a
different event type. Optional-key handling and a regression test now cover a
warning followed by a grasp achievement. The failed attempt is preserved in
`pick/`; it is not evidence of physical pick failure.

The corrected pick gate (`pick_v2/`) passed in 16 simulated seconds. Exact
evaluation replay passed for all 480 ticks. The task evaluator correctly reports
50% macro progress and no full-task success: this was a pick-only gate.

The full non-video gate (`full/`) passed: 3,964 control ticks; 132.13 simulated
seconds including retreat; first stable placement at 124.93 seconds. Release
clearance was 6.20 mm; final XY error was 6.11 mm. Route clearance passed,
retention was released, and the independent passive evaluator reported success.
All ten evaluation events replay exactly.

| Diagnostic | Earlier can leg | New full non-video trial |
|---|---:|---:|
| Duration through retreat | 245.23 s | 132.13 s |
| Base traversal phase | 61.40 s | 35.73 s |
| Base path length | 3.104 m | 3.132 m |
| Peak measured arm jerk | 37.58 rad/s³ | 26.85 rad/s³ |
| Peak commanded arm jerk | 251.55 rad/s³ | 147.48 rad/s³ |
| Peak measured base yaw acceleration | 1.970 rad/s² | 0.217 rad/s² |
| Peak measured base yaw jerk | 29.50 rad/s³ | 9.22 rad/s³ |

The reference is the can portion of
`passive_evaluation/can_and_cube_seed0`, stopping before its empty return for the
cube. This is a descriptive comparison across different episode contexts, not a
controlled A/B, human reference, or optimality claim. Use
`examples/analyze_xgrids2_can_diagnostic.py` to reproduce it.

Three commanded-arm advisory episodes remain, with the largest inside the
approach trajectory at 6.53 seconds. There were zero measured-arm or base-turning
advisory episodes in this trial. Endpoint easing cannot remove all internal
polyline derivative discontinuities. The report deliberately retains those
warnings rather than claiming jerk-free motion. Measured base jerk can exceed
the command-limiter bound because contacts and arm/payload forces participate
in the dynamics; these are different signals.

Validation also passed 47 simulator-backed mobile-controller/scenario tests and
17 focused CPU evaluator/trajectory/dashboard tests. The playback regression
still checks forward play and backward seek do not expose future achievements;
its expected text was updated to the new plain-language labels.

### Independent recorded trial

`video/` passed both controller acceptance and passive task correctness:

- 3,982 frames, 1440×730, 30 FPS, 132.73 seconds at **1x** playback; approximately
  50.7 MiB. Stable placement was first observed at 125.77 simulated seconds.
- Release clearance 8.32 mm, final XY error 6.14 mm, base path 3.130 m. Route
  clearance passed; retention was disabled at the end; no legacy motion-quality
  gate violation occurred.
- Two sharp-command advisory episodes: approach at 6.53 seconds and retract at
  22.17 seconds. Peak target jerk was 148.15 rad/s³. No measured-arm jerk exceeded
  the 50 rad/s³ advisory threshold, and no base yaw acceleration exceeded 1 rad/s².
- **A release transient remains:** peak measured arm jerk was 41.47 rad/s³ at
  124.60 seconds, shortly after retention release (first released sample 124.30 s).
  This is higher than the earlier can leg's 37.58 rad/s³ peak and the new non-video
  trial's 26.85. Therefore, the evidence does **not** support a claim that all arm
  jolts decreased or disappeared. Threshold compliance is not perceptual proof
  of smoothness. The local constraint/contact unloading is the next targeted
  refinement, rather than replacing the entire simulator or policy stack.
- Peak measured base yaw acceleration was 0.124 rad/s², and yaw jerk 2.685 rad/s³.
  The full recorded can sequence is 1.85x faster than the earlier sequential
  episode's can portion, with the comparison limitation above.

Exact replay reproduced every row and all nine events. All 3,982 encoded frames
decoded and matched contiguous trace ticks and the frame map. All three camera
panels changed during arm motion after excluding overlays. Grasp, release, peak
jerk, and loaded-traverse frame sheets are saved under `video/audit/`; sampled
grasp/release and final-footer frames were visually inspected. This is not a
claim that every frame was manually reviewed or that pixel changes prove
smoothness.

The dashboard source is `video/dashboard`. Real dashboard API checks confirmed
the handoff diagnosis, warning events, applied-base signals, and tool-link
signals; its MP4 is the same synchronized evidence, not a separately rendered
or time-compressed demonstration.

The initial failed pick temporarily replaced the accepted can manifest entry
under the runner's old behavior. That entry and its aggregate flag were restored
from the untouched accepted `benchmark_pack/can_dining_to_snack/result.json`.
Subsequent custom-output trials keep their manifest outside the accepted pack.
