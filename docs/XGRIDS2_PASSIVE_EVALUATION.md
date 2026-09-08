# XGrids passive evaluation and synchronized evidence

For the newer single-can execution profile, readable live outcomes, and
transition-quality diagnostics, see [Single-can diagnosis](XGRIDS2_CAN_DIAGNOSTIC.md).
The original two-object evidence below remains a historical reference.

## Scope and architecture

The first integrated scenario is `can_and_cube_dining_to_snack`. Its validated
controller, scene physics, routes, retention joints, and placement strategy are
unchanged. Other four scenarios still use their existing evaluation paths.

The predicate graph is an observational evaluation structure, not an execution
state machine; it never gates or advances policy execution.

```text
policy / planner → env.step(action) → physics
                                      │ post-control-step observation
                                      ▼
                             measured evaluation snapshot
                                      │
                    passive evaluator + timestamped events
                         ┌────────────┼──────────────┐
                         ▼            ▼              ▼
                    JSONL trace    MP4 overlay   RoboLab HDF5/log export
                                                      │
                                            existing dashboard replay
```

Evaluation observes **both objects at every control tick**, including while the
robot is returning empty for the cube. Previously placed objects remain under
observation. Oracle phase labels are useful annotations but are not evaluator
inputs. Privileged poses/contact measurements are evaluation-only data and are
not added to a policy observation space.

## Files and responsibilities

- `robolab/tasks/mobile_manipulation/xgrids2_passive_evaluation.py`: pure snapshot
  evaluation, configurable numerical thresholds, oriented-footprint geometry,
  historical/maintained predicate semantics, progress, handoffs, quality metrics.
  It imports neither Isaac nor a planner and cannot issue actions.
- `xgrids2_evaluation_observer.py`: scoped `PostStepObserver` plus the Isaac
  snapshot adapter. The wrapper calls the original environment step once and
  observes after it returns. It restores the original method on exit, including
  exceptions. The sampler reads joint names and action-term slices explicitly;
  it never assumes that the first seven mobile-robot actions are arm targets.
- `xgrids2_evaluation_export.py`: bounded-batch JSONL-to-HDF5 conversion and
  standard RoboLab run/task/log export. Does not invoke the legacy offline arm
  metric calculator, whose action ordering is unsuitable for this mobile robot.
- `examples/validate_xgrids2_lab_scenario.py`: enables the passive observer for
  the MVP scenario, writes traces, overlays the evaluator's current values on
  the three-camera video, and exports the completed episode.
- `dashboard/static/passive-evaluation.js`: playback-time capability and handoff
  panel; final summary is explicitly separated from playback-time values.
  `dashboard/app.py` carries the evaluation and predicate transitions through
  its existing events endpoint. Existing score bar, timeline, chart seeking,
  camera synchronization, and experiment browsing are reused.
- `examples/audit_xgrids2_evaluation.py`: recomputes evaluation from every saved
  observation, requires exact row/event/summary agreement, and checks the
  frame/control/evaluation clock mapping.
- `examples/audit_xgrids2_recording.py`: full MP4 decode and rendered-panel motion
  checks. The new bottom metric overlay is excluded from pixel-change scoring,
  so changing numbers cannot make a frozen camera appear valid.

## Correctness and progress

Each leg observes the following dependencies without prescribing robot motion:

```text
measured fingertip contact → contact-associated retention → geometric lift
                                                              │
                                   object over receiving surface while held
                                                              │
                                          detached, contained, settled placement
```

Contact uses the scenario's configured `either`/`bilateral` mode and force
threshold. Retention must coincide with contact within 0.25 seconds. This is an
**assisted grasp achievement**, not evidence of an unassisted stable grasp.

Lift means the configured oriented object bounding box clears the source by
5 cm. Transport means the object footprint is inside the destination footprint
while held, after a lift. It does not require the baseline's base staging pose.

Placement requires:

- Object footprint inside the support with the configured margin; rectangular
  supports use their yaw, and round supports check every projected box corner.
- Configured target-center XY and settled-root-height tolerances.
- Retention disabled.
- Linear speed at most 0.05 m/s and angular speed at most 0.3 rad/s continuously
  for one second.

The geometry uses configured conservative dimensions, not a per-frame mesh
collision query. Root settled height is a separate calibrated value, so it is
not assumed that the asset origin is exactly its geometric center.

Grasp/lift/transport are historical achievements. Placement is **maintained**:
moving a previously placed object away revokes its placement score and final
success. Restoring it and re-establishing settling produces a recovery event.
Final correctness requires both placements, each with its observed prerequisites.

Four explicit equally weighted macro groups are used: **Grasp, Lift, Transport,
Place**. Each group averages its specified outcomes across the two objects;
the four group scores are then averaged. Grasp has contact and assisted-grasp
subconditions. Adding navigation or other diagnostic predicates does not change
the denominator. An acquire/perception group is not scored because this
baseline receives privileged target information.

This MVP does not parse arbitrary language into constraints, enforce an exact
action sequence, or implement all five scenario geometries. The existing
baseline executes the requested can-then-cube sequence. A general declarative
constraint/DAG authoring layer remains a follow-up.

## Capability and handoff diagnosis

For each object, report:

- `navigation_to_manipulation`: observed source base-goal arrival → assisted grasp.
- `grasp_to_transport`: assisted grasp → object transported to receiver.
- `transport_to_place`: transported object → settled, released placement.

Each report includes eligible legs, observed successes, numerator/denominator,
rate, per-leg status and latency. A zero denominator yields `null`, not zero.
These are **within-episode, per-leg achievement handoffs**, not yet an attempt-
level probability estimate over repeated failures or a statistically established
policy success rate. A historical successful handoff remains recorded if later
placement regresses; maintained task correctness will nevertheless fail.

Navigation arrival checks measured pose, yaw and low base speed against the
configured goal region. It does not claim successful localization, an IK
certificate, or a globally optimal approach. A different approach can succeed
without satisfying this diagnostic, and is not penalized in task correctness.
Interaction-pose feasibility is explicitly uninstrumented. Goal-region reentries
before grasp are counted as a limited repositioning proxy, not semantic repair
attempts inferred from the planner.

Failure output names each leg's first unsatisfied grasp/lift/transport/place
outcome. It is **structured failure localization**, not an assertion that a
particular perception/controller/physics fault has been causally established.
Unreached downstream outcomes must be interpreted with handoff eligibility and
the event timeline. No human-intervention or perception success is invented.

## Quality metrics and validity

Recorded now: simulated elapsed time, first complete-task time (only reported
as completion time if final correctness holds), measured base XY path length,
absolute yaw travel, branch-aware arm command-tracking RMSE, control-rate
integrated squared joint jerk and peak jerk, source-goal reentries before grasp,
predicate recoveries, geometric release clearance, final XY/Z placement error,
and containment. Joint diagnostics preserve explicitly named arm coordinates.

The existing controller/physics audit is reported separately: motion-quality
violation, route-clearance gate, jerk and oscillation diagnostics. It is not
used to manufacture a successful task outcome. The runner's original `passed`
continues to describe its legacy controller gates; `passive_evaluation.task_success`
describes measured transfer correctness. Inspect both, rather than overwriting
one with the other. A staged pick-only run is naturally not a full-task success.

Important limitations:

- Whole-body collision counting is **not instrumented** and is `null`, not zero.
  Existing source/destination collision-filter warnings mean it would be unsafe
  to advertise complete collision coverage from those sensor pairs. Route
  clearance is a separate geometric diagnostic, not measured collision count.
- Fixed-joint retention invalidates unassisted grasp stability, slip resistance,
  drop resistance and load-bearing grasp/lift quality. Geometric lift/transport,
  base trajectories, timing and post-release placement remain measurable under
  this explicitly assisted configuration. Compare like retention setups.
- Release clearance makes the existing elevated cube release visible. This work
  does not refine that placement strategy or silently relabel it gentle placement.
- No reference demonstration is supplied: reference-normalized path efficiency
  is `null`. There is no mathematical optimal-path claim.
- The snapshot rate is 30 Hz control time, not the 120 Hz PhysX substep rate.
  These traces do not establish the absence of every substep contact transient.
- Time metrics use the simulated execution clock, not render time or time spent
  computing plans while physics is paused. They do not measure end-to-end policy
  compute latency; compare consistent execution/timing configurations.
- Attach a fresh evaluator after reset. This MVP runner does not auto-reset
  during its full episode. A general auto-reset policy runner must capture the
  terminal pre-reset state through its recorder lifecycle before reusing these
  predicates; a post-return `env.step` wrapper alone cannot recover erased state.
- No hardware energy, wheel traction, odometry-drift or battery claims are made.

## Recording and viewing

From the repository root:

```bash
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/validate_xgrids2_lab_scenario.py \
  --scenario can_and_cube_dining_to_snack --stage full --seed 0 \
  --record-video --video-fps 30 --capture-every 1 --headless \
  --output-dir output/xgrids2_lab/passive_evaluation/can_and_cube_seed0
```

Passive evaluation is automatically enabled for this scenario, even without
video or the legacy control trace. Recorded passive evidence currently requires
one frame per control tick at real simulated speed; incompatible FPS/decimation
options fail explicitly rather than silently misalign plots.

```text
can_and_cube_seed0/
  result.json                     legacy result plus passive evaluation
  evaluation.json                 correctness, diagnosis, quality, validity
  evaluation_trace.jsonl          every measured snapshot and evaluated values
  evaluation_events.json         edge-triggered transitions
  control_trace.jsonl             existing controller diagnostics
  video_frame_map.json            frame ↔ global tick ↔ evaluation tick
  task.mp4                       three cameras with evaluator-derived overlay
  montage.png / contact_sheet.jpg
  evaluation_audit.json           exact replay and clock check
  audit/recording_audit.json      full decode and physical-view motion check
  dashboard/passive_transfer/
    episode_results.jsonl
    XGRIDS2LabCanAndCubeTransferTask/
      data.hdf5 / env_cfg.json / log_0_env0.json
      video_0_env0.mp4             link to original video, not a second encoding
```

Launch the existing dashboard using its installed Python environment:

```bash
/home/robocup/miniconda3/envs/env_isaaclab/bin/python -m dashboard.cli \
  --host 127.0.0.1 --port 8080 \
  --output-dir output/xgrids2_lab/passive_evaluation/can_and_cube_seed0/dashboard
```

Open `http://127.0.0.1:8080`. If the dashboard already has saved sources, add the
above `dashboard` directory through its source UI; the initial CLI source only
seeds an empty registry. Select `passive_transfer`, the can-and-cube task, and its
episode. Replay updates capabilities, handoffs, progress and plot playheads;
click an event or plot to seek. Default playback is now **1×**, not 2×.

The dashboard is a post-run research viewer. This implementation does not add
network streaming from an active simulation. Values are computed online during
execution, burned into the recording from that same evaluator, then replayed
from the saved trace. Final summaries are clearly labeled as final, not live.

Post-step sample tick 1 is simulated time `dt`, displayed at video time zero.
Every frame map explicitly records this one-sample origin convention. HDF5 and
event charts use video-relative time, while the video overlay says **SIM** time.

## Validation

Pure tests do not need a simulator; bypass the repository-wide conftest which
otherwise starts Isaac:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/robocup/miniconda3/envs/env_isaaclab/bin/python -m pytest --noconftest \
  tests/test_xgrids2_passive_evaluation.py -q
```

Audit a completed recording:

```bash
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/audit_xgrids2_evaluation.py \
  output/xgrids2_lab/passive_evaluation/can_and_cube_seed0
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/audit_xgrids2_recording.py \
  output/xgrids2_lab/passive_evaluation/can_and_cube_seed0
```

The replay audit fails on changed scores, changed events, missing ticks, altered
clock mappings or disagreement between frame counts and trace length. It is
separate from whether the robot succeeded: a correctly diagnosed failed task
can have valid evidence.

## Verified seed-0 evidence (2026-09-04 local / 2026-09-05 UTC)

The completed recording is in
`output/xgrids2_lab/passive_evaluation/can_and_cube_seed0/`.

| Check / metric | Measured result |
| --- | --- |
| Legacy controller gates / completed stage | Passed / full |
| Passive correctness / progress | True / 100% |
| Released objects | Can and cube; final retention off |
| Handoffs | Each of the three categories has 2 observed successes / 2 eligible legs |
| First complete-task simulated time | 671.567 s |
| Total simulated duration / MP4 duration | 680.033 s (11 min 20 s) |
| Base path / absolute yaw travel | 10.752 m / 27.936 rad |
| Arm tracking RMSE | 0.009899 rad |
| Peak measured arm jerk | 113.348 rad/s³ |
| Jerk / oscillation gate violations | None; peak rolling reversal count 0 |
| Route-clearance gate | Passed; not a substitute for measured collision count |
| Final can XY error / Z error | 7.18 mm / approximately zero |
| Final cube XY error / Z error | 32.05 mm / 9.14 mm, within configured tolerances |
| Predicate recoveries | 0 |
| Frame / control / evaluation samples | 20,401 each, at 30 FPS |
| Exact replay | Every row, summary value and all 15 events agree |

The test run covered **41 tests**: 13 focused passive-evaluation tests plus the
existing scenario and controller regressions. The log is `regression_tests.log`.
The completed episode's dashboard endpoints were checked with the actual data:
15 events, 19 plotted series and a successful HTTP 206 MP4 range response.
Full video decoding and non-overlay panel motion checks passed. Decoded frame
sheets around release and peak jerk were also visually inspected. This is one
evidence seed, not a robustness estimate.

### Important finding: correctness is not placement quality

The can's first post-retention-release sample has approximately zero clearance
above the table. The cube's corresponding sample has **0.398 m** clearance from
its lowest configured bounding-box point to the destination plane.

This is an actual high release in this trial, not just a retention-label change:
at simulated time 669.900 s retention is off, the cube root is at Z=1.159 m and
vertical speed is -0.226 m/s. Finger forces are near zero and become zero on the
next tick. By 670.100 s its vertical speed is -2.126 m/s; it reaches the table
and subsequently settles. See `audit/cube_release.jpg` and the corresponding
event at video time 669.867 s. Both final geometry and maintained placement pass,
but that must **not** be described as gentle or high-quality placement.

The configured 10 cm release-clearance parameter is therefore not an achieved
physical clearance guarantee. This implementation changes neither the placement
planner nor physics, and does not establish the root cause of that discrepancy.
The new measurement exposes it for a separate placement refinement/diagnosis.

Another useful distinction is that `can_transfer/navigation_destination_reached`
is false while the can is successfully transported and placed. The configured
reference base-goal diagnostic does not invalidate a measured successful object
transfer. Also, the initial can source goal is already satisfied at the first
sample; its navigation handoff is not evidence of solving a new navigation route.
