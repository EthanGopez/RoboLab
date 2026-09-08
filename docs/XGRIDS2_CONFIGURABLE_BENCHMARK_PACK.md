# Configurable XGrids2 Manipulation Benchmark Pack

For the maintained entry points and module ownership, see
[XGRIDS2_CODEBASE_GUIDE.md](XGRIDS2_CODEBASE_GUIDE.md).

The can-and-cube passive evaluator, capability/handoff diagnosis, and synchronized
dashboard/video evidence are documented in
[XGRIDS2_PASSIVE_EVALUATION.md](XGRIDS2_PASSIVE_EVALUATION.md).

## Scope

This package turns the validated XGrids2 can and plate demonstrations into a
typed, configuration-driven manipulation framework and adds three benchmark
scenarios.  It intentionally keeps the vendor USDZ scene as collision-free
appearance.  Tables, receivers, targets, and clutter are separate, clean
physics objects shared by PhysX, CuRobo, route checks, and result predicates.

The first pack contains:

| Scenario | Capability | Completion condition |
| --- | --- | --- |
| `can_dining_to_snack` | cylindrical surface transfer | released, settled can inside the snack-table target |
| `plate_dining_to_sink` | large thin object and basin placement | released plate settled inside the sink |
| `can_and_cube_dining_to_snack` | sequential multi-object episode | can and cube placed at separate targets without reset |
| `lime_clutter_to_bowl` | clutter extraction and container insertion | released lime fully inside the bowl; every distractor moves at most 3 cm |
| `mustard_pour_into_bowl` | spout alignment and in-hand reorientation | spout inside opening, sustained 110 degree tilt, upright return, stable release |

The mustard task is explicitly `pose_level_pour_no_fluid`.  It scores the
geometry and timing of a pouring motion; it does not claim liquid or particle
simulation.

## Architecture

The implementation has five layers.

1. `xgrids2_lab_scenarios.py` contains frozen dataclasses and the five pure
   scenario records.  Importing it does not start Isaac Sim, which makes the
   contracts statically testable.
2. `xgrids2_lab_scenario_task.py` converts a record into an Isaac Lab scene and
   task.  `auto_env_registrations_xgrids2_scenarios.py` registers generated
   environment classes directly and caches them by scenario name.
3. `xgrids2_lab_scenario_oracle.py` provides the public
   `XGRIDS2ScenarioOracle` facade.  It selects the unchanged validated can
   oracle, the explicit plate strategy, or the common configurable transfer /
   pour backend.
4. `examples/validate_xgrids2_lab_scenario.py` runs a scenario, captures all
   three cameras from the physical control-step callback, writes evidence, and
   updates the pack manifest.
5. `xgrids2_motion_diagnostics.py` provides the simulator-independent
   trajectory resampler and per-control-tick motion diagnostics used by the
   oracle and its unit tests.


The legacy can and plate environment names remain valid. Their validation
programs are now thin compatibility launchers for the configurable runner, so
recording and result-writing logic has one maintained implementation. Calling
`register_xgrids2_scenario()` for either reference scenario still returns its
established task and environment instead of generating a subtly different
replacement.

## Configuration records

All records are immutable dataclasses.

### `XGRIDS2ObjectSpec`

Defines the USD asset, dimensions, mass, spawn region and height, orientation,
collision approximation, held-object CuRobo spheres, and whether the object is
manipulated or a distractor.  A fixed seed samples a deterministic pose from
the region.

### `XGRIDS2SupportSpec`

Defines a rectangular or round surface, bowl, or sink.  It records clean
geometry, settled surface height, receiver opening, and optional display USD.
The display mesh is never used as an implicit collision oracle.

### `XGRIDS2GraspSpec`

Defines side, top, rim, or bottle-aware grasp posture; approach direction and
clearances; lift and retract offsets; contact threshold/mode; and the maximum
loaded navigation envelope.

### `XGRIDS2PlaceSpec` and `XGRIDS2RouteSpec`

Placement records describe a surface or container target, final height,
tolerances, containment margin, bounded placement-candidate count, and an
explicit option to release from a verified approach pose.  Route records hold
source/destination base poses, approach geometry, planning/runtime radii, and
conservative velocity limits.  These values are used by execution and
validation, not copied into a second hidden source of truth.

### `XGRIDS2TransferLegSpec`

A leg binds an object and grasp to its source, route, destination, placement
strategy, and optional return-to-source action.  Phase and result names are
prefixed by the leg name, so sequential episodes remain auditable.

### `XGRIDS2PourSpec`

Defines the bottle-local spout point, receiver, 110 degree target tilt, one
simulated-second hold, opening margin, height band, tilt tolerance, and upright
return tolerance.  The oracle deterministically selects the first reachable
horizontal tilt azimuth, aligns the spout to the same receiver target, and
reverses the verified CuRobo path before placing the bottle upright.

### Cameras and metadata

`XGRIDS2CameraShotSpec` supplies stable context views anchored to manipulation
regions.  A scenario also records its instruction, capability tags, difficulty,
strategy, episode length, and limitations.  The runner embeds the complete
record in both `result.json` and the pack manifest.

## Scene generation and physics ownership

The generated scene performs the following steps:

- load the XGrids vendor USDZ only through the existing collision-free NuRec
  appearance path;
- mask vendor map/table segments that overlap manipulation fixtures;
- spawn explicit table/receiver collision geometry;
- spawn every target and distractor as an individual dynamic rigid object with
  contact reporting;
- author one disabled `oracle_grasp_joint_<object-name>` per manipulated
  object; and
- reset all objects from their typed spawn contracts.

The retention joint is an acknowledged post-contact assist.  It begins
disabled, is transformed to the measured object/flange relationship only after
the configured fingertip contact is observed, and is disabled before success.
Sequential tasks therefore attach and release each object independently.
There is no manipulation-state teleportation.  The plate keeps its previously
accepted base-retained transport and guided lowering workaround.

## Oracle execution

The generic leg follows this state machine:

```text
reset -> base alignment -> branch anchor -> pregrasp -> approach
      -> measured contact -> retention -> lift -> retract -> carry tuck
      -> route traversal -> preplace -> place -> release -> retreat -> validate
```

The physical step callback owns joint-delta, speed, acceleration, jerk, and
oscillation diagnostics; base hold; retention/contact sampling; and camera
recording. CuRobo's interpolated path retains its 50 ms sampling period. The
oracle unwraps continuous joints, uniformly time-scales each path, caps free-space
target motion at `0.010 rad/control-step`, resamples onto the 30 Hz physical
control clock, and submits exactly one new target per tick. The clock never
pauses for intermediate following error; convergence waiting occurs only at the
trajectory endpoint. Direct reset and handoff moves use a quintic minimum-jerk
blend with zero endpoint velocity and acceleration. This replaces the former
stop--start behavior that waited at every CuRobo waypoint. Loaded trajectories
retain any smaller calibrated increment (the reference can profile uses
`0.004 rad/control-step`) and the existing physical
`0.05 rad/control-step` rejection gate.

Every tick is also checked against a `1500 rad/s^3` measured-joint jerk limit,
a `900 rad/s^3` target jerk limit, and an oscillation gate of more than eight
velocity-direction reversals above `0.08 rad/s` in a rolling one-second
window. The first two derivative samples are logged but excluded from maxima
because acceleration and jerk do not yet have a complete history. CuRobo sees
support geometry, distractors, and held-object spheres. Bowl and sink openings
remain valid approach volumes rather than being represented as solid cuboids.

Sequential execution releases the can, returns the empty robot over a checked
route, and then starts the independently namespaced cube leg without resetting
the scene.  The cube uses a dedicated buffered route and releases from its
verified 10 cm approach pose with a disclosed 2.5 cm inward XY target bias;
gravity settling, low final velocity, target tolerance, detachment, and
reverse-approach retreat are all still checked.  The
lime predicate additionally measures every distractor against its reset
transform.  The pour strategy replaces traversal with spout-aligned tilt, hold,
reverse, upright placement, and release.

## Public API

```python
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import (
    get_xgrids2_scenario,
    list_xgrids2_scenarios,
)
from robolab.registrations.kinova_mobile import register_xgrids2_scenario
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenario_oracle import (
    XGRIDS2ScenarioOracle,
)

for name in list_xgrids2_scenarios():
    spec = get_xgrids2_scenario(name)
    env_cfg_class, env_name = register_xgrids2_scenario(name)
```

After constructing the environment through the normal RoboLab runtime:

```python
oracle = XGRIDS2ScenarioOracle(env, spec, num_seeds=12)
result = oracle.run(stage="full")
assert result.as_dict()["passed"]
```

## Running and recording

Run from the repository root with the Isaac Lab Python interpreter:

```bash
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/validate_xgrids2_lab_scenario.py \
  --scenario lime_clutter_to_bowl \
  --stage full \
  --seed 0 \
  --record-video \
  --video-fps 30 \
  --capture-every 1 \
  --headless
```

`--stage pick|carry|place|full` provides calibration gates.  The final pack
uses `--stage full --seed 0`. `--capture-every 1 --video-fps 30` records every
30 Hz control step and plays it at real simulated speed. The runner sets the
render interval to one control period (four 120 Hz physics substeps), so camera
rendering is not confused with the physics timestep. Per-tick JSONL telemetry
is enabled by default; use `--no-record-control-trace` only when a trace is
deliberately unnecessary. GPU trials must be run sequentially.

Each final run writes:

```text
output/xgrids2_lab/benchmark_pack/<scenario>/
  result.json
  task.mp4
  montage.png
  contact_sheet.jpg
  control_trace.jsonl
output/xgrids2_lab/benchmark_pack/benchmark_manifest.json
```

The MP4 is synchronized as a large stable third-person view beside stacked
forward navigation and wrist-camera views.  Captures happen inside the
oracle's physical step callback, including the plate's special retained-base
callback.  Video diagnostics check readability, frame count, dimensions, and
changed pixels. `control_trace.jsonl` contains `global_tick`, backend-local
`tick` and `time_s`, phase, object name, joint target/position/velocity,
raw-PhysX velocity, acceleration, jerk, target velocity/acceleration/jerk,
retention state, and rolling oscillation reversals for every control tick. The
JSON remains authoritative for phase completion, motion quality, release state,
route clearance, containment, clutter disturbance, and pour geometry.

### Seed-0 evidence pack

The original seed-0 pack completed all five geometric task contracts. The 30 Hz
diagnostic video for `can_and_cube_dining_to_snack` was the only member rerun
with the stricter motion gate before the physics repair: it completed the
geometric task but failed jerk and oscillation quality. A post-repair non-video
seed-0 rerun now passes the complete task, route, release, and motion-quality
gates. An independent post-repair 30 FPS video trial also passes; see the
video-verification section below. This table remains an index of the original
video evidence. The other four videos have not been regenerated with the
repaired base.

| Scenario | Video frames | Maximum arm step (rad) | Key task-specific result |
| --- | ---: | ---: | --- |
| `can_dining_to_snack` | 876 | 0.02136 | can target error 0.0162 m |
| `plate_dining_to_sink` | 999 | 0.01583 | released plate passed sink geometry |
| `can_and_cube_dining_to_snack` | 18,147 | 0.03943 | pre-fix can/cube errors 0.00138/0.04220 m; quality gate failed |
| `lime_clutter_to_bowl` | 881 | 0.02788 | lime error 0.0092 m; maximum clutter displacement 0.0220 m |
| `mustard_pour_into_bowl` | 490 | 0.04131 | 110.1 degree tilt held 1 s; 1.92 degree upright return error |

These measurements are also stored machine-readably in each `result.json`;
the table is a human-readable evidence index, not a second scoring source.

### Can-and-cube motion diagnostic (2026-09-02)

The final diagnostic run took 1,845.65 seconds (30 minutes 45.65 seconds) on the
single RTX 5080 Laptop GPU. It recorded 18,147 control ticks and 18,147 frames
at 30 FPS, yielding 604.9 seconds of real-time playback. The MP4 is 1440 x 540
and readable; its first and last frames differ across 77.64 percent of pixels.

The continuous-execution cleanup reduced maximum commanded jerk from more than
1,000 to `251.51 rad/s^3`, below the `900 rad/s^3` target gate. The remaining
measured peak is `1828.27 rad/s^3`, with 14 rolling direction reversals against
a limit of eight. The worst measured event occurs during cube base alignment
while commanded arm velocity, acceleration, and jerk are all exactly zero; a
second `1570.21 rad/s^3` peak occurs during the empty return traversal under the
same stationary arm target. This separates the remaining defect from CuRobo
waypoint execution: it is in mobile-base/arm drive coupling and simulator
constraint response.

PhysX also repeatedly warns that each temporary retention joint has disjoint
body transforms when it is authored or enabled. The warning and the
stationary-target oscillations make retention-frame authoring, mobile-base
acceleration, articulation drive gains/damping, and solver settings the next
setup-level investigation. Planner refinement can improve path shape and place
style later, but it cannot explain motion that occurs under an unchanged arm
target.

### Physics-native base repair (2026-09-04)

#### Root causes

The old `HolonomicBaseVelocityAction` did not control a physical base. At every
120 Hz physics substep it integrated `(vx, vy, omega)` itself and then called
`write_root_link_pose_to_sim()` and `write_root_velocity_to_sim()`. PhysX
would solve gravity, arm motion, payload forces, contacts, and the floating
articulation, and the action term would overwrite part of that solved state
immediately afterward. The arm drives and the root overwrite therefore formed
two incompatible authorities. This explains why large measured arm jerk could
occur while the arm target was perfectly stationary.

The composite robot also inherited the Kinova mounting link's inertial
properties as its root: live parsing reported only `1.697 kg` and roughly
`0.002--0.005 kg m^2` diagonal inertia. That is plausible for one mounting
link but not for a mobile chassis carrying the arm. Arm reactions could
therefore accelerate and rotate the root unrealistically.

A PhysX D6 joint was considered for ideal planar motion, but an articulation
D6 requires all linear axes locked. It cannot provide free X/Y plus locked Z
inside this articulation. The repaired benchmark therefore keeps a floating
articulation and realizes the intentionally simplified planar base with finite,
bounded forces and torques.

#### Implementation

The action API remains body-frame `(vx, vy, omega)`. Internally:

1. The command is limited in velocity, acceleration, and jerk.
2. Body-frame X/Y velocity is converted to the current world yaw.
3. Bounded world X/Y forces regulate measured planar velocity.
4. A finite Z spring-damper regulates the nominal chassis height.
5. Finite roll/pitch torques regulate the chassis local Z axis toward world up.
6. A bounded yaw torque regulates measured angular velocity.

The force and torque are submitted through
`set_external_force_and_torque()` on `base_link`. Runtime motion contains no
root-pose or root-velocity write. Direct state writes remain only in reset/test
setup, where discontinuity is intentional.

The composite USD now gives `base_link` an explicit simplified benchmark
chassis model: `65 kg`, center of mass `(0, 0, -0.28) m`, and diagonal
inertia `(4, 4, 6) kg m^2`. These are stable abstraction parameters, not a
claim that they identify the physical CoBot drivetrain. Arm, payload, gravity,
and contact forces still participate in one PhysX solve.

#### Verification

Pure tests cover coordinate conversion, jerk-limited command evolution, vector
force saturation, quaternion tilt extraction, the wrench controller, chassis
mass schema, and the absence of runtime root-state writes. Live GPU gates then
check:

- unloaded hold at both dining and snack staging poses;
- the complete collision-inflated route and final pose;
- less than 2 degrees of chassis tilt;
- stationary-arm jerk and oscillation limits while the base moves; and
- route contact/clearance validity.

The post-repair seed-0 full `can_and_cube_dining_to_snack` run is stored in
`output/xgrids2_lab/physics_fix_diagnostics/can_and_cube_no_video/`. It
completed 22,081 control ticks, both transfers, reverse navigation, both
releases, and all geometric checks:

| Metric | Pre-fix diagnostic | Post-repair | Change |
| --- | ---: | ---: | ---: |
| Maximum arm step | 0.03943 rad | 0.01503 rad | -61.9% |
| Maximum control-rate arm speed | 1.1828 rad/s | 0.4508 rad/s | -61.9% |
| Maximum raw PhysX arm speed | 2.8669 rad/s | 0.4560 rad/s | -84.1% |
| Maximum arm acceleration | 34.308 rad/s2 | 5.813 rad/s2 | -83.1% |
| Maximum measured jerk | 1828.27 rad/s3 | 242.89 rad/s3 | -86.7% |
| Rolling oscillation reversals | 14 | 2 | -85.7% |
| Motion-quality violation | true | false | passed |

The can leg has zero detected reversals and a `60.21 rad/s3` jerk peak; the
cube leg has two reversals and a `242.89 rad/s3` peak. Both are below the
existing gates. This is strong evidence that the visible dancing was caused by
the simulator base/root setup, not by CuRobo's path itself.

PhysX still warns when the pre-authored disabled retention joints have
disjointed transforms. The full task now remains smooth across retention
activation and release, so that warning is not the dominant jitter source.
It should still be removed by authoring or reauthoring exact local joint frames
at attachment time before claiming that retention is physically clean.

#### Independent video verification (2026-09-05 UTC)

The full seed-0 can-and-cube task was rerun with all three cameras captured at
every 30 Hz control step and encoded at 30 FPS. Evidence is in
`output/xgrids2_lab/physics_fix_diagnostics/can_and_cube_video_20260905/`:

- `task.mp4`: full 1440 x 540 video, 21,207 frames, 706.9 seconds (11:46.9).
- `result.json`: passed full task, valid route clearance, both objects released,
  retention disabled at the end, no motion-quality violation.
- `control_trace.jsonl`: 21,207 contiguous control ticks.
- `audit/recording_audit.json`: complete decoding, frame/tick agreement,
  per-camera pixel-change checks excluding overlays, per-phase motion metrics,
  and frame indices/timestamps for attachment, travel, release, and peak jerk.
- `audit/diagnostic_highlights.mp4`: excerpts at original playback speed;
  `audit/highlights_index.json` maps excerpts to full-video timestamps.
- `audit/*.jpg`: inspected sequences of frames around those events.

| Metric | Earlier 30 FPS video | Repaired 30 FPS video |
| --- | ---: | ---: |
| Maximum measured jerk (rad/s3) | 1828.27 | 110.41 |
| Maximum arm acceleration (rad/s2) | 34.31 | 3.18 |
| Maximum control-rate arm speed (rad/s) | 1.1828 | 0.2998 |
| Maximum raw PhysX arm speed (rad/s) | 2.8669 | 0.2931 |
| Maximum target jerk (rad/s3) | 251.51 | 251.25 |
| Maximum rolling reversals | 14 | 0 |
| Motion-quality violation | true | false |

The approximately 94% decrease in measured peak jerk occurs with essentially
the same maximum target jerk. Phase comparisons are especially useful: can
loaded travel fell from 1144.91 to 19.16 rad/s3, empty return travel from 1570.21
to 28.20, cube base alignment from 1828.27 to 31.85, and cube loaded travel from
1266.17 to 13.27. All four phases have zero detected reversals in the repaired
recording. Can and cube final XY errors are 7.9 mm and 20.4 mm.

All frames decoded, frame count matched the trace and the runner's count,
all three rendered panels changed during execution, and the 43 recorded phase
intervals agreed with the runner's video phase list. Visual inspection used
trace-aligned frame sequences around both attachments, both loaded traversals,
both releases, and the largest measured jerk event. The sampled arm posture
remains steady during travel. The largest residual jerk occurs on cube release
at 697.13 seconds, without repeated oscillation.

This validates removal of the large simulator-induced jitter for this scenario
and seed, supported by both the non-video pass and this independent recorded
pass. It is not a robustness result across all tasks or policies. The video
still shows the cube being released above the tabletop and falling to settle:
the disclosed elevated-release placement strategy remains in place. Pouring
and the other scenarios were not rerun here.

The CPU-only audit is reproducible without launching Isaac:

```bash
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/audit_xgrids2_recording.py \
  output/xgrids2_lab/physics_fix_diagnostics/can_and_cube_video_20260905
```

The audit requires one video frame per control tick. Pixel-change measurements
check for frozen views and do not themselves establish smoothness. A small
synthetic recording checked its synchronization and panel-change logic.

## Adding a scenario

1. Reuse or define immutable object, support, grasp, place, and route records.
2. Add one uniquely named `XGRIDS2ScenarioSpec` to `SCENARIOS`.
3. Use `generic_transfer` if profile-driven transfer is sufficient.  Add an
   explicit strategy only when the task has materially different mechanics,
   as the plate does.
4. Run the static contract and factory tests.
5. Calibrate `pick`, `carry`, and `place`; then run one full non-video seed-0
   trial.
6. Run one independent full seed-0 video trial and inspect both its MP4 and
   `result.json`.

A new scenario must not introduce a second geometry definition in its oracle.
If a fixture matters to physics, planning, clearance, or scoring, it belongs in
the scenario contract and must be consumed by all applicable layers.

## Validation

Static/regression coverage is in:

- `tests/test_xgrids2_lab_scenarios.py`
- `tests/test_xgrids2_lab_scenario_factory.py`
- `tests/test_xgrids2_lab_scenario_controller.py`
- `tests/test_xgrids2_lab_manipulation_schema.py`
- `tests/test_xgrids2_lab_plate_sink_schema.py`
- `tests/test_fixed_base_manipulation.py`
- `tests/test_mobile_cobot.py`
- `tests/test_xgrids2_lab_physics_sim.py`

These check unique names, assets, immutable contracts, spawn/support geometry,
clutter separation, destination capacity, generated task classes, direct
registration, retention names, legacy geometry and CLI compatibility, and the
existing controller invariants.  Runtime results add task-specific geometric
and motion gates.

## Bugs found while building the pack

- Re-importing an expanded task module under another name duplicated Isaac
  class execution and broke plate scene expansion.  Generated classes are now
  built and registered directly with a per-scenario cache.
- A single hard-coded grasp joint could not support a two-object episode.  Each
  manipulated object now owns a disabled namespaced joint.
- Generic can assumptions required bilateral contact and named every object as
  a can.  The shared backend now receives an object name, joint name, grasp
  profile, contact mode, and held geometry.
- Loaded-object drive lag could cross the physical motion gate even with a
  smooth CuRobo plan.  Generic loaded execution now commands smaller steps and
  tighter convergence while retaining the original measured quality limit.
- A fixed bottle tilt axis reached a wrist joint limit midway through the pour.
  Tilt azimuth is now selected deterministically from collision-checked CuRobo
  candidates; the scored 110 degree tilt and spout target do not change.
- Recording only `_step()` omitted the plate's special transport callback.  The
  facade now observes both physical paths.
- Treating a receiver's complete bounds as one solid CuRobo box made valid bowl
  or sink insertion impossible.  Physics still owns its walls, while the
  planner approaches through the configured opening.

- An early lift retry rewrote the target from an assumed object pose, which
  could behave like a hidden state correction.  The retry now measures the
  live TCP height and commands only the missing vertical Cartesian distance;
  object state is never teleported.
- PhysX exposes a final integration-substep velocity that can spike even when
  every commanded control waypoint is safe.  The scored arm speed is now the
  finite difference between control ticks.  Raw PhysX speed is preserved as a
  separate diagnostic and cannot replace the physical 0.05 rad/tick gate.
- Freshly replanning the empty return after the first sequential placement
  found different continuous-joint branches and produced long, brittle
  retries.  The validated placement and navigation segments are recorded and
  replayed exactly in reverse after release.
- The cube originally shared the can's immediate work area and could be bumped
  before its own leg.  Its source region is now physically separated while
  remaining comfortably inside the top-grasp reach envelope.
- A coarse waypoint tolerance let the mobile base overshoot the final snack
  approach.  The departure uses a settled acceleration ramp; traversal uses a
  planning-buffered approach point, tighter waypoint/final tolerances, and a
  slower final segment.  Planning and runtime radii remain distinct and both
  are reported.
- CuRobo can return an apparently short path on the opposite equivalent branch
  of a continuous joint.  Candidate preflight now rejects trajectories that
  approach the physical branch boundary within a conservative profile-specific
  margin (0.25 rad normally and 0.10 rad for the calibrated lime top grasp).
  For branch-sensitive placement profiles, several complete collision-checked
  chains are compared before motion; a configured placement offset is honored
  first, then the smallest continuous-joint excursion is selected.
- The lime's generic top-grasp quaternion repeatedly selected an unstable
  equivalent wrist branch.  Its profile now records the deterministic
  collision-checked seed-0 quaternion, uses the narrower safe branch margin,
  and gives the long clutter/insertion episode an explicit 1200 s budget.
- The cube's final 5.5 cm CuRobo lowering repeatedly chose a joint-3 solution
  that became ambiguous when Isaac's wrapped +/-pi observation was mapped onto
  the robot's finite +/-2pi command range.  The cube now releases physically
  from the already verified approach pose and retreats by reversing that path;
  it passes only if gravity produces a stable, detached placement in tolerance.
- One valid cube tuck ended just 0.00012 rad outside a finite joint limit due
  to solver rounding.  Rebasing now clamps only numerical overflows up to
  0.001 rad to the existing interior limit; larger violations still fail.
- The first physics-settled cube release consistently drifted outward on the
  snack table.  The configured release target is biased 2.5 cm inward in X and
  Y and candidate selection prioritizes that offset.  The 8 cm success
  tolerance was not widened, and the assist is named in scenario limitations.
- A single solid proxy for a bowl both blocked insertion and disagreed with
  PhysX.  Generated bowls now use clean floor-and-wall fixtures; the same open
  receiver dimensions drive planning and containment scoring.

- Earlier evidence captured one frame per eight control ticks and encoded at
  8 FPS. That cadence hid seven out of every eight physical poses and could
  alias joint oscillation into apparent dancing. Evidence now records all
  30 Hz control ticks and encodes at 30 FPS.
- The first CuRobo executor treated interpolated samples as independent
  waypoints and waited for tolerance at each one. The timed path is now
  continuously resampled and streamed; convergence waiting remains only at its
  endpoint.
- Starting each segment or a chassis traversal from the measured pose could
  jump backward from the previous command and manufacture target jerk. Both now
  preserve the last commanded target. An attempted following-error backpressure
  mechanism then exposed a second 0--move--0--move cadence in telemetry, so it
  was removed; a conservative continuous speed cap now keeps the path clock
  moving every control tick.
- The observer previously ran only after a successful step, omitting precisely
  the tick that tripped a gate. A `finally` hook now logs and records it too.

## Diversity and limitations

The pack is more diverse than five table transfers: it varies object topology
(cylinder, cube, thin plate, near-sphere, bottle), destination geometry
(surface, basin, bowl), episode structure (single and sequential), obstacle
density, grasp direction, navigation, insertion, and in-hand reorientation.

It is still a deterministic seed-0 baseline.  Retention is assisted after
measured contact, the cube uses a disclosed 10 cm physics-settled release, the
plate has a documented transport workaround, and pouring has no fluid.  The
next useful tasks are shelf extraction, stacking,
drawer/cabinet articulation, and long-object navigation because they add
height constraints, contact-rich stability, articulation, and
orientation-dependent clearance rather than another tabletop variant.
