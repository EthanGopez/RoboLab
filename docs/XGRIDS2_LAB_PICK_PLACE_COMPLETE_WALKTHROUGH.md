# XGRIDS2 lab pick-and-place: complete implementation walkthrough

> Historical implementation walkthrough. Use the configurable runner described
> in [XGRIDS2_CODEBASE_GUIDE.md](XGRIDS2_CODEBASE_GUIDE.md) for current runs.
> The legacy command remains as a thin compatibility launcher.

**Last updated:** 2026-08-29

**Repository:** `/home/robocup/Downloads/RoboLab`
**Task:** move a tomato-soup can from the dining table to the round snack table
with the mobile Kinova, CuRobo arm planning, physical simulation, and recorded
video evidence.

## 1. Result and scope

The working system is no longer one monolithic USD scene or one monolithic
controller. It is a stack of explicit contracts:

```text
vendor USDZ / NuRec                  photorealistic appearance
        +
clean PhysX floor, map, and tables  deterministic contact geometry
        +
mobile Kinova articulation          driven robot, fixed camera mounts
        +
CuRobo                               fixed-base collision-aware arm paths
        +
closed-loop route follower          mobile-base transport
        +
phase oracle                         grasp, carry, place, release, validation
        +
video and robustness runners         evidence and repeatability gates
```

The validated deterministic run performs the complete sequence:

```text
reset -> pregrasp -> approach -> bilateral contact -> retention
      -> lift -> retract -> compact carry tuck
      -> collision-inflated A* traversal
      -> three-part place reorientation -> preplace -> place
      -> physical release -> reverse-path retreat -> settle -> validate
```

The best evidence video currently is:

```text
output/xgrids2_lab/curobo_full_video_evidence_forward_mount/pick_place_task.mp4
```

This is an **assisted planning oracle**, not a claim of an entirely unassisted
friction grasp. The fingers must first establish bilateral contact; a temporary
fixed joint then retains the can during transport. The joint is removed for the
physical release and settle.

## 2. Why the supplied USDZ is sufficient

The production visual source is the post-meeting vendor package:

```text
assets/scenes/xgrids2_lab/new_data_0827/isaacsim-import-package/
  ut_lab_1_scene_pack/0/ut_lab_1.usdz
```

That USDZ is the vendor-authored package entry point. It already resolves the
NuRec payload, its proxy relationship, and the renderer metadata needed to
display the reconstruction. The separate 68 GB raw capture would be needed to
retrain, regenerate, or deeply inspect the reconstruction, but it is not needed
to load the finished scene in Isaac Sim.

Before receiving this package, the integration used a locally assembled
`forward_compatible` wrapper around `mesh-model.usdz`. The wrapper referenced
the same kind of neural payload but did not reproduce the complete vendor
runtime contract. This was the root of the visible flashing: camera frames
alternated between the NuRec composite, its gray proxy, and incomplete
composites. The underlying scan was not rapidly changing; the render path was.

The fix was to load the vendor USDZ directly and apply the complete NuRec
configuration in `robolab/core/utils/nurec_rendering.py`:

- use one NVIDIA GPU;
- use Fabric scene delegation;
- enable registered compositing globally and on every render product;
- enable inverse tone mapping and inverse color correction;
- use synchronous rendering;
- keep navigation, wrist, and audit sensors RGB-only;
- preserve the audited `+1.075 m` vertical alignment.

The important architectural change was to stop asking the photorealistic scan
to also be the physics model. The USDZ supplies appearance. Independently
authored, simple geometry supplies collision and support.

## 3. Visual layer versus physics layer

### 3.1 Visual layer

`robolab/tasks/mobile_manipulation/xgrids2_lab_geometry.py` points
`XGRIDS2_VISUAL_USD` at the vendor `ut_lab_1.usdz`. The scene configuration
loads it under `xgrids2_lab_visual`, applies the audited Z offset, and disables
collision on the entire visual subtree.

Disabling visual-subtree collision matters because reconstruction/proxy meshes
contain noisy or incidental geometry. Letting PhysX collide against that mesh
caused unstable floor contact and phantom obstacles.

### 3.2 Static navigation collision

The LabApt vector map was rigidly registered into XGRIDS coordinates:

```text
p_xgrids = R(35.617718 deg) p_labapt + (-13.259476, 6.588388) m
```

`xgrids2_lab_geometry.py` converts the retained map segments into thin static
cuboids and exposes the same geometry to both A* and runtime clearance checks.
The visual scan therefore cannot disagree silently with the planner's actual
collision contract.

One registered segment (`map_175`) crosses a manipulation approach at its full
1.5 m height even though the physical structure is low. The manipulation scene
masks that segment down to approximately 0.70 m instead of allowing an
invisible chest-height wall to block the arm.

### 3.3 Floor and tabletop support

The clean floor is a static slab whose top is exactly `z=0`. The reconstructed
tables are visual only. Two simple collision fixtures in
`xgrids2_lab_manipulation_geometry.py` provide stable support:

- dining tabletop center `(0.84, -0.72, 0.843)`, size
  `(1.40, 0.85, 0.06)`, yaw `2.29876` rad;
- snack tabletop center `(5.60, -1.80, 0.682)`, size
  `(1.05, 1.05, 0.06)`.

The snack table is visibly round, so final placement is checked against a
circle, including a full can-radius margin. Merely landing inside the bounding
square is not accepted.

## 4. Robot embodiment and camera mount

`robolab/robots/kinova_mobile_cobot.py` defines the combined mobile base,
Kinova Gen3 arm, Robotiq gripper, and base-parented navigation camera.

The early “flying cobot” failure came from treating the base like a fixed-base
robot while also trying to drive it in the plane. The imported root joint and
articulation-root placement over-constrained the assembly. The mobile version
keeps `base_link` as the articulation root, disables the inappropriate fixed
root joint, and controls planar motion through the base action term. Gravity,
floor contact, and the wheel/base dynamics then determine Z normally.

The fixed camera mount initially appeared backward. The camera optical axis
and the route follower were using opposite forward conventions. The final
contract is:

- the mount faces robot `+X`;
- its quaternion uses the `+X` convention;
- `WaypointFollower.sensor_forward_yaw_offset = 0`.

The audit camera is independent of the robot and uses fixed world poses. It is
only for third-person evidence, never a policy observation.

## 5. Task geometry and reset

`xgrids2_lab_manipulation_geometry.py` is the single source of truth for:

- tabletop shapes and support heights;
- the dining-table pick region;
- the robot-facing snack-table place region;
- manipulation staging poses;
- can spawn and expected settled heights;
- “entire can is on table” predicates.

The pick region is deliberately small and reachable:

```text
x = [1.23, 1.27] m
y = [-1.20, -1.17] m
```

The place region stays in the robot-facing quadrant of the round table:

```text
x = [5.20, 5.24] m
y = [-1.60, -1.52] m
```

This avoids the Kinova's near-singular maximum reach while retaining a complete
can-radius margin from the table edge.

The robustness runner resets the environment, seeds Torch, writes the sampled
can pose, and optionally perturbs the floating base root. It does **not** write
joint state after reset. Directly teleporting arm joints while position drives
were active created a drive/state discontinuity and produced a fake robustness
failure. Arm perturbations are now commanded smoothly, then normalized to the
audited planning posture.

Small base perturbations are also recovered physically. The oracle uses a
closed-loop alignment motion back to the dining staging pose before it creates
the fixed-base CuRobo frame. That ensures the planner's base transform matches
the robot that executes the path.

## 6. CuRobo planner

`robolab/tasks/mobile_manipulation/curobo_fixed_base_planner.py` is the adapter
between RoboLab task geometry and CuRobo `MotionGen`.

It performs four jobs:

1. loads the seven-joint planner-only robot definition and conservative
   collision spheres from `assets/robots/kinova_mobile_cobot/curobo/`;
2. converts world targets into the current base frame;
3. converts the clean tabletop fixtures into CuRobo collision cuboids;
4. plans either end-effector pose targets or a compact joint-space target.

The mobile base remains fixed during each arm segment. This is intentional:
CuRobo plans the arm in one consistent base frame, while a separate controller
handles mobile traversal after the arm is tucked.

“Fixed” here means a low-speed closed-loop planar pose hold, not a USD fixed
joint and not state teleportation. The hold rejects base drift from arm and
payload reactions so the physical base frame stays equal to CuRobo's planning
frame. During transport, the route follower replaces the hold command.

After grasp, the held can is represented in planning through the measured
can-in-tool transform. Place tool targets are solved from the desired can pose,
not by pretending the tool center itself is the can center.

## 7. Oracle script, part by part

The main state machine is
`robolab/tasks/mobile_manipulation/xgrids2_lab_curobo_oracle.py`.

### 7.1 Result and instrumentation

`XGRIDS2CuroboResult` records:

- requested and completed stage;
- phase history;
- planner time and waypoint count for every CuRobo segment;
- measured seven-joint endpoint for every completed arm segment;
- measured pick and final can position;
- maximum physical arm-joint step and speed;
- final base position/yaw error;
- carry-posture radius and route-clearance validity;
- requested place target and final target error;
- failure reason with arm, base, TCP, and maximum-step diagnostics.

This instrumentation was essential: a video can look plausible while hiding a
joint discontinuity, route violation, or off-table placement.

### 7.2 Action construction and physical stepping

`_action()` assembles the base, seven arm joints, and two gripper commands into
the environment's 12-dimensional action tensor. `_step()` calls `env.step()`
and measures the actual circular joint displacement between consecutive
physical states.

The oracle never teleports arm state during task execution. CuRobo waypoints
are targets for damped position drives.

Every physical tick also enforces the benchmark's `0.05 rad` step and
`3.50 rad/s` instantaneous-speed limits. Exceeding either stops the episode at
the first unsafe tick instead of allowing an unstable branch to keep dancing.

### 7.3 Continuous-joint branch handling

Several Kinova joints are continuous. Numerically, `+pi` and `-pi` are nearly
the same physical angle, but naïve subtraction sees an approximately `2*pi`
jump. That was a major source of the earlier violent “dancing.”

Only joints 1, 3, 5, and 7 use circular topology. Joints 2, 4, and 6 retain
ordinary bounded differences. CuRobo normally receives the canonical circular
representation that produced the validated IK family. Inside a `0.40 rad`
guard band around a `+/-2*pi` hard limit, it receives the measured unwrapped
branch so planning moves away from the boundary. The executor selects the
nearest limit-valid equivalent target; a boundary crossing is rejected before
reaching the drive. Tracking and metrics wrap only the four continuous joints.

Default target increments are at most `0.020 rad`; loaded lift, retract, tuck,
and placement segments use `0.004 rad`. The slower loaded command prevents the
held can and compliant arm drives from exciting one another.

### 7.4 Reset normalization and pick-base alignment

`reset_robot_to_default()` smoothly restores the audited arm configuration.
For robustness trials the oracle first visits the commanded perturbed arm pose,
then normalizes it. Before Cartesian pick planning, it collision-plans to the
validated `PICK_BRANCH_ANCHOR_JOINTS` posture. This gives CuRobo a stable local
inverse-kinematics branch instead of allowing a millimetre-scale target change
to select a distant continuous-joint representation. `_align_pick_base()`
corrects small root perturbations with
the same closed-loop base follower used for navigation.

### 7.5 Pregrasp and approach

The can pose is read from Isaac Sim after settling. The oracle creates CuRobo
with the measured base pose, plans a side pregrasp 12 cm from the can, then
plans the short approach with hand contact allowed. Tabletop collision remains active.

Allowed contact disables only the conservative `tool_frame` and
`robotiq_85_base_link` hand envelopes for approach/lift/place planning. Wrist
and arm spheres, table collision, and all physical contacts remain active.

The exact deterministic control can position is
`(1.2459619045, -1.1793328524)`. It is retained as trial zero because the
geometric midpoint can select a different CuRobo branch despite differing by
only a few millimetres.

A CuRobo success flag is not accepted on its own. After executing the approach,
the oracle measures the physical TCP-to-can distance and requires it to be at
most `0.035 m` before closing. One failed experiment tried to correct a missed
endpoint with another Cartesian command; that pushed an ungrasped can and was
removed. A miss is now rejected safely instead of being converted into contact.

### 7.6 Bilateral contact and retention

`_close_and_attach()` closes the gripper and requires bilateral fingertip
contact for consecutive simulation frames. Only after that physical predicate
passes does `_attach()` enable the temporary fixed joint at coincident frames.

This prevents the assist from turning a miss into a successful grasp. The
fixed joint represents retention after contact because the supplied Robotiq
mimic-drive setup is not stable enough for a long, fully frictional mobile
carry.

At the instant retention is enabled, the oracle freezes the two driven
knuckles at their measured bilateral-contact preload. Earlier code continued
driving toward the full `+/-0.8 rad` close command. That squeezed an object
already fixed to the flange, sent competing contact impulses through the arm,
and made loaded tracking vary from run to run.

### 7.7 Lift, retract, and carry tuck

The loaded arm executes a vertical lift, a retract away from the dining table,
and a CuRobo joint-space plan to `CARRY_TUCK_TCP_BASE = (0.32, 0.03, 0.68)`.
The tuck is accepted only when measured link centers remain within
`0.350 m` of the base center. This remains 5 cm inside the independently
checked `0.40 m` carried-route footprint.

Loaded endpoint tolerances are looser than free-space tolerances because they
measure the physical drive response under payload, not CuRobo's numerical
endpoint. The motion-quality gate still rejects large physical steps or high
speed.

### 7.8 Mobile carry

`_traverse()` plans from the **measured current base position**, through
`(4.46, -1.14)`, to the snack manipulation staging pose. A* inflates map
geometry by `0.42 m`; every executed state is checked with a `0.40 m` carried
footprint.

The base moves at no more than `0.06 m/s`. The waypoint follower transforms
world-frame route velocity into body-frame `[vx, vy, wz]`, respects the `+X`
camera convention, and closes the loop on measured pose. Slowing transport and
tightening waypoint acceptance stopped the base from cutting corners while
carrying the extended arm.

### 7.9 Place reorientation

The required tool rotation is about 177 degrees. Sending that as one Cartesian
request caused branch changes and abrupt arm motion. The oracle uses quaternion
slerp to split it into three separately planned reorientation segments:

```text
place_reorient_1 -> place_reorient_2 -> place_reorient_3
```

Each segment starts from the measured endpoint of the preceding physical
motion.

### 7.10 Preplace and place

The requested place XY is validated against both `PLACE_REGION` and the actual
round tabletop with a can-radius margin. The oracle derives the corresponding
tool pose from the measured held transform, approaches above/outside the table,
then executes the final placement path with hand contact allowed.

The loaded preplace transition uses `0.004 rad` target increments, a longer
tracking window, a `0.25 rad` intermediate following-error envelope, and higher
wrist damping. The final place keeps its `0.050 rad` endpoint tolerance but
permits 160 convergence ticks. The measured step, speed, and placement gates remain unchanged. This lets a payload-lagged
drive accumulate enough restoring effort instead of falsely declaring a stall.

This fixed the earlier off-table result, which used a visually estimated point
and did not require the whole can footprint to lie on the round surface.

### 7.11 Release, retreat, and settle

Replanning immediately from the contact pose was unreliable: conservative
collision spheres considered the valid contact state already in collision.
The final constrained target keeps the can `0.040 m` above its support height.
The lower `0.015 m` target was at the edge of the arm workspace and could return
`IK_FAIL` after a valid preplace. A short gravity descent is both reachable and
physically checked. The solution then removes the retention constraint, opens
the fingers, and retreats by reversing the already validated place trajectory.

The can must then settle dynamically at the snack-table support height, remain
upright and slow, stay completely on the round table, and finish within
`0.08 m` of its requested XY target.

## 8. Why the original arm motion danced

The kitchen logic looked similar at a high level, but the XGRIDS task exposed
different execution conditions:

- the base is mobile and its measured frame can drift from a fixed planning
  frame;
- the arm carries a constrained payload for much longer;
- the route requires a large tool reorientation before placement;
- continuous joints make equivalent `+pi/-pi` solutions look discontinuous;
- direct state writes and exact waypoint replay fight compliant physical
  drives;
- a replanner starting at tabletop contact sees a different collision problem
  from one starting in free space.

The earlier code mixed planning state, commanded state, and physical state.
When a waypoint landed on another continuous-joint branch, the controller tried
to traverse almost a full revolution. When the payload lagged a target, the
next exact target arrived anyway. That generated the visible oscillation.

The working code separates these concepts:

```text
CuRobo path
   -> nearest equivalent circular branch
   -> small target increments
   -> damped joint drives
   -> measured physical state
   -> tolerance check before advancing
```

Loaded motion is slower; the base is aligned before planning; the large place
rotation is segmented; and retreat reuses a known-valid trajectory. Those are
the concrete reasons the robot now moves smoothly rather than dancing.

## 9. Video runner

`examples/validate_xgrids2_lab_manipulation.py` builds the environment, creates
the CuRobo oracle, and optionally records a composed evidence video.

Its main parts are:

- `_rgb()` converts Isaac camera tensors to NumPy RGB frames;
- `_panel()` labels and resizes views;
- `_set_audit_shot()` changes among a few fixed third-person compositions only
  at task-section boundaries;
- a step callback captures frames as the oracle advances physics;
- the third-person view is the main panel, with fixed-base and wrist views as
  diagnostic insets;
- the result JSON includes render settings, phase metrics, camera roles, and
  output paths.

The first bad video recorded frames outside the oracle's actual control loop,
so the base camera appeared static and only the manually moved audit camera
changed. The working runner captures through the oracle's physical step
callback. The audit shot changes only when the task changes location, which
keeps the third-person camera stable within each phase.

Run the deterministic evidence task with:

```bash
TERM=xterm \
PATH=/home/robocup/miniconda3/envs/env_isaaclab/bin:/usr/bin:/bin \
CUDA_HOME=/home/robocup/miniconda3/envs/env_isaaclab \
TORCH_CUDA_ARCH_LIST=12.0 ACCEPT_EULA=Y \
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/validate_xgrids2_lab_manipulation.py \
  --stage full --record-video --headless \
  --video-fps 10 --capture-every 8 \
  --output-dir output/xgrids2_lab/curobo_full_video_evidence_forward_mount
```

## 10. Robustness gate

The deterministic demonstration proves one complete trajectory. It does not
prove repeatability. The next step therefore adds a seeded 20-trial gate rather
than immediately adding perception or learning.

### 10.1 Trial generator

`robolab/tasks/mobile_manipulation/xgrids2_lab_robustness.py` defines
`XGRIDS2RobustnessTrial` and `robustness_trial_schedule()`.

Trial zero is the exact proven deterministic control. Trials 1–19 sample:

- can XY over the complete pick region;
- can yaw over `[-pi, pi]`;
- base XY by up to `±0.025 m`;
- base yaw by up to `±2 deg`;
- every arm joint by up to `±0.025 rad`;
- place XY over the valid place region.

The schedule uses a fixed master seed (`20260829` by default), and each record
stores its own seed and full sampled state. Any failure can therefore be rerun
by trial index.

### 10.2 Quality gates

A trial passes only if all of these are true:

- the oracle completes the full task;
- the bilateral-contact phase completed;
- every carried base pose remained in inflated free space;
- the complete can footprint ends on the round snack table;
- place-target error is at most `0.08 m`;
- maximum measured arm step is at most `0.05 rad`;
- maximum instantaneous PhysX arm speed is at most `3.50 rad/s`; the stricter
  `0.05 rad` control-tick displacement gate rejects visible discontinuities;
- final base position error is at most `0.055 m`;
- final base yaw error is at most `0.06 rad`.

The complete benchmark requires at least 90% success. Failures are categorized
as grasp, planning, arm tracking, transport, placement, motion quality, or
other.

### 10.3 Development tests versus the release gate

The oracle is scripted only at the task-sequencing level. It does not replay a
recorded joint trajectory and it does not use a placeholder IK controller:
every arm segment calls `CuroboFixedBasePlanner`, then executes the CuRobo path
through physical damped drives. The base route and the contact/release checks
remain explicit task logic because CuRobo is an arm motion planner, not a
complete mobile-manipulation state machine.

Use the test layers according to the question being asked:

- focused pytest checks code and geometry contracts in seconds;
- `--stage pick` checks one live CuRobo grasp before changing carry logic;
- one deterministic `--stage full` run checks end-to-end behavior and video;
- selected robustness indices diagnose a known seed;
- the complete 20-trial run is a milestone/release gate, not the inner
  development loop.

At the current deliberately conservative physical speeds, a full trial takes
roughly nine minutes. Parallel Isaac processes are not appropriate on the
available 16 GB GPU, and removing tracking, contact, or settle waits would
invalidate the exact dynamics that previously caused dancing and off-table
placement.

### 10.4 Runner and checkpointing

`examples/validate_xgrids2_lab_curobo_robustness.py`:

1. launches one Isaac environment;
2. creates the seeded schedule;
3. resets and applies each physical perturbation;
4. runs the full oracle;
5. applies benchmark-level quality checks;
6. atomically rewrites the JSON report after every trial;
7. frees oracle/CUDA cache before the next trial.

The atomic incremental report means a long GPU run retains every completed
record if Isaac Sim later exits unexpectedly.

If a run is interrupted, `--resume-from N` validates the existing report's
seed and full schedule, retains the contiguous records before `N`, and reruns
`N` onward. After every result, the runner disables the retention joint and
advances eight physics ticks before resetting. This matters because a USD
joint-disable edit reaches PhysX on a simulation step; resetting immediately
after a post-grasp failure can otherwise contaminate the next trial.

The robustness-only episode budget is 300 simulated seconds instead of the
task default of 180 seconds. The extra minute covers smooth perturbation
normalization and base alignment; it does not change any grasp, collision,
placement, or motion-quality predicate.

The headless gate intentionally keeps navigation, wrist, and audit RGB render
products alive at only `64x36`. During validation, removing all cameras—or
creating cameras without registering all three NuRec render products—changed
Fabric articulation-update behavior and made the previously stable arm jump.
This appears to be a runtime coupling in the current Isaac Sim 5.1/Fabric
stack, not task logic. The tiny products preserve the validated execution path
at a small fraction of video rendering cost.

Run the complete gate with:

```bash
TERM=xterm \
PATH=/home/robocup/miniconda3/envs/env_isaaclab/bin:/usr/bin:/bin \
CUDA_HOME=/home/robocup/miniconda3/envs/env_isaaclab \
TORCH_CUDA_ARCH_LIST=12.0 ACCEPT_EULA=Y \
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/validate_xgrids2_lab_curobo_robustness.py \
  --trials 20 --headless \
  --output output/xgrids2_lab/robustness/gate_20_final.json
```

Rerun selected schedule entries with:

```bash
# Generates the same 20-trial schedule but executes only trials 1 and 7.
.../env_isaaclab/bin/python \
  examples/validate_xgrids2_lab_curobo_robustness.py \
  --trials 20 --trial-indices 1,7 --headless \
  --output output/xgrids2_lab/robustness/trials_1_7.json
```

### 10.5 Final measured result

The authoritative report is
`output/xgrids2_lab/robustness/gate_20_final.json`. It completed all 20 seeded
trials and passed the requested gate with 18 successes, 2 failures, and an
exact 90% success rate.

Across the 18 passing trials, the worst measured arm step was `0.04053 rad`,
the worst instantaneous arm speed was `3.11055 rad/s`, the largest place error
was `0.02991 m`, the largest final base-position error was `0.0000128 m`, and
the largest final base-yaw error was `0.000239 rad`.

The two failures are retained rather than hidden:

- trial index 8 stopped during place when one physical step reached
  `0.05158 rad`, just above the `0.05 rad` limit;
- trial index 11 stopped before closing because the measured TCP remained
  `0.1512 m` from the can despite a nominal CuRobo approach result.

Both are safe rejections: no failed reach was converted into an attachment and
no motion-quality threshold was relaxed. An intermediate report with 17/20 is
preserved as `gate_20_pre_place_wait_fix.json`. Its third failure was a final
place endpoint that reached only `0.0573 rad` error within 100 convergence
ticks. The final code keeps the same `0.050 rad` endpoint tolerance but allows
160 ticks for the loaded final place. Trials 18 and 19 were then rerun from the
atomic checkpoint; both passed. Earlier passing trials are behaviorally
unchanged because execution returns immediately when the tolerance is met; only
the previously timed-out final place consumed the added ticks.

## 11. Bugs encountered and fixes

| Symptom | Root cause | Fix |
|---|---|---|
| Lab flashed between correct, gray, and partial frames | Locally assembled wrapper did not reproduce the vendor NuRec render-product contract | Load the vendor USDZ directly and configure Fabric, compositing, inverse mappings, sync rendering, and every RGB render product |
| Raw data seemed necessary | Finished package and training inputs were being conflated | Treat USDZ as the runtime product; retain raw capture only for reconstruction work |
| Cobot flew or was over-constrained | Fixed-root assumptions conflicted with a planar mobile articulation | Make `base_link` the mobile articulation root and remove the inappropriate fixed root joint |
| Robot did not move in the first video | Recorder sampled outside the oracle's physical stepping loop | Capture from the oracle step callback |
| Third-person camera shook | Audit pose was updated continuously | Use fixed audit shots and change only at section boundaries |
| Arm danced violently | Continuous-joint branch jumps, exact/fast target replay, and commanded/physical state mismatch | Rebase circular joints, rate-limit targets, use damped drives, and wait for physical tracking |
| Final place snapped by approximately pi in one tick | All seven joints were treated as circular; bounded joints received invalid equivalents, while one continuous target passed `-2*pi` | Wrap only joints 1/3/5/7, preserve unwrapped state only near hard limits, enforce every physical limit, and fail fast on the benchmark motion thresholds |
| Carry became unstable after grasp | Loaded motion was too fast and the posture too extended | Use `0.004 rad` loaded increments, measured tracking tolerances, and a compact CuRobo tuck |
| Loaded tracking varied between identical runs | Fingers kept squeezing toward full close after the can was retained by a fixed joint | Freeze the gripper at the measured bilateral-contact preload when retention activates |
| Failures cascaded across a multi-trial run | Reset occurred before PhysX consumed a post-failure retention-joint disable | Process eight cleanup physics ticks after every result and support validated checkpoint resume |
| Base cut route corners | Speed/tolerance were too aggressive for the carried footprint | Slow to `0.06 m/s`, tighten waypoint following, and validate every measured pose |
| Place was not on the table | Target was visually estimated and rectangular bounds ignored the round top | Author the round table predicate, use a robot-facing place region, and require full can-radius support |
| Large place rotation triggered a different branch | Approximately 177-degree tool reorientation was planned as one segment | Split it into three quaternion-slerp segments |
| A fixed place-branch anchor regressed the golden run | One joint posture was not a universally valid seed after mobile carry and payload-dependent settling | Remove the fixed place anchor; retain measured-state replanning plus three bounded quaternion-slerp segments |
| Retreat planning failed at contact | Conservative collision geometry considered the valid contact start colliding | Reverse the already validated place trajectory after release |
| Final place plan intermittently returned `IK_FAIL` | The `0.015 m`-high constrained pose was at the low workspace boundary | Plan to `0.040 m` above support, release physically, and require the can to settle at the exact tabletop height |
| A valid final place stopped at `0.0573 rad` tracking error | The loaded physical drive needed longer than the generic 100-tick endpoint window | Keep the `0.050 rad` tolerance and all motion gates, but give final place 160 convergence ticks; checkpointed rerun passed |
| Raising CuRobo to 32 seeds caused a later CUDA OOM | Persistent planners and CUDA graphs exceeded the 16 GB GPU budget | Keep 8 seeds and fix the task geometry instead of masking it with brute-force sampling |
| Fixed camera mount faced backward | Camera optical-axis and follower yaw conventions differed by pi | Face the mount along `+X` and use zero sensor-forward offset |
| Randomized reset made the arm jump | Joint state was teleported while drives were live | Apply arm perturbations through commands, then normalize smoothly |
| Perturbed base changed arm plans unpredictably | Fixed-base planner frame was created from a nominal, not measured, pose | Physically align the base, then instantiate CuRobo from measured state |
| Base drifted after reaching the snack table | Arm and payload reactions moved the dynamic base while place plans still used the nominal frame | Hold the dining/snack planar pose closed-loop during every arm phase |
| Headless no-camera run became unstable | Removing NuRec render products changed the Fabric update path | Retain three registered `64x36` RGB products in the robustness runner |
| Geometric pick center failed while control passed | A few millimetres selected a different CuRobo solution branch | Keep the exact proven pose as golden trial zero and test the rest as explicit perturbations |
| A randomized pick produced a continuous-joint branch jump | Cartesian IK was seeded from a posture with multiple distant equivalent branches | Collision-plan to the validated pick branch anchor before pregrasp planning |
| A smooth CuRobo approach stopped 15 cm from the can | Planner success described its numerical solve, not the measured physical TCP endpoint | Require a 3.5 cm physical task-space reach check and reject the miss; remove the unsafe Cartesian correction that pushed the can |
| A valid compact tuck missed the old envelope by 2.8 mm | The 0.340 m tuck limit had more margin than the separately checked 0.400 m carried footprint required | Validate up to 0.350 m while preserving 5 cm of route-footprint margin and all per-step collision checks |
| Broad local sampling reached unstable IK edges | The task used support-valid bounds that were wider than the demonstrated single grasp family | Use explicit reach-conditioned pick and place regions; broad coverage remains a future multi-grasp benchmark |

## 12. File-by-file guide

| File | Responsibility |
|---|---|
| `assets/scenes/xgrids2_lab/new_data_0827/.../ut_lab_1.usdz` | Vendor NuRec scene package and runtime visual source |
| `robolab/core/utils/nurec_rendering.py` | Global and per-render-product NuRec/Fabric settings |
| `robolab/tasks/mobile_manipulation/xgrids2_lab_geometry.py` | Visual asset path, registration, vector-map collision, staging, A* geometry |
| `robolab/tasks/mobile_manipulation/xgrids2_lab_navigation_task.py` | Visual scene, clean floor/map proxies, navigation cameras and task scene |
| `robolab/tasks/mobile_manipulation/xgrids2_lab_navigation_oracle.py` | Closed-loop holonomic route follower |
| `robolab/tasks/mobile_manipulation/xgrids2_lab_manipulation_geometry.py` | Tables, pick/place regions, support heights, containment predicates |
| `robolab/tasks/mobile_manipulation/xgrids2_lab_can_transfer_task.py` | Manipulation scene, tabletop colliders, can, contacts, actuators, actions |
| `robolab/robots/kinova_mobile_cobot.py` | Combined mobile Kinova embodiment and fixed camera mount |
| `robolab/tasks/mobile_manipulation/curobo_fixed_base_planner.py` | CuRobo adapter, transforms, collision world, pose/joint planning |
| `robolab/tasks/mobile_manipulation/xgrids2_lab_curobo_oracle.py` | Complete phase state machine and physical execution |
| `examples/validate_xgrids2_lab_manipulation.py` | Deterministic validation, metrics, video capture and evidence composition |
| `robolab/tasks/mobile_manipulation/xgrids2_lab_robustness.py` | Seeded trial schedule, quality checks, aggregation and failure taxonomy |
| `examples/validate_xgrids2_lab_curobo_robustness.py` | Long-running checkpointed randomized gate |
| `tests/test_xgrids2_lab_robustness.py` | Schedule, bounds, threshold and taxonomy unit tests |
| `tests/test_xgrids2_lab_manipulation_schema.py` | Manipulation geometry/config contract tests |
| `tests/test_xgrids2_lab_physics_schema.py` | Floor, route, staging and collision contract tests |
| `tests/test_mobile_cobot.py` | Mobile articulation, mount and action-contract tests |

## 13. Tests

Run the focused non-GUI suite with:

```bash
/home/robocup/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/test_xgrids2_lab_robustness.py \
  tests/test_xgrids2_lab_manipulation_schema.py \
  tests/test_xgrids2_lab_physics_schema.py \
  tests/test_mobile_cobot.py
```

These tests cover deterministic schedule generation, perturbation bounds,
quality thresholds, failure taxonomy, table containment, scene and actuator
schema, map/staging contracts, and the mobile camera convention. They are
necessary but not substitutes for the live Isaac/CuRobo gate.

## 14. Evidence and interpretation

Use these artifacts for different questions:

- `.../pick_place_task.mp4`: did the complete task visibly execute?
- the deterministic result JSON: which phases ran, how smooth was the measured
  motion, and where did the can/base finish?
- `robustness/gate_20_final.json`: does it repeat under bounded perturbations?
- focused pytest results: did a code change violate a static contract?

A passing video alone is not a robustness claim. A passing static test alone is
not a physical-execution claim. The system is considered ready for the next
research step only when the deterministic evidence, static tests, and live
repeatability gate agree.

## 15. Known limitations and next research boundary

- Retention after bilateral contact still uses a temporary fixed joint.
- Ground-truth simulation poses are used; perception is not yet in the loop.
- CuRobo plans fixed-base arm phases, not coupled base-and-arm trajectories.
- The current Isaac/Fabric build requires registered tiny render products even
  for reliable headless benchmarking.
- The 68 GB raw capture is not present here and is not needed for runtime, but
  would be needed for reconstruction changes.
- Perturbations are intentionally local. A broad workspace-wide grasp benchmark
  will require grasp-family selection rather than one approach family.

The 20-trial gate now meets its 90% threshold. The next best step is to replace
the temporary retention joint with calibrated Robotiq force/contact behavior.
That improves physical fidelity without changing the now-audited perception,
planning, route, placement, or evidence infrastructure.
