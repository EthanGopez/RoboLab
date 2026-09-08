# Milestone 2 kitchen oracle: complete implementation walkthrough

## 1. What was built, and what was not

Milestone 2 adds a deterministic, vectorized mobile-manipulation controller on
top of the physically valid Milestone-1 kitchen. It can:

1. wait for a randomized dynamic can to settle on the island;
2. approach it with the Kinova/Robotiq tool-center point (TCP);
3. require independent left and right fingertip contact;
4. lift it;
5. drive the mobile base across the aisle while holding an arm pose;
6. reconfigure the arm on the counter side;
7. place and release the can;
8. require stable dynamic support and target accuracy;
9. retreat and report success only if all final predicates remain true.

The historical assisted oracle passed 100/100 randomized trials. The newer
strict post-contact qualification path passed its exact 10-trial seed-101
reproduction. “Assisted” has a precise
meaning: after bilateral contact has been physically detected and held, a
pre-authored fixed joint preserves the flange-to-can transform during
transport. The contact acquisition, can drop, counter contact, gravity,
release, support, and collision checks are simulated. An unassisted
friction-only grasp is implemented as a mode, but it has not passed the
100-trial gate.

That distinction matters. This milestone proves the task geometry, action
interface, state machine, navigation, placement, predicates, and evaluation
harness. It does not yet prove that the current Robotiq collision and actuator
model can retain the 0.349 kg can for the whole trajectory without assistance.

## 2. End-to-end data flow

```text
measured geometry constants
        |
        v
visual kitchen + simple static collision overlay + dynamic can
        |
        v
registered RoboLab environment
  observation: arm/gripper/TCP proprioception
  action: [base twist 3 | flange pose 7 | gripper drives 2]
        |
        +----------------------+
        |                      |
        v                      v
strict task predicates    scripted oracle
        |                      |
        +----------+-----------+
                   v
        env.step(12-D action)
                   |
                   v
       PhysX contacts and next state
                   |
                   +----> next oracle transition
                   |
                   +----> trace / randomized gate / video
```

All simulation tensors are batched by environment. A ten-environment gate does
not run ten Python controllers sequentially; one oracle instance holds a phase
and state row for each environment and emits a `(10, 12)` action tensor.

## 3. Geometry contract

File: `robolab/tasks/mobile_manipulation/genmanip_kitchen_geometry.py`

This file is deliberately free of Isaac imports. Geometry can therefore be
tested without starting Kit or PhysX.

### `Bounds2D`

`Bounds2D` stores closed `xmin`, `xmax`, `ymin`, and `ymax` limits.

- `center` returns the arithmetic midpoint and is used for default pick/place
  targets.
- `size` returns width and depth.
- `inflated(radius)` grows all four sides. We use this to convert a robot disk
  into a point-path query: inflate furniture by the base radius plus margin,
  then test the base center.
- `contains(point, tolerance)` makes bounds checks explicit and reusable.
- `distance_to(point)` returns zero inside and Euclidean distance outside.

The dataclass is frozen so a test or controller cannot accidentally mutate the
authoritative geometry during a run.

### `CollisionBox`

Each collision primitive has a stable name, 3-D center, 3-D size, semantic
category, and optional visibility. `footprint` projects the box to XY;
`z_bounds` returns its vertical interval. The detailed EBench meshes remain the
visual source, while these boxes form the task-grade collision contract.

### Physical and execution regions

The measured physical support regions are:

```text
PICK_REGION  x [-0.06, 0.04]  y [ 0.10, 0.25]
PLACE_REGION x [ 1.76, 1.86]  y [-0.02, 0.22]
```

The present side-grasp controller samples narrower regions:

```text
ORACLE_PICK_REGION  x [-0.03, 0.04]  y [0.12, 0.23]
ORACLE_PLACE_REGION x [ 1.79, 1.83]  y [0.09, 0.11]
```

The broad regions answer “is there physical support here?” The oracle regions
answer “has this particular IK/state-machine implementation been validated
here?” Keeping them separate avoids changing the scene’s physical truth to
hide a controller limitation.

### Staging poses and base inflation

The base is modeled conservatively as a 0.30 m disk plus 0.05 m margin. Pick,
place, and post-release retreat stages all use the authored base height of
0.468 m. The retreat pose moves the base away from the counter while preserving
the arm pose in the robot frame; this proved more stable than commanding a
near-singular vertical arm withdrawal.

### Can and manipulation constants

The YCB can envelope is measured as radius `0.033927 m` and height
`0.102037 m`. The wrapper uses a plausible 0.349 kg mass, high but reasonable
static/dynamic friction, low restitution, a 2 mm contact offset, and gravity.

`GRASP_TCP_OFFSET_FLANGE = (0, 0, 0.1429)` locates the midpoint of the flat
finger pads relative to `robotiq_85_base_link`. Controlling the TCP rather than
the flange makes “put the fingers around the can center” a direct geometric
command.

The side-grasp quaternions are explicit `(w, x, y, z)` values:

- `SIDE_GRASP_QUAT_WXYZ` approaches the island can along world `-X`;
- `PLACE_SIDE_GRASP_QUAT_WXYZ` mirrors the yaw to approach the opposing
  counter from the aisle.

Motion constants separate geometry from controller behavior. Examples:

- 0.12 m side pregrasp offset;
- 0.04 m lift, with 0.02 m required physical clearance;
- slower 0.015 m/s lift and place motion;
- 0.05 m/s free carry reposition;
- 0.10 m pre-place clearance;
- 0.025 m final XY tolerance;
- consecutive-frame requirements for pose, grasp, carry settle, release, and
  placement settle.

Free-body settle limits are stricter than carried-body limits. A constrained
body driven by an articulation retains small solver velocities even when its
relative transform is stable; using the free-body threshold during carry would
reject physically acceptable constrained motion.

### Collision/path functions

`_segment_intersects_box_interior` is a two-dimensional slab intersection
test. It computes the interval of segment parameter `t` that lies inside an
axis-aligned box. Boundary-only contact is excluded so a tangent path is not
reported as penetrating.

`navigation_obstacle_footprints` returns one outer footprint per wall/furniture
unit instead of every constituent countertop slab. This prevents duplicate
boxes from complicating path reasoning.

`base_pose_is_clear` tests a base center against every safety-inflated
footprint. `base_path_is_clear` applies the segment test to the entire straight
path, not only its endpoints.

`swept_center_intersects_box` is the 3-D equivalent. It inflates a static box
by a moving proxy’s half extents and intersects the center trajectory against
the expanded box. This is a Minkowski-sum reduction of swept-volume collision
checking.

`swept_proxy_is_clear` applies that test to the complete static overlay. The
oracle calls it for the can’s cylinder-sized proxy and sampled arm-link
proxies, catching a between-frame fixture crossing that endpoint checks would
miss.

## 4. Physics task scene and reset

File: `robolab/tasks/mobile_manipulation/kitchen_can_transfer_task.py`

### `STATIC_MATERIAL`

Static fixtures use friction `0.70/0.60`, zero restitution, `max` friction
combination, and `min` restitution combination. The combine modes stop a low
default material elsewhere in the scene from unexpectedly reducing support
friction or increasing bounce.

### `author_disabled_oracle_grasp_joints`

The prestartup event creates one disabled `UsdPhysics.FixedJoint` under every
environment namespace. Body 0 is the Robotiq flange and body 1 is the can. The
joint disables collision only between those two declared bodies; the separate
fingertip bodies still collide with the can and provide the bilateral contact
evidence. Once enabled, the joint preserves the observed relative pose.

The joints are authored before PhysX starts because creating/destroying joint
schemas during GPU simulation caused illegal-memory failures. The event rejects
non-`None` `env_ids` because it is a one-time scene-construction operation, not
a reset operation. `ExcludeFromArticulation` keeps the object joint from
becoming part of the robot articulation topology.

### `_static_box_cfg`

This converts a pure `CollisionBox` into an Isaac `AssetBaseCfg` with collision
enabled and no rigid-body or mass API. Therefore fixtures are truly static.
Only room-shell boxes marked visible receive a neutral visual material; the
collision overlay under detailed furniture remains invisible.

### `make_physics_can_cfg`

This creates the dynamic can wrapper:

- loads `tomato_soup_can_physics.usda`;
- enables rigid-body simulation and gravity;
- sets small damping, finite velocity/depenetration limits, gyroscopic forces,
  and higher solver iterations;
- assigns 0.349 kg mass;
- activates contact sensing;
- starts upright 2 cm above the island with zero velocity.

The USD contains a simple convex collision mesh and a detailed YCB visual child
with collision disabled. That gives stable contact without losing appearance.

### `GenManipKitchenPhysicsSceneCfg`

The scene combines the untouched detailed kitchen visual, dynamic can, and
explicit collision entities. `replicate_physics=False` is required because
each cloned environment owns an independently enabled/disabled grasp joint.

The island and counter support slabs are named fields because RoboLab contact
registration needs stable entity names. The remaining boxes are attached to
the config class in a loop so geometry and scene construction consume the same
single source of truth.

### `KitchenKinovaMobileCobotCfg`

This task-specific robot config starts at the pick stage and increases arm
stiffness/damping while preserving authored effort limits. The original gentle
gains produced too much steady deflection under the constrained 0.349 kg load.
Increasing position-loop gains rather than effort ceilings keeps the model
closer to the original hardware limits.

### `reset_can_on_island`

For selected environment IDs, the reset function clones the can’s default root
state, uniformly samples XY inside `ORACLE_PICK_REGION`, sets the authored drop
height and upright quaternion, zeroes all six velocities, and writes pose and
velocity to PhysX. Sampling is batched and uses Torch’s seeded RNG.

### Events, termination, and task dataclass

`KitchenCanTransferEvents` authors joints at prestartup, resets the full scene,
then samples the can. `KitchenCanTransferTerminations` supplies only the normal
episode timeout. The `GenManipKitchenCanTransferTask` dataclass binds scene,
events, termination, contact-object names, instruction, 120-second episode
length, and a PhysX velocity-iteration override of eight.

There is intentionally no task reward here. Milestone 2 is an oracle/evaluation
layer; learned-policy reward design comes later.

## 5. Robot action interface

File: `robolab/robots/kinova_mobile_cobot.py`

### Contact names

`MOBILE_KINOVA_CONTACT_GRIPPER` exposes aggregate, left-tip, and right-tip
regexes. An aggregate “some gripper contact” signal cannot prove a bilateral
grasp; independent names allow the predicate to require both sides.

### `HolonomicBaseVelocityAction`

The custom action term consumes body-frame `[vx, vy, wz]`.

- `__init__` allocates raw, clamped, and actually applied action tensors;
  constructs speed and per-physics-step acceleration limits; and captures the
  planar pose integrator state.
- `action_dim`, `raw_actions`, and `processed_actions` implement Isaac Lab’s
  action-term interface.
- `process_actions` stores the caller’s values and clamps speed.
- `apply_actions` slew-rate-limits acceleration, converts body twist to world
  twist, integrates XY/yaw, projects the floating articulation root back to
  planar SE(2), fixes the root height, and writes pose and velocity to PhysX.
  A displacement above 0.10 m is treated as an external reset/teleport so the
  integrator resynchronizes instead of pulling the base back.
- `reset` clears commands and resynchronizes the planar integrator for selected
  environments.

This is an ideal holonomic-base abstraction, not wheel-level dynamics. It is a
policy-agnostic mobile-base action contract suitable for the current benchmark
stage; wheel/drive controllers can later implement the same command semantics.

### Robot configs

`KinovaMobileCobotCfg` switches to the mobile USD, unfixes the root, strengthens
the gripper position drive, sets the 0.468 m root height, and loads a measured
IK-friendly initial joint seed. It also defines the navigation camera.

`KinovaMobileIKActionCfg` orders actions as base velocity, absolute flange pose,
then gripper. Differential IK uses damped least squares, an absolute seven-value
pose command, and zero `body_offset`. The task performs its own explicit
TCP-to-flange transform, avoiding hidden offset-frame Jacobian conventions.

`KinovaMobileContinuousGripperIKActionCfg` replaces the binary gripper command
with direct positions for the two driven Robotiq outer knuckles; the four remaining joints follow authored mimic relationships. The oracle can therefore stop
at the actual contact width and add controlled preload instead of always
requesting generic fully closed geometry.

## 6. Environment registration

File: `robolab/registrations/kinova_mobile/auto_env_registrations_ik.py`

`register_genmanip_kitchen_mobile_ik_env` lazily imports the task and configs,
builds a proprioception observation group, instantiates the continuous-gripper
IK action config, and reduces base acceleration to `0.20 m/s²` for loaded
transport. It then calls RoboLab’s environment generator with:

- the kitchen task;
- task-specific mobile Kinova config;
- proprioceptive observations;
- 12-D action config;
- fingertip contact mappings;
- 120 Hz physics, render every four physics steps, action decimation four;
- one environment by default and 8 m clone spacing.

The resulting control step is `4 / 120 = 1/30 s`. That is why trial step 1519
corresponds to about 50.6 seconds.

## 7. Task predicates

File: `robolab/tasks/mobile_manipulation/kitchen_manipulation_predicates.py`

The `@atomic` decorator integrates these functions into RoboLab’s task/world
state machinery. Predicates return a vector for all environments or one Python
value when `env_id` is supplied.

### Helpers

`_select` implements that vector-or-single-env return convention.

`_stable_boolean` stores one counter per environment in RoboLab world state.
True increments the counter; false resets it to zero. The factory also exposes
a reset hook. Consecutive-frame gates prevent one-frame contact impulses or
velocity noise from becoming task events.

`_can_state` returns environment-local can position plus scalar linear and
angular speeds. Subtracting `env_origins` is essential in vectorized scenes;
otherwise environment clone offsets contaminate task coordinates.

`_inside_region` performs vectorized XY bounds checks.

`_instant_bilateral_grasp` requires both fingertip contact booleans and both
force norms above threshold. Checking booleans and norms is redundant by
design: it protects against differences between filtered contact state and raw
force aggregation.

### Public predicates

`can_supported_by_island` requires:

- can center inside the broad physical pick region;
- center Z within 6 mm of surface plus half-height;
- strict free-body speed limits;
- support force when available.

GPU PhysX can report zero filtered force after a body sleeps, so a fully
conjunctive position/height/velocity fallback accepts a clearly settled can.

`can_bilaterally_grasped` wraps instant contact in a configurable
consecutive-frame gate.

`can_lifted_clear` requires bilateral contact, can bottom above the island by
the configured clearance, and no island contact. In assisted mode the oracle
adds a constraint-active height fallback because fingertip force can change
once the joint carries the load.

`base_at_place_stage` computes XY error, extracts yaw from quaternion, wraps
yaw error into `[-pi, pi]`, and also requires low planar and yaw velocities.
Being momentarily near the waypoint while still moving is not arrival.

`gripper_released` requires neither fingertip touching the can and the primary
gripper joint near open for consecutive frames.

`can_supported_in_place_region` mirrors island support on the counter, adds an
optional no-finger-contact requirement, and requires 15 stable frames.

`robot_collision_free` rejects non-finite robot/can state, a can outside finite
workspace limits, a base center inside safety-inflated furniture, or any
forbidden collision flag supplied by the environment.

`kitchen_transfer_success` records ordered historical facts: grasp happened,
then lift happened, then the base arrived while the object was held. It finally
requires current released placement and collision freedom. This prevents
accidental success if a can somehow begins on the goal or is knocked there
without completing the task sequence.

## 8. Oracle controller

File: `robolab/tasks/mobile_manipulation/kitchen_manipulation_oracle.py`

### Phase enum and timeouts

`OraclePhase` assigns compact tensor-friendly integer states. `SUCCESS` and
`FAILURE` are terminal. `_PHASE_TIMEOUT_STEPS` bounds every active state so a
controller cannot silently wait forever.

`ALIGN` remains in the enum/target mapping for compatibility with an earlier
alignment split but the current transition goes directly from settle to
pregrasp. Keeping an unused state is harmless, though removing it is a small
cleanup opportunity.

### `KitchenManipulationOracle.__init__`

Initialization stores the environment, device, batch size, robot, and can.
It resolves one pre-authored grasp joint per environment and fails early if
scene construction omitted them.

It resolves the flange body index and samples arm-like body names for swept
checks. `_resolve_action_slices` converts action-manager term names and sizes
into exact base/arm/gripper slices. Initialization validates all three names,
the seven-value pose dimension, and two driven gripper joints; a silently
different policy interface would otherwise produce plausible but wrong motion.

The gripper section resolves joint indices, constructs the signed fully closed
command, and stores sign values used when adding preload to joints whose
directions differ.

The transform section batches TCP offsets and pick/place quaternions. A target
may be one XY pair broadcast to all environments or one pair per environment.
Shape and conservative-region bounds are checked before simulation begins.

Persistent tensors then store, per environment:

- phase and phase step;
- pose/carry/handoff stability counts;
- captured pick, secure, and lift TCP positions;
- rate-limited lift and general motion commands;
- approach orientation;
- arm pose to hold during carry/base motion;
- grasp-width and opening joint positions;
- failure text/diagnostics;
- grasp-constraint state and captured flange-to-can pose;
- previous can and arm-link positions for swept checks.

`control_dt` comes from `env.step_dt`, so Cartesian rate limits remain correct
if physics/decimation changes.

### Reset and grasp-joint methods

`reset` can reset all environments or selected rows. It disables assist joints,
returns phases/counters/targets to zero, refreshes swept-check history from the
new simulation state, and clears Python failure records.

`_attach_grasp_constraint` does nothing when assistance is off or already
active. Otherwise it computes the can pose in the current flange frame,
writes that exact transform into the joint, enables the joint, and marks it
active. Capturing the observed transform avoids snapping the can to a generic
grasp pose.

`_detach_grasp_constraints` disables selected joints and clears their active
flags. `_apply_grasp_assist` is intentionally empty: an earlier implementation
wrote can poses every frame, but the pre-authored PhysX joint now owns the
dynamics. `post_step` remains as a stable runner hook.

`finished`, `succeeded`, and `phase_names` expose batched terminal state without
requiring callers to interpret integer values.

### Frame-transform methods

`_current_flange_w` returns flange pose in world coordinates.

`current_tcp_w` rotates the local TCP offset by flange orientation, adds it to
flange position, and uses flange quaternion as TCP orientation.

`_tcp_world_to_flange_root` performs the inverse operation required by the IK
action term:

1. subtract rotated TCP offset to recover desired flange position in world;
2. subtract the mobile-root transform to express flange pose in robot-root
   coordinates;
3. concatenate XYZ and WXYZ into the seven-value action.

`_pose_reached` checks 12 mm position and 0.10 rad orientation errors for five
consecutive frames. `_set_phase` changes one row and resets phase/pose counters.

### Safety and diagnostic methods

`_swept_motion_clear` only runs detailed carried-motion checks in traverse,
reposition, reorient, and pre-place. It checks the previous-to-current can
segment with the measured can extents and each selected arm body with a 35 mm
proxy, updates history every call, and returns per-environment reasons.

`_fail` snapshots the failure before detaching the joint: phase, phase steps,
can pose/speeds, current and target TCP, TCP error, placement error, handoff
counter, and constraint state. It then stores the reason, safely detaches, and
enters `FAILURE`.

`_capture_pick_targets` records the settled randomized can center as the TCP
grasp target.

### `_target_tcp_w`

This function maps phase to desired world TCP pose.

- Pick phases use the island side-grasp quaternion.
- Counter phases use the mirrored place quaternion.
- The desired can center is sampled XY plus counter surface and can half-height.
- The desired TCP is not assumed equal to the can center. It is corrected using
  the TCP offset and the flange-to-can translation captured at secure grasp.
  This removed a systematic 14–15 mm placement bias.
- Carry reposition commands a high safe world Z relative to the mobile root.
- Pregrasp offsets 0.12 m along world X.
- Preplace adds 0.10 m vertical clearance.

The function initializes targets from current pose, then overwrites rows by
phase. Terminal/hold rows therefore retain safe values rather than uninitialized
commands.

### `_base_command` and `_base_at_retreat_stage`

The base moves only in `TRAVERSE` or `RETREAT`. `_base_command` chooses the
corresponding stage, computes world XY error, rotates it into the robot body
frame, uses `min(0.08, 0.6 * distance)` speed, and stops inside a fraction of
the final tolerance. The action term supplies acceleration limiting.

`_base_at_retreat_stage` requires both position tolerance and low planar speed.

### `_advance`: predicate evaluation and state transitions

At the top of every control step, `_advance` computes all predicates and live
errors in batch, including collision and swept checks. It then loops only over
environment rows to execute state-specific transitions:

| State | Command/meaning | Exit condition |
|---|---|---|
| `SETTLE` | Hold current arm/base | Can stably supported on island |
| `PREGRASP` | TCP 0.12 m away with side-grasp orientation | Pose stable |
| `DESCEND` | TCP to captured can center | Pose stable |
| `CLOSE` | Drive both outer knuckles closed; four joints mimic them | Instant bilateral contact |
| `SECURE` | Hold contact pose with bounded 0.04 rad preload | Pose stable and bilateral force stable for eight frames |
| `LIFT` | Rate-limited 0.04 m vertical lift | Can clear and command reaches lift target |
| `CARRY_SETTLE` | Hold root-frame flange pose | Lift remains true and carried speeds settle for five frames |
| `TRAVERSE` | Hold arm in root frame; command slow base motion | Base reaches place stage and stops |
| `CARRY_REPOSITION` | Rate-limited move to high counter-side position | Pose stable |
| `CARRY_REORIENT` | Rotate to mirrored side grasp | Pose stable |
| `PREPLACE` | Rate-limited move 0.10 m above target | Pose stable |
| `PLACE` | Slow descent | Can center in physical region, within 25 mm target error, and at release height |
| `HANDOFF_SETTLE` | Detach/open; hold arm | Free can supported/released/accurate and speed-stable |
| `OPEN` | Confirm final released support | Support, release, target accuracy |
| `RETREAT` | Hold arm in root frame; move base back | Base stopped at retreat and placement still valid |
| `SUCCESS` | Terminal | — |

Global collision, swept-collision, non-finite-state, and phase-timeout checks
run before the phase-specific transition.

During `CLOSE`, preload is computed from the observed contact width, signed per
joint, and clamped between open and fully closed limits. During `SECURE`, the
oracle captures the actual contact-deflected TCP before lifting and optionally
enables the fixed joint.

At release, the controller first verifies geometry, freezes the arm’s root-frame
pose, zeroes residual assisted can velocity, detaches, and opens. Only then does
it test the free body’s counter support. This ordering is essential; the older
attached-settle ordering overconstrained the contact solver.

### `compute_action`

`compute_action` is the public controller entry point.

1. It computes targets and calls `_advance`.
2. It recomputes targets so a new phase receives its new command immediately.
3. It rate-limits lift, carry reposition/pre-place, and final place descent in
   Cartesian space using speed times `control_dt`.
4. It converts commanded TCP world poses to flange poses in the mobile-root
   frame.
5. During carry settle, traversal, handoff, open, and retreat it replaces IK
   output with the captured root-frame flange hold pose. The arm therefore
   travels with the base rather than attempting to remain fixed in world.
6. It allocates the complete 12-D action tensor and fills base, arm, and
   gripper slices.
7. Gripper logic searches fully closed in `CLOSE`, holds captured preload from
   `SECURE` through `PLACE`, and rate-limits opening during handoff/open.
8. It increments phase timers only for unfinished rows and returns the action.

## 9. Single-trial trace runner

File: `examples/run_genmanip_kitchen_manipulation_oracle.py`

The import order is intentional. OpenCV is loaded before Isaac Lab to avoid
shared-library conflicts; `AppLauncher` starts Kit before Torch/Isaac modules
are imported. `# noqa: E402` documents that deliberate late-import pattern.

CLI flags control seed, trial count, placement randomization, grasp mode,
optional HDF5 recording, step budget, and JSON output.

`_snapshot` captures a human-debuggable state: phase, can/base/TCP poses and
speeds, fingertip poses, fingertip force vectors/contact booleans, assist state,
and all gripper joint positions.

`_run_trial` resets RoboLab evaluation and scene state, constructs an oracle,
steps `compute_action -> env.step -> post_step`, records every phase transition,
adds dense contact/carry snapshots for a single trial, and returns terminal
metadata plus the trace.

`main` validates arguments, seeds Torch and a separate Python target RNG,
registers/parses the IK environment, disables cameras and HDF5 unless requested,
runs one or more trials, writes JSON, and returns nonzero on any failure. The
outer `try/finally` always closes both environment and simulation app.

## 10. Randomized exit-gate runner

File: `examples/validate_genmanip_kitchen_oracle_gate.py`

This runner is optimized for evidence rather than visualization.

`_sample_place_targets` uses a private seeded Python RNG so target samples are
reproducible independently of Torch’s reset sampling.

`_final_state` captures terminal/final can pose and speeds, base pose, target
error vector, phase, and constraint state.

`_run_batch` pads the last partial batch to the fixed environment count and
marks padding rows successful so they do not block vectorized completion. It
resets the scene, captures initial can positions, constructs a per-row-target
oracle, records phase histories, converts unexpected RoboLab termination into
an oracle failure, snapshots state at the first terminal step, prints aggregate
phase counts every 100 steps, and returns one result object per real trial.

`_build_summary` records completion/success counts, seed, batch size, explicit
grasp mode, both conservative sampling domains, swept-check status, and all
trial records. `all_success` is true only when the requested number completed
and all succeeded; an interrupted partial run cannot look successful.

`_write_summary` writes to `output.tmp`, flushes Python buffers, calls `fsync`,
then atomically replaces the target. A kill or crash therefore leaves either
the previous valid checkpoint or the new valid checkpoint, not truncated JSON.

`main` validates counts, resolves explicit or sampled targets, creates a
camera-free/recorder-free non-Fabric environment, runs batches, checkpoints
after each batch, and optionally stops after the first failed batch. The final
summary is written again after environment close.

## 11. Visualization runner

File: `examples/visualize_genmanip_kitchen_oracle_trial.py`

The visualization is kept separate from the gate so RTX/material compilation
and video encoding cannot affect validation throughput.

Before AppLauncher, the script registers Isaac and EBench MDL search paths and
accepts output, still, montage, trace, seed, step, and capture-stride arguments.
It forces cameras on.

`_world_to_panel` maps kitchen XY coordinates into pixels. `_draw_bounds`
draws a physical or inflated region. `_phase_group` collapses detailed oracle
states into ten viewer-friendly labels.

`_top_down_panel` draws fixture footprints, base-inflated boundaries, physical
and conservative regions, staging paths, robot footprint, live can/TCP, and
target. `_annotate` combines that panel with the rendered RGB view and live
phase/pose text.

`_live_state` converts world tensors into environment-local values. `_save_rgb`
handles RGB-to-BGR conversion for OpenCV. `_save_montage` requires eight key
phases and creates a 2-by-4 contact sheet.

`main` builds the same IK environment but enables Fabric because RTX cameras
need moving articulation transforms synchronized to rendering. It disables the
unused robot cameras/recorder, adds a 960x540 independent scene camera, runs an
assisted center-target oracle, captures every second control step and every
phase change, validates success, and saves MP4/still/montage/JSON. At 15 fps
with stride two, video time matches the 30 Hz control loop.

## 12. Automated tests

`tests/test_genmanip_kitchen_physics_schema.py` checks, without relying on a
successful rollout:

- every overlay fixture is a static collider with positive size;
- the can is dynamic, massive, gravity-enabled, contact-enabled, convex, and
  contains a non-colliding visual child;
- collision slabs leave the sink opening free;
- physical and conservative regions are nested on valid supports;
- every staging pose and both base path segments clear inflated furniture;
- swept checks catch a between-frame wall crossing;
- the raised carry corridor is clear and a low corridor is rejected;
- joint-position and IK versions register as `RobolabEnv`;
- IK action, PhysX iteration count, and separate fingertip mappings are correct.

`tests/test_genmanip_kitchen_physics_sim.py` starts Isaac and checks:

- `_matrix_values` converts USD matrices for exact comparison;
- `_action_slices` resolves the configured action layout;
- `_hold_action` holds default arm pose with stopped base;
- `_step` asserts no unexpected termination;
- `_set_robot_stage` writes a precise root pose/velocity/joint state;
- the fixture creates one reusable module-scoped environment;
- a randomized can drops and settles on the island for three seconds;
- every live overlay prim has collision, no rigid-body API, and never moves;
- the robot settles correctly at pick and place stages;
- the base traverses the complete aisle path while remaining planar/clear and
  while the untouched can stays supported.

The current validation results are 16/16 static/schema tests, 5/5 physics
simulation tests, and 100/100 assisted oracle trials.

## 13. Visualized seed-101 trial

The fresh rendered run used the center target `(1.81, 0.10)` and succeeded at
step 1519. Its transition trace is:

| Step | Time (s) | Entered state | Meaning |
|---:|---:|---|---|
| 1 | 0.03 | settle | Can falling/settling under gravity |
| 3 | 0.10 | pregrasp | Island support verified |
| 39 | 1.30 | descend | Pregrasp pose stable |
| 52 | 1.73 | close | TCP aligned with can |
| 97 | 3.23 | secure | Bilateral contact found |
| 104 | 3.47 | lift | Bilateral preload confirmed; assist enabled |
| 181 | 6.03 | carry_settle | Can physically clear of island |
| 186 | 6.20 | traverse | Loaded motion settled |
| 453 | 15.10 | carry_reposition | Base stopped at place stage |
| 769 | 25.63 | carry_reorient | Counter-side high pose reached |
| 840 | 28.00 | preplace | Mirrored grasp orientation reached |
| 1287 | 42.90 | place | Pre-place pose reached |
| 1425 | 47.50 | handoff_settle | Release geometry verified; assist detached |
| 1446 | 48.20 | open | Free can stable and released |
| 1447 | 48.23 | retreat | Placement confirmed |
| 1519 | 50.63 | success | Retreat stage reached with placement intact |

Final can center was `(1.8084, 0.1035, 0.8963) m`; the grasp constraint was
off. The MP4 is H.264/yuv420p, 1380x540, 15 fps, 770 frames, and 51.33 seconds.

Artifacts:

```text
output/genmanip_kitchen/milestone2/oracle_trial_visualized.mp4
output/genmanip_kitchen/milestone2/oracle_trial_final.png
output/genmanip_kitchen/milestone2/oracle_trial_keyframes.png
output/genmanip_kitchen/milestone2/oracle_trial_visualized.json
```

## 14. Challenges, bugs, causes, and fixes

### Detailed scene meshes were unsuitable as direct collision

The EBench kitchen meshes contain seams, thin decorative triangles, and complex
topology. Using them directly produced poor contact and made navigation
reasoning opaque. The fix was to preserve them only for appearance and add a
measured primitive collision overlay used by PhysX and tests.

### Runtime joint/collision-schema changes crashed GPU PhysX

Creating or changing collision/joint topology after simulation startup caused
CUDA illegal-memory failures. GPU PhysX expects stable scene topology. The fix
was to author one disabled fixed joint per environment at prestartup, disable
physics replication, and only update joint frames plus the enabled attribute.

### Unassisted friction grasp dropped the can

The rounded source finger collisions, contact patch, position-drive behavior,
and carried acceleration did not provide a reliable 100-trial pinch grasp.
Higher gripper stiffness, damping, solver iterations, preload, slower base
acceleration, and rate-limited Cartesian motion improved it but did not make it
gate-reliable. The explicit assist was introduced only after bilateral
force-confirmed contact so the rest of the execution stack could be validated
without pretending friction fidelity was solved.

### A generic constraint orientation was unstable

Replacing the captured flange-to-can rotation with a normalized generic pose
failed 0/5 in one experiment. It introduced a transform discontinuity and
solver snap. The fix was to preserve the full observed relative translation
and quaternion at the secure-contact instant.

### Place target initially meant TCP rather than can center

The oracle first used sampled `place_xy` as a TCP target. Because the real
side-grasp contact leaves the TCP 14–15 mm from the can center, valid controller
motion produced systematically shifted object placement. The fix converts a
desired can center through the captured flange-to-can translation and TCP
offset before commanding the place TCP.

### Broad execution regions exposed controller conditioning limits

Early broad randomized placement passed only 5/10. A center target with broad
pick samples passed 10/10; a later broad-pick/conservative-place run passed
19/20, with the remaining far-edge sample timing out in lift. These were reach
and IK-conditioning issues, not absent support geometry. The fix introduced
separate conservative oracle regions while retaining broad physical predicates.
Future work must expand them rather than silently redefining support.

### Attached can and counter contact fought the solver

The original handoff waited for the can to become stationary on the counter
before detaching the fixed joint. One seed-101 trial reached about 1.03 m/s and
7.77 rad/s because the driven flange constraint and static counter contact
overconstrained the same rigid body. The fix was to verify release geometry,
zero residual assisted velocity, detach/open, and only then require free-body
settling. The exact reproduction changed from 9/10 to 10/10, followed by the
100/100 gate.

### Arm-only retreat approached a singular configuration

A vertical post-place arm withdrawal was unstable near this staging pose. The
fix holds the observed flange pose relative to the robot root and retreats the
base from `(1.12, 0.10)` to `(1.00, 0.10)`. The arm moves with the chassis and
the final predicates are checked throughout.

### Long gates lost evidence when interrupted

The first 100-trial attempt wrote JSON only after every batch and environment
shutdown. Once a failure made 100/100 impossible, stopping the long run risked
losing diagnostics. The gate now atomically checkpoints every batch and has a
fail-fast option.

### Cameras and HDF5 made validation unnecessarily expensive

The oracle needs proprioception and contacts, not pixels or full trajectory
recording. Leaving cameras/HDF5 enabled consumed RTX time and created a growing
data file. Gate defaults now disable both; the visualization and recorder are
explicit opt-ins.

### Rendering reports missing EBench textures

The imported scene references several absent/unresolved flattened JPG textures.
Isaac’s MDL translator emits warnings and uses fallbacks. This affects some
surface appearance, not collision or oracle success. The visualization adds
the known MDL search roots, but genuinely absent texture files remain an
upstream asset-packaging issue.

### PhysX prints “disjointed body transforms” for assist joints

The fixed joints exist disabled at startup with neutral frames, then receive
the observed grasp transform before being enabled. PhysX logs a topology/frame
warning when it rebuilds the joint. The 100-trial result and detached terminal
states show the runtime behavior is stable, but eliminating the warning through
a cleaner constraint API is technical debt.

## 15. Next step

### Optional embodiment-fidelity track: unassisted friction

This is useful when claiming Robotiq contact fidelity, but it is not a blocker
for the robot-agnostic task interface:

1. inspect and simplify Robotiq fingertip collision geometry;
2. measure contact normals, patch forces, slip velocity, and joint tracking;
3. tune friction/contact offsets and actuator gains against real gripper data;
4. use compliant close/force behavior instead of pure aggressive position
   closure;
5. tune lift/base jerk with the assist disabled;
6. run the exact same 10-trial reproduction, then 100-trial randomized gate
   with `grasp_mode=unassisted_friction`.

The assisted result is the supported reference mode and must be labeled as
constraint-assisted. The friction-only result remains a separate embodiment
validation metric.

### Gate B: expose a policy-ready task interface

After or alongside grasp-fidelity work:

1. define observations without leaking the oracle phase: base pose/velocity,
   arm/gripper state, TCP pose, can pose/velocity or camera observations,
   target pose, and optional collision-distance features;
2. keep the current policy-agnostic action semantics, or add a normalized
   wrapper around base twist, delta TCP pose, and gripper command;
3. add reward terms and terminations derived from the existing predicates;
4. export oracle trajectories with explicit `grasp_mode` metadata;
5. start with behavior cloning to verify observation/action plumbing;
6. evaluate every policy on the same reset seeds and report success beside the
   oracle, including drops, collisions, placement error, and completion time;
7. expand conservative pick/place domains only after controller/policy evidence
   supports it.

This preserves the intended architecture: the scene/task contract stays robot
and policy agnostic; robot configs implement action semantics; an oracle,
learned policy, teleoperator, or motion planner can all drive the same
registered RoboLab environment and be judged by the same predicates.
