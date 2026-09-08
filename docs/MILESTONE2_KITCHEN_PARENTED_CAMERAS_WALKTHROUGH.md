# Milestone 2 kitchen sensors and workspace-facing navigation: complete walkthrough

## 1. What this increment built

This increment makes the kitchen trial usable for later vision-policy evaluation
without changing the task into a Kinova-specific camera demo. It adds and
validates the following behavior:

1. the Kinect proxy is mounted once beneath the mobile robot's `base_link`;
2. the camera inherits every base translation and rotation through the USD
   parent transform;
3. no controller code teleports or rotates the policy camera at runtime;
4. the harder `robot_other_side` scenario drives around the north side of the
   island to the existing pick stage;
5. every carry route reaches the placement stage facing the opposing counter;
6. the wrist camera, Kinect RGB/depth, and fixed third-person view are recorded
   synchronously;
7. the renderer records the actual moving articulation rather than a stale
   authored USD pose;
8. the trace measures whether each sensor is producing changing RGB, valid
   depth, and a view of the can;
9. static regression tests prevent a return to a floating camera, camera
   teleport calls, or a place route with the wrong final heading.

The final hard-scenario trial succeeded at step 3008. The Kinect produced 765
sampled RGB/depth frames and saw the can in 495 of them, including the grasp,
carry, placement, release, and retreat phases.

This is a sensor and navigation validation increment, not a new grasp-fidelity
claim. The rendered trial uses `strict_post_contact_constraint`: physical
bilateral contact qualifies the grasp, after which the existing fixed-joint
transport assist is enabled. The 100-trial policy-camera gate has not been run.

## 2. The invariant that controls the design

The policy camera is part of the robot, not part of the oracle.

```text
World
  `-- env_0
      `-- robot
          `-- base_link
              |-- physical sensor mast and bracket
              `-- navigation_camera
```

Let `B` be the robot base frame and `C` the camera frame. The camera's local
mount is constant:

```text
p_C^B = (0.0656, -0.3507, 0.64)
q_C^B = (0.9961947, 0.0, 0.0871557, 0.0)
```

At any simulation step, the world pose follows from the base pose:

```text
p_C^W = p_B^W + R_B^W p_C^B
q_C^W = q_B^W tensor-product q_C^B
```

USD evaluates this relationship automatically for rendering because the camera
prim is a child of `base_link`. The visualization code repeats the same
calculation only for diagnostics. It never writes the result back to the
camera.

This separation matters for policy evaluation. The environment owns the robot
and sensor geometry. A navigation policy, scripted oracle, teleoperator, or
planner may rotate the base, but none of them gets a hidden operation that
points the camera directly at the target.

## 3. Reference implementations inspected

The final design follows the common robot-relative sensor pattern rather than
inventing a separate moving object for the camera.

### Isaac Lab G1 locomanipulation

The official G1 environment places its point-of-view camera at a path beneath
the robot torso and D435 link:

```text
/World/envs/env_.*/Robot/torso_link/d435_link/camera
```

It creates the camera with a local `CameraCfg.OffsetCfg`. The robot articulation
moves the parent links; the camera follows through the hierarchy. Source:
[G1 locomanipulation environment](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_mimic/isaaclab_mimic/locomanipulation_sdg/envs/g1_locomanipulation_sdg_env.py).

### Isaac Lab Galbot

Galbot creates wrist and ego cameras below authored robot camera frames, for
example:

```text
{ENV_REGEX_NS}/Robot/right_arm_camera_sim_view_frame/right_camera
{ENV_REGEX_NS}/Robot/head_camera_sim_view_frame/head_camera
```

It separately uses `FrameTransformerCfg` when it needs physical frame pose
information. That is an important distinction: the camera hierarchy controls
rendering, while a transform sensor can provide pose data for algorithms.
Source: [Galbot stack configuration](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/galbot/stack_rmp_rel_env_cfg.py).

### Local `cobot_moma_isaacsim`

The local reference config uses:

```text
/World/CoBotMoMa/base_link/rgbd_camera
/World/CoBotMoMa/base_link/chase_camera
```

`CoBotMoMa._create_camera` constructs the RGB-D sensor with one local
translation and orientation. The containing robot prim supplies subsequent
base motion. This was the closest local confirmation that the Kinect proxy
should be beneath `base_link`, not independently updated in a base action term.

Relevant local files:

```text
cobot_moma_isaacsim/config/cobot_moma_sim.toml
cobot_moma_isaacsim/src/cobot_moma.py
```

### InternRobotics and GenManip

InternRobotics' sensor interface likewise describes sensor prim paths relative
to robot links. GenManip uses multi-view camera randomization for data
generation, but that does not imply moving a physical robot sensor every
control step. Randomized extrinsics should be sampled at reset and then remain
fixed for the episode. References:
[InternRobotics sensor tutorial](https://internrobotics.github.io/user_guide/internutopia/tutorials/how-to-use-sensor.html)
and [GenManip](https://github.com/InternRobotics/GenManip).

## 4. Coordinate, camera, and timing conventions

### Robot and kitchen frames

- Kitchen/world axes are `X`, `Y`, and `Z`, with `Z` upward.
- Base commands are body-frame `[vx, vy, wz]`.
- The current mobile CoBot's navigation camera looks approximately along robot
  `-X`, not robot `+X`.
- Base yaw is stored in radians and quaternions use `wxyz` ordering.
- The robot root is kept at `z = 0.468 m`.

The `-X` optical direction explains why setting a placement yaw of zero made
the camera appear sideways or backward relative to the counter. The correct
fix was a base heading of pi radians at the placement side.

### Camera frame conventions

Isaac Lab exposes several camera conventions:

- the configured `world` offset convention is convenient for authoring the
  sensor mount;
- OpenGL camera geometry uses its own optical-axis convention;
- ROS camera projection treats positive `Z` as forward.

The diagnostic code converts the configured quaternion to OpenGL before
composing it with the parent body orientation, then converts the world camera
orientation from OpenGL to ROS before pinhole projection. Mixing these
conventions produces plausible images but incorrect projected can markers.

### Simulation and video timing

- `max_steps` is the maximum environment-control-step count;
- `capture_stride` controls how often frames are copied to CPU and encoded;
- phase transitions and the terminal frame are captured even if they fall
  between regular stride samples;
- output video uses 15 fps;
- camera validation counts captured frames, not every physics step.

## 5. Files involved

| File | Responsibility in this increment |
|---|---|
| `assets/robots/kinova_mobile_cobot/kinova_mobile_cobot.usda` | Physical mast and bracket; deliberately contains no camera rigid body or camera joint |
| `robolab/robots/kinova_mobile_cobot.py` | Direct-child Kinect sensor config and existing planar base action contract |
| `robolab/robots/kinova_gen3.py` | Existing wrist camera inherited by the mobile robot |
| `robolab/tasks/mobile_manipulation/genmanip_kitchen_geometry.py` | Placement and retreat headings |
| `robolab/tasks/mobile_manipulation/kitchen_scenarios.py` | Hard-scenario routes and the translate-then-rotate placement route |
| `robolab/tasks/mobile_manipulation/kitchen_manipulation_oracle.py` | Yaw-aware route follower and safe carried-arm reorientation |
| `examples/visualize_genmanip_kitchen_oracle_trial.py` | Synchronized rendering, visibility diagnostics, video, montage, and JSON trace |
| `tests/test_mobile_cobot.py` | Camera mount, sensor outputs, and absence-of-teleport regression tests |
| `tests/test_genmanip_kitchen_physics_schema.py` | Collision-clear scenario routes and workspace-facing final yaw tests |

## 6. Robot USD: physical mast without a fake camera body

File: `assets/robots/kinova_mobile_cobot/kinova_mobile_cobot.usda`

### `sensor_mast`

The mast remains a collision-enabled cube beneath `base_link`:

```text
scale     = (0.0125, 0.0125, 0.35)
translate = (0.0656, -0.3507, 0.30)
```

USD cube scales are half-extents, so the mast has a 25 mm by 25 mm cross
section and extends 0.70 m in `Z`. The mount is visibly and physically connected
to the robot rather than appearing as a floating optical viewpoint.

### `camera_bracket`

The bracket is another collision-enabled cube:

```text
scale     = (0.075, 0.045, 0.02)
translate = (0.0656, -0.3507, 0.60)
```

It represents a 150 mm by 90 mm by 40 mm physical camera support. Its center is
below the optical center, so the rendered sensor appears mounted rather than
balanced at the tip of the mast.

### What is intentionally absent

The USD does not contain:

```text
def Camera "navigation_camera"
def Xform "navigation_camera_link"
def PhysicsFixedJoint "navigation_camera_joint"
```

The sensor config spawns the camera beneath the already-existing rigid
`base_link`. Adding a 0.01 kg camera articulation link and a fixed joint was a
temporary debugging experiment. It did not fix the stale camera-pose metadata
and unnecessarily changed the robot's physical topology, so it was removed.

## 7. Mobile robot configuration

File: `robolab/robots/kinova_mobile_cobot.py`

### Imports

- `isaaclab.sim as sim_utils` supplies `PinholeCameraCfg`.
- `TiledCameraCfg` defines a batched Isaac Lab camera sensor.
- The existing Torch, action-manager, IK, and actuator imports remain for the
  mobile robot action stack.

### `HolonomicBaseVelocityAction`

This class still owns only base motion. That boundary is now deliberate.

#### `__init__`

The constructor allocates raw, bounded, and applied `[vx, vy, wz]` tensors for
each environment. It also stores velocity, acceleration, and jerk limits. The
initial root `XY` and yaw become the internal ideal-planar state.

No camera name, camera handle, or camera transform is stored here.

#### `action_dim`, `raw_actions`, and `processed_actions`

These properties expose the standard Isaac Lab action-term interface. The
dimension stays three, so sensor mounting does not alter the policy action
space.

#### `process_actions`

The requested body twist is copied to `_raw_actions` and independently clamped
against linear and angular speed limits. It does not inspect the task target or
camera view.

#### `apply_actions`

The method:

1. jerk-limits the requested velocity;
2. reads the current robot root pose;
3. detects external reset/teleport operations larger than 0.10 m;
4. converts the applied body twist to a world-frame twist;
5. integrates planar `XY` and yaw;
6. writes an upright quaternion and fixed root height;
7. writes the root pose and velocity to PhysX.

Because `navigation_camera` is a child of `base_link`, step 6 automatically
moves and rotates its rendered frame. There is no call to
`camera.set_world_poses()`.

#### `reset`

Reset clears all command history and resynchronizes the internal planar state
from the reset robot root. Again, no sensor-specific reset logic is necessary;
the child camera inherits the reset base transform.

### `HolonomicBaseVelocityActionCfg`

This config keeps the base's generic limits:

```text
linear speed        0.6 m/s
angular speed       1.0 rad/s
linear acceleration 1.2 m/s^2
angular acceleration 2.0 rad/s^2
linear jerk          2.4 m/s^3
angular jerk         4.0 rad/s^3
planar root height   0.468 m
```

There is intentionally no `attached_camera_name` field. Sensor movement is a
scene-graph responsibility, not an action-term option.

### `KinovaMobileCobotCfg.navigation_camera`

The final sensor declaration is:

```python
navigation_camera = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/base_link/navigation_camera",
    update_latest_camera_pose=True,
    height=360,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.0,
        horizontal_aperture=4.0,
        vertical_aperture=2.25,
        clipping_range=(0.1, 20.0),
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0656, -0.3507, 0.64),
        rot=(0.9961947, 0.0, 0.0871557, 0.0),
        convention="world",
    ),
)
```

Each field has a specific role:

- `prim_path` places the sensor directly beneath `base_link` in every cloned
  environment.
- `update_latest_camera_pose=True` requests current pose metadata. Rendering
  follows the parent correctly even though pose metadata can lag under Fabric;
  that issue is handled only in diagnostics.
- `360 x 640` gives a lightweight 16:9 policy stream.
- RGB is needed for visual policies and qualitative review.
- `distance_to_image_plane` gives metric depth along the optical axis.
- A 2 mm focal length with 4 mm horizontal aperture gives roughly a 90-degree
  horizontal field of view.
- A 2.25 mm vertical aperture gives roughly a 59-degree vertical field of view,
  close to a Kinect-style 16:9 RGB mode.
- The 0.1–20 m clipping interval covers manipulation and room-scale navigation.
- The local position reuses the calibrated CoBot horizontal mount and raises
  the optical center to 0.64 m above `base_link`.
- At nominal root `z = 0.468 m`, the lens is at `1.108 m`, about 6.3 cm above
  the `1.045 m` island surface.
- The quaternion points the sensor along robot `+X` with a mild downward pitch,
  matching the repository-wide robot-forward convention.

The height increase is intentionally small. Lowering the kitchen would change
fixture geometry, collision constants, reachability, and all task surfaces.
Raising the existing physical sensor mount by a few centimeters changes only
the embodiment configuration and is the lower-effort, lower-risk correction.

### Existing wrist camera

`KinovaMobileCobotCfg` inherits the wrist camera from `KinovaGen3Cfg`. It is
already a direct child of `bracelet_link`:

```text
prim path  {ENV_REGEX_NS}/robot/bracelet_link/wrist_cam
resolution 1280 x 720
outputs    RGB + distance_to_image_plane
offset     (0.0, -0.05639, -0.074575)
```

The same parenting principle therefore covers both policy sensors: the base
link carries the Kinect and the bracelet link carries the wrist camera.

## 8. Placement heading and hard-scenario route

### Staging poses

File: `robolab/tasks/mobile_manipulation/genmanip_kitchen_geometry.py`

```text
pick    = (0.52, 0.175, 0.468, 0)
place   = (1.12, 0.10, 0.468, pi)
retreat = (1.00, 0.10, 0.468, pi)
```

The pick stage remains at yaw zero. The place and retreat stages use yaw pi so
a Kinect that looks along robot `-X` faces world `+X`, toward the opposing
counter.

### `KitchenManipulationScenario`

File: `robolab/tasks/mobile_manipulation/kitchen_scenarios.py`

The frozen dataclass contains all reset- and route-specific differences:

- `initial_base_pose` includes `x`, `y`, root height, and yaw;
- `pick_region`, support name, grasp quaternion, and pregrasp offset describe
  which side of the island is used;
- `pre_pick_waypoints` route an initially remote base to the pick stage;
- `carry_waypoints` route a loaded robot to placement;
- separate speed and tolerance fields let hard scenarios remain conservative
  without modifying the generic base action.

Freezing the dataclass prevents a trial from silently changing its route.

### `robot_other_side`

The robot begins west of the island:

```text
start      (-1.18, 0.175, pi)
north-west (-1.18, 1.50,  pi)
north-east ( 0.58, 1.50,  0)
east pick  ( 0.52, 0.175, 0)
```

The north route is used because the L-shaped counter blocks the south passage.
All segments were checked against furniture footprints inflated by the 0.30 m
base radius plus the 0.05 m safety margin.

### Translate, then rotate at placement

Every carry route now ends with two waypoints at the same `XY`:

```text
place approach = (1.12, 0.10, 0)
place facing   = (1.12, 0.10, pi)
```

This is not redundant. It separates translation from the 180-degree turn. If
the robot translates and turns simultaneously near the counter, the carried
can follows a wider arc and can strike a fixture. The first waypoint finishes
translation with the pick-side orientation; the second rotates in place and
points the fixed Kinect at the placement workspace.

### `get_kitchen_scenario`

The lookup returns one immutable scenario from `SCENARIOS`. Unknown names raise
a `ValueError` listing valid options, so a misspelled evaluation case cannot
silently fall back to the baseline.

## 9. Oracle changes required by the base turn

File: `robolab/tasks/mobile_manipulation/kitchen_manipulation_oracle.py`

### `_route_complete`

This helper does more than check final `XY`. It:

1. converts the root quaternion to yaw;
2. compares the root to the final route waypoint;
3. wraps yaw error into `[-pi, pi]`;
4. requires scenario-specific position and yaw tolerances;
5. requires planar speed below 0.02 m/s;
6. requires angular speed below 0.03 rad/s.

The oracle therefore cannot begin arm placement while the base is still
turning and the policy camera is still moving.

### `_base_command`

The route follower selects one of three route sources:

- `pre_pick_waypoints` during `NAVIGATE_TO_PICK`;
- `carry_waypoints` during `TRAVERSE`;
- a single retreat waypoint during `RETREAT`.

For each environment it:

1. reads root position and yaw;
2. selects the current waypoint and speed limit;
3. computes world-frame position error and wrapped yaw error;
4. advances the waypoint index only when both errors are within tolerance;
5. rotates the world `XY` error into the robot body frame;
6. scales translation speed with remaining distance and clamps it to the
   scenario limit;
7. clamps angular command to `[-0.35, 0.35] rad/s`.

The body-frame conversion is what lets the same controller follow the route
after arbitrary turns. The camera is absent from this computation.

### `_target_tcp_w`: preserving the carried pose during base rotation

Adding a pi-radian place heading exposed a manipulation bug. The old target
logic forced the original world-frame pick quaternion immediately after
traversal. But the base had just rotated the whole arm and can by pi. Asking IK
to restore the old world quaternion while moving to the central high pose made
the wrist sweep the can through the counter.

The corrected orientation schedule is:

```text
PREGRASP through CARRY_SETTLE  -> authored pick quaternion
TRAVERSE                       -> hold flange pose in the base frame
CARRY_REPOSITION               -> preserve current physical world orientation
CARRY_REORIENT                 -> rotate at the high central pose
PREPLACE through RETREAT       -> authored place-side quaternion
```

`pick_motion` now ends at `CARRY_SETTLE`. `CARRY_REPOSITION` is intentionally
left with the current measured quaternion. Only `CARRY_REORIENT` switches to
the placement quaternion. This divides a risky coupled translation/rotation
into two easier motions, just as the base route does.

### State transitions

When `_route_complete` reports that traversal has settled, the oracle captures
the current TCP command and enters `CARRY_REPOSITION`. Reaching the high central
pose enters `CARRY_REORIENT`; reaching the new orientation enters `PREPLACE`.
This keeps existing collision checks active through all three phases.

## 10. Visualization script, function by function

File: `examples/visualize_genmanip_kitchen_oracle_trial.py`

### Startup and imports

- `cv2` is imported before Isaac Lab to avoid shared-library loading conflicts.
- `_REPO_ROOT`, `_EBENCH_ASSETS`, and `_MDL_PATHS` establish material search
  roots before Kit starts.
- `OMNI_KIT_ACCEPT_EULA` permits headless launch.
- `AppLauncher` consumes both script arguments and Isaac Lab arguments.
- `args_cli.enable_cameras = True` ensures RTX sensors are initialized.
- NumPy, Torch, Isaac Lab, and RoboLab modules are imported after the app is
  launched, following Isaac Lab's required startup order.

### Command-line arguments

- `--seed` fixes reset sampling and the oracle reproduction.
- `--max-steps` bounds the trial.
- `--scenario` selects baseline, remote-robot, or far-side-can geometry.
- `--pick-xy` and `--place-xy` optionally pin task samples.
- `--capture-stride` trades output density against rendering/encoding cost.
- `--sensor-views` enables wrist and Kinect streams in addition to overview.
- the four output arguments independently select video, still, montage, and
  JSON paths.

### `_world_to_panel`

This helper converts kitchen-frame `XY` to pixels in the schematic top-down
panel. It uses fixed room bounds and reverses the image `Y` axis so positive
world `Y` appears upward.

### `_draw_bounds`

It optionally inflates a `Bounds2D`, projects opposing corners, and draws a
rectangle. It is used for physical support regions, target regions, furniture,
and base-clearance envelopes.

### `_phase_group`

The state machine has more phases than a readable video legend needs. This
mapping groups related phases into reset, approach, grasp, lift, traverse,
reposition, pre-place, release, retreat, success, or failure labels.

### `_top_down_panel`

The panel draws the collision/task contract rather than another photorealistic
camera. It shows fixture footprints, inflated clearance, pick/place regions,
scenario routes, staging poses, and live robot/can/TCP positions. This remains
useful when the overview image makes a narrow aisle hard to judge.

### `_annotate`

For overview-only runs, this function overlays phase, step, grasp-assist state,
robot pose, can pose, and target pose. It concatenates the photorealistic RGB
image with `_top_down_panel`.

### `_live_state`

This reads one environment's current phase, robot `XY`, can `XYZ`, TCP `XYZ`,
and assist-joint state. Environment origins are subtracted so the trace reports
kitchen-local coordinates rather than cloned-world coordinates.

### `_save_rgb`

It creates the destination directory, converts RGB to OpenCV's BGR ordering,
and raises if `cv2.imwrite` fails. Silent artifact loss is therefore treated as
a test failure.

### `_can_visibility`

This is a geometry-plus-depth diagnostic, not an observation supplied to the
oracle.

The function:

1. accepts an optional externally composed physical camera pose;
2. subtracts camera position from can position;
3. rotates that vector into the ROS optical frame;
4. rejects points behind the camera;
5. projects with `u = fx*x/z + cx` and `v = fy*y/z + cy`;
6. estimates a pixel rectangle from the known can radius and half-height;
7. compares rendered depth in that patch with expected can depth;
8. declares visibility only when at least four pixels match.

The bounding-depth test is independent of Coke-can color, texture, semantic
labels, and instance IDs. The tolerance combines the can's bounding radius with
a 15 mm margin.

### `_sensor_composite`

The function resizes and concatenates three synchronized 640 by 360 panes:

```text
third-person overview | wrist RGB | Kinect/base RGB
```

It overlays the robot position on the overview and a visibility cross/status
on sensor views. The output is 1920 by 360 pixels.

### `_new_sensor_stats`

This initializes counters for:

- number and shape of frames;
- RGB range and mean;
- temporal frame differences;
- valid depth pixels and depth range;
- projected/in-frame can counts;
- depth-confirmed visibility;
- phase-by-phase visibility.

`previous_rgb` stores only a 160 by 90 downsample, limiting diagnostic memory.

### `_update_sensor_stats`

Each captured frame updates RGB extrema, mean, valid-depth totals, phase
counts, and visibility. The RGB frame is downsampled and compared with the
previous sample. A mean absolute difference above 0.25 counts as a changed
frame. This detects frozen streams without assuming how quickly the robot
moves.

### `_finalize_sensor_stats`

This converts accumulators to JSON-safe metrics and sets `working=True` only
when:

1. more than one frame exists;
2. RGB is not constant;
3. at least one temporal change exists;
4. at least one valid depth pixel exists.

Visibility is reported separately; a working camera can legitimately lose the
can while navigating.

### `_save_montage`

Representative frames are selected for settle, navigation, pregrasp, close,
lift, traverse, preplace, and success. Missing phases are filled from available
frames, the frames are resized, and the contact sheet is saved through
`_save_rgb`.

### `main`: configuration

`main` first validates positive step and stride arguments, fixes the Torch
seed, resolves the scenario, registers the IK environment, and creates one
environment.

`use_fabric=True` is essential for the video. The base controller writes
tensor-side articulation state. Without Fabric, Hydra may continue rendering
the robot at its authored USD transform even while oracle tensors say that the
base moved.

The episode length is expanded to cover `max_steps`. The selected scenario's
base pose initializes the articulation. An optional point pick is validated
against the scenario region before replacing the reset distribution.

Recorders are disabled because this script writes its own purpose-built trace.
If `--sensor-views` is absent, wrist and Kinect sensors are removed from the
scene to avoid unnecessary RTX work.

### `main`: third-person camera

`trial_camera` is a separate, static diagnostic camera under the environment,
not under the robot. Its high south-side view is intended to show the complete
west-north-east route. `set_world_poses_from_view` is called once after scene
creation to aim this overview. This is allowed because it is explicitly a
third-person visualization camera, not a policy sensor.

### `main`: sensor setup

When sensor views are enabled, the script:

1. retrieves `wrist_cam` and `navigation_camera` from the scene;
2. resolves exactly one `bracelet_link` and one `base_link` body index;
3. copies each immutable sensor offset to the simulation device;
4. converts each offset quaternion to OpenGL convention.

Requiring exactly one parent body catches malformed assets early.

### `main`: why diagnostic poses are composed manually

Isaac Lab issue [#745](https://github.com/isaac-sim/IsaacLab/issues/745)
documents a case where a tiled camera renders correctly with a moving parent
while camera pose or point-cloud metadata remains stale. That behavior occurred
here: RGB clearly followed the base, but `camera.data.pos_w` did not reliably
track tensor-driven root motion.

For visibility diagnostics only, the script reads the current parent body's
pose and composes the fixed local offset with `quat_apply` and `quat_mul`. The
result is converted to ROS convention and passed to `_can_visibility`.

This is not camera teleportation. No pose is written to USD, Hydra, or the
sensor object.

### `main`: simulation and capture loop

For each control step:

1. `oracle.compute_action()` produces base, arm, and gripper commands;
2. `env.step(action)` advances RoboLab and PhysX;
3. `oracle.post_step()` updates post-contact state;
4. `_live_state` records the phase and physical state;
5. phase transitions are appended once;
6. regular-stride, transition, and terminal frames are captured;
7. RGB and depth tensors are detached and copied to CPU;
8. parented camera diagnostic poses are composed;
9. visibility and health statistics are updated;
10. the three views are composed and written to MP4;
11. the first frame from every phase is retained for the montage.

An environment termination before oracle completion raises immediately. A
normal oracle terminal state exits the loop.

### `main`: artifacts and cleanup

The last frame and montage are saved first. The JSON result records task
success, seed, scenario, step count, targets, grasp mode, camera mode, camera
metrics, phase transitions, final physical state, and artifact paths.

The `finally` block always releases the video writer and closes the environment.
The outer `finally` closes the Isaac application even if `main` raises.

## 11. Regression tests

### `tests/test_mobile_cobot.py`

`test_mobile_asset_composes_existing_kinova_and_handoff_fixtures` now asserts
that the physical bracket exists while an authored camera, camera link, and
fixed joint do not.

`test_navigation_camera_uses_raised_forward_facing_real_mount` checks:

- the camera requests updated pose metadata;
- its nominal world height exceeds the island surface;
- it has a spawned pinhole model;
- position, quaternion, and convention match the audited mount;
- its prim path is beneath `base_link`;
- the base action config has no attached-camera option;
- robot source contains no `set_world_poses(` call.

`test_policy_cameras_expose_rgb_and_depth` verifies both the inherited wrist
camera and the Kinect proxy expose RGB plus metric image-plane depth.

The existing twist, acceleration, jerk, and gripper tests remain important:
camera correctness depends on predictable parent motion, and carried-object
visibility depends on smooth base commands.

### `tests/test_genmanip_kitchen_physics_schema.py`

`test_all_hard_scenario_base_routes_clear_inflated_fixtures` walks every
pre-pick and carry segment. It checks each waypoint and each connecting segment
against the base-inflated static collision geometry.

`test_place_route_rotates_base_to_face_counter_with_fixed_camera` asserts:

- pick yaw is zero;
- place and retreat yaws are pi;
- each scenario ends its carry route with two equal-`XY` waypoints;
- the first has yaw zero and the second has yaw pi.

This test turns the sensor-facing design into a task-schema invariant.

Focused result:

```text
20 passed
```

## 12. Hard-scenario reproduction

Command:

```bash
PYTHONPATH=/home/robocup/Downloads/RoboLab \
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
examples/visualize_genmanip_kitchen_oracle_trial.py \
  --headless \
  --scenario robot_other_side \
  --sensor-views \
  --seed 101 \
  --max-steps 6000 \
  --capture-stride 4 \
  --output output/genmanip_kitchen/milestone2/robot_other_side_parented_cameras.mp4 \
  --still-output output/genmanip_kitchen/milestone2/robot_other_side_parented_cameras_final.png \
  --montage-output output/genmanip_kitchen/milestone2/robot_other_side_parented_cameras_montage.png \
  --trace-output output/genmanip_kitchen/milestone2/robot_other_side_parented_cameras.json
```

### Phase trace

| Step | Entered phase |
|---:|---|
| 1 | settle |
| 3 | navigate_to_pick |
| 1114 | pregrasp |
| 1153 | descend |
| 1168 | close |
| 1226 | secure |
| 1234 | lift |
| 1257 | carry_settle |
| 1262 | traverse |
| 1970 | carry_reposition |
| 2286 | carry_reorient |
| 2307 | preplace |
| 2754 | place |
| 2892 | handoff_settle |
| 2912 | open |
| 2913 | retreat |
| 3008 | success |

Final state:

```text
robot XY = (1.024244, 0.100632) m
can XYZ  = (1.808775, 0.098276, 0.896318) m
TCP XYZ  = (1.723846, 0.071124, 0.912769) m
grasp constraint active = false
```

### Sensor evidence

| Metric | Wrist camera | Kinect camera |
|---|---:|---:|
| Captured frames | 765 | 765 |
| Changed frames | 754 | 762 |
| Valid depth fraction | 99.81% | 99.93% |
| RGB range | 0–245 | 0–245 |
| Depth range | 0.025–4.560 m | 0.146–5.177 m |
| Diagnostic can-visible frames | 4 | 495 |
| Working | yes | yes |

The Kinect visibility result is reliable and agrees with the montage. The
wrist camera visually shows the can during grasp and transport, but its current
projection diagnostic labels most of those frames out of view. That discrepancy
is discussed below and must not be interpreted as a dead wrist stream.

Artifacts:

```text
output/genmanip_kitchen/milestone2/robot_other_side_parented_cameras.mp4
output/genmanip_kitchen/milestone2/robot_other_side_parented_cameras_final.png
output/genmanip_kitchen/milestone2/robot_other_side_parented_cameras_montage.png
output/genmanip_kitchen/milestone2/robot_other_side_parented_cameras.json
```

## 13. Issues faced, how they were diagnosed, and how they were fixed

### The initial Kinect view could not see over the island

**Symptom:** the island occupied most of the image and hid the can/workspace.

**Cause:** the calibrated horizontal mount location was reasonable, but the
optical center was below or too close to the 1.045 m island top.

**Debugging:** the mount height was compared directly with the audited island
surface constant, and rendered frames were checked at pick and placement.

**Fix:** preserve the horizontal calibration and raise only the sensor's local
`Z` to 0.64 m. The resulting lens height is 1.108 m, 6.3 cm above the island.
The physical mast and bracket were extended to match.

**Why the kitchen was not scaled:** rescaling or lowering the kitchen would
invalidate collision boxes, surface heights, reachability, placement targets,
and the Milestone-0/1 audits.

### The Kinect looked sideways at the workspaces

**Symptom:** when the robot reached the island or counter, the Kinect faced
across the aisle rather than toward the task surface.

**Cause:** the camera's real optical direction is robot `-X`; the earlier base
route treated yaw as only a locomotion variable and left the place heading at
zero.

**Debugging:** first/last Kinect frames were compared with robot yaw and the
known local camera quaternion. The view was internally consistent with the
mount—it was the base heading that was wrong.

**Fix:** keep the mount fixed and make the navigation route end at yaw pi on the
counter side. This is the correct behavior for a real robot and for any visual
policy that must cope with embodiment geometry.

### A manually updated camera appeared to float

**Symptom:** an intermediate version recomputed and wrote a camera world pose
during base control. It could be aimed correctly, but the sensor was logically
detached from the robot and risked synchronization errors.

**Cause:** camera following was implemented as imperative control rather than a
scene-graph relationship.

**Debugging:** source inspection showed sensor-specific state and
`set_world_poses` behavior inside the base action. This also made the action
term responsible for a visual side effect outside its policy semantics.

**Fix:** delete all runtime policy-camera pose writes and spawn the camera under
`base_link`. A regression test searches the source for `set_world_poses(`.

### A rigid camera link and fixed joint did not solve metadata lag

**Symptom:** a temporary 0.01 kg `navigation_camera_link` connected by a fixed
joint moved visually, but `camera.data.pos_w` could still remain stale.

**Cause:** the problem was not an absent PhysX connection. It was the camera
sensor's pose metadata path under Fabric/tensor-driven articulation updates.

**Debugging:** rendered frames changed correctly while the reported camera pose
did not. This matched Isaac Lab issue #745.

**Fix:** remove the extra rigid body and joint. Use the standard direct-child
camera pattern for rendering and derive diagnostic pose from the articulation
body tensor plus immutable mount offset.

### The rendered robot initially appeared stationary

**Symptom:** task tensors and final base coordinates changed, but a rendered
third-person video could show the robot at its authored start pose.

**Cause:** Hydra was not consuming the same tensor-side articulation transforms
used by the ideal base controller.

**Debugging:** numeric root state was compared against overview frames. The
physics trace moved while the render did not.

**Fix:** create the visualization environment with `use_fabric=True` so RTX
rendering consumes the current articulation transforms.

### Turning the base caused the carried can to sweep into the counter

**Symptom:** after adding the correct pi place yaw, the arm/can trajectory could
collide during `CARRY_REPOSITION`.

**Cause:** the arm target immediately forced the original world-frame pick
orientation even though the base had rotated the whole grasp by pi.

**Debugging:** phase traces localized the failure after traversal. Inspecting
the target quaternion schedule showed `CARRY_REPOSITION` was included in the
pick-orientation mask.

**Fix:** preserve the actual post-turn orientation while translating to a high
central pose, then enter the explicit `CARRY_REORIENT` phase. The corrected
trial completed placement successfully.

### Simultaneous placement translation and rotation produced a wide carried arc

**Symptom:** asking the base to approach a rotated final pose could bring the
carried can unnecessarily close to furniture.

**Cause:** a holonomic proportional controller blended `XY` and yaw errors.

**Fix:** add two equal-position waypoints: approach at yaw zero, then rotate in
place to yaw pi. Static tests require this structure for every scenario.

### Camera images moved but reported camera pose stayed stale

**Symptom:** Kinect RGB clearly changed with navigation, but direct pose-based
projection could claim the can was in an impossible location.

**Cause:** `TiledCamera.data.pos_w`/orientation metadata can lag a moving parent
under Fabric even when RTX rendering follows correctly.

**Debugging:** a 300-step moving-base probe compared root motion, consecutive
RGB frames, and direct camera metadata. RGB changed in 31 of 32 samples while
the metadata did not consistently track the parent.

**Fix:** compute diagnostic world pose from `base_link` or `bracelet_link` body
pose and the fixed sensor offset. The camera itself remains purely parented.

### Wrist visibility annotation disagrees with the actual image

**Symptom:** the montage visibly contains a large can in wrist frames while the
red overlay says `CAN OUT OF VIEW` for many of them.

**Likely cause:** the inherited wrist sensor's authored optical convention and
the diagnostic reconstruction are not yet perfectly aligned. The underlying
RGB and depth streams are valid and moving.

**Decision:** do not alter or float the wrist camera merely to make a debug
marker agree. This remains a diagnostic-calibration issue. A future fix should
compare projected robot-frame landmarks or use an explicit
`FrameTransformerCfg` for the wrist optical frame.

### Missing EBench textures produce MDL warnings

**Symptom:** Isaac logs unresolved flattened JPG references and some material
warnings.

**Cause:** the imported EBench scene package does not contain every referenced
texture at the authored path.

**Fix/status:** known material search roots are added before Kit launch. Truly
absent files still use renderer fallbacks. This affects appearance, not camera
parenting, collision, depth validity, or oracle success.

### Low-resolution rendering produces DLSS warnings

**Symptom:** RTX may warn that an internal render dimension is below DLSS's
preferred minimum.

**Cause:** the policy Kinect is intentionally only 640 by 360 and the composed
layout uses resized panes.

**Status:** the produced images and depth are valid. Policy evaluation can
disable DLSS or choose a larger native sensor resolution later if image quality
requires it.

## 14. What the result proves and does not prove

The completed trial proves that:

- a fixed, physically plausible base-mounted Kinect follows the moving robot;
- navigation, not camera teleportation, controls which workspace is visible;
- the remote-start route is collision-audited and executable;
- the robot turns toward the placement counter before manipulation;
- RGB and metric depth are available from both policy cameras;
- the Kinect sees the can for most of the hard trajectory;
- the task still reaches a valid released placement after the new base turn.

It does not yet prove that:

- the wrist projection diagnostic is perfectly calibrated;
- Kinect intrinsics and noise exactly reproduce a particular Azure Kinect
  hardware mode;
- a learned policy can solve the task from pixels;
- the constraint-assisted grasp is equivalent to unassisted Robotiq friction;
- the camera-enabled hard case passes 100 randomized trials.

## 15. Recommended next step

Keep the current physical sensor transforms fixed and build the policy-facing
observation contract:

1. expose synchronized Kinect RGB/depth and wrist RGB/depth with timestamps;
2. define resize, normalization, depth clipping, and invalid-depth conventions;
3. add camera extrinsics/intrinsics to episode metadata, not to oracle actions;
4. fix the wrist diagnostic with a frame-transform sensor or landmark
   calibration test;
5. run a short deterministic multi-seed sensor gate before any 100-trial run;
6. export successful oracle trajectories with `grasp_mode`, scenario, camera
   mount, and task-success metadata;
7. use those trajectories to verify behavior-cloning observation/action
   plumbing before training a larger policy.

The 100-trial gate should only begin after the user explicitly approves it.
