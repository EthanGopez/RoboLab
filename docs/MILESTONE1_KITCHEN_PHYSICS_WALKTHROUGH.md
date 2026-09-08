# Milestone 1 kitchen physics: complete implementation walkthrough

## What Milestone 1 now is

The environment is registered as
`GenManipKitchenCanTransferKinovaMobileJointPosition`. It contains the original
EBench/GenManip kitchen as the rendering layer, a deliberately simple collision
overlay as the physics layer, a dynamic can, and the mobile Kinova. Resetting the
environment samples and drops the can on the island. The base action can move the
robot from the pick staging pose to the existing opposing-counter staging pose.
There is no dining table and there is not yet an arm grasp-and-place controller.

The central separation is:

```text
EBench scene.usd ------------------------------> visual appearance
                                                       |
measured geometry constants -> cuboid overlay --------+--> Isaac/PhysX scene
                                                       |
can dimensions + material -> dynamic cylinder --------+
                                                       |
mobile Kinova config -> base/arm/gripper actions ------+

the same constants -> schema tests + path tests + simulation tests + visualization
```

This makes the visual asset replaceable without silently changing the physics
contract, and makes the robot or policy replaceable without re-authoring the
kitchen.

## Coordinate and timing conventions

- All authored dimensions are metres.
- Geometry constants use the local coordinate frame of one kitchen environment.
- Isaac adds `env.scene.env_origins[i]` when cloning environment `i`.
- Quaternions use Isaac's `(w, x, y, z)` convention.
- The simulation timestep is `1/120 s`.
- An environment action is applied every four physics steps, so policy/control
  frequency is 30 Hz.
- The mobile base action is body-frame `[vx, vy, wz]`.
- The current staging yaws are zero, so the simple trial controller's XY direction
  is identical in body and world coordinates.

## 1. Authoritative geometry

File: `robolab/tasks/mobile_manipulation/genmanip_kitchen_geometry.py`

### Lines 1-12: module contract and imports

- Lines 1-5 state the most important design decision: the detailed kitchen remains
  the rendering source, while simple primitives define the physics contract.
- Line 8 enables postponed evaluation of type annotations. This keeps annotations
  such as `tuple[float, float]` lightweight in this pure-Python module.
- Line 10 imports `dataclass` for immutable geometry records.
- Line 11 imports `hypot`, used to compute Euclidean point-to-box distance without
  adding NumPy or a geometry dependency.

### Lines 14-52: `Bounds2D`

- `@dataclass(frozen=True)` makes a bounds object immutable. A measured region
  cannot be accidentally changed in place by a test or controller.
- `xmin/xmax/ymin/ymax` store a closed, axis-aligned rectangle.
- `center` returns the arithmetic midpoint. Pick and place targets therefore
  derive from their measured regions instead of duplicating coordinates.
- `size` returns width and depth.
- `inflated(radius)` subtracts the radius from lower bounds and adds it to upper
  bounds. This is a Minkowski-sum approximation: instead of moving a circular
  robot around furniture, we expand furniture and treat the base center as a
  point.
- `contains(point, tolerance=0)` performs inclusive bounds checking. Positive
  tolerance is used only where the simulator can produce millimetre-scale contact
  variation.
- `distance_to(point)` clamps the point against each axis. Points inside have
  `dx=dy=0`; points outside return their Euclidean distance to the rectangle.

### Lines 55-75: `CollisionBox`

- Each record has a stable `name`, a center, a full XYZ size, a semantic
  `category`, and a rendering flag.
- `visible=False` means most proxies affect PhysX but do not cover the detailed
  kitchen graphics.
- `footprint` converts center-plus-size into XY bounds. Geometry and navigation
  tests therefore consume the exact boxes used to spawn PhysX shapes.
- `z_bounds` similarly exposes the bottom and top heights for support tests.

### Lines 78-111: measured constants

- `KITCHEN_USD_RELATIVE` points to the checked-in EBench dishwasher scene.
  Keeping the path relative to the package makes it independent of the checkout
  location.
- `ISLAND_SURFACE_Z=1.0450` and `COUNTER_SURFACE_Z=0.8453` are the Milestone-0
  measured support heights.
- `PICK_REGION`, `PLACE_REGION`, and `SINK_OPENING` are conservative valid
  regions, not the full visible surfaces. Conservative bounds avoid edges,
  backsplash geometry, and the sink cutout.
- `ROBOT_BASE_RADIUS=0.30` approximates the mobile base footprint.
- `BASE_SAFETY_MARGIN=0.05` adds 5 cm of clearance.
- `BASE_CLEARANCE_INFLATION=0.35` is the resulting planning radius.
- `ROBOT_BASE_Z=0.468` is the measured offset from the drive-base bottom to the
  articulation root.
- `STAGING_POSES` stores `(x, y, z, yaw)` for pick and place. Both face along
  the kitchen's authored zero-yaw direction.
- `CAN_RADIUS` and `CAN_HEIGHT` are the measured YCB can envelope.
- `CAN_HALF_HEIGHT` is derived rather than copied, preventing inconsistent rest
  heights.
- `CAN_MASS_KG=0.40` is plausible for a full soup can and is explicit so policy
  behavior is reproducible.
- Static/dynamic friction of 0.60/0.50 gives stable countertop contact without
  making the object unrealistically glued down.
- Restitution 0.02 prevents a visibly bouncy food can.
- Contact offset 2 mm gives PhysX a small contact-generation buffer; rest offset
  zero means the intended resting surfaces still coincide.
- The can starts 2 cm above its correct resting center. That is high enough to
  prove gravity and collision work, but low enough to avoid an artificial impact.

### Lines 113-139: collision overlay boxes

- The floor is a 10 cm thick slab whose top is exactly at `z=0`.
- Back, left, and right walls close the room while leaving the camera-facing side
  open. Walls are visible because the source kitchen has no complete room shell.
- The island lower cabinet is one stable box instead of hundreds of decorative
  triangles.
- Four thin island-top boxes surround the measured sink opening. A single top box
  was intentionally rejected because it would create invisible collision across
  the sink.
- The sink has a bottom and four thin walls. These provide usable collision if a
  future object is dropped into it.
- The opposing counter is represented by right-run and back-run cabinet/top pairs.
  The current place region is supported by `counter_right_top`.
- There is deliberately no dining-table proxy.

### Lines 141-218: navigation geometry

- `STATIC_COLLIDER_NAMES` gives tests and tools stable names.
- `NAVIGATION_OBSTACLE_CATEGORIES` documents which categories matter to base
  motion; the floor is intentionally absent.
- `_segment_intersects_box_interior` is the slab-form line/AABB clipping test.
  It tracks the range of segment parameter `t` for which X and Y are inside the
  rectangle.
- The tolerance shrinks the tested interior slightly. Merely touching an inflated
  boundary is allowed; entering it is not.
- A nearly parallel segment is rejected only when its fixed coordinate lies
  strictly inside that axis interval.
- `navigation_obstacle_footprints` returns one outer footprint per physical
  fixture rather than every top/basin constituent. This prevents double counting
  and matches what a base planner needs.
- `base_pose_is_clear` inflates every footprint by 0.35 m and verifies that the
  base center is outside all of them.
- `base_path_is_clear` applies the exact segment test to every inflated
  footprint. This validates the complete straight path, not a few sampled points.

## 2. Physics task and reset behavior

File: `robolab/tasks/mobile_manipulation/kitchen_can_transfer_task.py`

### Lines 1-35: imports

- `copy` prevents the task-specific robot start pose from mutating the shared
  Kinova config.
- `os` resolves the package-relative kitchen asset.
- Standard `dataclass` is used for the RoboLab task description.
- Isaac Lab MDP, simulation, asset, manager, and `configclass` imports supply the
  scene/event/config types.
- `torch` is needed only for batched reset sampling and simulator writes.
- `PACKAGE_DIR` anchors the visual USD.
- `Task` is RoboLab's task metadata/config-generation contract.
- The mobile Kinova config is imported independently of the task, preserving robot
  modularity.
- All measurements are imported from the pure geometry module; this file contains
  no second copy of a dimension.

### Lines 38-52: physics materials

- `STATIC_MATERIAL` applies 0.70/0.60 friction to floor and furniture.
- `friction_combine_mode="max"` keeps counter contact stable when paired with the
  can material.
- Restitution is zero for immobile fixtures.
- `CAN_MATERIAL` takes every value from the authoritative constants.
- `restitution_combine_mode="min"` prevents another material from making the can
  bounce more than intended.

### Lines 55-81: `_static_box_cfg`

- The function converts one pure `CollisionBox` record into an Isaac
  `AssetBaseCfg`.
- Only boxes marked visible receive a brown-gray rough preview material.
- Every proxy is spawned directly under `{ENV_REGEX_NS}`. Isaac shape spawners
  cannot clone below a regex parent that does not already exist, which is why a
  nonexistent intermediate `/collision` scope was not used.
- `CuboidCfg(size=...)` creates simple, watertight contact geometry.
- `collision_enabled=True` is explicit.
- The same 2 mm contact and zero rest offsets are used consistently.
- No `rigid_props` and no `mass_props` are supplied. In USD/PhysX this makes
  the shape a static collider rather than dynamic furniture.
- `init_state.pos` places the cuboid at the measured center.

### Lines 84-136: `make_physics_can_cfg`

- A `RigidObjectCfg`, rather than an `AssetBaseCfg`, exposes runtime root pose
  and velocity buffers and supports batched state writes.
- The collision and temporary visual are a Z-axis cylinder with the measured
  envelope. This is more stable than using the detailed can render mesh directly.
- A semantic tag identifies it as a can for cameras/datasets.
- Rigid body is enabled, it is non-kinematic, and gravity remains enabled.
- Small damping removes unrealistic endless sliding/spinning without masking a
  bad contact.
- Linear/angular velocity caps protect the simulation if a contact explodes.
- Maximum depenetration velocity is limited to 1 m/s, reducing launch behavior if
  a reset begins with tiny overlap.
- Gyroscopic forces and elevated solver iterations improve cylindrical contact
  stability.
- Sleep/stabilization thresholds let a settled can become genuinely quiet.
- Explicit mass, collision, material, torsional friction radii, and contact sensor
  activation make grasp/contact behavior inspectable later.
- The red metallic preview clearly distinguishes the physics wrapper from the
  imported kitchen.
- Initial position uses the pick-region center and derived drop height.
- Identity quaternion and zero velocities produce deterministic initial state.

### Lines 139-160: scene composition

- `GenManipKitchenPhysicsSceneCfg` contains the original kitchen visual and the
  dynamic can.
- The kitchen visual is referenced unchanged at authored scale.
- The loop converts every collision-box record into a config field on the scene
  class. This avoids a manually maintained second list while still giving Isaac
  one named scene entity per proxy.

### Lines 163-169: task-specific robot

- `KitchenKinovaMobileCobotCfg` subclasses the reusable mobile robot config.
- The robot asset is deep-copied before changing its initial state.
- Its initial XYZ comes from the verified pick staging pose.
- Identity orientation matches the zero-yaw staging contract.

### Lines 172-190: `reset_can_on_island`

- Isaac calls this function with a batch of environment IDs.
- The scene entity name defaults to `can`, making the event configurable.
- Default root states are cloned so tensor shape, dtype, and device remain correct.
- `torch.rand(count, 2)` creates independent samples for every cloned environment.
- Each unit sample is affine-mapped into the measured pick rectangle.
- XY and Z are translated by each environment origin. Without this step, all
  cloned cans would reset into environment zero.
- Quaternion is reset to identity and all six root velocities are zeroed.
- Pose and velocity are written explicitly to PhysX.

### Lines 193-214: events, termination, and task

- Reset first restores the whole scene to defaults, then samples the can. Event
  declaration order is intentional.
- The only current termination is the 120-second timeout; task-success predicates
  belong to the next milestone.
- The task class tells RoboLab which scene, events, and termination configs to use.
- `contact_object_list=["can"]` makes the can available to RoboLab contact logic.
- The instruction names the existing opposing counter, not a dining table.

## 3. RoboLab registration

Files:

- `robolab/registrations/kinova_mobile/auto_env_registrations_jointpos.py`
- `robolab/registrations/kinova_mobile/__init__.py`

### Registration lines 1-46

- `ENV_NAME` is the stable Gym/RoboLab identifier.
- Imports inside the registration function avoid importing Isaac-heavy task
  modules merely by importing the package.
- `generate_env_cfg_from_task` converts task metadata into a
  `ManagerBasedRLEnvCfg` and registers the Gym entry point as
  `robolab.core.environments.env:RobolabEnv`.
- Kinova proprioception is the initial observation group. Task cameras are not
  forced into the policy interface.
- `task_path` lets RoboLab load the task class using its normal task-discovery
  path.
- `KitchenKinovaMobileCobotCfg` selects the mobile robot at the pick stage.
- `KinovaMobileJointPositionActionCfg` selects the combined base, arm, and
  gripper action interface.
- The existing contact-gripper implementation is reused.
- Camera, lighting, and background overrides are `None` because the imported
  scene and robot own their assets; the visualization script adds a camera only
  to its local config instance.
- `dt=1/120`, `decimation=4`, and `render_interval=4` produce 120 Hz physics
  and 30 Hz control/rendering.
- The deterministic seed is zero, the default is one environment, and 8 m spacing
  prevents cloned kitchens from overlapping.
- `__init__.py` re-exports only the environment name and registration function.

## 4. Mobile Kinova traversal stack

Files:

- `robolab/robots/mobile_base_math.py`
- `robolab/robots/kinova_mobile_cobot.py`

### `body_twist_to_world`

- The function accepts batched body-frame `[vx, vy, wz]` and root quaternions.
- Quaternion components are unpacked in `wxyz` order.
- The two quaternion expressions compute sine and cosine of yaw without converting
  the full orientation to Euler angles.
- Body X/Y velocity is rotated into world X/Y.
- A six-element PhysX twist is returned with planar velocity in indices 0/1 and
  yaw rate in index 5.

### `slew_rate_limit`

- The difference between target and current commands is clamped componentwise.
- Adding the clamped delta implements a deterministic acceleration limit and
  prevents instantaneous velocity jumps.

### `HolonomicBaseVelocityAction`

- Constructor buffers raw, processed, and applied commands for every environment.
- Speed limits and per-physics-step acceleration deltas are stored on the simulator
  device.
- `process_actions` records the policy command and clamps it to configured speed
  limits.
- `apply_actions` acceleration-limits the command.
- The floating root is projected back onto planar SE(2): current yaw is retained,
  roll/pitch are removed, and Z is fixed to the measured 0.468 m root height.
- The resulting body command is rotated into world coordinates and written as the
  root velocity.
- `reset` zeros all three command buffers so momentum from a previous episode is
  not inherited.
- Default limits are 0.6 m/s, 1 rad/s, 1.2 m/s², and 2 rad/s².

This controller is intentionally an ideal holonomic base interface. It is suitable
for policy-agnostic task development, but it is not yet a wheel-level drivetrain
model or a navigation planner.

### Combined action config

- `base` occupies three values: `vx, vy, wz`.
- `arm` occupies seven absolute Kinova joint-position values.
- `gripper` occupies one binary open/close value.
- Tests discover action slices by active term names rather than assuming hard-coded
  offsets.

## 5. Static tests

File: `tests/test_genmanip_kitchen_physics_schema.py`

- Imports deliberately include both Isaac config types and the pure geometry
  contract.
- `test_all_overlay_fixtures_are_static_colliders` checks every source box has a
  scene config, collision enabled, no rigid body, no mass, and positive dimensions.
- `test_can_wrapper_has_explicit_dynamic_physics` checks the can is dynamic,
  gravity-enabled, collision-enabled, and matches mass, dimensions, and friction
  constants.
- `test_sink_opening_is_not_filled_by_a_worktop_slab` protects against the
  invisible-sink-collision regression.
- `test_pick_and_place_centers_have_physical_support` ties task regions to their
  actual support proxies.
- `test_staging_poses_and_full_path_clear_inflated_fixtures` proves the 5 cm
  safety margin is included and checks both endpoints plus the whole segment.
- `test_environment_registers_as_robolab_env` checks the stable name, Gym entry
  point, and generated config class.

## 6. Isaac simulation tests

File: `tests/test_genmanip_kitchen_physics_sim.py`

### Helpers and fixture

- `_matrix_values` flattens a USD transform matrix so pytest can compare it.
- `_action_slices` derives slices from Isaac's active action terms.
- `_hold_action` initializes a zero action, fills the arm slice with the default
  joint pose, and leaves base/gripper neutral.
- `_step` advances the registered environment and fails immediately on unexpected
  termination or truncation.
- `_set_robot_stage` builds the root pose in `wxyz`, adds the clone origin,
  zeros root velocity, and restores default joint state.
- The module fixture creates the real registered environment on CUDA with
  `use_fabric=False`. Disabling Fabric lets the tests inspect authoritative live
  USD schemas and transforms.

### Runtime assertions

- The can-drop test simulates 90 control steps, or three seconds.
- It records minimum and maximum Z to catch both falling through and flying.
- Final XY, final Z, linear speed, and angular speed prove the can is supported and
  settled.
- The live-overlay test walks each named proxy's USD subtree because Isaac applies
  `CollisionAPI` to a generated geometry child, not the wrapper Xform.
- It checks collision is enabled, no descendant has `RigidBodyAPI`, and every
  proxy transform remains unchanged after simulation.
- The parameterized stage test runs independently at pick and place and checks
  finite state, clearance, XY error below 1 cm, and Z error below 3 mm.
- The traversal test first proves the geometric segment is clear.
- A bounded proportional command uses
  `speed=min(0.25, 1.2*distance)`.
- Ten seconds are allowed because the action term ramps velocity at physics-step
  acceleration limits.
- Every step checks finite state, planar height, and inflated collision clearance.
- Success requires entering a 2 cm goal radius and ending within 3 cm.
- The can must remain on the island throughout traversal.

## 7. The 100-trial acceptance validator

File: `examples/validate_genmanip_kitchen_physics.py`

### Launch and arguments

- The app is launched before importing Torch/Isaac runtime modules.
- Defaults encode the acceptance criterion: 100 trials, ten parallel cloned
  environments, and 90 settle steps.
- The report defaults to
  `output/genmanip_kitchen/milestone1/reset_and_settle.json`.

### Batch loop

- Positive CLI values are enforced before creating Isaac state.
- Batch size is capped by the number of requested trials.
- The real registered environment is used with 8 m clone spacing.
- Each batch receives a deterministic but different Torch seed.
- `reset_eval_state()` clears RoboLab's frozen-evaluation bookkeeping before an
  explicit reset.
- Per-environment minimum Z, maximum Z, finite-state, and early-termination buffers
  are accumulated through all 90 steps.
- Final can pose/speed and robot pose are then checked per trial.
- Failures are structured records containing trial number, check name, and value.
- The checks cover finite state, premature termination, pick-region containment,
  rest height, penetration, flying, residual linear/angular speed, and robot root
  height.
- Cleanup is in `finally`, so the environment closes on either pass or failure.
- JSON output is written after simulation, and the process returns nonzero if any
  check failed.

The completed report is 100 trials, zero failures.

## 8. Trial visualization

File: `examples/visualize_genmanip_kitchen_physics_trial.py`

### Why it is separate

The registered environment deliberately does not require a third-person camera or
OpenCV annotations. The visualizer instantiates the registered config and adds a
camera only to that local config. A policy therefore sees the same task and action
interface whether or not a demonstration is being recorded.

### Launch and camera

- EBench shared and scene-local MDL directories are put on `MDL_SYSTEM_PATH`
  before Kit starts.
- CLI arguments select MP4 and still-image paths.
- Camera support is enabled explicitly.
- The camera is 1280x720 with an 18 mm focal length and looks from the open side of
  the room toward the island/counter aisle.
- The recording enables Fabric so Isaac Sim 5.0 synchronizes moving articulation
  transforms into the RTX camera. Physics schema tests separately use
  `use_fabric=False` because they need authoritative live USD API inspection.

### Action and map helpers

- `_action_slices` and `_hold_action` use the same term-name discovery as the
  tests.
- `_world_to_panel` maps authored kitchen XY coordinates into pixels and flips Y
  because image coordinates grow downward.
- `_draw_bounds` draws either raw or inflated rectangles.
- `_top_down_panel` displays fixture collision footprints, 0.35 m inflated
  clearance boundaries, pick/place regions, the planned segment, robot footprint,
  and can position.
- The panel is rendered from the same constants as PhysX and the tests; it is not a
  hand-drawn approximation.
- `_annotate` adds phase and live pose telemetry to the Isaac RGB frame, then
  concatenates the map to its right.

### Trial loop

- The first 90 frames hold the robot while the can falls under gravity and settles.
- Traversal then computes the vector from current base position to the place stage.
- Speed is proportional to distance and capped at 0.40 m/s for a clearly visible
  demonstration, below the registered action term's 0.60 m/s limit.
- The command is written only to the base's linear slice; arm remains at home.
- Each frame fails if the task terminates, the robot enters an inflated obstacle,
  or the settled can leaves its support height.
- Once inside the 2 cm arrival radius, 45 stopped frames show the final state.
- The MP4 is streamed through H.264 rather than retaining all frames in memory.
- Final success again requires less than 3 cm position error.

The corrected rendered artifact is 330 frames at 30 FPS, 1710x720, lasting 11 seconds.
The extra 430 pixels contain the live collision/clearance map.

## 9. What a single trial proves

| Phase | Command | Physical evidence |
|---|---|---|
| Reset | Scene defaults, randomized can XY, zero velocities | No stale episode state |
| Drop | Base stopped; gravity enabled | Can contacts island instead of falling through |
| Settle | Arm held at home | Can reaches expected center Z and low velocity |
| Traverse | Bounded `vx, vy`; zero yaw | Base follows the audited clear segment |
| Arrival | Base command returns to zero | Acceleration limiter decelerates without collision |
| Final check | No mutation | Robot is within 3 cm and can remains supported |

The visualization is an execution of the registered environment, not an animation
of pre-authored transforms.

## 10. Known limitations

- The can currently uses a stable cylinder as both collision and temporary visual.
  The detailed YCB appearance can later be attached as a collision-free child.
- The mobile base controller is ideal planar velocity control, not wheel dynamics.
- The straight segment is sufficient for these two staging poses; general
  obstacle-aware navigation is not implemented.
- There is no grasp, lift, carry, place, or success predicate yet.
- Some EBench materials reference missing flattened textures and a few meshes have
  malformed normal buffers. These are source-asset rendering issues, not collision
  or dynamics failures.

## 11. Next step: Milestone 2 manipulation oracle

The right next step is a deterministic scripted oracle before training or attaching
a learned policy:

1. Add task predicates:
   - can supported by island;
   - can grasped by the gripper;
   - can lifted clear of the island;
   - base at place stage;
   - can supported inside the counter placement region;
   - gripper released and robot collision-free.
2. Add end-effector pose actions using differential IK while retaining the same
   base action term.
3. Implement a state machine:
   - pre-grasp;
   - descend;
   - close and confirm bilateral/contact grasp;
   - lift;
   - traverse while carrying;
   - pre-place;
   - descend;
   - open;
   - retreat.
4. Add swept collision checks for the arm and carried can.
5. Replace the temporary can visual with the detailed YCB visual as a non-colliding
   child of the same rigid-body wrapper.
6. Run a second 100-trial gate over randomized pick/place samples.
7. Only after the oracle is reliable, expose observations/actions to candidate
   policies and benchmark policy success against the oracle.

The exit criterion for that milestone should be 100 successful scripted
pick-carry-place trials with no dropped can, fixture penetration, self-collision,
or false-positive success.
