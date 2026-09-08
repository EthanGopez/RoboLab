# XGRIDS2 lab physics-authoring walkthrough

## Outcome

The XGRIDS2 scan is now a registered route-first `RobolabEnv`, not only a
renderer.  The implementation deliberately separates three contracts:

1. `xgrids2_lab_zup.usda` supplies appearance only.
2. A clean floor slab and 229 transformed LabApt vector-map segments supply
   static PhysX collision.
3. A robot-specific transport radius is passed to robot-independent navigation
   geometry and a deterministic waypoint follower.

This is sufficient to spawn the mobile Kinova at the dining/snack staging
poses and traverse between them.  It is not yet a manipulation environment:
the scan has no semantically authored tabletop, object-support, or grasp
collision surfaces.

## Gate 1: independent static collision overlay

`robolab/tasks/mobile_manipulation/xgrids2_lab_geometry.py` is the pure geometry
contract.  It loads `LabApt.vectormap.txt`, applies the audited rigid transform

```text
p_xgrids = R(35.617718 deg) p_labapt + (-13.259476, 6.588388)
```

and retains the 229 segments intersecting the route-first region
`x=[-1, 7]`, `y=[-4, 2]` metres.  Each segment becomes a hidden, static PhysX
cuboid with 6 cm width and 1.5 m height.  Static means collision is enabled but
neither rigid-body nor mass APIs are authored.

The visual XGRIDS USD unexpectedly contains `PhysicsCollisionAPI` on its neural
reconstruction mesh.  The scene therefore explicitly disables collision on
the complete visual subtree.  Without that override, clean probes collide with
noisy reconstruction triangles and roll/fall despite the floor slab.

Visualization:

- `output/xgrids2_lab/physics_gates/gate1_collision_overlay.png`

## Gate 2: clean floor and gravity support

The floor is an 8 m by 6 m by 0.10 m static cuboid centered at `z=-0.05 m`, so
its top is exactly `z=0`.  It has zero restitution and a high-friction static
material.

The live floor test drops nine independent 6 cm-radius dynamic probes along
the planned route.  All nine settle with their centers at approximately
`z=0.060 m`, negligible velocity, and no lateral escape.

Artifacts:

- `output/xgrids2_lab/physics_gates/gate2_floor_drop.png`
- `output/xgrids2_lab/physics_gates/gate2_floor_drop_summary.png`
- `output/xgrids2_lab/physics_gates/gate2_floor_drop_results.json`

## Gate 3: navigable-space inflation

The base collider alone has a 0.30 m radius, but the first live spawn exposed
why base-only inflation is unsafe: the old arm home pose extended the
fingertips 0.564 m from the base center and intersected a nearby proxy.

`examples/audit_mobile_kinova_transport_postures.py` measures every link-center
radius for candidate transport postures.  The selected, joint-limit-safe fold

```text
[joint_1 ... joint_7] = [0, -0.70, pi, -2.50, 0, 0.70, pi/2]
```

has a maximum link-center radius of 0.305 m.  Navigation rounds that up to a
0.36 m whole-robot radius, then adds a 0.05 m safety margin and the 0.03 m proxy
half-width:

```text
required centerline clearance = 0.36 + 0.05 + 0.03 = 0.44 m
```

The occupancy grid resolution is 0.08 m.  A* uses eight-connected moves and
the final route is resampled every 2 cm for its clearance assertion.  The raw
snack pose was only 0.269 m from an obstacle, so it moved 0.220 m to
`(4.56, -0.76)` while preserving its surveyed heading.  The dining pose remains
`(1.7656, -1.7639)`.

Visualization:

- `output/xgrids2_lab/physics_gates/gate3_inflated_navigation.png`

## Gate 4: registered robot spawning

`xgrids2_lab_navigation_task.py` defines the scene, reset event, timeout, robot
initial state, static overlay, and base contact sensor.

`auto_env_registrations_xgrids2_lab.py` registers
`XGRIDS2LabNavigationKinovaMobileJointPosition` through RoboLab's normal
environment generator.  The action contract remains the reusable mobile
Kinova contract:

```text
base: [vx_body, vy_body, wz]
arm:  seven absolute joint positions
gripper: binary open/close
```

The live gate teleports to each authored staging pose, restores the compact
transport posture, runs 45 simulation steps, and requires:

- less than 1 cm planar drift;
- base-link height within 4 mm of 0.468 m;
- linear speed below 0.02 m/s;
- valid depth from the base-parented navigation and wrist cameras.

Visualization:

- `output/xgrids2_lab/physics_gates/gate4_robot_spawn_montage.png`

The montage pairs a scan-supported third-person audit view with the actual
camera parented beneath `robot/base_link`.  The audit camera is not an
observation or policy sensor.

## Gate 5: closed-loop traversal

`xgrids2_lab_navigation_oracle.py` is deliberately small and robot-agnostic.
It accepts a world-frame route, a fixed sensor-forward yaw offset, and speed /
tolerance parameters.  On each control step it:

1. advances past reached A* waypoints;
2. computes a bounded world-frame velocity toward the next waypoint;
3. rotates that velocity into the current base frame;
4. rotates the base so the fixed camera faces along the route;
5. rotates to the authored staging yaw at the final point.

It never rewrites the camera pose.  Camera motion comes only from USD parenting
and robot base motion.

The registered base action applies acceleration and jerk limits before
integrating the body twist.  Each live step checks the 0.44 m geometry contract.
A `ContactSensor` also monitors net force on `base_link`.  GPU PhysX cannot
filter a dynamic body separately against hundreds of static roots, so the gate
uses a supported physical criterion: floor support/friction is accepted only
while total base force remains inside a 45-degree vertical cone.  A wall or
furniture impact produces positive `horizontal_force - abs(vertical_force)`.

The recorded trial passed with:

| Metric | Result |
|---|---:|
| Route points | 38 |
| Control steps | 629 |
| Required line clearance | 0.440 m |
| Minimum observed line clearance | 0.505 m |
| Maximum nonvertical force excess | 0.000 N |
| Final position error | 0.0347 m |
| Final yaw error | 0.0380 rad |

Artifacts:

- `output/xgrids2_lab/physics_gates/gate5_base_traversal.mp4`
- `output/xgrids2_lab/physics_gates/gate5_base_traversal_keyframes.png`
- `output/xgrids2_lab/physics_gates/gate5_base_traversal_final.png`
- `output/xgrids2_lab/physics_gates/gate5_base_traversal_results.json`

## Automated tests

Static/schema tests:

```bash
/home/robocup/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/test_xgrids2_lab_physics_schema.py
```

These ten tests cover map/schema constants, exact floor height, staging-pose
correction, footprint inflation, route clearance, task validation,
registration, world-to-body conversion, and fixed-camera yaw behavior.
Final result: **10 passed**.

Live Isaac tests:

```bash
ACCEPT_EULA=Y /home/robocup/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/test_xgrids2_lab_physics_sim.py
```

They cover the authored USD APIs, disabled scan collision, both spawn/settle
poses, parented camera depth, geometric traversal clearance, contact-force
cone, final pose, and floor height.
Final result: **5 passed, 0 failures, 0 errors, 0 skipped** (464.32 seconds).

## Problems found and resolved

- **Scan mesh competed with the clean floor.** Explicitly disabled collision
  on the visual USD subtree.
- **Old arm home pose exceeded the planned footprint.** Measured link envelopes,
  chose a compact transport posture, and planned for the whole robot.
- **An initially compact pose violated joint 4 limits.** Rejected it at the
  live articulation-config gate and selected the 0.305 m limit-safe fold.
- **Dynamic task loading failed on postponed annotations.** Kept the task file
  compatible with RoboLab's current loader rather than changing unrelated
  loader behavior.
- **A temporary class-level `yaw` scalar was interpreted as a scene asset.**
  Inlined the quaternion expression so every public scene field is a valid
  asset/sensor config.
- **GPU static contact filters are unsupported.** Replaced the 229-target
  filter with a net-force vertical-cone test plus exact geometric clearance.
- **The workstation had insufficient free VRAM for three audit cameras.** The
  visualization-only config omits the unused wrist stream and lowers audit
  resolution; the registered environment retains both policy cameras.
- **Novel scan viewpoints can become dark or blurred.** Third-person audit
  views use the exact validated View 01/View 05 capture poses.  Dark mid-route
  base-camera frames are preserved because they reveal a real coverage risk
  for visual policies.

## Current boundary and next gate

This route-first physics layer is valid for base spawning and traversal.  It
does not make the reconstructed tables manipulation-ready.  The next gate is
semantic fixture authoring: add clean dining/snack tabletop support proxies,
measure their pick/place regions, drop a dynamic object on each, and validate
the base camera's object visibility at the staging headings.  Only then should
the manipulation oracle be transferred from the kitchen environment.
