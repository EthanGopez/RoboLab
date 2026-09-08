# XGRIDS2 lab integration handoff

> This document preserves the original integration and debugging history. For
> maintained commands and module ownership, use
> [XGRIDS2_CODEBASE_GUIDE.md](XGRIDS2_CODEBASE_GUIDE.md).

**Last updated:** 2026-08-27

**Repository:** `/home/robocup/Downloads/RoboLab`
**Current scope:** production vendor NuRec rendering, XGRIDS physics/navigation,
and a completed CuRobo-assisted dining-to-snack can-transfer baseline.

## Completion addendum (supersedes the earlier status below)

The new vendor package received after the meeting is now the production visual source:

```text
assets/scenes/xgrids2_lab/new_data_0827/isaacsim-import-package/
  ut_lab_1_scene_pack/0/ut_lab_1.usdz
```

This USDZ is sufficient because it is the vendor-authored package entry point: it resolves the NuRec payload, proxy relationship, and render metadata as one asset. RoboLab adds physics separately; the 68 GB raw capture is not required to run the reconstruction. The earlier `forward_compatible` wrapper is historical and is no longer the runtime visual source.

The flashing problem is resolved by reproducing the vendor rendering contract: Fabric scene delegate, single GPU, registered compositing globally and on every render product, inverse tone mapping, inverse color correction, synchronous rendering, and RGB-only navigation/wrist/audit sensors. The visual uses the audited `+1.075 m` Z offset. Physics remains an independent floor/proxy layer.

The completed manipulation sequence is:

```text
pregrasp -> approach -> grasp assist -> lift -> CuRobo carry fold
         -> 0.40 m inflated A* route -> preplace -> place
         -> release -> physical settle -> success
```

This is deliberately an **assisted planner oracle**, not yet a fully dynamic grasp controller:

- CuRobo collision-checks all arm trajectories and the compact carry fold.
- The arm replays CuRobo joint waypoints exactly while fixed-base staging is enforced.
- The sampled can is pinned until the reach-verified grasp-assist boundary.
- A runtime fixed joint represents retention because the supplied Robotiq mimic drives become unstable during bilateral closure.
- The integrated gate replays the independently audited A* route with a 0.40 m carried footprint; the wheel controller remains verified by the standalone navigation gate.
- Release removes the assist, clears arm overlap, initializes the can at the CuRobo-validated pose, and then requires free PhysX settling on the snack support.

Passing artifacts:

```text
output/xgrids2_lab/manipulation_gate/results.json
output/xgrids2_lab/manipulation_gate/pick_place_montage.png
```

Run with the Isaac environment CUDA 12.8 compiler, required for RTX 5080 / `sm_120` CuRobo extensions:

```bash
TERM=xterm \
PATH=/home/robocup/miniconda3/envs/env_isaaclab/bin:/usr/bin:/bin \
CUDA_HOME=/home/robocup/miniconda3/envs/env_isaaclab \
TORCH_CUDA_ARCH_LIST=12.0 ACCEPT_EULA=Y \
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/validate_xgrids2_lab_manipulation.py --headless
```

The next milestone is embodiment calibration: identify stable Robotiq mimic-joint contact behavior and mobile-arm tracking gains, then replace reach assistance and exact waypoint/route replay with measured bilateral grasp and fully dynamic controller execution.

## 1. Historical executive status

> The remainder of this document records the earlier route-first investigation.
> Statements about active assets, blockers, or unfinished manipulation are superseded by the completion addendum above.

The XGRIDS2 lab scan is no longer only a collection of visual files. It is
referenced through a floor-corrected Z-up USD wrapper, registered against the
existing LabApt navigation frame, and used by a registered RoboLab environment
with an independent PhysX floor, static collision proxies, collision-aware
staging poses, a mobile Kinova, and a validated dining-to-snack-table base
route.

The physics/navigation result is successful:

- the scan floor is aligned to `Z=0`;
- the LabApt-to-XGRIDS planar transform has been measured and audited;
- nine floor-drop probes settle correctly;
- the mobile Kinova settles at both staging poses;
- a 38-point route completes in 629 control steps;
- minimum measured line clearance is 0.505 m against a 0.440 m requirement;
- final position error is 0.0347 m and final yaw error is 0.0380 rad;
- static tests were rerun on 2026-08-27 and passed 10/10; the last live Isaac
  run passed 5/5.

The remaining blocking issue is visual, not physical: the supplied NuRec
reconstruction renders deterministically in a small two-camera scene, but in
the loaded robot task on the local Isaac Sim 5.1 runtime, camera products
intermittently alternate between the neural composite and the gray proxy or an
incomplete composite. A stationary third-person camera also flashes, which
rules out ordinary camera motion or scan coverage as the immediate cause.

The current direction is **not** to require a textured mesh prematurely. The
XGRIDS team reports that their native splat pipeline supports the intended
rendering. The immediate next step is to reproduce their exact Isaac
Sim/Omniverse NuRec configuration, version, extensions, initialization order,
and camera capture loop. The independently authored collision layer remains
necessary regardless of whether the visual layer is splats or a mesh.

## 2. What we were trying to build

The long-term task is a physical, policy-ready lab environment in which a
mobile manipulator can:

1. spawn at known semantic staging locations;
2. travel through collision-free lab space;
3. observe the scene through a camera rigidly mounted to the base and,
   eventually, a wrist camera;
4. approach semantically authored work surfaces;
5. pick an object at one fixture and place it at another;
6. expose stable RGB/depth observations for scripted and learned policies.

This handoff covers the route-first foundation. It does **not** claim that the
dining or snack table is manipulation-ready yet. Clean support surfaces,
pick/place regions, task objects, reachability, and manipulation predicates
still need to be authored.

## 3. Supplied XGRIDS2 data

The previous source asset was:

```text
assets/scenes/xgrids2_lab/forward_compatible/usdz-result/mesh-model.usdz
```

It is 849,480,392 bytes. Its embedded archive contents are:

| Embedded file | Size | Role |
|---|---:|---|
| `model.nurec` | 841,372,190 bytes | NVIDIA neural reconstruction |
| `mesh.ply` | 4,997,698 bytes | NuRec proxy geometry |
| `mesh.usd` | 3,106,832 bytes | USD layer for the proxy |
| `gauss.usda` | 2,338 bytes | NuRec volume and proxy relationship |
| `default.usda` | 834 bytes | package entry point and render settings |

`gauss.usda` declares an `OmniNuRecFieldAsset`, points its density and
emissive-color fields at `model.nurec`, marks the volume with
`omni:nurec:isNuRecVolume = true`, and relates the volume to
`/World/mesh` as its proxy.

Other supplied assets are:

- `forward_compatible/usdz-result/model.usdz` (841,375,585 bytes);
- `forward_compatible/mesh-files/UT_Robtics_lab_int.obj`;
- the duplicate `mesh-files/UT_Robtics_lab_int.obj`;
- `usd-result/UT_Robtics_lab_int.usd`, a 399 MB binary USD crate;
- Gaussian PLY files under
  `ply-result/point_cloud/iteration_100/`;
- `ply-result/cameras.json`;
- `ply-result/cfg_args`.

### 3.1 What the PLY contains

The primary `point_cloud.ply` is 484,477,377 bytes and contains 7,124,654
binary little-endian vertices. Each entry has position, normal placeholders,
DC color, opacity, anisotropic scale, and quaternion rotation:

```text
x y z
nx ny nz
f_dc_0 f_dc_1 f_dc_2
opacity
scale_0 scale_1 scale_2
rot_0 rot_1 rot_2 rot_3
```

It does not contain high-order spherical-harmonic fields. Density here means
many Gaussian primitives; it does not mean millions of conventional textured
mesh vertices.

### 3.2 What the OBJ contains

`UT_Robtics_lab_int.obj` is about 9.6 MB and contains:

- 141,772 vertex-position records;
- 253,553 faces;
- no `vt` UV-coordinate records;
- no material library or texture bindings found in the file.

The OBJ is therefore useful for metric geometry analysis and registration, but
it is not the source of the photorealistic appearance. NuRec supplies that
appearance.

### 3.3 Camera metadata limitation

`ply-result/cameras.json` contains only one entry named `startup.jpg` at
position `[0, 0, 1]`. It is not a complete capture trajectory. We currently
do not have the original capture images, full camera poses/intrinsics, or a
coverage map. This is a separate issue from the observed frame flashing:
coverage metadata is needed to predict novel-view quality, while stationary
camera flashing identifies a runtime/compositing failure.

## 4. Terminology and rendering model

### 4.1 NuRec

NuRec is NVIDIA's neural-reconstruction rendering path. For a camera pose, it
projects and blends a neural/Gaussian representation to reconstruct the
scanned lab appearance. The `model.nurec` payload, not the small proxy mesh,
contains most of the visual information.

### 4.2 Ordinary RTX geometry

The Kinova, mobile base, gripper, task objects, lights, and debug objects are
ordinary USD geometry. PhysX updates their rigid bodies and articulation
states; Isaac's RTX renderer draws their triangle meshes and materials.

### 4.3 Full camera composition

The intended render path is:

```text
NuRec lab color/depth ───────┐
                            ├─ registered compositing ─ camera RGB/depth tensor
RTX robot/object color/depth ┘
```

The robot is not converted into splats. Its conventional rendering must be
correctly depth-composited with the neural lab for every camera product.

WebGL is not part of the current RoboLab pipeline. A WebGL viewer can establish
that splats are visually valid, but it does not by itself provide synchronized
Isaac physics, dynamic USD robot occlusion, multiple Isaac Lab camera tensors,
or headless policy evaluation. Likewise, `.usdc` only means binary USD Crate
encoding; it does not indicate whether a layer contains splats, meshes,
materials, physics, or semantics.

## 5. How the scan entered RoboLab

### 5.1 Floor-corrected wrapper

We created:

```text
assets/scenes/xgrids2_lab/xgrids2_lab_zup.usda
```

The wrapper:

- declares metres;
- declares Z-up;
- references the original `mesh-model.usdz` rather than modifying it;
- translates the referenced scan by `+1.075 m` in Z.

The dominant source floor was measured at approximately `Z=-1.075 m`; after
the wrapper translation it lies at `Z=0`.

### 5.2 Numbered view audit

`examples/audit_xgrids2_lab_views.py` renders reproducible first-person and
overhead views and records camera/depth metadata. The initial six views are in:

```text
output/xgrids2_lab/numbered_views/
```

All initial views had approximately 99.73% to 99.96% valid depth. These views
were used to identify the dining-table, snack-table, living-room, and work
areas and to discuss approximate robot staging poses.

## 6. LabApt coordinate registration

The existing robot/navigation system on the cobot computer had labeled
semantic navigation poses for:

- kitchen sink;
- microwave;
- kitchen shelf;
- living room;
- snack table;
- study desk;
- dining table.

The relevant map data was copied read-only into:

```text
assets/scenes/xgrids2_lab/labapt_reference/
```

The labels are robot staging poses `[x, y, yaw]`, not asset centers. Visual
landmarks alone were sufficient for validation but not precise enough to fit a
metric transform, so we used geometry-based registration.

### 6.1 Registration method

`examples/register_labapt_to_xgrids2.py`:

1. parses vertices and faces from the conventional OBJ;
2. intersects the mesh with horizontal planes at corrected-world heights
   0.25, 0.45, 0.75, 1.10, and 1.55 m;
3. samples the intersection lines and LabApt vector-map segments at 0.08 m;
4. searches rotations and estimates translation with FFT correlation;
5. refines rotation and translation against nearest-surface distance;
6. transforms all semantic staging poses;
7. saves the transform, registration geometry, and overlay.

No camera image, splat appearance, or manually selected semantic point was
used to fit the transform.

### 6.2 Result

```text
p_xgrids = R(35.617718 deg) p_labapt + (-13.259476, 6.588388) m
yaw_xgrids = wrap(yaw_labapt + 35.617718 deg)
```

Fit metrics:

| Metric | Result |
|---|---:|
| Fraction within 0.24 m | 77.24% |
| Median nearest-surface distance | 0.0875 m |
| 75th-percentile distance | 0.2128 m |

The saved transform is:

```text
output/xgrids2_lab/labapt_registration/labapt_to_xgrids2_transform.json
```

Seven aligned semantic views passed RGB/depth validation with 99.96% to 100%
valid depth. The visual labels agreed with the expected lab zones.

## 7. Physics and navigation architecture

The key design decision is to keep appearance, collision, and robot navigation
as separate contracts:

```text
XGRIDS2 NuRec USD        → appearance only
LabApt-derived cuboids   → static obstacle collision
clean floor slab         → gravity support
robot footprint argument → navigable-space inflation
mobile Kinova USD        → robot visual, articulation, and collision
```

### 7.1 Why the reconstruction does not directly drive PhysX

The detailed reconstructed mesh contains seams, noise, small components,
decorative geometry, and unexpectedly authored `PhysicsCollisionAPI`. Direct
contact against those triangles caused clean probes to compete with the
intended floor and roll/fall unpredictably.

`xgrids2_lab_navigation_task.py` therefore disables collision on the full
visual subtree. A clean static slab and simplified cuboids are the only lab
geometry participating in the route-first PhysX scene.

This remains required even if the XGRIDS team provides fully stable splat
rendering. Splats are a visual/radiance representation, not a robust PhysX
collision contract.

### 7.2 Collision region

The current route-first physics scope is:

```text
x = [-1, 7] m
y = [-4, 2] m
```

There are 229 transformed LabApt vector-map segments. Each becomes a hidden
static cuboid:

- width: 0.06 m;
- height: 1.50 m;
- collision enabled;
- no dynamic rigid-body or mass APIs.

The floor is an 8 m by 6 m by 0.10 m static cuboid centered at `Z=-0.05 m`,
placing its upper surface exactly at `Z=0`.

### 7.3 Robot footprint and staging poses

The drive-base collision radius is 0.30 m, but navigation cannot use the base
alone. The original arm home posture extended approximately 0.564 m from the
base and collided near a staging pose.

A joint-limit-safe folded transport posture was measured:

```text
[joint_1 ... joint_7] = [0, -0.70, pi, -2.50, 0, 0.70, pi/2]
```

Its maximum link-center radius was 0.305 m. Navigation uses:

```text
0.36 m whole-robot radius
+ 0.05 m safety margin
+ 0.03 m proxy half-width
= 0.44 m required centerline clearance
```

The transformed raw snack pose had only 0.269 m clearance, so it was shifted
0.220 m to `(4.56, -0.76)` while preserving its heading. The dining pose
remains `(1.7656, -1.7639)`.

### 7.4 Registered task

The environment name is:

```text
XGRIDS2LabNavigationKinovaMobileJointPosition
```

Its action contract is:

```text
base     [vx_body, vy_body, wz]
arm      seven absolute joint positions
gripper  binary open/close
```

The waypoint follower turns world-frame A* directions into body-frame base
commands and rotates the base so the physically fixed navigation camera faces
the route. It never moves the camera independently.

## 8. Robot and camera rendering implementation

The base navigation camera is authored at:

```text
{ENV_REGEX_NS}/robot/base_link/navigation_camera
```

It is a child of `base_link`, so the sensor follows base translation and yaw
through USD parenting. Its configured outputs are RGB and
`distance_to_image_plane`. The camera is not artificially floated or rotated
during traversal.

The visualization script adds a separate static third-person audit camera. At
each `env.step(action)`:

1. the base/arm action is applied;
2. PhysX advances the robot;
3. articulation transforms and the parented camera pose update;
4. RTX renders conventional robot geometry;
5. NuRec renders the lab reconstruction;
6. registered compositing should merge them;
7. Isaac Lab exposes the camera tensors;
8. the script joins third-person RGB, robot-camera RGB, and a navigation map
   into the MP4 frame.

The robot traversal visualization disables the unused wrist stream to reduce
VRAM and uses two active RGB render products: the third-person camera and the
base-parented navigation camera. The registered robot configuration retains
the policy sensor definitions.

## 9. Validation gates

### Gate 1: collision overlay

Confirmed the registered transform, route-scoped map geometry, hidden static
colliders, and visual/collision separation.

Artifact:

```text
output/xgrids2_lab/physics_gates/gate1_collision_overlay.png
```

### Gate 2: floor drop

Nine dynamic 6 cm-radius probes were dropped along the route for 180 steps. All
settled with center height approximately 0.060 m and negligible velocity.

Artifacts:

```text
output/xgrids2_lab/physics_gates/gate2_floor_drop.png
output/xgrids2_lab/physics_gates/gate2_floor_drop_summary.png
output/xgrids2_lab/physics_gates/gate2_floor_drop_results.json
```

### Gate 3: inflated navigation

Built an 0.08 m occupancy grid from the same obstacle centerlines used by
PhysX, planned an eight-connected A* route, and verified the continuously
resampled path against the 0.44 m clearance requirement.

Artifact:

```text
output/xgrids2_lab/physics_gates/gate3_inflated_navigation.png
```

### Gate 4: robot spawning

Spawned and settled the mobile Kinova at dining and snack staging poses,
restored the compact transport posture, and validated pose, floor height,
velocity, and parented sensor output.

Artifact:

```text
output/xgrids2_lab/physics_gates/gate4_robot_spawn_montage.png
```

### Gate 5: closed-loop traversal

| Metric | Result |
|---|---:|
| Route points | 38 |
| Control steps | 629 |
| Required line clearance | 0.440 m |
| Minimum observed line clearance | 0.5049 m |
| Maximum nonvertical force excess | 0.000 N |
| Final position error | 0.0347 m |
| Final yaw error | 0.0380 rad |

Artifacts:

```text
output/xgrids2_lab/physics_gates/gate5_base_traversal.mp4
output/xgrids2_lab/physics_gates/gate5_base_traversal_keyframes.png
output/xgrids2_lab/physics_gates/gate5_base_traversal_final.png
output/xgrids2_lab/physics_gates/gate5_base_traversal_results.json
```

The JSON records a physics pass. The MP4 also exposes the unresolved NuRec
rendering instability and must not be described as a visually stable result.

## 10. Problems found and how they were handled

### 10.1 Asset orientation and floor height

**Symptom:** the dominant floor was approximately 1.075 m below the desired
world floor.

**Cause:** the supplied reconstruction's source coordinate frame placed the
dominant floor near `Z=-1.075`.

**Resolution:** created a non-destructive Z-up, metre-scale wrapper referencing
the original USDZ and translated its scan child by `+1.075 m`.

### 10.2 LabApt and scan did not share a usable frame directly

**Symptom:** semantic poses from the real cobot system could not be applied
directly to the scan.

**Cause:** the systems used different XY origins and headings. The semantic
coordinates were staging poses, not exact furniture centers.

**Resolution:** registered LabApt vector geometry to horizontal slices of the
metric scan mesh; saved and visually audited the rigid transform.

### 10.3 Reconstructed collision competed with the clean floor

**Symptom:** probes rolled or fell despite a clean floor slab.

**Cause:** the visual reconstruction unexpectedly carried collision APIs and
its noisy triangles participated in PhysX.

**Resolution:** explicitly disabled collision on the visual subtree and used a
clean floor plus simplified static proxies.

### 10.4 Base-only inflation was unsafe

**Symptom:** the robot arm/gripper could intersect an obstacle while the base
center appeared clear.

**Cause:** the initial arm posture extended far outside the 0.30 m base radius.

**Resolution:** measured candidate link envelopes, selected a limit-safe
transport fold, and inflated navigation using the complete robot envelope.

### 10.5 First compact posture violated a joint limit

**Symptom:** live articulation validation rejected an initially selected fold.

**Cause:** joint 4 was outside the authored robot limit.

**Resolution:** rejected that posture and selected the measured 0.305 m
link-center-radius posture listed above.

### 10.6 Raw snack staging pose was in inflated collision

**Symptom:** the mapped snack pose was only 0.269 m from an obstacle.

**Cause:** surveyed navigation goals do not automatically satisfy the new
whole-robot 0.44 m clearance contract.

**Resolution:** shifted it 0.220 m to the nearest suitable region while
preserving the surveyed heading.

### 10.7 GPU PhysX could not filter against hundreds of static roots

**Symptom:** a contact sensor could not conveniently filter the dynamic base
against 229 independent static collider roots.

**Resolution:** combined exact geometric clearance checks with a supported net
base-force criterion. Floor support/friction is accepted while the force lies
inside a 45-degree cone around vertical; lateral obstacle contact produces
positive `horizontal_force - abs(vertical_force)`.

### 10.8 Camera originally risked being treated as a floating diagnostic

**Requirement:** the navigation sensor must be fixed to the physical robot,
and navigation should rotate the base toward its destination rather than
artificially rotating the camera.

**Resolution:** authored the camera once beneath `robot/base_link` and made
the waypoint follower command base yaw. No runtime navigation-camera pose
writes are used.

### 10.9 Audit VRAM pressure

**Symptom:** the workstation could not reliably support the complete sensor set
plus diagnostic cameras at high resolution alongside other Isaac processes.

**Resolution:** the visualization-only script disables the unused wrist stream
and lowers audit resolution. This improves resource use but does not solve the
NuRec compositor failure described below.

### 10.10 Loader/config integration errors

During task authoring:

- postponed annotations conflicted with the current dynamic task loader, so the
  task file was kept compatible without changing unrelated loader behavior;
- a class-level temporary `yaw` scalar was interpreted as a scene asset, so
  the quaternion calculation was inlined;
- the visual subtree collision override was made explicit rather than relying
  on the source USD's authored APIs.

## 11. NuRec rendering investigation

### 11.1 Initial symptom

The full traversal video alternates between a bright photorealistic lab and a
dark/gray proxy-like lab. The robot physics and route continue correctly.

An early hypothesis attributed dark moving-camera frames to unsupported novel
views. Novel-view coverage remains a legitimate future policy concern, but it
does **not** explain the immediate flashing: a completely stationary
third-person camera changes state at the same time.

### 11.2 Supplied NuRec render settings

The USDZ's `default.usda` authors settings including:

- Raytraced Lighting;
- histogram disabled;
- registered-compositing inverse color correction enabled;
- registered-compositing inverse tonemapping enabled;
- tonemap operator 2.

The local helper `robolab/core/utils/nurec_rendering.py` applies the relevant
baseline before loading the NuRec stage:

- single GPU;
- automatic multi-GPU disabled;
- Gaussian tonemapping behavior;
- histogram disabled;
- registered-compositing inverse color/tone settings;
- tonemap operator 2;
- temporal antialiasing disabled for the diagnostic.

It also explicitly authors registered-compositing attributes on every camera
render product. These settings remove configuration ambiguity; they are not
claimed to repair a loaded-runtime integration bug.

### 11.3 Minimal scene that passes

`examples/diagnose_xgrids2_nurec_stability.py` loads only the wrapped scan and
two camera products. In the saved 60-frame moving-camera run:

- camera A was stationary and pixel-identical for all frames;
- stationary mean luminance was 134.3027;
- stationary luminance range was 0.0;
- maximum stationary mean absolute frame difference was 0.0;
- camera B moved continuously and remained in the neural-rendered brightness
  range;
- the diagnostic passed.

Artifacts:

```text
output/xgrids2_lab/nurec_stability_moving/fixed_camera.mp4
output/xgrids2_lab/nurec_stability_moving/fixed_camera.png
output/xgrids2_lab/nurec_stability_moving/results.json
```

This proves that the supplied neural asset and baseline configuration can
produce stable frames in a lightweight multi-camera scene.

### 11.4 Full robot scene that fails visually

The fixed-pose full-environment probe measured:

| Metric | Tiled-camera probe | Standard-camera probe |
|---|---:|---:|
| Frames | 60 | 30 |
| Mean luminance minimum | 54.446 | 54.446 |
| Mean luminance maximum | 161.383 | 161.383 |
| Luminance range | 106.937 | 106.937 |
| Maximum mean absolute frame difference | 115.561 | 115.561 |
| Passed | No | No |

Because the camera pose was fixed, the two-state output is a
rendering/compositing failure, not motion blur or novel-view coverage.

### 11.5 Experiments that did not solve it

The following were tested and ruled out as simple fixes:

| Experiment | Result |
|---|---|
| Correct single-GPU and pre-stage settings | Required baseline; full task still flashes |
| Explicit per-render-product compositing attributes | Minimal scene passes; full task still flashes |
| Disable Fabric | No meaningful improvement |
| Use standard Camera instead of TiledCamera | No meaningful improvement |
| Additional `sim.render()` and forced camera update | No meaningful improvement |
| Reduce active sensors/resolution | Reduced load but did not remove flashing |
| Replicator `rt_subframes=4` with blocking capture | Impractically slow, tens of seconds per frame |

One out-of-memory event occurred while multiple Isaac processes and diagnostic
runs were resident. Resource pressure can reduce throughput, but it does not
explain a fixed camera alternating between two specific visual states, and the
failure persisted in controlled probes.

Diagnostic/failed-experiment folders retained as evidence:

```text
output/xgrids2_lab/nurec_robot_env_probe/
output/xgrids2_lab/nurec_standard_camera_probe/
output/xgrids2_lab/physics_gates_stable/
output/xgrids2_lab/physics_gates_stable_final/
output/xgrids2_lab/physics_gates_smooth/
```

They must not be presented as successful fixes.

### 11.6 Current rendering conclusion

The most precise conclusion supported by the evidence is:

> The supplied NuRec asset renders stably in a lightweight two-camera scene,
> but the local Isaac Sim 5.1 runtime intermittently fails to deliver the
> registered neural composite to camera render products in the loaded robot
> task.

We do not have the internal NVIDIA plugin source needed to name the exact
failing function. The failure boundary is isolated to the loaded
NuRec/RTX/render-product integration, not the MP4 encoder, robot physics,
camera parenting, or simple lack of splat density.

Official current Isaac Sim documentation says NuRec setup must occur before the
first Hydra synchronization and that NuRec assets require single-GPU
rendering:

<https://docs.isaacsim.omniverse.nvidia.com/latest/assets/nurec_utils.html>

The current documentation exposes `isaacsim.replicator.nurec_utils` helpers,
but the same helper module was not present in the local Isaac Sim 5.1
installation. A version/integration mismatch is therefore plausible and must
be checked against the XGRIDS team's known-good setup.

## 12. Why a textured mesh was discussed, and the corrected decision

A UV-textured conventional mesh was initially proposed as a robust fallback
for policy RGB/depth because ordinary RTX rendering avoids NuRec registered
compositing. The supplied OBJ is geometry-only and cannot provide the
photorealistic lab appearance by itself.

The XGRIDS team subsequently stated that they can perform the required work
properly using only splats. That is credible and changes the immediate request:

- a textured visual mesh is **not inherently required**;
- a WebGL viewer alone is not sufficient evidence for the Isaac use case;
- we need their native Isaac Sim/Omniverse integration and exact configuration;
- simplified collision geometry remains necessary, but we already author it
  independently.

Current decision:

```text
Primary path:
reproduce the XGRIDS team's known-good native Isaac/NuRec splat pipeline

Fallback only:
obtain or bake a textured mesh if stable policy-camera integration is not
available in the target Isaac runtime
```

## 13. Information needed from the XGRIDS team

The required use case must be stated precisely:

- this exact `mesh-model.usdz`;
- a dynamic articulated USD robot;
- correct robot/lab occlusion;
- a camera parented to a moving robot link;
- a simultaneous third-person camera;
- later, a simultaneous wrist camera;
- continuous physics stepping over hundreds of frames;
- stable synchronized RGB and metric depth tensors;
- Isaac Lab/Replicator or an equivalent headless API;
- reproducibility across resets.

Ask for:

1. exact Isaac Sim build/version;
2. exact XGRIDS/NuRec extension versions;
3. GPU/driver configuration and single-/multi-GPU mode;
4. the intended entry point: `mesh-model.usdz` or `model.usdz`;
5. required extensions and launch arguments;
6. setup calls that must run before the first Hydra synchronization;
7. a minimal native Isaac script loading this exact asset;
8. camera/render-product creation code;
9. the physics-step/render/camera-read ordering;
10. a demonstration with a moving conventional USD object or robot;
11. two or more simultaneous moving cameras;
12. stable RGB and metric depth capture for several hundred frames;
13. whether headless rendering is supported;
14. any registered-compositing, tone-map, exposure, gamma, or color-space
    requirements;
15. whether splat and RTX depth buffers are merged for correct occlusion;
16. expected resolution/frame rate for two or three cameras;
17. full capture camera poses/intrinsics or a coverage map, if available;
18. any simplified floor/wall/furniture collision export, if available.

If their successful system is WebGL-only, ask how its color/depth framebuffer
is synchronized and composited with Isaac's dynamic RTX robot and exposed as
policy tensors. The preferred deliverable is a native Isaac/Omniverse example,
not WebGL materials.

## 14. Current limitations

1. NuRec rendering is not stable in the full local robot task.
2. The original capture trajectory and coverage metadata are missing.
3. Only the route-first `x=[-1,7], y=[-4,2]` region has collision proxies.
4. The tables are not clean support/collision surfaces.
5. No dynamic task object has been added to the XGRIDS lab.
6. No pick/place regions or manipulation staging poses have been validated.
7. No arm reachability or swept-collision test has been performed in this lab.
8. No scripted pick-carry-place oracle has been transferred to this environment.
9. Stable simultaneous Kinect/wrist/third-person policy capture has not passed.
10. The XGRIDS2 work and assets are currently untracked by Git.

## 15. Recommended next steps

### Gate R1: reproduce the team's known-good renderer

Obtain their exact script/version/configuration and run it unchanged against
the supplied asset. Record version information and a 300-frame fixed-camera
and moving-camera test.

Acceptance:

- no proxy/composite state switches;
- no unexplained fixed-camera frame changes;
- stable headless capture;
- exact asset path and settings recorded.

### Gate R2: add dynamic RTX geometry

Add one moving conventional USD primitive, then the mobile Kinova.

Acceptance:

- correct depth occlusion between splats and RTX geometry;
- no flashing;
- stable RGB and metric depth.

### Gate R3: policy sensor set

Run base-parented, wrist, and third-person cameras simultaneously during the
existing traversal.

Acceptance:

- at least 300 continuous frames per camera;
- no proxy fallback;
- valid metric depth;
- camera transforms exactly follow their physical parents;
- practical target frame rate/resolution documented.

### Gate M0: semantic fixtures

After rendering is stable:

1. author clean dining/snack tabletop support proxies;
2. measure valid pick/place bounds;
3. validate object visibility at staging headings;
4. drop a dynamic object on each surface;
5. validate Kinova reachability;
6. extend collision proxies only where the manipulation route requires them.

### Gate M1: scripted manipulation

Transfer the kitchen's deterministic manipulation oracle only after fixture,
camera, and reachability gates pass. Keep environment geometry robot-agnostic
and supply robot radius/reach parameters through the robot/task layer.

Do not launch a 100-trial manipulation gate before the visual and semantic
fixture gates are reliable.

## 16. Reproduction commands

Use the Isaac Lab environment:

```bash
ISAAC_PY=/home/robocup/miniconda3/envs/env_isaaclab/bin/python
```

### Recompute LabApt registration

```bash
$ISAAC_PY examples/register_labapt_to_xgrids2.py
```

### Render numbered views

```bash
$ISAAC_PY examples/audit_xgrids2_lab_views.py --headless \
  --view-set numbered \
  --output-dir output/xgrids2_lab/numbered_views
```

### Render aligned semantic views

```bash
$ISAAC_PY examples/audit_xgrids2_lab_views.py --headless \
  --view-set labapt_aligned \
  --output-dir output/xgrids2_lab/labapt_aligned_views
```

### Render physics gates

```bash
$ISAAC_PY examples/visualize_xgrids2_lab_physics_gates.py --gate all
```

### Run robot spawn/traversal visualization

Use a new output directory when diagnosing changes so the known evidence is
not overwritten:

```bash
$ISAAC_PY examples/visualize_xgrids2_lab_robot_gates.py --headless \
  --output-dir output/xgrids2_lab/physics_gates_new
```

### Run minimal NuRec regression

```bash
$ISAAC_PY examples/diagnose_xgrids2_nurec_stability.py --headless \
  --frames 60 \
  --output-dir output/xgrids2_lab/nurec_stability_new
```

### Static/schema tests

```bash
$ISAAC_PY -m pytest -q tests/test_xgrids2_lab_physics_schema.py
```

Rerun on 2026-08-27: **10 passed**.

### Live Isaac tests

```bash
ACCEPT_EULA=Y $ISAAC_PY -m pytest -q tests/test_xgrids2_lab_physics_sim.py
```

Last recorded result: **5 passed, 0 failures, 0 errors, 0 skipped** in
464.32 seconds.

## 17. Implementation file map

### Source assets

```text
assets/scenes/xgrids2_lab/forward_compatible/usdz-result/mesh-model.usdz
assets/scenes/xgrids2_lab/forward_compatible/usdz-result/model.usdz
assets/scenes/xgrids2_lab/forward_compatible/mesh-files/UT_Robtics_lab_int.obj
assets/scenes/xgrids2_lab/usd-result/UT_Robtics_lab_int.usd
assets/scenes/xgrids2_lab/ply-result/point_cloud/iteration_100/
assets/scenes/xgrids2_lab/ply-result/cameras.json
```

### Wrapper and map data

```text
assets/scenes/xgrids2_lab/xgrids2_lab_zup.usda
assets/scenes/xgrids2_lab/labapt_reference/LabApt.vectormap.txt
assets/scenes/xgrids2_lab/labapt_reference/LabApt.vectormap.json
assets/scenes/xgrids2_lab/labapt_reference/LabApt.navigation.json
assets/scenes/xgrids2_lab/labapt_reference/labapt_gridmap.npz
```

### Runtime implementation

```text
robolab/core/utils/nurec_rendering.py
robolab/robots/kinova_mobile_cobot.py
robolab/registrations/kinova_mobile/auto_env_registrations_xgrids2_lab.py
robolab/tasks/mobile_manipulation/xgrids2_lab_geometry.py
robolab/tasks/mobile_manipulation/xgrids2_lab_navigation_task.py
robolab/tasks/mobile_manipulation/xgrids2_lab_navigation_oracle.py
```

### Scripts

```text
examples/register_labapt_to_xgrids2.py
examples/audit_xgrids2_lab_views.py
examples/visualize_xgrids2_lab_physics_gates.py
examples/visualize_xgrids2_lab_robot_gates.py
examples/diagnose_xgrids2_nurec_stability.py
examples/audit_mobile_kinova_transport_postures.py
```

### Tests

```text
tests/test_xgrids2_lab_physics_schema.py
tests/test_xgrids2_lab_physics_sim.py
```

### Detailed supporting documents

```text
docs/XGRIDS2_LABAPT_ALIGNMENT.md
docs/XGRIDS2_LAB_PHYSICS_WALKTHROUGH.md
```

The alignment document is authoritative for coordinate registration. The
physics walkthrough is authoritative for Gates 1-5, except that its early
novel-view explanation is superseded by Section 11 of this handoff.

## 18. Output/artifact map

### Initial scan audit

```text
output/xgrids2_lab/numbered_views/
```

### Registration

```text
output/xgrids2_lab/labapt_registration/labapt_to_xgrids2_transform.json
output/xgrids2_lab/labapt_registration/labapt_xgrids_overlay.png
output/xgrids2_lab/labapt_registration/registration_geometry.npz
output/xgrids2_lab/labapt_aligned_views/
```

### Physics/navigation

```text
output/xgrids2_lab/physics_gates/gate1_collision_overlay.png
output/xgrids2_lab/physics_gates/gate2_floor_drop.png
output/xgrids2_lab/physics_gates/gate2_floor_drop_summary.png
output/xgrids2_lab/physics_gates/gate2_floor_drop_results.json
output/xgrids2_lab/physics_gates/gate3_inflated_navigation.png
output/xgrids2_lab/physics_gates/gate4_robot_spawn_montage.png
output/xgrids2_lab/physics_gates/gate5_base_traversal.mp4
output/xgrids2_lab/physics_gates/gate5_base_traversal_keyframes.png
output/xgrids2_lab/physics_gates/gate5_base_traversal_final.png
output/xgrids2_lab/physics_gates/gate5_base_traversal_results.json
```

### Rendering comparison

Known-good lightweight diagnostic:

```text
output/xgrids2_lab/nurec_stability_moving/fixed_camera.mp4
output/xgrids2_lab/nurec_stability_moving/fixed_camera.png
output/xgrids2_lab/nurec_stability_moving/results.json
```

Known-bad full-scene evidence:

```text
output/xgrids2_lab/physics_gates/gate5_base_traversal.mp4
output/xgrids2_lab/nurec_robot_env_probe/nurec_robot_env_probe.json
output/xgrids2_lab/nurec_standard_camera_probe/nurec_robot_env_probe.json
```

## 19. Version-control and portability warning

At the time of this handoff, the XGRIDS2 assets, XGRIDS-specific documents,
scripts, registration package, task modules, utility, and tests all appear as
untracked files in `git status`. The large scan assets may be intentionally
excluded from normal Git, but the implementation and documentation have also
not been committed.

A fresh clone will not reproduce this environment unless the team:

1. commits the code, tests, and documentation;
2. chooses an approved storage mechanism for the large XGRIDS assets;
3. records checksums or a manifest for external asset retrieval;
4. preserves the generated transform JSON or reruns registration;
5. records the target Isaac Sim/NuRec runtime once confirmed.

Do not delete or overwrite the current output directories until the important
comparison artifacts have been archived.

## 20. Do-not-repeat list

- Do not treat the 811 MB USDZ as an 811 MB conventional textured mesh.
- Do not use the reconstruction triangles directly as the route-first PhysX
  collision layer.
- Do not infer that millions of splats guarantee arbitrary-view coverage.
- Do not blame camera parenting for flashing seen in a stationary camera.
- Do not artificially rotate or float the navigation camera; rotate the base.
- Do not claim `physics_gates_stable*` or `physics_gates_smooth` solved the
  renderer.
- Do not use blocking `rt_subframes=4` as a real-time robotics solution.
- Do not require a textured visual mesh before attempting the XGRIDS team's
  known-good native NuRec integration.
- Do not run the 100-trial manipulation gate before semantic fixture,
  reachability, and stable policy-camera gates pass.

## 21. Immediate handoff checklist

The next owner should:

1. read this document, `XGRIDS2_LABAPT_ALIGNMENT.md`, and
   `XGRIDS2_LAB_PHYSICS_WALKTHROUGH.md`;
2. preserve the current known-good and known-bad rendering artifacts;
3. obtain the XGRIDS team's exact native Isaac example and version matrix;
4. run Gates R1-R3 without changing physics/navigation code;
5. update this handoff with the confirmed renderer setup and results;
6. only then begin semantic tabletop and object manipulation authoring.
