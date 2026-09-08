# Milestone 0 kitchen audit: complete code and design walkthrough

This document explains the code used to measure the EBench/GenManip dishwasher
kitchen, select the initial pick/place regions, stage the Kinova mobile cobot,
verify positional reach, and render the resulting scene. Line references match
the source as it existed when this audit was completed.

## What was proven, and what was not

The audit establishes four things:

1. The kitchen island contains a conservative, flat, unobstructed can-pick
   rectangle at the authored scene scale.
2. The opposing counter contains a conservative placement rectangle.
3. Two mobile-base staging footprints fit geometrically in the measured aisle.
4. The authored Kinova articulation can position `robotiq_85_base_link` over the
   center and four corners of both rectangles without reaching a joint limit.

The audit does **not** yet prove a collision-free arm trajectory, a feasible 6D
gripper orientation, stable contact, grasp closure, object attachment, or mobile
navigation under PhysX. The EBench kitchen USD is currently being consumed as
render geometry; a task-grade collision overlay has not yet been authored.

## Coordinate and scale conventions

- USD and Isaac Lab use metres.
- `z` points upward.
- World `x` crosses the aisle from the island toward the opposing counter in the
  selected workspace.
- The mobile robot root is `base_link`; its planar authored height is 0.468 m.
- Quaternion tuples are scalar-first `(w, x, y, z)` in the relevant Isaac Lab
  configuration.
- Yaw zero is the authored robot orientation. The navigation mast/camera lies on
  the local `-Y` side, so keeping yaw zero prevents that projection from pointing
  into either counter at the selected staging poses.

## Script 1: `examples/audit_genmanip_kitchen_milestone0.py`

### Lines 1–10: license and contract

- Lines 1–2 retain the repository's NVIDIA/Apache-2.0 source header.
- Line 4 begins the module docstring.
- Lines 6–9 state the exact contract: inspect the authored USD, find upward
  horizontal triangles, convert them into a top-height grid, and produce a JSON
  report plus image. The final sentence is important: this script measures
  visual geometry; it does not silently pretend that visual meshes are good
  collision meshes.

### Lines 12–31: imports and Isaac/ USD startup

- Line 12 imports `argparse` for command-line configuration.
- Line 13 imports `json` for the machine-readable report.
- Line 14 imports `os` for repository-independent path construction.
- Line 15 imports `deque`. It is unused in the final version and can be removed;
  it remained from an early pure-Python flood-fill implementation that was
  replaced by OpenCV connected components.
- Line 17 imports OpenCV. It supplies polygon rasterization, connected-component
  labeling, the diagnostic color map, text labels, and PNG writing.
- Line 18 imports NumPy for transformed vertices, triangle normals, masks, and
  raster arrays.
- Line 19 imports Isaac Lab's `AppLauncher`. USD plugins and extensions are more
  reliably initialized through the same Kit runtime used by the rest of RoboLab.
- Line 22 creates the argument parser.
- Line 23 defines the output directory while keeping it overridable.
- Line 24 adds standard Isaac arguments such as `--headless` and device selection.
- Line 25 parses the complete argument set.
- Line 26 forces headless mode because this is a geometry audit, not an
  interactive viewer.
- Line 27 starts Kit/Isaac before importing `pxr` APIs.
- Line 29 imports USD geometry types after application startup. `Gf` supplies
  vector math, `Usd` opens/traverses the stage, and `UsdGeom` identifies meshes
  and computes transforms.
- Line 31 imports `PACKAGE_DIR`, avoiding dependence on the shell's current
  directory.

### Lines 34–50: immutable audit inputs

- Lines 34–45 assemble the exact published EBench dishwasher-scene path. We use
  the complete `scene.usd`, not manually reconstructed furniture, so all authored
  transforms and scales are preserved.
- Line 47 sets a 0.02 m raster resolution. Two centimetres is fine enough to
  resolve a roughly 68 mm can and counter cut-outs, while keeping the grid small
  and deterministic.
- Line 48 restricts measurement to `x=[-1.0, 2.5]`, `y=[-2.5, 1.5]`. This covers
  the task kitchen and excludes unrelated scene space.
- Line 49 restricts candidate work surfaces to `z=[0.70, 1.20]` m. This includes
  normal counters and the higher island, but rejects the floor and most overhead
  cabinetry.
- Line 50 requires `normal.z >= 0.985`, approximately within 9.9 degrees of
  vertical. This retains slightly imperfect triangulated worktops but excludes
  walls, cabinet faces, and steep decorations.

### Lines 53–91: `_triangles_from_stage`

- Line 53 declares a stage-to-triangle function. Returning ordinary dictionaries
  keeps the next stages independent of USD objects.
- Line 54 documents that output positions are in the world frame.
- Line 56 creates one `XformCache`, avoiding repeated hierarchy evaluation for
  every vertex.
- Line 57 creates the result list.
- Line 58 traverses every composed prim, including referenced sub-USD assets.
- Lines 59–60 reject non-mesh prims immediately.
- Line 61 wraps the prim as `UsdGeom.Mesh`.
- Lines 62–64 read vertices, per-face vertex counts, and the flattened index list.
- Lines 65–66 skip malformed or empty meshes.
- Line 67 retrieves the complete local-to-world transform. This is what preserves
  every authored parent transform and scale.
- Lines 68–70 transform each point to world coordinates and store it as
  double-precision NumPy data. Double precision prevents millimetre-scale drift
  while nested USD transforms are composed.
- Line 71 starts a cursor into the flattened face-index array.
- Lines 72–74 recover each polygon's own vertex indices and advance the cursor.
- Lines 75–76 reject degenerate faces with fewer than three vertices.
- Line 77 chooses the first polygon vertex as a triangle-fan anchor.
- Lines 78–79 triangulate every polygon as `(anchor, vertex_i, vertex_i+1)`. This
  handles quads and larger worktop faces without requiring a separate mesh tool.
- Line 80 calculates the unnormalized cross-product normal.
- Lines 81–84 reject zero-area triangles and normalize valid normals.
- Lines 85–86 apply the upward-facing threshold.
- Line 87 uses the mean triangle height. Candidate triangles are already nearly
  horizontal, so one representative height is sufficient for a 2 cm grid.
- Lines 88–89 apply the work-surface height band.
- Line 90 records XY vertices, height, and source prim path. The source path was
  kept for traceability even though the current JSON summarizes components.
- Line 91 returns all accepted triangles. The completed scene contained 103,437.

### Lines 94–109: `_rasterize`

- Line 94 begins conversion from irregular triangles to a regular top-down grid.
- Line 95 unpacks world bounds.
- Lines 96–97 derive grid dimensions and add the inclusive endpoint cell.
- Line 98 initializes top heights as `NaN`; `NaN` means no candidate surface.
- Line 99 initializes triangle ownership to `-1` for later traceability.
- Line 100 iterates with a stable triangle index.
- Line 101 allocates the three integer image coordinates.
- Line 102 maps world X into image columns.
- Line 103 maps world Y into rows while flipping Y because image rows increase
  downward.
- Line 104 creates a temporary binary mask for one triangle.
- Line 105 fills that projected triangle.
- Line 106 updates only covered pixels that are empty or lower than this triangle.
  Taking the maximum Z is deliberate: from above, the physically available top
  surface should hide geometry below it.
- Lines 107–108 write the height and owning triangle.
- Line 109 returns both grids. In the checked-in function `owner` is computed but
  not yet written; during the audit it was also saved as
  `work_surface_owner.npy` for debugging.

### Lines 112–140: `_components`

- Line 112 begins grouping adjacent top-surface pixels.
- Line 113 converts finite heights into a binary occupancy map.
- Line 114 runs 8-connected component labeling. Diagonal connectivity avoids
  fragmenting a single triangulated worktop at raster corners.
- Line 115 unpacks bounds for pixel-to-world conversion.
- Line 116 initializes the component report.
- Line 117 skips label zero, which OpenCV reserves for background.
- Line 118 reads component area in cells.
- Lines 119–120 discard components smaller than 20 cells, or 0.008 m². These are
  small props/triangulation fragments, not useful task surfaces.
- Line 121 retrieves every component pixel.
- Lines 122–125 convert its pixel extrema back to world XY, including the Y flip.
- Lines 126–139 append label, cell count, area, bounds, height range, and centroid.
  The height range helps reveal that raw connected components may include a sink,
  cooktop, or objects at several heights; it must not be confused with a final
  valid placement mask.
- Line 140 sorts largest-first so the island and counter appear at the top.

### Lines 143–156: `_write_heightmap`

- Line 143 declares the diagnostic image writer.
- Line 144 starts with an off-white RGB background.
- Line 145 identifies measured cells.
- Line 146 allocates an 8-bit height image.
- Line 147 linearly maps the 0.70–1.20 m band into 0–255 and clips outliers.
- Line 148 applies OpenCV's Turbo color map only where geometry exists.
- Line 149 unpacks bounds.
- Line 150 labels at most the twelve largest components to keep the image legible.
- Lines 151–153 map each centroid to a pixel.
- Lines 154–155 draw a black text outline followed by white foreground so labels
  remain visible on any color.
- Line 156 writes the PNG.

### Lines 159–187: orchestration and shutdown

- Line 159 declares the entry point.
- Line 160 opens the complete USD and loads all referenced payloads.
- Lines 161–163 execute extraction, rasterization, and grouping in order.
- Line 164 makes the requested output path absolute.
- Line 165 creates it safely if missing.
- Lines 166–172 assemble the report, including exact input path, resolution,
  bounds, triangle count, and component list.
- Lines 173–175 write indented JSON for both machines and humans.
- Lines 176–177 create the heightmap filename and image.
- Lines 178–180 echo results and artifact locations.
- Line 183 prevents execution when the module is imported.
- Lines 184–187 guarantee Kit shutdown even if the audit raises an exception.

### How the final valid rectangles were selected

The script above deliberately reports raw top-surface components. The final
region selection was a second numerical filtering pass over
`work_surface_top_height.npy`:

1. Select cells within 4 mm of the dominant plane: 1.0450 m for the island and
   0.8453 m for the opposing counter.
2. Erode that binary mask with a 9×9 elliptical kernel. At 2 cm/cell, the kernel
   radius is four cells, giving approximately 8 cm clearance from an edge, sink,
   cooktop, or other height discontinuity.
3. Find connected clear regions.
4. Choose small axis-aligned rectangles around the intended can centers:
   pick `x=[-0.06,0.04], y=[0.10,0.25]`; place
   `x=[1.76,1.86], y=[-0.02,0.22]`.

The 67.9×67.8 mm can footprint fits both rectangles. The rectangles are smaller
than the total clear components by design: Milestone 0 needs a defensible first
task, not the largest possible operating envelope.

## Script 2: `examples/verify_genmanip_kitchen_reach.py`

### Lines 1–33: startup and dependencies

- Lines 1–4 provide licensing and state that this is a positional reach audit.
- Lines 6–9 import CLI handling, deep copying, JSON, and path utilities.
- Line 11 imports OpenCV before Isaac Lab. In this environment that import order
  avoids a shared-library conflict; it is intentionally unused afterward.
- Line 12 imports `AppLauncher`.
- Lines 15–17 define output and convergence-step options. At 240 control steps
  and a 30 Hz environment step, each target receives up to eight simulated
  seconds to settle.
- Lines 18–20 add Isaac arguments, parse them, and start the application before
  importing simulator-dependent modules.
- Lines 22–31 import Torch and the Isaac Lab managers/configuration objects used
  to construct a minimal environment. `sim_utils` on line 24 is unused and can
  be removed without changing behavior.
- Line 33 imports the real RoboLab mobile Kinova configuration; the audit does
  not substitute a geometric reach sphere or simplified kinematic chain.

### Lines 36–54: measured task constants

- Lines 36–39 encode the two selected base-link positions.
- Lines 41–50 encode the conservative XY rectangles and measured surface heights.
- Line 52 stores the measured YCB can half-height, 51.0185 mm.
- Line 53 adds 50 mm between the can center and the controlled gripper-base point.
  This is a pre-grasp-style clearance target, not a final finger-contact pose.
- Line 54 sets a deliberately loose 15 mm pass threshold. The achieved errors
  were below 0.5 mm, but 15 mm is an appropriate engineering acceptance band for
  this preliminary position-only test.

### Lines 57–99: minimal Isaac Lab environment

- Line 57 turns the scene class into an Isaac config object.
- Line 58 subclasses `InteractiveSceneCfg`.
- Line 59 deep-copies the robot so audit-specific modifications cannot mutate the
  shared global robot configuration.
- Line 60 fixes the root. The base is intentionally frozen at each staging pose
  while arm reach is tested; otherwise base motion could hide arm infeasibility.
- Line 61 adds a flat ground plane. The kitchen itself is not needed for this
  kinematic convergence test because its visual geometry has no task collision
  overlay yet.
- Lines 64–66 define an empty observation config because no policy consumes
  observations in this audit.
- Lines 69–79 define one three-dimensional arm action.
- Line 72 binds the action to `robot`.
- Line 73 selects all seven Kinova arm joints by regular expression.
- Line 74 controls `robotiq_85_base_link`, the gripper attachment/base body.
- Lines 75–77 configure absolute, position-only, damped-least-squares IK. DLS is
  numerically stable near singularities. Position-only avoids claiming a grasp
  orientation that has not yet been designed.
- Line 78 keeps command scaling at one, so inputs are metres in the controller's
  expected frame.
- Lines 82–84 reset the scene to default state on environment reset.
- Lines 87–92 combine scene, empty observations, action, and reset event into one
  `ManagerBasedEnvCfg` with a single non-replicated environment.
- Line 94 begins post-initialization.
- Line 95 applies one policy action every four physics steps.
- Line 96 uses 120 Hz physics, producing a 30 Hz environment/control step.
- Line 97 renders at the same decimated rate even though the run is headless.
- Line 98 disables Fabric to keep state reads consistent with USD/PhysX behavior
  used elsewhere in this imported-asset workflow.

### Lines 101–110: `_targets`

- Line 101 declares deterministic target generation for any rectangle.
- Line 102 unpacks XY bounds.
- Line 103 builds the gripper-base target Z from surface height, can half-height,
  and 5 cm clearance.
- Lines 104–110 return the center and four corners. Reaching only the center would
  not prove that the complete declared region lies inside the positional
  workspace.

### Lines 113–159: `_run_region`

- Line 113 begins one independent region run.
- Line 114 creates a fresh environment config.
- Line 115 moves the fixed robot root to that region's staging position.
- Line 116 uses the identity quaternion, i.e. authored yaw zero.
- Lines 117–118 construct and reset the live Isaac environment.
- Line 119 retrieves the articulation object.
- Line 120 finds the controlled body index by name instead of relying on an
  unstable numeric ordering.
- Line 121 resolves all seven arm indices explicitly.
- Line 122 initializes results.
- Line 123 begins a `try/finally` so simulator resources are released.
- Line 124 loops over the five deterministic targets.
- Line 125 creates the target on the same Torch device as simulation.
- Line 126 creates the robot-root world position.
- Line 127 subtracts root translation because this Isaac action interprets an
  absolute position command in robot-root coordinates for this articulation.
  An earlier diagnostic run exposed this convention: sending world coordinates
  displaced every achieved point by exactly the base translation.
- Lines 128–129 repeatedly step the absolute command until the controller settles.
- Line 130 reads the controlled body's world position and removes only the
  parallel-environment origin, leaving authored kitchen/world coordinates.
- Line 131 computes Euclidean position error against the world target.
- Lines 132–133 read arm positions and soft limits.
- Lines 134–135 compute the smallest distance to any lower and upper joint limit.
- Line 136 rejects NaN/Inf states.
- Lines 137–149 record target, root-frame command, achieved world point, error,
  finite state, limit margins, and pass/fail.
- Lines 150–151 always close the environment.
- Lines 152–159 summarize base pose, region, tolerance, individual results, and
  whether all points passed.

### Lines 162–182: complete report

- Line 162 declares the executable entry point.
- Lines 163–169 describe the method and run both regions.
- Lines 170–173 resolve/create the output path and write JSON.
- Lines 174–175 print the same evidence and saved location.
- Lines 178–182 execute only as a script and always close the Isaac application.

The pick sweep passed 5/5 points with 0.131 mm worst error and 0.893 rad minimum
joint-limit margin. The place sweep passed 5/5 with 0.461 mm worst error and
1.089 rad minimum margin.

## Script 3: `examples/render_genmanip_kitchen.py`

This renderer predates the numerical audit but was updated to consume the audited can and base poses.

### Lines 1–32: material-safe startup

- Lines 1–4 provide licensing and the module contract.
- Lines 6–12 import CLI, copying, paths, traceback, OpenCV, and `AppLauncher`. OpenCV is intentionally loaded before Isaac Lab to avoid a shared-library ordering conflict in this environment.
- Lines 14–16 explain why MDL paths must be registered before Kit starts: EBench uses `KooPbr` modules, and missing paths produce fallback or black materials.
- Lines 17–19 locate the repository, EBench assets, and shared MDL library.
- Lines 20–28 locate scene-local generated materials.
- Lines 29–32 combine Isaac, EBench, scene, and caller-supplied MDL paths without discarding the caller's environment.

### Lines 35–61: CLI and simulator imports

- Lines 35–44 define frame count, MP4 path, can pose, optional still, and robot visibility.
- Lines 45–48 add Isaac options, parse them, enable cameras, and start Kit.
- Lines 50–57 import NumPy, Torch, simulation, assets, scene, camera, and config APIs after Kit startup.
- Lines 59–61 import RoboLab paths, video writing, and the mobile Kinova configuration.

### Lines 64–91: authored assets and audited poses

- Lines 64–75 construct the exact complete kitchen USD path.
- Line 76 selects RoboLab's known-scale YCB tomato-soup-can USD.
- Lines 77–82 place the can at each audited region center. Each Z is measured worktop height plus the 51.0185 mm can half-height, so the bottom rests on the plane.
- Lines 83–86 duplicate the audited base positions for visualization.
- Lines 87–91 choose a camera target for the full room, pick workspace, or place workspace.

### Lines 94–168: `GenManipKitchenSceneCfg`

- Lines 94–101 define the scene and spawn the original kitchen without rescaling or manually rebuilding its furniture.
- Lines 102–133 add three neutral, rough visual walls because the published dishwasher USD has no complete room shell. The camera-facing side stays open.
- Lines 134–144 conditionally spawn the can as a gravity-disabled kinematic rigid body. This makes a stable before/after scene; it is not evidence of a physical grasp.
- Lines 145–151 deep-copy and stage the cobot at the selected audited pose, identity yaw, and a fixed root for a stable still.
- Lines 152–168 define a 1280×720 RGB/depth pinhole camera. The 16 mm focal length gives a wide room view and the 5 cm near plane avoids clipping legitimate foreground geometry.

### Lines 171–182: `camera_eye`

- Line 174 uses the middle of the camera arc for a still and normalized time for video.
- Lines 175–181 create a short 16-degree interior-safe arc with small radius and height variation.
- Line 182 returns the eye point as a batched, device-local Torch tensor.

### Lines 185–227: `annotate`

- Line 186 converts the RGB result into a contiguous three-channel array.
- Lines 187–211 draw the renderer title and a task/robot-specific subtitle.
- Lines 212–225 draw a frame-progress bar.
- Line 227 returns the annotated frame.

### Lines 230–307: rendering and shutdown

- Lines 230–232 reject a missing kitchen asset early.
- Lines 234–244 create a 60 Hz, single-environment, non-replicated scene with eager sensor updates and Fabric disabled.
- Lines 245–253 reset, retrieve the camera, construct its look-at target, and open a 30 fps video.
- Lines 255–259 warm up eight frames so scene, materials, sensors, and articulation state settle.
- Lines 261–269 update camera pose and advance simulation for every output frame.
- Lines 270–282 read RGB and validate the first frame using dynamic range and finite depth, catching blank renders.
- Lines 283–290 annotate, encode, and optionally save the final still with checked RGB-to-BGR conversion.
- Lines 292–295 report provenance and guarantee video flushing.
- Lines 298–304 print a traceback and force a failing process status after any render error.
- Lines 305–307 use `os._exit(0)` after success because Isaac Sim 5.0 can hang during plugin shutdown after this MDL-heavy imported scene has already flushed its output.

## Robot support: `robolab/robots/kinova_mobile_cobot.py`

### Lines 1–27: composition source

- Lines 1–24 import Isaac managers, sensors, Torch, the established Kinova/Robotiq configuration, and planar-base helpers.
- Line 27 instantiates that trusted fixed Kinova configuration once. The mobile version clones it instead of duplicating seven-joint and gripper definitions.

### Lines 30–89: `HolonomicBaseVelocityAction`

- Lines 30–47 define a body-frame `[vx, vy, wz]` action and allocate raw, bounded, applied, speed-limit, and per-step acceleration-limit tensors.
- Lines 49–59 expose the three-dimensional action and its raw/processed buffers.
- Lines 61–63 save and speed-clamp incoming commands.
- Lines 65–68 apply acceleration limiting so policy commands cannot create instantaneous base velocity changes.
- Lines 69–82 read the floating root, extract yaw, remove roll/pitch, pin height to 0.468 m, and convert the body twist to world velocity. This models an ideal planar holonomic base.
- Lines 84–89 clear action history at reset.

### Lines 92–100: base limits

- Lines 92–95 connect the config to the robot and action class.
- Lines 96–100 set conservative indoor limits: 0.6 m/s, 1.0 rad/s, 1.2 m/s², 2.0 rad/s², and a 0.468 m planar root height.

### Lines 103–147: `KinovaMobileCobotCfg`

- Lines 103–110 clone the fixed Kinova and substitute the composed mobile-cobot USD.
- Line 111 makes the root floating for navigation tasks.
- Lines 112–126 set the authored root height and an upright seven-joint mobile home pose.
- Lines 128–147 attach a 640×360 RGB/depth navigation camera on the local `-Y` mast side, facing outward and slightly downward. The inline comment records why the previous orientation looked through the chassis.

### Lines 150–166: policy action configuration

- Line 154 adds the three-value base velocity action.
- Lines 155–160 add seven ordered arm joint-position actions without default offsets.
- Lines 161–166 add the established binary Robotiq open/close action.

The nominal policy interface is `[base_vx, base_vy, base_wz, arm_joint_1..7, gripper]`. The reach audit used Cartesian IK instead because it tests workspace feasibility without freezing the eventual policy adapter.

## Visual evidence and how to interpret it

- `pick_scene.png` shows the can on the island and the robot at the pick staging pose.
- `place_scene.png` shows the can on the opposing counter and the robot at the place staging pose.
- `pick_place_scenes.png` puts both 1280×720 Isaac renders side-by-side.
- `milestone0_staging_map.svg` is a scale-accurate top-down schematic of valid bounds, 0.30 m base footprints, aisle edges, and the staging transition.

The scene images prove authored scale and base staging. Their arms remain in the upright home configuration. The IK audit ran in a separate minimal environment, so the images are not arm-trajectory collision evidence.

## Why the important choices were conservative

| Decision | Choice | Reason |
|---|---:|---|
| Surface grid | 2 cm | Resolves the approximately 68 mm can and fixture cut-outs efficiently. |
| Horizontal threshold | `normal.z >= 0.985` | Tolerates mild mesh imperfection while rejecting walls/slopes. |
| Plane tolerance | 4 mm | Separates worktops from sinks, cooktops, props, and nearby layers. |
| Edge erosion | about 8 cm | Exceeds the can radius and provides a placement buffer. |
| Base footprint | 0.30 m radius | Matches the authored drive-base extent. |
| Residual fixture clearance | 0.06 m | Keeps the footprint outside either measured edge. |
| IK samples | center plus four corners | Tests the complete declared rectangle, not one lucky point. |
| IK mode | position-only DLS | Tests reach without inventing a grasp orientation. |
| Root during IK | fixed | Prevents base motion from hiding arm infeasibility. |
| Pass threshold | 15 mm | Suitable preliminary band; measured worst case was below 0.5 mm. |

## Next step: Milestone 1

The next step should be a **collision-aware scripted pick-and-place**, before any policy training:

1. Author a simple PhysX collision overlay for floor, island, opposing counter, sink/cooktop exclusions, and walls. Keep the high-detail EBench USD for rendering.
2. Make the can dynamic with gravity, mass, friction, contact, and resets sampled inside the audited pick bounds.
3. Define 6D pre-grasp, grasp, lift, transport, pre-place, place, retreat, and home transforms with a valid gripper orientation.
4. Use cuRobo, RMPflow, or equivalent collision-aware motion generation instead of direct position-only IK.
5. Execute a scripted oracle: stage base, approach, close/grasp, lift, move base, approach place, release, and retreat.
6. Add observations, actions, reset randomization, safety conditions, and success/termination definitions.
7. After the scripted oracle is reliable, record demonstrations and connect policy-specific adapters to the same task contract.

The next visual checkpoint should show the arm at actual 6D pre-grasp and pre-place configurations inside the collision-enabled kitchen, with collision geometry/debug contacts visible. That will combine the currently separate staging render and positional-IK evidence.
