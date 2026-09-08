# XGRIDS2 dining-plate-to-sink task

> Historical implementation walkthrough. Use the configurable runner described
> in [XGRIDS2_CODEBASE_GUIDE.md](XGRIDS2_CODEBASE_GUIDE.md) for current runs.
> The legacy command remains as a thin compatibility launcher.

**Last updated:** 2026-08-30

**Task:** pick a plate from the dining table, carry it through the XGRIDS2 lab,
and place it inside the kitchen sink.

## 1. What this implementation is

This is a staged, physics-executed test oracle:

```text
dynamic plate on clean table collider
  -> CuRobo rim pregrasp and approach
  -> measured fingertip contact
  -> temporary fixed retention joint
  -> CuRobo lift and staged vertical carry posture
  -> plate-inflated A* base route
  -> CuRobo reorientation and lowering into modeled basin
  -> retention release, retreat, settle, and geometric validation
```

CuRobo plans the collision-aware arm segments, including the calibrated joint
anchors used to keep the carry roll on one continuous-joint branch. Isaac Sim
position drives execute those joint waypoints; the validation code does not
teleport the manipulation motion. A separate closed-loop holonomic controller
moves the base along an A* route.

The fixed joint is an explicit temporary retention aid. It is enabled only
after a real fingertip-to-plate contact force of at least 1 N is measured and
is disabled before the plate settles in the sink. This task does not claim a
fully frictional two-finger carry.

## 2. Files and responsibilities

### Geometry contract

`robolab/tasks/mobile_manipulation/xgrids2_lab_plate_sink_geometry.py` owns:

- the plate USD, measured diameter, thickness, mass, spawn, and support height;
- the reachable near-edge pick region and intentional rim overhang;
- the audited sink center, yaw, opening, counter height, and basin-bottom height;
- clean counter slabs, basin bottom, and four basin wall collision boxes;
- the expanded vector-map physics region needed to reach the kitchen;
- fixture masks that remove coarse map segments around the clean table/sink;
- a collision-inflated A* grid and runtime route-clearance checks;
- the nine-sphere approximation used for the held plate in CuRobo.

The plate diameter is 0.2976294 m. Its center remains supported by the table,
but the robot-facing rim overhangs enough for a lower fingertip to reach under
it without intersecting the tabletop. The sink opening is 0.56 by 0.42 m, so
the horizontal plate fits with a positive wall margin.

### Isaac Sim task

`robolab/tasks/mobile_manipulation/xgrids2_lab_plate_to_sink_task.py` creates:

- the vendor XGRIDS2/NuRec scene as the visual layer;
- an expanded invisible floor and vector-map collision layer;
- a rigid, gravity-enabled, contact-sensed plate;
- a clean dining-table support collider;
- four counter slabs, a basin bottom, and basin walls;
- plate reset sampling and the disabled retention joint;
- a plate-only robot configuration using the proven kitchen IK arm gains.

The visual reconstruction remains collision-free. PhysX and CuRobo share the
same simple, explicit fixture geometry, avoiding the noisy scan/proxy collision
problem fixed earlier in the XGRIDS2 work.

### Planner adapter

`robolab/tasks/mobile_manipulation/curobo_fixed_base_planner.py` accepts an
optional held-object sphere set. Existing can callers keep the original can
spheres; the plate oracle passes its disk-shaped nine-sphere approximation.
The measured object-in-tool transform is retained, so desired plate poses are
converted into correct tool poses instead of treating the TCP as the plate
center.

### Task oracle

`robolab/tasks/mobile_manipulation/xgrids2_lab_plate_to_sink_curobo_oracle.py`
implements the phase machine and all physical gates.

The pick sequence plans a rim pregrasp, approaches with hand contact enabled,
closes slowly, and polls the independently named fingertip contact sensors. A
single measured pad contact is sufficient to enable the user-approved
post-contact retention aid. Proximity alone can never enable it.

After lift and retract, the plate is rolled toward vertical in several CuRobo
segments. A single large rotation had no IK solution at full reach, so the
working sequence rolls 20% and 40%, moves the partially tilted plate inward,
then uses three collision-checked CuRobo joint anchors for the final roll and
tuck. A disk is symmetric under its normal direction; the chosen normal and
anchors avoid winding continuous Kinova joint 3 into its `+2*pi` guard. The
reachable 70% roll is 63 degrees from horizontal and already reduces the
navigation footprint substantially.

The transport pose keeps the plate center 0.38 m forward and 0.62 m above the
robot base frame. Its measured total footprint must stay below 0.47 m. A 0.47 m
planning radius inflates the A* route, and every executed base pose is checked
again against that same clearance contract.

The measured carry-posture comparison permits 1 mm of numerical tolerance;
the route planner and runtime map-clearance checks remain fixed at 0.47 m. This
avoids rejecting a 0.470037 m floating-point measurement as a different
navigation envelope.

The base is holonomic, so it translates along route segments while holding a
constant carry yaw. Turning to face every short grid segment previously pumped
the retained plate and arm. Only the final sink alignment changes yaw.

At the sink, a fresh fixed-base CuRobo instance uses the measured sink-stage
base pose and the same held-plate spheres. It progressively returns the plate
to horizontal and moves over the basin center. Because the arm cannot reach a
reliable low release pose, retention is transferred to the base at coincident
frames, the gripper opens, and the plate is guided vertically in 2 mm fixed-
joint-frame increments to 0.025 m above its settled center. The joint is then
disabled, the arm retreats, and the plate settles under gravity.

Success requires all of the following:

- the full plate disk lies inside the opening with margin;
- final Z agrees with the modeled basin bottom within 3 cm;
- the plate normal is within 20 degrees of vertical;
- linear speed is below 0.05 m/s and angular speed below 0.20 rad/s.

The center error is recorded as a diagnostic, but it is not a second success
gate: full disk containment is the physical requirement for placing a plate
inside this rectangular sink.

### Registration and runner

`robolab/registrations/kinova_mobile/auto_env_registrations_xgrids2_lab_plate_sink.py`
registers `XGRIDS2LabPlateToSinkKinovaMobileCurobo`. It builds the environment
directly from the imported task class. The generic file loader was deliberately
not used: it executed the task module twice, and the expanded scene's hundreds
of generated map fields caused Kit to terminate during duplicate config-class
generation.

`examples/validate_xgrids2_lab_plate_to_sink.py` supports four gates:

```bash
# Contact, retention, and lift only
python examples/validate_xgrids2_lab_plate_to_sink.py --stage pick --headless

# Pick plus vertical carry and route traversal
python examples/validate_xgrids2_lab_plate_to_sink.py --stage carry --headless

# Pick, carry, sink insertion, and release validation
python examples/validate_xgrids2_lab_plate_to_sink.py --stage place --headless

# The same end-to-end task, labeled as the full evidence run
python examples/validate_xgrids2_lab_plate_to_sink.py --stage full --headless
```

Add `--record-video` for synchronized dining, moving third-person, and sink
views. Without video, the runner uses small audit cameras to reduce validation
cost while still producing a result JSON and evidence montage.

## 3. Bugs found and how they were fixed

### Thin curved plate did not produce stable bilateral contact

One pad commonly reached 20–25 N while the other missed as the curved plate
slid. Requiring bilateral contact made a real rim touch fail nondeterministically.
The accepted test contract is therefore measured contact on either fingertip,
followed by temporary fixed retention. The result JSON states this exactly as
`measured_fingertip_contact_then_fixed_joint_retention`.

### Full vertical rotation failed IK

Rolling the plate 90 degrees at its original full-reach position had no valid
collision-free solution. The roll was split, and the partially tilted plate is
moved inward before completing the reachable 80% orientation.

### Equivalent disk orientation crossed a continuous-joint boundary

The first vertical normal made joint 3 approach `+2*pi`. Flipping the disk
normal produces the same physical plate orientation for transport but keeps
the arm on a valid continuous-joint branch.

### Original route radius described the robot, not the held plate

The exact projected plate rim and every robot body are now measured around the
base after tucking. Planning uses a 0.47 m inflated footprint, and the route's
minimum map clearance is checked against the required 0.55 m including safety
margin and half proxy width.

### Base yaw-following excited the arm

The generic waypoint follower rotated toward every A* segment. That behavior
was harmless for a compact can but repeatedly changed the retained plate's
inertial load. Plate traversal now holds its transport yaw while translating
and performs only the final sink alignment.

The first plate route exposed all 57 eight-connected A* cells directly to the
follower. The arm stayed stable on straight cells but drifted at the first
west-to-southwest switch near `x=0.92 m`. The route is now greedily shortened
to four farthest-visible segments; every shortcut is sampled at 2 cm spacing
and must retain the full inflated clearance. Plate-only traversal is capped at
0.06 m/s with 0.06 m/s^2 acceleration and 0.12 m/s^3 jerk. This keeps
the explicitly locked transport state stable while leaving enough distance in
the 3,600-step traversal budget to reach the sink.

### Damping-only tuning left the loaded posture too compliant

The first plate configuration inherited can stiffness of 120/80 N m/rad and
raised damping to 30/24. A carry run showed joint 1 drifting over 0.3 rad from
its target, then moving 0.074 rad in one physical control tick. Raising the
quality limit would have hidden visible dancing. Instead, the plate task now
uses the kitchen IK gains of 300/240 stiffness and 30/24 damping and requires a
90-tick stationary loaded settle before traversal begins.

Steady base translation then showed large shoulder and wrist tracking error
even with paired peak effort limits. The cause was the off-axis retained disk
reacting against direct floating-root motion, not ordinary arm torque demand.
After physical pick and CuRobo tuck, the enabled retention joint is therefore
retargeted at coincident measured frames from the flange to `base_link` for
traversal. The gripper opens so its fingers can leave the now-base-retained
plate without a physical self-jam. Within this explicit transport assist, the
runner interpolates the unloaded arm to its stable default posture in bounded
0.005 rad state increments with zero velocity, then holds that state after each
base control tick. This prevents the floating-root controller from re-solving
even an unloaded arm. At the sink the same bounded interpolation returns the
open arm to its saved carry posture on the same continuous-joint branches. It
then closes around the plate and retargets retention from the base to the
flange at another coincident measured frame. Physical drive execution is used
again for all sink manipulation; the can task is unchanged.

### Loose loaded-waypoint tracking caused a catch-up spike

The first high-stiffness run reduced the maximum one-tick displacement but
failed during the 80% roll at 3.96 rad/s. Its diagnostics showed that the
physical arm had been allowed to trail the interpolated CuRobo target by about
0.3 rad. The high-gain drive then corrected that accumulated error quickly.
The plate executor now limits intermediate tracking error to 0.040 rad and
segment endpoint error to 0.045 rad. Contact-loaded retract uses a slightly
wider 0.050/0.055 rad convergence band. During lift only, a 0.100 rad waypoint
band provides the position-error torque needed to break static tabletop
contact, followed by a 0.065 rad endpoint gate. The physical step and speed
gates remain unchanged. This fixes the control contract instead of hiding the
transient by raising those quality thresholds.

The accurately tracked branch also made the marginal 75--80% roll targets fail
intermittently. The final carry roll is now 70%, still within the physically
measured 0.47 m footprint gate, and the plate planner uses 16 seeds. One later
run still selected an equivalent carry-tuck solution beside joint 3's
`+2*pi` boundary. The last three roll/tuck poses therefore use measured,
collision-checked CuRobo joint anchors from the successful interior branch.
They are still planned and physically drive-executed; only the ambiguous IK
branch choice is removed.

### The final sink descent was unreachable and the original success gate was too strict

The first complete traversal centered and flattened the plate over the basin,
then failed IK for an extra target only 0.055 m above the modeled settled
center. Releasing from the preceding 0.10 m target worked once but tipped on a
repeat, so the high drop was not accepted as robust. The sink staging pose was
moved 0.038 m closer while retaining 0.552 m of measured route clearance
against a 0.550 m requirement. The low target still failed with and without
the conservative nine-sphere held-plate approximation, proving that it was a
real arm-workspace limitation rather than a false collision.

The final stand-in therefore uses the user-approved retention aid for the
unreachable part only. After CuRobo has made the plate horizontal and centered
it over the opening, retention is transferred from the flange to `base_link`
at coincident measured frames. The gripper opens, and the joint's local frame
is lowered vertically in 2 mm increments until the plate center is 0.025 m
above its settled height. A contained-pose gate runs before the constraint is
released. PhysX then handles the remaining short drop and basin contact. This
phase is labeled `basin_guided_lowering` in the result rather than being
misrepresented as CuRobo motion.

One high-release physical run settled flat and fully inside the basin, but its
center drifted 0.0527 m from the nominal center and missed an unrelated 0.05 m
radial cutoff. The rectangular full-disk containment test is the correct
geometric contract, so the redundant radial cutoff was removed. Height,
orientation, and velocity gates remain unchanged.

### Sink physics did not exist in the original can-task crop

The dining-to-snack collision overlay intentionally stopped before the kitchen.
This task builds an isolated expanded map region and a real basin rather than
altering the proven can scene or pretending the sink is only a target point.

## 4. Regression gates

`tests/test_xgrids2_lab_plate_sink_schema.py` checks:

- the plate asset exists and the pick overhang keeps its center supported;
- the full plate fits inside the sink opening;
- counter, basin-bottom, and all wall colliders are present;
- the held-plate A* route meets its required clearance;
- the dynamic plate and expanded collision map are configured;
- the task validates and registers under the expected environment name.

The existing XGRIDS2 can schema and robustness tests remain part of the suite,
because plate-specific generalizations must not change the proven can task.
The combined static suite currently passes all 22 tests.

## 5. Validation evidence

Every run writes a stage result JSON containing phase order, planner timings,
trajectory sizes, measured arm endpoints, maximum physical joint step/speed,
route-clearance status, base errors, plate pose, and failure diagnostics.

The accepted headless end-to-end run is:

```text
output/xgrids2_lab/plate_to_sink_guided_validation/
  plate_to_sink_results.json
  plate_to_sink_montage.png
```

It completed with `passed=true`, `completed_stage=full`, and `full_success`.
The result records:

- all route-clearance checks valid;
- 0.44898 m measured carry radius against the 0.47 m contract;
- 64 bounded guided-lowering increments;
- 0.000883 m final XY center error;
- `0.9999994` upright cosine;
- 0.02839 rad maximum measured arm step and 1.4032 rad/s maximum arm speed;
- 1.33 micrometers final base-position error and 7.21 microradians yaw error.

The final plate center was `(-2.43627, 2.23808, 0.73585)` m. It passed full
disk containment, bottom-height, orientation, and settled-velocity gates.
The combined plate, existing can-task schema, and XGRIDS2 robustness suite
passed all 22 tests after this run.

The accepted synchronized video run is:

```text
output/xgrids2_lab/plate_to_sink_video_evidence/
  plate_to_sink_task.mp4
  plate_to_sink_results.json
  plate_to_sink_montage.png
```

That independent run also completed with `passed=true` and `full_success`.
The MP4 is 1440 by 540 pixels, 537 frames at 8 fps, and 67.125 seconds long.
Frames at the beginning, quarter points, midpoint, and end all decoded during
verification. The fixed dining/sink third-person view, navigation view, and
wrist view remain synchronized throughout the recording.
