# Milestone 2 kitchen manipulation oracle

## Status

The supported benchmark path is now `strict_post_contact_constraint`: it first
requires bilateral contact, sufficient opposing normal force, low tangential
slip, joint tracking, and a stable flange-to-can transform. Only after that
qualification does a fixed constraint preserve transport. The exact seed-101,
ten-environment reproduction passed 10/10; its result is
`output/genmanip_kitchen/milestone2/strict_post_contact_seed101_10.json`.
All terminal constraints were detached, the step range was 1,602--1,733, and
the worst final XY placement error was 0.00251 m. The strict path has not yet
run its 100-trial gate; the 100/100 result below belongs to the older, looser
assisted predicate.

This is the robot-agnostic boundary: scene support, contact semantics, release,
placement, and success are task predicates. The scripted Kinova oracle remains
an embodiment-specific policy and supplies its flange name, fingertip bodies,
pad offsets, and gripper joints to the reusable physical-grasp monitor.
`unassisted_friction` remains an optional hardware-fidelity experiment, not a
requirement for every robot supported by RoboLab.

A prior, looser `fixed_joint_after_bilateral_contact` implementation passed its
randomized assisted-grasp gate:

- 100 requested trials;
- 100 completed trials;
- 100 successful trials;
- seed `101`, ten vectorized environments per batch;
- randomized can starts inside the reach-conditioned island region;
- randomized targets inside the reach-conditioned counter region;
- no reported drop, fixture penetration, self/workspace collision, false
  success, or phase timeout;
- every terminal grasp constraint was detached;
- terminal target error ranged from 0.00063 m to 0.01731 m.

The machine-readable result is:

```text
output/genmanip_kitchen/milestone2/oracle_gate_100_seed101.json
```

That historical gate uses `fixed_joint_after_bilateral_contact`: the controller must first
obtain bilateral fingertip contact at the configured force threshold, then a
pre-authored fixed joint preserves the grasp during transport. The
`unassisted_friction` path remains available, but it has not passed this gate
and must not be described as reliable yet.

## Implementation

The main controller is
`robolab/tasks/mobile_manipulation/kitchen_manipulation_oracle.py`.
It supplies a complete 12-value RoboLab action each control step:

- three mobile-base velocity commands;
- seven absolute end-effector pose values consumed by differential IK;
- two driven Robotiq outer-knuckle commands; the remaining four joints follow
  authored mimic relationships.

The controller runs independently for every vectorized environment. Its
states are:

```text
settle
  -> pregrasp -> descend -> close -> secure -> lift -> carry_settle
  -> traverse -> carry_reposition -> carry_reorient
  -> preplace -> place -> handoff_settle -> open -> retreat -> success
```

`failure` is terminal and stores the phase, phase step, can pose and speeds,
TCP target error, placement error, settle counter, and grasp-constraint state.
Every nonterminal phase has a finite timeout.

Task success is conjunctive. It requires the free can to be stably supported
inside the physical counter region and within 0.025 m of the sampled target,
the gripper to be released, the base to reach its retreat stage, and the robot
and can to remain inside the collision-free workspace.

## Physical release sequence

The release transition is deliberately ordered as follows:

1. Descend until the desired can center is inside the counter bounds, within
   target tolerance, and no more than the release clearance above the surface.
2. Store the current arm pose, zero residual assisted rigid-body velocity,
   detach the grasp constraint, and open the gripper.
3. Let the now-dynamic can settle under PhysX gravity, collision, and friction.
4. Require stable counter support, release, target accuracy, and bounded linear
   and angular velocity for consecutive frames.
5. Retreat the base while holding the arm configuration in the robot frame.

The earlier ordering waited for a counter-contacting can to become stationary
while it was still fixed to a driven flange. One seed-101 trial exposed the
resulting overconstraint with approximately 1.03 m/s linear and 7.77 rad/s
angular can velocity. Releasing before the settle test removed that solver
fight; the exact ten-trial reproduction then passed 10/10 before the 100-trial
gate was run.

## Geometry domains

The broad regions remain the physical support contracts:

```text
PICK_REGION  = x [-0.06, 0.04], y [ 0.10, 0.25]
PLACE_REGION = x [ 1.76, 1.86], y [-0.02, 0.22]
```

The present side-grasp controller samples conservative, reach-conditioned
execution domains:

```text
ORACLE_PICK_REGION  = x [-0.03, 0.04], y [0.12, 0.23]
ORACLE_PLACE_REGION = x [ 1.79, 1.83], y [0.09, 0.11]
```

The distinction prevents a controller-conditioning limitation from being
misrepresented as missing physical support geometry. Expanding the execution
domain is future controller work.

## Gate runner

`examples/validate_genmanip_kitchen_oracle_gate.py` runs vectorized randomized
trials. It records the initial can position, sampled target, phase history,
terminal state, final state, and failure diagnostics for every trial.

Results are atomically checkpointed after every batch. `--fail-fast` stops
after the first batch containing a failure, so a long run preserves its useful
evidence even if it cannot reach 100/100.

Reproduce the completed gate with:

```bash
PYTHONPATH=/home/robocup/Downloads/RoboLab \
  /home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/validate_genmanip_kitchen_oracle_gate.py \
  --headless \
  --grasp-assist \
  --trials 100 \
  --batch-size 10 \
  --seed 101 \
  --max-steps 2400 \
  --fail-fast \
  --output output/genmanip_kitchen/milestone2/oracle_gate_100_seed101.json
```

The gate defaults to camera-free and recorder-free execution. Use
`--enable-cameras` and `--record-data` only when those outputs are explicitly
needed.

## Validation completed

Static/schema suite:

```bash
.venv-isaac50/bin/python -m pytest -q \
  tests/test_mobile_cobot.py \
  tests/test_genmanip_kitchen_physics_schema.py \
  tests/test_genmanip_kitchen_render.py \
  tests/test_replay_env_config.py
```

Result: 22 passed on the current strict-grasp code.

Physics simulation regression suite:

```bash
PYTHONPATH=/home/robocup/Downloads/RoboLab \
  /home/robocup/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/test_genmanip_kitchen_physics_sim.py
```

Result: 5 passed. This retains the Milestone-1 can-drop/settle and robot
staging-pose checks after the manipulation changes.

## Next step

The primary next gate is the policy-ready interface: normalize base twist, TCP
pose, and open/close semantics; expose observations without oracle phase; and
record demonstrations from this deterministic reference. Learned policies can
then be evaluated on the same reset seeds and task predicates.

`unassisted_friction` is a separate embodiment-validation track. It is valuable
when a benchmark specifically claims Robotiq contact fidelity, but it should
not block a robot- and policy-agnostic RoboLab task. Every result must retain
its explicit `grasp_mode` so assisted transport is never presented as a
friction-only grasp.
