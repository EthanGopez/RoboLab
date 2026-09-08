# Fixed-base CuRobo grasp and local-place milestone

## Result

The harder far-side kitchen case now passes the requested gate:

- 40/40 successful grasps
- 37/40 successful local placements
- required threshold: at least 35 of each
- report: `output/genmanip_kitchen/fixed_base_curobo/gate_40.json`

The gate uses a deterministic golden-angle schedule whose pick poses expand
from the center to 95% of the valid pick-region radius. The local place goal is
fixed at the center of the valid west-island region so the gate isolates
robustness to initial object-pose perturbation.

## Planner boundary and CuRobo version

Task geometry and phase semantics live behind the planner-neutral
`FixedBaseMotionPlanner` protocol. The current backend uses the CuRobo v1
`MotionGen` API at the commit already supported by Isaac Lab SkillGen
(`ebb7170`). CuRobo v2 is a substantial API rewrite and does not currently ship
an official Isaac Sim example, so it is not coupled directly into RoboLab. A v2
adapter can replace the backend later without changing the task or executor.

The reference `cobot_moma_isaacsim` checkout contains the mobile-base USD,
transport, and sensor-facing simulator integration, but no CuRobo planner
implementation to reuse.

## Files

- `robolab/tasks/mobile_manipulation/fixed_base_manipulation.py`
  defines planner-neutral poses, trajectories, phases, staging geometry, and
  the deterministic 40-trial schedule.
- `robolab/tasks/mobile_manipulation/curobo_fixed_base_planner.py`
  wraps CuRobo MotionGen, converts kitchen geometry into base-frame cuboids,
  and adds/removes held-can collision spheres.
- `robolab/tasks/mobile_manipulation/fixed_base_curobo_oracle.py`
  executes open, elevated pre-grasp, approach, bilateral-contact close, lift,
  retract, pre-place, place, release, and retreat.
- `assets/robots/kinova_mobile_cobot/curobo/`
  contains the planner-only seven-joint URDF, robot configuration, and
  conservative collision spheres.
- `examples/validate_genmanip_kitchen_fixed_base_curobo.py`
  runs targeted reproductions or the full checkpointed gate.
- `tests/test_fixed_base_manipulation.py`
  checks schedule bounds, base-frame transforms, robot schema/TCP agreement,
  and inflated staging clearance.

## Important design choices

1. **Ground-truth object pose.** Each trial reads the settled can pose directly
   from Isaac Sim. Perception is intentionally deferred.
2. **Fixed base with audited clearance.** The base is staged at
   `(-1.06, 0.14, 0.468, pi)`. Its center remains 0.36 m from the island face,
   exceeding the 0.30 m footprint plus 0.05 m safety margin.
3. **Elevated side pre-grasp.** The pre-grasp is 6 cm above the can before a
   diagonal allowed-contact approach. This avoids falsely declaring the
   conservative spherical hand envelope in collision with the worktop.
4. **Collision-aware carrying.** The static RoboLab collision overlay becomes
   CuRobo cuboids. After lift, three attached spheres cover the held can using
   its measured post-settle can-in-TCP transform.
5. **Simple gripper semantics.** The executor retains RoboLab's binary gripper
   action. It requires four frames of bilateral fingertip contact before using
   the existing post-contact fixed-joint transport assist. This gate is about
   planning and is not an unassisted physical-grasp fidelity claim.
6. **Placement success.** Success means the released dynamic can is upright at
   support height, settled, inside the valid island region, the gripper is
   released, and the robot is collision-free. Point-target error is logged as
   a diagnostic but is not the task predicate.
7. **Post-settle attachment transform.** The held-object transform is measured
   after PhysX settles the assist joint. Measuring it before settle caused
   inconsistent carried-object and placement targets.

## Validation

Focused tests:

```bash
PATH=/home/robocup/miniconda3/envs/env_isaaclab/bin:/usr/bin:/bin \
CUDA_HOME=/home/robocup/miniconda3/envs/env_isaaclab \
TORCH_CUDA_ARCH_LIST=12.0 \
/home/robocup/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  tests/test_fixed_base_manipulation.py
```

Result: `5 passed`.

Full gate:

```bash
TERM=xterm \
PATH=/home/robocup/miniconda3/envs/env_isaaclab/bin:/usr/bin:/bin \
CUDA_HOME=/home/robocup/miniconda3/envs/env_isaaclab \
TORCH_CUDA_ARCH_LIST=12.0 \
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/validate_genmanip_kitchen_fixed_base_curobo.py \
  --headless --trials 40 --num-seeds 8 \
  --output output/genmanip_kitchen/fixed_base_curobo/gate_40.json
```

All 40 grasps succeeded. Three trials (14, 18, and 25) reached the placement
target within 8 mm before release but were kicked or tipped outside the valid
region during unconstrained opening. The minimal follow-up is to open the
fingers while the verified transport constraint still holds, then remove the
constraint only after the fingertips clear the can.

## External references

- [Isaac Lab SkillGen and CuRobo setup](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html)
- [NVlabs CuRobo](https://github.com/NVlabs/curobo)
- [CuRobo v2 Isaac Sim integration status](https://github.com/NVlabs/curobo/discussions/616)
