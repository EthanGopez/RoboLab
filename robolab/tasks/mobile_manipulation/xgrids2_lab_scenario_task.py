"""Isaac scene/task generation for configurable XGRIDS2 scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from pxr import Gf, Sdf, Usd, UsdPhysics

from robolab.constants import PACKAGE_DIR
from robolab.core.task.task import Task
from robolab.robots.kinova_mobile_cobot import MOBILE_KINOVA_FLANGE_BODY
from robolab.tasks.mobile_manipulation.kitchen_can_transfer_task import (
    STATIC_MATERIAL,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_can_transfer_task import (
    _is_manipulation_table_segment,
    _round_support_cfg,
    _support_cfg,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
    COLLISION_SEGMENTS,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_manipulation_geometry import (
    Tabletop,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_navigation_task import (
    XGRIDS2LabPhysicsSceneCfg,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import (
    XGRIDS2ObjectSpec,
    XGRIDS2ScenarioSpec,
    XGRIDS2SupportSpec,
    get_xgrids2_scenario,
)


SCENARIO_GRASP_JOINT_PREFIX = "oracle_grasp_joint"


def scenario_grasp_joint_name(object_name: str) -> str:
    return f"{SCENARIO_GRASP_JOINT_PREFIX}_{object_name}"


def _physics_object_cfg(spec: XGRIDS2ObjectSpec) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{spec.name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(PACKAGE_DIR, spec.usd_relative),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=spec.kinematic,
                disable_gravity=spec.kinematic,
                linear_damping=0.08,
                angular_damping=0.10,
                max_linear_velocity=4.0,
                max_angular_velocity=30.0,
                max_depenetration_velocity=0.5,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.004,
                stabilization_threshold=0.001,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.003,
                rest_offset=0.0,
                torsional_patch_radius=0.02,
                min_torsional_patch_radius=0.005,
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(*spec.center_xy, spec.spawn_z),
            rot=spec.spawn_quaternion_wxyz,
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )


def _receiver_cfg(spec: XGRIDS2SupportSpec) -> RigidObjectCfg:
    """Load receiver appearance without trusting asset-authored collisions."""

    if spec.usd_relative is None:
        raise ValueError(f"receiver {spec.name!r} has no USD asset")
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{spec.name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(PACKAGE_DIR, spec.usd_relative),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=spec.center_xyz,
            rot=(math.cos(0.5 * spec.yaw_rad), 0.0, 0.0, math.sin(0.5 * spec.yaw_rad)),
        ),
    )


def _receiver_collision_cfgs(
    spec: XGRIDS2SupportSpec,
) -> dict[str, AssetBaseCfg]:
    """Build a deterministic open-top box proxy for a bowl/container.

    The visual YCB bowl is never allowed to define task physics. Its declared
    outer bounds and opening define a thin floor plus four walls instead. The
    same opening is consumed by the success predicate and CuRobo approach
    logic, so rendering, physics, and scoring cannot silently disagree.
    """

    opening = spec.opening_size_xy
    if opening is None:
        raise ValueError(f"receiver {spec.name!r} has no opening geometry")
    outer_x, outer_y, height = spec.size_xyz
    opening_x, opening_y = opening
    wall_x = max(0.004, 0.5 * (outer_x - opening_x))
    wall_y = max(0.004, 0.5 * (outer_y - opening_y))
    floor_height = min(0.008, max(0.004, 0.15 * height))
    bottom_z = spec.center_xyz[2] - 0.5 * height
    wall_z = bottom_z + 0.5 * height
    cosine, sine = math.cos(spec.yaw_rad), math.sin(spec.yaw_rad)

    def world_xy(local_x: float, local_y: float) -> tuple[float, float]:
        return (
            spec.center_xyz[0] + cosine * local_x - sine * local_y,
            spec.center_xyz[1] + sine * local_x + cosine * local_y,
        )

    components = {
        "floor": (
            (0.0, 0.0, bottom_z + 0.5 * floor_height),
            (outer_x, outer_y, floor_height),
        ),
        "left": (
            (-0.5 * (opening_x + wall_x), 0.0, wall_z),
            (wall_x, outer_y, height),
        ),
        "right": (
            (0.5 * (opening_x + wall_x), 0.0, wall_z),
            (wall_x, outer_y, height),
        ),
        "front": (
            (0.0, -0.5 * (opening_y + wall_y), wall_z),
            (opening_x, wall_y, height),
        ),
        "back": (
            (0.0, 0.5 * (opening_y + wall_y), wall_z),
            (opening_x, wall_y, height),
        ),
    }
    result: dict[str, AssetBaseCfg] = {}
    for component, (local_center, size) in components.items():
        center_x, center_y = world_xy(local_center[0], local_center[1])
        name = f"{spec.name}_{component}_collision"
        result[name] = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/collision_{name}",
            spawn=sim_utils.CuboidCfg(
                size=size,
                visible=False,
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.002,
                    rest_offset=0.0,
                ),
                physics_material=STATIC_MATERIAL,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(center_x, center_y, local_center[2]),
                rot=(
                    math.cos(0.5 * spec.yaw_rad),
                    0.0,
                    0.0,
                    math.sin(0.5 * spec.yaw_rad),
                ),
            ),
        )
    return result


def _table_cfg(spec: XGRIDS2SupportSpec) -> AssetBaseCfg:
    tabletop = Tabletop(spec.name, spec.center_xyz, spec.size_xyz, spec.yaw_rad)
    return _round_support_cfg(tabletop) if spec.kind == "circle" else _support_cfg(tabletop)


def author_disabled_scenario_grasp_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    *,
    object_names: tuple[str, ...],
) -> None:
    """Author one independently switchable flange/object joint per target."""

    if env_ids is not None:
        raise ValueError("prestartup grasp-joint authoring expects env_ids=None")
    stage = env.scene.stage
    for env_path in env.scene.env_prim_paths:
        robot_prim = stage.GetPrimAtPath(f"{env_path}/robot")
        flange_prim = next(
            (
                prim
                for prim in Usd.PrimRange(robot_prim)
                if prim.GetName() == MOBILE_KINOVA_FLANGE_BODY
            ),
            None,
        )
        if flange_prim is None:
            raise RuntimeError(
                f"Could not find {MOBILE_KINOVA_FLANGE_BODY!r} below {robot_prim.GetPath()}"
            )
        for object_name in object_names:
            object_path = Sdf.Path(f"{env_path}/{object_name}")
            joint_path = Sdf.Path(
                f"{env_path}/{scenario_grasp_joint_name(object_name)}"
            )
            joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            joint.CreateBody0Rel().SetTargets([flange_prim.GetPath()])
            joint.CreateBody1Rel().SetTargets([object_path])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
            joint.CreateCollisionEnabledAttr().Set(False)
            joint.CreateExcludeFromArticulationAttr().Set(True)
            joint.CreateJointEnabledAttr().Set(False)


def reset_scenario_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    scenario_name: str,
) -> None:
    """Reset every configured object, sampling only inside its declared region."""

    scenario = get_xgrids2_scenario(scenario_name)
    for spec in scenario.objects:
        asset = env.scene[spec.name]
        state = asset.data.default_root_state[env_ids].clone()
        count = len(env_ids)
        random_xy = torch.rand((count, 2), device=asset.device)
        xmin, xmax, ymin, ymax = spec.spawn_region_xy
        random_xy[:, 0] = xmin + random_xy[:, 0] * (xmax - xmin)
        random_xy[:, 1] = ymin + random_xy[:, 1] * (ymax - ymin)
        state[:, :2] = random_xy + env.scene.env_origins[env_ids, :2]
        state[:, 2] = spec.spawn_z + env.scene.env_origins[env_ids, 2]
        state[:, 3:7] = torch.tensor(
            spec.spawn_quaternion_wxyz,
            device=asset.device,
            dtype=state.dtype,
        )
        state[:, 7:13] = 0.0
        asset.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(state[:, 7:13], env_ids=env_ids)


@configclass
class XGRIDS2ScenarioTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


_GENERATED_TASKS: dict[str, type[Task]] = {}


def build_xgrids2_scenario_task(scenario_name: str) -> type[Task]:
    """Generate one direct-import task class from a validated scenario."""

    if scenario_name in _GENERATED_TASKS:
        return _GENERATED_TASKS[scenario_name]
    scenario = get_xgrids2_scenario(scenario_name)
    if scenario.strategy == "plate_sink":
        from robolab.tasks.mobile_manipulation.xgrids2_lab_plate_to_sink_task import (
            XGRIDS2LabPlateToSinkTask,
        )

        return XGRIDS2LabPlateToSinkTask
    if scenario_name == "can_dining_to_snack":
        from robolab.tasks.mobile_manipulation.xgrids2_lab_can_transfer_task import (
            XGRIDS2LabCanTransferTask,
        )

        return XGRIDS2LabCanTransferTask

    scene_name = f"{scenario.task_name}SceneCfg"
    scene_class = type(scene_name, (XGRIDS2LabPhysicsSceneCfg,), {})
    scene_class.replicate_physics = False
    for segment in COLLISION_SEGMENTS:
        if _is_manipulation_table_segment(segment):
            setattr(scene_class, segment.name, None)
    for support in scenario.supports:
        if support.kind in {"rectangle", "circle"}:
            setattr(scene_class, support.name, _table_cfg(support))
        elif support.kind == "bowl":
            setattr(scene_class, support.name, _receiver_cfg(support))
            for name, cfg in _receiver_collision_cfgs(support).items():
                setattr(scene_class, name, cfg)
    for object_spec in scenario.objects:
        setattr(scene_class, object_spec.name, _physics_object_cfg(object_spec))
    scene_class = configclass(scene_class)

    manipulated_names = tuple(
        item.name for item in scenario.objects if item.manipulated
    )
    events_name = f"{scenario.task_name}Events"
    events_class = type(
        events_name,
        (),
        {
            "author_oracle_grasp_joints": EventTerm(
                func=author_disabled_scenario_grasp_joints,
                params={"object_names": manipulated_names},
                mode="prestartup",
            ),
            "reset_scene": EventTerm(func=mdp.reset_scene_to_default, mode="reset"),
            "sample_objects": EventTerm(
                func=reset_scenario_objects,
                params={"scenario_name": scenario.name},
                mode="reset",
            ),
        },
    )
    events_class = configclass(events_class)
    contact_objects = [item.name for item in scenario.objects]
    contact_objects.extend(support.name for support in scenario.supports)
    task_class = type(
        scenario.task_name,
        (Task,),
        {
            "scene": scene_class,
            "events": events_class,
            "terminations": XGRIDS2ScenarioTerminations,
            "contact_object_list": contact_objects,
            "instruction": scenario.instruction,
            "episode_length_s": scenario.episode_length_s,
            "task_name": scenario.task_name,
            "sim_physx_overrides": {"max_velocity_iteration_count": 8},
            "subtasks": None,
            "attributes": list(scenario.capability_tags),
            "__module__": __name__,
        },
    )
    task_class = dataclass(task_class)
    _GENERATED_TASKS[scenario_name] = task_class
    return task_class
