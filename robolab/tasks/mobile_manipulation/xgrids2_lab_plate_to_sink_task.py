"""Physics-backed plate transfer from the XGRIDS2 dining table to the sink."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.constants import PACKAGE_DIR
from robolab.core.task.task import Task
from robolab.tasks.mobile_manipulation.kitchen_can_transfer_task import (
    STATIC_MATERIAL,
    author_disabled_oracle_grasp_joints,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_can_transfer_task import (
    XGRIDS2LabManipulationRobotCfg,
    _support_cfg,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
    COLLISION_SEGMENTS,
    FLOOR_THICKNESS_M,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_manipulation_geometry import (
    DINING_TABLE,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_navigation_task import (
    XGRIDS2LabPhysicsSceneCfg,
    make_proxy_cfg,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_plate_sink_geometry import (
    ACTIVE_COLLISION_SEGMENTS,
    PLATE_INITIAL_Z,
    PLATE_MASS_KG,
    PLATE_PICK_CENTER,
    PLATE_PICK_REGION,
    PLATE_USD_RELATIVE,
    SINK_COLLISION_BOXES,
    SINK_PHYSICS_REGION,
    CollisionBox,
)


def _expanded_floor_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/collision_floor",
        spawn=sim_utils.CuboidCfg(
            size=(*SINK_PHYSICS_REGION.size, FLOOR_THICKNESS_M),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=STATIC_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                *SINK_PHYSICS_REGION.center,
                -0.5 * FLOOR_THICKNESS_M,
            )
        ),
    )


def _plate_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/plate",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(PACKAGE_DIR, PLATE_USD_RELATIVE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=PLATE_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.003,
                rest_offset=0.0,
                torsional_patch_radius=0.04,
                min_torsional_patch_radius=0.01,
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(*PLATE_PICK_CENTER, PLATE_INITIAL_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )


def _sink_box_cfg(box: CollisionBox) -> AssetBaseCfg:
    half_yaw = 0.5 * box.yaw
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/collision_{box.name}",
        spawn=sim_utils.CuboidCfg(
            size=box.size_xyz,
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=STATIC_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=box.center_xyz,
            rot=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        ),
    )


class XGRIDS2LabPlateToSinkSceneCfg(XGRIDS2LabPhysicsSceneCfg):
    """Expanded kitchen route, clean supports, basin, and dynamic plate."""

    floor = _expanded_floor_cfg()
    plate: RigidObjectCfg = _plate_cfg()
    dining_table_top: AssetBaseCfg = _support_cfg(DINING_TABLE)


# Replace the cropped base map with the expanded task-specific map.  Coarse
# dining-table and sink-fixture outlines are omitted by ACTIVE_COLLISION_SEGMENTS
# and replaced by the clean tabletop/counter/basin geometry.
for _segment in COLLISION_SEGMENTS:
    setattr(XGRIDS2LabPlateToSinkSceneCfg, _segment.name, None)
for _segment in ACTIVE_COLLISION_SEGMENTS:
    setattr(XGRIDS2LabPlateToSinkSceneCfg, _segment.name, make_proxy_cfg(_segment))
for _box_contract in SINK_COLLISION_BOXES:
    setattr(
        XGRIDS2LabPlateToSinkSceneCfg,
        _box_contract.name,
        _sink_box_cfg(_box_contract),
    )

XGRIDS2LabPlateToSinkSceneCfg = configclass(XGRIDS2LabPlateToSinkSceneCfg)


@configclass
class XGRIDS2LabPlateRobotCfg(XGRIDS2LabManipulationRobotCfg):
    """Use the proven kitchen-IK arm drives and dining-table initialization."""

    robot = copy.deepcopy(XGRIDS2LabManipulationRobotCfg().robot)
    # The retained plate loads the arm off-axis.  Damping alone left the
    # can-tuned 120/80 N m/rad drives too compliant: joint 1 drifted more than
    # 0.3 rad in the stationary carry pose, then snapped back when traversal
    # began.  Match the already-proven kitchen IK position gains while keeping
    # the manufacturer effort and velocity ceilings unchanged.
    robot.actuators["arm_1_to_4"].stiffness = 300.0
    robot.actuators["arm_1_to_4"].damping = 30.0
    robot.actuators["arm_5_to_7"].stiffness = 240.0
    robot.actuators["arm_5_to_7"].damping = 24.0


def reset_plate_on_dining_table(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> None:
    plate = env.scene[asset_cfg.name]
    root_state = plate.data.default_root_state[env_ids].clone()
    random_xy = torch.rand((len(env_ids), 2), device=plate.device)
    xmin, xmax, ymin, ymax = PLATE_PICK_REGION
    random_xy[:, 0] = xmin + random_xy[:, 0] * (xmax - xmin)
    random_xy[:, 1] = ymin + random_xy[:, 1] * (ymax - ymin)
    root_state[:, 0:2] = random_xy + env.scene.env_origins[env_ids, 0:2]
    root_state[:, 2] = PLATE_INITIAL_Z + env.scene.env_origins[env_ids, 2]
    root_state[:, 3:7] = torch.tensor(
        (1.0, 0.0, 0.0, 0.0), device=plate.device
    )
    root_state[:, 7:13] = 0.0
    plate.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    plate.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)


@configclass
class XGRIDS2LabPlateToSinkEvents:
    author_oracle_grasp_joints = EventTerm(
        func=author_disabled_oracle_grasp_joints,
        params={"object_name": "plate"},
        mode="prestartup",
    )
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    sample_plate = EventTerm(func=reset_plate_on_dining_table, mode="reset")


@configclass
class XGRIDS2LabPlateToSinkTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class XGRIDS2LabPlateToSinkTask(Task):
    """Pick a plate from the dining table and place it inside the sink."""

    scene = XGRIDS2LabPlateToSinkSceneCfg
    events = XGRIDS2LabPlateToSinkEvents
    terminations = XGRIDS2LabPlateToSinkTerminations
    contact_object_list = [
        "plate",
        "dining_table_top",
        "sink_bottom",
        "sink_wall_left",
        "sink_wall_right",
        "sink_wall_front",
        "sink_wall_back",
    ]
    instruction: str = "Move the plate from the dining table into the sink."
    episode_length_s: int = 300
    task_name: str = "XGRIDS2LabPlateToSinkTask"
    sim_physx_overrides = {"max_velocity_iteration_count": 8}
