"""Physics-backed dining-to-snack manipulation task in the XGRIDS2 lab."""

import copy
import math
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

from robolab.core.task.task import Task
from robolab.tasks.mobile_manipulation.kitchen_can_transfer_task import (
    STATIC_MATERIAL,
    author_disabled_oracle_grasp_joints,
    make_physics_can_cfg,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_manipulation_geometry import (
    CAN_INITIAL_Z,
    DINING_TABLE,
    PICK_CENTER,
    PICK_REGION,
    PLACE_CENTER,
    SNACK_TABLE,
    Tabletop,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
    COLLISION_SEGMENTS,
    PHYSICS_REGION,
    required_line_clearance,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_navigation_task import (
    XGRIDS2LabKinovaMobileCobotCfg,
    XGRIDS2LabPhysicsSceneCfg,
)


DINING_TABLE_MAP_MASK_RADIUS_M = 0.95
# LabAPT's fragmented snack-table outline reaches 0.667 m from the fitted
# tabletop center.  Round up to 0.70 m so all of that furniture outline is
# replaced by the task's accurate cylindrical support instead of leaving an
# invisible 1.5 m collision segment across the manipulation approach.
SNACK_TABLE_MAP_MASK_RADIUS_M = 0.70


def _support_cfg(table: Tabletop) -> AssetBaseCfg:
    """Create the complete oriented tabletop used by physical manipulation."""

    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/collision_{table.name}",
        spawn=sim_utils.CuboidCfg(
            size=table.size_xyz,
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0
            ),
            physics_material=STATIC_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=table.center_xyz,
            rot=(math.cos(0.5 * table.yaw), 0.0, 0.0, math.sin(0.5 * table.yaw)),
        ),
    )


def _round_support_cfg(table: Tabletop) -> AssetBaseCfg:
    """Create a complete cylindrical support for a visible round tabletop."""

    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/collision_{table.name}",
        spawn=sim_utils.CylinderCfg(
            radius=0.5 * min(table.size_xyz[:2]),
            height=table.size_xyz[2],
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0
            ),
            physics_material=STATIC_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=table.center_xyz),
    )


def _can_cfg() -> RigidObjectCfg:
    cfg = make_physics_can_cfg()
    cfg.init_state.pos = (*PICK_CENTER, CAN_INITIAL_Z)
    return cfg


class XGRIDS2LabCanTransferSceneCfg(XGRIDS2LabPhysicsSceneCfg):
    """Navigation scene plus clean supports and one dynamic YCB can."""

    can: RigidObjectCfg = _can_cfg()
    dining_table_top: AssetBaseCfg = _support_cfg(DINING_TABLE)
    snack_table_top: AssetBaseCfg = _round_support_cfg(SNACK_TABLE)


def _distance_to_segment_xy(point, segment) -> float:
    px, py = point
    ax, ay = segment.start_xy
    bx, by = segment.end_xy
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared <= 1.0e-12:
        return math.hypot(px - ax, py - ay)
    fraction = max(
        0.0,
        min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_squared),
    )
    return math.hypot(px - (ax + fraction * dx), py - (ay + fraction * dy))


def _is_manipulation_table_segment(segment) -> bool:
    return (
        _distance_to_segment_xy(DINING_TABLE.center_xyz[:2], segment)
        <= DINING_TABLE_MAP_MASK_RADIUS_M
        or _distance_to_segment_xy(SNACK_TABLE.center_xyz[:2], segment)
        <= SNACK_TABLE_MAP_MASK_RADIUS_M
    )


def manipulation_base_pose_is_clear(
    point: tuple[float, float],
    *,
    robot_radius: float,
    safety_margin: float = 0.05,
) -> bool:
    """Check clearance against the collision segments retained by this scene."""

    required = required_line_clearance(robot_radius, safety_margin)
    return PHYSICS_REGION.contains(
        point, margin=robot_radius + safety_margin
    ) and all(
        _distance_to_segment_xy(point, segment) >= required
        for segment in COLLISION_SEGMENTS
        if not _is_manipulation_table_segment(segment)
    )


MANIPULATION_COLLISION_SEGMENTS = tuple(
    segment
    for segment in COLLISION_SEGMENTS
    if not _is_manipulation_table_segment(segment)
)


# The LabAPT navigation map represents furniture outlines as 1.5 m walls.
# Remove only the segments belonging to the two manipulation tables; the full
# physical tabletops above replace them while route obstacles remain intact.
for _segment in COLLISION_SEGMENTS:
    if _segment not in MANIPULATION_COLLISION_SEGMENTS:
        setattr(XGRIDS2LabCanTransferSceneCfg, _segment.name, None)

# Apply configclass only after masking the inherited navigation fields.
# Decorating first would freeze inherited AssetBaseCfg defaults into the
# generated initializer, so the colliders would still spawn at runtime.
XGRIDS2LabCanTransferSceneCfg = configclass(XGRIDS2LabCanTransferSceneCfg)


@configclass
class XGRIDS2LabManipulationRobotCfg(XGRIDS2LabKinovaMobileCobotCfg):
    """Stable RGB-only XGRIDS robot initialized at the dining stage."""

    robot = copy.deepcopy(XGRIDS2LabKinovaMobileCobotCfg().robot)
    # Keep joint 4 inside its authored safety band for both IK and planning.
    robot.init_state.joint_pos["joint_4"] = -2.40
    # Gentle physical drives support smooth collision-checked waypoint
    # tracking without saturating the wrist effort limits. The action
    # controller and PhysX are the only owners of arm state; validation code
    # never injects articulation joint poses.
    robot.actuators["arm_1_to_4"].stiffness = 120.0
    robot.actuators["arm_1_to_4"].damping = 12.0
    robot.actuators["arm_1_to_4"].effort_limit_sim = 39.0
    robot.actuators["arm_1_to_4"].velocity_limit_sim = 1.3963
    robot.actuators["arm_5_to_7"].stiffness = 80.0
    robot.actuators["arm_5_to_7"].damping = 14.0
    robot.actuators["arm_5_to_7"].effort_limit_sim = 9.0
    robot.actuators["arm_5_to_7"].velocity_limit_sim = 1.2218


@configclass
class XGRIDS2LabIKManipulationRobotCfg(XGRIDS2LabManipulationRobotCfg):
    """Task-facing XGRIDS arm seed and gains for local Cartesian control."""

    robot = copy.deepcopy(XGRIDS2LabManipulationRobotCfg().robot)
    # Bounded Pinocchio position-IK solution for the XGRIDS dining-table
    # pre-grasp center. It is close to the collision-safe transport posture,
    # has sub-nanometer FK residual, and avoids using local IK for a global
    # configuration transition. Runtime code never writes articulation state.
    robot.init_state.joint_pos.update(
        {
            "joint_1": -0.1202665002,
            "joint_2": 0.2163686709,
            "joint_3": 3.1756161042,
            "joint_4": -1.9628153204,
            "joint_5": 0.0365685478,
            "joint_6": 0.6087319635,
            "joint_7": 1.5334912904,
        }
    )
    # Match the gains proven by the kitchen differential-IK task while keeping
    # the Kinova manufacturer effort ceilings.
    robot.actuators["arm_1_to_4"].stiffness = 300.0
    robot.actuators["arm_1_to_4"].damping = 30.0
    robot.actuators["arm_5_to_7"].stiffness = 240.0
    robot.actuators["arm_5_to_7"].damping = 24.0


def reset_can_on_dining_table(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("can"),
) -> None:
    """Sample the can inside the reach-conditioned dining-table region."""

    can = env.scene[asset_cfg.name]
    root_state = can.data.default_root_state[env_ids].clone()
    random_xy = torch.rand((len(env_ids), 2), device=can.device)
    xmin, xmax, ymin, ymax = PICK_REGION
    random_xy[:, 0] = xmin + random_xy[:, 0] * (xmax - xmin)
    random_xy[:, 1] = ymin + random_xy[:, 1] * (ymax - ymin)
    root_state[:, 0:2] = random_xy + env.scene.env_origins[env_ids, 0:2]
    root_state[:, 2] = CAN_INITIAL_Z + env.scene.env_origins[env_ids, 2]
    root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=can.device)
    root_state[:, 7:13] = 0.0
    can.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    can.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)



@configclass
class XGRIDS2LabCanTransferEvents:
    author_oracle_grasp_joints = EventTerm(
        func=author_disabled_oracle_grasp_joints, mode="prestartup"
    )
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    sample_can = EventTerm(func=reset_can_on_dining_table, mode="reset")


@configclass
class XGRIDS2LabCanTransferTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class XGRIDS2LabCanTransferTask(Task):
    """Pick a can at the dining table and place it on the snack table."""

    scene = XGRIDS2LabCanTransferSceneCfg
    events = XGRIDS2LabCanTransferEvents
    terminations = XGRIDS2LabCanTransferTerminations
    contact_object_list = ["can", "dining_table_top", "snack_table_top"]
    instruction: str = "Move the can from the dining table to the snack table."
    episode_length_s: int = 180
    task_name: str = "XGRIDS2LabCanTransferTask"
    sim_physx_overrides = {"max_velocity_iteration_count": 8}
