"""Physics-enabled route-first navigation task in the XGRIDS2 lab scan."""

import copy
import math
import os
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from robolab.constants import PACKAGE_DIR
from robolab.core.task.task import Task
from robolab.robots.kinova_mobile_cobot import KinovaMobileCobotCfg
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
    COLLISION_SEGMENTS,
    FLOOR_CENTER_Z_M,
    FLOOR_THICKNESS_M,
    PHYSICS_REGION,
    PROXY_CENTER_Z_M,
    PROXY_HEIGHT_M,
    PROXY_WIDTH_M,
    STAGING_POSES,
    XGRIDS_VISUAL_USD_RELATIVE,
    XGRIDS_VISUAL_Z_OFFSET_M,
    CollisionSegment,
)


STATIC_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.80,
    dynamic_friction=0.70,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="min",
)


def make_floor_cfg(*, visible: bool = False) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/collision_floor",
        spawn=sim_utils.CuboidCfg(
            size=(*PHYSICS_REGION.size, FLOOR_THICKNESS_M),
            visible=visible,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=STATIC_MATERIAL,
            visual_material=(
                sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.20, 0.72, 0.28), opacity=0.35, roughness=0.9
                )
                if visible
                else None
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(*PHYSICS_REGION.center, FLOOR_CENTER_Z_M)
        ),
    )


def make_proxy_cfg(segment: CollisionSegment, *, visible: bool = False) -> AssetBaseCfg:
    half_yaw = 0.5 * segment.yaw
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/collision_{segment.name}",
        spawn=sim_utils.CuboidCfg(
            size=(segment.length, PROXY_WIDTH_M, PROXY_HEIGHT_M),
            visible=visible,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=STATIC_MATERIAL,
            visual_material=(
                sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.94, 0.35, 0.08), opacity=0.45, roughness=0.8
                )
                if visible
                else None
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(*segment.center_xy, PROXY_CENTER_Z_M),
            rot=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        ),
    )


@configclass
class XGRIDS2LabPhysicsSceneCfg:
    """Neural appearance asset plus an independent static collision layer."""

    replicate_physics: bool = False
    lab_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/xgrids2_lab_visual",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(PACKAGE_DIR, XGRIDS_VISUAL_USD_RELATIVE),
            # XGRIDS packaged the reconstruction mesh with PhysicsCollisionAPI.
            # Explicitly disable it so only the clean overlay participates in
            # simulation; otherwise noisy scan triangles compete with the slab.
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, XGRIDS_VISUAL_Z_OFFSET_M)
        ),
    )
    floor = make_floor_cfg()
    dome_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dome_light",
        spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.85, 0.88, 1.0)),
    )
    base_obstacle_contact = ContactSensorCfg(
        # GPU PhysX cannot filter a dynamic body against hundreds of static
        # collider roots. Monitor the supported net base-link contact force;
        # validation accepts floor support/friction only while the force stays
        # inside a 45-degree cone around vertical.
        prim_path="{ENV_REGEX_NS}/robot/base_link",
        update_period=0.0,
        history_length=3,
        force_threshold=0.5,
    )


for _segment in COLLISION_SEGMENTS:
    setattr(XGRIDS2LabPhysicsSceneCfg, _segment.name, make_proxy_cfg(_segment))


@configclass
class XGRIDS2LabKinovaMobileCobotCfg(KinovaMobileCobotCfg):
    """Mobile Kinova initialized at the collision-safe dining staging pose."""

    robot = copy.deepcopy(KinovaMobileCobotCfg().robot)
    # Stable vendor-volume hold gains, retaining the authored effort limits.
    robot.actuators["arm_1_to_4"].stiffness = 800.0
    robot.actuators["arm_1_to_4"].damping = 40.0
    robot.actuators["arm_5_to_7"].stiffness = 800.0
    robot.actuators["arm_5_to_7"].damping = 40.0
    robot.init_state.pos = STAGING_POSES["dining_table"][:3]
    robot.init_state.rot = (
        math.cos(0.5 * STAGING_POSES["dining_table"][3]),
        0.0,
        0.0,
        math.sin(0.5 * STAGING_POSES["dining_table"][3]),
    )
    # This limit-safe transport posture has a measured 0.305 m maximum
    # link-center radius.  Navigation inflates it to a conservative 0.36 m
    # whole-robot footprint.  The previous home pose reached 0.564 m sideways.
    robot.init_state.joint_pos.update(
        {
            "joint_1": 0.0,
            "joint_2": -0.70,
            "joint_3": math.pi,
            "joint_4": -2.50,
            "joint_5": 0.0,
            "joint_6": 0.70,
            "joint_7": math.pi / 2.0,
        }
    )
    # The packaged NuRec volume's sensor contract is RGB-only.
    navigation_camera = copy.deepcopy(KinovaMobileCobotCfg().navigation_camera)
    navigation_camera.data_types = ["rgb"]
    wrist_cam = copy.deepcopy(KinovaMobileCobotCfg().wrist_cam)
    wrist_cam.data_types = ["rgb"]


@configclass
class XGRIDS2LabEvents:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class XGRIDS2LabTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class XGRIDS2LabNavigationTask(Task):
    """Navigate between the dining-table and snack-table staging regions."""

    scene = XGRIDS2LabPhysicsSceneCfg
    events = XGRIDS2LabEvents
    terminations = XGRIDS2LabTerminations
    contact_object_list = []
    instruction: str = "Navigate from the dining-table stage to the snack-table stage."
    episode_length_s: int = 90
    task_name: str = "XGRIDS2LabNavigationTask"
