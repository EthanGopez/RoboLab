"""Physics-valid GenManip kitchen can-transfer task for the mobile Kinova."""

import copy
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
from pxr import Gf, Sdf, Usd, UsdPhysics

from robolab.constants import PACKAGE_DIR
from robolab.core.task.task import Task
from robolab.robots.kinova_mobile_cobot import KinovaMobileCobotCfg
from robolab.robots.kinova_mobile_cobot import MOBILE_KINOVA_FLANGE_BODY
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
    CAN_CONTACT_OFFSET,
    CAN_INITIAL_Z,
    CAN_MASS_KG,
    CAN_PHYSICS_USD_RELATIVE,
    CAN_REST_OFFSET,
    KITCHEN_USD_RELATIVE,
    ORACLE_PICK_REGION,
    PICK_REGION,
    STAGING_POSES,
    STATIC_COLLISION_BOXES,
    CollisionBox,
)


STATIC_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.70,
    dynamic_friction=0.60,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="min",
)

ORACLE_GRASP_JOINT_NAME = "oracle_grasp_joint"


def author_disabled_oracle_grasp_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    *,
    object_name: str = "can",
) -> None:
    """Create one disabled flange-to-object joint before PhysX starts."""

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
        object_path = Sdf.Path(f"{env_path}/{object_name}")
        joint = UsdPhysics.FixedJoint.Define(
            stage, Sdf.Path(f"{env_path}/{ORACLE_GRASP_JOINT_NAME}")
        )
        joint.CreateBody0Rel().SetTargets([flange_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([object_path])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint.CreateJointEnabledAttr().Set(False)

def _static_box_cfg(box: CollisionBox) -> AssetBaseCfg:
    """Create one non-rigid static collider from the collision contract."""

    visual_material = None
    if box.visible:
        visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.46, 0.43, 0.39),
            roughness=0.85,
        )
    return AssetBaseCfg(
        # Keep every collider directly under the cloned environment namespace.
        # Shape spawners require an intermediate regex parent (such as
        # ``/collision``) to exist before they can clone beneath it.
        prim_path=f"{{ENV_REGEX_NS}}/collision_{box.name}",
        spawn=sim_utils.CuboidCfg(
            size=box.size_xyz,
            visible=box.visible,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=STATIC_MATERIAL,
            visual_material=visual_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=box.center_xyz),
    )


def make_physics_can_cfg() -> RigidObjectCfg:
    """Build one convex can rigid body with a collision-free YCB visual child."""

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/can",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(PACKAGE_DIR, CAN_PHYSICS_USD_RELATIVE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.05,
                angular_damping=0.05,
                max_linear_velocity=5.0,
                max_angular_velocity=720.0,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CAN_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=CAN_CONTACT_OFFSET,
                rest_offset=CAN_REST_OFFSET,
                torsional_patch_radius=0.02,
                min_torsional_patch_radius=0.005,
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(*PICK_REGION.center, CAN_INITIAL_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )


@configclass
class GenManipKitchenPhysicsSceneCfg:
    """Authored kitchen visual plus explicit task-grade physics entities."""

    # The prestartup event authors an independent joint in every environment,
    # so the environment prims cannot share replicated physics schemas.
    replicate_physics: bool = False

    kitchen_visual: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/kitchen_visual",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(PACKAGE_DIR, KITCHEN_USD_RELATIVE),
        ),
    )
    can: RigidObjectCfg = make_physics_can_cfg()
    # These supports are explicit annotated fields because RoboLab's contact
    # sensor generator and task validator require stable entity names.
    island_top_east: AssetBaseCfg = _static_box_cfg(
        next(box for box in STATIC_COLLISION_BOXES if box.name == "island_top_east")
    )
    island_top_west: AssetBaseCfg = _static_box_cfg(
        next(box for box in STATIC_COLLISION_BOXES if box.name == "island_top_west")
    )
    counter_right_top: AssetBaseCfg = _static_box_cfg(
        next(box for box in STATIC_COLLISION_BOXES if box.name == "counter_right_top")
    )


# Configclass supports fields added after decoration, matching RoboLab's scene
# importer.  Keeping the primitive list in the pure geometry module makes tests
# and simulation consume exactly the same collision contract.
for _collision_box in STATIC_COLLISION_BOXES:
    if _collision_box.name in {
        "island_top_east",
        "island_top_west",
        "counter_right_top",
    }:
        continue
    setattr(
        GenManipKitchenPhysicsSceneCfg,
        _collision_box.name,
        _static_box_cfg(_collision_box),
    )


@configclass
class KitchenKinovaMobileCobotCfg(KinovaMobileCobotCfg):
    """Mobile cobot whose authored initial root is the verified pick stage."""

    robot = copy.deepcopy(KinovaMobileCobotCfg().robot)
    # The oracle's constrained 0.349 kg lift needs a stiffer task-space position
    # servo than the gentle renderer/default-policy setup.  Retain the
    # manufacturer's 39/9 N m effort ceilings; only reduce steady deflection.
    robot.actuators["arm_1_to_4"].stiffness = 300.0
    robot.actuators["arm_1_to_4"].damping = 30.0
    robot.actuators["arm_5_to_7"].stiffness = 240.0
    robot.actuators["arm_5_to_7"].damping = 24.0
    robot.init_state.pos = STAGING_POSES["pick"][:3]
    robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)


def reset_can_on_island(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    pick_region=ORACLE_PICK_REGION,
) -> None:
    """Uniformly sample can centers in the audited pick region and drop them."""

    can = env.scene[asset_cfg.name]
    count = len(env_ids)
    root_state = can.data.default_root_state[env_ids].clone()
    random_xy = torch.rand((count, 2), device=can.device)
    random_xy[:, 0] = pick_region.xmin + random_xy[:, 0] * (
        pick_region.xmax - pick_region.xmin
    )
    random_xy[:, 1] = pick_region.ymin + random_xy[:, 1] * (
        pick_region.ymax - pick_region.ymin
    )
    root_state[:, 0:2] = random_xy + env.scene.env_origins[env_ids, 0:2]
    root_state[:, 2] = CAN_INITIAL_Z + env.scene.env_origins[env_ids, 2]
    root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=can.device)
    root_state[:, 7:13] = 0.0
    can.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    can.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)


@configclass
class KitchenCanTransferEvents:
    author_oracle_grasp_joints = EventTerm(
        func=author_disabled_oracle_grasp_joints, mode="prestartup"
    )
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    sample_can = EventTerm(func=reset_can_on_island, mode="reset")


@configclass
class KitchenCanTransferTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@dataclass
class GenManipKitchenCanTransferTask(Task):
    """Move the can from the island to the opposing kitchen counter."""

    scene = GenManipKitchenPhysicsSceneCfg
    events = KitchenCanTransferEvents
    terminations = KitchenCanTransferTerminations
    contact_object_list = [
        "can",
        "island_top_east",
        "island_top_west",
        "counter_right_top",
    ]
    instruction: str = "Move the can from the kitchen island to the opposing counter."
    episode_length_s: int = 120
    task_name: str = "GenManipKitchenCanTransferTask"
    # The global default clamps velocity iterations to one. Frictional
    # grasping needs the robot/can actors' authored four iterations to survive
    # the clamp; eight leaves headroom without forcing every actor to use it.
    sim_physx_overrides = {"max_velocity_iteration_count": 8}
