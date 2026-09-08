"""Register the GenManip kitchen with mobile-base plus absolute-pose arm IK."""

from __future__ import annotations

import os


IK_ENV_NAME = "GenManipKitchenCanTransferKinovaMobileIK"


def register_genmanip_kitchen_mobile_ik_env() -> tuple[type, str]:
    """Register and return the deterministic-oracle control environment."""

    from robolab.core.environments.config import generate_env_cfg_from_task
    from robolab.core.observations.observation_utils import generate_obs_cfg
    from robolab.robots.kinova_gen3 import KinovaProprioceptionObservationCfg
    from robolab.robots.kinova_mobile_cobot import (
        KinovaMobileContinuousGripperIKActionCfg,
        MOBILE_KINOVA_CONTACT_GRIPPER,
    )
    from robolab.tasks.mobile_manipulation import kitchen_can_transfer_task
    from robolab.tasks.mobile_manipulation.kitchen_can_transfer_task import (
        KitchenKinovaMobileCobotCfg,
    )

    observation_cfg = generate_obs_cfg(
        {"proprio_obs": KinovaProprioceptionObservationCfg()}
    )
    action_cfg = KinovaMobileContinuousGripperIKActionCfg()
    from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
        CARRY_BASE_LINEAR_ACCELERATION,
        CARRY_BASE_LINEAR_JERK,
    )

    action_cfg.base.max_linear_acceleration = CARRY_BASE_LINEAR_ACCELERATION
    action_cfg.base.max_linear_jerk = CARRY_BASE_LINEAR_JERK
    task_path = os.path.abspath(kitchen_can_transfer_task.__file__)
    return generate_env_cfg_from_task(
        task_file_path=task_path,
        env_name=IK_ENV_NAME,
        robot_cfg=KitchenKinovaMobileCobotCfg,
        observations_cfg=observation_cfg(),
        actions_cfg=action_cfg,
        contact_gripper=MOBILE_KINOVA_CONTACT_GRIPPER,
        camera_cfg=None,
        lighting_cfg=None,
        background_cfg=None,
        dt=1.0 / 120.0,
        render_interval=4,
        decimation=4,
        seed=0,
        num_envs=1,
        env_spacing=8.0,
    )
