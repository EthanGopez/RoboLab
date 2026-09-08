"""Register the physics-valid GenManip kitchen with the mobile Kinova."""

from __future__ import annotations

import os


ENV_NAME = "GenManipKitchenCanTransferKinovaMobileJointPosition"


def register_genmanip_kitchen_mobile_env() -> tuple[type, str]:
    """Register and return the single Milestone-1 environment configuration."""

    from robolab.core.environments.config import generate_env_cfg_from_task
    from robolab.core.observations.observation_utils import generate_obs_cfg
    from robolab.robots.kinova_gen3 import KinovaProprioceptionObservationCfg
    from robolab.robots.kinova_mobile_cobot import (
        KinovaMobileJointPositionActionCfg,
        MOBILE_KINOVA_CONTACT_GRIPPER,
    )
    from robolab.tasks.mobile_manipulation import kitchen_can_transfer_task
    from robolab.tasks.mobile_manipulation.kitchen_can_transfer_task import (
        KitchenKinovaMobileCobotCfg,
    )

    observation_cfg = generate_obs_cfg(
        {"proprio_obs": KinovaProprioceptionObservationCfg()}
    )
    task_path = os.path.abspath(kitchen_can_transfer_task.__file__)
    return generate_env_cfg_from_task(
        task_file_path=task_path,
        env_name=ENV_NAME,
        robot_cfg=KitchenKinovaMobileCobotCfg,
        observations_cfg=observation_cfg(),
        actions_cfg=KinovaMobileJointPositionActionCfg(),
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
