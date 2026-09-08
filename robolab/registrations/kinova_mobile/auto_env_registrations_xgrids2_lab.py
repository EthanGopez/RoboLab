"""Register the physics-enabled XGRIDS2 lab with the mobile Kinova."""

from __future__ import annotations

import os


XGRIDS2_LAB_ENV_NAME = "XGRIDS2LabNavigationKinovaMobileJointPosition"


def register_xgrids2_lab_mobile_env() -> tuple[type, str]:
    from robolab.core.environments.config import generate_env_cfg_from_task
    from robolab.core.observations.observation_utils import generate_obs_cfg
    from robolab.robots.kinova_gen3 import KinovaProprioceptionObservationCfg
    from robolab.robots.kinova_mobile_cobot import KinovaMobileJointPositionActionCfg
    from robolab.tasks.mobile_manipulation import xgrids2_lab_navigation_task
    from robolab.tasks.mobile_manipulation.xgrids2_lab_navigation_task import (
        XGRIDS2LabKinovaMobileCobotCfg,
    )

    observation_cfg = generate_obs_cfg(
        {"proprio_obs": KinovaProprioceptionObservationCfg()}
    )
    task_path = os.path.abspath(xgrids2_lab_navigation_task.__file__)
    return generate_env_cfg_from_task(
        task_file_path=task_path,
        env_name=XGRIDS2_LAB_ENV_NAME,
        robot_cfg=XGRIDS2LabKinovaMobileCobotCfg,
        observations_cfg=observation_cfg(),
        actions_cfg=KinovaMobileJointPositionActionCfg(),
        contact_gripper=None,
        camera_cfg=None,
        lighting_cfg=None,
        background_cfg=None,
        dt=1.0 / 120.0,
        render_interval=4,
        decimation=4,
        seed=0,
        num_envs=1,
        env_spacing=12.0,
    )
