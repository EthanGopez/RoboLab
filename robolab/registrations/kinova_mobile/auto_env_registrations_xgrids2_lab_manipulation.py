"""Register the XGRIDS2 dining-to-snack can-transfer environment."""

from __future__ import annotations

import os


XGRIDS2_LAB_MANIP_ENV_NAME = "XGRIDS2LabCanTransferKinovaMobileCurobo"
XGRIDS2_LAB_MANIP_IK_ENV_NAME = "XGRIDS2LabCanTransferKinovaMobileIK"


def register_xgrids2_lab_manipulation_env() -> tuple[type, str]:
    from robolab.core.environments.config import generate_env_cfg_from_task
    from robolab.core.observations.observation_utils import generate_obs_cfg
    from robolab.robots.kinova_gen3 import KinovaProprioceptionObservationCfg
    from robolab.robots.kinova_mobile_cobot import (
        KinovaMobileContinuousGripperJointPositionActionCfg,
        MOBILE_KINOVA_CONTACT_GRIPPER,
    )
    from robolab.tasks.mobile_manipulation import xgrids2_lab_can_transfer_task
    from robolab.tasks.mobile_manipulation.xgrids2_lab_can_transfer_task import (
        XGRIDS2LabManipulationRobotCfg,
    )

    observation_cfg = generate_obs_cfg(
        {"proprio_obs": KinovaProprioceptionObservationCfg()}
    )
    action_cfg = KinovaMobileContinuousGripperJointPositionActionCfg()
    action_cfg.base.max_linear_acceleration = 0.20
    action_cfg.base.max_linear_jerk = 0.50
    task_path = os.path.abspath(xgrids2_lab_can_transfer_task.__file__)
    return generate_env_cfg_from_task(
        task_file_path=task_path,
        env_name=XGRIDS2_LAB_MANIP_ENV_NAME,
        robot_cfg=XGRIDS2LabManipulationRobotCfg,
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
        env_spacing=12.0,
    )


def register_xgrids2_lab_manipulation_ik_env() -> tuple[type, str]:
    """Register the clean task-space IK XGRIDS manipulation environment."""

    from robolab.core.environments.config import generate_env_cfg_from_task
    from robolab.core.observations.observation_utils import generate_obs_cfg
    from robolab.robots.kinova_gen3 import KinovaProprioceptionObservationCfg
    from robolab.robots.kinova_mobile_cobot import (
        KinovaMobileContinuousGripperIKActionCfg,
        MOBILE_KINOVA_CONTACT_GRIPPER,
    )
    from robolab.tasks.mobile_manipulation import xgrids2_lab_can_transfer_task
    from robolab.tasks.mobile_manipulation.xgrids2_lab_can_transfer_task import (
        XGRIDS2LabIKManipulationRobotCfg,
    )

    observation_cfg = generate_obs_cfg(
        {"proprio_obs": KinovaProprioceptionObservationCfg()}
    )
    action_cfg = KinovaMobileContinuousGripperIKActionCfg()
    action_cfg.base.max_linear_acceleration = 0.20
    action_cfg.base.max_linear_jerk = 0.50
    task_path = os.path.abspath(xgrids2_lab_can_transfer_task.__file__)
    return generate_env_cfg_from_task(
        task_file_path=task_path,
        env_name=XGRIDS2_LAB_MANIP_IK_ENV_NAME,
        robot_cfg=XGRIDS2LabIKManipulationRobotCfg,
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
        env_spacing=12.0,
    )
