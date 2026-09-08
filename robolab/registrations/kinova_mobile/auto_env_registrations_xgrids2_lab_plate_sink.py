"""Register the XGRIDS2 dining-plate-to-sink environment."""

from __future__ import annotations

XGRIDS2_LAB_PLATE_SINK_ENV_NAME = "XGRIDS2LabPlateToSinkKinovaMobileCurobo"


def register_xgrids2_lab_plate_sink_env() -> tuple[type, str]:
    from robolab.core.environments.config import (
        generate_scene_env_cfg,
        generate_task_env_cfg,
        register_generated_env,
    )
    from robolab.core.observations.observation_utils import generate_obs_cfg
    from robolab.robots.kinova_gen3 import KinovaProprioceptionObservationCfg
    from robolab.robots.kinova_mobile_cobot import (
        KinovaMobileContinuousGripperJointPositionActionCfg,
        MOBILE_KINOVA_CONTACT_GRIPPER,
    )
    from robolab.tasks.mobile_manipulation.xgrids2_lab_plate_to_sink_task import (
        XGRIDS2LabPlateRobotCfg,
        XGRIDS2LabPlateToSinkTask,
    )

    observations = generate_obs_cfg(
        {"proprio_obs": KinovaProprioceptionObservationCfg()}
    )
    actions = KinovaMobileContinuousGripperJointPositionActionCfg()
    actions.base.max_linear_acceleration = 0.06
    actions.base.max_linear_jerk = 0.12
    actions.base.max_angular_acceleration = 0.15
    actions.base.max_angular_jerk = 0.30
    scene_cfg = generate_scene_env_cfg(
        XGRIDS2LabPlateToSinkTask,
        XGRIDS2LabPlateRobotCfg,
    )
    env_cfg = generate_task_env_cfg(
        XGRIDS2LabPlateToSinkTask,
        scene_cfg,
        observations(),
        actions,
        MOBILE_KINOVA_CONTACT_GRIPPER,
        dt=1.0 / 120.0,
        render_interval=4,
        decimation=4,
        seed=0,
        num_envs=1,
        env_spacing=12.0,
    )
    env_cfg.__name__ = f"{XGRIDS2_LAB_PLATE_SINK_ENV_NAME}EnvCfg"
    register_generated_env(env_cfg, XGRIDS2_LAB_PLATE_SINK_ENV_NAME)
    return env_cfg, XGRIDS2_LAB_PLATE_SINK_ENV_NAME
