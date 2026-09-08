"""Direct registration for the configurable XGRIDS2 scenario registry."""

from __future__ import annotations


_ENV_CACHE: dict[str, tuple[type, str]] = {}


def register_xgrids2_scenario(scenario_name: str) -> tuple[type, str]:
    """Register one scenario without re-importing its generated task module."""

    if scenario_name in _ENV_CACHE:
        return _ENV_CACHE[scenario_name]

    if scenario_name == "can_dining_to_snack":
        from .auto_env_registrations_xgrids2_lab_manipulation import (
            register_xgrids2_lab_manipulation_env,
        )

        result = register_xgrids2_lab_manipulation_env()
        _ENV_CACHE[scenario_name] = result
        return result
    if scenario_name == "plate_dining_to_sink":
        from .auto_env_registrations_xgrids2_lab_plate_sink import (
            register_xgrids2_lab_plate_sink_env,
        )

        result = register_xgrids2_lab_plate_sink_env()
        _ENV_CACHE[scenario_name] = result
        return result

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
    from robolab.tasks.mobile_manipulation.xgrids2_lab_can_transfer_task import (
        XGRIDS2LabManipulationRobotCfg,
    )
    from robolab.tasks.mobile_manipulation.xgrids2_lab_scenario_task import (
        build_xgrids2_scenario_task,
    )
    from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import (
        get_xgrids2_scenario,
    )

    scenario = get_xgrids2_scenario(scenario_name)
    task_class = build_xgrids2_scenario_task(scenario_name)
    observations = generate_obs_cfg(
        {"proprio_obs": KinovaProprioceptionObservationCfg()}
    )
    actions = KinovaMobileContinuousGripperJointPositionActionCfg()
    actions.base.max_linear_acceleration = 0.20
    actions.base.max_linear_jerk = 0.50
    if scenario_name == "can_and_cube_dining_to_snack":
        actions.base.max_linear_acceleration = 0.10
        actions.base.max_linear_jerk = 0.20
    scene_cfg = generate_scene_env_cfg(task_class, XGRIDS2LabManipulationRobotCfg)
    env_cfg = generate_task_env_cfg(
        task_class,
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
    env_cfg.__name__ = f"{scenario.env_name}EnvCfg"
    register_generated_env(env_cfg, scenario.env_name)
    result = (env_cfg, scenario.env_name)
    _ENV_CACHE[scenario_name] = result
    return result
