"""Isaac configuration gates for generated XGRIDS2 scenarios."""

import pytest

from robolab.core.task.task import verify_task_valid
from robolab.registrations.kinova_mobile import register_xgrids2_scenario
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenario_task import (
    build_xgrids2_scenario_task,
    scenario_grasp_joint_name,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import (
    get_xgrids2_scenario,
)


GENERATED_SCENARIOS = (
    "can_and_cube_dining_to_snack",
    "lime_clutter_to_bowl",
    "mustard_pour_into_bowl",
)


@pytest.mark.parametrize("scenario_name", GENERATED_SCENARIOS)
def test_generated_task_and_environment_are_valid(scenario_name):
    scenario = get_xgrids2_scenario(scenario_name)
    task = build_xgrids2_scenario_task(scenario_name)
    valid, error = verify_task_valid(task)
    assert valid, error
    env_cfg_class, env_name = register_xgrids2_scenario(scenario_name)
    assert env_name == scenario.env_name
    env_cfg = env_cfg_class()
    assert env_cfg.scene.replicate_physics is False
    assert env_cfg.episode_length_s == scenario.episode_length_s
    assert set(item.name for item in scenario.objects) <= set(
        env_cfg.contact_object_list
    )


def test_generated_scene_has_all_multi_object_targets_and_supports():
    task = build_xgrids2_scenario_task("can_and_cube_dining_to_snack")
    scene = task.scene()
    assert scene.can.spawn.rigid_props.rigid_body_enabled is True
    assert scene.cube.spawn.rigid_props.rigid_body_enabled is True
    assert scene.dining_table_top.spawn.collision_props.collision_enabled is True
    assert scene.snack_table_top.spawn.collision_props.collision_enabled is True
    assert task.events().author_oracle_grasp_joints.params["object_names"] == (
        "can",
        "cube",
    )


def test_clutter_scene_contains_dynamic_distractors_and_kinematic_bowl():
    task = build_xgrids2_scenario_task("lime_clutter_to_bowl")
    scene = task.scene()
    assert scene.lime.spawn.rigid_props.kinematic_enabled is False
    assert scene.clutter_can.spawn.rigid_props.kinematic_enabled is False
    assert scene.target_bowl.spawn.rigid_props.kinematic_enabled is True


def test_scenario_retention_joint_names_are_independent():
    assert scenario_grasp_joint_name("can") == "oracle_grasp_joint_can"
    assert scenario_grasp_joint_name("cube") == "oracle_grasp_joint_cube"
    assert scenario_grasp_joint_name("can") != scenario_grasp_joint_name("cube")


def test_reference_scenarios_keep_legacy_registration_names():
    for name in ("can_dining_to_snack", "plate_dining_to_sink"):
        scenario = get_xgrids2_scenario(name)
        _, env_name = register_xgrids2_scenario(name)
        assert env_name == scenario.env_name
