"""Static contracts for the configurable XGRIDS2 benchmark pack."""

from dataclasses import FrozenInstanceError

import pytest

from robolab.constants import PACKAGE_DIR
from robolab.tasks.mobile_manipulation.xgrids2_lab_can_transfer_task import (
    MANIPULATION_COLLISION_SEGMENTS,
    manipulation_base_pose_is_clear,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
    planned_route_between,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import (
    CAN,
    DINING_TABLE,
    SCENARIOS,
    SNACK_TABLE,
    TOP_GRASP_DEPARTURE_POSE,
    XGRIDS2ScenarioSpec,
    get_xgrids2_scenario,
    list_xgrids2_scenarios,
    validate_xgrids2_scenario,
    xgrids2_scenario_manifest_entry,
)


EXPECTED_SCENARIOS = (
    "can_dining_to_snack",
    "plate_dining_to_sink",
    "can_and_cube_dining_to_snack",
    "lime_clutter_to_bowl",
    "mustard_pour_into_bowl",
)


def test_registry_contains_the_five_task_pack_in_stable_order():
    assert list_xgrids2_scenarios() == EXPECTED_SCENARIOS
    assert tuple(SCENARIOS) == EXPECTED_SCENARIOS
    assert len({scenario.env_name for scenario in SCENARIOS.values()}) == 5
    assert len({scenario.task_name for scenario in SCENARIOS.values()}) == 5


@pytest.mark.parametrize("name", EXPECTED_SCENARIOS)
def test_every_scenario_is_complete_and_uses_existing_assets(name):
    scenario = get_xgrids2_scenario(name)
    assert isinstance(scenario, XGRIDS2ScenarioSpec)
    assert validate_xgrids2_scenario(scenario, package_root=PACKAGE_DIR) == ()
    assert scenario.legs
    assert scenario.camera_shots
    assert scenario.instruction.endswith(".")


def test_scenario_records_are_immutable_and_unknown_names_fail_helpfully():
    with pytest.raises(FrozenInstanceError):
        CAN.mass_kg = 1.0
    with pytest.raises(ValueError, match="Unknown XGRIDS2 scenario"):
        get_xgrids2_scenario("not_a_scenario")


def test_reference_can_scenario_preserves_validated_geometry():
    scenario = get_xgrids2_scenario("can_dining_to_snack")
    leg = scenario.legs[0]
    assert scenario.objects == (CAN,)
    assert scenario.supports == (DINING_TABLE, SNACK_TABLE)
    assert CAN.spawn_region_xy == (1.23, 1.27, -1.20, -1.17)
    assert leg.place.target_region_xy == (5.20, 5.24, -1.60, -1.52)
    assert leg.route is not None
    assert leg.route.runtime_radius_m == 0.40
    assert leg.route.planning_radius_m == 0.42
    assert scenario.episode_length_s == 600


def test_configured_top_grasp_route_endpoints_are_clear():
    scenario = get_xgrids2_scenario("lime_clutter_to_bowl")
    route = scenario.legs[0].route
    assert route is not None
    assert manipulation_base_pose_is_clear(
        route.approach_xy, robot_radius=route.planning_radius_m
    )
    destination = route.destination_base_pose_xyzyaw[:2]
    assert manipulation_base_pose_is_clear(
        destination, robot_radius=route.runtime_radius_m
    )
    for index in range(21):
        fraction = index / 20.0
        point = tuple(
            start + fraction * (end - start)
            for start, end in zip(route.approach_xy, destination)
        )
        assert manipulation_base_pose_is_clear(
            point, robot_radius=route.runtime_radius_m
        )
    planned_route_between(
        TOP_GRASP_DEPARTURE_POSE[:2],
        route.approach_xy,
        robot_radius=route.planning_radius_m,
        collision_segments=MANIPULATION_COLLISION_SEGMENTS,
    )


def test_multi_object_scenario_is_one_ordered_episode():
    scenario = get_xgrids2_scenario("can_and_cube_dining_to_snack")
    assert scenario.strategy == "sequential"
    assert tuple(leg.object_name for leg in scenario.legs) == ("can", "cube")
    assert scenario.legs[0].return_to_source_after is True
    assert scenario.legs[1].return_to_source_after is False
    assert scenario.legs[0].place.center_xy != scenario.legs[1].place.center_xy
    assert scenario.legs[0].place.reorientation_steps == 3
    assert scenario.legs[0].place.tool_quaternion_wxyz is not None
    assert scenario.legs[0].place.intermediate_clearances_m == (0.09,)
    assert scenario.legs[0].place.preplace_tracking_tolerance_rad == 0.060
    assert scenario.legs[0].place.placement_target_step_rad == 0.001
    assert scenario.legs[0].place.placement_minimum_tracking_steps == 2
    cube_leg = scenario.legs[1]
    assert cube_leg.route is not None
    assert cube_leg.route.planning_radius_m == 0.43
    assert cube_leg.route.max_angular_speed_rad_s == 0.10
    assert cube_leg.place.release_from_approach is True
    assert cube_leg.place.center_xy == pytest.approx((5.30, -1.64))
    assert cube_leg.place.preferred_center_offset_xy == (-0.025, -0.025)
    assert "cube_10cm_physics_settled_release_with_xy_bias" in (
        scenario.limitations
    )
    assert cube_leg.place.preplace_tracking_tolerance_rad == 0.075
    assert cube_leg.place.preplace_endpoint_tolerance_rad == 0.075
    assert cube_leg.place.release_clearance_m == cube_leg.place.approach_clearance_m
    assert cube_leg.place.placement_preflight_candidate_count == 6
    assert scenario.episode_length_s == 3600


def test_clutter_and_pour_contracts_are_explicit():
    clutter = get_xgrids2_scenario("lime_clutter_to_bowl")
    assert clutter.episode_length_s == 1200
    assert clutter.legs[0].place.preplace_target_step_rad == 0.002
    assert clutter.legs[0].place.preplace_tracking_tolerance_rad == 0.040
    assert clutter.legs[0].place.preplace_endpoint_tolerance_rad == 0.050
    assert sum(item.distractor for item in clutter.objects) == 6
    assert {
        item.name for item in clutter.objects if item.name.startswith("destination_")
    } == {"destination_cube", "destination_can", "destination_lemon"}
    assert clutter.legs[0].place.kind == "container"
    assert clutter.support("target_bowl").opening_size_xy == (0.145, 0.145)
    assert clutter.legs[0].place.preferred_orientation_mode == "side_release"
    assert clutter.legs[0].place.preferred_yaw_offset_degrees == -135.0

    pouring = get_xgrids2_scenario("mustard_pour_into_bowl")
    assert pouring.strategy == "pour"
    assert pouring.pour is not None
    assert pouring.pour.tilt_degrees == 110.0
    assert pouring.pour.hold_control_steps == 30
    assert "pose_level_pour_no_fluid" in pouring.limitations


def test_manifest_entries_are_json_safe_and_disclose_benchmark_eligibility():
    entry = xgrids2_scenario_manifest_entry(
        get_xgrids2_scenario("mustard_pour_into_bowl")
    )
    assert entry["retention_mode"] == "development_only_not_benchmark_eligible"
    assert xgrids2_scenario_manifest_entry(
        get_xgrids2_scenario("can_dining_to_snack")
    )["retention_mode"] == "forbidden"
    assert entry["pouring_fidelity"] == "pose_level_pour_no_fluid"
    assert isinstance(entry["objects"], tuple)
    provenance = entry["asset_provenance"]
    assert provenance["vendor_appearance"]["usage"] == (
        "collision_free_appearance_only"
    )
    entities = {item["entity"] for item in provenance["spawned_physics_assets"]}
    assert {"mustard_bottle", "pour_bowl"} <= entities
