"""Static gates for the XGRIDS2 dining-plate-to-sink task."""

import os

from robolab.constants import PACKAGE_DIR
from robolab.core.task.task import verify_task_valid
from robolab.registrations.kinova_mobile import (
    XGRIDS2_LAB_PLATE_SINK_ENV_NAME,
    register_xgrids2_lab_plate_sink_env,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
    DEFAULT_SAFETY_MARGIN_M,
    PROXY_WIDTH_M,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_plate_sink_geometry import (
    ACTIVE_COLLISION_SEGMENTS,
    EXPANDED_COLLISION_SEGMENTS,
    PLATE_DIAMETER_M,
    PLATE_HELD_SPHERES,
    PLATE_PICK_CENTER,
    PLATE_RADIUS_M,
    PLATE_SETTLED_Z,
    PLATE_USD_RELATIVE,
    SINK_BOTTOM_SURFACE_Z_M,
    SINK_COLLISION_BOXES,
    SINK_OPENING_SIZE_M,
    SINK_PLACE_CENTER,
    SINK_PLACE_SETTLED_Z,
    plate_overhang_m,
    planned_sink_route,
    required_line_clearance,
    route_minimum_clearance,
    sink_contains_plate,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_plate_to_sink_task import (
    XGRIDS2LabPlateToSinkSceneCfg,
    XGRIDS2LabPlateToSinkTask,
)


def test_plate_asset_and_pick_pose_support_contact_grasp():
    assert os.path.isfile(os.path.join(PACKAGE_DIR, PLATE_USD_RELATIVE))
    overhang = plate_overhang_m(PLATE_PICK_CENTER)
    assert 0.04 < overhang < PLATE_RADIUS_M
    assert PLATE_SETTLED_Z > 0.0


def test_sink_opening_contains_horizontal_plate_and_has_full_collision_model():
    assert min(SINK_OPENING_SIZE_M) > PLATE_DIAMETER_M
    assert sink_contains_plate(SINK_PLACE_CENTER, margin=0.02)
    assert SINK_PLACE_SETTLED_Z > SINK_BOTTOM_SURFACE_Z_M
    names = {box.name for box in SINK_COLLISION_BOXES}
    assert names == {
        "sink_counter_left",
        "sink_counter_right",
        "sink_counter_front",
        "sink_counter_back",
        "sink_bottom",
        "sink_wall_left",
        "sink_wall_right",
        "sink_wall_front",
        "sink_wall_back",
    }


def test_held_plate_route_reserves_the_required_clearance():
    carry_radius = 0.47
    route = planned_sink_route(robot_radius=carry_radius)
    required = required_line_clearance(carry_radius, DEFAULT_SAFETY_MARGIN_M)
    assert required == carry_radius + DEFAULT_SAFETY_MARGIN_M + 0.5 * PROXY_WIDTH_M
    assert route_minimum_clearance(route) >= required
    assert 3 <= len(route) <= 6
    assert len(PLATE_HELD_SPHERES) == 9


def test_plate_scene_replaces_cropped_map_and_enables_plate_physics():
    scene = XGRIDS2LabPlateToSinkSceneCfg()
    assert scene.plate.spawn.rigid_props.rigid_body_enabled is True
    assert scene.plate.spawn.collision_props.collision_enabled is True
    assert scene.dining_table_top.spawn.rigid_props is None
    assert scene.sink_bottom.spawn.collision_props.collision_enabled is True
    assert scene.map_000 is None
    assert len(ACTIVE_COLLISION_SEGMENTS) < len(EXPANDED_COLLISION_SEGMENTS)
    for segment in ACTIVE_COLLISION_SEGMENTS:
        assert getattr(scene, segment.name) is not None


def test_plate_task_is_valid_and_registered():
    valid, error = verify_task_valid(XGRIDS2LabPlateToSinkTask)
    assert valid, error
    _, name = register_xgrids2_lab_plate_sink_env()
    assert name == XGRIDS2_LAB_PLATE_SINK_ENV_NAME
