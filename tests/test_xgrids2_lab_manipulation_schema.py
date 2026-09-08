"""Static gates for XGRIDS2 manipulation geometry and registration."""

from robolab.core.task.task import verify_task_valid
from robolab.registrations.kinova_mobile import (
    XGRIDS2_LAB_MANIP_ENV_NAME,
    register_xgrids2_lab_manipulation_env,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_can_transfer_task import (
    DINING_TABLE_MAP_MASK_RADIUS_M,
    SNACK_TABLE_MAP_MASK_RADIUS_M,
    XGRIDS2LabCanTransferSceneCfg,
    XGRIDS2LabCanTransferTask,
    _distance_to_segment_xy,
    manipulation_base_pose_is_clear,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
    COLLISION_SEGMENTS,
    STAGING_POSES,
    base_pose_is_clear,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_manipulation_geometry import (
    CAN_HALF_HEIGHT,
    CAN_RADIUS,
    CAN_SETTLED_Z,
    DINING_TABLE,
    PICK_CENTER,
    PLACE_CENTER,
    PLACE_REGION,
    SNACK_MANIPULATION_BASE_POSE,
    SNACK_TABLE,
    snack_table_contains_xy,
    tabletop_contains_xy,
)


def test_tabletop_supports_cover_reach_conditioned_object_regions():
    assert tabletop_contains_xy(DINING_TABLE, PICK_CENTER, margin=0.04)
    assert tabletop_contains_xy(SNACK_TABLE, PLACE_CENTER, margin=0.04)
    assert snack_table_contains_xy(PLACE_CENTER, margin=0.04)
    for x in PLACE_REGION[:2]:
        for y in PLACE_REGION[2:]:
            assert snack_table_contains_xy((x, y), margin=CAN_RADIUS)
    assert CAN_SETTLED_Z == DINING_TABLE.surface_z + CAN_HALF_HEIGHT
    assert DINING_TABLE.surface_z > SNACK_TABLE.surface_z


def test_manipulation_keeps_navigation_stages_collision_safe():
    assert base_pose_is_clear(STAGING_POSES["dining_table"][:2])
    assert base_pose_is_clear(STAGING_POSES["snack_table"][:2])
    assert base_pose_is_clear(SNACK_MANIPULATION_BASE_POSE[:2], robot_radius=0.40)
    assert manipulation_base_pose_is_clear(
        SNACK_MANIPULATION_BASE_POSE[:2], robot_radius=0.40
    )


def test_manipulation_scene_has_dynamic_can_and_static_supports():
    scene = XGRIDS2LabCanTransferSceneCfg()
    assert scene.can.spawn.rigid_props.rigid_body_enabled is True
    assert scene.can.spawn.collision_props.collision_enabled is True
    assert scene.dining_table_top.spawn.rigid_props is None
    assert scene.snack_table_top.spawn.rigid_props is None
    assert scene.dining_table_top.spawn.collision_props.collision_enabled is True
    assert scene.snack_table_top.spawn.collision_props.collision_enabled is True


def test_manipulation_scene_masks_table_map_walls_before_instantiation():
    scene = XGRIDS2LabCanTransferSceneCfg()
    removed = 0
    retained = 0
    for segment in COLLISION_SEGMENTS:
        belongs_to_table = (
            _distance_to_segment_xy(DINING_TABLE.center_xyz[:2], segment)
            <= DINING_TABLE_MAP_MASK_RADIUS_M
            or _distance_to_segment_xy(SNACK_TABLE.center_xyz[:2], segment)
            <= SNACK_TABLE_MAP_MASK_RADIUS_M
        )
        if belongs_to_table:
            assert getattr(scene, segment.name) is None
            removed += 1
        else:
            assert getattr(scene, segment.name) is not None
            retained += 1
    assert removed > 0
    assert retained > 0
    assert getattr(scene, "map_175") is None


def test_manipulation_task_is_valid_and_registered():
    valid, error = verify_task_valid(XGRIDS2LabCanTransferTask)
    assert valid, error
    _, name = register_xgrids2_lab_manipulation_env()
    assert name == XGRIDS2_LAB_MANIP_ENV_NAME
