"""Audited scenario variants for the GenManip kitchen transfer task."""

from __future__ import annotations

import math
from dataclasses import dataclass

from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
    Bounds2D,
    ORACLE_PICK_REGION,
    PLACE_SIDE_GRASP_QUAT_WXYZ,
    ROBOT_BASE_Z,
    SIDE_GRASP_QUAT_WXYZ,
    STAGING_POSES,
)


@dataclass(frozen=True)
class KitchenManipulationScenario:
    """Task reset plus collision-audited base routes for one scenario."""

    name: str
    initial_base_pose: tuple[float, float, float, float]
    pick_region: Bounds2D
    pick_support_name: str
    pick_grasp_quat_wxyz: tuple[float, float, float, float]
    pick_pregrasp_offset_xy: tuple[float, float]
    pre_pick_waypoints: tuple[tuple[float, float, float], ...] = ()
    carry_waypoints: tuple[tuple[float, float, float], ...] = ()
    pre_pick_base_speed: float = 0.18
    carry_base_speed: float = 0.05
    pregrasp_position_tolerance: float = 0.012
    pick_base_position_tolerance: float = 0.025
    pick_base_yaw_tolerance: float = 0.06
    initial_arm_joint_pos: tuple[float, ...] | None = None


# The west worktop itself spans x [-0.70, -0.48]. This conservative interior
# mirrors the y-range used by the east-side oracle and keeps a can radius away
# from both x edges.
WEST_ORACLE_PICK_REGION = Bounds2D(-0.63, -0.55, 0.12, 0.23)

# A base center fits between the west island clearance boundary (x=-1.05) and
# the left-wall clearance boundary (x=-1.39). The north route is required:
# the L-shaped counter closes the corresponding south passage.
ROBOT_OTHER_SIDE_START = (-1.18, 0.175, ROBOT_BASE_Z, math.pi)
WEST_PICK_STAGE = (-1.10, 0.14, ROBOT_BASE_Z, math.pi)
NORTH_WEST_WAYPOINT = (-1.18, 1.50, math.pi)
NORTH_EAST_WAYPOINT = (0.58, 1.50, 0.0)
EAST_PICK_WAYPOINT = (*STAGING_POSES["pick"][:2], 0.0)
# Reach the placement stage with the pick-facing heading first, then rotate in
# place. Keeping translation and rotation as separate waypoints avoids a wide
# carried-object arc near the fixture while making the fixed Kinect face the
# counter before arm placement starts.
PLACE_APPROACH_WAYPOINT = (*STAGING_POSES["place"][:2], 0.0)
PLACE_WAYPOINT = (*STAGING_POSES["place"][:2], STAGING_POSES["place"][3])
BASELINE_SCENARIO = KitchenManipulationScenario(
    name="baseline",
    initial_base_pose=STAGING_POSES["pick"],
    pick_region=ORACLE_PICK_REGION,
    pick_support_name="island_top_east",
    pick_grasp_quat_wxyz=SIDE_GRASP_QUAT_WXYZ,
    pick_pregrasp_offset_xy=(0.12, 0.0),
    carry_waypoints=(PLACE_APPROACH_WAYPOINT, PLACE_WAYPOINT),
)

ROBOT_OTHER_SIDE_SCENARIO = KitchenManipulationScenario(
    name="robot_other_side",
    initial_base_pose=ROBOT_OTHER_SIDE_START,
    pick_region=ORACLE_PICK_REGION,
    pick_support_name="island_top_east",
    pick_grasp_quat_wxyz=SIDE_GRASP_QUAT_WXYZ,
    pick_pregrasp_offset_xy=(0.12, 0.0),
    pre_pick_waypoints=(
        NORTH_WEST_WAYPOINT,
        NORTH_EAST_WAYPOINT,
        EAST_PICK_WAYPOINT,
    ),
    carry_waypoints=(PLACE_APPROACH_WAYPOINT, PLACE_WAYPOINT),
)

CAN_OTHER_SIDE_SCENARIO = KitchenManipulationScenario(
    name="can_other_side",
    initial_base_pose=STAGING_POSES["pick"],
    pick_region=WEST_ORACLE_PICK_REGION,
    pick_support_name="island_top_west",
    # This is the same upright side grasp approached from world -X.
    pick_grasp_quat_wxyz=PLACE_SIDE_GRASP_QUAT_WXYZ,
    pick_pregrasp_offset_xy=(-0.12, 0.0),
    pre_pick_waypoints=(
        NORTH_EAST_WAYPOINT,
        NORTH_WEST_WAYPOINT,
        (*WEST_PICK_STAGE[:2], math.pi),
    ),
    carry_waypoints=(
        NORTH_WEST_WAYPOINT,
        NORTH_EAST_WAYPOINT,
        PLACE_APPROACH_WAYPOINT,
        PLACE_WAYPOINT,
    ),
    carry_base_speed=0.10,
    pregrasp_position_tolerance=0.016,
    pick_base_position_tolerance=0.005,
    pick_base_yaw_tolerance=0.015,
)

SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        BASELINE_SCENARIO,
        ROBOT_OTHER_SIDE_SCENARIO,
        CAN_OTHER_SIDE_SCENARIO,
    )
}


def get_kitchen_scenario(name: str) -> KitchenManipulationScenario:
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown kitchen scenario {name!r}; choose one of {sorted(SCENARIOS)}"
        ) from error
