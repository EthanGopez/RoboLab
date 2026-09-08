"""Audited task geometry for XGRIDS2 dining-to-snack can transfer."""

from __future__ import annotations

import math
from dataclasses import dataclass

from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
    CAN_HALF_HEIGHT,
    CAN_RADIUS,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import ROBOT_BASE_Z_M


@dataclass(frozen=True)
class Tabletop:
    name: str
    center_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    yaw: float = 0.0

    @property
    def surface_z(self) -> float:
        return self.center_xyz[2] + 0.5 * self.size_xyz[2]


DINING_TABLE = Tabletop("dining_table_top", (0.84, -0.72, 0.843), (1.40, 0.85, 0.06), 2.29876)
SNACK_TABLE = Tabletop("snack_table_top", (5.60, -1.80, 0.682), (1.05, 1.05, 0.06), 0.0)
TABLETOPS = (DINING_TABLE, SNACK_TABLE)

# Reach-conditioned near-edge regions; each is inside its physical support.
PICK_REGION = (1.23, 1.27, -1.20, -1.17)
SNACK_TABLE_RADIUS_M = 0.5 * min(SNACK_TABLE.size_xyz[:2])
# The visible snack table is round.  Use its robot-facing quadrant so the
# loaded Kinova does not have to place at its near-singular maximum reach.
# Every corner keeps at least one complete can radius inside the tabletop edge.
PLACE_REGION = (5.20, 5.24, -1.60, -1.52)
PICK_CENTER = (0.5 * (PICK_REGION[0] + PICK_REGION[1]), 0.5 * (PICK_REGION[2] + PICK_REGION[3]))
PLACE_CENTER = (0.5 * (PLACE_REGION[0] + PLACE_REGION[1]), 0.5 * (PLACE_REGION[2] + PLACE_REGION[3]))
SNACK_MANIPULATION_BASE_POSE = (
    4.48,
    -1.14,
    ROBOT_BASE_Z_M,
    math.atan2(PLACE_CENTER[1] + 1.14, PLACE_CENTER[0] - 4.48),
)
CAN_DROP_HEIGHT = 0.02
CAN_INITIAL_Z = DINING_TABLE.surface_z + CAN_HALF_HEIGHT + CAN_DROP_HEIGHT
CAN_SETTLED_Z = DINING_TABLE.surface_z + CAN_HALF_HEIGHT
CAN_PLACE_SETTLED_Z = SNACK_TABLE.surface_z + CAN_HALF_HEIGHT


def tabletop_contains_xy(table: Tabletop, point: tuple[float, float], margin: float = 0.0) -> bool:
    """Test an XY point against a yawed tabletop rectangle."""

    dx, dy = point[0] - table.center_xyz[0], point[1] - table.center_xyz[1]
    cosine, sine = math.cos(table.yaw), math.sin(table.yaw)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    return (
        abs(local_x) <= 0.5 * table.size_xyz[0] - margin
        and abs(local_y) <= 0.5 * table.size_xyz[1] - margin
    )


def snack_table_contains_xy(point: tuple[float, float], margin: float = 0.0) -> bool:
    """Test a point against the visible round snack tabletop."""

    return math.dist(point, SNACK_TABLE.center_xyz[:2]) <= SNACK_TABLE_RADIUS_M - margin
