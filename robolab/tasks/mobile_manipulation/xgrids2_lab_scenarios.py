"""Configuration contracts and registry for XGRIDS2 manipulation scenarios.

The records in this module deliberately avoid Isaac Sim imports.  They are the
single, statically testable description consumed by scene generation, oracle
execution, evidence recording, and benchmark manifests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Literal


Quaternion = tuple[float, float, float, float]
Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
Region2D = tuple[float, float, float, float]


@dataclass(frozen=True)
class XGRIDS2ObjectSpec:
    """One spawned object and its conservative manipulation geometry."""

    name: str
    usd_relative: str
    dimensions_xyz: Vector3
    mass_kg: float
    spawn_region_xy: Region2D
    spawn_z: float
    spawn_quaternion_wxyz: Quaternion = (1.0, 0.0, 0.0, 0.0)
    collision_approximation: Literal["asset", "convex_hull", "bounding_box"] = "asset"
    manipulated: bool = False
    distractor: bool = False
    kinematic: bool = False
    held_spheres: tuple[tuple[float, float, float, float], ...] = ()

    @property
    def center_xy(self) -> Vector2:
        xmin, xmax, ymin, ymax = self.spawn_region_xy
        return (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))


@dataclass(frozen=True)
class XGRIDS2SupportSpec:
    """A clean support or receiver used by physics and success checks."""

    name: str
    kind: Literal["rectangle", "circle", "bowl", "sink"]
    center_xyz: Vector3
    size_xyz: Vector3
    yaw_rad: float = 0.0
    usd_relative: str | None = None
    opening_size_xy: Vector2 | None = None
    collision_enabled: bool = True

    @property
    def surface_z(self) -> float:
        return self.center_xyz[2] + 0.5 * self.size_xyz[2]


@dataclass(frozen=True)
class XGRIDS2GraspSpec:
    """Object-aware grasp and loaded-motion contract."""

    kind: Literal["side", "top", "rim", "spout"]
    approach_direction_xy: Vector2
    grasp_quaternion_wxyz: Quaternion
    pregrasp_distance_m: float
    pregrasp_height_m: float
    lift_height_m: float
    retract_distance_m: float
    contact_mode: Literal["bilateral", "either"] = "either"
    contact_force_n: float = 1.0
    carry_radius_m: float = 0.42
    continuous_branch_margin_rad: float = 0.25


@dataclass(frozen=True)
class XGRIDS2PlaceSpec:
    """Destination pose and geometric success contract."""

    kind: Literal["surface", "container", "sink", "pour"]
    support_name: str
    target_region_xy: Region2D
    settled_z: float
    target_quaternion_wxyz: Quaternion = (1.0, 0.0, 0.0, 0.0)
    xy_tolerance_m: float = 0.08
    z_tolerance_m: float = 0.03
    containment_margin_m: float = 0.005
    preplace_clearance_m: float = 0.20
    approach_clearance_m: float = 0.10
    release_clearance_m: float = 0.045
    preferred_orientation_mode: Literal[
        "preserve_carried_tilt",
        "side_release",
        "oblique_top_50",
        "oblique_top_65",
        "oblique_top_80",
    ] | None = None
    preferred_yaw_offset_degrees: float | None = None
    preferred_center_offset_xy: Vector2 | None = None
    tool_quaternion_wxyz: Quaternion | None = None
    approach_direction_xy: Vector2 = (0.0, 0.0)
    preplace_offset_m: float = 0.0
    intermediate_clearances_m: tuple[float, ...] = ()
    reorientation_steps: int = 0
    preplace_target_step_rad: float = 0.002
    preplace_tracking_tolerance_rad: float = 0.050
    preplace_endpoint_tolerance_rad: float = 0.060
    placement_target_step_rad: float = 0.004
    placement_tracking_tolerance_rad: float = 0.050
    placement_endpoint_tolerance_rad: float = 0.060
    placement_minimum_tracking_steps: int = 1
    placement_preflight_candidate_count: int = 1
    release_from_approach: bool = False

    @property
    def center_xy(self) -> Vector2:
        xmin, xmax, ymin, ymax = self.target_region_xy
        return (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))


@dataclass(frozen=True)
class XGRIDS2RouteSpec:
    """Mobile-base staging and collision-inflation parameters."""

    source_base_pose_xyzyaw: tuple[float, float, float, float]
    destination_base_pose_xyzyaw: tuple[float, float, float, float]
    approach_xy: Vector2
    planning_radius_m: float
    runtime_radius_m: float
    max_linear_speed_m_s: float = 0.06
    max_angular_speed_rad_s: float = 0.35


@dataclass(frozen=True)
class XGRIDS2TransferLegSpec:
    """One pick/carry/place leg in a possibly sequential scenario."""

    name: str
    object_name: str
    source_support_name: str
    grasp: XGRIDS2GraspSpec
    place: XGRIDS2PlaceSpec
    route: XGRIDS2RouteSpec | None
    return_to_source_after: bool = False
    staging_poses: tuple[tuple[float, float, float, float], ...] = ()
    success_predicate: Literal[
        "settled_surface", "contained_receiver", "pose_level_pour_no_fluid"
    ] = "settled_surface"


@dataclass(frozen=True)
class XGRIDS2PourSpec:
    """Pose-level pouring baseline; it intentionally models no liquid."""

    object_name: str
    receiver_name: str
    spout_local_xyz: Vector3
    tilt_degrees: float = 110.0
    tilt_tolerance_degrees: float = 10.0
    return_upright_tolerance_degrees: float = 15.0
    hold_control_steps: int = 30
    spout_height_range_m: tuple[float, float] = (0.04, 0.12)
    opening_margin_m: float = 0.015
    return_to_pick_pose: bool = True


@dataclass(frozen=True)
class XGRIDS2CameraShotSpec:
    name: str
    eye: Vector3
    target: Vector3
    base_anchor_xy: Vector2 | None = None
    activation_radius_m: float = 0.25


@dataclass(frozen=True)
class XGRIDS2ScenarioSpec:
    """Complete scenario definition shared by every XGRIDS2 subsystem."""

    name: str
    task_name: str
    env_name: str
    instruction: str
    difficulty: Literal["reference", "moderate", "complex", "baseline"]
    capability_tags: tuple[str, ...]
    objects: tuple[XGRIDS2ObjectSpec, ...]
    supports: tuple[XGRIDS2SupportSpec, ...]
    legs: tuple[XGRIDS2TransferLegSpec, ...]
    camera_shots: tuple[XGRIDS2CameraShotSpec, ...]
    strategy: Literal["generic_transfer", "plate_sink", "sequential", "pour"]
    pour: XGRIDS2PourSpec | None = None
    episode_length_s: int = 300
    limitations: tuple[str, ...] = (
        "contact_verified_temporary_fixed_joint_retention",
    )
    metadata: dict[str, str] = field(default_factory=dict)

    def object(self, name: str) -> XGRIDS2ObjectSpec:
        return next(item for item in self.objects if item.name == name)

    def support(self, name: str) -> XGRIDS2SupportSpec:
        return next(item for item in self.supports if item.name == name)


DINING_TABLE = XGRIDS2SupportSpec(
    "dining_table_top",
    "rectangle",
    (0.84, -0.72, 0.843),
    (1.40, 0.85, 0.06),
    2.29876,
)
SNACK_TABLE = XGRIDS2SupportSpec(
    "snack_table_top",
    "circle",
    (5.60, -1.80, 0.682),
    (1.05, 1.05, 0.06),
)
SINK = XGRIDS2SupportSpec(
    "sink_basin",
    "sink",
    (-2.4369233721, 2.2386716370, 0.726),
    (0.56, 0.42, 0.02),
    0.7508501299,
    opening_size_xy=(0.56, 0.42),
)

DINING_BASE_POSE = (1.7656, -1.7639, 0.13335, 2.29876)
TOP_GRASP_DINING_BASE_POSE = (1.30, -1.45, 0.13335, 2.1587989303)
TOP_GRASP_DEPARTURE_POSE = (1.45, -1.60, 0.13335, 2.1587989303)
SNACK_BASE_POSE = (4.48, -1.14, 0.13335, 0.3487710036)
TOP_GRASP_SNACK_BASE_POSE = (4.58, -1.31, 0.13335, -0.3655101930)
CUBE_SNACK_BASE_POSE = (4.56, -1.29, 0.13335, -0.3655101930)
SINK_BASE_POSE = (-1.7888, 1.5441, 0.13335, 2.3216464567)

CAN_GRASP = XGRIDS2GraspSpec(
    "side",
    (-0.6658, 0.7461),
    (-0.2520023, -0.2520023, 0.6606776, 0.6606776),
    0.12,
    0.12,
    0.14,
    0.12,
    "bilateral",
    carry_radius_m=0.40,
)
TOP_GRASP = XGRIDS2GraspSpec(
    "top",
    (-0.6658, 0.7461),
    (0.0, 0.7071068, 0.7071068, 0.0),
    0.12,
    0.10,
    0.22,
    0.10,
    "either",
    carry_radius_m=0.36,
)
LIME_TOP_GRASP = XGRIDS2GraspSpec(
    "top",
    (-0.6658, 0.7461),
    (-0.1569520197, 0.2834480893, 0.8518831934, 0.4114830844),
    0.12,
    0.10,
    0.22,
    0.10,
    "either",
    carry_radius_m=0.36,
    continuous_branch_margin_rad=0.10,
)
BOTTLE_GRASP = XGRIDS2GraspSpec(
    "spout",
    (-0.6658, 0.7461),
    (-0.2520023, -0.2520023, 0.6606776, 0.6606776),
    0.12,
    0.08,
    0.18,
    0.10,
    "either",
    carry_radius_m=0.42,
)

CAN = XGRIDS2ObjectSpec(
    "can",
    "assets/objects/ycb/tomato_soup_can_physics.usda",
    (0.067854, 0.067750, 0.102037),
    0.45,
    (1.23, 1.27, -1.20, -1.17),
    DINING_TABLE.surface_z + 0.0510185 + 0.02,
    manipulated=True,
    held_spheres=tuple((0.0, 0.0, z, 0.0349) for z in (-0.026, 0.0, 0.026)),
)
PLATE = XGRIDS2ObjectSpec(
    "plate",
    "assets/objects/vomp/plate_small/plate_small.usd",
    (0.2976294, 0.2976294, 0.0211513),
    0.25,
    (1.260, 1.272, -1.208, -1.196),
    DINING_TABLE.surface_z + 0.01057565 + 0.012,
    manipulated=True,
)
CUBE = XGRIDS2ObjectSpec(
    "cube",
    "assets/objects/hot3d/rubiks_cube.usd",
    (0.058286, 0.057771, 0.057964),
    0.20,
    (0.995, 1.005, -1.005, -0.995),
    DINING_TABLE.surface_z + 0.028982 + 0.012,
    manipulated=True,
    held_spheres=((0.0, 0.0, 0.0, 0.051),),
)
LIME = XGRIDS2ObjectSpec(
    "lime",
    "assets/objects/fruits_veggies/lime.usd",
    (0.075507, 0.061338, 0.060036),
    0.10,
    (1.235, 1.245, -1.185, -1.175),
    DINING_TABLE.surface_z + 0.030018 + 0.012,
    manipulated=True,
    held_spheres=((0.0, 0.0, 0.0, 0.039),),
)
MUSTARD = XGRIDS2ObjectSpec(
    "mustard_bottle",
    "assets/objects/hope/mustard_bottle.usd",
    (0.048574, 0.065133, 0.160050),
    0.30,
    (1.235, 1.245, -1.185, -1.175),
    DINING_TABLE.surface_z + 0.080025 + 0.012,
    manipulated=True,
    held_spheres=tuple((0.0, 0.0, z, 0.034) for z in (-0.055, -0.018, 0.018, 0.055)),
)


def _distractor(
    name: str,
    usd_relative: str,
    dimensions_xyz: Vector3,
    mass_kg: float,
    xy: Vector2,
    support_z: float,
) -> XGRIDS2ObjectSpec:
    return XGRIDS2ObjectSpec(
        name,
        usd_relative,
        dimensions_xyz,
        mass_kg,
        (xy[0], xy[0], xy[1], xy[1]),
        support_z + 0.5 * dimensions_xyz[2] + 0.002,
        distractor=True,
    )


SOURCE_CLUTTER = (
    _distractor(
        "clutter_can",
        CAN.usd_relative,
        CAN.dimensions_xyz,
        CAN.mass_kg,
        (1.10, -1.03),
        DINING_TABLE.surface_z,
    ),
    _distractor(
        "clutter_cube",
        CUBE.usd_relative,
        CUBE.dimensions_xyz,
        CUBE.mass_kg,
        (1.095, -1.31),
        DINING_TABLE.surface_z,
    ),
    _distractor(
        "clutter_lemon", "assets/objects/fruits_veggies/lemon2.usd",
        (0.060369, 0.039673, 0.039534), 0.08, (1.385, -1.05),
        DINING_TABLE.surface_z,
    ),
)

BOWL_SNACK = XGRIDS2SupportSpec(
    "target_bowl",
    "bowl",
    (5.24, -1.58, SNACK_TABLE.surface_z + 0.027485),
    (0.161444, 0.161113, 0.054970),
    usd_relative="assets/objects/ycb/bowl.usd",
    opening_size_xy=(0.145, 0.145),
)
DESTINATION_CLUTTER = (
    _distractor(
        "destination_cube",
        CUBE.usd_relative,
        CUBE.dimensions_xyz,
        CUBE.mass_kg,
        (5.14, -1.72),
        SNACK_TABLE.surface_z,
    ),
    _distractor(
        "destination_can",
        CAN.usd_relative,
        CAN.dimensions_xyz,
        CAN.mass_kg,
        (5.39, -1.45),
        SNACK_TABLE.surface_z,
    ),
    _distractor(
        "destination_lemon",
        "assets/objects/fruits_veggies/lemon2.usd",
        (0.060369, 0.039673, 0.039534),
        0.08,
        (5.36, -1.77),
        SNACK_TABLE.surface_z,
    ),
)
BOWL_DINING = XGRIDS2SupportSpec(
    "pour_bowl",
    "bowl",
    (1.12, -1.02, DINING_TABLE.surface_z + 0.027485),
    (0.161444, 0.161113, 0.054970),
    usd_relative="assets/objects/ycb/bowl.usd",
    opening_size_xy=(0.145, 0.145),
)

TRANSFER_ROUTE = XGRIDS2RouteSpec(
    DINING_BASE_POSE,
    SNACK_BASE_POSE,
    (4.46, -1.14),
    0.42,
    0.40,
    max_linear_speed_m_s=0.04,
    max_angular_speed_rad_s=0.20,
)
TOP_GRASP_TRANSFER_ROUTE = XGRIDS2RouteSpec(
    TOP_GRASP_DINING_BASE_POSE,
    TOP_GRASP_SNACK_BASE_POSE,
    (4.55, -1.25),
    0.41,
    0.36,
    max_linear_speed_m_s=0.04,
    max_angular_speed_rad_s=0.20,
)
CUBE_TRANSFER_ROUTE = XGRIDS2RouteSpec(
    TOP_GRASP_DINING_BASE_POSE,
    CUBE_SNACK_BASE_POSE,
    (4.45, -1.125),
    0.43,
    0.36,
    max_linear_speed_m_s=0.04,
    max_angular_speed_rad_s=0.10,
)
PLATE_ROUTE = XGRIDS2RouteSpec(
    DINING_BASE_POSE,
    SINK_BASE_POSE,
    (-1.80, 1.52),
    0.47,
    0.47,
    max_angular_speed_rad_s=0.10,
)

COMMON_SHOTS = (
    XGRIDS2CameraShotSpec("dining", (2.10, -2.25, 1.75), (1.20, -1.13, 0.90), DINING_BASE_POSE[:2]),
    XGRIDS2CameraShotSpec("route", (0.0, 0.0, 1.625), (4.0, -1.5, 0.8)),
    XGRIDS2CameraShotSpec("snack", (4.45, -2.25, 1.65), (5.331, -1.531, 0.75), SNACK_BASE_POSE[:2]),
)
SINK_SHOTS = (
    COMMON_SHOTS[0],
    XGRIDS2CameraShotSpec("route", (0.15, 0.20, 2.15), (-0.20, 0.10, 0.72)),
    XGRIDS2CameraShotSpec("sink", (-1.25, 1.20, 1.70), (-2.44, 2.24, 0.82), SINK_BASE_POSE[:2]),
)

CAN_PLACE = XGRIDS2PlaceSpec(
    "surface",
    "snack_table_top",
    (5.20, 5.24, -1.60, -1.52),
    SNACK_TABLE.surface_z + 0.0510185,
    preplace_clearance_m=0.14,
    approach_clearance_m=0.14,
    release_clearance_m=0.040,
    tool_quaternion_wxyz=(0.6532815, 0.6532815, 0.2705981, 0.2705981),
    approach_direction_xy=(0.7071068, -0.7071068),
    preplace_offset_m=0.12,
    intermediate_clearances_m=(0.09,),
    reorientation_steps=3,
    preplace_target_step_rad=0.004,
    preplace_tracking_tolerance_rad=0.060,
    preplace_endpoint_tolerance_rad=0.060,
    placement_target_step_rad=0.001,
    placement_tracking_tolerance_rad=0.050,
    placement_endpoint_tolerance_rad=0.060,
    placement_minimum_tracking_steps=2,
)
CAN_LEG = XGRIDS2TransferLegSpec(
    "can_transfer", "can", "dining_table_top", CAN_GRASP, CAN_PLACE, TRANSFER_ROUTE
)


SCENARIOS: dict[str, XGRIDS2ScenarioSpec] = {
    "can_dining_to_snack": XGRIDS2ScenarioSpec(
        "can_dining_to_snack",
        "XGRIDS2LabCanTransferTask",
        "XGRIDS2LabCanTransferKinovaMobileCurobo",
        "Move the can from the dining table to the snack table.",
        "reference",
        ("cylinder", "surface-placement", "mobile-manipulation"),
        (CAN,),
        (DINING_TABLE, SNACK_TABLE),
        (CAN_LEG,),
        COMMON_SHOTS,
        "generic_transfer",
        episode_length_s=600,
    ),
    "plate_dining_to_sink": XGRIDS2ScenarioSpec(
        "plate_dining_to_sink",
        "XGRIDS2LabPlateToSinkTask",
        "XGRIDS2LabPlateToSinkKinovaMobileCurobo",
        "Move the plate from the dining table into the sink.",
        "reference",
        ("thin-large-object", "rim-grasp", "container-placement", "mobile-manipulation"),
        (PLATE,),
        (DINING_TABLE, SINK),
        (
            XGRIDS2TransferLegSpec(
                "plate_transfer",
                "plate",
                "dining_table_top",
                XGRIDS2GraspSpec(
                    "rim",
                    (-0.66596903, 0.74597939),
                    (-0.6453621139, 0.2889770614, 0.6453621139, 0.2889770614),
                    0.11,
                    0.05,
                    0.24,
                    0.11,
                    "either",
                    carry_radius_m=0.47,
                ),
                XGRIDS2PlaceSpec(
                    "sink", "sink_basin", (-2.4369233721, -2.4369233721, 2.2386716370, 2.2386716370), 0.74657565
                ),
                PLATE_ROUTE,
            ),
        ),
        SINK_SHOTS,
        "plate_sink",
        limitations=(
            "contact_verified_temporary_fixed_joint_retention",
            "base_retained_transport_and_guided_sink_lowering",
        ),
    ),
    "can_and_cube_dining_to_snack": XGRIDS2ScenarioSpec(
        "can_and_cube_dining_to_snack",
        "XGRIDS2LabCanAndCubeTransferTask",
        "XGRIDS2LabCanAndCubeTransferKinovaMobileCurobo",
        "Move the can and then the cube from the dining table to separate positions on the snack table.",
        "complex",
        ("multi-object", "sequential", "cylinder", "cube", "return-navigation"),
        (CAN, CUBE),
        (DINING_TABLE, SNACK_TABLE),
        (
            XGRIDS2TransferLegSpec(
                **{**CAN_LEG.__dict__, "return_to_source_after": True}
            ),
            XGRIDS2TransferLegSpec(
                "cube_transfer",
                "cube",
                "dining_table_top",
                TOP_GRASP,
                XGRIDS2PlaceSpec(
                    "surface",
                    "snack_table_top",
                    (5.27, 5.33, -1.68, -1.60),
                    SNACK_TABLE.surface_z + 0.028982,
                    preferred_orientation_mode="side_release",
                    preferred_yaw_offset_degrees=-135.0,
                    preferred_center_offset_xy=(-0.025, -0.025),
                    release_clearance_m=0.10,
                    preplace_tracking_tolerance_rad=0.075,
                    preplace_endpoint_tolerance_rad=0.075,
                    placement_preflight_candidate_count=6,
                    release_from_approach=True,
                ),
                CUBE_TRANSFER_ROUTE,
                staging_poses=(TOP_GRASP_DEPARTURE_POSE,),
            ),
        ),
        COMMON_SHOTS,
        "sequential",
        episode_length_s=3600,
        limitations=(
            "contact_verified_temporary_fixed_joint_retention",
            "cube_10cm_physics_settled_release_with_xy_bias",
        ),
    ),
    "lime_clutter_to_bowl": XGRIDS2ScenarioSpec(
        "lime_clutter_to_bowl",
        "XGRIDS2LabLimeClutterToBowlTask",
        "XGRIDS2LabLimeClutterToBowlKinovaMobileCurobo",
        "Pick the lime from the dining-table clutter and place it in the bowl on the cluttered snack table.",
        "complex",
        ("clutter", "target-selection", "near-sphere", "container-placement"),
        (LIME, *SOURCE_CLUTTER, *DESTINATION_CLUTTER),
        (DINING_TABLE, SNACK_TABLE, BOWL_SNACK),
        (
            XGRIDS2TransferLegSpec(
                "lime_clutter_transfer",
                "lime",
                "dining_table_top",
                LIME_TOP_GRASP,
                XGRIDS2PlaceSpec(
                    "container",
                    "target_bowl",
                    (5.24, 5.24, -1.58, -1.58),
                    SNACK_TABLE.surface_z + 0.030,
                    xy_tolerance_m=0.045,
                    containment_margin_m=0.005,
                    preplace_clearance_m=0.24,
                    approach_clearance_m=0.18,
                    release_clearance_m=0.11,
                    preferred_orientation_mode="side_release",
                    preferred_yaw_offset_degrees=-135.0,
                    preferred_center_offset_xy=(0.0, 0.0),
                    preplace_target_step_rad=0.002,
                    preplace_tracking_tolerance_rad=0.040,
                    preplace_endpoint_tolerance_rad=0.050,
                ),
                TOP_GRASP_TRANSFER_ROUTE,
                staging_poses=(TOP_GRASP_DEPARTURE_POSE,),
                success_predicate="contained_receiver",
            ),
        ),
        COMMON_SHOTS,
        "generic_transfer",
        episode_length_s=1200,
    ),
    "mustard_pour_into_bowl": XGRIDS2ScenarioSpec(
        "mustard_pour_into_bowl",
        "XGRIDS2LabMustardPourTask",
        "XGRIDS2LabMustardPourKinovaMobileCurobo",
        "Pick up the mustard bottle, pour it over the bowl, return it upright, and place it down.",
        "baseline",
        ("pouring", "reorientation", "spout-alignment", "same-surface"),
        (MUSTARD,),
        (DINING_TABLE, BOWL_DINING),
        (
            XGRIDS2TransferLegSpec(
                "mustard_pour",
                "mustard_bottle",
                "dining_table_top",
                BOTTLE_GRASP,
                XGRIDS2PlaceSpec(
                    "pour", "pour_bowl", (1.12, 1.12, -1.02, -1.02), DINING_TABLE.surface_z + 0.080025
                ),
                None,
                success_predicate="pose_level_pour_no_fluid",
            ),
        ),
        (COMMON_SHOTS[0],),
        "pour",
        pour=XGRIDS2PourSpec(
            "mustard_bottle", "pour_bowl", (0.0, 0.0, 0.080025)
        ),
        episode_length_s=180,
        limitations=(
            "contact_verified_temporary_fixed_joint_retention",
            "pose_level_pour_no_fluid",
        ),
    ),
}


def list_xgrids2_scenarios() -> tuple[str, ...]:
    """Return stable benchmark scenario names in pack order."""

    return tuple(SCENARIOS)


def get_xgrids2_scenario(name: str) -> XGRIDS2ScenarioSpec:
    """Resolve one scenario or fail with the complete valid-name set."""

    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown XGRIDS2 scenario {name!r}; choose one of {list_xgrids2_scenarios()}"
        ) from error


def _xy_radius(item: XGRIDS2ObjectSpec) -> float:
    return 0.5 * max(item.dimensions_xyz[:2])


def _support_contains_xy(
    support: XGRIDS2SupportSpec,
    point_xy: Vector2,
    *,
    margin: float = 0.0,
) -> bool:
    dx = point_xy[0] - support.center_xyz[0]
    dy = point_xy[1] - support.center_xyz[1]
    if support.kind == "circle":
        return math.hypot(dx, dy) + margin <= 0.5 * min(support.size_xyz[:2])
    if support.kind in {"bowl", "sink"}:
        opening = support.opening_size_xy
        if opening is None:
            return False
        return (
            abs(dx) + margin <= 0.5 * opening[0]
            and abs(dy) + margin <= 0.5 * opening[1]
        )
    cosine, sine = math.cos(support.yaw_rad), math.sin(support.yaw_rad)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    return (
        abs(local_x) + margin <= 0.5 * support.size_xyz[0]
        and abs(local_y) + margin <= 0.5 * support.size_xyz[1]
    )


def _region_corners(region: Region2D) -> tuple[Vector2, ...]:
    xmin, xmax, ymin, ymax = region
    return ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))


def validate_xgrids2_scenario(
    scenario: XGRIDS2ScenarioSpec,
    *,
    package_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Return all static contract violations for one scenario."""

    errors: list[str] = []
    object_names = [item.name for item in scenario.objects]
    support_names = [item.name for item in scenario.supports]
    if len(object_names) != len(set(object_names)):
        errors.append("object names are not unique")
    if len(support_names) != len(set(support_names)):
        errors.append("support names are not unique")
    for item in scenario.objects:
        if any(value <= 0.0 for value in item.dimensions_xyz):
            errors.append(f"{item.name}: dimensions must be positive")
        if item.mass_kg <= 0.0:
            errors.append(f"{item.name}: mass must be positive")
        if item.manipulated == item.distractor:
            errors.append(
                f"{item.name}: exactly one of manipulated/distractor must be true"
            )
        xmin, xmax, ymin, ymax = item.spawn_region_xy
        if xmin > xmax or ymin > ymax:
            errors.append(f"{item.name}: invalid spawn region")
        norm = math.sqrt(sum(value * value for value in item.spawn_quaternion_wxyz))
        if not math.isclose(norm, 1.0, abs_tol=1.0e-4):
            errors.append(f"{item.name}: spawn quaternion is not normalized")
        if item.collision_approximation not in {
            "asset",
            "convex_hull",
            "bounding_box",
        }:
            errors.append(f"{item.name}: unsupported collision approximation")
        if package_root is not None and not (
            Path(package_root) / item.usd_relative
        ).is_file():
            errors.append(f"{item.name}: missing asset {item.usd_relative}")
    for support in scenario.supports:
        if not support.collision_enabled:
            errors.append(f"{support.name}: benchmark support collision is disabled")
        if any(value <= 0.0 for value in support.size_xyz):
            errors.append(f"{support.name}: dimensions must be positive")
        if support.usd_relative and package_root is not None and not (
            Path(package_root) / support.usd_relative
        ).is_file():
            errors.append(f"{support.name}: missing asset {support.usd_relative}")
        if support.kind in {"bowl", "sink"} and support.opening_size_xy is None:
            errors.append(f"{support.name}: receiver opening is required")
    for leg in scenario.legs:
        if leg.object_name not in object_names:
            errors.append(f"{leg.name}: unknown object {leg.object_name}")
        if leg.source_support_name not in support_names:
            errors.append(f"{leg.name}: unknown source {leg.source_support_name}")
        if leg.place.support_name not in support_names:
            errors.append(f"{leg.name}: unknown destination {leg.place.support_name}")
            continue
        xmin, xmax, ymin, ymax = leg.place.target_region_xy
        if xmin > xmax or ymin > ymax:
            errors.append(f"{leg.name}: invalid destination region")
        grasp_norm = math.sqrt(
            sum(value * value for value in leg.grasp.grasp_quaternion_wxyz)
        )
        if not math.isclose(grasp_norm, 1.0, abs_tol=1.0e-4):
            errors.append(f"{leg.name}: grasp quaternion is not normalized")
        direction_norm = math.hypot(*leg.grasp.approach_direction_xy)
        if not math.isclose(direction_norm, 1.0, abs_tol=1.0e-3):
            errors.append(f"{leg.name}: approach direction is not unit length")
        if leg.route is not None:
            if leg.route.planning_radius_m < leg.route.runtime_radius_m:
                errors.append(f"{leg.name}: planning radius is below runtime radius")
            if min(
                leg.route.max_linear_speed_m_s,
                leg.route.max_angular_speed_rad_s,
            ) <= 0.0:
                errors.append(f"{leg.name}: route speeds must be positive")
        if leg.object_name in object_names and leg.source_support_name in support_names:
            item = scenario.object(leg.object_name)
            source = scenario.support(leg.source_support_name)
            radius = _xy_radius(item)
            # Large thin objects may intentionally overhang a source surface
            # while remaining stably supported; cap the required source inset.
            source_margin = min(radius, 0.045)
            if not all(
                _support_contains_xy(source, corner, margin=source_margin)
                for corner in _region_corners(item.spawn_region_xy)
            ):
                errors.append(f"{leg.name}: object spawn region is outside its support")
            destination = scenario.support(leg.place.support_name)
            destination_margin = radius + (
                leg.place.containment_margin_m
                if leg.place.kind in {"container", "sink"}
                else 0.0
            )
            if not all(
                _support_contains_xy(destination, corner, margin=destination_margin)
                for corner in _region_corners(leg.place.target_region_xy)
            ):
                errors.append(f"{leg.name}: destination cannot contain the object")
            source_base = (
                leg.route.source_base_pose_xyzyaw[:2]
                if leg.route is not None
                else DINING_BASE_POSE[:2]
            )
            destination_base = (
                leg.route.destination_base_pose_xyzyaw[:2]
                if leg.route is not None
                else source_base
            )
            if not 0.25 <= math.dist(source_base, item.center_xy) <= 1.25:
                errors.append(f"{leg.name}: source region is outside arm reach envelope")
            if not 0.25 <= math.dist(destination_base, leg.place.center_xy) <= 1.25:
                errors.append(
                    f"{leg.name}: destination region is outside arm reach envelope"
                )

    # Every distractor must begin on one of the declared surface supports and
    # must not overlap another spawned object or a receiver fixture.
    surface_supports = tuple(
        support for support in scenario.supports if support.kind in {"rectangle", "circle"}
    )
    for item in scenario.objects:
        if item.distractor and not any(
            _support_contains_xy(support, item.center_xy, margin=_xy_radius(item))
            for support in surface_supports
        ):
            errors.append(f"{item.name}: distractor spawn is outside every support")
    for index, left in enumerate(scenario.objects):
        for right in scenario.objects[index + 1 :]:
            if math.dist(left.center_xy, right.center_xy) < _xy_radius(left) + _xy_radius(right):
                errors.append(f"{left.name}/{right.name}: spawn regions overlap")
    receivers = tuple(
        support for support in scenario.supports if support.kind in {"bowl", "sink"}
    )
    for item in scenario.objects:
        if not item.distractor:
            continue
        for receiver in receivers:
            receiver_radius = 0.5 * math.hypot(*receiver.size_xyz[:2])
            if math.dist(item.center_xy, receiver.center_xyz[:2]) < (
                _xy_radius(item) + receiver_radius
            ):
                errors.append(f"{item.name}/{receiver.name}: clutter overlaps receiver")
    if scenario.strategy == "pour" and scenario.pour is None:
        errors.append("pour strategy requires a pour contract")
    if scenario.pour is not None:
        if scenario.pour.object_name not in object_names:
            errors.append("pour object is unknown")
        if scenario.pour.receiver_name not in support_names:
            errors.append("pour receiver is unknown")
        if not 90.0 < scenario.pour.tilt_degrees < 180.0:
            errors.append("pour tilt must be between 90 and 180 degrees")
        if scenario.pour.hold_control_steps < 1:
            errors.append("pour hold must contain at least one control step")
    return tuple(errors)


def xgrids2_scenario_manifest_entry(scenario: XGRIDS2ScenarioSpec) -> dict:
    """Convert a scenario to stable JSON-safe benchmark metadata."""

    from robolab.tasks.mobile_manipulation.xgrids2_lab_geometry import (
        XGRIDS_VISUAL_USD_RELATIVE,
    )

    payload = asdict(scenario)
    physics_assets = [
        {
            "entity": item.name,
            "path": item.usd_relative,
            "collection": item.usd_relative.split("/")[2],
        }
        for item in scenario.objects
    ]
    physics_assets.extend(
        {
            "entity": support.name,
            "path": support.usd_relative,
            "collection": support.usd_relative.split("/")[2],
        }
        for support in scenario.supports
        if support.usd_relative is not None
    )
    payload["asset_provenance"] = {
        "vendor_appearance": {
            "path": XGRIDS_VISUAL_USD_RELATIVE,
            "usage": "collision_free_appearance_only",
        },
        "spawned_physics_assets": physics_assets,
    }
    payload["retention_mode"] = (
        "measured_fingertip_contact_then_temporary_fixed_joint"
    )
    payload["pouring_fidelity"] = (
        "pose_level_pour_no_fluid" if scenario.pour is not None else None
    )
    return payload
