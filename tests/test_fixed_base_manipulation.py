"""CPU-only schema tests for the fixed-base manipulation boundary."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from robolab.tasks.mobile_manipulation.curobo_fixed_base_planner import (
    world_pose_to_base,
)
from robolab.tasks.mobile_manipulation.genmanip_kitchen_geometry import (
    BASE_CLEARANCE_INFLATION,
    STATIC_COLLISION_BOXES,
)
from robolab.tasks.mobile_manipulation.fixed_base_manipulation import (
    ARM_JOINT_NAMES,
    FIXED_BASE_ISLAND_STAGE,
    FIXED_BASE_LOCAL_PLACE_REGION,
    FIXED_BASE_PICK_REGION,
    Pose,
    fixed_base_trial_schedule,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "assets/robots/kinova_mobile_cobot/curobo"


def test_40_trial_schedule_is_center_to_edge_and_in_bounds():
    trials = fixed_base_trial_schedule(40)
    assert len(trials) == 40
    assert trials[0].scale == 0.0
    assert all(left.scale <= right.scale for left, right in zip(trials, trials[1:]))
    assert trials[-1].scale == pytest.approx(0.95)
    assert all(FIXED_BASE_PICK_REGION.contains(trial.pick_xy) for trial in trials)
    assert all(
        trial.place_xy == FIXED_BASE_LOCAL_PLACE_REGION.center for trial in trials
    )


def test_fixed_base_stage_preserves_inflated_island_clearance():
    cabinet = next(
        box for box in STATIC_COLLISION_BOXES if box.name == "island_lower_cabinet"
    )
    assert cabinet.footprint.distance_to(FIXED_BASE_ISLAND_STAGE[:2]) >= (
        BASE_CLEARANCE_INFLATION
    )


def test_world_pose_to_fixed_base_handles_pi_yaw():
    bx, by, bz, _ = FIXED_BASE_ISLAND_STAGE
    converted = world_pose_to_base(
        Pose((bx + 0.50, by - 0.20, bz + 0.70), (1.0, 0.0, 0.0, 0.0)),
        FIXED_BASE_ISLAND_STAGE,
    )
    assert converted.position == pytest.approx((-0.50, 0.20, 0.70), abs=1.0e-6)
    assert converted.quaternion_wxyz == pytest.approx(
        (0.0, 0.0, 0.0, -1.0), abs=1.0e-6
    )


def test_planner_urdf_and_config_share_the_seven_arm_joint_contract():
    urdf = ET.parse(CONFIG_DIR / "kinova_gen3_robolab.urdf").getroot()
    joint_names = {joint.attrib["name"] for joint in urdf.findall("joint")}
    config = yaml.safe_load(
        (CONFIG_DIR / "kinova_gen3_robolab.yml").read_text(encoding="utf-8")
    )["robot_cfg"]["kinematics"]
    assert tuple(config["cspace"]["joint_names"]) == ARM_JOINT_NAMES
    assert set(ARM_JOINT_NAMES).issubset(joint_names)
    assert config["ee_link"] == "tool_frame"
    assert config["extra_links"]["attached_object"]["parent_link_name"] == "tool_frame"


def test_tool_frame_uses_measured_robolab_tcp_offset():
    urdf = ET.parse(CONFIG_DIR / "kinova_gen3_robolab.urdf").getroot()
    joint = next(
        joint
        for joint in urdf.findall("joint")
        if joint.attrib["name"] == "tool_frame_joint"
    )
    xyz = tuple(float(value) for value in joint.find("origin").attrib["xyz"].split())
    assert xyz == pytest.approx((0.0, 0.0, 0.1429))
    assert math.isfinite(sum(xyz))
