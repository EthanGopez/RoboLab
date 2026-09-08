# SPDX-License-Identifier: Apache-2.0

"""Structural checks for the GenManip kitchen import and RoboLab renderer."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "assets" / "third_party" / "ebench"
SCENE = (
    ASSET_ROOT
    / "assets"
    / "scene_usds"
    / "ebench"
    / "long_horizon"
    / "scene_dishwasher"
    / "scene.usd"
)
RUNNER = ROOT / "examples" / "render_genmanip_kitchen.py"
ORACLE_VISUALIZER = ROOT / "examples" / "visualize_genmanip_kitchen_oracle_trial.py"
PREDICATES = ROOT / "robolab" / "tasks" / "mobile_manipulation" / "kitchen_manipulation_predicates.py"
ORACLE = ROOT / "robolab" / "tasks" / "mobile_manipulation" / "kitchen_manipulation_oracle.py"
GATE_RUNNER = ROOT / "examples" / "validate_genmanip_kitchen_oracle_gate.py"


def test_complete_kitchen_dependency_set_is_present():
    assert SCENE.is_file()
    dependency_dir = SCENE.parent / "SubUSDs"
    for name in (
        "layout_flattened.usd",
        "model_d887aa45d2bf3c36b5797da16bf28a0d_0.usd",
        "_8dd440442874c8e8ae35494b4aff0a2.usd",
        "_964b7f2_2f9d_4034_9932_c208191a30e6.usd",
    ):
        assert (dependency_dir / name).is_file()
    shared_mdl = ASSET_ROOT / "assets" / "miscs" / "mdl" / "ebench" / "mdl"
    assert (shared_mdl / "KooPbr.mdl").is_file()
    assert (shared_mdl / "KooPbr_maps.mdl").is_file()
    assert (ASSET_ROOT / "PROVENANCE.md").is_file()


def test_renderer_preserves_authored_scene_and_checks_rgb_depth():
    text = RUNNER.read_text()
    assert 'KITCHEN_USD = os.path.join(KITCHEN_DIR, "scene.usd")' in text
    assert "UsdFileCfg(usd_path=KITCHEN_USD)" in text
    assert 'os.environ["MDL_SYSTEM_PATH"]' in text
    assert "_SHARED_MDL_DIR" in text
    assert "_SCENE_MDL_DIR" in text
    assert 'prim_path="{ENV_REGEX_NS}/back_wall"' in text
    assert 'prim_path="{ENV_REGEX_NS}/left_wall"' in text
    assert 'prim_path="{ENV_REGEX_NS}/right_wall"' in text
    assert text.count("sim_utils.CuboidCfg(") == 3
    assert "PreviewSurfaceCfg" in text
    assert '"--can-pose"' in text
    assert 'assets", "objects", "ycb", "tomato_soup_can.usd"' in text
    assert '"pick":' in text and '"place":' in text
    assert "kinematic_enabled=True" in text
    assert "cv2.imwrite" in text
    assert '"--show-cobot"' in text
    assert "KinovaMobileCobotCfg" in text
    assert '"pick": (0.50, 0.175, 0.468)' in text
    assert "robot.init_state.pos = BASE_POSES.get" in text
    assert "robot.spawn.scale" not in text
    assert "set_world_poses_from_view" in text
    assert 'camera.data.output["rgb"]' in text
    assert 'camera.data.output["distance_to_image_plane"]' in text
    assert "HomeOfficeBackgroundCfg" not in text


def test_oracle_visualizer_uses_registered_task_and_live_oracle():
    text = ORACLE_VISUALIZER.read_text()
    assert "register_genmanip_kitchen_mobile_ik_env()" in text
    assert "IK_ENV_NAME" in text
    assert "use_fabric=True" in text
    assert "KitchenManipulationOracle(" in text
    assert "use_grasp_assist=True" in text
    assert 'env.step(action)' in text
    assert 'camera.data.output["rgb"]' in text
    assert "video.write(final_frame)" in text
    assert "_save_montage(montage_path, phase_frames)" in text
    assert 'env_cfg.scene.wrist_cam = None' in text
    assert 'env_cfg.scene.navigation_camera = None' in text


def test_grasp_task_semantics_are_separate_from_the_kinova_adapter():
    predicate_text = PREDICATES.read_text()
    oracle_text = ORACLE.read_text()
    gate_text = GATE_RUNNER.read_text()
    assert "robolab.robots" not in predicate_text
    assert "flange_body_name: str" in predicate_text
    assert "fingertip_pad_offsets" in predicate_text
    assert "Whether a particular hand is mechanically" in predicate_text
    assert "MOBILE_KINOVA_FINGERTIP_PAD_OFFSETS" in oracle_text
    assert "use_grasp_assist: bool = True" in oracle_text
    assert "parser.set_defaults(grasp_assist=True)" in gate_text
    assert "strict_post_contact_constraint" in gate_text
    assert "unassisted_friction" in gate_text
