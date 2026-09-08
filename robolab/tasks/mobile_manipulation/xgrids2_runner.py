"""Execute one XGrids2 scenario and write its evaluation evidence.

Import this module only after ``AppLauncher`` has started Isaac Sim.
"""

import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
import numpy as np
import torch

from robolab.core.environments.runtime import create_env
from robolab.core.utils.nurec_rendering import configure_xgrids_vendor_render_products
from robolab.core.utils.video_utils import VideoWriter
from robolab.registrations.kinova_mobile import register_xgrids2_scenario
from robolab.tasks.mobile_manipulation.xgrids2_evidence import (
    camera_rgb,
    choose_camera_shot,
    inspect_video,
    make_contact_sheet,
    render_camera_panel,
    set_camera_shot,
    update_benchmark_manifest,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenario_oracle import (
    XGRIDS2ScenarioOracle,
)
from robolab.tasks.mobile_manipulation.xgrids2_lab_scenarios import (
    get_xgrids2_scenario,
    list_xgrids2_scenarios,
    xgrids2_scenario_manifest_entry,
)


def run_xgrids2_scenario(
    args_cli: Any, *, repo_root: Path, nurec_settings: dict[str, Any]
) -> None:
    """Run one configured scenario after the caller has initialized Isaac."""
    if args_cli.video_fps < 1 or args_cli.capture_every < 1:
        raise ValueError("--video-fps and --capture-every must be positive")
    if args_cli.num_seeds < 1:
        raise ValueError("--num-seeds must be positive")
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    scenario = get_xgrids2_scenario(args_cli.scenario)
    env_cfg_class, env_name = register_xgrids2_scenario(scenario.name)
    env_cfg = env_cfg_class()
    env_cfg.seed = args_cli.seed
    env_cfg.recorders = None
    env_cfg.episode_length_s = scenario.episode_length_s
    env_cfg.sim.use_fabric = True
    # Camera data must update once per captured control tick. Render interval
    # is expressed in physics steps, while capture_every is in control steps.
    env_cfg.sim.render_interval = (
        env_cfg.decimation * args_cli.capture_every
        if args_cli.record_video
        else 8
    )
    env_cfg.sim.render.antialiasing_mode = "OFF"
    camera_height = 270 if args_cli.record_video else 72
    camera_width = 480 if args_cli.record_video else 128
    env_cfg.scene.navigation_camera.height = camera_height
    env_cfg.scene.navigation_camera.width = camera_width
    env_cfg.scene.wrist_cam.height = camera_height
    env_cfg.scene.wrist_cam.width = camera_width
    env_cfg.scene.audit_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/scenario_audit_camera",
        height=540 if args_cli.record_video else 144,
        width=960 if args_cli.record_video else 256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, 30.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(convention="opengl"),
    )
    env, _ = create_env(env_cfg)
    pack_root = (repo_root / "output/xgrids2_lab/benchmark_pack").resolve()
    output_dir = (
        Path(args_cli.output_dir).resolve()
        if args_cli.output_dir
        else pack_root / scenario.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args_cli.output_dir:
        # Diagnostic trials must not replace the accepted pack manifest entry.
        pack_root = output_dir.parent
    result_path = output_dir / "result.json"
    video_path = output_dir / "task.mp4"
    montage_path = output_dir / "montage.png"
    contact_sheet_path = output_dir / "contact_sheet.jpg"
    control_trace_path = output_dir / "control_trace.jsonl"
    video_writer = None
    trace_file = None
    evaluation_file = None
    evaluator = None
    evaluation_row = None
    frame_map = []
    try:
        env.reset_eval_state()
        env.reset()
        products = configure_xgrids_vendor_render_products(
            env.scene["navigation_camera"],
            env.scene["wrist_cam"],
            env.scene["audit_camera"],
        )
        oracle = XGRIDS2ScenarioOracle(
            env, scenario, num_seeds=args_cli.num_seeds,
            execution_profile=("can_diagnostic" if scenario.name == "can_dining_to_snack" else "legacy")
                              if args_cli.execution_profile == "auto" else args_cli.execution_profile,
        )
        # Evaluation attaches to env.step, independently of oracle phases.
        # Other scenarios retain their accepted behavior until migrated.
        evaluation_observer = nullcontext()
        if scenario.name in {"can_and_cube_dining_to_snack", "can_dining_to_snack"}:
            from robolab.tasks.mobile_manipulation.xgrids2_passive_evaluation import PassiveTransferEvaluator
            from robolab.tasks.mobile_manipulation.xgrids2_evaluation_observer import (
                IsaacTransferSampler,
                PostStepObserver,
            )

            real_time_video = abs(
                args_cli.video_fps * float(env.step_dt) - 1.0
            ) <= 1e-6
            if args_cli.record_video and (
                args_cli.capture_every != 1 or not real_time_video
            ):
                raise ValueError(
                    "Passive evidence requires every control tick at 30 FPS"
                )
            evaluator = PassiveTransferEvaluator(scenario, dt=float(env.step_dt))
            sampler = IsaacTransferSampler(env, scenario)
            evaluation_file = (output_dir / "evaluation_trace.jsonl").open("w", buffering=1)

            def evaluate_step(action, step_result):
                nonlocal evaluation_row
                evaluation_row = evaluator.observe(sampler.sample(action))
                evaluation_file.write(json.dumps(evaluation_row, separators=(",", ":"), allow_nan=False)+"\n")

            evaluation_observer = PostStepObserver(env, evaluate_step)
        audit_camera = env.scene["audit_camera"]
        current_shot = scenario.camera_shots[0]
        set_camera_shot(audit_camera, device=oracle.device, shot=current_shot)
        env.sim.render()
        initial_frame = camera_rgb(audit_camera)
        sampled_third_person = [initial_frame.copy()]
        observed_control_ticks = 0
        written_video_frames = 0
        captured_phases: list[str] = []
        last_captured_phase = None
        last_camera_cut_tick = -100
        if args_cli.record_video:
            video_writer = VideoWriter(str(video_path), fps=args_cli.video_fps)
        if args_cli.record_control_trace:
            trace_file = control_trace_path.open("w", buffering=1)
        if args_cli.record_video or args_cli.record_control_trace:

            def observe_step(facade, backend):
                nonlocal observed_control_ticks, written_video_frames
                nonlocal current_shot, last_captured_phase
                nonlocal last_camera_cut_tick
                observed_control_ticks += 1
                if trace_file is not None:
                    trace_sample = dict(backend._latest_motion_sample)
                    trace_sample["global_tick"] = observed_control_ticks
                    trace_sample["object_name"] = backend.object_name
                    trace_file.write(json.dumps(trace_sample, separators=(",", ":")) + "\n")
                if not args_cli.record_video:
                    return
                root = facade.robot.data.root_link_pos_w[0] - env.scene.env_origins[0]
                xy = (float(root[0]), float(root[1]))
                next_shot = choose_camera_shot(scenario, xy)
                if next_shot.name != current_shot.name:
                    last_camera_cut_tick = observed_control_ticks
                    current_shot = next_shot
                    set_camera_shot(
                        audit_camera, device=facade.device, shot=current_shot
                    )
                phase_changed = backend._stage_label != last_captured_phase
                if observed_control_ticks % args_cli.capture_every and not phase_changed:
                    return
                last_captured_phase = backend._stage_label
                if not captured_phases or captured_phases[-1] != backend._stage_label:
                    captured_phases.append(backend._stage_label)
                subtitle = (
                    f"{backend._stage_label.upper()} | robot x={xy[0]:.2f} y={xy[1]:.2f}"
                )
                if scenario.pour is not None:
                    subtitle = f"{subtitle} | pose_level_pour_no_fluid"
                third_raw = camera_rgb(audit_camera)
                # Stream every frame to the encoder, retaining only a bounded
                # phase sample for the contact sheet.
                if len(sampled_third_person) < 12 or phase_changed:
                    if len(sampled_third_person) < 64:
                        sampled_third_person.append(third_raw.copy())
                third = render_camera_panel(
                    third_raw,
                    f"THIRD-PERSON | {scenario.name}",
                    subtitle,
                    size=(960, 540),
                )
                if evaluation_row is not None and scenario.name != "can_dining_to_snack":
                    cv2.rectangle(third, (0, 454), (960, 540), (15, 19, 25), -1)
                    held = [
                        name
                        for name, obj in evaluation_row["observation"]["objects"].items()
                        if obj["retention_active"]
                    ]
                    lines = [
                        f"SIM {evaluation_row['sim_time_s']:.1f}s | "
                        f"PROGRESS {100 * evaluation_row['progress']:.0f}% | "
                        f"BASE PATH {evaluation_row['base_path_length_m']:.2f}m",
                        " | ".join(f"{name} {100*value:.0f}%" for name, value in evaluation_row["groups"].items()),
                        f"RETENTION ASSIST: {','.join(held) or 'OFF'} | "
                        "collisions: uninstrumented | perception: privileged oracle",
                    ]
                    for line_index, text in enumerate(lines):
                        cv2.putText(
                            third,
                            text,
                            (14, 477 + 24 * line_index),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.47,
                            (245, 245, 245),
                            1,
                        )
                robot_views = np.concatenate(
                    (
                        render_camera_panel(
                            camera_rgb(env.scene["navigation_camera"]),
                            "NAVIGATION CAMERA",
                            "forward-facing fixed base mount",
                        ),
                        render_camera_panel(
                            camera_rgb(env.scene["wrist_cam"]),
                            "WRIST CAMERA",
                            "physical manipulation view",
                        ),
                    ),
                    axis=0,
                )
                frame = np.concatenate((third, robot_views), axis=1)
                if evaluation_row is not None and scenario.name == "can_dining_to_snack":
                    from robolab.tasks.mobile_manipulation.xgrids2_can_diagnostic import can_diagnostic_lines
                    footer = np.full((190, 1440, 3), (15, 19, 25), dtype=np.uint8)
                    for index, line in enumerate(can_diagnostic_lines(evaluation_row,
                            camera_cut=observed_control_ticks-last_camera_cut_tick < 30)):
                        cv2.putText(footer, line, (14, 25+30*index), cv2.FONT_HERSHEY_SIMPLEX,
                                    .55, (245, 245, 245) if index != 4 else (255, 210, 100), 1, cv2.LINE_AA)
                    frame = np.concatenate((frame, footer), axis=0)
                video_writer.write(frame)
                frame_map.append({"frame": written_video_frames, "global_tick": observed_control_ticks,
                                  "evaluation_tick": evaluation_row["tick"] if evaluation_row else None,
                                  "video_time_s": written_video_frames/args_cli.video_fps,
                                  "sim_time_s": observed_control_ticks*float(env.step_dt)})
                written_video_frames += 1

            oracle.set_step_observer(observe_step)

        try:
            with evaluation_observer:
                result = oracle.run(stage=args_cli.stage)
            result_payload = result.as_dict()
        except Exception as error:
            result_payload = {
                "scenario": scenario.name,
                "passed": False,
                "requested_stage": args_cli.stage,
                "completed_stage": "",
                "failure_reason": f"unhandled_oracle_error: {error}",
                "phase_history": [],
                "leg_results": [],
                "maximum_arm_step_rad": 0.0,
                "maximum_arm_speed_rad_s": 0.0,
                "minimum_route_clearance_valid": False,
                "released_objects": [],
                "retention_active_end": False,
            }
        retention_active_end = bool(
            oracle.backend is not None
            and oracle.backend.grasp_joint.GetJointEnabledAttr().Get()
        )
        result_payload["retention_active_end"] = retention_active_end
        if "released_objects" not in result_payload:
            result_payload["released_objects"] = (
                [item.name for item in scenario.objects if item.manipulated]
                if result_payload.get("passed")
                and args_cli.stage in {"place", "full"}
                and not retention_active_end
                else []
            )
        maximum_step = float(result_payload.get("maximum_arm_step_rad", 0.0))
        maximum_speed = float(
            result_payload.get("maximum_arm_speed_rad_s", 0.0)
        )
        extended_motion = result_payload.get("motion_diagnostics", {})
        result_payload["motion_quality_violation"] = (
            maximum_step > 0.05
            or maximum_speed > 3.5
            or bool(extended_motion.get("jerk_violation", False))
            or bool(extended_motion.get("oscillation_violation", False))
        )
        result_payload["route_clearance_valid"] = bool(
            result_payload.get("minimum_route_clearance_valid", True)
        )
        result_payload["video_observed_phases"] = captured_phases
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        if trace_file is not None:
            trace_file.close()
            trace_file = None
        evaluation_summary = None
        dashboard_source = None
        if evaluation_file is not None:
            evaluation_file.close()
            evaluation_file = None
            from robolab.tasks.mobile_manipulation.xgrids2_evaluation_export import export_dashboard_episode
            evaluation_summary = evaluator.summary()
            result_payload["task_correctness_passed"] = evaluation_summary["task_success"]
            if (
                args_cli.stage in {"full", "place"}
                and result_payload.get("passed")
                and not evaluation_summary["task_success"]
            ):
                result_payload["passed"] = False
                result_payload["failure_reason"] = "passive final placement criterion unsatisfied"
            evaluation_summary["physics_audit"] = {
                "motion_quality_violation": result_payload["motion_quality_violation"],
                "route_clearance_valid": result_payload["route_clearance_valid"],
                "motion_diagnostics": extended_motion,
                "interpretation": "controller/physics diagnostics, separate from task correctness",
            }
            (output_dir / "evaluation.json").write_text(json.dumps(evaluation_summary, indent=2, allow_nan=False)+"\n")
            (output_dir / "evaluation_events.json").write_text(json.dumps(evaluator.events, indent=2)+"\n")
            (output_dir / "video_frame_map.json").write_text(json.dumps(frame_map, indent=2)+"\n")
            dashboard_source = export_dashboard_episode(
                output_dir,
                scenario,
                evaluation_summary,
                evaluator.events,
                dt=float(env.step_dt),
                video_path=video_path if args_cli.record_video else None,
                seed=args_cli.seed,
            )
        env.sim.render()
        final_frame = camera_rgb(audit_camera)
        sampled_third_person.append(final_frame.copy())
        montage = np.concatenate((initial_frame, final_frame), axis=1)
        if not cv2.imwrite(
            str(montage_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
        ):
            raise RuntimeError(f"could not write {montage_path}")
        sheet = make_contact_sheet(sampled_third_person)
        if not cv2.imwrite(
            str(contact_sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
        ):
            raise RuntimeError(f"could not write {contact_sheet_path}")
        video_diagnostics = (
            inspect_video(video_path) if args_cli.record_video else None
        )
        payload = {
            **result_payload,
            "environment": env_name,
            "execution_profile": oracle.execution_profile,
            "execution_profile_config": (
                asdict(oracle.backend.diagnostic_profile)
                if oracle.backend is not None
                and getattr(oracle.backend, "diagnostic_profile", None) is not None
                else None
            ),
            "seed": args_cli.seed,
            "scenario_contract": xgrids2_scenario_manifest_entry(scenario),
            "nurec_render_settings": nurec_settings,
            "nurec_render_products": products,
            "video_diagnostics": video_diagnostics,
            "passive_evaluation": evaluation_summary,
            "dashboard_source": dashboard_source,
            "recording_diagnostics": {
                "control_dt_s": float(env.step_dt),
                "metrics_panel_layout": (
                    "appended_footer"
                    if scenario.name == "can_dining_to_snack"
                    else "third_person_overlay"
                ),
                "capture_every_control_ticks": args_cli.capture_every,
                "video_fps": args_cli.video_fps if args_cli.record_video else None,
                "observed_control_ticks": observed_control_ticks,
                "written_video_frames": written_video_frames,
                "playback_speed_ratio": (
                    args_cli.video_fps * float(env.step_dt) * args_cli.capture_every
                    if args_cli.record_video
                    else None
                ),
                "control_trace_enabled": args_cli.record_control_trace,
            },
            "artifacts": {
                "result": str(result_path),
                "video": str(video_path) if args_cli.record_video else None,
                "montage": str(montage_path),
                "contact_sheet": str(contact_sheet_path),
                "control_trace": (
                    str(control_trace_path) if args_cli.record_control_trace else None
                ),
            },
        }
        result_path.write_text(json.dumps(payload, indent=2) + "\n")
        manifest_path = update_benchmark_manifest(
            pack_root,
            scenario.name,
            {
                "scenario": xgrids2_scenario_manifest_entry(scenario),
                "result": payload,
            },
            list_xgrids2_scenarios(),
        )
        print(json.dumps(payload, indent=2), flush=True)
        print(f"Saved result: {result_path}")
        print(f"Saved montage: {montage_path}")
        print(f"Saved contact sheet: {contact_sheet_path}")
        if args_cli.record_video:
            print(f"Saved video: {video_path}")
        if args_cli.record_control_trace:
            print(f"Saved control trace: {control_trace_path}")
        print(f"Updated pack manifest: {manifest_path}")
        if not result_payload.get("passed", False):
            raise RuntimeError(
                result_payload.get("failure_reason") or "scenario validation failed"
            )
        if video_diagnostics is not None and (
            not video_diagnostics["readable"]
            or video_diagnostics["frame_count"] < 2
            or video_diagnostics["maximum_adjacent_changed_fraction"] <= 0.001
        ):
            raise RuntimeError(
                f"video evidence failed readability/change gate: {video_diagnostics}"
            )
    finally:
        if video_writer is not None:
            video_writer.release()
        if trace_file is not None:
            trace_file.close()
        if evaluation_file is not None:
            evaluation_file.close()
        env.close()
