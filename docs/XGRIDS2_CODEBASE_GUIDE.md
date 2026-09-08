# XGrids2 code map

The configurable scenario runner is the single maintained entry point:

```bash
python examples/validate_xgrids2_lab_scenario.py \
  --scenario can_dining_to_snack --stage full --headless
```

Use `--record-video --video-fps 30 --capture-every 1` for synchronized
control-rate evidence. The runner owns environment creation, oracle execution,
passive evaluation, recording, and the benchmark manifest.

## Maintained modules

| Responsibility | File |
| --- | --- |
| Typed task and scene records | `xgrids2_lab_scenarios.py` |
| Generated Isaac task | `xgrids2_lab_scenario_task.py` |
| Scenario execution and artifact lifecycle | `xgrids2_runner.py` |
| Scenario strategy selection | `xgrids2_lab_scenario_oracle.py` |
| Validated can controller | `xgrids2_lab_curobo_oracle.py` |
| Validated plate controller | `xgrids2_lab_plate_to_sink_curobo_oracle.py` |
| Passive task evaluator | `xgrids2_passive_evaluation.py` |
| Simulator snapshot adapter | `xgrids2_evaluation_observer.py` |
| Evidence image/video helpers | `xgrids2_evidence.py` |
| Dashboard export | `xgrids2_evaluation_export.py` |

Scenario definitions describe the world and the current reference execution.
The passive evaluator remains independent of oracle phase labels and does not
advance the controller.

## Public scripts

- `validate_xgrids2_lab_scenario.py`: run any registered scenario.
- `validate_xgrids2_lab_manipulation.py`: compatibility launcher for the can.
- `validate_xgrids2_lab_plate_to_sink.py`: compatibility launcher for the plate.
- `audit_xgrids2_evaluation.py`: replay and verify passive-evaluation output.
- `audit_xgrids2_recording.py`: fully decode and verify synchronized video.
- `build_xgrids2_diagnostic_viewer.py`: build the offline HTML report.
- `export_xgrids2_can_video_variants.py`: make the two shareable can videos.

The remaining `diagnose_`, `visualize_`, `register_`, and specialized
`validate_` programs are maintenance tools for asset alignment, rendering,
physics gates, reachability, and robustness. They are not alternative benchmark
runners.

## Cleanup rules

- Add scenarios to the typed registry; do not copy the runner.
- Add reusable logic under `robolab/tasks/mobile_manipulation/`; keep example
  scripts limited to argument parsing and orchestration.
- Record one source of truth for thresholds and geometry in the scenario or
  controller configuration.
- Comments should explain physical assumptions, safety gates, or non-obvious
  simulator constraints. Avoid comments that merely restate the next line.
- Preserve evaluator-only privileged data boundaries and explicitly report
  unsupported metrics as unavailable.
- Keep generated evidence under `output/`; do not commit temporary `.orig`
  files or one-off diagnostic copies.

Two obsolete pre-integration rendering prototypes were removed after the
proxy-free vendor scene and current recording audits passed. The maintained
NuRec stability diagnostic remains available as the rendering regression gate.
