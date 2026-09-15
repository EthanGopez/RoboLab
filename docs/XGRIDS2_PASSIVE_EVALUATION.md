# XGrids2 passive evaluation

The evaluator is an observational layer around continuous simulator execution.
It reads a post-control-step snapshot after every `env.step`; it never advances,
gates, or prescribes the policy's phase ordering.

## Current scope

`can_dining_to_snack` is the only benchmark-eligible task. It uses unassisted
physical gripper contact. Fixed-joint retention is forbidden, and observing it
at any tick invalidates task success. The additional scenario records are not
scored until their manipulation implementations are migrated.

## Predicate structure

```text
sustained fingertip contact -> physical grasp -> lift -> transport -> placement
                                      ^              ^             ^
                                      |              |             |
                            navigation handoff   drop check   release/settling
```

The graph expresses dependencies for evaluation, not a required action plan.
A policy may perceive, navigate, and move its arm concurrently.

The grasp predicate requires qualifying contact for 0.25 seconds. Lift requires
the object's conservative geometry to clear its source support by 5 cm without
retention. Transport requires a preceding physical lift and arrival over the
destination. Placement requires containment, target tolerances, released
contact, low linear/angular velocity, and one second of settling.

Grasp, lift, and transport are historical achievements. Placement is maintained:
moving an object away revokes placement and task success until it is restored
and settles again.

## Progress and diagnosis

Four explicit macro groups—Grasp, Lift, Transport, and Place—receive equal
weight. Diagnostic predicates do not change the denominator. The report includes
the following handoffs:

- navigation to physical grasp;
- physical grasp to transport;
- transport to settled placement.

A handoff is reported only when its upstream event was observed. Failure output
identifies the first unsatisfied capability; it is structured failure
localization, not causal inference about the physical root cause.

## Quality metrics

The evaluator records completion time, base path length, base yaw travel, arm
tracking RMSE, integrated squared jerk, maximum jerk, source-goal reentries,
predicate recoveries, release clearance, and final placement geometry. Controller
motion-quality and route-clearance audits remain separately visible.

Unavailable values remain `null`: perception for the privileged oracle,
whole-body collision count, and reference-normalized path efficiency. The
controller's object-to-tool translation diagnostic is a separation proxy, not
a complete measurement of contact-surface slip.

## Evidence and replay

The runner writes `evaluation_trace.jsonl`, `evaluation_events.json`,
`evaluation.json`, `control_trace.jsonl`, synchronized camera video, and a
frame-to-control-tick map. Recording evidence requires one frame per 30 Hz
control tick and 30 FPS encoding, producing 1x simulated-time playback.

Use:

```bash
python examples/audit_xgrids2_evaluation.py <run-directory>
python examples/audit_xgrids2_recording.py <run-directory>
python examples/build_xgrids2_diagnostic_viewer.py <run-directory> <output.html>
```

The dashboard and offline viewer display only achievements observed by the
selected playback time. Final episode results are labeled separately.
