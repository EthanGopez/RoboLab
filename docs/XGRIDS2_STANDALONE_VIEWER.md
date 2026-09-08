# Standalone can diagnostic viewer

Open `output/xgrids2_lab/can_diagnostic_20260905/video/viewer/index.html`
in a browser. It works directly from a local file: no server, network libraries,
Isaac session, or RoboLab integration is required. Keep the HTML in its evidence
folder so its relative `../task.mp4` link resolves.

The original MP4 and traces are unchanged. The old burned-in metrics footer is
hidden by cropping the video element in CSS; all three camera panels remain.
The new view is a synchronized replay of recorded physical samples, not a live
connection to the simulator.

## What to try

- Around **10 seconds**: finger contact and source clearance during grasping.
- Around **45 seconds**: requested, controller-applied, and measured base speed
  and yaw rate. The requested-speed ramps and smoother physical response are
  visibly different; neither should be inferred from a single jerk number.
- Around **124–130 seconds**: receiving-table clearance, can speed, retention
  release, and observed settling. Placement error appears after release, not
  during pickup.
- Expand **Event navigator** and click a motion advisory. This seeks to the
  recorded event and opens the jerk plot. Step backward/forward by one frame.
- Choose a joint or pin a plot category for manual inspection. This changes
  only the display, not the measured context or evaluation result.

Only two primary plots are shown at once, with 5/10/20-second history windows.
They draw past samples only, with a playhead tied to decoded video-frame time.
The sample at video time zero is the first post-step sample at simulation time
1/30 second. Seeking, pausing, frame stepping, and changing playback speed use
the same frame mapping. Fixed scales are calculated from the recorded episode,
not rescaled on each moving window. Pale green shading indicates observed
retention activation; vertical markers indicate outcomes and warnings.

Automatic display context uses observed base motion, retention, contact, and
task predicates, not the oracle's scripted phase names. Display selection has
no effect on correctness or execution. Clearance above the destination is null
until the object has been observed over that surface. Base arrival alone does
not expose placement measurements.

Episode totals and limitations are under the collapsed **Episode report**.
The event navigator intentionally includes future events for offline inspection
and is labeled accordingly. Opening either is an explicit look at episode-level
information; the live status and plots do not reveal future achievements.

Jerk thresholds are read from the saved evaluation metadata. They remain
development advisories, not validated perceptual limits or task-failure gates.
Grasp robustness under assisted retention, privileged perception, uninstrumented
collision counts, and unavailable reference efficiencies are not scored here.

## Build and test

```bash
/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/build_xgrids2_diagnostic_viewer.py \
  output/xgrids2_lab/can_diagnostic_20260905/video

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  -m pytest --noconftest tests/test_xgrids2_diagnostic_viewer.py -q

/home/robocup/miniconda3/envs/env_isaaclab/bin/python \
  examples/audit_xgrids2_diagnostic_viewer.py \
  output/xgrids2_lab/can_diagnostic_20260905/video/viewer/index.html
```

The builder validates consecutive trace ticks and the video frame map, then
embeds selected full-rate signals into a roughly 3 MB HTML file. It intentionally
supports only this single-can evidence format for now. Presentation source is
in `examples/xgrids2_diagnostic_viewer/`; no dashboard source was modified.

Seven CPU tests passed. An isolated headless Firefox audit verified local-file
video playback; pickup/navigation/settling contexts; hidden inapplicable values;
backward seeking without future outcomes; manual plot selection; warning-event
seeking; single-frame stepping; half-speed playback; and a responsive layout.
`viewer/browser_audit.json` records the results. `viewer/preview.png` and
`viewer/placement_preview.png` show actual browser screenshots.

The browser audit uses Selenium plus the locally installed Firefox/geckodriver;
override `--firefox` and `--geckodriver` on a different installation. These are
testing dependencies only, not dependencies of the viewer itself.

## Shareable MP4 variants

`video/video_versions/` also contains two ordinary H.264 MP4s:

- `can_pick_place_clean.mp4`: camera names only, with the original numeric
  headers and metrics footer removed.
- `can_pick_place_current_metrics.mp4`: the same views with an activity label
  and at most three present-time measurements. No progress totals or cumulative
  warning counts. Placement error is hidden until transport and release.

Both use all 3,982 original frames, at 30 FPS and 1x playback (132.73 seconds).
No simulation was rerun. All frames were decoded for verification, and rendered
camera regions were compared to the source every 30 frames to check alignment
while allowing lossy re-encoding. The source SHA-256 remained unchanged.
`video_variants.json` records the checks and annotation intervals.

Reproduce with `examples/export_xgrids2_can_video_variants.py <evidence-dir>`;
use `--output <new-directory>` to preserve existing exports. The three focused
tests in `tests/test_xgrids2_video_variants.py` cover measurement applicability,
backward-context correctness, camera-pixel preservation before encoding, and
annotation fit. These exports are intentionally uncluttered videos; the HTML
viewer remains the option for synchronized interactive plots.
