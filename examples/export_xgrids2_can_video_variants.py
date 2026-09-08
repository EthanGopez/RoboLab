#!/usr/bin/env python3
"""Export clean and instant-relevant MP4s from the same recorded physical trial."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from examples.build_xgrids2_diagnostic_viewer import build_payload
from robolab.core.utils.video_utils import VideoWriter


def current_annotation(columns, i):
    """Only present-time facts; no totals, past warning counts, or future outcomes."""
    c = columns
    mode, held = c['context'][i], c['held'][i]
    retention = ('Retention assist', 'ON' if held else 'OFF')
    error = ('Arm tracking error (largest joint)', f"{math.degrees(c['max_joint_error'][i]):.2f} deg")
    speed = ('Can speed', f"{c['object_speed'][i]:.3f} m/s")
    if mode == 'navigation':
        return 'Transporting the can' if held else 'Moving the base', [
            ('Measured base speed', f"{c['base_measured'][i]:.2f} m/s"),
            ('Measured turning rate', f"{math.degrees(c['yaw_measured'][i]):.1f} deg/s"), retention]
    if mode == 'grasp':
        if held:
            return 'Lifting the can with retention assistance', [
                ('Clearance above dining table', f"{c['source_clearance'][i]:.1f} cm"), speed, retention]
        return 'Establishing finger contact', [
            ('Left finger contact', f"{c['left_contact'][i]:.1f} N"),
            ('Right finger contact', f"{c['right_contact'][i]:.1f} N"), retention]
    if mode in {'placement', 'settling', 'settled'}:
        if not c['transported'][i]:
            return 'Positioning the can above the snack table', [error, speed, retention]
        if held:
            return 'Positioning the can for release', [
                ('Clearance above snack table', f"{c['destination_clearance'][i]:.1f} cm"), speed, retention]
        return ('Can placed and settled' if c['placed'][i] else 'Can released - checking settling'), [
            speed, ('Placement XY error', f"{c['placement_error'][i]:.1f} cm"), retention]
    return ('Moving the arm with the can held' if held else 'Positioning the arm for pickup'), (
        [error, speed, retention] if held else [error])


def clean_frame(source):
    """Keep every original camera pixel below its existing label band."""
    clean = source[:540, :1440].copy()
    for x, y, width, label in ((0, 0, 960, 'THIRD-PERSON CAMERA'),
                                (960, 0, 480, 'NAVIGATION CAMERA'),
                                (960, 270, 480, 'WRIST CAMERA')):
        # The source header was painted through row 52 inclusive.
        cv2.rectangle(clean, (x, y), (x+width-1, y+52), (25, 19, 15), -1)
        cv2.putText(clean, label, (x+14, y+32), cv2.FONT_HERSHEY_SIMPLEX,
                    .60, (230, 235, 240), 1, cv2.LINE_AA)
    return clean


def annotated_frame(clean, title, facts):
    footer = np.full((140, 1440, 3), (28, 23, 18), dtype=np.uint8)
    cv2.putText(footer, title, (22, 33), cv2.FONT_HERSHEY_SIMPLEX,
                .76, (193, 223, 98), 2, cv2.LINE_AA)
    for j, (label, value) in enumerate(facts):
        x = 22+j*474
        cv2.putText(footer, label, (x, 73), cv2.FONT_HERSHEY_SIMPLEX,
                    .51, (185, 177, 166), 1, cv2.LINE_AA)
        cv2.putText(footer, value, (x, 114), cv2.FONT_HERSHEY_SIMPLEX,
                    .88, (240, 239, 232), 2, cv2.LINE_AA)
    return np.concatenate((clean, footer), axis=0)


def audit_videos(source, outputs, *, expected_frames, fps):
    captures = [cv2.VideoCapture(str(path)) for path in [source, *outputs]]
    counts = [0]*len(captures)
    max_camera_mae = [0.] * len(outputs)
    max_motion_change = [0.] * len(outputs)
    previous = [None] * len(outputs)
    try:
        for cap in captures:
            assert cap.isOpened()
            assert abs(cap.get(cv2.CAP_PROP_FPS)-fps) < 1e-6
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_frames
        while True:
            decoded = [cap.read() for cap in captures]
            for n, (ok, _) in enumerate(decoded):
                counts[n] += int(ok)
            if not all(ok for ok, _ in decoded):
                assert not any(ok for ok, _ in decoded), 'Unequal decoded video lengths'
                break
            index = counts[0]-1
            if index % 30:
                continue
            original = decoded[0][1]
            for n, (_, frame) in enumerate(decoded[1:]):
                regions = [(53, 540, 0, 960), (53, 270, 960, 1440), (323, 540, 960, 1440)]
                for y0, y1, x0, x1 in regions:
                    mae = float(cv2.absdiff(original[y0:y1, x0:x1], frame[y0:y1, x0:x1]).mean())
                    max_camera_mae[n] = max(max_camera_mae[n], mae)
                small = cv2.resize(frame[:540], (360, 135))
                if previous[n] is not None:
                    max_motion_change[n] = max(max_motion_change[n], float(cv2.absdiff(small, previous[n]).mean()))
                previous[n] = small
        assert counts == [expected_frames]*len(captures)
        assert all(v < 12 for v in max_camera_mae), 'Camera content differs beyond re-encoding tolerance'
        assert all(v > 1 for v in max_motion_change), 'Unexpected frozen recording'
        return {'passed': True, 'decoded_frames': counts, 'fps': fps,
                'duration_s': expected_frames/fps, 'maximum_sampled_camera_pixel_error': max_camera_mae,
                'maximum_sampled_motion_change': max_motion_change,
                'pixel_error_interpretation': 'Same-index comparisons of rendered camera areas, every 30th frame; lossy re-encoding allowed. All frames decoded.'}
    finally:
        for cap in captures:
            cap.release()


def export(directory, output=None):
    directory = Path(directory).resolve()
    data = build_payload(directory)
    if data['video_width'] != 1440 or data['visible_video_height'] != 540:
        raise ValueError('Expected the validated three-panel can recording')
    source = directory/'task.mp4'
    output = Path(output).resolve() if output else directory/'video_versions'
    output.mkdir(parents=True, exist_ok=True)
    names = ['can_pick_place_clean.mp4', 'can_pick_place_current_metrics.mp4']
    paths = [output/name for name in names]
    if any(path.exists() for path in paths):
        raise FileExistsError('Choose a new output directory; existing exports are preserved')
    with source.open('rb') as stream:
        source_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    fps = round(1/data['dt'])
    writers = [VideoWriter(str(path), fps) for path in paths]
    capture = cv2.VideoCapture(str(source))
    count = 0
    intervals = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if count >= data['frames']:
                raise ValueError('Video longer than synchronized trace')
            title, facts = current_annotation(data['columns'], count)
            clean = clean_frame(frame)
            annotated = annotated_frame(clean, title, facts)
            for writer, image in zip(writers, (clean, annotated)):
                writer.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not intervals or intervals[-1]['title'] != title:
                intervals.append({'first_frame': count, 'time_s': count/fps, 'title': title,
                                  'fields': [label for label, _ in facts]})
            if count in (300, 1350, 3735, 3900):
                cv2.imwrite(str(output/f'current_metrics_frame_{count}.jpg'), annotated)
            count += 1
            if count % 900 == 0:
                print(f'Encoded {count}/{data["frames"]} frames in both variants', flush=True)
    finally:
        capture.release()
        for writer in writers:
            writer.release()
    assert count == data['frames']
    audit = audit_videos(source, paths, expected_frames=count, fps=fps)
    with source.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == source_digest
    result = {'source_video': str(source), 'source_sha256': source_digest,
              'same_physical_trial': True, 'simulation_rerun': False, 'playback_speed_ratio': 1,
              'clean_video': str(paths[0]), 'current_metrics_video': str(paths[1]),
              'clean_overlay': 'Camera names only; no coordinates, task metrics, or diagnostic footer',
              'metrics_overlay': 'At most three current-context facts; no totals or cumulative warning counts',
              'context_intervals': intervals, 'audit': audit}
    (output/'video_variants.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.directory, args.output), indent=2))
