#!/usr/bin/env python3
"""Compare can-only phase traces; does not conflate quality with task success."""
import argparse
import json
from pathlib import Path

import numpy as np


def measure(directory):
    directory = Path(directory)
    result = json.loads((directory/'result.json').read_text())
    dt = result['recording_diagnostics']['control_dt_s']
    rows = []
    with (directory/'control_trace.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['object_name'] != 'can' or row['phase'].startswith('return_'):
                break
            rows.append(row)
    phases = {}
    for row in rows:
        phases[row['phase']] = phases.get(row['phase'], 0.) + dt
    def peak(key):
        index = max(range(3, len(rows)), key=lambda i: max(map(abs, rows[i][key])))
        row = rows[index]
        return {'value': max(map(abs, row[key])), 'phase': row['phase'],
                'sim_time_s': row['global_tick']*dt}
    base = []
    for line in (directory/'evaluation_trace.jsonl').open():
        row = json.loads(line)
        if row['tick'] > rows[-1]['global_tick']:
            break
        base.append(row['observation']['base'])
    yaw_speed = np.array([b['angular_velocity_xyz'][2] for b in base])
    yaw_accel = np.diff(yaw_speed)/dt
    yaw_jerk = np.diff(yaw_accel)/dt
    return {'source': str(directory), 'can_phase_duration_s': len(rows)*dt,
            'phase_duration_s': phases,
            'peak_measured_arm_jerk_rad_s3': peak('jerk_rad_s3'),
            'peak_target_arm_jerk_rad_s3': peak('target_jerk_rad_s3'),
            'peak_measured_base_yaw_acceleration_rad_s2': float(np.max(np.abs(yaw_accel))),
            'peak_measured_base_yaw_jerk_rad_s3': float(np.max(np.abs(yaw_jerk))),
            'base_path_length_m': sum(np.linalg.norm(np.array(b['position_xyz'][:2])-a['position_xyz'][:2]) for a,b in zip(base,base[1:])),
            'phase_sample_count': len(rows)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    report = {'current': measure(args.directory)}
    if args.reference:
        report['reference'] = measure(args.reference)
        report['duration_ratio_reference_over_current'] = report['reference']['can_phase_duration_s']/report['current']['can_phase_duration_s']
        report['comparison_limitation'] = 'Earlier sequential-task can leg versus new standalone can task; descriptive comparison, not a controlled A/B or optimality reference.'
    (args.directory/'diagnostic_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
