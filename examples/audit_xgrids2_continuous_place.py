#!/usr/bin/env python3
"""Compare placement near-stops from physical control traces, not video edits."""
import argparse
import json
from pathlib import Path


def near_stops(rows, key, start, stop, dt, threshold=.005, minimum_duration=.25):
    intervals = []
    beginning = None
    for index in range(start, stop+1):
        low = index < stop and max(map(abs, rows[index][key])) < threshold
        if low and beginning is None:
            beginning = index
        if not low and beginning is not None:
            if (index-beginning)*dt >= minimum_duration:
                intervals.append({'start_s': beginning*dt, 'end_s': index*dt,
                                  'duration_s': (index-beginning)*dt})
            beginning = None
    return intervals


def measure(directory):
    directory = Path(directory)
    result = json.loads((directory/'result.json').read_text())
    rows = [json.loads(line) for line in (directory/'control_trace.jsonl').open()]
    dt = result['recording_diagnostics']['control_dt_s']
    continuous = any(r['phase']=='place_continuous' for r in rows)
    phases = {'place_continuous'} if continuous else {'place_approach','place_lower','place'}
    indices = [i for i,r in enumerate(rows) if r['phase'] in phases]
    if not indices:
        raise ValueError('Trial did not reach placement')
    first, last = indices[0], indices[-1]
    # Find the final moving target sample to exclude the deliberate terminal
    # ease-out and safety dwell. Start/end ramps are not intermediate stops.
    moving = [i for i in indices if max(map(abs,rows[i]['target_velocity_rad_s'])) >= .005]
    if not moving:
        raise ValueError('No moving placement command samples')
    begin = first+round(.5/dt)
    end = moving[-1]+1
    return {'source':str(directory), 'continuous_single_plan':continuous,
            'placement_start_video_s':first*dt, 'release_video_s':(last+1)*dt,
            'placement_phase_duration_s':len(indices)*dt,
            'inspection_interval_s':[begin*dt,end*dt],
            'near_stop_definition':'all seven absolute joint speeds below 0.005 rad/s for >=0.25 seconds; initial 0.5s and final ease-out/dwell excluded',
            'intermediate_command_near_stops':near_stops(rows,'target_velocity_rad_s',begin,end,dt),
            'intermediate_measured_near_stops':near_stops(rows,'velocity_rad_s',begin,end,dt),
            'task_passed':result['passed'], 'route_clearance_valid':result['route_clearance_valid'],
            'retention_active_end':result['retention_active_end'],
            'profile':result.get('execution_profile_config'),
            'release_clearances_m':result['passive_evaluation']['metrics']['release_clearances_above_destination_m'],
            'final_placement':result['passive_evaluation']['metrics']['final_placement']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args()
    current=measure(args.directory)
    report={'current':current}
    if args.reference:
        report['reference']=measure(args.reference)
    report['passed']=(current['continuous_single_plan'] and current['task_passed']
                      and not current['intermediate_command_near_stops']
                      and not current['intermediate_measured_near_stops']
                      and current['route_clearance_valid'] and not current['retention_active_end'])
    (args.directory/'continuous_place_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
