"""Pure checks for placement profile selection and measured-stop auditing."""
from dataclasses import replace

from examples.audit_xgrids2_continuous_place import near_stops
from robolab.tasks.mobile_manipulation.xgrids2_can_diagnostic import CanDiagnosticProfile


def test_continuous_profile_is_explicit_and_keeps_release_guard():
    profile=CanDiagnosticProfile()
    assert profile.continuous_place
    assert profile.name=='can_diagnostic_v2_continuous_place'
    assert profile.release_max_clearance_m==.015
    assert not replace(profile,continuous_place=False).continuous_place


def test_near_stop_detector_does_not_confuse_terminal_hold_with_interior_stop():
    rows=[{'speed':[.1]*7} for _ in range(100)]
    for i in range(20,30):rows[i]['speed']=[0.]*7
    for i in range(80,100):rows[i]['speed']=[0.]*7
    found=near_stops(rows,'speed',0,80,1/30)
    assert len(found)==1
    assert found[0]['start_s']==20/30
    assert found[0]['duration_s']==10/30


def test_single_stationary_joint_is_not_a_whole_arm_stop():
    rows=[{'speed':[0.,.1,0.,0.,0.,0.,0.]} for _ in range(100)]
    assert near_stops(rows,'speed',0,100,1/30)==[]
