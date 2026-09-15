"""Pure display-context and evidence export regression checks."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('viewer_builder',ROOT/'examples/build_xgrids2_diagnostic_viewer.py')
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.mark.parametrize('updates,expected',[
    ({},'arm'),({'contact':True},'grasp'),({'held':True,'moving':True},'navigation'),
    ({'held':True,'destination':True},'placement'),
    ({'transported':True},'settling'),({'placed':True},'settled'),
])
def test_observational_context(updates,expected):
    state=dict(held=False,lifted=False,transported=False,placed=False,destination=False,contact=False,moving=False)
    state.update(updates)
    assert builder.context_for(**state)==expected


def test_legacy_assisted_evidence_is_rejected(tmp_path):
    evidence=ROOT/'output/xgrids2_lab/can_diagnostic_20260905/video'
    if not (evidence/'result.json').exists():
        pytest.skip('Local can evidence unavailable')
    with pytest.raises(ValueError, match="forbidden fixed-joint retention"):
        builder.build_payload(evidence)
