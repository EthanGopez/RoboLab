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


def test_offline_export_and_placement_visibility(tmp_path):
    evidence=ROOT/'output/xgrids2_lab/can_diagnostic_20260905/video'
    if not (evidence/'result.json').exists():
        pytest.skip('Local can evidence unavailable')
    data=builder.build_payload(evidence)
    c=data['columns']
    assert len(c['time'])==data['frames']==3982
    assert c['context'][0]=='arm'
    assert c['context'][1350]=='navigation'
    assert c['context'][3900]=='settled'
    for i,transported in enumerate(c['transported']):
        assert (c['placement_error'][i] is None)==(not transported)
        assert (c['destination_clearance'][i] is None)==(not transported)
    output=builder.build(evidence,tmp_path/'viewer.html')
    text=output.read_text()
    assert '<script src=' not in text and 'fetch(' not in text
    assert '__DATA__' not in text and '__SCRIPT__' not in text
    assert 'application/json' in text
    assert 'includes future events' in text
    assert 'final results, not live values' in text
    embedded=text.split('<script id="viewer-data" type="application/json">')[1].split('</script>')[0]
    assert json.loads(embedded)['columns']==data['columns']
