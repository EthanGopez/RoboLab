"""Video annotations must not reintroduce irrelevant or future task metrics."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('video_variants', ROOT/'examples/export_xgrids2_can_video_variants.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample(context='arm', **changes):
    values = dict(context=context, held=False, transported=False, placed=False,
        max_joint_error=.01, object_speed=.02, base_measured=.1, yaw_measured=.2,
        source_clearance=1., left_contact=1.2, right_contact=1.3,
        destination_clearance=None, placement_error=None)
    values.update(changes)
    return {key:[value] for key,value in values.items()}


def test_navigation_and_prearrival_do_not_display_placement():
    for context in ('arm','navigation','grasp','placement'):
        title,facts = module.current_annotation(sample(context),0)
        assert len(facts) <= 3
        assert not any('Placement XY' in label or 'snack table' in label for label,_ in facts)


def test_placement_geometry_only_after_transport_and_error_after_release():
    c=sample('placement',held=True,transported=True,destination_clearance=.8,placement_error=.6)
    _,facts=module.current_annotation(c,0)
    assert any('Clearance above snack' in label for label,_ in facts)
    assert not any('Placement XY' in label for label,_ in facts)
    c['held']=[False];c['context']=['settling']
    title,facts=module.current_annotation(c,0)
    assert 'settled' not in title
    assert any('Placement XY' in label for label,_ in facts)
    c['placed']=[True];c['context']=['settled']
    assert module.current_annotation(c,0)[0]=='Can placed and settled'
    assert 'settled' not in module.current_annotation(sample(),0)[0]


def test_cropping_preserves_camera_pixels_and_annotations_fit():
    image=np.random.default_rng(0).integers(0,256,(730,1440,3),dtype=np.uint8)
    clean=module.clean_frame(image)
    assert clean.shape==(540,1440,3)
    for y0,y1,x0,x1 in ((53,540,0,960),(53,270,960,1440),(323,540,960,1440)):
        np.testing.assert_array_equal(clean[y0:y1,x0:x1],image[y0:y1,x0:x1])
    for context in ('arm','navigation','grasp','placement','settling','settled'):
        c=sample(context,transported=True,destination_clearance=.8,placement_error=.6)
        title,facts=module.current_annotation(c,0)
        frame=module.annotated_frame(clean,title,facts)
        assert frame.shape==(680,1440,3)
        np.testing.assert_array_equal(frame[:540],clean)
        for label,_ in facts:
            assert module.cv2.getTextSize(label,module.cv2.FONT_HERSHEY_SIMPLEX,.51,1)[0][0]<450
