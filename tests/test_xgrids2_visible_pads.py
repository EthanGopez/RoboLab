# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run with Isaac USD libraries available; tests verify composed asset state."""
from pathlib import Path

import numpy as np
import pytest

Usd=pytest.importorskip('pxr.Usd')
from pxr import UsdGeom, UsdPhysics, UsdShade


@pytest.fixture(scope='module')
def stage():
    return Usd.Stage.Open(str(Path(__file__).resolve().parents[1]/
        'assets/robots/kinova_mobile_cobot/kinova_mobile_cobot.usda'))


@pytest.mark.parametrize('side,sign',[('left',-1),('right',1)])
def test_visible_contact_insert_and_separate_materials(stage,side,sign):
    path=f'/KinovaMobileCobot/robotiq_85_{side}_finger_tip_link/flat_contact_pad_collision'
    prim=stage.GetPrimAtPath(path)
    assert prim.IsA(UsdGeom.Cube)
    assert UsdGeom.Imageable(prim).ComputeVisibility() == 'inherited'
    assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    render,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    physics,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
    assert str(render.GetPath()) == '/KinovaMobileCobot/gripper_pad_visual'
    assert render.ComputeSurfaceSource()[0]
    assert str(physics.GetPath()) == '/KinovaMobileCobot/gripper_pad_physics'
    np.testing.assert_allclose(prim.GetAttribute('xformOp:translate').Get(),[sign*.008,0,.030])
    np.testing.assert_allclose(prim.GetAttribute('xformOp:scale').Get(),[.003175,.011,.01875])
    assert prim.GetAttribute('physxCollision:contactOffset').Get() == pytest.approx(.001)
    assert prim.GetAttribute('physxCollision:restOffset').Get() == 0


def test_no_duplicate_collider_and_unchanged_friction(stage):
    for side in ('left','right'):
        body=stage.GetPrimAtPath(f'/KinovaMobileCobot/robotiq_85_{side}_finger_tip_link')
        active=[p for p in Usd.PrimRange(body,Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        assert len(active)==1
        assert active[0].GetName()=='flat_contact_pad_collision'
    material=stage.GetPrimAtPath('/KinovaMobileCobot/gripper_pad_physics')
    assert material.GetAttribute('physics:staticFriction').Get() == pytest.approx(1.5)
    assert material.GetAttribute('physics:dynamicFriction').Get() == pytest.approx(1.2)


@pytest.mark.parametrize('side',['left','right'])
def test_one_visible_fingertip_matching_the_only_collider(stage,side):
    body=stage.GetPrimAtPath(f'/KinovaMobileCobot/robotiq_85_{side}_finger_tip_link')
    visible=[p for p in Usd.PrimRange(body,Usd.TraverseInstanceProxies())
             if p.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(p).ComputeVisibility()!='invisible'
             and UsdGeom.Imageable(p).ComputePurpose() in ('default','render')]
    assert [p.GetName() for p in visible]==['flat_contact_pad_collision']
    assert UsdPhysics.CollisionAPI(visible[0]).GetCollisionEnabledAttr().Get()
