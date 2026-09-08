# XGRIDS2 / LabApt planar alignment

## Result

The active LabApt navigation frame is related to the floor-corrected XGRIDS2
wrapper frame by the planar rigid transform

```text
p_xgrids = R(35.617718 deg) p_labapt + [-13.259476, 6.588388] m
yaw_xgrids = wrap(yaw_labapt + 35.617718 deg)
```

This transform is for the Z-up wrapper whose scan is translated by +1.075 m,
placing the dominant floor at Z=0. The Z correction does not affect the planar
registration.

## Inputs

- XGRIDS2 triangle mesh:
  `assets/scenes/xgrids2_lab/forward_compatible/mesh-files/UT_Robtics_lab_int.obj`
- LabApt vector map copied read-only from the cobot computer:
  `assets/scenes/xgrids2_lab/labapt_reference/LabApt.vectormap.txt`
- LabApt occupancy grid and map metadata:
  `assets/scenes/xgrids2_lab/labapt_reference/labapt_gridmap.npz`
- Active semantic navigation poses originally read from
  `/home/ros/cobot_demo/src/codebotler_amrl_impl/data.yaml`.

The semantic poses are robot navigation/staging goals `[x, y, yaw]`; they are
not object centers. They were therefore reserved for validation and were not
used as exact point correspondences during registration.

## Method

`examples/register_labapt_to_xgrids2.py` performs the registration:

1. It parses metric vertices and triangle faces from the scan OBJ.
2. It intersects the mesh with horizontal planes at Z = 0.25, 0.45, 0.75,
   1.10, and 1.55 m in the corrected wrapper frame. Multiple heights retain
   walls and partitions while reducing dependence on any one furniture height.
3. It samples both the intersection segments and LabApt vector-map segments at
   the native map resolution of 0.08 m.
4. For each coarse rotation hypothesis, it rasterizes both geometries and uses
   FFT cross-correlation to estimate translation.
5. It refines the best hypotheses continuously in rotation and translation by
   minimizing robust nearest-surface distance.
6. It applies the winning transform to all active semantic staging poses and
   writes the transform, aligned geometry, and an overlay.

No camera image, Gaussian appearance data, or hand-selected semantic point was
used to fit the transform.

## Fit quality

- Rotation: 35.617718 degrees
- Translation: (-13.259476, 6.588388) m
- Fraction of sampled LabApt geometry within 0.24 m of a scan slice: 77.24%
- Median nearest-surface distance: 0.0875 m
- 75th-percentile nearest-surface distance: 0.2128 m

The unmatched regions are expected because the navigation map and reconstructed
mesh contain different extents, movable furniture, scan noise, and geometry
observed at different heights.

## Transformed semantic staging poses

| Goal | X (m) | Y (m) | Yaw (rad) |
|---|---:|---:|---:|
| kitchen_sink | -1.721 | 1.471 | 2.322 |
| microwave | -1.217 | 2.691 | 2.202 |
| kitchen_shelf | 0.986 | 3.496 | 2.282 |
| living_room | 6.125 | -7.437 | 0.662 |
| snack_table | 4.752 | -0.867 | -0.948 |
| study_desk | 2.575 | 1.399 | 0.772 |
| dining_table | 1.766 | -1.764 | 2.192 |

## Validation

`examples/audit_xgrids2_lab_views.py --view-set labapt_aligned` reads the saved
transform and renders from each transformed staging pose, looking along its
transformed robot heading. All seven views passed RGB/depth validation with
99.96% to 100% valid depth.

Visual checks agree with the labels:

- Kitchen poses land along the red kitchen fixture/shelving wall.
- Living-room staging lands at the curtain/living zone.
- Snack-table staging lands by the round-table/red-chair area.
- Dining-table staging looks along the long dining table toward the kitchen.

The snack tabletop is low in the horizontal 1.625 m audit-camera view. That is
a view-pitch/framing limitation; it does not indicate a planar alignment error.

## Generated artifacts

- `output/xgrids2_lab/labapt_registration/labapt_to_xgrids2_transform.json`
- `output/xgrids2_lab/labapt_registration/labapt_xgrids_overlay.png`
- `output/xgrids2_lab/labapt_registration/registration_geometry.npz`
- `output/xgrids2_lab/labapt_aligned_views/00_numbered_views_montage.png`
- `output/xgrids2_lab/labapt_aligned_views/numbered_views_manifest.json`

## Remaining limitation

This establishes a metric coordinate bridge for visualization, semantic pose
transfer, and later collision authoring. It does not make the reconstructed
appearance mesh a physics-ready environment. Static collision proxies, a clean
navigable floor, clearance validation, and robot spawn/traversal tests are still
required before using these poses in a RoboLab task.
