"""Render setup shared by scenes containing NVIDIA NuRec volumes.

NuRec's neural image is composited with ordinary RTX geometry. These settings
establish the supported single-GPU, pre-Hydra-sync baseline. They make the
standalone regression scene deterministic, but do not by themselves guarantee
stable multi-camera compositing in a heavily loaded Isaac Sim 5.1 task.
"""

from __future__ import annotations


def require_single_gpu(app_launcher_args) -> None:
    """Apply NuRec's launch-time single-GPU requirement."""

    app_launcher_args.multi_gpu = False


def configure_xgrids_vendor_nurec_volume() -> dict[str, object]:
    """Configure Nucleus4D's August 27 package for Isaac Lab cameras.

    The vendor loader provides the Fabric, single-GPU, registered-compositing,
    and synchronous-rendering baseline. Replicator camera tensors additionally
    require both inversion flags enabled; otherwise they are nearly black even
    while the standalone viewport is valid.
    """

    import carb

    settings = carb.settings.get_settings()
    requested = {
        "/app/useFabricSceneDelegate": True,
        "/renderer/multiGpu/enabled": False,
        "/rtx/post/registeredCompositing/enabled": True,
        "/rtx/post/registeredCompositing/invertToneMap": True,
        "/rtx/post/registeredCompositing/invertColorCorrection": True,
        "/app/asyncRendering": False,
        "/app/asyncRenderingLowLatency": False,
    }
    for path, value in requested.items():
        settings.set(path, value)
    return {path: settings.get(path) for path in requested}


def configure_xgrids_vendor_render_products(
    *cameras,
) -> dict[str, dict[str, bool]]:
    """Author the required NuRec compositing flags on Replicator products."""

    from pxr import Sdf

    authored = {}
    attributes = {
        "omni:rtx:post:registeredCompositing:enabled": True,
        "omni:rtx:post:registeredCompositing:invertColorCorrection": True,
        "omni:rtx:post:registeredCompositing:invertToneMap": True,
    }
    for camera in cameras:
        for path in camera.render_product_paths:
            prim = camera.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(
                    f"NuRec camera RenderProduct does not exist: {path}"
                )
            for name, value in attributes.items():
                prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(value)
            authored[path] = {
                name: bool(prim.GetAttribute(name).Get()) for name in attributes
            }
    return authored


def configure_plain_nurec_volume() -> dict[str, object]:
    """Configure the standard (non-SPG/PPISP) NuRec volume render path.

    Returns the values read back from Carb so diagnostics can record the exact
    runtime configuration. This function must run after Isaac Sim starts but
    before the stage containing the NuRec volume is created.
    """

    import carb

    settings = carb.settings.get_settings()
    requested = {
        "/renderer/multiGpu/enabled": False,
        "/renderer/multiGpu/autoEnable": False,
        "/rtx/rtpt/gaussian/skipTonemapping/enabled": False,
        "/rtx/post/histogram/enabled": False,
        "/rtx/post/registeredCompositing/invertColorCorrection": True,
        "/rtx/post/registeredCompositing/invertToneMap": True,
        "/rtx/post/tonemap/op": 2,
        # DLSS's temporal history is undesirable for a neural volume and was
        # also running below its minimum internal resolution in our cameras.
        "/rtx/post/aa/op": 0,
    }
    for path, value in requested.items():
        settings.set(path, value)
    return {path: settings.get(path) for path in requested}


def configure_nurec_render_products(*cameras) -> dict[str, dict[str, bool]]:
    """Author NuRec compositing controls on each camera RenderProduct.

    Carb settings are only defaults. Replicator creates one RenderProduct for
    every TiledCamera, and each product can carry its own RTX opinions. Making
    those opinions explicit removes configuration ambiguity; it is a baseline
    check, not a workaround for renderer overload or runtime integration bugs.
    """

    from pxr import Sdf

    authored = {}
    attributes = {
        "omni:rtx:post:registeredCompositing:enabled": True,
        "omni:rtx:post:registeredCompositing:invertColorCorrection": True,
        "omni:rtx:post:registeredCompositing:invertToneMap": True,
    }
    for camera in cameras:
        for path in camera.render_product_paths:
            prim = camera.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"NuRec camera RenderProduct does not exist: {path}")
            for name, value in attributes.items():
                prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(value)
            authored[path] = {
                name: bool(prim.GetAttribute(name).Get()) for name in attributes
            }
    return authored
