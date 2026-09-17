"""Host-side Universal Scene Description (USD) authoring for vehicle assets: bake the First Person
View (FPV) camera + migrate rotor actuator params.

``pxr``-only, plus ``numpy`` for the rotation compose, so this module imports and runs anywhere
with ``usd-core``; it's host-side and unit-tested, with no Kit dependency. The photoreal
*world* converters, photogrammetry mesh + Gaussian-splat, are Kit-only and live in
``scripts/assets/convert_world.py``.
"""

from __future__ import annotations

# --- FPV camera mount math: the single source of truth for the authored FpvCam -----------
# This module bakes the FPV camera into the vehicle USD as a body-child once; the renderer,
# runtimes/isaacsim/fpv_renderer.py, just creates a render product on that authored prim, with no
# in-code camera and no per-frame posing. The mount/intrinsics below mirror the old FpvConfig
# defaults; change them here + re-author the vehicle USD to move the camera. The constants live
# here to keep this assets module decoupled from the Isaac runtime package.
#
# Forward Right Down (FRD) body → USD camera basis; USD cameras look down -Z, up +Y, right +X:
#   forward, +X body → cam -Z ; up, -Z body → cam +Y ; right, +Y body → cam +X.
_R_BODY_CAM = [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
_FOCAL_LENGTH = 12.0  # FpvConfig.focal_length
_H_APERTURE = 36.0  # FpvConfig.h_aperture; ~112° Horizontal Field Of View (HFOV) at the spike resolution
_MOUNT_OFFSET = (0.15, 0.0, 0.0)  # FpvConfig.mount_offset: 15 cm forward at the nose, body FRD
_DOWN_TILT_DEG = 10.0  # FpvConfig.down_tilt_deg: slight down-tilt about the camera's right axis
# Default render resolution, FpvConfig.width/height; the in-code camera bakes verticalAperture =
# h_aperture * height/width so the vertical Field Of View (FOV) matches the feed aspect, see
# fpv_renderer.py:216. This module bakes the same value here so the authored camera and the in-code
# fallback match bit for bit.
_DEFAULT_WIDTH = 1280.0  # FpvConfig.width
_DEFAULT_HEIGHT = 720.0  # FpvConfig.height


def _body_cam_rotation(down_tilt_deg: float):
    """The camera's body-relative rotation matrix R_body_cam, in the column-vector convention
    ``v_body = R_body_cam @ v_cam``, the same as fpv_renderer's
    ``_R_BODY_CAM @ Rot.from_euler("x", down_tilt_deg).as_matrix()``.

    _R_BODY_CAM is body_from_cam; right-composing a rotation about +X applies the tilt in the
    camera frame about its own right axis = pitch the view down.
    """
    import numpy as np

    t = np.radians(down_tilt_deg)
    c, s = np.cos(t), np.sin(t)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return np.asarray(_R_BODY_CAM) @ rx


def _find_base_body(stage):
    """Locate the vehicle's base body prim: the root rigid-body Xform physics treats as the body.

    Heuristic, robust + documented: the first prim, in stage order under the default prim, that
    carries the rigid-body physics API, ``UsdPhysics.RigidBodyAPI``; if none carries it, for
    example a stripped/synthetic USD, fall back to the default prim itself, else the first
    ``UsdGeom.Xform`` on the stage.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    default = stage.GetDefaultPrim()
    root = default if (default and default.IsValid()) else stage.GetPseudoRoot()
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    if default and default.IsValid():
        return default
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.IsA(UsdGeom.Xform):
            return prim
    raise ValueError("could not locate a base body prim in the vehicle USD")


def author_vehicle_camera(usdz, out, *, mount=None, intrinsics=None) -> str:
    """Bake an FPV camera as a child of the vehicle's base body so it follows the vehicle with
    no per-frame posing; fpv_renderer poses its in-code camera every frame, and this is the
    same camera authored into the USD instead. ``pxr``-only: runs host-side.

    Args:
        usdz: the vehicle USD, ``.usd`` / ``.usdz`` / ``.usda``, to read.
        out: output path to write, ``.usd`` / ``.usdz`` / ``.usda``.
        mount: optional ``{"offset": (x,y,z) m (body FRD), "down_tilt_deg": deg}``; missing keys
            default to the fpv_renderer values, ``(0.15,0,0)`` and ``10°``.
        intrinsics: optional ``{"focal_length": mm, "h_aperture": mm}`` plus, to match a non-default
            render resolution, either ``{"v_aperture": mm}`` directly or ``{"width": px, "height": px}``;
            the vertical aperture is then ``h_aperture * height/width``. Defaults: ``12.0`` / ``36.0``
            focal/horizontal and ``36*720/1280 = 20.25`` vertical, the 720p feed.

    The baked local transform = translate(mount_offset) ∘ rotate(R_body_cam), where R_body_cam =
    ``_R_BODY_CAM @ Rx(down_tilt_deg)``; see :func:`_body_cam_rotation`. Stored as a single USD
    ``xformOp:transform`` matrix. USD matrices are row-vector / row-major, so the upper-left 3x3
    is ``R_body_cam.T`` and the translation sits in row 3; composing under the body's world
    transform reproduces the in-code ``cam_pos = p + R_body·mount`` / ``q = R_body·R_body_cam``.

    The function bakes the vertical aperture too, matching fpv_renderer.py:216: without it a consumer
    reading ``verticalAperture`` directly, or an RTX render product that honours the authored vertical
    FOV, would fall back to the UsdGeomCamera schema default of 15.29 mm and crop the view tighter than
    the in-code fallback, vertical FOV 65° instead of 80° at focal 12, an A/B framing divergence.

    NOTE: the authored FpvCam is a child of the base body, so it inherits the body's full world
    transform including scale, unlike the in-code camera, which the renderer poses absolutely each
    frame. This assumes the base body carries identity scale, as rigid bodies normally do; a non-unit
    scale would scale the mount offset and a non-uniform scale would shear the view basis, so verify
    identity scale on the real vehicle USD during the operational A/B.

    Returns the output path.
    """
    from pxr import Gf, UsdGeom

    m = dict(mount or {})
    offset = tuple(m.get("offset", _MOUNT_OFFSET))
    down_tilt_deg = float(m.get("down_tilt_deg", _DOWN_TILT_DEG))
    intr = dict(intrinsics or {})
    focal = float(intr.get("focal_length", _FOCAL_LENGTH))
    h_aperture = float(intr.get("h_aperture", _H_APERTURE))
    if "v_aperture" in intr:
        v_aperture = float(intr["v_aperture"])
    else:
        width = float(intr.get("width", _DEFAULT_WIDTH))
        height = float(intr.get("height", _DEFAULT_HEIGHT))
        v_aperture = h_aperture * height / width

    stage = _open_stage(usdz)
    body = _find_base_body(stage)

    cam_path = body.GetPath().AppendChild("FpvCam")
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.GetFocalLengthAttr().Set(focal)
    cam.GetHorizontalApertureAttr().Set(h_aperture)
    cam.GetVerticalApertureAttr().Set(v_aperture)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e6))

    # Body-relative camera rotation in the column-vector convention → the USD row-vector matrix needs R.T.
    rbc = _body_cam_rotation(down_tilt_deg)
    rt = rbc.T
    mtx = Gf.Matrix4d(
        float(rt[0, 0]),
        float(rt[0, 1]),
        float(rt[0, 2]),
        0.0,
        float(rt[1, 0]),
        float(rt[1, 1]),
        float(rt[1, 2]),
        0.0,
        float(rt[2, 0]),
        float(rt[2, 1]),
        float(rt[2, 2]),
        0.0,
        float(offset[0]),
        float(offset[1]),
        float(offset[2]),
        1.0,
    )
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(mtx)
    # RTX render params as sensor:* custom attrs; the vehicle USD is the single authority, so
    # resolution/rate live here, not in code; constructor-kwarg naming, as in the analytic suite.
    from pxr import Sdf

    prim = cam.GetPrim()
    prim.CreateAttribute("sensor:width", Sdf.ValueTypeNames.Int, custom=True).Set(
        int(intr.get("width", _DEFAULT_WIDTH))
    )
    prim.CreateAttribute("sensor:height", Sdf.ValueTypeNames.Int, custom=True).Set(
        int(intr.get("height", _DEFAULT_HEIGHT))
    )
    prim.CreateAttribute("sensor:rate_hz", Sdf.ValueTypeNames.Float, custom=True).Set(float(intr.get("rate_hz", 24.0)))

    out = str(out)
    stage.GetRootLayer().Export(out)
    return out


def _open_stage(path):
    """Open a USD or usdz stage: a thin wrapper so the lazy ``pxr`` import stays inside the handler."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"could not open USD stage: {path}")
    return stage


# The unified-actuator param attrs authored on each rotor REVOLUTE joint, and the legacy
# ``freefly:actuator:*`` keys they migrate from; see physics/builders/usd.parse_rotor_joint_params.
_ROTOR_PARAM_MAP = {
    "propeller:ct": "ct",
    "propeller:cd": "cd",
    "propeller:rpm_max": "rpm_max",
    "propeller:aero_h": "aero_h",
    "propeller:aero_hforce": "aero_hforce",
    "motor:tau": "tau",
}


def author_rotor_params(usdz, out, *, params=None) -> str:
    """Re-author the vehicle USD onto the unified-actuator schema: write the motor + propeller params as
    ``motor:*`` / ``propeller:*`` custom Float attrs on each rotor ``PhysicsRevoluteJoint``, then delete
    the legacy ``NewtonActuator`` prims. ``pxr``-only: runs host-side. Everything else survives: geometry,
    joints, the authored ``FpvCam``.

    ``params`` is an optional ``{ct, cd, rpm_max, aero_h, aero_hforce, tau}`` dict; if omitted, the function
    reads the uniform values from the existing ``freefly:actuator:*`` attrs on the ``NewtonActuator`` prims,
    so the migration keeps the same values. Returns the output path.
    """
    from pxr import Sdf

    stage = _open_stage(usdz)
    if params is None:
        params = _read_legacy_actuator_params(stage)

    joints, actuators = [], []
    for prim in stage.Traverse():
        tn = prim.GetTypeName()
        if tn == "PhysicsRevoluteJoint":
            joints.append(prim)
        elif tn == "NewtonActuator":
            actuators.append(prim.GetPath())
    if not joints:
        raise ValueError("no PhysicsRevoluteJoint rotor joints found to author motor:*/propeller:* onto")

    for joint in joints:
        for attr_name, key in _ROTOR_PARAM_MAP.items():
            attr = joint.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float, custom=True)
            attr.Set(float(params[key]))
    for path in actuators:
        stage.RemovePrim(path)  # drop the legacy NewtonActuator schema, which went away with newton.actuators

    out = str(out)
    if out.endswith(".usdz"):
        # usdz is a package, and pxr forbids a direct Export to it. Export the edited, self-contained, stage to
        # a sidecar .usdc, then zip it into the .usdz via UsdUtils; the astro-max usdz is a single usdc, no
        # external textures, so this is a faithful repackage.
        import os

        from pxr import UsdUtils

        tmp_usdc = out[: -len(".usdz")] + ".usdc"
        stage.GetRootLayer().Export(tmp_usdc)
        if not UsdUtils.CreateNewUsdzPackage(tmp_usdc, out):
            raise RuntimeError(f"UsdUtils.CreateNewUsdzPackage failed writing {out}")
        os.remove(tmp_usdc)
    else:
        stage.GetRootLayer().Export(out)
    return out


def _read_legacy_actuator_params(stage) -> dict:
    """Read the uniform ``freefly:actuator:*`` params off the ``NewtonActuator`` prims of an as-built USD."""
    prefix = "freefly:actuator:"
    per_rotor = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "NewtonActuator":
            continue
        p = {a.GetName()[len(prefix) :]: a.Get() for a in prim.GetAttributes() if a.GetName().startswith(prefix)}
        if p:
            per_rotor.append(p)
    if not per_rotor:
        raise ValueError("no freefly:actuator:* params found on NewtonActuator prims; pass params= explicitly")
    first = {k: float(v) for k, v in per_rotor[0].items()}
    missing = [k for k in _ROTOR_PARAM_MAP.values() if k not in first]
    if missing:
        raise ValueError(f"legacy actuator params missing keys {missing} (have {sorted(first)})")
    return first
