"""Author the five Astro Max Universal Scene Description (USD) variants, Peter's sensor-configuration
matrix, saved locally.

1. ``astro_max_base.usdz``: base body + rotor bodies/joints only, no RTX sensors
2. ``astro_max_fpv.usdz``: 1 + the FpvCam, which is today's registry USD
3. ``astro_max_fpv_lr1.usdz``: 2 + the LR1 payload camera a fixed distance below the body, +z in
   the Forward Right Down (FRD) frame, forward-looking: aligned with the body x, no tilt
4. ``astro_max_fpv_flux.usdz``: 2 + the Flux payload, an RTX OmniLidar, at the LR1 pose. This
   script doesn't author the OmniLidar itself: a schema-complete OmniLidar needs Kit, via
   ``isaacsim.sensors.experimental.rtx``, so run scripts/assets/author_lidar_variant.py in the
   container once; it inherits the sensor suite from variant 2. When the Kit-authored file already
   exists, this script re-authors the analytic sensor suite onto it in place.
5. ``astro_max_fpv_wiris.usdz``: 2 + the Wiris IR payload: an IrCam beside the FpvCam on the same
   nose mount, carrying ``sensor:modality = "ir"``, the thermal sensor class, at 1280x1024, which
   is the 640x512 Long Wave Infrared (LWIR) core once the Arbitrary Output Variable (AOV) comes
   back at the Deep Learning Super Sampling (DLSS) internal resolution.

Every variant carries the analytic PX4 sensor suite, :data:`SENSOR_SUITE`: Inertial Measurement
Unit (IMU) / mag / baro / Global Positioning System (GPS) as ``sensor:*`` Xform prims on the base
body, and the rotor motors, :data:`ROTOR_MOTOR`: one ``NewtonActuator`` prim per rotor revolute
joint, a ``NewtonPIDControlAPI`` velocity servo + a ``NewtonDCMotorClampingAPI`` envelope, parsed
by ``add_usd`` onto ``model.actuators``, which is what the core :class:`ArticulatedRotors` steps.
The vehicle USD is the single authority for all sensors and actuators, and the assemblies build
both from it, with no fallback.

Source: the resolved registry ``astro_max_base.usdz``, sha-cached, so run a sim once. Output:
``assets/local/``, gitignored; push to S3 via scripts/assets/prepare_asset_upload.py. Host-side pxr
only, + the codeless ``newton_usd_schemas`` plugin for the NewtonActuator prims.

Run:  uv run python scripts/assets/author_astro_variants.py
"""

from __future__ import annotations

import glob
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.assets.convert import (  # noqa: E402
    _body_cam_rotation,
    _find_base_body,
)

OUT_DIR = REPO / "assets" / "local"
LR1_OFFSET = (0.0, 0.0, 0.10)  # 10 cm below the body, FRD +z is down, forward-looking

# The analytic PX4 sensor suite, authored on every variant: (prim name, sensor:type, sensor:* params).
# Param names are exactly the sensor constructors' kwargs in nexus/_src/vehicle/sensors/sensors.py; the
# prim's local translation is the mount offset, all origin-mounted today. Mag sigma is the realistic
# ~300 nT: PX4's per-sample World Magnetic Model (WMM) strength gate trips on the ~10x noisier bridge
# default, and refuses to arm.
SENSOR_SUITE = (
    ("Imu", "imu", {"acc_noise": 0.02, "gyro_noise": 0.02}),
    ("Mag", "mag", {"mag_offset": (0.0, 0.0, 0.0), "noise": (0.003, 0.003, 0.003)}),
    ("Baro", "baro", {"noise": 0.02}),
    ("Gps", "gps", {"fix_type": 3}),
)

# The rotor motor: one NewtonActuator per rotor joint, a kd-only ControllerPID velocity servo with
# kp = ki = 0 → effort = kd·(Ω_cmd − Ω) + feedforward, clamped by the ClampingDCMotor four-quadrant
# torque–speed envelope. Values from the validated conformed-actuator spike on Astro Max rotors:
# kd 0.025 N·m·s, saturation/max effort 8 N·m. The velocity limit derives per vehicle from the
# authored propeller:rpm_max, the DC envelope's no-load speed.
ROTOR_MOTOR = {"kp": 0.0, "ki": 0.0, "kd": 0.025, "integralMax": 0.0,
               "saturationEffort": 8.0, "maxMotorEffort": 8.0}  # + velocityLimit from rpm_max  # fmt: skip
_RPM_PER_RADS = 60.0 / (2.0 * 3.141592653589793)


def _load_flat(src: str):
    """Open *src* fully, with LoadAll, and flatten into a single-layer stage: the registry usdz composes
    the camera through a layer/payload the plain root-layer export drops; flattening bakes everything,
    and single-layer edits such as RemovePrim then stick.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(src), Usd.Stage.LoadAll)
    if stage is None:
        raise ValueError(f"could not open USD stage: {src}")
    return Usd.Stage.Open(stage.Flatten())


def _export_usdz(stage, out: pathlib.Path) -> None:
    """Write *stage* as a .usdz; layer.Export can't write usdz directly, so usdc temp -> package."""
    import tempfile

    from pxr import UsdUtils

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / (out.stem + ".usdc")
        stage.GetRootLayer().Export(str(tmp))
        out.unlink(missing_ok=True)  # replace in place; a container-authored file might be root-owned
        ok = UsdUtils.CreateNewUsdzPackage(str(tmp), str(out))
        if not ok:
            raise RuntimeError(f"usdz packaging failed for {out}")


def _source_usdz() -> str:
    """The current registry astro_max_base, its sha read from registry.yaml, in the asset cache."""
    import yaml

    from nexus._src.assets.resolver import default_cache

    reg = yaml.safe_load((REPO / "nexus" / "_src" / "config" / "registry.yaml").read_text())
    sha = next(v["usd"]["sha256"] for v in reg["vehicles"] if v["name"] == "astro_max_base")
    for root in (default_cache(), pathlib.Path(os.path.expanduser("~/.cache/nexus/assets"))):
        cands = glob.glob(str(root / "**" / "astro_max_base*.usdz"), recursive=True)
        target = [c for c in cands if sha in c or sha in pathlib.Path(c).parent.name]
        if target:
            return target[0]
    raise SystemExit(f"registry astro_max_base.usdz (sha {sha[:8]}…) not in the asset cache: run a sim once")


def _author_px4_sensors(stage, body) -> None:
    """Author the analytic PX4 sensor suite as ``sensor:*`` Xform prims on the base body. Idempotent
    by replacement: the function removes any pre-existing sensor prim first, since re-authoring onto
    an existing variant must not leave stale prims/attrs behind: duplicates fail the build.
    """
    from pxr import Gf, Sdf, UsdGeom

    for path in [p.GetPath() for p in stage.Traverse() if p.GetAttribute("sensor:type").HasAuthoredValue()]:
        stage.RemovePrim(path)
    value_types = {float: Sdf.ValueTypeNames.Float, int: Sdf.ValueTypeNames.Int, tuple: Sdf.ValueTypeNames.Float3}
    for name, kind, params in SENSOR_SUITE:
        prim = UsdGeom.Xform.Define(stage, body.GetPath().AppendChild(name)).GetPrim()
        prim.CreateAttribute("sensor:type", Sdf.ValueTypeNames.Token, custom=True).Set(kind)
        for key, val in params.items():
            attr = prim.CreateAttribute(f"sensor:{key}", value_types[type(val)], custom=True)
            attr.Set(Gf.Vec3f(*val) if isinstance(val, tuple) else val)


def _author_rotor_motors(stage) -> None:
    """Author one ``NewtonActuator`` per rotor revolute joint, a sibling prim named ``<joint>_motor``:
    the NewtonPIDControlAPI velocity servo + the NewtonDCMotorClampingAPI envelope, targets via
    ``rel newton:targets``. ``add_usd`` parses these onto ``model.actuators``, the core
    ``ArticulatedRotors``' motors. Idempotent by replacement. The velocity limit derives from the
    joint's authored ``propeller:rpm_max``.
    """
    import newton_usd_schemas  # noqa: F401  # registers the codeless NewtonActuator schema via Plug
    from pxr import Sdf

    for path in [p.GetPath() for p in stage.Traverse() if p.GetTypeName() == "NewtonActuator"]:
        stage.RemovePrim(path)
    joints = [p for p in stage.Traverse() if p.GetTypeName() == "PhysicsRevoluteJoint"]
    if not joints:
        raise SystemExit("no PhysicsRevoluteJoint rotor joints found: wrong source USD?")
    for j in joints:
        rpm_max = j.GetAttribute("propeller:rpm_max").Get()
        if not rpm_max:
            raise SystemExit(f"{j.GetPath()} lacks propeller:rpm_max, the motor's velocity limit derives from it")
        prim = stage.DefinePrim(j.GetPath().GetParentPath().AppendChild(j.GetName() + "_motor"), "NewtonActuator")
        prim.AddAppliedSchema("NewtonPIDControlAPI")
        prim.AddAppliedSchema("NewtonDCMotorClampingAPI")
        prim.CreateRelationship("newton:targets").SetTargets([j.GetPath()])
        for key, val in ROTOR_MOTOR.items():
            prim.CreateAttribute(f"newton:{key}", Sdf.ValueTypeNames.Float).Set(float(val))
        vlim = float(rpm_max) / _RPM_PER_RADS  # rad/s at the authored rpm_max
        prim.CreateAttribute("newton:velocityLimit", Sdf.ValueTypeNames.Float).Set(vlim)


def _author_camera(
    stage,
    body,
    name: str,
    offset,
    down_tilt_deg: float,
    *,
    width: int = 1280,
    height: int = 720,
    rate_hz: float = 24.0,
    modality: str | None = None,
) -> None:
    """Author one camera as a body child, via the author_vehicle_camera transform recipe.

    The render params default to the EO payload's, so a variant that passes none comes out
    exactly as before. The function writes ``modality`` only when given: a missing ``sensor:modality``
    means EO, which is what keeps every vehicle authored before the attribute existed unchanged.
    """
    from pxr import Gf, Sdf, UsdGeom

    cam = UsdGeom.Camera.Define(stage, body.GetPath().AppendChild(name))
    cam.GetFocalLengthAttr().Set(12.0)
    cam.GetHorizontalApertureAttr().Set(36.0)
    cam.GetVerticalApertureAttr().Set(36.0 * height / width)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e6))
    # RTX render params as sensor:* custom attrs with constructor-kwarg naming, as in the analytic
    # suite: the vehicle USD is the single authority; resolution/rate live here, not in code.
    prim = cam.GetPrim()
    prim.CreateAttribute("sensor:width", Sdf.ValueTypeNames.Int, custom=True).Set(int(width))
    prim.CreateAttribute("sensor:height", Sdf.ValueTypeNames.Int, custom=True).Set(int(height))
    prim.CreateAttribute("sensor:rate_hz", Sdf.ValueTypeNames.Float, custom=True).Set(float(rate_hz))
    if modality is not None:
        prim.CreateAttribute("sensor:modality", Sdf.ValueTypeNames.Token, custom=True).Set(modality)
    rt = _body_cam_rotation(down_tilt_deg).T
    mtx = Gf.Matrix4d(
        float(rt[0][0]), float(rt[0][1]), float(rt[0][2]), 0.0,
        float(rt[1][0]), float(rt[1][1]), float(rt[1][2]), 0.0,
        float(rt[2][0]), float(rt[2][1]), float(rt[2][2]), 0.0,
        float(offset[0]), float(offset[1]), float(offset[2]), 1.0,
    )  # fmt: skip
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(mtx)


def main() -> None:
    src = _source_usdz()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"source: {src}")

    # 1. base: strip the FpvCam
    stage = _load_flat(src)
    body = _find_base_body(stage)
    cam = stage.GetPrimAtPath(body.GetPath().AppendChild("FpvCam"))
    if cam.IsValid():
        stage.RemovePrim(cam.GetPath())
    _author_px4_sensors(stage, body)
    _author_rotor_motors(stage)
    out1 = OUT_DIR / "astro_max_base.usdz"
    _export_usdz(stage, out1)
    print(f"1. {out1}")

    # 2. fpv: the registry USD already carries the FpvCam; re-author it through the same recipe,
    # nose mount and 10° down-tilt, so it carries the full authored sensor surface: intrinsics and
    # the sensor:width/height/rate_hz render params, since the vehicle USD is the single authority.
    out2 = OUT_DIR / "astro_max_fpv.usdz"
    stage = _load_flat(src)
    body = _find_base_body(stage)
    _author_px4_sensors(stage, body)
    _author_rotor_motors(stage)
    _author_camera(stage, body, "FpvCam", (0.15, 0.0, 0.0), down_tilt_deg=10.0)
    _export_usdz(stage, out2)
    print(f"2. {out2}")

    # 3. fpv + the LR1 payload camera: below the body, forward-looking / x-aligned, zero tilt;
    # suite inherited from 2. Prim names follow the payload: FpvCam / Lr1Cam / Flux.
    stage = _load_flat(str(out2))
    body = _find_base_body(stage)
    _author_camera(stage, body, "Lr1Cam", LR1_OFFSET, down_tilt_deg=0.0)
    out3 = OUT_DIR / "astro_max_fpv_lr1.usdz"
    _export_usdz(stage, out3)
    print(f"3. {out3}")

    # 5. fpv + the Wiris IR payload: an IrCam beside the FpvCam on the same nose mount, 3 cm right,
    # same 10° down-tilt, so the two feeds show the same scene. Suite inherited from 2. Authored at
    # 1280x1024 because the thermal AOV comes back at the DLSS internal resolution, half, which is
    # 640x512, the real LWIR core size. sensor:modality picks the thermal sensor class at launch.
    stage = _load_flat(str(out2))
    body = _find_base_body(stage)
    _author_camera(
        stage, body, "IrCam", (0.15, 0.03, 0.0), down_tilt_deg=10.0,
        width=1280, height=1024, rate_hz=24.0, modality="ir",
    )  # fmt: skip
    out5 = OUT_DIR / "astro_max_fpv_wiris.usdz"
    _export_usdz(stage, out5)
    print(f"5. {out5}")

    # 4. fpv + the Flux payload, an RTX OmniLidar: the OmniLidar prim needs Kit, so run
    # author_lidar_variant.py once in the container; it sources variant 2, so a fresh run
    # inherits the suite. An existing Kit-authored file only needs host-side re-authoring here.
    out4 = OUT_DIR / "astro_max_fpv_flux.usdz"
    legacy4 = OUT_DIR / "astro_max_fpv_lidar.usdz"  # pre-payload-naming artifact
    src4 = out4 if out4.exists() else legacy4
    if src4.exists():
        stage = _load_flat(str(src4))
        body = _find_base_body(stage)
        _author_px4_sensors(stage, body)
        _author_rotor_motors(stage)
        _author_camera(stage, body, "FpvCam", (0.15, 0.0, 0.0), down_tilt_deg=10.0)
        from pxr import Sdf

        # Payload naming: the lidar prim is the Flux; rename a legacy 'Lidar' prim in place.
        legacy_prim = stage.GetPrimAtPath(body.GetPath().AppendChild("Lidar"))
        if legacy_prim.IsValid():
            edit = Sdf.BatchNamespaceEdit()
            edit.Add(Sdf.NamespaceEdit.Rename(legacy_prim.GetPath(), "Flux"))
            stage.GetRootLayer().Apply(edit)
        flux = stage.GetPrimAtPath(body.GetPath().AppendChild("Flux"))
        if flux.IsValid():
            # The OmniLidar schema itself is Kit-authored; its sensor:* render params are plain
            # custom attrs, host-authored: one sample per full scan.
            flux.CreateAttribute("sensor:rate_hz", Sdf.ValueTypeNames.Float, custom=True).Set(10.0)
        _export_usdz(stage, out4)
        print(f"4. {out4}")
    else:
        print(f"4. {out4} -> run scripts/assets/author_lidar_variant.py in the isaacsim container")


if __name__ == "__main__":
    main()
