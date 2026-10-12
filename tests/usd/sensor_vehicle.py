"""The fixture vehicle of the sensor tests, and the stand-ins a run of it needs.

The fixture is a local layer over ``fixture_vehicle.usda`` beside this module: a base body with a box
collider, four rotor bodies with propellers on revolute joints, and the PX4 controller declared on its scope, with
no mesh and no sensor. A test adds its own sensor prims. A run flies it in ``fixture_scene.usda``, an
empty world, so a test reads no hosted asset. A stand-in controller answers at once with zero
commands. A stand-in estimator, which the stand-in schema `StandInEstimatorAPI` declares on a scope of its
own, keeps what each signal it reads holds, so a test reads what a sensor wrote. A fake PX4 peer that keeps
every message it receives lets a test read what PX4 receives.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp

from nexus_uav._src.config import LaunchConfig
from nexus_uav._src.core.clock import DeviceClock
from nexus_uav._src.core.interfaces import Component, Stage
from nexus_uav._src.core.registry import Registry
from nexus_uav._src.core.signals import Signal, wire
from nexus_uav._src.core.stages import Bound
from nexus_uav._src.peers.px4_sitl.fake import Px4Fake

BASE = Path(__file__).with_name("fixture_vehicle.usda")
# The scene a fixture run flies in: an empty world, local, so a test fetches nothing.
SCENE = str(Path(__file__).with_name("fixture_scene.usda"))
# The fixture scene as a catalog entry's file, for a test catalog that names a scene: `{"usd": SCENE_REF}`.
SCENE_REF = {
    "url": Path(SCENE).as_uri(),
    "sha256": sha256(Path(SCENE).read_bytes()).hexdigest(),
    "filename": "fixture_scene.usda",
}
ROOT = "/vehicle"
# The base body, model body 0.
BODY = f"{ROOT}/body"
# A second rigid body with the base body's leaf name, fixed 1 m over it and rolled 180 degrees about its
# forward axis. The base body's frame is Forward Right Down (FRD), so up is its -Z.
MAST = f"{ROOT}/Mast/body"

_MAST = """
    def Xform "Mast"
    {
        def Xform "body" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        )
        {
            float physics:mass = 0.01
            float3 physics:diagonalInertia = (0.0001, 0.0001, 0.0001)
            double3 xformOp:translate = (0, 0, -1)
            float xformOp:rotateX = 180
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateX"]
__MAST_PRIMS__
        }
    }
"""

_MAST_JOINT = f"""
        def PhysicsFixedJoint "mast_joint"
        {{
            rel physics:body0 = <{BODY}>
            rel physics:body1 = <{MAST}>
            point3f physics:localPos0 = (0, 0, -1)
            quatf physics:localRot0 = (0, 1, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
        }}
"""


def prim(name: str, schema: str | None, attrs: str = "", *, kind: str = "Sphere", translate=None) -> str:
    """The text of one prim named `name` of type `kind` that applies `schema`, or none, and authors `attrs`.

    `attrs` is one attribute per line, such as `float nexus:noise = 0.09`. `translate` moves the prim
    from its parent's origin. The default kind is a `Sphere` of 1 cm, the geometry prim an analytic
    sensor's schema sits on, which Newton loads as a site.
    """
    lines = [line.strip() for line in attrs.strip().splitlines() if line.strip()]
    if kind == "Sphere" and not any(line.startswith("double radius") for line in lines):
        lines.insert(0, "double radius = 0.01")
    if translate is not None:
        lines += [
            f"double3 xformOp:translate = {tuple(translate)}",
            'uniform token[] xformOpOrder = ["xformOp:translate"]',
        ]
    metadata = f' (\n    prepend apiSchemas = ["{schema}"]\n)' if schema else ""
    body = "".join(f"    {line}\n" for line in lines)
    return f'def {kind} "{name}"{metadata}\n{{\n{body}}}\n'


def vehicle(
    tmp_path: Path,
    body: str = "",
    *,
    root: str = "",
    mast: str | None = None,
    px4: bool = False,
    controller: str = "",
    estimator: str | None = None,
) -> str:
    """Write the fixture vehicle under `tmp_path` and return its path.

    `body` is the text of the prims under the base body, and `root` the text of more prims under the root prim. `mast`, when given, adds the second body and
    is the text of the prims under it. `px4` declares the PX4 Software In The Loop (SITL) peer, for a
    run that maps it to its fake. `controller` is more text the controller's scope authors, such as a
    connection. `estimator`, when given, declares the stand-in estimator with `StandInEstimatorAPI` on the
    scope `Estimator`, and is the text that scope authors, such as a connection.
    """
    geometry = "" if mast is None else _MAST.replace("__MAST_PRIMS__", mast)
    applied = ' (\n        prepend apiSchemas = ["NexusPx4SitlAPI"]\n    )' if px4 else ""
    peer = f'    over "Controller"{applied}\n    {{\n        {controller}\n    }}\n' if px4 or controller else ""
    if estimator is not None:
        root += prim("Estimator", "StandInEstimatorAPI", estimator, kind="Scope")
    joint = "" if mast is None else _MAST_JOINT
    path = tmp_path / "sensor_vehicle.usda"
    path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "vehicle"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [
        @{BASE}@
    ]
)

over "vehicle"
{{
{peer}{root}{geometry}
    over "body"
    {{
{body}
    }}
    over "Physics"
    {{
{joint}
    }}
}}
"""
    )
    return str(path)


class HostController(Component):
    """A controller with one host stage, ``exchange``, over :meth:`exchange`: the stage writes what the exchange
    returns to the controls signal, ``width`` wide, and reports the peer gone on ``None``, which ends the run. A
    stand-in subclasses it and overrides :meth:`exchange`, and ``reads`` for the signals the exchange reads.
    """

    width = 4
    reads: tuple = ()

    def __init__(self):
        super().__init__()
        self.controls = Signal("controls", wp.float32, shape=(1, self.width))
        self._cmd = np.zeros((1, self.width), dtype=np.float32)

    def stages(self):
        return [Stage("exchange", "host", self._exchange, reads=tuple(self.reads), writes=(self.controls,))]

    def _exchange(self, tick):
        command = self.exchange(tick.t, tick.timeout)
        if command is None:
            return False
        self._cmd[0, :] = command
        self.controls.write(self._cmd)
        return True

    def exchange(self, t, timeout=None):
        return np.zeros(self.width, dtype=np.float32)


class Controller(HostController):
    """A stand-in controller: it answers at once with zero commands."""

    def __init__(self, **kwargs):
        super().__init__()
        self.airframe = kwargs.get("airframe")


class StandInSensor(Component):
    """A stand-in sensor: it keeps the model, its prim's path and its keyword arguments, and its one host stage does
    nothing.
    """

    def __init__(self, model, path, **kwargs):
        self.model = model
        self.path = path
        self.kwargs = kwargs

    def stages(self):
        return [Stage("stand_in", "host", lambda tick: None)]


class _Every(dict):
    """The fake's last message of each kind, which also keeps, in order, every message the fake sets in it."""

    def __init__(self):
        super().__init__()
        self.every: list = []

    def __setitem__(self, kind, msg):
        self.every.append(msg)
        super().__setitem__(kind, msg)


class KeepingFake(Px4Fake):
    """The fake PX4 peer, which keeps every message it receives, in order, in `last.every`, beside the last of each kind."""

    def __init__(self, **run):
        super().__init__(**run)
        self.last = _Every()


def estimator(*reads: Signal) -> type:
    """A stand-in estimator class, for a registry to map `StandInEstimatorAPI` to: its host stage reads the signals `reads`.

    Each tick the host stage keeps the tick's sim time and what each signal read gave, one tuple a tick,
    in the class's `kept`. Its warm device stage launches nothing. Each of its two stages counts its runs in
    the class's `runs`, so a run that stops before any stage runs leaves it at zero. Each call makes a
    class of its own, for one run.
    """

    class StandInEstimator(Component):
        kept: ClassVar[list] = []
        runs = 0

        def __init__(self, **kwargs):
            pass  # the stand-in schema's keyword arguments, which it doesn't read

        def stages(self) -> list[Stage]:
            cls = type(self)

            def count(tick):
                cls.runs += 1

            def keep(tick):
                cls.runs += 1
                cls.kept.append((tick.t.sim_time, *(signal.read() for signal in reads)))

            return [Stage("count", "device", count), Stage("estimate", "host", keep, reads=tuple(reads))]

    return StandInEstimator


def wired(sensor, time: float = 0.0):
    """`sensor`, its signals wired as a run wires them and the tick's sim time on the device at `time`, for a
    test that samples it outside a run.
    """
    clock = DeviceClock(0.004)
    pairs = ((clock, "clock"), (sensor, "sensor"))
    wire([Bound(stage, component, role) for component, role in pairs for stage in component.stages()])
    clock.time.write([time])
    clock.time_ns.write([round(time * 1e9)])
    return sensor


def components(**entries) -> Registry:
    """A registry of the test's own: the stand-in controller for PX4, then `entries`; the install answers the rest."""
    return Registry({"NexusPx4API": Controller, **entries})


def scenario() -> dict:
    """The default scenario in the fixture scene, an empty world whose ``PhysicsScene`` authors its origin, for
    a test that builds its run's cfg by hand: the model takes the origin from the scene.
    """
    from nexus_uav._src.build.assembly import build_scenario

    cfg = build_scenario()
    cfg["scene_usd_path"] = SCENE
    return cfg


# The world's values a hand-built model holds: an origin at 47.6, -122.3, 5 m, and a field of
# (0.21, 0.05, 0.43) gauss as a North East Down (NED) vector.
ORIGIN = (47.6, -122.3, 5.0)
FIELD_NED = (0.21, 0.05, 0.43)


def model(
    *,
    bodies: int = 1,
    mount=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0, 1.0),
    com=(0.0, 0.0, 0.0),
    origin=ORIGIN,
    field_ned=FIELD_NED,
    pressure: float = 1013.25,
    temperature: float = 25.0,
):
    """A model of `bodies` free bodies, each with a site under it, for a sensor built by hand, and the labels of
    the sites, one per body.

    Body 0 carries the fixture's base body's label, and each body after it that label with its index appended.
    Each body's site sits at `mount`, turned by `rotation`, a quaternion (x, y, z, w), in the body's axes, and
    each body's center of mass at `com`. The world's values come from the arguments: `origin`, the field
    `field_ned` turned into the world's axes, `pressure` at sea level and `temperature`; gravity is the world's
    9.81.
    """
    import newton

    from nexus_uav._src.scene import world

    builder = newton.ModelBuilder()
    world.declare(builder)
    n, e, d = field_ned
    builder.custom_attributes[world.ORIGIN].values[0] = wp.vec3d(*(float(x) for x in origin))
    builder.custom_attributes[world.FIELD].values[0] = wp.vec3(n, -e, -d)
    builder.custom_attributes[world.PRESSURE].values[0] = wp.float32(pressure)
    builder.custom_attributes[world.TEMPERATURE].values[0] = wp.float32(temperature)
    sites = []
    for i in range(bodies):
        label = BODY if i == 0 else f"{BODY}{i}"
        body = builder.add_body(
            com=wp.vec3(*(float(x) for x in com)), inertia=wp.mat33(np.eye(3) * 0.01), mass=1.0, label=label
        )
        sites.append(f"{label}/site")
        builder.add_site(body, xform=wp.transform(tuple(mount), wp.quat(*rotation)), label=sites[-1])
    builder.gravity = -world.GRAVITY
    return builder.finalize(), sites


def mounted(**kw):
    """A one-body model of :func:`model` and the label of its site, for ``cls(*mounted(), ...)``."""
    built, (site,) = model(bodies=1, **kw)
    return built, site


def build(
    vehicle_path: str, *, seed: int = 42, scene: str = SCENE, catalog=None, device: str = "cpu", fall_from=None, **kw
):
    """Build a run of the vehicle at `vehicle_path` on `device`, `cpu` or `cuda`, flown by the stand-in controller.

    `catalog` is the catalog, for a scene of the test's own. `fall_from`, a position, places the vehicle there
    in free flight, where it falls, in place of resting on the ground. The rest goes to the build:
    `components` replaces the registry of :func:`components`, and `peers` is the peer mapping.
    """
    import nexus_uav._src.build.launch as launch_mod

    launch = LaunchConfig.from_dict(
        {"vehicle": vehicle_path, "scene": scene, "runtime": {"device": device, "seed": seed}}
    )
    if fall_from is not None:
        _, _, cfg = launch_mod.resolve_scenario(launch, catalog=catalog)
        cfg["physics"]["spawn"] = {"pos": tuple(float(v) for v in fall_from)}
        kw["cfg"] = cfg
    kw.setdefault("components", components())
    return launch_mod.build_from_launch(launch, catalog=catalog, preroll_timeout=10.0, **kw)


def steps(loop, ticks: int) -> int:
    """Step `loop` for `ticks` control ticks, close it, and return how many it ran."""
    n = 0
    while n < ticks and loop.step():
        n += 1
    loop.close()
    return n
