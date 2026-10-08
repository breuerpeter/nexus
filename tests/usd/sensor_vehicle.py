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

from importlib.metadata import entry_points
from pathlib import Path
from typing import ClassVar

import numpy as np

from nexus_sim._src.config import LaunchConfig
from nexus_sim._src.core.clock import DeviceClock
from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.registry import ComponentRegistry
from nexus_sim._src.core.schema import Controls
from nexus_sim._src.core.signals import Signal, wire
from nexus_sim._src.core.stages import Bound, peer_stages
from nexus_sim._src.peers.px4_sitl.fake import Px4Fake
from nexus_sim._src.usd import ENTRY_POINT_GROUP

BASE = Path(__file__).with_name("fixture_vehicle.usda")
# The scene a fixture run flies in: an empty world, local, so a test fetches nothing.
SCENE = str(Path(__file__).with_name("fixture_scene.usda"))
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


def prim(name: str, schema: str | None, attrs: str = "", *, kind: str = "Xform", translate=None) -> str:
    """The text of one prim named `name` of type `kind` that applies `schema`, or none, and authors `attrs`.

    `attrs` is one attribute per line, such as `float nexus:noise = 0.09`. `translate` moves the prim
    from its parent's origin.
    """
    lines = [line.strip() for line in attrs.strip().splitlines() if line.strip()]
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


class Controller:
    """A stand-in controller: it answers at once with zero commands."""

    def __init__(self, **kwargs):
        self.airframe = kwargs.get("airframe")

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return peer_stages(self)

    def exchange(self, t, timeout=None):
        return Controls(command=np.zeros(4))


class StandInSensor:
    """A stand-in sensor: it keeps the run's values and its keyword arguments, and its one host stage does nothing."""

    def __init__(self, run, **kwargs):
        self.run = run
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

    class StandInEstimator:
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
    return sensor


def components(**entries) -> ComponentRegistry:
    """A registry of the test's own: every shipped entry, the stand-in controller for PX4, then `entries`."""
    shipped = {entry.name: entry for entry in entry_points(group=ENTRY_POINT_GROUP)}
    return ComponentRegistry({**shipped, "NexusPx4API": Controller, **entries})


def build(
    vehicle_path: str, *, seed: int = 42, scene: str = SCENE, catalog=None, device: str = "cpu", fall_from=None, **kw
):
    """Build a run of the vehicle at `vehicle_path` on `device`, `cpu` or `cuda`, flown by the stand-in controller.

    `catalog` is the catalog, for a scene of the test's own. `fall_from`, a position, places the vehicle there
    in free flight, where it falls, in place of resting on the ground. The rest goes to the build:
    `components` replaces the registry of :func:`components`, and `peers` is the peer mapping.
    """
    import nexus_sim._src.build.launch as launch_mod

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
