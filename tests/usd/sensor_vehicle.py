"""The fixture vehicle of the sensor tests, and the stand-ins a run of it needs.

The fixture is a local layer over the shipped `astro_max_base`, so it flies real bodies and rotors. It
deactivates the shipped sensor prims and drops the PX4 Software In The Loop (SITL) peer, and a test adds
its own sensor prims, under names the shipped vehicle leaves free. A stand-in controller answers at once and keeps every `Measurement` it
receives, so a test reads what a controller reads.
"""

from __future__ import annotations

import dataclasses
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np

from nexus._src.config import LaunchConfig, resolve
from nexus._src.core.interfaces import Stage
from nexus._src.core.registry import ComponentRegistry
from nexus._src.core.schema import Controls
from nexus._src.core.stages import peer_stages
from nexus._src.usd import ENTRY_POINT_GROUP

ROOT = "/astro_max"
# The base body, model body 0.
BODY = f"{ROOT}/Geometry/body_frd"
# A second rigid body with the base body's leaf name, fixed 1 m over it and rolled 180 degrees about its
# forward axis. The base body's frame is Forward Right Down (FRD), so up is its -Z.
MAST = f"{ROOT}/Geometry/Mast/body_frd"

_MAST = """
        def Xform "Mast"
        {
            def Xform "body_frd" (
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


def vehicle(tmp_path: Path, body: str = "", *, mast: str | None = None, px4: bool = False) -> str:
    """Write the fixture vehicle under `tmp_path` and return its path.

    `body` is the text of the prims under the base body. `mast`, when given, adds the second body and
    is the text of the prims under it. `px4` keeps the PX4 SITL peer declared, for a run that maps it to
    its fake.
    """
    shipped = resolve(LaunchConfig.from_dict({"vehicle": "astro_max_base", "scene": "empty"})).vehicle_usd_path
    geometry = "" if mast is None else _MAST.replace("__MAST_PRIMS__", mast)
    peer = "" if px4 else ' (\n    delete apiSchemas = ["NexusPx4SitlAPI"]\n)'
    joint = "" if mast is None else _MAST_JOINT
    path = tmp_path / "sensor_vehicle.usda"
    path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "astro_max"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [
        @{shipped}@
    ]
)

over "astro_max"{peer}
{{
    over "Geometry"
    {{
{geometry}
        over "body_frd"
        {{
            over "Imu" (
                active = false
            )
            {{
            }}
            over "Mag" (
                active = false
            )
            {{
            }}
            over "Baro" (
                active = false
            )
            {{
            }}
            over "Gps" (
                active = false
            )
            {{
            }}
{body}
        }}
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
    """A stand-in controller: it answers at once with zero commands and keeps each `Measurement` it receives."""

    def __init__(self, **kwargs):
        self.airframe = kwargs.get("airframe")
        self.received = []

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return peer_stages(self)

    def exchange(self, meas, t, timeout=None):
        self.received.append(dataclasses.replace(meas))
        return Controls(command=np.zeros(4))


class StandInSensor:
    """A stand-in sensor: it keeps the run's values and its keyword arguments, and its one host stage does nothing."""

    def __init__(self, run, **kwargs):
        self.run = run
        self.kwargs = kwargs

    def stages(self):
        return [Stage("stand_in", "host", lambda tick: None)]


def components(**entries) -> ComponentRegistry:
    """A registry of the test's own: every shipped entry, the stand-in controller for PX4, then `entries`."""
    shipped = {entry.name: entry for entry in entry_points(group=ENTRY_POINT_GROUP)}
    return ComponentRegistry({**shipped, "NexusPx4API": Controller, **entries})


def build(vehicle_path: str, *, seed: int = 42, scene: str = "empty", registry=None, **kw):
    """Build a run of the vehicle at `vehicle_path` on the CPU, flown by the stand-in controller.

    `registry` is the catalog, for a scene of the test's own. The rest goes to the build: `components`
    replaces the registry of :func:`components`, and `peers` is the peer mapping.
    """
    import nexus._src.build.launch as launch_mod

    launch = LaunchConfig.from_dict(
        {"vehicle": vehicle_path, "scene": scene, "runtime": {"device": "cpu", "seed": seed}}
    )
    kw.setdefault("components", components())
    return launch_mod.build_from_launch(launch, registry=registry, preroll_timeout=10.0, **kw)


def fly(loop, ticks: int) -> list:
    """Step `loop` for `ticks` control ticks, close it, and return each `Measurement` its controller received in them."""
    n = 0
    while n < ticks and loop.step():
        n += 1
    received = loop.controller.received[-n:] if n else []
    loop.close()
    return received
