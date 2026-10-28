"""The core loop assembly, shared by every run, per §12.

Controller-agnostic by construction: the assembly wires the plant, ``NewtonPhysics`` from nexus_uav's
own ModelBuilder + solvers, every command element and force element the vehicle declares, each rotor's ESC and
propeller among them, around the Newton motors physics steps, the sensor suite authored in
Universal Scene Description (USD) over the model's sites and world values, and the Rerun sink around a
**caller-supplied controller**. The registry builds each of them from its declaration; the assembly names no
class. Which controller flies is a decision one layer up: the run's builder
builds the controller the vehicle USD declares, and the example controllers self-assemble beside their flight
scripts in ``nexus_uav/examples/controllers/*/assembly.py``, reusing the helpers here,
``build_scenario`` / ``resolve_device``.

What a class requires by name, a peer's link or the render stage the RTX sensors require, enters through
``peer_links``: the loop hands each component what its class requires at ``connect``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import warp as wp

from nexus_uav._src.core import Clock, Loop, SeedTree, logger
from nexus_uav._src.core.components import resolve_components, run_keywords, sibling_keywords
from nexus_uav._src.core.peer_inputs import PeerInputs
from nexus_uav._src.physics import NewtonPhysics
from nexus_uav._src.recording import Recorder


@dataclass(slots=True)
class Assembly:
    """The shared component bundle: everything but the run loop."""

    clock: Any
    physics: Any
    commands: list  # the command elements the vehicle declares, each rotor's ESC among them
    forces: list  # the force elements the vehicle declares, each rotor's propeller among them
    sensors: list
    controller: Any
    logger: Any  # the Rerun sink, or None for no recording
    prims: dict  # each declared component by the path of its prim, for the vehicle's connections


def declared_elements(physics, usd_path, components, instance: int) -> list[tuple]:
    """Every command element and force element the vehicle declares, built through the registry as a sensor is:
    the model and its prim's path, then the schema's attributes, the run's values its class states it needs, and
    the attributes of each schema beside its own that its class names in ``siblings``. Returns each spec with its
    component, in the order of the prims.
    """
    elements = [spec for spec in resolve_components(usd_path, components) if spec.role in ("command", "force")]
    return [
        (
            spec,
            spec.cls(
                physics.model,
                spec.prim,
                **spec.kwargs,
                **run_keywords(spec.cls, instance),
                **sibling_keywords(spec, elements),
            ),
        )
        for spec in elements
    ]


def elements(physics, vehicle_usd, components=None, instance: int = 0) -> tuple[list, list]:
    """The command elements and the force elements the vehicle declares, each rotor's ESC and propeller among them,
    as the two lists a loop takes: what an example's own assembly builds beside its controller.
    """
    built = declared_elements(physics, vehicle_usd.cfg["usd_path"], components, instance)
    return (
        [component for spec, component in built if spec.role == "command"],
        [component for spec, component in built if spec.role == "force"],
    )


def assemble(
    physics,
    vehicle_usd,
    cfg: dict,
    *,
    controller,
    rerun: bool = False,
    viewer: bool = True,
    debug: bool = False,
    components=None,
    settings: dict | None = None,
    record_to_rrd: str | None = None,
    instance: int = 0,
) -> Assembly:
    """Assemble the shared core components around the constructed ``physics`` and the
    caller-supplied ``controller``; which controller flies is the launch layer's decision.
    ``components`` is the registry that resolves each sensor's schema, ``None`` for the default one.
    ``settings`` is the run's effective configuration for the viewer's Settings tab; the launch
    glue passes the tested-config receipt. ``record_to_rrd`` is where the recording goes when the run
    writes one, ``None`` for a new run folder of its own. ``instance`` is the run's instance, which each declared
    sensor and element whose class states ``needs_instance`` takes.
    """
    from nexus_uav._src.vehicle.sensors.declared import build_sensors, sensor_specs

    dt = cfg["physics"]["dt"]
    rtf = cfg["physics"].get("rtf", 0)
    # Every command and force element the vehicle declares, each rotor's ESC and propeller among them, from the
    # vehicle USD's declarations, the single, hash-pinned source, not cfg.
    usd_path = vehicle_usd.cfg["usd_path"]
    built = declared_elements(physics, usd_path, components, instance)
    commands = [component for spec, component in built if spec.role == "command"]
    forces = [component for spec, component in built if spec.role == "force"]

    seedtree = SeedTree(cfg.get("seed", 42))  # the run's builder threads runtime.seed; string-name path keeps 42
    # The sensors come from the schemas the vehicle USD applies, the single, hash-pinned source, the
    # same as the elements. Each reads its site and the world's values, which the scene's
    # PhysicsScene authors, from the model. A vehicle that declares no sensor builds: which sensors a
    # flight needs is its controller's matter.
    specs = sensor_specs(usd_path, components)
    sensors = build_sensors(specs, usd_path=usd_path, model=physics.model, seedtree=seedtree, instance=instance)
    # The central Rerun recording, §10: built here because it needs the physics Model; the import is
    # lazy so rerun is only pulled in when logging is on. Resilient: a logging stack that fails to
    # build must never block the flight; warn and fly without the sink.
    sink = None
    if rerun:
        try:
            from nexus_uav._src.logging import build_logger

            sink = build_logger(
                physics.model, viewer=viewer, base=physics.base_index, record_to_rrd=record_to_rrd, debug=debug,
                settings=settings,
            )  # fmt: skip
        except Exception as exc:
            from nexus_uav._src.core import logger

            logger.warning(f"Rerun recording disabled (logging stack unavailable): {exc}")

    return Assembly(
        clock=Clock(dt, rtf=rtf),
        physics=physics,
        commands=commands,
        forces=forces,
        sensors=sensors,
        controller=controller,
        logger=sink,
        prims={
            **dict(zip([spec.prim for spec in specs], sensors, strict=True)),
            **{spec.prim: component for spec, component in built},
        },
    )


# Default scenario. No sensor entry, by design: the scene's PhysicsScene authors its geodetic origin
# and the world's values, which the model holds, and the sensor suite, which sensors exist + their
# noise/mount params, lives in the vehicle USD as applied sensor schemas, the same as the rotors.
DEFAULT_SCENARIO = {
    "physics": {"enabled": True, "dt": 0.004, "force_cpu": False, "rtf": 0, "solver": "mujoco"},
    # No rotor entry, by design. Each rotor's values, its channel, its ct, cd, aero_h and aero_hforce, and its
    # motor's no-load speed, live in the vehicle USD, declared on its body and its motor, and VehicleUsd.rotors()
    # reads them; since the USD is content-hashed, the vehicle hash pins the rotors, not any config.
}


def build_scenario() -> dict:
    return copy.deepcopy(DEFAULT_SCENARIO)


def _link_ends(peers: dict, connections: dict, by_prim: dict) -> dict:
    """The loop end of each declared peer's wired inputs, by the peer's prim: for each connection on a peer's prim,
    the outputs its targets' components declare under the signal's name, from which the end declares what it reads.
    A peer whose prim wires nothing has no end.

    Raises:
        ValueError: A target declares no component, or its component writes no signal of that name; the message
            names the peer's prim, the signal and the target.
    """
    ends = {}
    for prim, peer in peers.items():
        inputs = {}
        for signal, targets in connections.get(prim, {}).items():
            outputs = []
            for target in targets:
                component = by_prim.get(target)
                output = _output(component, signal) if component is not None else None
                if output is None:
                    what = "declares no component" if component is None else f"writes no {signal!r}"
                    raise ValueError(
                        f"{prim} connects {signal!r} to {target}, which {what}: a peer's input targets the prims "
                        "whose components write the signal"
                    )
                outputs.append(output)
            inputs[signal] = outputs
        if inputs:
            ends[prim] = PeerInputs(peer, inputs)
    return ends


def _output(component, name: str):
    """The signal `name` among the outputs the stages of `component` declare, or ``None``."""
    return next((signal for stage in component.stages() for signal in stage.writes if signal.name == name), None)


def resolve_device(cfg: dict) -> str:
    """Select + activate the Warp device for this build, returning its name. ``physics.force_cpu``
    forces CPU, the bit-exact determinism authority; otherwise prefer CUDA when available so
    the live sim runs on the GPU and the captured strategy can engage, falling
    back to CPU when there is no CUDA device. Must run *before* ``NewtonPhysics`` builds the model, as
    the model + state live on the active device. Newton's own default device is CPU, so a non-CPU
    run must set CUDA explicitly here, not rely on the Warp default.
    """
    if cfg["physics"].get("force_cpu") or not wp.is_cuda_available():
        wp.set_device("cpu")
        return "cpu"
    wp.set_device("cuda:0")
    return "cuda:0"


def build_loop(
    vehicle: str,
    cfg: dict,
    vehicle_usd=None,
    *,
    controller,
    estimator=None,
    guidance=None,
    rerun: bool = False,
    viewer: bool = True,
    debug: bool = False,
    peers=(),
    peer_links=None,
    peer_inputs=None,
    connections=None,
    prims=None,
    preroll_timeout: float = 30.0,
    max_steps: int | None = None,
    settings: dict | None = None,
    components=None,
    record_to_rrd: str | None = None,
    instance: int = 0,
) -> Loop:
    """The core loop: the one ``NewtonPhysics`` + the one shared assembly around the
    caller-supplied ``controller``, ``estimator`` and ``guidance``, ``None`` for none.

    ``components`` resolves each sensor's schema to its class.
    ``peers`` are what the build started for this run, which the loop stops when the run ends.
    ``peer_links`` holds what the run made for each name a class requires, by the name: a peer's link, or the
    render stage the RTX sensors require.
    ``peer_inputs`` maps the prim of each declared peer to the peer: for each input the vehicle wires on that
    prim, the loop end of the link is a component in the ``link`` role, which hands the peer each writer's value.
    ``connections`` are the vehicle's, by prim: for a reader's prim, each signal it reads and the prims
    whose components write it, one for a component's input and one or more for a peer's, or the body's or joint's
    prim whose output the plant writes; ``prims`` names
    the controller, the estimator and the guidance by their prims, as the assembly names each sensor and
    element, so the loop takes each
    connection between the two components, or between the reader and the plant's output.
    ``settings`` is the run's effective configuration, which the loop keeps for a caller to read back
    and the logger shows in the viewer's Settings tab; the run's builder passes the tested-config receipt.
    ``record_to_rrd`` is where the recording goes when the run writes one: the run's builder passes
    ``nexus.rrd`` in the run's folder, and ``None`` takes a new run folder.
    ``preroll_timeout`` covers the boot of a controller with a peer, since an autopilot in a container
    needs a generous window. ``instance`` is the run's instance, which each declared component built here takes
    when its class states ``needs_instance``.

    Raises:
        ValueError: A connection names a prim that declares no component and is no body or joint of the
            model; the message names both prims.
    """
    logger.info(f"device: {resolve_device(cfg)}")
    if vehicle_usd is None:
        raise ValueError(f"vehicle_usd is required for {vehicle!r} (resolve it with resolve_vehicle_usd)")
    physics = NewtonPhysics(vehicle_usd=vehicle_usd, cfg=cfg)
    a = assemble(
        physics, vehicle_usd, cfg,
        controller=controller, rerun=rerun, viewer=viewer, debug=debug, components=components, settings=settings,
        record_to_rrd=record_to_rrd, instance=instance,
    )  # fmt: skip
    by_prim = {**a.prims, **(prims or {})}
    ends = _link_ends(peer_inputs or {}, connections or {}, by_prim)
    by_prim.update(ends)
    links = []
    for reader, inputs in (connections or {}).items():
        for signal, writers in inputs.items():
            if reader not in by_prim:
                raise ValueError(
                    f"{reader} connects {signal!r} to {' and '.join(writers)}, and declares no component itself"
                )
            for writer in writers:
                # A body's or a joint's prim names the plant's output of it, even where an element sits on that
                # prim, as a rotor's ESC and propeller sit on its body: an element writes no pose.
                if writer in physics.outputs.by_label:
                    links.append((by_prim[reader], signal, physics, physics.outputs.by_label[writer].name))
                elif writer in by_prim:
                    links.append((by_prim[reader], signal, by_prim[writer], signal))
                else:
                    who = type(by_prim[reader]).__name__
                    raise ValueError(
                        f"{who} on {reader} connects {signal!r} to {writer}, which declares no component and is no "
                        "body or joint of the model"
                    )
    return Loop(
        clock=a.clock,
        physics=a.physics,
        commands=a.commands,
        forces=a.forces,
        sensors=a.sensors,
        controller=a.controller,
        estimator=estimator,
        guidance=guidance,
        logger=a.logger,
        recorder=Recorder(),
        links=dict(peer_links or {}),
        peers=peers,
        peer_inputs=list(ends.values()),
        connections=links,
        preroll_timeout=preroll_timeout,
        max_steps=max_steps,
        settings=settings,
    )
