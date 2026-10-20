"""The run's builder: a ``LaunchConfig`` → a running ``Loop``, the one path every run builds through.

Resolves the launch against the catalog, sha-verifying every asset, wraps the vehicle Universal Scene
Description (USD) file in a :class:`VehicleUsd`, threads the resolved scene, its USD plus start, into the
scenario cfg, and builds the run from the vehicle's declarations. The registry maps each applied schema to its
class, so the builder names no component and no peer: it builds the controller, the estimator and the guidance
by the role their schemas state, the sensors and the command and force elements in the assembly, and each
peer the vehicle declares, a schema whose class is a peer. A peer a component's class requires, such as the
Kit render peer an RTX sensor requires, it starts once by the name the class requires it by.

A component's schema attributes are its class's keyword arguments, and so are a peer's. What only the run
knows reaches a peer as keywords beside them: its pin, the run's instance, the run's folder, the files the
run reads, the asset cache and, for a peer with a serial line, the path of each of its ends. A component whose
class states ``needs_instance`` takes the instance too. The run claims one instance on this machine when it
starts a declared peer, which every address its peers use counts from, and instance 0 when it starts none,
for an autopilot started elsewhere.

A serial line between two peers, as a companion computer's line to the flight controller, is wiring on the
vehicle, so the declared peer's prim declares it: a relationship ``nexus:serial:<line>`` its schema defines,
which targets the prim of the peer at the other end, as a component's prim declares its inputs. The run owns
the link as it owns every port: it makes one :class:`~nexus_uav._src.peers.serial.SerialLink` per line before
any peer starts, hands each of the two peers the path of its end in the keyword ``serial``, the declaring peer
by the line's name and the other by the declaring prim's name, and stops the links with the peers at the run's
end.

Before any peer starts, the build resolves and checks the pin of each peer it starts, through
:mod:`nexus_uav._src.peers.pins` and the peer's own ``check_pin``, and the receipt records each started
peer's pin and each declaration with the class it resolved to. The required peers start first, so Kit boots
while the declared peers, PX4 among them, build and the physics compiles.
"""

from __future__ import annotations

import pathlib

import warp as wp

from nexus_uav._src.assets.resolver import default_cache
from nexus_uav._src.config import Catalog, LaunchConfig, ResolvedLaunch, Runtime, resolve
from nexus_uav._src.core import Loop
from nexus_uav._src.core.components import ComponentSpec, PeerSpec, resolve_declarations, run_keywords
from nexus_uav._src.core.errors import RunError
from nexus_uav._src.core.registry import Registry
from nexus_uav._src.core.run_folder import run_folder
from nexus_uav._src.peers import CLOCKS, instance, pins
from nexus_uav._src.peers.files import run_files
from nexus_uav._src.peers.serial import SerialLink
from nexus_uav._src.physics import VehicleUsd
from nexus_uav._src.rendering import PEER as KIT
from nexus_uav._src.rendering import RtxRendererFactory
from nexus_uav._src.usd.reader import read_connections, read_serial_lines

from .assembly import build_loop, build_scenario
from .components import declared_controller, declared_estimator, declared_guidance


def resolve_vehicle_usd(
    launch: LaunchConfig,
    catalog: Catalog | None = None,
    *,
    cache_dir: str | pathlib.Path | None = None,
) -> tuple[VehicleUsd, ResolvedLaunch]:
    """Resolve *launch*, fetching and sha-verifying assets, and wrap the vehicle USD in a VehicleUsd.

    The receipt records the device the run picks: ``cpu`` for an explicit ``cpu`` or a machine with no CUDA
    device, else ``cuda``. The pick sits here, not in ``resolve``, so a caller that only resolves an asset path
    never starts Warp.
    """
    resolved = resolve(launch, catalog, cache_dir=cache_dir)
    rt = resolved.tested_config.runtime
    picked = "cpu" if rt.device == "cpu" or not wp.is_cuda_available() else "cuda"
    resolved.tested_config.runtime = rt.model_copy(update={"device": picked})
    vehicle_usd = VehicleUsd({"usd_path": resolved.vehicle_usd_path})
    return vehicle_usd, resolved


def _scenario_from_receipt(rt: Runtime) -> dict:
    """Map the receipt's runtime fields onto the scenario cfg dict, so the run applies what its receipt
    records: dt, device, solver, rtf and seed. ``max_steps`` goes to the loop.
    """
    cfg = build_scenario()
    cfg["physics"]["dt"] = rt.dt
    # The receipt names the device the run picked, "cpu" or "cuda".
    cfg["physics"]["force_cpu"] = rt.device == "cpu"
    cfg["physics"]["solver"] = rt.solver  # mujoco, the default | semi_implicit | featherstone
    cfg["physics"]["rtf"] = rt.rtf  # 0 = unthrottled; 1.0 = pace to wall-clock, for interactive flying
    cfg["seed"] = rt.seed  # honored by build_loop's SeedTree
    return cfg


def _thread_scene(cfg: dict, resolved: ResolvedLaunch) -> None:
    """Thread the resolved scene into the cfg, uniformly, for every run.

    The physics model build loads the scene USD plus its start point exactly the same way as the
    vehicle USD: physics takes the UsdPhysics-authored prims, see ``scene.add_scene``, and the
    importer reads the geodetic origin the scene's ``PhysicsScene`` prim authors into the model,
    which anchors *both* the render world and the sensors: the scene is the single source of the
    "where on Earth" answer, so the Ground Control Station (GCS) minimap and the camera feed agree.
    When the vehicle renders, the Kit peer opens the same file as its render stage and finds by its
    content whether it needs live machinery, for example the cesium globe.
    """
    cfg["scene_usd_path"] = resolved.scene_usd_path
    cfg["scene_start"] = resolved.tested_config.scene_start


def resolve_scenario(
    launch: LaunchConfig,
    *,
    catalog: Catalog | None = None,
    cache_dir: str | pathlib.Path | None = None,
) -> tuple[VehicleUsd, ResolvedLaunch, dict]:
    """Resolve *launch* to ``(vehicle_usd, resolved, scenario cfg)``: the shared front half of
    any loop assembly. ``build_from_launch`` builds the vehicle's declarations on it; a
    self-assembled example, with its own controller plus actuator plus physics, starts from this and
    hands the built loop to ``Sim._from_loop``.
    """
    vehicle_usd, resolved = resolve_vehicle_usd(launch, catalog, cache_dir=cache_dir)
    cfg = _scenario_from_receipt(resolved.tested_config.runtime)
    _thread_scene(cfg, resolved)
    return vehicle_usd, resolved, cfg


def build_from_launch(
    launch: LaunchConfig,
    *,
    catalog: Catalog | None = None,
    cache_dir: str | pathlib.Path | None = None,
    cfg: dict | None = None,
    preroll_timeout: float = 30.0,
    components: Registry | None = None,
    run_dir: str | pathlib.Path | None = None,
) -> Loop:
    """Resolve *launch* and build the run the vehicle declares around its one controller.

    ``components`` is the registry that resolves each name the run resolves to its class: each schema the
    vehicle applies, a component's or a peer's, and the name a class requires a peer by. Its own rows answer
    first and the install answers the rest; ``None`` takes the install alone. A test sends a peer to its fake
    with a row of its own. ``run_dir`` is the folder the run writes its files to, see
    :func:`~nexus_uav._src.core.run_folder.run_folder`; ``None`` takes a new one.

    Raises:
        ValueError: The vehicle declares no controller or two, two estimators or guidances, one off its
            ``Scope``, a schema no class claims, a component schema that states no role or two, a peer that
            fails its own check of the values its schema authors, or a class requires a peer no class
            claims, or a peer's serial line targets a prim that declares no other peer, or two lines reach one
            peer from prims of one name, or a pin file beside a catalog names no peer, or a peer's pin isn't in
            its form; raised before any peer starts.
        FileNotFoundError: The launch names an override layer with no file behind it.
        FileExistsError: ``run_dir`` holds files already; raised before any peer starts.
        RunError: A peer couldn't start, the Kit peer for RTX sensors or the PX4 peer among them; the
            message names the peer and its error.
    """
    vehicle_usd, resolved = resolve_vehicle_usd(launch, catalog, cache_dir=cache_dir)
    if cfg is None:
        cfg = _scenario_from_receipt(resolved.tested_config.runtime)
    _thread_scene(cfg, resolved)
    label = resolved.tested_config.vehicle or "vehicle"
    registry = components if components is not None else Registry()
    usd = resolved.vehicle_usd_path
    specs, declared = resolve_declarations(usd, registry)
    roles = {
        "controller": declared_controller(usd, registry),
        "estimator": declared_estimator(usd, registry),
        "guidance": declared_guidance(usd, registry),
    }
    required = _required(specs, registry)
    # Every pin file beside a catalog names a peer, and the pin of each peer the run starts is in its form,
    # before any peer starts, so a stale or broken pin file fails each run, not only the runs of its peer.
    layers = resolved.tested_config.catalogs
    classes = [spec.cls for spec in declared] + list(required.values())
    pins.check(layers, sorted({*pins.shipped(), *(cls.name for cls in classes)}))
    found = {cls.name: (cls, pins.resolve(cls.name, layers)) for cls in classes}
    peer_pins = {name: cls.check_pin(pin) if pin is not None else None for name, (cls, pin) in found.items()}
    _check_clocks(classes, launch.runtime.rtf)
    # Every file the run writes lands in one folder: the recording and each peer's logs.
    logs = run_folder(run_dir)
    # The run numbers every address its declared peers use from one instance; with none, an autopilot started
    # elsewhere dials instance 0's.
    claim = instance.claim(spec.cls for spec in declared) if declared else None
    run = {
        "instance": claim.instance if claim is not None else 0,
        "log_dir": logs,
        "files": run_files(usd, resolved.scene_usd_path),
        "cache_dir": cache_dir or default_cache(),
    }
    resolved.tested_config.peers = {name: pin for name, pin in peer_pins.items() if pin is not None}
    resolved.tested_config.declarations = [_declaration(spec) for spec in [*specs, *declared]]
    started: list = []
    links: list = []
    try:
        # Every serial link exists before any peer starts, so each end's path is a device when its peer opens it.
        links, ends = _serial_links(declared, read_serial_lines(usd))
        # The build makes each peer before any starts, so a peer that fails its own check fails the run first.
        made = [_peer(spec.cls, spec.kwargs, peer_pins[spec.cls.name], run, spec.prim, ends) for spec in declared]
        needed = {name: _peer(cls, {}, peer_pins[cls.name], run, None, ends) for name, cls in required.items()}
        # The required peers start first: Kit boots while the declared peers build and the physics compiles.
        for peer in needed.values():
            _start(peer, started)
        renderer_factory = RtxRendererFactory(needed[KIT], log_dir=logs, components=registry) if KIT in needed else None
        built = {role: _component(spec, run["instance"]) for role, spec in roles.items() if spec is not None}
        # A declared peer starts here, before the assembly: PX4's start builds PX4 incrementally, which must
        # stay outside the sim's preroll window, GH #39, and PX4 boots while the physics compiles.
        for peer in made:
            _start(peer, started)
        # output.log/view → the Logger, which writes the .rrd or serves :9876; neither → no recording.
        return build_loop(
            label,
            cfg,
            vehicle_usd=vehicle_usd,
            controller=built["controller"],
            instance=run["instance"],
            estimator=built.get("estimator"),
            guidance=built.get("guidance"),
            # The serial links close with the peers, and the run's instance stays held until the loop has stopped
            # every peer.
            peers=[*started, *links, *([claim] if claim is not None else [])],
            peer_links={name: peer for name, peer in needed.items() if name != KIT},
            peer_inputs={spec.prim: peer for spec, peer in zip(declared, made, strict=True)},
            connections=read_connections(usd),
            prims={roles[role].prim: component for role, component in built.items()},
            rerun=launch.output.log or launch.output.view,
            viewer=launch.output.view,
            debug=launch.output.debug,
            renderer_factory=renderer_factory,
            preroll_timeout=preroll_timeout,
            max_steps=launch.runtime.max_steps,
            # The viewer's Settings tab shows the tested-config receipt, "every input that affects the
            # simulation" per config.receipt, so a recording says what produced it.
            settings=resolved.tested_config.model_dump(mode="json"),
            components=registry,
            record_to_rrd=str(logs / "nexus.rrd"),
        )
    except BaseException:
        # The loop never took the peers over, so they stop here, and the instance is free again.
        for peer in [*started, *links]:
            peer.stop()
        if claim is not None:
            claim.stop()
        raise


def _required(specs: list[ComponentSpec], registry: Registry) -> dict[str, type]:
    """Each peer a declared component's class requires, by the name it requires it by, to the class that answers
    that name.

    Raises:
        ValueError: No class answers a name a class requires; the message names the class and the name.
    """
    required = {}
    for spec in specs:
        for name in getattr(spec.cls, "requires", ()):
            cls = registry.resolve(name)
            if cls is None:
                raise ValueError(
                    f"{spec.prim}: {spec.cls.__name__} requires the peer {name!r}, and no class claims it; map it "
                    "with register or a nexus.registry entry point"
                )
            required[name] = cls
    return dict(sorted(required.items()))


def _serial_links(
    declared: list[PeerSpec], lines: dict[str, dict[str, str]]
) -> tuple[list[SerialLink], dict[str, dict[str, str]]]:
    """One serial link per line a declared peer's prim wires, `lines` by prim as :func:`read_serial_lines` reads
    them, and each peer's ends by its prim: ``ends[prim][line]`` is the path of the end of the peer whose prim wires
    `line`, and ``ends[target][name]`` the path of the other peer's end, by the name of the prim that wires it.

    Checks every line before it makes any link, so a line the build rejects leaks no device.

    Raises:
        ValueError: A line targets a prim that declares no other peer this run starts, or two lines reach one peer
            from prims of one name, which couldn't tell their ends apart; the message names the prim and the line.
    """
    peers = {spec.prim for spec in declared}
    wanted: list[tuple[str, str, str]] = []
    taken: dict[tuple[str, str], str] = {}  # the line that named an end of `target` `name`
    for spec in declared:
        for line, target in sorted(lines.get(spec.prim, {}).items()):
            if target not in peers or target == spec.prim:
                raise ValueError(
                    f"{spec.prim}: nexus:serial:{line} targets {target}, which declares no other peer this run starts"
                )
            name = spec.prim.rsplit("/", 1)[-1]
            if (target, name) in taken:
                raise ValueError(
                    f"{spec.prim}: nexus:serial:{line} and {taken[target, name]} both reach {target} from a prim "
                    f"named {name}, so {target} can't tell their ends apart"
                )
            taken[target, name] = f"{spec.prim}.nexus:serial:{line}"
            wanted.append((spec.prim, line, target))
    links: list[SerialLink] = []
    ends: dict[str, dict[str, str]] = {}
    for prim, line, target in wanted:
        link = SerialLink(f"{prim}.nexus:serial:{line}")
        links.append(link)
        ends.setdefault(prim, {})[line] = link.ends[0]
        ends.setdefault(target, {})[prim.rsplit("/", 1)[-1]] = link.ends[1]
    return links, ends


def _check_clocks(classes: list[type], rtf: float) -> None:
    """Fail a run that can't keep a peer's time, before any peer starts: a free-running peer keeps its own wall
    clock, which agrees with sim time only at a real-time factor of 1.

    Raises:
        RunError: A peer class states no ``clock``, or one outside the three, or runs free at an `rtf` other than 1;
            the message names the peer and the value or the factor.
    """
    for cls in classes:
        clock = getattr(cls, "clock", None)
        if clock not in CLOCKS:
            stated = "no clock" if clock is None else f"the clock {clock!r}"
            raise RunError(
                f"the {cls.name} peer states {stated}; a peer class states how it keeps time in `clock`, one of {CLOCKS}"
            )
        if clock == "free" and rtf != 1:
            raise RunError(
                f"the {cls.name} peer runs free on its own clock, and this run paces at a real-time factor of "
                f"{rtf:g}; pass --rtf 1 to fly it"
            )


def _start(peer, started: list) -> None:
    """Start `peer` and add it to `started`, the peers a failed build stops again.

    Raises:
        RunError: The peer couldn't start, such as PX4 with no tree or no docker daemon; the message names the
            peer and its error, which it chains. A peer's own ``RunError``, the Kit peer's, passes through unchanged.
    """
    try:
        peer.start()
    except RunError:
        raise
    except Exception as exc:
        name = getattr(peer, "name", None) or type(peer).__name__
        raise RunError(f"the {name} peer couldn't start: {exc}") from exc
    started.append(peer)


def _peer(cls: type, kwargs: dict, pin, run: dict, prim: str | None, ends: dict[str, dict[str, str]]):
    """The peer `cls` makes from its schema's attributes, `kwargs`, its checked `pin`, the run's values and, when
    `prim` has an end of a serial line, the keyword ``serial``, its ends by name.

    Raises:
        ValueError: The peer fails its own check of its values; the message names its prim.
    """
    serial = {"serial": ends[prim]} if prim in ends else {}
    try:
        return cls(**kwargs, pin=pin, **run, **serial)
    except ValueError as exc:
        raise ValueError(f"{prim}: {exc}" if prim else str(exc)) from exc


def _component(spec: ComponentSpec, run_instance: int):
    """The component `spec` declares: its class with its schema's attributes, and the run's instance when the
    class needs it.
    """
    return spec.cls(**spec.kwargs, **run_keywords(spec.cls, run_instance))


def _declaration(spec: ComponentSpec | PeerSpec) -> dict:
    """The receipt's record of one declaration: its prim, its schema, its class and its attributes' values."""
    cls = spec.cls
    return {
        "prim": spec.prim,
        "schema": spec.schema,
        "class": f"{cls.__module__}:{cls.__qualname__}",
        "values": {name: _plain(value) for name, value in spec.kwargs.items()},
    }


def _plain(value):
    """`value` as plain data the receipt holds: a number, a string, a bool, or a list of them."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    try:
        return [_plain(v) for v in value]
    except TypeError:
        return str(value)
