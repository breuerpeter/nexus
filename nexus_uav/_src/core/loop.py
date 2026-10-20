"""The fixed-order, deterministic loop.

A tick is an ordered ring of stages, per docs/design/execution.md. Every component states its
per-tick work as a list of ``Stage``: a device stage joins a CUDA graph, a host stage runs between
graph replays. The loop lays the stages out in the canonical order, clock -> [clear -> the command elements
-> the force elements -> step] per physics substep -> state -> sensors -> estimator -> guidance ->
controller -> record, and cuts the ring at its host stages, so each maximal run of device stages becomes
one CUDA graph, the Real Time Factor (RTF) lever, and the host stages run between replays. The physics
step opens a tick, so every row the tick records describes the state at the tick's time. One partition
and one loop serve every arrangement: PX4, whose ``exchange`` host stage blocks on its peer, a Model
Predictive Control (MPC) solver that runs on the host, and a device-native Proportional Integral
Derivative (PID) law whose whole ring is one graph. On a CPU device the same segments run stage by stage,
the bit-exact determinism gate. GPU physics is tolerance-gated, not bit-exact, so capture is an RTF optimization
layered over the same component semantics.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import warp as wp

from nexus_uav._src.diagnostics import diagnostics

from .clock import DeviceClock
from .errors import RunError
from .interfaces import Stage, Tick
from .interrupts import hold_interrupts, stop_signals
from .logging import logger
from .ports import PortMap
from .profiling import LoopProfiler
from .schema import PoseTwist
from .signals import Signal, _declared, wire
from .stages import SUBSTEP_ROLES, TICK_ROLES, Bound, build_ring, partition, plan_line, start_stages

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import newton

    from nexus_uav._src.logging import Logger
    from nexus_uav._src.peers import Peer
    from nexus_uav._src.recording import Recorder

    from .interfaces import Clock, Component, Physics


def _on_cuda() -> bool:
    """Whether the active Warp device captures graphs: a CUDA device."""
    return bool(wp.get_device().is_cuda)


def _instance(component) -> str:
    """A component's instance name, the leaf of its path: its ``name``, else its class's name."""
    return str(component.name or type(component).__name__)


def _peer_name(peer) -> str:
    """A peer's name, which is optional: its ``name``, else its class's name."""
    return str(getattr(peer, "name", None) or type(peer).__name__)


def _sensor_names(sensors: list) -> list[str]:
    """Each sensor's instance name, in order.

    Raises:
        ValueError: Two sensors share a name; the message names the class of each. The builder checks the
            same for the sensors a vehicle declares, and names their prims.
    """
    first: dict[str, object] = {}
    for sensor in sensors:
        name = _instance(sensor)
        if name in first:
            declared = [type(s).__name__ for s in (first[name], sensor)]
            raise ValueError(
                f"sensors {declared[0]} and {declared[1]} share the name {name!r}: a recorded row's path "
                "ends in its sensor's name, so name one apart"
            )
        first[name] = sensor
    return [_instance(sensor) for sensor in sensors]


class Loop:
    """Fixed-order, deterministic per-tick driver over typed component boundaries.

    Runs the ring of stages every component states, actuate and step -> sensors -> estimator -> guidance ->
    controller -> record, per docs/design/execution.md. Before any stage runs, it wires the signals the stages
    declare: each input meets the one output of its name, and a pair that disagrees on the type or
    the shape stops the run. On a CUDA device each maximal run of device stages
    replays as one CUDA graph and the host stages run between replays; on a CPU device the same
    segments run stage by stage. The partition is component-agnostic: it reads each stage's stated
    kind, never a component's type.

    Attributes:
        sensors: The configured sensors as a ``list``, because the constructor copies the
            ``sensors`` iterable, each reading the live ``newton.State`` per tick.
        run_stats: Set when a run finishes: ``{"control_steps", "rtf",
            "full_rtf"}``: total control steps, the steady-window real-time factor,
            sim-time advanced / wall-time, post-warmup and compilation-free, and
            the full RTF including warmup/compile. Missing until ``run()`` has looped.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        physics: Physics,
        sensors: Iterable[Component],
        controller: Component,
        estimator: Component | None = None,
        guidance: Component | None = None,
        commands: Iterable[Component] = (),
        forces: Iterable[Component] = (),
        actuator: Component | None = None,
        renderer: object | None = None,
        links: Mapping[str, object] | None = None,
        peers: Iterable[Peer] = (),
        peer_inputs: Iterable[Component] = (),
        connections: Iterable[tuple[Component, str, Component]] | None = None,
        logger: Logger | None = None,
        recorder: Recorder | None = None,
        on_tick: Callable[[newton.State, float, int], None] | None = None,
        preroll_timeout: float = 2.0,
        exchange_timeout: float = 2.0,
        max_steps: int | None = None,
        physics_substeps: int = 1,
        settings: dict | None = None,
    ):
        """Wire the components and run options into a single tick driver.

        Args:
            clock: Simulation clock. ``advance()`` advances and returns the sim
                time ``t``; ``dt`` is the control timestep; ``throttle()`` paces the
                loop to wall-time, a no-op unless real-time throttling is on.
            physics: Physics component. ``reset()`` builds and settles the vehicle at
                the North East Down (NED) origin and returns the initial state; its ``clear``
                and ``step`` stages zero the shared ``body_f`` and integrate: collide + solver step
                + double-buffer.
            sensors: Iterable of sensors, each a device stage that writes its sample to a signal of its
                own type, or a host stage for a sensor whose work leaves the process. Stored as a list.
            controller: The controller: its stages write the controls the command elements read. One with
                a peer reads ``attached`` false until the peer dials in, which holds the sim clock, as any
                component that waits on a peer does.
            estimator: Optional estimator, for a guidance or a controller that reads the vehicle's state:
                its stages run each tick after the sensors' and before the guidance's, and write the
                estimate, the base body's pose and twist, which the guidance and the controller read in
                place of the physics state. A run whose guidance or controller reads the estimate fails
                without one. The loop reads the attribute when it builds the ring, at the first step.
                ``None`` for PX4, whose estimator runs in its peer.
            guidance: Optional guidance, for a controller that reads a setpoint: its stages run each
                tick after the estimator's and before the controller's, and read the vehicle's
                position from the estimate. It holds no controller: its
                stage writes the setpoint signal the controller reads, so the controller reads a new
                setpoint on the tick the guidance writes it. Its stage marks the tick done when its
                mission is over, which ends the run. A flight can also set the ``guidance`` attribute
                before the first step. ``None`` for PX4, which flies its own missions.
            commands: The command elements: each turns the controls the controller writes into Newton's
                control inputs, the rotors' speed targets and feedforward, in its device stages.
            forces: The force elements: each adds its body wrenches to the shared ``state.body_f``
                from the current state, the propellers' thrust and drag, in its device stages.
            actuator: The old actuator stage: one device stage that writes the shared body forces from
                the controls the controller writes. No shipped component fills it; ``None`` for a run on
                command elements and force elements.
            renderer: Optional render-lifecycle object, for example the Kit render peer's
                :class:`~nexus_uav._src.rendering.KitRenderer`: the loop calls only
                ``on_physics_ready()``/``close()``; the host-rate RTX camera *sensors* drive
                rendering itself in a host stage.
            links: The link of each peer the run holds, by the peer's name, such as the Kit render peer's
                :class:`~nexus_uav._src.rendering.KitRenderer` under ``kit``: the loop hands a component
                the links of the peers its ``requires`` names when it connects it, and fails a component
                that requires a peer with no link here. ``None`` holds no link.
            peers: The peers the build started for this run, for example the PX4 Software In The
                Loop (SITL) container. The loop stops each when the run ends, whether it ran out,
                stopped early or never stepped; it starts none, since a peer boots while the build
                goes on.
            peer_inputs: The loop end of each declared peer's wired inputs, a
                :class:`~nexus_uav._src.core.peer_inputs.PeerInputs` per peer the vehicle plugs a writer into:
                its one host stage runs after every writer of the tick, in the ``link`` role, and hands the peer
                each value. The builder makes them from the vehicle's connections; empty for a run that wires
                none.
            connections: Quadruples of a reader, an input's name, the writer it reads that input from and
                the name of the writer's output, which the build resolves from the vehicle's
                ``nexus:inputs:`` relationships: where more than one component writes a signal, a reader
                takes the one its connection names, and a reader of a body's or joint's prim takes that
                output of the plant. An input of the vehicle's pose type that no component writes and no
                connection names takes the plant's base body. ``None`` connects nothing.
            logger: Optional :class:`~nexus_uav._src.logging.Logger`: the recording
                sink + shared log calls. ``None`` ⇒ no recording, for max speed. When present, the
                loop hands each component a view scoped to its path, and writes the Recorder's
                histories into it, a block at a time.
            recorder: The run's :class:`~nexus_uav._src.recording.Recorder`, which keeps every row of
                each device signal a component writes, the plant's outputs among them. The loop reads the
                attribute at the first step, so a host can set it after construction; ``None`` records nothing.
            on_tick: Optional post-step observer called as ``on_tick(state, t, steps)``
                after recording: an output-only hook for demos and tests to read the live state,
                for example to capture a trajectory.
            preroll_timeout: Seconds to wait for a peer to dial in, and again for its first answer,
                before raising ``RunError``.
            exchange_timeout: Per-tick timeout in seconds for the blocking
                ``controller.exchange`` in the steady loop.
            max_steps: Bound on control steps for the steady loop. ``None`` runs until
                the controller ends the run, the PX4 default, or a guidance's mission is over; a
                finite run, test/demo/episode, sets it.
            physics_substeps: Physics steps per control exchange, zero-order-hold over
                finer physics, cf. multi-rate. ``1`` is lockstep, the PX4 default;
                ``>1`` lets a policy control at a coarse rate over finer integration.
            settings: The run's effective configuration as plain data, what a caller reads
                back to know what flew and what the viewer's Settings tab shows: the build
                passes the tested-config receipt. ``None`` records nothing.
        """
        self.clock = clock
        self.physics = physics
        self.commands = list(commands)
        self.forces = list(forces)
        self.actuator = actuator
        self.sensors = list(sensors)
        self.controller = controller
        self.estimator = estimator  # the loop reads it when it builds the ring, at the first step
        self.guidance = guidance  # the loop reads it when it builds the ring, at the first step
        self.renderer = renderer
        self.links = dict(links or {})
        self.peers = list(peers)
        self.peer_inputs = list(peer_inputs)
        self.connections = list(connections or ())
        self.run_stats: dict = {}  # set when a run finishes; empty until then
        self._connected: list = []  # the components the run connected, which the teardown closes
        # The single logging switch: a `Logger`, the recording sink + the shared log calls, or None.
        # None ⇒ no recording → max benchmark/CI speed.
        self.logger = logger
        # Each component's path under the sim's root, resolved once, here, where the loop wires it:
        # its role folder, then its instance name. A recorded row's path starts with it, a signal's
        # history sits under it, so two sensors that share one fail the build.
        wired = [
            (physics, "vehicle"),
            *((c, f"vehicle/commands/{_instance(c)}") for c in self.commands),
            *((f, f"vehicle/forces/{_instance(f)}") for f in self.forces),
            *([(actuator, f"vehicle/actuators/{_instance(actuator)}")] if actuator is not None else []),
            *([(estimator, f"vehicle/estimators/{_instance(estimator)}")] if estimator is not None else []),
            (controller, f"vehicle/controllers/{_instance(controller)}"),
            *(
                (s, f"vehicle/sensors/{name}")
                for s, name in zip(self.sensors, _sensor_names(self.sensors), strict=True)
            ),
        ]
        self._paths = {id(c): path for c, path in wired}
        # The loop hands each component a view of the one Logger scoped to the component's path, so no
        # call site names an entity; None when off, the single switch. A component logs a value with no
        # fixed width live, in its own code: the guidance its markers at a mission event, an MPC
        # controller its horizon when it refreshes, a camera its frames; the guidance via `add_loggable`
        # when the loop builds its ring, since a flight can hand it over after construction. The scene
        # and every fixed-width value come from the Recorder's histories, which the loop writes a block
        # at a time. Physics is a fixed part, not a component: it logs nothing itself.
        self._loggables = [c for c, _ in wired if c is not physics]
        for c, path in wired:
            if c is not physics:
                c.set_logger(self._scoped(path))
        # Live-RTF anchors, wall and sim: a rolling window the width of the ~1 s emission interval.
        # Distinct from _last_log_t, the sim-time scene decimation, and from run_stats' post-hoc RTF.
        self._rtf_wall_prev: float | None = None
        self._rtf_sim_prev = 0.0
        # The peers that keep their own wall clock, which only a run paced at 1 keeps up with: the loop warns
        # once when its live real-time factor falls below 0.95, since pacing can't stop it falling behind.
        self._free_peers = [_peer_name(peer) for peer in self.peers if getattr(peer, "clock", None) == "free"]
        self._pace_warned = False
        # The Recorder records everything itself: every device signal a component writes, the plant's
        # outputs among them, which the loop hands it once it wires the ring. Its taps are device launches in the record
        # stage, so recording never adds a host stage. The loop reads the attribute at the first step, so
        # `Sim` can hand a self-assembled loop its Recorder after construction.
        self.recorder = recorder
        self._written = 0  # rows of every history the Logger has written
        # Every signal a component writes, device and host, by its path: the writer's path then the signal's
        # name, as its history's key. The loop fills it once it wires the ring, and a script reads a live value
        # through it.
        self.signals: dict[str, Signal] = {}
        # Hosting hooks; Sim drives these. on_tick is the per-tick observer,
        # post-step, fired as on_tick(state, t, steps) after recording: a demo and test hook to read the
        # live state, output-only. _stop is the cooperative teardown the host sets to end the steady loop.
        self.on_tick = on_tick
        self._stop = False
        # The tick generator backing run()/step(), lazily created on the
        # first step(), None between runs. run() exhausts it; Sim.step() advances it one tick.
        self._ticks_iter = None
        self._stepped = False  # a first step() ran, or close() tore down a run that never stepped
        self.preroll_timeout = preroll_timeout
        self.exchange_timeout = exchange_timeout
        # Bound the steady loop; None = run until the controller ends it, the PX4 default.
        # A controller with no peer, for example the trained policy, never returns None, so a
        # finite run, test/demo/episode, sets this.
        self.max_steps = max_steps
        # Physics steps per control exchange, zero-order-hold control over finer physics,
        # cf. multi-rate: 1 = lockstep, the PX4 default. >1 lets a policy control at a
        # coarse rate, for example 50 Hz, over finer physics integration, matching its training.
        self.physics_substeps = int(physics_substeps)
        self.settings = settings
        # The tick's sim time on the device, which the ring's first stage advances and the record stage stamps.
        self._device_clock = DeviceClock(self.clock.dt)

    # -- component-owned logging --------------------------------------------------
    def _scoped(self, path: str):
        """The Logger's view for the component at ``path`` under the sim's root; None when off."""
        return self.logger.scoped(path) if self.logger is not None else None

    def add_loggable(self, component, path: str) -> None:
        """Register a logging component handed over after construction, the guidance, and hand it the
        Logger scoped to ``path``, the component's path under the sim's root: ``_logger``, None when off,
        the gate for its event-driven logging. Idempotent.
        """
        if component not in self._loggables:
            self._loggables.append(component)
        self._paths[id(component)] = path
        component.set_logger(self._scoped(path))

    # -- recording, the read-side twin of logging -----------------------------------
    def _tap_signals(self, ring: list[Bound]) -> None:
        """Key every signal a component's stage writes by its path, the writer's path then the signal's name, and
        hand the Recorder each device one, so it registers one history each. After the loop wires the ring and
        before the start, so every history's buffer exists before any capture. The loop's own clock has no
        path, so its ``time`` keeps no history: every row carries the tick's time.
        """
        seen: set[int] = set()
        for bound in ring:
            path = self._paths.get(id(bound.component))
            if path is None or id(bound.stage) in seen:  # the ring repeats the stages of each physics substep
                continue
            seen.add(id(bound.stage))
            for signal in bound.stage.writes:
                self.signals[f"{path}/{signal.name}"] = signal
                if signal.device and self.recorder is not None:
                    self.recorder.tap(path, signal, source=type(bound.component).__name__)

    def _default_links(self, ring: list[Bound]) -> list[tuple]:
        """A link for each input of the vehicle's pose type that no component writes and no connection names:
        the plant's base body output, ``body/<label>``. So a component that reads the vehicle's pose, the
        passthrough estimator or PX4's ground truth, needs no connection on its prim, and a run that hands its
        estimator over with no prim, as the examples do, reads the base body.
        """
        writes, reads = _declared(ring)
        named = {(id(reader), name) for reader, name, *_ in self.connections}
        return [
            (bound.component, name, self.physics, f"body/{self.physics.base_body}")
            for name, readers in reads.items()
            if name not in writes
            for bound, signal in readers
            if signal.type is PoseTwist and (id(bound.component), name) not in named
        ]

    def _record_tick(self) -> None:
        """The record stage: the Recorder snapshots every tapped signal into its history, stamped with the
        device clock's time. A device stage, no D2H, so it replays with its graph. A no-op with no Recorder,
        leaving that graph the same byte for byte.
        """
        if self.recorder is None:
            return
        self.recorder.record(self._device_clock.time.buffer)

    def _commit(self) -> None:
        """Count the row the record stage wrote this tick. When that fills the staging buffers, the
        Recorder drains them onto its host blocks, and the Logger writes the drained block.
        """
        if self.recorder is None:
            return
        block = self.recorder.commit()
        if block is not None and self.logger is not None:
            self.logger.write(self.recorder.histories, *block)
            self._written = block[1]

    def _begin_log(self, t) -> None:
        """Set the shared ``time`` timeline once at the start of each tick, right after the clock advances
        and BEFORE ``exchange``, so every component's overlay this tick, the controller's horizon in
        ``exchange``, the guidance's markers in its stage, physics' scene, lands at the same timestamp,
        and no component touches ``set_time`` itself. A no-op when not recording.
        """
        if self.logger is not None:
            self.logger.set_time(float(t.sim_time))

    def _log_rtf(self, st: float) -> None:
        """Take the live real-time factor ~1 Hz wall-clock: sim seconds over wall seconds since the
        last reading, a rolling window exactly one interval wide. Only completed ticks reach
        this call, so a stalled exchange shows as a *low* next reading, not a live countdown. The reading
        goes to the recording when there is one, and a run with a free-running peer warns once when it
        falls below 0.95, naming the peer: sim time then trails that peer's wall clock.
        """
        if self.logger is None and (not self._free_peers or self._pace_warned):
            return
        now = time.monotonic()
        if self._rtf_wall_prev is None:
            self._rtf_wall_prev, self._rtf_sim_prev = now, st
            return
        wall = now - self._rtf_wall_prev
        if wall < 1.0:
            return
        rtf = (st - self._rtf_sim_prev) / wall
        if self.logger is not None:
            self.logger.log_rtf(rtf)
        if self._free_peers and rtf < 0.95 and not self._pace_warned:
            self._pace_warned = True
            logger.warning(
                f"the live real-time factor fell to {rtf:.2f}, and {' and '.join(self._free_peers)} runs free on "
                "its own clock: sim time now trails it"
            )
        self._rtf_wall_prev, self._rtf_sim_prev = now, st

    def _close_logs(self) -> None:
        """Teardown: write the rows still staged, the last block of every history, the run's profile,
        *then* close the sink, so every row lands before the ``.rrd`` finalizes.
        """
        if self.logger is None:
            return
        if self.recorder is not None:
            self.logger.write(self.recorder.histories, self._written)  # the last block: the staged rows
        # The final steady RTF from run_stats, whose loop's finally ran first, stamped at end-of-run
        # sim time; a run faster than the ~1 s live cadence would otherwise never show a reading.
        stats = self.run_stats  # empty until a run has looped
        if stats.get("rtf"):
            self.logger.set_time(self.clock.now().sim_time)
            self.logger.log_rtf(float(stats["rtf"]))
        if stats:
            self.logger.log_profile(stats)  # steps + RTF + the loop profiler's breakdown, for the Profile tab
        self.logger.close()

    def _record_rtf(self, steps: int, t0: float, t_warm: float | None, steps_warm: int) -> None:
        """Stamp ``run_stats`` and log the live RTF, the sim speed, sim-time advanced / wall-time,
        shared by the eager and captured loops. The steady window, post-``warmup_steps``, excludes lazy
        kernel compilation + initial settle; the full RTF, including warmup/compile, lands alongside.
        For the decoupled PX4 flight this is the speed *with* PX4 in the lockstep loop.
        """
        now = time.monotonic()
        dt = self.clock.dt
        full_rtf = round(steps * dt / (now - t0), 3) if now > t0 else 0.0
        if t_warm is not None and now > t_warm:
            rtf = round((steps - steps_warm) * dt / (now - t_warm), 3)
            window = steps - steps_warm
        else:  # never reached the warmup window
            rtf, window = full_rtf, steps
        self.run_stats = {"control_steps": steps, "rtf": rtf, "full_rtf": full_rtf}
        if steps:
            logger.info(
                f"sim RTF {rtf:.2f}x (steady, over {window} steps); full {full_rtf:.2f}x incl. warmup/compile",
                extra={"console_only": True},  # in-viewer this lives in the RTF pane + Profile tab
            )

    def _make_profiler(self, label: str) -> LoopProfiler:
        """Always-on loop profiler, integer-ns marks, noise at 250 Hz. The shared ``--profile``
        flag, ``diagnostics.profile``, adds periodic reports; ``--trace <path>`` buffers a
        Chrome/Perfetto trace exported at the end of the run.
        """
        deep = diagnostics.profile
        prof = LoopProfiler(
            label,
            dt=self.clock.dt,
            report_every_s=5.0 if deep else 0.0,
            trace_path=diagnostics.trace,
        )
        if self.renderer is not None and hasattr(self.renderer, "set_profiler"):
            self.renderer.set_profiler(prof)  # detail spans inside the renderer: render/grab
        return prof

    def stop(self) -> None:
        """Cooperatively end the steady loop; checked at the top of each tick.

        ``Sim`` sets this to tear a run down: its lifecycle verb. Takes effect within one ``exchange_timeout``, since the loop might be
        waiting in ``controller.exchange``.
        """
        self._stop = True

    def _ticks(self):
        """The single execution path, as a generator that yields once per control tick, driving both
        ``run()``, which exhausts it, and ``Sim.step()``, which advances it one tick.

        Resets physics, which builds and settles the vehicle at the NED origin, builds the ring of
        stages and partitions it, runs the start over the settled state at sim time zero, which records the
        first row, captures each device segment on CUDA, waits for a peer to dial in, then loops. Every
        controller advances one tick per ``next()``, but the first, which runs a peer's preroll ticks too.
        Teardown, renderer,
        controller and logs, runs in the ``finally``, whether the generator runs out, on run to
        completion, raises, on a peer lost, or closes early, via :meth:`close` on ``stop()`` mid-step.
        The live RTF lands in ``run_stats``. For the span of the run SIGTERM and SIGHUP end it as
        Ctrl-C does, so each reaches this teardown.
        """
        with stop_signals():
            yield from self._run()

    def _run(self):
        try:
            # Inside the try from the first line: the renderer's peer started at build, so a reset or a
            # peer start that fails must still reach the `finally`, which removes its container, and
            # from the connect on the controller owns a live peer too, the PX4 container it launches.
            state = self.physics.reset()  # build + settle the vehicle at the NED origin
            # Renderer warm-up hook, before the peer's lockstep starts: the Kit peer connects and warms
            # its stage here, which takes seconds and would stall the lockstep.
            if self.renderer is not None and hasattr(self.renderer, "on_physics_ready"):
                self.renderer.on_physics_ready()
            record = Stage("record", "device", lambda tick: self._record_tick())
            if self.estimator is not None and id(self.estimator) not in self._paths:
                # Handed over after construction, as a flight can do: the loop resolves its path here.
                self._paths[id(self.estimator)] = f"vehicle/estimators/{_instance(self.estimator)}"
            if self.guidance is not None:
                # Here, since a flight can hand the guidance over after construction.
                self.add_loggable(self.guidance, "guidance")
            ring = build_ring(
                clock=self._device_clock,
                physics=self.physics,
                roles=self.roles(),
                record=record,
                substeps=self.physics_substeps,
            )
            self._check_estimator(ring)
            # Every signal a stage declares gets its buffer before any stage runs.
            wire(ring, [*self.connections, *self._default_links(ring)])
            self._tap_signals(ring)  # every written signal by its path, and one history per device one
            self._check_guidance(ring)
            segments = partition(ring)
            captured = _on_cuda()
            logger.info(plan_line(segments, captured))
            tick = Tick(
                state=state,
                t=self.clock.now(),
                dt=self.clock.dt / self.physics_substeps,
                timeout=self.exchange_timeout,  # the start's host stages wait on a peer no longer than a tick's
                base=getattr(self.physics, "base_index", 0),  # a physics that finds none has body 0 as base
                control=getattr(self.physics, "control", None),  # the control inputs a command element writes
            )
            self._device_clock.set(tick.t)
            # A component attached from the start has no peer: it connects first, which reserves its
            # device buffers before the start. One that waits on a peer connects after the capture,
            # so every kernel the loop launches has loaded before the peer is up: from the moment PX4
            # dials in, it logs a ``poll timeout`` error for each second of wall time that brings no
            # message. The component's own ``attached`` says which.
            components = self.components()
            self._owners = {id(b.stage): b.component for b in ring}
            waiting = [c for c in components if not c.attached]
            for c in components:
                if not any(c is w for w in waiting):
                    c.connect(self._links_for(c))
                    self._connected.append(c)
            # The start: the ring without the clock and the physics substeps, once, over the settled state at
            # sim time zero, so the first tick's step applies the controls it computes and its row is the first.
            # The log opens at the settled state's time, so the markers a guidance logs here sit on the timeline.
            self._begin_log(tick.t)
            for st in start_stages(ring, waiting):
                st.run(tick)
            self._commit()
            graphs = self._capture(segments, tick) if captured else None
            for c in waiting:
                c.connect(self._links_for(c))  # bind the peer's port and start listening
                self._connected.append(c)
            self._attach(ring, tick, waiting)
            yield from self._loop(segments, graphs, tick, waiting)
        finally:
            try:
                if self.renderer is not None and hasattr(self.renderer, "close"):
                    self.renderer.close()  # take the last frame, then stop the Kit peer
            finally:
                self._teardown()  # whatever the renderer's close did: a stop that lands in it, or an error

    def _teardown(self) -> None:
        """The recording first: write the last block of every history and close the file, with a stop
        that lands meanwhile held until the file closes, then close the controller and stop the
        peers. The recording is what a failed run is *for*, so a controller or a peer that raises on
        the way out never costs it, and a second Ctrl-C never cuts it.
        """
        try:
            with hold_interrupts():
                self._close_logs()
        finally:
            try:
                self._close_components()
            finally:
                self._stop_peers()

    def _peer_report(self) -> str:
        """What each peer the run started says about itself, as a clause to end an error with: the files it leaves,
        such as its console log, each after the peer's name, then its own error lines, one a line after the peer's
        name; empty for a run whose peers leave neither. A peer's ``name``, its ``artifacts`` and its ``errors`` are
        optional, so a peer with no name goes by its class's, and an artifact the peer didn't write, ``None``, names
        no file.
        """
        files = [
            f"{_peer_name(peer)}'s {key} {path}"
            for peer in self.peers
            for key, path in getattr(peer, "artifacts", dict)().items()
            if path is not None
        ]
        said = [f"\n{_peer_name(peer)}: {line}" for peer in self.peers for line in getattr(peer, "errors", list)()]
        return ("; see " + ", ".join(files) if files else "") + "".join(said)

    def _check_peers_alive(self, names: str) -> None:
        """Fail the setup at once when a peer the run started has exited, rather than wait out ``preroll_timeout``
        for a component that waits on it.

        Raises:
            RunError: A peer reads dead; the message names it and ends with the report of every peer.
        """
        dead = [_peer_name(peer) for peer in self.peers if not peer.alive()]
        if dead:
            raise RunError(f"{' and '.join(dead)} exited while {names} waited on it{self._peer_report()}")

    def _links_for(self, component) -> dict:
        """The link of each peer `component` requires, by the peer's name, which its ``connect`` takes.

        Raises:
            ValueError: The component requires a peer this run holds no link to; the message names both.
        """
        missing = [name for name in component.requires if name not in self.links]
        if missing:
            raise ValueError(
                f"{_instance(component)} requires the peer {' and '.join(missing)}, and this run holds no link to it"
            )
        return {name: self.links[name] for name in component.requires}

    @property
    def ports(self) -> PortMap:
        """The run's port map: each link that leaves the run, by name, to the address a script opens its client
        on, as each component and then each peer states it, so a fake peer keeps out a link its component names.
        The run owns every address: the build hands each its share.
        """
        stated = [*self.components(), *self.peers]
        return PortMap.merge(source.ports() for source in stated if callable(getattr(source, "ports", None)))

    def roles(self) -> dict[str, list]:
        """Each role's components, by role, each list in the order its components run."""
        one = lambda c: [c] if c is not None else []  # noqa: E731
        return {
            "sensor": list(self.sensors),
            "estimator": one(self.estimator),
            "guidance": one(self.guidance),
            "controller": one(self.controller),
            "link": list(self.peer_inputs),
            "command": list(self.commands),
            "force": list(self.forces),
            "actuator": one(self.actuator),
        }

    def components(self) -> list:
        """Every component of the run, in ring order: the sensors, the estimator, the guidance, the controller,
        the loop end of each declared peer's inputs, the command elements, the force elements and the old
        actuator stage.
        """
        roles = self.roles()
        return [c for role in (*TICK_ROLES, *SUBSTEP_ROLES) for c in roles[role]]

    def _close_components(self) -> None:
        """Close every component the run connected, in reverse order, each even when an earlier one raises:
        the first error propagates after the last close.
        """
        components = list(reversed(self._connected))
        self._connected = []

        def close(i: int) -> None:
            if i == len(components):
                return
            try:
                components[i].close()
            finally:
                close(i + 1)

        close(0)

    def _controller_name(self) -> str:
        """The controller's class, so a run's end names the peer that left it: ``Px4MavlinkController``."""
        return type(self.controller).__name__

    def _check_estimator(self, ring) -> None:
        """Check that the run has an estimator when a guidance or a controller reads the estimate: a run takes
        an estimator for a guidance or a controller that reads the vehicle's state, and the estimate is the
        estimator's, whatever other component writes a signal of its type.

        Raises:
            ValueError: A stage reads the estimate, and the run has no estimator, as on a run built with none.
                The message names each reader's class and the estimator.
        """
        if any(b.role == "estimator" for b in ring):
            return
        readers = list(
            dict.fromkeys(type(b.component).__name__ for b in ring for s in b.stage.reads if s.name == "estimate")
        )
        if readers:
            raise ValueError(
                f"{' and '.join(readers)} read the estimate, and no estimator writes it: a run whose guidance or "
                "controller reads the vehicle's state takes an estimator, and PX4 estimates in its own peer"
            )

    def _check_guidance(self, ring) -> None:
        """Check that the controller reads a signal the guidance writes, when the run has a guidance: a
        run takes a guidance only for a controller that reads a setpoint.

        Raises:
            TypeError: The controller reads nothing the guidance writes, as on a PX4 run, whose controller
                reads no setpoint, whatever other component reads it. The message names the controller.
        """
        if self.guidance is None:
            return
        written = {s.name for b in ring if b.role == "guidance" for s in b.stage.writes}
        if not written & {s.name for b in ring if b.role == "controller" for s in b.stage.reads}:
            raise TypeError(
                f"{self._controller_name()} reads no setpoint, so this run takes no guidance: a guidance "
                "writes the setpoint a controller reads, and PX4 flies its own missions"
            )

    def _stop_peers(self) -> None:
        """Stop each peer the build started. A peer whose stop fails, a docker daemon that went
        away, must not cost the rest of the teardown: the recording still has to flush.
        """
        for peer in self.peers:
            try:
                peer.stop()
            except Exception as exc:
                logger.warning(f"stopping a peer failed ({exc})")

    def _attach(self, ring, tick, waiting: list) -> None:
        """Hold the sim clock until every component in ``waiting`` has attached, running each one's host stages
        meanwhile, its way to its peer: time spent waiting would start a peer's clock late, and PX4 times its
        boot checks from its first stamp.

        Raises:
            RunError: A component never attached within ``preroll_timeout``, or a peer exited before it did.
        """
        if not waiting:
            return
        names = " and ".join(_instance(c) for c in waiting)
        logger.info(f"Waiting for {names} to attach...")
        tick.timeout = 0.05
        polls = [b.stage for b in ring if b.stage.kind == "host" and any(b.component is c for c in waiting)]
        deadline = time.monotonic() + self.preroll_timeout
        while not all(c.attached for c in waiting):
            if time.monotonic() > deadline:
                raise RunError(f"{names} did not attach within {self.preroll_timeout}s{self._peer_report()}")
            self._check_peers_alive(names)
            for st in polls:
                st.run(tick)
        logger.info(f"{names} attached")

    def _open_tick(self, tick) -> bool:
        """Advance the clock and open the tick's log, before any host stage, so the Model Predictive
        Control (MPC) example can log its horizon inside ``exchange``. On a graph the loop calls this
        once it has launched the tick's first replay, so the host work overlaps the device work.
        """
        tick.t = self.clock.advance()
        self._begin_log(tick.t)
        return True

    def _capture(self, segments, tick) -> list:
        """Capture each device segment into its own CUDA graph, serially, before the first tick. Any
        other stream operation during a capture kills it, so the start completes first, and nothing else
        touches the device until the loop replays.
        """
        wp.synchronize()  # complete the start before a capture opens
        graphs = []
        for seg in segments:
            if seg.kind != "device":
                graphs.append(None)
                continue
            with wp.ScopedCapture() as cap:
                for st in seg.stages:
                    st.run(tick)
            graphs.append(cap.graph)
        return graphs

    def _loop(self, segments, graphs, tick, waiting: list):
        """The steady loop as a generator: one control tick per ``yield``, which step() drives.
        Each tick advances the clock, opens its log, then runs the segments in ring order: a device
        segment replays its graph, or runs stage by stage without one, and a host stage runs on the
        host between replays. A host stage that reports its peer gone fails the run, and a tick a stage
        marked done ends it once the tick completes. While the peers of ``waiting`` boot, the preroll, a tick
        whose host stage gets no answer still steps, records its row and moves on, with no ``yield``, until
        the first tick that every host stage answers, within ``preroll_timeout``, or until a peer exits. The ``finally``
        stamps the RTF, so it runs whether the generator runs out or closes early on stop.

        Raises:
            RunError: A peer gave no first answer within ``preroll_timeout``, exited before it, or stopped
                answering after it.
        """
        count = 0
        t0 = time.monotonic()  # for the end-to-end RTF, sim-time advanced / wall-time, of the live run
        # Steady-state window: the RTF measurement starts here so it excludes lazy kernel compilation and
        # the initial settle.
        warmup_steps = 250
        t_warm = None
        steps_warm = 0
        prerolling = bool(waiting)
        tick.timeout = 0.05 if prerolling else self.exchange_timeout
        deadline = time.monotonic() + self.preroll_timeout
        replay = wp.capture_launch if graphs is not None else None
        prof = self._make_profiler("graph" if graphs is not None else "eager")
        try:
            while not self._stop and not tick.done and (self.max_steps is None or count < self.max_steps):
                count += 1
                prof.tick_begin()
                if count == warmup_steps:
                    t_warm, steps_warm = time.monotonic(), count
                opened = False
                answered = True
                for i, seg in enumerate(segments):
                    if seg.kind == "host":
                        if not opened:
                            opened = self._open_tick(tick)
                        st = seg.stages[0]
                        if st.run(tick) is False:
                            answered = False
                            if not prerolling:  # the peer stopped answering: the run lost it
                                count -= 1  # the lost tick didn't advance the sim
                                owner = _instance(self._owners.get(id(st), self.controller))
                                raise RunError(
                                    f"{owner} lost its peer: no answer in its {st.name} stage within "
                                    f"{tick.timeout}s{self._peer_report()}"
                                )
                        prof.mark(st.name)
                    elif replay is not None:
                        prof.gpu_begin()
                        replay(graphs[i])
                        prof.gpu_end()
                        if not opened:  # the clock advances while the first graph runs on the device
                            opened = self._open_tick(tick)
                        prof.mark("graph")
                    else:
                        if not opened:
                            opened = self._open_tick(tick)
                        for st in seg.stages:
                            st.run(tick)
                        prof.mark("stages")
                self._commit()  # the tick's row landed in every history; a full staging buffer drains
                self._log_rtf(float(tick.t.sim_time))  # the live RTF, about once a second of wall time
                if self.on_tick is not None:
                    self.on_tick(tick.state, tick.t, count)  # post-step observer, a demo and test hook
                self.clock.throttle()  # no-op unless rtf>0, the interactive real-time throttle
                prof.mark("log")
                prof.tick_end()
                if prerolling:
                    names = " and ".join(_instance(c) for c in waiting)
                    if not answered:
                        if time.monotonic() > deadline:
                            raise RunError(
                                f"{names} did not answer within {self.preroll_timeout}s{self._peer_report()}"
                            )
                        self._check_peers_alive(names)
                        continue  # a preroll tick: the peer is still booting
                    prerolling = False
                    tick.timeout = self.exchange_timeout
                    logger.info(f"{names}: lockstep established")
                yield  # one control tick complete: step() returns here
        finally:
            self._record_rtf(count, t0, t_warm, steps_warm)
            prof.close()
            self.run_stats["profile"] = prof.stats()

    def step(self) -> bool:
        """Advance the run one control tick, returning ``True`` if a tick ran or ``False`` when the run has
        ended: mission complete / ``max_steps`` / ``stop()``. Setup, reset, connect and
        graph capture, runs on the first call; teardown runs automatically when the run ends.
        ``Sim.step`` drives this for every controller, PX4 included. Deterministic:
        no wall-clock.

        Raises:
            RunError: The setup failed, for example a peer that never answered, or the run lost a peer;
                teardown has run.
        """
        if self._ticks_iter is None:
            self._stepped = True
            self._ticks_iter = self._ticks()
        try:
            next(self._ticks_iter)
            return True
        except StopIteration:  # the generator ran its finally, the teardown, on the way out
            self._ticks_iter = None
            return False

    def run(self) -> None:
        """Drive the full run to completion: exhaust the tick generator, setup, loop and teardown.

        Side effects only: advances the simulation and populates ``run_stats``. The same as stepping
        until :meth:`step` returns ``False``; every non-stepped run uses it: ``nexus run``, the
        examples' ``sim.run()``.

        Raises:
            RunError: The setup failed, for example a peer that never answered, or the run lost a peer;
                teardown has run.
        """
        while self.step():
            pass

    def close(self) -> None:
        """Tear down the run, ``Sim.stop()``. A partially stepped run closes its tick generator, running
        its ``finally``, RTF stamp + renderer/logs/controller/peers teardown. A run that never stepped
        closes the renderer, closes the logs, then stops the peers, which started at build. Idempotent:
        a no-op once the run has finished, ``run()`` completed / ``step()`` returned ``False``, or closed.
        """
        if self._ticks_iter is not None:
            self._ticks_iter.close()  # GeneratorExit → the loop's finally, RTF, + _ticks' finally, close
            self._ticks_iter = None
        elif not self._stepped:
            self._stepped = True  # torn down: a later close() is a no-op
            try:
                if self.renderer is not None and hasattr(self.renderer, "close"):
                    self.renderer.close()
            finally:
                try:
                    with hold_interrupts():
                        self._close_logs()
                finally:
                    self._stop_peers()
