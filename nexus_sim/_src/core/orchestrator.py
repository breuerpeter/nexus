"""The fixed-order, deterministic orchestrator.

A tick is an ordered ring of stages, per docs/design/execution.md. Every component states its
per-tick work as a list of ``Stage``: a device stage joins a CUDA graph, a host stage runs between
graph replays. The loop lays the stages out in the canonical order, sensors -> estimator -> guidance ->
controller -> [clear -> the command elements -> the force elements -> step] per physics substep -> record,
cuts the ring at its host stages and
rotates it to start after the last cut, so each maximal run of device stages becomes one CUDA
graph, the Real Time Factor (RTF) lever, and the host stages run between
replays. One partition and one loop serve every arrangement: PX4, whose ``exchange`` host stage
blocks on its peer, a Model Predictive Control (MPC) solver that runs on the host, and a
device-native Proportional Integral Derivative (PID) law whose whole ring is one graph. On a CPU device the same segments run stage by stage, the bit-exact
determinism gate. GPU physics is tolerance-gated, not bit-exact, so capture is an RTF optimization
layered over the same component semantics.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import warp as wp

from nexus_sim._src.diagnostics import diagnostics

from .clock import DeviceClock
from .interfaces import Stage, Tick
from .interrupts import hold_interrupts, stop_signals
from .logging import logger
from .ports import PortMap
from .profiling import LoopProfiler
from .schema import PoseTwist
from .signals import wire
from .stages import Bound, build_ring, opening, partition, plan_line, seed_stages, stages_of, warm_stages

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import newton

    from nexus_sim._src.logging import Logger
    from nexus_sim._src.peers import Peer
    from nexus_sim._src.recording import Recorder

    from .interfaces import (
        Actuator,
        Clock,
        Controller,
        Physics,
        Renderer,
        Sensor,
    )


def _on_cuda() -> bool:
    """Whether the active Warp device captures graphs: a CUDA device."""
    return bool(wp.get_device().is_cuda)


def _instance(component) -> str:
    """A component's instance name, the leaf of its path: its ``name``, else its class's name."""
    return str(getattr(component, "name", None) or type(component).__name__)


def _sensor_names(sensors: list) -> list[str]:
    """Each sensor's instance name, in order.

    Raises:
        ValueError: Two sensors share a name; the message names the prim that declares each.
    """
    first: dict[str, object] = {}
    for sensor in sensors:
        name = _instance(sensor)
        if name in first:
            declared = [getattr(s, "prim_path", None) or type(s).__name__ for s in (first[name], sensor)]
            raise ValueError(
                f"sensors {declared[0]} and {declared[1]} share the name {name!r}: a recorded row's path "
                "ends in its sensor's name, so rename one prim"
            )
        first[name] = sensor
    return [_instance(sensor) for sensor in sensors]


class Orchestrator:
    """Fixed-order, deterministic per-tick driver over typed component boundaries.

    Runs the ring of stages every component states, sensors -> estimator -> guidance -> controller ->
    actuate and step -> record, per docs/design/execution.md. Before any stage runs, it wires the signals the stages
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
        sensors: Iterable[Sensor],
        controller: Controller,
        estimator: object | None = None,
        guidance: object | None = None,
        commands: Iterable[object] = (),
        forces: Iterable[object] = (),
        actuator: Actuator | None = None,
        renderer: Renderer | None = None,
        peers: Iterable[Peer] = (),
        ports: Mapping[str, dict] | None = None,
        connections: Mapping[str, Mapping[str, str]] | None = None,
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
            controller: Control boundary. ``connect()`` binds and starts the peer, and the
                seed pass waits for it; its stages set the command buffer; ``close()`` tears down.
                A controller with a peer exposes ``attached``, false until the peer dials in,
                which holds the seed pass's clock.
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
            actuator: The old actuator seam, the examples' single-body ``Rotors``: one device stage
                that writes the shared body forces from the controls the controller writes. ``None``
                for a run on the command and force seams.
            renderer: Optional render-lifecycle object, for example the Kit render peer's
                :class:`~nexus_sim._src.rendering.KitRenderer`: the loop calls only
                ``on_physics_ready()``/``close()``; the host-rate RTX camera *sensors* drive
                rendering itself in a host stage.
            peers: The peers the build started for this run, for example the PX4 Software In The
                Loop (SITL) container. The loop stops each when the run ends, whether it ran out,
                stopped early or never stepped; it starts none, since a peer boots while the build
                goes on.
            ports: The run's port map, a :class:`~nexus_sim._src.core.ports.PortMap`: each link that
                leaves the run, by name, to the address a script opens its client on. The run
                owns every address, and the build names them here; ``None`` names no link.
            connections: For a component's prim, each signal it reads and the prim whose component writes
                it, from the vehicle's ``nexus:inputs:`` relationships: where more than one component
                writes a signal, a reader takes the one its connection names. A component's prim is its
                ``prim_path``, which the build sets. ``None`` connects nothing.
            logger: Optional :class:`~nexus_sim._src.logging.Logger`: the recording
                sink + shared log calls. ``None`` ⇒ no recording, for max speed. When present, the
                loop hands each component that states ``set_logger`` a view scoped to its path, and
                writes the Recorder's histories into it, a block at a time.
            recorder: The run's :class:`~nexus_sim._src.recording.Recorder`, which keeps every row of
                the plant's state and of each sensor's output. The loop reads the attribute at the first
                step, so a host can set it after construction; ``None`` records nothing.
            on_tick: Optional post-step observer called as ``on_tick(state, t, steps)``
                after recording: an output-only hook for demos and tests to read the live state,
                for example to capture a trajectory.
            preroll_timeout: Seconds to wait for the controller to establish lockstep
                during preroll before raising ``ConnectionError``.
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
        self.peers = list(peers)
        self.ports = PortMap() if ports is None else ports
        self.connections = dict(connections or {})
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
        # Logging components expose `set_logger(logger)`. The orchestrator hands each a view of the one
        # Logger scoped to the component's path, so no call site names an entity; None when off, the
        # single switch. A component logs a value with no fixed width live, in its own code: the
        # guidance its markers at a mission event, an MPC controller its horizon when it refreshes, a
        # camera its frames; the guidance via `add_loggable` when the loop builds its ring, since a
        # flight can hand it over after construction. The scene and every fixed-width value come from
        # the Recorder's histories, which the loop writes a block at a time.
        self._loggables = [c for c, _ in wired if hasattr(c, "set_logger")]
        for c, path in wired:
            if hasattr(c, "set_logger"):
                c.set_logger(self._scoped(path))
        # Live-RTF anchors, wall and sim: a rolling window the width of the ~1 s emission interval.
        # Distinct from _last_log_t, the sim-time scene decimation, and from run_stats' post-hoc RTF.
        self._rtf_wall_prev: float | None = None
        self._rtf_sim_prev = 0.0
        # The Recorder records everything itself: the plant's state and every device signal a component
        # writes, which the loop hands it once it wires the ring. Its taps are device launches in the record
        # stage, so recording never adds a host stage. The loop reads the attribute at the first step, so
        # `Sim` can hand a self-assembled loop its Recorder after construction.
        self.recorder = recorder
        self._written = 0  # rows of every history the Logger has written
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
        # The tick's sim time on the device, for a run with a stage that reads it; the ring sets it.
        self._device_clock: DeviceClock | None = None

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
    def _watch_plant(self) -> None:
        """Hand the Recorder the plant, so it registers one history per body and joint. At setup, right after
        the reset, so a script reads the plant's histories even when the wiring stops the run.
        """
        if self.recorder is not None:
            self.recorder.watch(self.physics)

    def _tap_signals(self, ring: list[Bound]) -> None:
        """Hand the Recorder every device signal a component's stage writes, with the writer's path, so it
        registers one history each. After the loop wires the ring and before the warm pass, so every history's
        buffer exists before any capture. The loop's own clock has no path, so its ``time`` keeps no history: every
        row carries the tick's time.
        """
        if self.recorder is None:
            return
        seen: set[int] = set()
        for bound in ring:
            path = self._paths.get(id(bound.component))
            if path is None or id(bound.stage) in seen:  # the ring repeats the stages of each physics substep
                continue
            seen.add(id(bound.stage))
            for signal in bound.stage.writes:
                if signal.device:
                    self.recorder.tap(path, signal, source=type(bound.component).__name__)

    def _record_tick(self) -> None:
        """The record stage: the Recorder snapshots the plant's state and every tapped signal into their
        histories. A device stage, no D2H, so it replays with its graph. A no-op with no Recorder, leaving
        that graph the same byte for byte.
        """
        if self.recorder is None:
            return
        self.recorder.record()

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
        """Emit the live real-time factor ~1 Hz wall-clock: sim seconds over wall seconds since the
        last emission, a rolling window exactly one emission interval wide. Only completed ticks reach
        this call, so a stalled exchange shows as a *low* next reading, not a live countdown.
        """
        log_rtf = getattr(self.logger, "log_rtf", None)  # a custom sink might not have it, and None is off
        if log_rtf is None:
            return
        now = time.monotonic()
        if self._rtf_wall_prev is None:
            self._rtf_wall_prev, self._rtf_sim_prev = now, st
            return
        wall = now - self._rtf_wall_prev
        if wall < 1.0:
            return
        log_rtf((st - self._rtf_sim_prev) / wall)
        self._rtf_wall_prev, self._rtf_sim_prev = now, st

    def _close_logs(self) -> None:
        """Teardown: write the rows still staged, the last block of every history, the run's profile,
        *then* close the sink, so every row lands before the ``.rrd`` finalizes.
        """
        if self.logger is None:
            return
        if self.recorder is not None and hasattr(self.logger, "write"):
            self.logger.write(self.recorder.histories, self._written)  # the last block: the staged rows
        # The final steady RTF from run_stats, whose loop's finally ran first, stamped at end-of-run
        # sim time; a run faster than the ~1 s live cadence would otherwise never show a reading.
        stats = getattr(self, "run_stats", None) or {}  # missing until a run has looped
        if stats.get("rtf") and hasattr(self.logger, "log_rtf"):
            self.logger.set_time(stats["control_steps"] * self.clock.dt)
            self.logger.log_rtf(float(stats["rtf"]))
        if stats and hasattr(self.logger, "log_profile"):
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
        stages and partitions it, runs the warm pass over the settled state, records the seed row,
        captures each device segment on CUDA, runs the seed pass of the controller's host stages, the
        preroll for a controller with a peer, then loops. Every controller advances one tick per
        ``next()``. Teardown, renderer,
        controller and logs, runs in the ``finally``, whether the generator runs out, on run to
        completion or peer disconnect, or closes early, via :meth:`close` on ``stop()`` mid-step.
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
            self._watch_plant()
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
                self._paths[id(self.guidance)] = "guidance"
                if hasattr(self.guidance, "set_logger"):
                    self.add_loggable(self.guidance, "guidance")
            ring = build_ring(
                sensors=self.sensors,
                estimator=self.estimator,
                guidance=self.guidance,
                controller=self.controller,
                physics=self.physics,
                commands=self.commands,
                forces=self.forces,
                actuator=self.actuator,
                record=record,
                substeps=self.physics_substeps,
            )
            self._check_estimator(ring)
            # The tick's sim time on the device, for a stage that reads it: the clock's stage opens each tick.
            reads_time = any(s.name == "time" for b in ring for s in b.stage.reads)
            self._device_clock = DeviceClock(self.clock.dt) if reads_time else None
            if self._device_clock is not None:
                ring = opening(ring, stages_of(self._device_clock, "clock"))
            wire(ring, self.connections)  # every signal a stage declares gets its buffer before any stage runs
            self._tap_signals(ring)  # one history per device signal the ring's stages write
            self._check_guidance(ring)
            segments = partition(ring)
            captured = _on_cuda()
            logger.info(plan_line(segments, captured))
            tick = Tick(
                state=state,
                t=self.clock.now(),
                dt=self.clock.dt / self.physics_substeps,
                base=getattr(self.physics, "base_index", 0),  # a physics that finds none has body 0 as base
            )
            if self._device_clock is not None:
                self._device_clock.time.write([tick.t.sim_time])
            # A controller without a peer connects first, which reserves its device buffers before the
            # warm pass. One with a peer connects after the capture, so every kernel the loop launches has
            # loaded before the peer is up: from the moment PX4 dials in, it logs a ``poll timeout``
            # error for each second of wall time that brings no message. The peer's own signal,
            # ``attached``, says which, until #37 moves it to the peer.
            peer = hasattr(self.controller, "attached")
            if not peer:
                self.controller.connect()
            # The warm pass, over the settled state. A guidance's warm stage runs between the sensors'
            # and the controller's and writes its setpoint, so the controller holds the first goal
            # before its own first stage. The log opens at the settled state's time, so the markers a
            # guidance logs here sit on the timeline.
            self._begin_log(tick.t)
            for st in warm_stages(ring):
                st.run(tick)
            self._record_tick()  # the settled pre-flight row, the datum every climb measures against
            self._commit()
            graphs = self._capture(segments, tick) if captured else None
            if peer:
                self.controller.connect()  # bind the peer's port and start it
            self._seed(ring, tick, peer)
            yield from self._loop(segments, graphs, tick)
        except ConnectionError as e:
            logger.info(
                str(e)
            )  # the autopilot's disconnect: a run's normal end. A dying Kit peer raises KitPeerError, which ends the run with it
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
                self.controller.close()
            finally:
                self._stop_peers()

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
            dict.fromkeys(type(b.component).__name__ for b in ring for s in b.stage.reads if s.type is PoseTwist)
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

    def _seed(self, ring, tick, peer: bool) -> None:
        """The seed pass, for a controller with a host stage: one pass of the sensors' warm stages and
        the controller's stages over the settled state, whose final exchange writes the controls the
        first tick applies. Each pass advances the sim clock, and its mirror on the device with it. A
        controller with a peer holds the sim clock until the peer attaches, so the peer's first stamp is
        near zero, and the pass repeats, re-sampling the static settled state so noise dithers into a
        live feed, until the first controls arrive or the preroll times out. A controller whose stages
        are all device stages gets no pass: the warm pass seeded it.

        Raises:
            ConnectionError: No controls arrived within ``preroll_timeout``.
        """
        stages = seed_stages(ring)
        if all(st.kind == "device" for st in stages):
            return
        if peer:
            logger.info("Waiting for the controller to start lockstep...")
        tick.timeout = 0.05
        deadline = time.monotonic() + self.preroll_timeout
        while time.monotonic() < deadline:
            # Hold the sim clock until the peer has dialed in: time spent waiting would start the peer's
            # clock late, and PX4 times its boot checks from its first stamp.
            attached = self.controller.attached if peer else True
            if attached:
                tick.t = self.clock.advance()
                if self._device_clock is not None:
                    self._device_clock.advance()
            else:
                tick.t = self.clock.now()
            if all(st.run(tick) is not False for st in stages):
                if peer:
                    logger.info("controller lockstep established")
                return
        raise ConnectionError(f"controller did not respond within {self.preroll_timeout}s")

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
        other stream operation during a capture kills it, so the warm pass and the seed row complete
        first, and nothing else touches the device until the loop replays.
        """
        wp.synchronize()  # complete the warm pass and the seed row before a capture opens
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

    def _loop(self, segments, graphs, tick):
        """The steady loop as a generator: one control tick per ``yield``, which step() drives.
        Each tick advances the clock, opens its log, then runs the segments in ring order: a device
        segment replays its graph, or runs stage by stage without one, and a host stage runs on the
        host between replays. A host stage that reports its peer gone ends the run, and so does a tick
        a stage marked done, once it completes. The ``finally``
        stamps the RTF, so it runs whether the generator runs out or closes early on stop.
        """
        count = 0
        t0 = time.monotonic()  # for the end-to-end RTF, sim-time advanced / wall-time, of the live run
        # Steady-state window: the RTF measurement starts here so it excludes lazy kernel compilation and
        # the initial settle.
        warmup_steps = 250
        t_warm = None
        steps_warm = 0
        tick.timeout = self.exchange_timeout
        replay = wp.capture_launch if graphs is not None else None
        prof = self._make_profiler("graph" if graphs is not None else "eager")
        try:
            while not self._stop and not tick.done and (self.max_steps is None or count < self.max_steps):
                count += 1
                prof.tick_begin()
                if count == warmup_steps:
                    t_warm, steps_warm = time.monotonic(), count
                opened = False
                for i, seg in enumerate(segments):
                    if seg.kind == "host":
                        if not opened:
                            opened = self._open_tick(tick)
                        st = seg.stages[0]
                        if st.run(tick) is False:  # the peer stopped answering: a run's normal end
                            count -= 1  # the disconnect tick didn't advance the sim
                            raise ConnectionError(
                                f"{self._controller_name()} disconnected (no actuator controls received)"
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
                yield  # one control tick complete: step() returns here
        finally:
            self._record_rtf(count, t0, t_warm, steps_warm)
            prof.close()
            self.run_stats["profile"] = prof.stats()

    def step(self) -> bool:
        """Advance the run one control tick, returning ``True`` if a tick ran or ``False`` when the run has
        ended: mission complete / ``max_steps`` / ``stop()`` / peer disconnect. Setup, reset, connect and
        graph capture, runs on the first call; teardown runs automatically when the run ends.
        ``Sim.step`` drives this for every controller, PX4 included. Deterministic:
        no wall-clock.
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
