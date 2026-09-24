"""The fixed-order, deterministic orchestrator of architecture.md §3, §4, §5.

Reproduces the bridge's exact per-tick sequence, the behavioral reference, but
behind typed component boundaries. ``run()`` selects the **execution strategy**
automatically, in ``_execution_strategy``: there is no separate knob, and the rule
is single: **capture everything capturable; host-exchange only the components
that need it.** On a CUDA device with the whole device region, physics + actuator
+ in-graph sensors, ``capturable``, the loop records the per-tick device region into a
CUDA graph once and replays it: the captured strategy, the Real Time Factor (RTF) lever
of architecture.md §5. The controller then just picks its seam: a ``capturable``
in-process controller, Proportional Integral Derivative (PID) or policy, with a device-native
``exchange``, joins the graph so the *whole* tick captures, in ``_loop_captured_inprocess``; any other
controller, a ``host_boundary`` peer such as PX4, whose blocking MAVLink ``exchange`` is
uncapturable, or an in-process host solver such as a Model Predictive Control (MPC) controller's
per-tick optimization, exchanges at the host seam, read -> exchange -> write_controls, between
replays, in ``_loop_captured_host_exchange``. A non-capturable controller never forces the
rest of the tick eager.

The strategy is controller-agnostic: it routes on the generic ``host_boundary`` /
``capturable`` markers, not on any specific controller type. The **eager** strategy,
``_loop``, where Python steps components one-by-one, is the fallback: a CPU device,
because capture needs CUDA, or a non-capturable device-region component, and the
bit-exact CPU determinism gate. It stays the debugging/parity path. GPU physics
is tolerance-gated, not bit-exact, so capture is an RTF optimization layered over
the same component semantics.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from nexus._src.diagnostics import diagnostics

from .logging import logger
from .profiling import LoopProfiler
from .schema import Measurement

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import newton

    from nexus._src.logging import Logger

    from .interfaces import (
        Actuator,
        Clock,
        Controller,
        Environment,
        Physics,
        Renderer,
        Sensor,
    )


class Orchestrator:
    """Fixed-order, deterministic per-tick driver over typed component boundaries.

    Reproduces the bridge's exact per-tick sequence, sense -> control exchange ->
    actuate -> step, behind typed components, and picks the execution strategy
    automatically from the active device, per architecture.md §3, §4, §5: capture
    everything capturable, host-exchange only the components that need it. On a
    CUDA device with the whole device region, physics + actuator + in-graph
    sensors, ``capturable``, the loop records the per-tick device region into a CUDA
    graph once and replays it, the captured strategy, the RTF lever; otherwise it
    steps components one-by-one in Python, the eager fallback / parity path. The
    controller picks its seam: an in-process ``capturable`` controller joins the
    graph, so the whole tick captures, while any other controller, a
    ``host_boundary`` peer such as PX4 MAVLink lockstep, or an in-process host solver
    such as an MPC, exchanges at the host seam between replays. Strategy selection
    is controller-agnostic: it routes on the generic ``host_boundary`` /
    ``capturable`` markers, not on any specific type.

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
        environment: Environment,
        physics: Physics,
        actuator: Actuator,
        sensors: Iterable[Sensor],
        controller: Controller,
        renderer: Renderer | None = None,
        logger: Logger | None = None,
        on_tick: Callable[[newton.State, float, int], None] | None = None,
        preroll_timeout: float = 2.0,
        exchange_timeout: float = 2.0,
        max_steps: int | None = None,
        physics_substeps: int = 1,
    ):
        """Wire the components and run options into a single tick driver.

        Args:
            clock: Simulation clock. ``advance()`` advances and returns the sim
                time ``t``; ``dt`` is the control timestep; ``throttle()`` paces the
                loop to wall-time, a no-op unless real-time throttling is on.
            environment: Environment model. ``sample(state, t)`` returns the per-tick
                ``env`` context, for example wind/gravity/air density, passed to the
                actuator, physics, and sensors.
            physics: Physics component. ``reset()`` builds and settles the vehicle at
                the North East Down (NED) origin and returns the initial state; ``clear_forces(state)``
                zeroes the shared ``body_f`` and ``step(state, env, dt)`` integrates:
                collide + solver step + double-buffer. Can expose ``capturable``.
            actuator: The actuator. ``forces(controls, state, env)`` writes the
                shared body forces, eager; ``write_controls`` / ``forces_wp(state)``
                are the host-boundary captured seam. Can expose ``capturable``.
            sensors: Iterable of sensors producing the Forward Right Down (FRD) ``Measurement``. Each
                exposes ``sample(state, env, t, meas)``, eager, or
                ``sample_wp(state, env, t)`` + ``read(meas)``, captured. Stored as a
                list; each can expose ``capturable``.
            controller: Control boundary. ``connect()`` binds and waits for the peer;
                ``exchange(meas, t, timeout)`` returns controls or ``None``;
                ``close()`` tears down. Can expose ``host_boundary`` / ``capturable``.
            renderer: Optional render-lifecycle object, for example the Kit render peer's
                :class:`~nexus._src.rendering.KitRenderer`: the loop calls only
                ``on_physics_ready()``/``close()``; the host-rate RTX camera *sensors* drive
                rendering itself at the host seam.
            logger: Optional :class:`~nexus._src.logging.Logger`: the recording
                sink + shared log calls. ``None`` ⇒ no recording and no per-tick log
                fan-out, for max speed. When present, each loggable component's
                ``log(t, logger)`` / ``flush(logger)`` runs at the host seam, §10.
            on_tick: Optional post-step observer called as ``on_tick(state, t, steps)``
                after recording: an output-only seam to read the live state and
                react, for example advance a waypoint goal.
            preroll_timeout: Seconds to wait for the controller to establish lockstep
                during preroll before raising ``ConnectionError``.
            exchange_timeout: Per-tick timeout in seconds for the blocking
                ``controller.exchange`` in the steady loop.
            max_steps: Bound on control steps for the steady loop. ``None`` runs until
                the controller ends the run, the PX4 default; a finite run,
                test/demo/episode, sets it.
            physics_substeps: Physics steps per control exchange, zero-order-hold over
                finer physics, cf. multi-rate. ``1`` is lockstep, the PX4 default;
                ``>1`` lets a policy control at a coarse rate over finer integration.
        """
        self.clock = clock
        self.environment = environment
        self.physics = physics
        self.actuator = actuator
        self.sensors = list(sensors)
        # Sensor partition; architecture: sensors are the one abstraction, renderables are just sensors:
        # * in-graph sensors: the physics-rate suite, Inertial Measurement Unit (IMU), Global Positioning
        #   System (GPS), baro, mag and obs: sampled every tick, join the captured CUDA graph when capturable.
        # * host-rate sensors, ``host_rate = True``: RTX renderables, camera and lidar: inherently
        #   low-rate and host-bound, because their frames come from the Kit peer over a socket, sampled at
        #   the host seam of every loop and self-decimating to their own rate. They never veto the
        #   captured strategy.
        self._graph_sensors = [s for s in self.sensors if not getattr(s, "host_rate", False)]
        self._host_sensors = [s for s in self.sensors if getattr(s, "host_rate", False)]
        self.controller = controller
        self.renderer = renderer
        # The single logging switch: a `Logger`, the recording sink + the shared log_state/log_image
        # calls, or None. None ⇒ no recording and no per-tick log fan-out → max benchmark/CI speed.
        self.logger = logger
        # Logging components expose `set_logger(logger)`. The orchestrator hands each the one Logger, the
        # single switch, None when off. Physics has a per-tick `log(t)` + a teardown `flush()`; its
        # scene/trail change every tick and it's captured, so the orchestrator must call it from outside the
        # graph. The operator + MPC controllers log their overlays event-driven in their own methods, at a
        # mission event / when the horizon refreshes, gated on the handed-over logger; they just need it
        # set, the operator via `add_loggable`, as it's wired on_tick, not a core component.
        self._loggables = [c for c in (physics, actuator, controller, *self.sensors) if hasattr(c, "set_logger")]
        for c in self._loggables:
            c.set_logger(logger)
        # The per-tick log fan-out decimates to the Logger's log_hz; people scrub an .rrd at human rates, and
        # full-rate scene logging tanks recorded-run RTF, esp. PX4 lockstep. The clock lives here: one
        # throttle for every per-tick logger, not per-component. Event-driven overlays self-rate, untouched.
        self._log_interval = logger.log_interval if logger is not None else 0.0
        self._last_log_t: float | None = None
        # Live-RTF anchors, wall and sim: a rolling window the width of the ~1 s emission interval.
        # Distinct from _last_log_t, the sim-time scene decimation, and from run_stats' post-hoc RTF.
        self._rtf_wall_prev: float | None = None
        self._rtf_sim_prev = 0.0
        # Component-owned observation seam, the read-side twin of logging: a recordable component exposes
        # a capturable `record_wp()` that snapshots its quantities into a Recorder channel. Sim's `observe`
        # attaches the Recorder post-build, via `attach_recorder`, since observation is a
        # control-surface concern. The taps are device-only, so observing never forces the eager strategy.
        self._recordables = [c for c in (physics, actuator, controller, *self.sensors) if hasattr(c, "record_wp")]
        self._recorder = None
        # Hosting hooks; the control surface, Sim, drives these. on_tick is the per-tick observer,
        # post-step, fired as on_tick(state, t, steps) after recording: a demo/test seam to read the
        # live state and react, for example advance a waypoint goal, output-only. _stop is the cooperative
        # teardown the host sets to end the steady loop.
        self.on_tick = on_tick
        self._stop = False
        # The tick generator backing run()/step(), the in-process driving seam, lazily created on the
        # first step(), None between runs. run() exhausts it; Sim.step() advances it one tick.
        self._ticks_iter = None
        self.preroll_timeout = preroll_timeout
        self.exchange_timeout = exchange_timeout
        # Bound the steady loop; None = run until the controller ends it, the PX4 default.
        # An in-process controller, for example the trained policy, never returns None, so a
        # finite run, test/demo/episode, sets this.
        self.max_steps = max_steps
        # Physics steps per control exchange, zero-order-hold control over finer physics,
        # cf. multi-rate: 1 = lockstep, the PX4 default. >1 lets a policy control at a
        # coarse rate, for example 50 Hz, over finer physics integration, matching its training.
        self.physics_substeps = int(physics_substeps)

    # -- component-owned logging seam, architecture.md §10 ------------------------
    def add_loggable(self, component) -> None:
        """Register a logging component that isn't a core component, the operator, wired as ``on_tick``,
        and hand it the Logger: ``_logger``, None when off, the gate for its event-driven logging.
        Idempotent.
        """
        if component not in self._loggables:
            self._loggables.append(component)
        component.set_logger(self.logger)

    # -- component-owned observation seam, the read-side twin of logging -----------
    def attach_recorder(self, recorder) -> None:
        """Attach the observation sink, Sim's ``observe=True``, and hand each recordable component the
        Recorder, so it registers its channel. The read-side mirror of the Logger wiring; the taps are
        capturable, device-only, so observing never forces the eager strategy. Call before ``run()``;
        the captured graph records the taps at capture time.
        """
        self._recorder = recorder
        for c in self._recordables:
            c.set_recorder(recorder)

    def _record_tick(self) -> None:
        """Per-tick observation fan-out: each recordable's capturable tap snapshots its own device buffers,
        physics' ``state0``, a sensor's measurement, into its Recorder channels. Launched INSIDE the
        captured device region, device-only, no D2H, so it replays with the graph. A no-op when not
        observing, leaving the not-observing graph the same byte for byte.
        """
        if self._recorder is None:
            return
        for c in self._recordables:
            c.record_wp()

    def _begin_log(self, t) -> None:
        """Set the shared ``time`` timeline once at the start of each tick, right after the clock advances
        and BEFORE ``exchange``, so every component's overlay this tick, the controller's horizon in
        ``exchange``, the operator's markers in ``on_tick``, physics' scene, lands at the same timestamp,
        and no component touches ``set_time`` itself. A no-op when not recording.
        """
        if self.logger is not None:
            self.logger.set_time(float(t.sim_time))

    def _log_tick(self, t) -> None:
        """Host-seam per-tick log fan-out, outside any captured graph: call each loggable that has a
        per-tick ``log(t)``, physics' scene + trail. Decimated to the Logger's ``log_hz`` here, so every
        per-tick logger rides one clock. The single off-switch: a no-op when ``logger is None``. The
        operator + controllers log event-driven in their own methods, not here.
        """
        if self.logger is None:
            return
        st = float(t.sim_time)
        self._log_rtf(st)  # wall-clock gated; must run BEFORE the sim-time decimation early return
        if self._last_log_t is not None and self._log_interval and (st - self._last_log_t) < self._log_interval:
            return  # decimated to log_hz
        self._last_log_t = st
        for c in self._loggables:
            log = getattr(c, "log", None)
            if log is not None:
                log(t)

    def _log_rtf(self, st: float) -> None:
        """Emit the live real-time factor ~1 Hz wall-clock: sim seconds over wall seconds since the
        last emission, a rolling window exactly one emission interval wide. Only completed ticks reach
        this seam, so a stalled exchange shows as a *low* next reading, not a live countdown.
        """
        log_rtf = getattr(self.logger, "log_rtf", None)  # a custom sink might not have it
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
        """Teardown: flush each loggable's accumulated emission, dump the Recorder's ring buffers as debug
        time series plus the flown path when observing, *then* close the sink, so every ``send_columns``
        lands before the ``.rrd`` finalizes. Fault-isolated: a dump hiccup must never eat the teardown.
        """
        if self.logger is None:
            return
        for c in self._loggables:
            flush = getattr(c, "flush", None)
            if flush is not None:
                flush()
        # The final steady RTF from run_stats, whose loop's finally ran first, stamped at end-of-run
        # sim time; a run faster than the ~1 s live cadence would otherwise never show a reading.
        stats = getattr(self, "run_stats", None) or {}  # missing until a run has looped
        if stats.get("rtf") and hasattr(self.logger, "log_rtf"):
            self.logger.set_time(stats["control_steps"] * self.clock.dt)
            self.logger.log_rtf(float(stats["rtf"]))
        if stats and hasattr(self.logger, "log_profile"):
            self.logger.log_profile(stats)  # steps + RTF + the loop profiler's breakdown, for the Profile tab
        dump_to = getattr(self._recorder, "dump_to", None)
        if dump_to is not None:
            try:
                dump_to(self.logger)  # the Recorder's own Rerun adapter: the rings + the flown path
            except Exception as exc:
                logger.warning(f"recorder debug dump failed: {exc}")
        self.logger.close()

    # -- pre-roll, init/handshake: settle is in physics.reset; here the loop samples the
    #    settled state and exchanges until PX4 returns its first actuators, so PX4
    #    establishes lockstep at the settled NED origin, architecture.md §3. --
    def _sample(self, state, env, t) -> Measurement:
        meas = Measurement()
        for s in self._graph_sensors:
            s.sample(state, env, t, meas)  # sensors read the live newton.State directly
        return meas

    def _sample_host(self, state, env, t, meas) -> None:
        """Host-rate sensors, RTX camera/lidar with ``host_rate = True``, sampled at every loop's host
        seam. Each self-decimates to its own rate, so calling per tick is cheap; their work, the
        exchange with the Kit peer, is host-bound and must never enter the captured graph.
        """
        for s in self._host_sensors:
            s.sample(state, env, t, meas)

    def _preroll(self, state) -> None:
        host = getattr(self.controller, "host_boundary", False)
        if host:  # an external peer such as PX4 takes seconds to dial in; an in-process controller answers at once
            logger.info("Waiting for the controller to start lockstep...")
        deadline = time.monotonic() + self.preroll_timeout
        while time.monotonic() < deadline:
            t = self.clock.advance()
            env = self.environment.sample(None, t)
            meas = self._sample(state, env, t)
            controls = self.controller.exchange(meas, t, timeout=0.05)
            if controls is not None:
                if host:
                    logger.info("controller lockstep established")
                return
        raise ConnectionError(f"controller did not respond within {self.preroll_timeout}s")

    def _loop(self, state):
        """Eager per-tick loop as a generator: one control tick per ``yield``, the step() driving seam.
        ``run()`` exhausts it; ``Sim.step()`` advances it one tick. The ``finally`` stamps the RTF, so
        it runs whether the generator runs out or closes early on stop.
        """
        dt = self.clock.dt
        steps = 0
        t0 = time.monotonic()  # for the end-to-end RTF, sim-time advanced / wall-time, of the live run
        # Steady-state window: the RTF measurement starts here so it excludes lazy kernel compilation, the
        # first actuator/solver launches, for example the ~1.2 s ClampingDCMotor compile, and the initial settle.
        warmup_steps = 250
        t_warm = None
        steps_warm = 0
        prof = self._make_profiler("eager")
        try:
            while not self._stop and (self.max_steps is None or steps < self.max_steps):
                steps += 1
                prof.tick_begin()
                if steps == warmup_steps:
                    t_warm, steps_warm = time.monotonic(), steps
                t = self.clock.advance()
                self._begin_log(t)  # set the timeline before exchange; the MPC controller logs its horizon there
                env = self.environment.sample(None, t)
                meas = self._sample(state, env, t)  # IMU/GPS/baro/mag -> FRD Measurement
                prof.mark("sensors.sample")  # per-sensor kernels + D2H reads -> Measurement

                controls = self.controller.exchange(meas, t, timeout=self.exchange_timeout)
                if controls is None:  # blocking-lockstep liveness: a lost peer ends the run
                    raise ConnectionError("controller disconnected (no actuator controls received)")
                prof.mark("exchange")  # the host seam, for example the PX4 MAVLink round-trip

                # Zero-order-hold the controls over physics_substeps finer physics steps.
                sub_dt = dt / self.physics_substeps
                for _ in range(self.physics_substeps):
                    self.physics.clear_forces(state)  # physics owns the body_f clear
                    self.actuator.forces(controls, state, env)  # writes shared body_f, + the actuator joints
                    state = self.physics.step(state, env, sub_dt)  # collide + solver.step + double-buffer
                prof.mark("actuate+step")  # actuator kernels + solver step

                self._record_tick()  # observation tap, post-step groundtruth; no-op if not observing
                self._sample_host(state, env, t, meas)  # host-rate sensors, RTX cameras, self-decimated
                self._log_tick(t)  # host-seam log fan-out: scene+trail, horizon, …; no-op if logging off
                if self.on_tick is not None:
                    self.on_tick(state, t, steps)  # post-step observer, for example waypoint advance
                self.clock.throttle()
                prof.mark("record+log")
                prof.tick_end()
                yield  # one control tick complete: the step() driving seam
        finally:
            # Report the live RTF; for the decoupled PX4 flight this is the sim speed *with* PX4 in the
            # lockstep loop, the meaningful end-to-end number; the controller exchange paces the loop.
            # Steady RTF, post-warmup and compilation-free, is the headline; full RTF is for reference.
            now = time.monotonic()
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
            prof.close()
            self.run_stats["profile"] = prof.stats()

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

    def _loop_captured_inprocess(self, state, steps: int | None = None):
        """Captured strategy for an **in-process controller**, architecture.md §5: the loop records the
        whole per-tick device region, sensors -> controller.exchange -> actuator -> physics, into a CUDA
        graph once and replays it, with no per-tick Python / kernel-launch overhead. The Python glue,
        Measurement / Controls, runs once at capture; replay re-runs only the recorded kernels over the
        components' persistent buffers, advancing ``state``. ``steps`` defaults to ``max_steps``.

        A generator: one control tick per ``yield``, the step() driving seam: ``run()`` exhausts it,
        ``Sim.step()`` advances it one tick. The loop captures the graph once before the first yield.
        """
        import warp as wp

        steps = self.max_steps if steps is None else steps
        dt = self.clock.dt
        t = self.clock.advance()
        env = self.environment.sample(None, t)
        meas = Measurement()
        # Warm pass, outside the graph: every device buffer must exist before capture.
        # Memory allocated during CUDA stream capture is graph-owned; referencing it from outside
        # the graph reads as stable right up until another consumer allocates between replays. The
        # Kit renderer exposed this: the PID's lazily built mixer buffers silently diverged from the
        # graph's and the flight froze at the settle pose. Sampling + exchange only write their
        # persistent buffers, state-free, so the captured semantics stay the same.
        for s in self._graph_sensors:
            s.sample(state, env, t, meas)
        self.controller.exchange(meas, t, None)
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            for s in self._graph_sensors:
                s.sample(state, env, t, meas)  # device-native obs -> meas.observation, a Warp array
            controls = self.controller.exchange(meas, t, None)  # in-process: Warp Controls, no host wait
            self.physics.clear_forces(state)
            self.actuator.forces(controls, state, env)
            self.physics.step(state, env, dt)
            self._record_tick()  # capturable observation tap; joins the graph, device-only; no-op if not observing
        graph = cap.graph
        count = 0
        t0 = time.monotonic()
        warmup_steps = 250
        t_warm = None
        steps_warm = 0
        prof = self._make_profiler("captured-inprocess")
        try:
            while not self._stop and (steps is None or count < steps):
                count += 1
                prof.tick_begin()
                if count == warmup_steps:
                    t_warm, steps_warm = time.monotonic(), count
                prof.gpu_begin()
                wp.capture_launch(graph)
                prof.gpu_end()
                prof.mark("replay.launch")
                t = self.clock.advance()
                self._begin_log(t)  # set the timeline for this tick's overlays
                self._sample_host(state, env, t, meas)  # host-rate sensors, RTX cameras, self-decimated
                prof.mark("sensors.host")
                self._log_tick(t)  # host-seam log fan-out, outside the graph; no-op if logging off
                if self.on_tick is not None:
                    self.on_tick(state, t, count)  # post-step observer, for example waypoint advance
                self.clock.throttle()
                prof.mark("log")
                prof.tick_end()
                yield  # one control tick complete: the step() driving seam
        finally:
            self._record_rtf(count, t0, t_warm, steps_warm)
            prof.close()
            self.run_stats["profile"] = prof.stats()

    def _loop_captured_host_exchange(self, state, steps: int | None = None):
        """Captured strategy for a controller that exchanges at the **host seam**, architecture.md §5:
        any controller whose ``exchange`` can't join the graph: a ``host_boundary`` peer, PX4, whose
        blocking MAVLink lockstep is an off-device round-trip, or an in-process host solver, an MPC's
        per-tick optimization, a torch policy. Everything else still captures; the graph is the
        device region, reordered contiguous: ``actuator.forces_wp -> physics.step ->
        sensors.sample_wp``. Per tick the host seam does one D2H, sensors ``read`` ->
        ``Measurement``, the ``exchange``, and one H2D, ``actuator.write_controls``; the graph
        replays between. Uses only the generic component interfaces, no controller-specific code.

        Load-bearing: sensor noise must **dither per replay**; the sensors increment a device step
        counter inside the graph. A frozen captured stream reads to a state-estimator, PX4's Extended
        Kalman Filter (EKF), as a stuck sensor, so position fusion never starts and the vehicle won't arm.
        The preroll re-samples each iteration likewise. ``steps`` defaults to ``max_steps``; ``None``
        runs until the controller ends the run, on disconnect / ``stop()``.

        A generator: one control tick per ``yield``, the step() driving seam: ``run()`` exhausts it,
        ``Sim.step()`` advances it one tick. The loop captures the graph once before the first yield.
        """
        import warp as wp

        steps = self.max_steps if steps is None else steps
        dt = self.clock.dt
        meas = Measurement()
        # Pre-roll: sense the settled state once, which seeds the IMU finite-diff so capture runs first=0,
        # then stream the sensor feed until the controller's first exchange completes; an external
        # host-boundary peer takes seconds to dial in; an in-process controller answers on the first try.
        t = self.clock.advance()
        env = self.environment.sample(None, t)
        for s in self._graph_sensors:
            s.sample_wp(state, env, t)
        wp.synchronize()
        controls = None
        host = getattr(self.controller, "host_boundary", False)
        if host:
            logger.info("Waiting for the controller to start lockstep...")
        deadline = time.monotonic() + self.preroll_timeout
        while time.monotonic() < deadline:
            t = self.clock.advance()
            for s in self._graph_sensors:
                s.sample_wp(state, env, t)  # re-sample, state static, noise dithers -> live feed
                s.read(meas)
            controls = self.controller.exchange(meas, t, timeout=0.05)
            if controls is not None:
                break
        if controls is None:
            raise ConnectionError(f"controller did not establish lockstep within {self.preroll_timeout}s")
        if host:
            logger.info("controller lockstep established")
        # Seed one observation row: the settled pre-flight state, which is a datum in its own right,
        # the pose every climb measures against. The eager loop seeds one too, so both strategies
        # record the same pre-flight row. It also warms the record kernels' module load, so that
        # happens outside the capture below.
        self._record_tick()
        wp.synchronize()  # complete the seed row's launches before the capture below opens

        # Capture the device region, with controls already seeded into the actuator buffer, before the
        # first tick: any other stream op during CUDA stream capture kills the capture, and the run
        # would then die silently with PX4 lockstep frozen mid-boot. Replays are safe.
        self.actuator.write_controls(controls)
        with wp.ScopedCapture() as cap:
            self.physics.clear_forces(state)
            self.actuator.forces_wp(state)
            self.physics.step(state, env, dt)
            for s in self._graph_sensors:
                s.sample_wp(state, env, t)
            self._record_tick()  # capturable observation tap, post-step groundtruth; no-op if not observing
        graph = cap.graph
        count = 0
        t0 = time.monotonic()
        warmup_steps = 250
        t_warm = None
        steps_warm = 0
        prof = self._make_profiler("captured-host")
        try:
            while not self._stop and (steps is None or count < steps):
                count += 1
                prof.tick_begin()
                if count == warmup_steps:
                    t_warm, steps_warm = time.monotonic(), count
                prof.gpu_begin()
                wp.capture_launch(graph)  # apply controls -> step -> sense, into device meas buffers
                prof.gpu_end()
                prof.mark("replay.launch")
                t = self.clock.advance()
                self._begin_log(t)  # set the timeline for this tick's overlays
                for s in self._graph_sensors:
                    s.read(meas)  # D2H the new measurement; also the sync point that completes the replay
                prof.mark("gpu+read")  # the D2H read forces the replay's sync; device time lands here
                controls = self.controller.exchange(meas, t, timeout=self.exchange_timeout)
                prof.mark("exchange")  # the host seam: the PX4 MAVLink round-trip / the MPC solve
                if controls is None:  # controller disconnected -> end the run
                    count -= 1  # the disconnect tick didn't advance the sim
                    raise ConnectionError("controller disconnected (no actuator controls received)")
                self.actuator.write_controls(controls)  # H2D for the next replay, host seam
                prof.mark("write")
                self._sample_host(state, env, t, meas)  # host-rate sensors, RTX cameras, self-decimated
                prof.mark("sensors.host")
                self._log_tick(t)  # host-seam log fan-out: scene+trail, …; no-op if logging off
                if self.on_tick is not None:
                    self.on_tick(state, t, count)  # post-step observer, for example the operator's waypoint advance
                self.clock.throttle()  # no-op unless rtf>0, the interactive real-time throttle
                prof.mark("log")
                prof.tick_end()
                yield  # one control tick complete: the step() driving seam
        finally:
            self._record_rtf(count, t0, t_warm, steps_warm)
            prof.close()
            self.run_stats["profile"] = prof.stats()

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
            self.renderer.set_profiler(prof)  # detail spans inside the render seam: render/grab
        return prof

    def stop(self) -> None:
        """Cooperatively end the steady loop; checked at the top of each tick.

        The in-process Sim host sets this to tear a run down: the control surface's
        lifecycle verb. Takes effect within one ``exchange_timeout``, since the loop might be
        waiting in ``controller.exchange``.
        """
        self._stop = True

    def _execution_strategy(self) -> str:
        """Resolve the execution strategy for this run from the active device, architecture.md §5:
        ``"captured-inprocess"``, ``"captured-host-exchange"``, or ``"eager"``. No separate knob, one
        rule, capture everything capturable and host-exchange only what needs it: on a CUDA device with
        the whole device region, physics + actuator + in-graph sensors, ``capturable``, the tick
        captures; the controller then picks its seam: a ``capturable`` in-process controller joins
        the graph, captured-inprocess; any other, a ``host_boundary`` peer such as PX4, or an in-process
        host solver such as an MPC, exchanges between replays, captured-host-exchange. A CPU device or a
        non-capturable device-region component -> eager.
        """
        try:
            import warp as wp

            on_cuda = wp.get_device().is_cuda
        except Exception:
            on_cuda = False
        device_capturable = (
            on_cuda
            and getattr(self.physics, "capturable", False)
            and getattr(self.actuator, "capturable", False)
            and bool(self._graph_sensors)
            and all(getattr(s, "capturable", False) for s in self._graph_sensors)
        )
        if not device_capturable:
            return "eager"
        controller_in_graph = getattr(self.controller, "capturable", False) and not getattr(
            self.controller, "host_boundary", False
        )
        return "captured-inprocess" if controller_in_graph else "captured-host-exchange"

    def _ticks(self):
        """The single execution path, as a generator that yields once per control tick, driving both
        ``run()``, which exhausts it, and ``Sim.step()``, which advances it one tick.

        Resets physics, which builds + settles the vehicle at the NED origin, connects the controller,
        which binds its port and starts its peer; the wait for the peer is the preroll, not this call.
        Then it resolves the execution strategy and delegates to the matching loop.
        Every loop is a generator; it ``yield``s per tick and step-drives, for every control kind
        alike: a host-boundary peer, PX4, and an in-process autopilot both advance one tick per
        ``next()``. Common setup runs before the first tick; teardown, renderer/controller/logs close,
        runs in the ``finally``, whether the generator runs out, on run to completion / peer
        disconnect, or closes early, via :meth:`close` on ``stop()`` mid-step. The live RTF lands in
        ``run_stats``.
        """
        state = self.physics.reset()  # build + settle the vehicle at the NED origin
        try:
            # Renderer warm-up hook, before PX4 lockstep starts: the Kit peer connects and warms its
            # stage here, which takes seconds and would stall the lockstep. Inside the try: a peer
            # that fails its start must still reach the `finally`, which removes its container.
            if self.renderer is not None and hasattr(self.renderer, "on_physics_ready"):
                self.renderer.on_physics_ready()
            # From here on the controller owns a live peer, the PX4 container it launches, so every
            # exit path has to reach the `finally`'s controller.close().
            self.controller.connect()  # bind tcpin:4560 and return listening, then start the peer
            strategy = self._execution_strategy()
            logger.info(f"execution strategy: {strategy}")
            if strategy == "captured-host-exchange":
                # The host-exchange captured loop owns its own streaming preroll, since it must re-sample
                # the sensor feed while the peer dials in, and its own seed row.
                yield from self._loop_captured_host_exchange(state)
            elif strategy == "captured-inprocess":
                yield from self._loop_captured_inprocess(state)
            else:
                self._preroll(state)
                self._record_tick()  # seed one observation row, the same contract as captured
                yield from self._loop(state)
        except ConnectionError as e:
            logger.info(str(e))
        finally:
            if self.renderer is not None and hasattr(self.renderer, "close"):
                self.renderer.close()  # for example flush+close the First Person View (FPV) encoder, an output-only seam
            # Lifecycle teardown of the controller, then the logging teardown: _close_logs flushes each
            # loggable's accumulated emission, the MPC horizon, into the recording, dumps the Recorder's
            # rings and closes the sink last, so every send_columns lands before the .rrd finalizes. The recording is
            # what a failed run is *for*, so a controller that raises on the way out must not lose it.
            try:
                self.controller.close()
            finally:
                self._close_logs()

    def step(self) -> bool:
        """Advance the run one control tick, returning ``True`` if a tick ran or ``False`` when the run has
        ended: mission complete / ``max_steps`` / ``stop()`` / peer disconnect. Setup, reset, connect and
        graph capture, runs on the first call; teardown runs automatically when the run ends. The
        control surface, ``Sim.step``, drives this for every control kind, PX4 included. Deterministic:
        no wall-clock.
        """
        if self._ticks_iter is None:
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
        """Tear down a partially stepped run, ``Sim.stop()`` after ``step()``s: close the tick
        generator, running its ``finally``, RTF stamp + controller/logs teardown. Idempotent: a no-op if
        the run already finished, ``run()`` completed / ``step()`` returned ``False``.
        """
        if self._ticks_iter is not None:
            self._ticks_iter.close()  # GeneratorExit → the loop's finally, RTF, + _ticks' finally, close
            self._ticks_iter = None
