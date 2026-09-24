"""Sim: the in-process control-surface handle, the program-driving API.

Owns the sim: builds it from a ``LaunchConfig`` via ``build_from_launch`` and drives the
``Orchestrator`` loop on the *caller's* thread, one driving model for both control kinds. A
script steps the sim, with ``step``, ``run``, ``wait_until`` or ``sleep``, whether the autopilot
is in-process or a host boundary such as PX4: the orchestrator's tick generator yields once per
control tick either way, so a PX4 run is something you drive, not something you watch. Tears down
cooperatively. ``observe=True`` attaches a ``Recorder`` so ``physics`` and ``sensors`` read sim
ground truth.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from nexus._src.api.args import save_run_artifacts, sim_argparser  # noqa: F401  # re-export; defs are import-light
from nexus._src.config import LaunchConfig
from nexus._src.core import logger
from nexus._src.operator import InProcessOperator
from nexus._src.recording import ChannelMap, Recorder
from nexus._src.runtimes.launch import build_from_launch as _neutral_build_from_launch

if TYPE_CHECKING:
    from nexus._src.core import Orchestrator
    from nexus._src.core.interfaces import Controller
    from nexus._src.operator import Px4Offboard

# Control kinds whose controller is a host boundary, the PX4 Hardware In The Loop (HIL) lockstep.
# This selects the *operator*, Px4Offboard over MAVLink :14540 compared to the in-process
# InProcessOperator, never the driving path: every kind is step-driven on the caller's thread.
_HOST_BOUNDARY_KINDS = ("px4-sitl",)

# Wall-clock budget for PX4 to answer on the operator link, :14540, once the sim starts stepping
# for it: the same 30 s Px4Offboard's own blocking connect allows.
_PX4_LINK_TIMEOUT_S = 30.0


def build_from_launch(launch, **kwargs):
    """Build the orchestrator through the glue of the runtime this process is *in*.

    Inside a booted Kit app, since the Isaac Sim container runs examples under Kit's Python, route
    through the isaacsim launch glue so the vehicle's authored RTX sensors render: the *same*
    neutral routing plus the renderer factory. Per stage 3, any orchestrator setup runs identically
    in both runtimes, and a runtime differs only in its injected renderer. Everywhere else: the
    neutral, renderless glue. Detection is by the booted app, not importability: `import isaacsim`
    exists but is unusable before SimulationApp starts.
    """
    import sys

    if "isaacsim" in sys.modules and "omni.kit.app" in sys.modules:
        from nexus._src.runtimes.isaacsim.launch import build_from_launch as _isaac_build

        return _isaac_build(launch, **kwargs)
    return _neutral_build_from_launch(launch, **kwargs)


class Sim:
    """In-process control-surface handle for a Newton sim run.

    Builds an ``Orchestrator`` from a ``LaunchConfig`` and drives its loop on the *caller's*
    thread, one control tick per :meth:`step`, the same for an in-process autopilot and for
    a host-boundary one, the PX4 HIL lockstep. :meth:`start` drives setup as far as lockstep;
    :meth:`run` drives the whole run; :meth:`wait_until` and :meth:`sleep` step until a
    predicate or a sim-time budget. Use it as a context manager: ``__enter__`` builds,
    ``__exit__`` calls :meth:`stop` to tear the run down cooperatively.

    With ``observe=True`` the handle attaches a :class:`~nexus._src.recording.Recorder`,
    making :attr:`physics` and :attr:`sensors` read sim ground truth, in the world frame,
    Z-up Forward-Left-Up (FLU), off the components' observation channels.

    Args:
        vehicle: Registry vehicle *name*, for example ``"astro_max_fpv"``, or a local .usd path;
            ``None`` takes the registry's default vehicle.
        registry: Path to a catalog that extends the bundled one. ``None`` takes the nearest
            ``nexus.registry.yaml`` in the working directory or a directory over it, and only the
            catalog bundled in the wheel when no directory holds one.
        control: Control kind: ``"px4-sitl"``. PX4 is the one first-class controller; an
            example controller self-assembles its orchestrator and enters via
            :meth:`from_orchestrator` instead.
        scene: Optional registry scene to load, for example the ``"slalom"`` obstacle pillars;
            ``None`` = the default empty scene.
        device: Compute device for the runtime: ``"auto"``, the default, which picks CUDA when
            present, or an explicit ``"cpu"``, for bit-exact determinism, or ``"cuda"``.
        observe: Attach a ``Recorder`` so :attr:`physics` and :attr:`sensors` read ground truth.
        log: Record the run to a ``.rrd``; ``False`` runs headless with no recorder, at max speed.
        view: Serve the Rerun recording live on ``:9876`` instead of writing a ``.rrd`` file.
        cache_dir: Override the asset cache directory; ``None`` = the framework default.
        max_steps: Cap the run at this many control steps; ``None`` = the launch-config default.
        rtf: Real-time-factor throttle. ``0``, the default, runs unthrottled, as fast as the
            controller keeps up. ``1.0`` paces the loop to wall-clock for human-in-the-loop
            flying: 1:1 stick feel, and sim-time protocol timeouts align with wall-clock peers.
        reached_m: Mission-waypoint arrival threshold [m] for the in-process operator.
        final_hold_s: Keep running this long, in sim-time, after the final goal before stopping.

    Example:
        >>> with Sim("astro_max_base", control="px4-sitl") as sim:
        ...     sim.start()
        ...     sim.wait_until(lambda: sim.physics[sim.base_body].latest().altitude_m > 1.0, sim_timeout=30.0)
    """

    def __init__(
        self,
        vehicle: str | None = None,
        *,
        registry: str | None = None,
        control: str = "px4-sitl",
        scene: str | None = None,
        geo: str | None = None,
        device: str = "auto",
        observe: bool = True,
        log: bool = False,
        view: bool = False,
        debug: bool = False,
        cache_dir: str | None = None,
        solver: str | None = None,
        max_steps: int | None = None,
        rtf: float = 0.0,
        reached_m: float = 0.3,
        final_hold_s: float = 2.0,
    ):
        self._launch = LaunchConfig().set_control(control)
        # --vehicle is a registry *name*, a local .usd path, or None for the registry's default.
        self._launch.set_vehicle(vehicle)
        self._launch.registry = registry  # None: the run finds its own catalog, see load_registry
        if solver is not None:  # physics integrator override: mujoco | semi_implicit | featherstone
            self._launch.runtime.solver = solver
        if scene is not None:
            self._launch.set_scene(scene)  # registry scene, for example the 'slalom' obstacle pillars for sampling-mpc
        if geo is not None:  # override the scene's geodetic origin, for example to fly cesium over any lat/lon
            parts = [float(x) for x in geo.split(",")]
            self._launch.set_geodetic_origin(*parts)  # lat,lon[,alt]; alt is the WGS84 ellipsoidal surface height
        # Resolve the device selector: 'auto'/'gpu' -> 'cuda' if a CUDA device is present, else 'cpu'. The
        # command-line tool plus sim_argparser default to 'auto'; the launch path itself only distinguishes 'cpu'.
        if device in ("auto", "gpu"):
            import warp as wp

            have_gpu = wp.is_cuda_available()
            if device == "gpu" and not have_gpu:
                logger.warning("device=gpu requested but no CUDA device found, falling back to CPU")
            device = "cuda" if have_gpu else "cpu"
        self._launch.runtime.device = device
        self._reached_m = float(reached_m)  # operator advance threshold: mission waypoint arrival
        self._final_hold_s = float(final_hold_s)  # keep running this long, in sim-time, after the final goal
        if max_steps is not None:
            self._launch.runtime.max_steps = int(max_steps)
        self._launch.runtime.rtf = float(rtf)  # 0 = unthrottled; 1.0 = wall-clock pacing
        # Rerun, serve or file but not both: view serves live on :9876; log writes the .rrd. Omitting *both*
        # builds no Logger at all: no recording, no per-tick log fan-out → max headless speed. The two are exclusive.
        if log and view:
            raise ValueError("Sim(log=, view=): write-.rrd and serve-viewer are mutually exclusive: pick one")
        self._launch.output.log = log
        self._launch.output.view = view
        self._launch.output.debug = debug  # axes-only scene: coordinate triads, no meshes → a small .rrd
        self._cache_dir = cache_dir
        self._observe = observe
        self._in_process = self._launch.control.kind not in _HOST_BOUNDARY_KINDS
        self._orch = None
        self._operator = None
        self._recorder: Recorder | None = None
        self._base_ch = None  # cached base body channel: the default "vehicle" entity, for wait_until/sleep
        self._ran = False
        self._stopped = False
        self._prebuilt_orch = None  # set by from_orchestrator, the self-assembled-example entry
        self._ext_operator = None

    @classmethod
    def from_orchestrator(
        cls,
        orch: Orchestrator,
        *,
        operator: object | None = None,
        reached_m: float = 0.3,
        final_hold_s: float = 2.0,
        observe: bool = True,
    ) -> Sim:
        """Host a self-assembled :class:`~nexus._src.core.Orchestrator`: the examples' entry.

        An example builds its own orchestrator, its controller plus actuator plus sensors around
        the core components, see ``nexus/examples/controllers/*/assembly.py``, and hands it
        over; the ``Sim`` adds the control surface: the observation ``Recorder``, ``sim.physics``
        and ``sim.sensors``; the operator plane, where by default a core
        :class:`~nexus._src.operator.InProcessOperator` over the controller's
        ``accept_setpoint`` gets wired, with the orchestrator's ``reference_planner`` passed through
        for a tracking controller; and the run lifecycle,
        ``run``, ``step``, ``stop``, ``results`` and ``artifacts``.

        Args:
            orch: The built orchestrator; its components already carry the renderer and logger.
            operator: Override the operator commanding this run; ``None`` wires the default
                ``InProcessOperator`` when the controller exposes ``accept_setpoint``.
            reached_m: The default operator's waypoint-arrival threshold [m].
            final_hold_s: Keep running this long, in sim-time, after the final goal.
            observe: Attach the ``Recorder`` behind ``sim.physics`` and ``sim.sensors``.

        Returns:
            The ``Sim`` handle: use as a context manager, then ``sim.operator.set_mission`` plus
            ``sim.run()``.
        """
        sim = cls.__new__(cls)
        sim._launch = None
        sim._cache_dir = None
        sim._reached_m = float(reached_m)
        sim._final_hold_s = float(final_hold_s)
        sim._observe = observe
        sim._in_process = not getattr(orch.controller, "host_boundary", False)
        sim._orch = None
        sim._prebuilt_orch = orch
        sim._ext_operator = operator
        sim._operator = None
        sim._recorder = None
        sim._base_ch = None
        sim._ran = False
        sim._stopped = False
        return sim

    @classmethod
    def from_args(cls, args: argparse.Namespace, **overrides) -> Sim:
        """Construct a ``Sim`` from a :func:`sim_argparser` namespace, ignoring script-specific extras.

        ``overrides`` win over the namespace, for the ``Sim`` kwargs the shared parser doesn't cover:
        ``reached_m``, ``final_hold_s`` and ``gains``. So a script does
        ``Sim.from_args(args, final_hold_s=3.0)``.
        """
        from nexus._src.diagnostics import diagnostics

        diagnostics.configure(args)  # the shared --profile/--trace/--benchmark flags, process-wide
        kw = {
            "vehicle": getattr(args, "vehicle", None),
            "registry": getattr(args, "registry", None),
            "control": getattr(args, "control", "px4-sitl"),
            "device": getattr(args, "device", "auto"),
            "scene": getattr(args, "scene", None),
            "geo": getattr(args, "geo", None),
            "solver": getattr(args, "solver", None),
            "log": getattr(args, "log", False),
            "view": getattr(args, "view", False),
            "debug": getattr(args, "debug", False),
            "max_steps": getattr(args, "max_steps", None),
            "rtf": getattr(args, "rtf", 0.0),
        }
        kw.update(overrides)
        return cls(**kw)

    # -- context manager: build only; start()/step()/run() drive the run on the caller's thread --
    def __enter__(self) -> Sim:
        if self._prebuilt_orch is not None:
            self._orch = self._prebuilt_orch  # a self-assembled example's orchestrator, via from_orchestrator
        else:
            self._orch = build_from_launch(self._launch, cache_dir=self._cache_dir)
        # The API can't teleport the caller's process into the Kit container the way the command-line
        # tool re-execs itself: an RTX vehicle on the plain host runs physics-correct but *renderless*.
        # Say so loudly. The fix: `nexus script <your-script>`, which auto-launches the
        # container and boots Kit before handing off.
        if getattr(self._orch, "renderer", None) is None:
            usd_path = getattr(getattr(self._orch.physics, "vehicle_builder", None), "cfg", {}).get("usd_path")
            if usd_path:
                from nexus._src.vehicle.sensors.usd import vehicle_rtx_sensor_prims

                prims = vehicle_rtx_sensor_prims(usd_path)
                if prims:
                    logger.warning(
                        f"vehicle authors RTX sensors {prims} but no renderer is available in this "
                        "process: flying RENDERLESS. Run it under the Isaac Sim runtime: "
                        "`nexus script <your-script> [args]` (auto-launches the Kit container)."
                    )
        if self._observe:
            # Attach the observation sink: each recordable component registers its capturable channels;
            # physics → one per body plus per joint. dt → the per-row snapshot time, counter × dt.
            # The ring must cover the *whole* run, since post-run evaluation reads the full trajectory, so
            # size it from max_steps when the launch sets one, plus margin for the pre-flight seed rows.
            if self._launch is not None:
                max_steps, dt = self._launch.runtime.max_steps, self._launch.runtime.dt
            else:  # a self-assembled orchestrator carries its own bounds/clock
                max_steps = getattr(self._orch, "max_steps", None)
                dt = getattr(getattr(self._orch, "clock", None), "dt", 0.004)
            # Unbounded runs, interactive PX4 with max_steps=None, get ~2 min at 250 Hz: enough that the
            # end-of-run debug dump covers a whole interactive flight, ~40-80 MB device total across
            # the usual ~15 channels; the dump warns when the ring still wrapped.
            maxlen = max(4096, int(max_steps) + 64) if max_steps else 30_000
            self._recorder = Recorder(dt=dt, maxlen=maxlen)
            self._orch.attach_recorder(self._recorder)
            # Cache the base body channel, the discovered base, for the wait_until/sleep sim clock.
            self._base_ch = self._recorder.channels[f"physics/body/{self._orch.physics.base_body}"]
        if self._in_process:
            # In-process autopilot: wire the operator over the controller's thin setpoint surface at
            # the host seam, Orchestrator.on_tick, the post-step slot between graph replays. The
            # run itself is synchronous, sim.run(); the operator advances the mission and ends the
            # run. Nothing mutates inside the captured region, per the capture contract. An example can
            # pass its own operator, from_orchestrator(operator=...); the default is the core
            # InProcessOperator whenever the controller exposes accept_setpoint.
            if self._ext_operator is not None:
                self._operator = self._ext_operator
            elif hasattr(self._orch.controller, "accept_setpoint"):
                self._operator = InProcessOperator(
                    self._orch.controller,
                    stop=self._orch.stop,
                    reached_m=self._reached_m,
                    final_hold_s=self._final_hold_s,
                    # a tracking controller, acados, exposes a reference planner; the operator plans
                    # the whole-path ReferenceTrajectory and hands it over instead of sequencing goals.
                    planner=getattr(self._orch, "reference_planner", None),
                )
            if self._operator is not None:
                if hasattr(self._operator, "tick"):
                    self._orch.on_tick = self._operator.tick  # sequencing seam: advance the mission
                if hasattr(self._operator, "set_logger"):
                    self._orch.add_loggable(self._operator)  # logging seam: re-emit the mission viz
        # Host-boundary, PX4, wires nothing here: its operator is a remote Ground Control Station (GCS),
        # Px4Offboard, built lazily on first access after start(), and the run is step-driven the same
        # way as any other.
        return self

    # -- the operator, Plane 5, plus the controller's thin surface --
    @property
    def operator(self) -> InProcessOperator | Px4Offboard:
        """The Operator commanding this sim, Plane 5. In-process control, policy, pid, mpc or acados, →
        an :class:`InProcessOperator` over the controller's ``accept_setpoint``. PX4 → a
        :class:`Px4Offboard` over MAVLink :14540, PX4's offboard and onboard link, separate from the
        controller's HIL :4560, constructed plus connected lazily on first access, so access it
        *after* PX4 is up, for example after ``sim.start()``; cached, and closed on ``sim.stop()``.

        Connecting the PX4 link **steps the sim**, because PX4's clock is the sim's under lockstep:
        a caller that merely slept here would stop the sim, and PX4 would never send the heartbeat
        the caller waits for. The budget stays wall-clock: peer liveness is a property of the PX4
        process, and this must work under ``observe=False`` too, where there is no sim clock.

        Returns:
            The :class:`InProcessOperator`, in-process, or :class:`Px4Offboard`, PX4, driving this run.

        Raises:
            RuntimeError: Accessed before entering the ``Sim`` context, in the in-process case, or the
                run ended while the PX4 link was connecting.
            TimeoutError: PX4 didn't answer on :14540 within ``_PX4_LINK_TIMEOUT_S``.
        """
        if self._in_process:
            if self._operator is None:
                raise RuntimeError("enter the Sim context first (`with na.Sim(...) as sim:`)")
            return self._operator
        if self._operator is None:
            from nexus._src.operator import Px4Offboard

            op = Px4Offboard()
            op.open()  # bind the MAVLink link + start the pump; returns at once
            self._operator = op  # cache BEFORE the wait, so a failed connect is still closed by stop()
            deadline = time.monotonic() + _PX4_LINK_TIMEOUT_S
            while not op.connected:
                if not self.step():  # keep PX4's clock moving, or its heartbeat never comes
                    raise RuntimeError("the run ended before PX4 answered on the operator link (:14540)")
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"no PX4 heartbeat on the operator link (:14540) within {_PX4_LINK_TIMEOUT_S:.0f}s"
                    )
        return self._operator

    @property
    def controller(self) -> Controller | None:
        """The autopilot's thin control surface, ``accept_setpoint`` plus config, or ``None`` for PX4,
        which owns its mission in the external process and takes commands via ``sim.operator``.

        Returns:
            The in-process controller, which exposes ``accept_setpoint``, or ``None`` for a PX4 run.
        """
        ctrl = getattr(self._orch, "controller", None) if self._orch is not None else None
        return ctrl if (ctrl is not None and hasattr(ctrl, "accept_setpoint")) else None

    def start(self, timeout: float | None = None) -> None:
        """Drive the run's setup, returning once the first control tick has completed.

        One step: the orchestrator's first ``step()`` runs physics reset, the renderer warm-up,
        the multi-second Kit shader compile, the seed row and the graph capture,
        ``controller.connect()``, which binds ``:4560`` and launches the PX4 container, and the
        preroll wait for the peer, then flies one tick.
        So for a PX4 sim this is the "lockstep is up" verb, and it returns with one observation
        row already recorded and the captured graph replayed once.

        Idempotent: a second call is a no-op once the run has started. Optional for an
        in-process sim, where :meth:`run` and :meth:`step` drive the same setup.

        Args:
            timeout: Override the assembly's ``preroll_timeout``: seconds to wait for the peer
                to establish lockstep, 30 s standalone, 120 s under Isaac Sim. ``None`` keeps
                it. This is where the unbounded time is: physics reset and the Kit warm-up have
                bounds by their own nature.

        Raises:
            RuntimeError: Called outside the ``Sim`` context manager, or the run ended during
                setup, for example when PX4 never connected on ``:4560``.
        """
        if self._orch is None:
            raise RuntimeError("Sim.start() called outside the context manager (use `with na.Sim(...) as sim:`)")
        if self._ran:
            return
        if timeout is not None:
            self._orch.preroll_timeout = float(timeout)
        if not self.step():
            raise RuntimeError("the sim ended during setup (did PX4 connect on :4560?)")

    def run(self) -> None:
        """Run the sim **synchronously** to completion on the calling thread: exhaust the tick
        generator, setup, loop and teardown. Bounded by ``max_steps``, the operator's mission end, or,
        for PX4, the peer disconnecting; ``stop()`` ends it early.

        Set the mission via ``sim.operator`` first, then read ``sim.physics[name]`` or
        ``sim.sensors[name]``, with ``.latest()`` or ``.history()``, after it returns. A script that
        wants to observe or command mid-flight uses :meth:`step` or :meth:`wait_until` instead.

        Raises:
            RuntimeError: Called outside the ``Sim`` context manager.
        """
        if self._orch is None:
            raise RuntimeError("Sim.run() called outside the context manager (use `with na.Sim(...) as sim:`)")
        self._ran = True
        # Blocks: fires on_tick, operator.tick, each step; closes the recorder plus controller in its finally.
        self._orch.run()

    def step(self) -> bool:
        """Advance the sim one control tick on the calling thread. Deterministic: the predicate or state
        you read between steps lands at exact tick boundaries, with no wall-clock.

        The one driving verb for both control kinds: an in-process autopilot and a host-boundary one,
        PX4, alike advance one tick per call. Set the mission via ``sim.operator`` first; then step and
        read ``sim.physics[...]`` between steps. Returns ``False`` when the run has ended: mission
        complete, ``max_steps``, stopped, or the peer disconnected.

        Raises:
            RuntimeError: Called outside the ``Sim`` context manager.
        """
        if self._orch is None:
            raise RuntimeError("sim.step() called outside the context manager (use `with na.Sim(...) as sim:`)")
        self._ran = True
        return self._orch.step()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- ground-truth observation: North-East-Down (NED) free; world Z-up --
    @property
    def physics(self) -> ChannelMap:
        """Name-keyed view over the full recorded model state: every body and joint, by the model's
        labels. Component-kind access: what the *physics* component records.

        ``sim.physics["body_frd"]`` or ``sim.physics["rotor_1"]`` → a body channel, whose ``.latest()``
        and ``.history()`` give :class:`~nexus._src.recording.BodyState`;
        ``sim.physics["rotor_1_joint"]`` → a joint channel,
        :class:`~nexus._src.recording.JointState`. Ground truth in the world frame, Z-up FLU, not an
        autopilot's Extended Kalman Filter (EKF) estimate. Use :attr:`base_body` for the base body
        without hardcoding a name: ``sim.physics[sim.base_body].latest()``.

        Raises:
            RuntimeError: ``observe=False``, so no ``Recorder`` attached.
        """
        if self._recorder is None:
            raise RuntimeError("Sim(observe=False): no Recorder attached")
        return ChannelMap(self._recorder.channels, ("physics/body/", "physics/joint/"))

    @property
    def sensors(self) -> ChannelMap:
        """Name-keyed view over the recorded sensor channels: ``sim.sensors["imu"]`` and so on.
        Component-kind access: what each *sensor* instance records; keys are flat instance names,
        so a future redundant setup reads ``sim.sensors["imu_bosch"]`` beside ``sim.sensors["imu_murata"]``.

        Each value is a :class:`~nexus._src.recording.RecordChannel`, with ``.latest()`` and ``.history()``,
        of that sensor's recorded measurement stream.

        Raises:
            RuntimeError: ``observe=False``, so no ``Recorder`` attached.
        """
        if self._recorder is None:
            raise RuntimeError("Sim(observe=False): no Recorder attached")
        return ChannelMap(self._recorder.channels, ("sensors/",))

    @property
    def base_body(self) -> str:
        """The base body's label, discovered from the model, not hardcoded: the default vehicle
        entity for ``sim.physics[sim.base_body]`` and the sim clock :meth:`wait_until` and :meth:`sleep` use.
        """
        return self._orch.physics.base_body

    # -- sim-time waits, robust to the sim's real-time factor --
    def _clock_t(self) -> float:
        """Current sim time, the lockstep clock: the sim-time budget basis for the waits that follow.

        Read straight off the orchestrator's ``Clock``, which holds it host-side, so this is free:
        :meth:`wait_until` and :meth:`sleep` call it *every* control tick, and reading it off the
        recorder's base body channel instead meant a device-to-host copy per tick, ~0.08 ms against
        a 4 ms budget, 6% of a PX4 run's wall time. Same quantity either way: the recorder writes
        ``t = dt × row counter`` and the clock advances ``dt`` per tick, and both waits use it
        differentially, as ``t - t0``, so the one-row offset between them cancels.

        The Recorder is still required, unchanged: waiting on a sim you can't observe is a
        programming error, and the predicate almost always reads ``sim.physics[...]``.

        Returns 0.0 before the first tick; sim time starts at 0.
        """
        if self._base_ch is None:
            raise RuntimeError("Sim(observe=False): no Recorder attached (wait_until/sleep need observation)")
        return self._orch.clock.now().sim_time

    def wait_until(self, predicate: Callable[[], bool], sim_timeout: float) -> None:
        """Step the sim until ``predicate()`` is true, or ``sim_timeout`` sim seconds elapse.

        Deterministic: this drives the sim itself, so it evaluates the predicate at exact tick
        boundaries and there is no wall-clock in the loop. Budgets in sim time, the lockstep clock
        from ``sim.physics[sim.base_body].latest().t``, so a driver's timeouts are invariant to how
        fast the sim actually runs.

        Args:
            predicate: Zero-arg callable evaluated after each control tick.
            sim_timeout: Budget in sim seconds, measured by advance of the sim clock, the lockstep,
                before giving up.

        Raises:
            TimeoutError: ``sim_timeout`` sim seconds elapsed without the predicate becoming
                true, or the run ended first: mission complete, ``max_steps``, or the peer
                disconnected.
        """
        t0 = self._clock_t()
        while True:
            ran = self.step()
            if predicate():
                return
            if not ran:
                raise TimeoutError("the run ended before the condition was met")
            if self._clock_t() - t0 >= sim_timeout:
                raise TimeoutError(f"condition not met within {sim_timeout}s sim-time")

    def sleep(self, sim_seconds: float) -> None:
        """Step the sim until ``sim_seconds`` of sim time elapse, not wall-clock.

        Measures elapsed time by the advance of the sim clock, the lockstep, so the wait is
        invariant to the sim's real-time factor. Returns early if the run ends.

        Args:
            sim_seconds: Amount of sim time to wait, in seconds.
        """
        t0 = self._clock_t()
        while self._clock_t() - t0 < sim_seconds:
            if not self.step():
                break

    def results(self) -> dict:
        """Run stats from the last in-process ``run()``: the orchestrator's ``run_stats``, which are
        ``control_steps``, steady ``rtf`` and ``full_rtf``. Empty before a run completes.

        Returns:
            dict: The run-stats mapping, for example ``control_steps``, ``rtf`` and ``full_rtf``;
                empty if no run has completed.
        """
        return dict(getattr(self._orch, "run_stats", {}) or {}) if self._orch is not None else {}

    def artifacts(self) -> dict:
        """Return the artifacts produced by this run.

        The Rerun ``.rrd`` this run wrote, merged with whatever the active controller
        contributes. Stays generic: it doesn't name controller-specific artifacts
        such as the PX4 ``.ulg``; each controller declares its own via an optional
        ``artifacts()`` method.

        Returns:
            dict: At least ``{"rrd": <path or None>}``, the ``.rrd`` path, or ``None``
                if no recorder ran, updated with any controller-contributed entries.
        """
        orch = self._orch
        logger = getattr(orch, "logger", None) if orch is not None else None
        out = {"rrd": logger.rrd_path if logger is not None else None}
        controller = getattr(orch, "controller", None) if orch is not None else None
        contribute = getattr(controller, "artifacts", None) if controller is not None else None
        if callable(contribute):
            out.update(contribute())
        return out

    def stop(self) -> None:
        """Tear the run down cooperatively: one path for both control kinds.

        Signals the orchestrator to stop, then closes the tick generator so its teardown runs:
        the Real-Time Factor (RTF) stamp, ``controller.close()``, which for PX4 kills the
        container, and the logging flush. Idempotent: repeat calls are no-ops. ``__exit__`` calls
        it automatically.
        """
        if self._stopped:
            return
        self._stopped = True
        # Close a connected PX4 operator first, since Px4Offboard holds a MAVLink link plus pump thread;
        # the in-process operator holds no resources and has no close().
        op = self._operator
        if op is not None and hasattr(op, "close"):
            op.close()
        if self._orch is None:
            return
        self._orch.stop()
        if self._ran:
            # Step/run-driven: close the tick generator so its teardown, the RTF stamp plus controller
            # and logs close, runs. A no-op if run() already drove to completion and exhausted the generator.
            self._orch.close()
        else:
            # Entered but never driven: close the Logger built at construction so the .rrd flushes or
            # the :9876 server releases; the run's own finally would otherwise have done this.
            self._orch._close_logs()
