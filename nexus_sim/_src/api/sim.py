"""Sim: the one public entry to a run, the program-driving API.

Owns the sim: builds it from a ``LaunchConfig`` via ``build_from_launch`` and drives the ``Loop`` on the
*caller's* thread, one driving model for every controller. A script steps the sim, with ``step``, ``run``,
``wait_until`` or ``sleep``, whether the controller takes setpoints or is PX4: the loop's tick generator
yields once per control tick either way, so a PX4 run is something you drive, not something you watch.
Tears down cooperatively. Every run has a ``Recorder``, and a script reads any recorded history, a
component's or the plant's, by indexing the run with the history's path: ``sim["vehicle/sensors/imu/imu"]``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from nexus_sim._src.api.args import save_run_artifacts, sim_argparser  # noqa: F401  # re-export; defs are import-light
from nexus_sim._src.build.launch import build_from_launch
from nexus_sim._src.config import LaunchConfig
from nexus_sim._src.core import logger
from nexus_sim._src.recording import History, Recorder

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nexus_sim._src.core import Loop
    from nexus_sim._src.core.registry import Registry


class Sim:
    """The one public entry to a Newton sim run.

    Builds an ``Loop`` from a ``LaunchConfig`` and drives its loop on the *caller's*
    thread, one control tick per :meth:`step`, the same for a controller that takes setpoints
    and for PX4 on its Hardware In The Loop (HIL) lockstep. :meth:`start` drives setup as far as lockstep;
    :meth:`run` drives the whole run; :meth:`wait_until` and :meth:`sleep` step until a
    predicate or a sim-time budget. Use it as a context manager: ``__enter__`` builds,
    ``__exit__`` calls :meth:`stop` to tear the run down cooperatively.

    Every run has a :class:`~nexus_sim._src.recording.Recorder`, so :attr:`physics` and
    :attr:`sensors` read sim ground truth, in the world frame, Z-up Forward-Left-Up (FLU), off the
    recorded histories.

    Args:
        vehicle: Catalog vehicle *name*, for example ``"astro_max_fpv"``, or a local .usd path.
        catalog: Path to a catalog that extends the bundled one. ``None`` takes the nearest
            ``nexus.catalog.yaml`` in the working directory or a directory over it, and only the
            catalog bundled in the wheel when no directory holds one.
        scene: Catalog scene *name* to fly in, for example ``"empty"`` for flat ground or the
            ``"slalom"`` obstacle pillars, or a local scene .usd path.
        device: Compute device for the runtime: ``"auto"``, the default, which picks CUDA when
            present, or an explicit ``"cpu"``, for bit-exact determinism, or ``"cuda"``.
        log: Record the run to a ``.rrd``; ``False`` writes no recording, at max speed.
        view: Serve the Rerun recording live on ``:9876`` instead of writing a ``.rrd`` file.
        cache_dir: Override the asset cache directory; ``None`` = the framework default.
        max_steps: Cap the run at this many control steps; ``None`` = the launch-config default.
        rtf: Real-time-factor throttle. ``0``, the default, runs unthrottled, as fast as the
            controller keeps up. ``1.0`` paces the loop to wall-clock for human-in-the-loop
            flying: 1:1 stick feel, and sim-time protocol timeouts align with wall-clock peers.
        layer: Path to an override layer, a local Universal Scene Description (USD) file the run
            composes over the vehicle: it changes a declared value, selects a variant, or drops a
            declaration. A layer that drops the PX4 Software In The Loop (SITL) peer's,
            ``NexusPx4SitlAPI``, flies an autopilot started elsewhere, which dials port 4560.
        registry: The :class:`~nexus_sim.Registry` the build resolves each schema with, its own rows first
            and the install's for the rest, so a test maps a schema to a stand-in or a peer to its fake;
            ``None`` takes the install alone.

    Example:
        ```python
        with Sim("astro_max_base", scene="empty") as sim:
            sim.start()
            body = f"vehicle/body/{sim.base_body}"
            sim.wait_until(lambda: sim.read(body).position[2] > 1.0, sim_timeout=30.0)
        ```
    """

    def __init__(
        self,
        vehicle: str,
        *,
        scene: str,
        catalog: str | None = None,
        geo: str | None = None,
        device: str = "auto",
        log: bool = False,
        view: bool = False,
        debug: bool = False,
        cache_dir: str | None = None,
        solver: str | None = None,
        max_steps: int | None = None,
        rtf: float = 0.0,
        layer: str | None = None,
        registry: Registry | None = None,
    ):
        self._registry = registry  # the registry the build resolves each schema with, or None for the default
        self._launch = LaunchConfig()
        self._launch.set_vehicle(vehicle)  # a catalog *name* or a local .usd path
        self._launch.layer = layer  # an override layer the run composes over the vehicle, or None
        self._launch.catalog = catalog  # None: the run finds its own catalog, see load_catalog
        if solver is not None:  # physics integrator override: mujoco | semi_implicit | featherstone
            self._launch.runtime.solver = solver
        self._launch.set_scene(scene)  # catalog scene, for example the 'slalom' obstacle pillars for sampling-mpc
        if geo is not None:  # override the scene's geodetic origin, for example to fly cesium over any lat/lon
            parts = [float(x) for x in geo.split(",")]
            self._launch.set_geodetic_origin(*parts)  # lat,lon[,alt]; alt is the WGS84 ellipsoidal surface height
        # 'gpu' is an alias of 'cuda'. The launch takes 'auto', 'cpu' or 'cuda', and the resolve picks the
        # device: CUDA when a CUDA device is present, else the CPU. The receipt records the pick.
        if device == "gpu":
            import warp as wp

            if not wp.is_cuda_available():
                logger.warning("device=gpu requested but no CUDA device found, falling back to CPU")
            device = "cuda"
        self._launch.runtime.device = device
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
        self._orch = None
        self._guidance = None  # a launch-built run has none: a flight hands one to _from_loop
        self._ran = False
        self._stopped = False
        self._prebuilt_orch = None  # set by _from_loop, the self-assembled-example entry

    @classmethod
    def _from_loop(
        cls,
        orch: Loop,
        *,
        guidance: object | None = None,
    ) -> Sim:
        """Host a self-assembled :class:`~nexus_sim._src.core.Loop`: the examples' entry.

        An example builds its own loop, its controller plus actuator plus sensors around
        the core components, see ``nexus_sim/examples/controllers/*/assembly.py``, and hands it
        over; the ``Sim`` adds what a script reads and drives: the ``Recorder``, when the loop has
        none, the histories by path; the guidance the flight constructed, which joins the loop; and
        the run lifecycle, ``run``, ``step``, ``stop``, ``results`` and ``artifacts``. Internal: the
        examples' way in until they build through the one builder.

        Args:
            orch: The built loop; its components already carry the renderer and logger.
            guidance: The guidance the flight constructed, for example
                ``MissionGuidance(reached_m=0.3)``. Its stage runs each tick before the controller's
                and writes the setpoint the controller reads. ``None`` flies the controller to the
                setpoint's default. The flight keeps the object it built and sets the mission on it.

        Returns:
            The ``Sim`` handle: use as a context manager, then ``sim.run()``.
        """
        sim = cls.__new__(cls)
        sim._launch = None
        sim._cache_dir = None
        sim._orch = None
        sim._prebuilt_orch = orch
        sim._guidance = guidance
        sim._ran = False
        sim._stopped = False
        return sim

    @classmethod
    def from_args(cls, args: argparse.Namespace, **overrides) -> Sim:
        """Construct a ``Sim`` from a :func:`sim_argparser` namespace, ignoring script-specific extras.

        ``overrides`` win over the namespace, for the ``Sim`` kwargs the shared parser doesn't cover.
        So a script does ``Sim.from_args(args, max_steps=500)``.
        """
        from nexus_sim._src.diagnostics import diagnostics

        diagnostics.configure(args)  # the shared --profile/--trace/--benchmark flags, process-wide
        kw = {
            "vehicle": args.vehicle,
            "catalog": getattr(args, "catalog", None),
            "device": getattr(args, "device", "auto"),
            "scene": args.scene,
            "geo": getattr(args, "geo", None),
            "solver": getattr(args, "solver", None),
            "log": getattr(args, "log", False),
            "view": getattr(args, "view", False),
            "debug": getattr(args, "debug", False),
            "max_steps": getattr(args, "max_steps", None),
            "rtf": getattr(args, "rtf", 0.0),
            "layer": getattr(args, "layer", None),
        }
        kw.update(overrides)
        return cls(**kw)

    # -- context manager: build only; start()/step()/run() drive the run on the caller's thread --
    def __enter__(self) -> Sim:
        if self._prebuilt_orch is not None:
            self._orch = self._prebuilt_orch  # a self-assembled example's loop, via _from_loop
            if self._guidance is not None:
                # The flight's guidance joins the loop's ring: its stage runs before the controller's.
                # The loop builds its ring at the first step, so handing it over here is in time.
                self._orch.guidance = self._guidance
        else:
            # A vehicle that authors RTX sensors starts the Kit render peer here, from the host.
            self._orch = build_from_launch(self._launch, cache_dir=self._cache_dir, components=self._registry)
        if self._orch.recorder is None:
            # A self-assembled loop that brought no Recorder gets one: the loop hands it the plant and
            # each sensor at its first step. dt → the per-row sim time, counter × dt.
            self._orch.recorder = Recorder(dt=self._orch.clock.dt)
        # A run whose controller takes no setpoint, as PX4's does, wires nothing here: a script opens
        # its own client on the offboard link, from sim.ports, after start(), and the run is
        # step-driven the same way as any other.
        return self

    # -- the run's port map --
    @property
    def ports(self) -> Mapping[str, dict]:
        """The run's port map: each link that leaves the run, by name, to the address a script opens
        its client on. The run owns every address. The builder builds a link's end inside the run
        from them, and names here each link whose other end a script holds. An entry is a plain
        mapping of what that client takes, such as ``{"protocol": "udp", "port": 14540,
        "system_id": 1}`` for the ``"offboard"`` link of a PX4 run, which the PX4 page of the
        reference documents. Open a client after ``sim.start()``, and step the sim while it waits
        for an autopilot that runs on the sim's clock. A link with nothing behind it stays out of
        the map, and its lookup raises with the reason.

        Returns:
            The port map, link name to entry.

        Raises:
            RuntimeError: Accessed before entering the ``Sim`` context.
        """
        if self._orch is None:
            raise RuntimeError("enter the Sim context first (`with nx.Sim(...) as sim:`)")
        return self._orch.ports

    def start(self, timeout: float | None = None) -> None:
        """Drive the run's setup, returning once the first control tick has completed.

        One step: the loop's first ``step()`` runs physics reset, the wait for the Kit
        render peer to boot and warm its stage when the vehicle renders, the seed row and the graph
        capture, ``controller.connect()``, which binds the HIL port PX4 dials, and the preroll wait
        for the peer, then flies one tick.
        So for a PX4 sim this is the "lockstep is up" verb, and it returns with one observation
        row already recorded and the captured graph replayed once.

        Idempotent: a second call is a no-op once the run has started. Optional for a run with
        a setpoint controller, where :meth:`run` and :meth:`step` drive the same setup.

        Args:
            timeout: Override the assembly's ``preroll_timeout``: seconds to wait for the peer
                to establish lockstep, 30 s by default. ``None`` keeps it. This is where the
                unbounded time is: physics reset and the Kit peer's start have bounds of their own.

        Raises:
            RuntimeError: Called outside the ``Sim`` context manager, or the run ended during
                setup, for example when PX4 never connected on the HIL port.
            KitPeerError: The vehicle renders and its Kit render peer failed to start.
        """
        if self._orch is None:
            raise RuntimeError("Sim.start() called outside the context manager (use `with nx.Sim(...) as sim:`)")
        if self._ran:
            return
        if timeout is not None:
            self._orch.preroll_timeout = float(timeout)
        if not self.step():
            raise RuntimeError("the sim ended during setup (did PX4 connect on the HIL port?)")

    def run(self) -> None:
        """Run the sim **synchronously** to completion on the calling thread: exhaust the tick
        generator, setup, loop and teardown. Bounded by ``max_steps``, the guidance's mission end, or,
        for PX4, the peer disconnecting; ``stop()`` ends it early.

        Read any history, ``sim[path]``, with ``.history()``, after it returns. A script
        that wants to observe or command mid-flight uses :meth:`step` or :meth:`wait_until` instead.

        Raises:
            RuntimeError: Called outside the ``Sim`` context manager.
            KitPeerError: The vehicle renders and its Kit render peer died mid-flight; the run's
                teardown has run and closed the recording.
        """
        if self._orch is None:
            raise RuntimeError("Sim.run() called outside the context manager (use `with nx.Sim(...) as sim:`)")
        self._ran = True
        # Blocks: runs every stage, the guidance's among them, each step; closes the recorder plus controller in its finally.
        self._orch.run()

    def step(self) -> bool:
        """Advance the sim one control tick on the calling thread. Deterministic: the predicate or state
        you read between steps lands at exact tick boundaries, with no wall-clock.

        The one driving verb for every controller: one that takes setpoints and PX4 alike advance
        one tick per call. Step, then read ``sim[path]`` between steps. Returns ``False`` when the run
        has ended: mission complete, ``max_steps``, stopped, or the peer disconnected.

        Raises:
            RuntimeError: Called outside the ``Sim`` context manager.
            KitPeerError: The vehicle renders and its Kit render peer died mid-flight; the run's
                teardown has run and closed the recording.
        """
        if self._orch is None:
            raise RuntimeError("sim.step() called outside the context manager (use `with nx.Sim(...) as sim:`)")
        self._ran = True
        return self._orch.step()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- the recorded histories, by path: the run is a mapping --
    def _loop(self) -> Loop:
        if self._orch is None:
            raise RuntimeError("enter the Sim context first (`with nx.Sim(...) as sim:`)")
        return self._orch

    def _histories(self) -> dict[str, History]:
        return self._loop().recorder.histories

    def __getitem__(self, path: str) -> History:
        """The recorded history at ``path``, its path below the run's root, as the recording names it.

        ``sim["vehicle/sensors/imu/imu"]`` is the Inertial Measurement Unit (IMU)'s sample stream,
        ``sim["vehicle/controllers/px4/controls"]``
        the controls PX4 wrote, ``sim["vehicle/body/body_frd"]`` the base body's pose and twist and
        ``sim["vehicle/joints/rotor_1_joint"]`` a joint's, ground truth in the world frame, Z-up FLU. Each is a
        :class:`~nexus_sim._src.recording.History`, with ``.history()`` and ``.arrays()``.
        Every history reads one way: a row is a record of ``t``, the tick's sim time, then the fields the
        signal's type declares, ``row["accel"]`` for the IMU and ``row["position"]``, ``row["orientation"]``,
        ``row["linear_velocity"]`` and ``row["angular_velocity"]`` for a body, ``row["q"]`` and ``row["qd"]``
        for a joint; ``history()`` gives every row as one structured array, ``rows["position"][:, 2]``, and
        ``history()[-1]`` the newest row. :meth:`read` gives a signal's live value. Use :attr:`base_body` for the
        base body without hardcoding its name: ``sim[f"vehicle/body/{sim.base_body}"].history()``.

        Raises:
            RuntimeError: Accessed before entering the ``Sim`` context.
            KeyError: No history at ``path``; ``for path in sim`` lists them.
        """
        return self._histories()[path]

    def __iter__(self) -> Iterator[str]:
        """Each recorded history's path, in the order the run registered them."""
        return iter(self._histories())

    def __len__(self) -> int:
        return len(self._histories())

    def __contains__(self, path: object) -> bool:
        return path in self._histories()

    def read(self, path: str) -> Any:
        """The live value of the signal at ``path``, typed by the signal: what its writer wrote on the last tick.

        ``path`` is the writer's path below the run's root then the signal's name, as a history's key, and a
        signal that keeps no history, a camera's frame say, has one too. ``sim.read(f"vehicle/body/{sim.base_body}")``
        is the base body's :class:`~nexus_sim.PoseTwist`, so ``.position[2]`` is its height, and
        ``sim.read("vehicle/controllers/px4/controls")`` the 16 controls PX4 wrote, one array. A device signal
        reads without its leading axis of one, as an instance of its struct or an array of its other axes, and
        a host signal as the object its writer wrote. The value is a copy, which later ticks leave alone.

        Raises:
            RuntimeError: Called before entering the ``Sim`` context.
            KeyError: No component writes a signal at ``path``.
        """
        return self._loop().signals[path].read()

    @property
    def time(self) -> float:
        """The sim time of the last tick, seconds: 0.0 before the first tick.

        Read off the loop's host clock, so it costs no device copy, and :meth:`wait_until` and :meth:`sleep`
        read it every tick. Every history's newest row carries the same time.

        Raises:
            RuntimeError: Read before entering the ``Sim`` context.
        """
        return self._loop().clock.now().sim_time

    @property
    def base_body(self) -> str:
        """The base body's label, discovered from the model, not hardcoded: the body under ``vehicle/body/``
        a script reads the vehicle's pose from, and the one the sim clock :meth:`wait_until` and :meth:`sleep` use.
        """
        return self._orch.physics.base_body

    # -- sim-time waits, robust to the sim's real-time factor --
    def wait_until(self, predicate: Callable[[], bool], sim_timeout: float) -> None:
        """Step the sim until ``predicate()`` is true, or ``sim_timeout`` sim seconds elapse.

        Deterministic: this drives the sim itself, so it evaluates the predicate at exact tick
        boundaries and there is no wall-clock in the loop. Budgets in sim time, the lockstep clock,
        so a driver's timeouts are invariant to how fast the sim actually runs.

        Args:
            predicate: Zero-arg callable evaluated after each control tick.
            sim_timeout: Budget in sim seconds, measured by advance of the sim clock, the lockstep,
                before giving up.

        Raises:
            TimeoutError: ``sim_timeout`` sim seconds elapsed without the predicate becoming
                true, or the run ended first: mission complete, ``max_steps``, or the peer
                disconnected.
        """
        t0 = self.time
        while True:
            ran = self.step()
            if predicate():
                return
            if not ran:
                raise TimeoutError("the run ended before the condition was met")
            if self.time - t0 >= sim_timeout:
                raise TimeoutError(f"condition not met within {sim_timeout}s sim-time")

    def sleep(self, sim_seconds: float) -> None:
        """Step the sim until ``sim_seconds`` of sim time elapse, not wall-clock.

        Measures elapsed time by the advance of the sim clock, the lockstep, so the wait is
        invariant to the sim's real-time factor. Returns early if the run ends.

        Args:
            sim_seconds: Amount of sim time to wait, in seconds.
        """
        t0 = self.time
        while self.time - t0 < sim_seconds:
            if not self.step():
                break

    def results(self) -> dict:
        """Run stats from the last ``run()``: the loop's ``run_stats``, which are
        ``control_steps``, steady ``rtf`` and ``full_rtf``. Empty before a run completes.

        Returns:
            dict: The run-stats mapping, for example ``control_steps``, ``rtf`` and ``full_rtf``;
                empty if no run has completed.
        """
        return dict(self._orch.run_stats) if self._orch is not None else {}

    def artifacts(self) -> dict:
        """Return the artifacts produced by this run.

        The Rerun ``.rrd`` this run wrote, merged with what every component and every peer of the run
        contributes through its ``artifacts()``: PX4's controller names its ``.ulg``, the PX4 peer its console
        log. Stays generic: it names no artifact itself.

        Returns:
            dict: At least ``{"rrd": <path or None>}``, the ``.rrd`` path, or ``None``
                if no recorder ran, updated with the entries the components and the peers contribute.
        """
        if self._orch is None:
            return {"rrd": None}
        out = {"rrd": self._orch.logger.rrd_path if self._orch.logger is not None else None}
        for source in (*self._orch.components(), *self._orch.peers):
            out.update(source.artifacts())
        return out

    def stop(self) -> None:
        """Tear the run down cooperatively: one path for every controller.

        Signals the loop to stop, then closes the tick generator so its teardown runs:
        the Real-Time Factor (RTF) stamp, ``controller.close()``, the stop of every peer the build
        started, the PX4 container, and the logging flush. Idempotent: repeat calls are no-ops.
        ``__exit__`` calls it automatically.
        """
        if self._stopped:
            return
        self._stopped = True
        if self._orch is None:
            return
        self._orch.stop()
        # Step/run-driven, this closes the tick generator so its teardown, the RTF stamp plus renderer,
        # controller, peers and logs close, runs; a no-op if run() drove to completion. Entered but
        # never driven, it stops the peers started at build, the renderer's and PX4, and closes the
        # Logger built at construction, so the .rrd flushes or the :9876 server releases.
        self._orch.close()
