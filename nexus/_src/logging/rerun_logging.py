"""The one Rerun recording the whole sim logs to, per architecture.md §10.

`newton-logging` owns the entire Rerun stack so the rest of the framework stays
logging-agnostic:

* **the scene**: NVIDIA Newton's ``ViewerRerun`` logs the physics ``Model``/``State``,
  geometry + how it evolves, via ``log_state``. This module reuses it rather than reinvent
  scene logging, and crucially it logs **in-process** to the recording it also
  serves, so the physics state never round-trips over a gRPC socket to a server in
  the same process.
* **events**: a `TextLog` handler attached to the ``newton`` logger, the one every
  component already uses via ``from nexus._src.core import logger``, routes all log
  records into the same recording's ``logs/sim`` panel. Components need no changes:
  they keep calling ``logger.info(...)``.

The recording, app ID ``nexus`` and recording ID ``nexus``, either serves
over gRPC on :9876, where a native Rerun viewer connects with
``rerun --connect rerun+http://127.0.0.1:9876/proxy``, or goes to an
``.rrd`` file for replay/forensics. Every producer writes through this one Logger,
in-process on the component-owned logging seam, so serve and file mode carry the same
content: no second producer, no sidecar files, no merge.

`RerunLogger` fills the orchestrator's optional ``recorder`` seam: the core loop
hands it each tick's state, and it drives ``ViewerRerun.log_state``. It's
output-only and so can't perturb determinism.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from newton.viewer import ViewerRerun
from rerun.archetypes import TextLog

from nexus._src.core import logger

APP_ID = "nexus"
RECORDING_ID = "nexus"
GRPC_PORT = 9876
WEB_PORT = 9090
SERVER_URI = f"rerun+http://127.0.0.1:{GRPC_PORT}/proxy"
LOG_ENTITY = "logs/sim"
RTF_ENTITY = "run/rtf"  # live real-time-factor readout: a small markdown doc, re-logged ~1 Hz
SETTINGS_ENTITY = "run/settings"  # the run's effective config: one static markdown doc
PROFILE_ENTITY = "run/profile"  # end-of-run stats: steps, Real Time Factor (RTF), the loop profiler's breakdown
# End-of-run Recorder dump root: every declared channel quantity lands at
# recording/<channel key>/<field> as a Scalars time series via send_columns, at zero per-tick cost.
RECORDING_ROOT = "recording"
# Cameras nest under the per-tick base-body entity so the pinhole frustum rides the body pose at the
# body-log rate, since a camera-rate world transform visibly lags the mesh: vehicle/body gets a Transform3D
# every logged tick; each camera is a STATIC child, authored mount + Pinhole, under it.
BODY_ENTITY = "vehicle/body"
FPV_ENTITY = f"{BODY_ENTITY}/cameras"  # RtxCameraSensor -> log_image at <FPV_ENTITY>/<name>
# Debug coordinate-axis triads: the base body, index 0, gets a longer triad so it stands out; the other
# bodies, the actuator links, get a shorter one. Shaft radius is the rerun default, the same for every body.
DEBUG_BASE_AXIS_LENGTH = 0.3  # m: base-body triad length
DEBUG_LIMB_AXIS_LENGTH = 0.16  # m: other bodies' triad length, a bit shorter
DEBUG_LABEL_RADIUS = 0.02  # m: the body-name label marker, a dot at each body origin, same for every body


def _default_rrd_path() -> str:
    return os.path.expanduser(f"~/.cache/nexus/logs/nexus-{time.strftime('%Y%m%d-%H%M%S')}.rrd")


def recording_path(name: str) -> str:
    """Stable path for a named recording, for examples and CI artifacts, under the nexus cache, which
    keeps ``.rrd`` files out of the working tree. ``RerunLogger`` creates the parent dir.
    """
    return os.path.expanduser(f"~/.cache/nexus/recordings/{name}.rrd")


# The vehicle's mesh entity in NVIDIA Newton's ViewerRerun scene: shape_0 is the ground plane,
# excluded; shape_1 is the base body. The 3D eye tracks it so the vehicle stays centered.
VEHICLE_SHAPE_ENTITY = "/model/shapes/shape_1"


# Channel-key first segment → the debug group tab title. Any other prefix a future component
# records under, for example actuators/, becomes its own capitalized group tab automatically.
_GROUP_TITLES = {"physics": "Physics", "sensors": "Sensors"}


def _recording_tabs(recording: dict | None, cameras: dict | None) -> list:
    """The component-kind debug tab tree, group ▸ instance ▸ quantity, derived from the Recorder's
    channel keys, ``physics/body/<name>``, ``sensors/<name>``, and so on, so the tabs mirror the access
    surface: ``sim.physics["body_frd"]`` → Physics ▸ body_frd; ``sim.sensors["imu"]`` → Sensors ▸ imu.

    ``recording`` maps channel key → ``(source, [field names])``, from the recorder's Rerun adapter; each
    quantity tab is a ``TimeSeriesView`` on its ``recording/<key>/<field>`` entity. ``cameras`` maps a
    registered camera name → its source class: cameras are sensors whose one "quantity" is the live
    feed, since frames can't ride the device ring, they're already logged at sensor rate, so each becomes
    an instance tab under Sensors holding its 2D view.
    """
    groups: dict[str, dict[str, object]] = {}
    for key, (source, fields) in (recording or {}).items():
        parts = key.split("/")
        group = _GROUP_TITLES.get(parts[0], parts[0].capitalize())
        # Sensor instance tabs carry the impl class, as in imu followed by ImuSensor in parentheses:
        # provenance without a click layer; physics instances are all the one plant, so the suffix
        # would be noise there.
        title = f"{parts[-1]} ({source})" if group == "Sensors" and source else parts[-1]
        inst = groups.setdefault(group, {})
        if title in inst:  # a body and a joint sharing a leaf name: disambiguate by the kind segment
            title = f"{parts[-1]} ({'/'.join(parts[1:-1])})"
        inst[title] = rrb.Tabs(
            *[rrb.TimeSeriesView(origin=f"{RECORDING_ROOT}/{key}/{f}", name=f) for f in fields],
            name=title,
        )
    for name, source in (cameras or {}).items():
        title = f"{name} ({source})" if source else name
        groups.setdefault("Sensors", {})[title] = rrb.Spatial2DView(origin=f"{FPV_ENTITY}/{name}", name=title)
    order = ["Physics", "Sensors"]
    titles = [g for g in order if g in groups] + sorted(g for g in groups if g not in order)
    return [rrb.Tabs(*groups[g].values(), name=g) for g in titles]


def _blueprint(
    exclude: list[str] | None = None,
    cameras: dict[str, str | None] | None = None,
    *,
    recording: dict | None = None,
    has_settings: bool = False,
    has_profile: bool = False,
) -> rrb.Blueprint:
    """Two rows: [RTF-over-Scene | Logs|Settings] on top, the full-width debug tab tree below.

    The top row is two equal columns: the left stacks the one-line live RTF readout over the 3D
    scene, with [1, 8] shares, proportional, since rerun has no fixed-pixel rows, so the RTF row is
    "about one line of markdown" tall; the right holds the event log + the run-settings doc, when
    provided, as tabs. The bottom row is the component-kind debug tab tree, see :func:`_recording_tabs`,
    at full viewer width: time series are wide, so they get the whole span. Camera feeds live there from
    registration; every recorded quantity joins after the end-of-run dump. Each camera view's origin
    sits AT its pinhole entity, ``cameras/<name>``: a 2D view whose root sits higher in the tree than
    a Pinhole makes rerun refuse the image with "Can't visualize 2D content with a pinhole ancestor
    that's embedded within the 2D view" as the error. Registrations arrive over the run, via
    ``log_camera`` and ``show_recording``, so the Logger re-sends the blueprint then; before any, a plain
    First Person View (FPV) placeholder keeps the layout for the standalone path. The scene's 3D eye
    TRACKS the vehicle mesh, so the flight stays centered.
    """
    tabs = _recording_tabs(recording, cameras)
    bottom = rrb.Tabs(*tabs) if tabs else rrb.Spatial2DView(origin=FPV_ENTITY, name="FPV")
    log_tabs: list = [
        rrb.TextLogView(
            origin="logs",
            name="Logs",
            # No EntityPath column: every in-process row lives at logs/sim, so it's pure noise;
            # out-of-process producers merge under logs/ too, and the level + body carry the story.
            columns=rrb.TextLogColumns(text_log_columns=["loglevel", "body"]),
        )
    ]
    if has_settings:
        log_tabs.append(rrb.TextDocumentView(origin=SETTINGS_ENTITY, name="Settings"))
    if has_profile:  # teardown-only: the loop profiler's stats exist once the run ended
        log_tabs.append(rrb.TextDocumentView(origin=PROFILE_ENTITY, name="Profile"))
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.TextDocumentView(origin=RTF_ENTITY, name="RTF"),
                    # Static, world-attached shapes, chiefly the ground plane, are large flat
                    # sheets that occlude the vehicle; exclude them from the Scene view. Their entity
                    # names differ per runtime, by batch ordinals, so the exclusions come from a read
                    # back off the viewer. The run/ + recording/ namespaces are non-spatial, docs and
                    # scalar series, so they're scoped out too.
                    rrb.Spatial3DView(
                        origin="/",
                        name="Scene",
                        contents=[
                            "+ $origin/**",
                            *(exclude or ["- /model/shapes/shape_0"]),
                            f"- /{RECORDING_ROOT}/**",
                            "- /run/**",
                        ],
                        eye_controls=rrb.archetypes.EyeControls3D(tracking_entity=VEHICLE_SHAPE_ENTITY),
                    ),
                    row_shares=[1, 8],
                ),
                rrb.Tabs(*log_tabs),
                column_shares=[1, 1],
            ),
            bottom,
            row_shares=[1, 1],
        ),
        rrb.TimePanel(state="collapsed"),
        collapse_panels=True,
    )


def scene_only_blueprint() -> rrb.Blueprint:
    """Just the 3D scene: no event-log panel, side panels collapsed. A clean layout for
    docs/example recordings where the timeline scene is the whole story.

    Known cosmetic noise, upstream and not fixable here: NVIDIA Newton's ``ViewerRerun`` caches each
    geometry asset at ``/geometry/{kind}_N`` then clears it, since the mesh inlines under ``/model``,
    leaving dangling transform-less entities. The viewer spams a "failed to query transformations:
    missing transform on entity '/geometry/{kind}_N'" warning. Scoping this view away from
    ``/geometry`` doesn't silence the warning, because it's emitted recording-wide, not per view, so
    this view no longer scopes it away. The fix belongs upstream; see the post-refactor-followups plan.
    """
    return rrb.Blueprint(
        rrb.Spatial3DView(origin="/", name="Scene"),
        collapse_panels=True,
    )


class _RerunHandler(rr.LoggingHandler):
    """Route the stdlib ``newton`` logger into the recording's ``logs/sim`` panel,
    tagged with the emitting module, mirroring the bridge's handler.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "console_only", False):
            return  # for example the profiler/RTF end-of-run reports: the Profile tab carries them in-viewer
        level = self.LVL2NAME.get(record.levelno)
        if level is None:
            level = rr.components.TextLogLevel(record.levelname)
        # "sim/" namespaces the sim's rows next to PX4's "[px4/" prefixed ones in the shared Logs pane.
        rr.log(LOG_ENTITY, TextLog(f"[sim/{record.module}] {record.getMessage()}", level=level))


def attach_log_handler() -> None:
    """Tee the ``newton`` logger into the active recording; idempotent. Call after
    the recording exists; ``RerunLogger`` does this for you.
    """
    if not any(isinstance(h, _RerunHandler) for h in logger.handlers):
        logger.addHandler(_RerunHandler(LOG_ENTITY))


class Logger:
    """The recording apparatus + the shared log calls: infra, owns no semantic quantity.

    Two roles. First, it owns the one Rerun **recording** every component writes into: serve or file
    but never both, the blueprint, and routing the ``newton`` logger's events into ``logs/sim``. Second,
    it exposes the **shared** log calls components use directly: :meth:`log_state`, the generic scene
    via NVIDIA ``ViewerRerun``, :meth:`set_time`, :meth:`log_image`. Semantic overlays such as
    ``physics/trajectory``, ``operator/reference``, ``controller/mpc_horizon`` live in each component's
    own log step, not here. The orchestrator holds one ``Logger | None``; ``None`` is the single, clean
    off-switch: no recording, no per-tick log fan-out, max benchmark/CI speed.

    **serve or file, never both**: a live gRPC server and a *complete* ``.rrd`` can't both come out
    of one process, because rerun's serve and file sinks are mutually exclusive. The public surface is
    one knob, ``viewer``: Output / the command-line ``--viewer``/``--no-viewer`` / ``Sim(viewer=)``.

    * **viewer**, ``serve=True``, the default: serve the recording live on :9876 for a native
      viewer, ``rerun --connect rerun+http://127.0.0.1:9876/proxy``. No file gets written;
      to keep one, save it from the connected viewer.
    * **not viewer**, ``serve=False``: write the full ``.rrd`` to disk.

    Every producer writes through this one Logger, in-process on this seam, so both modes
    carry the same content.

    The scene logs **in-process**, with no self-gRPC, via NVIDIA Newton's
    ``ViewerRerun.log_state`` and every ``newton`` logger event lands in ``logs/sim``.

    Parameters
    ----------
    model: NVIDIA Newton ``Model`` to visualise; ``None`` → events only, no scene.
    serve: ``True`` serves live on :9876; ``False`` writes the ``.rrd``.
    record_to_rrd: the ``.rrd`` path when ``serve=False``; defaults to a timestamped path
        under ``~/.cache/nexus/logs/``.
    blueprint: a Rerun ``rrb.Blueprint`` for the viewer layout; the default is the standard
        scene+events layout. Pass a custom one to, for example, show only the 3D scene without the
        event-log panel: ``scene_only_blueprint``, which the blueprint-shape tests use; examples run
        the default.
    settings: the run's effective configuration, a mapping or a pre-rendered markdown string,
        logged once, statically, as the viewer's Settings tab. ``None`` = no Settings tab.
    """

    def __init__(
        self,
        model=None,
        *,
        serve: bool = True,
        record_to_rrd: str | None = None,
        blueprint: rrb.Blueprint | None = None,
        log_hz: float | None = 50.0,
        debug: bool = False,
        settings=None,
    ):
        # log_hz, default 50 Hz, caps the per-tick log rate: the orchestrator decimates its per-tick log
        # fan-out, physics' scene + trail, to this interval, so every per-tick logger rides one clock. The
        # full scene serialize + state read is the dominant per-tick cost and a human scrubs an .rrd at
        # human rates, so full-rate logging just bloats the recording + tanks recorded-run RTF, PX4
        # lockstep most visibly. ``None`` = no cap. The horizon keeps its own snapshot_every; markers are
        # event-driven.
        self.log_interval = (1.0 / log_hz) if log_hz else 0.0
        # Camera frames: the producing SENSOR owns the rate, by sim-time decimation; log_image just hands
        # the frame to a background encode worker: the JPEG compress, ~10-20 ms at 720p, must never run
        # on the lockstep thread, and a newest-wins mailbox drops frames if the encoder falls behind.
        self._img_log_warned = False
        self._sim_time: float | None = None
        self._img_queue: queue.Queue | None = None
        self._img_worker = None
        self._serve = serve
        # serve or file, never both: a live gRPC server and a *complete* .rrd can't both come out of
        # one process. Rerun's serve and file sinks are mutually exclusive: `set_sinks` replaces
        # `serve_grpc`, and a serving endpoint isn't a `set_sinks` sink; ViewerRerun's serve-mode
        # record_to_rrd writes only a stub. So serve=True serves only; serve=False writes the
        # .rrd. To keep a file of a live session, save it from the connected viewer.
        self._rrd_path: str | None = None
        self._blueprint = blueprint  # None -> built after set_model, with exclusions read from the viewer
        self._auto_blueprint = blueprint is None
        self._exclusions: list[str] | None = None
        # Registered RTX cameras, name -> impl class: instance tabs under Sensors, see _recording_tabs.
        self._cameras: dict[str, str | None] = {}
        self._recording: dict | None = None  # the end-of-run dump's tab tree, from show_recording
        self._has_settings = settings is not None
        self._has_profile = False  # set by log_profile at teardown, when the Profile tab shows up
        # Debug, axes-only, mode: log each body's coordinate-frame triad instead of its mesh. The meshes make
        # up nearly the whole of an .rrd's size, the astro-max geometry, so a debug recording is a small
        # fraction the size, for quick flight-shape inspection where the frames tell the story. This skips
        # ViewerRerun.set_model, so no geometry gets registered, and log_state draws a Red Green Blue (RGB)
        # axis triad per body from the state, in _log_axes.
        self._debug = bool(debug)
        # Body Universal Scene Description (USD) prim paths, Newton's ``body_label``, for example
        # ``/astro_max/Geometry/body_frd/rotor_1``, used verbatim as the Rerun entity paths so the debug
        # scene mirrors the USD hierarchy; the leaf is the label.
        self._body_paths = (
            [str(k) for k in (getattr(model, "body_label", None) or [])] if (debug and model is not None) else None
        )

        # The one timeline is duration-typed, so it reads as 14.2 s rather than an epoch date, and every
        # producer stamps it. NVIDIA Newton's ViewerRerun hardcodes
        # timestamp-typed stamps on the same "time" timeline, in the viewer_rerun.py ctor + begin_frame,
        # and mixed types malform the stream when it closes, so shim rr.set_time to rewrite them: the
        # same neuter-the-side-channel pattern as _build_viewer; the real fix belongs upstream: a
        # parametrizable timeline.
        if not getattr(rr.set_time, "_nexus_duration_shim", False):
            _orig_set_time = rr.set_time

            def _set_time(timeline, *args, **kwargs):
                if timeline == "time" and "timestamp" in kwargs:
                    kwargs["duration"] = kwargs.pop("timestamp")
                return _orig_set_time(timeline, *args, **kwargs)

            _set_time._nexus_duration_shim = True
            rr.set_time = _set_time

        if serve:
            # ViewerRerun(serve_web_viewer=True) starts the in-process gRPC server *and* a web
            # viewer on :9090, which opens a browser. The sim wants only the native-viewer gRPC server,
            # since the web viewer is "too slow at these data rates" per §10, so neuter the web-viewer
            # leg: serve_grpc(:9876) still runs, in-process, no self-gRPC.
            self._viewer = self._build_viewer("serve_web_viewer", record_to_rrd=None)
        else:
            # serve_web_viewer=False would launch a native viewer, which needs a display; neuter the native-viewer launch
            # so ViewerRerun only does rr.init + rr.save(.rrd): the full recording on disk.
            self._rrd_path = record_to_rrd or _default_rrd_path()
            os.makedirs(os.path.dirname(self._rrd_path), exist_ok=True)
            self._viewer = self._build_viewer("spawn", record_to_rrd=self._rrd_path)

        if model is not None and not self._debug:
            self._viewer.set_model(model)  # register the geometry, the meshes; skipped in debug → axes-only
        if settings is not None:
            self._log_settings(settings)  # one static markdown doc: the Settings tab's content
        if self._blueprint is None:
            self._exclusions = _viewer_static_exclusions(self._viewer) if self._viewer else None
            self._blueprint = _blueprint(self._exclusions, has_settings=self._has_settings)
        rr.send_blueprint(self._blueprint, make_active=True)
        attach_log_handler()  # all later `newton` logger events land in logs/sim

        if serve:
            logger.info(f"Rerun: serving on :{GRPC_PORT}, native viewer: `rerun --connect {SERVER_URI}`")
        else:
            logger.info(f"Rerun: recording to {self._rrd_path}")

    @property
    def rrd_path(self) -> str | None:
        """The ``.rrd`` written this run; ``None`` in serve mode, since serve and a full file are
        mutually exclusive; save a live session from the viewer.
        """
        return self._rrd_path

    def _build_viewer(self, neuter: str, *, record_to_rrd):
        """Construct ViewerRerun with its side-channels neutered, so it does exactly one clean thing.

        This neuters two things, both by temporarily swapping a ``rerun`` module attribute that
        ViewerRerun's ctor calls; ViewerRerun looks these up on ``rr`` at call time, so the swap takes:

        * ``neuter``: the ``serve_web_viewer`` web leg or the native-viewer launch, replaced
          with a no-op so ViewerRerun doesn't open a window / web server.
        * the **blueprint**: ViewerRerun's ctor logs its own ``_get_blueprint()``, an ``origin="/"`` 3D
          view, as ``default_blueprint`` on ``rr.init`` and ``rr.save``. RerunLogger *owns* the blueprint,
          it sends ``self._blueprint`` immediately after, so this strips ``default_blueprint`` from those
          calls. Otherwise the recording carries two competing blueprints, ViewerRerun's default plus
          RerunLogger's, and a viewer that honours the default re-introduces the panels/entities, the
          event log and the ``/geometry`` cache husks, that RerunLogger deliberately scoped out.
        """
        import inspect

        orig_neuter = getattr(rr, neuter)
        orig_init, orig_save = rr.init, rr.save

        def _drop_default_blueprint(fn):
            def wrapped(*a, **k):
                k.pop("default_blueprint", None)
                return fn(*a, **k)

            return wrapped

        setattr(rr, neuter, lambda *a, **k: None)
        rr.init, rr.save = _drop_default_blueprint(orig_init), _drop_default_blueprint(orig_save)
        try:
            kwargs = {
                "app_id": APP_ID,
                "rec_id": RECORDING_ID,
                "grpc_port": GRPC_PORT,
                "web_port": WEB_PORT,
                "serve_web_viewer": (neuter == "serve_web_viewer"),
                "keep_historical_data": True,  # keep history so the timeline is scrubbable
                "record_to_rrd": record_to_rrd,
            }
            # NVIDIA Newton's ViewerRerun signature varies by version: the workspace pins one
            # version, but the Isaac Sim runtime uses Kit's bundled Newton, 1.2.0, whose ctor
            # drops some kwargs, for example ``rec_id``. Pass only what this build accepts; it still
            # serves grpc on :9876.
            supported = set(inspect.signature(ViewerRerun.__init__).parameters)
            return ViewerRerun(**{k: v for k, v in kwargs.items() if k in supported})
        finally:
            setattr(rr, neuter, orig_neuter)
            rr.init, rr.save = orig_init, orig_save

    # ----- shared log calls: components pass data; the Logger owns all rerun interaction -----
    def set_time(self, sim_time: float) -> None:
        """Set the shared ``time`` timeline for this tick, duration-typed: the viewer renders sim
        seconds, not an epoch date. The orchestrator calls it once at tick start,
        so every component's later overlay lands at the timestamp the scene mesh is on. Components
        never call this; they just log their data.
        """
        self._sim_time = float(sim_time)
        rr.set_time("time", duration=self._sim_time)

    @property
    def debug(self) -> bool:
        """True in axes-only debug mode: sensors add their own debug overlays, axes triads, when set."""
        return self._debug

    def _log_settings(self, settings) -> None:
        """One static markdown ``TextDocument`` at ``run/settings``: the Settings tab's content.

        A mapping renders as one flat dotted-path table, see :func:`_settings_markdown`,
        after a JSON round-trip with ``default=str``, where Paths, numpy scalars, tuples all become
        plain values; a string counts as pre-rendered markdown.
        """
        try:
            import json

            if isinstance(settings, str):
                md = settings
            else:
                md = _settings_markdown(json.loads(json.dumps(settings, default=str)))
            rr.log(SETTINGS_ENTITY, rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN), static=True)
        except Exception as exc:
            self._has_settings = False
            logger.warning(f"Rerun settings doc disabled: {exc}")

    def log_profile(self, stats) -> None:
        """One static markdown table at ``run/profile``: the run's end-of-run stats, meaning control steps,
        steady/full RTF, the always-on loop profiler's per-phase breakdown, rendered the same way as the
        Settings tab. Called at teardown by the orchestrator, since the stats only exist once the loop
        ended; rebuilds the blueprint so the Profile tab shows up.
        """
        try:
            import json

            md = _settings_markdown(json.loads(json.dumps(stats, default=str)), key_header="Property")
            rr.log(PROFILE_ENTITY, rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN), static=True)
            self._has_profile = True
            if self._auto_blueprint:
                self._blueprint = _blueprint(
                    self._exclusions, cameras=self._cameras, recording=self._recording,
                    has_settings=self._has_settings, has_profile=True,
                )  # fmt: skip
                rr.send_blueprint(self._blueprint, make_active=True)
        except Exception as exc:
            logger.warning(f"Rerun profile doc disabled: {exc}")

    def log_rtf(self, rtf: float) -> None:
        """Log the live real-time factor as a one-line markdown doc at ``run/rtf``, stamped on the
        timeline, so the viewer's RTF row always shows the value AT the timeline cursor and scrubbing
        replays it. The orchestrator calls this ~1 Hz wall-clock from its per-tick log seam; the
        timeline is already set.
        """
        try:
            rr.log(RTF_ENTITY, rr.TextDocument(f"RTF: {rtf:.2f}", media_type=rr.MediaType.MARKDOWN))
        except Exception as exc:
            if not getattr(self, "_rtf_log_warned", False):
                self._rtf_log_warned = True
                logger.warning(f"Rerun RTF logging disabled: {exc}")

    def show_recording(self, tree) -> None:
        """Show the recorder's dumped channels in the viewer: the debug tab tree, Physics ▸ instance ▸
        quantity, Sensors ▸ …, rebuilt into the blueprint.

        The recorder's Rerun adapter dumps the rings and hands the tree over, ``channel key →
        (source, [field names])``; the rows themselves are the adapter's, not the Logger's.
        """
        if not tree:
            return
        self._recording = tree
        if self._auto_blueprint:
            self._blueprint = _blueprint(
                self._exclusions, cameras=self._cameras, recording=tree,
                has_settings=self._has_settings, has_profile=self._has_profile,
            )  # fmt: skip
            rr.send_blueprint(self._blueprint, make_active=True)

    def log_static_frame(self, entity: str, local_translation, local_mat3_rows, *, label: str | None = None) -> None:
        """One STATIC coordinate frame under *entity*, a child of a body entity: a fixed local
        ``Transform3D`` with fixed-length dedicated frame axes plus an origin dot carrying the
        centered label.

        Logged once at setup: the debug scene logs a per-tick ``Transform3D`` on each body entity, and
        children inherit the parent transform, so a rigidly mounted sensor frame needs no re-logging.
        ``local_mat3_rows`` is the row-vector local rotation, with rows = axes in the parent frame.
        """
        import numpy as np

        rr.log(
            entity,
            _transform3d(
                np.asarray(local_translation, dtype=np.float32),
                np.asarray(local_mat3_rows, dtype=np.float32).T,  # rerun wants column-vector
                DEBUG_LIMB_AXIS_LENGTH,
            ),
            static=True,
        )
        _log_frame_axes(entity, DEBUG_LIMB_AXIS_LENGTH)
        rr.log(
            entity,
            rr.Points3D(
                [[0.0, 0.0, 0.0]],
                labels=[label or entity.rsplit("/", 1)[-1]],
                show_labels=True,
                radii=[DEBUG_LABEL_RADIUS],
            ),
            static=True,
        )

    def log_camera(
        self,
        entity: str,
        *,
        width: int,
        height: int,
        focal_length_mm: float,
        h_aperture_mm: float,
        v_aperture_mm: float | None = None,
        local_translation=None,
        local_quat_xyzw=None,
        source: str | None = None,
    ) -> str:
        """Log the camera's ``Pinhole``, the frustum and Field Of View (FOV) visualization: once per
        camera, static.

        With ``local_translation``/``local_quat_xyzw``, the authored mount relative to the base
        body, the camera also gets a static local ``Transform3D``: it then rides the per-tick
        ``vehicle/body`` pose with zero per-frame transform traffic. Returns the camera's entity
        path, the ``log_image`` target. ``source`` names the registering sensor class: the camera's
        instance tab under Sensors carries it, as every recorded sensor channel's does.

        USD cameras look down -Z with +Y up = Rerun's Right Up Back (RUB) view coordinates. Pixel focal lengths:
        ``fx = width * focalLength / horizontalAperture`` and ``fy = height * focalLength /
        verticalAperture``, passed separately so a camera with an independently authored vertical
        aperture, non-square pixels, still gets the right vertical FOV.
        """
        name = entity.rsplit("/", 1)[-1]
        entity = f"{FPV_ENTITY}/{name}"
        if local_translation is not None and local_quat_xyzw is not None:
            rr.log(
                entity,
                rr.Transform3D(
                    translation=[float(v) for v in local_translation],
                    rotation=rr.Quaternion(xyzw=[float(v) for v in local_quat_xyzw]),
                ),
                static=True,
            )
        if self._auto_blueprint and name not in self._cameras:
            self._cameras[name] = source
            self._blueprint = _blueprint(
                self._exclusions, cameras=self._cameras, recording=self._recording, has_settings=self._has_settings
            )
            rr.send_blueprint(self._blueprint, make_active=True)
        _log_frame_axes(entity, DEBUG_LIMB_AXIS_LENGTH)  # fixed-length camera frame axes
        fx = float(width) * float(focal_length_mm) / float(h_aperture_mm)
        v_ap = float(v_aperture_mm) if v_aperture_mm else float(h_aperture_mm) * height / width
        fy = float(height) * float(focal_length_mm) / v_ap
        rr.log(
            entity,
            rr.Pinhole(
                focal_length=[fx, fy],
                resolution=[int(width), int(height)],
                camera_xyz=rr.ViewCoordinates.RUB,
                image_plane_distance=0.3,
            ),
            static=True,
        )
        return entity

    def log_transform(self, entity: str, translation, mat3_rows, *, sim_time: float | None = None) -> None:
        """Log a world-frame ``Transform3D`` for *entity*, which places a Pinhole frustum at the sensor pose.

        ``mat3_rows`` is the row-vector rotation, with rows = the frame axes in world; Rerun wants
        column-vector, so this transposes it. ``sim_time`` stamps the pose at its render epoch; pass the
        same time as the paired image, else the frustum lags the frame when scrubbing.
        """
        import numpy as np

        if sim_time is not None:
            rr.set_time("time", duration=sim_time)
        rr.log(
            entity,
            _transform3d(
                np.asarray(translation, dtype=np.float32),
                np.asarray(mat3_rows, dtype=np.float32).T,
                DEBUG_LIMB_AXIS_LENGTH,
            ),
        )

    def log_points(self, entity: str, positions, *, colors=None, radii=None, labels=None, sim_time=None) -> None:
        """Log marker points, for example the operator's waypoint spheres, at the current timeline time.
        ``sim_time`` stamps the thread-local timeline first: pass it when logging off the lockstep
        thread, as with ``log_transform``.
        """
        if sim_time is not None:
            rr.set_time("time", duration=float(sim_time))
        rr.log(
            entity, rr.Points3D(positions, colors=colors, radii=radii, labels=labels, show_labels=labels is not None)
        )

    def log_arrows(self, entity: str, vectors, *, origins=None, colors=None, labels=None, sim_time=None) -> None:
        """Log 3D arrows at *entity*: body-frame overlays, such as a magnetometer vector or an axis
        triad, logged under a posed parent entity ride its transform.
        ``sim_time`` as in :meth:`log_points`.
        """
        if sim_time is not None:
            rr.set_time("time", duration=float(sim_time))
        rr.log(
            entity,
            rr.Arrows3D(vectors=vectors, origins=origins, colors=colors, labels=labels, show_labels=labels is not None),
        )

    def log_text(self, entity: str, text: str, *, level=None, sim_time=None) -> None:
        """One ``TextLog`` row at *entity*: the framework's own events land at ``logs/sim`` in the
        Logs pane, and another producer's adapter writes its rows through the same seam.
        ``level`` is a rerun ``TextLogLevel`` name in Rerun's uppercase spelling; ``sim_time`` as in :meth:`log_points`.
        """
        if sim_time is not None:
            rr.set_time("time", duration=float(sim_time))
        rr.log(entity, TextLog(text, level=level))

    def log_strip(self, entity: str, points, *, color=None, radius=None) -> None:
        """Log one polyline, a reference path or a Model Predictive Control (MPC) horizon, at the current
        timeline time. Rerun keeps the newest value per entity path at each time, so re-logging the same
        entity just replaces it.
        """
        rr.log(entity, rr.LineStrips3D([points], colors=None if color is None else [color], radii=radius))

    def log_trail(self, entity: str, positions, times, *, color, stride: int = 4) -> None:
        """Emit a GROWING flown-path trail: one columnar ``send_columns`` where each timeline row's strip is
        one segment longer, so as the viewer shows the newest strip, scrubbing grows the trail with the
        vehicle. The data grows linearly with the path, against re-logging the whole path every tick. A
        no-op below 2 points.
        """
        import numpy as np

        pos = np.asarray(positions, dtype=np.float32)
        t = np.asarray(times)
        if len(pos) < 2:
            return
        idx = list(range(0, len(pos), stride))  # stride the growing-strip vertices to bound the data
        if idx[-1] != len(pos) - 1:
            idx.append(len(pos) - 1)  # anchor the final point so the trail reaches the last pose
        # ``.partition([1]*n)`` puts one strip per timeline row; else all strips collapse into one row.
        rr.send_columns(
            entity,
            indexes=[rr.TimeColumn("time", duration=t[idx])],
            columns=rr.LineStrips3D.columns(
                strips=[pos[: i + 1] for i in idx], colors=[list(color)] * len(idx)
            ).partition([1] * len(idx)),
        )

    def log_state(self, state, sim_time: float) -> None:
        """Draw the generic 3D scene, every body's mesh at its current pose, via NVIDIA Newton's
        ``ViewerRerun``: the one reused scene logger. Logs on every call; the orchestrator decimates the
        per-tick fan-out via ``log_hz``, so this stays dumb. Output-only + fault-isolated: a scene hiccup,
        for example a ViewerRerun API drift against Kit's bundled Newton, must never break the loop.
        """
        try:
            if self._debug:
                self._log_axes(state, float(sim_time))  # coordinate-frame triads, no meshes, small .rrd
            else:
                self._viewer.begin_frame(float(sim_time))
                self._viewer.log_state(state)
                self._viewer.end_frame()
            # the canonical base-body entity: cameras, and other static children, ride this pose
            bq0 = state.body_q.numpy()[0]
            rr.set_time("time", duration=float(sim_time))
            try:  # axis_length=0: no frame-axes visual on the body anchor; SDK versions differ
                tf = rr.Transform3D(
                    translation=[float(v) for v in bq0[:3]],
                    rotation=rr.Quaternion(xyzw=[float(v) for v in bq0[3:7]]),
                    axis_length=0.0,
                )
            except TypeError:
                tf = rr.Transform3D(
                    translation=[float(v) for v in bq0[:3]],
                    rotation=rr.Quaternion(xyzw=[float(v) for v in bq0[3:7]]),
                )
            rr.log(BODY_ENTITY, tf)
        except Exception as exc:
            if not getattr(self, "_log_warned", False):
                self._log_warned = True
                logger.warning(f"Rerun scene logging disabled (viewer API mismatch): {exc}")

    def _log_axes(self, state, sim_time: float) -> None:
        """Debug scene: one fixed-length RGB coordinate frame per body, logged HIERARCHICALLY.

        The frame visual, the dedicated axes archetype + an origin dot with the body's leaf-name label,
        logs once, statically, at each body's USD prim path; per tick only a ``Transform3D`` per
        body moves it. The base body gets its world pose; bodies whose entity path nests under the base
        body's, the actuator links, get their pose RELATIVE to the base: rerun composes parent∘child, so their
        world pose is exact and rigidly mounted children, sensor frames, see ``log_static_frame``, ride
        along at no cost. No mesh geometry ⇒ the recording is a small fraction of the full-mesh scene.
        """
        import numpy as np

        rr.set_time("time", duration=sim_time)
        bq = np.asarray(state.body_q.numpy(), dtype=np.float32)  # (nbodies, 7): translation(3) + quat xyzw(4)

        def pose_mat(i):  # column-vector homogeneous world transform of body i
            x, y, z, w = (float(v) for v in bq[i, 3:7])
            m = np.eye(4, dtype=np.float64)
            m[:3, :3] = np.array([
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ])  # fmt: skip
            m[:3, 3] = bq[i, :3]
            return m

        paths = [
            ((self._body_paths[i] if self._body_paths and i < len(self._body_paths) else "") or f"/body_{i}")
            for i in range(len(bq))
        ]
        if not getattr(self, "_axes_static_done", False):
            self._axes_static_done = True
            for i, path in enumerate(paths):
                # the DEDICATED axes archetype draws the fixed-length RGB frame, see _log_frame_axes;
                # the origin dot carries the centered body-name label
                _log_frame_axes(path, DEBUG_BASE_AXIS_LENGTH if i == 0 else DEBUG_LIMB_AXIS_LENGTH)
                rr.log(
                    path,
                    rr.Points3D(
                        [[0.0, 0.0, 0.0]],
                        labels=[path.rsplit("/", 1)[-1] or f"body_{i}"],
                        show_labels=True,
                        radii=[DEBUG_LABEL_RADIUS],
                    ),
                    static=True,
                )
        base = pose_mat(0)
        base_inv = np.linalg.inv(base)
        for i, path in enumerate(paths):
            if i == 0:
                m = base
            elif path.startswith(paths[0] + "/"):
                m = base_inv @ pose_mat(i)  # entity nests under the base -> log the RELATIVE pose
            else:
                m = pose_mat(i)
            length = DEBUG_BASE_AXIS_LENGTH if i == 0 else DEBUG_LIMB_AXIS_LENGTH
            rr.log(
                path,
                _transform3d(m[:3, 3].astype(np.float32), m[:3, :3].astype(np.float32), length),
            )

    def log_image(self, entity: str, rgb, *, sim_time: float | None = None) -> None:
        """Log one rendered frame, an (H,W,3) uint8 RGB array, to *entity*, ``cameras/<name>``.

        The producing sensor owns the rate, by sim-time decimation; this call is cheap on the lockstep
        thread: copy the frame + current sim time into a newest-wins mailbox. A background worker does
        the JPEG compress, q=85, ~30x smaller than raw, since raw 720p ballooned .rrds into the GBs, and the
        ``rr.log`` with an explicit timeline stamp, because rerun timelines are thread-local. Fault-isolated.
        """
        try:
            if self._img_worker is None:
                self._img_queue = queue.Queue(maxsize=2)
                self._img_worker = threading.Thread(target=self._img_encode_loop, name="rrd-img", daemon=True)
                self._img_worker.start()
            item = (entity, np.ascontiguousarray(rgb), self._sim_time if sim_time is None else float(sim_time))
            try:
                self._img_queue.put_nowait(item)
            except queue.Full:  # encoder behind: drop the oldest, keep the freshest
                try:
                    self._img_queue.get_nowait()
                except queue.Empty:
                    pass
                self._img_queue.put_nowait(item)
        except Exception as exc:
            if not self._img_log_warned:
                self._img_log_warned = True
                logger.warning(f"Rerun image logging disabled: {exc}")

    def _img_encode_loop(self) -> None:
        while True:
            item = self._img_queue.get()
            if item is None:
                return
            entity, rgb, sim_time = item
            try:
                if sim_time is not None:
                    rr.set_time("time", duration=float(sim_time))  # thread-local timeline
                rr.log(entity, rr.Image(rgb).compress(jpeg_quality=85))
            except Exception as exc:
                if not self._img_log_warned:
                    self._img_log_warned = True
                    logger.warning(f"Rerun image logging disabled: {exc}")
                return

    def _drain_images(self) -> None:
        if self._img_worker is not None and self._img_queue is not None:
            try:
                self._img_queue.put(None, timeout=1.0)  # sentinel
                self._img_worker.join(timeout=5.0)
            except Exception:
                pass
            self._img_worker = None

    def close(self) -> None:
        self._drain_images()
        try:
            self._viewer.close()
        except Exception:
            pass
        if self._rrd_path:
            logger.info(f"Rerun: session saved to {self._rrd_path}")


def _settings_markdown(data: dict, key_header: str = "Setting") -> str:
    """Render a settings mapping as one flat two-column table: every leaf keyed by its fully dotted
    config path, such as ``runtime.solver``, with the value code-styled. A config is tabular
    data, not prose: heading-per-nest-level read poorly, since rerun's renderer compresses heading sizes,
    so adjacent levels looked the same, while the dotted path carries the nesting exactly.
    Scalar lists print inline comma-separated; a list of mappings inlines each entry,
    ``;``-separated. Values arrive JSON-sanitized, by ``_log_settings``, so only dict/list/scalar
    occur.
    """
    rows: list[tuple[str, str]] = []

    def leaf(value) -> str:
        if isinstance(value, list):
            if any(isinstance(item, dict) for item in value):  # entries ;-separated, their pairs inlined
                return "; ".join(
                    ", ".join(f"{k}: {v}" for k, v in item.items()) if isinstance(item, dict) else str(item)
                    for item in value
                )
            return ", ".join(str(item) for item in value) or "(none)"
        return str(value)

    def walk(path: str, value) -> None:
        if isinstance(value, dict):
            if not value:  # keep an empty section visible: the receipt stays complete
                rows.append((path, "(empty)"))
            for k, v in value.items():
                walk(f"{path}.{k}" if path else str(k), v)
        else:
            rows.append((path, leaf(value)))

    walk("", data)
    esc = lambda t: t.replace("|", "\\|")  # a pipe in a cell would split the table column  # noqa: E731
    body = "\n".join(f"| {esc(k)} | {esc(v)} |" for k, v in rows)
    return f"| {key_header} | Value |\n|---|---|\n" + body  # the Profile tab passes "Property"


def _transform3d(translation, mat3x3, axis_length: float):
    """A ``Transform3D`` whose entity shows fixed-length RGB frame axes, per SDK generation.

    rerun >= 0.28 split the axes visualization out of ``Transform3D`` into the dedicated
    ``TransformAxes3D`` archetype, which :func:`_log_frame_axes` logs separately; older SDKs,
    as the isaacsim container's pinned 0.27, carry ``axis_length`` on ``Transform3D`` itself.
    """
    if hasattr(rr, "TransformAxes3D"):
        return rr.Transform3D(translation=translation, mat3x3=mat3x3)
    return rr.Transform3D(translation=translation, mat3x3=mat3x3, axis_length=axis_length)


def _log_frame_axes(entity: str, axis_length: float) -> None:
    """Log the dedicated fixed-length axes archetype once, static: rerun >= 0.28 only; on older
    SDKs the length rides on every ``Transform3D``, see :func:`_transform3d`.
    """
    if hasattr(rr, "TransformAxes3D"):
        rr.log(entity, rr.TransformAxes3D(axis_length=float(axis_length)), static=True)


def _viewer_static_exclusions(viewer) -> list[str] | None:
    """Blueprint exclusions for every STATIC shape batch, chiefly the ground plane.

    Read from the ViewerRerun instance itself after ``set_model``: the viewer names entities per
    INSTANCING batch, ``/model/shapes/shape_N`` where N = order of first appearance of a unique
    geometry, static, flags batch, not per model shape index: 18 model shapes become ~9 batches,
    so any index computed from the model is wrong; the isaac ground landed at batch 8 while its model
    shape index was 12. Each batch object carries its own ``name`` and ``static`` flag: the exact
    source of truth.
    """
    try:
        batches = getattr(viewer, "_shape_instances", None)
        if not batches:
            return None
        out = [f"- {b.name}" for b in batches.values() if getattr(b, "static", False)]
        return out or None
    except Exception:
        return None  # viewer API drift -> the default shape_0 exclusion


def build_logger(
    model,
    *,
    viewer: bool,
    record_to_rrd: str | None = None,
    debug: bool = False,
    sink=None,
    settings=None,
):
    """The one Rerun-sink factory: every runtime/orchestrator builds its recording sink here instead of
    inlining ``Logger(...)``, so the serve-or-file and axes-only-``debug`` policy lives in a single
    place, which used to be a copy-paste across the orchestrator builders. Call it inside the builder's
    ``if rerun:`` guard, which keeps the optional ``rerun`` dependency lazily imported; headless/CI never
    pulls it.

    Returns a caller-supplied ``sink`` untouched, since a runtime can inject its own, else a fresh
    :class:`Logger`: ``serve=viewer`` serves on :9876 when viewing, else writes the ``.rrd``; ``debug`` →
    axes-only, small .rrd; ``settings`` → the run's effective config, shown in the viewer's Settings tab.
    """
    if sink is not None:
        return sink
    return Logger(model, serve=viewer, record_to_rrd=record_to_rrd, debug=debug, settings=settings)
