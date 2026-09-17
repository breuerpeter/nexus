"""The Recorder's Rerun adapter: the teardown dump of every channel ring into the recording.

Recorder vocabulary, not Logger vocabulary: it reads ``fields``, ``source`` and ``history_arrays()``
off the channels, so it belongs beside the Recorder that declares them. The Logger stays the sink:
it takes the tab tree for the blueprint and draws the flown path with its own growing-strip call.

The flown path is the base body channel's own ``position`` series. Physics used to accumulate it per
tick only because the Logger called it every tick; the rows were already on the ring.
"""

from __future__ import annotations

import numpy as np
import rerun as rr

from nexus._src.core import logger
from nexus._src.logging.rerun_logging import RECORDING_ROOT


def _series_names(field: str, width: int) -> list[str]:
    """Legend names for a multi-column quantity's series, one per column: kinematic vectors get
    their conventional component names; anything else gets an index.
    """
    if field in ("position", "velocity"):
        return ["x", "y", "z"]
    if field == "angular_velocity":
        return ["wx", "wy", "wz"]
    if field == "quat_xyzw":
        return ["qx", "qy", "qz", "qw"]
    return [f"{field}[{i}]" for i in range(width)]


TRAJECTORY_ENTITY = "physics/trajectory"
TRAIL_COLOR = (40, 160, 255)


def dump(channels, sink, *, base_body=None) -> None:
    """Dump every declared channel's ring into the recording, then draw the flown path.

    Each channel becomes time-series entities, ``recording/<channel key>/<field>``, one columnar
    ``send_columns`` per quantity, at zero per-tick cost: the rings are the history. Carries its own
    time indexes, since a columnar send takes an explicit index, not the global timeline. Channels
    with no declared schema, and never-written ones, as in a Sim entered but not run, are skipped.

    Args:
        channels: The Recorder's channels, keyed by entity name.
        sink: The Logger: it takes the tab tree and draws the trail.
        base_body: The base body's channel, whose ``position`` series is the flown path; ``None``
            draws no trail.
    """
    tree: dict[str, tuple[str | None, list[str]]] = {}
    wrapped: list[str] = []
    wrapped_s = 0.0
    warned = False
    for key, ch in channels.items():
        fields = getattr(ch, "fields", None)
        if not fields:
            continue  # no declared schema: nothing to plot
        try:
            arrays = ch.history_arrays()
            t = np.asarray(arrays.pop("t"), dtype=np.float64)
            if t.size == 0:
                continue
            if int(ch.counter.numpy()[0]) > ch.maxlen:
                wrapped.append(key)
                wrapped_s = ch.maxlen * ch.dt
            for name, width in fields:
                entity = f"{RECORDING_ROOT}/{key}/{name}"
                if width > 1:  # one series per column, named once for the legend, statically
                    rr.log(entity, rr.SeriesLines(names=_series_names(name, width)), static=True)
                rr.send_columns(
                    entity,
                    indexes=[rr.TimeColumn("time", duration=t)],
                    columns=rr.Scalars.columns(scalars=arrays[name]),
                )
            tree[key] = (getattr(ch, "source", None), [n for n, _ in fields])
        except Exception as exc:
            if not warned:
                warned = True
                logger.warning(f"Rerun recorder dump failed at {key!r}: {exc}")
    if wrapped:
        logger.warning(
            f"recorder ring wrapped for {len(wrapped)} channel(s) (e.g. {wrapped[0]!r}): "
            f"debug plots cover only the last {wrapped_s:.0f}s of the run"
        )
    _draw_trail(base_body, sink)
    sink.show_recording(tree)


def _draw_trail(base_body, sink) -> None:
    """Draw the flown path from the base body channel's recorded positions. Output-only."""
    if base_body is None:
        return
    try:
        arrays = base_body.history_arrays()
        t = np.asarray(arrays["t"], dtype=np.float64)
        if t.size:
            sink.log_trail(TRAJECTORY_ENTITY, arrays["position"], t, color=TRAIL_COLOR)
    except Exception as exc:
        logger.warning(f"Rerun flown-path dump failed: {exc}")
