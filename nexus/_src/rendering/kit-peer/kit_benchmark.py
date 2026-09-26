"""Opt-in Kit benchmarking through Isaac's ``isaacsim.benchmark.services`` recorders.

The shared ``--benchmark`` flag records what the host's loop profiler can't see from outside this
process: the render GPU frametime from HydraEngineStats, Video Random Access Memory (VRAM) and host
memory, process CPU load, app-update frametime and windowed Real Time Factor (RTF) stability. It
writes one JSON the host names in the setup message, plus one summary line on the console. The
recorders come straight from the registry, not through ``BaseIsaacBenchmark``, whose init bails
without a Nucleus assets root and assumes World.

The PhysX recorders, physics_frametime and physics_step_interval, stay out: Kit simulates nothing.
render_frametime stays out: it needs async rendering.
"""

from __future__ import annotations

import json
import time

_RECORDERS = ("hardware", "runtime", "app_frametime", "gpu_frametime", "memory", "cpu_continuous", "rtf_stability")


def _log(msg: str) -> None:
    print(f"[kit-peer] benchmark: {msg}", flush=True)


class KitBenchmark:
    """Curated Isaac benchmark recorders around the steady flight window."""

    def __init__(self):
        self._recs: list = []
        self._t0 = time.time()
        try:
            import omni.kit.app

            em = omni.kit.app.get_app().get_extension_manager()
            em.set_extension_enabled_immediate("isaacsim.benchmark.services", True)
            from isaacsim.benchmark.services.datarecorders.interface import (
                InputContext,
                MeasurementDataRecorderRegistry,
            )

            ctx = InputContext(phase="flight")
            for name in _RECORDERS:
                cls = MeasurementDataRecorderRegistry.get(name)
                if cls is None:
                    continue
                try:
                    rec = cls(ctx)
                    getattr(rec, "start_collecting", lambda: None)()
                    self._recs.append((name, rec))
                except Exception as exc:
                    _log(f"recorder {name!r} unavailable ({exc!r})")
            _log(f"recording [{', '.join(n for n, _ in self._recs)}]")
        except Exception as exc:
            _log(f"isaacsim.benchmark.services unavailable ({exc!r})")

    def finish(self, out_path: str) -> None:
        """Stop the recorders, log a one-line summary and write the full JSON to ``out_path``."""
        metrics: dict = {"wall_s": round(time.time() - self._t0, 1)}
        for name, rec in self._recs:
            try:
                getattr(rec, "stop_collecting", lambda: None)()
                for m in rec.get_data().measurements:
                    val = getattr(m, "value", None)
                    if isinstance(val, float):
                        val = round(val, 3)
                    metrics[getattr(m, "name", f"{name}?")] = val
            except Exception as exc:
                metrics[name] = f"failed: {exc!r}"
        headline = {
            k: v
            for k, v in metrics.items()
            if any(s in k.lower() for s in ("mean fps", "real time factor", "gpu memory", "rss", "frametime"))
            and not any(s in k.lower() for s in ("stdev", "min", "max", "samples"))
        }
        _log(json.dumps(headline))
        try:
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=1, default=str)
            _log(f"full metrics -> {out_path}")
        except Exception as exc:
            _log(f"write failed ({exc!r})")
