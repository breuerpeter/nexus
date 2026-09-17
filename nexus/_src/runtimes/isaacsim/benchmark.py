"""Opt-in system-envelope benchmarking via Isaac's ``isaacsim.benchmark.services`` recorders.

The shared ``--benchmark`` flag records what the loop profiler structurally can't see: real render GPU
frametime from HydraEngineStats, Video Random Access Memory (VRAM) + host memory, process CPU%,
app-update frametime and windowed Real Time Factor (RTF) stability. It writes one JSON next to the
flight logs plus one info-level summary line. Complements, and doesn't replace,
``core/profiling.LoopProfiler``: that one partitions the sim's own tick at the loop seams; these see
the Kit/system envelope. The recorders come straight from the registry, not via ``BaseIsaacBenchmark``,
whose init silently bails without a Nucleus assets root and assumes World.

PhysX-bound recorders, physics_frametime and physics_step_interval, stay out: physics here is
Newton/Warp, their PhysX zones never fire. render_frametime stays out: it needs async rendering.
"""

from __future__ import annotations

import json
import time

from nexus._src.core import logger

_RECORDERS = ("hardware", "runtime", "app_frametime", "gpu_frametime", "memory", "cpu_continuous", "rtf_stability")


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
                    logger.warning(f"benchmark: recorder {name!r} unavailable ({exc!r})")
            logger.info(f"benchmark: recording [{', '.join(n for n, _ in self._recs)}]")
        except Exception as exc:
            logger.warning(f"benchmark: isaacsim.benchmark.services unavailable ({exc!r})")

    def finish(self, out_path: str | None = None) -> dict:
        """Stop recorders, log a one-line summary, optionally write the full JSON."""
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
        logger.info(f"benchmark: {json.dumps(headline)}")
        if out_path:
            try:
                with open(out_path, "w") as f:
                    json.dump(metrics, f, indent=1, default=str)
                logger.info(f"benchmark: full metrics -> {out_path}")
            except Exception as exc:
                logger.warning(f"benchmark: write failed ({exc!r})")
        return metrics
