"""Loop profiling: where does the wall-clock go per tick, on any runtime, at negligible cost.

The design follows the established simulator patterns: MuJoCo's per-phase timer partition with an
explicit residual, Gazebo's sliding-window Real Time Factor (RTF), microprofile's integer-nanosecond
ring buffers.

* **Phases partition the tick.** ``mark(phase)`` records the time since the last mark into a named
  top-level bucket; whatever no bucket claims lands in ``other``, so the buckets always reconcile
  to the measured tick time.
* **Spans add nested detail.** ``span(name)``, a context manager, times a sub-region inside a phase,
  for example the Kit render inside ``sensors.host``; spans inform, they don't partition.
* **Always-on and cheap.** Accumulation is two ``perf_counter_ns`` reads and an integer add per mark,
  about 1 µs/tick for the whole loop, noise at 250 Hz. Percentiles come from a fixed ring of per-tick
  totals; nothing allocates or formats on the hot path.
* **GPU time via CUDA events, read late.** A small pool of event pairs brackets the device batch;
  the harvest of elapsed times comes a few ticks later, when the events are long complete, never a sync.
* **Deep traces on demand.** ``trace_path`` buffers spans as integer tuples and exports Chrome/Perfetto
  trace-event JSON, ``ph:"X"`` spans + ``ph:"C"`` counters, on ``close()``; drag it onto ui.perfetto.dev.
  Inside Kit, ``use_carb`` mirrors phases into ``carb.profiler`` zones, so Kit's tracy/nvtx/cpu
  backends, ``/app/profilerBackend``, see the sim loop with zero extra plumbing.

One report line per run, and periodically with ``report_every_s``::

    profile [captured-host] 7113 ticks | tick p50 3.9 ms p95 4.4 ms | rtf 0.79 | exchange 2.71 ms 68% |
    sensors.host 0.83 ms 21% | replay 0.27 ms 7% | ... | other 0.04 ms 1% | gpu.batch 0.26 ms

``stats()`` returns the same numbers as a dict, merged into ``Orchestrator.run_stats``.
"""

from __future__ import annotations

import contextlib
import json
import time

from nexus._src.core import logger

_RING = 4096  # per-tick totals kept for percentiles; about 16 s at 250 Hz
_WIN = 1000  # ticks per window of the whole-run RTF series; about 4 s at 250 Hz
_TRACE_CAP = 400_000  # max buffered trace spans, about 50 MB JSON; deep traces are for short runs


class LoopProfiler:
    """Per-tick phase profiler for the orchestrator loops; see the module docstring."""

    def __init__(
        self,
        label: str,
        *,
        dt: float,
        report_every_s: float = 0.0,
        trace_path: str | None = None,
        use_carb: bool = False,
    ):
        self.label = label
        self._dt_ns = int(dt * 1e9)
        self._report_every_ns = int(report_every_s * 1e9)
        self._phase_ns: dict[str, int] = {}
        self._phase_n: dict[str, int] = {}
        self._span_ns: dict[str, int] = {}
        self._span_n: dict[str, int] = {}
        self._ring = [0] * _RING
        self._win_t0 = 0
        self._win_ticks = 0
        self._win_rtfs: list[float] = []
        self._ticks = 0
        self._t_tick0 = 0
        self._t_last = 0
        self._t_run0 = 0
        self._t_report = 0
        self._trace: list | None = [] if trace_path else None
        self._trace_path = trace_path
        self._trace_t0 = time.perf_counter_ns()
        self._carb = None
        if use_carb:
            try:  # inside Kit only; zones then reach any /app/profilerBackend: cpu, tracy or nvtx
                import carb.profiler

                self._carb = carb.profiler
            except Exception:
                self._carb = None
        # GPU event pool; gpu_begin fills it lazily on a CUDA device
        self._gpu_pool: list | None = None
        self._gpu_pending: list = []
        self._gpu_ns = 0
        self._gpu_n = 0

    # -- hot path -------------------------------------------------------------------------
    def tick_begin(self) -> None:
        now = time.perf_counter_ns()
        if self._t_run0 == 0:
            self._t_run0 = self._t_report = self._win_t0 = now
        self._t_tick0 = self._t_last = now
        if self._carb is not None:
            self._carb.begin(1, "tick")

    def mark(self, phase: str) -> None:
        """Close the current phase segment: everything since the last mark belongs to *phase*."""
        now = time.perf_counter_ns()
        self._phase_ns[phase] = self._phase_ns.get(phase, 0) + (now - self._t_last)
        self._phase_n[phase] = self._phase_n.get(phase, 0) + 1
        self._t_last = now

    @contextlib.contextmanager
    def span(self, name: str):
        """Nested detail timing; doesn't partition the tick."""
        t0 = time.perf_counter_ns()
        if self._carb is not None:
            self._carb.begin(1, name)
        try:
            yield
        finally:
            t1 = time.perf_counter_ns()
            if self._carb is not None:
                self._carb.end(1)
            self._span_ns[name] = self._span_ns.get(name, 0) + (t1 - t0)
            self._span_n[name] = self._span_n.get(name, 0) + 1
            if self._trace is not None and len(self._trace) < _TRACE_CAP:
                self._trace.append((name, t0, t1 - t0, 1))

    def tick_end(self) -> None:
        now = time.perf_counter_ns()
        total = now - self._t_tick0
        self._ring[self._ticks % _RING] = total
        # Window RTF over wall-clock boundary-to-boundary spans; summed in-tick time would exclude
        # the inter-tick seam, for example Kit render/present between step() calls, and overstate speed.
        self._win_ticks += 1
        if self._win_ticks == _WIN:
            self._win_rtfs.append(_WIN * self._dt_ns / max(1, now - self._win_t0))
            self._win_t0 = now
            self._win_ticks = 0
        self._ticks += 1
        if self._carb is not None:
            self._carb.end(1)
        if self._trace is not None and len(self._trace) < _TRACE_CAP:
            self._trace.append(("tick", self._t_tick0, total, 0))
        self._harvest_gpu()
        if self._report_every_ns and (now - self._t_report) >= self._report_every_ns:
            self._t_report = now
            self.report()

    # -- GPU: CUDA events around the device batch; harvested ticks later, never a sync ----
    def gpu_begin(self) -> None:
        try:
            import warp as wp

            if self._gpu_pool is None:
                self._gpu_pool = [(wp.Event(enable_timing=True), wp.Event(enable_timing=True)) for _ in range(8)]
            pair = self._gpu_pool[self._ticks % 8]
            wp.record_event(pair[0])
            self._gpu_open = pair
        except Exception:
            self._gpu_pool = None  # non-CUDA device; GPU timing off
            self.gpu_begin = lambda: None  # type: ignore[method-assign]
            self.gpu_end = lambda: None  # type: ignore[method-assign]

    def gpu_end(self) -> None:
        if self._gpu_pool is None:
            return
        import warp as wp

        wp.record_event(self._gpu_open[1])
        self._gpu_pending.append((self._ticks, self._gpu_open))

    def _harvest_gpu(self) -> None:
        if not self._gpu_pending or self._ticks - self._gpu_pending[0][0] < 4:
            return  # events from 4+ ticks ago are complete; each tick's D2H read synced the stream
        import warp as wp

        while self._gpu_pending and self._ticks - self._gpu_pending[0][0] >= 4:
            _, (e0, e1) = self._gpu_pending.pop(0)
            try:
                self._gpu_ns += int(wp.get_event_elapsed_time(e0, e1) * 1e6)  # ms -> ns
                self._gpu_n += 1
            except Exception:
                pass

    # -- reporting ------------------------------------------------------------------------
    def stats(self) -> dict:
        n = max(1, self._ticks)
        window = sorted(self._ring[: min(self._ticks, _RING)])

        def pct(p):
            return window[min(len(window) - 1, int(p * len(window)))] if window else 0

        wall = max(1, time.perf_counter_ns() - self._t_run0)
        phases = {k: v / n for k, v in self._phase_ns.items()}
        tick_mean = sum(self._ring[: min(self._ticks, _RING)]) / max(1, len(window))
        phases["other"] = max(0.0, tick_mean - sum(phases.values()))
        out = {
            "ticks": self._ticks,
            "rtf_window": round(len(window) * self._dt_ns / max(1, sum(window)), 3),
            "rtf_full": round(self._ticks * self._dt_ns / wall, 3),
            "tick_p50_us": round(pct(0.50) / 1e3, 1),
            "tick_p95_us": round(pct(0.95) / 1e3, 1),
            "tick_p99_us": round(pct(0.99) / 1e3, 1),
            "phases_us": {k: round(v / 1e3, 1) for k, v in sorted(phases.items(), key=lambda kv: -kv[1])},
            "spans_us": {
                k: round(v / max(1, self._span_n[k]) / 1e3, 1)
                for k, v in sorted(self._span_ns.items(), key=lambda kv: -kv[1])
            },
            "spans_n": dict(sorted(self._span_n.items(), key=lambda kv: -kv[1])),
        }
        if self._gpu_n:
            out["gpu_batch_us"] = round(self._gpu_ns / self._gpu_n / 1e3, 1)
        # Whole-run RTF spread over fixed windows; drop the first window as the compile/settle
        # transient, the same exclusion the orchestrator's steady rtf makes via warmup_steps.
        steady = self._win_rtfs[1:] if len(self._win_rtfs) > 1 else self._win_rtfs
        if steady:
            mean = sum(steady) / len(steady)
            out["rtf_win"] = {
                "min": round(min(steady), 3),
                "max": round(max(steady), 3),
                "avg": round(mean, 3),
                "std": round((sum((x - mean) ** 2 for x in steady) / len(steady)) ** 0.5, 3),
                "n": len(steady),
            }
        return out

    def report(self) -> None:
        if not self._ticks:
            return
        s = self.stats()
        tick_mean_us = sum(s["phases_us"].values())
        parts = " | ".join(
            f"{k} {v / 1e3:.2f} ms {v / max(1e-9, tick_mean_us) * 100:.0f}%" for k, v in s["phases_us"].items()
        )
        detail = " | ".join(
            f"{k} {v / 1e3:.2f} ms x{s['spans_n'].get(k, 0)}" for k, v in list(s["spans_us"].items())[:8]
        )
        gpu = f" | gpu.batch {s['gpu_batch_us'] / 1e3:.2f} ms" if "gpu_batch_us" in s else ""
        logger.info(
            f"profile [{self.label}] {s['ticks']} ticks | tick p50 {s['tick_p50_us'] / 1e3:.2f} ms "
            f"p95 {s['tick_p95_us'] / 1e3:.2f} ms | rtf {s['rtf_window']:.2f} | {parts}{gpu}"
            + (f" || {detail}" if detail else ""),
            extra={"console_only": True},  # the viewer shows this as the Profile tab, not a log row
        )

    def close(self) -> None:
        """Final report + deep-trace export, if enabled."""
        self.report()
        if self._trace is not None and self._trace_path:
            events = [
                {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": f"newton [{self.label}]"}},
                {"ph": "M", "name": "thread_name", "pid": 1, "tid": 0, "args": {"name": "loop"}},
                {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1, "args": {"name": "detail"}},
            ]
            t0 = self._trace_t0
            events += [
                {"ph": "X", "name": name, "cat": "sim", "ts": (ts - t0) / 1e3, "dur": dur / 1e3, "pid": 1, "tid": tid}
                for name, ts, dur, tid in self._trace
            ]
            with open(self._trace_path, "w") as f:
                json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)
            logger.info(f"profile trace written: {self._trace_path} ({len(self._trace)} spans): ui.perfetto.dev")
