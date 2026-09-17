"""Honest .rrd flight-video analysis: real frame rate, duplicate fraction, pinhole/body coincidence.

Reader correctness, the lesson this tool encodes: when iterating rerun chunks via
``chunk.to_record_batch()``, you must extract per-row values with ``column[i].as_py()``.
Reaching through ``cell.values`` returns the chunk-shared flattened child buffer, so every row
in a chunk then reads identically, which fabricates ~75-95% "duplicate frames" and quantizes
pose traces to one value per chunk. This artifact once masqueraded as a render-pipeline
throttle and burned a full investigation. Pixel-equality verdicts additionally need the
JPEG bytes themselves: raw-pixel hashes flip on denoiser dither every frame.

Usage:
  uv run --extra policy python tools/analyze_rrd_video.py [path/to.rrd]   # default: newest
"""

from __future__ import annotations

import glob
import hashlib
import os
import sys

import numpy as np


def _rows(path: str, entity: str, colpred):
    """(time, as_py value) rows for *entity* columns matching *colpred*, extracted per row."""
    from rerun.experimental import RrdReader

    out = []
    for chunk in RrdReader(path).stream():
        if str(chunk.entity_path) != entity:
            continue
        rb = chunk.to_record_batch()
        tcols = [c for c in rb.schema.names if c == "time" or "timestamp" in c.lower()]
        if not tcols:
            continue
        times = [s.value / 1e9 for s in rb[tcols[0]]]
        for c in [c for c in rb.schema.names if colpred(c)]:
            col = rb[c]
            for i in range(len(col)):
                py = col[i].as_py()  # load-bearing: per-row scalar, never cell.values
                if py is not None:
                    out.append((times[i], py))
    out.sort(key=lambda x: x[0])
    return out


def main() -> None:
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else max(glob.glob(os.path.expanduser("~/.cache/newton/logs/*.rrd")), key=os.path.getmtime)
    )
    print(f"rrd: {path} ({os.path.getsize(path) // 2**20} MB)")

    cam = "/cameras/fpvcam"
    imgs = _rows(path, cam, lambda c: "blob" in c.lower())
    if imgs:
        hashes = [hashlib.md5(bytes(v[0] if isinstance(v, list) and len(v) == 1 else v)).hexdigest() for _, v in imgs]
        ts = [t for t, _ in imgs]
        span = max(ts) - min(ts) or 1.0
        dup = float(np.mean([hashes[i] == hashes[i - 1] for i in range(1, len(hashes))]))
        print(
            f"video {cam}: {len(hashes)} frames over {span:.0f}s = {len(hashes) / span:.1f} fps logged, "
            f"{len(set(hashes))} distinct ({len(set(hashes)) / span:.1f} real fps), "
            f"consecutive-duplicates {dup * 100:.0f}%"
        )

    def vec3(entity):
        rows = _rows(path, entity, lambda c: "translation" in c.lower())
        return [(t, np.asarray(v, dtype=np.float64).reshape(-1)[:3]) for t, v in rows]

    pin, body = vec3(cam), vec3("/World/Vehicle/Geometry/body_frd")
    if pin and body:
        bt = np.array([t for t, _ in body])
        bz = np.array([v[2] for _, v in body])
        dz, vz = [], []
        for t, v in pin:
            if bt[0] < t < bt[-1]:
                i = np.searchsorted(bt, t)
                vel = (bz[min(i + 1, len(bz) - 1)] - bz[max(i - 2, 0)]) / (
                    bt[min(i + 1, len(bt) - 1)] - bt[max(i - 2, 0)] + 1e-9
                )
                dz.append(v[2] - float(np.interp(t, bt, bz)))
                vz.append(vel)
        dz, vz = np.array(dz), np.array(vz)
        m = np.abs(vz) > 0.2
        if m.any():
            print(
                f"pinhole-vs-body: climb offset {dz[m].mean() * 1000:+.1f} mm "
                f"(slope {np.polyfit(vz[m], dz[m], 1)[0] * 1000:+.1f} ms of velocity), "
                f"at rest {dz[~m].mean() * 1000:+.1f} mm: rigid pairing => ~0/~0"
            )


if __name__ == "__main__":
    main()
