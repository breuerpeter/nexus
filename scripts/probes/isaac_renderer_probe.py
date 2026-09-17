"""Stage-2 verification probe: Isaac Sim as a pure renderer over the one ``NewtonPhysics``.

Validates, headless in the runtime container, the four things the renderer collapse rests on:

1. **Vehicle pose writes + epoch pairing**: the frame writes the vehicle body prims' Fabric
   ``worldMatrix`` from nexus's ``body_q``, and meshes compose beneath; a body-mounted camera
   must stay rigid to the airframe under large teleports, corner-patch frame diff ~0; if the mesh
   writes didn't take, the motor pods would fly out of frame and the diff explodes; and the render
   must display the earlier write, the one-frame latch the epoch pairing rests on: the reported
   camera pose matches the earlier teleport, not this one.
2. **Captured CUDA graph over nexus's solver inside Kit**: clear/actuate/step recorded once,
   replayed at high rate: the vehicle must actually climb, a motion-REQUIRED check, since a frozen
   state under replay is the classic silent capture failure, with the per-replay cost printed.
3. **Render cost**: mean ``frame.update`` wall time, the Real-Time Factor (RTF) lever, printed for
   comparison against the pre-collapse numbers; expect an improvement: no double model, no stage rebuild.
4. **The timeline clock tracks the lockstep sim time**: the Kit timeline is the clock authored
   Universal Scene Description (USD) time-samples resolve against, and left undriven Kit's playback
   paces it off the RENDER frame count, so scene animation runs at the render rate and drifts away
   from sim time without bound, GH #62. After the render loop the clock must track the last
   ``sim_time`` driven, to within a few frames.

Run:
  uv run nexus script scripts/probes/isaac_renderer_probe.py
"""

import time
from types import SimpleNamespace

import numpy as np
import warp as wp

from nexus._src.config import LaunchConfig
from nexus._src.runtimes.isaacsim.launch import build_from_launch

orch = build_from_launch(LaunchConfig().set_vehicle("assets/local/astro_max_fpv.usdz").set_control("px4-sitl"))
phys = orch.physics
frame = orch.renderer
cams = [s for s in orch.sensors if type(s).__name__ == "RtxCameraSensor"]
assert frame is not None and cams, f"renderer={frame} cams={cams}"
cam = cams[0]

state = phys.reset()  # build + settle, solver kernels compile, BEFORE any render
frame.on_physics_ready()

import omni.replicator.core as rep  # noqa: E402  # Kit extension module, imported at its point of use

params = rep.AnnotatorRegistry.get_annotator("camera_params")
params.attach(cam._rp)

base_bq = state.body_q.numpy().copy()
sim_t = {"t": 0.0}


def teleport(dx: float):
    bq = base_bq.copy()
    bq[:, 0] += dx
    bq[:, 2] += 200.0  # pure-sky background: corner pixels are only the motor pods
    state.body_q.assign(bq)


def render():
    sim_t["t"] += 1.0 / 30.0
    frame._last_update = None
    frame.update(sim_time=sim_t["t"])


def cam_world_x():
    try:
        m = np.asarray(params.get_data()["cameraViewTransform"]).reshape(4, 4)
        return float((-m[3, :3] @ np.linalg.inv(m[:3, :3]))[0])
    except Exception:
        return float("nan")


def grab():
    return np.asarray(cam._rgb.get_data())[:, :, :3].astype(np.float32)


# --- 1: rigidity + earlier-pose pairing under externally written vehicle poses ---
teleport(0.0)
for _ in range(3):  # settle the pipeline on the start pose
    render()
prev_img = grab()
diffs, err_this, err_prev = [], [], []
prev_x = base_bq[0, 0]
for k in range(1, 7):
    this_x = base_bq[0, 0] + 10.0 * k
    teleport(10.0 * k)
    render()
    img = grab()
    cx = cam_world_x()
    off = float(cam._local[3][0])
    err_this.append(abs(cx - (this_x + off)))
    err_prev.append(abs(cx - (prev_x + off)))
    h, w = img.shape[:2]
    ph, pw = h // 5, w // 5

    def corners(a, ph=ph, pw=pw):
        return np.concatenate([a[:ph, :pw], a[:ph, -pw:], a[-ph:, :pw], a[-ph:, -pw:]], axis=None)

    diffs.append(float(np.mean(np.abs(corners(img) - corners(prev_img)))))
    prev_img = img
    prev_x = this_x
print(
    f"[R1] rigidity corner-diff={np.mean(diffs):6.2f} (rigid ~0) | render-used cam-x err: "
    f"vs THIS teleport={np.nanmean(err_this):6.3f} m, vs PREV={np.nanmean(err_prev):6.3f} m "
    f"(pairing correct => PREV ~0)",
    flush=True,
)

# --- 3: render cost, mean frame.update wall ms ---
teleport(0.0)
render()
t0 = time.time()
n_frames = 30
for _ in range(n_frames):
    render()
print(f"[R3] render.update mean {1000.0 * (time.time() - t0) / n_frames:.1f} ms over {n_frames} frames", flush=True)

# --- 4: the Kit timeline clock tracks the lockstep sim time ---
# Reads the clock the preceding renders just drove. Authored USD time-samples resolve against it,
# and left undriven Kit's playback paces it off the RENDER frame count, so scene animation runs
# at the render rate and walks away from sim time; measured on main: +0.5 s of lead by here.
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402

tl_t = float(omni.timeline.get_timeline_interface().get_current_time())
driven = sim_t["t"]
# A few frames of tolerance, not exact equality: the clock lands one frame ahead of the value the
# probe sets, because Kit's playback advances it during the update that follows the write. That
# offset is constant; the failure this gates on is a lead that grows with the render count, which
# no frame-scale tolerance can hide.
tcps = float(omni.usd.get_context().get_stage().GetTimeCodesPerSecond() or 24.0)
tracking = abs(tl_t - driven) <= 3.0 / tcps
print(
    f"[R4] timeline clock {tl_t:.3f} s vs last driven sim time {driven:.3f} s "
    f"(lead {tl_t - driven:+.3f} s, tol {3.0 / tcps:.3f} s @ {tcps:g} fps) "
    f"({'TRACKING' if tracking else 'NOT SIM-LOCKED: animation is paced by render count'})",
    flush=True,
)

# --- 2: captured CUDA graph over nexus's solver inside Kit, motion-REQUIRED ---
state = phys.reset()  # back on the ground, settled
env = orch.environment.sample(None, None)
dt = phys.sim_dt
act = orch.actuator
act.write_controls(SimpleNamespace(command=np.full(4, 0.9, dtype=np.float32)))
z0 = float(state.body_q.numpy()[0, 2])
with wp.ScopedCapture() as cap:
    phys.clear_forces(state)
    act.forces_wp(state)
    phys.step(state, env, dt)
n_replays = 500
wp.synchronize()
t0 = time.time()
for _ in range(n_replays):
    wp.capture_launch(cap.graph)
wp.synchronize()
us = 1e6 * (time.time() - t0) / n_replays
z1 = float(state.body_q.numpy()[0, 2])
print(
    f"[R2] captured replay x{n_replays} @0.9 cmd: climb {z1 - z0:+.2f} m "
    f"({'MOTION OK' if (z1 - z0) > 1.0 else 'FROZEN: CAPTURE BROKEN'}), {us:.0f} us/replay",
    flush=True,
)
print("[R] DONE", flush=True)
