"""Physical-parameter recovery, system-ID, by differentiable simulation: recover an unknown body **mass**
from an open-loop trajectory.

Self-contained example: a fixed world-+z thrust acts on a lumped single free body under
``SolverSemiImplicit``; a heavier body rises less. Starting from a wrong mass guess, gradient descent on
``body_inv_mass``, the differentiable leaf the integrator actually consumes, minimizes the trajectory
Mean Squared Error (MSE) against a reference rollout generated at the true mass, recovering the mass to
<5%. This is a math gate the multi-body Featherstone path can't give, since it exposes no mass gradient.

The companion ``gain_tuning.py`` is the other differentiable-design gate, controller auto-tuning.

Run:

    uv run -m nexus.examples mass_recovery   # asserts the recovered mass; runs on CPU
"""

from __future__ import annotations

import newton
import newton.solvers
import numpy as np
import warp as wp

import nexus as na
from nexus.examples._lib import dump_stats
from nexus.examples._lib.single_body import collapse_to_single_body  # the one single-body seam

GRAVITY = 9.81


@wp.kernel
def const_wrench_world(thrust_z: float, body_f: wp.array(dtype=wp.spatial_vector)):
    """A fixed world-+z thrust, open-loop: the system-ID excitation."""
    body_f[0] = wp.spatial_vector(wp.vec3(0.0, 0.0, thrust_z), wp.vec3(0.0, 0.0, 0.0))


@wp.kernel
def traj_mse(
    body_q: wp.array(dtype=wp.transform),
    ref: wp.array(dtype=wp.vec3),
    step: int,
    cost: wp.array(dtype=float),
):
    p = wp.transform_get_translation(body_q[0])
    d = p - ref[step]
    wp.atomic_add(cost, 0, wp.dot(d, d))


def recover_mass(
    *,
    vehicle: str = "astro_max_base",
    true_mass_scale: float = 1.4,
    init_guess_scale: float = 0.7,
    n_steps: int = 200,
    dt: float = 0.004,
    iters: int = 120,
    lr: float = 0.05,
):
    """Known-parameter recovery: recover the body mass from an open-loop trajectory.

    A fixed world-+z thrust acts on the lumped body; a heavier body rises less. Starting from a wrong
    mass guess, descend ``d/d(inv_mass)`` of the trajectory MSE against a reference rollout generated at the
    true mass. Returns the convergence history. ``inv_mass`` is the differentiable leaf the SemiImplicit
    integrator actually consumes; reported back as mass = 1/inv_mass.
    """
    from nexus._src.build.launch import resolve_to_vehicle_builder
    from nexus._src.config import LaunchConfig

    # Single rigid body collapsed from the vehicle Universal Scene Description (USD): the open-loop world-+z
    # thrust is frame-agnostic, so the Forward Right Down (FRD) body works unchanged here; only the body
    # mass matters for the system-ID.
    vb, _ = resolve_to_vehicle_builder(LaunchConfig().set_vehicle(vehicle))
    sb = collapse_to_single_body(vb, requires_grad=True)
    model, nominal_mass = sb.model, sb.mass
    solver = newton.solvers.SolverSemiImplicit(model)
    true_mass = nominal_mass * true_mass_scale
    thrust = 1.3 * nominal_mass * GRAVITY  # fixed, mass-independent open-loop thrust

    states = [model.state(requires_grad=True) for _ in range(n_steps + 1)]

    def reset_state():
        newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
        bq = states[0].body_q.numpy()
        bq[0, 0:3] = [0.0, 0.0, 0.5]
        bq[0, 3:7] = [0, 0, 0, 1]
        states[0].body_q.assign(bq)
        bqd = states[0].body_qd.numpy()
        bqd[0, :] = 0.0
        states[0].body_qd.assign(bqd)

    def open_loop_forward(cost=None, ref=None):
        reset_state()
        for t in range(n_steps):
            states[t].clear_forces()
            wp.launch(const_wrench_world, dim=1, inputs=(thrust,), outputs=(states[t].body_f,))
            solver.step(states[t], states[t + 1], None, None, dt)
            if cost is not None:
                wp.launch(traj_mse, dim=1, inputs=(states[t + 1].body_q, ref, t), outputs=(cost,))

    # reference trajectory at the true mass
    model.body_mass.assign(np.array([true_mass], dtype=np.float32))
    model.body_inv_mass.assign(np.array([1.0 / true_mass], dtype=np.float32))
    open_loop_forward()
    wp.synchronize()
    ref = wp.array(np.array([states[t].body_q.numpy()[0, :3] for t in range(1, n_steps + 1)]), dtype=wp.vec3)

    # recover: optimize inv_mass from a wrong guess
    inv_mass = model.body_inv_mass
    inv_mass.assign(np.array([1.0 / (nominal_mass * init_guess_scale)], dtype=np.float32))
    opt = wp.optim.Adam([inv_mass], lr=lr * (1.0 / nominal_mass))
    cost = wp.zeros(1, dtype=float, requires_grad=True)
    history = []
    for _ in range(iters):
        cost.zero_()
        tape = wp.Tape()
        with tape:
            open_loop_forward(cost=cost, ref=ref)
        tape.backward(cost)
        opt.step([inv_mass.grad])
        tape.zero()
        inv_mass.assign(np.clip(inv_mass.numpy(), 1e-3, 10.0))
        recovered = float(1.0 / inv_mass.numpy()[0])
        history.append({"mass": recovered, "loss": float(cost.numpy()[0])})
    return {"true_mass": float(true_mass), "recovered_mass": history[-1]["mass"], "history": history}


def main():
    import time

    # The system-ID math gate: lumped single body, CPU backend, light + bit-exact.
    wp.set_device("cpu")
    t0 = time.time()
    mr = recover_mass(true_mass_scale=1.4, init_guess_scale=0.7, n_steps=200, iters=150, lr=0.05)
    err = abs(mr["recovered_mass"] - mr["true_mass"]) / mr["true_mass"]
    na.logger.info(f"mass recovery: true={mr['true_mass']:.3f} recovered={mr['recovered_mass']:.3f} ({err * 100:.2f}%)")
    # Evaluation artifacts first, before the gate can raise, since a failed run must still leave its
    # stats for diagnosis; no flight, so the stats JSON only.
    dump_stats(
        "mass_recovery",
        {
            "mass_rel_err": round(err, 4),
            "true_mass_kg": round(mr["true_mass"], 4),
            "recovered_mass_kg": round(mr["recovered_mass"], 4),
            "wall_s": round(time.time() - t0, 1),
        },
    )
    assert err < 0.05, "mass not recovered"
    na.logger.info("OK: mass recovered from an open-loop trajectory (differentiable system-ID)")


if __name__ == "__main__":
    main()
