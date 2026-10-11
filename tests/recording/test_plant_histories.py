"""The plant writes signals, and every history reads as records, #275.

Real builds of the fixture quad in ``tests/vehicle/quad.py`` on the Warp CPU backend: a cube body of 0.2 m on
four rotor bodies, each on a revolute joint under a free base joint. A test reads the plant's histories
through ``Sim``, as a script does. Skipped without newton or pxr.
"""

import contextlib
import itertools

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import warp as wp

import nexus_sim as nx
from tests.vehicle import quad

pytestmark = pytest.mark.usefixtures("warp_cpu")

TICKS = 5
# The quad's base body is a cube of 0.2 m, so at rest on the ground its origin sits 0.1 m over it.
REST_HEIGHT = 0.1
BODY_FIELDS = {"t", "position", "orientation", "linear_velocity", "angular_velocity"}


def _shapes(rows) -> dict[str, tuple[int, ...]]:
    """Each field of structured rows with its shape per row, or nothing when the rows aren't structured."""
    dtype = getattr(rows, "dtype", None)
    if dtype is None or not dtype.names:
        return {}
    return {name: dtype[name].shape for name in dtype.names}


@contextlib.contextmanager
def _flown(path, ticks: int, controller=None):
    """A run of the quad at `path` under `controller`, full throttle by default, stepped `ticks` ticks on the
    CPU device and open for a test to read, then closed.
    """
    with wp.ScopedDevice("cpu"), nx.Sim._from_loop(quad.build(path, steps=ticks + 1, controller=controller)) as sim:
        for _ in range(ticks):
            assert sim.step()
        yield sim


def test_a_bodys_history_reads_as_a_signals_does_one_structured_array_one_record_one_array_per_quantity(tmp_path):
    """A body's history reads as a signal's does: `history()` one structured array of `t` then `position`,
    `orientation`, `linear_velocity` and `angular_velocity`, `latest()` one record and `arrays()` one array per
    quantity, holding the state the plant holds.

    Given the fixture quad at rest on the CPU device, when the run steps 5 ticks, then
    `sim[f"vehicle/body/{sim.base_body}"].history()` has 6 rows whose dtype names `t`, `position (3,)`,
    `orientation (4,)`, `linear_velocity (3,)` and `angular_velocity (3,)`, float32, `latest()["position"][2]`
    is the base body's height over the ground and `arrays()["position"]` has shape `(6, 3)`.
    """
    at_rest = quad.Commands((0.0, 0.0, 0.0, 0.0))
    with _flown(quad.author(tmp_path / "quad.usda"), TICKS, controller=at_rest) as sim:
        body = sim[f"vehicle/body/{sim.base_body}"]
        rows, latest, arrays = body.history(), body.latest(), body.arrays()
    shapes = _shapes(rows)
    height = float(np.ravel(latest["position"])[2]) if _shapes(latest) else None

    assert (
        len(rows),
        shapes,
        {name: str(rows.dtype[name].base) for name in shapes if name != "t"},
        height,
        arrays["position"].shape,
    ) == (
        TICKS + 1,
        {"t": (), "position": (3,), "orientation": (4,), "linear_velocity": (3,), "angular_velocity": (3,)},
        dict.fromkeys(BODY_FIELDS - {"t"}, "float32"),
        pytest.approx(REST_HEIGHT, abs=5e-3),
        (TICKS + 1, 3),
    )


def test_a_joints_history_reads_as_records_of_t_q_and_qd_each_of_that_joints_own_width(tmp_path):
    """A joint's history reads as records of `t`, `q` and `qd`, each of that joint's own width.

    Given the fixture quad with its gimbal joint at full throttle, when the run steps 5 ticks, then
    `sim["vehicle/joints/joint_1"].history()` has `q (7,)` and `qd (6,)`,
    `sim["vehicle/joints/rotor_0_joint"].history()` has `q (1,)` and `qd (1,)`, and the rotor joint's `qd`
    rows rise from the second row on.
    """
    with _flown(quad.author(tmp_path / "quad.usda", gimbal=True), TICKS) as sim:
        base, rotor = sim["vehicle/joints/joint_1"].history(), sim["vehicle/joints/rotor_0_joint"].history()
    spin = [float(np.ravel(row["qd"])[0]) for row in rotor] if "qd" in _shapes(rotor) else []
    rising = [b > a for a, b in itertools.pairwise(spin[1:])]

    assert (_shapes(base), _shapes(rotor), len(spin), rising) == (
        {"t": (), "q": (7,), "qd": (6,)},
        {"t": (), "q": (1,), "qd": (1,)},
        TICKS + 1,
        [True] * (TICKS - 1),
    )
