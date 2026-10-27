"""A run's render mode: synchronous unless `Sim` or the command line asks for the pipelined render, and the receipt,
which the recording holds as its settings document, says which.

Each test builds the fixture vehicle on the Warp CPU backend, flown by the stand-in controller, steps it once and
reads the settings document from the `.rrd` the run wrote. Skipped without newton, pxr or rerun.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")
pytest.importorskip("rerun")

from nexus_uav._src.api.args import sim_argparser
from nexus_uav._src.api.sim import Sim
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")


def _from_sim(vehicle, run_dir, **kw):
    return Sim(vehicle, scene=sv.SCENE, device="cpu", log=True, registry=sv.components(), run_dir=run_dir, **kw)


def _from_command_line(vehicle, run_dir, *flags):
    args = sim_argparser().parse_args(["--vehicle", vehicle, "--scene", sv.SCENE, "--device", "cpu", "--log", *flags])
    return Sim.from_args(args, registry=sv.components(), run_dir=run_dir)


ENTRIES = {
    "default": lambda vehicle, run_dir: _from_sim(vehicle, run_dir),
    "sim": lambda vehicle, run_dir: _from_sim(vehicle, run_dir, pipelined_render=True),
    "command_line": lambda vehicle, run_dir: _from_command_line(vehicle, run_dir, "--pipelined-render"),
}


def _receipt_value(rrd: str, key: str, rrd_rows) -> str | None:
    """The value the recording's settings document gives `key`, a dotted path such as `runtime.seed`, or None when
    the document has no row for it.
    """
    for entity, _, columns in rrd_rows(rrd):
        if entity == "/sim/run/settings" and "TextDocument:text" in columns:
            cell = columns["TextDocument:text"].to_pylist()[0]
            for line in (cell[0] if isinstance(cell, list) else cell).splitlines():
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                if len(cells) == 2 and cells[0].strip("`") == key:
                    return cells[1].strip("`")
    return None


@pytest.mark.parametrize(("entry", "expected"), [("default", "False"), ("sim", "True"), ("command_line", "True")])
def test_a_runs_receipt_records_whether_it_pipelines_the_render_off_unless_asked(entry, expected, tmp_path, rrd_rows):
    """`Runtime.pipelined_render` is off by default. `Sim(pipelined_render=True)` and `nexus run --pipelined-render`
    turn it on, and the run's receipt records the value.

    Given the fixture vehicle, when a run built by default, by `Sim(pipelined_render=True)` or from the command line
    with `--pipelined-render` steps once, then the receipt in its recording gives `runtime.pipelined_render` as
    `False`, `True` and `True`.
    """
    with ENTRIES[entry](sv.vehicle(tmp_path), tmp_path / "run") as sim:
        sim.step()
        rrd = sim.artifacts()["rrd"]

    assert _receipt_value(rrd, "runtime.pipelined_render", rrd_rows) == expected
