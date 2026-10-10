"""Each controller declares its controls signal with its own width, and the command elements read its leading
part, #44.

A real build of the fixture vehicle in ``tests/usd/sensor_vehicle.py`` on the Warp CPU backend, flown by a
stand-in controller 20 channels wide. Skipped without newton or pxr.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import warp as wp

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.signals import Signal
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

TICKS = 5
WIDTH = 20


class _Wide:
    """A stand-in controller named `wide` whose host stage writes a 20-wide controls signal, 0.5 on every channel."""

    name = "wide"

    def __init__(self, **kwargs):
        self.controls = Signal("controls", wp.float32, shape=(1, WIDTH))

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("act", "host", self._act, writes=(self.controls,))]

    def _act(self, tick):
        self.controls.write(np.full((1, WIDTH), 0.5, dtype=np.float32))
        return True


def test_a_controller_declares_its_controls_signal_with_its_own_width(tmp_path):
    """Each controller declares its controls signal with its own width, and the command elements read its leading
    part.

    Given a stand-in controller whose host stage writes a 20-wide controls signal on the fixture vehicle, when
    the run flies 5 ticks on the CPU device, then it completes them and the recorded controls history is 20 wide.
    """
    loop = sv.build(sv.vehicle(tmp_path), components=sv.components(NexusPx4API=_Wide))
    ran = sv.steps(loop, TICKS)
    rows = loop.recorder.histories["vehicle/controllers/wide/controls"].history()

    assert (ran, np.asarray(rows["controls"]).shape[-1]) == (TICKS, WIDTH)
