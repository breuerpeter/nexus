"""The loop resolves each component's path once, where it wires the component, and hands it a
logger scoped to that path. Warp-free stand-ins, as in test_orchestrator_seams.py.
"""

import pytest

from nexus._src.core.interfaces import Stage
from nexus._src.core.orchestrator import Orchestrator
from nexus._src.core.schema import SimTime


class _Clock:
    dt = 0.004

    def now(self):
        return SimTime()


class _Sink:
    """A Logger stand-in whose scoped view is the path the loop asked for."""

    log_interval = 0.0

    def scoped(self, path):
        return path


class _Component:
    """A component that keeps the logger the loop hands it."""

    def __init__(self, name=None, prim_path=None):
        if name is not None:
            self.name = name
        if prim_path is not None:
            self.prim_path = prim_path
        self.handed = "nothing"

    def set_logger(self, logger):
        self.handed = logger

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("work", "device", lambda tick: None)]


def _loop(logger, *, sensors=()):
    physics, actuator, controller = _Component(), _Component("rotors"), _Component("pid")
    loop = Orchestrator(
        clock=_Clock(), physics=physics, actuator=actuator, sensors=list(sensors), controller=controller, logger=logger
    )
    return loop, physics, actuator, controller


def test_the_loop_hands_each_component_a_logger_scoped_to_its_role_folder_and_its_name():
    """The loop hands each component a logger scoped to its role folder and its name: `vehicle` for
    physics, `vehicle/actuators/<name>`, `vehicle/controllers/<name>` and `vehicle/sensors/<name>`.
    """
    imu = _Component("imu")
    _, physics, actuator, controller = _loop(_Sink(), sensors=[imu])

    assert (physics.handed, actuator.handed, controller.handed, imu.handed) == (
        "vehicle",
        "vehicle/actuators/rotors",
        "vehicle/controllers/pid",
        "vehicle/sensors/imu",
    )


def test_a_component_added_after_the_build_gets_a_logger_scoped_to_the_path_it_is_added_at():
    """A component added after the build, the operator, gets a logger scoped to the path it's added at."""
    loop, *_ = _loop(_Sink())
    operator = _Component()
    loop.add_loggable(operator, "operator")

    assert operator.handed == "operator"


def test_with_no_logger_the_loop_hands_each_component_none():
    """With no logger, the loop hands each component `None`: the one switch that turns logging off."""
    imu = _Component("imu")
    _, physics, actuator, controller = _loop(None, sensors=[imu])

    assert (physics.handed, actuator.handed, controller.handed, imu.handed) == (None, None, None, None)


def test_two_sensors_that_share_a_name_fail_the_build_and_the_error_names_both_prims():
    """Two sensors that share a name fail the build, with or without a logger, and the error names
    the prim that declares each.
    """
    sensors = [_Component("cam", "/vehicle/body/cam"), _Component("cam", "/vehicle/body/rotor_1/cam")]
    with pytest.raises(ValueError) as err:
        _loop(None, sensors=sensors)

    assert ("/vehicle/body/cam" in str(err.value), "/vehicle/body/rotor_1/cam" in str(err.value)) == (True, True)
