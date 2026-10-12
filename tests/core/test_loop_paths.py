"""The loop resolves each component's path once, where it wires the component, and hands it a
logger scoped to that path. Warp-free stand-ins, as in test_loop_ambient.py.
"""

import pytest

from nexus_sim._src.core.interfaces import Component, Stage
from nexus_sim._src.core.loop import Loop
from nexus_sim._src.core.schema import SimTime


class _Clock:
    dt = 0.004

    def now(self):
        return SimTime()


class _Sink:
    """A Logger stand-in whose scoped view is the path the loop asked for."""

    def scoped(self, path):
        return path

    def write(self, histories, start, stop=None):
        pass

    def log_rtf(self, rtf):
        pass

    def log_profile(self, stats):
        pass


class _Component(Component):
    """A component that keeps the logger the loop hands it."""

    def __init__(self, name=None):
        if name is not None:
            self.name = name
        self.handed = "nothing"

    def set_logger(self, logger):
        self.handed = logger

    def connect(self, links):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("work", "device", lambda tick: None)]


def _loop(logger, *, sensors=()):
    physics, actuator, controller = _Component(), _Component("rotors"), _Component("pid")
    loop = Loop(
        clock=_Clock(), physics=physics, actuator=actuator, sensors=list(sensors), controller=controller, logger=logger
    )
    return loop, physics, actuator, controller


def test_the_loop_hands_each_component_a_logger_scoped_to_its_role_folder_and_its_name():
    """The loop hands each component a logger scoped to its role folder and its name,
    `vehicle/actuators/<name>`, `vehicle/controllers/<name>` and `vehicle/sensors/<name>`, and physics, a
    fixed part, none.
    """
    imu = _Component("imu")
    _, physics, actuator, controller = _loop(_Sink(), sensors=[imu])

    assert (physics.handed, actuator.handed, controller.handed, imu.handed) == (
        "nothing",
        "vehicle/actuators/rotors",
        "vehicle/controllers/pid",
        "vehicle/sensors/imu",
    )


def test_the_loop_hands_a_command_element_and_a_force_element_a_logger_scoped_to_their_role_folders():
    """The loop hands a command element and a force element a logger scoped to `vehicle/commands/<name>`
    and `vehicle/forces/<name>`, their role folders.
    """
    command, force = _Component("rotors"), _Component("propellers")
    Loop(
        clock=_Clock(),
        physics=_Component(),
        commands=[command],
        forces=[force],
        sensors=[],
        controller=_Component("pid"),
        logger=_Sink(),
    )

    assert (command.handed, force.handed) == ("vehicle/commands/rotors", "vehicle/forces/propellers")


def test_a_component_added_after_the_build_gets_a_logger_scoped_to_the_path_it_is_added_at():
    """A component added after the build, the guidance, gets a logger scoped to the path it's added at."""
    loop, *_ = _loop(_Sink())
    guidance = _Component()
    loop.add_loggable(guidance, "guidance")

    assert guidance.handed == "guidance"


def test_with_no_logger_the_loop_hands_each_component_none():
    """With no logger, the loop hands each component `None`: the one switch that turns logging off."""
    imu = _Component("imu")
    _, _, actuator, controller = _loop(None, sensors=[imu])

    assert (actuator.handed, controller.handed, imu.handed) == (None, None, None)


def test_two_sensors_that_share_a_name_fail_the_build_and_the_error_names_both():
    """Two sensors that share a name fail the build, with or without a logger, and the error names the name
    and the class of each; the builder names the prims for the sensors a vehicle declares.
    """
    sensors = [_Component("cam"), _Component("cam")]
    with pytest.raises(ValueError) as err:
        _loop(None, sensors=sensors)

    assert ("'cam'" in str(err.value), str(err.value).count("_Component")) == (True, 2)
