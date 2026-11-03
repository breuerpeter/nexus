"""The rotors' Electronic Speed Controllers (ESCs): the controller's command to each rotor's terminal voltage.

A brushless drive sets nothing but the fraction of its bus it switches onto the windings. The ESC reads the
controller's normalized command ``u`` on its channel and writes the terminal voltage ``V = d(u)·V_bus``, where
``d(u) = idleDuty + (maxDuty − idleDuty)·clamp(u, 0, 1)`` is the duty map its own attributes state: ``idleDuty``
is the idle a real drive spins at under command 0, and ``maxDuty`` below 1 holds back the top. ``V_bus`` is the
``bus_voltage`` signal, a battery's in flight, and the ESC's own ``busVoltage`` until a component writes it. The
rotor's motor, :class:`~nexus_uav._src.vehicle.actuators.motor.Motor` on the same body, turns the voltage into
torque in the actuator stage after it, so the ESC reads nothing of its propeller's or its motor's.

Each rotor's rigid body declares its ESC with the ``NexusEscAPI`` schema, beside its motor's and its propeller's,
and the registry builds it as :class:`Esc`, a command element with one device stage, ``esc``, so a captured graph
replays it over the controls the controller writes. :class:`Escs` runs the same kernel over the rotors of many
vehicles at once, as the RL task and the gradient rollout step them.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from nexus_uav._src.core.interfaces import Component, Stage
from nexus_uav._src.core.signals import Signal


@wp.kernel
def _terminal_voltage(
    cmd: wp.array2d(dtype=float),  # (vehicles, n) normalized [0, 1] commands, one row per vehicle
    bus: wp.array2d(dtype=float),  # (vehicles, nr) the bus voltage each rotor's ESC switches [V]
    channels: wp.array(dtype=wp.int32),  # (nr,) the entry of its vehicle's row each rotor reads
    idle_duty: wp.array(dtype=float),  # (nr,) each ESC's duty at command 0
    max_duty: wp.array(dtype=float),  # (nr,) each ESC's duty at command 1
    voltage: wp.array2d(dtype=float),  # (vehicles, nr) out: each rotor's terminal voltage [V]
):
    """Normalized command → the duty the map gives → the terminal voltage, that fraction of the bus."""
    v, i = wp.tid()
    u = wp.clamp(cmd[v, channels[i]], 0.0, 1.0)
    voltage[v, i] = (idle_duty[i] + (max_duty[i] - idle_duty[i]) * u) * bus[v, i]


class Escs:
    """The ESCs of rotors the caller has found: one row of commands per vehicle, as the RL task steps many
    vehicles in one model, each rotor with its own duty map.

    Args:
        channels: ``(nr,)`` the entry of its vehicle's command row each rotor reads.
        idle_duty: ``(nr,)`` each ESC's duty at command 0.
        max_duty: ``(nr,)`` each ESC's duty at command 1.
        device: The Warp device the arrays live on; the current one by default.
    """

    def __init__(self, *, channels, idle_duty, max_duty, device=None):
        self.nr = len(channels)
        self._channels = wp.array(np.asarray(channels, dtype=np.int32), dtype=wp.int32, device=device)
        self._idle_duty = wp.array(np.asarray(idle_duty, dtype=np.float32), dtype=float, device=device)
        self._max_duty = wp.array(np.asarray(max_duty, dtype=np.float32), dtype=float, device=device)

    def write(self, cmd: wp.array, bus: wp.array, voltage: wp.array) -> None:
        """Write each rotor's terminal voltage into ``voltage``, ``(vehicles, nr)``, from ``cmd``, one row of
        normalized commands per vehicle, and ``bus``, ``(vehicles, nr)``, the bus voltage each ESC switches.
        """
        wp.launch(
            _terminal_voltage,
            dim=(cmd.shape[0], self.nr),
            inputs=(cmd, bus, self._channels, self._idle_duty, self._max_duty),
            outputs=(voltage,),
        )


class Esc(Component):
    """The ESC of one rotor, which its rigid body declares with ``NexusEscAPI``: the controller's normalized
    command on its channel to the terminal voltage its duty map switches onto the rotor's motor.

    Args:
        model: The finalized articulated model; the ESC reads nothing of it.
        path: The rotor's rigid body, the prim that applies the schema.
        channel: The entry of the controls the ESC reads, ``nexus:channel``.
        idle_duty: The duty at command 0, ``nexus:idleDuty``.
        max_duty: The duty at command 1, ``nexus:maxDuty``.
        bus_voltage: The bus voltage [V] until a component writes the ``bus_voltage`` signal, ``nexus:busVoltage``.

    Raises:
        ValueError: ``channel`` is below 0, the duty map is outside ``0 <= idle_duty < max_duty <= 1``, or
            ``bus_voltage`` isn't positive; the message names the prim and the attribute.
    """

    def __init__(
        self, model, path: str, *, channel: int = 0, idle_duty: float = 0.0, max_duty: float = 1.0, bus_voltage: float = 0.0
    ):  # fmt: skip
        if int(channel) < 0:
            raise ValueError(
                f"{path}: nexus:channel is {channel}, and an ESC reads an entry of the controls, 0 or above"
            )
        if not 0.0 <= idle_duty < max_duty <= 1.0:
            raise ValueError(
                f"{path}: nexus:idleDuty is {idle_duty:g} and nexus:maxDuty {max_duty:g}, and a duty map runs from an "
                "idle at or above 0 to a max above it, at or below 1"
            )
        if not bus_voltage > 0.0:
            raise ValueError(f"{path}: nexus:busVoltage is {bus_voltage:g}, and the ESC's bus must be positive")
        self.channel = int(channel)
        self._escs = Escs(channels=[self.channel], idle_duty=[idle_duty], max_duty=[max_duty])
        # The controller's commands, of which this ESC reads one: the builder checks that the controller writes
        # at least that many, as PX4 streams 16 HIL_ACTUATOR_CONTROLS channels.
        self.controls = Signal("controls", wp.float32, shape=(1, self.channel + 1))
        self.bus_voltage = Signal("bus_voltage", wp.float32, shape=(1,), default=[float(bus_voltage)])
        self.voltage = Signal("voltage", wp.float32, shape=(1,))

    def stages(self) -> list[Stage]:
        """One device stage, ``esc``, reading the controls and the bus voltage and writing the terminal voltage."""
        return [
            Stage(
                "esc",
                "device",
                lambda tick: self._escs.write(
                    self.controls.buffer, self.bus_voltage.buffer.reshape((1, 1)), self.voltage.buffer.reshape((1, 1))
                ),
                reads=(self.controls, self.bus_voltage),
                writes=(self.voltage,),
            )
        ]


__all__ = ["Esc", "Escs"]
