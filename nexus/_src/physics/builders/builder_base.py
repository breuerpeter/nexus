from abc import ABC, abstractmethod
from pathlib import Path

from newton import Model, ModelBuilder

from nexus._src.core import logger


class BuilderBase(ABC):
    def __init__(self, cfg: dict, vehicle_dir: Path):
        self.cfg = cfg
        self.vehicle_dir = vehicle_dir

    @abstractmethod
    def build(self, builder: ModelBuilder) -> None:
        """Build the model using Newton's model builder."""

    def actuator_params(self) -> dict:
        """Per-vehicle actuator aero/thrust-map params such as ``ct``, ``cd``, ``rpm_max`` and ``tau``, the
        canonical single source for the actuator. Universal Scene Description (USD) vehicles declare them
        per rotor: the propeller's with the ``NexusPropellerAPI`` schema on the rotor's rigid body, and the
        speed at full command on its ``NewtonActuator`` motor; see
        :class:`~nexus._src.physics.builders.usd.USDBuilder`. A builder with no authored params returns
        ``{}`` and the consumer falls back to its cfg/defaults.
        """
        return {}

    def rotor_joints(self) -> list[str]:
        """The joint path of each rotor the vehicle declares, which
        :func:`~nexus._src.vehicle.actuators.layout.find_rotor_joints` maps onto the built model. A builder
        that declares no rotor returns ``[]``.
        """
        return []

    def model_debug_print(self, model: Model) -> None:
        for i, key in enumerate(model.body_label):
            mass = model.body_mass.numpy()[i]
            inertia = model.body_inertia.numpy()[i]
            logger.debug(f"Body {i} ({key}): mass = {mass}, inertia =\n{inertia}")
