"""The controllers: each turns what it reads into its controls, one command per actuator it drives.

A controller's stages run once per tick, after the guidance. Two ship. PX4's, in ``px4/``, flies the vehicle over
its lockstep link. The gimbal's, :class:`~nexus_uav._src.vehicle.controllers.gimbal.GimbalController`, holds a
camera's attitude through the torque motors of three joints, with a law that runs as one kernel inside the
captured graph. A vehicle that carries both authors ``nexus:inputs:controls`` on every reader of the controls.
"""
