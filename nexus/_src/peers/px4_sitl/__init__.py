"""The PX4 Software In The Loop (SITL) peer: one PX4 autopilot in a container.

:mod:`.runner` starts and stops the container, :mod:`.checkout` fetches the PX4 tree from the pin,
``px4.ref``, and ``image/`` is the build context of the toolchain image.

PX4 numbers every link from its instance ``i``: it dials the sim's Hardware In The Loop (HIL)
server on ``HIL_PORT + i``, streams its offboard link to ``OFFBOARD_PORT + i`` and its
ground-station link from ``GCS_PORT + i``, and takes ``i + 1`` as its MAVLink system id.

The base ports below mirror PX4's own numbering at the pinned commit and aren't settings. PX4
reads none of them: the runner hands it only ``-i``, and no environment variable names a port. So
changing one moves only the host's end of a link, and they change only with the pin. The run
config, the controller and the offboard operator read them here, so the host's ends follow PX4.
"""

HIL_PORT = 4560
"""The TCP port of instance 0's HIL link: the sim listens, PX4 dials."""
OFFBOARD_PORT = 14540
"""The User Datagram Protocol (UDP) port instance 0 streams its offboard link to, where the operator listens."""
GCS_PORT = 18570
"""The UDP port instance 0 sends its ground-station link from."""

__all__ = ["GCS_PORT", "HIL_PORT", "OFFBOARD_PORT"]
