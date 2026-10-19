"""The serial link the run owns between two peers: bytes written to one end come out of the other, whole and in
order, and a stop ends the relay and both devices. The two ends are the system boundary: the tests open them as a
peer would, as serial devices by path.
"""

import os
import select
import threading
import tty

from nexus_uav._src.peers.serial import SerialLink


def _open(path: str) -> int:
    """`path` opened as a peer opens its end: read-write, raw, no controlling terminal."""
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
    tty.setraw(fd)
    return fd


def _read(fd: int, count: int, seconds: float = 2.0) -> bytes:
    """Up to `count` bytes from `fd`, waiting at most `seconds` for them."""
    out = b""
    while len(out) < count and select.select([fd], [], [], seconds)[0]:
        out += os.read(fd, count - len(out))
    return out


def _is_device(path: str) -> bool:
    """Whether `path` opens as a terminal."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        return os.isatty(fd)
    finally:
        os.close(fd)


def test_bytes_written_to_one_end_come_out_of_the_other_whole_and_in_order():
    """Bytes written to one end come out of the other, whole and in order, both ways.

    Given a link between `radio` and `px4_sitl` with both ends open, when `radio` writes 100 KiB of a counting pattern
    and `px4_sitl` writes `hello`, then `px4_sitl` reads the 100 KiB unchanged and `radio` reads `hello`.
    """
    link = SerialLink("radio", "px4_sitl")
    pattern = bytes(range(256)) * 400
    radio, px4 = _open(link.paths["radio"]), _open(link.paths["px4_sitl"])
    try:
        writer = threading.Thread(target=lambda: os.write(radio, pattern))
        writer.start()
        os.write(px4, b"hello")
        seen = (_read(px4, len(pattern), 5.0), _read(radio, 5))
        writer.join(timeout=5.0)
    finally:
        os.close(radio)
        os.close(px4)
        link.stop()

    assert seen == (pattern, b"hello")


def test_a_stop_ends_the_relay_and_both_devices():
    """A stop ends the relay and both devices.

    Given a link whose two ends are devices and whose relay is alive, when it stops, then it reads dead and neither
    path opens as a terminal; a second stop changes nothing.
    """
    link = SerialLink("radio", "px4_sitl")
    before = ([_is_device(p) for p in link.paths.values()], link.alive())

    link.stop()
    link.stop()

    after = ([_is_device(p) for p in link.paths.values()], link.alive())
    assert (before, after) == (([True, True], True), ([False, False], False))
