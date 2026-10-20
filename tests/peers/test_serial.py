"""The serial link the run owns between two peers: bytes written to one end come out of the other, whole and in
order, and a stop ends the relay and both devices. The two ends are the system boundary: the tests open them as a
peer would, as serial devices by path.
"""

import os
import select
import threading
import tty

from nexus_uav._src.peers.serial import SerialLink

LINE = "/vehicle/Companion.nexus:serial:autopilot"
"""The USD name of the line a test's link stands for."""


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

    Given a link with both ends open, when the first end writes 100 KiB of a counting pattern and the second writes
    `hello`, then the second reads the 100 KiB unchanged and the first reads `hello`.
    """
    link = SerialLink(LINE)
    pattern = bytes(range(256)) * 400
    first, second = _open(link.ends[0]), _open(link.ends[1])
    try:
        writer = threading.Thread(target=lambda: os.write(first, pattern))
        writer.start()
        os.write(second, b"hello")
        seen = (_read(second, len(pattern), 5.0), _read(first, 5))
        writer.join(timeout=5.0)
    finally:
        os.close(first)
        os.close(second)
        link.stop()

    assert seen == (pattern, b"hello")


def test_a_stop_ends_the_relay_and_both_devices():
    """A stop ends the relay and both devices.

    Given a link whose two ends are devices and whose relay is alive, when it stops, then it reads dead and neither
    path opens as a terminal; a second stop changes nothing.
    """
    link = SerialLink(LINE)
    before = ([_is_device(p) for p in link.ends], link.alive())

    link.stop()
    link.stop()

    after = ([_is_device(p) for p in link.ends], link.alive())
    assert (before, after) == (([True, True], True), ([False, False], False))
