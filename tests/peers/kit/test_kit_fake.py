"""The fake Kit peer: the class a peer mapping sends the Kit peer to, which starts no container and answers
each ``frame`` request with a synthetic frame. No RTX sensor requires the Kit peer any more, so a run never reaches
it; the fake stays with the peer until the issue that removes both lands.
"""

import time

from nexus_uav._src.peers.kit.fake import KitFake


def test_the_kit_fake_stops_at_once_when_no_run_connected_to_it():
    """The Kit fake stops at once when no run connected to it."""
    fake = KitFake()
    fake.start()
    began = time.monotonic()
    fake.stop()
    assert time.monotonic() - began < 1.0
