"""The run's port map: each link that leaves the run, by name, to the address its client opens.

A lookup of a link the map holds returns its entry. A lookup of one it lacks fails at once,
with the reason the builder gave for keeping it out when it gave one, else with the links the map
holds, so a script learns why before it opens a client on nothing.
"""

import pytest

from nexus._src.core.ports import PortMap


def test_a_link_the_builder_left_out_fails_with_its_reason():
    """A lookup of a link the builder left out with a reason fails with that reason."""
    ports = PortMap(
        {"viewer": {"protocol": "grpc", "port": 9876}}, missing={"offboard": "the fake answers no offboard link"}
    )

    with pytest.raises(LookupError, match="the fake answers no offboard link"):
        ports["offboard"]


def test_a_link_the_map_never_heard_of_fails_naming_the_links_it_holds():
    """A lookup of a link the map holds no entry and no reason for fails naming the links it holds."""
    ports = PortMap({"viewer": {"protocol": "grpc", "port": 9876}})

    with pytest.raises(LookupError, match=r"no 'offboard' link.*'viewer'"):
        ports["offboard"]


def test_a_link_the_map_holds_reads_as_its_entry():
    """A lookup of a link the map holds returns its entry, and the map reads as a mapping of the links it holds."""
    ports = PortMap({"offboard": {"protocol": "udp", "port": 14540, "system_id": 1}}, missing={"viewer": "no viewer"})

    assert (ports["offboard"], dict(ports), "offboard" in ports, "viewer" in ports) == (
        {"protocol": "udp", "port": 14540, "system_id": 1},
        {"offboard": {"protocol": "udp", "port": 14540, "system_id": 1}},
        True,
        False,
    )
