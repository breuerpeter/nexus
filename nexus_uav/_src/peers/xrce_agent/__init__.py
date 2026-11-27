"""The Micro XRCE-DDS agent peer: the bridge between PX4's topics and the Data Distribution Service (DDS).

PX4 runs the client end of the bridge. The agent is the other end: it takes each topic PX4 publishes and puts
it on DDS, and hands PX4 each topic a DDS writer sends it. On a real vehicle the agent runs on the companion
computer. A vehicle declares it with ``NexusXrceAgentAPI``, and a run then starts one agent in a container.

:mod:`.runner` starts and stops the container, ``pin.ref`` names the agent's commit, and ``image/`` is the
build context of its image, which builds that commit.

The agent listens on ``AGENT_PORT`` plus the run's instance, and PX4 dials the same port, so two runs on one
machine each have their own agent.
"""

AGENT_PORT = 8888
"""The User Datagram Protocol (UDP) port instance 0's agent listens on, PX4's own default."""

__all__ = ["AGENT_PORT"]
