"""The Kit render peer: NVIDIA's Isaac Sim image, running the program in ``peer-src/``.

:mod:`.runner` pulls the image and runs the container. ``peer-src/`` is package data the container
mounts read-only, and no host module imports it, so no host tier ever loads Kit. Its ``link.py``
defines the render link, the framing and the message set, and the host end,
:mod:`nexus_sim._src.rendering.link`, runs that file by path.
"""
