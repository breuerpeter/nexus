"""Renderer glue: the one shared Kit frame service every RTX sensor rides.

Kit boot, the container plumbing and the benchmarking stay runtime glue under
``runtimes/isaacsim``. The RTX sensors are vehicle components and live under ``vehicle/sensors``,
with the stage queries and the pose math they share with the frame.
"""
