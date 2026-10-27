"""The RTX sensors and the render stage they require.

A camera, a thermal camera and a lidar each fill the sensor role over a prim the vehicle file authors, :mod:`.camera`,
:mod:`.thermal` and :mod:`.lidar` on the base in :mod:`.sensor`. Each class requires the render stage, :mod:`.stage`,
by the name ``ovrtx``: the build makes it once per run and the loop hands it to each sensor at ``connect``.
:mod:`.fake` is its fake, for a run with no renderer. No module here imports ovrtx or ovstage at its own import.
"""
