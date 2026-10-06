"""The nexus Universal Scene Description (USD) schema plugin and its reader.

The plugin is codeless: `plugInfo.json` and `generatedSchema.usda` beside this file, and no compiled code.
"""

from __future__ import annotations

import json
from importlib.metadata import entry_points
from importlib.util import find_spec
from pathlib import Path

_PLUGIN = Path(__file__).parent

# The entry-point group a package declares its component classes in, one entry per schema:
# `NexusImuAPI = "nexus_sim._src.vehicle.sensors:ImuSensor"`.
ENTRY_POINT_GROUP = "nexus.components"


def register_plugins() -> None:
    """Register the nexus schema plugin with OpenUSD, then the plugin of each package with a component entry.

    A package with an entry in the `nexus.components` group ships its plugin's `plugInfo.json` at the root
    of its top-level package, as Newton's schema package does. This finds the package without importing
    it, so a class with heavy imports behind it costs nothing until a run declares its schema.

    OpenUSD builds its schema registry once, on first use, and a plugin registered after that never shows
    in it, so `import nexus_sim` calls this before any caller can open a stage.
    """
    from pxr import Plug

    paths = [str(_PLUGIN)]
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        spec = find_spec(entry.module.partition(".")[0])
        locations = (spec.submodule_search_locations or []) if spec else []
        paths += [location for location in locations if (Path(location) / "plugInfo.json").is_file()]
    Plug.Registry().RegisterPlugins(paths)


def schema_names() -> tuple[str, ...]:
    """The identifiers of every applied API schema the nexus plugin defines, from its `plugInfo.json`."""
    (plugin,) = json.loads((_PLUGIN / "plugInfo.json").read_text())["Plugins"]
    return tuple(info["schemaIdentifier"] for info in plugin["Info"]["Types"].values())
