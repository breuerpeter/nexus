"""The check that a released schema version doesn't change in place.

`released.json` beside this file records each released version's attributes with their type and unit. A
version changes only by a new version, so an attribute in the record keeps its name, type and unit until
its version retires.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RECORD = Path(__file__).with_name("released.json")

_UNITS = re.compile(r"^\s*Units:\s*(.+?)\s*$", re.MULTILINE)


def schema_record(plugin: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Each schema the plugin folder at `plugin` defines, with the type and unit of each of its attributes."""
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(plugin / "generatedSchema.usda"))
    record = {}
    for schema in layer.rootPrims:
        record[schema.name] = {}
        for attribute in schema.properties:
            units = _UNITS.search(attribute.documentation)
            record[schema.name][attribute.name] = {
                "type": str(attribute.typeName),
                "units": units.group(1) if units else "",
            }
    return record


def in_place_changes(plugin: Path) -> list[str]:
    """One message per released attribute the plugin folder at `plugin` lost, renamed, retyped or gave another unit.

    Each message names the schema version and the attribute. An attribute that joins a released version
    with a fallback is no change.
    """
    now = schema_record(plugin)
    changes = []
    for schema, attributes in json.loads(RECORD.read_text()).items():
        if schema not in now:
            changes.append(f"{schema}: the plugin no longer defines this released version")
            continue
        for name, released in attributes.items():
            found = now[schema].get(name)
            if found is None:
                changes.append(f"{schema}: {name} is gone or renamed")
            elif found != released:
                changes.append(f"{schema}: {name} was {released}, and is now {found}")
    return changes
