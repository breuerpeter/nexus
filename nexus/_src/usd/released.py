"""The check that a released schema version doesn't change in place.

`released.json` beside this file records each released schema version: each attribute's type,
fallback and unit. A version changes only by a new version, so an attribute in the record keeps its name,
type, fallback and unit until its version retires.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PLUGIN = Path(__file__).parent
RECORD = PLUGIN / "released.json"

_UNITS = re.compile(r"^\s*Units:\s*(.+?)\s*$", re.MULTILINE)


def schema_record(plugin: Path) -> dict[str, dict[str, dict[str, str | None]]]:
    """Each schema the plugin folder at `plugin` defines, with the type, fallback and unit of each of its attributes.

    A fallback is its text in the schema file, and `None` when the attribute has none.
    """
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(plugin / "generatedSchema.usda"))
    record = {}
    for schema in layer.rootPrims:
        record[schema.name] = {}
        for attribute in schema.properties:
            units = _UNITS.search(attribute.documentation)
            fallback = attribute.default
            record[schema.name][attribute.name] = {
                "type": str(attribute.typeName),
                "fallback": None
                if fallback is None
                else f"{fallback:g}"
                if isinstance(fallback, float)
                else str(fallback),
                "units": units.group(1) if units else "",
            }
    return record


def in_place_changes(plugin: Path) -> list[str]:
    """One message per released attribute the plugin folder at `plugin` lost, renamed, retyped, or gave another fallback or unit.

    Each message names the schema version and the attribute. An attribute that joins a released version
    with a fallback is no change, and one that joins with none is.
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
        for name, found in now[schema].items():
            if name not in attributes and found["fallback"] is None:
                changes.append(f"{schema}: {name} joins with no fallback")
    return changes
