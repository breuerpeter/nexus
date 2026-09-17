"""Scenario/config loading, per architecture.md §7: a declarative YAML + a JSON env
override, lifted from the bridge's ``get_cfg`` / ``_deep_merge``.
"""

from __future__ import annotations

from .logging import logger


def deep_merge(base: dict, override: dict, path: str = "") -> dict:
    for k, v in override.items():
        key_path = f"{path}.{k}" if path else k
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v, key_path)
        else:
            logger.info(f"Config override: {key_path} = {v!r}")
            base[k] = v
    return base
