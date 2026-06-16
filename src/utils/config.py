"""Configuration loading utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config" / "experiment.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override dict into base config."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_with_overrides(
    config_path: str | Path | None = None,
    override_path: str | Path | None = None,
) -> dict[str, Any]:
    base = load_config(config_path)
    if override_path is None:
        return base
    with open(override_path, encoding="utf-8") as f:
        override = yaml.safe_load(f) or {}
    return merge_config(base, override)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
