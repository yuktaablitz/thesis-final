"""
Single loader for the YAML configs in configs/. approach2.yaml declares
`extends: approach1.yaml`; load_config follows that chain and merges child
keys over parent keys (shallow-merge on top-level, deep-merge one level
into dict-valued keys) so the target/split definition is never duplicated.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(name: str) -> dict[str, Any]:
    """
    Load a config by filename (e.g. "approach1.yaml" or "approach2.yaml") or
    bare name ("approach1"). Follows `extends:` chains.
    """
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    path = CONFIG_DIR / name
    with open(path) as f:
        cfg = yaml.safe_load(f)

    parent_name = cfg.get("extends")
    if parent_name:
        parent_cfg = load_config(parent_name)
        cfg = _deep_merge(parent_cfg, cfg)

    return cfg


if __name__ == "__main__":
    import json

    print(json.dumps(load_config("approach2.yaml"), indent=2, default=str))
