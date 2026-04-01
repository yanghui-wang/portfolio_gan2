from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ConfigBundle:
    paths: dict[str, Any]
    data: dict[str, Any]
    model: dict[str, Any]
    train: dict[str, Any]
    eval: dict[str, Any]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be mapping: {path}")
    return loaded


def load_config_bundle(config_dir: Path) -> ConfigBundle:
    return ConfigBundle(
        paths=load_yaml(config_dir / "paths.yaml"),
        data=load_yaml(config_dir / "data.yaml"),
        model=load_yaml(config_dir / "model.yaml"),
        train=load_yaml(config_dir / "train.yaml"),
        eval=load_yaml(config_dir / "eval.yaml"),
    )


def resolve_path(project_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate

