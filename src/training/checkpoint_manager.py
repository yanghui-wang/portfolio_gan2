from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointState:
    epoch: int
    global_step: int
    best_metric: float


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        payload: dict[str, Any],
    ) -> Path:
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(payload, path)
        return path

    def load(self, path: Path) -> dict[str, Any]:
        return torch.load(path, map_location="cpu")

