from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class DeviceDiagnostics:
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    cudnn_enabled: bool
    device_name: str
    total_gpu_memory_gb: float | None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_device_diagnostics(device: torch.device) -> DeviceDiagnostics:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_gb = round(props.total_memory / (1024**3), 2)
        name = props.name
    else:
        total_gb = None
        name = "cpu"

    return DeviceDiagnostics(
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        cudnn_enabled=torch.backends.cudnn.enabled,
        device_name=name,
        total_gpu_memory_gb=total_gb,
    )


def diagnostics_as_dict(diag: DeviceDiagnostics) -> dict[str, object]:
    return asdict(diag)

