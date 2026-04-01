from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.utils.io import save_csv


def evaluate_reconstruction(w_hat: torch.Tensor, w_true: torch.Tensor) -> dict[str, float]:
    abs_err = (w_hat - w_true).abs()
    l_count = (abs_err > 1e-3).float().sum(dim=-1).mean().item()
    l_concentration = abs_err.max(dim=-1).values.mean().item()
    l_turnover = abs_err.sum(dim=-1).mean().item()
    return {
        "L_count": float(l_count),
        "L_concentration": float(l_concentration),
        "L_turnover": float(l_turnover),
    }


def run_evaluation_stub(outputs_dir: Path) -> None:
    diagnostics = pd.DataFrame(
        {
            "metric": ["status", "note"],
            "value": [
                "scaffold_only",
                "Full paper metric formulas pending exact field mapping.",
            ],
        }
    )
    save_csv(diagnostics, outputs_dir / "diagnostics" / "evaluation_stub_summary.csv")

