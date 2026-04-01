from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BaselineOutput:
    weights: torch.Tensor
    name: str


def zero_trade_baseline(w_prev: torch.Tensor) -> BaselineOutput:
    return BaselineOutput(weights=w_prev.clone(), name="zero_trade")


def turnover_matched_random(w_prev: torch.Tensor, turnover_scale: float = 0.01) -> BaselineOutput:
    noise = torch.randn_like(w_prev) * turnover_scale
    candidate = torch.relu(w_prev + noise)
    normalized = candidate / (candidate.sum(dim=-1, keepdim=True) + 1e-8)
    return BaselineOutput(weights=normalized, name="turnover_matched_random")


def factor_tilt_matched(w_prev: torch.Tensor, factor_scores: torch.Tensor) -> BaselineOutput:
    tilt = torch.softmax(factor_scores, dim=-1)
    adjusted = torch.relu(w_prev + 0.05 * tilt)
    normalized = adjusted / (adjusted.sum(dim=-1, keepdim=True) + 1e-8)
    return BaselineOutput(weights=normalized, name="factor_tilt_matched")


def generator_only_ablation(weights: torch.Tensor) -> BaselineOutput:
    normalized = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    return BaselineOutput(weights=normalized, name="generator_only")

