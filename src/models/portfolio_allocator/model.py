from __future__ import annotations

import torch
from torch import nn


class PortfolioAllocator(nn.Module):
    """Decoder scaffold mapping market state and strategy latent to weights."""

    def __init__(
        self,
        num_assets: int,
        num_features: int,
        latent_dim: int,
        hidden_dim: int = 256,
        output_mode: str = "softmax",
    ) -> None:
        super().__init__()
        self.output_mode = output_mode
        self.net = nn.Sequential(
            nn.Linear(num_assets * (num_features + 2) + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_assets),
        )

    def _normalize(self, logits: torch.Tensor) -> torch.Tensor:
        if self.output_mode == "softmax":
            weights = torch.softmax(logits, dim=-1)
        elif self.output_mode == "thresholded":
            sparse = torch.relu(logits)
            weights = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            raise ValueError(f"Unsupported allocator output_mode: {self.output_mode}")

        if not torch.isfinite(weights).all():
            raise ValueError("Allocator produced non-finite weights")
        return weights

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        phi: torch.Tensor,
        w_prev: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or r.ndim != 2 or phi.ndim != 2 or w_prev.ndim != 2:
            raise ValueError("Unexpected input dimensions for portfolio allocator")
        batch_size = x.shape[0]
        features = torch.cat([x, r.unsqueeze(-1), w_prev.unsqueeze(-1)], dim=-1)
        flat = features.reshape(batch_size, -1)
        in_vec = torch.cat([flat, phi], dim=-1)
        logits = self.net(in_vec)
        weights = self._normalize(logits)
        if not torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-4):
            raise AssertionError("Portfolio weights do not sum to 1")
        return weights
