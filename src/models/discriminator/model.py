from __future__ import annotations

import torch
from torch import nn


class PortfolioDiscriminator(nn.Module):
    """WGAN-style discriminator scaffold over real/fake portfolio tuples."""

    def __init__(
        self,
        num_assets: int,
        num_features: int,
        latent_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
    ) -> None:
        super().__init__()
        input_dim = num_assets * (num_features + 3) + latent_dim
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.LeakyReLU(0.2)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        w_prev: torch.Tensor,
        w_t: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or r.ndim != 2 or w_prev.ndim != 2 or w_t.ndim != 2 or phi.ndim != 2:
            raise ValueError("Unexpected input dimensions for discriminator")
        batch_size = x.shape[0]
        merged = torch.cat([x, r.unsqueeze(-1), w_prev.unsqueeze(-1), w_t.unsqueeze(-1)], dim=-1)
        flat = merged.reshape(batch_size, -1)
        out = self.net(torch.cat([flat, phi], dim=-1))
        if not torch.isfinite(out).all():
            raise ValueError("Discriminator produced non-finite scores")
        return out
