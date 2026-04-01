from __future__ import annotations

import torch
from torch import nn


class MarketGenerator(nn.Module):
    def __init__(
        self,
        num_assets: int,
        num_features: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        factor_dim: int = 4,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.num_features = num_features
        self.factor_dim = factor_dim

        self.encoder = nn.Sequential(
            nn.Linear(num_assets * (num_features + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.factor_head = nn.Linear(latent_dim, factor_dim)
        self.decoder_x = nn.Linear(latent_dim + factor_dim, num_assets * num_features)
        self.decoder_r = nn.Linear(latent_dim + factor_dim, num_assets)

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        flat = torch.cat([x.reshape(batch_size, -1), r], dim=-1)
        z = self.encoder(flat)
        factors = self.factor_head(z)
        merged = torch.cat([z, factors], dim=-1)

        x_hat = self.decoder_x(merged).reshape(batch_size, self.num_assets, self.num_features)
        r_hat = self.decoder_r(merged)

        if not torch.isfinite(x_hat).all() or not torch.isfinite(r_hat).all():
            raise ValueError("Market generator produced invalid tensors")

        return x_hat, r_hat, factors

