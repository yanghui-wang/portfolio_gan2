from __future__ import annotations

import torch
from torch import nn


class StrategyEncoder(nn.Module):
    """Three-lane strategy encoder scaffold producing latent posterior params."""

    def __init__(
        self,
        num_assets: int,
        num_features: int,
        latent_dim: int = 8,
        hidden_dim: int = 128,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.latent_dim = latent_dim

        self.char_proj = nn.Linear(num_features + 1, hidden_dim)
        self.ret_proj = nn.Linear(2, hidden_dim)
        self.turnover_proj = nn.Linear(1, hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        w_prev: torch.Tensor,
        w_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or r.ndim != 2 or w_prev.ndim != 2 or w_t.ndim != 2:
            raise ValueError("Unexpected input dimensions for strategy encoder")
        if not torch.isfinite(x).all() or not torch.isfinite(r).all() or not torch.isfinite(w_t).all():
            raise ValueError("Non-finite detected in strategy encoder inputs")

        char_lane = torch.cat([x, w_t.unsqueeze(-1)], dim=-1)
        ret_lane = torch.stack([r, w_t], dim=-1)
        turnover = (w_t - w_prev).abs().unsqueeze(-1)

        h_char = self.char_proj(char_lane)
        h_ret = self.ret_proj(ret_lane)
        h_turn = self.turnover_proj(turnover)

        h = (h_char + h_ret + h_turn) / 3.0
        attended, _ = self.attn(h, h, h)
        pooled = self.norm(attended).mean(dim=1)

        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled)
        if not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
            raise ValueError("Non-finite latent posterior in strategy encoder")
        return mu, logvar
