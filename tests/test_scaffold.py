from pathlib import Path

import torch

from src.models.discriminator import PortfolioDiscriminator
from src.models.portfolio_allocator import PortfolioAllocator
from src.models.strategy_encoder import StrategyEncoder
from src.utils.config import load_config_bundle


def test_config_bundle_loads() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config_bundle(project_root / "config")
    assert "target_universe_size" in config.data["filters"]
    assert config.model["latent_dim"] == 8


def test_model_forward_shapes() -> None:
    batch_size = 2
    num_assets = 10
    num_features = 8
    latent_dim = 8

    x = torch.randn(batch_size, num_assets, num_features)
    r = torch.randn(batch_size, num_assets)
    w_prev = torch.softmax(torch.randn(batch_size, num_assets), dim=-1)
    w_t = torch.softmax(torch.randn(batch_size, num_assets), dim=-1)

    encoder = StrategyEncoder(num_assets=num_assets, num_features=num_features, latent_dim=latent_dim)
    allocator = PortfolioAllocator(
        num_assets=num_assets,
        num_features=num_features,
        latent_dim=latent_dim,
    )
    discriminator = PortfolioDiscriminator(
        num_assets=num_assets,
        num_features=num_features,
        latent_dim=latent_dim,
    )

    mu, logvar = encoder(x, r, w_prev, w_t)
    phi = mu
    w_hat = allocator(x, r, phi, w_prev)
    scores = discriminator(x, r, w_prev, w_hat, phi)

    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    assert w_hat.shape == (batch_size, num_assets)
    assert scores.shape == (batch_size, 1)

