from __future__ import annotations

import torch
from torch import autograd


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def generator_loss(
    fake_scores: torch.Tensor,
    w_hat: torch.Tensor,
    w_real: torch.Tensor,
    lambda_replication: float,
    lambda_exposure: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    replication = -fake_scores.mean()
    exposure = (w_hat - w_real).abs().mean()
    total = lambda_replication * replication + lambda_exposure * exposure
    return total, {
        "L_replication": replication.detach().item(),
        "L_exposure": exposure.detach().item(),
        "L_generator": total.detach().item(),
    }


def compute_gradient_penalty(
    discriminator,
    x: torch.Tensor,
    r: torch.Tensor,
    w_prev: torch.Tensor,
    w_real: torch.Tensor,
    w_fake: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    batch_size = w_real.shape[0]
    alpha = torch.rand(batch_size, 1, device=w_real.device)
    interpolated = (alpha * w_real + (1.0 - alpha) * w_fake).requires_grad_(True)
    d_interpolated = discriminator(x, r, w_prev, interpolated, phi)
    grad_outputs = torch.ones_like(d_interpolated)
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


def discriminator_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    gradient_penalty: torch.Tensor,
    lambda_gp: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    wasserstein = real_scores.mean() - fake_scores.mean()
    loss = fake_scores.mean() - real_scores.mean() + lambda_gp * gradient_penalty
    return loss, {
        "L_discriminator": loss.detach().item(),
        "score_real": real_scores.mean().detach().item(),
        "score_fake": fake_scores.mean().detach().item(),
        "wasserstein": wasserstein.detach().item(),
        "gradient_penalty": gradient_penalty.detach().item(),
    }
