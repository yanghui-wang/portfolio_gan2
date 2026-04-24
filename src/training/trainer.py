from __future__ import annotations

from collections import deque
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast

from src.models.discriminator import PortfolioDiscriminator
from src.models.portfolio_allocator import PortfolioAllocator
from src.models.strategy_encoder import StrategyEncoder
from src.training.checkpoint_manager import CheckpointManager
from src.training.losses import (
    compute_gradient_penalty,
    discriminator_loss,
    generator_loss,
    reparameterize,
)
from src.training.metrics_writer import MetricsWriter

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    best_val: float = float("inf")
    last_checkpoint_time: float = 0.0


class GANTrainer:
    """Resumable GAN training scaffold with checkpoint and metric logging."""

    def __init__(
        self,
        model_cfg: dict,
        train_cfg: dict,
        artifacts_dir: Path,
        outputs_dir: Path,
        logger,
        device: torch.device,
        num_features: int = 8,
    ) -> None:
        self.logger = logger
        self.device = device
        self.train_cfg = train_cfg
        latent_dim = int(model_cfg.get("latent_dim", 8))
        num_assets = int(model_cfg.get("num_assets", 500))
        self.lambda_gp = float(model_cfg.get("discriminator", {}).get("gradient_penalty_lambda", 10.0))
        self.num_assets = num_assets
        self.num_features = num_features

        self.encoder = StrategyEncoder(num_assets, num_features, latent_dim=latent_dim).to(device)
        self.allocator = PortfolioAllocator(
            num_assets,
            num_features,
            latent_dim=latent_dim,
            output_mode=model_cfg.get("portfolio_allocator", {}).get("output_mode", "softmax"),
        ).to(device)
        self.discriminator = PortfolioDiscriminator(num_assets, num_features, latent_dim=latent_dim).to(device)

        optim_cfg = train_cfg.get("optimizer", {})
        betas = tuple(optim_cfg.get("betas", [0.9, 0.999]))
        self.opt_g = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.allocator.parameters()),
            lr=float(optim_cfg.get("lr_generator", 1e-4)),
            betas=betas,
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=float(optim_cfg.get("lr_discriminator", 1e-4)),
            betas=betas,
        )

        self.ckpt_manager = CheckpointManager(artifacts_dir / "checkpoints")
        self.metrics_writer = MetricsWriter(outputs_dir / "metrics")
        self.use_amp = bool(train_cfg.get("runtime", {}).get("mixed_precision", True)) and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        tensorboard_enabled = train_cfg.get("logging", {}).get("tensorboard_enabled", True)
        self.tb_writer = (
            SummaryWriter(log_dir=str(outputs_dir / "logs" / "tensorboard"))
            if tensorboard_enabled and SummaryWriter is not None
            else None
        )

        self.state = TrainingState(last_checkpoint_time=time.time())

    def resume_if_needed(self) -> None:
        resume_path = self.train_cfg.get("checkpoint", {}).get("resume_from", "")
        if not resume_path:
            return
        payload = self.ckpt_manager.load(Path(resume_path))
        self.encoder.load_state_dict(payload["encoder"])
        self.allocator.load_state_dict(payload["allocator"])
        self.discriminator.load_state_dict(payload["discriminator"])
        self.opt_g.load_state_dict(payload["opt_g"])
        self.opt_d.load_state_dict(payload["opt_d"])
        if "scaler" in payload and payload["scaler"] is not None:
            self.scaler.load_state_dict(payload["scaler"])
        self.state = TrainingState(
            epoch=int(payload.get("epoch", 0)) + 1,
            global_step=payload.get("global_step", 0),
            best_val=payload.get("best_val", float("inf")),
            last_checkpoint_time=time.time(),
        )
        self.logger.info(f"Resumed checkpoint from {resume_path}")

    def _save_checkpoint(self, tag: str) -> Path:
        payload = {
            "encoder": self.encoder.state_dict(),
            "allocator": self.allocator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "best_val": self.state.best_val,
            "train_cfg": self.train_cfg,
        }
        return self.ckpt_manager.save(tag, payload)

    def _gpu_memory_stats(self) -> dict[str, float]:
        if self.device.type != "cuda":
            return {"gpu_mem_allocated_mb": 0.0, "gpu_mem_reserved_mb": 0.0}
        return {
            "gpu_mem_allocated_mb": round(torch.cuda.memory_allocated(self.device) / (1024**2), 2),
            "gpu_mem_reserved_mb": round(torch.cuda.memory_reserved(self.device) / (1024**2), 2),
        }

    @staticmethod
    def _module_grad_norm(module: nn.Module) -> float:
        sq_norm = 0.0
        for p in module.parameters():
            if p.grad is None:
                continue
            sq_norm += float(p.grad.detach().data.norm(2).item() ** 2)
        return sq_norm ** 0.5

    def _validate(self, val_loader) -> dict[str, float]:
        if val_loader is None or len(val_loader) == 0:
            return {"val_loss": float("nan"), "val_exposure": float("nan")}

        self.encoder.eval()
        self.allocator.eval()
        self.discriminator.eval()

        val_losses: list[float] = []
        val_exposures: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(self.device)
                r = batch["r"].to(self.device)
                w_prev = batch["w_prev"].to(self.device)
                w_t = batch["w_t"].to(self.device)

                with autocast("cuda", enabled=self.use_amp):
                    mu, logvar = self.encoder(x, r, w_prev, w_t)
                    phi = reparameterize(mu, logvar)
                    w_hat = self.allocator(x, r, phi, w_prev)
                    fake_scores = self.discriminator(x, r, w_prev, w_hat, phi)
                    g_loss, g_stats = generator_loss(
                        fake_scores,
                        w_hat,
                        w_t,
                        lambda_replication=float(self.train_cfg.get("loss", {}).get("lambda_replication", 1.0)),
                        lambda_exposure=float(self.train_cfg.get("loss", {}).get("lambda_exposure", 1.0)),
                    )
                val_losses.append(float(g_loss.detach().item()))
                val_exposures.append(float(g_stats["L_exposure"]))

        self.encoder.train()
        self.allocator.train()
        self.discriminator.train()

        return {
            "val_loss": float(sum(val_losses) / max(1, len(val_losses))),
            "val_exposure": float(sum(val_exposures) / max(1, len(val_exposures))),
        }

    def fit(self, train_loader, val_loader=None) -> None:
        """Run adversarial training loop scaffold."""

        self.resume_if_needed()
        epochs = int(self.train_cfg.get("epochs", 1))
        d_steps = int(self.train_cfg.get("adversarial", {}).get("discriminator_steps_per_generator_step", 3))
        log_interval = int(self.train_cfg.get("logging", {}).get("log_interval_steps", 10))
        heartbeat_seconds = int(self.train_cfg.get("logging", {}).get("heartbeat_seconds", 60))
        val_interval = int(self.train_cfg.get("logging", {}).get("validation_interval_steps", 500))
        lambda_rep = float(self.train_cfg.get("loss", {}).get("lambda_replication", 1.0))
        lambda_exp = float(self.train_cfg.get("loss", {}).get("lambda_exposure", 1.0))
        lambda_gp = self.lambda_gp

        gradient_clip = float(self.train_cfg.get("runtime", {}).get("gradient_clip_norm", 1.0))
        save_every_steps = int(self.train_cfg.get("checkpoint", {}).get("save_every_n_steps", 0))
        save_every_minutes = int(self.train_cfg.get("checkpoint", {}).get("save_every_n_minutes", 0))

        self.logger.info("Start training scaffold with real tensors")
        recent_batch_times: deque[float] = deque(maxlen=200)
        recent_data_times: deque[float] = deque(maxlen=200)
        recent_compute_times: deque[float] = deque(maxlen=200)
        recent_g_losses: deque[float] = deque(maxlen=200)
        recent_d_losses: deque[float] = deque(maxlen=200)
        last_heartbeat = time.time()

        for epoch in range(self.state.epoch, epochs):
            self.state.epoch = epoch
            epoch_start = time.time()
            epoch_g_losses: list[float] = []
            epoch_d_losses: list[float] = []
            epoch_exposure_losses: list[float] = []
            instability_flags: list[str] = []

            last_step_end = time.time()
            for batch_index, batch in enumerate(train_loader):
                data_seconds = time.time() - last_step_end
                step_start = time.time()
                self.state.global_step += 1
                x = batch["x"].to(self.device)
                r = batch["r"].to(self.device)
                w_prev = batch["w_prev"].to(self.device)
                w_t = batch["w_t"].to(self.device)

                if not torch.isfinite(x).all() or not torch.isfinite(r).all() or not torch.isfinite(w_t).all():
                    instability_flags.append("non_finite_input")
                    raise ValueError(
                        f"Non-finite input detected at epoch={epoch}, step={self.state.global_step}, batch={batch_index}"
                    )

                last_d_stats: dict[str, float] = {}
                for _ in range(d_steps):
                    self.opt_d.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        mu_d, logvar_d = self.encoder(x, r, w_prev, w_t)
                        phi_d = reparameterize(mu_d, logvar_d)
                        w_hat_d = self.allocator(x, r, phi_d, w_prev)

                    with autocast("cuda", enabled=self.use_amp):
                        real_scores = self.discriminator(x, r, w_prev, w_t, phi_d)
                        fake_scores = self.discriminator(x, r, w_prev, w_hat_d.detach(), phi_d)
                        gp = compute_gradient_penalty(
                            discriminator=self.discriminator,
                            x=x,
                            r=r,
                            w_prev=w_prev,
                            w_real=w_t,
                            w_fake=w_hat_d.detach(),
                            phi=phi_d.detach(),
                        )
                        d_loss, d_stats = discriminator_loss(
                            real_scores=real_scores,
                            fake_scores=fake_scores,
                            gradient_penalty=gp,
                            lambda_gp=lambda_gp,
                        )
                    self.scaler.scale(d_loss).backward()
                    self.scaler.unscale_(self.opt_d)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=gradient_clip)
                    self.scaler.step(self.opt_d)
                    self.scaler.update()
                    last_d_stats = d_stats

                self.opt_g.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=self.use_amp):
                    mu, logvar = self.encoder(x, r, w_prev, w_t)
                    phi = reparameterize(mu, logvar)
                    w_hat = self.allocator(x, r, phi, w_prev)
                    fake_scores_g = self.discriminator(x, r, w_prev, w_hat, phi)
                    g_loss, g_stats = generator_loss(
                        fake_scores_g,
                        w_hat,
                        w_t,
                        lambda_replication=lambda_rep,
                        lambda_exposure=lambda_exp,
                    )

                self.scaler.scale(g_loss).backward()
                self.scaler.unscale_(self.opt_g)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.allocator.parameters()),
                    max_norm=gradient_clip,
                )
                self.scaler.step(self.opt_g)
                self.scaler.update()

                if not torch.isfinite(w_hat).all():
                    instability_flags.append("non_finite_w_hat")
                    raise ValueError(
                        f"Non-finite generated weights at epoch={epoch}, step={self.state.global_step}, batch={batch_index}"
                    )

                duration = time.time() - step_start
                compute_seconds = duration
                last_step_end = time.time()

                recent_batch_times.append(duration)
                recent_data_times.append(data_seconds)
                recent_compute_times.append(compute_seconds)
                recent_g_losses.append(float(g_stats["L_generator"]))
                recent_d_losses.append(float(last_d_stats.get("L_discriminator", 0.0)))

                memory_stats = self._gpu_memory_stats()
                eta_seconds = (epochs - (epoch + 1)) * len(train_loader) + (len(train_loader) - batch_index - 1)
                avg_batch = sum(recent_batch_times) / max(1, len(recent_batch_times))
                eta_seconds = max(0.0, eta_seconds * avg_batch)

                grad_norm_encoder = self._module_grad_norm(self.encoder)
                grad_norm_allocator = self._module_grad_norm(self.allocator)
                grad_norm_discriminator = self._module_grad_norm(self.discriminator)

                payload = {
                    "epoch": epoch,
                    "global_step": self.state.global_step,
                    "batch_index": batch_index,
                    "g_loss": float(g_stats["L_generator"]),
                    "d_loss": float(last_d_stats.get("L_discriminator", 0.0)),
                    "exposure_loss": float(g_stats["L_exposure"]),
                    "replication_loss": float(g_stats["L_replication"]),
                    "gradient_penalty": float(last_d_stats.get("gradient_penalty", 0.0)),
                    "wasserstein": float(last_d_stats.get("wasserstein", 0.0)),
                    "score_real": float(last_d_stats.get("score_real", 0.0)),
                    "score_fake": float(last_d_stats.get("score_fake", 0.0)),
                    "grad_norm": float(grad_norm),
                    "grad_norm_encoder": grad_norm_encoder,
                    "grad_norm_allocator": grad_norm_allocator,
                    "grad_norm_discriminator": grad_norm_discriminator,
                    "lr_generator": float(self.opt_g.param_groups[0]["lr"]),
                    "lr_discriminator": float(self.opt_d.param_groups[0]["lr"]),
                    "batch_seconds": round(duration, 4),
                    "data_seconds": round(data_seconds, 4),
                    "compute_seconds": round(compute_seconds, 4),
                    "eta_seconds": round(eta_seconds, 2),
                    "weights_mean": float(w_hat.mean().item()),
                    "weights_std": float(w_hat.std().item()),
                    "weights_min": float(w_hat.min().item()),
                    "weights_max": float(w_hat.max().item()),
                    "nan_or_inf": int((~torch.isfinite(w_hat)).any().item()),
                    "device": str(self.device),
                }
                payload.update(memory_stats)
                self.metrics_writer.log_step(payload)

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("train/g_loss", payload["g_loss"], self.state.global_step)
                    self.tb_writer.add_scalar("train/d_loss", payload["d_loss"], self.state.global_step)
                    self.tb_writer.add_scalar("train/grad_norm", payload["grad_norm"], self.state.global_step)
                    self.tb_writer.add_scalar("train/gpu_mem_allocated_mb", payload["gpu_mem_allocated_mb"], self.state.global_step)
                    self.tb_writer.add_scalar("train/batch_seconds", payload["batch_seconds"], self.state.global_step)
                    self.tb_writer.add_scalar("train/exposure_loss", payload["exposure_loss"], self.state.global_step)

                epoch_g_losses.append(float(payload["g_loss"]))
                epoch_d_losses.append(float(payload["d_loss"]))
                epoch_exposure_losses.append(float(payload["exposure_loss"]))

                if self.state.global_step % log_interval == 0:
                    self.logger.info(
                        (
                            "epoch=%s step=%s batch=%s g_loss=%.6f d_loss=%.6f exposure=%.6f gp=%.6f "
                            "grad_norm=%.4f batch=%.3fs data=%.3fs eta=%.1fs device=%s"
                        ),
                        epoch,
                        self.state.global_step,
                        batch_index,
                        payload["g_loss"],
                        payload["d_loss"],
                        payload["exposure_loss"],
                        payload["gradient_penalty"],
                        payload["grad_norm"],
                        payload["batch_seconds"],
                        payload["data_seconds"],
                        payload["eta_seconds"],
                        payload["device"],
                    )

                now = time.time()
                if heartbeat_seconds > 0 and now - last_heartbeat >= heartbeat_seconds:
                    heartbeat_payload = {
                        "epoch": epoch,
                        "global_step": self.state.global_step,
                        "batches_processed": batch_index + 1,
                        "avg_batch_seconds": round(sum(recent_batch_times) / max(1, len(recent_batch_times)), 4),
                        "avg_data_seconds": round(sum(recent_data_times) / max(1, len(recent_data_times)), 4),
                        "avg_compute_seconds": round(sum(recent_compute_times) / max(1, len(recent_compute_times)), 4),
                        "recent_g_loss": round(sum(recent_g_losses) / max(1, len(recent_g_losses)), 6),
                        "recent_d_loss": round(sum(recent_d_losses) / max(1, len(recent_d_losses)), 6),
                        "eta_seconds": round(eta_seconds, 2),
                        "device": str(self.device),
                    }
                    heartbeat_payload.update(memory_stats)
                    self.metrics_writer.log_heartbeat(heartbeat_payload)
                    self.logger.info(
                        (
                            "HEARTBEAT epoch=%s step=%s batches=%s avg_batch=%.3fs avg_data=%.3fs avg_compute=%.3fs "
                            "eta=%.1fs gpu_alloc=%.1fMB gpu_reserved=%.1fMB recent_g=%.6f recent_d=%.6f"
                        ),
                        heartbeat_payload["epoch"],
                        heartbeat_payload["global_step"],
                        heartbeat_payload["batches_processed"],
                        heartbeat_payload["avg_batch_seconds"],
                        heartbeat_payload["avg_data_seconds"],
                        heartbeat_payload["avg_compute_seconds"],
                        heartbeat_payload["eta_seconds"],
                        heartbeat_payload["gpu_mem_allocated_mb"],
                        heartbeat_payload["gpu_mem_reserved_mb"],
                        heartbeat_payload["recent_g_loss"],
                        heartbeat_payload["recent_d_loss"],
                    )
                    last_heartbeat = now

                if val_loader is not None and val_interval > 0 and self.state.global_step % val_interval == 0:
                    val_metrics = self._validate(val_loader)
                    self.metrics_writer.log_epoch(
                        {
                            "epoch": epoch,
                            "global_step": self.state.global_step,
                            "val_loss": val_metrics["val_loss"],
                            "val_exposure": val_metrics["val_exposure"],
                            "event": "step_validation",
                        }
                    )
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar("val/loss", val_metrics["val_loss"], self.state.global_step)
                        self.tb_writer.add_scalar("val/exposure", val_metrics["val_exposure"], self.state.global_step)
                    if val_metrics["val_loss"] < self.state.best_val:
                        self.state.best_val = val_metrics["val_loss"]
                        best_path = self._save_checkpoint("best")
                        self.logger.info(
                            "New best checkpoint at step=%s val_loss=%.6f path=%s",
                            self.state.global_step,
                            val_metrics["val_loss"],
                            best_path,
                        )

                should_save_by_step = save_every_steps > 0 and self.state.global_step % save_every_steps == 0
                should_save_by_time = (
                    save_every_minutes > 0 and (time.time() - self.state.last_checkpoint_time) >= save_every_minutes * 60
                )
                if should_save_by_step or should_save_by_time:
                    latest_path = self._save_checkpoint("latest")
                    self.state.last_checkpoint_time = time.time()
                    self.logger.info(
                        "Checkpoint saved during epoch at step=%s path=%s",
                        self.state.global_step,
                        latest_path,
                    )

            epoch_seconds = time.time() - epoch_start
            val_metrics = self._validate(val_loader) if val_loader is not None else {"val_loss": float("nan"), "val_exposure": float("nan")}
            best_updated = False
            if val_metrics["val_loss"] < self.state.best_val:
                self.state.best_val = val_metrics["val_loss"]
                best_updated = True

            ckpt_path = self._save_checkpoint(f"epoch_{epoch:03d}")
            if best_updated:
                self._save_checkpoint("best")
            epoch_payload = {
                "epoch": epoch,
                "global_step": self.state.global_step,
                "epoch_seconds": round(epoch_seconds, 2),
                "train_g_loss_mean": float(sum(epoch_g_losses) / max(1, len(epoch_g_losses))),
                "train_d_loss_mean": float(sum(epoch_d_losses) / max(1, len(epoch_d_losses))),
                "train_exposure_mean": float(sum(epoch_exposure_losses) / max(1, len(epoch_exposure_losses))),
                "val_loss": float(val_metrics["val_loss"]),
                "val_exposure": float(val_metrics["val_exposure"]),
                "best_val": float(self.state.best_val),
                "instability_flags": "|".join(sorted(set(instability_flags))) if instability_flags else "none",
                "checkpoint": str(ckpt_path),
            }
            self.metrics_writer.log_epoch(epoch_payload)
            self.logger.info(
                (
                    "epoch=%s done duration=%.2fs train_g=%.6f train_d=%.6f val_loss=%.6f "
                    "best_val=%.6f checkpoint=%s"
                ),
                epoch,
                epoch_seconds,
                epoch_payload["train_g_loss_mean"],
                epoch_payload["train_d_loss_mean"],
                epoch_payload["val_loss"],
                epoch_payload["best_val"],
                ckpt_path,
            )

        self._save_checkpoint("latest")
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
