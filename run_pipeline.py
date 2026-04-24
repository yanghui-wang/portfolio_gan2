from __future__ import annotations

import argparse
import uuid
from pathlib import Path
import math

from src.evaluation.evaluator import run_evaluation
from src.features.tensor_builder import (
    build_dataloader,
    build_model_input_index,
    build_real_dataset_bundle,
)
from src.ingest.data_inventory import build_inventory, write_inventory_reports
from src.ingest.data_loader import load_raw_frames
from src.preprocess.sample_construction import construct_sample_panels
from src.preprocess.variable_crosswalk import build_variable_crosswalk, write_crosswalk_outputs
from src.training.trainer import GANTrainer
from src.training.evaluation_exporter import export_evaluation_artifacts
from src.utils.config import load_config_bundle, resolve_path
from src.utils.logging_utils import build_logger
from src.utils.runtime import collect_device_diagnostics, detect_device, diagnostics_as_dict, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portfolio GAN replication pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["inventory", "sample", "tensors", "train", "evaluate", "all"],
    )
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--training-mode", type=str, default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def missing_inputs_from_inventory(inventory_df) -> list[str]:
    missing_rows = inventory_df.loc[~inventory_df["exists"]]
    return [f"{row.dataset_key}:{row.configured_path}" for row in missing_rows.itertuples(index=False)]


def log_batch_debug(loader, logger, split: str, max_batches: int = 2) -> None:
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        x = batch["x"]
        r = batch["r"]
        w_prev = batch["w_prev"]
        w_t = batch["w_t"]
        logger.info(
            "%s batch[%s] x=%s r=%s w_prev=%s w_t=%s w_t[min,max,mean]=[%.6f, %.6f, %.6f]",
            split,
            batch_index,
            tuple(x.shape),
            tuple(r.shape),
            tuple(w_prev.shape),
            tuple(w_t.shape),
            float(w_t.min().item()),
            float(w_t.max().item()),
            float(w_t.mean().item()),
        )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_dir = project_root / "config"
    configs = load_config_bundle(config_dir)

    if args.training_mode:
        configs.train["training_mode"] = args.training_mode
    if args.epochs > 0:
        configs.train["epochs"] = int(args.epochs)
    if args.batch_size > 0:
        configs.train["batch_size"] = int(args.batch_size)

    checkpoint_cfg = configs.train.setdefault("checkpoint", {})
    resume_from = checkpoint_cfg.get("resume_from", "")
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = project_root / resume_path
        checkpoint_cfg["resume_from"] = str(resume_path)

    run_id = uuid.uuid4().hex[:10]
    logs_dir = resolve_path(project_root, configs.paths.get("logs_dir", "outputs/logs"))
    logger = build_logger(run_id=run_id, logs_dir=logs_dir)

    device = detect_device(configs.train.get("device", "auto"))
    diag = diagnostics_as_dict(collect_device_diagnostics(device))
    logger.info(f"Device diagnostics: {diag}")
    seed_everything(int(configs.train.get("seed", 42)))

    stages = [args.stage] if args.stage != "all" else ["inventory", "sample", "tensors", "train", "evaluate"]

    derived_dir = resolve_path(project_root, configs.paths["derived_dir"])
    docs_dir = project_root / "docs"
    diagnostics_dir = resolve_path(project_root, configs.paths["diagnostics_dir"])
    artifacts_dir = resolve_path(project_root, configs.paths["artifacts_dir"])
    outputs_dir = resolve_path(project_root, configs.paths["outputs_dir"])

    inventory_df = None
    raw_frames = None
    sample_outputs = None
    skip_raw_keys = {"holdings_file"}

    if "inventory" in stages:
        inventory_df = build_inventory(configs.data, project_root)
        write_inventory_reports(inventory_df, derived_dir, docs_dir)
        missing = missing_inputs_from_inventory(inventory_df)
        if missing:
            logger.warning("Missing inputs: %s", missing)
        else:
            logger.info("All placeholder raw inputs found")

    if "sample" in stages:
        raw_frames = load_raw_frames(project_root, configs.data, skip_keys=skip_raw_keys)
        crosswalk_df = build_variable_crosswalk(raw_frames)
        write_crosswalk_outputs(crosswalk_df, derived_dir, docs_dir)
        sample_outputs = construct_sample_panels(raw_frames, configs.data, derived_dir, diagnostics_dir)
        logger.info(
            "Sample stage done: fund_sample=%s holdings_panel=%s universe_panel=%s mapped_vars=%s",
            len(sample_outputs.fund_sample),
            len(sample_outputs.holdings_panel),
            len(sample_outputs.stock_universe_panel),
            int((crosswalk_df["status"] == "mapped").sum()),
        )

    if "tensors" in stages:
        if sample_outputs is None:
            raw_frames = raw_frames or load_raw_frames(project_root, configs.data, skip_keys=skip_raw_keys)
            sample_outputs = construct_sample_panels(raw_frames, configs.data, derived_dir, diagnostics_dir)
        index_df = build_model_input_index(sample_outputs.holdings_panel, derived_dir)
        logger.info("Tensor index built: rows=%s", len(index_df))

    if "train" in stages:
        raw_frames = raw_frames or load_raw_frames(project_root, configs.data, skip_keys=skip_raw_keys)
        dataset_bundle = build_real_dataset_bundle(
            project_root=project_root,
            data_cfg=configs.data,
            model_cfg=configs.model,
            train_cfg=configs.train,
            raw_frames=raw_frames,
            derived_dir=derived_dir,
            diagnostics_dir=diagnostics_dir,
            logger=logger,
        )
        if len(dataset_bundle.train_dataset) == 0:
            raise RuntimeError("Real-data train dataset is empty after filtering; check diagnostics/tensor_build_summary.csv")

        num_workers = int(configs.train.get("num_workers", 0))
        pin_memory = bool(configs.train.get("pin_memory", True))
        if device.type == "cpu":
            num_workers = 0
            pin_memory = False

        train_loader = build_dataloader(
            dataset_bundle.train_dataset,
            batch_size=int(configs.train.get("batch_size", 32)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
        val_loader = build_dataloader(
            dataset_bundle.val_dataset,
            batch_size=max(1, int(configs.train.get("batch_size", 32))),
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

        logger.info(
            "Dataset summary train=%s val=%s test=%s batches_per_epoch=%s",
            len(dataset_bundle.train_dataset),
            len(dataset_bundle.val_dataset),
            len(dataset_bundle.test_dataset),
            math.ceil(len(dataset_bundle.train_dataset) / max(1, int(configs.train.get("batch_size", 32)))),
        )
        log_batch_debug(train_loader, logger, split="train")
        if len(dataset_bundle.val_dataset) > 0:
            log_batch_debug(val_loader, logger, split="val")

        trainer = GANTrainer(
            model_cfg=configs.model,
            train_cfg=configs.train,
            artifacts_dir=artifacts_dir,
            outputs_dir=outputs_dir,
            logger=logger,
            device=device,
            num_features=int(dataset_bundle.train_dataset.x.shape[-1]),
        )
        trainer.fit(train_loader=train_loader, val_loader=val_loader if len(dataset_bundle.val_dataset) > 0 else None)
        export_evaluation_artifacts(
            trainer,
            dataset_bundle,
            project_root=project_root,
            derived_dir=derived_dir,
            eval_cfg=configs.eval,
            run_id=run_id,
            logger=logger,
        )
        logger.info("Training stage complete")

    if "evaluate" in stages:
        run_evaluation(project_root, configs.eval, outputs_dir, artifacts_dir=artifacts_dir, logger=logger)
        logger.info("Evaluation stage complete")


if __name__ == "__main__":
    main()
