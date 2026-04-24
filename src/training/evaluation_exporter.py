from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.features.tensor_builder import DatasetBundle, RealPortfolioDataset
from src.utils.io import ensure_dir, save_parquet


def export_evaluation_artifacts(
    trainer: Any,
    dataset_bundle: DatasetBundle,
    *,
    project_root: Path,
    derived_dir: Path,
    eval_cfg: Mapping[str, Any],
    run_id: str,
    logger: Any,
) -> dict[str, Path | None]:
    """Export generated weights and strategy embeddings for the evaluation stage.

    The trainer is responsible for learning the encoder/allocator. This function
    runs deterministic inference on configured splits and writes the normalized
    long-table artifacts consumed by `src.evaluation.evaluator`.
    """

    cfg = _evaluation_cfg(eval_cfg)
    export_cfg = dict(cfg.get("export", {}))
    if not bool(export_cfg.get("enabled", True)):
        logger.info("Evaluation artifact export disabled by config")
        return {"portfolio_predictions": None, "strategy_embeddings": None}

    splits = list(export_cfg.get("splits", ["val", "test"]))
    if not splits:
        splits = ["val", "test"]
    batch_size = int(export_cfg.get("batch_size", 128))
    model_name = str(cfg.get("model_name", "portfolio_gan"))
    prediction_source = str(cfg.get("prediction_source", "model"))
    configured_run_id = str(cfg.get("run_id", "unknown"))
    artifact_run_id = run_id if configured_run_id == "unknown" else configured_run_id

    universe = _load_universe(derived_dir)
    portfolio_parts: list[pd.DataFrame] = []
    embedding_parts: list[pd.DataFrame] = []

    for split in splits:
        dataset = _dataset_for_split(dataset_bundle, split)
        if dataset is None or len(dataset) == 0:
            logger.info("Skipping evaluation export for empty split=%s", split)
            continue
        if not isinstance(dataset, RealPortfolioDataset):
            logger.warning("Skipping evaluation export for unsupported dataset type split=%s", split)
            continue

        pred, embed = _predict_dataset(
            trainer,
            dataset,
            split=split,
            batch_size=batch_size,
            universe=universe,
            model_name=model_name,
            run_id=artifact_run_id,
            prediction_source=prediction_source,
        )
        if not pred.empty:
            portfolio_parts.append(pred)
        if not embed.empty:
            embedding_parts.append(embed)
        logger.info(
            "Prepared evaluation export split=%s samples=%s portfolio_rows=%s embedding_rows=%s",
            split,
            len(dataset),
            len(pred),
            len(embed),
        )

    paths = dict(cfg.get("inputs", {}))
    portfolio_path = _resolve_project_path(project_root, str(paths.get("portfolio_predictions", "artifacts/evaluation/portfolio_predictions.parquet")))
    embeddings_path = _resolve_project_path(project_root, str(paths.get("representation_embeddings", "artifacts/embeddings/strategy_embeddings.parquet")))

    if portfolio_parts:
        portfolio_df = pd.concat(portfolio_parts, ignore_index=True)
        portfolio_df = _attach_lipper_labels(portfolio_df, project_root=project_root, cfg=cfg)
        ensure_dir(portfolio_path.parent)
        save_parquet(portfolio_df, portfolio_path)
        logger.info("Wrote evaluation portfolio predictions: %s rows=%s", portfolio_path, len(portfolio_df))
    else:
        portfolio_path = None
        logger.warning("No portfolio predictions exported for evaluation")

    if embedding_parts:
        embedding_df = pd.concat(embedding_parts, ignore_index=True)
        embedding_df = _attach_lipper_labels(embedding_df, project_root=project_root, cfg=cfg)
        ensure_dir(embeddings_path.parent)
        save_parquet(embedding_df, embeddings_path)
        logger.info("Wrote strategy embeddings: %s rows=%s", embeddings_path, len(embedding_df))
    else:
        embeddings_path = None
        logger.warning("No strategy embeddings exported for evaluation")

    return {"portfolio_predictions": portfolio_path, "strategy_embeddings": embeddings_path}


def _predict_dataset(
    trainer: Any,
    dataset: RealPortfolioDataset,
    *,
    split: str,
    batch_size: int,
    universe: pd.DataFrame,
    model_name: str,
    run_id: str,
    prediction_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    trainer.encoder.eval()
    trainer.allocator.eval()

    portfolio_parts: list[pd.DataFrame] = []
    embedding_rows: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(trainer.device)
            r = batch["r"].to(trainer.device)
            w_prev = batch["w_prev"].to(trainer.device)
            w_t = batch["w_t"].to(trainer.device)

            mu, _ = trainer.encoder(x, r, w_prev, w_t)
            # Use posterior mean for deterministic evaluation artifact export.
            w_pred = trainer.allocator(x, r, mu, w_prev)

            batch_size_actual = x.shape[0]
            meta = dataset.meta.iloc[offset : offset + batch_size_actual].reset_index(drop=True)
            offset += batch_size_actual

            portfolio_parts.append(
                _long_prediction_frame(
                    meta=meta,
                    w_true=w_t.detach().cpu().numpy(),
                    w_pred=w_pred.detach().cpu().numpy(),
                    w_prev=w_prev.detach().cpu().numpy(),
                    universe=universe,
                    split=split,
                    model_name=model_name,
                    run_id=run_id,
                    prediction_source=prediction_source,
                )
            )
            embedding_rows.extend(
                _embedding_rows(
                    meta=meta,
                    phi=mu.detach().cpu().numpy(),
                    split=split,
                    model_name=model_name,
                    run_id=run_id,
                    prediction_source=prediction_source,
                )
            )

    trainer.encoder.train()
    trainer.allocator.train()

    portfolio = pd.concat(portfolio_parts, ignore_index=True) if portfolio_parts else pd.DataFrame()
    embeddings = pd.DataFrame(embedding_rows)
    return portfolio, embeddings


def _long_prediction_frame(
    *,
    meta: pd.DataFrame,
    w_true: np.ndarray,
    w_pred: np.ndarray,
    w_prev: np.ndarray,
    universe: pd.DataFrame,
    split: str,
    model_name: str,
    run_id: str,
    prediction_source: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    n_assets = w_true.shape[1]
    for sample_idx, sample in meta.reset_index(drop=True).iterrows():
        date = pd.Timestamp(sample["date"])
        asset_ids = _asset_ids_for_date(universe, date=date, n_assets=n_assets)
        rows.append(
            pd.DataFrame(
                {
                    "model_name": model_name,
                    "run_id": run_id,
                    "prediction_source": prediction_source,
                    "split": str(sample.get("split", split) or split),
                    "fund_id": sample["fund_id"],
                    "date": date,
                    "asset_id": asset_ids,
                    "w_true": w_true[sample_idx],
                    "w_pred": w_pred[sample_idx],
                    "w_prev": w_prev[sample_idx],
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _embedding_rows(
    *,
    meta: pd.DataFrame,
    phi: np.ndarray,
    split: str,
    model_name: str,
    run_id: str,
    prediction_source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_idx, sample in meta.reset_index(drop=True).iterrows():
        row: dict[str, Any] = {
            "model_name": model_name,
            "run_id": run_id,
            "prediction_source": prediction_source,
            "split": str(sample.get("split", split) or split),
            "fund_id": sample["fund_id"],
            "date": pd.Timestamp(sample["date"]),
        }
        row.update({f"phi_{idx + 1}": float(value) for idx, value in enumerate(phi[sample_idx])})
        rows.append(row)
    return rows


def _asset_ids_for_date(universe: pd.DataFrame, *, date: pd.Timestamp, n_assets: int) -> np.ndarray:
    if universe.empty:
        return np.arange(n_assets).astype(str)
    sub = universe.loc[universe["date"] == date].sort_values("asset_idx")
    if sub.shape[0] < n_assets:
        fallback = pd.DataFrame({"asset_idx": np.arange(n_assets), "stock_id": np.arange(n_assets).astype(str)})
        sub = fallback.merge(sub, on="asset_idx", how="left", suffixes=("_fallback", ""))
        stock = sub["stock_id"].fillna(sub["stock_id_fallback"])
        return stock.astype(str).to_numpy()
    return sub.head(n_assets)["stock_id"].astype(str).to_numpy()


def _load_universe(derived_dir: Path) -> pd.DataFrame:
    path = derived_dir / "stock_universe_panel.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["date", "asset_idx", "stock_id"])
    universe = pd.read_parquet(path)
    if "date" in universe.columns:
        universe["date"] = pd.to_datetime(universe["date"], errors="coerce")
    return universe


def _dataset_for_split(dataset_bundle: DatasetBundle, split: str) -> RealPortfolioDataset | None:
    if split == "train":
        return dataset_bundle.train_dataset  # type: ignore[return-value]
    if split == "val":
        return dataset_bundle.val_dataset  # type: ignore[return-value]
    if split == "test":
        return dataset_bundle.test_dataset  # type: ignore[return-value]
    return None


def _evaluation_cfg(eval_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = eval_cfg.get("evaluation") if isinstance(eval_cfg.get("evaluation"), Mapping) else None
    return nested or eval_cfg


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _attach_lipper_labels(df: pd.DataFrame, *, project_root: Path, cfg: Mapping[str, Any]) -> pd.DataFrame:
    labels_cfg = dict(cfg.get("labels", {}))
    if not bool(labels_cfg.get("enabled", True)) or df.empty:
        return df

    label_col = str(labels_cfg.get("output_column", cfg.get("columns", {}).get("label", "style_label")))
    if label_col in df.columns and df[label_col].notna().any():
        return df

    labels = _load_lipper_labels(project_root, labels_cfg=labels_cfg)
    if labels.empty:
        return df

    working = df.copy()
    working["fund_id"] = pd.to_numeric(working["fund_id"], errors="coerce")
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    sample_keys = working[["fund_id", "date"]].drop_duplicates().dropna().sort_values(["fund_id", "date"])
    if sample_keys.empty:
        return df

    method = str(labels_cfg.get("merge_method", "asof_backward"))
    if method == "exact":
        keyed = sample_keys.merge(labels, on=["fund_id", "date"], how="left")
    else:
        keyed_parts: list[pd.DataFrame] = []
        for fund_id, samples in sample_keys.groupby("fund_id", sort=False):
            fund_labels = labels.loc[labels["fund_id"] == fund_id].sort_values("date")
            if fund_labels.empty:
                part = samples.copy()
                part[label_col] = np.nan
            else:
                part = pd.merge_asof(
                    samples.sort_values("date"),
                    fund_labels[["date", label_col]].sort_values("date"),
                    on="date",
                    direction="backward",
                )
                part["fund_id"] = fund_id
            keyed_parts.append(part)
        keyed = pd.concat(keyed_parts, ignore_index=True) if keyed_parts else pd.DataFrame()

    if keyed.empty or label_col not in keyed.columns:
        return df
    keyed = keyed[["fund_id", "date", label_col]].drop_duplicates(subset=["fund_id", "date"])
    working = working.merge(keyed, on=["fund_id", "date"], how="left")
    return working


def _load_lipper_labels(project_root: Path, *, labels_cfg: Mapping[str, Any]) -> pd.DataFrame:
    path = _resolve_project_path(project_root, str(labels_cfg.get("path", "raw/lipper.csv")))
    if not path.exists():
        return pd.DataFrame(columns=["fund_id", "date", str(labels_cfg.get("output_column", "style_label"))])

    fund_col = str(labels_cfg.get("fund_id_column", "crsp_fundno"))
    date_col = str(labels_cfg.get("date_column", "caldt"))
    source_label_col = str(labels_cfg.get("source_label_column", "lipper_class"))
    output_label_col = str(labels_cfg.get("output_column", "style_label"))
    encoding = str(labels_cfg.get("encoding", "latin1"))
    usecols = [fund_col, date_col, source_label_col]
    labels = pd.read_csv(path, usecols=usecols, dtype=str, encoding=encoding)
    labels = labels.rename(columns={fund_col: "fund_id", date_col: "date", source_label_col: output_label_col})
    labels["fund_id"] = pd.to_numeric(labels["fund_id"], errors="coerce")
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    labels[output_label_col] = labels[output_label_col].astype("string").str.strip()
    labels = labels.dropna(subset=["fund_id", "date", output_label_col])
    labels = labels.loc[labels[output_label_col] != ""].copy()
    labels = labels.sort_values(["fund_id", "date"]).drop_duplicates(subset=["fund_id", "date"], keep="last")
    return labels[["fund_id", "date", output_label_col]]
