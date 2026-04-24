from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluation.aggregation import markdown_table, metric_summary_columns, summarize_metric_columns
from src.evaluation.frontier import (
    FRONTIER_STATUS,
    compute_frontier_metrics,
    frontier_by_sample_columns,
)
from src.evaluation.factor_exposures import build_or_load_carhart_factor_exposures
from src.evaluation.io import (
    merge_factor_exposures,
    normalize_factor_frame,
    normalize_portfolio_frame,
    normalize_return_frame,
    read_optional_table,
    write_json,
    write_markdown,
    write_table_outputs,
)
from src.evaluation.metrics_behavior import (
    STABILITY_STATUS,
    compute_strategy_stability,
    stability_by_fund_columns,
    stability_summary_columns,
)
from src.evaluation.metrics_counterfactual import (
    COUNTERFACTUAL_STATUS,
    compute_counterfactual_metrics,
    counterfactual_by_case_columns,
)
from src.evaluation.metrics_portfolio import (
    PORTFOLIO_METRIC_STATUS,
    concentration_error,
    count_error,
    portfolio_metric_columns,
    turnover_error,
)
from src.evaluation.metrics_representation import (
    REPRESENTATION_STATUS,
    representation_per_class_columns,
    run_linear_probe,
)


DEFAULT_EVALUATION_CONFIG: dict[str, Any] = {
    "model_name": "portfolio_gan",
    "run_id": "unknown",
    "prediction_source": "model",
    "split": "test",
    "holding_threshold": 0.0001,
    "normalize_weights": True,
    "require_full_metrics": False,
    "factor_columns": ["market_beta", "SMB", "HML", "UMD"],
    "columns": {
        "fund_id": "fund_id",
        "date": "date",
        "asset_id": "asset_id",
        "w_true": "w_true",
        "w_pred": "w_pred",
        "w_prev": "w_prev",
        "label": "style_label",
        "model_name": "model_name",
        "run_id": "run_id",
        "split": "split",
        "prediction_source": "prediction_source",
        "return": "ret",
        "case_id": "case_id",
        "w_original": "w_original",
        "w_transferred": "w_transferred",
        "w_prev_transferred": "w_prev_transferred",
    },
    "inputs": {
        "portfolio_predictions": "artifacts/evaluation/portfolio_predictions.parquet",
        "representation_embeddings": "artifacts/embeddings/strategy_embeddings.parquet",
        "factor_exposures": "raw/stock_characteristics.parquet",
        "carhart_factors": "raw/carhart_factors.parquet",
        "asset_returns": "raw/stock_returns.parquet",
        "counterfactual_transfers": "artifacts/evaluation/counterfactual_transfers.parquet",
    },
    "factor_exposure_estimation": {
        "enabled": True,
        "output_path": "artifacts/evaluation/carhart_betas.parquet",
        "lookback_periods": 36,
        "min_periods": 24,
        "ridge_penalty": 1e-8,
        "use_cache": True,
        "asset_id_column": "permno",
        "date_column": "date",
        "return_column": "ret",
        "risk_free_column": "rf",
        "factor_column_map": {
            "mktrf": "market_beta",
            "smb": "SMB",
            "hml": "HML",
            "umd": "UMD",
        },
    },
    "representation": {
        "classifier": "linear_svm",
        "average_embeddings_over_time": False,
        "train_split": "train",
        "eval_split": "test",
        "random_seed": 42,
    },
    "frontier": {
        "enabled": True,
        "method": "mean_variance_long_only_random_frontier",
        "weight_column": "w_pred",
        "num_random_portfolios": 1000,
        "num_random_reference_portfolios": 200,
        "covariance_shrinkage": 0.0,
        "lookback_periods": 36,
        "min_periods": 3,
        "long_only": True,
        "random_seed": 42,
    },
    "counterfactual": {
        "enabled": True,
    },
}


def evaluate_reconstruction(
    w_hat: torch.Tensor,
    w_true: torch.Tensor,
    w_prev: torch.Tensor | None = None,
    *,
    threshold: float = 1e-4,
) -> dict[str, float]:
    """Tensor convenience wrapper around the paper reconstruction metrics."""

    pred = w_hat.detach().cpu().numpy()
    true = w_true.detach().cpu().numpy()
    prev = w_prev.detach().cpu().numpy() if w_prev is not None else None
    rows: list[dict[str, float]] = []
    for idx in range(pred.shape[0]):
        row = {
            "L_count": count_error(true[idx], pred[idx], threshold),
            "L_concentration": concentration_error(true[idx], pred[idx]),
            "L_turnover": np.nan,
        }
        if prev is not None:
            row["L_turnover"] = turnover_error(true[idx], pred[idx], prev[idx])
        rows.append(row)
    frame = pd.DataFrame(rows)
    return {col: float(frame[col].mean()) for col in ["L_count", "L_concentration", "L_turnover"]}


def run_evaluation(
    project_root: Path,
    eval_cfg: dict[str, Any],
    outputs_dir: Path,
    artifacts_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Run the config-driven evaluation stage and write all required artifacts."""

    logger = logger or logging.getLogger(__name__)
    cfg = _build_evaluation_cfg(eval_cfg)
    columns = cfg["columns"]
    factor_columns = list(cfg.get("factor_columns", []))
    eval_dir = outputs_dir / "evaluation"
    skipped: list[dict[str, Any]] = []

    defaults = {
        "model_name": str(cfg.get("model_name", "portfolio_gan")),
        "run_id": str(cfg.get("run_id", "unknown")),
        "prediction_source": str(cfg.get("prediction_source", "model")),
        "split": str(cfg.get("split", "")),
    }

    paths = _evaluation_paths(eval_dir)
    portfolio_df = pd.DataFrame()
    portfolio_metrics = pd.DataFrame(columns=portfolio_metric_columns())
    portfolio_summary = pd.DataFrame(columns=metric_summary_columns())

    portfolio_raw, portfolio_path = read_optional_table(project_root, cfg["inputs"].get("portfolio_predictions"))
    if portfolio_raw.empty:
        _skip(skipped, logger, "portfolio_reconstruction", "EXACT", "missing portfolio prediction artifact")
    else:
        try:
            portfolio_df = normalize_portfolio_frame(portfolio_raw, columns_cfg=columns, defaults=defaults)
            portfolio_metrics = _compute_portfolio_metrics(portfolio_df, cfg)
            portfolio_summary = summarize_metric_columns(
                portfolio_metrics,
                ["L_count", "L_concentration", "L_turnover"],
                status_by_metric=PORTFOLIO_METRIC_STATUS,
            )
            logger.info("Portfolio reconstruction metrics computed from %s", portfolio_path)
        except Exception as exc:
            _skip(skipped, logger, "portfolio_reconstruction", "EXACT", str(exc))
            portfolio_df = pd.DataFrame()

    write_table_outputs(portfolio_metrics, paths["portfolio_by_sample"])
    write_table_outputs(portfolio_summary, paths["portfolio_summary"])

    representation_metrics, representation_per_class = _run_representation(project_root, cfg, columns, logger, skipped)
    write_json(representation_metrics, paths["representation_metrics"])
    write_table_outputs(representation_per_class, paths["representation_per_class"])

    stability_by_fund, stability_summary = _run_stability(
        project_root,
        cfg,
        columns,
        factor_columns,
        portfolio_df,
        logger,
        skipped,
    )
    write_table_outputs(stability_by_fund, paths["stability_by_fund"])
    write_table_outputs(stability_summary, paths["stability_summary"])

    frontier_by_sample, frontier_summary = _run_frontier(
        project_root,
        cfg,
        columns,
        portfolio_df,
        logger,
        skipped,
    )
    write_table_outputs(frontier_by_sample, paths["frontier_by_sample"])
    write_table_outputs(frontier_summary, paths["frontier_summary"])

    counterfactual_by_case, counterfactual_summary = _run_counterfactual(
        project_root,
        cfg,
        columns,
        factor_columns,
        logger,
        skipped,
    )
    write_table_outputs(counterfactual_by_case, paths["counterfactual_by_case"])
    write_table_outputs(counterfactual_summary, paths["counterfactual_summary"])

    skipped_payload = {"skipped": skipped}
    write_json(skipped_payload, paths["skipped"])
    report = _build_report(
        cfg=cfg,
        portfolio_summary=portfolio_summary,
        representation_metrics=representation_metrics,
        representation_per_class=representation_per_class,
        stability_summary=stability_summary,
        frontier_summary=frontier_summary,
        counterfactual_summary=counterfactual_summary,
        skipped=skipped,
    )
    write_markdown(report, paths["report"])

    if skipped and bool(cfg.get("require_full_metrics", False)):
        raise RuntimeError(f"Evaluation skipped {len(skipped)} metric groups; see {paths['skipped']}")


def run_evaluation_stub(outputs_dir: Path) -> None:
    """Backward-compatible entry point retained for older scripts."""

    run_evaluation(Path("."), {}, outputs_dir)


def _compute_portfolio_metrics(portfolio_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    from src.evaluation.metrics_portfolio import compute_portfolio_metrics_by_sample

    return compute_portfolio_metrics_by_sample(
        portfolio_df,
        threshold=float(cfg.get("holding_threshold", 1e-4)),
        normalize_weights=bool(cfg.get("normalize_weights", True)),
        columns=cfg["columns"],
        default_model_name=str(cfg.get("model_name", "portfolio_gan")),
        default_run_id=str(cfg.get("run_id", "unknown")),
        default_prediction_source=str(cfg.get("prediction_source", "model")),
    )


def _run_representation(
    project_root: Path,
    cfg: dict[str, Any],
    columns: dict[str, str],
    logger: logging.Logger,
    skipped: list[dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    embeddings, path = read_optional_table(project_root, cfg["inputs"].get("representation_embeddings"))
    if embeddings.empty:
        _skip(skipped, logger, "strategy_representation_linear_probe", REPRESENTATION_STATUS, "missing embedding artifact")
        return _empty_representation_metrics("missing embedding artifact"), pd.DataFrame(columns=representation_per_class_columns())
    metrics, per_class = run_linear_probe(embeddings, columns=columns, cfg=cfg.get("representation", {}))
    if metrics.get("status") == "SKIPPED":
        _skip(skipped, logger, "strategy_representation_linear_probe", REPRESENTATION_STATUS, str(metrics.get("reason", "")))
    else:
        logger.info("Representation linear probe computed from %s", path)
    return metrics, per_class


def _run_stability(
    project_root: Path,
    cfg: dict[str, Any],
    columns: dict[str, str],
    factor_columns: list[str],
    portfolio_df: pd.DataFrame,
    logger: logging.Logger,
    skipped: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if portfolio_df.empty:
        _skip(skipped, logger, "factor_tilt_stability", STABILITY_STATUS, "missing normalized portfolio predictions")
        return (
            pd.DataFrame(columns=stability_by_fund_columns(factor_columns)),
            pd.DataFrame(columns=stability_summary_columns()),
        )
    factor_df = _load_factor_exposure_frame(
        project_root=project_root,
        cfg=cfg,
        columns=columns,
        factor_columns=factor_columns,
        logger=logger,
    )
    merged = merge_factor_exposures(portfolio_df, factor_df, factor_columns=factor_columns)
    missing = [factor for factor in factor_columns if factor not in merged.columns]
    if missing:
        _skip(skipped, logger, "factor_tilt_stability", STABILITY_STATUS, f"missing factor columns: {missing}")
        return (
            pd.DataFrame(columns=stability_by_fund_columns(factor_columns)),
            pd.DataFrame(columns=stability_summary_columns()),
        )
    try:
        by_fund, summary = compute_strategy_stability(
            merged,
            factor_columns=factor_columns,
            columns=columns,
            normalize_weights=bool(cfg.get("normalize_weights", True)),
            weight_columns=[columns.get("w_true", "w_true"), columns.get("w_pred", "w_pred")],
        )
    except Exception as exc:
        _skip(skipped, logger, "factor_tilt_stability", STABILITY_STATUS, str(exc))
        return (
            pd.DataFrame(columns=stability_by_fund_columns(factor_columns)),
            pd.DataFrame(columns=stability_summary_columns()),
        )
    if by_fund.empty:
        _skip(skipped, logger, "factor_tilt_stability", STABILITY_STATUS, "no funds with enough periods for drift")
    return by_fund, summary


def _run_frontier(
    project_root: Path,
    cfg: dict[str, Any],
    columns: dict[str, str],
    portfolio_df: pd.DataFrame,
    logger: logging.Logger,
    skipped: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frontier_cfg = dict(cfg.get("frontier", {}))
    if not bool(frontier_cfg.get("enabled", True)):
        _skip(skipped, logger, "markowitz_optimal_proximity", FRONTIER_STATUS, "disabled by config")
        return pd.DataFrame(columns=frontier_by_sample_columns()), pd.DataFrame(columns=metric_summary_columns())
    if portfolio_df.empty:
        _skip(skipped, logger, "markowitz_optimal_proximity", FRONTIER_STATUS, "missing normalized portfolio predictions")
        return pd.DataFrame(columns=frontier_by_sample_columns()), pd.DataFrame(columns=metric_summary_columns())
    returns, path = read_optional_table(project_root, cfg["inputs"].get("asset_returns"))
    if returns.empty:
        _skip(skipped, logger, "markowitz_optimal_proximity", FRONTIER_STATUS, "missing asset return panel")
        return pd.DataFrame(columns=frontier_by_sample_columns()), pd.DataFrame(columns=metric_summary_columns())
    try:
        returns = normalize_return_frame(returns, columns_cfg=columns)
        frontier_cfg["holding_threshold"] = float(cfg.get("holding_threshold", 1e-4))
        by_sample, summary = compute_frontier_metrics(
            portfolio_df,
            returns,
            columns=columns,
            cfg=frontier_cfg,
            normalize_weights=bool(cfg.get("normalize_weights", True)),
        )
        if by_sample.empty:
            _skip(skipped, logger, "markowitz_optimal_proximity", FRONTIER_STATUS, "no valid return windows")
        else:
            logger.info("Frontier metrics computed from %s", path)
        return by_sample, summary
    except Exception as exc:
        _skip(skipped, logger, "markowitz_optimal_proximity", FRONTIER_STATUS, str(exc))
        return pd.DataFrame(columns=frontier_by_sample_columns()), pd.DataFrame(columns=metric_summary_columns())


def _run_counterfactual(
    project_root: Path,
    cfg: dict[str, Any],
    columns: dict[str, str],
    factor_columns: list[str],
    logger: logging.Logger,
    skipped: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not bool(cfg.get("counterfactual", {}).get("enabled", True)):
        _skip(skipped, logger, "counterfactual_transfer", COUNTERFACTUAL_STATUS, "disabled by config")
        return pd.DataFrame(columns=counterfactual_by_case_columns(factor_columns)), pd.DataFrame(columns=metric_summary_columns())
    counterfactual, path = read_optional_table(project_root, cfg["inputs"].get("counterfactual_transfers"))
    if counterfactual.empty:
        _skip(skipped, logger, "counterfactual_transfer", COUNTERFACTUAL_STATUS, "missing counterfactual transfer artifact")
        return pd.DataFrame(columns=counterfactual_by_case_columns(factor_columns)), pd.DataFrame(columns=metric_summary_columns())
    try:
        normalized = normalize_portfolio_frame(counterfactual, columns_cfg=columns, defaults={"prediction_source": "counterfactual"})
        factor_df = _load_factor_exposure_frame(
            project_root=project_root,
            cfg=cfg,
            columns=columns,
            factor_columns=factor_columns,
            logger=logger,
        )
        normalized = merge_factor_exposures(normalized, factor_df, factor_columns=factor_columns)
        by_case, summary = compute_counterfactual_metrics(
            normalized,
            factor_columns=factor_columns,
            columns=columns,
            threshold=float(cfg.get("holding_threshold", 1e-4)),
            normalize_weights=bool(cfg.get("normalize_weights", True)),
        )
        logger.info("Counterfactual metrics computed from %s", path)
        return by_case, summary
    except Exception as exc:
        _skip(skipped, logger, "counterfactual_transfer", COUNTERFACTUAL_STATUS, str(exc))
        return pd.DataFrame(columns=counterfactual_by_case_columns(factor_columns)), pd.DataFrame(columns=metric_summary_columns())


def _build_evaluation_cfg(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_EVALUATION_CONFIG)
    user_cfg = eval_cfg.get("evaluation", eval_cfg)
    _deep_update(cfg, user_cfg)
    return cfg


def _load_factor_exposure_frame(
    *,
    project_root: Path,
    cfg: dict[str, Any],
    columns: dict[str, str],
    factor_columns: list[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    factor_df, factor_path = read_optional_table(project_root, cfg["inputs"].get("factor_exposures"))
    factor_df = normalize_factor_frame(
        factor_df,
        columns_cfg=columns,
        factor_columns=factor_columns,
        factor_aliases=cfg.get("factor_aliases", {}),
    )
    if factor_df.empty:
        missing = factor_columns
    else:
        missing = [factor for factor in factor_columns if factor not in factor_df.columns]
    if not missing:
        return factor_df

    estimate_cfg = dict(cfg.get("factor_exposure_estimation", {}))
    if not bool(estimate_cfg.get("enabled", True)):
        return factor_df

    returns, returns_path = read_optional_table(project_root, cfg["inputs"].get("asset_returns"))
    carhart, carhart_path = read_optional_table(project_root, cfg["inputs"].get("carhart_factors"))
    if returns.empty or carhart.empty:
        logger.warning(
            "Cannot estimate Carhart betas; returns_path=%s carhart_path=%s",
            returns_path,
            carhart_path,
        )
        return factor_df

    betas = build_or_load_carhart_factor_exposures(
        project_root=project_root,
        stock_returns=returns,
        carhart_factors=carhart,
        cfg=estimate_cfg,
    )
    betas = normalize_factor_frame(
        betas,
        columns_cfg=columns,
        factor_columns=factor_columns,
        factor_aliases=cfg.get("factor_aliases", {}),
    )
    if not betas.empty:
        logger.info("Using Carhart rolling beta factor exposures for missing columns: %s", missing)
        return betas

    logger.warning("Carhart rolling beta estimation produced no rows; falling back to configured factor exposure frame")
    return factor_df


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _skip(
    skipped: list[dict[str, Any]],
    logger: logging.Logger,
    metric: str,
    status: str,
    reason: str,
) -> None:
    skipped.append({"metric": metric, "status": status, "reason": reason})
    logger.warning("Skipping %s (%s): %s", metric, status, reason)


def _empty_representation_metrics(reason: str) -> dict[str, Any]:
    return {
        "status": "SKIPPED",
        "reason": reason,
        "classifier": "",
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "accuracy": None,
        "labels": [],
        "confusion_matrix": [],
    }


def _evaluation_paths(eval_dir: Path) -> dict[str, Path]:
    return {
        "portfolio_by_sample": eval_dir / "portfolio_metrics_by_sample.parquet",
        "portfolio_summary": eval_dir / "portfolio_metrics_summary.csv",
        "representation_metrics": eval_dir / "representation_metrics.json",
        "representation_per_class": eval_dir / "representation_per_class.csv",
        "stability_by_fund": eval_dir / "stability_by_fund.parquet",
        "stability_summary": eval_dir / "stability_summary.csv",
        "frontier_by_sample": eval_dir / "frontier_metrics_by_sample.parquet",
        "frontier_summary": eval_dir / "frontier_summary.csv",
        "counterfactual_by_case": eval_dir / "counterfactual_metrics_by_case.parquet",
        "counterfactual_summary": eval_dir / "counterfactual_summary.csv",
        "skipped": eval_dir / "skipped_metrics.json",
        "report": eval_dir / "evaluation_report.md",
    }


def _build_report(
    *,
    cfg: dict[str, Any],
    portfolio_summary: pd.DataFrame,
    representation_metrics: dict[str, Any],
    representation_per_class: pd.DataFrame,
    stability_summary: pd.DataFrame,
    frontier_summary: pd.DataFrame,
    counterfactual_summary: pd.DataFrame,
    skipped: list[dict[str, Any]],
) -> str:
    lines = [
        "# Evaluation Report",
        "",
        "This report is produced by the scaffold evaluation layer. It does not claim scientific replication.",
        "",
        "## Run Context",
        "",
        f"- Model: `{cfg.get('model_name', 'portfolio_gan')}`",
        f"- Run ID: `{cfg.get('run_id', 'unknown')}`",
        f"- Default split: `{cfg.get('split', '')}`",
        f"- Holding threshold: `{cfg.get('holding_threshold', '')}`",
        f"- Normalize weights: `{cfg.get('normalize_weights', '')}`",
        "",
        "## Metric Status",
        "",
        "| Metric group | Status | Notes |",
        "|---|---|---|",
        "| Portfolio reconstruction (`L_count`, `L_concentration`, `L_turnover`) | EXACT | Paper formulas implemented directly when `w_true`, `w_pred`, and optional `w_prev` are available. |",
        "| Strategy representation linear probe | EXACT | Linear SVM by default; logistic regression fallback keeps a linear decision boundary. |",
        "| Factor tilt stability | EXACT | Formula implemented directly over configured factor exposure columns. |",
        "| Markowitz optimal-proximity | CLOSE | Uses sampled long-only mean-variance frontier because exact optimization details are not fully specified. |",
        "| Counterfactual transfer preservation | CLOSE | Computes configured exposure deltas and structural deltas for available transfer artifacts. |",
        "",
        "## Portfolio Reconstruction Summary",
        "",
        markdown_table(portfolio_summary),
        "",
        "## Representation Metrics",
        "",
        markdown_table(pd.DataFrame([_flatten_representation_metrics(representation_metrics)])),
        "",
        "## Representation Per-Class Metrics",
        "",
        markdown_table(representation_per_class),
        "",
        "## Strategy Stability Summary",
        "",
        markdown_table(stability_summary),
        "",
        "## Frontier Proximity Summary",
        "",
        markdown_table(frontier_summary),
        "",
        "## Counterfactual Summary",
        "",
        markdown_table(counterfactual_summary),
        "",
        "## Skipped Metrics",
        "",
        markdown_table(pd.DataFrame(skipped) if skipped else pd.DataFrame(columns=["metric", "status", "reason"])),
        "",
        "## Caveats",
        "",
        "- Missing generated prediction, embedding, or transfer artifacts cause metric groups to be skipped rather than fabricated.",
        "- Frontier proximity is an implementation approximation and should not be compared to paper tables without confirming optimization details and data alignment.",
        "- Factor-based metrics are only as faithful as the configured factor exposure columns and source-field mapping.",
    ]
    return "\n".join(lines) + "\n"


def _flatten_representation_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(metrics)
    if "embedding_columns" in flattened:
        flattened["embedding_columns"] = ",".join(flattened["embedding_columns"])
    if "labels" in flattened:
        flattened["labels"] = ",".join(map(str, flattened["labels"]))
    if "confusion_matrix" in flattened:
        flattened["confusion_matrix"] = str(flattened["confusion_matrix"])
    return flattened
