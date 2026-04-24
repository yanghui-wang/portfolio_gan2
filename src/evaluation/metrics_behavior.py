from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.aggregation import summary_stats
from src.evaluation.metrics_portfolio import normalize_weight_vector


STABILITY_STATUS = "EXACT"


def stability_by_fund_columns(factor_columns: Sequence[str] | None = None) -> list[str]:
    factor_columns = list(factor_columns or [])
    return [
        "model_name",
        "run_id",
        "prediction_source",
        "split",
        "portfolio_type",
        "fund_id",
        "n_periods",
        "factor_tilt_stability",
        "status",
    ] + [f"mean_abs_drift_{col}" for col in factor_columns]


def stability_summary_columns() -> list[str]:
    return [
        "model_name",
        "run_id",
        "prediction_source",
        "split",
        "portfolio_type",
        "metric",
        "status",
        "count",
        "mean",
        "median",
        "std",
        "p05",
        "p25",
        "p75",
        "p95",
    ]


def compute_factor_exposures(
    portfolios: pd.DataFrame,
    *,
    weight_col: str,
    factor_columns: Sequence[str],
    columns: Mapping[str, str],
    normalize_weights: bool,
) -> pd.DataFrame:
    """Compute beta_{a,t} = w_{a,t}' X_t for each fund/date portfolio."""

    if portfolios.empty:
        return pd.DataFrame()
    factor_columns = [col for col in factor_columns if col in portfolios.columns]
    if not factor_columns:
        raise ValueError("No configured factor columns are present in the portfolio/factor table")
    if weight_col not in portfolios.columns:
        raise ValueError(f"Missing weight column for exposure computation: {weight_col}")

    fund_col = columns.get("fund_id", "fund_id")
    date_col = columns.get("date", "date")
    split_col = columns.get("split", "split")
    model_col = columns.get("model_name", "model_name")
    run_col = columns.get("run_id", "run_id")
    source_col = columns.get("prediction_source", "prediction_source")

    df = portfolios.copy()
    for optional, default in [(model_col, "portfolio_gan"), (run_col, "unknown"), (source_col, "model"), (split_col, "")]:
        if optional not in df.columns:
            df[optional] = default
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    group_cols = [model_col, run_col, source_col, split_col, fund_col, date_col]
    rows: list[dict[str, Any]] = []
    for key, group in df.dropna(subset=[date_col]).groupby(group_cols, dropna=False, sort=True):
        weights = pd.to_numeric(group[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if normalize_weights:
            weights = normalize_weight_vector(weights)
        factors = group[list(factor_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        exposure = weights @ factors
        row = {
            "model_name": key[0],
            "run_id": key[1],
            "prediction_source": key[2],
            "split": key[3],
            "fund_id": key[4],
            "date": key[5],
        }
        row.update({factor: float(value) for factor, value in zip(factor_columns, exposure, strict=False)})
        rows.append(row)
    return pd.DataFrame(rows)


def compute_strategy_stability(
    portfolios: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
    columns: Mapping[str, str],
    normalize_weights: bool,
    weight_columns: Sequence[str] = ("w_true", "w_pred"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the paper's factor tilt drift metric.

    For each weight column, the implementation computes beta_{a,t}=w_{a,t}'X_t,
    subtracts the cross-sectional date average beta_bar_t, then averages
    sum(abs(relative_beta_t - relative_beta_{t-1})) by fund.
    """

    if portfolios.empty:
        return pd.DataFrame(columns=stability_by_fund_columns(factor_columns)), pd.DataFrame(columns=stability_summary_columns())

    by_fund_parts: list[pd.DataFrame] = []
    for weight_col in weight_columns:
        if weight_col not in portfolios.columns:
            continue
        exposures = compute_factor_exposures(
            portfolios,
            weight_col=weight_col,
            factor_columns=factor_columns,
            columns=columns,
            normalize_weights=normalize_weights,
        )
        if exposures.empty:
            continue
        present_factors = [col for col in factor_columns if col in exposures.columns]
        exposures["portfolio_type"] = _portfolio_type(weight_col, columns)
        avg_cols = ["model_name", "run_id", "prediction_source", "split", "portfolio_type", "date"]
        beta_bar = exposures.groupby(avg_cols, dropna=False)[present_factors].transform("mean")
        rel = exposures.copy()
        for factor in present_factors:
            rel[f"relative_{factor}"] = rel[factor] - beta_bar[factor]

        rel_cols = [f"relative_{factor}" for factor in present_factors]
        group_cols = ["model_name", "run_id", "prediction_source", "split", "portfolio_type", "fund_id"]
        rows: list[dict[str, Any]] = []
        for key, group in rel.groupby(group_cols, dropna=False, sort=True):
            ordered = group.sort_values("date")
            diffs = ordered[rel_cols].diff().abs()
            total_drift = diffs.sum(axis=1).iloc[1:]
            row = dict(zip(group_cols, key, strict=False))
            row["n_periods"] = int(ordered.shape[0])
            row["factor_tilt_stability"] = float(total_drift.mean()) if not total_drift.empty else np.nan
            row["status"] = STABILITY_STATUS
            factor_drift = diffs.iloc[1:].mean(axis=0)
            for factor in present_factors:
                row[f"mean_abs_drift_{factor}"] = float(factor_drift.get(f"relative_{factor}", np.nan))
            rows.append(row)
        by_fund_parts.append(pd.DataFrame(rows))

    if not by_fund_parts:
        return pd.DataFrame(columns=stability_by_fund_columns(factor_columns)), pd.DataFrame(columns=stability_summary_columns())

    by_fund = pd.concat(by_fund_parts, ignore_index=True)
    by_fund = by_fund.reindex(columns=stability_by_fund_columns(factor_columns))
    summary = _summarize_stability(by_fund)
    return by_fund, summary


def _portfolio_type(weight_col: str, columns: Mapping[str, str]) -> str:
    if weight_col == columns.get("w_true", "w_true"):
        return "real"
    if weight_col == columns.get("w_pred", "w_pred"):
        return "generated"
    return weight_col


def _summarize_stability(by_fund: pd.DataFrame) -> pd.DataFrame:
    if by_fund.empty:
        return pd.DataFrame(columns=stability_summary_columns())
    rows: list[dict[str, Any]] = []
    group_cols = ["model_name", "run_id", "prediction_source", "split", "portfolio_type"]
    for key, group in by_fund.groupby(group_cols, dropna=False, sort=True):
        meta = dict(zip(group_cols, key, strict=False))
        row: dict[str, Any] = {
            **meta,
            "metric": "factor_tilt_stability",
            "status": STABILITY_STATUS,
        }
        row.update(summary_stats(group["factor_tilt_stability"]))
        rows.append(row)
    return pd.DataFrame(rows, columns=stability_summary_columns())
