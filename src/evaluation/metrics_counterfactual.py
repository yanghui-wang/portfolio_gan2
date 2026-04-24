from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.aggregation import summarize_metric_columns
from src.evaluation.metrics_portfolio import normalize_weight_vector, structural_deltas


COUNTERFACTUAL_STATUS = "CLOSE"


def counterfactual_by_case_columns(factor_columns: Sequence[str] | None = None) -> list[str]:
    factor_columns = list(factor_columns or [])
    return [
        "model_name",
        "run_id",
        "prediction_source",
        "split",
        "case_id",
        "source_fund_id",
        "target_fund_id",
        "source_date",
        "target_date",
        "status",
    ] + [f"{factor}_original" for factor in factor_columns] + [f"{factor}_transferred" for factor in factor_columns] + [
        f"{factor}_delta" for factor in factor_columns
    ] + [
        "count_delta",
        "concentration_delta",
        "turnover_delta",
    ]


def compute_counterfactual_metrics(
    counterfactual: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
    columns: Mapping[str, str],
    threshold: float,
    normalize_weights: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute strategy-transfer exposure and structural preservation metrics."""

    if counterfactual.empty:
        empty = pd.DataFrame(columns=counterfactual_by_case_columns(factor_columns))
        return empty, _empty_counterfactual_summary()

    asset_col = columns.get("asset_id", "asset_id")
    model_col = columns.get("model_name", "model_name")
    run_col = columns.get("run_id", "run_id")
    source_col = columns.get("prediction_source", "prediction_source")
    split_col = columns.get("split", "split")
    case_col = columns.get("case_id", "case_id")
    original_col = columns.get("w_original", "w_original")
    transferred_col = columns.get("w_transferred", "w_transferred")
    prev_col = columns.get("w_prev_transferred", "w_prev_transferred")

    required = [asset_col, original_col, transferred_col]
    missing = [col for col in required if col not in counterfactual.columns]
    if missing:
        raise ValueError(f"Counterfactual metrics missing columns: {missing}")
    present_factors = [factor for factor in factor_columns if factor in counterfactual.columns]
    if not present_factors:
        raise ValueError("Counterfactual metrics require at least one configured factor exposure column")

    df = counterfactual.copy()
    for optional, default in [(model_col, "portfolio_gan"), (run_col, "unknown"), (source_col, "counterfactual"), (split_col, "")]:
        if optional not in df.columns:
            df[optional] = default
    if case_col not in df.columns:
        df[case_col] = _default_case_id(df, columns)

    group_cols = [model_col, run_col, source_col, split_col, case_col]
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_cols, dropna=False, sort=True):
        original = _asset_weight_series(group, asset_col=asset_col, weight_col=original_col)
        transferred = _asset_weight_series(group, asset_col=asset_col, weight_col=transferred_col)
        if normalize_weights:
            original = pd.Series(normalize_weight_vector(original), index=original.index)
            transferred = pd.Series(normalize_weight_vector(transferred), index=transferred.index)

        factors = group.drop_duplicates(subset=[asset_col]).set_index(group.drop_duplicates(subset=[asset_col])[asset_col].astype(str))
        factors = factors.reindex(original.index.union(transferred.index))
        factor_matrix = factors[present_factors].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        original_aligned = original.reindex(factor_matrix.index, fill_value=0.0).to_numpy(dtype=float)
        transferred_aligned = transferred.reindex(factor_matrix.index, fill_value=0.0).to_numpy(dtype=float)
        factor_values = factor_matrix.to_numpy(dtype=float)
        original_exp = original_aligned @ factor_values
        transferred_exp = transferred_aligned @ factor_values

        meta = dict(zip(group_cols, key, strict=False))
        row: dict[str, Any] = {
            "model_name": meta[model_col],
            "run_id": meta[run_col],
            "prediction_source": meta[source_col],
            "split": meta[split_col],
            "case_id": meta[case_col],
            "source_fund_id": _first_existing(group, ["source_fund_id", columns.get("fund_id", "fund_id")]),
            "target_fund_id": _first_existing(group, ["target_fund_id"]),
            "source_date": _first_existing(group, ["source_date", columns.get("date", "date")]),
            "target_date": _first_existing(group, ["target_date"]),
            "status": COUNTERFACTUAL_STATUS,
        }
        for factor, value in zip(present_factors, original_exp, strict=False):
            row[f"{factor}_original"] = float(value)
        for factor, value in zip(present_factors, transferred_exp, strict=False):
            row[f"{factor}_transferred"] = float(value)
        for factor in present_factors:
            row[f"{factor}_delta"] = abs(float(row[f"{factor}_transferred"]) - float(row[f"{factor}_original"]))

        prev = _asset_weight_series(group, asset_col=asset_col, weight_col=prev_col) if prev_col in group.columns else None
        row.update(structural_deltas(original, transferred, threshold=threshold, w_prev=prev))
        if "turnover_delta" not in row:
            row["turnover_delta"] = np.nan
        rows.append(row)

    by_case = pd.DataFrame(rows)
    by_case = by_case.reindex(columns=counterfactual_by_case_columns(present_factors))
    metric_cols = [f"{factor}_delta" for factor in present_factors] + ["count_delta", "concentration_delta", "turnover_delta"]
    summary = summarize_metric_columns(
        by_case,
        metric_cols,
        status_by_metric={metric: COUNTERFACTUAL_STATUS for metric in metric_cols},
    )
    return by_case, summary


def _asset_weight_series(group: pd.DataFrame, *, asset_col: str, weight_col: str) -> pd.Series:
    values = pd.to_numeric(group[weight_col], errors="coerce").fillna(0.0)
    return pd.Series(values.to_numpy(dtype=float), index=group[asset_col].astype(str)).groupby(level=0).sum()


def _default_case_id(df: pd.DataFrame, columns: Mapping[str, str]) -> pd.Series:
    pieces: list[pd.Series] = []
    for col in [columns.get("fund_id", "fund_id"), columns.get("date", "date")]:
        if col in df.columns:
            pieces.append(df[col].astype(str))
    if not pieces:
        return pd.Series(np.arange(df.shape[0]).astype(str), index=df.index)
    out = pieces[0]
    for part in pieces[1:]:
        out = out + "_" + part
    return out


def _first_existing(group: pd.DataFrame, columns: Sequence[str | None]) -> Any:
    for col in columns:
        if col and col in group.columns:
            non_null = group[col].dropna()
            if not non_null.empty:
                return non_null.iloc[0]
    return ""


def _empty_counterfactual_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_name",
            "run_id",
            "prediction_source",
            "split",
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
    )
