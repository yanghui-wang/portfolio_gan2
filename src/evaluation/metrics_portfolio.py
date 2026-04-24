from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


PORTFOLIO_METRIC_STATUS = {
    "L_count": "EXACT",
    "L_concentration": "EXACT",
    "L_turnover": "EXACT",
}


def _as_numeric_array(weights: Any, *, name: str = "weights") -> np.ndarray:
    """Coerce dense or sparse weights to a finite float array with missing values set to zero."""

    if isinstance(weights, pd.Series):
        values = pd.to_numeric(weights, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    elif isinstance(weights, Mapping):
        values = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        values = pd.to_numeric(pd.Series(np.asarray(weights).ravel()), errors="coerce").fillna(0.0)
        arr = values.to_numpy(dtype=float).reshape(np.asarray(weights).shape)
        values = arr

    arr = np.asarray(values, dtype=float)
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains infinite values")
    return arr


def _align_weight_inputs(*weights: Any) -> list[np.ndarray]:
    """Align sparse pandas/dict weights by index and fill missing assets with zero."""

    sparse_like = any(isinstance(w, (pd.Series, Mapping)) for w in weights)
    if not sparse_like:
        arrays = [_as_numeric_array(w) for w in weights]
        base_shape = arrays[0].shape
        for arr in arrays[1:]:
            if arr.shape != base_shape:
                raise ValueError(f"Weight arrays must share shape; got {base_shape} and {arr.shape}")
        return arrays

    series_list: list[pd.Series] = []
    union_index = pd.Index([])
    for weight in weights:
        if isinstance(weight, pd.Series):
            series = pd.to_numeric(weight, errors="coerce")
        elif isinstance(weight, Mapping):
            series = pd.to_numeric(pd.Series(weight), errors="coerce")
        else:
            series = pd.Series(_as_numeric_array(weight))
        series = series.fillna(0.0)
        if np.isinf(series.to_numpy(dtype=float)).any():
            raise ValueError("weights contain infinite values")
        series_list.append(series)
        union_index = union_index.union(series.index)

    return [series.reindex(union_index, fill_value=0.0).to_numpy(dtype=float) for series in series_list]


def normalize_weight_vector(weights: Any) -> np.ndarray:
    """Normalize a weight vector to sum to one when the sum is non-zero."""

    arr = _as_numeric_array(weights)
    total = float(arr.sum())
    if np.isclose(total, 0.0):
        return arr
    return arr / total


def maybe_normalize(weights: Any, normalize: bool) -> np.ndarray:
    arr = _as_numeric_array(weights)
    if not normalize:
        return arr
    return normalize_weight_vector(arr)


def holding_count(weights: Any, threshold: float) -> int | np.ndarray:
    """Count assets whose weight exceeds the configured holding threshold."""

    arr = _as_numeric_array(weights)
    counts = (arr > float(threshold)).sum(axis=-1)
    if np.ndim(counts) == 0:
        return int(counts)
    return counts


def herfindahl_index(weights: Any) -> float | np.ndarray:
    """Compute the Herfindahl concentration index sum(w_i^2)."""

    arr = _as_numeric_array(weights)
    values = np.square(arr).sum(axis=-1)
    if np.ndim(values) == 0:
        return float(values)
    return values


def portfolio_turnover(w_current: Any, w_prev: Any) -> float | np.ndarray:
    """Compute sum(abs(w_current - w_prev)) after sparse asset alignment."""

    current, prev = _align_weight_inputs(w_current, w_prev)
    values = np.abs(current - prev).sum(axis=-1)
    if np.ndim(values) == 0:
        return float(values)
    return values


def count_error(w_true: Any, w_pred: Any, threshold: float) -> float:
    """Paper Lcount: absolute difference in holding counts above threshold."""

    true, pred = _align_weight_inputs(w_true, w_pred)
    return float(abs(int(holding_count(true, threshold)) - int(holding_count(pred, threshold))))


def concentration_error(w_true: Any, w_pred: Any) -> float:
    """Paper Lconcentration: absolute difference in Herfindahl concentration."""

    true, pred = _align_weight_inputs(w_true, w_pred)
    return float(abs(float(herfindahl_index(pred)) - float(herfindahl_index(true))))


def turnover_error(w_true: Any, w_pred: Any, w_prev: Any) -> float:
    """Paper Lturnover: absolute difference in turnover relative to previous weights."""

    true, pred, prev = _align_weight_inputs(w_true, w_pred, w_prev)
    return float(abs(float(portfolio_turnover(pred, prev)) - float(portfolio_turnover(true, prev))))


def _series_for_group(group: pd.DataFrame, asset_col: str | None, weight_col: str) -> pd.Series:
    if weight_col not in group.columns:
        raise ValueError(f"Missing required weight column: {weight_col}")
    values = pd.to_numeric(group[weight_col], errors="coerce").fillna(0.0)
    if asset_col and asset_col in group.columns:
        return pd.Series(values.to_numpy(dtype=float), index=group[asset_col].astype(str))
    return pd.Series(values.to_numpy(dtype=float), index=np.arange(len(group)))


def compute_portfolio_metrics_by_sample(
    portfolios: pd.DataFrame,
    *,
    threshold: float,
    normalize_weights: bool,
    columns: Mapping[str, str],
    default_model_name: str = "portfolio_gan",
    default_run_id: str = "unknown",
    default_prediction_source: str = "model",
) -> pd.DataFrame:
    """Compute paper reconstruction metrics for each fund-date-model sample.

    The function expects a normalized long table, but column names remain
    configurable so tests and later generated artifacts can use the same metric
    implementation.
    """

    if portfolios.empty:
        return pd.DataFrame(columns=portfolio_metric_columns())

    w_true_col = columns.get("w_true", "w_true")
    w_pred_col = columns.get("w_pred", "w_pred")
    w_prev_col = columns.get("w_prev", "w_prev")
    fund_col = columns.get("fund_id", "fund_id")
    date_col = columns.get("date", "date")
    asset_col = columns.get("asset_id", "asset_id")
    split_col = columns.get("split", "split")
    model_col = columns.get("model_name", "model_name")
    run_col = columns.get("run_id", "run_id")
    source_col = columns.get("prediction_source", "prediction_source")

    required = [fund_col, date_col, w_true_col, w_pred_col]
    missing = [col for col in required if col not in portfolios.columns]
    if missing:
        raise ValueError(f"Portfolio metrics missing required columns: {missing}")

    df = portfolios.copy()
    if model_col not in df.columns:
        df[model_col] = default_model_name
    if run_col not in df.columns:
        df[run_col] = default_run_id
    if source_col not in df.columns:
        df[source_col] = default_prediction_source
    if split_col not in df.columns:
        df[split_col] = ""

    group_cols = [model_col, run_col, source_col, split_col, fund_col, date_col]
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_cols, dropna=False, sort=True):
        meta = dict(zip(group_cols, key, strict=False))
        true = _series_for_group(group, asset_col, w_true_col)
        pred = _series_for_group(group, asset_col, w_pred_col)
        if normalize_weights:
            true = pd.Series(normalize_weight_vector(true), index=true.index)
            pred = pd.Series(normalize_weight_vector(pred), index=pred.index)

        row: dict[str, Any] = {
            "model_name": meta[model_col],
            "run_id": meta[run_col],
            "prediction_source": meta[source_col],
            "split": meta[split_col],
            "fund_id": meta[fund_col],
            "date": meta[date_col],
            "L_count": count_error(true, pred, threshold),
            "L_count_status": PORTFOLIO_METRIC_STATUS["L_count"],
            "L_concentration": concentration_error(true, pred),
            "L_concentration_status": PORTFOLIO_METRIC_STATUS["L_concentration"],
            "L_turnover": np.nan,
            "L_turnover_status": PORTFOLIO_METRIC_STATUS["L_turnover"],
        }

        if w_prev_col in group.columns:
            prev = _series_for_group(group, asset_col, w_prev_col)
            if normalize_weights:
                prev = pd.Series(normalize_weight_vector(prev), index=prev.index)
            row["L_turnover"] = turnover_error(true, pred, prev)
        rows.append(row)

    return pd.DataFrame(rows, columns=portfolio_metric_columns())


def portfolio_metric_columns() -> list[str]:
    return [
        "model_name",
        "run_id",
        "prediction_source",
        "split",
        "fund_id",
        "date",
        "L_count",
        "L_count_status",
        "L_concentration",
        "L_concentration_status",
        "L_turnover",
        "L_turnover_status",
    ]


def structural_deltas(
    original: Sequence[float] | pd.Series | Mapping[Any, float],
    transferred: Sequence[float] | pd.Series | Mapping[Any, float],
    *,
    threshold: float,
    w_prev: Sequence[float] | pd.Series | Mapping[Any, float] | None = None,
) -> dict[str, float]:
    """Compute optional structural preservation deltas for counterfactual cases."""

    original_arr, transferred_arr = _align_weight_inputs(original, transferred)
    result = {
        "count_delta": count_error(original_arr, transferred_arr, threshold),
        "concentration_delta": concentration_error(original_arr, transferred_arr),
    }
    if w_prev is not None:
        original_arr, transferred_arr, prev_arr = _align_weight_inputs(original, transferred, w_prev)
        result["turnover_delta"] = turnover_error(original_arr, transferred_arr, prev_arr)
    return result
