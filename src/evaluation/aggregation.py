from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_GROUP_COLS = ["model_name", "run_id", "prediction_source", "split"]


def metric_summary_columns() -> list[str]:
    return [
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


def summarize_metric_columns(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    *,
    status_by_metric: Mapping[str, str],
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
) -> pd.DataFrame:
    """Summarize metric columns by model/run/source/split."""

    if df.empty:
        return pd.DataFrame(columns=metric_summary_columns())

    present_group_cols = [col for col in group_cols if col in df.columns]
    if not present_group_cols:
        working = df.copy()
        working["_all"] = "all"
        present_group_cols = ["_all"]
    else:
        working = df.copy()

    rows: list[dict[str, Any]] = []
    for key, group in working.groupby(present_group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        meta = dict(zip(present_group_cols, key, strict=False))
        for metric in metric_cols:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row = {
                "model_name": meta.get("model_name", "all"),
                "run_id": meta.get("run_id", "all"),
                "prediction_source": meta.get("prediction_source", "all"),
                "split": meta.get("split", "all"),
                "metric": metric,
                "status": status_by_metric.get(metric, ""),
                "count": int(values.shape[0]),
                "mean": float(values.mean()) if not values.empty else np.nan,
                "median": float(values.median()) if not values.empty else np.nan,
                "std": float(values.std(ddof=0)) if values.shape[0] > 1 else 0.0 if values.shape[0] == 1 else np.nan,
                "p05": float(values.quantile(0.05)) if not values.empty else np.nan,
                "p25": float(values.quantile(0.25)) if not values.empty else np.nan,
                "p75": float(values.quantile(0.75)) if not values.empty else np.nan,
                "p95": float(values.quantile(0.95)) if not values.empty else np.nan,
            }
            rows.append(row)

    return pd.DataFrame(rows, columns=metric_summary_columns())


def summary_stats(
    values: pd.Series,
    *,
    prefix: str = "",
) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    name = f"{prefix}_" if prefix else ""
    return {
        f"{name}count": int(clean.shape[0]),
        f"{name}mean": float(clean.mean()) if not clean.empty else np.nan,
        f"{name}median": float(clean.median()) if not clean.empty else np.nan,
        f"{name}std": float(clean.std(ddof=0)) if clean.shape[0] > 1 else 0.0 if clean.shape[0] == 1 else np.nan,
        f"{name}p05": float(clean.quantile(0.05)) if not clean.empty else np.nan,
        f"{name}p25": float(clean.quantile(0.25)) if not clean.empty else np.nan,
        f"{name}p75": float(clean.quantile(0.75)) if not clean.empty else np.nan,
        f"{name}p95": float(clean.quantile(0.95)) if not clean.empty else np.nan,
    }


def markdown_table(df: pd.DataFrame, *, max_rows: int = 20) -> str:
    """Render a small markdown table without requiring optional tabulate."""

    if df.empty:
        return "_No rows._"
    display = df.head(max_rows).copy()
    display = display.where(pd.notna(display), "")
    headers = [str(col) for col in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(_format_markdown_cell(row[col]) for col in display.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def _format_markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.6g}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
