from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.aggregation import summarize_metric_columns
from src.evaluation.metrics_portfolio import holding_count, normalize_weight_vector


FRONTIER_STATUS = "CLOSE"


def frontier_by_sample_columns() -> list[str]:
    return [
        "model_name",
        "run_id",
        "prediction_source",
        "split",
        "fund_id",
        "date",
        "portfolio_return",
        "portfolio_risk",
        "frontier_distance",
        "random_reference_distance_mean",
        "random_reference_distance_median",
        "random_reference_type",
        "status",
    ]


def estimate_portfolio_return(portfolio_weights: np.ndarray, expected_returns: np.ndarray) -> float:
    return float(np.asarray(portfolio_weights, dtype=float) @ np.asarray(expected_returns, dtype=float))


def estimate_portfolio_risk(portfolio_weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    weights = np.asarray(portfolio_weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    variance = float(weights @ cov @ weights.T)
    return float(np.sqrt(max(variance, 0.0)))


def build_efficient_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    *,
    num_random_portfolios: int,
    long_only: bool,
    random_seed: int,
) -> pd.DataFrame:
    """Approximate an ex-post frontier using random long-only portfolios.

    This is intentionally tagged CLOSE because the paper's exact optimization
    procedure and constraints are not fully specified in the scaffold inputs.
    """

    expected = np.asarray(expected_returns, dtype=float)
    n_assets = expected.shape[0]
    if n_assets == 0:
        return pd.DataFrame(columns=["risk", "return"])

    rng = np.random.default_rng(random_seed)
    samples = max(int(num_random_portfolios), n_assets + 1)
    if long_only:
        weights = rng.dirichlet(np.ones(n_assets), size=samples)
    else:
        weights = rng.normal(size=(samples, n_assets))
        weights = weights / (np.abs(weights).sum(axis=1, keepdims=True) + 1e-12)

    equal_weight = np.full((1, n_assets), 1.0 / n_assets)
    one_hot = np.eye(n_assets)
    weights = np.vstack([weights, equal_weight, one_hot])

    returns = weights @ expected
    risks = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", weights, cov_matrix, weights), 0.0))
    frontier = pd.DataFrame({"risk": risks, "return": returns}).replace([np.inf, -np.inf], np.nan).dropna()
    if frontier.empty:
        return frontier
    frontier = frontier.sort_values(["risk", "return"], ascending=[True, False])
    frontier["max_return_so_far"] = frontier["return"].cummax()
    efficient = frontier.loc[frontier["return"] >= frontier["max_return_so_far"] - 1e-12, ["risk", "return"]]
    return efficient.drop_duplicates().reset_index(drop=True)


def distance_to_frontier(portfolio_return: float, portfolio_risk: float, frontier: pd.DataFrame) -> float:
    if frontier.empty:
        return float("nan")
    risk_scale = max(float(frontier["risk"].max() - frontier["risk"].min()), 1e-12)
    return_scale = max(float(frontier["return"].max() - frontier["return"].min()), 1e-12)
    distances = np.sqrt(
        np.square((float(portfolio_risk) - frontier["risk"].to_numpy(dtype=float)) / risk_scale)
        + np.square((float(portfolio_return) - frontier["return"].to_numpy(dtype=float)) / return_scale)
    )
    return float(np.min(distances))


def compute_frontier_metrics(
    portfolios: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    columns: Mapping[str, str],
    cfg: Mapping[str, Any],
    normalize_weights: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Markowitz optimal-proximity approximation for generated portfolios."""

    if portfolios.empty or returns.empty:
        empty = pd.DataFrame(columns=frontier_by_sample_columns())
        return empty, _empty_frontier_summary()

    date_col = columns.get("date", "date")
    asset_col = columns.get("asset_id", "asset_id")
    fund_col = columns.get("fund_id", "fund_id")
    split_col = columns.get("split", "split")
    model_col = columns.get("model_name", "model_name")
    run_col = columns.get("run_id", "run_id")
    source_col = columns.get("prediction_source", "prediction_source")
    weight_col = str(cfg.get("weight_column", columns.get("w_pred", "w_pred")))
    ret_col = str(cfg.get("return_column", "ret"))
    style_col = str(cfg.get("style_label_column", columns.get("label", "style_label")))

    required_portfolio = [date_col, asset_col, fund_col, weight_col]
    missing_portfolio = [col for col in required_portfolio if col not in portfolios.columns]
    required_returns = [date_col, asset_col, ret_col]
    missing_returns = [col for col in required_returns if col not in returns.columns]
    if missing_portfolio:
        raise ValueError(f"Frontier metrics missing portfolio columns: {missing_portfolio}")
    if missing_returns:
        raise ValueError(f"Frontier metrics missing return columns: {missing_returns}")

    p = portfolios.copy()
    r = returns.copy()
    for optional, default in [(model_col, "portfolio_gan"), (run_col, "unknown"), (source_col, "model"), (split_col, "")]:
        if optional not in p.columns:
            p[optional] = default
    p[date_col] = pd.to_datetime(p[date_col], errors="coerce")
    r[date_col] = pd.to_datetime(r[date_col], errors="coerce")
    r[ret_col] = pd.to_numeric(r[ret_col], errors="coerce")
    r = r.dropna(subset=[date_col, asset_col, ret_col])

    lookback = int(cfg.get("lookback_periods", 36))
    min_periods = int(cfg.get("min_periods", 3))
    shrinkage = float(cfg.get("covariance_shrinkage", 0.0))
    num_random = int(cfg.get("num_random_portfolios", 1000))
    random_reference = int(cfg.get("num_random_reference_portfolios", min(200, max(20, num_random // 5))))
    long_only = bool(cfg.get("long_only", True))
    threshold = float(cfg.get("holding_threshold", 1e-4))
    random_seed = int(cfg.get("random_seed", 42))

    rows: list[dict[str, Any]] = []
    returns_by_date = {
        pd.Timestamp(date): group[[asset_col, ret_col]].copy()
        for date, group in r.groupby(date_col, dropna=True, sort=True)
    }
    sorted_return_dates = sorted(returns_by_date.keys())
    date_groups = p.dropna(subset=[date_col]).groupby(date_col, sort=True)

    for date, date_portfolios in date_groups:
        window_dates = [d for d in sorted_return_dates if d <= pd.Timestamp(date)]
        if lookback > 0:
            window_dates = window_dates[-lookback:]
        if len(window_dates) < min_periods:
            continue
        window = pd.concat([returns_by_date[d].assign(_date=d) for d in window_dates], ignore_index=True)
        pivot = window.pivot_table(index="_date", columns=asset_col, values=ret_col, aggfunc="mean").sort_index()
        pivot = pivot.dropna(axis=1, how="all").fillna(0.0)
        if pivot.shape[1] == 0:
            continue

        assets = pivot.columns.astype(str).tolist()
        expected = pivot.mean(axis=0).to_numpy(dtype=float)
        cov = _covariance_matrix(pivot.to_numpy(dtype=float), shrinkage=shrinkage)
        frontier = build_efficient_frontier(
            expected,
            cov,
            num_random_portfolios=num_random,
            long_only=long_only,
            random_seed=random_seed + int(pd.Timestamp(date).strftime("%Y%m%d")),
        )
        if frontier.empty:
            continue

        sample_group_cols = [model_col, run_col, source_col, split_col, fund_col]
        for key, sample in date_portfolios.groupby(sample_group_cols, dropna=False, sort=True):
            weight_series = (
                sample.assign(_asset=sample[asset_col].astype(str))
                .groupby("_asset", dropna=False)[weight_col]
                .sum()
                .reindex(assets, fill_value=0.0)
            )
            weights = pd.to_numeric(weight_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if normalize_weights:
                weights = normalize_weight_vector(weights)
            if np.isclose(np.abs(weights).sum(), 0.0):
                continue

            port_return = estimate_portfolio_return(weights, expected)
            port_risk = estimate_portfolio_risk(weights, cov)
            distance = distance_to_frontier(port_return, port_risk, frontier)
            reference_distances, reference_type = _random_reference_distances(
                weights=weights,
                expected=expected,
                cov=cov,
                frontier=frontier,
                all_assets=np.array(assets, dtype=object),
                date_portfolios=date_portfolios,
                asset_col=asset_col,
                weight_col=weight_col,
                style_col=style_col if style_col in date_portfolios.columns else None,
                sample_style=_sample_style(sample, style_col),
                n_samples=random_reference,
                threshold=threshold,
                seed=random_seed + len(rows),
            )
            meta = dict(zip(sample_group_cols, key, strict=False))
            rows.append(
                {
                    "model_name": meta[model_col],
                    "run_id": meta[run_col],
                    "prediction_source": meta[source_col],
                    "split": meta[split_col],
                    "fund_id": meta[fund_col],
                    "date": date,
                    "portfolio_return": port_return,
                    "portfolio_risk": port_risk,
                    "frontier_distance": distance,
                    "random_reference_distance_mean": float(np.nanmean(reference_distances))
                    if reference_distances.size
                    else np.nan,
                    "random_reference_distance_median": float(np.nanmedian(reference_distances))
                    if reference_distances.size
                    else np.nan,
                    "random_reference_type": reference_type,
                    "status": FRONTIER_STATUS,
                }
            )

    by_sample = pd.DataFrame(rows, columns=frontier_by_sample_columns())
    summary = summarize_metric_columns(
        by_sample,
        ["frontier_distance", "random_reference_distance_mean"],
        status_by_metric={"frontier_distance": FRONTIER_STATUS, "random_reference_distance_mean": FRONTIER_STATUS},
    )
    return by_sample, summary


def _covariance_matrix(values: np.ndarray, *, shrinkage: float) -> np.ndarray:
    if values.shape[0] < 2:
        cov = np.zeros((values.shape[1], values.shape[1]), dtype=float)
    elif values.shape[1] == 1:
        cov = np.array([[float(np.var(values[:, 0], ddof=1)) if values.shape[0] > 1 else 0.0]])
    else:
        cov = np.cov(values, rowvar=False)
        cov = np.asarray(cov, dtype=float)
    shrinkage = min(max(float(shrinkage), 0.0), 1.0)
    if shrinkage > 0:
        diag = np.diag(np.diag(cov))
        cov = (1.0 - shrinkage) * cov + shrinkage * diag
    return np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)


def _random_reference_distances(
    *,
    weights: np.ndarray,
    expected: np.ndarray,
    cov: np.ndarray,
    frontier: pd.DataFrame,
    all_assets: np.ndarray,
    date_portfolios: pd.DataFrame,
    asset_col: str,
    weight_col: str,
    style_col: str | None,
    sample_style: Any,
    n_samples: int,
    threshold: float,
    seed: int,
) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    n_assets = len(all_assets)
    if n_assets == 0 or n_samples <= 0:
        return np.array([], dtype=float), "none"
    support_count = int(max(1, min(n_assets, holding_count(weights, threshold))))
    candidate_indices = np.arange(n_assets)
    reference_type = "count_matched_random"
    if style_col is not None and sample_style is not None and not pd.isna(sample_style):
        same_style = date_portfolios.loc[date_portfolios[style_col] == sample_style]
        style_assets = same_style.loc[pd.to_numeric(same_style[weight_col], errors="coerce").fillna(0.0) > threshold, asset_col]
        style_assets = style_assets.astype(str).unique()
        style_indices = np.where(np.isin(all_assets.astype(str), style_assets.astype(str)))[0]
        if style_indices.size >= support_count:
            candidate_indices = style_indices
            reference_type = "style_matched_random"

    distances: list[float] = []
    for _ in range(n_samples):
        support = rng.choice(candidate_indices, size=min(support_count, candidate_indices.size), replace=False)
        random_weights = np.zeros(n_assets, dtype=float)
        random_weights[support] = rng.dirichlet(np.ones(len(support)))
        ret = estimate_portfolio_return(random_weights, expected)
        risk = estimate_portfolio_risk(random_weights, cov)
        distances.append(distance_to_frontier(ret, risk, frontier))
    return np.asarray(distances, dtype=float), reference_type


def _sample_style(sample: pd.DataFrame, style_col: str) -> Any:
    if style_col not in sample.columns:
        return None
    non_null = sample[style_col].dropna()
    return non_null.iloc[0] if not non_null.empty else None


def _empty_frontier_summary() -> pd.DataFrame:
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
