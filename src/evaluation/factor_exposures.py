from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, save_parquet


CARHART_BETA_STATUS = "CLOSE"
DEFAULT_OUTPUT_COLUMNS = {
    "mktrf": "market_beta",
    "smb": "SMB",
    "hml": "HML",
    "umd": "UMD",
}


def build_carhart_factor_exposures(
    stock_returns: pd.DataFrame,
    carhart_factors: pd.DataFrame,
    *,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Estimate rolling Carhart factor betas for each asset-month.

    The paper asks for factor tilts `w'X_t`; the scaffold does not ship
    precomputed asset-level Carhart exposures, so this module estimates them
    from available return and factor panels. This is tagged CLOSE because
    rolling-window choices are configurable approximations.
    """

    if stock_returns.empty or carhart_factors.empty:
        return _empty_exposures()

    asset_col = str(cfg.get("asset_id_column", "permno"))
    date_col = str(cfg.get("date_column", "date"))
    return_col = str(cfg.get("return_column", "ret"))
    rf_col = str(cfg.get("risk_free_column", "rf"))
    factor_map = dict(cfg.get("factor_column_map", DEFAULT_OUTPUT_COLUMNS))
    factor_cols = list(factor_map.keys())

    missing_returns = [col for col in [asset_col, date_col, return_col] if col not in stock_returns.columns]
    missing_factors = [col for col in [date_col, rf_col] + factor_cols if col not in carhart_factors.columns]
    if missing_returns:
        raise ValueError(f"Carhart beta estimation missing stock return columns: {missing_returns}")
    if missing_factors:
        raise ValueError(f"Carhart beta estimation missing factor columns: {missing_factors}")

    returns = stock_returns[[asset_col, date_col, return_col]].copy()
    returns = returns.rename(columns={asset_col: "asset_id", date_col: "date", return_col: "ret"})
    returns["asset_id"] = returns["asset_id"].astype(str)
    returns["date"] = _to_month_end(returns["date"])
    returns["ret"] = pd.to_numeric(returns["ret"], errors="coerce")
    returns = returns.dropna(subset=["asset_id", "date", "ret"])

    factors = carhart_factors[[date_col, rf_col] + factor_cols].copy()
    factors = factors.rename(columns={date_col: "date", rf_col: "rf"})
    factors["date"] = _to_month_end(factors["date"])
    for col in ["rf"] + factor_cols:
        factors[col] = pd.to_numeric(factors[col], errors="coerce")
    factors = factors.dropna(subset=["date"] + factor_cols)
    factors = factors.drop_duplicates(subset=["date"], keep="last")

    merged = returns.merge(factors, on="date", how="inner")
    if merged.empty:
        return _empty_exposures(factor_map.values())
    merged["excess_ret"] = merged["ret"] - merged["rf"].fillna(0.0)
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["excess_ret"] + factor_cols)
    if merged.empty:
        return _empty_exposures(factor_map.values())

    lookback = int(cfg.get("lookback_periods", 36))
    min_periods = int(cfg.get("min_periods", 24))
    ridge = float(cfg.get("ridge_penalty", 1e-8))
    include_intercept = bool(cfg.get("include_intercept", True))
    max_assets = int(cfg.get("max_assets", 0))
    if max_assets > 0:
        keep_assets = merged["asset_id"].drop_duplicates().head(max_assets)
        merged = merged.loc[merged["asset_id"].isin(keep_assets)].copy()

    rows: list[pd.DataFrame] = []
    for asset_id, group in merged.sort_values(["asset_id", "date"]).groupby("asset_id", sort=False):
        exposures = _rolling_betas_for_asset(
            group,
            factor_cols=factor_cols,
            output_map=factor_map,
            lookback=lookback,
            min_periods=min_periods,
            ridge=ridge,
            include_intercept=include_intercept,
        )
        if not exposures.empty:
            exposures.insert(0, "asset_id", asset_id)
            rows.append(exposures)

    if not rows:
        return _empty_exposures(factor_map.values())
    result = pd.concat(rows, ignore_index=True)
    result["status"] = CARHART_BETA_STATUS
    return result


def build_or_load_carhart_factor_exposures(
    *,
    project_root: Path,
    stock_returns: pd.DataFrame,
    carhart_factors: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Load cached beta estimates or build them from raw return/factor panels."""

    output_path = _resolve_path(project_root, str(cfg.get("output_path", "artifacts/evaluation/carhart_betas.parquet")))
    if bool(cfg.get("use_cache", True)) and output_path.exists():
        return pd.read_parquet(output_path)

    exposures = build_carhart_factor_exposures(stock_returns, carhart_factors, cfg=cfg)
    if not exposures.empty:
        ensure_dir(output_path.parent)
        save_parquet(exposures, output_path)
    return exposures


def _rolling_betas_for_asset(
    group: pd.DataFrame,
    *,
    factor_cols: Sequence[str],
    output_map: Mapping[str, str],
    lookback: int,
    min_periods: int,
    ridge: float,
    include_intercept: bool,
) -> pd.DataFrame:
    dates = group["date"].to_numpy()
    y = group["excess_ret"].to_numpy(dtype=float)
    x = group[list(factor_cols)].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for idx in range(group.shape[0]):
        start = max(0, idx - lookback + 1) if lookback > 0 else 0
        x_window = x[start : idx + 1]
        y_window = y[start : idx + 1]
        valid = np.isfinite(y_window) & np.isfinite(x_window).all(axis=1)
        if int(valid.sum()) < min_periods:
            continue
        x_valid = x_window[valid]
        y_valid = y_window[valid]
        design = np.column_stack([np.ones(x_valid.shape[0]), x_valid]) if include_intercept else x_valid
        betas = _ridge_lstsq(design, y_valid, ridge=ridge)
        slopes = betas[1:] if include_intercept else betas
        row: dict[str, Any] = {"date": pd.Timestamp(dates[idx])}
        row.update({output_map[factor]: float(value) for factor, value in zip(factor_cols, slopes, strict=False)})
        rows.append(row)
    columns = ["date"] + [output_map[factor] for factor in factor_cols]
    return pd.DataFrame(rows, columns=columns)


def _ridge_lstsq(x: np.ndarray, y: np.ndarray, *, ridge: float) -> np.ndarray:
    ridge = max(float(ridge), 0.0)
    xtx = x.T @ x
    if ridge > 0:
        penalty = np.eye(xtx.shape[0]) * ridge
        penalty[0, 0] = 0.0
        xtx = xtx + penalty
    xty = x.T @ y
    try:
        return np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(x, y, rcond=None)[0]


def _to_month_end(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def _empty_exposures(factor_columns: Sequence[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=["asset_id", "date"] + list(factor_columns or DEFAULT_OUTPUT_COLUMNS.values()) + ["status"])


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path
