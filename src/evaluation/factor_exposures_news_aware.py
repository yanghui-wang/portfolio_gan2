"""Load news-aware factor exposures (Carhart 4-factor + 5 orthogonalized news factors = 9 factors)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.factor_exposures import build_or_load_carhart_factor_exposures
from src.evaluation.io import read_optional_table


# Default 4 Carhart + 5 news factors
CARHART_FACTORS = ["market_beta", "SMB", "HML", "UMD"]
NEWS_FACTORS = [
    "ortho_sentiment",
    "ortho_risk",
    "ortho_uncertainty",
    "ortho_macro_credit_pressure",
    "ortho_corporate_market_activity",
]
ALL_FACTORS = CARHART_FACTORS + NEWS_FACTORS


def build_news_aware_factor_exposures(
    project_root: Path,
    stock_returns: pd.DataFrame,
    carhart_factors: pd.DataFrame,
    news_factors: pd.DataFrame,
    *,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Combine Carhart 4-factor rolling betas with monthly news factors.
    
    News factors are broadcasted to all assets in each month.
    """

    # Build/load Carhart rolling betas
    carhart_betas = build_or_load_carhart_factor_exposures(
        project_root=project_root,
        stock_returns=stock_returns,
        carhart_factors=carhart_factors,
        cfg=cfg,
    )

    if carhart_betas.empty:
        return pd.DataFrame()

    # Normalize news factors to asset-month granularity
    carhart_betas["month"] = (
        pd.to_datetime(carhart_betas["date"], errors="coerce")
        .dt.strftime("%Y-%m")
    )
    
    news_factors_norm = news_factors[["month"] + NEWS_FACTORS].copy()
    news_factors_norm["month"] = pd.to_datetime(news_factors_norm["month"], format="%Y-%m", errors="coerce").dt.strftime("%Y-%m")
    
    # Merge Carhart betas + news factors by month
    merged = pd.merge(
        carhart_betas,
        news_factors_norm,
        on="month",
        how="left",
    )

    missing_news = [f for f in NEWS_FACTORS if f not in merged.columns]
    if missing_news:
        raise ValueError(f"Missing news factor columns after merge: {missing_news}")

    # Ensure every asset-month has mapped news factors.
    null_news = merged[NEWS_FACTORS].isna().any(axis=1)
    if bool(null_news.any()):
        missing_months = sorted(merged.loc[null_news, "month"].dropna().astype(str).unique().tolist())
        preview = ", ".join(missing_months[:6])
        suffix = "..." if len(missing_months) > 6 else ""
        raise ValueError(
            f"News factors missing for some asset-month rows; missing months: {preview}{suffix}"
        )

    # Drop temporary month column if it came from merge
    result = merged.copy()
    result = result[["asset_id", "date"] + CARHART_FACTORS + NEWS_FACTORS + ["status"]]
    
    return result


def build_or_load_news_aware_factor_exposures(
    *,
    project_root: Path,
    stock_returns: pd.DataFrame,
    carhart_factors: pd.DataFrame,
    news_factors_path: str | None,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Load or build news-aware 9-factor exposures."""

    output_path = Path(
        cfg.get("output_path_news_aware")
        or cfg.get("output_path")
        or "artifacts/evaluation/news_aware_betas.parquet"
    )
    if output_path.is_absolute():
        output_full = output_path
    else:
        output_full = project_root / output_path

    # Try to load from cache
    if bool(cfg.get("use_cache", True)) and output_full.exists():
        return pd.read_parquet(output_full)

    # Load news factors
    if not news_factors_path:
        raise ValueError("news_factors_path is required for news_aware model")
    
    news_factors, _ = read_optional_table(project_root, news_factors_path)
    if news_factors.empty:
        raise ValueError(f"Failed to load news factors from {news_factors_path}")

    # Build combined exposures
    exposures = build_news_aware_factor_exposures(
        project_root,
        stock_returns,
        carhart_factors,
        news_factors,
        cfg=cfg,
    )

    if not exposures.empty:
        output_full.parent.mkdir(parents=True, exist_ok=True)
        exposures.to_parquet(output_full)

    return exposures
