from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import save_csv, save_parquet


@dataclass
class SampleOutputs:
    """Container for sample-construction outputs saved to derived datasets."""

    fund_sample: pd.DataFrame
    holdings_panel: pd.DataFrame
    stock_universe_panel: pd.DataFrame


def _to_percent(series: pd.Series) -> pd.Series:
    """Normalize ratio-like or percent-like series into percentage scale [0, 100+]."""
    clean = pd.to_numeric(series, errors="coerce")
    if clean.dropna().empty:
        return clean
    if clean.quantile(0.95) <= 1.5:
        return clean * 100.0
    return clean


def construct_sample_panels(
    raw_frames: dict[str, pd.DataFrame],
    data_cfg: dict,
    derived_dir: Path,
    diagnostics_dir: Path,
) -> SampleOutputs:
    """Construct fund sample, holdings panel, and stock-universe panel.

    This is a scaffold implementation and should be upgraded once CRSP
    field mappings are finalized.
    """

    holdings = raw_frames.get("holdings_file", pd.DataFrame()).copy()
    metadata = raw_frames.get("fund_meta_file", pd.DataFrame()).copy()
    market_cap = raw_frames.get("market_cap_file", pd.DataFrame()).copy()
    eligible_fund_months = raw_frames.get("eligible_fund_months_file", pd.DataFrame()).copy()

    filters = data_cfg.get("filters", {})

    required_holdings = {"fund_id", "stock_id", "date"}
    required_market_cap = {"stock_id", "date", "mkt_cap"}
    can_run_full = (
        not holdings.empty
        and not metadata.empty
        and not market_cap.empty
        and required_holdings.issubset(set(holdings.columns))
        and {"fund_id"}.issubset(set(metadata.columns))
        and required_market_cap.issubset(set(market_cap.columns))
    )

    if can_run_full:
        fund_id_col = data_cfg.get("identifiers", {}).get("fund_id", "fund_id")
        date_col = data_cfg.get("identifiers", {}).get("date", "date")
        stock_col = data_cfg.get("identifiers", {}).get("stock_id", "stock_id")

        holdings[date_col] = pd.to_datetime(holdings[date_col])
        market_cap[date_col] = pd.to_datetime(market_cap[date_col])

        top_n = int(filters.get("target_universe_size", 500))
        universe = (
            market_cap.sort_values([date_col, "mkt_cap"], ascending=[True, False])
            .groupby(date_col)
            .head(top_n)
            .assign(in_universe=True)
        )

        holdings_panel = holdings.merge(
            universe[[date_col, stock_col, "in_universe"]],
            on=[date_col, stock_col],
            how="left",
        )
        holdings_panel["in_universe"] = holdings_panel["in_universe"].fillna(False)

        fund_sample = metadata[[fund_id_col]].drop_duplicates().rename(columns={fund_id_col: "fund_id"})
        fund_sample["status"] = "scaffold_only"
        fund_sample["reason"] = "Full CRSP field mapping pending"

        stock_universe_panel = universe.rename(columns={date_col: "date", stock_col: "stock_id"})
        holdings_panel = holdings_panel.rename(columns={fund_id_col: "fund_id", stock_col: "stock_id", date_col: "date"})
    elif not eligible_fund_months.empty and {
        "crsp_fundno",
        "report_dt",
        "total_reported_weight",
        "weight_in_top500",
    }.issubset(set(eligible_fund_months.columns)):
        eligible_fund_months["report_dt"] = pd.to_datetime(eligible_fund_months["report_dt"])
        sample_start = pd.to_datetime(data_cfg.get("sample_start", "2010-01-01"))
        sample_end = pd.to_datetime(data_cfg.get("sample_end", "2024-12-31"))
        monthly = eligible_fund_months[
            (eligible_fund_months["report_dt"] >= sample_start)
            & (eligible_fund_months["report_dt"] <= sample_end)
        ].copy()

        monthly["coverage_pct"] = _to_percent(monthly["total_reported_weight"])
        monthly["in_top500_pct"] = _to_percent(monthly["weight_in_top500"])

        min_months = int(filters.get("min_months_holdings", 12))
        min_cov = float(filters.get("min_reported_weight_coverage", 0.75)) * 100.0
        min_univ = float(filters.get("min_in_universe_weight_share", 0.75)) * 100.0

        monthly["rule_weight_coverage"] = monthly["coverage_pct"] >= min_cov
        monthly["rule_in_top500"] = monthly["in_top500_pct"] >= min_univ
        monthly["eligible_observation"] = monthly["rule_weight_coverage"] & monthly["rule_in_top500"]

        grouped_total = (
            monthly.groupby("crsp_fundno", dropna=True)
            .agg(
                n_months_total=("report_dt", "nunique"),
                avg_total_reported_weight=("coverage_pct", "mean"),
                avg_weight_in_top500=("in_top500_pct", "mean"),
                first_obs=("report_dt", "min"),
                last_obs=("report_dt", "max"),
            )
            .reset_index()
        )
        grouped_pass = (
            monthly.loc[monthly["eligible_observation"]]
            .groupby("crsp_fundno", dropna=True)
            .agg(
                n_months_eligible=("report_dt", "nunique"),
            )
            .reset_index()
        )
        grouped = grouped_total.merge(grouped_pass, on="crsp_fundno", how="left")
        grouped["n_months_eligible"] = grouped["n_months_eligible"].fillna(0).astype(int)
        grouped["rule_min_months"] = grouped["n_months_eligible"] >= min_months
        grouped["eligible"] = grouped["rule_min_months"]
        grouped["status"] = "close_from_eligible_fund_months"
        grouped["reason"] = "Observation-level 75% filters applied; stock-level holdings still missing"

        fund_sample = grouped.rename(columns={"crsp_fundno": "fund_id"})
        holdings_panel = monthly.loc[monthly["eligible_observation"]].rename(
            columns={
                "crsp_fundno": "fund_id",
                "report_dt": "date",
                "coverage_pct": "reported_weight",
                "in_top500_pct": "weight_in_top500_pct",
            }
        )
        eligible_funds = set(fund_sample.loc[fund_sample["eligible"], "fund_id"].tolist())
        holdings_panel = holdings_panel.loc[holdings_panel["fund_id"].isin(eligible_funds)].copy()
        stock_universe_panel = pd.DataFrame(columns=["date", "stock_id", "mkt_cap", "in_universe"])
    else:
        fund_sample = pd.DataFrame(columns=["fund_id", "status", "reason"])
        holdings_panel = pd.DataFrame(columns=["fund_id", "stock_id", "date", "weight"])
        stock_universe_panel = pd.DataFrame(columns=["date", "stock_id", "mkt_cap", "in_universe"])

    save_parquet(fund_sample, derived_dir / "fund_sample.parquet")
    save_parquet(holdings_panel, derived_dir / "holdings_panel.parquet")
    save_parquet(stock_universe_panel, derived_dir / "stock_universe_panel.parquet")

    summary = pd.DataFrame(
        {
            "metric": [
                "n_funds",
                "n_eligible_funds",
                "n_holdings_rows",
                "n_universe_rows",
                "min_months_holdings_rule",
                "coverage_rule",
                "in_universe_rule",
            ],
            "value": [
                len(fund_sample),
                int(fund_sample["eligible"].sum()) if "eligible" in fund_sample.columns else 0,
                len(holdings_panel),
                len(stock_universe_panel),
                filters.get("min_months_holdings", 12),
                filters.get("min_reported_weight_coverage", 0.75),
                filters.get("min_in_universe_weight_share", 0.75),
            ],
        }
    )
    save_csv(summary, diagnostics_dir / "sample_summary.csv")

    return SampleOutputs(
        fund_sample=fund_sample,
        holdings_panel=holdings_panel,
        stock_universe_panel=stock_universe_panel,
    )
