from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import save_csv


def _row(
    canonical_variable: str,
    description: str,
    source_dataset: str,
    source_column: str,
    transform: str,
    label: str,
    status: str,
    notes: str,
) -> dict[str, str]:
    return {
        "canonical_variable": canonical_variable,
        "description": description,
        "source_dataset": source_dataset,
        "source_column": source_column,
        "transform": transform,
        "label": label,
        "status": status,
        "notes": notes,
    }


def build_variable_crosswalk(raw_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    eligible = raw_frames.get("eligible_fund_months_file", pd.DataFrame())
    cols = set(eligible.columns)

    def add_if_exists(
        canonical_variable: str,
        description: str,
        source_column: str,
        transform: str,
        label: str,
        notes: str,
    ) -> None:
        if source_column in cols:
            rows.append(
                _row(
                    canonical_variable=canonical_variable,
                    description=description,
                    source_dataset="eligible_fund_months_file",
                    source_column=source_column,
                    transform=transform,
                    label=label,
                    status="mapped",
                    notes=notes,
                )
            )

    add_if_exists("fund_id", "Fund identifier", "crsp_fundno", "rename -> fund_id", "CLOSE", "Share-class aggregation still unresolved")
    add_if_exists("portfolio_id", "Portfolio identifier", "crsp_portno", "rename -> portfolio_id", "CLOSE", "Portfolio to fund mapping assumptions pending")
    add_if_exists("date", "Holdings report date", "report_dt", "to_datetime", "CLOSE", "Monthly timestamp alignment to returns pending")
    add_if_exists("lipper_class", "Lipper class code", "lipper_class", "as-is", "CLOSE", "Class cleaning for probe not finalized")
    add_if_exists("lipper_class_name", "Lipper class name", "lipper_class_name", "as-is", "CLOSE", "Potentially multi-version naming across years")
    add_if_exists(
        "reported_weight_coverage",
        "Portfolio weight coverage metric",
        "total_reported_weight",
        "scale check (0-1 or 0-100)",
        "CLOSE",
        "Observed values suggest percentage-like scale with outliers >100",
    )
    add_if_exists(
        "in_universe_weight_share",
        "Weight allocated in top-500 universe",
        "weight_in_top500",
        "scale check (0-1 or 0-100)",
        "CLOSE",
        "Universe definition source still indirect",
    )
    add_if_exists("n_holdings", "Number of holdings", "n_holdings", "as-is", "CLOSE", "Holdings table unavailable for exact reconciliation")
    add_if_exists("n_distinct_stocks", "Distinct stock identifiers count", "n_permnos", "as-is", "CLOSE", "Exact stock panel unavailable")

    missing_targets = [
        ("stock_id", "Security identifier", "PROXY", "Missing stock-level holdings file"),
        ("weight_t", "Current portfolio weight by stock", "PROXY", "Missing holdings-by-stock table"),
        ("weight_t_minus_1", "Lagged portfolio weight by stock", "PROXY", "Requires holdings-by-stock time panel"),
        ("ret_1m", "Stock return", "PROXY", "Missing stock returns panel"),
        ("mkt_cap", "Market cap", "PROXY", "Missing market cap panel"),
        ("factor_mkt_smb_hml_umd", "Carhart factors", "PROXY", "Missing factor table"),
        ("stock_characteristics", "Characteristics matrix X", "PROXY", "Missing characteristics panel"),
    ]
    for canonical_variable, description, label, notes in missing_targets:
        rows.append(
            _row(
                canonical_variable=canonical_variable,
                description=description,
                source_dataset="",
                source_column="",
                transform="",
                label=label,
                status="missing",
                notes=notes,
            )
        )

    return pd.DataFrame(rows)


def write_crosswalk_outputs(crosswalk_df: pd.DataFrame, derived_dir: Path, docs_dir: Path) -> None:
    save_csv(crosswalk_df, derived_dir / "variable_crosswalk.csv")

    lines = ["# Variable Crosswalk", "", "Auto-generated from currently available files.", ""]
    lines.append("| canonical_variable | description | source_dataset | source_column | transform | label | status | notes |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in crosswalk_df.itertuples(index=False):
        lines.append(
            "| {a} | {b} | {c} | {d} | {e} | {f} | {g} | {h} |".format(
                a=row.canonical_variable,
                b=row.description,
                c=row.source_dataset,
                d=row.source_column,
                e=row.transform,
                f=row.label,
                g=row.status,
                h=row.notes,
            )
        )

    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "variable_crosswalk.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

