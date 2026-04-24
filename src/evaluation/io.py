from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, save_csv, save_parquet


STANDARD_COLUMNS = {
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
    "case_id": "case_id",
    "w_original": "w_original",
    "w_transferred": "w_transferred",
    "w_prev_transferred": "w_prev_transferred",
}


COLUMN_ALIASES = {
    "fund_id": ["fund_id", "crsp_fundno", "source_fund_id"],
    "date": ["date", "report_dt", "month", "source_date"],
    "asset_id": ["asset_id", "stock_id", "permno", "asset_idx"],
    "w_true": ["w_true", "true_weight", "weight_true", "w_t", "weight"],
    "w_pred": ["w_pred", "w_hat", "pred_weight", "predicted_weight", "generated_weight", "baseline_weight"],
    "w_prev": ["w_prev", "previous_weight", "prev_weight", "weight_prev"],
    "label": ["style_label", "label", "fund_style", "lipper_class", "objective"],
    "model_name": ["model_name", "model", "baseline_name"],
    "run_id": ["run_id", "experiment_id"],
    "split": ["split", "eval_split"],
    "prediction_source": ["prediction_source", "source"],
    "case_id": ["case_id", "transfer_id"],
    "w_original": ["w_original", "original_weight", "source_weight"],
    "w_transferred": ["w_transferred", "transferred_weight", "target_weight"],
    "w_prev_transferred": ["w_prev_transferred", "target_prev_weight", "w_prev"],
    "ret": ["ret", "return", "asset_return", "realized_return"],
}


def read_optional_table(project_root: Path, configured_path: str | None) -> tuple[pd.DataFrame, Path | None]:
    if not configured_path:
        return pd.DataFrame(), None
    for path in _candidate_paths(project_root, configured_path):
        if path.exists():
            return _read_table(path), path
    return pd.DataFrame(), None


def write_json(payload: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)


def write_markdown(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_table_outputs(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        save_parquet(df, path)
    elif path.suffix == ".csv":
        save_csv(df, path)
    else:
        raise ValueError(f"Unsupported output table suffix: {path}")


def normalize_portfolio_frame(
    df: pd.DataFrame,
    *,
    columns_cfg: dict[str, str],
    defaults: dict[str, str],
) -> pd.DataFrame:
    normalized = _rename_to_standard(df, columns_cfg=columns_cfg)
    for col, value in defaults.items():
        if col not in normalized.columns:
            normalized[col] = value
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    if "asset_id" in normalized.columns:
        normalized["asset_id"] = normalized["asset_id"].astype(str)
    return normalized


def normalize_return_frame(df: pd.DataFrame, *, columns_cfg: dict[str, str]) -> pd.DataFrame:
    normalized = _rename_to_standard(df, columns_cfg=columns_cfg)
    ret_col = _resolve_column(normalized, columns_cfg.get("return", "ret"), COLUMN_ALIASES["ret"])
    if ret_col and ret_col != "ret":
        normalized = normalized.rename(columns={ret_col: "ret"})
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    if "asset_id" in normalized.columns:
        normalized["asset_id"] = normalized["asset_id"].astype(str)
    return normalized


def normalize_factor_frame(
    df: pd.DataFrame,
    *,
    columns_cfg: dict[str, str],
    factor_columns: list[str],
    factor_aliases: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    normalized = _rename_to_standard(df, columns_cfg=columns_cfg)
    factor_aliases = factor_aliases or {}
    renames: dict[str, str] = {}
    for factor in factor_columns:
        if factor in normalized.columns:
            continue
        candidates = factor_aliases.get(factor, []) + _default_factor_aliases(factor)
        resolved = _resolve_column(normalized, factor, candidates)
        if resolved and resolved != factor:
            renames[resolved] = factor
    if renames:
        normalized = normalized.rename(columns=renames)
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    if "asset_id" in normalized.columns:
        normalized["asset_id"] = normalized["asset_id"].astype(str)
    return normalized


def merge_factor_exposures(
    portfolios: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    factor_columns: list[str],
) -> pd.DataFrame:
    if portfolios.empty:
        return portfolios
    missing_factors = [factor for factor in factor_columns if factor not in portfolios.columns]
    if not missing_factors or factors.empty:
        return portfolios
    if not {"date", "asset_id"}.issubset(portfolios.columns) or not {"date", "asset_id"}.issubset(factors.columns):
        return portfolios
    available = [factor for factor in missing_factors if factor in factors.columns]
    if not available:
        return portfolios
    factor_subset = factors[["date", "asset_id"] + available].drop_duplicates(subset=["date", "asset_id"])
    return portfolios.merge(factor_subset, on=["date", "asset_id"], how="left")


def _rename_to_standard(df: pd.DataFrame, *, columns_cfg: dict[str, str]) -> pd.DataFrame:
    normalized = df.copy()
    renames: dict[str, str] = {}
    for field, default_col in STANDARD_COLUMNS.items():
        target = default_col
        configured = columns_cfg.get(field, default_col)
        resolved = _resolve_column(normalized, configured, COLUMN_ALIASES.get(field, [default_col]))
        if resolved and resolved != target and target not in normalized.columns:
            renames[resolved] = target
    if renames:
        normalized = normalized.rename(columns=renames)
    return normalized


def _resolve_column(df: pd.DataFrame, configured: str, candidates: list[str]) -> str | None:
    if configured in df.columns:
        return configured
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lower_lookup = {str(col).lower(): col for col in df.columns}
    if configured.lower() in lower_lookup:
        return str(lower_lookup[configured.lower()])
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return str(lower_lookup[candidate.lower()])
    return None


def _candidate_paths(project_root: Path, configured_path: str) -> list[Path]:
    rel = Path(configured_path)
    candidates = [rel if rel.is_absolute() else project_root / rel]
    if not rel.is_absolute():
        candidates.append(project_root / rel.name)

    def add_variants(path: Path) -> None:
        if path.suffix == ".parquet":
            csv_path = path.with_suffix(".csv")
            candidates.extend([csv_path, Path(f"{csv_path}.gz"), Path(f"{csv_path}.zip")])
        elif path.suffix == ".csv":
            candidates.extend([path.with_suffix(".parquet"), Path(f"{path}.gz"), Path(f"{path}.zip")])

    base_candidates = list(candidates)
    for candidate in base_candidates:
        add_variants(candidate)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if path.name.endswith(".csv.zip"):
        with zipfile.ZipFile(path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv")
                and not name.startswith("__MACOSX/")
                and not Path(name).name.startswith("._")
            ]
            if not members:
                raise ValueError(f"No CSV member found in {path}")
            with archive.open(sorted(members)[0]) as handle:
                return pd.read_csv(handle)
    return pd.DataFrame()


def _default_factor_aliases(factor: str) -> list[str]:
    lookup = {
        "market_beta": ["mkt_beta", "MKT", "mkt", "mktrf", "Mkt-RF", "beta_mkt"],
        "SMB": ["smb"],
        "HML": ["hml"],
        "UMD": ["umd", "mom", "momentum"],
    }
    return lookup.get(factor, [factor.lower(), factor.upper()])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
