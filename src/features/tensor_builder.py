from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.io import save_csv, save_parquet


def build_model_input_index(
    holdings_panel: pd.DataFrame,
    derived_dir: Path,
) -> pd.DataFrame:
    """Build sample-level tensor index for downstream training tensors."""

    if holdings_panel.empty:
        index_df = pd.DataFrame(columns=["fund_id", "date", "sample_id"])
    else:
        base_cols = [c for c in ["fund_id", "date"] if c in holdings_panel.columns]
        index_df = holdings_panel[base_cols].drop_duplicates().reset_index(drop=True)
        index_df["sample_id"] = np.arange(len(index_df))
    save_parquet(index_df, derived_dir / "model_input_index.parquet")
    return index_df


def _to_month_end(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def _to_int_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("Int64")


def _safe_float(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype(float)


def _to_percent_like(values: pd.Series) -> pd.Series:
    clean = _safe_float(values)
    if clean.dropna().empty:
        return clean
    if clean.quantile(0.95) <= 1.5:
        return clean * 100.0
    return clean


@dataclass
class DatasetBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    summary: pd.DataFrame


@dataclass
class TensorBatch:
    """Typed batch container for portfolio training tensors."""

    x: torch.Tensor
    r: torch.Tensor
    w_prev: torch.Tensor
    w_t: torch.Tensor


class RealPortfolioDataset(Dataset[dict[str, torch.Tensor]]):
    """Real-data dataset for portfolio GAN training/evaluation."""

    def __init__(
        self,
        x: np.ndarray,
        r: np.ndarray,
        w_prev: np.ndarray,
        w_t: np.ndarray,
        meta: pd.DataFrame,
    ) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.r = torch.from_numpy(r.astype(np.float32))
        self.w_prev = torch.from_numpy(w_prev.astype(np.float32))
        self.w_t = torch.from_numpy(w_t.astype(np.float32))
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "r": self.r[index],
            "w_prev": self.w_prev[index],
            "w_t": self.w_t[index],
        }


def _empty_real_dataset(num_assets: int, num_features: int) -> RealPortfolioDataset:
    return RealPortfolioDataset(
        x=np.zeros((0, num_assets, num_features), dtype=np.float32),
        r=np.zeros((0, num_assets), dtype=np.float32),
        w_prev=np.zeros((0, num_assets), dtype=np.float32),
        w_t=np.zeros((0, num_assets), dtype=np.float32),
        meta=pd.DataFrame(columns=["fund_id", "date", "split"]),
    )


def _select_feature_columns(chars: pd.DataFrame, train_cfg: dict[str, Any], target_features: int) -> list[str]:
    explicit = train_cfg.get("dataset", {}).get("feature_columns", [])
    if explicit:
        cols = [c for c in explicit if c in chars.columns]
    else:
        char_cols = [c for c in chars.columns if c.startswith("char_")]
        base_candidates = [c for c in ["ret", "mkt_cap"] if c in chars.columns]
        cols = base_candidates + char_cols
    if not cols:
        raise ValueError("No usable feature columns found in stock_characteristics")
    unique_cols: list[str] = []
    for col in cols:
        if col not in unique_cols:
            unique_cols.append(col)
    return unique_cols[:target_features]


def _build_universe_panel(
    market_cap: pd.DataFrame,
    num_assets: int,
) -> pd.DataFrame:
    stock_col = "permno" if "permno" in market_cap.columns else "stock_id"
    universe = market_cap[[stock_col, "date", "mkt_cap"]].copy()
    universe[stock_col] = _to_int_series(universe[stock_col])
    universe["date"] = _to_month_end(universe["date"])
    universe["mkt_cap"] = _safe_float(universe["mkt_cap"])
    universe = universe.dropna(subset=[stock_col, "date", "mkt_cap"])
    universe = (
        universe.sort_values(["date", "mkt_cap"], ascending=[True, False])
        .groupby("date", group_keys=False)
        .head(num_assets)
    )
    universe["asset_idx"] = universe.groupby("date").cumcount().astype(int)
    universe = universe.rename(columns={stock_col: "stock_id"})
    return universe[["date", "stock_id", "asset_idx", "mkt_cap"]]


def _build_feature_maps(
    universe_panel: pd.DataFrame,
    stock_chars: pd.DataFrame,
    stock_returns: pd.DataFrame,
    feature_cols: list[str],
    num_assets: int,
) -> tuple[dict[pd.Timestamp, np.ndarray], dict[pd.Timestamp, np.ndarray]]:
    universe_lookup = universe_panel[["date", "stock_id", "asset_idx"]].copy()
    stock_col_chars = "permno" if "permno" in stock_chars.columns else "stock_id"
    stock_col_ret = "permno" if "permno" in stock_returns.columns else "stock_id"

    chars = stock_chars[[stock_col_chars, "date"] + [c for c in feature_cols if c in stock_chars.columns]].copy()
    chars[stock_col_chars] = _to_int_series(chars[stock_col_chars])
    chars["date"] = _to_month_end(chars["date"])
    for col in feature_cols:
        if col not in chars.columns:
            chars[col] = 0.0
        chars[col] = _safe_float(chars[col]).fillna(0.0)

    rets = stock_returns[[stock_col_ret, "date", "ret"]].copy()
    rets[stock_col_ret] = _to_int_series(rets[stock_col_ret])
    rets["date"] = _to_month_end(rets["date"])
    rets["ret"] = _safe_float(rets["ret"]).fillna(0.0)

    char_join = universe_lookup.merge(
        chars.rename(columns={stock_col_chars: "stock_id"}),
        on=["date", "stock_id"],
        how="left",
    )
    for col in feature_cols:
        char_join[col] = _safe_float(char_join[col]).fillna(0.0)

    ret_join = universe_lookup.merge(
        rets.rename(columns={stock_col_ret: "stock_id"}),
        on=["date", "stock_id"],
        how="left",
    )
    ret_join["ret"] = _safe_float(ret_join["ret"]).fillna(0.0)

    x_by_date: dict[pd.Timestamp, np.ndarray] = {}
    r_by_date: dict[pd.Timestamp, np.ndarray] = {}
    for date, sub in char_join.groupby("date"):
        x = np.zeros((num_assets, len(feature_cols)), dtype=np.float32)
        valid = sub["asset_idx"].to_numpy(dtype=int)
        x[valid] = sub[feature_cols].to_numpy(dtype=np.float32)
        x_by_date[pd.Timestamp(date)] = x
    for date, sub in ret_join.groupby("date"):
        r = np.zeros((num_assets,), dtype=np.float32)
        valid = sub["asset_idx"].to_numpy(dtype=int)
        r[valid] = sub["ret"].to_numpy(dtype=np.float32)
        r_by_date[pd.Timestamp(date)] = r

    return x_by_date, r_by_date


def _build_eligible_months(data_cfg: dict[str, Any], eligible_fund_months: pd.DataFrame) -> pd.DataFrame:
    if eligible_fund_months.empty:
        return pd.DataFrame(columns=["fund_id", "crsp_portno", "date", "pass_obs"])

    required = {
        "crsp_fundno",
        "crsp_portno",
        "report_dt",
        "total_reported_weight",
        "weight_in_top500",
    }
    if not required.issubset(set(eligible_fund_months.columns)):
        return pd.DataFrame(columns=["fund_id", "crsp_portno", "date", "pass_obs"])

    efm = eligible_fund_months.copy()
    efm["fund_id"] = _to_int_series(efm["crsp_fundno"])
    efm["crsp_portno"] = _to_int_series(efm["crsp_portno"])
    efm["date"] = _to_month_end(efm["report_dt"])
    efm["total_reported_weight"] = _to_percent_like(efm["total_reported_weight"])
    efm["weight_in_top500"] = _to_percent_like(efm["weight_in_top500"])
    efm = efm.dropna(subset=["fund_id", "crsp_portno", "date"])

    filters = data_cfg.get("filters", {})
    min_cov = float(filters.get("min_reported_weight_coverage", 0.75)) * 100.0
    min_universe = float(filters.get("min_in_universe_weight_share", 0.75)) * 100.0
    min_months = int(filters.get("min_months_holdings", 12))

    efm["pass_obs"] = (efm["total_reported_weight"] >= min_cov) & (efm["weight_in_top500"] >= min_universe)
    fund_counts = efm.groupby("fund_id")["pass_obs"].sum()
    eligible_funds = set(fund_counts[fund_counts >= min_months].index.tolist())
    efm = efm.loc[efm["pass_obs"] & efm["fund_id"].isin(eligible_funds)].copy()
    return efm[["fund_id", "crsp_portno", "date", "pass_obs"]]


def _resolve_sample_caps(train_cfg: dict[str, Any]) -> dict[str, int]:
    mode = train_cfg.get("training_mode", "debug_train")
    defaults = {
        "smoke_test": {"train": 64, "val": 0, "test": 0},
        "debug_train": {"train": 1024, "val": 128, "test": 0},
        "full_train": {"train": 30000, "val": 3000, "test": 0},
        "profile_run": {"train": 128, "val": 0, "test": 0},
    }
    cfg = train_cfg.get("dataset", {})
    base = defaults.get(mode, defaults["debug_train"])
    base_override_source = cfg if mode == "full_train" else {}
    mode_overrides = cfg.get("mode_overrides", {}).get(mode, {}) if isinstance(cfg.get("mode_overrides", {}), dict) else {}

    def _pick(key: str, default_value: int) -> int:
        if key in mode_overrides:
            return int(mode_overrides[key])
        if key in base_override_source:
            return int(base_override_source[key])
        return default_value

    return {
        "train": _pick("max_train_samples", base["train"]),
        "val": _pick("max_val_samples", base["val"]),
        "test": _pick("max_test_samples", base["test"]),
    }


def _split_eligible_months(data_cfg: dict[str, Any], eligible_months: pd.DataFrame, caps: dict[str, int]) -> pd.DataFrame:
    split_cfg = data_cfg.get("split", {})
    train_start = pd.Timestamp(split_cfg.get("train_start", "2010-01-01"))
    train_end = pd.Timestamp(split_cfg.get("train_end", "2018-12-31"))
    val_start = pd.Timestamp(split_cfg.get("val_start", "2019-01-01"))
    val_end = pd.Timestamp(split_cfg.get("val_end", "2019-12-31"))
    test_start = pd.Timestamp(split_cfg.get("test_start", "2020-01-01"))
    test_end = pd.Timestamp(split_cfg.get("test_end", "2024-12-31"))

    key_df = eligible_months[["fund_id", "date"]].drop_duplicates().sort_values(["date", "fund_id"])
    key_df["split"] = ""
    key_df.loc[(key_df["date"] >= train_start) & (key_df["date"] <= train_end), "split"] = "train"
    key_df.loc[(key_df["date"] >= val_start) & (key_df["date"] <= val_end), "split"] = "val"
    key_df.loc[(key_df["date"] >= test_start) & (key_df["date"] <= test_end), "split"] = "test"
    key_df = key_df.loc[key_df["split"] != ""].copy()

    parts: list[pd.DataFrame] = []
    for split_name in ["train", "val", "test"]:
        sub = key_df.loc[key_df["split"] == split_name].copy()
        cap = caps[split_name]
        if cap >= 0:
            sub = sub.head(cap)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def _extract_year_from_path(path: Path) -> int | None:
    match = re.search(r"(20\d{2}|19\d{2})", path.name)
    if not match:
        return None
    return int(match.group(0))


def _discover_holdings_sources(project_root: Path, data_cfg: dict[str, Any]) -> list[Path]:
    def unique_existing(paths: list[Path]) -> list[Path]:
        unique: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            if path.exists():
                unique.append(path)
        return unique

    placeholders = data_cfg.get("placeholders", {})
    configured_path = Path(placeholders.get("holdings_file", "raw/holdings.csv"))

    configured_candidates: list[Path] = [project_root / configured_path, project_root / configured_path.name]
    if configured_path.suffix == ".parquet":
        csv_rel = configured_path.with_suffix(".csv")
        configured_candidates.extend(
            [
                project_root / csv_rel,
                project_root / csv_rel.name,
                project_root / f"{csv_rel}.gz",
                project_root / f"{csv_rel.name}.gz",
                project_root / f"{csv_rel}.zip",
                project_root / f"{csv_rel.name}.zip",
            ]
        )
    if configured_path.suffix == ".csv":
        configured_candidates.extend(
            [
                project_root / f"{configured_path}.gz",
                project_root / f"{configured_path.name}.gz",
                project_root / f"{configured_path}.zip",
                project_root / f"{configured_path.name}.zip",
            ]
        )
    if configured_path.suffix == ".gz" and configured_path.name.endswith(".csv.gz"):
        csv_name = configured_path.name[:-3]
        configured_candidates.extend([project_root / csv_name, project_root / f"{csv_name}.zip"])
    if configured_path.suffix == ".zip" and configured_path.name.endswith(".csv.zip"):
        csv_name = configured_path.name[:-4]
        configured_candidates.extend([project_root / csv_name, project_root / f"{csv_name}.gz"])

    configured_sources = unique_existing(configured_candidates)
    if configured_sources:
        return configured_sources

    candidates: list[Path] = []
    raw_dir = project_root / "raw"
    if raw_dir.exists():
        candidates.extend(
            sorted(raw_dir.glob("holdings_raw_*.csv"))
            + sorted(raw_dir.glob("holdings_raw_*.csv.gz"))
            + sorted(raw_dir.glob("holdings_raw_*.csv.zip"))
            + sorted(raw_dir.glob("holdings.csv"))
            + sorted(raw_dir.glob("holdings.csv.gz"))
            + sorted(raw_dir.glob("holdings.csv.zip"))
        )
        year_subdir = raw_dir / "holdings_by_year"
        if year_subdir.exists():
            candidates.extend(
                sorted(year_subdir.glob("holdings_raw_*.csv"))
                + sorted(year_subdir.glob("holdings_raw_*.csv.gz"))
                + sorted(year_subdir.glob("holdings_raw_*.csv.zip"))
            )

    raw_sources = unique_existing(candidates)
    if raw_sources:
        return raw_sources

    candidates = []
    legacy_dir = project_root.parent / "export_replication" / "holdings_by_year"
    if legacy_dir.exists():
        candidates.extend(
            sorted(legacy_dir.glob("holdings_raw_*.csv"))
            + sorted(legacy_dir.glob("holdings_raw_*.csv.gz"))
            + sorted(legacy_dir.glob("holdings_raw_*.csv.zip"))
        )

    return unique_existing(candidates)


def _zip_csv_member(source: Path) -> str:
    with zipfile.ZipFile(source) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
            and not Path(name).name.startswith("._")
        ]
    if not members:
        raise ValueError(f"No readable CSV member found in zip: {source}")
    preferred = [name for name in members if Path(name).name.lower() == "holdings.csv"]
    return preferred[0] if preferred else sorted(members)[0]


def _read_holdings_header(source: Path, compression: str | None, zip_member: str | None) -> pd.DataFrame:
    if zip_member is not None:
        with zipfile.ZipFile(source) as archive:
            with archive.open(zip_member) as handle:
                return pd.read_csv(handle, nrows=0)
    return pd.read_csv(source, nrows=0, compression=compression)


def _iter_holdings_chunks(
    source: Path,
    usecols: list[str],
    chunksize: int,
    compression: str | None,
    zip_member: str | None,
) -> Any:
    if zip_member is not None:
        with zipfile.ZipFile(source) as archive:
            with archive.open(zip_member) as handle:
                for chunk in pd.read_csv(
                    handle,
                    usecols=usecols,
                    chunksize=chunksize,
                    low_memory=False,
                ):
                    yield chunk
        return

    for chunk in pd.read_csv(
        source,
        usecols=usecols,
        chunksize=chunksize,
        compression=compression,
        low_memory=False,
    ):
        yield chunk


def _load_chunked_holdings(
    source: Path,
    chunksize: int,
) -> tuple[str, list[str], Any]:
    compression = "gzip" if source.name.endswith(".gz") else None
    zip_member = _zip_csv_member(source) if source.name.endswith(".zip") else None
    header = _read_holdings_header(source=source, compression=compression, zip_member=zip_member)
    cols = header.columns.tolist()

    if {"fund_id", "date", "stock_id", "weight"}.issubset(set(cols)):
        mode = "standardized"
        usecols = ["fund_id", "date", "stock_id", "weight"]
    else:
        mode = "wrds_raw"
        usecols = [c for c in ["crsp_portno", "report_dt", "permno", "percent_tna"] if c in cols]
        if len(usecols) < 4:
            raise ValueError(f"Unsupported holdings schema in {source}")

    iterator = _iter_holdings_chunks(
        source=source,
        usecols=usecols,
        chunksize=chunksize,
        compression=compression,
        zip_member=zip_member,
    )
    return mode, usecols, iterator


def _build_holdings_asset_weights(
    sources: list[Path],
    split_keys: pd.DataFrame,
    eligible_months: pd.DataFrame,
    universe_panel: pd.DataFrame,
    chunk_size: int,
    max_chunks_per_source: int,
    logger,
) -> pd.DataFrame:
    if split_keys.empty:
        return pd.DataFrame(columns=["fund_id", "date", "asset_idx", "weight", "split"])

    split_map = split_keys[["fund_id", "date", "split"]].copy()
    month_map = eligible_months[["fund_id", "crsp_portno", "date"]].drop_duplicates()
    month_map = month_map.merge(split_map, on=["fund_id", "date"], how="inner")

    required_keys = {
        (int(row.fund_id), pd.Timestamp(row.date), str(row.split))
        for row in split_map.itertuples(index=False)
    }
    seen_keys: set[tuple[int, pd.Timestamp, str]] = set()

    universe_lookup = universe_panel[["date", "stock_id", "asset_idx"]].drop_duplicates()
    split_years = set(split_map["date"].dt.year.unique().tolist())

    aggregated_parts: list[pd.DataFrame] = []
    for source in sources:
        source_year = _extract_year_from_path(source)
        if source_year is not None and source_year not in split_years:
            continue

        logger.info("Reading holdings source: %s", source)
        mode, _, chunk_iter = _load_chunked_holdings(source, chunk_size)
        chunk_counter = 0
        kept_rows = 0
        for chunk in chunk_iter:
            chunk_counter += 1
            if max_chunks_per_source > 0 and chunk_counter > max_chunks_per_source:
                logger.info(
                    "Reached max_chunks_per_source=%s for source=%s",
                    max_chunks_per_source,
                    source.name,
                )
                break
            if mode == "standardized":
                chunk = chunk.rename(columns={"stock_id": "permno"})
                chunk["fund_id"] = _to_int_series(chunk["fund_id"])
                chunk["date"] = _to_month_end(chunk["date"])
                chunk["permno"] = _to_int_series(chunk["permno"])
                chunk["weight"] = _safe_float(chunk["weight"])
                merged = chunk.merge(split_map, on=["fund_id", "date"], how="inner")
            else:
                chunk["crsp_portno"] = _to_int_series(chunk["crsp_portno"])
                chunk["date"] = _to_month_end(chunk["report_dt"])
                chunk["permno"] = _to_int_series(chunk["permno"])
                chunk["weight"] = _safe_float(chunk["percent_tna"]) / 100.0
                merged = chunk.merge(month_map, on=["crsp_portno", "date"], how="inner")

            if merged.empty:
                continue
            merged = merged.merge(
                universe_lookup,
                left_on=["date", "permno"],
                right_on=["date", "stock_id"],
                how="inner",
            )
            if merged.empty:
                continue

            merged = merged[["fund_id", "date", "asset_idx", "weight", "split"]].copy()
            merged["weight"] = merged["weight"].clip(lower=0.0)
            merged = merged.dropna(subset=["fund_id", "date", "asset_idx", "weight"])
            if merged.empty:
                continue

            grouped = (
                merged.groupby(["fund_id", "date", "asset_idx", "split"], as_index=False)["weight"].sum()
            )
            seen_keys.update(
                {
                    (int(row.fund_id), pd.Timestamp(row.date), str(row.split))
                    for row in grouped[["fund_id", "date", "split"]].drop_duplicates().itertuples(index=False)
                }
            )
            kept_rows += len(grouped)
            aggregated_parts.append(grouped)

            if chunk_counter % 20 == 0:
                logger.info(
                    "holdings source=%s chunks=%s kept_rows=%s",
                    source.name,
                    chunk_counter,
                    kept_rows,
                )

            if required_keys and required_keys.issubset(seen_keys):
                logger.info(
                    "Collected all required split keys; stopping early at source=%s chunk=%s",
                    source.name,
                    chunk_counter,
                )
                break

        logger.info(
            "Finished holdings source=%s chunks=%s kept_rows=%s",
            source.name,
            chunk_counter,
            kept_rows,
        )

        if required_keys and required_keys.issubset(seen_keys):
            break

    if not aggregated_parts:
        return pd.DataFrame(columns=["fund_id", "date", "asset_idx", "weight", "split"])

    all_weights = pd.concat(aggregated_parts, ignore_index=True)
    all_weights = (
        all_weights.groupby(["fund_id", "date", "asset_idx", "split"], as_index=False)["weight"].sum()
    )
    return all_weights


def _build_split_dataset(
    split_name: str,
    split_keys: pd.DataFrame,
    weights_long: pd.DataFrame,
    x_by_date: dict[pd.Timestamp, np.ndarray],
    r_by_date: dict[pd.Timestamp, np.ndarray],
    num_assets: int,
    num_features: int,
) -> RealPortfolioDataset:
    if split_keys.empty:
        return _empty_real_dataset(num_assets, num_features)

    holdings_map: dict[tuple[int, pd.Timestamp], pd.DataFrame] = {}
    sub_weights = weights_long.loc[weights_long["split"] == split_name]
    for (fund_id, date), group in sub_weights.groupby(["fund_id", "date"]):
        holdings_map[(int(fund_id), pd.Timestamp(date))] = group[["asset_idx", "weight"]]

    split_keys = split_keys.sort_values(["fund_id", "date"]).reset_index(drop=True)
    x_rows: list[np.ndarray] = []
    r_rows: list[np.ndarray] = []
    w_prev_rows: list[np.ndarray] = []
    w_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    prev_by_fund: dict[int, np.ndarray] = {}
    for row in split_keys.itertuples(index=False):
        fund_id = int(row.fund_id)
        date = pd.Timestamp(row.date)
        if date not in x_by_date or date not in r_by_date:
            continue

        current = np.zeros((num_assets,), dtype=np.float32)
        h = holdings_map.get((fund_id, date))
        if h is not None and not h.empty:
            idx = h["asset_idx"].to_numpy(dtype=int)
            w = h["weight"].to_numpy(dtype=np.float32)
            current[idx] = w
        weight_sum = float(current.sum())
        if weight_sum > 0:
            current = current / weight_sum

        prev = prev_by_fund.get(fund_id)
        if prev is None:
            prev = current.copy()
        prev_by_fund[fund_id] = current.copy()

        x_rows.append(x_by_date[date])
        r_rows.append(r_by_date[date])
        w_prev_rows.append(prev)
        w_rows.append(current)
        meta_rows.append({"fund_id": fund_id, "date": date, "split": split_name})

    if not x_rows:
        return _empty_real_dataset(num_assets, num_features)

    return RealPortfolioDataset(
        x=np.stack(x_rows),
        r=np.stack(r_rows),
        w_prev=np.stack(w_prev_rows),
        w_t=np.stack(w_rows),
        meta=pd.DataFrame(meta_rows),
    )


def build_real_dataset_bundle(
    project_root: Path,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    raw_frames: dict[str, pd.DataFrame],
    derived_dir: Path,
    diagnostics_dir: Path,
    logger,
) -> DatasetBundle:
    num_assets = int(model_cfg.get("num_assets", 500))
    target_num_features = int(model_cfg.get("num_features", 8))

    market_cap = raw_frames.get("market_cap_file", pd.DataFrame())
    stock_chars = raw_frames.get("stock_chars_file", pd.DataFrame())
    stock_returns = raw_frames.get("stock_returns_file", pd.DataFrame())
    eligible_fund_months = raw_frames.get("eligible_fund_months_file", pd.DataFrame())

    if market_cap.empty or stock_chars.empty or stock_returns.empty or eligible_fund_months.empty:
        raise ValueError(
            "Missing required raw frames for real-data tensor build: "
            "market_cap_file, stock_chars_file, stock_returns_file, eligible_fund_months_file"
        )

    logger.info("Building real-data tensor datasets")
    universe_panel = _build_universe_panel(market_cap, num_assets=num_assets)
    feature_cols = _select_feature_columns(stock_chars, train_cfg=train_cfg, target_features=target_num_features)
    num_features = len(feature_cols)
    x_by_date, r_by_date = _build_feature_maps(
        universe_panel=universe_panel,
        stock_chars=stock_chars,
        stock_returns=stock_returns,
        feature_cols=feature_cols,
        num_assets=num_assets,
    )

    eligible_months = _build_eligible_months(data_cfg, eligible_fund_months)
    split_keys_full = _split_eligible_months(
        data_cfg,
        eligible_months,
        caps={"train": -1, "val": -1, "test": -1},
    )

    sources = _discover_holdings_sources(project_root, data_cfg)
    if not sources:
        raise FileNotFoundError(
            "No holdings source found. Expected year files in export_replication/holdings_by_year or raw/holdings.csv(.gz)."
        )

    mode = str(train_cfg.get("training_mode", "debug_train"))
    max_chunks_per_source = 0
    if mode in {"smoke_test", "profile_run"}:
        sources = sources[:1]
        max_chunks_per_source = int(train_cfg.get("dataset", {}).get("max_chunks_smoke", 60))
    elif mode == "debug_train":
        sources = sources[:5]
        max_chunks_per_source = int(train_cfg.get("dataset", {}).get("max_chunks_debug", 200))

    chunk_size = int(train_cfg.get("dataset", {}).get("holdings_chunksize", 250_000))
    weights_long = _build_holdings_asset_weights(
        sources=sources,
        split_keys=split_keys_full,
        eligible_months=eligible_months,
        universe_panel=universe_panel,
        chunk_size=chunk_size,
        max_chunks_per_source=max_chunks_per_source,
        logger=logger,
    )

    caps = _resolve_sample_caps(train_cfg)
    available_keys = (
        weights_long[["fund_id", "date", "split"]]
        .drop_duplicates()
        .sort_values(["split", "date", "fund_id"])
        .reset_index(drop=True)
    )
    split_parts: list[pd.DataFrame] = []
    for split_name in ["train", "val", "test"]:
        sub = available_keys.loc[available_keys["split"] == split_name, ["fund_id", "date", "split"]]
        cap = caps[split_name]
        if cap >= 0:
            sub = sub.head(cap)
        split_parts.append(sub)
    split_keys = pd.concat(split_parts, ignore_index=True)

    train_keys = split_keys.loc[split_keys["split"] == "train", ["fund_id", "date"]]
    val_keys = split_keys.loc[split_keys["split"] == "val", ["fund_id", "date"]]
    test_keys = split_keys.loc[split_keys["split"] == "test", ["fund_id", "date"]]

    train_dataset = _build_split_dataset(
        split_name="train",
        split_keys=train_keys,
        weights_long=weights_long,
        x_by_date=x_by_date,
        r_by_date=r_by_date,
        num_assets=num_assets,
        num_features=num_features,
    )
    val_dataset = _build_split_dataset(
        split_name="val",
        split_keys=val_keys,
        weights_long=weights_long,
        x_by_date=x_by_date,
        r_by_date=r_by_date,
        num_assets=num_assets,
        num_features=num_features,
    )
    test_dataset = _build_split_dataset(
        split_name="test",
        split_keys=test_keys,
        weights_long=weights_long,
        x_by_date=x_by_date,
        r_by_date=r_by_date,
        num_assets=num_assets,
        num_features=num_features,
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "feature_columns",
                "eligible_fund_months",
                "split_keys_total",
                "holdings_weight_rows",
                "train_samples",
                "val_samples",
                "test_samples",
                "num_assets",
                "num_features",
            ],
            "value": [
                ",".join(feature_cols),
                len(eligible_months),
                len(split_keys),
                len(weights_long),
                len(train_dataset),
                len(val_dataset),
                len(test_dataset),
                num_assets,
                num_features,
            ],
        }
    )
    save_csv(summary, diagnostics_dir / "tensor_build_summary.csv")

    save_parquet(universe_panel, derived_dir / "stock_universe_panel.parquet")
    if hasattr(train_dataset, "meta") and not train_dataset.meta.empty:
        save_parquet(train_dataset.meta, derived_dir / "train_tensor_index.parquet")
    if hasattr(val_dataset, "meta") and not val_dataset.meta.empty:
        save_parquet(val_dataset.meta, derived_dir / "val_tensor_index.parquet")

    return DatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        summary=summary,
    )


class PortfolioDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic dataset used for smoke/debug training while data is missing."""

    def __init__(
        self,
        n_samples: int,
        num_assets: int,
        num_features: int,
        seed: int = 42,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n_samples, num_assets, num_features, generator=generator)
        self.r = torch.randn(n_samples, num_assets, generator=generator)
        raw_prev = torch.randn(n_samples, num_assets, generator=generator)
        raw_target = torch.randn(n_samples, num_assets, generator=generator)
        self.w_prev = torch.softmax(raw_prev, dim=-1)
        self.w_t = torch.softmax(raw_target, dim=-1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "r": self.r[index],
            "w_prev": self.w_prev[index],
            "w_t": self.w_t[index],
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader with safe worker/pinning defaults."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        persistent_workers=num_workers > 0,
    )
