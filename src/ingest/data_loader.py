from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd


def _candidate_paths(project_root: Path, configured_path: str) -> list[Path]:
    rel = Path(configured_path)
    candidates: list[Path] = []

    candidates.append(project_root / rel)
    candidates.append(project_root / rel.name)

    def add_csv_variants(path: Path) -> None:
        if path.suffix == ".csv":
            candidates.append(project_root / f"{path}.gz")
            candidates.append(project_root / f"{path.name}.gz")
            candidates.append(project_root / f"{path}.zip")
            candidates.append(project_root / f"{path.name}.zip")

    if rel.suffix == ".parquet":
        csv_rel = rel.with_suffix(".csv")
        candidates.append(project_root / csv_rel)
        candidates.append(project_root / csv_rel.name)
        add_csv_variants(csv_rel)
    elif rel.suffix == ".csv":
        pq_rel = rel.with_suffix(".parquet")
        candidates.append(project_root / pq_rel)
        candidates.append(project_root / pq_rel.name)
        add_csv_variants(rel)

    if rel.suffix == ".gz" and rel.name.endswith(".csv.gz"):
        csv_name = rel.name[:-3]
        candidates.append(project_root / csv_name)
        candidates.append(project_root / rel.name)
        candidates.append(project_root / f"{csv_name}.zip")

    if rel.suffix == ".zip" and rel.name.endswith(".csv.zip"):
        csv_name = rel.name[:-4]
        candidates.append(project_root / csv_name)
        candidates.append(project_root / f"{csv_name}.gz")

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _zip_csv_member(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
            and not Path(name).name.startswith("._")
        ]
    if not members:
        raise ValueError(f"No readable CSV member found in zip: {path}")
    preferred = [name for name in members if Path(name).name.lower() == "holdings.csv"]
    return preferred[0] if preferred else sorted(members)[0]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if path.name.endswith(".csv.zip"):
        member = _zip_csv_member(path)
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as handle:
                return pd.read_csv(handle)
    return pd.DataFrame()


def read_table_if_exists(project_root: Path, configured_path: str) -> pd.DataFrame:
    for candidate in _candidate_paths(project_root, configured_path):
        if candidate.exists():
            return _read_table(candidate)
    return pd.DataFrame()


def load_raw_frames(
    project_root: Path,
    data_cfg: dict,
    skip_keys: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    placeholders = data_cfg.get("placeholders", {})
    frames: dict[str, pd.DataFrame] = {}
    skip_keys = skip_keys or set()
    for key, rel_path in placeholders.items():
        if key in skip_keys:
            frames[key] = pd.DataFrame()
            continue
        frames[key] = read_table_if_exists(project_root, rel_path)
    return frames
