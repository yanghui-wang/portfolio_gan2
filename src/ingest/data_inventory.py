from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import save_csv


@dataclass
class DataAsset:
    """Metadata record for one required dataset input."""

    dataset_key: str
    file_path: str
    exists: bool
    status: str


def _candidate_paths(project_root: Path, configured_path: str) -> list[Path]:
    rel = Path(configured_path)
    candidates: list[Path] = [project_root / rel, project_root / rel.name]

    def add_csv_variants(path: Path) -> None:
        if path.suffix == ".csv":
            candidates.extend([project_root / f"{path}", project_root / f"{path}.gz"])
            candidates.extend([project_root / path.name, project_root / f"{path.name}.gz"])
            candidates.extend([project_root / f"{path}.zip", project_root / f"{path.name}.zip"])

    if rel.suffix == ".parquet":
        csv_rel = rel.with_suffix(".csv")
        candidates.extend([project_root / csv_rel, project_root / csv_rel.name])
        add_csv_variants(csv_rel)
    elif rel.suffix == ".csv":
        pq_rel = rel.with_suffix(".parquet")
        candidates.extend([project_root / pq_rel, project_root / pq_rel.name])
        add_csv_variants(rel)

    if rel.suffix == ".gz" and rel.name.endswith(".csv.gz"):
        csv_name = rel.name[:-3]
        candidates.extend([project_root / csv_name, project_root / rel.name, project_root / f"{csv_name}.zip"])

    if rel.suffix == ".zip" and rel.name.endswith(".csv.zip"):
        csv_name = rel.name[:-4]
        candidates.extend([project_root / csv_name, project_root / f"{csv_name}.gz"])

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def build_inventory(data_cfg: dict, project_root: Path) -> pd.DataFrame:
    """Build a file-level inventory from configured placeholder paths."""

    rows: list[dict[str, object]] = []
    placeholders = data_cfg.get("placeholders", {})
    for dataset_key, file_path in placeholders.items():
        matched = None
        for candidate in _candidate_paths(project_root, file_path):
            if candidate.exists():
                matched = candidate.resolve()
                break
        exists = matched is not None
        rows.append(
            {
                "dataset_key": dataset_key,
                "configured_path": file_path,
                "resolved_path": str(matched) if matched else "",
                "exists": exists,
                "status": "ready" if exists else "missing",
            }
        )
    return pd.DataFrame(rows)


def write_inventory_reports(
    inventory_df: pd.DataFrame,
    derived_dir: Path,
    docs_dir: Path,
) -> None:
    """Persist inventory CSV and a simple missing-input report."""

    save_csv(inventory_df, derived_dir / "data_inventory.csv")

    missing = inventory_df[~inventory_df["exists"]].copy()
    lines = ["# Missing Inputs Report", "", "## Missing data assets", ""]
    if missing.empty:
        lines.append("All placeholder files are available.")
    else:
        for _, row in missing.iterrows():
            lines.append(f"- `{row['dataset_key']}`: `{row['configured_path']}`")

    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "missing_inputs_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
