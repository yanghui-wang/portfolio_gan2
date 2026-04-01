from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


class MetricsWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_jsonl = self.output_dir / "train_steps.jsonl"
        self.epoch_jsonl = self.output_dir / "val_epochs.jsonl"
        self.heartbeat_jsonl = self.output_dir / "heartbeat.jsonl"
        self.metrics_csv = self.output_dir / "metrics.csv"
        self.csv_columns: list[str] = []
        if self.metrics_csv.exists():
            try:
                old = pd.read_csv(self.metrics_csv, nrows=1)
                self.csv_columns = old.columns.tolist()
            except Exception:
                self.csv_columns = []

    def log_step(self, payload: dict[str, object]) -> None:
        with self.step_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        self._append_csv(payload)

    def log_epoch(self, payload: dict[str, object]) -> None:
        with self.epoch_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        self._append_csv(payload)

    def log_heartbeat(self, payload: dict[str, object]) -> None:
        with self.heartbeat_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        self._append_csv(payload)

    def _append_csv(self, payload: dict[str, object]) -> None:
        payload_keys = list(payload.keys())
        if not self.csv_columns:
            self.csv_columns = payload_keys
            self._append_row(payload)
            return

        new_keys = [k for k in payload_keys if k not in self.csv_columns]
        if new_keys:
            self.csv_columns.extend(new_keys)
            existing = pd.read_csv(self.metrics_csv) if self.metrics_csv.exists() else pd.DataFrame()
            for key in new_keys:
                if key not in existing.columns:
                    existing[key] = None
            existing = existing.reindex(columns=self.csv_columns)
            existing.to_csv(self.metrics_csv, index=False)

        self._append_row(payload)

    def _append_row(self, payload: dict[str, object]) -> None:
        row = {col: payload.get(col) for col in self.csv_columns}
        mode = "a" if self.metrics_csv.exists() else "w"
        with self.metrics_csv.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.csv_columns)
            if mode == "w":
                writer.writeheader()
            writer.writerow(row)
