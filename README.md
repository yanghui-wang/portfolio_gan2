# Portfolio GAN Replication

Replication codebase for **“Learning to Manage Investment Portfolios beyond Simple Utility Functions”** (Scholl, Mahfouz, Calinescu, Farmer).

This repository is intentionally staged for transparent replication, with strict separation between:

1. **EXACT** (paper-specified, directly implementable),
2. **CLOSE** (reasonable inference consistent with paper intent),
3. **PROXY** (fallback when data/definitions are unavailable).

## Current scope (first build)

This initial build provides:

- End-to-end project scaffold.
- Config templates for paths/data/model/train/eval.
- Documentation stubs for replication boundary and data gaps.
- Data inventory and schema-mapping templates.
- Modular code skeletons for ingestion, sample construction, tensor building, models, training, baselines, and evaluation.

It does **not** claim empirical replication yet.

## Repository layout

```text
portfolio_gan_replication/
  config/
  raw/
  interim/
  derived/
  artifacts/
  outputs/
  docs/
  src/
  tests/
  run_pipeline.py
```

## Quick start

1. Create environment.
2. Copy `.env.example` to `.env` and update if needed.
3. Put raw source files into `raw/`.
4. Fill `derived/data_inventory_template.csv`.
5. Run:

```bash
python run_pipeline.py --stage inventory
python run_pipeline.py --stage sample
python run_pipeline.py --stage tensors
python run_pipeline.py --stage train
python run_pipeline.py --stage evaluate
```

## Pipeline stages

- `inventory`: detect files, write inventory skeleton and missing-column diagnostics.
- `sample`: sample filters and universe construction scaffold.
- `tensors`: model input index and tensor conversion scaffold.
- `train`: adversarial training scaffold.
- `evaluate`: metrics/baseline evaluation scaffold.

## Reproducibility rules

- All major parameters come from YAML configs.
- Run metadata and config snapshots should be archived per experiment.
- Intermediate datasets are intended to be saved as parquet.
- Assumptions must be recorded with labels `EXACT` / `CLOSE` / `PROXY`.

## Missing inputs (blocking full replication)

The following are not bundled and must be provided:

- CRSP mutual fund holdings snapshots.
- CRSP mutual fund metadata and classification fields.
- Stock-level characteristics panel and market cap history.
- Return histories and Carhart factor-related fields.
- Confirmed mappings for fund/stock identifiers and class filters.

See `docs/data_gaps.md` and `docs/variable_crosswalk.md`.
