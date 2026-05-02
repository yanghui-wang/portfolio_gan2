# Portfolio GAN Replication

Replication codebase for **“Learning to Manage Investment Portfolios beyond Simple Utility Functions”** (Scholl, Mahfouz, Calinescu, Farmer).

This repository is intentionally staged for transparent replication, with strict separation between:

1. **EXACT** (paper-specified, directly implementable),
2. **CLOSE** (reasonable inference consistent with paper intent),
3. **PROXY** (fallback when data/definitions are unavailable).

## Current scope (first build) + News-Aware Extension

This build provides:

- End-to-end project scaffold with **dual-model support** (Carhart 4-factor baseline + News-Aware 9-factor extension).
- Config templates for paths/data/model/train/eval, with `model_type` parameter for factor model selection.
- Documentation stubs for replication boundary and data gaps.
- Data inventory and schema-mapping templates.
- Modular code skeletons for ingestion, sample construction, tensor building, models, training, baselines, and evaluation.
- **News-Aware factor model integration**: combines Carhart rolling betas with 5 orthogonalized news factors for extended evaluation.

### News-Aware Extension (New)

This codebase now supports running parallel evaluations with two factor models:

1. **Carhart 4-factor** (default): market beta, SMB, HML, UMD — paper-original setup.
2. **News-Aware 9-factor** (extended): Carhart 4 + 5 orthogonalized news factors (sentiment, risk, uncertainty, macro credit pressure, corporate market activity).

The GAN training core remains **unchanged**; the extension only affects the evaluation layer. Use `--model-type` to switch between them without re-training.

It does **not** claim empirical replication of the original paper (4-factor only); the 9-factor mode is an extension research configuration.

## Repository layout

```text
portfolio_gan_replication/
  config/
    paths.yaml           # paths for checkpoints, logs, outputs
    data.yaml
    train.yaml
    eval.yaml            # evaluation config (includes model_type selection)
  raw/                   # raw data inputs
  interim/
  derived/
  artifacts/
    News_aware/          # (if model_type == news_aware)
  outputs/
    news_aware/          # (if model_type == news_aware)
  docs/
    data_gaps.md
    variable_crosswalk.md
    news_aware_PGAN.md   # News-aware extension guide
  src/
    training/
    evaluation/
      evaluator.py       # dispatches by model_type
      factor_exposures_news_aware.py  # 9-factor builder
  tests/
  run_pipeline.py        # supports --model-type argument
  requirements-all.txt   # consolidated dependencies
```

## Quick start

1. Create environment and install dependencies.
   ```bash
   pip install -r requirements-all.txt
   ```

2. Copy `.env.example` to `.env` and update if needed.

3. Put raw source files into `raw/`.

4. Fill `derived/data_inventory_template.csv`.

5. Run the full pipeline with your chosen factor model:

   **Carhart 4-factor (paper-original):**
   ```bash
   PYTHONPATH=. python run_pipeline.py --stage inventory --model-type carhart
   PYTHONPATH=. python run_pipeline.py --stage sample --model-type carhart
   PYTHONPATH=. python run_pipeline.py --stage tensors --model-type carhart
   PYTHONPATH=. python run_pipeline.py --stage train --model-type carhart
   PYTHONPATH=. python run_pipeline.py --stage evaluate --model-type carhart
   ```

   **News-Aware 9-factor (extended):**
   ```bash
   PYTHONPATH=. python run_pipeline.py --stage train --model-type news_aware
   PYTHONPATH=. python run_pipeline.py --stage evaluate --model-type news_aware
   ```

   Note: `--model-type` defaults to `carhart`. The `train` stage does not depend on factor model selection (GAN training is identical); the extension only affects the `evaluate` stage.

## Pipeline stages

- `inventory`: detect files, write inventory skeleton and missing-column diagnostics.
- `sample`: sample filters and universe construction scaffold.
- `tensors`: model input index and tensor conversion scaffold.
- `train`: adversarial training scaffold.
- `evaluate`: metrics/baseline evaluation scaffold.

## Factor Model Selection

This project supports **two factor models** via the `--model-type` parameter:

### Carhart 4-Factor (paper-original, default)

```bash
python run_pipeline.py --stage evaluate --model-type carhart
```

Uses the classic Carhart (1997) factors:
- Market beta
- SMB (Small Minus Big)
- HML (High Minus Low)
- UMD (Up Minus Down / Momentum)

**Outputs**: `outputs/evaluation/*` (standard naming)

### News-Aware 9-Factor (extended)

```bash
python run_pipeline.py --stage evaluate --model-type news_aware
```

Extends Carhart 4-factor with 5 orthogonalized news-based factors:
- Market beta, SMB, HML, UMD (from Carhart)
- ortho_sentiment (market sentiment)
- ortho_risk (tail risk)
- ortho_uncertainty (macro uncertainty)
- ortho_macro_credit_pressure (credit pressure)
- ortho_corporate_market_activity (corporate activity)

News factors data source: `../../LLM news scoring/scores_monthly/news_factors_orthogonalized_2010_2024.csv` (must be provided).

**Outputs**: `outputs/evaluation/*_news_aware` (suffixed to avoid overwriting carhart results)  
**Artifacts**: `artifacts/news_aware/` (isolated checkpoint/embedding directory)

### Configuration

To change the default factor model, edit `config/eval.yaml`:

```yaml
evaluation:
  model_type: news_aware  # or "carhart"
```

Command-line override takes precedence:

```bash
python run_pipeline.py --stage evaluate --model-type news_aware
```

## Reproducibility rules

- All major parameters come from YAML configs.
- Run metadata and config snapshots should be archived per experiment.
- Intermediate datasets are intended to be saved as parquet.
- Assumptions must be recorded with labels `EXACT` / `CLOSE` / `PROXY`.
- Use `--model-type` to select factor model; results are automatically isolated by suffix/directory to prevent overwrites.

## Evaluation outputs by model type

### Carhart evaluation (default)

When running `--model-type carhart`, outputs land in:
- `outputs/evaluation/portfolio_metrics_summary.csv`
- `outputs/evaluation/stability_summary.csv`
- `outputs/evaluation/frontier_metrics_by_sample.parquet`
- `outputs/evaluation/evaluation_report.md`

### News-Aware evaluation

When running `--model-type news_aware`, outputs land in:
- `outputs/evaluation/portfolio_metrics_summary_news_aware.csv`
- `outputs/evaluation/stability_summary_news_aware.csv`
- `outputs/evaluation/frontier_metrics_by_sample_news_aware.parquet`
- `outputs/evaluation/evaluation_report_news_aware.md`

(Note the `_news_aware` suffix to prevent collision with carhart results.)

## News-Aware Extension Details

For a detailed guide on the 9-factor news-aware extension, see [docs/news_aware_PGAN.md](docs/news_aware_PGAN.md). This document explains:

- Factor definitions and orthogonalization.
- Configuration changes and data requirements.
- Expected data flows and output file naming.
- Alignment with the paper evaluation framework.

## Missing inputs (blocking full replication)

The following are not bundled and must be provided:

- CRSP mutual fund holdings snapshots.
- CRSP mutual fund metadata and classification fields.
- Stock-level characteristics panel and market cap history.
- Return histories and Carhart factor-related fields.
- Confirmed mappings for fund/stock identifiers and class filters.

See `docs/data_gaps.md` and `docs/variable_crosswalk.md`.
