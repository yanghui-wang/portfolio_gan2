# Replication Checklist

## Phase 0 — Paper audit boundary

- [ ] Extract sample construction rules from paper text/tables.
- [ ] Extract variable definitions and notation mapping.
- [ ] Extract model architecture details (all components).
- [ ] Extract training objectives and coefficients.
- [ ] Extract baseline definitions.
- [ ] Extract evaluation metrics and formulas.
- [ ] Extract target tables/figures with expected values.

## Phase 1 — Data inventory and schema mapping

- [ ] Build file-level inventory into `derived/data_inventory.csv`.
- [ ] Map raw columns into replication schema (`docs/variable_crosswalk.md`).
- [ ] Produce missing columns report (`docs/missing_columns_report.md`).
- [ ] Verify identifier linkages across fund and stock tables.

## Phase 2 — Sample construction

- [ ] Active U.S. equity fund filter implemented.
- [ ] 12-month holdings minimum rule implemented.
- [ ] 75% reported holdings coverage rule implemented.
- [ ] 75% in-universe allocation rule implemented.
- [ ] Top-500 universe by market cap implemented.
- [ ] Train/val/test temporal split validated (2010-2018, 2019, 2020-2024).

## Phase 3 — Feature and tensor construction

- [ ] Build `X`, `r`, `w_(t-1)`, `w_t` panel index.
- [ ] Save `derived/model_input_index.parquet`.
- [ ] Fit and save scalers into `artifacts/scalers/`.
- [ ] Document tensor schema in `docs/tensor_schema.md`.

## Phase 4-8 — Model stack and training

- [ ] Market generator scaffold completed.
- [ ] Strategy encoder scaffold completed.
- [ ] Portfolio allocator scaffold completed.
- [ ] Discriminator scaffold completed.
- [ ] Adversarial losses implemented with configurable lambdas.
- [ ] Training observability (logs/metrics/checkpoints) enabled.

## Phase 9-12 — Baselines, evaluation, reporting

- [ ] Baseline classes implemented.
- [ ] Evaluation metrics implemented.
- [ ] Table/figure scripts implemented.
- [ ] Validation report template completed.

## Missing inputs summary

Current blocking inputs are tracked in `docs/data_gaps.md` and should be reviewed before Phase 2.
