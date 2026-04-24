# Evaluation Metrics

This note documents the implementation contract for `python run_pipeline.py --stage evaluate`.
The evaluation layer writes machine-readable tables under `outputs/evaluation/` and an
`evaluation_report.md` summary. It skips unavailable metric groups rather than fabricating
results unless `evaluation.require_full_metrics: true`.

## Input Contract

Evaluation inputs are configured in `config/eval.yaml`. Tables are normalized through
column aliases before metric functions run. Supported fields include:

- `fund_id`, `date`, `asset_id`
- `w_true`, `w_pred`, `w_prev`
- `style_label`
- `phi_1 ... phi_k` or other configured embedding columns
- `market_beta`, `SMB`, `HML`, `UMD` or configured factor columns
- `ret`
- counterfactual weights `w_original`, `w_transferred`, and optional `w_prev_transferred`

The same interface supports generated portfolios and baselines through `model_name`,
`run_id`, `split`, and `prediction_source`.

After training, `src.training.evaluation_exporter` runs deterministic inference with the
posterior mean strategy representation and writes:

- `artifacts/evaluation/portfolio_predictions.parquet`
- `artifacts/embeddings/strategy_embeddings.parquet`

Those files are the default inputs for the `evaluate` stage. When `raw/lipper.csv` is
available, the exporter attaches `style_label` from the configured Lipper class column.
The default label merge is backward-looking by fund, so a sample uses the latest
available Lipper label at or before its month-end date.
The default export includes train/validation/test splits so smoke runs with no validation
or test observations still produce a small artifact for pipeline verification.

## Metrics

| Metric group | Status | Implementation |
|---|---|---|
| Count error | EXACT | `abs(count(w_pred > threshold) - count(w_true > threshold))` |
| Concentration error | EXACT | `abs(sum(w_pred^2) - sum(w_true^2))` |
| Turnover error | EXACT | `abs(sum(abs(w_pred - w_prev)) - sum(abs(w_true - w_prev)))` |
| Linear probe | EXACT | Linear SVM by default, logistic regression fallback, with macro precision/recall/F1 and per-class rows. |
| Strategy stability | EXACT | Computes `w'X`, demeans by date, then averages absolute temporal drift by fund. |
| Carhart beta construction | CLOSE | If asset-level factor exposures are missing, estimates rolling stock-level betas from `stock_returns` and `carhart_factors`. |
| Markowitz optimal-proximity | CLOSE | Uses an ex-post sampled long-only mean-variance frontier. Exact optimization details are not available in the scaffold. |
| Counterfactual exposure preservation | CLOSE | Computes configured factor exposure deltas and optional structural deltas for transfer artifacts. |

## Approximation Choices

- Frontier distance is normalized Euclidean distance in risk-return space to a sampled
  efficient frontier. The number of samples, covariance shrinkage, lookback window, and
  long-only constraint are config-driven.
- Random frontier references are count-matched by default and style-matched when a usable
  style label exists in the portfolio table.
- Weight vectors can be normalized before evaluation through
  `evaluation.normalize_weights`.
- Lipper labels are read from `evaluation.labels.path`; default source columns are
  `crsp_fundno`, `caldt`, and `lipper_class`.
- When `market_beta`, `SMB`, `HML`, and `UMD` are absent from the configured
  factor exposure table, evaluation can estimate rolling Carhart betas from
  `raw/stock_returns` and `raw/carhart_factors`. Defaults are a 36-month lookback
  and 24-month minimum window, cached at `artifacts/evaluation/carhart_betas.parquet`.
- Missing factor, return, embedding, or counterfactual artifacts are recorded in
  `outputs/evaluation/skipped_metrics.json`.

## Known Deviations

- The module implements the paper-aligned evaluation layer but does not claim empirical
  replication of reported tables.
- Counterfactual evaluation depends on separately generated transfer artifacts; the
  evaluate stage does not generate those portfolios.
- Factor fidelity is limited by the available factor exposure columns and the configured
  source-field mapping.
- Representation classification depends on Lipper label coverage after matching each
  fund-month to a configured style label.
