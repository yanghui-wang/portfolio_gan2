# Evaluation Report

This report is produced by the scaffold evaluation layer. It does not claim scientific replication.

## Run Context

- Model: `portfolio_gan`
- Run ID: `unknown`
- Default split: `test`
- Holding threshold: `0.0001`
- Normalize weights: `True`

## Metric Status

| Metric group | Status | Notes |
|---|---|---|
| Portfolio reconstruction (`L_count`, `L_concentration`, `L_turnover`) | EXACT | Paper formulas implemented directly when `w_true`, `w_pred`, and optional `w_prev` are available. |
| Strategy representation linear probe | EXACT | Linear SVM by default; logistic regression fallback keeps a linear decision boundary. |
| Factor tilt stability | EXACT | Formula implemented directly over configured factor exposure columns. |
| Markowitz optimal-proximity | CLOSE | Uses sampled long-only mean-variance frontier because exact optimization details are not fully specified. |
| Counterfactual transfer preservation | CLOSE | Computes configured exposure deltas and structural deltas for available transfer artifacts. |

## Portfolio Reconstruction Summary

| model_name | run_id | prediction_source | split | metric | status | count | mean | median | std | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| portfolio_gan | 3874bbbb05 | model | train | L_count | EXACT | 23029 | 109.674 | 64 | 112.672 | 26 | 45 | 107 | 392 |
| portfolio_gan | 3874bbbb05 | model | train | L_concentration | EXACT | 23029 | 0.973472 | 0.976266 | 0.0159459 | 0.947841 | 0.968369 | 0.983012 | 0.991242 |
| portfolio_gan | 3874bbbb05 | model | train | L_turnover | EXACT | 0 |  |  |  |  |  |  |  |

## Representation Metrics

| classifier | macro_precision | macro_recall | macro_f1 | accuracy | labels | confusion_matrix | status | split_source | train_rows | eval_rows | embedding_columns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_svm | 0.00870251 | 0.0357143 | 0.0139949 | 0.24367 | ABR,BM,CG,CME,CS,DL,EIEI,ELCC,EMN,FS,GEI,H,ID,LCCE,LCGE,LCVE,LSE,MCGE,MLCE,MLGE,MLVE,NR,S,SESE,SPSP,TK,TL,UT | [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 347, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1126, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1203, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 610, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 360, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 322, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 293, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] | EXACT | deterministic_fallback_split | 11518 | 4937 | phi_1,phi_2,phi_3,phi_4,phi_5,phi_6,phi_7,phi_8 |

## Representation Per-Class Metrics

| label | precision | recall | f1 | support | status |
| --- | --- | --- | --- | --- | --- |
| ABR | 0 | 0 | 0 | 9 | EXACT |
| BM | 0 | 0 | 0 | 14 | EXACT |
| CG | 0 | 0 | 0 | 23 | EXACT |
| CME | 0 | 0 | 0 | 1 | EXACT |
| CS | 0 | 0 | 0 | 46 | EXACT |
| DL | 0 | 0 | 0 | 21 | EXACT |
| EIEI | 0 | 0 | 0 | 347 | EXACT |
| ELCC | 0 | 0 | 0 | 16 | EXACT |
| EMN | 0 | 0 | 0 | 9 | EXACT |
| FS | 0 | 0 | 0 | 58 | EXACT |
| GEI | 0 | 0 | 0 | 3 | EXACT |
| H | 0 | 0 | 0 | 25 | EXACT |
| ID | 0 | 0 | 0 | 15 | EXACT |
| LCCE | 0 | 0 | 0 | 1126 | EXACT |
| LCGE | 0.24367 | 1 | 0.391857 | 1203 | EXACT |
| LCVE | 0 | 0 | 0 | 610 | EXACT |
| LSE | 0 | 0 | 0 | 30 | EXACT |
| MCGE | 0 | 0 | 0 | 12 | EXACT |
| MLCE | 0 | 0 | 0 | 360 | EXACT |
| MLGE | 0 | 0 | 0 | 322 | EXACT |

_Showing 20 of 28 rows._

## Strategy Stability Summary

| model_name | run_id | prediction_source | split | portfolio_type | metric | status | count | mean | median | std | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| portfolio_gan | 3874bbbb05 | model | train | generated | factor_tilt_stability | EXACT | 2409 | 1.1905e-05 | 6.31876e-06 | 2.42682e-05 | 0 | 8.97646e-07 | 1.53036e-05 | 3.28918e-05 |
| portfolio_gan | 3874bbbb05 | model | train | real | factor_tilt_stability | EXACT | 2409 | 0.0750483 | 0.0669796 | 0.063709 | 0 | 0.0381159 | 0.0961738 | 0.180373 |

## Frontier Proximity Summary

| model_name | run_id | prediction_source | split | metric | status | count | mean | median | std | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| portfolio_gan | 3874bbbb05 | model | train | frontier_distance | CLOSE | 23029 | 0.0179955 | 0.0129796 | 0.0146283 | 0.00832744 | 0.00945856 | 0.0177115 | 0.0635376 |
| portfolio_gan | 3874bbbb05 | model | train | random_reference_distance_mean | CLOSE | 23029 | 0.0250163 | 0.0135603 | 0.0192855 | 0.00864517 | 0.0114854 | 0.0402274 | 0.0614241 |

## Counterfactual Summary

_No rows._

## Skipped Metrics

| metric | status | reason |
| --- | --- | --- |
| counterfactual_transfer | CLOSE | missing counterfactual transfer artifact |

## Caveats

- Missing generated prediction, embedding, or transfer artifacts cause metric groups to be skipped rather than fabricated.
- Frontier proximity is an implementation approximation and should not be compared to paper tables without confirming optimization details and data alignment.
- Factor-based metrics are only as faithful as the configured factor exposure columns and source-field mapping.
