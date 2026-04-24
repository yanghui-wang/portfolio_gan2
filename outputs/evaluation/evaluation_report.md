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
| portfolio_gan | 44d17330e4 | model | train | L_count | EXACT | 1024 | 100.288 | 63 | 102.636 | 28 | 46 | 98.25 | 403 |
| portfolio_gan | 44d17330e4 | model | train | L_concentration | EXACT | 1024 | 0.976186 | 0.977787 | 0.0137286 | 0.95418 | 0.973425 | 0.983551 | 0.991787 |
| portfolio_gan | 44d17330e4 | model | train | L_turnover | EXACT | 0 |  |  |  |  |  |  |  |

## Representation Metrics

| status | reason | classifier | macro_precision | macro_recall | macro_f1 | accuracy | labels | confusion_matrix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SKIPPED | no complete embedding rows after cleaning |  |  |  |  |  |  | [] |

## Representation Per-Class Metrics

_No rows._

## Strategy Stability Summary

| model_name | run_id | prediction_source | split | portfolio_type | metric | status | count | mean | median | std | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| portfolio_gan | 44d17330e4 | model | train | generated | factor_tilt_stability | EXACT | 246 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| portfolio_gan | 44d17330e4 | model | train | real | factor_tilt_stability | EXACT | 246 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Frontier Proximity Summary

| model_name | run_id | prediction_source | split | metric | status | count | mean | median | std | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| portfolio_gan | 44d17330e4 | model | train | frontier_distance | CLOSE | 1024 | 0.0360504 | 0.0441439 | 0.0111725 | 0.0229407 | 0.0229407 | 0.0460468 | 0.0460468 |
| portfolio_gan | 44d17330e4 | model | train | random_reference_distance_mean | CLOSE | 1024 | 0.0417179 | 0.0374034 | 0.00838096 | 0.0317647 | 0.0342614 | 0.0504612 | 0.0539044 |

## Counterfactual Summary

_No rows._

## Skipped Metrics

| metric | status | reason |
| --- | --- | --- |
| strategy_representation_linear_probe | EXACT | no complete embedding rows after cleaning |
| counterfactual_transfer | CLOSE | missing counterfactual transfer artifact |

## Caveats

- Missing generated prediction, embedding, or transfer artifacts cause metric groups to be skipped rather than fabricated.
- Frontier proximity is an implementation approximation and should not be compared to paper tables without confirming optimization details and data alignment.
- Factor-based metrics are only as faithful as the configured factor exposure columns and source-field mapping.
