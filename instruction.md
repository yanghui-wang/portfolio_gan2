# Codex Instruction — Implement Evaluation Metrics Module

You are working inside an existing runnable replication scaffold for the paper:

**“Learning to Manage Investment Portfolios beyond Simple Utility Functions”**

The repository already has a working project structure and pipeline stages documented in the README. Do **not** redesign the repo. Your job is to **complete the evaluation metrics part** in a way that is faithful to the paper, modular, testable, and compatible with the current scaffold.

---

## 0. Objective

Implement the `evaluate` stage so that the project can compute, save, and report the paper-aligned evaluation metrics for generated portfolios and baselines.

Focus on the metrics described in the paper's **Section 4.3 Evaluation Metrics**, and structure the implementation so that:

- metrics can be computed for:
  - real vs generated portfolios,
  - real vs baseline portfolios,
  - replication setting (same market state),
  - counterfactual / transfer setting where supported by available data;
- outputs are reproducible and saved to disk;
- the code is robust to partially missing inputs;
- each metric is explicitly tagged as:
  - `EXACT` = directly specified by the paper,
  - `CLOSE` = reasonable implementation inference,
  - `PROXY` = fallback when exact inputs are unavailable.

Do not claim scientific replication. Only implement the evaluation layer cleanly and transparently.

---

## 1. Constraints

### 1.1 Keep repo structure stable
Follow the existing project structure from the README. Do not move major directories or introduce a new framework. Integrate into the current scaffold.

### 1.2 Prefer small, composable modules
Add or complete code in `src/` using small pure functions where possible.

### 1.3 Config-driven
All thresholds, column names, file paths, and evaluation options must come from YAML config or a central config object where consistent with the repo.

### 1.4 No hidden assumptions
Any assumption not fully specified by the paper must be documented in code comments and in a markdown note under `docs/`.

### 1.5 Graceful degradation
If a metric cannot be computed because the required inputs are unavailable, do not crash the whole pipeline unless the config explicitly says so.
Instead:
- emit a warning,
- write the metric as missing / skipped,
- record the reason in a machine-readable artifact.

---

## 2. Paper-aligned scope to implement

Implement the evaluation metrics corresponding to the paper's evaluation section.

### 2.1 Portfolio Reconstruction Quality
Implement:

1. **Count Error (`Lcount`)**  
   Absolute difference in the number of assets held above a holding threshold.

2. **Concentration Error (`Lconcentration`)**  
   Absolute difference in Herfindahl index:
   `abs(sum(w_hat^2) - sum(w^2))`

3. **Turnover Error (`Lturnover`)**  
   Compare turnover relative to previous weights:
   `abs(sum(abs(w_hat_t - w_t_minus_1)) - sum(abs(w_t - w_t_minus_1)))`
   
Use the exact formula above unless an existing tensor convention requires equivalent handling.

### 2.2 Strategy Representation Quality
Implement a latent representation evaluation module that supports:

1. **Linear probe classification**
   - train a linear classifier on strategy embeddings,
   - predict fund style labels,
   - report:
     - macro recall,
     - macro precision,
     - macro F1,
     - per-class precision / recall / F1 / support.

Preferred default: linear SVM if available in project dependencies; otherwise logistic regression with equivalent linear decision boundary.

### 2.3 Behavioral Fidelity
Implement:

1. **Strategy Stability**
   - compute factor tilt drift over time;
   - for each fund:
     - factor tilt at time `t`: `beta_{a,t} = w_{a,t}^T X_t`
     - drift:
       `u_a = mean_t sum(abs((beta_{a,t} - beta_bar_t) - (beta_{a,t-1} - beta_bar_{t-1})))`
   - report summary stats across funds.

2. **Markowitz Optimal-Proximity**
   Implement a configurable approximation of the paper's frontier-distance idea:
   - for each period:
     - compute realized mean returns and covariance from available asset return panel;
     - compute ex-post efficient frontier or a practical approximation;
     - measure distance of each portfolio to the frontier in risk-return space;
     - compare against a style-matched random portfolio sample if style labels are available.

This metric is likely `CLOSE` unless all exact paper inputs are available. Make that explicit.

### 2.4 Counterfactual Analysis
Implement support for strategy transfer evaluation where generated portfolios are produced by applying one strategy representation in another market state.

At minimum compute:

1. **Factor exposure preservation**
   - differences in exposures between original and transferred portfolios:
     - market beta delta,
     - SMB delta,
     - HML delta,
     - UMD delta.

2. **Optional structural preservation**
   - count,
   - Herfindahl concentration,
   - turnover if previous weights are defined in the transferred setting.

This module must be optional and skip cleanly if counterfactual generation artifacts are not yet available.

---

## 3. Recommended file layout inside `src/`

Adapt to the existing scaffold, but prefer something close to:

 
src/
  evaluation/
    __init__.py
    metrics_portfolio.py
    metrics_behavior.py
    metrics_representation.py
    metrics_counterfactual.py
    frontier.py
    aggregation.py
    io.py
 


## 4. Data contract

Define and document a clear evaluation input contract.

The evaluation stage should work with tables / tensors that can provide, at minimum, some subset of:

* fund_id
* date
* asset_id
* w_true
* w_pred
* w_prev
* label or style_label
* asset characteristics matrix / columns for factor exposures
* asset returns history or realized forward returns
* latent embedding columns, e.g. phi_1 ... phi_k
* evaluation split indicator
* model name / baseline name

Implement adapter functions if the current pipeline stores these in different shapes.

Do not hardcode a single format deep inside metric functions.
Metric functions should operate on normalized in-memory structures.

⸻

## 5. Implementation details

### 5.1 Portfolio metric functions

Implement pure functions:

* holding_count(weights, threshold)
* herfindahl_index(weights)
* portfolio_turnover(w_current, w_prev)
* count_error(w_true, w_pred, threshold)
* concentration_error(w_true, w_pred)
* turnover_error(w_true, w_pred, w_prev)

Requirements:

* validate weights are numeric;
* fill missing weights with zero where appropriate;
* tolerate sparse portfolios;
* normalize weights if config says to normalize before evaluation.

### 5.2 Representation metrics

Implement:

* extraction of embedding matrix X_embed
* label vector y
* train/validation/test split handling
* classifier training
* metrics report generation

Requirements:

* default to test split reporting;
* macro metrics required;
* per-class table required;
* confusion matrix optional but useful.

### 5.3 Strategy stability

Implement a function that:

* computes portfolio factor tilt w^T X,
* computes cross-sectional average beta_bar_t,
* computes per-fund temporal drift,
* returns:
    * per-fund results,
    * aggregate summary (mean, median, std, percentiles).

### 5.4 Markowitz optimal-proximity

Implement this carefully and transparently.

Recommended design:

* estimate_portfolio_return(portfolio_weights, realized_asset_returns)
* estimate_portfolio_risk(portfolio_weights, cov_matrix)
* build_efficient_frontier(...)
* distance_to_frontier(...)
* sample_style_matched_random_portfolios(...)
* optimal_proximity_score(...)

Because exact paper details may be unavailable, expose config for:

* covariance shrinkage,
* long-only constraint,
* weight sum constraint,
* number of random portfolios,
* style-matching rule,
* frontier grid density.

This module must include very clear docstrings stating what is exact vs approximate.

### 5.5 Counterfactual transfer metrics

Implement functions for:

* original exposure computation,
* transferred exposure computation,
* absolute deltas by factor,
* aggregate summary table.

Return both:

* per-case results,
* aggregate model-level summary.

⸻

## 6. Output artifacts

The evaluate stage must save artifacts under the repo’s outputs / artifacts convention.

At minimum produce:

### 6.1 Machine-readable outputs

* outputs/evaluation/portfolio_metrics_by_sample.parquet
* outputs/evaluation/portfolio_metrics_summary.csv
* outputs/evaluation/representation_metrics.json
* outputs/evaluation/representation_per_class.csv
* outputs/evaluation/stability_by_fund.parquet
* outputs/evaluation/stability_summary.csv
* outputs/evaluation/frontier_metrics_by_sample.parquet
* outputs/evaluation/frontier_summary.csv
* outputs/evaluation/counterfactual_metrics_by_case.parquet
* outputs/evaluation/counterfactual_summary.csv
* outputs/evaluation/skipped_metrics.json

### 6.2 Human-readable report

Generate:

* outputs/evaluation/evaluation_report.md

This markdown report should contain:

* what model / baseline was evaluated,
* data split used,
* which metrics ran,
* which metrics were skipped,
* summary tables,
* key caveats,
* EXACT / CLOSE / PROXY status per metric.

⸻

## 7. Baseline compatibility

The paper compares against baseline models such as zero-trade, turnover-matched random, factor-tilt matched, and generator-only ablation.

Your implementation should not assume only the full GAN exists.

Design evaluation so it can compare multiple model outputs with a common interface:

* model_name
* run_id
* split
* prediction_source

If baselines are already scaffolded, wire evaluation to them.
If some baselines are not yet implemented, make evaluation compatible with them without fabricating results.

⸻

## 8. Config additions

If needed, add evaluation config under something like:
evaluation:
  holding_threshold: 0.0001
  normalize_weights: true
  require_full_metrics: false
  representation:
    classifier: linear_svm
    average_embeddings_over_time: false
  frontier:
    enabled: true
    method: mean_variance_long_only
    num_random_portfolios: 1000
    covariance_shrinkage: 0.0
  counterfactual:
    enabled: true

    Only add keys that are actually used.

⸻

## 9. Tests

Add focused tests under tests/.

Minimum required tests:

### 9.1 Unit tests

* count metric on sparse vectors
* Herfindahl calculation sanity checks
* turnover calculation sanity checks
* exposure computation shape and value checks
* drift metric on a tiny handcrafted two-period example

### 9.2 Integration-style tests

* evaluate stage runs on a toy dataset
* skipped metrics are recorded when inputs are missing
* report files are created

Avoid brittle tests tied to exact floating-point values unless the case is analytically simple.

⸻

## 10. Documentation to add

Add a short markdown note, e.g.:

* docs/evaluation_metrics.md

It should explain:

* metric definitions,
* exact formulas used,
* required inputs,
* approximation choices,
* which metrics are EXACT / CLOSE / PROXY,
* known deviations from the paper.

Keep it concise and implementation-facing.

⸻

## 11. Coding style expectations

* Use type hints where consistent with repository style.
* Prefer pandas / numpy implementations unless the repo already uses something else.
* Keep functions short and named by business meaning.
* Raise informative errors for malformed inputs.
* Use logging, not print.
* Do not introduce heavyweight dependencies unless already present.

⸻

## 12. Deliverables

When done, provide:

1. a short summary of files changed,
2. a list of implemented metrics,
3. a list of skipped / approximate metrics,
4. any assumptions that still block exact replication.

Do not overstate scientific validity. This is implementation completion for the evaluation layer, not a claim of reproducing the paper’s empirical tables.