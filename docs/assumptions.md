# Assumptions Register

Every assumption must be labeled as:

- **EXACT**: directly stated in paper and fully implemented as written.
- **CLOSE**: inferred implementation consistent with paper intent.
- **PROXY**: fallback due to missing data/specification.

## Current assumptions

| ID | Topic | Assumption | Label | Rationale | Review needed |
|---|---|---|---|---|---|
| A1 | Sample period | Use 2010-01 to 2024-12 monthly holdings snapshots | EXACT | Explicit paper period | No |
| A2 | Temporal split | Train 2010-2018, val 2019, test 2020-2024 | EXACT | Explicit paper split | No |
| A3 | Latent dimension | `phi_dim = 8` | EXACT | Explicit paper detail | No |
| A4 | Optimizer/lr | Adam with `1e-4` for G and D | EXACT | Explicit paper detail | No |
| A5 | D:G step ratio | 3 discriminator updates per 1 generator update | EXACT | Explicit paper detail | No |
| A6 | Active U.S. equity filter fields | Use best-available class fields until CRSP mapping confirmed | CLOSE | Paper states rule, fields underspecified | Yes |
| A7 | Top-500 universe schedule | Rebuild top-500 monthly using lagged market cap | CLOSE | Consistent with no look-ahead | Yes |
| A8 | Missing metrics formula details | Use standard definitions and document formulas | CLOSE | Paper-level ambiguity | Yes |
| A9 | Default training data path | Real-data tensors are built from eligible fund-month mappings + holdings + stock characteristics/returns; synthetic dataset remains as legacy utility only | CLOSE | Uses available WRDS exports, but full CRSP engineering details still being verified | Yes |
| A10 | Eligible-fund-month fallback | Apply 75% filters at observation level then require >=12 eligible months per fund using `eligible_fund_months.csv` | CLOSE | Closest available implementation without full holdings table | Yes |
| A11 | Fast debug subsampling | `smoke_test` / `debug_train` limit holdings chunks and sample caps for quick fault detection before full run | PROXY | Improves debuggability and runtime observability; not intended for headline results | No |
| A12 | Frontier proximity | Approximate Markowitz optimal-proximity with a sampled long-only ex-post frontier and normalized risk-return distance | CLOSE | Paper intent is clear, but exact optimization details and constraints are not specified in scaffold inputs | Yes |
| A13 | Counterfactual transfer metrics | Evaluate provided transfer artifacts with factor exposure deltas and structural deltas; do not generate transfers in evaluation | CLOSE | Matches evaluation need while keeping generation as a separate artifact-producing stage | Yes |
| A14 | Lipper label matching | Attach style labels to fund-month artifacts using the latest available `lipper_class` for the same fund at or before the sample date | CLOSE | Avoids look-ahead when Lipper label dates are less frequent than monthly portfolio samples | Yes |
| A15 | Carhart asset exposures | Estimate `market_beta`, `SMB`, `HML`, and `UMD` with rolling stock-level regressions when precomputed exposure columns are unavailable | CLOSE | Supplies paper-aligned factor tilt inputs from available returns/factor panels, but window and estimation choices are scaffold approximations | Yes |

## Governance

- New assumptions must be appended with unique ID.
- Any change from CLOSE/PROXY to EXACT requires citation to paper section/table and source-field mapping confirmation.
