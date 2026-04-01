# Variable Crosswalk

Auto-generated from currently available files.

| canonical_variable | description | source_dataset | source_column | transform | label | status | notes |
|---|---|---|---|---|---|---|---|
| fund_id | Fund identifier | eligible_fund_months_file | crsp_fundno | rename -> fund_id | CLOSE | mapped | Share-class aggregation still unresolved |
| portfolio_id | Portfolio identifier | eligible_fund_months_file | crsp_portno | rename -> portfolio_id | CLOSE | mapped | Portfolio to fund mapping assumptions pending |
| date | Holdings report date | eligible_fund_months_file | report_dt | to_datetime | CLOSE | mapped | Monthly timestamp alignment to returns pending |
| lipper_class | Lipper class code | eligible_fund_months_file | lipper_class | as-is | CLOSE | mapped | Class cleaning for probe not finalized |
| lipper_class_name | Lipper class name | eligible_fund_months_file | lipper_class_name | as-is | CLOSE | mapped | Potentially multi-version naming across years |
| reported_weight_coverage | Portfolio weight coverage metric | eligible_fund_months_file | total_reported_weight | scale check (0-1 or 0-100) | CLOSE | mapped | Observed values suggest percentage-like scale with outliers >100 |
| in_universe_weight_share | Weight allocated in top-500 universe | eligible_fund_months_file | weight_in_top500 | scale check (0-1 or 0-100) | CLOSE | mapped | Universe definition source still indirect |
| n_holdings | Number of holdings | eligible_fund_months_file | n_holdings | as-is | CLOSE | mapped | Holdings table unavailable for exact reconciliation |
| n_distinct_stocks | Distinct stock identifiers count | eligible_fund_months_file | n_permnos | as-is | CLOSE | mapped | Exact stock panel unavailable |
| stock_id | Security identifier |  |  |  | PROXY | missing | Missing stock-level holdings file |
| weight_t | Current portfolio weight by stock |  |  |  | PROXY | missing | Missing holdings-by-stock table |
| weight_t_minus_1 | Lagged portfolio weight by stock |  |  |  | PROXY | missing | Requires holdings-by-stock time panel |
| ret_1m | Stock return |  |  |  | PROXY | missing | Missing stock returns panel |
| mkt_cap | Market cap |  |  |  | PROXY | missing | Missing market cap panel |
| factor_mkt_smb_hml_umd | Carhart factors |  |  |  | PROXY | missing | Missing factor table |
| stock_characteristics | Characteristics matrix X |  |  |  | PROXY | missing | Missing characteristics panel |
