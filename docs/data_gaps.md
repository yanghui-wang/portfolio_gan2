# Data Gaps and Blocking Items

This file tracks unresolved data requirements. Do not proceed to full replication claims until these are resolved.

## Required datasets and status

| Data group | Required for | Status | Gap | Label |
|---|---|---|---|---|
| Mutual fund holdings snapshots | sample construction, targets `w_t` | PARTIAL | Yearly holdings extracts available; still need strict CRSP engineering parity checks | CLOSE required |
| Fund metadata and classifications | active U.S. equity filter, Lipper labels | PARTIAL | Metadata present, but active/passive and class filtering rules still need exact field validation | CLOSE required |
| Stock characteristics panel | `X` construction | PARTIAL | Panel available; feature definitions still need EXACT mapping confirmation | CLOSE required |
| Stock returns panel | `r` construction | PARTIAL | Panel available; return horizon alignment vs paper still needs verification | CLOSE required |
| Factor data (MKT/SMB/HML/UMD) | Carhart structure | PARTIAL | Factor file present; merge-key/date harmonization and generator usage not finalized | CLOSE required |
| Market cap panel | top-500 universe | PARTIAL | Universe build implemented monthly; lag/rebalance convention needs explicit confirmation | CLOSE required |

## Specific unresolved fields

- Fund identifier mapping (`crsp_fundno`, share class aggregation logic).
- Holdings coverage variable for "75% holdings by weight reported".
- In-universe allocation definition for 75% rule.
- Rebalancing timestamp alignment between holdings and returns.
- Explicit formulas for concentration/turnover/count metrics if not fully stated.

## Available partial input already used

- `raw/eligible_fund_months.csv` is now integrated as a **CLOSE** fallback source.
- Holdings-by-year files are now integrated into train tensor construction (`export_replication/holdings_by_year`).
- Real-data tensors are generated for smoke/debug/full modes, but strict paper-faithful field parity remains an open validation task.

## Action items

1. Populate `derived/data_inventory_template.csv` with actual raw files.
2. Build `docs/variable_crosswalk.md` with exact source columns.
3. Freeze a versioned data dictionary before running sample construction.

## Replication integrity note

Any implementation made before the above is resolved must be tagged **PROXY** and excluded from final match claims.
