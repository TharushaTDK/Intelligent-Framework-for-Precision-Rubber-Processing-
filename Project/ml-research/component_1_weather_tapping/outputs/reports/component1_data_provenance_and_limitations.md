# Component 1 - Data Provenance and Limitations

- File: `data/raw/RubberSense_RawStyle_Synthetic_Dataset.xlsx`, sheet `RawStyle_Data_600`.
- SHA-256: `0bd6c3b2b5de0a0d53c9309f52805c9de9ae3042395207039251d4a925119534`.
- 600 rows, synthetic raw-style weather/estate records for demonstration.
- `estate_id` (120 synthetic values) and `village_code` (4 synthetic values) are NOT real estates.
- Targets: `latex_drc_pct`, `price_lkr_per_kg` (simulated), `tapping_suitability_proxy`
  (rule-derived with ~30% simulated label noise, seed 340).
- No real observations are fabricated or implied; the raw workbook is otherwise read-only.
- `estate_name` (added 2026-10-17) contains temporary placeholders generated from the synthetic
  `estate_id` (e.g. `EST-001` -> `TBD_EST-001`), one per estate and identical across all rows of the
  same estate. These are **not** verified real-world estate names and must not be interpreted as
  identifying actual estates; `estate_id` remains the stable identifier. Verified real names may
  replace the placeholders in a future revision. `estate_name` is documentation-only and is **not**
  an input feature of any model.
- Collection-time prediction is intentionally absent.
- Every downstream report restates the synthetic provenance and the row-wise-split caveat.

## Change history
- 2026-10-17 - `estate_name` placeholders added. The only intended dataset schema additions are the
  `estate_name` column in `RawStyle_Data_600` (column W) and in `Estate_Wise_DRC_Summary` (column H),
  plus this note in the `Data_Provenance` sheet. No existing rows, columns, sheets, splits, models or
  preprocessing artifacts were changed, and no model was retrained.
  - Original workbook SHA-256 (superseded): `31eea4269e6aec28f109271b1323ad8ce42400d4a8fcd543eccdf1e9871c92e9`.
  - Current workbook SHA-256: `0bd6c3b2b5de0a0d53c9309f52805c9de9ae3042395207039251d4a925119534`.
