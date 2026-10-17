# Component 1 - Data Provenance and Limitations

- File: `data/raw/RubberSense_RawStyle_Synthetic_Dataset.xlsx`, sheet `RawStyle_Data_600`.
- SHA-256: `31eea4269e6aec28f109271b1323ad8ce42400d4a8fcd543eccdf1e9871c92e9`.
- 600 rows, synthetic raw-style weather/estate records for demonstration.
- `estate_id` (120 synthetic values) and `village_code` (4 synthetic values) are NOT real estates.
- Targets: `latex_drc_pct`, `price_lkr_per_kg` (simulated), `tapping_suitability_proxy`
  (rule-derived with ~30% simulated label noise, seed 340).
- No real observations are fabricated or implied; the raw workbook is read-only.
- Collection-time prediction is intentionally absent.
- Every downstream report restates the synthetic provenance and the row-wise-split caveat.
