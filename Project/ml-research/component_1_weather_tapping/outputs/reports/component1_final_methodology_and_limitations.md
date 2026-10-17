# Component 1 - Methodology and Limitations

## Scope
RubberSense weather / tapping advisory. Three supervised tasks:
1. `drc_prediction` - regress `latex_drc_pct` (dry rubber content, %).
2. `price_estimation` - regress `price_lkr_per_kg`.
3. `tapping_advisory` - 3-class classification of `tapping_suitability_proxy`.

**Collection-time prediction is removed**: no collection-time target exists and no such model,
comparison, report or artefact is produced.

## Data
- Source: `data/raw/RubberSense_RawStyle_Synthetic_Dataset.xlsx` (sheet `RawStyle_Data_600`).
- Raw workbook SHA-256: `0bd6c3b2b5de0a0d53c9309f52805c9de9ae3042395207039251d4a925119534` (not rewritten by any pipeline stage).
- 600 rows; `estate_id` values are synthetic and unverified; `village_code` synthetic.
- **All targets are simulated.** DRC and price are not measured; tapping labels are rule-derived
  with ~30% simulated label noise (seed 340) and are not expert agronomic advice.
- `estate_name` (added 2026-10-17) holds temporary `TBD_EST-xxx` placeholders derived from the
  synthetic `estate_id`. It is documentation-only, is **not** a model feature for any task, and is
  not a verified real estate name.
- Change history: 2026-10-17 - added `estate_name` placeholder columns to `RawStyle_Data_600`
  (column W) and `Estate_Wise_DRC_Summary` (column H) and a `Data_Provenance` note. No existing data,
  columns, sheets, splits, or model / preprocessing artifacts were changed; no retraining occurred.
  Original workbook SHA-256 (superseded): `31eea4269e6aec28f109271b1323ad8ce42400d4a8fcd543eccdf1e9871c92e9`.

## Protocol
- One deterministic 80/20 split per task is stored under `data/splits/` (seed 340).
- Preprocessing (imputation, scaling, one-hot) lives inside sklearn Pipelines and is fitted on the
  training partition only.
- Candidate selection uses k-fold CV on the training partition (5-fold; regression scored by RMSE,
  classification by macro-F1, stratified).
- The held-out test partition is loaded once, in stage 31.
- `30_estate_group_cross_validation.py` adds a group-aware (by `estate_id`) estimate because the
  stored split is row-wise.

## Selection
- Regression: lowest mean CV RMSE (tie-breakers MAE asc, R2 desc).
- Classification: highest mean CV macro-F1 (tie-breakers macro-recall desc, accuracy desc).
- The exported deployment pipeline is the CV-selected, train-fitted model; it was **not** refit on
  the full dataset, so it does not imply an independently tested full-data refit.

## Limitations
- Synthetic data: none of the reported metrics constitute real-world evidence.
- Row-wise split: test metrics describe new observations from known estates, not unseen estates.
- Tapping labels are simulated proxies, not verified advice.
- Within-tolerance percentage is not accuracy.
- Permutation importance describes the simulation; correlation is not causation. SHAP was not used.
