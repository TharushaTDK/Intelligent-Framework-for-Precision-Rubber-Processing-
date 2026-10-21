# Component 1 - Methodology and Limitations

## Scope
RubberSense weather / tapping advisory. Three supervised tasks:
1. `drc_prediction` - regress `latex_drc_pct` (dry rubber content, %).
2. `price_estimation` - regress `price_lkr_per_kg` (price per kg of field latex, LKR).
3. `tapping_advisory` - 3-class classification of `tapping_suitability_proxy`.

**Collection-time prediction is removed**: no collection-time target exists and no such model,
comparison, report or artefact is produced.

## Data
- Source workbook: `data/raw/Combined_Rubber_Latex_Dataset_Condition_Matched.xlsx`
  (sheet `Combined_Dataset`, SHA-256 `3b9fc3a3b45dc78b17059f0db1c68a0a44a781e113e7506069045c404c56c919`, never rewritten).
- Prepared by stage 00 into `data/validated/component1_realistic_prepared.csv`
  (SHA-256 `05e05544758fef988d9ed8c7deef90070ffda2734149adce1fe56cb569b76631`).
- Training rows: the 610 farmer records (Estate 01-05, June-September 2026) with Open-Meteo hourly
  weather joined by the dataset author. The other 600 workbook rows are the earlier synthetic set
  and have no weather, so weather-based models cannot use them.
- DRC: chart-derived from Metrolac readings, but normalized by the dataset author to valid Kudaligama
  chart entries near 41%; visible latex condition was aligned with DRC. Not original measurements.
- `forecast_rain_next_6h_mm` is the Open-Meteo rain forecast for the next 6 hours issued about a
  day earlier (Previous Runs API, `data/reference/openmeteo_rain_forecast_issued_1day_before.csv`).
  The workbook's `rainfall_next_6h_observed_mm` (rain that actually fell) is used only to judge the
  tapping label, never as a model input.
- `soil_moisture_pct` is the farmer-recorded value.

## Targets
- **DRC** inputs: weather, farmer soil moisture, soil pH, visible latex condition, estate and date.
  The Metrolac reading and latex temperature are excluded because they determine DRC exactly
  through the Kudaligama chart.
- **Price** = farm-gate latex price (Rs per kg dry rubber) x DRC / 100, using
  `data/reference/latex_farmgate_price_reference.csv` (SHA-256 `45e40a132d2865370746b06df2793828c6b3ef0601e0e62bf918266a1f7df1d7`).
  Field latex is paid on its dry rubber content, so watery (low-DRC) latex is worth less per kg.
  Farm-gate levels: Rubber Development Department where published; June-September 2026 are
  estimates scaled by Central Bank of Sri Lanka rubber FOB prices.
  The price model's `latex_drc_pct` input is the estimated DRC (stage 04b: out-of-fold on train,
  train-fitted on test), because the app always prices latex from an estimated DRC.
- **Tapping** proxy: Low if >= 5 mm rain in the last 6 h or >= 5 mm in the next 6 h; Moderate for
  >= 1.5 mm / >= 2 mm or >= 60 mm over 3 days; otherwise High. "Next 6 h" uses rain that actually
  fell; the model only sees the forecast. No noise is added.

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
- Small sample: 610 records from 5 estates over 4 wet-season months; results may not carry over
  to other estates or seasons.
- DRC values were normalized and aligned with latex condition by the dataset author, so the DRC
  model mainly learns that alignment; weather alone shows almost no relationship with DRC here.
- Price and tapping targets are constructed (formula and rule). Their scores are realistic only
  because the models get what the app gets (estimated DRC, an issued forecast), not because they
  discovered real market or agronomic behaviour.
- Row-wise split: test metrics describe new observations from known estates, not unseen estates.
- Within-tolerance percentage is not accuracy.
- Permutation importance describes these constructed targets; correlation is not causation.
  SHAP was not used.
