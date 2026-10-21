# Component 1 - Data Provenance and Limitations

- Source workbook: `data/raw/Combined_Rubber_Latex_Dataset_Condition_Matched.xlsx`,
  sheet `Combined_Dataset`, SHA-256 `3b9fc3a3b45dc78b17059f0db1c68a0a44a781e113e7506069045c404c56c919` (read-only).
- Prepared copy (stage 00): `data/validated/component1_realistic_prepared.csv`,
  SHA-256 `05e05544758fef988d9ed8c7deef90070ffda2734149adce1fe56cb569b76631`.
- 1210 workbook rows: 610 farmer records (Estate 01-05, Jun-Sep 2026, Open-Meteo weather) and 600
  model-development rows (the earlier synthetic set, no weather). Models train on the farmer rows.
- Weather and soil temperature/moisture estimates come from Open-Meteo (modelled, not sensors);
  `soil_moisture_pct` used by the models is the farmer-recorded value.
- `latex_drc_pct` and Metrolac readings were normalized to Kudaligama chart entries by the dataset
  author; `visible_latex_condition` was aligned with DRC. Neither is an original field measurement.
- `price_lkr_per_kg` (field latex) = farm-gate latex price x DRC / 100. Sources per month are in
  `data/reference/latex_farmgate_price_reference.csv`: Rubber Development Department figures where
  published, Central Bank of Sri Lanka FOB-scaled estimates for June-September 2026.
- `forecast_rain_next_6h_mm` is the Open-Meteo forecast issued about a day earlier
  (`data/reference/openmeteo_rain_forecast_issued_1day_before.csv`).
- `tapping_suitability_proxy` is rule-derived from rainfall that actually fell (no added noise),
  not expert advice.
- Collection-time prediction is intentionally absent.
