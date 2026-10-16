# Component 2 — Latex Coagulation Recommendation

## Raw data

The raw CSV (`data/raw/latex_coagulation_dataset_final.csv`) carries `batch_id`, `record_date`,
`metrolac_reading`, `latex_temperature_C`, `drc`, `latex_volume_L`, `water_amount_L`,
`prepared_1pct_formic_acid_solution_mL`, `coagulation_time_hours`. `metrolac_reading` and
`latex_temperature_C` are only the raw inputs used to arrive at `drc` — they are not used as
model features directly; the models consume `drc` itself.

## Models

| | Model 1 — Water & Acid Amount | Model 2 — Coagulation Time |
|---|---|---|
| Inputs | `drc`, `latex_volume_L` | `drc`, `latex_volume_L`, `water_amount_L`, `prepared_1pct_formic_acid_solution_mL` |
| Targets | `water_amount_L`, `prepared_1pct_formic_acid_solution_mL` | `coagulation_time_hours` |
| Algorithm | ExtraTrees | Ridge |

## Final accuracy (held-out test set)

| Target | MAE | RMSE | R² |
|---|---|---|---|
| water_amount_L | 0.31 | 0.38 | **0.997** |
| prepared_1pct_formic_acid_solution_mL | 4.24 | 5.53 | **0.999** |
| coagulation_time_hours | 0.64 | 0.81 | **0.704** |

## Artifacts

The trained models, model card, input schema, metrics and plots are in `artifacts/`.
See `artifacts/README.md` for details and how to run a prediction.
