# Component 2 — Latex Coagulation Recommendation

## Raw data

The raw CSV (`data/raw/latex_coagulation_dataset_raw.csv`) is synthetic data with 2,606 rows and
14 columns. It contains data quality issues that are handled in step 4 (`src/04_preprocessing.py`):

| Issue | Rows | Handling |
|---|---|---|
| Unwanted columns (`Unnamed: 0`, `entry_no`, `recorded_by`, `weather_note`, `remarks`) | all | dropped |
| Units inside numbers (`35.2 %`, `10 L`), extra spaces and lower case in `batch_id`, mixed date formats | many | cleaned |
| Duplicate entries | 48 | removed |
| Missing or unreadable values | 41 | removed (targets are not filled in) |
| Impossible values (DRC 0 or above 100, non-positive volume or time, negative amounts) | 17 | removed |

Each removed row and its reason is listed in `outputs/reports/04_removed_rows.csv`.

The cleaned dataset (`data/processed/latex_coagulation_dataset_final.csv`, 2,500 rows) carries
`batch_id`, `record_date`, `metrolac_reading`, `latex_temperature_C`, `drc`, `latex_volume_L`,
`water_amount_L`, `prepared_1pct_formic_acid_solution_mL`, `coagulation_time_hours`.
`metrolac_reading` and `latex_temperature_C` are only the raw inputs used to arrive at `drc` — they
are not used as model features directly; the models consume `drc` itself.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Data understanding | `01_data_understanding.py` | overview and column summary of the raw data |
| 2. Target validation | `02_target_validation.py` | target and input column checks |
| 3. Data quality analysis | `03_data_quality_analysis.py` | issues found in the raw data |
| 4. Preprocessing | `04_preprocessing.py` | cleaned final dataset |
| 5. EDA | `05_eda.py` | statistics, correlations, distribution plots |
| 6. Temporal analysis | `06_temporal_layer_analysis.py` | monthly summary |
| 7. Feature engineering | `07_feature_engineering.py` | modelling table |
| 8. Feature selection | `08_feature_analysis_selection.py` | feature and target correlations |
| 9. Batch aware split | `09_batch_aware_split.py` | train 1,750 / validation 375 / test 375 (70/15/15, no batch in two sets) |
| 10. Train only preprocessing | `10_train_only_preprocessing.py` | imputer and scaler fitted on train data only |
| 29. Group cross validation | `29_group_cross_validation.py` | algorithm comparison |
| 30. Final test evaluation | `30_final_test_evaluation.py` | test metrics and predictions |
| 31. SHAP | `31_shap_explainability.py` | feature importance |
| 32. Export models | `32_export_final_models.py` | final model files in `artifacts/` |
| 33. Backend runtime export | `33_export_backend_runtime.py` | model files for the backend |

## Models

| | Model 1 — Water & Acid Amount | Model 2 — Coagulation Time |
|---|---|---|
| Inputs | `drc`, `latex_volume_L` | `drc`, `latex_volume_L`, `water_amount_L`, `prepared_1pct_formic_acid_solution_mL` |
| Targets | `water_amount_L`, `prepared_1pct_formic_acid_solution_mL` | `coagulation_time_hours` |
| Algorithm | ExtraTrees | Ridge |

Training data range: DRC 19.4 – 45.56 %, latex volume 5.13 – 199.92 L. Inputs outside this range
are flagged as less reliable.

## Algorithm selection (5-fold group cross validation, mean MAE)

| Algorithm | Model 1 — Water & Acid | Model 2 — Coagulation Time |
|---|---|---|
| LinearRegression | 18.55 | 0.632 |
| Ridge | 18.55 | **0.632** |
| RandomForest | 2.80 | 0.662 |
| ExtraTrees | **2.18** | 0.664 |
| HistGradientBoosting | 2.97 | 0.666 |

The algorithm with the lowest error was selected for each model.

## Final accuracy

These are regression models, so they are measured with MAE, RMSE and R² instead of a
classification accuracy.

| Target | Split | MAE | RMSE | R² |
|---|---|---|---|---|
| water_amount_L | validation | 0.32 L | 0.42 L | 0.997 |
| water_amount_L | **test** | **0.31 L** | **0.38 L** | **0.997** |
| prepared_1pct_formic_acid_solution_mL | validation | 4.45 mL | 6.47 mL | 0.999 |
| prepared_1pct_formic_acid_solution_mL | **test** | **4.24 mL** | **5.53 mL** | **0.999** |
| coagulation_time_hours | validation | 0.64 h | 0.80 h | 0.726 |
| coagulation_time_hours | **test** | **0.64 h** | **0.81 h** | **0.704** |

- Model 1 explains almost all of the variation in water and acid (R² above 0.99).
- Model 2 explains about 70 % of the variation in coagulation time, with an average error of
  about 38 minutes. The time also depends on factors that are not in the inputs.
- Validation and test results are close, so the models do not overfit.
- The dataset is synthetic, so these results describe the project dataset and are not proof of
  real-world accuracy.

## Artifacts

The trained models, model card, input schema, metrics and plots are in `artifacts/`.
See `artifacts/README.md` for details and how to run a prediction.
