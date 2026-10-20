# Component 2 — Latex Coagulation Recommendation

## Raw data

The raw CSV (`data/raw/latex_coagulation_dataset_raw.csv`) is synthetic data with 2,614 rows and
15 columns (2026-08-01 to 2026-09-30, latex volume 1.0 – 2.0 L). It contains data quality issues
that are handled in step 4 (`src/04_preprocessing.py`):

| Issue | Rows | Handling |
|---|---|---|
| Unwanted columns (`Unnamed: 0`, `entry_no`, `recorded_by`, `weather_note`, `tank_no`, `remarks`) | all | dropped |
| Units inside numbers (`44.9 %`, `1.5 L`, `222 mL`, `28.7 °C`, `205 min`), extra spaces and lower case in `batch_id`, mixed date formats | many | cleaned |
| Duplicate entries | 52 | removed |
| Empty `drc` | 260 | filled from the ready-reckoner chart (see below) |
| Other missing or unreadable values | 43 | removed (targets are not filled in) |
| Impossible values (non-positive Metrolac reading, volume, acid or time, temperature outside 15–35 °C, negative water) | 19 | removed |

Each removed row and its reason is listed in `outputs/reports/04_removed_rows.csv`.

The cleaned dataset (`data/processed/latex_coagulation_dataset_final.csv`, 2,500 rows) carries
`batch_id`, `record_date`, `metrolac_reading`, `latex_temperature_C`, `drc`, `latex_volume_L`,
`water_amount_L`, `prepared_1pct_formic_acid_solution_mL`, `coagulation_time_minutes`.
`metrolac_reading` and `latex_temperature_C` are only the raw inputs used to arrive at `drc` — they
are not used as model features directly; the models consume `drc` itself. In the deployed system
`drc` comes from Component 1.

## DRC from the Metrolac reading

`drc` is worked out with the temperature corrected Metrolac ready-reckoner chart (Table 1 of
Kudaligama et al., 2014, Journal of the Rubber Research Institute of Sri Lanka 94, 1-8; latex
diluted 1:2 with water). The chart is stored in `src/drc_ready_reckoner.py`. The temperature is
interpolated between the chart columns. The chart rises by a constant 0.02 per 10 Metrolac units,
so readings above 150 (this dataset has 155 – 183) follow the same straight line. The temperatures
(26.8 – 31.9 °C) are inside the chart range. The resulting DRC is 40.9 – 46.9 %.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Data understanding | `01_data_understanding.py` | overview and column summary of the raw data |
| 2. Target validation | `02_target_validation.py` | target and input column checks |
| 3. Data quality analysis | `03_data_quality_analysis.py` | issues found in the raw data |
| 4. Preprocessing | `04_preprocessing.py` | cleaned final dataset with every `drc` filled |
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
| Targets | `water_amount_L`, `prepared_1pct_formic_acid_solution_mL` | `coagulation_time_minutes` |
| Algorithm | XGBoost | RandomForest |

Training data range: DRC 40.9 – 46.9 %, latex volume 1.0 – 2.0 L, water 1.8 – 4.4 L, acid
solution 102 – 507 mL. Inputs outside this range are flagged as less reliable.

## Algorithm selection (5-fold cross validation on the training split, mean MAE)

| Algorithm | Model 1 — Water & Acid | Model 2 — Coagulation Time (min) |
|---|---|---|
| XGBoost | **3.11** | 18.46 |
| RandomForest | 3.25 | **18.09** |
| ExtraTrees | 3.25 | 18.39 |
| Ridge | 4.51 | 22.78 |
| LinearRegression | 4.52 | 22.79 |
| KNN | – | 18.45 |
| Huber | – | 22.01 |
| RANSAC | – | 23.13 |
| SVR | – | 26.66 |

The algorithm with the lowest error was selected for each model.

## Final accuracy

These are regression models, so they are measured with MAE, RMSE and R². Accuracy is given as
100 % − mean absolute percentage error.

Summary (test set, 375 records):

| | Model 1 — Water & Acid | Model 2 — Coagulation Time |
|---|---|---|
| Best algorithm | XGBoost | RandomForest |
| Accuracy | water 97.4 %, acid 97.5 % | 91.6 % |
| Average error | ±0.08 L water, ±5.5 mL acid | ±18 min |
| R² | water 0.979, acid 0.983 | 0.923 |

Detailed results:

| Target | Split | MAE | RMSE | R² |
|---|---|---|---|---|
| water_amount_L | validation | 0.07 L | 0.09 L | 0.979 |
| water_amount_L | **test** | **0.08 L** | **0.10 L** | **0.979** |
| prepared_1pct_formic_acid_solution_mL | validation | 6.81 mL | 8.57 mL | 0.976 |
| prepared_1pct_formic_acid_solution_mL | **test** | **5.54 mL** | **7.17 mL** | **0.983** |
| coagulation_time_minutes | validation | 19.8 min | 26.5 min | 0.870 |
| coagulation_time_minutes | **test** | **17.9 min** | **23.9 min** | **0.923** |

- Model 1 explains about 98 % of the variation in water and acid.
- Model 2 explains about 92 % of the variation in coagulation time, with an average error of
  about 18 minutes. The model works in minutes; predicted times are shown as hours and minutes
  (for example 121 minutes is shown as 2 hours 1 minute).
- Validation and test results are close, so the models do not overfit.

## Artifacts

The trained models, model card, input schema, metrics and plots are in `artifacts/`.
See `artifacts/README.md` for details and how to run a prediction.
