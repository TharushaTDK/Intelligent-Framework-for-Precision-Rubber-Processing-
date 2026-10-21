# Component 2 artifacts

This folder has the two trained models and the files you need to use or check them.
Everything here is produced by `src/32_export_final_models.py`, so if you retrain, run
that script again and this folder gets refreshed.

## What's in here

- `models/model_1_water_acid_best.joblib`: XGBoost model. Give it `drc` and `latex_volume_L`
  and it returns water (L) and prepared 1% formic acid solution (mL).
- `models/model_2_coagulation_time_best.joblib`: RandomForest model. Takes `drc`, `latex_volume_L`,
  `water_amount_L` and `prepared_1pct_formic_acid_solution_mL`, returns coagulation time in minutes, also shown as hours and minutes.
- `model_card.json`: algorithm, inputs, targets, CV score, validation and test scores,
  library versions and a SHA256 hash for each model file.
- `input_schema.json`: min, max, mean and median of every input in the training data.
- `requirements-lock.txt`: exact library versions used for training. Joblib files can fail
  to load on a different scikit-learn version, so install these if you get load errors.
- `metrics/`: validation and test scores, test predictions vs. actual values, the algorithm
  comparison tables and SHAP feature importance.
- `plots/`: predicted vs. actual and residual plots for each target, plus SHAP bar charts.
- `predict.py` and `sample_input.csv`: a quick way to run both models.

## Trying it out

```
cd artifacts
python predict.py sample_input.csv
```

Model 1 suggests the water and acid, then model 2 estimates the coagulation time.
Rows with `drc` or latex volume outside the training range get flagged, because
predictions there are less reliable.

## Results

Trained on 1,750 batches, checked on 375 validation and 375 test batches.

| Target | Split | MAE | RMSE | R² |
|---|---|---|---|---|
| water_amount_L | validation | 0.07 | 0.09 | 0.979 |
| water_amount_L | test | 0.08 | 0.10 | 0.979 |
| prepared_1pct_formic_acid_solution_mL | validation | 6.81 | 8.57 | 0.976 |
| prepared_1pct_formic_acid_solution_mL | test | 5.54 | 7.17 | 0.983 |
| coagulation_time_minutes | validation | 19.8 | 26.5 | 0.870 |
| coagulation_time_minutes | test | 17.9 | 23.9 | 0.923 |

Model 1 explains about 98 % of the variation in water and acid. Model 2 is off by about
18 minutes on average and explains about 92 % of the variation in coagulation time.

The dataset is synthetic, so these numbers show the pipeline works. They aren't proof
of how the models will do on real factory batches.
