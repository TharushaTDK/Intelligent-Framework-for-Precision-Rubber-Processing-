# Component 2 artifacts

This folder has the two trained models and the files you need to use or check them.
Everything here is produced by `src/32_export_final_models.py`, so if you retrain, run
that script again and this folder gets refreshed.

## What's in here

- `models/model_1_water_acid_best.joblib`: ExtraTrees model. Give it `drc` and `latex_volume_L`
  and it returns water (L) and prepared 1% formic acid solution (mL).
- `models/model_2_coagulation_time_best.joblib`: Ridge model. Takes `drc`, `latex_volume_L`,
  `water_amount_L` and `prepared_1pct_formic_acid_solution_mL`, returns coagulation time in hours.
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
| water_amount_L | validation | 0.32 | 0.42 | 0.997 |
| water_amount_L | test | 0.31 | 0.38 | 0.997 |
| prepared_1pct_formic_acid_solution_mL | validation | 4.45 | 6.47 | 0.999 |
| prepared_1pct_formic_acid_solution_mL | test | 4.24 | 5.53 | 0.999 |
| coagulation_time_hours | validation | 0.64 | 0.80 | 0.726 |
| coagulation_time_hours | test | 0.64 | 0.81 | 0.704 |

Model 1 is very accurate. Model 2 is off by about 40 minutes on average. DRC has by far
the biggest effect on coagulation time, and the other inputs only explain part of what's
left. Temperature and similar conditions probably account for some of the rest.

The dataset is synthetic, so these numbers show the pipeline works. They aren't proof
of how the models will do on real factory batches.
