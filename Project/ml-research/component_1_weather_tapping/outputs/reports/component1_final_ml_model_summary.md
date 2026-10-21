# Component 1 - Final ML Model Summary

**Dataset provenance:** condition-matched farmer records (5 estates, Jun-Sep 2026) with Open-Meteo weather; DRC/Metrolac values normalized by the dataset author; latex price derived from farm-gate levels (Jun-Sep 2026 levels estimated).

Models were selected by cross-validation on the training partition and evaluated once on the
held-out test partition (stage 31). xgboost available at export time: `True`.

## Regression

| Task | Selected model | CV RMSE | Test MAE | Test RMSE | Test R2 | Within tol | Tolerance |
|---|---|---|---|---|---|---|---|
| drc_prediction | tuned_gradient_boosting | 1.021 | 0.837 | 1.019 | 0.770 | 61.5% | +/-1.0 |
| price_estimation | extra_trees | 7.958 | 6.701 | 8.139 | 0.901 | 45.1% | +/-5.0 |

## Classification

| Task | Selected model | CV macro-F1 | Test accuracy | Test macro-P | Test macro-R | Test macro-F1 |
|---|---|---|---|---|---|---|
| tapping_advisory | gradient_boosting | 0.968 | 0.934 | 0.944 | 0.919 | 0.925 |

## Notes

- Within-tolerance % is **not** accuracy and **not** R-squared.
- The split is row-wise, so most test estates also appear in training; see
  `outputs/reports/30_*_estate_grouped_cv.csv` for the unseen-estate estimate.
- Tapping labels come from a rainfall rule judged on rain that actually fell, while the model
  only sees the forecast issued a day earlier; its errors are real forecast misses. Labels
  are a proxy, not expert-verified tapping advice.
- Price is farm-gate latex price x DRC / 100. The model gets the estimated DRC (stage 04b),
  as in the app, so its error includes DRC estimation error. June-September 2026
  farm-gate levels are estimates.
- DRC values were normalized by the dataset author; they are not original field measurements.
- Collection-time prediction is removed from the project entirely.