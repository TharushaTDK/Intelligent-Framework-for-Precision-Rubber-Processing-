# Component 1 - Final ML Model Summary

**Dataset provenance: SYNTHETIC. No metric here is real-world evidence.**

Models were selected by cross-validation on the training partition and evaluated once on the
held-out test partition (stage 31). xgboost available at export time: `True`.

## Regression

| Task | Selected model | CV RMSE | Test MAE | Test RMSE | Test R2 | Within tol | Tolerance |
|---|---|---|---|---|---|---|---|
| drc_prediction | tuned_gradient_boosting | 0.848 | 0.686 | 0.863 | 0.801 | 77.3% | +/-1.0 |
| price_estimation | tuned_random_forest | 7.485 | 6.229 | 7.663 | 0.726 | 74.2% | +/-9.0 |

## Classification

| Task | Selected model | CV macro-F1 | Test accuracy | Test macro-P | Test macro-R | Test macro-F1 |
|---|---|---|---|---|---|---|
| tapping_advisory | tuned_gradient_boosting | 0.661 | 0.775 | 0.795 | 0.759 | 0.768 |

## Notes

- Within-tolerance % is **not** accuracy and **not** R-squared.
- The split is row-wise, so most test estates also appear in training; see
  `outputs/reports/30_*_estate_grouped_cv.csv` for the unseen-estate estimate.
- Tapping labels are rule-derived with simulated label noise; not expert advice.
- Collection-time prediction is removed from the project entirely.