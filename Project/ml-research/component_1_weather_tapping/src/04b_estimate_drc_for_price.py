"""Stage 04b: the DRC the price model gets as input.

In the app the price is worked out from the DRC model's estimate, never from a lab
DRC. Training the price model on the true DRC would only teach it the price formula
(farm-gate price x DRC / 100) and score a meaningless 100 %. So the price model's
latex_drc_pct input is replaced by an estimated DRC, made without peeking at the row:

  * training rows: 5-fold out-of-fold predictions inside the price training split
  * test rows:     a DRC model fitted on the whole price training split

The price target itself still uses the true DRC. Nothing from a test row is used to
estimate anything. Must run after stage 04 (splits).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from common import PROCESSED, REPORTS, REG_TASKS, ensure_dirs, load_raw, clean_basic
from train_candidate import SEED, CV_FOLDS, prep, regression_model, split_indices

ESTIMATED_DRC = PROCESSED / "estimated_drc_for_price.csv"
# Same family the DRC search settles on; its exact tuning does not matter much here.
DRC_FAMILY = "xgboost"


def drc_pipe(X):
    return Pipeline([("preprocess", prep(X, scale=False)), ("model", regression_model(DRC_FAMILY))])


def main():
    ensure_dirs()
    target, features, _ = REG_TASKS["drc_prediction"]
    df = clean_basic(load_raw())
    tr, te = split_indices("price_estimation")
    X, y = df[features], pd.to_numeric(df[target], errors="coerce")

    estimate = pd.Series(np.nan, index=df.index)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    for fit_pos, pred_pos in kf.split(tr):
        fit_rows, pred_rows = tr[fit_pos], tr[pred_pos]
        estimate.loc[pred_rows] = drc_pipe(X.loc[fit_rows]).fit(X.loc[fit_rows], y.loc[fit_rows]).predict(X.loc[pred_rows])
    estimate.loc[te] = drc_pipe(X.loc[tr]).fit(X.loc[tr], y.loc[tr]).predict(X.loc[te])

    out = pd.DataFrame({"row_index": df.index, "estimated_latex_drc_pct": estimate.round(3)})
    out = out[out["estimated_latex_drc_pct"].notna()]
    out.to_csv(ESTIMATED_DRC, index=False)

    lines = [
        "Stage 04b - estimated DRC used as the price model's input",
        f"DRC model family: {DRC_FAMILY}; features: {', '.join(features)}",
        f"train rows (out-of-fold): {len(tr)}; test rows (fitted on train only): {len(te)}",
        f"MAE vs true DRC, train out-of-fold: {mean_absolute_error(y.loc[tr], estimate.loc[tr]):.3f} pp",
        f"MAE vs true DRC, test: {mean_absolute_error(y.loc[te], estimate.loc[te]):.3f} pp",
    ]
    (REPORTS / "04b_estimated_drc_for_price.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
