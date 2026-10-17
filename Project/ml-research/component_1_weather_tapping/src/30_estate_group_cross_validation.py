"""Stage 30 (estate-grouped cross-validation).

The project's stored split is row-wise, so most test estates also appear in
training. This stage provides the *unseen-estate* generalisation estimate by
grouping fold assignment on ``estate_id`` for every task, using the same
candidate factories as the k-fold comparison.

Writes ``outputs/reports/30_<task>_estate_grouped_cv.csv`` and a summary JSON.
The held-out test partition is not loaded.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    REPORTS, REG_TASKS, CLS_TASK, ensure_dirs, load_raw, clean_basic, write_json,
)
from train_candidate import (  # noqa: E402
    prep, split_indices, _reg_data, _cls_data,
    regression_model, classification_model,
    REG_CANDIDATES, CLS_CANDIDATES, SEED,
    LabelEncoder, Pipeline, mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def _agg(rows):
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in rows[0]}


def grouped_regression():
    out = {}
    for task, (target, features, tol) in REG_TASKS.items():
        X, y, tol = _reg_data(task)
        tr, _ = split_indices(task)
        tr = [i for i in tr if i in X.index]
        Xtr, ytr = X.loc[tr], y.loc[tr]
        groups = clean_basic(load_raw()).loc[tr, "estate_id"].astype(str).to_numpy()
        gkf = GroupKFold(n_splits=5)
        rows = []
        for name in REG_CANDIDATES:
            fold_rows = []
            for a, b in gkf.split(Xtr, ytr, groups):
                pipe = Pipeline([("preprocess", prep(Xtr.iloc[a], scale=False)), ("model", regression_model(name))])
                pipe.fit(Xtr.iloc[a], ytr.iloc[a])
                pred = pipe.predict(Xtr.iloc[b]); yv = np.asarray(ytr.iloc[b], float)
                fold_rows.append({
                    "MAE": float(mean_absolute_error(yv, pred)),
                    "RMSE": float(np.sqrt(mean_squared_error(yv, pred))),
                    "R2": float(r2_score(yv, pred)),
                    "within_tolerance_pct": float(100 * np.mean(np.abs(yv - pred) <= tol)),
                })
            agg = _agg(fold_rows)
            rows.append({"task": task, "model": name, "grouping": "estate_id",
                         "cv_RMSE_mean": agg["RMSE"]["mean"], "cv_RMSE_std": agg["RMSE"]["std"],
                         "cv_MAE_mean": agg["MAE"]["mean"], "cv_R2_mean": agg["R2"]["mean"],
                         "cv_within_tolerance_pct_mean": agg["within_tolerance_pct"]["mean"]})
        df = pd.DataFrame(rows).sort_values("cv_RMSE_mean").reset_index(drop=True)
        df.to_csv(REPORTS / f"30_{task}_estate_grouped_cv.csv", index=False)
        out[task] = {"best_grouped_model": df.iloc[0]["model"],
                     "best_grouped_cv_RMSE": float(df.iloc[0]["cv_RMSE_mean"]),
                     "models": df.to_dict("records")}
    return out


def grouped_classification():
    X, y = _cls_data()
    tr, _ = split_indices(CLS_TASK)
    tr = [i for i in tr if i in X.index]
    Xtr, ytr = X.loc[tr], y.loc[tr]
    groups = clean_basic(load_raw()).loc[tr, "estate_id"].astype(str).to_numpy()
    enc = LabelEncoder(); ytr_enc = enc.fit_transform(ytr)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name in CLS_CANDIDATES:
        fold_rows = []
        for a, b in sgkf.split(Xtr, ytr_enc, groups):
            pipe = Pipeline([("preprocess", prep(Xtr.iloc[a], scale=True)), ("model", classification_model(name))])
            pipe.fit(Xtr.iloc[a], ytr_enc[a])
            pred = pipe.predict(Xtr.iloc[b]); yv = ytr_enc[b]
            fold_rows.append({"accuracy": float(accuracy_score(yv, pred)),
                              "precision_macro": float(precision_score(yv, pred, average="macro", zero_division=0)),
                              "recall_macro": float(recall_score(yv, pred, average="macro", zero_division=0)),
                              "f1_macro": float(f1_score(yv, pred, average="macro", zero_division=0))})
        agg = _agg(fold_rows)
        rows.append({"task": CLS_TASK, "model": name, "grouping": "estate_id",
                     "cv_macro_f1_mean": agg["f1_macro"]["mean"], "cv_macro_f1_std": agg["f1_macro"]["std"],
                     "cv_accuracy_mean": agg["accuracy"]["mean"],
                     "cv_macro_recall_mean": agg["recall_macro"]["mean"]})
    df = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    df.to_csv(REPORTS / f"30_{CLS_TASK}_estate_grouped_cv.csv", index=False)
    return {CLS_TASK: {"best_grouped_model": df.iloc[0]["model"],
                       "best_grouped_cv_macro_f1": float(df.iloc[0]["cv_macro_f1_mean"]),
                       "models": df.to_dict("records")}}


def main():
    ensure_dirs()
    summary = {
        "stage": "30_estate_group_cross_validation",
        "purpose": "Unseen-estate generalisation estimate. Folds are grouped on estate_id so an "
                   "estate never appears in both the fit and the evaluation fold.",
        "dataset_provenance": "synthetic; not real-world evidence",
        "test_loaded": False,
        "results": {},
    }
    summary["results"].update(grouped_regression())
    summary["results"].update(grouped_classification())
    write_json(REPORTS / "30_estate_group_cross_validation_summary.json", summary)
    for task, r in summary["results"].items():
        print(f"[{task}] estate-grouped best: {r['best_grouped_model']}")
    print("Stage 30 estate-grouped CV complete.")


if __name__ == "__main__":
    main()
