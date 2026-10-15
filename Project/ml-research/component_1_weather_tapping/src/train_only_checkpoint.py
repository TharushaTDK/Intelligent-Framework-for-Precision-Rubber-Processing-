"""Training-only checkpoint harness for Component 1.

Runs every candidate model for the three tasks with a leakage-safe protocol:

* one consistent, deterministic split reused from data/splits/*.csv;
* preprocessing fitted on the training partition only (inside a Pipeline);
* model selection by cross-validation *within the training partition*;
* the held-out test partition is touched exactly once, for the selected model.

It never writes to the project's outputs/ or artifacts/ directories. Set
RUBBERSENSE_TRAIN_ONLY_DIR to a temporary directory and run:

    python src/train_only_checkpoint.py

All checkpoints land under $RUBBERSENSE_TRAIN_ONLY_DIR/train_only_checkpoints/.
"""

from pathlib import Path
import sys, json, os
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, SPLITS, REPORTS, load_raw, clean_basic, write_json)
from train_candidate import prep, regression_model, classification_model, HAS_XGB

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)

SEED = 340
RNG = np.random.default_rng(SEED)

REG_TASKS = {
    "drc_prediction": ("latex_drc_pct",
        ["temperature_c", "relative_humidity_pct", "rainfall_last_6h_mm", "forecast_rain_next_6h_mm",
         "rainfall_last_3days_mm", "wind_speed_kmh", "soil_moisture_pct", "soil_ph", "estate_id",
         "tree_count", "tapping_system", "days_since_last_tap", "previous_collection_yield_litres",
         "month", "day_of_year"], 1.0),
    "price_estimation": ("price_lkr_per_kg",
        ["latex_drc_pct", "temperature_c", "relative_humidity_pct", "rainfall_last_6h_mm",
         "soil_moisture_pct", "soil_ph", "tree_count", "tapping_system", "month", "day_of_year"], 9.0),
}
CLS_TASK = "tapping_advisory"
CLS_TARGET = "tapping_suitability_proxy"
CLS_FEATURES = ["temperature_c", "relative_humidity_pct", "rainfall_last_6h_mm", "forecast_rain_next_6h_mm",
                "rainfall_last_3days_mm", "wind_speed_kmh", "soil_moisture_pct", "soil_ph", "tree_count",
                "tapping_system", "days_since_last_tap", "month", "day_of_year"]

REG_CANDIDATES = ["dummy_mean", "linear_regression", "random_forest", "extra_trees",
                  "gradient_boosting", "hist_gradient_boosting"] + (["xgboost"] if HAS_XGB else [])
CLS_CANDIDATES = ["dummy_most_frequent", "logistic_regression", "random_forest", "extra_trees",
                  "gradient_boosting"] + (["xgboost"] if HAS_XGB else [])

TOL_JUSTIFICATION = ("Tolerance bands are the project-declared illustrative evaluation bands: "
                     "DRC +/-1.0 percentage point and price +/-LKR 9/kg. They are not validated "
                     "acceptance thresholds. The within-tolerance percentage is distinct from "
                     "classification accuracy and is never reported as accuracy.")

WARNINGS = [
    "Dataset is SYNTHETIC; estate_id values are not verified estates and price/DRC values are simulated.",
    "tapping_suitability_proxy is rule-derived with ~30% simulated label noise (seed 340); NOT expert-verified agronomic ground truth.",
    "Collection-time prediction is intentionally excluded; no collection-time target or model exists.",
    "Default split is row-wise, so most test estates also appear in training: results are for NEW OBSERVATIONS FROM KNOWN ESTATES unless the estate-grouped CV is used.",
    "latex_drc_pct is used as a price input; it is a physical price covariate, not derived from price_lkr_per_kg.",
]


def split_indices(task):
    tr = pd.read_csv(SPLITS / f"{task}_train_indices.csv")["row_index"].to_numpy()
    te = pd.read_csv(SPLITS / f"{task}_test_indices.csv")["row_index"].to_numpy()
    return tr, te


def reg_metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


def cls_metrics(y, p):
    return {"accuracy": float(accuracy_score(y, p)),
            "precision_macro": float(precision_score(y, p, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y, p, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y, p, average="macro", zero_division=0))}


def cv_regression(X, y, name, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for tr, va in kf.split(X):
        pipe = Pipeline([("preprocess", prep(X.iloc[tr], scale=False)), ("model", regression_model(name))])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[va])
        m = reg_metrics(y.iloc[va], pred)
        m["within_tolerance_pct"] = float(100 * np.mean(np.abs(np.asarray(y.iloc[va], float) - pred) <= _TOL))
        rows.append(m)
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in rows[0]}


def cv_regression_grouped(X, y, groups, name, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for tr, va in gkf.split(X, y, groups):
        pipe = Pipeline([("preprocess", prep(X.iloc[tr], scale=False)), ("model", regression_model(name))])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[va])
        m = reg_metrics(y.iloc[va], pred)
        m["within_tolerance_pct"] = float(100 * np.mean(np.abs(np.asarray(y.iloc[va], float) - pred) <= _TOL))
        rows.append(m)
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in rows[0]}


def cv_classification(X, y, name, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for tr, va in skf.split(X, y):
        pipe = Pipeline([("preprocess", prep(X.iloc[tr], scale=True)), ("model", classification_model(name))])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[va])
        rows.append(cls_metrics(y.iloc[va], pred))
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in rows[0]}


_TOL = None


def run_regression_task(task):
    target, features, tol = REG_TASKS[task]
    df = clean_basic(load_raw())
    valid = pd.to_numeric(df[target], errors="coerce").notna()
    X_all = df.loc[valid, features].copy()
    y_all = pd.to_numeric(df.loc[valid, target], errors="coerce")
    tr, te = split_indices(task)
    tr = [i for i in tr if i in X_all.index]; te = [i for i in te if i in X_all.index]
    Xtr, ytr = X_all.loc[tr], y_all.loc[tr]
    Xte, yte = X_all.loc[te], y_all.loc[te]
    groups = df.loc[tr, "estate_id"].astype(str).to_numpy()

    global _TOL; _TOL = tol
    task_out = {"task": task, "target": target, "type": "regression", "tolerance": tol,
                "tolerance_justification": TOL_JUSTIFICATION,
                "n_train": len(Xtr), "n_test": len(Xte), "n_features": len(features),
                "dataset_provenance": "synthetic", "candidates": {}, "skipped": [], "warnings": WARNINGS}

    for name in REG_CANDIDATES:
        entry = {"model": name, "training_completed": False}
        try:
            cv = cv_regression(Xtr, ytr, name)
            gcv = cv_regression_grouped(Xtr, ytr, groups, name)
            pipe = Pipeline([("preprocess", prep(Xtr, scale=False)), ("model", regression_model(name))])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            test = reg_metrics(yte, pred)
            test["within_tolerance_pct"] = float(100 * np.mean(np.abs(np.asarray(yte, float) - pred) <= tol))
            entry.update({"training_completed": True, "cv_kfold": cv, "cv_estate_grouped": gcv,
                          "test_metrics": test, "n_train": len(Xtr), "n_test": len(Xte),
                          "cv_folds": 5, "dataset_provenance": "synthetic"})
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        task_out["candidates"][name] = entry

    completed = {k: v for k, v in task_out["candidates"].items() if v.get("training_completed")}
    if completed:
        best = min(completed, key=lambda k: completed[k]["cv_kfold"]["RMSE"]["mean"])
        task_out["selected_model"] = best
        task_out["selection_basis"] = "lowest mean 5-fold CV RMSE on the training partition"
        task_out["selected_test_metrics"] = completed[best]["test_metrics"]
    return task_out


def run_classification_task():
    df = clean_basic(load_raw())
    valid = df[CLS_TARGET].notna()
    X_all = df.loc[valid, CLS_FEATURES].copy()
    y_all = df.loc[valid, CLS_TARGET].astype(str)
    tr, te = split_indices(CLS_TASK)
    Xtr, ytr = X_all.loc[tr], y_all.loc[tr]
    Xte, yte = X_all.loc[te], y_all.loc[te]

    task_out = {"task": CLS_TASK, "target": CLS_TARGET, "type": "classification",
                "n_train": len(Xtr), "n_test": len(Xte), "n_features": len(CLS_FEATURES),
                "dataset_provenance": "synthetic",
                "label_provenance": "rule-derived proxy with ~30% simulated label noise (seed 340); NOT expert-verified",
                "classes": sorted(y_all.unique().tolist()), "candidates": {}, "skipped": [], "warnings": WARNINGS}

    for name in CLS_CANDIDATES:
        entry = {"model": name, "training_completed": False}
        try:
            cv = cv_classification(Xtr, ytr, name)
            pipe = Pipeline([("preprocess", prep(Xtr, scale=True)), ("model", classification_model(name))])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            test = cls_metrics(yte, pred)
            cm = confusion_matrix(yte, pred, labels=task_out["classes"])
            per_class = classification_report(yte, pred, output_dict=True, zero_division=0)
            entry.update({"training_completed": True, "cv_stratified": cv, "test_metrics": test,
                          "confusion_matrix": cm.tolist(),
                          "confusion_matrix_labels": task_out["classes"],
                          "per_class": {c: per_class[c] for c in task_out["classes"] if c in per_class},
                          "n_train": len(Xtr), "n_test": len(Xte), "cv_folds": 5,
                          "dataset_provenance": "synthetic"})
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        task_out["candidates"][name] = entry

    completed = {k: v for k, v in task_out["candidates"].items() if v.get("training_completed")}
    if completed:
        best = max(completed, key=lambda k: completed[k]["cv_stratified"]["f1_macro"]["mean"])
        task_out["selected_model"] = best
        task_out["selection_basis"] = "highest mean 5-fold stratified CV macro-F1 on the training partition"
        task_out["selected_test_metrics"] = completed[best]["test_metrics"]
    return task_out


def main():
    out_dir = (Path(os.environ["RUBBERSENSE_TRAIN_ONLY_DIR"]) / "train_only_checkpoints"
               if os.environ.get("RUBBERSENSE_TRAIN_ONLY_DIR") else REPORTS / "train_only_checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["drc_prediction"] = run_regression_task("drc_prediction")
    results["price_estimation"] = run_regression_task("price_estimation")
    results["tapping_advisory"] = run_classification_task()

    checkpoint = {"project": "RubberSense Component 1", "phase": "training-only checkpoint",
                  "seed": SEED, "dataset_provenance": "synthetic",
                  "rng_note": "All model fits use random_state=340; CV shuffles use random_state=340.",
                  "collection_time": "removed (not trained, compared, reported or exported)",
                  "tolerance_bands": "DRC +/-1.0 pp; price +/-LKR 9/kg (project-declared, illustrative)",
                  "results": results}
    write_json(out_dir / "training_only_checkpoint.json", checkpoint)

    rows = []
    for task, t in results.items():
        for name, c in t["candidates"].items():
            r = {"task": task, "model": name, "training_completed": c.get("training_completed")}
            if c.get("training_completed"):
                if t["type"] == "regression":
                    cv = c["cv_kfold"]; tm = c["test_metrics"]
                    r.update({"cv_MAE_mean": cv["MAE"]["mean"], "cv_RMSE_mean": cv["RMSE"]["mean"],
                              "cv_R2_mean": cv["R2"]["mean"], "cv_within_tol_pct_mean": cv["within_tolerance_pct"]["mean"],
                              "cv_estate_grouped_RMSE_mean": c["cv_estate_grouped"]["RMSE"]["mean"],
                              "test_MAE": tm["MAE"], "test_RMSE": tm["RMSE"], "test_R2": tm["R2"],
                              "test_within_tol_pct": tm["within_tolerance_pct"]})
                else:
                    cv = c["cv_stratified"]; tm = c["test_metrics"]
                    r.update({"cv_accuracy_mean": cv["accuracy"]["mean"], "cv_macro_precision_mean": cv["precision_macro"]["mean"],
                              "cv_macro_recall_mean": cv["recall_macro"]["mean"], "cv_macro_f1_mean": cv["f1_macro"]["mean"],
                              "test_accuracy": tm["accuracy"], "test_macro_precision": tm["precision_macro"],
                              "test_macro_recall": tm["recall_macro"], "test_macro_f1": tm["f1_macro"]})
            else:
                r["error"] = c.get("error")
            rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "training_only_checkpoint.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nCheckpoint written to {out_dir}")


if __name__ == "__main__":
    main()
