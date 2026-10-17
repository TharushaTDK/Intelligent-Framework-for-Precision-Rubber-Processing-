"""Stage 18 / 28 (hyperparameter tuning): tune a shortlist of families with
grid search, evaluated by cross-validation **on the training partition only**.

Results are written to ``artifacts/models/<kind>/<task>/tuned/<family>/`` and
summarised in ``outputs/reports/tuned_model_comparison.csv``. The held-out test
partition is never loaded.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    REG_TASKS, CLS_TASK, CLS_TARGET, CLS_FEATURES, DATASET_PROVENANCE, TARGET_WARNING,
    REPORTS, ensure_dirs, write_json, role_dir, sha256_frame, sha256_file,
)
from train_candidate import (  # noqa: E402
    prep, split_indices, _reg_data, _cls_data, _explicit_params,
    RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier,
    GradientBoostingClassifier, LabelEncoder, Pipeline,
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    KFold, StratifiedKFold, SEED, HAS_XGB,
)

from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

REGRESSION_SHORTLIST = ["random_forest", "gradient_boosting"] + (["xgboost"] if HAS_XGB else [])
CLASSIFICATION_SHORTLIST = ["random_forest", "gradient_boosting"] + (["xgboost"] if HAS_XGB else [])

REG_GRIDS = {
    "random_forest": {"model__n_estimators": [100], "model__max_depth": [None, 8],
                      "model__min_samples_leaf": [2, 4]},
    "gradient_boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.06],
                          "model__max_depth": [2, 3]},
    "xgboost": {"model__n_estimators": [100], "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.06]},
}
CLS_GRIDS = dict(REG_GRIDS)


def _fresh(family, kind):
    if kind == "regression":
        base = {"random_forest": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1),
                "gradient_boosting": GradientBoostingRegressor(random_state=SEED)}
        if HAS_XGB:
            from xgboost import XGBRegressor
            base["xgboost"] = XGBRegressor(objective="reg:squarederror", random_state=SEED, n_jobs=1, verbosity=0)
    else:
        base = {"random_forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=1),
                "gradient_boosting": GradientBoostingClassifier(random_state=SEED)}
        if HAS_XGB:
            from xgboost import XGBClassifier
            base["xgboost"] = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                                            random_state=SEED, n_jobs=1, verbosity=0)
    return base[family]


def _cv_regression(pipeline, X, y, tol):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for tr, va in kf.split(X):
        p = clone(pipeline); p.fit(X.iloc[tr], y.iloc[tr]); pred = p.predict(X.iloc[va])
        yv = np.asarray(y.iloc[va], float)
        rows.append({"MAE": float(mean_absolute_error(yv, pred)),
                     "RMSE": float(np.sqrt(mean_squared_error(yv, pred))),
                     "R2": float(r2_score(yv, pred)),
                     "within_tolerance_pct": float(100 * np.mean(np.abs(yv - pred) <= tol))})
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in rows[0]}


def _cv_classification(pipeline, X, y_enc):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for tr, va in skf.split(X, y_enc):
        p = clone(pipeline); p.fit(X.iloc[tr], y_enc[tr]); pred = p.predict(X.iloc[va]); yv = y_enc[va]
        rows.append({"accuracy": float(accuracy_score(yv, pred)),
                     "precision_macro": float(precision_score(yv, pred, average="macro", zero_division=0)),
                     "recall_macro": float(recall_score(yv, pred, average="macro", zero_division=0)),
                     "f1_macro": float(f1_score(yv, pred, average="macro", zero_division=0))})
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in rows[0]}


def tune_regression():
    results = []
    for task, (target, features, tol) in REG_TASKS.items():
        X, y, tol = _reg_data(task)
        tr, _ = split_indices(task)
        tr = [i for i in tr if i in X.index]
        Xtr, ytr = X.loc[tr], y.loc[tr]
        for family in REGRESSION_SHORTLIST:
            pipe = Pipeline([("preprocess", prep(Xtr, scale=False)), ("model", _fresh(family, "regression"))])
            search = GridSearchCV(pipe, REG_GRIDS[family], scoring="neg_root_mean_squared_error",
                                  cv=3, n_jobs=1, refit=True)
            search.fit(Xtr, ytr)
            cv = _cv_regression(search.best_estimator_, Xtr, ytr, tol)
            dest = role_dir("regression", task, "tuned", family)
            model_path = dest / "model.joblib"
            joblib.dump(search.best_estimator_, model_path)
            meta = {
                "stage": "18_regression_tuning", "task": task, "target": target,
                "model_family": family, "model_name": f"tuned_{family}", "role": "tuned",
                "model_class": type(search.best_estimator_.named_steps["model"]).__name__,
                "fit_scope": "TRAIN_ONLY", "validation_scope": "CV_ON_TRAIN_ONLY",
                "test_loaded": False, "hyperparameter_tuning_performed": True,
                "search": {"method": "GridSearchCV", "cv_folds": 3,
                           "scoring": "neg_root_mean_squared_error", "grid": REG_GRIDS[family]},
                "best_params": search.best_params_, "best_search_score_cv_rmse": float(-search.best_score_),
                "cv": {"folds": 5, "primary_metric": "RMSE", **cv},
                "parameters": _explicit_params(search.best_estimator_.named_steps["model"]),
                "tolerance": tol, "train_rows": int(len(ytr)),
                "feature_names": features, "dataset_provenance": DATASET_PROVENANCE,
                "x_train_sha256": sha256_frame(Xtr), "y_train_sha256": sha256_frame(ytr.to_frame()),
                "model_sha256": sha256_file(model_path),
            }
            write_json(dest / "metadata.json", meta)
            results.append({"kind": "regression", "task": task, "model": f"tuned_{family}",
                            "cv_primary_metric": "RMSE", "cv_primary_mean": cv["RMSE"]["mean"],
                            "best_params": str(search.best_params_)})
    return results


def tune_classification():
    X, y = _cls_data()
    tr, _ = split_indices(CLS_TASK)
    tr = [i for i in tr if i in X.index]
    Xtr, ytr = X.loc[tr], y.loc[tr]
    enc = LabelEncoder(); ytr_enc = enc.fit_transform(ytr)
    results = []
    for family in CLASSIFICATION_SHORTLIST:
        pipe = Pipeline([("preprocess", prep(Xtr, scale=True)), ("model", _fresh(family, "classification"))])
        search = GridSearchCV(pipe, CLS_GRIDS[family], scoring="f1_macro", cv=3, n_jobs=1, refit=True)
        search.fit(Xtr, ytr_enc)
        cv = _cv_classification(search.best_estimator_, Xtr, ytr_enc)
        dest = role_dir("classification", CLS_TASK, "tuned", family)
        model_path = dest / "model.joblib"
        joblib.dump({"pipeline": search.best_estimator_, "label_encoder": enc,
                     "classes": list(enc.classes_), "warning": TARGET_WARNING}, model_path)
        meta = {
            "stage": "28_classification_tuning", "task": CLS_TASK, "target": CLS_TARGET,
            "model_family": family, "model_name": f"tuned_{family}", "role": "tuned",
            "model_class": type(search.best_estimator_.named_steps["model"]).__name__,
            "fit_scope": "TRAIN_ONLY", "validation_scope": "CV_ON_TRAIN_ONLY",
            "test_loaded": False, "hyperparameter_tuning_performed": True,
            "search": {"method": "GridSearchCV", "cv_folds": 3, "scoring": "f1_macro",
                       "grid": CLS_GRIDS[family]},
            "best_params": search.best_params_, "best_search_score_cv_macro_f1": float(search.best_score_),
            "cv": {"folds": 5, "primary_metric": "f1_macro", **cv},
            "parameters": _explicit_params(search.best_estimator_.named_steps["model"]),
            "classes": list(enc.classes_), "train_rows": int(len(ytr)),
            "feature_names": CLS_FEATURES, "dataset_provenance": DATASET_PROVENANCE,
            "target_warning": TARGET_WARNING,
            "x_train_sha256": sha256_frame(Xtr), "y_train_sha256": sha256_frame(ytr.to_frame()),
            "model_sha256": sha256_file(model_path),
        }
        write_json(dest / "metadata.json", meta)
        results.append({"kind": "classification", "task": CLS_TASK, "model": f"tuned_{family}",
                        "cv_primary_metric": "f1_macro", "cv_primary_mean": cv["f1_macro"]["mean"],
                        "best_params": str(search.best_params_)})
    return results


def main():
    ensure_dirs()
    results = tune_regression() + tune_classification()
    df = pd.DataFrame(results)
    df.to_csv(REPORTS / "tuned_model_comparison.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nTuned {len(results)} candidates. xgboost_available={HAS_XGB}")


if __name__ == "__main__":
    main()
