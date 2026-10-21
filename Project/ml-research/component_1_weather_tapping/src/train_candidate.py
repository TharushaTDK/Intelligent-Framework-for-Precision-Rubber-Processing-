"""Candidate-model fitting utilities for Component 1.

Every candidate is fitted on the **stored training partition only** and evaluated
with k-fold cross-validation *inside that partition*. The held-out test partition
is never loaded here; it is reserved for stage 31.

Artefacts are written using the reference-project layout:

    artifacts/models/<kind>/<task>/{baseline|candidates}/<family>/
        model.joblib
        metadata.json

(Filenames are generic inside each family folder because the workspace path is
long and ``<family>_model.joblib`` would exceed the Windows MAX_PATH limit.)

The functions are reused by the per-family stage scripts (11-16, 20, 22-26) and by
the comparison / selection stages.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    SPLITS, REPORTS, PROCESSED, REG_TASKS, CLS_TASK, CLS_TARGET, CLS_FEATURES, CLS_CLASSES,
    DATASET_PROVENANCE, TARGET_WARNING,
    ensure_dirs, load_raw, clean_basic, write_json, role_dir, sha256_frame, sha256_file,
)

from sklearn.model_selection import train_test_split  # noqa: F401  (kept for API compatibility)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover - exercised only when xgboost is absent
    HAS_XGB = False

SEED = 340
CV_FOLDS = 5

BASELINE_REG = "dummy_mean"
BASELINE_CLS = "dummy_most_frequent"

REG_CANDIDATES = ["dummy_mean", "linear_regression", "random_forest", "extra_trees",
                  "gradient_boosting", "hist_gradient_boosting"] + (["xgboost"] if HAS_XGB else [])
CLS_CANDIDATES = ["dummy_most_frequent", "logistic_regression", "random_forest", "extra_trees",
                  "gradient_boosting"] + (["xgboost"] if HAS_XGB else [])

TOLERANCE = {task: spec[2] for task, spec in REG_TASKS.items()}


def prep(X, scale=False):
    cats = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    nums = [c for c in X.columns if c not in cats]
    nsteps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        nsteps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(nsteps), nums),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cats),
    ], remainder="drop")


def regression_model(name):
    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=SEED, n_jobs=1),
        "extra_trees": ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, random_state=SEED, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=SEED),
    }
    if name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed; install requirements.txt to train this candidate")
        return XGBRegressor(n_estimators=180, max_depth=3, learning_rate=.04, subsample=.9,
                            colsample_bytree=.9, objective="reg:squarederror",
                            random_state=SEED, n_jobs=1, verbosity=0)
    return models[name]


def classification_model(name):
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=1),
        "extra_trees": ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }
    if name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed; install requirements.txt to train this candidate")
        return XGBClassifier(n_estimators=180, max_depth=3, learning_rate=.04, subsample=.9,
                             colsample_bytree=.9, objective="multi:softprob", eval_metric="mlogloss",
                             random_state=SEED, n_jobs=1, verbosity=0)
    return models[name]


def _explicit_params(estimator):
    return {k: v for k, v in estimator.get_params(deep=False).items()
            if isinstance(v, (int, float, str, bool, type(None)))}


def split_indices(task):
    tr = pd.read_csv(SPLITS / f"{task}_train_indices.csv")["row_index"].to_numpy()
    te = pd.read_csv(SPLITS / f"{task}_test_indices.csv")["row_index"].to_numpy()
    return tr, te


def _reg_data(task):
    target, features, tol = REG_TASKS[task]
    df = clean_basic(load_raw())
    if task == "price_estimation":
        # The price model gets the DRC as the app would estimate it (stage 04b),
        # not the true DRC its target was computed from.
        est = pd.read_csv(PROCESSED / "estimated_drc_for_price.csv").set_index("row_index")
        df["latex_drc_pct"] = est["estimated_latex_drc_pct"].reindex(df.index)
    valid = pd.to_numeric(df[target], errors="coerce").notna()
    X = df.loc[valid, features].copy()
    y = pd.to_numeric(df.loc[valid, target], errors="coerce")
    good = y.notna()
    return X.loc[good], y.loc[good], tol


def _cls_data():
    df = clean_basic(load_raw())
    valid = df[CLS_TARGET].notna()
    X = df.loc[valid, CLS_FEATURES].copy()
    y = df.loc[valid, CLS_TARGET].astype(str)
    return X, y


def cv_regression_metrics(Xtr, ytr, name, tol, folds=CV_FOLDS):
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for tr, va in kf.split(Xtr):
        pipe = Pipeline([("preprocess", prep(Xtr.iloc[tr], scale=False)), ("model", regression_model(name))])
        pipe.fit(Xtr.iloc[tr], ytr.iloc[tr])
        pred = pipe.predict(Xtr.iloc[va])
        yv = np.asarray(ytr.iloc[va], float)
        rows.append({
            "MAE": float(mean_absolute_error(yv, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(yv, pred))),
            "R2": float(r2_score(yv, pred)),
            "within_tolerance_pct": float(100 * np.mean(np.abs(yv - pred) <= tol)),
        })
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in rows[0]}


def cv_classification_metrics(Xtr, ytr_enc, name, folds=CV_FOLDS):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for tr, va in skf.split(Xtr, ytr_enc):
        pipe = Pipeline([("preprocess", prep(Xtr.iloc[tr], scale=True)), ("model", classification_model(name))])
        pipe.fit(Xtr.iloc[tr], ytr_enc[tr])
        pred = pipe.predict(Xtr.iloc[va])
        yv = ytr_enc[va]
        rows.append({
            "accuracy": float(accuracy_score(yv, pred)),
            "precision_macro": float(precision_score(yv, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(yv, pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(yv, pred, average="macro", zero_division=0)),
        })
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in rows[0]}


def save_regression_candidate(name, stage="candidate"):
    """Fit one regression family on the train split of every regression task."""
    ensure_dirs()
    results = []
    for task in REG_TASKS:
        X, y, tol = _reg_data(task)
        tr, _ = split_indices(task)
        tr = [i for i in tr if i in X.index]
        Xtr, ytr = X.loc[tr], y.loc[tr]
        cv = cv_regression_metrics(Xtr, ytr, name, tol)
        pipe = Pipeline([("preprocess", prep(Xtr, scale=False)), ("model", regression_model(name))])
        pipe.fit(Xtr, ytr)
        role = "baseline" if name == BASELINE_REG else "candidates"
        dest = role_dir("regression", task, role, name)
        model_path = dest / "model.joblib"
        joblib.dump(pipe, model_path)
        meta = {
            "stage": stage,
            "task": task,
            "target": REG_TASKS[task][0],
            "model_family": name,
            "model_class": type(regression_model(name)).__name__,
            "role": role,
            "fit_scope": "TRAIN_ONLY",
            "validation_scope": "CV_ON_TRAIN_ONLY",
            "test_loaded": False,
            "hyperparameter_tuning_performed": False,
            "parameters": _explicit_params(regression_model(name)),
            "cv": {"folds": CV_FOLDS, "primary_metric": "RMSE", **cv},
            "tolerance": tol,
            "within_tolerance_note": "Within-tolerance % is NOT accuracy and is not R-squared.",
            "train_rows": int(len(ytr)),
            "n_features": len(REG_TASKS[task][1]),
            "feature_names": REG_TASKS[task][1],
            "dataset_provenance": DATASET_PROVENANCE,
            "x_train_sha256": sha256_frame(Xtr),
            "y_train_sha256": sha256_frame(ytr.to_frame()),
            "model_sha256": sha256_file(model_path),
        }
        write_json(dest / "metadata.json", meta)
        results.append(meta)
    return results


def save_classification_candidate(name, stage="candidate"):
    """Fit one classification family on the train split of the tapping task."""
    ensure_dirs()
    X, y = _cls_data()
    tr, _ = split_indices(CLS_TASK)
    tr = [i for i in tr if i in X.index]
    Xtr, ytr = X.loc[tr], y.loc[tr]
    enc = LabelEncoder()
    ytr_enc = enc.fit_transform(ytr)
    cv = cv_classification_metrics(Xtr, ytr_enc, name)
    pipe = Pipeline([("preprocess", prep(Xtr, scale=True)), ("model", classification_model(name))])
    pipe.fit(Xtr, ytr_enc)
    role = "baseline" if name == BASELINE_CLS else "candidates"
    dest = role_dir("classification", CLS_TASK, role, name)
    model_path = dest / "model.joblib"
    joblib.dump({"pipeline": pipe, "label_encoder": enc, "classes": list(enc.classes_),
                 "warning": TARGET_WARNING}, model_path)
    meta = {
        "stage": stage,
        "task": CLS_TASK,
        "target": CLS_TARGET,
        "model_family": name,
        "model_class": type(classification_model(name)).__name__,
        "role": role,
        "fit_scope": "TRAIN_ONLY",
        "validation_scope": "CV_ON_TRAIN_ONLY",
        "test_loaded": False,
        "hyperparameter_tuning_performed": False,
        "parameters": _explicit_params(classification_model(name)),
        "cv": {"folds": CV_FOLDS, "primary_metric": "f1_macro", **cv},
        "classes": list(enc.classes_),
        "train_rows": int(len(ytr)),
        "n_features": len(CLS_FEATURES),
        "feature_names": CLS_FEATURES,
        "dataset_provenance": DATASET_PROVENANCE,
        "target_warning": TARGET_WARNING,
        "x_train_sha256": sha256_frame(Xtr),
        "y_train_sha256": sha256_frame(ytr.to_frame()),
        "model_sha256": sha256_file(model_path),
    }
    write_json(dest / "metadata.json", meta)
    return meta


# Backwards-compatible entry points used by the per-family stage scripts.
def fit_regression_candidate(name):
    results = save_regression_candidate(name, stage=f"{name}_regression")
    for r in results:
        print(f"[regression] {r['task']} / {name}: CV RMSE={r['cv']['RMSE']['mean']:.4f} "
              f"MAE={r['cv']['MAE']['mean']:.4f} R2={r['cv']['R2']['mean']:.4f}")
    return results


def fit_classification_candidate(name):
    r = save_classification_candidate(name, stage=f"{name}_classification")
    print(f"[classification] {CLS_TASK} / {name}: CV macro-F1={r['cv']['f1_macro']['mean']:.4f} "
          f"accuracy={r['cv']['accuracy']['mean']:.4f}")
    return r
