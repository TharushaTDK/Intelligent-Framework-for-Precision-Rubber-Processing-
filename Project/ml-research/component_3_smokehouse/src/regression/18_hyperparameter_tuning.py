from pathlib import Path
import hashlib
import json
import math
import os
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "\nXGBoost is required for Stage 18.\n"
        "Install it inside the active virtual environment with:\n"
        "    python -m pip install xgboost\n"
        "Then rerun Stage 18.\n"
    ) from exc


# ============================================================
# COMPONENT 3 - STAGE 18: REGRESSION HYPERPARAMETER TUNING
# ============================================================
#
# Shortlisted families from Stage 17:
#   1) Gradient Boosting
#   2) XGBoost
#
# Method:
#   - fit every configuration on TRAIN only
#   - evaluate configurations on VALIDATION only
#   - primary tuning metric: Validation MAE
#   - tie-breakers: RMSE, then R²
#   - TEST is not loaded
#
# Stage 18 produces tuned development candidates only.
# Stage 19 performs final development-set model selection.
# Stage 30 performs group-aware robustness analysis.
# Stage 31 performs one-time final Test evaluation.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_READY_DIR = ROOT / "data" / "model_ready"
SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "outputs" / "reports"

MODEL_DIR = ROOT / "artifacts" / "models" / "regression" / "tuned"
GB_MODEL_DIR = MODEL_DIR / "gradient_boosting"
XGB_MODEL_DIR = MODEL_DIR / "xgboost"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
GB_MODEL_DIR.mkdir(parents=True, exist_ok=True)
XGB_MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = MODEL_READY_DIR / "Component3_X_Train.csv"
X_VALIDATION_PATH = MODEL_READY_DIR / "Component3_X_Validation.csv"
Y_TRAIN_PATH = MODEL_READY_DIR / "Component3_y_regression_Train.csv"
Y_VALIDATION_PATH = MODEL_READY_DIR / "Component3_y_regression_Validation.csv"
META_TRAIN_PATH = MODEL_READY_DIR / "Component3_metadata_Train.csv"
META_VALIDATION_PATH = MODEL_READY_DIR / "Component3_metadata_Validation.csv"

TEST_PATH = SPLIT_DIR / "Component3_Test.csv"

SHORTLIST_PATH = REPORT_DIR / "17_regression_tuning_shortlist.csv"
STAGE13_METRICS_PATH = REPORT_DIR / "13_gradient_boosting_regression_metrics.csv"
STAGE15_METRICS_PATH = REPORT_DIR / "15_xgboost_regression_metrics.csv"

REPORT_PATH = REPORT_DIR / "18_regression_tuning_report.txt"
CHECKS_PATH = REPORT_DIR / "18_regression_tuning_checks.csv"
TRIALS_PATH = REPORT_DIR / "18_regression_tuning_trials.csv"
BEST_RESULTS_PATH = REPORT_DIR / "18_regression_tuning_best_results.csv"
IMPROVEMENT_PATH = REPORT_DIR / "18_regression_tuning_improvement_comparison.csv"
LAYER_PATH = REPORT_DIR / "18_tuned_validation_by_layer.csv"
BATCH_PATH = REPORT_DIR / "18_tuned_validation_by_batch.csv"
PREDICTIONS_PATH = REPORT_DIR / "18_tuned_validation_predictions.csv"
SUMMARY_PATH = REPORT_DIR / "18_regression_tuning_summary.json"

GB_MODEL_PATH = GB_MODEL_DIR / "tuned_gradient_boosting_regressor.joblib"
GB_METADATA_PATH = GB_MODEL_DIR / "tuned_gradient_boosting_metadata.json"
XGB_MODEL_PATH = XGB_MODEL_DIR / "tuned_xgboost_regressor.joblib"
XGB_METADATA_PATH = XGB_MODEL_DIR / "tuned_xgboost_metadata.json"


# ============================================================
# 2. EXPECTED STRUCTURE
# ============================================================

EXPECTED_TRAIN_ROWS = 16246
EXPECTED_VALIDATION_ROWS = 4159
EXPECTED_FEATURES = 20
EXPECTED_SHORTLIST = {"Gradient_Boosting", "XGBoost"}

# Internal smoke-test switch used only to test script mechanics.
# Normal local execution does not set this variable.
SMOKE_TEST = os.environ.get("STAGE18_SMOKE_TEST", "0") == "1"


# ============================================================
# 3. START + REQUIRED FILES
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 18: REGRESSION HYPERPARAMETER TUNING")
print("=" * 78)

required_files = [
    X_TRAIN_PATH,
    X_VALIDATION_PATH,
    Y_TRAIN_PATH,
    Y_VALIDATION_PATH,
    META_TRAIN_PATH,
    META_VALIDATION_PATH,
    TEST_PATH,
    SHORTLIST_PATH,
    STAGE13_METRICS_PATH,
    STAGE15_METRICS_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

print("\nRequired development files found.")
print("Final Test dataset existence verified, but contents were NOT loaded.")


# ============================================================
# 4. LOAD DEVELOPMENT DATA ONLY
# ============================================================

X_train = pd.read_csv(X_TRAIN_PATH)
X_validation = pd.read_csv(X_VALIDATION_PATH)
y_train_df = pd.read_csv(Y_TRAIN_PATH)
y_validation_df = pd.read_csv(Y_VALIDATION_PATH)
meta_train = pd.read_csv(META_TRAIN_PATH)
meta_validation = pd.read_csv(META_VALIDATION_PATH)

shortlist_df = pd.read_csv(SHORTLIST_PATH)
stage13_metrics = pd.read_csv(STAGE13_METRICS_PATH)
stage15_metrics = pd.read_csv(STAGE15_METRICS_PATH)

y_train = pd.to_numeric(
    y_train_df["Remaining_Drying_Time"],
    errors="raise",
).to_numpy()

y_validation = pd.to_numeric(
    y_validation_df["Remaining_Drying_Time"],
    errors="raise",
).to_numpy()


# ============================================================
# 5. INPUT VALIDATION
# ============================================================

if X_train.shape != (EXPECTED_TRAIN_ROWS, EXPECTED_FEATURES):
    raise RuntimeError(f"Unexpected X_train shape: {X_train.shape}")

if X_validation.shape != (EXPECTED_VALIDATION_ROWS, EXPECTED_FEATURES):
    raise RuntimeError(f"Unexpected X_validation shape: {X_validation.shape}")

if len(y_train) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError("Unexpected y_train row count.")

if len(y_validation) != EXPECTED_VALIDATION_ROWS:
    raise RuntimeError("Unexpected y_validation row count.")

if len(meta_train) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError("Unexpected Train metadata row count.")

if len(meta_validation) != EXPECTED_VALIDATION_ROWS:
    raise RuntimeError("Unexpected Validation metadata row count.")

if list(X_train.columns) != list(X_validation.columns):
    raise RuntimeError(
        "Train and Validation transformed feature columns differ.")

if not np.isfinite(X_train.to_numpy(dtype=float)).all():
    raise RuntimeError("X_train contains missing/non-finite values.")

if not np.isfinite(X_validation.to_numpy(dtype=float)).all():
    raise RuntimeError("X_validation contains missing/non-finite values.")

if "Model" not in shortlist_df.columns:
    raise RuntimeError(
        "Stage-17 tuning shortlist is missing the Model column.")

actual_shortlist = set(shortlist_df["Model"].tolist())

if actual_shortlist != EXPECTED_SHORTLIST:
    raise RuntimeError(
        "Unexpected Stage-17 tuning shortlist.\n"
        f"Expected: {sorted(EXPECTED_SHORTLIST)}\n"
        f"Found: {sorted(actual_shortlist)}"
    )


# ============================================================
# 6. FINGERPRINTS + UNTUNED RESULTS
# ============================================================

def sha256_file(path):
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


x_train_sha256 = sha256_file(X_TRAIN_PATH)
x_validation_sha256 = sha256_file(X_VALIDATION_PATH)


def validation_result(df, model_name):
    row = df[
        (df["Model"] == model_name)
        &
        (df["Split"] == "Validation")
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Could not uniquely locate Validation result for {model_name}."
        )

    row = row.iloc[0]

    return {
        "MAE": float(row["MAE"]),
        "RMSE": float(row["RMSE"]),
        "R2": float(row["R2"]),
    }


UNTUNED_GB = validation_result(
    stage13_metrics,
    "Gradient_Boosting",
)

UNTUNED_XGB = validation_result(
    stage15_metrics,
    "XGBoost",
)


# ============================================================
# 7. BOUNDED PREDEFINED SEARCH SPACES
# ============================================================
#
# Trial 01 in each family exactly reproduces the corresponding
# Stage 13 / Stage 15 fixed model.
#
# The remaining trials explore meaningful nearby choices without
# creating an excessively large search over only two Validation
# batches.

GB_CONFIGS = [
    # Stage 13 reference
    dict(learning_rate=0.05, n_estimators=200, max_depth=3,
         min_samples_leaf=1, subsample=1.00),

    dict(learning_rate=0.03, n_estimators=300, max_depth=3,
         min_samples_leaf=1, subsample=1.00),

    dict(learning_rate=0.03, n_estimators=400, max_depth=3,
         min_samples_leaf=1, subsample=1.00),

    dict(learning_rate=0.05, n_estimators=300, max_depth=3,
         min_samples_leaf=1, subsample=1.00),

    dict(learning_rate=0.08, n_estimators=200, max_depth=3,
         min_samples_leaf=1, subsample=1.00),

    dict(learning_rate=0.05, n_estimators=300, max_depth=2,
         min_samples_leaf=1, subsample=1.00),

    dict(learning_rate=0.05, n_estimators=300, max_depth=4,
         min_samples_leaf=1, subsample=1.00),

    dict(learning_rate=0.05, n_estimators=300, max_depth=3,
         min_samples_leaf=5, subsample=1.00),

    dict(learning_rate=0.05, n_estimators=300, max_depth=3,
         min_samples_leaf=10, subsample=1.00),

    dict(learning_rate=0.03, n_estimators=400, max_depth=3,
         min_samples_leaf=5, subsample=0.90),
]

XGB_CONFIGS = [
    # Stage 15 reference
    dict(learning_rate=0.05, n_estimators=300, max_depth=3,
         min_child_weight=1.0, subsample=0.90, colsample_bytree=0.90,
         reg_alpha=0.0, reg_lambda=1.0, gamma=0.0),

    dict(learning_rate=0.03, n_estimators=400, max_depth=3,
         min_child_weight=1.0, subsample=0.90, colsample_bytree=0.90,
         reg_alpha=0.0, reg_lambda=1.0, gamma=0.0),

    dict(learning_rate=0.03, n_estimators=500, max_depth=3,
         min_child_weight=1.0, subsample=0.90, colsample_bytree=0.90,
         reg_alpha=0.0, reg_lambda=1.0, gamma=0.0),

    dict(learning_rate=0.05, n_estimators=400, max_depth=3,
         min_child_weight=1.0, subsample=0.90, colsample_bytree=0.90,
         reg_alpha=0.0, reg_lambda=1.0, gamma=0.0),

    dict(learning_rate=0.08, n_estimators=250, max_depth=3,
         min_child_weight=1.0, subsample=0.90, colsample_bytree=0.90,
         reg_alpha=0.0, reg_lambda=1.0, gamma=0.0),

    dict(learning_rate=0.05, n_estimators=400, max_depth=2,
         min_child_weight=1.0, subsample=0.90, colsample_bytree=0.90,
         reg_alpha=0.0, reg_lambda=1.0, gamma=0.0),

    dict(learning_rate=0.05, n_estimators=400, max_depth=4,
         min_child_weight=1.0, subsample=0.90, colsample_bytree=0.90,
         reg_alpha=0.0, reg_lambda=1.0, gamma=0.0),

    dict(learning_rate=0.05, n_estimators=400, max_depth=3,
         min_child_weight=5.0, subsample=0.90, colsample_bytree=0.90,
         reg_alpha=0.0, reg_lambda=1.0, gamma=0.0),

    dict(learning_rate=0.05, n_estimators=400, max_depth=3,
         min_child_weight=5.0, subsample=0.90, colsample_bytree=0.90,
         reg_alpha=0.0, reg_lambda=5.0, gamma=0.0),

    dict(learning_rate=0.03, n_estimators=500, max_depth=3,
         min_child_weight=5.0, subsample=0.85, colsample_bytree=0.85,
         reg_alpha=0.10, reg_lambda=5.0, gamma=0.0),
]

if SMOKE_TEST:
    GB_CONFIGS = GB_CONFIGS[:1]
    XGB_CONFIGS = XGB_CONFIGS[:1]


# ============================================================
# 8. METRICS / RANKING HELPERS
# ============================================================

def regression_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Negative_Predictions": int((y_pred < 0).sum()),
        "Predictions_Above_126h": int((y_pred > 126).sum()),
        "Prediction_Min": float(np.min(y_pred)),
        "Prediction_Max": float(np.max(y_pred)),
    }


def rank_key(metrics):
    return (
        metrics["MAE"],
        metrics["RMSE"],
        -metrics["R2"],
    )


trial_rows = []

best_gb_model = None
best_gb_metrics = None
best_gb_config = None
best_gb_trial = None

best_xgb_model = None
best_xgb_metrics = None
best_xgb_config = None
best_xgb_trial = None


# ============================================================
# 9. GRADIENT BOOSTING TUNING
# ============================================================

print(f"\nRunning {len(GB_CONFIGS)} Gradient Boosting tuning trials...")

for index, config in enumerate(GB_CONFIGS, start=1):
    trial_id = f"GB_{index:02d}"

    model = GradientBoostingRegressor(
        loss="squared_error",
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
        subsample=config["subsample"],
        min_samples_split=2,
        min_samples_leaf=config["min_samples_leaf"],
        max_depth=config["max_depth"],
        max_features=None,
        random_state=42,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_validation)

    train_m = regression_metrics(y_train, train_pred)
    val_m = regression_metrics(y_validation, val_pred)

    trial_rows.append({
        "Model_Family": "Gradient_Boosting",
        "Trial_ID": trial_id,
        "learning_rate": config["learning_rate"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "min_samples_leaf": config["min_samples_leaf"],
        "subsample": config["subsample"],
        "min_child_weight": np.nan,
        "colsample_bytree": np.nan,
        "reg_alpha": np.nan,
        "reg_lambda": np.nan,
        "gamma": np.nan,
        "Train_MAE": train_m["MAE"],
        "Train_RMSE": train_m["RMSE"],
        "Train_R2": train_m["R2"],
        "Validation_MAE": val_m["MAE"],
        "Validation_RMSE": val_m["RMSE"],
        "Validation_R2": val_m["R2"],
        "MAE_Generalization_Gap": val_m["MAE"] - train_m["MAE"],
        "Negative_Predictions": val_m["Negative_Predictions"],
        "Predictions_Above_126h": val_m["Predictions_Above_126h"],
        "Fit_Seconds": fit_seconds,
    })

    if (
        best_gb_metrics is None
        or rank_key(val_m) < rank_key(best_gb_metrics)
    ):
        best_gb_model = model
        best_gb_metrics = val_m
        best_gb_config = config.copy()
        best_gb_trial = trial_id

    print(
        f"  {trial_id}: "
        f"MAE={val_m['MAE']:.4f} h, "
        f"RMSE={val_m['RMSE']:.4f} h, "
        f"R2={val_m['R2']:.4f}"
    )


# ============================================================
# 10. XGBOOST TUNING
# ============================================================

print(f"\nRunning {len(XGB_CONFIGS)} XGBoost tuning trials...")

for index, config in enumerate(XGB_CONFIGS, start=1):
    trial_id = f"XGB_{index:02d}"

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_child_weight=config["min_child_weight"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        gamma=config["gamma"],
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train, verbose=False)
    fit_seconds = time.perf_counter() - start

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_validation)

    train_m = regression_metrics(y_train, train_pred)
    val_m = regression_metrics(y_validation, val_pred)

    trial_rows.append({
        "Model_Family": "XGBoost",
        "Trial_ID": trial_id,
        "learning_rate": config["learning_rate"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "min_samples_leaf": np.nan,
        "subsample": config["subsample"],
        "min_child_weight": config["min_child_weight"],
        "colsample_bytree": config["colsample_bytree"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
        "gamma": config["gamma"],
        "Train_MAE": train_m["MAE"],
        "Train_RMSE": train_m["RMSE"],
        "Train_R2": train_m["R2"],
        "Validation_MAE": val_m["MAE"],
        "Validation_RMSE": val_m["RMSE"],
        "Validation_R2": val_m["R2"],
        "MAE_Generalization_Gap": val_m["MAE"] - train_m["MAE"],
        "Negative_Predictions": val_m["Negative_Predictions"],
        "Predictions_Above_126h": val_m["Predictions_Above_126h"],
        "Fit_Seconds": fit_seconds,
    })

    if (
        best_xgb_metrics is None
        or rank_key(val_m) < rank_key(best_xgb_metrics)
    ):
        best_xgb_model = model
        best_xgb_metrics = val_m
        best_xgb_config = config.copy()
        best_xgb_trial = trial_id

    print(
        f"  {trial_id}: "
        f"MAE={val_m['MAE']:.4f} h, "
        f"RMSE={val_m['RMSE']:.4f} h, "
        f"R2={val_m['R2']:.4f}"
    )


# ============================================================
# 11. ALL-TRIAL RANKING
# ============================================================

trials_df = pd.DataFrame(trial_rows)

trials_df = (
    trials_df
    .sort_values(
        by=["Validation_MAE", "Validation_RMSE", "Validation_R2"],
        ascending=[True, True, False],
    )
    .reset_index(drop=True)
)

trials_df.insert(
    0,
    "Overall_Tuning_Rank",
    np.arange(1, len(trials_df) + 1),
)

trials_df.to_csv(TRIALS_PATH, index=False)


# ============================================================
# 12. REPRODUCIBILITY CHECK FOR REFERENCE TRIALS
# ============================================================

def trial_row(trial_id):
    row = trials_df[trials_df["Trial_ID"] == trial_id]

    if len(row) != 1:
        raise RuntimeError(f"Could not locate tuning trial {trial_id}.")

    return row.iloc[0]


gb_reference = trial_row("GB_01")
xgb_reference = trial_row("XGB_01")

GB_REFERENCE_MATCH = bool(
    abs(float(gb_reference["Validation_MAE"]) - UNTUNED_GB["MAE"]) <= 1e-6
    and abs(float(gb_reference["Validation_RMSE"]) - UNTUNED_GB["RMSE"]) <= 1e-6
    and abs(float(gb_reference["Validation_R2"]) - UNTUNED_GB["R2"]) <= 1e-6
)

XGB_REFERENCE_MATCH = bool(
    abs(float(xgb_reference["Validation_MAE"]) - UNTUNED_XGB["MAE"]) <= 1e-6
    and abs(float(xgb_reference["Validation_RMSE"]) - UNTUNED_XGB["RMSE"]) <= 1e-6
    and abs(float(xgb_reference["Validation_R2"]) - UNTUNED_XGB["R2"]) <= 1e-6
)


# ============================================================
# 13. BEST RESULT PER FAMILY
# ============================================================

def find_best_row(family):
    family_df = (
        trials_df[
            trials_df["Model_Family"] == family
        ]
        .sort_values(
            by=["Validation_MAE", "Validation_RMSE", "Validation_R2"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )

    return family_df.iloc[0].copy()


best_gb_row = find_best_row("Gradient_Boosting")
best_xgb_row = find_best_row("XGBoost")

best_results_df = pd.DataFrame(
    [best_gb_row, best_xgb_row]
)

best_results_df = (
    best_results_df
    .sort_values(
        by=["Validation_MAE", "Validation_RMSE", "Validation_R2"],
        ascending=[True, True, False],
    )
    .reset_index(drop=True)
)

best_results_df.insert(
    0,
    "Tuned_Model_Rank",
    np.arange(1, len(best_results_df) + 1),
)

best_results_df.to_csv(BEST_RESULTS_PATH, index=False)


# ============================================================
# 14. UNTUNED VS TUNED
# ============================================================

improvement_df = pd.DataFrame([
    {
        "Model_Family": "Gradient_Boosting",
        "Untuned_MAE": UNTUNED_GB["MAE"],
        "Tuned_MAE": best_gb_metrics["MAE"],
        "MAE_Improvement_pct": (
            (UNTUNED_GB["MAE"] - best_gb_metrics["MAE"])
            / UNTUNED_GB["MAE"] * 100
        ),
        "Untuned_RMSE": UNTUNED_GB["RMSE"],
        "Tuned_RMSE": best_gb_metrics["RMSE"],
        "RMSE_Improvement_pct": (
            (UNTUNED_GB["RMSE"] - best_gb_metrics["RMSE"])
            / UNTUNED_GB["RMSE"] * 100
        ),
        "Untuned_R2": UNTUNED_GB["R2"],
        "Tuned_R2": best_gb_metrics["R2"],
        "R2_Change": best_gb_metrics["R2"] - UNTUNED_GB["R2"],
        "Best_Trial_ID": best_gb_trial,
    },
    {
        "Model_Family": "XGBoost",
        "Untuned_MAE": UNTUNED_XGB["MAE"],
        "Tuned_MAE": best_xgb_metrics["MAE"],
        "MAE_Improvement_pct": (
            (UNTUNED_XGB["MAE"] - best_xgb_metrics["MAE"])
            / UNTUNED_XGB["MAE"] * 100
        ),
        "Untuned_RMSE": UNTUNED_XGB["RMSE"],
        "Tuned_RMSE": best_xgb_metrics["RMSE"],
        "RMSE_Improvement_pct": (
            (UNTUNED_XGB["RMSE"] - best_xgb_metrics["RMSE"])
            / UNTUNED_XGB["RMSE"] * 100
        ),
        "Untuned_R2": UNTUNED_XGB["R2"],
        "Tuned_R2": best_xgb_metrics["R2"],
        "R2_Change": best_xgb_metrics["R2"] - UNTUNED_XGB["R2"],
        "Best_Trial_ID": best_xgb_trial,
    },
])

improvement_df.to_csv(IMPROVEMENT_PATH, index=False)


# ============================================================
# 15. SAVE BEST TUNED MODELS
# ============================================================

joblib.dump(best_gb_model, GB_MODEL_PATH)
joblib.dump(best_xgb_model, XGB_MODEL_PATH)


# ============================================================
# 16. DETAILED VALIDATION OUTPUTS FOR THE TWO BEST TUNED MODELS
# ============================================================

prediction_frames = []
layer_rows = []
batch_rows = []

for family, model in [
    ("Gradient_Boosting", best_gb_model),
    ("XGBoost", best_xgb_model),
]:
    val_pred = model.predict(X_validation)

    pred_df = meta_validation.copy()
    pred_df["Model_Family"] = family
    pred_df["Actual_Remaining_Drying_Time"] = y_validation
    pred_df["Predicted_Remaining_Drying_Time"] = val_pred
    pred_df["Residual"] = y_validation - val_pred
    pred_df["Absolute_Error"] = np.abs(y_validation - val_pred)
    prediction_frames.append(pred_df)

    layer_temp = pd.DataFrame({
        "Layer_ID": meta_validation["Layer_ID"],
        "Actual": y_validation,
        "Predicted": val_pred,
    })

    for layer_id, group in layer_temp.groupby("Layer_ID", sort=True):
        m = regression_metrics(
            group["Actual"].to_numpy(),
            group["Predicted"].to_numpy(),
        )

        layer_rows.append({
            "Model_Family": family,
            "Layer_ID": layer_id,
            "Rows": len(group),
            **m,
        })

    batch_temp = pd.DataFrame({
        "Smokehouse_ID": meta_validation["Smokehouse_ID"],
        "Batch_ID": meta_validation["Batch_ID"],
        "Actual": y_validation,
        "Predicted": val_pred,
    })

    for (smokehouse_id, batch_id), group in batch_temp.groupby(
        ["Smokehouse_ID", "Batch_ID"],
        sort=True,
    ):
        m = regression_metrics(
            group["Actual"].to_numpy(),
            group["Predicted"].to_numpy(),
        )

        batch_rows.append({
            "Model_Family": family,
            "Smokehouse_ID": smokehouse_id,
            "Batch_ID": batch_id,
            "Rows": len(group),
            **m,
        })


pd.concat(
    prediction_frames,
    ignore_index=True,
).to_csv(
    PREDICTIONS_PATH,
    index=False,
)

pd.DataFrame(layer_rows).to_csv(
    LAYER_PATH,
    index=False,
)

pd.DataFrame(batch_rows).to_csv(
    BATCH_PATH,
    index=False,
)


# ============================================================
# 17. MODEL METADATA
# ============================================================

gb_model_sha256 = sha256_file(GB_MODEL_PATH)
xgb_model_sha256 = sha256_file(XGB_MODEL_PATH)

gb_metadata = {
    "stage": "18_regression_tuning",
    "model_family": "Gradient_Boosting",
    "best_trial_id": best_gb_trial,
    "best_parameters": best_gb_config,
    "validation": best_gb_metrics,
    "fit_scope": "TRAIN_ONLY",
    "validation_scope": "HYPERPARAMETER_SELECTION",
    "test_loaded": False,
    "x_train_sha256": x_train_sha256,
    "x_validation_sha256": x_validation_sha256,
    "model_sha256": gb_model_sha256,
}

xgb_metadata = {
    "stage": "18_regression_tuning",
    "model_family": "XGBoost",
    "best_trial_id": best_xgb_trial,
    "best_parameters": best_xgb_config,
    "validation": best_xgb_metrics,
    "fit_scope": "TRAIN_ONLY",
    "validation_scope": "HYPERPARAMETER_SELECTION",
    "test_loaded": False,
    "x_train_sha256": x_train_sha256,
    "x_validation_sha256": x_validation_sha256,
    "model_sha256": xgb_model_sha256,
}

with open(GB_METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(gb_metadata, file, indent=2)

with open(XGB_METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(xgb_metadata, file, indent=2)


# ============================================================
# 18. STAGE-18 DEVELOPMENT LEADER
# ============================================================

leader = best_results_df.iloc[0]

LEADER_FAMILY = str(leader["Model_Family"])
LEADER_TRIAL = str(leader["Trial_ID"])
LEADER_MAE = float(leader["Validation_MAE"])
LEADER_RMSE = float(leader["Validation_RMSE"])
LEADER_R2 = float(leader["Validation_R2"])


# ============================================================
# 19. CHECKS
# ============================================================

expected_trial_count = len(GB_CONFIGS) + len(XGB_CONFIGS)

all_trials_finite = bool(
    np.isfinite(
        trials_df[
            [
                "Train_MAE",
                "Train_RMSE",
                "Train_R2",
                "Validation_MAE",
                "Validation_RMSE",
                "Validation_R2",
            ]
        ].to_numpy(dtype=float)
    ).all()
)

best_families_correct = bool(
    set(best_results_df["Model_Family"]) == EXPECTED_SHORTLIST
)

checks = pd.DataFrame([
    {
        "Check": "Stage-17 shortlist matches expected models",
        "Status": "PASS" if actual_shortlist == EXPECTED_SHORTLIST else "FAIL",
        "Details": ", ".join(sorted(actual_shortlist)),
    },
    {
        "Check": "Expected Train/Validation shapes",
        "Status": "PASS",
        "Details": f"Train={X_train.shape}; Validation={X_validation.shape}",
    },
    {
        "Check": "All model inputs finite",
        "Status": "PASS",
        "Details": "No missing/non-finite transformed inputs.",
    },
    {
        "Check": "Expected tuning trial count",
        "Status": "PASS" if len(trials_df) == expected_trial_count else "FAIL",
        "Details": f"Trials={len(trials_df)}; expected={expected_trial_count}",
    },
    {
        "Check": "All tuning metrics finite",
        "Status": "PASS" if all_trials_finite else "FAIL",
        "Details": "Train/Validation MAE, RMSE and R² verified.",
    },
    {
        "Check": "Best result retained for both shortlisted families",
        "Status": "PASS" if best_families_correct else "FAIL",
        "Details": ", ".join(best_results_df["Model_Family"].tolist()),
    },
    {
        "Check": "Gradient Boosting reference reproduces Stage 13",
        "Status": "PASS" if GB_REFERENCE_MATCH else "FAIL",
        "Details": (
            f"Stage13 MAE={UNTUNED_GB['MAE']:.8f}; "
            f"GB_01 MAE={float(gb_reference['Validation_MAE']):.8f}"
        ),
    },
    {
        "Check": "XGBoost reference reproduces Stage 15",
        "Status": "PASS" if XGB_REFERENCE_MATCH else "FAIL",
        "Details": (
            f"Stage15 MAE={UNTUNED_XGB['MAE']:.8f}; "
            f"XGB_01 MAE={float(xgb_reference['Validation_MAE']):.8f}"
        ),
    },
    {
        "Check": "All configurations fitted on Train only",
        "Status": "PASS",
        "Details": "Validation used for tuning evaluation only.",
    },
    {
        "Check": "No Test rows loaded",
        "Status": "PASS",
        "Details": "Only Component3_Test.csv existence checked.",
    },
    {
        "Check": "Final regression model not declared",
        "Status": "PASS",
        "Details": "Stage 19 remains responsible for final development selection.",
    },
])

checks.to_csv(CHECKS_PATH, index=False)

overall_status = (
    "PASS"
    if not (checks["Status"] == "FAIL").any()
    else "FAIL"
)


# ============================================================
# 20. SUMMARY JSON
# ============================================================

summary = {
    "stage": "18_regression_hyperparameter_tuning",
    "overall_status": overall_status,
    "search_strategy": "bounded_predefined_validation_search",
    "primary_metric": "Validation_MAE",
    "gradient_boosting_trials": len(GB_CONFIGS),
    "xgboost_trials": len(XGB_CONFIGS),
    "best_gradient_boosting": {
        "trial_id": best_gb_trial,
        "parameters": best_gb_config,
        "MAE": best_gb_metrics["MAE"],
        "RMSE": best_gb_metrics["RMSE"],
        "R2": best_gb_metrics["R2"],
    },
    "best_xgboost": {
        "trial_id": best_xgb_trial,
        "parameters": best_xgb_config,
        "MAE": best_xgb_metrics["MAE"],
        "RMSE": best_xgb_metrics["RMSE"],
        "R2": best_xgb_metrics["R2"],
    },
    "stage18_validation_leader": {
        "model_family": LEADER_FAMILY,
        "trial_id": LEADER_TRIAL,
        "MAE": LEADER_MAE,
        "RMSE": LEADER_RMSE,
        "R2": LEADER_R2,
    },
    "test_loaded": False,
    "final_model_selected": False,
    "final_model_selection_stage": 19,
    "group_cross_validation_stage": 30,
    "final_test_evaluation_stage": 31,
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)


# ============================================================
# 21. CONSOLE OUTPUT
# ============================================================

print("\n--- BEST TUNED MODEL FROM EACH FAMILY ---")

display_columns = [
    "Tuned_Model_Rank",
    "Model_Family",
    "Trial_ID",
    "Validation_MAE",
    "Validation_RMSE",
    "Validation_R2",
    "Train_MAE",
    "MAE_Generalization_Gap",
    "Negative_Predictions",
    "Predictions_Above_126h",
]

print(
    best_results_df[
        display_columns
    ]
    .round(4)
    .to_string(index=False)
)

print("\n--- UNTUNED VS TUNED ---")
print(
    improvement_df
    .round(4)
    .to_string(index=False)
)

print("\n--- BEST GRADIENT BOOSTING PARAMETERS ---")
for key, value in best_gb_config.items():
    print(f"{key:<24}: {value}")

print("\n--- BEST XGBOOST PARAMETERS ---")
for key, value in best_xgb_config.items():
    print(f"{key:<24}: {value}")

print("\n--- STAGE 18 VALIDATION LEADER ---")
print(f"Model family                    : {LEADER_FAMILY}")
print(f"Trial                           : {LEADER_TRIAL}")
print(f"Validation MAE                  : {LEADER_MAE:.4f} h")
print(f"Validation RMSE                 : {LEADER_RMSE:.4f} h")
print(f"Validation R2                   : {LEADER_R2:.4f}")

print("\n--- REPRODUCIBILITY ---")
print(f"GB_01 reproduces Stage 13       : {GB_REFERENCE_MATCH}")
print(f"XGB_01 reproduces Stage 15      : {XGB_REFERENCE_MATCH}")

print("\n--- IMPORTANT INTERPRETATION ---")
print("This is the Stage-18 tuning leader, NOT yet the final regression model.")
print("Stage 19 will perform final development-set model selection.")
print("Stage 30 will later assess group-aware robustness.")

print("\n--- TEST PROTECTION ---")
print("Component3_Test.csv content read: False")
print("Final Test remains frozen until Stage 31.")


# ============================================================
# 22. HUMAN-READABLE REPORT
# ============================================================

with open(REPORT_PATH, "w", encoding="utf-8") as report:
    report.write(
        "COMPONENT 3 - STAGE 18: REGRESSION HYPERPARAMETER TUNING\n"
    )
    report.write("=" * 78 + "\n\n")
    report.write(f"OVERALL STATUS: {overall_status}\n\n")

    report.write("PURPOSE\n-------\n")
    report.write(
        "Stage 18 tunes only the two model families shortlisted by "
        "Stage 17: Gradient Boosting and XGBoost. Every configuration "
        "is fitted on Train only and evaluated on Validation only. "
        "Final Test is not loaded.\n\n"
    )

    report.write("SEARCH DESIGN\n-------------\n")
    report.write(f"Gradient Boosting trials: {len(GB_CONFIGS)}\n")
    report.write(f"XGBoost trials: {len(XGB_CONFIGS)}\n")
    report.write(f"Total trials: {len(trials_df)}\n")
    report.write(
        "Primary metric: Validation MAE\n"
        "Tie-breakers: Validation RMSE, then Validation R²\n\n"
    )
    report.write(
        "The search is intentionally bounded because only two whole "
        "batches form the development Validation set. A very large "
        "search could over-optimize that specific holdout. Stage 30 "
        "will later provide group-aware robustness evidence.\n\n"
    )

    report.write(
        "BEST TUNED RESULT PER FAMILY\n----------------------------\n")
    report.write(
        best_results_df[
            display_columns
        ].round(6).to_string(index=False)
    )
    report.write("\n\n")

    report.write("UNTUNED VS TUNED\n----------------\n")
    report.write(
        improvement_df.round(6).to_string(index=False)
    )
    report.write("\n\n")

    report.write(
        "BEST GRADIENT BOOSTING PARAMETERS\n---------------------------------\n")
    for key, value in best_gb_config.items():
        report.write(f"{key}: {value}\n")
    report.write("\n")

    report.write("BEST XGBOOST PARAMETERS\n-----------------------\n")
    for key, value in best_xgb_config.items():
        report.write(f"{key}: {value}\n")
    report.write("\n")

    report.write("STAGE-18 VALIDATION LEADER\n--------------------------\n")
    report.write(f"Model family: {LEADER_FAMILY}\n")
    report.write(f"Trial ID: {LEADER_TRIAL}\n")
    report.write(f"Validation MAE: {LEADER_MAE:.6f} h\n")
    report.write(f"Validation RMSE: {LEADER_RMSE:.6f} h\n")
    report.write(f"Validation R²: {LEADER_R2:.6f}\n\n")

    report.write("REPRODUCIBILITY\n---------------\n")
    report.write(
        f"GB_01 reproduces Stage 13: {GB_REFERENCE_MATCH}\n"
    )
    report.write(
        f"XGB_01 reproduces Stage 15: {XGB_REFERENCE_MATCH}\n\n"
    )

    report.write("METHODOLOGICAL CAUTION\n----------------------\n")
    report.write(
        "The Validation set has now been used for model-family comparison "
        "and bounded hyperparameter selection. Its metrics are development "
        "metrics, not an unbiased final generalization estimate. The final "
        "Test set remains untouched for Stage 31. Stage 30 will also provide "
        "group-aware cross-validation evidence.\n\n"
    )

    report.write("LEAKAGE PROTECTION\n------------------\n")
    report.write("- Every tuning trial was fitted on Train only.\n")
    report.write("- Validation was used for development tuning only.\n")
    report.write("- Final Test rows were not loaded.\n")
    report.write("- No Test-based model or parameter decision was made.\n\n")

    report.write("PASS / FAIL CHECKS\n------------------\n")
    report.write(checks.to_string(index=False))
    report.write("\n\n")
    report.write(f"STAGE 18 STATUS: {overall_status}\n")


# ============================================================
# 23. COMPLETE
# ============================================================

print("\n" + "=" * 78)
print(f"STAGE 18 REGRESSION HYPERPARAMETER TUNING: {overall_status}")
print("=" * 78)

print(f"\nReports saved in:\n{REPORT_DIR}")
print(f"\nTuned models saved in:\n{MODEL_DIR}")

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 18 failed. Review 18_regression_tuning_report.txt."
    )
