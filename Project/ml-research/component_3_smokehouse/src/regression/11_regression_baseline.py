from pathlib import Path
import hashlib
import json
import math
import pandas as pd
import numpy as np
import joblib

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# COMPONENT 3 - STAGE 11: REGRESSION BASELINE
# ============================================================
#
# Purpose:
#   Establish simple, leakage-safe regression benchmarks before
#   training stronger ensemble models.
#
# Baselines:
#   1) DummyRegressor(mean)
#      -> predicts the Train target mean for every row.
#
#   2) Elapsed-only Linear Regression
#      -> uses only transformed Elapsed_Drying_Hours.
#      -> tests how much performance comes simply from drying age.
#
#   3) Full Linear Regression
#      -> uses all 20 Stage-10 transformed predictors.
#      -> simple multivariable benchmark before tree ensembles.
#
# Evaluation:
#   TRAIN and VALIDATION only.
#   FINAL TEST remains untouched until Stage 31.
#
# Metrics:
#   MAE
#   RMSE
#   R²
#   plus validation error by Layer_ID and Batch_ID.
#
# No tuning is performed here.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_READY_DIR = ROOT / "data" / "model_ready"
SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "outputs" / "reports"
MODEL_DIR = ROOT / "artifacts" / "models" / "regression" / "baseline"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. INPUT FILES
# ============================================================

X_TRAIN_PATH = MODEL_READY_DIR / "Component3_X_Train.csv"
X_VALIDATION_PATH = MODEL_READY_DIR / "Component3_X_Validation.csv"
Y_TRAIN_PATH = MODEL_READY_DIR / "Component3_y_regression_Train.csv"
Y_VALIDATION_PATH = MODEL_READY_DIR / "Component3_y_regression_Validation.csv"
META_TRAIN_PATH = MODEL_READY_DIR / "Component3_metadata_Train.csv"
META_VALIDATION_PATH = MODEL_READY_DIR / "Component3_metadata_Validation.csv"
TEST_PATH = SPLIT_DIR / "Component3_Test.csv"


# ============================================================
# 3. OUTPUT FILES
# ============================================================

REPORT_PATH = REPORT_DIR / "11_regression_baseline_report.txt"
CHECKS_PATH = REPORT_DIR / "11_regression_baseline_checks.csv"
METRICS_PATH = REPORT_DIR / "11_regression_baseline_metrics.csv"
PREDICTIONS_PATH = REPORT_DIR / "11_regression_baseline_validation_predictions.csv"
LAYER_METRICS_PATH = REPORT_DIR / "11_regression_baseline_by_layer.csv"
BATCH_METRICS_PATH = REPORT_DIR / "11_regression_baseline_by_batch.csv"

DUMMY_MODEL_PATH = MODEL_DIR / "dummy_mean_regressor.joblib"
ELAPSED_MODEL_PATH = MODEL_DIR / "elapsed_only_linear_regression.joblib"
FULL_LINEAR_MODEL_PATH = MODEL_DIR / "full_linear_regression.joblib"
METADATA_PATH = MODEL_DIR / "baseline_model_metadata.json"


# ============================================================
# 4. EXPECTED SHAPES
# ============================================================

EXPECTED_TRAIN_ROWS = 16246
EXPECTED_VALIDATION_ROWS = 4159
EXPECTED_FEATURES = 20


# ============================================================
# 5. LOAD INPUTS
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 11: REGRESSION BASELINE")
print("=" * 78)

required_files = [
    X_TRAIN_PATH,
    X_VALIDATION_PATH,
    Y_TRAIN_PATH,
    Y_VALIDATION_PATH,
    META_TRAIN_PATH,
    META_VALIDATION_PATH,
    TEST_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

# IMPORTANT:
# TEST file existence is checked only. Its contents are NOT read.

X_train = pd.read_csv(X_TRAIN_PATH)
X_validation = pd.read_csv(X_VALIDATION_PATH)
y_train_df = pd.read_csv(Y_TRAIN_PATH)
y_validation_df = pd.read_csv(Y_VALIDATION_PATH)
meta_train = pd.read_csv(META_TRAIN_PATH)
meta_validation = pd.read_csv(META_VALIDATION_PATH)

y_train = pd.to_numeric(
    y_train_df["Remaining_Drying_Time"],
    errors="raise",
).to_numpy()

y_validation = pd.to_numeric(
    y_validation_df["Remaining_Drying_Time"],
    errors="raise",
).to_numpy()

print("\nTrain and Validation model-ready data loaded.")
print("Final Test dataset was NOT loaded.")


# ============================================================
# 6. INPUT FINGERPRINTS
# ============================================================

def sha256_file(path):
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


x_train_sha256 = sha256_file(X_TRAIN_PATH)
x_validation_sha256 = sha256_file(X_VALIDATION_PATH)
y_train_sha256 = sha256_file(Y_TRAIN_PATH)
y_validation_sha256 = sha256_file(Y_VALIDATION_PATH)


# ============================================================
# 7. BASIC VALIDATION
# ============================================================

if X_train.shape != (EXPECTED_TRAIN_ROWS, EXPECTED_FEATURES):
    raise RuntimeError(f"Unexpected X_train shape: {X_train.shape}")

if X_validation.shape != (EXPECTED_VALIDATION_ROWS, EXPECTED_FEATURES):
    raise RuntimeError(
        f"Unexpected X_validation shape: {X_validation.shape}"
    )

if len(y_train) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError("Unexpected y_train row count.")

if len(y_validation) != EXPECTED_VALIDATION_ROWS:
    raise RuntimeError("Unexpected y_validation row count.")

if len(meta_train) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError("Unexpected Train metadata row count.")

if len(meta_validation) != EXPECTED_VALIDATION_ROWS:
    raise RuntimeError("Unexpected Validation metadata row count.")

if X_train.isna().any().any():
    raise RuntimeError("X_train contains missing values.")

if X_validation.isna().any().any():
    raise RuntimeError("X_validation contains missing values.")

if not np.isfinite(X_train.to_numpy(dtype=float)).all():
    raise RuntimeError("X_train contains non-finite values.")

if not np.isfinite(X_validation.to_numpy(dtype=float)).all():
    raise RuntimeError("X_validation contains non-finite values.")


# ============================================================
# 8. LOCATE ELAPSED FEATURE
# ============================================================

elapsed_feature = "Elapsed_Drying_Hours"

if elapsed_feature not in X_train.columns:
    raise RuntimeError(
        f"Required transformed feature '{elapsed_feature}' not found."
    )

X_train_elapsed = X_train[[elapsed_feature]]
X_validation_elapsed = X_validation[[elapsed_feature]]


# ============================================================
# 9. DEFINE BASELINE MODELS
# ============================================================

models = {
    "Dummy_Mean": (
        DummyRegressor(strategy="mean"),
        X_train,
        X_validation,
    ),
    "Elapsed_Only_Linear": (
        LinearRegression(),
        X_train_elapsed,
        X_validation_elapsed,
    ),
    "Full_Linear": (
        LinearRegression(),
        X_train,
        X_validation,
    ),
}


# ============================================================
# 10. METRIC FUNCTION
# ============================================================

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


# ============================================================
# 11. TRAIN + EVALUATE BASELINES
# ============================================================

results = []
fitted_models = {}
validation_predictions = {}
train_predictions = {}

for model_name, (model, train_features, validation_features) in models.items():
    model.fit(train_features, y_train)
    fitted_models[model_name] = model

    train_pred = model.predict(train_features)
    validation_pred = model.predict(validation_features)

    train_predictions[model_name] = train_pred
    validation_predictions[model_name] = validation_pred

    train_metrics = regression_metrics(y_train, train_pred)
    validation_metrics = regression_metrics(y_validation, validation_pred)

    results.append({
        "Model": model_name,
        "Split": "Train",
        **train_metrics,
        "Negative_Predictions": int((train_pred < 0).sum()),
        "Predictions_Above_126h": int((train_pred > 126).sum()),
    })

    results.append({
        "Model": model_name,
        "Split": "Validation",
        **validation_metrics,
        "Negative_Predictions": int((validation_pred < 0).sum()),
        "Predictions_Above_126h": int((validation_pred > 126).sum()),
    })

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(METRICS_PATH, index=False)


# ============================================================
# 12. DEFINE BENCHMARK FOR LATER MODELS
# ============================================================
#
# We report all three baselines. For later regression model
# comparison, the benchmark is the BETTER of the two genuine
# linear baselines on Validation MAE (Dummy is sanity-only).
# This selection uses Validation only; Test remains untouched.

validation_linear = (
    metrics_df[
        (metrics_df["Split"] == "Validation")
        & (metrics_df["Model"].isin(["Elapsed_Only_Linear", "Full_Linear"]))
    ]
    .sort_values(["MAE", "RMSE"], ascending=True)
    .reset_index(drop=True)
)

MAIN_BASELINE_NAME = str(validation_linear.loc[0, "Model"])
main_validation_metrics = validation_linear.iloc[0]


# ============================================================
# 13. VALIDATION PREDICTION TABLE
# ============================================================

pred_df = meta_validation.copy()
pred_df["Actual_Remaining_Drying_Time"] = y_validation

for model_name, predictions in validation_predictions.items():
    pred_df[f"Predicted_{model_name}"] = predictions
    pred_df[f"Absolute_Error_{model_name}"] = np.abs(
        y_validation - predictions
    )

pred_df.to_csv(PREDICTIONS_PATH, index=False)


# ============================================================
# 14. LAYER-WISE VALIDATION METRICS
# ============================================================

layer_rows = []

for model_name, predictions in validation_predictions.items():
    temp = meta_validation[["Layer_ID"]].copy()
    temp["Actual"] = y_validation
    temp["Predicted"] = predictions

    for layer_id, group in temp.groupby("Layer_ID", sort=True):
        group_metrics = regression_metrics(
            group["Actual"].to_numpy(),
            group["Predicted"].to_numpy(),
        )

        layer_rows.append({
            "Model": model_name,
            "Layer_ID": layer_id,
            "Rows": len(group),
            **group_metrics,
        })

layer_metrics_df = pd.DataFrame(layer_rows)
layer_metrics_df.to_csv(LAYER_METRICS_PATH, index=False)


# ============================================================
# 15. BATCH-WISE VALIDATION METRICS
# ============================================================

batch_rows = []

for model_name, predictions in validation_predictions.items():
    temp = meta_validation[["Smokehouse_ID", "Batch_ID"]].copy()
    temp["Actual"] = y_validation
    temp["Predicted"] = predictions

    for (smokehouse_id, batch_id), group in temp.groupby(
        ["Smokehouse_ID", "Batch_ID"],
        sort=True,
    ):
        group_metrics = regression_metrics(
            group["Actual"].to_numpy(),
            group["Predicted"].to_numpy(),
        )

        batch_rows.append({
            "Model": model_name,
            "Smokehouse_ID": smokehouse_id,
            "Batch_ID": batch_id,
            "Rows": len(group),
            **group_metrics,
        })

batch_metrics_df = pd.DataFrame(batch_rows)
batch_metrics_df.to_csv(BATCH_METRICS_PATH, index=False)


# ============================================================
# 16. SAVE MODELS
# ============================================================

joblib.dump(fitted_models["Dummy_Mean"], DUMMY_MODEL_PATH)
joblib.dump(fitted_models["Elapsed_Only_Linear"], ELAPSED_MODEL_PATH)
joblib.dump(fitted_models["Full_Linear"], FULL_LINEAR_MODEL_PATH)


# ============================================================
# 17. BASELINE IMPROVEMENT CALCULATIONS
# ============================================================

validation_only = (
    metrics_df[metrics_df["Split"] == "Validation"]
    .set_index("Model")
)

dummy_mae = float(validation_only.loc["Dummy_Mean", "MAE"])
elapsed_mae = float(validation_only.loc["Elapsed_Only_Linear", "MAE"])
full_mae = float(validation_only.loc["Full_Linear", "MAE"])

elapsed_vs_dummy_mae_improvement = (
    (dummy_mae - elapsed_mae) / dummy_mae * 100
)

full_vs_elapsed_mae_improvement = (
    (elapsed_mae - full_mae) / elapsed_mae * 100
)

full_vs_dummy_mae_improvement = (
    (dummy_mae - full_mae) / dummy_mae * 100
)


# ============================================================
# 18. SANITY CHECKS
# ============================================================

dummy_prediction_constant = bool(
    np.allclose(
        validation_predictions["Dummy_Mean"],
        validation_predictions["Dummy_Mean"][0],
    )
)

dummy_prediction_matches_train_mean = bool(
    np.isclose(
        validation_predictions["Dummy_Mean"][0],
        np.mean(y_train),
    )
)

all_prediction_lengths_correct = all(
    len(predictions) == EXPECTED_VALIDATION_ROWS
    for predictions in validation_predictions.values()
)

all_predictions_finite = all(
    np.isfinite(predictions).all()
    for predictions in validation_predictions.values()
)

metrics_finite = bool(
    np.isfinite(
        metrics_df[["MAE", "RMSE", "R2"]].to_numpy()
    ).all()
)


# ============================================================
# 19. PASS / FAIL CHECKS
# ============================================================

checks = pd.DataFrame([
    {
        "Check": "Expected Train shape",
        "Status": "PASS" if X_train.shape == (
            EXPECTED_TRAIN_ROWS,
            EXPECTED_FEATURES,
        ) else "FAIL",
        "Details": f"Shape={X_train.shape}",
    },
    {
        "Check": "Expected Validation shape",
        "Status": "PASS" if X_validation.shape == (
            EXPECTED_VALIDATION_ROWS,
            EXPECTED_FEATURES,
        ) else "FAIL",
        "Details": f"Shape={X_validation.shape}",
    },
    {
        "Check": "No missing/non-finite model inputs",
        "Status": "PASS" if (
            np.isfinite(X_train.to_numpy(dtype=float)).all()
            and np.isfinite(X_validation.to_numpy(dtype=float)).all()
        ) else "FAIL",
        "Details": "Stage-10 model-ready matrices verified.",
    },
    {
        "Check": "Dummy prediction equals Train target mean",
        "Status": "PASS" if (
            dummy_prediction_constant
            and dummy_prediction_matches_train_mean
        ) else "FAIL",
        "Details": f"Train target mean={np.mean(y_train):.6f}",
    },
    {
        "Check": "All validation prediction lengths correct",
        "Status": "PASS" if all_prediction_lengths_correct else "FAIL",
        "Details": f"Expected={EXPECTED_VALIDATION_ROWS}",
    },
    {
        "Check": "All predictions finite",
        "Status": "PASS" if all_predictions_finite else "FAIL",
        "Details": "No NaN or infinite predictions.",
    },
    {
        "Check": "All regression metrics finite",
        "Status": "PASS" if metrics_finite else "FAIL",
        "Details": "MAE/RMSE/R² verified.",
    },
    {
        "Check": "Validation metadata alignment",
        "Status": "PASS" if len(meta_validation) == len(y_validation) else "FAIL",
        "Details": f"Rows={len(meta_validation):,}",
    },
    {
        "Check": "Final Test not loaded",
        "Status": "PASS",
        "Details": (
            "Stage 11 checks only that Component3_Test.csv exists; "
            "its contents are not read."
        ),
    },
])

checks.to_csv(CHECKS_PATH, index=False)

overall_status = (
    "PASS"
    if not (checks["Status"] == "FAIL").any()
    else "FAIL"
)


# ============================================================
# 20. SAVE MODEL METADATA
# ============================================================

metadata = {
    "stage": "11_regression_baseline",
    "test_loaded": False,
    "train_rows": len(X_train),
    "validation_rows": len(X_validation),
    "transformed_feature_count": X_train.shape[1],
    "elapsed_only_feature": elapsed_feature,
    "benchmark_model_for_later_comparison": MAIN_BASELINE_NAME,
    "models": [
        "Dummy_Mean",
        "Elapsed_Only_Linear",
        "Full_Linear",
    ],
    "x_train_sha256": x_train_sha256,
    "x_validation_sha256": x_validation_sha256,
    "y_train_sha256": y_train_sha256,
    "y_validation_sha256": y_validation_sha256,
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)


# ============================================================
# 21. CONSOLE OUTPUT
# ============================================================

print("\n--- INPUT ---")
print(f"X_train shape                   : {X_train.shape}")
print(f"X_validation shape              : {X_validation.shape}")
print(f"Train target mean               : {np.mean(y_train):.3f} h")
print(f"Validation target mean          : {np.mean(y_validation):.3f} h")

print("\n--- BASELINE RESULTS ---")

display_metrics = metrics_df[
    metrics_df["Split"] == "Validation"
].copy()

print(
    display_metrics[
        [
            "Model",
            "MAE",
            "RMSE",
            "R2",
            "Negative_Predictions",
            "Predictions_Above_126h",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print("\n--- BASELINE IMPROVEMENTS (VALIDATION MAE) ---")
print(
    f"Elapsed-only vs Dummy           : "
    f"{elapsed_vs_dummy_mae_improvement:.2f}%"
)
print(
    f"Full Linear vs Elapsed-only     : "
    f"{full_vs_elapsed_mae_improvement:.2f}%"
)
print(
    f"Full Linear vs Dummy            : "
    f"{full_vs_dummy_mae_improvement:.2f}%"
)

print("\n--- BENCHMARK FOR LATER REGRESSION MODELS ---")
print(f"Model                           : {MAIN_BASELINE_NAME}")
print(
    f"Validation MAE                  : "
    f"{float(main_validation_metrics['MAE']):.4f} h"
)
print(
    f"Validation RMSE                 : "
    f"{float(main_validation_metrics['RMSE']):.4f} h"
)
print(
    f"Validation R2                   : "
    f"{float(main_validation_metrics['R2']):.4f}"
)

print("\n--- TEST PROTECTION ---")
print("Component3_Test.csv content read: False")
print("Final Test remains frozen until Stage 31.")


# ============================================================
# 22. HUMAN-READABLE REPORT
# ============================================================

with open(REPORT_PATH, "w", encoding="utf-8") as report:
    report.write("COMPONENT 3 - STAGE 11: REGRESSION BASELINE\n")
    report.write("=" * 78 + "\n\n")
    report.write(f"OVERALL STATUS: {overall_status}\n\n")

    report.write("PURPOSE\n")
    report.write("-------\n")
    report.write(
        "Stage 11 establishes simple regression reference models "
        "before stronger tree-based ensembles are trained. This "
        "prevents later models from being judged without a meaningful "
        "benchmark.\n\n"
    )

    report.write("BASELINES\n")
    report.write("---------\n")
    report.write("1. Dummy_Mean: predicts the Train target mean.\n")
    report.write(
        "2. Elapsed_Only_Linear: linear regression using only "
        "Elapsed_Drying_Hours.\n"
    )
    report.write(
        "3. Full_Linear: linear regression using all 20 "
        "Stage-10 transformed predictors.\n\n"
    )

    report.write("VALIDATION RESULTS\n")
    report.write("------------------\n")
    report.write(
        display_metrics[
            [
                "Model",
                "MAE",
                "RMSE",
                "R2",
                "Negative_Predictions",
                "Predictions_Above_126h",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    report.write("\n\n")

    report.write("VALIDATION MAE IMPROVEMENTS\n")
    report.write("---------------------------\n")
    report.write(
        f"Elapsed-only vs Dummy: "
        f"{elapsed_vs_dummy_mae_improvement:.4f}%\n"
    )
    report.write(
        f"Full Linear vs Elapsed-only: "
        f"{full_vs_elapsed_mae_improvement:.4f}%\n"
    )
    report.write(
        f"Full Linear vs Dummy: "
        f"{full_vs_dummy_mae_improvement:.4f}%\n\n"
    )

    report.write("BENCHMARK FOR LATER MODEL COMPARISON\n")
    report.write("------------------------------------\n")
    report.write(f"Model: {MAIN_BASELINE_NAME}\n")
    report.write(
        f"Validation MAE: "
        f"{float(main_validation_metrics['MAE']):.6f} h\n"
    )
    report.write(
        f"Validation RMSE: "
        f"{float(main_validation_metrics['RMSE']):.6f} h\n"
    )
    report.write(
        f"Validation R²: "
        f"{float(main_validation_metrics['R2']):.6f}\n\n"
    )

    report.write("INTERPRETATION NOTE\n")
    report.write("-------------------\n")
    report.write(
        "Row-level validation metrics measure prediction error across "
        "10-minute observations, but observations are nested within "
        "complete drying batches. Therefore Stage 11 also reports "
        "Layer_ID and Batch_ID metrics, and Stage 30 will provide "
        "GroupKFold robustness analysis across batches.\n\n"
    )

    report.write("LEAKAGE PROTECTION\n")
    report.write("------------------\n")
    report.write("- Only Stage-10 Train and Validation matrices were used.\n")
    report.write("- No model was fitted on Validation.\n")
    report.write("- No hyperparameter tuning was performed.\n")
    report.write("- Final Test contents were not loaded.\n")
    report.write("- Final Test remains frozen until Stage 31.\n\n")

    report.write("INPUT FINGERPRINTS\n")
    report.write("------------------\n")
    report.write(f"X Train SHA-256: {x_train_sha256}\n")
    report.write(f"X Validation SHA-256: {x_validation_sha256}\n")
    report.write(f"y Train SHA-256: {y_train_sha256}\n")
    report.write(f"y Validation SHA-256: {y_validation_sha256}\n\n")

    report.write("PASS / FAIL CHECKS\n")
    report.write("------------------\n")
    report.write(checks.to_string(index=False))
    report.write("\n\n")
    report.write(f"STAGE 11 STATUS: {overall_status}\n")


# ============================================================
# 23. COMPLETE
# ============================================================

print("\n" + "=" * 78)
print(f"STAGE 11 REGRESSION BASELINE: {overall_status}")
print("=" * 78)
print(f"\nReports saved in:\n{REPORT_DIR}")
print(f"\nModels saved in:\n{MODEL_DIR}")

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 11 failed. Review 11_regression_baseline_report.txt."
    )
