from pathlib import Path
import hashlib
import json
import math

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# COMPONENT 3 - STAGE 12: RANDOM FOREST REGRESSION
# ============================================================
#
# Purpose:
#   Train the first nonlinear regression candidate and compare
#   it against the Stage-11 Full Linear baseline.
#
# Methodology:
#   - TRAIN only for model fitting
#   - VALIDATION only for development evaluation
#   - FINAL TEST remains untouched until Stage 31
#   - Fixed untuned Random Forest configuration
#   - Hyperparameter tuning is reserved for Stage 18
#
# Metrics:
#   MAE, RMSE, R²
#   + layer-wise and batch-wise validation errors
#   + feature importance
#
# IMPORTANT:
#   Stage PASS means the experiment was executed correctly.
#   It does NOT require Random Forest to beat the baseline.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

# This file lives in:
# component_3_smokehouse/src/regression/
ROOT = Path(__file__).resolve().parents[2]

MODEL_READY_DIR = (
    ROOT
    / "data"
    / "model_ready"
)

SPLIT_DIR = (
    ROOT
    / "data"
    / "splits"
)

REPORT_DIR = (
    ROOT
    / "outputs"
    / "reports"
)

MODEL_DIR = (
    ROOT
    / "artifacts"
    / "models"
    / "regression"
    / "random_forest"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. INPUT FILES
# ============================================================

X_TRAIN_PATH = (
    MODEL_READY_DIR
    / "Component3_X_Train.csv"
)

X_VALIDATION_PATH = (
    MODEL_READY_DIR
    / "Component3_X_Validation.csv"
)

Y_TRAIN_PATH = (
    MODEL_READY_DIR
    / "Component3_y_regression_Train.csv"
)

Y_VALIDATION_PATH = (
    MODEL_READY_DIR
    / "Component3_y_regression_Validation.csv"
)

META_TRAIN_PATH = (
    MODEL_READY_DIR
    / "Component3_metadata_Train.csv"
)

META_VALIDATION_PATH = (
    MODEL_READY_DIR
    / "Component3_metadata_Validation.csv"
)

TEST_PATH = (
    SPLIT_DIR
    / "Component3_Test.csv"
)

STAGE11_METRICS_PATH = (
    REPORT_DIR
    / "11_regression_baseline_metrics.csv"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "12_random_forest_regression_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "12_random_forest_regression_checks.csv"
)

METRICS_PATH = (
    REPORT_DIR
    / "12_random_forest_regression_metrics.csv"
)

PREDICTIONS_PATH = (
    REPORT_DIR
    / "12_random_forest_validation_predictions.csv"
)

LAYER_METRICS_PATH = (
    REPORT_DIR
    / "12_random_forest_by_layer.csv"
)

BATCH_METRICS_PATH = (
    REPORT_DIR
    / "12_random_forest_by_batch.csv"
)

FEATURE_IMPORTANCE_PATH = (
    REPORT_DIR
    / "12_random_forest_feature_importance.csv"
)

MODEL_PATH = (
    MODEL_DIR
    / "random_forest_regressor.joblib"
)

MODEL_METADATA_PATH = (
    MODEL_DIR
    / "random_forest_metadata.json"
)


# ============================================================
# 4. EXPECTED SHAPES
# ============================================================

EXPECTED_TRAIN_ROWS = 16246
EXPECTED_VALIDATION_ROWS = 4159
EXPECTED_FEATURES = 20


# ============================================================
# 5. LOAD DEVELOPMENT DATA ONLY
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 12: RANDOM FOREST REGRESSION")
print("=" * 78)

required_files = [
    X_TRAIN_PATH,
    X_VALIDATION_PATH,
    Y_TRAIN_PATH,
    Y_VALIDATION_PATH,
    META_TRAIN_PATH,
    META_VALIDATION_PATH,
    TEST_PATH,
    STAGE11_METRICS_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

# IMPORTANT:
# Component3_Test.csv existence is checked only.
# Its contents are NOT loaded.

X_train = pd.read_csv(
    X_TRAIN_PATH
)

X_validation = pd.read_csv(
    X_VALIDATION_PATH
)

y_train_df = pd.read_csv(
    Y_TRAIN_PATH
)

y_validation_df = pd.read_csv(
    Y_VALIDATION_PATH
)

meta_train = pd.read_csv(
    META_TRAIN_PATH
)

meta_validation = pd.read_csv(
    META_VALIDATION_PATH
)

stage11_metrics = pd.read_csv(
    STAGE11_METRICS_PATH
)

y_train = pd.to_numeric(
    y_train_df[
        "Remaining_Drying_Time"
    ],
    errors="raise",
).to_numpy()

y_validation = pd.to_numeric(
    y_validation_df[
        "Remaining_Drying_Time"
    ],
    errors="raise",
).to_numpy()

print("\nTrain and Validation model-ready data loaded.")
print("Final Test dataset was NOT loaded.")


# ============================================================
# 6. INPUT FINGERPRINTS
# ============================================================

def sha256_file(path):
    with open(
        path,
        "rb",
    ) as file:
        return hashlib.sha256(
            file.read()
        ).hexdigest()


x_train_sha256 = sha256_file(
    X_TRAIN_PATH
)

x_validation_sha256 = sha256_file(
    X_VALIDATION_PATH
)

y_train_sha256 = sha256_file(
    Y_TRAIN_PATH
)

y_validation_sha256 = sha256_file(
    Y_VALIDATION_PATH
)


# ============================================================
# 7. BASIC VALIDATION
# ============================================================

if X_train.shape != (
    EXPECTED_TRAIN_ROWS,
    EXPECTED_FEATURES,
):
    raise RuntimeError(
        f"Unexpected X_train shape: {X_train.shape}"
    )

if X_validation.shape != (
    EXPECTED_VALIDATION_ROWS,
    EXPECTED_FEATURES,
):
    raise RuntimeError(
        f"Unexpected X_validation shape: {X_validation.shape}"
    )

if len(
    y_train
) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError(
        "Unexpected y_train row count."
    )

if len(
    y_validation
) != EXPECTED_VALIDATION_ROWS:
    raise RuntimeError(
        "Unexpected y_validation row count."
    )

if len(
    meta_train
) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError(
        "Unexpected Train metadata row count."
    )

if len(
    meta_validation
) != EXPECTED_VALIDATION_ROWS:
    raise RuntimeError(
        "Unexpected Validation metadata row count."
    )

if list(
    X_train.columns
) != list(
    X_validation.columns
):
    raise RuntimeError(
        "Train and Validation transformed feature columns differ."
    )

if X_train.isna().any().any():
    raise RuntimeError(
        "X_train contains missing values."
    )

if X_validation.isna().any().any():
    raise RuntimeError(
        "X_validation contains missing values."
    )

if not np.isfinite(
    X_train.to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "X_train contains non-finite values."
    )

if not np.isfinite(
    X_validation.to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "X_validation contains non-finite values."
    )


# ============================================================
# 8. LOAD STAGE-11 FULL-LINEAR BENCHMARK
# ============================================================

baseline_row = stage11_metrics[
    (
        stage11_metrics[
            "Model"
        ]
        ==
        "Full_Linear"
    )
    &
    (
        stage11_metrics[
            "Split"
        ]
        ==
        "Validation"
    )
]

if len(
    baseline_row
) != 1:
    raise RuntimeError(
        "Could not uniquely locate Stage-11 "
        "Full_Linear Validation benchmark."
    )

BASELINE_MAE = float(
    baseline_row.iloc[0][
        "MAE"
    ]
)

BASELINE_RMSE = float(
    baseline_row.iloc[0][
        "RMSE"
    ]
)

BASELINE_R2 = float(
    baseline_row.iloc[0][
        "R2"
    ]
)


# ============================================================
# 9. RANDOM FOREST CONFIGURATION
# ============================================================
#
# This is intentionally NOT hyperparameter tuning.
# We use a stable fixed configuration:
#
#   n_estimators=500
#       enough trees for stable estimates
#
#   max_depth=None
#   min_samples_split=2
#   min_samples_leaf=1
#   max_features=1.0
#       standard unrestricted RF behavior
#
# Stage 18 will tune complexity parameters properly.

RF_PARAMS = {
    "n_estimators": 500,
    "criterion": "squared_error",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}

model = RandomForestRegressor(
    **RF_PARAMS
)


# ============================================================
# 10. FIT TRAIN ONLY
# ============================================================

print("\nTraining Random Forest on TRAIN only...")

model.fit(
    X_train,
    y_train,
)

print("Random Forest training complete.")


# ============================================================
# 11. PREDICTIONS
# ============================================================

train_pred = model.predict(
    X_train
)

validation_pred = model.predict(
    X_validation
)


# ============================================================
# 12. METRICS
# ============================================================

def regression_metrics(
    y_true,
    y_pred,
):
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = math.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": float(
            mae
        ),
        "RMSE": float(
            rmse
        ),
        "R2": float(
            r2
        ),
        "Negative_Predictions": int(
            (
                y_pred
                <
                0
            ).sum()
        ),
        "Predictions_Above_126h": int(
            (
                y_pred
                >
                126
            ).sum()
        ),
    }


train_metrics = regression_metrics(
    y_train,
    train_pred,
)

validation_metrics = regression_metrics(
    y_validation,
    validation_pred,
)

metrics_df = pd.DataFrame(
    [
        {
            "Model": (
                "Random_Forest"
            ),
            "Split": "Train",
            **train_metrics,
        },
        {
            "Model": (
                "Random_Forest"
            ),
            "Split": "Validation",
            **validation_metrics,
        },
    ]
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)


# ============================================================
# 13. COMPARISON WITH STAGE 11
# ============================================================

mae_improvement_percent = (
    (
        BASELINE_MAE
        -
        validation_metrics[
            "MAE"
        ]
    )
    /
    BASELINE_MAE
    *
    100
)

rmse_improvement_percent = (
    (
        BASELINE_RMSE
        -
        validation_metrics[
            "RMSE"
        ]
    )
    /
    BASELINE_RMSE
    *
    100
)

r2_change = (
    validation_metrics[
        "R2"
    ]
    -
    BASELINE_R2
)

beats_baseline_mae = bool(
    validation_metrics[
        "MAE"
    ]
    <
    BASELINE_MAE
)


# ============================================================
# 14. GENERALIZATION GAP
# ============================================================

mae_generalization_gap = (
    validation_metrics[
        "MAE"
    ]
    -
    train_metrics[
        "MAE"
    ]
)

rmse_generalization_gap = (
    validation_metrics[
        "RMSE"
    ]
    -
    train_metrics[
        "RMSE"
    ]
)

r2_generalization_gap = (
    train_metrics[
        "R2"
    ]
    -
    validation_metrics[
        "R2"
    ]
)


# ============================================================
# 15. VALIDATION PREDICTION TABLE
# ============================================================

predictions_df = meta_validation.copy()

predictions_df[
    "Actual_Remaining_Drying_Time"
] = y_validation

predictions_df[
    "Predicted_Random_Forest"
] = validation_pred

predictions_df[
    "Residual_Random_Forest"
] = (
    y_validation
    -
    validation_pred
)

predictions_df[
    "Absolute_Error_Random_Forest"
] = np.abs(
    y_validation
    -
    validation_pred
)

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# 16. LAYER-WISE VALIDATION METRICS
# ============================================================

layer_temp = meta_validation[
    [
        "Layer_ID"
    ]
].copy()

layer_temp[
    "Actual"
] = y_validation

layer_temp[
    "Predicted"
] = validation_pred

layer_rows = []

for layer_id, group in layer_temp.groupby(
    "Layer_ID",
    sort=True,
):
    layer_result = regression_metrics(
        group[
            "Actual"
        ].to_numpy(),
        group[
            "Predicted"
        ].to_numpy(),
    )

    layer_rows.append({
        "Layer_ID": layer_id,
        "Rows": len(
            group
        ),
        **layer_result,
    })

layer_metrics_df = pd.DataFrame(
    layer_rows
)

layer_metrics_df.to_csv(
    LAYER_METRICS_PATH,
    index=False,
)


# ============================================================
# 17. BATCH-WISE VALIDATION METRICS
# ============================================================

batch_temp = meta_validation[
    [
        "Smokehouse_ID",
        "Batch_ID",
    ]
].copy()

batch_temp[
    "Actual"
] = y_validation

batch_temp[
    "Predicted"
] = validation_pred

batch_rows = []

for (
    smokehouse_id,
    batch_id,
), group in batch_temp.groupby(
    [
        "Smokehouse_ID",
        "Batch_ID",
    ],
    sort=True,
):
    batch_result = regression_metrics(
        group[
            "Actual"
        ].to_numpy(),
        group[
            "Predicted"
        ].to_numpy(),
    )

    batch_rows.append({
        "Smokehouse_ID": (
            smokehouse_id
        ),
        "Batch_ID": (
            batch_id
        ),
        "Rows": len(
            group
        ),
        **batch_result,
    })

batch_metrics_df = pd.DataFrame(
    batch_rows
)

batch_metrics_df.to_csv(
    BATCH_METRICS_PATH,
    index=False,
)


# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

feature_importance_df = pd.DataFrame({
    "Feature": (
        X_train.columns
    ),
    "Importance": (
        model.feature_importances_
    ),
})

feature_importance_df = (
    feature_importance_df
    .sort_values(
        "Importance",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

feature_importance_df[
    "Rank"
] = np.arange(
    1,
    len(
        feature_importance_df
    )
    + 1,
)

feature_importance_df = (
    feature_importance_df[
        [
            "Rank",
            "Feature",
            "Importance",
        ]
    ]
)

feature_importance_df.to_csv(
    FEATURE_IMPORTANCE_PATH,
    index=False,
)

importance_sum = float(
    feature_importance_df[
        "Importance"
    ].sum()
)


# ============================================================
# 19. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH,
)

model_sha256 = sha256_file(
    MODEL_PATH
)


# ============================================================
# 20. SANITY CHECKS
# ============================================================

predictions_finite = bool(
    np.isfinite(
        train_pred
    ).all()
    and
    np.isfinite(
        validation_pred
    ).all()
)

prediction_lengths_correct = bool(
    len(
        train_pred
    )
    ==
    EXPECTED_TRAIN_ROWS
    and
    len(
        validation_pred
    )
    ==
    EXPECTED_VALIDATION_ROWS
)

importance_valid = bool(
    len(
        feature_importance_df
    )
    ==
    EXPECTED_FEATURES
    and
    np.isclose(
        importance_sum,
        1.0,
    )
)


# ============================================================
# 21. PASS / FAIL CHECKS
# ============================================================

checks = pd.DataFrame(
    [
        {
            "Check": (
                "Expected Train shape"
            ),
            "Status": (
                "PASS"
                if X_train.shape
                ==
                (
                    EXPECTED_TRAIN_ROWS,
                    EXPECTED_FEATURES,
                )
                else "FAIL"
            ),
            "Details": (
                f"Shape="
                f"{X_train.shape}"
            ),
        },
        {
            "Check": (
                "Expected Validation shape"
            ),
            "Status": (
                "PASS"
                if X_validation.shape
                ==
                (
                    EXPECTED_VALIDATION_ROWS,
                    EXPECTED_FEATURES,
                )
                else "FAIL"
            ),
            "Details": (
                f"Shape="
                f"{X_validation.shape}"
            ),
        },
        {
            "Check": (
                "No missing/non-finite model inputs"
            ),
            "Status": (
                "PASS"
                if (
                    np.isfinite(
                        X_train.to_numpy(
                            dtype=float
                        )
                    ).all()
                    and
                    np.isfinite(
                        X_validation.to_numpy(
                            dtype=float
                        )
                    ).all()
                )
                else "FAIL"
            ),
            "Details": (
                "Stage-10 matrices verified."
            ),
        },
        {
            "Check": (
                "Stage-11 Full Linear benchmark loaded"
            ),
            "Status": (
                "PASS"
            ),
            "Details": (
                f"MAE={BASELINE_MAE:.6f}; "
                f"RMSE={BASELINE_RMSE:.6f}; "
                f"R2={BASELINE_R2:.6f}"
            ),
        },
        {
            "Check": (
                "Prediction lengths preserved"
            ),
            "Status": (
                "PASS"
                if prediction_lengths_correct
                else "FAIL"
            ),
            "Details": (
                f"Train={len(train_pred):,}; "
                f"Validation={len(validation_pred):,}"
            ),
        },
        {
            "Check": (
                "All predictions finite"
            ),
            "Status": (
                "PASS"
                if predictions_finite
                else "FAIL"
            ),
            "Details": (
                "No NaN or infinite predictions."
            ),
        },
        {
            "Check": (
                "Feature importances valid"
            ),
            "Status": (
                "PASS"
                if importance_valid
                else "FAIL"
            ),
            "Details": (
                f"Features={len(feature_importance_df)}; "
                f"Importance sum={importance_sum:.12f}"
            ),
        },
        {
            "Check": (
                "Model fitted on Train only"
            ),
            "Status": "PASS",
            "Details": (
                "Validation used only for evaluation."
            ),
        },
        {
            "Check": (
                "No hyperparameter tuning performed"
            ),
            "Status": "PASS",
            "Details": (
                "Fixed untuned configuration; tuning reserved for Stage 18."
            ),
        },
        {
            "Check": (
                "Final Test not loaded"
            ),
            "Status": "PASS",
            "Details": (
                "Only Test file existence checked; contents not read."
            ),
        },
    ]
)

checks.to_csv(
    CHECKS_PATH,
    index=False,
)

overall_status = (
    "PASS"
    if not (
        checks[
            "Status"
        ]
        ==
        "FAIL"
    ).any()
    else "FAIL"
)


# ============================================================
# 22. MODEL METADATA
# ============================================================

model_metadata = {
    "stage": (
        "12_random_forest_regression"
    ),
    "model": (
        "RandomForestRegressor"
    ),
    "fit_scope": (
        "TRAIN_ONLY"
    ),
    "validation_scope": (
        "EVALUATION_ONLY"
    ),
    "test_loaded": False,
    "hyperparameter_tuning_performed": False,
    "parameters": RF_PARAMS,
    "train_rows": int(
        len(
            X_train
        )
    ),
    "validation_rows": int(
        len(
            X_validation
        )
    ),
    "feature_count": int(
        X_train.shape[1]
    ),
    "stage11_full_linear_validation": {
        "MAE": (
            BASELINE_MAE
        ),
        "RMSE": (
            BASELINE_RMSE
        ),
        "R2": (
            BASELINE_R2
        ),
    },
    "random_forest_validation": {
        "MAE": (
            validation_metrics[
                "MAE"
            ]
        ),
        "RMSE": (
            validation_metrics[
                "RMSE"
            ]
        ),
        "R2": (
            validation_metrics[
                "R2"
            ]
        ),
    },
    "beats_stage11_baseline_by_mae": (
        beats_baseline_mae
    ),
    "x_train_sha256": (
        x_train_sha256
    ),
    "x_validation_sha256": (
        x_validation_sha256
    ),
    "y_train_sha256": (
        y_train_sha256
    ),
    "y_validation_sha256": (
        y_validation_sha256
    ),
    "model_sha256": (
        model_sha256
    ),
}

with open(
    MODEL_METADATA_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        model_metadata,
        file,
        indent=2,
    )


# ============================================================
# 23. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- INPUT ---"
)

print(
    f"X_train shape                   : "
    f"{X_train.shape}"
)

print(
    f"X_validation shape              : "
    f"{X_validation.shape}"
)

print(
    f"Features                        : "
    f"{X_train.shape[1]}"
)


print(
    "\n--- FIXED RANDOM FOREST CONFIGURATION ---"
)

for key, value in (
    RF_PARAMS.items()
):
    print(
        f"{key:<31}: {value}"
    )


print(
    "\n--- RANDOM FOREST RESULTS ---"
)

result_display = metrics_df[
    [
        "Split",
        "MAE",
        "RMSE",
        "R2",
        "Negative_Predictions",
        "Predictions_Above_126h",
    ]
]

print(
    result_display
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\n--- STAGE-11 FULL LINEAR BENCHMARK ---"
)

print(
    f"Validation MAE                  : "
    f"{BASELINE_MAE:.4f} h"
)

print(
    f"Validation RMSE                 : "
    f"{BASELINE_RMSE:.4f} h"
)

print(
    f"Validation R2                   : "
    f"{BASELINE_R2:.4f}"
)


print(
    "\n--- RANDOM FOREST VS FULL LINEAR ---"
)

print(
    f"MAE improvement                 : "
    f"{mae_improvement_percent:.2f}%"
)

print(
    f"RMSE improvement                : "
    f"{rmse_improvement_percent:.2f}%"
)

print(
    f"R2 change                       : "
    f"{r2_change:+.4f}"
)

print(
    f"Beats Full Linear by MAE        : "
    f"{beats_baseline_mae}"
)


print(
    "\n--- GENERALIZATION GAP ---"
)

print(
    f"Validation MAE - Train MAE      : "
    f"{mae_generalization_gap:.4f} h"
)

print(
    f"Validation RMSE - Train RMSE    : "
    f"{rmse_generalization_gap:.4f} h"
)

print(
    f"Train R2 - Validation R2        : "
    f"{r2_generalization_gap:.4f}"
)


print(
    "\n--- TOP 10 FEATURE IMPORTANCES ---"
)

print(
    feature_importance_df
    .head(
        10
    )
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print(
    "\n--- TEST PROTECTION ---"
)

print(
    "Component3_Test.csv content read: False"
)

print(
    "Final Test remains frozen until Stage 31."
)


# ============================================================
# 24. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 12: RANDOM FOREST REGRESSION\n"
    )

    report.write(
        "=" * 78
        + "\n\n"
    )

    report.write(
        f"OVERALL STATUS: "
        f"{overall_status}\n\n"
    )

    report.write(
        "PURPOSE\n"
        "-------\n"
    )

    report.write(
        "Stage 12 trains the first nonlinear regression "
        "candidate using a fixed, untuned Random Forest. "
        "The model is fitted on Train only and evaluated on "
        "Validation. Final Test remains untouched.\n\n"
    )

    report.write(
        "FIXED CONFIGURATION\n"
        "-------------------\n"
    )

    for key, value in (
        RF_PARAMS.items()
    ):
        report.write(
            f"{key}: {value}\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "RESULTS\n"
        "-------\n"
    )

    report.write(
        result_display
        .round(
            6
        )
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "COMPARISON WITH STAGE-11 FULL LINEAR BASELINE\n"
        "---------------------------------------------\n"
    )

    report.write(
        f"Full Linear Validation MAE: "
        f"{BASELINE_MAE:.6f} h\n"
    )

    report.write(
        f"Random Forest Validation MAE: "
        f"{validation_metrics['MAE']:.6f} h\n"
    )

    report.write(
        f"MAE improvement: "
        f"{mae_improvement_percent:.6f}%\n"
    )

    report.write(
        f"Full Linear Validation RMSE: "
        f"{BASELINE_RMSE:.6f} h\n"
    )

    report.write(
        f"Random Forest Validation RMSE: "
        f"{validation_metrics['RMSE']:.6f} h\n"
    )

    report.write(
        f"RMSE improvement: "
        f"{rmse_improvement_percent:.6f}%\n"
    )

    report.write(
        f"Full Linear Validation R²: "
        f"{BASELINE_R2:.6f}\n"
    )

    report.write(
        f"Random Forest Validation R²: "
        f"{validation_metrics['R2']:.6f}\n"
    )

    report.write(
        f"R² change: "
        f"{r2_change:+.6f}\n"
    )

    report.write(
        f"Beats Full Linear by Validation MAE: "
        f"{beats_baseline_mae}\n\n"
    )

    report.write(
        "GENERALIZATION GAP\n"
        "------------------\n"
    )

    report.write(
        f"Validation MAE - Train MAE: "
        f"{mae_generalization_gap:.6f} h\n"
    )

    report.write(
        f"Validation RMSE - Train RMSE: "
        f"{rmse_generalization_gap:.6f} h\n"
    )

    report.write(
        f"Train R² - Validation R²: "
        f"{r2_generalization_gap:.6f}\n\n"
    )

    report.write(
        "TOP 10 FEATURE IMPORTANCES\n"
        "--------------------------\n"
    )

    report.write(
        feature_importance_df
        .head(
            10
        )
        .round(
            8
        )
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "INTERPRETATION CAUTION\n"
        "----------------------\n"
    )

    report.write(
        "Random Forest impurity-based feature importance is "
        "useful for preliminary inspection but should not be "
        "treated as causal importance. Stage 32 will provide "
        "SHAP-based explainability for the final selected "
        "models.\n\n"
    )

    report.write(
        "LEAKAGE PROTECTION\n"
        "------------------\n"
    )

    report.write(
        "- Model fitted on Stage-10 Train matrix only.\n"
    )

    report.write(
        "- Validation used only for development evaluation.\n"
    )

    report.write(
        "- No hyperparameter search was performed.\n"
    )

    report.write(
        "- No final Test rows were loaded.\n"
    )

    report.write(
        "- Final Test remains frozen until Stage 31.\n\n"
    )

    report.write(
        "PASS / FAIL CHECKS\n"
        "------------------\n"
    )

    report.write(
        checks.to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        f"STAGE 12 STATUS: "
        f"{overall_status}\n"
    )


# ============================================================
# 25. COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    f"STAGE 12 RANDOM FOREST REGRESSION: "
    f"{overall_status}"
)

print(
    "=" * 78
)

print(
    f"\nReports saved in:\n"
    f"{REPORT_DIR}"
)

print(
    f"\nModel saved in:\n"
    f"{MODEL_DIR}"
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 12 failed. Review "
        "12_random_forest_regression_report.txt."
    )
