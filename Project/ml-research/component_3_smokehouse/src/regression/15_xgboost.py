from pathlib import Path
import hashlib
import json
import math

import joblib
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "\nXGBoost is not installed in the active virtual environment.\n"
        "Install it with:\n"
        "    python -m pip install xgboost\n"
        "Then rerun Stage 15.\n"
    ) from exc

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# COMPONENT 3 - STAGE 15: XGBOOST REGRESSION
# ============================================================
#
# Purpose:
#   Train a fixed, untuned XGBoost regression candidate and
#   compare it with completed regression models:
#     - Stage 11 Full Linear
#     - Stage 12 Random Forest
#     - Stage 13 Gradient Boosting
#     - Stage 14 Extra Trees
#
# Methodology:
#   - TRAIN only for model fitting
#   - VALIDATION only for development evaluation
#   - FINAL TEST remains untouched until Stage 31
#   - No hyperparameter search in this stage
#   - Stage 18 is reserved for tuning
#
# Metrics:
#   MAE, RMSE, R²
#   + generalization gap
#   + layer-wise validation metrics
#   + batch-wise validation metrics
#   + feature importance
#
# IMPORTANT:
#   Stage PASS means the experiment executed correctly.
#   It does NOT require XGBoost to beat the current best model.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

# File location:
# component_3_smokehouse/src/regression/15_xgboost.py
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
    / "xgboost"
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

STAGE12_METRICS_PATH = (
    REPORT_DIR
    / "12_random_forest_regression_metrics.csv"
)

STAGE13_METRICS_PATH = (
    REPORT_DIR
    / "13_gradient_boosting_regression_metrics.csv"
)

STAGE14_METRICS_PATH = (
    REPORT_DIR
    / "14_extra_trees_regression_metrics.csv"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "15_xgboost_regression_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "15_xgboost_regression_checks.csv"
)

METRICS_PATH = (
    REPORT_DIR
    / "15_xgboost_regression_metrics.csv"
)

PREDICTIONS_PATH = (
    REPORT_DIR
    / "15_xgboost_validation_predictions.csv"
)

LAYER_METRICS_PATH = (
    REPORT_DIR
    / "15_xgboost_by_layer.csv"
)

BATCH_METRICS_PATH = (
    REPORT_DIR
    / "15_xgboost_by_batch.csv"
)

FEATURE_IMPORTANCE_PATH = (
    REPORT_DIR
    / "15_xgboost_feature_importance.csv"
)

MODEL_PATH = (
    MODEL_DIR
    / "xgboost_regressor.joblib"
)

MODEL_METADATA_PATH = (
    MODEL_DIR
    / "xgboost_metadata.json"
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
print("COMPONENT 3 - STAGE 15: XGBOOST REGRESSION")
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
    STAGE12_METRICS_PATH,
    STAGE13_METRICS_PATH,
    STAGE14_METRICS_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

# IMPORTANT:
# Test file existence is checked only.
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

stage12_metrics = pd.read_csv(
    STAGE12_METRICS_PATH
)

stage13_metrics = pd.read_csv(
    STAGE13_METRICS_PATH
)

stage14_metrics = pd.read_csv(
    STAGE14_METRICS_PATH
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
        "Train and Validation feature columns differ."
    )

if not np.isfinite(
    X_train.to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "X_train contains missing/non-finite values."
    )

if not np.isfinite(
    X_validation.to_numpy(
        dtype=float
    )
).all():
    raise RuntimeError(
        "X_validation contains missing/non-finite values."
    )


# ============================================================
# 8. LOAD PREVIOUS VALIDATION RESULTS
# ============================================================

def get_validation_row(
    metrics_df,
    model_name,
):
    row = metrics_df[
        (
            metrics_df[
                "Model"
            ]
            ==
            model_name
        )
        &
        (
            metrics_df[
                "Split"
            ]
            ==
            "Validation"
        )
    ]

    if len(
        row
    ) != 1:
        raise RuntimeError(
            f"Could not uniquely locate Validation result "
            f"for {model_name}."
        )

    return row.iloc[0]


linear_row = get_validation_row(
    stage11_metrics,
    "Full_Linear",
)

rf_row = get_validation_row(
    stage12_metrics,
    "Random_Forest",
)

gb_row = get_validation_row(
    stage13_metrics,
    "Gradient_Boosting",
)

et_row = get_validation_row(
    stage14_metrics,
    "Extra_Trees",
)

LINEAR_MAE = float(
    linear_row[
        "MAE"
    ]
)
LINEAR_RMSE = float(
    linear_row[
        "RMSE"
    ]
)
LINEAR_R2 = float(
    linear_row[
        "R2"
    ]
)

RF_MAE = float(
    rf_row[
        "MAE"
    ]
)
RF_RMSE = float(
    rf_row[
        "RMSE"
    ]
)
RF_R2 = float(
    rf_row[
        "R2"
    ]
)

GB_MAE = float(
    gb_row[
        "MAE"
    ]
)
GB_RMSE = float(
    gb_row[
        "RMSE"
    ]
)
GB_R2 = float(
    gb_row[
        "R2"
    ]
)

ET_MAE = float(
    et_row[
        "MAE"
    ]
)
ET_RMSE = float(
    et_row[
        "RMSE"
    ]
)
ET_R2 = float(
    et_row[
        "R2"
    ]
)


# ============================================================
# 9. FIXED XGBOOST CONFIGURATION
# ============================================================
#
# This is intentionally ONE fixed untuned experiment.
#
# max_depth=3:
#   shallow trees to control model complexity.
#
# learning_rate=0.05, n_estimators=300:
#   moderate shrinkage with sufficient boosting rounds.
#
# subsample / colsample_bytree=0.90:
#   introduce controlled stochastic regularization.
#
# reg_lambda=1.0:
#   standard L2 regularization.
#
# tree_method="hist":
#   efficient histogram-based tree construction.
#
# Stage 18 remains the dedicated hyperparameter tuning stage.

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 1.0,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

model = XGBRegressor(
    **XGB_PARAMS
)


# ============================================================
# 10. FIT TRAIN ONLY
# ============================================================

print("\nTraining XGBoost on TRAIN only...")

model.fit(
    X_train,
    y_train,
    verbose=False,
)

print("XGBoost training complete.")


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
                "XGBoost"
            ),
            "Split": "Train",
            **train_metrics,
        },
        {
            "Model": (
                "XGBoost"
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
# 13. COMPARISON WITH CURRENT BEST
# ============================================================

mae_improvement_vs_linear = (
    (
        LINEAR_MAE
        -
        validation_metrics[
            "MAE"
        ]
    )
    /
    LINEAR_MAE
    *
    100
)

mae_improvement_vs_rf = (
    (
        RF_MAE
        -
        validation_metrics[
            "MAE"
        ]
    )
    /
    RF_MAE
    *
    100
)

mae_improvement_vs_gb = (
    (
        GB_MAE
        -
        validation_metrics[
            "MAE"
        ]
    )
    /
    GB_MAE
    *
    100
)

rmse_improvement_vs_gb = (
    (
        GB_RMSE
        -
        validation_metrics[
            "RMSE"
        ]
    )
    /
    GB_RMSE
    *
    100
)

r2_change_vs_gb = (
    validation_metrics[
        "R2"
    ]
    -
    GB_R2
)

beats_gradient_boosting_mae = bool(
    validation_metrics[
        "MAE"
    ]
    <
    GB_MAE
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
    "Predicted_XGBoost"
] = validation_pred

predictions_df[
    "Residual_XGBoost"
] = (
    y_validation
    -
    validation_pred
)

predictions_df[
    "Absolute_Error_XGBoost"
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
        atol=1e-6,
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
                f"Shape={X_train.shape}"
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
                f"Shape={X_validation.shape}"
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
                "Previous validation results loaded"
            ),
            "Status": "PASS",
            "Details": (
                f"Linear={LINEAR_MAE:.6f}; "
                f"RF={RF_MAE:.6f}; "
                f"GB={GB_MAE:.6f}; "
                f"ET={ET_MAE:.6f}"
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
                "Validation used only for development evaluation."
            ),
        },
        {
            "Check": (
                "No hyperparameter tuning performed"
            ),
            "Status": "PASS",
            "Details": (
                "Single fixed XGBoost configuration; "
                "Stage 18 reserved for tuning."
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
        "15_xgboost_regression"
    ),
    "model": (
        "XGBRegressor"
    ),
    "fit_scope": (
        "TRAIN_ONLY"
    ),
    "validation_scope": (
        "EVALUATION_ONLY"
    ),
    "test_loaded": False,
    "hyperparameter_tuning_performed": False,
    "parameters": XGB_PARAMS,
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
    "previous_validation_results": {
        "Full_Linear": {
            "MAE": LINEAR_MAE,
            "RMSE": LINEAR_RMSE,
            "R2": LINEAR_R2,
        },
        "Random_Forest": {
            "MAE": RF_MAE,
            "RMSE": RF_RMSE,
            "R2": RF_R2,
        },
        "Gradient_Boosting": {
            "MAE": GB_MAE,
            "RMSE": GB_RMSE,
            "R2": GB_R2,
        },
        "Extra_Trees": {
            "MAE": ET_MAE,
            "RMSE": ET_RMSE,
            "R2": ET_R2,
        },
    },
    "xgboost_validation": {
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
    "beats_gradient_boosting_by_mae": (
        beats_gradient_boosting_mae
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
    "\n--- FIXED XGBOOST CONFIGURATION ---"
)

for key, value in (
    XGB_PARAMS.items()
):
    print(
        f"{key:<31}: {value}"
    )


print(
    "\n--- XGBOOST RESULTS ---"
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
    "\n--- PREVIOUS VALIDATION RESULTS ---"
)

print(
    f"Full Linear MAE / RMSE / R2     : "
    f"{LINEAR_MAE:.4f} / "
    f"{LINEAR_RMSE:.4f} / "
    f"{LINEAR_R2:.4f}"
)

print(
    f"Random Forest MAE / RMSE / R2   : "
    f"{RF_MAE:.4f} / "
    f"{RF_RMSE:.4f} / "
    f"{RF_R2:.4f}"
)

print(
    f"Gradient Boost MAE / RMSE / R2  : "
    f"{GB_MAE:.4f} / "
    f"{GB_RMSE:.4f} / "
    f"{GB_R2:.4f}"
)

print(
    f"Extra Trees MAE / RMSE / R2     : "
    f"{ET_MAE:.4f} / "
    f"{ET_RMSE:.4f} / "
    f"{ET_R2:.4f}"
)


print(
    "\n--- XGBOOST COMPARISON ---"
)

print(
    f"MAE improvement vs Full Linear  : "
    f"{mae_improvement_vs_linear:.2f}%"
)

print(
    f"MAE improvement vs Random Forest: "
    f"{mae_improvement_vs_rf:.2f}%"
)

print(
    f"MAE improvement vs Grad Boost   : "
    f"{mae_improvement_vs_gb:.2f}%"
)

print(
    f"RMSE improvement vs Grad Boost  : "
    f"{rmse_improvement_vs_gb:.2f}%"
)

print(
    f"R2 change vs Gradient Boosting  : "
    f"{r2_change_vs_gb:+.4f}"
)

print(
    f"Beats Gradient Boosting by MAE  : "
    f"{beats_gradient_boosting_mae}"
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
        "COMPONENT 3 - STAGE 15: XGBOOST REGRESSION\n"
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
        "Stage 15 trains a fixed, untuned XGBoost regressor. "
        "XGBoost is a regularized gradient-boosted tree method "
        "that sequentially corrects residual error while using "
        "row/feature subsampling and explicit regularization. "
        "The model is fitted on Train only and evaluated on "
        "Validation only.\n\n"
    )

    report.write(
        "FIXED CONFIGURATION\n"
        "-------------------\n"
    )

    for key, value in (
        XGB_PARAMS.items()
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
        "COMPARISON WITH PREVIOUS MODELS\n"
        "-------------------------------\n"
    )

    report.write(
        f"Full Linear Validation MAE: "
        f"{LINEAR_MAE:.6f} h\n"
    )

    report.write(
        f"Random Forest Validation MAE: "
        f"{RF_MAE:.6f} h\n"
    )

    report.write(
        f"Gradient Boosting Validation MAE: "
        f"{GB_MAE:.6f} h\n"
    )

    report.write(
        f"Extra Trees Validation MAE: "
        f"{ET_MAE:.6f} h\n"
    )

    report.write(
        f"XGBoost Validation MAE: "
        f"{validation_metrics['MAE']:.6f} h\n"
    )

    report.write(
        f"MAE improvement vs Full Linear: "
        f"{mae_improvement_vs_linear:.6f}%\n"
    )

    report.write(
        f"MAE improvement vs Random Forest: "
        f"{mae_improvement_vs_rf:.6f}%\n"
    )

    report.write(
        f"MAE improvement vs Gradient Boosting: "
        f"{mae_improvement_vs_gb:.6f}%\n"
    )

    report.write(
        f"RMSE improvement vs Gradient Boosting: "
        f"{rmse_improvement_vs_gb:.6f}%\n"
    )

    report.write(
        f"R² change vs Gradient Boosting: "
        f"{r2_change_vs_gb:+.6f}\n"
    )

    report.write(
        f"Beats Gradient Boosting by Validation MAE: "
        f"{beats_gradient_boosting_mae}\n\n"
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
        "XGBoost feature importance is useful for preliminary "
        "inspection but should not be interpreted as causal "
        "importance. Final explainability is reserved for "
        "Stage 32 using SHAP.\n\n"
    )

    report.write(
        "LEAKAGE PROTECTION\n"
        "------------------\n"
    )

    report.write(
        "- XGBoost fitted on Train only.\n"
    )

    report.write(
        "- Validation used only for development evaluation.\n"
    )

    report.write(
        "- No hyperparameter search was performed.\n"
    )

    report.write(
        "- Final Test rows were not loaded.\n"
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
        f"STAGE 15 STATUS: "
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
    f"STAGE 15 XGBOOST REGRESSION: "
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
        "Stage 15 failed. Review "
        "15_xgboost_regression_report.txt."
    )
