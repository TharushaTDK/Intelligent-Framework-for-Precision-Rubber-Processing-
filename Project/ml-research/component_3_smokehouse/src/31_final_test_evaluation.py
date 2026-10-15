from pathlib import Path
import ast
import json
import math

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


# ============================================================
# COMPONENT 3 - STAGE 31:
# LOCKED FINAL TEST EVALUATION
# ============================================================
#
# Purpose:
#   Evaluate the already-selected regression and classification
#   configurations ONCE on the frozen final holdout batches.
#
# Frozen selections:
#
#   Regression
#   ----------
#   Stage 19: Gradient Boosting GB_02
#
#   Classification
#   --------------
#   Stage 29: XGBoost XGB_03
#
# Method:
#
#   1) Load Train + Validation only.
#   2) Combine them into the full DEVELOPMENT dataset.
#   3) Fit Stage-10 preprocessing on DEVELOPMENT only.
#   4) Refit the frozen selected models on DEVELOPMENT only.
#   5) Only after every training step is complete:
#         OPEN Component3_Test.csv.
#   6) Transform Test with the DEVELOPMENT-fitted preprocessor.
#   7) Predict and evaluate.
#
# IMPORTANT:
#   - Test is NEVER used to fit preprocessing.
#   - Test is NEVER used to fit models.
#   - Test metrics are NEVER used for tuning or reselection.
#   - No SMOTE / resampling.
#   - XGBoost balanced sample weights are derived from
#     DEVELOPMENT labels only.
#   - Predictions are NOT clipped before evaluation.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "outputs" / "reports"

REG_FINAL_DIR = (
    ROOT
    / "artifacts"
    / "models"
    / "regression"
    / "final"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_PATH = (
    SPLIT_DIR
    / "Component3_Train.csv"
)

VALIDATION_PATH = (
    SPLIT_DIR
    / "Component3_Validation.csv"
)

TEST_PATH = (
    SPLIT_DIR
    / "Component3_Test.csv"
)

REG_FEATURE_LIST_PATH = (
    REPORT_DIR
    / "08_regression_selected_features.csv"
)

CLS_FEATURE_LIST_PATH = (
    REPORT_DIR
    / "08_classification_selected_features.csv"
)

REG_SELECTION_SUMMARY_PATH = (
    REPORT_DIR
    / "19_final_regression_selection_summary.json"
)

CLS_SELECTION_CONFIG_PATH = (
    REPORT_DIR
    / "29_selected_classification_configuration.csv"
)

REG_SELECTED_ARTIFACT_PATH = (
    REG_FINAL_DIR
    / "selected_regression_model.joblib"
)

STAGE30_SUMMARY_PATH = (
    REPORT_DIR
    / "30_group_cross_validation_summary.json"
)


# ============================================================
# 2. OUTPUTS
# ============================================================

REG_METRICS_PATH = (
    REPORT_DIR
    / "31_regression_final_test_metrics.csv"
)

REG_PREDICTIONS_PATH = (
    REPORT_DIR
    / "31_regression_final_test_predictions.csv"
)

REG_BATCH_PATH = (
    REPORT_DIR
    / "31_regression_final_test_by_batch.csv"
)

REG_LAYER_PATH = (
    REPORT_DIR
    / "31_regression_final_test_by_layer.csv"
)

CLS_METRICS_PATH = (
    REPORT_DIR
    / "31_classification_final_test_metrics.csv"
)

CLS_PREDICTIONS_PATH = (
    REPORT_DIR
    / "31_classification_final_test_predictions.csv"
)

CLS_CM_PATH = (
    REPORT_DIR
    / "31_classification_final_test_confusion_matrix.csv"
)

CLS_CLASS_REPORT_PATH = (
    REPORT_DIR
    / "31_classification_final_test_per_class.csv"
)

CLS_BATCH_PATH = (
    REPORT_DIR
    / "31_classification_final_test_by_batch.csv"
)

CLS_LAYER_PATH = (
    REPORT_DIR
    / "31_classification_final_test_by_layer.csv"
)

COMPARISON_PATH = (
    REPORT_DIR
    / "31_stage30_vs_final_test_comparison.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "31_final_test_evaluation_summary.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "31_final_test_evaluation_report.txt"
)


# ============================================================
# 3. CONSTANTS
# ============================================================

CLASS_ORDER_TEXT = [
    "Normal",
    "Warning",
    "Critical",
]

CLASS_TO_INT = {
    "Normal": 0,
    "Warning": 1,
    "Critical": 2,
}

INT_TO_CLASS = {
    value: key
    for key, value in CLASS_TO_INT.items()
}

MIN_WARNING_RECALL_REFERENCE = 0.80
MIN_CRITICAL_RECALL_REFERENCE = 0.50


# ============================================================
# 4. HELPERS
# ============================================================

def make_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def make_preprocessor(
    numeric_features,
    categorical_features,
):
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                make_one_hot_encoder(),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def rmse(
    y_true,
    y_pred,
):
    return float(
        math.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def parse_dict_like(value):
    if isinstance(value, dict):
        return value

    text = str(value).strip()

    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}


def recover_selected_xgb_configuration(
    path,
):
    frame = pd.read_csv(path)

    if frame.empty:
        raise RuntimeError(
            "Stage-29 selected classification "
            "configuration is empty."
        )

    selected = frame.iloc[0]

    family = (
        str(selected["Model_Family"])
        if "Model_Family" in frame.columns
        else "Unknown"
    )

    trial = (
        str(selected["Trial"])
        if "Trial" in frame.columns
        else "Unknown"
    )

    params = {}

    for column in [
        "Parameters_JSON",
        "Best_Parameters",
        "Parameters",
        "Params",
    ]:
        if column in frame.columns:
            params = parse_dict_like(
                selected[column]
            )

            if params:
                break

    required = [
        "n_estimators",
        "learning_rate",
        "max_depth",
    ]

    missing = [
        parameter
        for parameter in required
        if parameter not in params
    ]

    if missing:
        raise RuntimeError(
            "Could not recover exact Stage-29 "
            "XGBoost configuration. "
            f"Missing: {missing}"
        )

    return (
        family,
        trial,
        params,
    )


def fixed_three_class_metrics(
    y_true,
    y_pred,
):
    return {
        "Accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "Balanced_Accuracy": float(
            recall_score(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "Macro_Precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "Macro_Recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "Macro_F1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
    }


def class_recall(
    y_true,
    y_pred,
    class_id,
):
    support = int(
        np.sum(
            np.asarray(y_true)
            == class_id
        )
    )

    if support == 0:
        return np.nan

    return float(
        recall_score(
            y_true,
            y_pred,
            labels=[class_id],
            average="macro",
            zero_division=0,
        )
    )


def regression_group_metrics(
    frame,
    group_column,
):
    rows = []

    for group_value, group in frame.groupby(
        group_column,
        sort=True,
    ):
        y_true = group[
            "Remaining_Drying_Time"
        ].to_numpy()

        y_pred = group[
            "Predicted_Remaining_Drying_Time"
        ].to_numpy()

        rows.append({
            group_column:
                group_value,
            "Rows":
                len(group),
            "MAE":
                float(
                    mean_absolute_error(
                        y_true,
                        y_pred,
                    )
                ),
            "RMSE":
                rmse(
                    y_true,
                    y_pred,
                ),
            "R2":
                float(
                    r2_score(
                        y_true,
                        y_pred,
                    )
                ),
            "Negative_Predictions":
                int(
                    np.sum(
                        y_pred < 0
                    )
                ),
        })

    return pd.DataFrame(rows)


def classification_group_metrics(
    frame,
    group_column,
):
    rows = []

    for group_value, group in frame.groupby(
        group_column,
        sort=True,
    ):
        y_true = group[
            "y_true_int"
        ].to_numpy()

        y_pred = group[
            "y_pred_int"
        ].to_numpy()

        metrics = (
            fixed_three_class_metrics(
                y_true,
                y_pred,
            )
        )

        rows.append({
            group_column:
                group_value,
            "Rows":
                len(group),
            "Normal_Support":
                int(
                    np.sum(
                        y_true == 0
                    )
                ),
            "Warning_Support":
                int(
                    np.sum(
                        y_true == 1
                    )
                ),
            "Critical_Support":
                int(
                    np.sum(
                        y_true == 2
                    )
                ),
            **metrics,
            "Warning_Recall":
                class_recall(
                    y_true,
                    y_pred,
                    1,
                ),
            "Critical_Recall":
                class_recall(
                    y_true,
                    y_pred,
                    2,
                ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. VERIFY PRE-TEST REQUIREMENTS
# ============================================================

required_files = [
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
    REG_FEATURE_LIST_PATH,
    CLS_FEATURE_LIST_PATH,
    REG_SELECTION_SUMMARY_PATH,
    CLS_SELECTION_CONFIG_PATH,
    REG_SELECTED_ARTIFACT_PATH,
    STAGE30_SUMMARY_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# 6. VERIFY STAGE 30 BEFORE TEST ACCESS
# ============================================================

with open(
    STAGE30_SUMMARY_PATH,
    "r",
    encoding="utf-8",
) as handle:
    stage30_summary = json.load(
        handle
    )

stage30_checks = (
    stage30_summary.get(
        "methodological_checks",
        {},
    )
)

stage30_batch_overlap = (
    stage30_checks.get(
        "batch_overlap_folds"
    )
)

stage30_fold_preprocessing = (
    stage30_checks.get(
        "all_fold_preprocessors_fit_train_only"
    )
)

stage30_test_read = (
    stage30_checks.get(
        "final_test_content_read"
    )
)

if (
    stage30_batch_overlap != 0
    or stage30_fold_preprocessing is not True
    or stage30_test_read is not False
):
    raise RuntimeError(
        "Stage 30 methodological checks are not clean. "
        "Final Test will NOT be opened."
    )


# ============================================================
# 7. LOAD FROZEN FEATURE DEFINITIONS
# ============================================================

reg_feature_df = pd.read_csv(
    REG_FEATURE_LIST_PATH
)

cls_feature_df = pd.read_csv(
    CLS_FEATURE_LIST_PATH
)

REGRESSION_FEATURES = (
    reg_feature_df[
        "Feature"
    ]
    .astype(str)
    .tolist()
)

CLASSIFICATION_FEATURES = (
    cls_feature_df[
        "Feature"
    ]
    .astype(str)
    .tolist()
)

if (
    REGRESSION_FEATURES
    != CLASSIFICATION_FEATURES
):
    raise RuntimeError(
        "Frozen Stage-08 regression and "
        "classification feature lists do not match."
    )

SELECTED_FEATURES = (
    REGRESSION_FEATURES
)

if len(SELECTED_FEATURES) != 17:
    raise RuntimeError(
        "Expected 17 frozen Stage-08 features, "
        f"found {len(SELECTED_FEATURES)}."
    )

CATEGORICAL_FEATURES = [
    "Smokehouse_ID",
    "Layer_ID",
]

NUMERIC_FEATURES = [
    feature
    for feature in SELECTED_FEATURES
    if feature
    not in CATEGORICAL_FEATURES
]

if len(NUMERIC_FEATURES) != 15:
    raise RuntimeError(
        "Expected 15 numeric selected features, "
        f"found {len(NUMERIC_FEATURES)}."
    )

for forbidden in [
    "Batch_ID",
    "Timestamp",
    "Actual_Total_Drying_Time",
    "Remaining_Drying_Time",
    "Fire_Risk_Label",
]:
    if forbidden in SELECTED_FEATURES:
        raise RuntimeError(
            "Forbidden leakage/group field found "
            f"in model features: {forbidden}"
        )


# ============================================================
# 8. RECOVER FROZEN SELECTED MODELS
# ============================================================

with open(
    REG_SELECTION_SUMMARY_PATH,
    "r",
    encoding="utf-8",
) as handle:
    regression_selection = json.load(
        handle
    )

selected_reg_family = (
    regression_selection.get(
        "selected_model_family"
    )
)

selected_reg_trial = (
    regression_selection.get(
        "selected_trial_id"
    )
)

selected_reg_artifact = (
    joblib.load(
        REG_SELECTED_ARTIFACT_PATH
    )
)

if not isinstance(
    selected_reg_artifact,
    GradientBoostingRegressor,
):
    raise RuntimeError(
        "Stage-19 selected regression artifact "
        "is not GradientBoostingRegressor."
    )

(
    selected_cls_family,
    selected_cls_trial,
    xgb_params,
) = recover_selected_xgb_configuration(
    CLS_SELECTION_CONFIG_PATH
)


# ============================================================
# 9. LOAD DEVELOPMENT DATA ONLY
# ============================================================

print("=" * 78)
print(
    "COMPONENT 3 - STAGE 31: "
    "LOCKED FINAL TEST EVALUATION"
)
print("=" * 78)

print(
    "\nStage 30 methodological checks: PASS"
)
print(
    "Final Test has NOT been opened yet in this script."
)

train_df = pd.read_csv(
    TRAIN_PATH
)

validation_df = pd.read_csv(
    VALIDATION_PATH
)

development = pd.concat(
    [
        train_df.assign(
            _Original_Split="Train"
        ),
        validation_df.assign(
            _Original_Split="Validation"
        ),
    ],
    ignore_index=True,
)

required_development_columns = (
    SELECTED_FEATURES
    + [
        "Batch_ID",
        "Layer_ID",
        "Remaining_Drying_Time",
        "Fire_Risk_Label",
    ]
)

missing_dev = [
    column
    for column in required_development_columns
    if column not in development.columns
]

if missing_dev:
    raise RuntimeError(
        "Development data missing required columns: "
        f"{missing_dev}"
    )

for feature in NUMERIC_FEATURES:
    development[
        feature
    ] = pd.to_numeric(
        development[
            feature
        ],
        errors="coerce",
    )

development[
    "Remaining_Drying_Time"
] = pd.to_numeric(
    development[
        "Remaining_Drying_Time"
    ],
    errors="raise",
)

development[
    "Fire_Risk_Label"
] = (
    development[
        "Fire_Risk_Label"
    ]
    .astype(str)
    .str.strip()
)

unexpected_dev_labels = sorted(
    set(
        development[
            "Fire_Risk_Label"
        ].unique()
    )
    - set(CLASS_ORDER_TEXT)
)

if unexpected_dev_labels:
    raise RuntimeError(
        "Unexpected DEVELOPMENT classification labels: "
        f"{unexpected_dev_labels}"
    )

development[
    "Fire_Risk_Int"
] = (
    development[
        "Fire_Risk_Label"
    ]
    .map(
        CLASS_TO_INT
    )
)

development_batches = set(
    development[
        "Batch_ID"
    ]
    .astype(str)
    .unique()
)

X_development_raw = (
    development[
        SELECTED_FEATURES
    ].copy()
)

y_reg_development = (
    development[
        "Remaining_Drying_Time"
    ]
    .to_numpy(
        dtype=float
    )
)

y_cls_development = (
    development[
        "Fire_Risk_Int"
    ]
    .to_numpy(
        dtype=int
    )
)

print("\n--- FULL DEVELOPMENT DATA ---")
print(
    f"Train rows          : {len(train_df):,}"
)
print(
    f"Validation rows     : {len(validation_df):,}"
)
print(
    f"Development rows    : {len(development):,}"
)
print(
    f"Development batches : "
    f"{len(development_batches)}"
)


# ============================================================
# 10. FIT PREPROCESSOR ON DEVELOPMENT ONLY
# ============================================================

preprocessor = (
    make_preprocessor(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
    )
)

preprocessor.fit(
    X_development_raw
)

X_development = (
    preprocessor.transform(
        X_development_raw
    )
)

transformed_feature_names = (
    preprocessor
    .get_feature_names_out()
    .tolist()
)

print(
    "\nPreprocessor fitted on DEVELOPMENT "
    "(Train + Validation) ONLY."
)
print(
    "Transformed feature count:",
    len(
        transformed_feature_names
    ),
)


# ============================================================
# 11. REFIT FROZEN MODELS ON DEVELOPMENT ONLY
# ============================================================

regression_model = clone(
    selected_reg_artifact
)

regression_model.fit(
    X_development,
    y_reg_development,
)

development_sample_weights = (
    compute_sample_weight(
        class_weight="balanced",
        y=y_cls_development,
    )
)

classification_model = (
    XGBClassifier(
        **xgb_params
    )
)

classification_model.fit(
    X_development,
    y_cls_development,
    sample_weight=(
        development_sample_weights
    ),
)

print("\n--- FROZEN CONFIGURATIONS REFIT ---")
print(
    "Regression     : "
    f"{selected_reg_family} / "
    f"{selected_reg_trial}"
)
print(
    "Classification : "
    f"{selected_cls_family} / "
    f"{selected_cls_trial}"
)
print(
    "Classification balanced weights derived "
    "from DEVELOPMENT labels only."
)

print(
    "\nAll fitting is complete."
)
print(
    "No test row has influenced preprocessing, "
    "weighting, or model fitting."
)


# ============================================================
# 12. OPEN LOCKED FINAL TEST
# ============================================================

print("\n" + "=" * 78)
print("OPENING LOCKED FINAL TEST FOR STAGE 31")
print("=" * 78)

# This is the FIRST point in this script where Test CONTENT
# is loaded.
test_df = pd.read_csv(
    TEST_PATH
)

final_test_content_read = True

print(
    "Component3_Test.csv content read:",
    final_test_content_read,
)
print(
    "This dataset is now being used for FINAL "
    "HOLDOUT EVALUATION ONLY."
)


# ============================================================
# 13. VALIDATE TEST STRUCTURE / GROUP ISOLATION
# ============================================================

required_test_columns = (
    SELECTED_FEATURES
    + [
        "Batch_ID",
        "Layer_ID",
        "Remaining_Drying_Time",
        "Fire_Risk_Label",
    ]
)

missing_test = [
    column
    for column in required_test_columns
    if column not in test_df.columns
]

if missing_test:
    raise RuntimeError(
        "Final Test missing required columns: "
        f"{missing_test}"
    )

for feature in NUMERIC_FEATURES:
    test_df[
        feature
    ] = pd.to_numeric(
        test_df[
            feature
        ],
        errors="coerce",
    )

test_df[
    "Remaining_Drying_Time"
] = pd.to_numeric(
    test_df[
        "Remaining_Drying_Time"
    ],
    errors="raise",
)

test_df[
    "Fire_Risk_Label"
] = (
    test_df[
        "Fire_Risk_Label"
    ]
    .astype(str)
    .str.strip()
)

unexpected_test_labels = sorted(
    set(
        test_df[
            "Fire_Risk_Label"
        ].unique()
    )
    - set(CLASS_ORDER_TEXT)
)

if unexpected_test_labels:
    raise RuntimeError(
        "Unexpected Final-Test labels: "
        f"{unexpected_test_labels}"
    )

test_df[
    "Fire_Risk_Int"
] = (
    test_df[
        "Fire_Risk_Label"
    ]
    .map(
        CLASS_TO_INT
    )
)

test_batches = set(
    test_df[
        "Batch_ID"
    ]
    .astype(str)
    .unique()
)

batch_overlap = (
    development_batches
    & test_batches
)

if batch_overlap:
    raise RuntimeError(
        "FATAL: Development / Final-Test "
        "Batch_ID overlap detected: "
        f"{sorted(batch_overlap)}"
    )

combined_keys = pd.concat(
    [
        development[
            [
                "Smokehouse_ID",
                "Batch_ID",
                "Layer_ID",
                "Timestamp",
            ]
        ],
        test_df[
            [
                "Smokehouse_ID",
                "Batch_ID",
                "Layer_ID",
                "Timestamp",
            ]
        ],
    ],
    ignore_index=True,
)

duplicate_cross_split_keys = int(
    combined_keys.duplicated(
        subset=[
            "Smokehouse_ID",
            "Batch_ID",
            "Layer_ID",
            "Timestamp",
        ],
        keep=False,
    ).sum()
)

if duplicate_cross_split_keys > 0:
    raise RuntimeError(
        "Observation-key overlap exists across "
        "Development and Final Test."
    )

print("\n--- FINAL TEST STRUCTURE ---")
print(
    f"Test rows       : {len(test_df):,}"
)
print(
    f"Test batches    : {len(test_batches)}"
)
print(
    "Test batch IDs  : "
    + ", ".join(
        sorted(
            test_batches
        )
    )
)
print(
    "Development/Test Batch_ID overlap:",
    len(batch_overlap),
)
print(
    "Development/Test observation-key overlap:",
    duplicate_cross_split_keys,
)

print("\n--- FINAL TEST CLASS SUPPORT ---")
for label in CLASS_ORDER_TEXT:
    print(
        f"{label:<8}: "
        f"{int((test_df['Fire_Risk_Label'] == label).sum())}"
    )


# ============================================================
# 14. TRANSFORM FINAL TEST
# ============================================================

X_test_raw = (
    test_df[
        SELECTED_FEATURES
    ].copy()
)

X_test = (
    preprocessor.transform(
        X_test_raw
    )
)

if (
    X_test.shape[1]
    != len(
        transformed_feature_names
    )
):
    raise RuntimeError(
        "Final-Test transformed feature count "
        "does not match development feature count."
    )

print(
    "\nFinal Test transformed with the already-fitted "
    "DEVELOPMENT preprocessor."
)
print(
    "Test was NOT used in preprocessor.fit()."
)


# ============================================================
# 15. REGRESSION FINAL TEST
# ============================================================

y_reg_test = (
    test_df[
        "Remaining_Drying_Time"
    ]
    .to_numpy(
        dtype=float
    )
)

reg_pred = (
    regression_model.predict(
        X_test
    )
)

reg_mae = float(
    mean_absolute_error(
        y_reg_test,
        reg_pred,
    )
)

reg_rmse = rmse(
    y_reg_test,
    reg_pred,
)

reg_r2 = float(
    r2_score(
        y_reg_test,
        reg_pred,
    )
)

reg_negative = int(
    np.sum(
        reg_pred < 0
    )
)

development_target_max = float(
    np.max(
        y_reg_development
    )
)

reg_above_dev_max = int(
    np.sum(
        reg_pred
        > development_target_max
    )
)

reg_metrics_df = pd.DataFrame([
    {
        "Model_Family":
            selected_reg_family,
        "Trial":
            selected_reg_trial,
        "Test_Rows":
            len(test_df),
        "Test_Batches":
            len(test_batches),
        "MAE":
            reg_mae,
        "RMSE":
            reg_rmse,
        "R2":
            reg_r2,
        "Negative_Predictions":
            reg_negative,
        "Predictions_Above_Development_Target_Max":
            reg_above_dev_max,
        "Development_Target_Max":
            development_target_max,
    }
])

reg_prediction_columns = [
    column
    for column in [
        "Timestamp",
        "Smokehouse_ID",
        "Batch_ID",
        "Layer_ID",
    ]
    if column in test_df.columns
]

reg_predictions_df = (
    test_df[
        reg_prediction_columns
    ].copy()
)

reg_predictions_df[
    "Remaining_Drying_Time"
] = y_reg_test

reg_predictions_df[
    "Predicted_Remaining_Drying_Time"
] = reg_pred

reg_predictions_df[
    "Error"
] = (
    reg_pred
    - y_reg_test
)

reg_predictions_df[
    "Absolute_Error"
] = np.abs(
    reg_predictions_df[
        "Error"
    ]
)

reg_by_batch_df = (
    regression_group_metrics(
        reg_predictions_df,
        "Batch_ID",
    )
)

reg_by_layer_df = (
    regression_group_metrics(
        reg_predictions_df,
        "Layer_ID",
    )
)


# ============================================================
# 16. CLASSIFICATION FINAL TEST
# ============================================================

y_cls_test = (
    test_df[
        "Fire_Risk_Int"
    ]
    .to_numpy(
        dtype=int
    )
)

cls_pred = (
    classification_model.predict(
        X_test
    )
    .astype(int)
)

cls_metrics = (
    fixed_three_class_metrics(
        y_cls_test,
        cls_pred,
    )
)

warning_recall = (
    class_recall(
        y_cls_test,
        cls_pred,
        1,
    )
)

critical_recall = (
    class_recall(
        y_cls_test,
        cls_pred,
        2,
    )
)

warning_reference_pass = (
    False
    if np.isnan(
        warning_recall
    )
    else bool(
        warning_recall
        >= MIN_WARNING_RECALL_REFERENCE
    )
)

critical_reference_pass = (
    False
    if np.isnan(
        critical_recall
    )
    else bool(
        critical_recall
        >= MIN_CRITICAL_RECALL_REFERENCE
    )
)

cls_metrics_df = pd.DataFrame([
    {
        "Model_Family":
            selected_cls_family,
        "Trial":
            selected_cls_trial,
        "Test_Rows":
            len(test_df),
        "Test_Batches":
            len(test_batches),
        **cls_metrics,
        "Warning_Recall":
            warning_recall,
        "Critical_Recall":
            critical_recall,
        "Warning_Reference_0_80_Pass":
            warning_reference_pass,
        "Critical_Reference_0_50_Pass":
            critical_reference_pass,
    }
])

cls_prediction_columns = [
    column
    for column in [
        "Timestamp",
        "Smokehouse_ID",
        "Batch_ID",
        "Layer_ID",
    ]
    if column in test_df.columns
]

cls_predictions_df = (
    test_df[
        cls_prediction_columns
    ].copy()
)

cls_predictions_df[
    "y_true_int"
] = y_cls_test

cls_predictions_df[
    "y_pred_int"
] = cls_pred

cls_predictions_df[
    "Actual_Fire_Risk_Label"
] = [
    INT_TO_CLASS[
        int(value)
    ]
    for value in y_cls_test
]

cls_predictions_df[
    "Predicted_Fire_Risk_Label"
] = [
    INT_TO_CLASS[
        int(value)
    ]
    for value in cls_pred
]

cm = confusion_matrix(
    y_cls_test,
    cls_pred,
    labels=[0, 1, 2],
)

cls_cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_Normal",
        "Actual_Warning",
        "Actual_Critical",
    ],
    columns=[
        "Predicted_Normal",
        "Predicted_Warning",
        "Predicted_Critical",
    ],
)

(
    per_class_precision,
    per_class_recall,
    per_class_f1,
    per_class_support,
) = precision_recall_fscore_support(
    y_cls_test,
    cls_pred,
    labels=[0, 1, 2],
    zero_division=0,
)

cls_per_class_df = pd.DataFrame({
    "Class":
        CLASS_ORDER_TEXT,
    "Precision":
        per_class_precision,
    "Recall":
        per_class_recall,
    "F1":
        per_class_f1,
    "Support":
        per_class_support,
})

cls_by_batch_df = (
    classification_group_metrics(
        cls_predictions_df,
        "Batch_ID",
    )
)

cls_by_layer_df = (
    classification_group_metrics(
        cls_predictions_df,
        "Layer_ID",
    )
)


# ============================================================
# 17. STAGE 30 VS FINAL TEST COMPARISON
# ============================================================

stage30_reg = (
    stage30_summary.get(
        "regression",
        {},
    )
)

stage30_cls = (
    stage30_summary.get(
        "classification",
        {},
    )
)

comparison_rows = [
    {
        "Task":
            "Regression",
        "Metric":
            "MAE",
        "Stage30_OOF":
            stage30_reg.get(
                "pooled_oof_mae"
            ),
        "Stage31_Final_Test":
            reg_mae,
    },
    {
        "Task":
            "Regression",
        "Metric":
            "RMSE",
        "Stage30_OOF":
            stage30_reg.get(
                "pooled_oof_rmse"
            ),
        "Stage31_Final_Test":
            reg_rmse,
    },
    {
        "Task":
            "Regression",
        "Metric":
            "R2",
        "Stage30_OOF":
            stage30_reg.get(
                "pooled_oof_r2"
            ),
        "Stage31_Final_Test":
            reg_r2,
    },
    {
        "Task":
            "Classification",
        "Metric":
            "Balanced_Accuracy",
        "Stage30_OOF":
            stage30_cls.get(
                "pooled_oof_balanced_accuracy"
            ),
        "Stage31_Final_Test":
            cls_metrics[
                "Balanced_Accuracy"
            ],
    },
    {
        "Task":
            "Classification",
        "Metric":
            "Macro_F1",
        "Stage30_OOF":
            stage30_cls.get(
                "pooled_oof_macro_f1"
            ),
        "Stage31_Final_Test":
            cls_metrics[
                "Macro_F1"
            ],
    },
    {
        "Task":
            "Classification",
        "Metric":
            "Warning_Recall",
        "Stage30_OOF":
            stage30_cls.get(
                "pooled_oof_warning_recall"
            ),
        "Stage31_Final_Test":
            warning_recall,
    },
    {
        "Task":
            "Classification",
        "Metric":
            "Critical_Recall",
        "Stage30_OOF":
            stage30_cls.get(
                "pooled_oof_critical_recall"
            ),
        "Stage31_Final_Test":
            critical_recall,
    },
]

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df[
    "Final_Minus_Stage30"
] = (
    pd.to_numeric(
        comparison_df[
            "Stage31_Final_Test"
        ],
        errors="coerce",
    )
    -
    pd.to_numeric(
        comparison_df[
            "Stage30_OOF"
        ],
        errors="coerce",
    )
)


# ============================================================
# 18. SAVE ALL EVIDENCE
# ============================================================

reg_metrics_df.to_csv(
    REG_METRICS_PATH,
    index=False,
)

reg_predictions_df.to_csv(
    REG_PREDICTIONS_PATH,
    index=False,
)

reg_by_batch_df.to_csv(
    REG_BATCH_PATH,
    index=False,
)

reg_by_layer_df.to_csv(
    REG_LAYER_PATH,
    index=False,
)

cls_metrics_df.to_csv(
    CLS_METRICS_PATH,
    index=False,
)

cls_predictions_df.to_csv(
    CLS_PREDICTIONS_PATH,
    index=False,
)

cls_cm_df.to_csv(
    CLS_CM_PATH,
    index=True,
)

cls_per_class_df.to_csv(
    CLS_CLASS_REPORT_PATH,
    index=False,
)

cls_by_batch_df.to_csv(
    CLS_BATCH_PATH,
    index=False,
)

cls_by_layer_df.to_csv(
    CLS_LAYER_PATH,
    index=False,
)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False,
)


# ============================================================
# 19. FINAL SUMMARY JSON
# ============================================================

summary = {
    "stage": 31,
    "evaluation_scope":
        "LOCKED_FINAL_TEST",
    "method": {
        "development_refit":
            "Train + Validation combined",
        "preprocessing_fit":
            "Development only",
        "test_used_for_fit":
            False,
        "test_used_for_tuning":
            False,
        "resampling":
            "None",
        "classification_weighting":
            (
                "Balanced sample weights computed "
                "from Development labels only"
            ),
    },
    "data": {
        "development_rows":
            len(development),
        "development_batches":
            sorted(
                development_batches
            ),
        "test_rows":
            len(test_df),
        "test_batches":
            sorted(
                test_batches
            ),
        "batch_overlap_count":
            len(batch_overlap),
        "cross_split_observation_key_overlap":
            duplicate_cross_split_keys,
    },
    "features": {
        "selected_raw_feature_count":
            len(SELECTED_FEATURES),
        "transformed_feature_count":
            len(
                transformed_feature_names
            ),
        "selected_features":
            SELECTED_FEATURES,
    },
    "regression": {
        "selected_family":
            selected_reg_family,
        "selected_trial":
            selected_reg_trial,
        "MAE":
            reg_mae,
        "RMSE":
            reg_rmse,
        "R2":
            reg_r2,
        "negative_predictions":
            reg_negative,
        "predictions_above_development_target_max":
            reg_above_dev_max,
    },
    "classification": {
        "selected_family":
            selected_cls_family,
        "selected_trial":
            selected_cls_trial,
        **cls_metrics,
        "Warning_Recall":
            None
            if np.isnan(
                warning_recall
            )
            else float(
                warning_recall
            ),
        "Critical_Recall":
            None
            if np.isnan(
                critical_recall
            )
            else float(
                critical_recall
            ),
        "Warning_Reference_0_80_Pass":
            warning_reference_pass,
        "Critical_Reference_0_50_Pass":
            critical_reference_pass,
    },
    "methodological_status":
        "PASS",
    "performance_note":
        (
            "Final-test results are evidence only. "
            "No model tuning/reselection is permitted "
            "from Stage-31 results."
    ),
}

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        summary,
        handle,
        indent=2,
    )


# ============================================================
# 20. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 31: "
        "LOCKED FINAL TEST EVALUATION\n"
    )
    report.write(
        "=" * 78 + "\n\n"
    )

    report.write(
        "METHODOLOGY\n"
        "-----------\n"
    )
    report.write(
        "Train and Validation were combined into the "
        "full development dataset.\n"
    )
    report.write(
        "Median imputation, StandardScaler and "
        "OneHotEncoder were fitted on Development only.\n"
    )
    report.write(
        "Frozen Stage-19 and Stage-29 model "
        "configurations were refitted on Development only.\n"
    )
    report.write(
        "The locked Final Test was opened only after "
        "all fitting steps were complete.\n"
    )
    report.write(
        "No Test row was used for preprocessing fit, "
        "model fit, weighting, tuning or reselection.\n\n"
    )

    report.write(
        "DATA ISOLATION\n"
        "--------------\n"
    )
    report.write(
        f"Development rows: {len(development):,}\n"
    )
    report.write(
        f"Development batches: "
        f"{len(development_batches)}\n"
    )
    report.write(
        f"Final-Test rows: {len(test_df):,}\n"
    )
    report.write(
        f"Final-Test batches: {len(test_batches)}\n"
    )
    report.write(
        f"Batch overlap: {len(batch_overlap)}\n"
    )
    report.write(
        "Observation-key overlap: "
        f"{duplicate_cross_split_keys}\n\n"
    )

    report.write(
        "REGRESSION FINAL TEST\n"
        "---------------------\n"
    )
    report.write(
        f"Model: {selected_reg_family} "
        f"/ {selected_reg_trial}\n"
    )
    report.write(
        f"MAE: {reg_mae:.6f} h\n"
    )
    report.write(
        f"RMSE: {reg_rmse:.6f} h\n"
    )
    report.write(
        f"R2: {reg_r2:.6f}\n"
    )
    report.write(
        f"Negative predictions: "
        f"{reg_negative}\n"
    )
    report.write(
        "Predictions above Development target max: "
        f"{reg_above_dev_max}\n\n"
    )

    report.write(
        "CLASSIFICATION FINAL TEST\n"
        "-------------------------\n"
    )
    report.write(
        f"Model: {selected_cls_family} "
        f"/ {selected_cls_trial}\n"
    )
    report.write(
        f"Accuracy: "
        f"{cls_metrics['Accuracy']:.6f}\n"
    )
    report.write(
        "Balanced Accuracy: "
        f"{cls_metrics['Balanced_Accuracy']:.6f}\n"
    )
    report.write(
        "Macro Precision: "
        f"{cls_metrics['Macro_Precision']:.6f}\n"
    )
    report.write(
        "Macro Recall: "
        f"{cls_metrics['Macro_Recall']:.6f}\n"
    )
    report.write(
        f"Macro F1: "
        f"{cls_metrics['Macro_F1']:.6f}\n"
    )

    report.write(
        "Warning Recall: "
        + (
            "N/A\n"
            if np.isnan(
                warning_recall
            )
            else f"{warning_recall:.6f}\n"
        )
    )

    report.write(
        "Critical Recall: "
        + (
            "N/A\n"
            if np.isnan(
                critical_recall
            )
            else f"{critical_recall:.6f}\n"
        )
    )

    report.write(
        "Warning >= 0.80 reference: "
        f"{'PASS' if warning_reference_pass else 'BELOW REFERENCE'}\n"
    )
    report.write(
        "Critical >= 0.50 reference: "
        f"{'PASS' if critical_reference_pass else 'BELOW REFERENCE'}\n\n"
    )

    report.write(
        "CONFUSION MATRIX\n"
        "----------------\n"
    )
    report.write(
        cls_cm_df.to_string()
        + "\n\n"
    )

    report.write(
        "FINAL STATUS\n"
        "------------\n"
    )
    report.write(
        "STAGE 31 LOCKED FINAL TEST EVALUATION: COMPLETE\n"
    )
    report.write(
        "Methodological status: PASS\n"
    )
    report.write(
        "No tuning, model reselection or threshold "
        "adjustment is permitted from these test results.\n"
    )


# ============================================================
# 21. CONSOLE RESULTS
# ============================================================

print("\n" + "=" * 78)
print("FINAL TEST — REGRESSION")
print("=" * 78)

print(
    f"Model : "
    f"{selected_reg_family} / "
    f"{selected_reg_trial}"
)
print(
    f"MAE   : {reg_mae:.4f} h"
)
print(
    f"RMSE  : {reg_rmse:.4f} h"
)
print(
    f"R2    : {reg_r2:.4f}"
)
print(
    "Negative predictions:",
    reg_negative,
)
print(
    "Predictions above Development target max:",
    reg_above_dev_max,
)

print("\n" + "=" * 78)
print("FINAL TEST — CLASSIFICATION")
print("=" * 78)

print(
    f"Model              : "
    f"{selected_cls_family} / "
    f"{selected_cls_trial}"
)
print(
    f"Accuracy           : "
    f"{cls_metrics['Accuracy']:.4f}"
)
print(
    f"Balanced Accuracy  : "
    f"{cls_metrics['Balanced_Accuracy']:.4f}"
)
print(
    f"Macro Precision    : "
    f"{cls_metrics['Macro_Precision']:.4f}"
)
print(
    f"Macro Recall       : "
    f"{cls_metrics['Macro_Recall']:.4f}"
)
print(
    f"Macro-F1           : "
    f"{cls_metrics['Macro_F1']:.4f}"
)
print(
    "Warning Recall     : "
    + (
        "N/A"
        if np.isnan(
            warning_recall
        )
        else f"{warning_recall:.4f}"
    )
)
print(
    "Critical Recall    : "
    + (
        "N/A"
        if np.isnan(
            critical_recall
        )
        else f"{critical_recall:.4f}"
    )
)

print(
    "\nReference checks:"
)
print(
    "Warning Recall >= 0.80 : "
    + (
        "PASS"
        if warning_reference_pass
        else "BELOW REFERENCE"
    )
)
print(
    "Critical Recall >= 0.50: "
    + (
        "PASS"
        if critical_reference_pass
        else "BELOW REFERENCE"
    )
)

print(
    "\n--- FINAL TEST CONFUSION MATRIX ---"
)
print(
    cls_cm_df.to_string()
)

print(
    "\n--- PER-CLASS PERFORMANCE ---"
)
print(
    cls_per_class_df.to_string(
        index=False
    )
)

print("\n" + "=" * 78)
print(
    "STAGE 31 LOCKED FINAL TEST "
    "EVALUATION: COMPLETE"
)
print("=" * 78)

print(
    "Methodological status: PASS"
)
print(
    "IMPORTANT: Do NOT tune, reselect, resample, "
    "or adjust thresholds using these test results."
)

print("\nOutputs:")
for path in [
    REG_METRICS_PATH,
    REG_PREDICTIONS_PATH,
    REG_BATCH_PATH,
    REG_LAYER_PATH,
    CLS_METRICS_PATH,
    CLS_PREDICTIONS_PATH,
    CLS_CM_PATH,
    CLS_CLASS_REPORT_PATH,
    CLS_BATCH_PATH,
    CLS_LAYER_PATH,
    COMPARISON_PATH,
    SUMMARY_PATH,
    REPORT_PATH,
]:
    print(path)
