from pathlib import Path
import ast
import json
import math

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import joblib
from xgboost import XGBClassifier


# ============================================================
# COMPONENT 3 - STAGE 30:
# LEAKAGE-SAFE GROUP / BATCH CROSS-VALIDATION
# ============================================================
#
# This is the FINAL Stage-30 implementation.
#
# It validates BOTH development-selected models:
#
#   Regression:
#     Stage-19 selected Gradient Boosting GB_02
#
#   Classification:
#     Stage-29 selected XGBoost XGB_03
#
# Critical methodological rule:
#   Stage-10 preprocessing cannot simply be reused here because
#   Stage 30 creates NEW GroupKFold train/validation partitions.
#
# Therefore, in EACH fold:
#
#   Fold Train
#       ↓
#   fit median imputer + StandardScaler + OneHotEncoder
#       ↓
#   transform Fold Train
#       ↓
#   transform Fold Validation with SAME fold-fitted transformer
#
# No fold-validation batch contributes to preprocessing fit.
#
# FINAL TEST:
#   Component3_Test.csv existence is checked only.
#   Its contents are NEVER loaded here.
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

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = SPLIT_DIR / "Component3_Train.csv"
VALIDATION_PATH = SPLIT_DIR / "Component3_Validation.csv"
TEST_PATH = SPLIT_DIR / "Component3_Test.csv"

REG_FEATURE_LIST_PATH = (
    REPORT_DIR / "08_regression_selected_features.csv"
)
CLS_FEATURE_LIST_PATH = (
    REPORT_DIR / "08_classification_selected_features.csv"
)

REG_SELECTION_SUMMARY_PATH = (
    REPORT_DIR / "19_final_regression_selection_summary.json"
)

CLS_SELECTION_CONFIG_PATH = (
    REPORT_DIR / "29_selected_classification_configuration.csv"
)

REG_FINAL_MODEL_PATH = (
    REG_FINAL_DIR / "selected_regression_model.joblib"
)

STAGE28_SOURCE_PATH = (
    ROOT
    / "src"
    / "classification"
    / "28_hyperparameter_tuning.py"
)


# ============================================================
# 2. OUTPUTS
# ============================================================

REG_FOLDS_PATH = (
    REPORT_DIR / "30_regression_group_cv_folds.csv"
)
REG_OOF_PATH = (
    REPORT_DIR / "30_regression_group_cv_oof_predictions.csv"
)
REG_BATCH_PATH = (
    REPORT_DIR / "30_regression_group_cv_by_batch.csv"
)
REG_SUMMARY_PATH = (
    REPORT_DIR / "30_regression_group_cv_summary.csv"
)

CLS_FOLDS_PATH = (
    REPORT_DIR / "30_classification_group_cv_folds.csv"
)
CLS_OOF_PATH = (
    REPORT_DIR / "30_classification_group_cv_oof_predictions.csv"
)
CLS_CM_PATH = (
    REPORT_DIR / "30_classification_group_cv_confusion_matrix.csv"
)
CLS_BATCH_PATH = (
    REPORT_DIR / "30_classification_group_cv_by_batch.csv"
)
CLS_SUMMARY_PATH = (
    REPORT_DIR / "30_classification_group_cv_summary.csv"
)

COMBINED_SUMMARY_PATH = (
    REPORT_DIR / "30_group_cross_validation_summary.json"
)
REPORT_PATH = (
    REPORT_DIR / "30_group_cross_validation_report.txt"
)


# ============================================================
# 3. CONSTANTS
# ============================================================

N_SPLITS = 5

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

# Stage-29 development selection guardrails.
# These are reported as reference robustness checks.
# Stage 30 itself is a validation stage; results are not altered
# to force these thresholds to pass.
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


def rmse(y_true, y_pred):
    return float(
        math.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def safe_class_recall(
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


def supported_macro_f1(
    y_true,
    y_pred,
):
    supported = sorted(
        np.unique(
            np.asarray(y_true)
        ).tolist()
    )

    return float(
        f1_score(
            y_true,
            y_pred,
            labels=supported,
            average="macro",
            zero_division=0,
        )
    )


def supported_balanced_accuracy(
    y_true,
    y_pred,
):
    supported = sorted(
        np.unique(
            np.asarray(y_true)
        ).tolist()
    )

    return float(
        recall_score(
            y_true,
            y_pred,
            labels=supported,
            average="macro",
            zero_division=0,
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


def recover_selected_xgb_configuration(path):
    frame = pd.read_csv(path)

    if frame.empty:
        raise RuntimeError(
            "Stage-29 selected classification configuration is empty."
        )

    selected = frame.iloc[0]

    if "Model_Family" in frame.columns:
        family = str(
            selected["Model_Family"]
        )
    else:
        family = "Unknown"

    if "Trial" in frame.columns:
        trial = str(
            selected["Trial"]
        )
    else:
        trial = "Unknown"

    possible_parameter_columns = [
        "Parameters_JSON",
        "Best_Parameters",
        "Parameters",
        "Params",
    ]

    params = {}

    for column in possible_parameter_columns:
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
            "Could not recover exact selected XGBoost parameters "
            f"from Stage 29. Missing: {missing}"
        )

    return family, trial, params


def classification_batch_metrics(
    frame,
):
    rows = []

    for batch_id, batch in frame.groupby(
        "Batch_ID",
        sort=True,
    ):
        y_true = batch["y_true_int"].to_numpy()
        y_pred = batch["y_pred_int"].to_numpy()

        rows.append({
            "Batch_ID": batch_id,
            "Rows": len(batch),
            "Normal_Support": int(
                np.sum(y_true == 0)
            ),
            "Warning_Support": int(
                np.sum(y_true == 1)
            ),
            "Critical_Support": int(
                np.sum(y_true == 2)
            ),
            "Accuracy": float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "Supported_Class_Macro_F1":
                supported_macro_f1(
                    y_true,
                    y_pred,
            ),
            "All_Three_Class_Macro_F1":
                float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=[0, 1, 2],
                        average="macro",
                        zero_division=0,
                    )
            ),
            "Supported_Class_Balanced_Accuracy":
                supported_balanced_accuracy(
                    y_true,
                    y_pred,
            ),
            "Warning_Recall":
                safe_class_recall(
                    y_true,
                    y_pred,
                    1,
            ),
            "Critical_Recall":
                safe_class_recall(
                    y_true,
                    y_pred,
                    2,
            ),
        })

    return pd.DataFrame(rows)


def regression_batch_metrics(
    frame,
):
    rows = []

    for batch_id, batch in frame.groupby(
        "Batch_ID",
        sort=True,
    ):
        y_true = batch[
            "Remaining_Drying_Time"
        ].to_numpy()

        y_pred = batch[
            "Predicted_Remaining_Drying_Time"
        ].to_numpy()

        rows.append({
            "Batch_ID": batch_id,
            "Rows": len(batch),
            "MAE": float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),
            "RMSE": rmse(
                y_true,
                y_pred,
            ),
            "R2": float(
                r2_score(
                    y_true,
                    y_pred,
                )
            ),
            "Negative_Predictions": int(
                np.sum(
                    y_pred < 0
                )
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. REQUIRED INPUTS / FINAL TEST PROTECTION
# ============================================================

required_files = [
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
    REG_FEATURE_LIST_PATH,
    CLS_FEATURE_LIST_PATH,
    REG_SELECTION_SUMMARY_PATH,
    CLS_SELECTION_CONFIG_PATH,
    REG_FINAL_MODEL_PATH,
    STAGE28_SOURCE_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

# IMPORTANT:
# Test existence is verified but its CONTENT IS NEVER READ.
final_test_content_read = False


# ============================================================
# 6. START
# ============================================================

print("=" * 78)
print(
    "COMPONENT 3 - STAGE 30: "
    "LEAKAGE-SAFE GROUP / BATCH CROSS-VALIDATION"
)
print("=" * 78)

print("\n--- TEST PROTECTION ---")
print(
    "Final Test dataset existence verified, "
    "but contents were NOT loaded."
)
print(
    "Component3_Test.csv content read:",
    final_test_content_read,
)
print(
    "Final Test remains frozen until Stage 31."
)


# ============================================================
# 7. LOAD STAGE-08 FEATURE DEFINITIONS
# ============================================================

reg_feature_df = pd.read_csv(
    REG_FEATURE_LIST_PATH
)

cls_feature_df = pd.read_csv(
    CLS_FEATURE_LIST_PATH
)

REGRESSION_FEATURES = (
    reg_feature_df["Feature"]
    .astype(str)
    .tolist()
)

CLASSIFICATION_FEATURES = (
    cls_feature_df["Feature"]
    .astype(str)
    .tolist()
)

if (
    REGRESSION_FEATURES
    != CLASSIFICATION_FEATURES
):
    raise RuntimeError(
        "Stage-08 regression and classification feature lists "
        "do not match, but Stage 10 expected one common feature set."
    )

SELECTED_FEATURES = (
    REGRESSION_FEATURES
)

if len(SELECTED_FEATURES) != 17:
    raise RuntimeError(
        "Expected exactly 17 Stage-08 selected features, "
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
        "Expected 15 numeric features, "
        f"found {len(NUMERIC_FEATURES)}."
    )

for categorical_feature in CATEGORICAL_FEATURES:
    if categorical_feature not in SELECTED_FEATURES:
        raise RuntimeError(
            f"Required categorical feature is not selected: "
            f"{categorical_feature}"
        )

for forbidden in [
    "Batch_ID",
    "Timestamp",
    "Fire_Risk_Assessment_Timestamp",
    "Actual_Total_Drying_Time",
    "Remaining_Drying_Time",
    "Fire_Risk_Label",
]:
    if forbidden in SELECTED_FEATURES:
        raise RuntimeError(
            "Leakage/grouping field unexpectedly appears in "
            f"selected model features: {forbidden}"
        )

print("\n--- STAGE-08 / STAGE-10 FEATURE SPACE ---")
print(
    f"Selected raw features : {len(SELECTED_FEATURES)}"
)
print(
    f"Numeric features      : {len(NUMERIC_FEATURES)}"
)
print(
    f"Categorical features  : {len(CATEGORICAL_FEATURES)}"
)
print(
    "Categorical           : "
    + ", ".join(CATEGORICAL_FEATURES)
)


# ============================================================
# 8. LOAD DEVELOPMENT DATA ONLY
# ============================================================

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

required_columns = (
    SELECTED_FEATURES
    + [
        "Batch_ID",
        "Remaining_Drying_Time",
        "Fire_Risk_Label",
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in development.columns
]

if missing_columns:
    raise RuntimeError(
        "Development data missing required columns: "
        f"{missing_columns}"
    )

for feature in NUMERIC_FEATURES:
    development[feature] = pd.to_numeric(
        development[feature],
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

unexpected_labels = sorted(
    set(
        development[
            "Fire_Risk_Label"
        ].unique()
    )
    - set(CLASS_ORDER_TEXT)
)

if unexpected_labels:
    raise RuntimeError(
        "Unexpected Fire_Risk_Label values: "
        f"{unexpected_labels}"
    )

development[
    "Fire_Risk_Int"
] = development[
    "Fire_Risk_Label"
].map(
    CLASS_TO_INT
)

groups = (
    development[
        "Batch_ID"
    ]
    .astype(str)
)

unique_batches = sorted(
    groups.unique().tolist()
)

if len(unique_batches) < N_SPLITS:
    raise RuntimeError(
        "Not enough development batches for "
        f"{N_SPLITS}-fold GroupKFold."
    )

print("\n--- DEVELOPMENT DATA ---")
print(
    f"Train rows       : {len(train_df):,}"
)
print(
    f"Validation rows  : {len(validation_df):,}"
)
print(
    f"Combined rows    : {len(development):,}"
)
print(
    f"Development batches: {len(unique_batches)}"
)
print(
    f"GroupKFold splits   : {N_SPLITS}"
)

print("\nDevelopment batches:")
print(
    ", ".join(unique_batches)
)

print("\n--- DEVELOPMENT CLASS SUPPORT ---")
for label in CLASS_ORDER_TEXT:
    print(
        f"{label:<8}: "
        f"{int((development['Fire_Risk_Label'] == label).sum())}"
    )


# ============================================================
# 9. RECOVER EXACT SELECTED REGRESSION MODEL
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

selected_reg_artifact = joblib.load(
    REG_FINAL_MODEL_PATH
)

if not isinstance(
    selected_reg_artifact,
    GradientBoostingRegressor,
):
    raise RuntimeError(
        "Stage-19 selected regression artifact is not "
        "GradientBoostingRegressor as expected."
    )

regression_params = (
    selected_reg_artifact.get_params(
        deep=False
    )
)

print("\n--- SELECTED REGRESSION MODEL ---")
print(
    f"Family : {selected_reg_family}"
)
print(
    f"Trial  : {selected_reg_trial}"
)
print(
    "Class  : "
    f"{selected_reg_artifact.__class__.__name__}"
)

print("\nRecovered regression parameters:")
for key in sorted(
    regression_params.keys()
):
    print(
        f"  {key:<20}: "
        f"{regression_params[key]}"
    )


# ============================================================
# 10. RECOVER EXACT SELECTED CLASSIFIER CONFIG
# ============================================================

(
    selected_cls_family,
    selected_cls_trial,
    xgb_params,
) = recover_selected_xgb_configuration(
    CLS_SELECTION_CONFIG_PATH
)

print("\n--- SELECTED CLASSIFIER ---")
print(
    f"Family : {selected_cls_family}"
)
print(
    f"Trial  : {selected_cls_trial}"
)

print("\nRecovered XGBoost parameters:")
for key in sorted(
    xgb_params.keys()
):
    print(
        f"  {key:<20}: "
        f"{xgb_params[key]}"
    )


# ============================================================
# 11. VERIFY STAGE-28 CLASS-IMBALANCE RECIPE
# ============================================================

stage28_source = (
    STAGE28_SOURCE_PATH
    .read_text(
        encoding="utf-8"
    )
)

stage28_lower = (
    stage28_source.lower()
)

# Stage 28 does NOT call sklearn.compute_sample_weight directly.
# Instead, it loads the Stage-21 row-level Sample_Weight values,
# verifies those against the saved Balanced_Class_Weight map,
# and passes train_sample_weights into Gradient Boosting / XGBoost.
#
# For Stage 30's NEW GroupKFold partitions, reusing the old
# Stage-21 Train weights would be incorrect because each fold has
# a different training class distribution. Therefore Stage 30
# reconstructs the same "balanced" weighting rule independently
# from EACH fold-training target using compute_sample_weight().
stage28_compact = "".join(
    stage28_lower.split()
)

uses_stage21_sample_weights = (
    "stage21_sample_weights_df"
    in stage28_lower
    and "train_sample_weights"
    in stage28_lower
)

uses_balanced_class_weight_map = (
    "balanced_class_weight"
    in stage28_lower
)

uses_xgb_sample_weights = (
    "sample_weight=train_sample_weights"
    in stage28_compact
)

documents_no_smote = (
    "no smote or resampling"
    in stage28_lower
)

if not (
    uses_stage21_sample_weights
    and uses_balanced_class_weight_map
    and uses_xgb_sample_weights
    and documents_no_smote
):
    raise RuntimeError(
        "Stage-28 source does not clearly verify the expected "
        "Stage-21 balanced sample-weight recipe for XGBoost. "
        "Stage 30 stopped before cross-validation."
    )

print(
    "\n--- CLASSIFICATION IMBALANCE RECIPE ---"
)
print(
    "Stage-28 loads Stage-21 sample weights       :",
    uses_stage21_sample_weights,
)
print(
    "Stage-28 verifies Balanced_Class_Weight map  :",
    uses_balanced_class_weight_map,
)
print(
    "Stage-28 passes weights to XGBoost.fit       :",
    uses_xgb_sample_weights,
)
print(
    "Stage-28 documents no SMOTE/resampling       :",
    documents_no_smote,
)
print(
    "Stage 30 will reproduce the same BALANCED "
    "weighting principle inside EACH fold using only "
    "that fold's training labels."
)


# ============================================================
# 12. GROUPKFOLD
# ============================================================

X_raw = development[
    SELECTED_FEATURES
].copy()

y_reg = development[
    "Remaining_Drying_Time"
].to_numpy(
    dtype=float
)

y_cls = development[
    "Fire_Risk_Int"
].to_numpy(
    dtype=int
)

group_values = groups.to_numpy()

group_kfold = GroupKFold(
    n_splits=N_SPLITS
)

reg_fold_rows = []
cls_fold_rows = []

reg_oof_records = []
cls_oof_records = []

fold_overlap_count = 0
all_fold_preprocessing_train_only = True

print("\n" + "=" * 78)
print("RUNNING LEAKAGE-SAFE GROUPKFOLD")
print("=" * 78)

for fold_number, (
    train_index,
    validation_index,
) in enumerate(
    group_kfold.split(
        X_raw,
        y_reg,
        groups=group_values,
    ),
    start=1,
):
    X_fold_train_raw = (
        X_raw.iloc[
            train_index
        ].copy()
    )

    X_fold_validation_raw = (
        X_raw.iloc[
            validation_index
        ].copy()
    )

    y_reg_fold_train = (
        y_reg[
            train_index
        ]
    )

    y_reg_fold_validation = (
        y_reg[
            validation_index
        ]
    )

    y_cls_fold_train = (
        y_cls[
            train_index
        ]
    )

    y_cls_fold_validation = (
        y_cls[
            validation_index
        ]
    )

    fold_train_groups = set(
        group_values[
            train_index
        ]
    )

    fold_validation_groups = set(
        group_values[
            validation_index
        ]
    )

    overlap = (
        fold_train_groups
        & fold_validation_groups
    )

    if overlap:
        fold_overlap_count += 1
        raise RuntimeError(
            f"Fold {fold_number} has Batch_ID overlap: "
            f"{sorted(overlap)}"
        )

    # --------------------------------------------------------
    # FOLD-LOCAL PREPROCESSING
    # --------------------------------------------------------

    fold_preprocessor = (
        make_preprocessor(
            NUMERIC_FEATURES,
            CATEGORICAL_FEATURES,
        )
    )

    # Fit ONLY on this fold's training portion.
    fold_preprocessor.fit(
        X_fold_train_raw
    )

    X_fold_train = (
        fold_preprocessor.transform(
            X_fold_train_raw
        )
    )

    X_fold_validation = (
        fold_preprocessor.transform(
            X_fold_validation_raw
        )
    )

    transformed_feature_names = (
        fold_preprocessor
        .get_feature_names_out()
        .tolist()
    )

    fold_preprocessing_train_only = True

    all_fold_preprocessing_train_only = (
        all_fold_preprocessing_train_only
        and fold_preprocessing_train_only
    )

    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------

    regression_model = clone(
        selected_reg_artifact
    )

    regression_model.fit(
        X_fold_train,
        y_reg_fold_train,
    )

    reg_pred = regression_model.predict(
        X_fold_validation
    )

    fold_reg_mae = float(
        mean_absolute_error(
            y_reg_fold_validation,
            reg_pred,
        )
    )

    fold_reg_rmse = rmse(
        y_reg_fold_validation,
        reg_pred,
    )

    fold_reg_r2 = float(
        r2_score(
            y_reg_fold_validation,
            reg_pred,
        )
    )

    fold_reg_negative = int(
        np.sum(
            reg_pred < 0
        )
    )

    fold_target_max = float(
        np.max(
            y_reg_fold_validation
        )
    )

    fold_reg_above_dev_max = int(
        np.sum(
            reg_pred
            > float(
                np.max(y_reg)
            )
        )
    )

    reg_fold_rows.append({
        "Fold": fold_number,
        "Train_Rows": len(train_index),
        "Validation_Rows": len(validation_index),
        "Train_Batches":
            " | ".join(
                sorted(
                    fold_train_groups
                )
        ),
        "Validation_Batches":
            " | ".join(
                sorted(
                    fold_validation_groups
                )
        ),
        "Batch_Overlap_Count": len(overlap),
        "Raw_Feature_Count":
            len(SELECTED_FEATURES),
        "Transformed_Feature_Count":
            len(
                transformed_feature_names
        ),
        "Preprocessor_Fit_On_Fold_Train_Only":
            fold_preprocessing_train_only,
        "MAE": fold_reg_mae,
        "RMSE": fold_reg_rmse,
        "R2": fold_reg_r2,
        "Negative_Predictions":
            fold_reg_negative,
        "Predictions_Above_Development_Target_Max":
            fold_reg_above_dev_max,
        "Validation_Target_Max":
            fold_target_max,
    })

    for local_position, row_index in enumerate(
        validation_index
    ):
        reg_oof_records.append({
            "Original_Row_Index":
                int(row_index),
            "Fold":
                fold_number,
            "Batch_ID":
                str(
                    group_values[
                        row_index
                    ]
                ),
            "Remaining_Drying_Time":
                float(
                    y_reg[
                        row_index
                    ]
                ),
            "Predicted_Remaining_Drying_Time":
                float(
                    reg_pred[
                        local_position
                    ]
                ),
            "Absolute_Error":
                float(
                    abs(
                        y_reg[
                            row_index
                        ]
                        - reg_pred[
                            local_position
                        ]
                    )
                ),
        })

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    classification_model = XGBClassifier(
        **xgb_params
    )

    fold_sample_weight = (
        compute_sample_weight(
            class_weight="balanced",
            y=y_cls_fold_train,
        )
    )

    classification_model.fit(
        X_fold_train,
        y_cls_fold_train,
        sample_weight=fold_sample_weight,
    )

    cls_pred = (
        classification_model.predict(
            X_fold_validation
        )
        .astype(int)
    )

    cls_accuracy = float(
        accuracy_score(
            y_cls_fold_validation,
            cls_pred,
        )
    )

    cls_supported_macro_f1 = (
        supported_macro_f1(
            y_cls_fold_validation,
            cls_pred,
        )
    )

    cls_all_three_macro_f1 = float(
        f1_score(
            y_cls_fold_validation,
            cls_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )

    cls_balanced = (
        supported_balanced_accuracy(
            y_cls_fold_validation,
            cls_pred,
        )
    )

    warning_recall = (
        safe_class_recall(
            y_cls_fold_validation,
            cls_pred,
            1,
        )
    )

    critical_recall = (
        safe_class_recall(
            y_cls_fold_validation,
            cls_pred,
            2,
        )
    )

    normal_support = int(
        np.sum(
            y_cls_fold_validation
            == 0
        )
    )

    warning_support = int(
        np.sum(
            y_cls_fold_validation
            == 1
        )
    )

    critical_support = int(
        np.sum(
            y_cls_fold_validation
            == 2
        )
    )

    cls_fold_rows.append({
        "Fold": fold_number,
        "Train_Rows": len(train_index),
        "Validation_Rows": len(validation_index),
        "Train_Batches":
            " | ".join(
                sorted(
                    fold_train_groups
                )
        ),
        "Validation_Batches":
            " | ".join(
                sorted(
                    fold_validation_groups
                )
        ),
        "Batch_Overlap_Count":
            len(overlap),
        "Raw_Feature_Count":
            len(SELECTED_FEATURES),
        "Transformed_Feature_Count":
            len(
                transformed_feature_names
        ),
        "Preprocessor_Fit_On_Fold_Train_Only":
            fold_preprocessing_train_only,
        "Normal_Support":
            normal_support,
        "Warning_Support":
            warning_support,
        "Critical_Support":
            critical_support,
        "Accuracy":
            cls_accuracy,
        "Supported_Class_Macro_F1":
            cls_supported_macro_f1,
        "All_Three_Class_Macro_F1":
            cls_all_three_macro_f1,
        "Supported_Class_Balanced_Accuracy":
            cls_balanced,
        "Warning_Recall":
            warning_recall,
        "Critical_Recall":
            critical_recall,
    })

    for local_position, row_index in enumerate(
        validation_index
    ):
        true_int = int(
            y_cls[
                row_index
            ]
        )

        pred_int = int(
            cls_pred[
                local_position
            ]
        )

        cls_oof_records.append({
            "Original_Row_Index":
                int(row_index),
            "Fold":
                fold_number,
            "Batch_ID":
                str(
                    group_values[
                        row_index
                    ]
                ),
            "y_true_int":
                true_int,
            "y_pred_int":
                pred_int,
            "Actual_Fire_Risk_Label":
                INT_TO_CLASS[
                    true_int
                ],
            "Predicted_Fire_Risk_Label":
                INT_TO_CLASS[
                    pred_int
                ],
        })

    print(
        f"\nFold {fold_number}/{N_SPLITS}"
    )
    print(
        "Validation batches : "
        + ", ".join(
            sorted(
                fold_validation_groups
            )
        )
    )
    print(
        "Fold-local transformed features : "
        f"{len(transformed_feature_names)}"
    )

    print("\n  Regression")
    print(
        f"    MAE  : "
        f"{fold_reg_mae:.4f} h"
    )
    print(
        f"    RMSE : "
        f"{fold_reg_rmse:.4f} h"
    )
    print(
        f"    R2   : "
        f"{fold_reg_r2:.4f}"
    )

    print("\n  Classification")
    print(
        "    Support: "
        f"Normal={normal_support}, "
        f"Warning={warning_support}, "
        f"Critical={critical_support}"
    )
    print(
        f"    Macro-F1 (all 3) : "
        f"{cls_all_three_macro_f1:.4f}"
    )
    print(
        f"    Balanced Accuracy: "
        f"{cls_balanced:.4f}"
    )
    print(
        "    Warning Recall   : "
        + (
            "N/A"
            if np.isnan(
                warning_recall
            )
            else f"{warning_recall:.4f}"
        )
    )
    print(
        "    Critical Recall  : "
        + (
            "N/A"
            if np.isnan(
                critical_recall
            )
            else f"{critical_recall:.4f}"
        )
    )


# ============================================================
# 13. BUILD OOF DATAFRAMES
# ============================================================

reg_folds_df = pd.DataFrame(
    reg_fold_rows
)

cls_folds_df = pd.DataFrame(
    cls_fold_rows
)

reg_oof_df = pd.DataFrame(
    reg_oof_records
).sort_values(
    "Original_Row_Index"
).reset_index(
    drop=True
)

cls_oof_df = pd.DataFrame(
    cls_oof_records
).sort_values(
    "Original_Row_Index"
).reset_index(
    drop=True
)

if len(reg_oof_df) != len(development):
    raise RuntimeError(
        "Regression OOF row count does not equal "
        "development row count."
    )

if len(cls_oof_df) != len(development):
    raise RuntimeError(
        "Classification OOF row count does not equal "
        "development row count."
    )

if (
    reg_oof_df[
        "Original_Row_Index"
    ].nunique()
    != len(development)
):
    raise RuntimeError(
        "Regression OOF rows are not one-to-one."
    )

if (
    cls_oof_df[
        "Original_Row_Index"
    ].nunique()
    != len(development)
):
    raise RuntimeError(
        "Classification OOF rows are not one-to-one."
    )


# ============================================================
# 14. POOLED REGRESSION PERFORMANCE
# ============================================================

reg_y_true = reg_oof_df[
    "Remaining_Drying_Time"
].to_numpy()

reg_y_pred = reg_oof_df[
    "Predicted_Remaining_Drying_Time"
].to_numpy()

pooled_reg_mae = float(
    mean_absolute_error(
        reg_y_true,
        reg_y_pred,
    )
)

pooled_reg_rmse = rmse(
    reg_y_true,
    reg_y_pred,
)

pooled_reg_r2 = float(
    r2_score(
        reg_y_true,
        reg_y_pred,
    )
)

pooled_reg_negative = int(
    np.sum(
        reg_y_pred < 0
    )
)

development_target_max = float(
    np.max(
        reg_y_true
    )
)

pooled_reg_above_max = int(
    np.sum(
        reg_y_pred
        > development_target_max
    )
)

reg_batch_df = (
    regression_batch_metrics(
        reg_oof_df
    )
)

reg_summary_df = pd.DataFrame([
    {
        "Selected_Model_Family":
            selected_reg_family,
        "Selected_Trial":
            selected_reg_trial,
        "Development_Rows":
            len(development),
        "Development_Batches":
            len(unique_batches),
        "GroupKFold_Splits":
            N_SPLITS,
        "Pooled_OOF_MAE":
            pooled_reg_mae,
        "Pooled_OOF_RMSE":
            pooled_reg_rmse,
        "Pooled_OOF_R2":
            pooled_reg_r2,
        "Negative_Predictions":
            pooled_reg_negative,
        "Predictions_Above_Development_Target_Max":
            pooled_reg_above_max,
        "Development_Target_Max":
            development_target_max,
        "Batch_Overlap_Folds":
            fold_overlap_count,
        "All_Fold_Preprocessors_Fit_Train_Only":
            all_fold_preprocessing_train_only,
        "Final_Test_Content_Read":
            final_test_content_read,
    }
])


# ============================================================
# 15. POOLED CLASSIFICATION PERFORMANCE
# ============================================================

cls_y_true = cls_oof_df[
    "y_true_int"
].to_numpy()

cls_y_pred = cls_oof_df[
    "y_pred_int"
].to_numpy()

pooled_cls_accuracy = float(
    accuracy_score(
        cls_y_true,
        cls_y_pred,
    )
)

pooled_cls_balanced = float(
    balanced_accuracy_score(
        cls_y_true,
        cls_y_pred,
    )
)

pooled_cls_macro_f1 = float(
    f1_score(
        cls_y_true,
        cls_y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
)

pooled_cls_macro_precision = float(
    precision_score(
        cls_y_true,
        cls_y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
)

pooled_cls_macro_recall = float(
    recall_score(
        cls_y_true,
        cls_y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
)

pooled_warning_recall = (
    safe_class_recall(
        cls_y_true,
        cls_y_pred,
        1,
    )
)

pooled_critical_recall = (
    safe_class_recall(
        cls_y_true,
        cls_y_pred,
        2,
    )
)

warning_reference_pass = bool(
    pooled_warning_recall
    >= MIN_WARNING_RECALL_REFERENCE
)

critical_reference_pass = bool(
    pooled_critical_recall
    >= MIN_CRITICAL_RECALL_REFERENCE
)

cm = confusion_matrix(
    cls_y_true,
    cls_y_pred,
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

cls_batch_df = (
    classification_batch_metrics(
        cls_oof_df
    )
)

folds_with_warning = int(
    (
        cls_folds_df[
            "Warning_Support"
        ]
        > 0
    ).sum()
)

folds_with_critical = int(
    (
        cls_folds_df[
            "Critical_Support"
        ]
        > 0
    ).sum()
)

cls_summary_df = pd.DataFrame([
    {
        "Selected_Model_Family":
            selected_cls_family,
        "Selected_Trial":
            selected_cls_trial,
        "Development_Rows":
            len(development),
        "Development_Batches":
            len(unique_batches),
        "GroupKFold_Splits":
            N_SPLITS,
        "Pooled_OOF_Accuracy":
            pooled_cls_accuracy,
        "Pooled_OOF_Balanced_Accuracy":
            pooled_cls_balanced,
        "Pooled_OOF_Macro_F1":
            pooled_cls_macro_f1,
        "Pooled_OOF_Macro_Precision":
            pooled_cls_macro_precision,
        "Pooled_OOF_Macro_Recall":
            pooled_cls_macro_recall,
        "Pooled_OOF_Warning_Recall":
            pooled_warning_recall,
        "Pooled_OOF_Critical_Recall":
            pooled_critical_recall,
        "Warning_Reference_0_80_Pass":
            warning_reference_pass,
        "Critical_Reference_0_50_Pass":
            critical_reference_pass,
        "Folds_With_Warning":
            folds_with_warning,
        "Folds_With_Critical":
            folds_with_critical,
        "Batch_Overlap_Folds":
            fold_overlap_count,
        "All_Fold_Preprocessors_Fit_Train_Only":
            all_fold_preprocessing_train_only,
        "Final_Test_Content_Read":
            final_test_content_read,
    }
])


# ============================================================
# 16. SAVE OUTPUTS
# ============================================================

reg_folds_df.to_csv(
    REG_FOLDS_PATH,
    index=False,
)

reg_oof_df.to_csv(
    REG_OOF_PATH,
    index=False,
)

reg_batch_df.to_csv(
    REG_BATCH_PATH,
    index=False,
)

reg_summary_df.to_csv(
    REG_SUMMARY_PATH,
    index=False,
)

cls_folds_df.to_csv(
    CLS_FOLDS_PATH,
    index=False,
)

cls_oof_df.to_csv(
    CLS_OOF_PATH,
    index=False,
)

cls_cm_df.to_csv(
    CLS_CM_PATH,
    index=True,
)

cls_batch_df.to_csv(
    CLS_BATCH_PATH,
    index=False,
)

cls_summary_df.to_csv(
    CLS_SUMMARY_PATH,
    index=False,
)

combined_summary = {
    "stage": 30,
    "method": (
        "5-fold GroupKFold by Batch_ID "
        "with fold-local Stage-10 preprocessing"
    ),
    "development_rows":
        len(development),
    "development_batches":
        unique_batches,
    "selected_features":
        SELECTED_FEATURES,
    "numeric_features":
        NUMERIC_FEATURES,
    "categorical_features":
        CATEGORICAL_FEATURES,
    "preprocessing": {
        "numeric": (
            "Median imputation + StandardScaler "
            "fitted inside each fold"
        ),
        "categorical": (
            "OneHotEncoder(handle_unknown='ignore') "
            "fitted inside each fold"
        ),
        "fold_local_fit":
            bool(
                all_fold_preprocessing_train_only
        ),
    },
    "regression": {
        "selected_family":
            selected_reg_family,
        "selected_trial":
            selected_reg_trial,
        "pooled_oof_mae":
            pooled_reg_mae,
        "pooled_oof_rmse":
            pooled_reg_rmse,
        "pooled_oof_r2":
            pooled_reg_r2,
        "negative_predictions":
            pooled_reg_negative,
        "predictions_above_development_target_max":
            pooled_reg_above_max,
    },
    "classification": {
        "selected_family":
            selected_cls_family,
        "selected_trial":
            selected_cls_trial,
        "training_imbalance_recipe":
            (
                "balanced sample weights computed "
                "inside each fold from fold-training labels"
            ),
        "pooled_oof_accuracy":
            pooled_cls_accuracy,
        "pooled_oof_balanced_accuracy":
            pooled_cls_balanced,
        "pooled_oof_macro_f1":
            pooled_cls_macro_f1,
        "pooled_oof_warning_recall":
            pooled_warning_recall,
        "pooled_oof_critical_recall":
            pooled_critical_recall,
        "warning_reference_0_80_pass":
            warning_reference_pass,
        "critical_reference_0_50_pass":
            critical_reference_pass,
        "folds_with_warning":
            folds_with_warning,
        "folds_with_critical":
            folds_with_critical,
    },
    "methodological_checks": {
        "batch_overlap_folds":
            fold_overlap_count,
        "all_fold_preprocessors_fit_train_only":
            bool(
                all_fold_preprocessing_train_only
            ),
        "final_test_content_read":
            final_test_content_read,
    },
}

with open(
    COMBINED_SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        combined_summary,
        handle,
        indent=2,
    )


# ============================================================
# 17. CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("POOLED OUT-OF-FOLD REGRESSION PERFORMANCE")
print("=" * 78)

print(
    f"MAE                  : "
    f"{pooled_reg_mae:.4f} h"
)
print(
    f"RMSE                 : "
    f"{pooled_reg_rmse:.4f} h"
)
print(
    f"R2                   : "
    f"{pooled_reg_r2:.4f}"
)
print(
    f"Negative predictions : "
    f"{pooled_reg_negative}"
)
print(
    "Above development target max: "
    f"{pooled_reg_above_max}"
)

print("\n" + "=" * 78)
print("POOLED OUT-OF-FOLD CLASSIFICATION PERFORMANCE")
print("=" * 78)

print(
    f"Accuracy             : "
    f"{pooled_cls_accuracy:.4f}"
)
print(
    f"Balanced Accuracy    : "
    f"{pooled_cls_balanced:.4f}"
)
print(
    f"Macro-F1             : "
    f"{pooled_cls_macro_f1:.4f}"
)
print(
    f"Warning Recall       : "
    f"{pooled_warning_recall:.4f}"
)
print(
    f"Critical Recall      : "
    f"{pooled_critical_recall:.4f}"
)

print(
    "\nReference to Stage-29 selection guardrails:"
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

print("\n--- POOLED CONFUSION MATRIX ---")
print(
    cls_cm_df.to_string()
)

print("\n--- MINORITY-CLASS FOLD COVERAGE ---")
print(
    f"Folds containing Warning  : "
    f"{folds_with_warning}/{N_SPLITS}"
)
print(
    f"Folds containing Critical : "
    f"{folds_with_critical}/{N_SPLITS}"
)

if folds_with_critical < N_SPLITS:
    print(
        "\nNOTE:"
    )
    print(
        "Some validation folds contain no Critical observations. "
        "Critical Recall for those folds is correctly recorded as N/A."
    )


# ============================================================
# 18. METHODOLOGICAL STATUS
# ============================================================

methodological_pass = (
    fold_overlap_count == 0
    and all_fold_preprocessing_train_only
    and not final_test_content_read
    and len(reg_oof_df)
    == len(development)
    and len(cls_oof_df)
    == len(development)
)

if not methodological_pass:
    raise RuntimeError(
        "Stage 30 methodological validation failed."
    )

print("\n--- TEST PROTECTION ---")
print(
    "Component3_Test.csv content read:",
    final_test_content_read,
)
print(
    "Final Test remains frozen until Stage 31."
)

print("\n" + "=" * 78)
print(
    "STAGE 30 GROUP / BATCH "
    "CROSS-VALIDATION: PASS"
)
print("=" * 78)
print(
    "PASS here means the validation procedure is "
    "group-safe and leakage-safe."
)
print(
    "Model robustness must still be interpreted "
    "from the reported metrics."
)


# ============================================================
# 19. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 30: "
        "LEAKAGE-SAFE GROUP / BATCH CROSS-VALIDATION\n"
    )
    report.write(
        "=" * 78 + "\n\n"
    )

    report.write(
        "METHOD\n"
        "------\n"
    )
    report.write(
        "5-fold GroupKFold using Batch_ID across the combined "
        "development Train + Validation batches.\n"
    )
    report.write(
        "For every fold, median imputation, StandardScaler and "
        "OneHotEncoder were freshly fitted on fold-training rows only. "
        "Fold-validation rows never contributed to preprocessing fit.\n"
    )
    report.write(
        "Final Test contents were NOT loaded.\n\n"
    )

    report.write(
        "REGRESSION\n"
        "----------\n"
    )
    report.write(
        f"Selected family: {selected_reg_family}\n"
    )
    report.write(
        f"Selected trial: {selected_reg_trial}\n"
    )
    report.write(
        f"Pooled OOF MAE: {pooled_reg_mae:.6f} h\n"
    )
    report.write(
        f"Pooled OOF RMSE: {pooled_reg_rmse:.6f} h\n"
    )
    report.write(
        f"Pooled OOF R2: {pooled_reg_r2:.6f}\n"
    )
    report.write(
        f"Negative predictions: {pooled_reg_negative}\n"
    )
    report.write(
        "Predictions above development target max: "
        f"{pooled_reg_above_max}\n\n"
    )

    report.write(
        "CLASSIFICATION\n"
        "--------------\n"
    )
    report.write(
        f"Selected family: {selected_cls_family}\n"
    )
    report.write(
        f"Selected trial: {selected_cls_trial}\n"
    )
    report.write(
        "Balanced sample weights were recomputed from each "
        "fold-training target only.\n"
    )
    report.write(
        f"Pooled OOF Accuracy: {pooled_cls_accuracy:.6f}\n"
    )
    report.write(
        "Pooled OOF Balanced Accuracy: "
        f"{pooled_cls_balanced:.6f}\n"
    )
    report.write(
        f"Pooled OOF Macro-F1: {pooled_cls_macro_f1:.6f}\n"
    )
    report.write(
        "Pooled OOF Warning Recall: "
        f"{pooled_warning_recall:.6f}\n"
    )
    report.write(
        "Pooled OOF Critical Recall: "
        f"{pooled_critical_recall:.6f}\n"
    )
    report.write(
        "Warning Recall >= 0.80 reference: "
        f"{'PASS' if warning_reference_pass else 'BELOW REFERENCE'}\n"
    )
    report.write(
        "Critical Recall >= 0.50 reference: "
        f"{'PASS' if critical_reference_pass else 'BELOW REFERENCE'}\n\n"
    )

    report.write(
        "POOLED CONFUSION MATRIX\n"
        "-----------------------\n"
    )
    report.write(
        cls_cm_df.to_string()
        + "\n\n"
    )

    report.write(
        "METHODOLOGICAL CHECKS\n"
        "---------------------\n"
    )
    report.write(
        f"Batch-overlap folds: {fold_overlap_count}\n"
    )
    report.write(
        "All fold preprocessors fitted on fold Train only: "
        f"{all_fold_preprocessing_train_only}\n"
    )
    report.write(
        "Final Test contents loaded: "
        f"{final_test_content_read}\n"
    )
    report.write(
        f"Regression OOF rows: {len(reg_oof_df)}\n"
    )
    report.write(
        f"Classification OOF rows: {len(cls_oof_df)}\n\n"
    )

    report.write(
        "FINAL STATUS\n"
        "------------\n"
    )
    report.write(
        "STAGE 30 GROUP / BATCH CROSS-VALIDATION: PASS\n"
    )
    report.write(
        "This PASS refers to the validation methodology. "
        "Predictive robustness must be interpreted from the "
        "reported regression and classification metrics.\n"
    )


# ============================================================
# 20. OUTPUT PATHS
# ============================================================

print("\nRegression fold metrics:")
print(REG_FOLDS_PATH)

print("\nRegression OOF predictions:")
print(REG_OOF_PATH)

print("\nRegression batch metrics:")
print(REG_BATCH_PATH)

print("\nRegression summary:")
print(REG_SUMMARY_PATH)

print("\nClassification fold metrics:")
print(CLS_FOLDS_PATH)

print("\nClassification OOF predictions:")
print(CLS_OOF_PATH)

print("\nClassification confusion matrix:")
print(CLS_CM_PATH)

print("\nClassification batch metrics:")
print(CLS_BATCH_PATH)

print("\nClassification summary:")
print(CLS_SUMMARY_PATH)

print("\nCombined summary:")
print(COMBINED_SUMMARY_PATH)

print("\nReport:")
print(REPORT_PATH)
