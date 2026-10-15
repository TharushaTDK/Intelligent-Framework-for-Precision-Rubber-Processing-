from pathlib import Path
import ast
import hashlib
import json
import platform
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# ============================================================
# COMPONENT 3 - STAGE 33:
# EXPORT FINAL INFERENCE PIPELINES
# ============================================================
#
# Purpose:
#   Package the exact frozen Stage-31 model behavior for backend
#   inference.
#
# Exported runtime artifacts:
#
#   drying_time_pipeline.pkl
#   fire_risk_pipeline.pkl
#
# Each pipeline includes:
#   - median imputation for numeric features
#   - StandardScaler for numeric features
#   - OneHotEncoder for Smokehouse_ID / Layer_ID
#   - frozen final model
#
# IMPORTANT:
#   Stage-07 temporal/rolling feature construction is NOT inside
#   these sklearn pipelines.
#
#   The backend must construct the frozen 17 selected model input
#   features first, then pass those features into these pipelines.
#
# Training scope:
#   Train + Validation = Development only.
#
# The Final Test is NOT used for fitting.
# It is used here only for an artifact-integrity check against the
# already-saved Stage-31 predictions. No new model evaluation,
# tuning, threshold adjustment, or model selection occurs.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROJECT_ROOT = ROOT.parents[1]

SPLIT_DIR = (
    ROOT / "data" / "splits"
)

REPORT_DIR = (
    ROOT / "outputs" / "reports"
)

REG_FINAL_DIR = (
    ROOT
    / "artifacts"
    / "models"
    / "regression"
    / "final"
)

PIPELINE_ARTIFACT_DIR = (
    ROOT
    / "artifacts"
    / "pipelines"
)

BACKEND_MODEL_DIR = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "ml"
    / "component_3_smokehouse"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PIPELINE_ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BACKEND_MODEL_DIR.mkdir(
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

STAGE31_SUMMARY_PATH = (
    REPORT_DIR
    / "31_final_test_evaluation_summary.json"
)

STAGE31_REG_PREDICTIONS_PATH = (
    REPORT_DIR
    / "31_regression_final_test_predictions.csv"
)

STAGE31_CLS_PREDICTIONS_PATH = (
    REPORT_DIR
    / "31_classification_final_test_predictions.csv"
)

STAGE32_SUMMARY_PATH = (
    REPORT_DIR
    / "32_shap_explainability_summary.json"
)


# ============================================================
# 2. EXPORT PATHS
# ============================================================

RESEARCH_REG_PIPELINE_PATH = (
    PIPELINE_ARTIFACT_DIR
    / "drying_time_pipeline.pkl"
)

RESEARCH_CLS_PIPELINE_PATH = (
    PIPELINE_ARTIFACT_DIR
    / "fire_risk_pipeline.pkl"
)

BACKEND_REG_PIPELINE_PATH = (
    BACKEND_MODEL_DIR
    / "drying_time_pipeline.pkl"
)

BACKEND_CLS_PIPELINE_PATH = (
    BACKEND_MODEL_DIR
    / "fire_risk_pipeline.pkl"
)

BACKEND_METADATA_PATH = (
    BACKEND_MODEL_DIR
    / "component3_model_metadata.json"
)

BACKEND_CONTRACT_PATH = (
    BACKEND_MODEL_DIR
    / "component3_inference_contract.json"
)

CHECKS_PATH = (
    REPORT_DIR
    / "33_export_final_models_checks.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "33_export_final_models_summary.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "33_export_final_models_report.txt"
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

MODEL_VERSION = "component3-stage33-v1"


# ============================================================
# 4. HELPERS
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


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


def parse_dict_like(value):
    if isinstance(
        value,
        dict,
    ):
        return value

    text = str(
        value
    ).strip()

    if not text:
        return {}

    try:
        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed
    except Exception:
        pass

    return {}


def recover_selected_xgb_configuration(
    path,
):
    frame = pd.read_csv(
        path
    )

    if frame.empty:
        raise RuntimeError(
            "Stage-29 selected classification "
            "configuration is empty."
        )

    selected = frame.iloc[
        0
    ]

    family = (
        str(
            selected[
                "Model_Family"
            ]
        )
        if "Model_Family"
        in frame.columns
        else "Unknown"
    )

    trial = (
        str(
            selected[
                "Trial"
            ]
        )
        if "Trial"
        in frame.columns
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
            params = (
                parse_dict_like(
                    selected[
                        column
                    ]
                )
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
        if parameter
        not in params
    ]

    if missing:
        raise RuntimeError(
            "Could not recover exact "
            "Stage-29 XGBoost parameters. "
            f"Missing: {missing}"
        )

    return (
        family,
        trial,
        params,
    )


def normalize_label_column(
    series,
):
    return (
        series
        .astype(str)
        .str.strip()
    )


# ============================================================
# 5. REQUIRED INPUTS
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
    STAGE31_SUMMARY_PATH,
    STAGE31_REG_PREDICTIONS_PATH,
    STAGE31_CLS_PREDICTIONS_PATH,
    STAGE32_SUMMARY_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# 6. VERIFY STAGES 31 + 32
# ============================================================

with open(
    STAGE31_SUMMARY_PATH,
    "r",
    encoding="utf-8",
) as handle:
    stage31_summary = json.load(
        handle
    )

with open(
    STAGE32_SUMMARY_PATH,
    "r",
    encoding="utf-8",
) as handle:
    stage32_summary = json.load(
        handle
    )

if (
    stage31_summary.get(
        "methodological_status"
    )
    != "PASS"
):
    raise RuntimeError(
        "Stage 31 methodological "
        "status is not PASS."
    )

if (
    stage31_summary.get(
        "evaluation_scope"
    )
    != "LOCKED_FINAL_TEST"
):
    raise RuntimeError(
        "Stage 31 is not recorded "
        "as locked final-test evaluation."
    )

if (
    stage32_summary.get(
        "methodological_status"
    )
    != "PASS"
):
    raise RuntimeError(
        "Stage 32 methodological "
        "status is not PASS."
    )

stage32_reproduction = (
    stage32_summary.get(
        "stage31_predictions_reproduced",
        {},
    )
)

if not (
    stage32_reproduction.get(
        "regression"
    )
    is True
    and
    stage32_reproduction.get(
        "classification"
    )
    is True
):
    raise RuntimeError(
        "Stage 32 did not verify exact "
        "Stage-31 prediction reproduction."
    )


# ============================================================
# 7. FROZEN FEATURE CONTRACT
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
        "Frozen regression and "
        "classification feature "
        "lists do not match."
    )

SELECTED_FEATURES = (
    REGRESSION_FEATURES
)

if len(
    SELECTED_FEATURES
) != 17:
    raise RuntimeError(
        "Expected 17 selected "
        f"features, found "
        f"{len(SELECTED_FEATURES)}."
    )

CATEGORICAL_FEATURES = [
    "Smokehouse_ID",
    "Layer_ID",
]

NUMERIC_FEATURES = [
    feature
    for feature in (
        SELECTED_FEATURES
    )
    if feature
    not in (
        CATEGORICAL_FEATURES
    )
]

if len(
    NUMERIC_FEATURES
) != 15:
    raise RuntimeError(
        "Expected 15 numeric features, "
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
            "Forbidden field found "
            "in exported inference "
            f"feature contract: {forbidden}"
        )


# ============================================================
# 8. RECOVER FROZEN MODEL CONFIGURATIONS
# ============================================================

with open(
    REG_SELECTION_SUMMARY_PATH,
    "r",
    encoding="utf-8",
) as handle:
    regression_selection = (
        json.load(
            handle
        )
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
        "Frozen regression artifact "
        "is not GradientBoostingRegressor."
    )

(
    selected_cls_family,
    selected_cls_trial,
    xgb_params,
) = (
    recover_selected_xgb_configuration(
        CLS_SELECTION_CONFIG_PATH
    )
)


# ============================================================
# 9. LOAD DEVELOPMENT DATA FOR FINAL EXPORT FIT
# ============================================================

print("=" * 78)
print(
    "COMPONENT 3 - STAGE 33: "
    "EXPORT FINAL INFERENCE PIPELINES"
)
print("=" * 78)

print(
    "\nStage 31: PASS"
)
print(
    "Stage 32: PASS"
)
print(
    "Exporting the frozen "
    "Stage-31 model behavior."
)

train_df = pd.read_csv(
    TRAIN_PATH
)

validation_df = pd.read_csv(
    VALIDATION_PATH
)

development = pd.concat(
    [
        train_df,
        validation_df,
    ],
    ignore_index=True,
)

required_development_columns = (
    SELECTED_FEATURES
    + [
        "Remaining_Drying_Time",
        "Fire_Risk_Label",
    ]
)

missing_dev = [
    column
    for column in (
        required_development_columns
    )
    if column
    not in (
        development.columns
    )
]

if missing_dev:
    raise RuntimeError(
        "Development data missing "
        f"required columns: {missing_dev}"
    )

for feature in (
    NUMERIC_FEATURES
):
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
] = normalize_label_column(
    development[
        "Fire_Risk_Label"
    ]
)

unexpected_dev_labels = sorted(
    set(
        development[
            "Fire_Risk_Label"
        ].unique()
    )
    - set(
        CLASS_ORDER_TEXT
    )
)

if unexpected_dev_labels:
    raise RuntimeError(
        "Unexpected Development labels: "
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

development_sample_weights = (
    compute_sample_weight(
        class_weight="balanced",
        y=y_cls_development,
    )
)

print(
    "\nDevelopment rows:",
    f"{len(development):,}",
)
print(
    "Raw inference features:",
    len(
        SELECTED_FEATURES
    ),
)


# ============================================================
# 10. BUILD COMPLETE INFERENCE PIPELINES
# ============================================================

regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            make_preprocessor(
                NUMERIC_FEATURES,
                CATEGORICAL_FEATURES,
            ),
        ),
        (
            "model",
            clone(
                selected_reg_artifact
            ),
        ),
    ]
)

classification_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            make_preprocessor(
                NUMERIC_FEATURES,
                CATEGORICAL_FEATURES,
            ),
        ),
        (
            "model",
            XGBClassifier(
                **xgb_params
            ),
        ),
    ]
)

regression_pipeline.fit(
    X_development_raw,
    y_reg_development,
)

classification_pipeline.fit(
    X_development_raw,
    y_cls_development,
    model__sample_weight=(
        development_sample_weights
    ),
)

reg_transformed_names = (
    regression_pipeline[
        "preprocessor"
    ]
    .get_feature_names_out()
    .tolist()
)

cls_transformed_names = (
    classification_pipeline[
        "preprocessor"
    ]
    .get_feature_names_out()
    .tolist()
)

if (
    reg_transformed_names
    != cls_transformed_names
):
    raise RuntimeError(
        "Regression and classification "
        "transformed feature orders differ."
    )

if len(
    reg_transformed_names
) != 20:
    raise RuntimeError(
        "Expected 20 transformed "
        f"features, found "
        f"{len(reg_transformed_names)}."
    )

print(
    "\nPipelines fitted on "
    "Development ONLY."
)
print(
    "Transformed features:",
    len(
        reg_transformed_names
    ),
)


# ============================================================
# 11. SAVE RESEARCH PIPELINE ARTIFACTS
# ============================================================

joblib.dump(
    regression_pipeline,
    RESEARCH_REG_PIPELINE_PATH,
)

joblib.dump(
    classification_pipeline,
    RESEARCH_CLS_PIPELINE_PATH,
)

research_reg_hash = (
    sha256_file(
        RESEARCH_REG_PIPELINE_PATH
    )
)

research_cls_hash = (
    sha256_file(
        RESEARCH_CLS_PIPELINE_PATH
    )
)


# ============================================================
# 12. COPY TO BACKEND RUNTIME LOCATION
# ============================================================

shutil.copy2(
    RESEARCH_REG_PIPELINE_PATH,
    BACKEND_REG_PIPELINE_PATH,
)

shutil.copy2(
    RESEARCH_CLS_PIPELINE_PATH,
    BACKEND_CLS_PIPELINE_PATH,
)

backend_reg_hash = (
    sha256_file(
        BACKEND_REG_PIPELINE_PATH
    )
)

backend_cls_hash = (
    sha256_file(
        BACKEND_CLS_PIPELINE_PATH
    )
)

reg_copy_identical = (
    research_reg_hash
    == backend_reg_hash
)

cls_copy_identical = (
    research_cls_hash
    == backend_cls_hash
)

if not (
    reg_copy_identical
    and cls_copy_identical
):
    raise RuntimeError(
        "Backend pipeline copy hash "
        "does not match research artifact."
    )


# ============================================================
# 13. ARTIFACT INTEGRITY:
#     REPRODUCE STAGE-31 PREDICTIONS
# ============================================================
#
# Test was already legitimately opened in Stage 31.
# Here it is used ONLY to verify that exported artifacts reproduce
# the saved Stage-31 predictions. No metrics or tuning are done.

test_df = pd.read_csv(
    TEST_PATH
)

for feature in (
    NUMERIC_FEATURES
):
    test_df[
        feature
    ] = pd.to_numeric(
        test_df[
            feature
        ],
        errors="coerce",
    )

X_test_raw = (
    test_df[
        SELECTED_FEATURES
    ].copy()
)

stage31_reg_predictions = (
    pd.read_csv(
        STAGE31_REG_PREDICTIONS_PATH
    )
)

stage31_cls_predictions = (
    pd.read_csv(
        STAGE31_CLS_PREDICTIONS_PATH
    )
)

exported_reg_pred = (
    regression_pipeline.predict(
        X_test_raw
    )
)

exported_cls_pred = (
    classification_pipeline.predict(
        X_test_raw
    )
    .astype(int)
)

saved_reg_pred = (
    pd.to_numeric(
        stage31_reg_predictions[
            "Predicted_Remaining_Drying_Time"
        ],
        errors="raise",
    )
    .to_numpy()
)

if "y_pred_int" in (
    stage31_cls_predictions.columns
):
    saved_cls_pred = (
        pd.to_numeric(
            stage31_cls_predictions[
                "y_pred_int"
            ],
            errors="raise",
        )
        .to_numpy(
            dtype=int
        )
    )
else:
    saved_cls_pred = (
        stage31_cls_predictions[
            "Predicted_Fire_Risk_Label"
        ]
        .map(
            CLASS_TO_INT
        )
        .to_numpy(
            dtype=int
        )
    )

reg_prediction_match = bool(
    np.allclose(
        exported_reg_pred,
        saved_reg_pred,
        rtol=1e-10,
        atol=1e-10,
    )
)

cls_prediction_match = bool(
    np.array_equal(
        exported_cls_pred,
        saved_cls_pred,
    )
)

if not (
    reg_prediction_match
    and cls_prediction_match
):
    raise RuntimeError(
        "Exported pipelines do not "
        "reproduce Stage-31 predictions."
    )

# Reload backend copies and verify once more.
backend_reg_pipeline = (
    joblib.load(
        BACKEND_REG_PIPELINE_PATH
    )
)

backend_cls_pipeline = (
    joblib.load(
        BACKEND_CLS_PIPELINE_PATH
    )
)

backend_reg_pred = (
    backend_reg_pipeline.predict(
        X_test_raw
    )
)

backend_cls_pred = (
    backend_cls_pipeline.predict(
        X_test_raw
    )
    .astype(int)
)

backend_reg_reload_match = bool(
    np.allclose(
        backend_reg_pred,
        saved_reg_pred,
        rtol=1e-10,
        atol=1e-10,
    )
)

backend_cls_reload_match = bool(
    np.array_equal(
        backend_cls_pred,
        saved_cls_pred,
    )
)

if not (
    backend_reg_reload_match
    and backend_cls_reload_match
):
    raise RuntimeError(
        "Reloaded backend artifacts "
        "do not reproduce Stage-31 predictions."
    )

print(
    "\n--- STAGE-31 ARTIFACT REPRODUCTION ---"
)
print(
    "Research regression pipeline match      :",
    reg_prediction_match,
)
print(
    "Research classification pipeline match  :",
    cls_prediction_match,
)
print(
    "Backend regression reload match         :",
    backend_reg_reload_match,
)
print(
    "Backend classification reload match     :",
    backend_cls_reload_match,
)


# ============================================================
# 14. FINAL PERFORMANCE / LIMITATION METADATA
# ============================================================

stage31_regression = (
    stage31_summary.get(
        "regression",
        {},
    )
)

stage31_classification = (
    stage31_summary.get(
        "classification",
        {},
    )
)

critical_recall = (
    stage31_classification.get(
        "Critical_Recall"
    )
)

warning_recall = (
    stage31_classification.get(
        "Warning_Recall"
    )
)

critical_limitation = (
    critical_recall is None
    or float(
        critical_recall
    )
    < 0.50
)


# ============================================================
# 15. BACKEND INFERENCE CONTRACT
# ============================================================

feature_contract = []

for feature in (
    SELECTED_FEATURES
):
    feature_contract.append({
        "name":
            feature,
        "type":
            (
                "categorical"
                if feature
                in CATEGORICAL_FEATURES
                else "numeric"
            ),
        "required":
            True,
    })

inference_contract = {
    "component":
        "component_3_smokehouse",
    "model_version":
        MODEL_VERSION,
    "input_contract": {
        "description":
            (
                "The backend must construct these 17 "
                "post-feature-engineering inputs before "
                "calling either exported sklearn pipeline."
            ),
        "feature_count":
            len(
                SELECTED_FEATURES
            ),
        "ordered_features":
            SELECTED_FEATURES,
        "features":
            feature_contract,
        "categorical_features":
            CATEGORICAL_FEATURES,
        "numeric_features":
            NUMERIC_FEATURES,
        "excluded_runtime_fields": [
            "Batch_ID",
            "Timestamp",
            "Actual_Total_Drying_Time",
            "Remaining_Drying_Time",
            "Fire_Risk_Label",
        ],
        "note":
            (
                "Batch_ID/Timestamp may be needed by the backend "
                "to construct temporal context, but they are not "
                "passed as physical model predictors unless they "
                "appear through an explicitly engineered selected "
                "feature."
            ),
    },
    "preprocessing_inside_pipeline": {
        "numeric":
            "SimpleImputer(strategy='median') + StandardScaler",
        "categorical":
            "OneHotEncoder(handle_unknown='ignore')",
        "transformed_feature_count":
            len(
                reg_transformed_names
            ),
        "transformed_feature_order":
            reg_transformed_names,
    },
    "regression_output": {
        "pipeline_file":
            "drying_time_pipeline.pkl",
        "target":
            "Remaining_Drying_Time",
        "unit":
            "hours",
        "model_family":
            selected_reg_family,
        "selected_trial":
            selected_reg_trial,
        "postprocessing_note":
            (
                "Do not silently alter research evaluation outputs. "
                "For runtime UI safety, a separate service-layer "
                "rule may clamp impossible negative displayed time "
                "to zero if explicitly documented."
            ),
    },
    "classification_output": {
        "pipeline_file":
            "fire_risk_pipeline.pkl",
        "target":
            "Fire_Risk_Label",
        "model_family":
            selected_cls_family,
        "selected_trial":
            selected_cls_trial,
        "pipeline_predict_returns":
            "integer class id",
        "class_id_to_label":
            {
                str(key):
                    value
                for key, value
                in INT_TO_CLASS.items()
            },
        "predict_proba_class_order": [
            {
                "class_id":
                    class_id,
                "label":
                    INT_TO_CLASS[
                        class_id
                    ],
            }
            for class_id
            in sorted(
                INT_TO_CLASS
            )
        ],
        "safety_limitation":
            (
                "Locked final-test Critical recall was "
                f"{critical_recall}. "
                "This classifier is a research/prototype "
                "decision-support model and must NOT be treated "
                "as a proven standalone Critical fire alarm."
            ),
    },
}

with open(
    BACKEND_CONTRACT_PATH,
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        inference_contract,
        handle,
        indent=2,
    )


# ============================================================
# 16. MODEL METADATA
# ============================================================

metadata = {
    "component":
        "Component 3 - Smart Layer-Wise Rubber Smokehouse "
        "Drying-Time & Fire-Risk Prediction System",
    "model_version":
        MODEL_VERSION,
    "stage":
        33,
    "status":
        "FINAL_RESEARCH_EXPORT",
    "training_scope": {
        "fit_data":
            "Stage-09 Train + Validation (Development)",
        "development_rows":
            len(
                development
            ),
        "final_test_included_in_fit":
            False,
        "note":
            (
                "Final Test was used only for locked Stage-31 "
                "evaluation and Stage-33 artifact reproduction "
                "verification, never for fitting/tuning."
            ),
    },
    "feature_contract": {
        "raw_selected_feature_count":
            len(
                SELECTED_FEATURES
            ),
        "selected_features":
            SELECTED_FEATURES,
        "numeric_features":
            NUMERIC_FEATURES,
        "categorical_features":
            CATEGORICAL_FEATURES,
        "transformed_feature_count":
            len(
                reg_transformed_names
            ),
        "transformed_feature_names":
            reg_transformed_names,
    },
    "regression": {
        "pipeline":
            str(
                BACKEND_REG_PIPELINE_PATH
            ),
        "model_family":
            selected_reg_family,
        "trial":
            selected_reg_trial,
        "locked_final_test": {
            "MAE":
                stage31_regression.get(
                    "MAE"
                ),
            "RMSE":
                stage31_regression.get(
                    "RMSE"
                ),
            "R2":
                stage31_regression.get(
                    "R2"
                ),
            "negative_predictions":
                stage31_regression.get(
                    "negative_predictions"
                ),
        },
        "sha256":
            backend_reg_hash,
    },
    "classification": {
        "pipeline":
            str(
                BACKEND_CLS_PIPELINE_PATH
            ),
        "model_family":
            selected_cls_family,
        "trial":
            selected_cls_trial,
        "class_mapping":
            {
                str(key):
                    value
                for key, value
                in INT_TO_CLASS.items()
            },
        "locked_final_test": {
            "Accuracy":
                stage31_classification.get(
                    "Accuracy"
                ),
            "Balanced_Accuracy":
                stage31_classification.get(
                    "Balanced_Accuracy"
                ),
            "Macro_F1":
                stage31_classification.get(
                    "Macro_F1"
                ),
            "Warning_Recall":
                warning_recall,
            "Critical_Recall":
                critical_recall,
        },
        "critical_detection_limitation":
            (
                "Major limitation: locked final-test Critical "
                "Recall did not meet the 0.50 reference. "
                "All Critical-risk performance claims must "
                "preserve this limitation."
                if critical_limitation
                else None
            ),
        "sha256":
            backend_cls_hash,
    },
    "research_integrity": {
        "dataset_type":
            "synthetic/simulated development data",
        "field_collected_claim":
            False,
        "test_based_retuning_after_stage31":
            False,
        "stage32_used_for_model_changes":
            False,
        "deployment_scope":
            (
                "Research/prototype decision support pending "
                "independent field/expert validation."
            ),
    },
    "artifact_reproduction": {
        "research_regression_matches_stage31":
            reg_prediction_match,
        "research_classification_matches_stage31":
            cls_prediction_match,
        "backend_regression_reload_matches_stage31":
            backend_reg_reload_match,
        "backend_classification_reload_matches_stage31":
            backend_cls_reload_match,
    },
    "software_environment": {
        "python":
            sys.version.split()[0],
        "platform":
            platform.platform(),
        "numpy":
            np.__version__,
        "pandas":
            pd.__version__,
        "scikit_learn":
            sklearn.__version__,
        "xgboost":
            xgboost.__version__,
        "joblib":
            joblib.__version__,
    },
}

with open(
    BACKEND_METADATA_PATH,
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        metadata,
        handle,
        indent=2,
    )


# ============================================================
# 17. FINAL CHECKS
# ============================================================

checks = [
    {
        "Check":
            "Stage 31 methodological status",
        "Status":
            "PASS",
    },
    {
        "Check":
            "Stage 32 methodological status",
        "Status":
            "PASS",
    },
    {
        "Check":
            "17-feature inference contract",
        "Status":
            (
                "PASS"
                if len(
                    SELECTED_FEATURES
                )
                == 17
                else "FAIL"
            ),
    },
    {
        "Check":
            "20 transformed features",
        "Status":
            (
                "PASS"
                if len(
                    reg_transformed_names
                )
                == 20
                else "FAIL"
            ),
    },
    {
        "Check":
            "Regression research artifact reproduces Stage 31",
        "Status":
            (
                "PASS"
                if reg_prediction_match
                else "FAIL"
            ),
    },
    {
        "Check":
            "Classification research artifact reproduces Stage 31",
        "Status":
            (
                "PASS"
                if cls_prediction_match
                else "FAIL"
            ),
    },
    {
        "Check":
            "Regression backend artifact hash copy",
        "Status":
            (
                "PASS"
                if reg_copy_identical
                else "FAIL"
            ),
    },
    {
        "Check":
            "Classification backend artifact hash copy",
        "Status":
            (
                "PASS"
                if cls_copy_identical
                else "FAIL"
            ),
    },
    {
        "Check":
            "Reloaded backend regression reproduces Stage 31",
        "Status":
            (
                "PASS"
                if backend_reg_reload_match
                else "FAIL"
            ),
    },
    {
        "Check":
            "Reloaded backend classification reproduces Stage 31",
        "Status":
            (
                "PASS"
                if backend_cls_reload_match
                else "FAIL"
            ),
    },
    {
        "Check":
            "Final Test excluded from export fitting",
        "Status":
            "PASS",
    },
    {
        "Check":
            "Critical-class limitation preserved in metadata",
        "Status":
            (
                "PASS"
                if (
                    critical_limitation
                    and
                    metadata[
                        "classification"
                    ][
                        "critical_detection_limitation"
                    ]
                    is not None
                )
                else (
                    "PASS"
                    if not critical_limitation
                    else "FAIL"
                )
            ),
    },
]

checks_df = pd.DataFrame(
    checks
)

checks_df.to_csv(
    CHECKS_PATH,
    index=False,
)

if (
    checks_df[
        "Status"
    ]
    != "PASS"
).any():
    raise RuntimeError(
        "Stage 33 export checks "
        "contain FAIL."
    )


# ============================================================
# 18. SUMMARY JSON
# ============================================================

summary = {
    "stage":
        33,
    "status":
        "PASS",
    "model_version":
        MODEL_VERSION,
    "development_rows":
        len(
            development
        ),
    "final_test_used_for_fit":
        False,
    "research_artifacts": {
        "drying_time_pipeline":
            str(
                RESEARCH_REG_PIPELINE_PATH
            ),
        "fire_risk_pipeline":
            str(
                RESEARCH_CLS_PIPELINE_PATH
            ),
    },
    "backend_artifacts": {
        "drying_time_pipeline":
            str(
                BACKEND_REG_PIPELINE_PATH
            ),
        "fire_risk_pipeline":
            str(
                BACKEND_CLS_PIPELINE_PATH
            ),
        "metadata":
            str(
                BACKEND_METADATA_PATH
            ),
        "inference_contract":
            str(
                BACKEND_CONTRACT_PATH
            ),
    },
    "hashes": {
        "drying_time_pipeline_sha256":
            backend_reg_hash,
        "fire_risk_pipeline_sha256":
            backend_cls_hash,
    },
    "stage31_reproduction": {
        "regression":
            backend_reg_reload_match,
        "classification":
            backend_cls_reload_match,
    },
    "critical_class_limitation_preserved":
        bool(
            critical_limitation
        ),
    "next_phase":
        "FastAPI backend integration",
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
# 19. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 33: "
        "EXPORT FINAL INFERENCE PIPELINES\n"
    )

    report.write(
        "=" * 78 + "\n\n"
    )

    report.write(
        "EXPORT STRATEGY\n"
        "---------------\n"
    )

    report.write(
        "The frozen Stage-31 regression and classification "
        "configurations were refitted on the full Development "
        "dataset (Train + Validation) only.\n"
    )

    report.write(
        "The Final Test was not included in fitting. "
        "It was used only to verify exact reproduction of the "
        "already-saved Stage-31 predictions.\n\n"
    )

    report.write(
        "PIPELINES\n"
        "---------\n"
    )

    report.write(
        f"Regression: {selected_reg_family} / "
        f"{selected_reg_trial}\n"
    )

    report.write(
        f"Classification: {selected_cls_family} / "
        f"{selected_cls_trial}\n"
    )

    report.write(
        "Each pipeline contains numeric median imputation, "
        "StandardScaler, categorical OneHotEncoder and the "
        "frozen selected model.\n"
    )

    report.write(
        "Stage-07 feature engineering is NOT inside the sklearn "
        "pipeline and must be constructed in backend logic before "
        "calling the pipeline.\n\n"
    )

    report.write(
        "ARTIFACT VALIDATION\n"
        "-------------------\n"
    )

    report.write(
        f"Regression Stage-31 reproduction: "
        f"{backend_reg_reload_match}\n"
    )

    report.write(
        f"Classification Stage-31 reproduction: "
        f"{backend_cls_reload_match}\n"
    )

    report.write(
        f"Regression SHA256: {backend_reg_hash}\n"
    )

    report.write(
        f"Classification SHA256: {backend_cls_hash}\n\n"
    )

    report.write(
        "CLASSIFICATION LIMITATION\n"
        "-------------------------\n"
    )

    report.write(
        f"Locked final-test Warning Recall: "
        f"{warning_recall}\n"
    )

    report.write(
        f"Locked final-test Critical Recall: "
        f"{critical_recall}\n"
    )

    if critical_limitation:
        report.write(
            "Critical-risk detection remains a major documented "
            "limitation. The exported classifier is a research/"
            "prototype decision-support artifact and must not be "
            "described as a proven standalone Critical alarm.\n\n"
        )
    else:
        report.write(
            "The configured Critical-recall reference was met.\n\n"
        )

    report.write(
        "BACKEND LOCATION\n"
        "----------------\n"
    )

    report.write(
        f"{BACKEND_REG_PIPELINE_PATH}\n"
    )

    report.write(
        f"{BACKEND_CLS_PIPELINE_PATH}\n"
    )

    report.write(
        f"{BACKEND_METADATA_PATH}\n"
    )

    report.write(
        f"{BACKEND_CONTRACT_PATH}\n\n"
    )

    report.write(
        "FINAL STATUS\n"
        "------------\n"
    )

    report.write(
        "STAGE 33 EXPORT FINAL MODELS: PASS\n"
    )

    report.write(
        "Component 3 ML research lifecycle stages "
        "01-33 are complete.\n"
    )

    report.write(
        "Next phase: FastAPI backend integration.\n"
    )


# ============================================================
# 20. CONSOLE SUMMARY
# ============================================================

print(
    "\n--- FINAL EXPORTED PIPELINES ---"
)

print(
    "Research regression:"
)
print(
    RESEARCH_REG_PIPELINE_PATH
)

print(
    "\nResearch classification:"
)
print(
    RESEARCH_CLS_PIPELINE_PATH
)

print(
    "\nBackend regression:"
)
print(
    BACKEND_REG_PIPELINE_PATH
)

print(
    "\nBackend classification:"
)
print(
    BACKEND_CLS_PIPELINE_PATH
)

print(
    "\n--- SHA256 ---"
)

print(
    "Regression:",
    backend_reg_hash,
)

print(
    "Classification:",
    backend_cls_hash,
)

print(
    "\n--- FINAL-TEST PERFORMANCE METADATA ---"
)

print(
    "Regression MAE:",
    stage31_regression.get(
        "MAE"
    ),
)

print(
    "Regression RMSE:",
    stage31_regression.get(
        "RMSE"
    ),
)

print(
    "Regression R2:",
    stage31_regression.get(
        "R2"
    ),
)

print(
    "Classification Warning Recall:",
    warning_recall,
)

print(
    "Classification Critical Recall:",
    critical_recall,
)

if critical_limitation:
    print(
        "Critical detection limitation: DOCUMENTED"
    )

print(
    "\n" + "=" * 78
)

print(
    "STAGE 33 EXPORT FINAL MODELS: PASS"
)

print(
    "=" * 78
)

print(
    "Component 3 ML research lifecycle "
    "Stages 01-33 are complete."
)

print(
    "Next phase: FastAPI backend integration."
)

print(
    "\nReports:"
)

for path in [
    CHECKS_PATH,
    SUMMARY_PATH,
    REPORT_PATH,
    BACKEND_METADATA_PATH,
    BACKEND_CONTRACT_PATH,
]:
    print(
        path
    )
