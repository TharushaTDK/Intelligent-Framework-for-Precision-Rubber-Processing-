from pathlib import Path
import ast
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "The 'shap' package is required for Stage 32.\n"
        "Install it inside the active virtual environment with:\n"
        "pip install shap"
    ) from exc


# ============================================================
# COMPONENT 3 - STAGE 32:
# SHAP EXPLAINABILITY
# ============================================================
#
# Purpose:
#   Explain the FINAL frozen model behavior after Stage 31.
#
# This stage DOES NOT:
#   - tune hyperparameters
#   - reselect models
#   - adjust thresholds
#   - rebalance/resample based on test results
#   - modify targets
#
# It DOES:
#   - reconstruct the exact Stage-31 frozen pipelines
#   - verify predictions match saved Stage-31 predictions
#   - compute SHAP for regression
#   - compute SHAP for multiclass classification
#   - explain actual Critical test rows separately
#
# Research interpretation:
#   SHAP explains associations used by the trained model.
#   It does NOT prove causal relationships.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = (
    ROOT
    / "outputs"
    / "figures"
    / "explainability"
)

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

FIGURE_DIR.mkdir(
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


# ============================================================
# 2. OUTPUT FILES
# ============================================================

REG_IMPORTANCE_PATH = (
    REPORT_DIR
    / "32_regression_shap_importance.csv"
)

REG_RAW_IMPORTANCE_PATH = (
    REPORT_DIR
    / "32_regression_shap_importance_aggregated_raw_features.csv"
)

CLS_IMPORTANCE_PATH = (
    REPORT_DIR
    / "32_classification_shap_importance_by_class.csv"
)

CLS_RAW_IMPORTANCE_PATH = (
    REPORT_DIR
    / "32_classification_shap_importance_aggregated_raw_features.csv"
)

CLS_OVERALL_IMPORTANCE_PATH = (
    REPORT_DIR
    / "32_classification_shap_overall_importance.csv"
)

CRITICAL_DIAGNOSTIC_PATH = (
    REPORT_DIR
    / "32_actual_critical_shap_diagnostic.csv"
)

CRITICAL_ROWS_PATH = (
    REPORT_DIR
    / "32_actual_critical_row_explanations.csv"
)

CHECKS_PATH = (
    REPORT_DIR
    / "32_shap_explainability_checks.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "32_shap_explainability_summary.json"
)

REPORT_PATH = (
    REPORT_DIR
    / "32_shap_explainability_report.txt"
)

REG_SUMMARY_FIGURE = (
    FIGURE_DIR
    / "32_regression_shap_summary.png"
)

REG_BAR_FIGURE = (
    FIGURE_DIR
    / "32_regression_shap_importance_bar.png"
)

CLS_NORMAL_FIGURE = (
    FIGURE_DIR
    / "32_classification_shap_summary_normal.png"
)

CLS_WARNING_FIGURE = (
    FIGURE_DIR
    / "32_classification_shap_summary_warning.png"
)

CLS_CRITICAL_FIGURE = (
    FIGURE_DIR
    / "32_classification_shap_summary_critical.png"
)

CLS_OVERALL_BAR_FIGURE = (
    FIGURE_DIR
    / "32_classification_shap_overall_importance_bar.png"
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

RANDOM_STATE = 42


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
            "XGBoost parameters. "
            f"Missing: {missing}"
        )

    return (
        family,
        trial,
        params,
    )


def transformed_to_raw_feature(
    transformed_feature,
    numeric_features,
):
    if transformed_feature in numeric_features:
        return transformed_feature

    if transformed_feature.startswith(
        "Smokehouse_ID_"
    ):
        return "Smokehouse_ID"

    if transformed_feature.startswith(
        "Layer_ID_"
    ):
        return "Layer_ID"

    return transformed_feature


def normalize_multiclass_shap(
    shap_result,
    n_samples,
    n_features,
    n_classes,
):
    """
    Convert common SHAP multiclass formats into:
        shape = (n_classes, n_samples, n_features)
    """

    # Older SHAP: list[class] -> (samples, features)
    if isinstance(
        shap_result,
        list,
    ):
        arrays = [
            np.asarray(item)
            for item in shap_result
        ]

        if len(arrays) != n_classes:
            raise RuntimeError(
                "Unexpected number of SHAP "
                f"class arrays: {len(arrays)}"
            )

        result = np.stack(
            arrays,
            axis=0,
        )

        if result.shape != (
            n_classes,
            n_samples,
            n_features,
        ):
            raise RuntimeError(
                "Unexpected list-based multiclass "
                f"SHAP shape: {result.shape}"
            )

        return result

    # Newer SHAP Explanation or ndarray.
    if hasattr(
        shap_result,
        "values",
    ):
        values = np.asarray(
            shap_result.values
        )
    else:
        values = np.asarray(
            shap_result
        )

    # Possible:
    # (samples, features, classes)
    if values.shape == (
        n_samples,
        n_features,
        n_classes,
    ):
        return np.moveaxis(
            values,
            -1,
            0,
        )

    # Possible:
    # (classes, samples, features)
    if values.shape == (
        n_classes,
        n_samples,
        n_features,
    ):
        return values

    # Possible:
    # (samples, classes, features)
    if values.shape == (
        n_samples,
        n_classes,
        n_features,
    ):
        return np.moveaxis(
            values,
            1,
            0,
        )

    raise RuntimeError(
        "Unsupported multiclass SHAP output "
        f"shape: {values.shape}"
    )


def save_summary_plot(
    shap_values,
    feature_data,
    output_path,
    title,
):
    plt.figure()

    shap.summary_plot(
        shap_values,
        feature_data,
        show=False,
        max_display=15,
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_bar_plot(
    importance_df,
    feature_column,
    importance_column,
    output_path,
    title,
    top_n=15,
):
    plot_df = (
        importance_df
        .sort_values(
            importance_column,
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            importance_column,
            ascending=True,
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.barh(
        plot_df[
            feature_column
        ],
        plot_df[
            importance_column
        ],
    )

    plt.xlabel(
        "Mean |SHAP value|"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 5. REQUIRED FILES
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
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# 6. VERIFY STAGE 31
# ============================================================

with open(
    STAGE31_SUMMARY_PATH,
    "r",
    encoding="utf-8",
) as handle:
    stage31_summary = json.load(
        handle
    )

if (
    stage31_summary.get(
        "methodological_status"
    )
    != "PASS"
):
    raise RuntimeError(
        "Stage 31 methodological status is not PASS."
    )

if (
    stage31_summary.get(
        "evaluation_scope"
    )
    != "LOCKED_FINAL_TEST"
):
    raise RuntimeError(
        "Stage 31 summary is not a locked final-test evaluation."
    )


# ============================================================
# 7. FROZEN FEATURES
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
        "Stage-08 feature lists do not match."
    )

SELECTED_FEATURES = (
    REGRESSION_FEATURES
)

if len(
    SELECTED_FEATURES
) != 17:
    raise RuntimeError(
        "Expected 17 frozen raw features."
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

if len(
    NUMERIC_FEATURES
) != 15:
    raise RuntimeError(
        "Expected 15 numeric features."
    )


# ============================================================
# 8. FROZEN MODELS
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
        "Frozen regression artifact is not "
        "GradientBoostingRegressor."
    )

(
    selected_cls_family,
    selected_cls_trial,
    xgb_params,
) = recover_selected_xgb_configuration(
    CLS_SELECTION_CONFIG_PATH
)


# ============================================================
# 9. LOAD DEVELOPMENT + FINAL TEST
# ============================================================

print("=" * 78)
print(
    "COMPONENT 3 - STAGE 32: "
    "SHAP EXPLAINABILITY"
)
print("=" * 78)

print("\nStage 31 methodological status: PASS")
print(
    "Stage 32 is EXPLAINABILITY ONLY."
)
print(
    "No tuning/reselection will be performed."
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

test_df = pd.read_csv(
    TEST_PATH
)

for frame_name, frame in [
    (
        "Development",
        development,
    ),
    (
        "Final Test",
        test_df,
    ),
]:
    required_columns = (
        SELECTED_FEATURES
        + [
            "Batch_ID",
            "Layer_ID",
            "Remaining_Drying_Time",
            "Fire_Risk_Label",
        ]
    )

    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"{frame_name} missing columns: "
            f"{missing}"
        )

    for feature in NUMERIC_FEATURES:
        frame[
            feature
        ] = pd.to_numeric(
            frame[
                feature
            ],
            errors="coerce",
        )

    frame[
        "Remaining_Drying_Time"
    ] = pd.to_numeric(
        frame[
            "Remaining_Drying_Time"
        ],
        errors="raise",
    )

    frame[
        "Fire_Risk_Label"
    ] = (
        frame[
            "Fire_Risk_Label"
        ]
        .astype(str)
        .str.strip()
    )

    unexpected = sorted(
        set(
            frame[
                "Fire_Risk_Label"
            ].unique()
        )
        - set(
            CLASS_ORDER_TEXT
        )
    )

    if unexpected:
        raise RuntimeError(
            f"{frame_name} has unexpected labels: "
            f"{unexpected}"
        )

    frame[
        "Fire_Risk_Int"
    ] = (
        frame[
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

test_batches = set(
    test_df[
        "Batch_ID"
    ]
    .astype(str)
    .unique()
)

if (
    development_batches
    & test_batches
):
    raise RuntimeError(
        "Development/Test batch overlap detected."
    )


# ============================================================
# 10. RECONSTRUCT EXACT STAGE-31 FIT
# ============================================================

X_development_raw = (
    development[
        SELECTED_FEATURES
    ].copy()
)

X_test_raw = (
    test_df[
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

X_test = (
    preprocessor.transform(
        X_test_raw
    )
)

transformed_feature_names = (
    preprocessor
    .get_feature_names_out()
    .tolist()
)

if len(
    transformed_feature_names
) != 20:
    raise RuntimeError(
        "Expected 20 transformed features, "
        f"found {len(transformed_feature_names)}."
    )

X_test_transformed_df = pd.DataFrame(
    X_test,
    columns=(
        transformed_feature_names
    ),
)

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


# ============================================================
# 11. VERIFY EXACT STAGE-31 PREDICTIONS
# ============================================================

stage32_reg_pred = (
    regression_model.predict(
        X_test
    )
)

stage32_cls_pred = (
    classification_model.predict(
        X_test
    )
    .astype(int)
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

if (
    len(stage31_reg_predictions)
    != len(test_df)
):
    raise RuntimeError(
        "Stage-31 regression prediction row count "
        "does not match Final Test."
    )

if (
    len(stage31_cls_predictions)
    != len(test_df)
):
    raise RuntimeError(
        "Stage-31 classification prediction row count "
        "does not match Final Test."
    )

saved_reg_pred = pd.to_numeric(
    stage31_reg_predictions[
        "Predicted_Remaining_Drying_Time"
    ],
    errors="raise",
).to_numpy()

reg_predictions_match = bool(
    np.allclose(
        stage32_reg_pred,
        saved_reg_pred,
        rtol=1e-10,
        atol=1e-10,
    )
)

if "y_pred_int" in (
    stage31_cls_predictions.columns
):
    saved_cls_pred = pd.to_numeric(
        stage31_cls_predictions[
            "y_pred_int"
        ],
        errors="raise",
    ).to_numpy(
        dtype=int
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

cls_predictions_match = bool(
    np.array_equal(
        stage32_cls_pred,
        saved_cls_pred,
    )
)

if not (
    reg_predictions_match
    and cls_predictions_match
):
    raise RuntimeError(
        "Stage-32 reconstructed models do not "
        "reproduce Stage-31 predictions. "
        "SHAP explanation stopped."
    )

print("\n--- STAGE-31 REPRODUCTION CHECK ---")
print(
    "Regression predictions match Stage 31     :",
    reg_predictions_match,
)
print(
    "Classification predictions match Stage 31 :",
    cls_predictions_match,
)
print(
    "SHAP will explain the exact final-evaluated behavior."
)


# ============================================================
# 12. REGRESSION SHAP
# ============================================================

print("\n--- REGRESSION SHAP ---")

reg_explainer = shap.TreeExplainer(
    regression_model
)

reg_shap_result = (
    reg_explainer.shap_values(
        X_test
    )
)

reg_shap_values = np.asarray(
    reg_shap_result
)

if reg_shap_values.shape != (
    len(test_df),
    len(transformed_feature_names),
):
    raise RuntimeError(
        "Unexpected regression SHAP shape: "
        f"{reg_shap_values.shape}"
    )

reg_mean_abs = np.mean(
    np.abs(
        reg_shap_values
    ),
    axis=0,
)

reg_importance_df = pd.DataFrame({
    "Transformed_Feature":
        transformed_feature_names,
    "Mean_Absolute_SHAP":
        reg_mean_abs,
})

reg_importance_df[
    "Raw_Feature"
] = [
    transformed_to_raw_feature(
        feature,
        NUMERIC_FEATURES,
    )
    for feature in (
        transformed_feature_names
    )
]

reg_importance_df = (
    reg_importance_df
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

reg_raw_importance_df = (
    reg_importance_df
    .groupby(
        "Raw_Feature",
        as_index=False,
    )[
        "Mean_Absolute_SHAP"
    ]
    .sum()
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

reg_importance_df.to_csv(
    REG_IMPORTANCE_PATH,
    index=False,
)

reg_raw_importance_df.to_csv(
    REG_RAW_IMPORTANCE_PATH,
    index=False,
)

save_summary_plot(
    reg_shap_values,
    X_test_transformed_df,
    REG_SUMMARY_FIGURE,
    (
        "Stage 32 — Regression SHAP Summary\n"
        "Remaining Drying Time"
    ),
)

save_bar_plot(
    reg_importance_df,
    "Transformed_Feature",
    "Mean_Absolute_SHAP",
    REG_BAR_FIGURE,
    (
        "Regression Global Feature Importance "
        "(Mean |SHAP|)"
    ),
)

print(
    "Regression SHAP rows:",
    reg_shap_values.shape[0],
)

print("\nTop regression features:")
print(
    reg_importance_df[
        [
            "Transformed_Feature",
            "Mean_Absolute_SHAP",
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 13. CLASSIFICATION SHAP
# ============================================================

print("\n--- CLASSIFICATION SHAP ---")

cls_explainer = shap.TreeExplainer(
    classification_model
)

cls_raw_shap = (
    cls_explainer.shap_values(
        X_test
    )
)

cls_shap_values = (
    normalize_multiclass_shap(
        cls_raw_shap,
        n_samples=len(test_df),
        n_features=len(
            transformed_feature_names
        ),
        n_classes=3,
    )
)

# Shape:
#   class x samples x features
print(
    "Classification SHAP normalized shape:",
    cls_shap_values.shape,
)

classification_rows = []

for class_id, class_name in enumerate(
    CLASS_ORDER_TEXT
):
    class_values = (
        cls_shap_values[
            class_id
        ]
    )

    class_mean_abs = np.mean(
        np.abs(
            class_values
        ),
        axis=0,
    )

    for feature, importance in zip(
        transformed_feature_names,
        class_mean_abs,
    ):
        classification_rows.append({
            "Class":
                class_name,
            "Class_ID":
                class_id,
            "Transformed_Feature":
                feature,
            "Raw_Feature":
                transformed_to_raw_feature(
                    feature,
                    NUMERIC_FEATURES,
                ),
            "Mean_Absolute_SHAP":
                float(
                    importance
                ),
        })

cls_importance_df = pd.DataFrame(
    classification_rows
)

cls_importance_df = (
    cls_importance_df
    .sort_values(
        [
            "Class_ID",
            "Mean_Absolute_SHAP",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)

cls_importance_df.to_csv(
    CLS_IMPORTANCE_PATH,
    index=False,
)

cls_raw_importance_df = (
    cls_importance_df
    .groupby(
        [
            "Class",
            "Class_ID",
            "Raw_Feature",
        ],
        as_index=False,
    )[
        "Mean_Absolute_SHAP"
    ]
    .sum()
    .sort_values(
        [
            "Class_ID",
            "Mean_Absolute_SHAP",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)

cls_raw_importance_df.to_csv(
    CLS_RAW_IMPORTANCE_PATH,
    index=False,
)

overall_cls_mean_abs = np.mean(
    np.abs(
        cls_shap_values
    ),
    axis=(0, 1),
)

cls_overall_importance_df = (
    pd.DataFrame({
        "Transformed_Feature":
            transformed_feature_names,
        "Mean_Absolute_SHAP_Across_Classes":
            overall_cls_mean_abs,
    })
)

cls_overall_importance_df[
    "Raw_Feature"
] = [
    transformed_to_raw_feature(
        feature,
        NUMERIC_FEATURES,
    )
    for feature in (
        transformed_feature_names
    )
]

cls_overall_importance_df = (
    cls_overall_importance_df
    .sort_values(
        "Mean_Absolute_SHAP_Across_Classes",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

cls_overall_importance_df.to_csv(
    CLS_OVERALL_IMPORTANCE_PATH,
    index=False,
)

save_summary_plot(
    cls_shap_values[
        0
    ],
    X_test_transformed_df,
    CLS_NORMAL_FIGURE,
    (
        "Stage 32 — Classification SHAP\n"
        "Normal Output"
    ),
)

save_summary_plot(
    cls_shap_values[
        1
    ],
    X_test_transformed_df,
    CLS_WARNING_FIGURE,
    (
        "Stage 32 — Classification SHAP\n"
        "Warning Output"
    ),
)

save_summary_plot(
    cls_shap_values[
        2
    ],
    X_test_transformed_df,
    CLS_CRITICAL_FIGURE,
    (
        "Stage 32 — Classification SHAP\n"
        "Critical Output"
    ),
)

save_bar_plot(
    cls_overall_importance_df,
    "Transformed_Feature",
    "Mean_Absolute_SHAP_Across_Classes",
    CLS_OVERALL_BAR_FIGURE,
    (
        "Classification Global Feature Importance "
        "(Across Classes)"
    ),
)

for class_id, class_name in enumerate(
    CLASS_ORDER_TEXT
):
    print(
        f"\nTop features for {class_name}:"
    )

    print(
        cls_importance_df[
            cls_importance_df[
                "Class"
            ]
            == class_name
        ][
            [
                "Transformed_Feature",
                "Mean_Absolute_SHAP",
            ]
        ]
        .head(8)
        .to_string(
            index=False
        )
    )


# ============================================================
# 14. ACTUAL-CRITICAL DIAGNOSTIC
# ============================================================
#
# Stage 31 had 18 actual Critical rows, all predicted Warning.
#
# This section explains those rows POST-HOC.
# It does NOT use them to alter the model.

actual_critical_mask = (
    test_df[
        "Fire_Risk_Int"
    ]
    .to_numpy(
        dtype=int
    )
    == CLASS_TO_INT[
        "Critical"
    ]
)

critical_count = int(
    actual_critical_mask.sum()
)

critical_pred_counts = (
    pd.Series(
        stage32_cls_pred[
            actual_critical_mask
        ]
    )
    .map(
        INT_TO_CLASS
    )
    .value_counts()
    .to_dict()
)

critical_diagnostic_rows = []

if critical_count > 0:
    warning_values = (
        cls_shap_values[
            CLASS_TO_INT[
                "Warning"
            ]
        ][
            actual_critical_mask
        ]
    )

    critical_values = (
        cls_shap_values[
            CLASS_TO_INT[
                "Critical"
            ]
        ][
            actual_critical_mask
        ]
    )

    for feature_index, feature in enumerate(
        transformed_feature_names
    ):
        critical_diagnostic_rows.append({
            "Transformed_Feature":
                feature,
            "Raw_Feature":
                transformed_to_raw_feature(
                    feature,
                    NUMERIC_FEATURES,
                ),
            "Actual_Critical_Rows":
                critical_count,
            "Mean_SHAP_for_Warning_Output":
                float(
                    np.mean(
                        warning_values[
                            :,
                            feature_index,
                        ]
                    )
                ),
            "Mean_Absolute_SHAP_for_Warning_Output":
                float(
                    np.mean(
                        np.abs(
                            warning_values[
                                :,
                                feature_index,
                            ]
                        )
                    )
                ),
            "Mean_SHAP_for_Critical_Output":
                float(
                    np.mean(
                        critical_values[
                            :,
                            feature_index,
                        ]
                    )
                ),
            "Mean_Absolute_SHAP_for_Critical_Output":
                float(
                    np.mean(
                        np.abs(
                            critical_values[
                                :,
                                feature_index,
                            ]
                        )
                    )
                ),
        })

critical_diagnostic_df = pd.DataFrame(
    critical_diagnostic_rows
)

if not critical_diagnostic_df.empty:
    critical_diagnostic_df[
        "Warning_Minus_Critical_Mean_SHAP"
    ] = (
        critical_diagnostic_df[
            "Mean_SHAP_for_Warning_Output"
        ]
        -
        critical_diagnostic_df[
            "Mean_SHAP_for_Critical_Output"
        ]
    )

    critical_diagnostic_df = (
        critical_diagnostic_df
        .sort_values(
            "Mean_Absolute_SHAP_for_Warning_Output",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

critical_diagnostic_df.to_csv(
    CRITICAL_DIAGNOSTIC_PATH,
    index=False,
)

critical_test_rows = (
    test_df.loc[
        actual_critical_mask
    ].copy()
)

critical_explanation_columns = [
    column
    for column in [
        "Timestamp",
        "Smokehouse_ID",
        "Batch_ID",
        "Layer_ID",
        "Temperature_C",
        "Humidity_%",
        "Elapsed_Drying_Hours",
        "Fire_Risk_Label",
    ]
    if column in (
        critical_test_rows.columns
    )
]

critical_rows_df = (
    critical_test_rows[
        critical_explanation_columns
    ].copy()
)

critical_rows_df[
    "Predicted_Fire_Risk_Label"
] = [
    INT_TO_CLASS[
        int(value)
    ]
    for value in (
        stage32_cls_pred[
            actual_critical_mask
        ]
    )
]

critical_rows_df.to_csv(
    CRITICAL_ROWS_PATH,
    index=False,
)

print(
    "\n--- ACTUAL CRITICAL FINAL-TEST DIAGNOSTIC ---"
)
print(
    "Actual Critical rows:",
    critical_count,
)
print(
    "Predicted labels for those rows:",
    critical_pred_counts,
)

if not critical_diagnostic_df.empty:
    print(
        "\nMost influential features for "
        "Warning output among actual Critical rows:"
    )

    print(
        critical_diagnostic_df[
            [
                "Transformed_Feature",
                "Mean_Absolute_SHAP_for_Warning_Output",
                "Mean_Absolute_SHAP_for_Critical_Output",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


# ============================================================
# 15. CHECKS
# ============================================================

checks = [
    {
        "Check":
            "Stage 31 methodological status PASS",
        "Status":
            "PASS",
    },
    {
        "Check":
            "Regression Stage-31 predictions reproduced",
        "Status":
            (
                "PASS"
                if reg_predictions_match
                else "FAIL"
            ),
    },
    {
        "Check":
            "Classification Stage-31 predictions reproduced",
        "Status":
            (
                "PASS"
                if cls_predictions_match
                else "FAIL"
            ),
    },
    {
        "Check":
            "No model tuning in Stage 32",
        "Status":
            "PASS",
    },
    {
        "Check":
            "No threshold adjustment in Stage 32",
        "Status":
            "PASS",
    },
    {
        "Check":
            "No resampling in Stage 32",
        "Status":
            "PASS",
    },
    {
        "Check":
            "Regression SHAP computed",
        "Status":
            (
                "PASS"
                if (
                    reg_shap_values.shape[
                        0
                    ]
                    == len(
                        test_df
                    )
                )
                else "FAIL"
            ),
    },
    {
        "Check":
            "Classification SHAP computed",
        "Status":
            (
                "PASS"
                if (
                    cls_shap_values.shape
                    == (
                        3,
                        len(test_df),
                        len(
                            transformed_feature_names
                        ),
                    )
                )
                else "FAIL"
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
        "Stage 32 checks contain FAIL."
    )


# ============================================================
# 16. SUMMARY
# ============================================================

summary = {
    "stage": 32,
    "purpose":
        "Post-hoc SHAP explainability of frozen Stage-31 models",
    "methodological_status":
        "PASS",
    "stage31_predictions_reproduced": {
        "regression":
            reg_predictions_match,
        "classification":
            cls_predictions_match,
    },
    "data_explained": {
        "source":
            "Locked Stage-31 Final Test",
        "rows":
            len(test_df),
        "batches":
            sorted(
                test_batches
            ),
        "actual_critical_rows":
            critical_count,
        "actual_critical_predicted_labels":
            critical_pred_counts,
    },
    "regression": {
        "model_family":
            selected_reg_family,
        "trial":
            selected_reg_trial,
        "top_transformed_features":
            reg_importance_df[
                [
                    "Transformed_Feature",
                    "Mean_Absolute_SHAP",
                ]
            ]
            .head(10)
            .to_dict(
                orient="records"
            ),
        "top_raw_features":
            reg_raw_importance_df
            .head(10)
            .to_dict(
                orient="records"
            ),
    },
    "classification": {
        "model_family":
            selected_cls_family,
        "trial":
            selected_cls_trial,
        "top_features_by_class": {
            class_name:
                cls_importance_df[
                    cls_importance_df[
                        "Class"
                    ]
                    == class_name
                ][
                    [
                        "Transformed_Feature",
                        "Mean_Absolute_SHAP",
                    ]
                ]
                .head(10)
                .to_dict(
                    orient="records"
                )
            for class_name in (
                CLASS_ORDER_TEXT
                    )
        },
    },
    "research_note":
        (
            "SHAP explanations are post-hoc model explanations. "
            "They do not establish causality and were not used "
            "to retune or reselect the models."
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
# 17. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 32: SHAP EXPLAINABILITY\n"
    )
    report.write(
        "=" * 78 + "\n\n"
    )

    report.write(
        "PURPOSE\n"
        "-------\n"
    )
    report.write(
        "Explain the frozen Stage-31 regression and "
        "classification model behavior without tuning, "
        "reselection, resampling or threshold adjustment.\n\n"
    )

    report.write(
        "MODEL REPRODUCTION\n"
        "------------------\n"
    )
    report.write(
        "Regression predictions reproduce Stage 31: "
        f"{reg_predictions_match}\n"
    )
    report.write(
        "Classification predictions reproduce Stage 31: "
        f"{cls_predictions_match}\n\n"
    )

    report.write(
        "REGRESSION GLOBAL SHAP IMPORTANCE\n"
        "---------------------------------\n"
    )
    report.write(
        reg_importance_df[
            [
                "Transformed_Feature",
                "Mean_Absolute_SHAP",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
        + "\n\n"
    )

    report.write(
        "CLASSIFICATION GLOBAL SHAP IMPORTANCE BY CLASS\n"
        "----------------------------------------------\n"
    )

    for class_name in (
        CLASS_ORDER_TEXT
    ):
        report.write(
            f"\n{class_name}\n"
        )
        report.write(
            "-" * len(
                class_name
            )
            + "\n"
        )

        report.write(
            cls_importance_df[
                cls_importance_df[
                    "Class"
                ]
                == class_name
            ][
                [
                    "Transformed_Feature",
                    "Mean_Absolute_SHAP",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
            + "\n"
        )

    report.write(
        "\nACTUAL CRITICAL TEST-ROW DIAGNOSTIC\n"
        "-----------------------------------\n"
    )

    report.write(
        f"Actual Critical rows: "
        f"{critical_count}\n"
    )

    report.write(
        "Predictions for actual Critical rows: "
        f"{critical_pred_counts}\n\n"
    )

    if not critical_diagnostic_df.empty:
        report.write(
            "Top SHAP features for Warning output "
            "on actual Critical rows:\n"
        )

        report.write(
            critical_diagnostic_df[
                [
                    "Transformed_Feature",
                    "Mean_Absolute_SHAP_for_Warning_Output",
                    "Mean_Absolute_SHAP_for_Critical_Output",
                ]
            ]
            .head(15)
            .to_string(
                index=False
            )
            + "\n\n"
        )

    report.write(
        "INTERPRETATION RULES\n"
        "--------------------\n"
    )
    report.write(
        "- Larger mean absolute SHAP means the model relied "
        "more strongly on that feature for the explained output.\n"
    )
    report.write(
        "- Positive/negative SHAP direction is model-output "
        "specific and should not be interpreted as causality.\n"
    )
    report.write(
        "- Classification SHAP is reported separately for "
        "Normal, Warning and Critical outputs.\n"
    )
    report.write(
        "- The actual-Critical diagnostic is post-hoc and "
        "must not be used to alter Stage-31 test results.\n\n"
    )

    report.write(
        "FINAL STATUS\n"
        "------------\n"
    )
    report.write(
        "STAGE 32 SHAP EXPLAINABILITY: PASS\n"
    )


# ============================================================
# 18. FINISH
# ============================================================

print("\n" + "=" * 78)
print(
    "STAGE 32 SHAP EXPLAINABILITY: PASS"
)
print("=" * 78)

print(
    "Regression and classification explanations "
    "were generated for the exact frozen "
    "Stage-31 model behavior."
)

print(
    "\nIMPORTANT:"
)
print(
    "SHAP explains model behavior; it does not "
    "prove causal relationships."
)
print(
    "No Stage-31 result was modified."
)

print("\nOutputs:")
for output_path in [
    REG_IMPORTANCE_PATH,
    REG_RAW_IMPORTANCE_PATH,
    CLS_IMPORTANCE_PATH,
    CLS_RAW_IMPORTANCE_PATH,
    CLS_OVERALL_IMPORTANCE_PATH,
    CRITICAL_DIAGNOSTIC_PATH,
    CRITICAL_ROWS_PATH,
    CHECKS_PATH,
    SUMMARY_PATH,
    REPORT_PATH,
    REG_SUMMARY_FIGURE,
    REG_BAR_FIGURE,
    CLS_NORMAL_FIGURE,
    CLS_WARNING_FIGURE,
    CLS_CRITICAL_FIGURE,
    CLS_OVERALL_BAR_FIGURE,
]:
    print(output_path)
