from pathlib import Path
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# COMPONENT 3 - STAGE 23: WEIGHTED RANDOM FOREST CLASSIFICATION
# ============================================================
#
# Target:
#   Fire_Risk_Label = Normal / Warning / Critical
#
# Purpose:
#   Evaluate a fixed, untuned nonlinear Random Forest classifier
#   using the Train-only imbalance strategy from Stage 21.
#
# Imbalance handling:
#   class_weight="balanced"
#
# Primary metric:
#   Macro-F1
#
# Important supporting metrics:
#   Balanced Accuracy
#   Warning Recall
#   Critical Recall
#   Per-class Precision / Recall / F1
#   Confusion Matrix
#
# Methodology:
#   - Fit on TRAIN only.
#   - Evaluate on VALIDATION only.
#   - No SMOTE or resampling.
#   - No hyperparameter tuning.
#   - No threshold tuning.
#   - Final TEST remains untouched until Stage 31.
#
# Balanced Accuracy implementation:
#   We compute the macro-average Recall across the fixed three
#   classes. This equals standard Balanced Accuracy when all three
#   classes are present and avoids subgroup warnings when a single
#   layer/batch contains no examples of one rare class.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_READY_DIR = ROOT / "data" / "model_ready"
SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "outputs" / "reports"

MODEL_DIR = (
    ROOT
    / "artifacts"
    / "models"
    / "classification"
    / "random_forest"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = MODEL_READY_DIR / "Component3_X_Train.csv"
X_VALIDATION_PATH = MODEL_READY_DIR / "Component3_X_Validation.csv"
Y_TRAIN_PATH = MODEL_READY_DIR / "Component3_y_classification_Train.csv"
Y_VALIDATION_PATH = MODEL_READY_DIR / \
    "Component3_y_classification_Validation.csv"
META_TRAIN_PATH = MODEL_READY_DIR / "Component3_metadata_Train.csv"
META_VALIDATION_PATH = MODEL_READY_DIR / "Component3_metadata_Validation.csv"

TEST_PATH = SPLIT_DIR / "Component3_Test.csv"

STAGE20_METRICS_PATH = (
    REPORT_DIR
    / "20_classification_baseline_metrics.csv"
)

STAGE21_WEIGHTS_PATH = (
    REPORT_DIR
    / "21_class_weights.csv"
)

STAGE21_STRATEGY_PATH = (
    REPORT_DIR
    / "21_class_imbalance_strategy.json"
)

STAGE22_METRICS_PATH = (
    REPORT_DIR
    / "22_logistic_regression_metrics.csv"
)

STAGE22_PER_CLASS_PATH = (
    REPORT_DIR
    / "22_logistic_regression_per_class_metrics.csv"
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "23_random_forest_classification_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "23_random_forest_classification_checks.csv"
)

METRICS_PATH = (
    REPORT_DIR
    / "23_random_forest_classification_metrics.csv"
)

PER_CLASS_PATH = (
    REPORT_DIR
    / "23_random_forest_classification_per_class_metrics.csv"
)

CONFUSION_PATH = (
    REPORT_DIR
    / "23_random_forest_classification_confusion_matrix.csv"
)

PREDICTIONS_PATH = (
    REPORT_DIR
    / "23_random_forest_classification_validation_predictions.csv"
)

LAYER_PATH = (
    REPORT_DIR
    / "23_random_forest_classification_by_layer.csv"
)

BATCH_PATH = (
    REPORT_DIR
    / "23_random_forest_classification_by_batch.csv"
)

IMPORTANCE_PATH = (
    REPORT_DIR
    / "23_random_forest_classification_feature_importance.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "23_random_forest_classification_summary.json"
)

MODEL_PATH = (
    MODEL_DIR
    / "random_forest_classifier.joblib"
)

MODEL_METADATA_PATH = (
    MODEL_DIR
    / "random_forest_metadata.json"
)


# ============================================================
# 3. EXPECTED STRUCTURE
# ============================================================

EXPECTED_TRAIN_ROWS = 16246
EXPECTED_VALIDATION_ROWS = 4159
EXPECTED_FEATURES = 20

CLASS_ORDER = [
    "Normal",
    "Warning",
    "Critical",
]

EXPECTED_CLASSES = set(CLASS_ORDER)

SMOKE_TEST = os.environ.get(
    "STAGE23_SMOKE_TEST",
    "0",
) == "1"


# ============================================================
# 4. START
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 23: WEIGHTED RANDOM FOREST CLASSIFICATION")
print("=" * 78)


# ============================================================
# 5. REQUIRED FILE CHECKS
# ============================================================

required_files = [
    X_TRAIN_PATH,
    X_VALIDATION_PATH,
    Y_TRAIN_PATH,
    Y_VALIDATION_PATH,
    META_TRAIN_PATH,
    META_VALIDATION_PATH,
    TEST_PATH,
    STAGE20_METRICS_PATH,
    STAGE21_WEIGHTS_PATH,
    STAGE21_STRATEGY_PATH,
    STAGE22_METRICS_PATH,
    STAGE22_PER_CLASS_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("\nTrain and Validation model-ready data found.")
print("Stage 21 imbalance strategy and Stage 22 Logistic evidence found.")
print("Final Test dataset existence verified, but contents were NOT loaded.")


# ============================================================
# 6. LOAD DEVELOPMENT DATA ONLY
# ============================================================

X_train = pd.read_csv(X_TRAIN_PATH)
X_validation = pd.read_csv(X_VALIDATION_PATH)

y_train_df = pd.read_csv(Y_TRAIN_PATH)
y_validation_df = pd.read_csv(Y_VALIDATION_PATH)

meta_train = pd.read_csv(META_TRAIN_PATH)
meta_validation = pd.read_csv(META_VALIDATION_PATH)

stage20_metrics = pd.read_csv(STAGE20_METRICS_PATH)
stage21_weights = pd.read_csv(STAGE21_WEIGHTS_PATH)
stage22_metrics = pd.read_csv(STAGE22_METRICS_PATH)
stage22_per_class = pd.read_csv(STAGE22_PER_CLASS_PATH)

with open(
    STAGE21_STRATEGY_PATH,
    "r",
    encoding="utf-8",
) as file:
    stage21_strategy = json.load(file)


# ============================================================
# 7. TARGET RESOLUTION
# ============================================================

def resolve_target_column(df):
    if "Fire_Risk_Label" in df.columns:
        return "Fire_Risk_Label"

    if len(df.columns) == 1:
        return df.columns[0]

    raise RuntimeError(
        "Could not determine Fire_Risk_Label target column."
    )


train_target_column = resolve_target_column(
    y_train_df
)

validation_target_column = resolve_target_column(
    y_validation_df
)

y_train = (
    y_train_df[
        train_target_column
    ]
    .astype(str)
    .to_numpy()
)

y_validation = (
    y_validation_df[
        validation_target_column
    ]
    .astype(str)
    .to_numpy()
)


# ============================================================
# 8. INPUT VALIDATION
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

if len(y_train) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError(
        "Unexpected y_train row count."
    )

if len(y_validation) != EXPECTED_VALIDATION_ROWS:
    raise RuntimeError(
        "Unexpected y_validation row count."
    )

if len(meta_train) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError(
        "Unexpected Train metadata row count."
    )

if len(meta_validation) != EXPECTED_VALIDATION_ROWS:
    raise RuntimeError(
        "Unexpected Validation metadata row count."
    )

if list(X_train.columns) != list(X_validation.columns):
    raise RuntimeError(
        "Train and Validation feature columns differ."
    )

if not np.isfinite(
    X_train.to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "X_train contains missing/non-finite values."
    )

if not np.isfinite(
    X_validation.to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "X_validation contains missing/non-finite values."
    )

train_classes = set(
    np.unique(y_train)
)

validation_classes = set(
    np.unique(y_validation)
)

if train_classes != EXPECTED_CLASSES:
    raise RuntimeError(
        "Unexpected Train classes."
    )

if validation_classes != EXPECTED_CLASSES:
    raise RuntimeError(
        "Unexpected Validation classes."
    )


# ============================================================
# 9. VERIFY STAGE-21 CLASS-WEIGHT STRATEGY
# ============================================================

expected_strategy = (
    "class_weight='balanced'"
)

actual_strategy = (
    stage21_strategy[
        "imbalance_strategy"
    ][
        "Random_Forest"
    ]
)

if actual_strategy != expected_strategy:
    raise RuntimeError(
        "Stage-21 Random Forest imbalance strategy mismatch.\n"
        f"Expected: {expected_strategy}\n"
        f"Found: {actual_strategy}"
    )

computed_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(
        CLASS_ORDER,
        dtype=object,
    ),
    y=y_train,
)

computed_weight_map = {
    class_name: float(weight)
    for class_name, weight in zip(
        CLASS_ORDER,
        computed_weights,
    )
}

reported_weight_map = {
    str(row["Class"]): float(
        row["Balanced_Class_Weight"]
    )
    for _, row in stage21_weights.iterrows()
}

weights_match_stage21 = bool(
    all(
        class_name in reported_weight_map
        and
        np.isclose(
            computed_weight_map[class_name],
            reported_weight_map[class_name],
            rtol=1e-10,
            atol=1e-10,
        )
        for class_name in CLASS_ORDER
    )
)

if not weights_match_stage21:
    raise RuntimeError(
        "Train-derived balanced class weights do not match Stage 21."
    )


# ============================================================
# 10. FIXED RANDOM FOREST CONFIGURATION
# ============================================================
#
# This stage is deliberately untuned.
#
# 500 trees:
#   stable ensemble estimate.
#
# max_features="sqrt":
#   standard classification RF behavior and tree diversity.
#
# class_weight="balanced":
#   exactly follows Stage 21.
#
# No pruning constraints are tuned here; Stage 28 handles tuning.

RF_PARAMS = {
    "n_estimators": (
        50
        if SMOKE_TEST
        else 500
    ),
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

model = RandomForestClassifier(
    **RF_PARAMS
)


# ============================================================
# 11. FIT TRAIN ONLY
# ============================================================

print("\nTraining weighted Random Forest on TRAIN only...")

model.fit(
    X_train,
    y_train,
)

print("Random Forest training complete.")


# ============================================================
# 12. PREDICTIONS
# ============================================================

train_pred = model.predict(
    X_train
)

validation_pred = model.predict(
    X_validation
)

validation_proba = model.predict_proba(
    X_validation
)


# ============================================================
# 13. METRIC HELPERS
# ============================================================

def fixed_balanced_accuracy(
    y_true,
    y_pred,
):
    return float(
        recall_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        )
    )


def overall_metrics(
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
        "Balanced_Accuracy": (
            fixed_balanced_accuracy(
                y_true,
                y_pred,
            )
        ),
        "Macro_Precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "Macro_Recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "Macro_F1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "Weighted_F1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASS_ORDER,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def per_class_rows(
    model_name,
    split_name,
    y_true,
    y_pred,
):
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for class_name in CLASS_ORDER:
        result = report[
            class_name
        ]

        rows.append({
            "Model": model_name,
            "Split": split_name,
            "Class": class_name,
            "Precision": float(
                result[
                    "precision"
                ]
            ),
            "Recall": float(
                result[
                    "recall"
                ]
            ),
            "F1": float(
                result[
                    "f1-score"
                ]
            ),
            "Support": int(
                result[
                    "support"
                ]
            ),
        })

    return rows


def confusion_rows(
    model_name,
    split_name,
    y_true,
    y_pred,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
    )

    rows = []

    for i, actual_class in enumerate(
        CLASS_ORDER
    ):
        for j, predicted_class in enumerate(
            CLASS_ORDER
        ):
            rows.append({
                "Model": model_name,
                "Split": split_name,
                "Actual_Class": actual_class,
                "Predicted_Class": predicted_class,
                "Count": int(
                    matrix[
                        i,
                        j
                    ]
                ),
            })

    return rows


# ============================================================
# 14. OVERALL + PER-CLASS METRICS
# ============================================================

MODEL_NAME = "Random_Forest"

metric_rows = []
class_rows = []
conf_rows = []

for split_name, y_true, y_pred in [
    (
        "Train",
        y_train,
        train_pred,
    ),
    (
        "Validation",
        y_validation,
        validation_pred,
    ),
]:
    metrics = overall_metrics(
        y_true,
        y_pred,
    )

    metric_rows.append({
        "Model": MODEL_NAME,
        "Split": split_name,
        **metrics,
    })

    class_rows.extend(
        per_class_rows(
            MODEL_NAME,
            split_name,
            y_true,
            y_pred,
        )
    )

    conf_rows.extend(
        confusion_rows(
            MODEL_NAME,
            split_name,
            y_true,
            y_pred,
        )
    )


metrics_df = pd.DataFrame(
    metric_rows
)

per_class_df = pd.DataFrame(
    class_rows
)

confusion_df = pd.DataFrame(
    conf_rows
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)

per_class_df.to_csv(
    PER_CLASS_PATH,
    index=False,
)

confusion_df.to_csv(
    CONFUSION_PATH,
    index=False,
)


# ============================================================
# 15. VALIDATION PREDICTION OUTPUT
# ============================================================

prediction_df = meta_validation.copy()

prediction_df[
    "Actual_Fire_Risk_Label"
] = y_validation

prediction_df[
    "Predicted_Fire_Risk_Label"
] = validation_pred

for class_index, class_name in enumerate(
    model.classes_
):
    prediction_df[
        f"Probability_{class_name}"
    ] = validation_proba[
        :,
        class_index
    ]

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# 16. LAYER-WISE VALIDATION METRICS
# ============================================================

layer_temp = pd.DataFrame({
    "Layer_ID": meta_validation[
        "Layer_ID"
    ],
    "Actual": y_validation,
    "Predicted": validation_pred,
})

layer_rows = []

for layer_id, group in layer_temp.groupby(
    "Layer_ID",
    sort=True,
):
    metrics = overall_metrics(
        group[
            "Actual"
        ].to_numpy(),
        group[
            "Predicted"
        ].to_numpy(),
    )

    layer_rows.append({
        "Layer_ID": layer_id,
        "Rows": len(group),
        **metrics,
    })

layer_df = pd.DataFrame(
    layer_rows
)

layer_df.to_csv(
    LAYER_PATH,
    index=False,
)


# ============================================================
# 17. BATCH-WISE VALIDATION METRICS
# ============================================================

batch_temp = pd.DataFrame({
    "Smokehouse_ID": meta_validation[
        "Smokehouse_ID"
    ],
    "Batch_ID": meta_validation[
        "Batch_ID"
    ],
    "Actual": y_validation,
    "Predicted": validation_pred,
})

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
    metrics = overall_metrics(
        group[
            "Actual"
        ].to_numpy(),
        group[
            "Predicted"
        ].to_numpy(),
    )

    batch_rows.append({
        "Smokehouse_ID": smokehouse_id,
        "Batch_ID": batch_id,
        "Rows": len(group),
        **metrics,
    })

batch_df = pd.DataFrame(
    batch_rows
)

batch_df.to_csv(
    BATCH_PATH,
    index=False,
)


# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_,
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

importance_df.insert(
    0,
    "Rank",
    np.arange(
        1,
        len(
            importance_df
        )
        + 1,
    ),
)

importance_df.to_csv(
    IMPORTANCE_PATH,
    index=False,
)


# ============================================================
# 19. EXTRACT VALIDATION METRICS
# ============================================================

train_metrics = (
    metrics_df[
        metrics_df[
            "Split"
        ]
        ==
        "Train"
    ]
    .iloc[
        0
    ]
)

validation_metrics = (
    metrics_df[
        metrics_df[
            "Split"
        ]
        ==
        "Validation"
    ]
    .iloc[
        0
    ]
)


def validation_class_metric(
    class_name,
    metric_name,
):
    row = per_class_df[
        (
            per_class_df[
                "Split"
            ]
            ==
            "Validation"
        )
        &
        (
            per_class_df[
                "Class"
            ]
            ==
            class_name
        )
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Could not locate Validation {class_name} metric."
        )

    return float(
        row.iloc[
            0
        ][
            metric_name
        ]
    )


warning_recall = validation_class_metric(
    "Warning",
    "Recall",
)

critical_recall = validation_class_metric(
    "Critical",
    "Recall",
)

warning_f1 = validation_class_metric(
    "Warning",
    "F1",
)

critical_f1 = validation_class_metric(
    "Critical",
    "F1",
)


# ============================================================
# 20. PREVIOUS-MODEL COMPARISON
# ============================================================

def get_validation_metric_row(
    df,
    model_name,
):
    row = df[
        (
            df[
                "Model"
            ]
            ==
            model_name
        )
        &
        (
            df[
                "Split"
            ]
            ==
            "Validation"
        )
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Could not uniquely locate {model_name} Validation metrics."
        )

    return row.iloc[
        0
    ]


most_frequent = get_validation_metric_row(
    stage20_metrics,
    "Dummy_Most_Frequent",
)

stratified = get_validation_metric_row(
    stage20_metrics,
    "Dummy_Stratified",
)

logistic = get_validation_metric_row(
    stage22_metrics,
    "Logistic_Regression",
)


def previous_class_metric(
    df,
    model_name,
    class_name,
    metric_name,
):
    row = df[
        (
            df[
                "Model"
            ]
            ==
            model_name
        )
        &
        (
            df[
                "Split"
            ]
            ==
            "Validation"
        )
        &
        (
            df[
                "Class"
            ]
            ==
            class_name
        )
    ]

    if len(row) != 1:
        raise RuntimeError(
            "Could not uniquely locate prior per-class metric."
        )

    return float(
        row.iloc[
            0
        ][
            metric_name
        ]
    )


logistic_warning_recall = previous_class_metric(
    stage22_per_class,
    "Logistic_Regression",
    "Warning",
    "Recall",
)

logistic_critical_recall = previous_class_metric(
    stage22_per_class,
    "Logistic_Regression",
    "Critical",
    "Recall",
)

macro_f1_change_vs_logistic = (
    float(
        validation_metrics[
            "Macro_F1"
        ]
    )
    -
    float(
        logistic[
            "Macro_F1"
        ]
    )
)

balanced_accuracy_change_vs_logistic = (
    float(
        validation_metrics[
            "Balanced_Accuracy"
        ]
    )
    -
    float(
        logistic[
            "Balanced_Accuracy"
        ]
    )
)

warning_recall_change_vs_logistic = (
    warning_recall
    -
    logistic_warning_recall
)

critical_recall_change_vs_logistic = (
    critical_recall
    -
    logistic_critical_recall
)


# ============================================================
# 21. GENERALIZATION GAP
# ============================================================

macro_f1_gap = (
    float(
        train_metrics[
            "Macro_F1"
        ]
    )
    -
    float(
        validation_metrics[
            "Macro_F1"
        ]
    )
)

balanced_accuracy_gap = (
    float(
        train_metrics[
            "Balanced_Accuracy"
        ]
    )
    -
    float(
        validation_metrics[
            "Balanced_Accuracy"
        ]
    )
)


# ============================================================
# 22. SAVE MODEL + HASHES
# ============================================================

joblib.dump(
    model,
    MODEL_PATH,
)


def sha256_file(path):
    with open(
        path,
        "rb",
    ) as file:
        return hashlib.sha256(
            file.read()
        ).hexdigest()


model_sha256 = sha256_file(
    MODEL_PATH
)

x_train_sha256 = sha256_file(
    X_TRAIN_PATH
)

x_validation_sha256 = sha256_file(
    X_VALIDATION_PATH
)


# ============================================================
# 23. CHECKS
# ============================================================

all_probabilities_valid = bool(
    np.isfinite(
        validation_proba
    ).all()
    and
    (
        validation_proba
        >=
        0
    ).all()
    and
    (
        validation_proba
        <=
        1
    ).all()
    and
    np.allclose(
        validation_proba.sum(
            axis=1
        ),
        1.0,
        rtol=1e-8,
        atol=1e-8,
    )
)

all_metrics_finite = bool(
    np.isfinite(
        metrics_df[
            [
                "Accuracy",
                "Balanced_Accuracy",
                "Macro_Precision",
                "Macro_Recall",
                "Macro_F1",
                "Weighted_F1",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all()
)

importance_sum_valid = bool(
    np.isclose(
        importance_df[
            "Importance"
        ].sum(),
        1.0,
        rtol=1e-8,
        atol=1e-8,
    )
)

predicted_classes_validation = sorted(
    np.unique(
        validation_pred
    ).tolist()
)

checks = pd.DataFrame(
    [
        {
            "Check": (
                "Expected Train/Validation shapes"
            ),
            "Status": "PASS",
            "Details": (
                f"Train={X_train.shape}; "
                f"Validation={X_validation.shape}"
            ),
        },
        {
            "Check": (
                "Expected three target classes"
            ),
            "Status": (
                "PASS"
                if (
                    train_classes
                    ==
                    EXPECTED_CLASSES
                    and
                    validation_classes
                    ==
                    EXPECTED_CLASSES
                )
                else "FAIL"
            ),
            "Details": (
                ", ".join(CLASS_ORDER)
            ),
        },
        {
            "Check": (
                "Stage-21 Random Forest strategy verified"
            ),
            "Status": (
                "PASS"
                if actual_strategy
                ==
                expected_strategy
                else "FAIL"
            ),
            "Details": actual_strategy,
        },
        {
            "Check": (
                "Balanced weights reproduce Stage 21"
            ),
            "Status": (
                "PASS"
                if weights_match_stage21
                else "FAIL"
            ),
            "Details": str(
                computed_weight_map
            ),
        },
        {
            "Check": (
                "Model fitted on Train only"
            ),
            "Status": "PASS",
            "Details": (
                "Validation used for evaluation only."
            ),
        },
        {
            "Check": (
                "No resampling or SMOTE"
            ),
            "Status": "PASS",
            "Details": (
                "class_weight='balanced' only."
            ),
        },
        {
            "Check": (
                "No hyperparameter tuning"
            ),
            "Status": "PASS",
            "Details": (
                "One fixed Random Forest configuration."
            ),
        },
        {
            "Check": (
                "All probabilities valid"
            ),
            "Status": (
                "PASS"
                if all_probabilities_valid
                else "FAIL"
            ),
            "Details": (
                "Probabilities finite, within [0,1], sum to 1."
            ),
        },
        {
            "Check": (
                "All overall metrics finite"
            ),
            "Status": (
                "PASS"
                if all_metrics_finite
                else "FAIL"
            ),
            "Details": (
                "Train and Validation metrics verified."
            ),
        },
        {
            "Check": (
                "Feature importances sum to 1"
            ),
            "Status": (
                "PASS"
                if importance_sum_valid
                else "FAIL"
            ),
            "Details": (
                f"Sum={importance_df['Importance'].sum():.12f}"
            ),
        },
        {
            "Check": (
                "Validation predicted-class coverage recorded"
            ),
            "Status": "PASS",
            "Details": (
                ", ".join(
                    predicted_classes_validation
                )
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
# 24. METADATA / SUMMARY
# ============================================================

metadata = {
    "stage": (
        "23_weighted_random_forest_classification"
    ),
    "model": (
        "RandomForestClassifier"
    ),
    "parameters": (
        RF_PARAMS
    ),
    "fit_scope": (
        "TRAIN_ONLY"
    ),
    "validation_scope": (
        "DEVELOPMENT_EVALUATION_ONLY"
    ),
    "imbalance_strategy": (
        "class_weight='balanced'"
    ),
    "train_only_balanced_weights": (
        computed_weight_map
    ),
    "smote_used": False,
    "hyperparameter_tuning_performed": False,
    "threshold_tuning_performed": False,
    "validation": {
        "Accuracy": float(
            validation_metrics[
                "Accuracy"
            ]
        ),
        "Balanced_Accuracy": float(
            validation_metrics[
                "Balanced_Accuracy"
            ]
        ),
        "Macro_F1": float(
            validation_metrics[
                "Macro_F1"
            ]
        ),
        "Warning_Recall": (
            warning_recall
        ),
        "Critical_Recall": (
            critical_recall
        ),
        "Warning_F1": (
            warning_f1
        ),
        "Critical_F1": (
            critical_f1
        ),
    },
    "comparison_vs_logistic": {
        "Macro_F1_change": (
            macro_f1_change_vs_logistic
        ),
        "Balanced_Accuracy_change": (
            balanced_accuracy_change_vs_logistic
        ),
        "Warning_Recall_change": (
            warning_recall_change_vs_logistic
        ),
        "Critical_Recall_change": (
            critical_recall_change_vs_logistic
        ),
    },
    "generalization": {
        "Macro_F1_Train_minus_Validation": (
            macro_f1_gap
        ),
        "Balanced_Accuracy_Train_minus_Validation": (
            balanced_accuracy_gap
        ),
    },
    "validation_predicted_classes": (
        predicted_classes_validation
    ),
    "x_train_sha256": (
        x_train_sha256
    ),
    "x_validation_sha256": (
        x_validation_sha256
    ),
    "model_sha256": (
        model_sha256
    ),
    "test_loaded": False,
}

with open(
    MODEL_METADATA_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        indent=2,
    )

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        indent=2,
    )


# ============================================================
# 25. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- FIXED RANDOM FOREST CONFIGURATION ---"
)

for key, value in RF_PARAMS.items():
    print(
        f"{key:<28}: {value}"
    )


print(
    "\n--- TRAIN-ONLY BALANCED CLASS WEIGHTS ---"
)

for class_name in CLASS_ORDER:
    print(
        f"{class_name:<10}: "
        f"{computed_weight_map[class_name]:.6f}"
    )


print(
    "\n--- RANDOM FOREST RESULTS ---"
)

print(
    metrics_df
    .round(4)
    .to_string(index=False)
)


print(
    "\n--- VALIDATION PER-CLASS RESULTS ---"
)

print(
    per_class_df[
        per_class_df[
            "Split"
        ]
        ==
        "Validation"
    ][
        [
            "Class",
            "Precision",
            "Recall",
            "F1",
            "Support",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


print(
    "\n--- COMPARISON WITH LOGISTIC REGRESSION ---"
)

print(
    f"Logistic Validation Macro-F1   : "
    f"{float(logistic['Macro_F1']):.4f}"
)

print(
    f"Random Forest Validation Macro-F1: "
    f"{float(validation_metrics['Macro_F1']):.4f}"
)

print(
    f"Macro-F1 change                : "
    f"{macro_f1_change_vs_logistic:+.4f}"
)

print(
    f"Balanced Accuracy change       : "
    f"{balanced_accuracy_change_vs_logistic:+.4f}"
)

print(
    f"Warning Recall change          : "
    f"{warning_recall_change_vs_logistic:+.4f}"
)

print(
    f"Critical Recall change         : "
    f"{critical_recall_change_vs_logistic:+.4f}"
)


print(
    "\n--- SAFETY-SENSITIVE VALIDATION METRICS ---"
)

print(
    f"Warning recall                  : "
    f"{warning_recall:.4f}"
)

print(
    f"Critical recall                 : "
    f"{critical_recall:.4f}"
)

print(
    f"Warning F1                      : "
    f"{warning_f1:.4f}"
)

print(
    f"Critical F1                     : "
    f"{critical_f1:.4f}"
)


print(
    "\n--- GENERALIZATION GAP ---"
)

print(
    f"Train Macro-F1 - Val Macro-F1  : "
    f"{macro_f1_gap:+.4f}"
)

print(
    f"Train BalAcc - Val BalAcc       : "
    f"{balanced_accuracy_gap:+.4f}"
)


print(
    "\n--- TOP 10 FEATURE IMPORTANCES ---"
)

print(
    importance_df
    .head(10)
    .round(6)
    .to_string(index=False)
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
# 26. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 23: "
        "WEIGHTED RANDOM FOREST CLASSIFICATION\n"
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
        "Stage 23 evaluates a fixed, untuned nonlinear Random "
        "Forest classifier for Fire_Risk_Label. The model follows "
        "the Stage-21 Train-only class-weighting strategy and is "
        "fitted on Train only. Validation is used only for "
        "development evaluation.\n\n"
    )

    report.write(
        "FIXED CONFIGURATION\n"
        "-------------------\n"
    )

    for key, value in RF_PARAMS.items():
        report.write(
            f"{key}: {value}\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "TRAIN-ONLY CLASS WEIGHTS\n"
        "------------------------\n"
    )

    for class_name in CLASS_ORDER:
        report.write(
            f"{class_name}: "
            f"{computed_weight_map[class_name]:.8f}\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "OVERALL RESULTS\n"
        "---------------\n"
    )

    report.write(
        metrics_df
        .round(6)
        .to_string(index=False)
    )

    report.write(
        "\n\n"
    )

    report.write(
        "VALIDATION PER-CLASS RESULTS\n"
        "----------------------------\n"
    )

    report.write(
        per_class_df[
            per_class_df[
                "Split"
            ]
            ==
            "Validation"
        ]
        .round(6)
        .to_string(index=False)
    )

    report.write(
        "\n\n"
    )

    report.write(
        "COMPARISON WITH LOGISTIC REGRESSION\n"
        "-----------------------------------\n"
    )

    report.write(
        f"Logistic Validation Macro-F1: "
        f"{float(logistic['Macro_F1']):.6f}\n"
    )

    report.write(
        f"Random Forest Validation Macro-F1: "
        f"{float(validation_metrics['Macro_F1']):.6f}\n"
    )

    report.write(
        f"Macro-F1 change: "
        f"{macro_f1_change_vs_logistic:+.6f}\n"
    )

    report.write(
        f"Balanced Accuracy change: "
        f"{balanced_accuracy_change_vs_logistic:+.6f}\n"
    )

    report.write(
        f"Warning Recall change: "
        f"{warning_recall_change_vs_logistic:+.6f}\n"
    )

    report.write(
        f"Critical Recall change: "
        f"{critical_recall_change_vs_logistic:+.6f}\n\n"
    )

    report.write(
        "SAFETY-SENSITIVE VALIDATION PERFORMANCE\n"
        "---------------------------------------\n"
    )

    report.write(
        f"Warning Recall: "
        f"{warning_recall:.6f}\n"
    )

    report.write(
        f"Critical Recall: "
        f"{critical_recall:.6f}\n"
    )

    report.write(
        f"Warning F1: "
        f"{warning_f1:.6f}\n"
    )

    report.write(
        f"Critical F1: "
        f"{critical_f1:.6f}\n\n"
    )

    report.write(
        "GENERALIZATION\n"
        "--------------\n"
    )

    report.write(
        f"Train Macro-F1 - Validation Macro-F1: "
        f"{macro_f1_gap:+.6f}\n"
    )

    report.write(
        f"Train Balanced Accuracy - Validation Balanced Accuracy: "
        f"{balanced_accuracy_gap:+.6f}\n\n"
    )

    report.write(
        "FEATURE IMPORTANCE NOTE\n"
        "-----------------------\n"
    )

    report.write(
        "Random Forest impurity-based feature importance is exported "
        "for descriptive model inspection only. It must not be used "
        "to reopen Stage-08 feature selection. Final explainability "
        "is reserved for Stage 32.\n\n"
    )

    report.write(
        "METHODOLOGY PROTECTION\n"
        "----------------------\n"
    )

    report.write(
        "- Model fitted on Train only.\n"
        "- class_weight='balanced' reproduced from Stage 21.\n"
        "- No SMOTE or resampling used.\n"
        "- No hyperparameter tuning performed.\n"
        "- No threshold tuning performed.\n"
        "- Validation used only for development evaluation.\n"
        "- Final Test rows were not loaded.\n"
        "- Final Test remains frozen until Stage 31.\n\n"
    )

    report.write(
        "NEXT STAGE\n"
        "----------\n"
    )

    report.write(
        "Stage 24 will evaluate weighted Extra Trees classification "
        "using the same frozen features, Train/Validation split, and "
        "Train-only imbalance strategy.\n\n"
    )

    report.write(
        "PASS / FAIL CHECKS\n"
        "------------------\n"
    )

    report.write(
        checks.to_string(index=False)
    )

    report.write(
        "\n\n"
    )

    report.write(
        f"STAGE 23 STATUS: "
        f"{overall_status}\n"
    )


# ============================================================
# 27. COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    f"STAGE 23 WEIGHTED RANDOM FOREST CLASSIFICATION: "
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
        "Stage 23 failed. Review "
        "23_random_forest_classification_report.txt."
    )
