from pathlib import Path
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# COMPONENT 3 - STAGE 25:
# WEIGHTED GRADIENT BOOSTING CLASSIFICATION
# ============================================================
#
# Target:
#   Fire_Risk_Label = Normal / Warning / Critical
#
# Purpose:
#   Evaluate a fixed, untuned Gradient Boosting classifier using
#   the TRAIN-derived sample weights created in Stage 21.
#
# Imbalance handling:
#   model.fit(..., sample_weight=train_sample_weights)
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
#   - Use Stage-21 TRAIN-only sample weights.
#   - No SMOTE or resampling.
#   - No hyperparameter tuning.
#   - No threshold tuning.
#   - Final TEST remains untouched until Stage 31.
#
# Balanced Accuracy:
#   Computed as macro recall across the fixed three classes.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_READY_DIR = ROOT / "data" / "model_ready"
SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "outputs" / "reports"

STAGE21_ARTIFACT_DIR = (
    ROOT
    / "artifacts"
    / "classification"
)

MODEL_DIR = (
    ROOT
    / "artifacts"
    / "models"
    / "classification"
    / "gradient_boosting"
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
    / "Component3_y_classification_Train.csv"
)

Y_VALIDATION_PATH = (
    MODEL_READY_DIR
    / "Component3_y_classification_Validation.csv"
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

STAGE21_WEIGHTS_PATH = (
    REPORT_DIR
    / "21_class_weights.csv"
)

STAGE21_STRATEGY_PATH = (
    REPORT_DIR
    / "21_class_imbalance_strategy.json"
)

STAGE21_SAMPLE_WEIGHTS_PATH = (
    STAGE21_ARTIFACT_DIR
    / "train_classification_sample_weights.csv"
)

STAGE22_METRICS_PATH = (
    REPORT_DIR
    / "22_logistic_regression_metrics.csv"
)

STAGE22_PER_CLASS_PATH = (
    REPORT_DIR
    / "22_logistic_regression_per_class_metrics.csv"
)

STAGE23_METRICS_PATH = (
    REPORT_DIR
    / "23_random_forest_classification_metrics.csv"
)

STAGE23_PER_CLASS_PATH = (
    REPORT_DIR
    / "23_random_forest_classification_per_class_metrics.csv"
)

STAGE24_METRICS_PATH = (
    REPORT_DIR
    / "24_extra_trees_classification_metrics.csv"
)

STAGE24_PER_CLASS_PATH = (
    REPORT_DIR
    / "24_extra_trees_classification_per_class_metrics.csv"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "25_gradient_boosting_classification_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "25_gradient_boosting_classification_checks.csv"
)

METRICS_PATH = (
    REPORT_DIR
    / "25_gradient_boosting_classification_metrics.csv"
)

PER_CLASS_PATH = (
    REPORT_DIR
    / "25_gradient_boosting_classification_per_class_metrics.csv"
)

CONFUSION_PATH = (
    REPORT_DIR
    / "25_gradient_boosting_classification_confusion_matrix.csv"
)

PREDICTIONS_PATH = (
    REPORT_DIR
    / "25_gradient_boosting_classification_validation_predictions.csv"
)

LAYER_PATH = (
    REPORT_DIR
    / "25_gradient_boosting_classification_by_layer.csv"
)

BATCH_PATH = (
    REPORT_DIR
    / "25_gradient_boosting_classification_by_batch.csv"
)

IMPORTANCE_PATH = (
    REPORT_DIR
    / "25_gradient_boosting_classification_feature_importance.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "25_gradient_boosting_classification_summary.json"
)

MODEL_PATH = (
    MODEL_DIR
    / "gradient_boosting_classifier.joblib"
)

MODEL_METADATA_PATH = (
    MODEL_DIR
    / "gradient_boosting_metadata.json"
)


# ============================================================
# 4. EXPECTED STRUCTURE
# ============================================================

EXPECTED_TRAIN_ROWS = 16246
EXPECTED_VALIDATION_ROWS = 4159
EXPECTED_FEATURES = 20

CLASS_ORDER = [
    "Normal",
    "Warning",
    "Critical",
]

EXPECTED_CLASSES = set(
    CLASS_ORDER
)

SMOKE_TEST = (
    os.environ.get(
        "STAGE25_SMOKE_TEST",
        "0",
    )
    ==
    "1"
)


# ============================================================
# 5. START
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 25: WEIGHTED GRADIENT BOOSTING CLASSIFICATION")
print("=" * 78)


# ============================================================
# 6. REQUIRED FILE CHECKS
# ============================================================

required_files = [
    X_TRAIN_PATH,
    X_VALIDATION_PATH,
    Y_TRAIN_PATH,
    Y_VALIDATION_PATH,
    META_TRAIN_PATH,
    META_VALIDATION_PATH,
    TEST_PATH,
    STAGE21_WEIGHTS_PATH,
    STAGE21_STRATEGY_PATH,
    STAGE21_SAMPLE_WEIGHTS_PATH,
    STAGE22_METRICS_PATH,
    STAGE22_PER_CLASS_PATH,
    STAGE23_METRICS_PATH,
    STAGE23_PER_CLASS_PATH,
    STAGE24_METRICS_PATH,
    STAGE24_PER_CLASS_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("\nTrain and Validation model-ready data found.")
print("Stage 21 sample-weight artifact and prior classifier evidence found.")
print("Final Test dataset existence verified, but contents were NOT loaded.")


# ============================================================
# 7. LOAD DEVELOPMENT DATA ONLY
# ============================================================

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

stage21_weights = pd.read_csv(
    STAGE21_WEIGHTS_PATH
)

stage21_sample_weights = pd.read_csv(
    STAGE21_SAMPLE_WEIGHTS_PATH
)

stage22_metrics = pd.read_csv(
    STAGE22_METRICS_PATH
)

stage22_per_class = pd.read_csv(
    STAGE22_PER_CLASS_PATH
)

stage23_metrics = pd.read_csv(
    STAGE23_METRICS_PATH
)

stage23_per_class = pd.read_csv(
    STAGE23_PER_CLASS_PATH
)

stage24_metrics = pd.read_csv(
    STAGE24_METRICS_PATH
)

stage24_per_class = pd.read_csv(
    STAGE24_PER_CLASS_PATH
)

with open(
    STAGE21_STRATEGY_PATH,
    "r",
    encoding="utf-8",
) as file:
    stage21_strategy = json.load(
        file
    )


# ============================================================
# 8. TARGET RESOLUTION
# ============================================================

def resolve_target_column(
    df,
):
    if "Fire_Risk_Label" in df.columns:
        return "Fire_Risk_Label"

    if len(
        df.columns
    ) == 1:
        return df.columns[
            0
        ]

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
    .astype(
        str
    )
    .to_numpy()
)

y_validation = (
    y_validation_df[
        validation_target_column
    ]
    .astype(
        str
    )
    .to_numpy()
)


# ============================================================
# 9. INPUT VALIDATION
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

train_classes = set(
    np.unique(
        y_train
    )
)

validation_classes = set(
    np.unique(
        y_validation
    )
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
# 10. VERIFY STAGE-21 STRATEGY
# ============================================================

expected_strategy = (
    "fit with TRAIN-derived sample_weight"
)

actual_strategy = (
    stage21_strategy[
        "imbalance_strategy"
    ][
        "Gradient_Boosting"
    ]
)

if actual_strategy != expected_strategy:
    raise RuntimeError(
        "Stage-21 Gradient Boosting imbalance strategy mismatch.\n"
        f"Expected: {expected_strategy}\n"
        f"Found: {actual_strategy}"
    )


# ============================================================
# 11. VERIFY STAGE-21 SAMPLE WEIGHTS
# ============================================================

required_sample_weight_columns = {
    "Train_Row_Index",
    "Fire_Risk_Label",
    "Sample_Weight",
}

missing_weight_columns = (
    required_sample_weight_columns
    -
    set(
        stage21_sample_weights.columns
    )
)

if missing_weight_columns:
    raise RuntimeError(
        "Stage-21 sample-weight file missing columns: "
        f"{sorted(missing_weight_columns)}"
    )

if len(
    stage21_sample_weights
) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError(
        "Stage-21 sample-weight row count does not match Train."
    )

expected_indices = np.arange(
    EXPECTED_TRAIN_ROWS
)

actual_indices = (
    stage21_sample_weights[
        "Train_Row_Index"
    ]
    .to_numpy(
        dtype=int
    )
)

sample_weight_indices_match = bool(
    np.array_equal(
        expected_indices,
        actual_indices,
    )
)

if not sample_weight_indices_match:
    raise RuntimeError(
        "Stage-21 sample-weight row order/index does not match Train."
    )

sample_weight_labels = (
    stage21_sample_weights[
        "Fire_Risk_Label"
    ]
    .astype(
        str
    )
    .to_numpy()
)

sample_weight_labels_match = bool(
    np.array_equal(
        sample_weight_labels,
        y_train,
    )
)

if not sample_weight_labels_match:
    raise RuntimeError(
        "Stage-21 sample-weight labels do not align with y_train."
    )

train_sample_weights = (
    stage21_sample_weights[
        "Sample_Weight"
    ]
    .to_numpy(
        dtype=float
    )
)

if not np.isfinite(
    train_sample_weights
).all():
    raise RuntimeError(
        "Stage-21 sample weights contain non-finite values."
    )

if not (
    train_sample_weights
    >
    0
).all():
    raise RuntimeError(
        "Stage-21 sample weights must all be positive."
    )


# ============================================================
# 12. VERIFY SAMPLE WEIGHT VALUES AGAINST STAGE-21 CLASS WEIGHTS
# ============================================================

reported_weight_map = {
    str(
        row[
            "Class"
        ]
    ): float(
        row[
            "Balanced_Class_Weight"
        ]
    )
    for _, row in stage21_weights.iterrows()
}

reconstructed_sample_weights = np.array(
    [
        reported_weight_map[
            label
        ]
        for label in y_train
    ],
    dtype=float,
)

sample_weight_values_match = bool(
    np.allclose(
        train_sample_weights,
        reconstructed_sample_weights,
        rtol=1e-12,
        atol=1e-12,
    )
)

if not sample_weight_values_match:
    raise RuntimeError(
        "Stage-21 sample-weight values do not match reported class weights."
    )


# ============================================================
# 13. FIXED GRADIENT BOOSTING CONFIGURATION
# ============================================================
#
# Stage 25 is deliberately untuned.
#
# This configuration is a moderate-capacity starting point.
# Stage 28 will handle bounded tuning only for shortlisted models.
#
# Note:
#   GradientBoostingClassifier does not expose class_weight.
#   Therefore Stage-21 TRAIN sample weights are supplied to fit().

GB_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "n_estimators": (
        30
        if SMOKE_TEST
        else 200
    ),
    "subsample": 1.0,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_depth": 3,
    "max_features": None,
    "random_state": 42,
}

model = GradientBoostingClassifier(
    **GB_PARAMS
)


# ============================================================
# 14. FIT TRAIN ONLY WITH TRAIN-DERIVED SAMPLE WEIGHTS
# ============================================================

print(
    "\nTraining weighted Gradient Boosting on TRAIN only..."
)

model.fit(
    X_train,
    y_train,
    sample_weight=train_sample_weights,
)

print(
    "Gradient Boosting training complete."
)


# ============================================================
# 15. PREDICTIONS
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
# 16. METRIC HELPERS
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
# 17. OVERALL + PER-CLASS METRICS
# ============================================================

MODEL_NAME = (
    "Gradient_Boosting"
)

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
# 18. VALIDATION PREDICTIONS
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
# 19. LAYER-WISE VALIDATION METRICS
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
        "Rows": len(
            group
        ),
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
# 20. BATCH-WISE VALIDATION METRICS
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
        "Rows": len(
            group
        ),
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
# 21. FEATURE IMPORTANCE
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
        +
        1,
    ),
)

importance_df.to_csv(
    IMPORTANCE_PATH,
    index=False,
)


# ============================================================
# 22. EXTRACT CURRENT RESULTS
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

    if len(
        row
    ) != 1:
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
# 23. PRIOR MODEL HELPERS
# ============================================================

def validation_metric_row(
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

    if len(
        row
    ) != 1:
        raise RuntimeError(
            f"Could not uniquely locate {model_name} Validation metrics."
        )

    return row.iloc[
        0
    ]


def prior_class_metric(
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

    if len(
        row
    ) != 1:
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


logistic = validation_metric_row(
    stage22_metrics,
    "Logistic_Regression",
)

random_forest = validation_metric_row(
    stage23_metrics,
    "Random_Forest",
)

extra_trees = validation_metric_row(
    stage24_metrics,
    "Extra_Trees",
)

logistic_warning_recall = prior_class_metric(
    stage22_per_class,
    "Logistic_Regression",
    "Warning",
    "Recall",
)

logistic_critical_recall = prior_class_metric(
    stage22_per_class,
    "Logistic_Regression",
    "Critical",
    "Recall",
)

rf_warning_recall = prior_class_metric(
    stage23_per_class,
    "Random_Forest",
    "Warning",
    "Recall",
)

rf_critical_recall = prior_class_metric(
    stage23_per_class,
    "Random_Forest",
    "Critical",
    "Recall",
)

et_warning_recall = prior_class_metric(
    stage24_per_class,
    "Extra_Trees",
    "Warning",
    "Recall",
)

et_critical_recall = prior_class_metric(
    stage24_per_class,
    "Extra_Trees",
    "Critical",
    "Recall",
)


# ============================================================
# 24. COMPARISONS
# ============================================================

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

macro_f1_change_vs_rf = (
    float(
        validation_metrics[
            "Macro_F1"
        ]
    )
    -
    float(
        random_forest[
            "Macro_F1"
        ]
    )
)

macro_f1_change_vs_et = (
    float(
        validation_metrics[
            "Macro_F1"
        ]
    )
    -
    float(
        extra_trees[
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
# 25. GENERALIZATION GAP
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
# 26. SAVE MODEL + HASHES
# ============================================================

joblib.dump(
    model,
    MODEL_PATH,
)


def sha256_file(
    path,
):
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

sample_weights_sha256 = sha256_file(
    STAGE21_SAMPLE_WEIGHTS_PATH
)


# ============================================================
# 27. CHECKS
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

all_three_predicted = bool(
    set(
        predicted_classes_validation
    )
    ==
    EXPECTED_CLASSES
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
                ", ".join(
                    CLASS_ORDER
                )
            ),
        },
        {
            "Check": (
                "Stage-21 Gradient Boosting strategy verified"
            ),
            "Status": (
                "PASS"
                if actual_strategy
                ==
                expected_strategy
                else "FAIL"
            ),
            "Details": (
                actual_strategy
            ),
        },
        {
            "Check": (
                "Sample-weight row indices align with Train"
            ),
            "Status": (
                "PASS"
                if sample_weight_indices_match
                else "FAIL"
            ),
            "Details": (
                f"Rows={len(stage21_sample_weights):,}"
            ),
        },
        {
            "Check": (
                "Sample-weight labels align with y_train"
            ),
            "Status": (
                "PASS"
                if sample_weight_labels_match
                else "FAIL"
            ),
            "Details": (
                "Exact row-order label match."
            ),
        },
        {
            "Check": (
                "Sample-weight values reproduce Stage-21 class weights"
            ),
            "Status": (
                "PASS"
                if sample_weight_values_match
                else "FAIL"
            ),
            "Details": (
                "All 16,246 Train weights verified."
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
                "Train-derived sample weighting only."
            ),
        },
        {
            "Check": (
                "No hyperparameter tuning"
            ),
            "Status": "PASS",
            "Details": (
                "One fixed Gradient Boosting configuration."
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
                "All three classes predicted on Validation"
            ),
            "Status": (
                "PASS"
                if all_three_predicted
                else "FAIL"
            ),
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
# 28. METADATA / SUMMARY
# ============================================================

metadata = {
    "stage": (
        "25_weighted_gradient_boosting_classification"
    ),
    "model": (
        "GradientBoostingClassifier"
    ),
    "parameters": (
        GB_PARAMS
    ),
    "fit_scope": (
        "TRAIN_ONLY"
    ),
    "validation_scope": (
        "DEVELOPMENT_EVALUATION_ONLY"
    ),
    "imbalance_strategy": (
        "TRAIN-derived sample_weight"
    ),
    "sample_weight_source": (
        str(
            STAGE21_SAMPLE_WEIGHTS_PATH
        )
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
    "comparison_vs_random_forest": {
        "Macro_F1_change": (
            macro_f1_change_vs_rf
        ),
    },
    "comparison_vs_extra_trees": {
        "Macro_F1_change": (
            macro_f1_change_vs_et
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
    "sample_weight_sha256": (
        sample_weights_sha256
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
# 29. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- FIXED GRADIENT BOOSTING CONFIGURATION ---"
)

for key, value in (
    GB_PARAMS.items()
):
    print(
        f"{key:<28}: {value}"
    )


print(
    "\n--- STAGE-21 SAMPLE-WEIGHT VERIFICATION ---"
)

print(
    f"Weighted Train rows             : "
    f"{len(train_sample_weights):,}"
)

print(
    f"Row indices aligned             : "
    f"{sample_weight_indices_match}"
)

print(
    f"Labels aligned                  : "
    f"{sample_weight_labels_match}"
)

print(
    f"Weight values match Stage 21    : "
    f"{sample_weight_values_match}"
)


print(
    "\n--- GRADIENT BOOSTING RESULTS ---"
)

print(
    metrics_df
    .round(
        4
    )
    .to_string(
        index=False
    )
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
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\n--- COMPARISON WITH PREVIOUS REAL CLASSIFIERS ---"
)

print(
    f"Logistic Validation Macro-F1    : "
    f"{float(logistic['Macro_F1']):.4f}"
)

print(
    f"Random Forest Validation Macro-F1: "
    f"{float(random_forest['Macro_F1']):.4f}"
)

print(
    f"Extra Trees Validation Macro-F1 : "
    f"{float(extra_trees['Macro_F1']):.4f}"
)

print(
    f"Gradient Boosting Val Macro-F1  : "
    f"{float(validation_metrics['Macro_F1']):.4f}"
)

print(
    f"Macro-F1 change vs Logistic     : "
    f"{macro_f1_change_vs_logistic:+.4f}"
)

print(
    f"Macro-F1 change vs RF           : "
    f"{macro_f1_change_vs_rf:+.4f}"
)

print(
    f"Macro-F1 change vs Extra Trees  : "
    f"{macro_f1_change_vs_et:+.4f}"
)

print(
    f"Balanced Acc. change vs Logistic: "
    f"{balanced_accuracy_change_vs_logistic:+.4f}"
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
    "\n--- RECALL CHANGES VS LOGISTIC REGRESSION ---"
)

print(
    f"Warning Recall change           : "
    f"{warning_recall_change_vs_logistic:+.4f}"
)

print(
    f"Critical Recall change          : "
    f"{critical_recall_change_vs_logistic:+.4f}"
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
# 30. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 25: "
        "WEIGHTED GRADIENT BOOSTING CLASSIFICATION\n"
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
        "Stage 25 evaluates a fixed, untuned Gradient Boosting "
        "classifier for Fire_Risk_Label. Because "
        "GradientBoostingClassifier does not expose class_weight, "
        "the exact Train-only sample weights generated in Stage 21 "
        "are supplied to fit(). Validation is used only for "
        "development evaluation.\n\n"
    )

    report.write(
        "FIXED CONFIGURATION\n"
        "-------------------\n"
    )

    for key, value in (
        GB_PARAMS.items()
    ):
        report.write(
            f"{key}: {value}\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "STAGE-21 SAMPLE WEIGHT VERIFICATION\n"
        "-----------------------------------\n"
    )

    report.write(
        f"Weighted Train rows: "
        f"{len(train_sample_weights):,}\n"
    )

    report.write(
        f"Row indices aligned: "
        f"{sample_weight_indices_match}\n"
    )

    report.write(
        f"Labels aligned: "
        f"{sample_weight_labels_match}\n"
    )

    report.write(
        f"Weight values reproduce Stage 21: "
        f"{sample_weight_values_match}\n\n"
    )

    report.write(
        "OVERALL RESULTS\n"
        "---------------\n"
    )

    report.write(
        metrics_df
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
        "COMPARISON WITH PREVIOUS REAL CLASSIFIERS\n"
        "-----------------------------------------\n"
    )

    report.write(
        f"Logistic Validation Macro-F1: "
        f"{float(logistic['Macro_F1']):.6f}\n"
    )

    report.write(
        f"Random Forest Validation Macro-F1: "
        f"{float(random_forest['Macro_F1']):.6f}\n"
    )

    report.write(
        f"Extra Trees Validation Macro-F1: "
        f"{float(extra_trees['Macro_F1']):.6f}\n"
    )

    report.write(
        f"Gradient Boosting Validation Macro-F1: "
        f"{float(validation_metrics['Macro_F1']):.6f}\n"
    )

    report.write(
        f"Macro-F1 change vs Logistic: "
        f"{macro_f1_change_vs_logistic:+.6f}\n\n"
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
        "Gradient Boosting impurity-based feature importance is "
        "exported only for descriptive inspection. It does not "
        "reopen the Stage-08 feature-selection decision. Formal "
        "explainability remains reserved for Stage 32.\n\n"
    )

    report.write(
        "METHODOLOGY PROTECTION\n"
        "----------------------\n"
    )

    report.write(
        "- Model fitted on Train only.\n"
        "- Exact Stage-21 Train sample weights used.\n"
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
        "Stage 26 will evaluate weighted XGBoost classification "
        "using the same Train-derived Stage-21 sample weights.\n\n"
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
        f"STAGE 25 STATUS: "
        f"{overall_status}\n"
    )


# ============================================================
# 31. COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    f"STAGE 25 WEIGHTED GRADIENT BOOSTING CLASSIFICATION: "
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
        "Stage 25 failed. Review "
        "25_gradient_boosting_classification_report.txt."
    )
