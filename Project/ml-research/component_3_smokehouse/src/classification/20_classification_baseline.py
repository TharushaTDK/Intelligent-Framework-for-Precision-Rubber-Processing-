from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# COMPONENT 3 - STAGE 20: CLASSIFICATION BASELINE
# ============================================================
#
# Target:
#   Fire_Risk_Label = Normal / Warning / Critical
#
# Purpose:
#   Establish naive classification baselines before imbalance
#   handling and real classifiers are introduced.
#
# Baselines:
#   1) Dummy_Most_Frequent
#      Always predicts the majority class observed in TRAIN.
#
#   2) Dummy_Stratified
#      Random predictions sampled according to TRAIN class priors.
#
# Why this matters:
#   The dataset is strongly imbalanced toward Normal. A classifier
#   that predicts only Normal can obtain high raw Accuracy while
#   completely failing to identify Warning and Critical events.
#
# Therefore the main baseline metrics include:
#   - Accuracy
#   - Balanced Accuracy
#   - Macro Precision
#   - Macro Recall
#   - Macro F1
#   - Weighted F1
#   - Per-class Precision / Recall / F1
#   - Confusion Matrix
#
# Methodology:
#   - Fit baselines using TRAIN only.
#   - Evaluate on VALIDATION only.
#   - No class weights.
#   - No SMOTE / oversampling / undersampling.
#   - No threshold tuning.
#   - Final TEST remains untouched until Stage 31.
#
# Note:
#   Fire-risk labels originate from hourly keeper/expert
#   assessments and are carried across the corresponding 10-minute
#   sensor rows. Primary model evaluation is row-level because the
#   operational system predicts every 10 minutes. Unique hourly
#   assessment counts are also documented for research integrity.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

# File location:
# component_3_smokehouse/src/classification/20_baseline.py
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
    / "classification"
    / "baseline"
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


# ============================================================
# 3. OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "20_classification_baseline_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "20_classification_baseline_checks.csv"
)

METRICS_PATH = (
    REPORT_DIR
    / "20_classification_baseline_metrics.csv"
)

PER_CLASS_PATH = (
    REPORT_DIR
    / "20_classification_baseline_per_class_metrics.csv"
)

CONFUSION_PATH = (
    REPORT_DIR
    / "20_classification_baseline_confusion_matrices.csv"
)

PREDICTIONS_PATH = (
    REPORT_DIR
    / "20_classification_baseline_validation_predictions.csv"
)

ROW_DISTRIBUTION_PATH = (
    REPORT_DIR
    / "20_classification_row_distribution.csv"
)

ASSESSMENT_DISTRIBUTION_PATH = (
    REPORT_DIR
    / "20_classification_assessment_distribution.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "20_classification_baseline_summary.json"
)

MOST_FREQUENT_MODEL_PATH = (
    MODEL_DIR
    / "dummy_most_frequent.joblib"
)

STRATIFIED_MODEL_PATH = (
    MODEL_DIR
    / "dummy_stratified.joblib"
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


# ============================================================
# 5. START
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 20: CLASSIFICATION BASELINE")
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
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("\nTrain and Validation classification data found.")
print("Final Test dataset existence verified, but contents were NOT loaded.")


# ============================================================
# 7. LOAD TRAIN + VALIDATION ONLY
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


# ============================================================
# 8. TARGET COLUMN VALIDATION
# ============================================================

TARGET_COLUMN = (
    "Fire_Risk_Label"
)

if TARGET_COLUMN not in y_train_df.columns:
    if len(
        y_train_df.columns
    ) == 1:
        TARGET_COLUMN = (
            y_train_df.columns[
                0
            ]
        )
    else:
        raise RuntimeError(
            "Could not determine classification target column "
            "in Train target file."
        )

if TARGET_COLUMN not in y_validation_df.columns:
    if len(
        y_validation_df.columns
    ) == 1:
        validation_target_column = (
            y_validation_df.columns[
                0
            ]
        )
    else:
        raise RuntimeError(
            "Could not determine classification target column "
            "in Validation target file."
        )
else:
    validation_target_column = (
        TARGET_COLUMN
    )

y_train = (
    y_train_df[
        TARGET_COLUMN
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
        "Unexpected Train classification classes.\n"
        f"Expected: {sorted(EXPECTED_CLASSES)}\n"
        f"Found: {sorted(train_classes)}"
    )

if validation_classes != EXPECTED_CLASSES:
    raise RuntimeError(
        "Unexpected Validation classification classes.\n"
        f"Expected: {sorted(EXPECTED_CLASSES)}\n"
        f"Found: {sorted(validation_classes)}"
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
# 10. ROW-LEVEL CLASS DISTRIBUTION
# ============================================================

distribution_rows = []

for split_name, y_values in [
    (
        "Train",
        y_train,
    ),
    (
        "Validation",
        y_validation,
    ),
]:
    total = len(
        y_values
    )

    for class_name in CLASS_ORDER:
        count = int(
            (
                y_values
                ==
                class_name
            ).sum()
        )

        distribution_rows.append({
            "Split": split_name,
            "Class": class_name,
            "Rows": count,
            "Percentage": (
                count
                /
                total
                *
                100
            ),
        })

row_distribution_df = pd.DataFrame(
    distribution_rows
)

row_distribution_df.to_csv(
    ROW_DISTRIBUTION_PATH,
    index=False,
)


# ============================================================
# 11. UNIQUE HOURLY ASSESSMENT DISTRIBUTION
# ============================================================

required_metadata_columns = {
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
    "Fire_Risk_Assessment_Timestamp",
}

missing_metadata_columns = (
    required_metadata_columns
    -
    set(
        meta_train.columns
    )
)

if missing_metadata_columns:
    raise RuntimeError(
        "Classification metadata missing required columns: "
        f"{sorted(missing_metadata_columns)}"
    )

assessment_rows = []

for split_name, metadata, y_values in [
    (
        "Train",
        meta_train,
        y_train,
    ),
    (
        "Validation",
        meta_validation,
        y_validation,
    ),
]:
    temp = metadata[
        [
            "Smokehouse_ID",
            "Batch_ID",
            "Layer_ID",
            "Fire_Risk_Assessment_Timestamp",
        ]
    ].copy()

    temp[
        "Fire_Risk_Label"
    ] = y_values

    assessment_key = [
        "Smokehouse_ID",
        "Batch_ID",
        "Layer_ID",
        "Fire_Risk_Assessment_Timestamp",
    ]

    label_consistency = (
        temp
        .groupby(
            assessment_key,
            dropna=False,
        )[
            "Fire_Risk_Label"
        ]
        .nunique()
    )

    inconsistent_assessments = int(
        (
            label_consistency
            >
            1
        ).sum()
    )

    if inconsistent_assessments != 0:
        raise RuntimeError(
            f"{split_name} has {inconsistent_assessments} "
            "assessment groups with inconsistent labels."
        )

    unique_assessments = (
        temp
        .drop_duplicates(
            subset=assessment_key,
            keep="first",
        )
    )

    total_assessments = len(
        unique_assessments
    )

    for class_name in CLASS_ORDER:
        count = int(
            (
                unique_assessments[
                    "Fire_Risk_Label"
                ]
                ==
                class_name
            ).sum()
        )

        assessment_rows.append({
            "Split": split_name,
            "Class": class_name,
            "Unique_Assessments": count,
            "Percentage": (
                count
                /
                total_assessments
                *
                100
            ),
        })

assessment_distribution_df = pd.DataFrame(
    assessment_rows
)

assessment_distribution_df.to_csv(
    ASSESSMENT_DISTRIBUTION_PATH,
    index=False,
)


# ============================================================
# 12. BASELINE MODELS
# ============================================================

baseline_models = {
    "Dummy_Most_Frequent": DummyClassifier(
        strategy="most_frequent"
    ),
    "Dummy_Stratified": DummyClassifier(
        strategy="stratified",
        random_state=42,
    ),
}


# ============================================================
# 13. METRIC HELPERS
# ============================================================

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
        "Balanced_Accuracy": float(
            balanced_accuracy_score(
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


def per_class_metrics(
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
        class_result = report[
            class_name
        ]

        rows.append({
            "Model": model_name,
            "Split": split_name,
            "Class": class_name,
            "Precision": float(
                class_result[
                    "precision"
                ]
            ),
            "Recall": float(
                class_result[
                    "recall"
                ]
            ),
            "F1": float(
                class_result[
                    "f1-score"
                ]
            ),
            "Support": int(
                class_result[
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

    for actual_index, actual_class in enumerate(
        CLASS_ORDER
    ):
        for pred_index, predicted_class in enumerate(
            CLASS_ORDER
        ):
            rows.append({
                "Model": model_name,
                "Split": split_name,
                "Actual_Class": actual_class,
                "Predicted_Class": predicted_class,
                "Count": int(
                    matrix[
                        actual_index,
                        pred_index,
                    ]
                ),
            })

    return rows


# ============================================================
# 14. FIT TRAIN ONLY + EVALUATE
# ============================================================

metric_rows = []
per_class_rows = []
confusion_output_rows = []
validation_prediction_df = meta_validation.copy()

fitted_models = {}

for model_name, model in baseline_models.items():

    print(
        f"\nFitting {model_name} on TRAIN only..."
    )

    model.fit(
        X_train,
        y_train,
    )

    fitted_models[
        model_name
    ] = model

    train_pred = model.predict(
        X_train
    )

    validation_pred = model.predict(
        X_validation
    )

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
            "Model": model_name,
            "Split": split_name,
            **metrics,
        })

        per_class_rows.extend(
            per_class_metrics(
                model_name,
                split_name,
                y_true,
                y_pred,
            )
        )

        confusion_output_rows.extend(
            confusion_rows(
                model_name,
                split_name,
                y_true,
                y_pred,
            )
        )

    validation_prediction_df[
        f"Prediction_{model_name}"
    ] = validation_pred


# ============================================================
# 15. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    metric_rows
)

per_class_df = pd.DataFrame(
    per_class_rows
)

confusion_df = pd.DataFrame(
    confusion_output_rows
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

validation_prediction_df[
    "Actual_Fire_Risk_Label"
] = y_validation

validation_prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# 16. SAVE BASELINE MODELS
# ============================================================

joblib.dump(
    fitted_models[
        "Dummy_Most_Frequent"
    ],
    MOST_FREQUENT_MODEL_PATH,
)

joblib.dump(
    fitted_models[
        "Dummy_Stratified"
    ],
    STRATIFIED_MODEL_PATH,
)


# ============================================================
# 17. EXTRACT VALIDATION RESULTS
# ============================================================

def get_metric_row(
    model_name,
    split_name,
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
            split_name
        )
    ]

    if len(
        row
    ) != 1:
        raise RuntimeError(
            f"Could not uniquely locate "
            f"{model_name}/{split_name} metrics."
        )

    return row.iloc[
        0
    ]


most_frequent_val = get_metric_row(
    "Dummy_Most_Frequent",
    "Validation",
)

stratified_val = get_metric_row(
    "Dummy_Stratified",
    "Validation",
)


# ============================================================
# 18. PER-CLASS VALIDATION RECALL
# ============================================================

def class_metric(
    model_name,
    class_name,
    metric_name,
):
    row = per_class_df[
        (
            per_class_df[
                "Model"
            ]
            ==
            model_name
        )
        &
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
            "Could not locate requested per-class result."
        )

    return float(
        row.iloc[
            0
        ][
            metric_name
        ]
    )


most_frequent_warning_recall = class_metric(
    "Dummy_Most_Frequent",
    "Warning",
    "Recall",
)

most_frequent_critical_recall = class_metric(
    "Dummy_Most_Frequent",
    "Critical",
    "Recall",
)

stratified_warning_recall = class_metric(
    "Dummy_Stratified",
    "Warning",
    "Recall",
)

stratified_critical_recall = class_metric(
    "Dummy_Stratified",
    "Critical",
    "Recall",
)


# ============================================================
# 19. BASELINE INTERPRETATION
# ============================================================

majority_class = str(
    fitted_models[
        "Dummy_Most_Frequent"
    ].classes_[
        np.argmax(
            fitted_models[
                "Dummy_Most_Frequent"
            ].class_prior_
        )
    ]
)

majority_train_percentage = float(
    (
        y_train
        ==
        majority_class
    ).mean()
    *
    100
)

high_accuracy_but_poor_minority_detection = bool(
    float(
        most_frequent_val[
            "Accuracy"
        ]
    )
    >
    0.90
    and
    most_frequent_warning_recall
    ==
    0.0
    and
    most_frequent_critical_recall
    ==
    0.0
)


# ============================================================
# 20. CHECKS
# ============================================================

row_distribution_complete = bool(
    len(
        row_distribution_df
    )
    ==
    6
)

assessment_distribution_complete = bool(
    len(
        assessment_distribution_df
    )
    ==
    6
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

both_baselines_present = bool(
    set(
        metrics_df[
            "Model"
        ].unique()
    )
    ==
    {
        "Dummy_Most_Frequent",
        "Dummy_Stratified",
    }
)

both_splits_present = bool(
    set(
        metrics_df[
            "Split"
        ].unique()
    )
    ==
    {
        "Train",
        "Validation",
    }
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
                "Expected classification classes"
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
                "Row-level distributions generated"
            ),
            "Status": (
                "PASS"
                if row_distribution_complete
                else "FAIL"
            ),
            "Details": (
                "Train and Validation × 3 classes."
            ),
        },
        {
            "Check": (
                "Unique assessment distributions generated"
            ),
            "Status": (
                "PASS"
                if assessment_distribution_complete
                else "FAIL"
            ),
            "Details": (
                "Hourly keeper/expert assessments documented."
            ),
        },
        {
            "Check": (
                "Both baseline models evaluated"
            ),
            "Status": (
                "PASS"
                if both_baselines_present
                else "FAIL"
            ),
            "Details": (
                "Most-Frequent and Stratified."
            ),
        },
        {
            "Check": (
                "Train and Validation metrics generated"
            ),
            "Status": (
                "PASS"
                if both_splits_present
                else "FAIL"
            ),
            "Details": (
                "No Test metrics generated."
            ),
        },
        {
            "Check": (
                "All classification metrics finite"
            ),
            "Status": (
                "PASS"
                if all_metrics_finite
                else "FAIL"
            ),
            "Details": (
                "zero_division=0 applied where necessary."
            ),
        },
        {
            "Check": (
                "No class balancing performed"
            ),
            "Status": "PASS",
            "Details": (
                "No weights, SMOTE, over/under-sampling."
            ),
        },
        {
            "Check": (
                "No real classifier trained"
            ),
            "Status": "PASS",
            "Details": (
                "Dummy baselines only; Logistic Regression starts Stage 22."
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
# 21. SUMMARY JSON
# ============================================================

summary = {
    "stage": (
        "20_classification_baseline"
    ),
    "overall_status": (
        overall_status
    ),
    "target": (
        "Fire_Risk_Label"
    ),
    "classes": (
        CLASS_ORDER
    ),
    "primary_operational_evaluation_unit": (
        "10-minute sensor row"
    ),
    "ground_truth_assessment_frequency": (
        "hourly keeper/expert assessment carried forward "
        "to associated 10-minute rows"
    ),
    "majority_class": (
        majority_class
    ),
    "majority_class_train_percentage": (
        majority_train_percentage
    ),
    "most_frequent_validation": {
        "Accuracy": float(
            most_frequent_val[
                "Accuracy"
            ]
        ),
        "Balanced_Accuracy": float(
            most_frequent_val[
                "Balanced_Accuracy"
            ]
        ),
        "Macro_F1": float(
            most_frequent_val[
                "Macro_F1"
            ]
        ),
        "Warning_Recall": (
            most_frequent_warning_recall
        ),
        "Critical_Recall": (
            most_frequent_critical_recall
        ),
    },
    "stratified_validation": {
        "Accuracy": float(
            stratified_val[
                "Accuracy"
            ]
        ),
        "Balanced_Accuracy": float(
            stratified_val[
                "Balanced_Accuracy"
            ]
        ),
        "Macro_F1": float(
            stratified_val[
                "Macro_F1"
            ]
        ),
        "Warning_Recall": (
            stratified_warning_recall
        ),
        "Critical_Recall": (
            stratified_critical_recall
        ),
    },
    "accuracy_warning_demonstrated": (
        high_accuracy_but_poor_minority_detection
    ),
    "class_weighting_used": False,
    "resampling_used": False,
    "test_loaded": False,
    "next_stage": (
        "21_class_imbalance_analysis"
    ),
}

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=2,
    )


# ============================================================
# 22. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- ROW-LEVEL CLASS DISTRIBUTION ---"
)

print(
    row_distribution_df
    .round(
        {
            "Percentage": 4
        }
    )
    .to_string(
        index=False
    )
)


print(
    "\n--- UNIQUE HOURLY ASSESSMENT DISTRIBUTION ---"
)

print(
    assessment_distribution_df
    .round(
        {
            "Percentage": 4
        }
    )
    .to_string(
        index=False
    )
)


print(
    "\n--- CLASSIFICATION BASELINE RESULTS ---"
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
    "\n--- VALIDATION MINORITY-CLASS RECALL ---"
)

print(
    f"Most-Frequent Warning recall    : "
    f"{most_frequent_warning_recall:.4f}"
)

print(
    f"Most-Frequent Critical recall   : "
    f"{most_frequent_critical_recall:.4f}"
)

print(
    f"Stratified Warning recall       : "
    f"{stratified_warning_recall:.4f}"
)

print(
    f"Stratified Critical recall      : "
    f"{stratified_critical_recall:.4f}"
)


print(
    "\n--- BASELINE INTERPRETATION ---"
)

print(
    f"TRAIN majority class            : "
    f"{majority_class}"
)

print(
    f"TRAIN majority proportion       : "
    f"{majority_train_percentage:.2f}%"
)

print(
    f"High accuracy can hide failure  : "
    f"{high_accuracy_but_poor_minority_detection}"
)

print(
    "Primary future comparison should emphasize Macro-F1, "
    "Balanced Accuracy, and Warning/Critical recall."
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
# 23. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 20: CLASSIFICATION BASELINE\n"
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
        "Stage 20 establishes naive baselines for the three-class "
        "Fire_Risk_Label target before class-imbalance handling or "
        "real classifiers are introduced. The baselines are "
        "Dummy_Most_Frequent and Dummy_Stratified.\n\n"
    )

    report.write(
        "TARGET SEMANTICS\n"
        "----------------\n"
    )

    report.write(
        "The target classes are Normal, Warning, and Critical. "
        "The source labels represent hourly keeper/expert fire-risk "
        "assessments carried forward across the associated 10-minute "
        "sensor rows. The operational classifier will produce a "
        "prediction every 10 minutes, so primary classification "
        "metrics are reported at row level. Unique hourly assessment "
        "counts are separately documented to avoid treating every "
        "10-minute row as an independent keeper decision.\n\n"
    )

    report.write(
        "ROW-LEVEL CLASS DISTRIBUTION\n"
        "----------------------------\n"
    )

    report.write(
        row_distribution_df
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
        "UNIQUE HOURLY ASSESSMENT DISTRIBUTION\n"
        "-------------------------------------\n"
    )

    report.write(
        assessment_distribution_df
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
        "BASELINE RESULTS\n"
        "----------------\n"
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
        "WHY ACCURACY IS NOT ENOUGH\n"
        "--------------------------\n"
    )

    report.write(
        f"The majority class in Train is {majority_class}, "
        f"representing approximately "
        f"{majority_train_percentage:.4f}% of Train rows. "
        "A Most-Frequent classifier can therefore obtain high raw "
        "Accuracy while producing zero recall for Warning and "
        "Critical. For this research task, Balanced Accuracy, "
        "Macro-F1, and especially Warning/Critical recall are more "
        "informative than Accuracy alone.\n\n"
    )

    report.write(
        "MOST-FREQUENT VALIDATION DETAILS\n"
        "--------------------------------\n"
    )

    report.write(
        f"Accuracy: "
        f"{float(most_frequent_val['Accuracy']):.6f}\n"
    )

    report.write(
        f"Balanced Accuracy: "
        f"{float(most_frequent_val['Balanced_Accuracy']):.6f}\n"
    )

    report.write(
        f"Macro-F1: "
        f"{float(most_frequent_val['Macro_F1']):.6f}\n"
    )

    report.write(
        f"Warning Recall: "
        f"{most_frequent_warning_recall:.6f}\n"
    )

    report.write(
        f"Critical Recall: "
        f"{most_frequent_critical_recall:.6f}\n\n"
    )

    report.write(
        "METHODOLOGY PROTECTION\n"
        "----------------------\n"
    )

    report.write(
        "- Baselines fitted using Train only.\n"
    )

    report.write(
        "- Validation used only for development evaluation.\n"
    )

    report.write(
        "- No class weighting was used.\n"
    )

    report.write(
        "- No SMOTE or other resampling was used.\n"
    )

    report.write(
        "- No threshold tuning was performed.\n"
    )

    report.write(
        "- Final Test rows were not loaded.\n"
    )

    report.write(
        "- Final Test remains frozen until Stage 31.\n\n"
    )

    report.write(
        "NEXT STAGE\n"
        "----------\n"
    )

    report.write(
        "Stage 21 will formally analyze class imbalance and define "
        "the scientifically appropriate imbalance-handling strategy "
        "before Logistic Regression is trained in Stage 22.\n\n"
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
        f"STAGE 20 STATUS: "
        f"{overall_status}\n"
    )


# ============================================================
# 24. COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    f"STAGE 20 CLASSIFICATION BASELINE: "
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
    f"\nBaseline models saved in:\n"
    f"{MODEL_DIR}"
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 20 failed. Review "
        "20_classification_baseline_report.txt."
    )
