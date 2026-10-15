from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# COMPONENT 3 - STAGE 22: WEIGHTED LOGISTIC REGRESSION
# ============================================================
#
# Target:
#   Fire_Risk_Label = Normal / Warning / Critical
#
# Purpose:
#   Train the first real Fire-Risk classifier using the
#   Train-only imbalance strategy established in Stage 21.
#
# Model:
#   Multiclass Logistic Regression
#
# Imbalance handling:
#   class_weight="balanced"
#
# Primary evaluation metric:
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
#   - No SMOTE / oversampling / undersampling.
#   - No hyperparameter tuning.
#   - No threshold tuning.
#   - Final TEST remains untouched until Stage 31.
#
# Stage 22 is a fixed untuned model experiment.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

# File location:
# component_3_smokehouse/src/classification/22_logistic_regression.py
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
    / "logistic_regression"
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

STAGE20_METRICS_PATH = (
    REPORT_DIR
    / "20_classification_baseline_metrics.csv"
)

STAGE20_PER_CLASS_PATH = (
    REPORT_DIR
    / "20_classification_baseline_per_class_metrics.csv"
)

STAGE21_WEIGHTS_PATH = (
    REPORT_DIR
    / "21_class_weights.csv"
)

STAGE21_STRATEGY_PATH = (
    REPORT_DIR
    / "21_class_imbalance_strategy.json"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "22_logistic_regression_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "22_logistic_regression_checks.csv"
)

METRICS_PATH = (
    REPORT_DIR
    / "22_logistic_regression_metrics.csv"
)

PER_CLASS_PATH = (
    REPORT_DIR
    / "22_logistic_regression_per_class_metrics.csv"
)

CONFUSION_PATH = (
    REPORT_DIR
    / "22_logistic_regression_confusion_matrix.csv"
)

PREDICTIONS_PATH = (
    REPORT_DIR
    / "22_logistic_regression_validation_predictions.csv"
)

LAYER_PATH = (
    REPORT_DIR
    / "22_logistic_regression_by_layer.csv"
)

BATCH_PATH = (
    REPORT_DIR
    / "22_logistic_regression_by_batch.csv"
)

COEFFICIENT_PATH = (
    REPORT_DIR
    / "22_logistic_regression_coefficients.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "22_logistic_regression_summary.json"
)

MODEL_PATH = (
    MODEL_DIR
    / "logistic_regression_classifier.joblib"
)

MODEL_METADATA_PATH = (
    MODEL_DIR
    / "logistic_regression_metadata.json"
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
print("COMPONENT 3 - STAGE 22: WEIGHTED LOGISTIC REGRESSION")
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
    STAGE20_METRICS_PATH,
    STAGE20_PER_CLASS_PATH,
    STAGE21_WEIGHTS_PATH,
    STAGE21_STRATEGY_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("\nTrain and Validation model-ready data found.")
print("Stage 20 baseline and Stage 21 imbalance evidence found.")
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

stage20_metrics = pd.read_csv(
    STAGE20_METRICS_PATH
)

stage20_per_class = pd.read_csv(
    STAGE20_PER_CLASS_PATH
)

stage21_weights = pd.read_csv(
    STAGE21_WEIGHTS_PATH
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
# 10. VERIFY STAGE-21 WEIGHT STRATEGY
# ============================================================

expected_strategy = (
    "class_weight='balanced'"
)

actual_strategy = (
    stage21_strategy[
        "imbalance_strategy"
    ][
        "Logistic_Regression"
    ]
)

if actual_strategy != expected_strategy:
    raise RuntimeError(
        "Stage-21 Logistic Regression imbalance strategy mismatch.\n"
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
    class_name: float(
        weight
    )
    for class_name, weight in zip(
        CLASS_ORDER,
        computed_weights,
    )
}

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

weights_match_stage21 = bool(
    all(
        class_name in reported_weight_map
        and
        np.isclose(
            computed_weight_map[
                class_name
            ],
            reported_weight_map[
                class_name
            ],
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
# 11. FIXED LOGISTIC REGRESSION CONFIGURATION
# ============================================================
#
# Stage 22 is intentionally untuned.
#
# The Stage-10 numeric features are already standardized and
# categorical features are one-hot encoded, which is appropriate
# for Logistic Regression.
#
# lbfgs supports multinomial multiclass classification.
# We do not explicitly set multi_class because recent sklearn
# versions handle multinomial behavior automatically for
# multiclass problems with compatible solvers.

LOGISTIC_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "max_iter": 5000,
    "tol": 1e-4,
    "random_state": 42,
}

model = LogisticRegression(
    **LOGISTIC_PARAMS
)


# ============================================================
# 12. FIT TRAIN ONLY
# ============================================================

print("\nTraining weighted Logistic Regression on TRAIN only...")

model.fit(
    X_train,
    y_train,
)

print("Logistic Regression training complete.")


# ============================================================
# 13. PREDICTIONS
# ============================================================

train_pred = model.predict(
    X_train
)

validation_pred = model.predict(
    X_validation
)

train_proba = model.predict_proba(
    X_train
)

validation_proba = model.predict_proba(
    X_validation
)


# ============================================================
# 14. METRIC HELPERS
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
# 15. OVERALL + PER-CLASS METRICS
# ============================================================

MODEL_NAME = (
    "Logistic_Regression"
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
# 16. VALIDATION PREDICTION OUTPUT
# ============================================================

prediction_df = meta_validation.copy()

prediction_df[
    "Actual_Fire_Risk_Label"
] = y_validation

prediction_df[
    "Predicted_Fire_Risk_Label"
] = validation_pred

# sklearn model.classes_ controls predict_proba column order.
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
# 17. LAYER-WISE VALIDATION METRICS
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
# 18. BATCH-WISE VALIDATION METRICS
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
# 19. COEFFICIENT EXPORT
# ============================================================
#
# Stage 10 standardized the numeric features, so coefficient
# magnitudes are somewhat more comparable than they would be on
# raw unscaled numeric inputs. One-hot categorical columns remain
# indicator variables, so interpretation still requires care.

coefficient_rows = []

for class_index, class_name in enumerate(
    model.classes_
):
    for feature_name, coefficient in zip(
        X_train.columns,
        model.coef_[
            class_index
        ],
    ):
        coefficient_rows.append({
            "Class": class_name,
            "Feature": feature_name,
            "Coefficient": float(
                coefficient
            ),
            "Absolute_Coefficient": float(
                abs(
                    coefficient
                )
            ),
        })

coefficient_df = pd.DataFrame(
    coefficient_rows
)

coefficient_df = (
    coefficient_df
    .sort_values(
        by=[
            "Class",
            "Absolute_Coefficient",
        ],
        ascending=[
            True,
            False,
        ],
    )
)

coefficient_df.to_csv(
    COEFFICIENT_PATH,
    index=False,
)


# ============================================================
# 20. BASELINE COMPARISON
# ============================================================

def baseline_metric_row(
    model_name,
):
    row = stage20_metrics[
        (
            stage20_metrics[
                "Model"
            ]
            ==
            model_name
        )
        &
        (
            stage20_metrics[
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
            f"Could not uniquely locate Stage-20 baseline: {model_name}"
        )

    return row.iloc[
        0
    ]


most_frequent_baseline = baseline_metric_row(
    "Dummy_Most_Frequent"
)

stratified_baseline = baseline_metric_row(
    "Dummy_Stratified"
)

validation_metrics = metrics_df[
    metrics_df[
        "Split"
    ]
    ==
    "Validation"
].iloc[
    0
]

train_metrics = metrics_df[
    metrics_df[
        "Split"
    ]
    ==
    "Train"
].iloc[
    0
]


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

macro_f1_improvement_vs_most_frequent = (
    float(
        validation_metrics[
            "Macro_F1"
        ]
    )
    -
    float(
        most_frequent_baseline[
            "Macro_F1"
        ]
    )
)

macro_f1_improvement_vs_stratified = (
    float(
        validation_metrics[
            "Macro_F1"
        ]
    )
    -
    float(
        stratified_baseline[
            "Macro_F1"
        ]
    )
)

balanced_accuracy_improvement_vs_most_frequent = (
    float(
        validation_metrics[
            "Balanced_Accuracy"
        ]
    )
    -
    float(
        most_frequent_baseline[
            "Balanced_Accuracy"
        ]
    )
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
# 22. CONVERGENCE
# ============================================================

iterations_used = int(
    np.max(
        model.n_iter_
    )
)

converged_before_limit = bool(
    iterations_used
    <
    LOGISTIC_PARAMS[
        "max_iter"
    ]
)


# ============================================================
# 23. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH,
)


# ============================================================
# 24. HASHES
# ============================================================

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
# 25. PASS / FAIL CHECKS
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

all_classes_predicted_validation = bool(
    set(
        np.unique(
            validation_pred
        )
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
                "Stage-21 Logistic Regression strategy verified"
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
                "Balanced weights reproduce Stage 21"
            ),
            "Status": (
                "PASS"
                if weights_match_stage21
                else "FAIL"
            ),
            "Details": (
                str(
                    computed_weight_map
                )
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
                "One fixed Logistic Regression configuration."
            ),
        },
        {
            "Check": (
                "Model converged before max_iter"
            ),
            "Status": (
                "PASS"
                if converged_before_limit
                else "FAIL"
            ),
            "Details": (
                f"Iterations={iterations_used}; "
                f"max_iter={LOGISTIC_PARAMS['max_iter']}"
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
                "All three classes predicted on Validation"
            ),
            "Status": (
                "PASS"
                if all_classes_predicted_validation
                else "FAIL"
            ),
            "Details": (
                ", ".join(
                    sorted(
                        np.unique(
                            validation_pred
                        )
                    )
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
# 26. MODEL METADATA + SUMMARY
# ============================================================

metadata = {
    "stage": (
        "22_weighted_logistic_regression"
    ),
    "model": (
        "LogisticRegression"
    ),
    "parameters": (
        LOGISTIC_PARAMS
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
    "iterations_used": (
        iterations_used
    ),
    "converged_before_limit": (
        converged_before_limit
    ),
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
    "generalization": {
        "Macro_F1_Train_minus_Validation": (
            macro_f1_gap
        ),
        "Balanced_Accuracy_Train_minus_Validation": (
            balanced_accuracy_gap
        ),
    },
    "baseline_comparison": {
        "Macro_F1_change_vs_Most_Frequent": (
            macro_f1_improvement_vs_most_frequent
        ),
        "Macro_F1_change_vs_Stratified": (
            macro_f1_improvement_vs_stratified
        ),
        "Balanced_Accuracy_change_vs_Most_Frequent": (
            balanced_accuracy_improvement_vs_most_frequent
        ),
    },
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
# 27. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- FIXED LOGISTIC REGRESSION CONFIGURATION ---"
)

for key, value in (
    LOGISTIC_PARAMS.items()
):
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
    "\n--- LOGISTIC REGRESSION RESULTS ---"
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
    "\n--- BASELINE COMPARISON ---"
)

print(
    f"Most-Frequent Macro-F1         : "
    f"{float(most_frequent_baseline['Macro_F1']):.4f}"
)

print(
    f"Stratified Macro-F1            : "
    f"{float(stratified_baseline['Macro_F1']):.4f}"
)

print(
    f"Logistic Validation Macro-F1   : "
    f"{float(validation_metrics['Macro_F1']):.4f}"
)

print(
    f"Macro-F1 change vs Most-Freq   : "
    f"{macro_f1_improvement_vs_most_frequent:+.4f}"
)

print(
    f"Macro-F1 change vs Stratified  : "
    f"{macro_f1_improvement_vs_stratified:+.4f}"
)

print(
    f"Balanced Acc. change vs MF     : "
    f"{balanced_accuracy_improvement_vs_most_frequent:+.4f}"
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
    "\n--- CONVERGENCE ---"
)

print(
    f"Iterations used                 : "
    f"{iterations_used}"
)

print(
    f"Converged before max_iter       : "
    f"{converged_before_limit}"
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
# 28. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 22: WEIGHTED LOGISTIC REGRESSION\n"
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
        "Stage 22 trains the first real Fire-Risk classifier. "
        "Weighted multiclass Logistic Regression is fitted on Train "
        "only using the class_weight='balanced' strategy established "
        "in Stage 21. Validation is used only for development "
        "evaluation.\n\n"
    )

    report.write(
        "FIXED CONFIGURATION\n"
        "-------------------\n"
    )

    for key, value in (
        LOGISTIC_PARAMS.items()
    ):
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
        "BASELINE COMPARISON\n"
        "-------------------\n"
    )

    report.write(
        f"Most-Frequent Validation Macro-F1: "
        f"{float(most_frequent_baseline['Macro_F1']):.6f}\n"
    )

    report.write(
        f"Stratified Validation Macro-F1: "
        f"{float(stratified_baseline['Macro_F1']):.6f}\n"
    )

    report.write(
        f"Logistic Regression Validation Macro-F1: "
        f"{float(validation_metrics['Macro_F1']):.6f}\n"
    )

    report.write(
        f"Macro-F1 change vs Most-Frequent: "
        f"{macro_f1_improvement_vs_most_frequent:+.6f}\n"
    )

    report.write(
        f"Macro-F1 change vs Stratified: "
        f"{macro_f1_improvement_vs_stratified:+.6f}\n\n"
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
        "INTERPRETATION\n"
        "--------------\n"
    )

    report.write(
        "Because the target is severely imbalanced, raw Accuracy "
        "must not be used as the main success criterion. Macro-F1 "
        "is the primary comparison metric, while Balanced Accuracy "
        "and Warning/Critical recall are especially important for "
        "fire-risk detection. The model remains an untuned "
        "development candidate; later tree-based classifiers may "
        "capture nonlinear risk patterns more effectively.\n\n"
    )

    report.write(
        "METHODOLOGY PROTECTION\n"
        "----------------------\n"
    )

    report.write(
        "- Model fitted on Train only.\n"
        "- class_weight='balanced' derived from Train only.\n"
        "- Stage-21 class weights reproduced exactly.\n"
        "- No SMOTE or resampling used.\n"
        "- No hyperparameter tuning performed.\n"
        "- No decision-threshold tuning performed.\n"
        "- Final Test rows were not loaded.\n"
        "- Final Test remains frozen until Stage 31.\n\n"
    )

    report.write(
        "NEXT STAGE\n"
        "----------\n"
    )

    report.write(
        "Stage 23 will train a weighted Random Forest classifier "
        "using the same frozen features, Train/Validation split, "
        "and Train-only imbalance strategy.\n\n"
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
        f"STAGE 22 STATUS: "
        f"{overall_status}\n"
    )


# ============================================================
# 29. COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    f"STAGE 22 WEIGHTED LOGISTIC REGRESSION: "
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
        "Stage 22 failed. Review "
        "22_logistic_regression_report.txt."
    )
