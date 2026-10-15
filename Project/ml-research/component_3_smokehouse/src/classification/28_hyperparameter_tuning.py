from pathlib import Path
import hashlib
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from xgboost import XGBClassifier


# ============================================================
# COMPONENT 3 - STAGE 28:
# BOUNDED CLASSIFICATION HYPERPARAMETER TUNING
# ============================================================
#
# Shortlisted model families from Stage 27:
#   1) XGBoost
#   2) Logistic Regression
#   3) Gradient Boosting
#
# PURPOSE
# -------
# Perform a deliberately bounded development-stage search over
# predeclared configurations for ONLY the shortlisted families.
#
# DEVELOPMENT METHODOLOGY
# -----------------------
# - Fit every trial on TRAIN only.
# - Evaluate every trial on VALIDATION only.
# - Validation is used for bounded hyperparameter selection.
# - No Test rows are loaded.
# - No SMOTE / resampling.
# - No decision-threshold tuning.
# - Logistic Regression uses class_weight="balanced".
# - Gradient Boosting and XGBoost use the exact Train-only
#   sample weights generated in Stage 21.
#
# IMPORTANT
# ---------
# Stage 28 does NOT make the final development-stage classifier
# selection. It identifies the best tuned configuration within
# each shortlisted family. Stage 29 makes the formal selection.
#
# Stage 30 later performs formal group-aware robustness analysis.
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

STAGE27_ARTIFACT_DIR = (
    ROOT
    / "artifacts"
    / "classification"
    / "comparison"
)

TUNED_MODEL_DIR = (
    ROOT
    / "artifacts"
    / "models"
    / "classification"
    / "tuned"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TUNED_MODEL_DIR.mkdir(
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

TEST_PATH = (
    SPLIT_DIR
    / "Component3_Test.csv"
)

STAGE21_WEIGHTS_PATH = (
    REPORT_DIR
    / "21_class_weights.csv"
)

STAGE21_SAMPLE_WEIGHTS_PATH = (
    STAGE21_ARTIFACT_DIR
    / "train_classification_sample_weights.csv"
)

STAGE27_SHORTLIST_PATH = (
    STAGE27_ARTIFACT_DIR
    / "classification_shortlist.json"
)

STAGE27_RANKING_PATH = (
    REPORT_DIR
    / "27_classification_real_model_ranking.csv"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "28_classification_tuning_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "28_classification_tuning_checks.csv"
)

ALL_TRIALS_PATH = (
    REPORT_DIR
    / "28_classification_tuning_all_trials.csv"
)

BEST_BY_FAMILY_PATH = (
    REPORT_DIR
    / "28_classification_tuning_best_by_family.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "28_classification_tuning_summary.json"
)

TUNING_ARTIFACT_PATH = (
    TUNED_MODEL_DIR
    / "classification_tuning_manifest.json"
)


# ============================================================
# 4. CONSTANTS
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

LABEL_TO_INT = {
    "Normal": 0,
    "Warning": 1,
    "Critical": 2,
}

INT_TO_LABEL = {
    value: key
    for key, value in LABEL_TO_INT.items()
}

EXPECTED_SHORTLIST = {
    "XGBoost",
    "Logistic_Regression",
    "Gradient_Boosting",
}

SMOKE_TEST = (
    os.environ.get(
        "STAGE28_SMOKE_TEST",
        "0",
    )
    ==
    "1"
)


# ============================================================
# 5. START
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 28: BOUNDED CLASSIFICATION TUNING")
print("=" * 78)


# ============================================================
# 6. REQUIRED FILE CHECKS
# ============================================================

required_files = [
    X_TRAIN_PATH,
    X_VALIDATION_PATH,
    Y_TRAIN_PATH,
    Y_VALIDATION_PATH,
    TEST_PATH,
    STAGE21_WEIGHTS_PATH,
    STAGE21_SAMPLE_WEIGHTS_PATH,
    STAGE27_SHORTLIST_PATH,
    STAGE27_RANKING_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("\nTrain and Validation model-ready data found.")
print("Stage 21 weights and Stage 27 shortlist found.")
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

stage21_weights_df = pd.read_csv(
    STAGE21_WEIGHTS_PATH
)

stage21_sample_weights_df = pd.read_csv(
    STAGE21_SAMPLE_WEIGHTS_PATH
)

stage27_ranking = pd.read_csv(
    STAGE27_RANKING_PATH
)

with open(
    STAGE27_SHORTLIST_PATH,
    "r",
    encoding="utf-8",
) as file:
    stage27_shortlist = json.load(
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
# 10. VERIFY STAGE 27 SHORTLIST
# ============================================================

shortlisted_models = set(
    stage27_shortlist[
        "shortlisted_models"
    ]
)

shortlist_matches_expected = bool(
    shortlisted_models
    ==
    EXPECTED_SHORTLIST
)

if not shortlist_matches_expected:
    raise RuntimeError(
        "Stage-27 shortlist differs from the expected locked shortlist.\n"
        f"Expected: {sorted(EXPECTED_SHORTLIST)}\n"
        f"Found: {sorted(shortlisted_models)}"
    )


# ============================================================
# 11. VERIFY TRAIN-ONLY SAMPLE WEIGHTS
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
        stage21_sample_weights_df.columns
    )
)

if missing_weight_columns:
    raise RuntimeError(
        "Stage-21 sample-weight file missing columns: "
        f"{sorted(missing_weight_columns)}"
    )

if len(
    stage21_sample_weights_df
) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError(
        "Stage-21 sample-weight row count does not match Train."
    )

sample_weight_indices_match = bool(
    np.array_equal(
        stage21_sample_weights_df[
            "Train_Row_Index"
        ]
        .to_numpy(
            dtype=int
        ),
        np.arange(
            EXPECTED_TRAIN_ROWS
        ),
    )
)

sample_weight_labels_match = bool(
    np.array_equal(
        stage21_sample_weights_df[
            "Fire_Risk_Label"
        ]
        .astype(
            str
        )
        .to_numpy(),
        y_train,
    )
)

train_sample_weights = (
    stage21_sample_weights_df[
        "Sample_Weight"
    ]
    .to_numpy(
        dtype=float
    )
)

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
    for _, row in stage21_weights_df.iterrows()
}

reconstructed_weights = np.array(
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
        reconstructed_weights,
        rtol=1e-12,
        atol=1e-12,
    )
)

if not (
    sample_weight_indices_match
    and
    sample_weight_labels_match
    and
    sample_weight_values_match
):
    raise RuntimeError(
        "Stage-21 sample-weight verification failed."
    )


# ============================================================
# 12. XGBOOST ENCODED TARGET
# ============================================================

y_train_encoded = np.array(
    [
        LABEL_TO_INT[
            label
        ]
        for label in y_train
    ],
    dtype=int,
)

y_validation_encoded = np.array(
    [
        LABEL_TO_INT[
            label
        ]
        for label in y_validation
    ],
    dtype=int,
)


# ============================================================
# 13. FIXED BOUNDED SEARCH SPACES
# ============================================================
#
# The first configuration for each family reproduces or closely
# reproduces its Stage 22/25/26 untuned setup so Stage 28 can
# verify continuity before testing nearby alternatives.
#
# These are manually bounded research configurations, not an
# exhaustive search and not an automated grid/random search.

LOGISTIC_TRIALS = [
    {
        "Trial": "LR_01",
        "C": 1.00,
    },
    {
        "Trial": "LR_02",
        "C": 0.05,
    },
    {
        "Trial": "LR_03",
        "C": 0.10,
    },
    {
        "Trial": "LR_04",
        "C": 0.25,
    },
    {
        "Trial": "LR_05",
        "C": 0.50,
    },
    {
        "Trial": "LR_06",
        "C": 2.00,
    },
    {
        "Trial": "LR_07",
        "C": 5.00,
    },
    {
        "Trial": "LR_08",
        "C": 10.00,
    },
]

GB_TRIALS = [
    {
        "Trial": "GB_01",
        "learning_rate": 0.05,
        "n_estimators": 200,
        "max_depth": 3,
        "min_samples_leaf": 1,
        "subsample": 1.00,
    },
    {
        "Trial": "GB_02",
        "learning_rate": 0.03,
        "n_estimators": 300,
        "max_depth": 3,
        "min_samples_leaf": 1,
        "subsample": 1.00,
    },
    {
        "Trial": "GB_03",
        "learning_rate": 0.03,
        "n_estimators": 250,
        "max_depth": 2,
        "min_samples_leaf": 2,
        "subsample": 1.00,
    },
    {
        "Trial": "GB_04",
        "learning_rate": 0.05,
        "n_estimators": 150,
        "max_depth": 2,
        "min_samples_leaf": 2,
        "subsample": 1.00,
    },
    {
        "Trial": "GB_05",
        "learning_rate": 0.03,
        "n_estimators": 300,
        "max_depth": 2,
        "min_samples_leaf": 4,
        "subsample": 0.90,
    },
    {
        "Trial": "GB_06",
        "learning_rate": 0.05,
        "n_estimators": 200,
        "max_depth": 2,
        "min_samples_leaf": 4,
        "subsample": 0.90,
    },
    {
        "Trial": "GB_07",
        "learning_rate": 0.02,
        "n_estimators": 350,
        "max_depth": 3,
        "min_samples_leaf": 2,
        "subsample": 0.90,
    },
    {
        "Trial": "GB_08",
        "learning_rate": 0.03,
        "n_estimators": 250,
        "max_depth": 3,
        "min_samples_leaf": 4,
        "subsample": 0.85,
    },
]

XGB_TRIALS = [
    {
        "Trial": "XGB_01",
        "learning_rate": 0.05,
        "n_estimators": 300,
        "max_depth": 3,
        "min_child_weight": 1.0,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "Trial": "XGB_02",
        "learning_rate": 0.03,
        "n_estimators": 400,
        "max_depth": 3,
        "min_child_weight": 1.0,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "Trial": "XGB_03",
        "learning_rate": 0.03,
        "n_estimators": 350,
        "max_depth": 2,
        "min_child_weight": 1.0,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "Trial": "XGB_04",
        "learning_rate": 0.05,
        "n_estimators": 250,
        "max_depth": 2,
        "min_child_weight": 2.0,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "Trial": "XGB_05",
        "learning_rate": 0.03,
        "n_estimators": 400,
        "max_depth": 3,
        "min_child_weight": 2.0,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "gamma": 0.05,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
    },
    {
        "Trial": "XGB_06",
        "learning_rate": 0.05,
        "n_estimators": 300,
        "max_depth": 3,
        "min_child_weight": 2.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.10,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
    },
    {
        "Trial": "XGB_07",
        "learning_rate": 0.02,
        "n_estimators": 450,
        "max_depth": 3,
        "min_child_weight": 1.0,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
        "gamma": 0.0,
        "reg_alpha": 0.05,
        "reg_lambda": 1.5,
    },
    {
        "Trial": "XGB_08",
        "learning_rate": 0.03,
        "n_estimators": 350,
        "max_depth": 4,
        "min_child_weight": 2.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.10,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
    },
]

if SMOKE_TEST:
    LOGISTIC_TRIALS = LOGISTIC_TRIALS[:2]

    GB_TRIALS = [
        {
            **GB_TRIALS[0],
            "n_estimators": 25,
        },
        {
            **GB_TRIALS[3],
            "n_estimators": 20,
        },
    ]

    XGB_TRIALS = [
        {
            **XGB_TRIALS[0],
            "n_estimators": 35,
        },
        {
            **XGB_TRIALS[3],
            "n_estimators": 30,
        },
    ]


# ============================================================
# 14. METRIC HELPERS
# ============================================================

def compute_metrics(
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
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
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
        "Warning_Precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=[
                    "Warning"
                ],
                average=None,
                zero_division=0,
            )[
                0
            ]
        ),
        "Warning_Recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=[
                    "Warning"
                ],
                average=None,
                zero_division=0,
            )[
                0
            ]
        ),
        "Warning_F1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[
                    "Warning"
                ],
                average=None,
                zero_division=0,
            )[
                0
            ]
        ),
        "Critical_Precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=[
                    "Critical"
                ],
                average=None,
                zero_division=0,
            )[
                0
            ]
        ),
        "Critical_Recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=[
                    "Critical"
                ],
                average=None,
                zero_division=0,
            )[
                0
            ]
        ),
        "Critical_F1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[
                    "Critical"
                ],
                average=None,
                zero_division=0,
            )[
                0
            ]
        ),
    }


def add_trial_row(
    rows,
    family,
    trial_name,
    params,
    train_metrics,
    val_metrics,
    elapsed_seconds,
):
    rows.append({
        "Model_Family": family,
        "Trial": trial_name,
        "Validation_Macro_F1": (
            val_metrics[
                "Macro_F1"
            ]
        ),
        "Validation_Balanced_Accuracy": (
            val_metrics[
                "Balanced_Accuracy"
            ]
        ),
        "Validation_Accuracy": (
            val_metrics[
                "Accuracy"
            ]
        ),
        "Validation_Macro_Precision": (
            val_metrics[
                "Macro_Precision"
            ]
        ),
        "Validation_Macro_Recall": (
            val_metrics[
                "Macro_Recall"
            ]
        ),
        "Validation_Weighted_F1": (
            val_metrics[
                "Weighted_F1"
            ]
        ),
        "Warning_Precision": (
            val_metrics[
                "Warning_Precision"
            ]
        ),
        "Warning_Recall": (
            val_metrics[
                "Warning_Recall"
            ]
        ),
        "Warning_F1": (
            val_metrics[
                "Warning_F1"
            ]
        ),
        "Critical_Precision": (
            val_metrics[
                "Critical_Precision"
            ]
        ),
        "Critical_Recall": (
            val_metrics[
                "Critical_Recall"
            ]
        ),
        "Critical_F1": (
            val_metrics[
                "Critical_F1"
            ]
        ),
        "Train_Macro_F1": (
            train_metrics[
                "Macro_F1"
            ]
        ),
        "Train_Balanced_Accuracy": (
            train_metrics[
                "Balanced_Accuracy"
            ]
        ),
        "Macro_F1_Gap": (
            train_metrics[
                "Macro_F1"
            ]
            -
            val_metrics[
                "Macro_F1"
            ]
        ),
        "Balanced_Accuracy_Gap": (
            train_metrics[
                "Balanced_Accuracy"
            ]
            -
            val_metrics[
                "Balanced_Accuracy"
            ]
        ),
        "Elapsed_Seconds": float(
            elapsed_seconds
        ),
        "Parameters_JSON": json.dumps(
            params,
            sort_keys=True,
        ),
    })


# ============================================================
# 15. TUNE LOGISTIC REGRESSION
# ============================================================

all_trial_rows = []

best_models = {}

print(
    "\n--- LOGISTIC REGRESSION TUNING ---"
)

for config in LOGISTIC_TRIALS:

    trial_name = config[
        "Trial"
    ]

    params = {
        "C": float(
            config[
                "C"
            ]
        ),
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 5000,
        "tol": 1e-4,
        "random_state": 42,
    }

    model = LogisticRegression(
        **params
    )

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    elapsed_seconds = (
        time.perf_counter()
        -
        start_time
    )

    train_pred = model.predict(
        X_train
    )

    val_pred = model.predict(
        X_validation
    )

    train_metrics = compute_metrics(
        y_train,
        train_pred,
    )

    val_metrics = compute_metrics(
        y_validation,
        val_pred,
    )

    add_trial_row(
        all_trial_rows,
        "Logistic_Regression",
        trial_name,
        params,
        train_metrics,
        val_metrics,
        elapsed_seconds,
    )

    print(
        f"{trial_name}: "
        f"Val Macro-F1={val_metrics['Macro_F1']:.4f}, "
        f"BalAcc={val_metrics['Balanced_Accuracy']:.4f}, "
        f"WarningRecall={val_metrics['Warning_Recall']:.4f}, "
        f"CriticalRecall={val_metrics['Critical_Recall']:.4f}"
    )


# ============================================================
# 16. TUNE GRADIENT BOOSTING
# ============================================================

print(
    "\n--- GRADIENT BOOSTING TUNING ---"
)

for config in GB_TRIALS:

    trial_name = config[
        "Trial"
    ]

    params = {
        "loss": "log_loss",
        "learning_rate": float(
            config[
                "learning_rate"
            ]
        ),
        "n_estimators": int(
            config[
                "n_estimators"
            ]
        ),
        "subsample": float(
            config[
                "subsample"
            ]
        ),
        "min_samples_split": 2,
        "min_samples_leaf": int(
            config[
                "min_samples_leaf"
            ]
        ),
        "max_depth": int(
            config[
                "max_depth"
            ]
        ),
        "max_features": None,
        "random_state": 42,
    }

    model = GradientBoostingClassifier(
        **params
    )

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        sample_weight=train_sample_weights,
    )

    elapsed_seconds = (
        time.perf_counter()
        -
        start_time
    )

    train_pred = model.predict(
        X_train
    )

    val_pred = model.predict(
        X_validation
    )

    train_metrics = compute_metrics(
        y_train,
        train_pred,
    )

    val_metrics = compute_metrics(
        y_validation,
        val_pred,
    )

    add_trial_row(
        all_trial_rows,
        "Gradient_Boosting",
        trial_name,
        params,
        train_metrics,
        val_metrics,
        elapsed_seconds,
    )

    print(
        f"{trial_name}: "
        f"Val Macro-F1={val_metrics['Macro_F1']:.4f}, "
        f"BalAcc={val_metrics['Balanced_Accuracy']:.4f}, "
        f"WarningRecall={val_metrics['Warning_Recall']:.4f}, "
        f"CriticalRecall={val_metrics['Critical_Recall']:.4f}"
    )


# ============================================================
# 17. TUNE XGBOOST
# ============================================================

print(
    "\n--- XGBOOST TUNING ---"
)

for config in XGB_TRIALS:

    trial_name = config[
        "Trial"
    ]

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "learning_rate": float(
            config[
                "learning_rate"
            ]
        ),
        "n_estimators": int(
            config[
                "n_estimators"
            ]
        ),
        "max_depth": int(
            config[
                "max_depth"
            ]
        ),
        "min_child_weight": float(
            config[
                "min_child_weight"
            ]
        ),
        "subsample": float(
            config[
                "subsample"
            ]
        ),
        "colsample_bytree": float(
            config[
                "colsample_bytree"
            ]
        ),
        "gamma": float(
            config[
                "gamma"
            ]
        ),
        "reg_alpha": float(
            config[
                "reg_alpha"
            ]
        ),
        "reg_lambda": float(
            config[
                "reg_lambda"
            ]
        ),
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = XGBClassifier(
        **params
    )

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train_encoded,
        sample_weight=train_sample_weights,
    )

    elapsed_seconds = (
        time.perf_counter()
        -
        start_time
    )

    train_pred_encoded = (
        model.predict(
            X_train
        )
        .astype(
            int
        )
    )

    val_pred_encoded = (
        model.predict(
            X_validation
        )
        .astype(
            int
        )
    )

    train_pred = np.array(
        [
            INT_TO_LABEL[
                int(
                    value
                )
            ]
            for value in train_pred_encoded
        ],
        dtype=object,
    )

    val_pred = np.array(
        [
            INT_TO_LABEL[
                int(
                    value
                )
            ]
            for value in val_pred_encoded
        ],
        dtype=object,
    )

    train_metrics = compute_metrics(
        y_train,
        train_pred,
    )

    val_metrics = compute_metrics(
        y_validation,
        val_pred,
    )

    add_trial_row(
        all_trial_rows,
        "XGBoost",
        trial_name,
        params,
        train_metrics,
        val_metrics,
        elapsed_seconds,
    )

    print(
        f"{trial_name}: "
        f"Val Macro-F1={val_metrics['Macro_F1']:.4f}, "
        f"BalAcc={val_metrics['Balanced_Accuracy']:.4f}, "
        f"WarningRecall={val_metrics['Warning_Recall']:.4f}, "
        f"CriticalRecall={val_metrics['Critical_Recall']:.4f}"
    )


# ============================================================
# 18. BUILD ALL-TRIAL TABLE
# ============================================================

all_trials_df = pd.DataFrame(
    all_trial_rows
)

all_trials_df = (
    all_trials_df
    .sort_values(
        by=[
            "Validation_Macro_F1",
            "Validation_Balanced_Accuracy",
            "Critical_Recall",
            "Warning_Recall",
            "Macro_F1_Gap",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)

all_trials_df[
    "Overall_Tuning_Rank"
] = np.arange(
    1,
    len(
        all_trials_df
    )
    +
    1,
)

rank_col = all_trials_df.pop(
    "Overall_Tuning_Rank"
)

all_trials_df.insert(
    0,
    "Overall_Tuning_Rank",
    rank_col,
)

all_trials_df.to_csv(
    ALL_TRIALS_PATH,
    index=False,
)


# ============================================================
# 19. SELECT BEST TRIAL WITHIN EACH FAMILY
# ============================================================
#
# Primary:
#   Validation Macro-F1
#
# Tie breakers:
#   1) Validation Balanced Accuracy
#   2) Critical Recall
#   3) Warning Recall
#   4) smaller Macro-F1 generalization gap
#
# This selects best hyperparameters WITHIN each model family.
# Final model-family selection is Stage 29.

best_family_rows = []

for family in [
    "XGBoost",
    "Logistic_Regression",
    "Gradient_Boosting",
]:

    family_df = (
        all_trials_df[
            all_trials_df[
                "Model_Family"
            ]
            ==
            family
        ]
        .copy()
        .sort_values(
            by=[
                "Validation_Macro_F1",
                "Validation_Balanced_Accuracy",
                "Critical_Recall",
                "Warning_Recall",
                "Macro_F1_Gap",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    best_family_rows.append(
        family_df.iloc[
            0
        ].to_dict()
    )

best_by_family_df = pd.DataFrame(
    best_family_rows
)

best_by_family_df = (
    best_by_family_df
    .sort_values(
        by=[
            "Validation_Macro_F1",
            "Validation_Balanced_Accuracy",
            "Critical_Recall",
            "Warning_Recall",
            "Macro_F1_Gap",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)

best_by_family_df[
    "Family_Rank"
] = np.arange(
    1,
    len(
        best_by_family_df
    )
    +
    1,
)

family_rank_col = best_by_family_df.pop(
    "Family_Rank"
)

best_by_family_df.insert(
    0,
    "Family_Rank",
    family_rank_col,
)

best_by_family_df.to_csv(
    BEST_BY_FAMILY_PATH,
    index=False,
)


# ============================================================
# 20. RE-FIT + SAVE BEST CONFIGURATION FROM EACH FAMILY
# ============================================================

saved_models = {}

for _, row in best_by_family_df.iterrows():

    family = str(
        row[
            "Model_Family"
        ]
    )

    trial = str(
        row[
            "Trial"
        ]
    )

    params = json.loads(
        row[
            "Parameters_JSON"
        ]
    )

    family_dir = (
        TUNED_MODEL_DIR
        / family.lower()
    )

    family_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        family_dir
        / f"{trial.lower()}_classifier.joblib"
    )

    metadata_path = (
        family_dir
        / f"{trial.lower()}_metadata.json"
    )

    if family == "Logistic_Regression":

        model = LogisticRegression(
            **params
        )

        model.fit(
            X_train,
            y_train,
        )

        label_mapping = None

    elif family == "Gradient_Boosting":

        model = GradientBoostingClassifier(
            **params
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=train_sample_weights,
        )

        label_mapping = None

    elif family == "XGBoost":

        model = XGBClassifier(
            **params
        )

        model.fit(
            X_train,
            y_train_encoded,
            sample_weight=train_sample_weights,
        )

        label_mapping = LABEL_TO_INT

    else:
        raise RuntimeError(
            f"Unexpected family: {family}"
        )

    joblib.dump(
        model,
        model_path,
    )

    with open(
        model_path,
        "rb",
    ) as file:
        model_sha256 = hashlib.sha256(
            file.read()
        ).hexdigest()

    model_metadata = {
        "stage": (
            "28_bounded_classification_tuning"
        ),
        "model_family": (
            family
        ),
        "selected_trial_within_family": (
            trial
        ),
        "parameters": (
            params
        ),
        "label_mapping": (
            label_mapping
        ),
        "validation_macro_f1": float(
            row[
                "Validation_Macro_F1"
            ]
        ),
        "validation_balanced_accuracy": float(
            row[
                "Validation_Balanced_Accuracy"
            ]
        ),
        "warning_recall": float(
            row[
                "Warning_Recall"
            ]
        ),
        "critical_recall": float(
            row[
                "Critical_Recall"
            ]
        ),
        "warning_f1": float(
            row[
                "Warning_F1"
            ]
        ),
        "critical_f1": float(
            row[
                "Critical_F1"
            ]
        ),
        "macro_f1_gap": float(
            row[
                "Macro_F1_Gap"
            ]
        ),
        "fit_scope": (
            "TRAIN_ONLY"
        ),
        "validation_role": (
            "BOUNDED_HYPERPARAMETER_SELECTION"
        ),
        "test_loaded": False,
        "model_sha256": (
            model_sha256
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model_metadata,
            file,
            indent=2,
        )

    saved_models[
        family
    ] = {
        "trial": (
            trial
        ),
        "model_path": str(
            model_path
        ),
        "metadata_path": str(
            metadata_path
        ),
        "model_sha256": (
            model_sha256
        ),
    }


# ============================================================
# 21. UNTUNED REFERENCE METRICS FROM STAGE 27
# ============================================================

untuned_reference = {}

for family in EXPECTED_SHORTLIST:

    row = stage27_ranking[
        stage27_ranking[
            "Model"
        ]
        ==
        family
    ]

    if len(
        row
    ) != 1:
        raise RuntimeError(
            f"Could not uniquely find Stage-27 ranking for {family}."
        )

    row = row.iloc[
        0
    ]

    untuned_reference[
        family
    ] = {
        "Validation_Macro_F1": float(
            row[
                "Validation_Macro_F1"
            ]
        ),
        "Validation_Balanced_Accuracy": float(
            row[
                "Validation_Balanced_Accuracy"
            ]
        ),
        "Warning_Recall": float(
            row[
                "Warning_Recall"
            ]
        ),
        "Critical_Recall": float(
            row[
                "Critical_Recall"
            ]
        ),
    }


# ============================================================
# 22. ADD IMPROVEMENT COLUMNS
# ============================================================

macro_improvements = []

balacc_improvements = []

for _, row in best_by_family_df.iterrows():

    family = str(
        row[
            "Model_Family"
        ]
    )

    macro_improvements.append(
        float(
            row[
                "Validation_Macro_F1"
            ]
        )
        -
        untuned_reference[
            family
        ][
            "Validation_Macro_F1"
        ]
    )

    balacc_improvements.append(
        float(
            row[
                "Validation_Balanced_Accuracy"
            ]
        )
        -
        untuned_reference[
            family
        ][
            "Validation_Balanced_Accuracy"
        ]
    )

best_by_family_df[
    "Macro_F1_Change_vs_Untuned"
] = macro_improvements

best_by_family_df[
    "Balanced_Accuracy_Change_vs_Untuned"
] = balacc_improvements

best_by_family_df.to_csv(
    BEST_BY_FAMILY_PATH,
    index=False,
)


# ============================================================
# 23. CONTINUITY CHECKS FOR FIRST TRIALS
# ============================================================

def get_trial_row(
    family,
    trial,
):
    row = all_trials_df[
        (
            all_trials_df[
                "Model_Family"
            ]
            ==
            family
        )
        &
        (
            all_trials_df[
                "Trial"
            ]
            ==
            trial
        )
    ]

    if len(
        row
    ) != 1:
        raise RuntimeError(
            f"Could not uniquely locate {family} {trial}."
        )

    return row.iloc[
        0
    ]


lr01 = get_trial_row(
    "Logistic_Regression",
    "LR_01",
)

gb01 = get_trial_row(
    "Gradient_Boosting",
    "GB_01",
)

xgb01 = get_trial_row(
    "XGBoost",
    "XGB_01",
)

# Small floating differences are acceptable.
lr_reproduces_stage22 = bool(
    np.isclose(
        float(
            lr01[
                "Validation_Macro_F1"
            ]
        ),
        untuned_reference[
            "Logistic_Regression"
        ][
            "Validation_Macro_F1"
        ],
        atol=5e-4,
        rtol=0,
    )
)

gb_reproduces_stage25 = bool(
    np.isclose(
        float(
            gb01[
                "Validation_Macro_F1"
            ]
        ),
        untuned_reference[
            "Gradient_Boosting"
        ][
            "Validation_Macro_F1"
        ],
        atol=5e-4,
        rtol=0,
    )
)

xgb_reproduces_stage26 = bool(
    np.isclose(
        float(
            xgb01[
                "Validation_Macro_F1"
            ]
        ),
        untuned_reference[
            "XGBoost"
        ][
            "Validation_Macro_F1"
        ],
        atol=5e-4,
        rtol=0,
    )
)

if SMOKE_TEST:
    # Smoke-test data are synthetic and the Stage-27 reference metrics
    # are mocked, so continuity against the user's real Stage-22/25/26
    # metrics is not meaningful in this local functional test.
    lr_reproduces_stage22 = True
    gb_reproduces_stage25 = True
    xgb_reproduces_stage26 = True


# ============================================================
# 24. CHECKS
# ============================================================

expected_trial_count = (
    len(
        LOGISTIC_TRIALS
    )
    +
    len(
        GB_TRIALS
    )
    +
    len(
        XGB_TRIALS
    )
)

trial_count_valid = bool(
    len(
        all_trials_df
    )
    ==
    expected_trial_count
)

all_metrics_finite = bool(
    np.isfinite(
        all_trials_df[
            [
                "Validation_Macro_F1",
                "Validation_Balanced_Accuracy",
                "Validation_Accuracy",
                "Warning_Recall",
                "Critical_Recall",
                "Warning_F1",
                "Critical_F1",
                "Train_Macro_F1",
                "Macro_F1_Gap",
                "Balanced_Accuracy_Gap",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all()
)

all_families_have_best = bool(
    set(
        best_by_family_df[
            "Model_Family"
        ]
    )
    ==
    EXPECTED_SHORTLIST
)

all_saved = bool(
    set(
        saved_models.keys()
    )
    ==
    EXPECTED_SHORTLIST
    and
    all(
        Path(
            info[
                "model_path"
            ]
        ).exists()
        for info in saved_models.values()
    )
)

checks = pd.DataFrame(
    [
        {
            "Check": (
                "Stage-27 shortlist exactly reproduced"
            ),
            "Status": (
                "PASS"
                if shortlist_matches_expected
                else "FAIL"
            ),
            "Details": (
                ", ".join(
                    sorted(
                        shortlisted_models
                    )
                )
            ),
        },
        {
            "Check": (
                "Stage-21 Train sample weights verified"
            ),
            "Status": (
                "PASS"
                if (
                    sample_weight_indices_match
                    and
                    sample_weight_labels_match
                    and
                    sample_weight_values_match
                )
                else "FAIL"
            ),
            "Details": (
                "Indices, labels, and values match."
            ),
        },
        {
            "Check": (
                "Expected bounded trial count"
            ),
            "Status": (
                "PASS"
                if trial_count_valid
                else "FAIL"
            ),
            "Details": (
                f"Trials={len(all_trials_df)}"
            ),
        },
        {
            "Check": (
                "All tuning metrics finite"
            ),
            "Status": (
                "PASS"
                if all_metrics_finite
                else "FAIL"
            ),
            "Details": (
                "Train/Validation metrics verified."
            ),
        },
        {
            "Check": (
                "Best configuration identified for all three families"
            ),
            "Status": (
                "PASS"
                if all_families_have_best
                else "FAIL"
            ),
            "Details": (
                ", ".join(
                    best_by_family_df[
                        "Model_Family"
                    ].tolist()
                )
            ),
        },
        {
            "Check": (
                "LR_01 reproduces Stage-22 untuned Macro-F1"
            ),
            "Status": (
                "PASS"
                if lr_reproduces_stage22
                else "FAIL"
            ),
            "Details": (
                f"LR_01={float(lr01['Validation_Macro_F1']):.6f}; "
                f"Stage22={untuned_reference['Logistic_Regression']['Validation_Macro_F1']:.6f}"
            ),
        },
        {
            "Check": (
                "GB_01 reproduces Stage-25 untuned Macro-F1"
            ),
            "Status": (
                "PASS"
                if gb_reproduces_stage25
                else "FAIL"
            ),
            "Details": (
                f"GB_01={float(gb01['Validation_Macro_F1']):.6f}; "
                f"Stage25={untuned_reference['Gradient_Boosting']['Validation_Macro_F1']:.6f}"
            ),
        },
        {
            "Check": (
                "XGB_01 reproduces Stage-26 untuned Macro-F1"
            ),
            "Status": (
                "PASS"
                if xgb_reproduces_stage26
                else "FAIL"
            ),
            "Details": (
                f"XGB_01={float(xgb01['Validation_Macro_F1']):.6f}; "
                f"Stage26={untuned_reference['XGBoost']['Validation_Macro_F1']:.6f}"
            ),
        },
        {
            "Check": (
                "Best tuned model artifact saved for each family"
            ),
            "Status": (
                "PASS"
                if all_saved
                else "FAIL"
            ),
            "Details": (
                ", ".join(
                    saved_models.keys()
                )
            ),
        },
        {
            "Check": (
                "No SMOTE or resampling"
            ),
            "Status": "PASS",
            "Details": (
                "Only class/sample weighting used."
            ),
        },
        {
            "Check": (
                "No threshold tuning"
            ),
            "Status": "PASS",
            "Details": (
                "Default multiclass decision rule retained."
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
# 25. PROVISIONAL TUNING LEADER
# ============================================================
#
# This is NOT the Stage-29 final selection. It is only the best
# tuned candidate according to Stage-28 Validation ranking.

provisional_leader = (
    best_by_family_df.iloc[
        0
    ]
)

provisional_family = str(
    provisional_leader[
        "Model_Family"
    ]
)

provisional_trial = str(
    provisional_leader[
        "Trial"
    ]
)


# ============================================================
# 26. SUMMARY / MANIFEST
# ============================================================

summary = {
    "stage": (
        "28_bounded_classification_hyperparameter_tuning"
    ),
    "overall_status": (
        overall_status
    ),
    "methodology": {
        "fit_data": (
            "TRAIN_ONLY"
        ),
        "selection_data": (
            "VALIDATION_ONLY"
        ),
        "primary_metric": (
            "Validation Macro-F1"
        ),
        "tie_breakers": [
            "Validation Balanced Accuracy",
            "Critical Recall",
            "Warning Recall",
            "Smaller Train-Validation Macro-F1 Gap",
        ],
        "group_cross_validation_used": False,
        "group_cross_validation_stage": 30,
        "smote_used": False,
        "threshold_tuning_used": False,
        "test_loaded": False,
    },
    "trial_counts": {
        "Logistic_Regression": len(
            LOGISTIC_TRIALS
        ),
        "Gradient_Boosting": len(
            GB_TRIALS
        ),
        "XGBoost": len(
            XGB_TRIALS
        ),
        "total": int(
            len(
                all_trials_df
            )
        ),
    },
    "best_by_family": [
        {
            "family": str(
                row[
                    "Model_Family"
                ]
            ),
            "trial": str(
                row[
                    "Trial"
                ]
            ),
            "validation_macro_f1": float(
                row[
                    "Validation_Macro_F1"
                ]
            ),
            "validation_balanced_accuracy": float(
                row[
                    "Validation_Balanced_Accuracy"
                ]
            ),
            "warning_recall": float(
                row[
                    "Warning_Recall"
                ]
            ),
            "critical_recall": float(
                row[
                    "Critical_Recall"
                ]
            ),
            "warning_f1": float(
                row[
                    "Warning_F1"
                ]
            ),
            "critical_f1": float(
                row[
                    "Critical_F1"
                ]
            ),
            "macro_f1_gap": float(
                row[
                    "Macro_F1_Gap"
                ]
            ),
            "macro_f1_change_vs_untuned": float(
                row[
                    "Macro_F1_Change_vs_Untuned"
                ]
            ),
            "parameters": json.loads(
                row[
                    "Parameters_JSON"
                ]
            ),
        }
        for _, row in (
            best_by_family_df.iterrows()
        )
    ],
    "provisional_tuning_leader": {
        "family": (
            provisional_family
        ),
        "trial": (
            provisional_trial
        ),
        "validation_macro_f1": float(
            provisional_leader[
                "Validation_Macro_F1"
            ]
        ),
        "validation_balanced_accuracy": float(
            provisional_leader[
                "Validation_Balanced_Accuracy"
            ]
        ),
        "warning_recall": float(
            provisional_leader[
                "Warning_Recall"
            ]
        ),
        "critical_recall": float(
            provisional_leader[
                "Critical_Recall"
            ]
        ),
    },
    "saved_models": (
        saved_models
    ),
    "next_stage": (
        "29_final_classification_selection"
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

with open(
    TUNING_ARTIFACT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=2,
    )


# ============================================================
# 27. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- BEST TUNED CONFIGURATION BY FAMILY ---"
)

display_columns = [
    "Family_Rank",
    "Model_Family",
    "Trial",
    "Validation_Macro_F1",
    "Validation_Balanced_Accuracy",
    "Warning_Recall",
    "Critical_Recall",
    "Warning_F1",
    "Critical_F1",
    "Macro_F1_Gap",
    "Macro_F1_Change_vs_Untuned",
]

print(
    best_by_family_df[
        display_columns
    ]
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\n--- BEST PARAMETERS ---"
)

for _, row in (
    best_by_family_df.iterrows()
):
    print(
        f"\n{row['Model_Family']} / {row['Trial']}"
    )

    params = json.loads(
        row[
            "Parameters_JSON"
        ]
    )

    for key, value in (
        params.items()
    ):
        print(
            f"  {key:<22}: {value}"
        )


print(
    "\n--- PROVISIONAL STAGE-28 LEADER ---"
)

print(
    f"Family                          : "
    f"{provisional_family}"
)

print(
    f"Trial                           : "
    f"{provisional_trial}"
)

print(
    f"Validation Macro-F1             : "
    f"{float(provisional_leader['Validation_Macro_F1']):.4f}"
)

print(
    f"Validation Balanced Accuracy    : "
    f"{float(provisional_leader['Validation_Balanced_Accuracy']):.4f}"
)

print(
    f"Warning Recall                  : "
    f"{float(provisional_leader['Warning_Recall']):.4f}"
)

print(
    f"Critical Recall                 : "
    f"{float(provisional_leader['Critical_Recall']):.4f}"
)

print(
    "\nNOTE: Stage 29 makes the formal development-stage final selection."
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
        "COMPONENT 3 - STAGE 28: "
        "BOUNDED CLASSIFICATION HYPERPARAMETER TUNING\n"
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
        "Stage 28 performs bounded hyperparameter tuning on the "
        "three model families formally shortlisted in Stage 27: "
        "XGBoost, Logistic Regression, and Gradient Boosting. "
        "Every trial is fitted on Train only and evaluated on "
        "Validation only. Final Test remains untouched.\n\n"
    )

    report.write(
        "WHY BOUNDED TUNING\n"
        "------------------\n"
    )

    report.write(
        "The search is intentionally limited to a small, predefined "
        "set of scientifically reasonable nearby configurations. "
        "This avoids an uncontrolled search over the same Validation "
        "set while still allowing each shortlisted family to improve "
        "its development-stage configuration.\n\n"
    )

    report.write(
        "SELECTION RULE WITHIN EACH FAMILY\n"
        "---------------------------------\n"
    )

    report.write(
        "Primary: Validation Macro-F1.\n"
        "Tie breakers: Validation Balanced Accuracy, Critical Recall, "
        "Warning Recall, then smaller Train-to-Validation Macro-F1 gap.\n\n"
    )

    report.write(
        "BEST TUNED CONFIGURATION BY FAMILY\n"
        "----------------------------------\n"
    )

    report.write(
        best_by_family_df[
            display_columns
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
        "PROVISIONAL TUNING LEADER\n"
        "-------------------------\n"
    )

    report.write(
        f"Family: {provisional_family}\n"
        f"Trial: {provisional_trial}\n"
        f"Validation Macro-F1: "
        f"{float(provisional_leader['Validation_Macro_F1']):.6f}\n"
        f"Validation Balanced Accuracy: "
        f"{float(provisional_leader['Validation_Balanced_Accuracy']):.6f}\n"
        f"Warning Recall: "
        f"{float(provisional_leader['Warning_Recall']):.6f}\n"
        f"Critical Recall: "
        f"{float(provisional_leader['Critical_Recall']):.6f}\n\n"
    )

    report.write(
        "IMPORTANT: This is not yet the formal final classifier. "
        "Stage 29 performs final development-stage selection using "
        "the tuned evidence across all three shortlisted families.\n\n"
    )

    report.write(
        "METHODOLOGY PROTECTION\n"
        "----------------------\n"
    )

    report.write(
        "- Only Stage-27 shortlisted families were tuned.\n"
        "- Every trial fitted on Train only.\n"
        "- Validation used for bounded hyperparameter selection.\n"
        "- Logistic Regression used class_weight='balanced'.\n"
        "- Gradient Boosting and XGBoost used exact Stage-21 "
        "Train-only sample weights.\n"
        "- No SMOTE or resampling performed.\n"
        "- No decision-threshold tuning performed.\n"
        "- No group cross-validation performed yet; reserved for "
        "Stage 30.\n"
        "- Final Test rows were not loaded.\n"
        "- Final Test remains frozen until Stage 31.\n\n"
    )

    report.write(
        "NEXT STAGE\n"
        "----------\n"
    )

    report.write(
        "Stage 29 will formally select the final development-stage "
        "classification model from the three best tuned family "
        "configurations. It will not use Final Test data.\n\n"
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
        f"STAGE 28 STATUS: "
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
    f"STAGE 28 BOUNDED CLASSIFICATION TUNING: "
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
    f"\nTuned model artifacts saved in:\n"
    f"{TUNED_MODEL_DIR}"
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 28 failed. Review "
        "28_classification_tuning_report.txt."
    )
