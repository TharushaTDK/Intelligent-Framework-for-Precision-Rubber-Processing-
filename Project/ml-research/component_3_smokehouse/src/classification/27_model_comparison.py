from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# COMPONENT 3 - STAGE 27:
# CLASSIFICATION MODEL COMPARISON & SHORTLISTING
# ============================================================
#
# Purpose:
#   Formally compare all completed classification experiments
#   and shortlist the strongest development-stage candidates
#   for bounded hyperparameter tuning in Stage 28.
#
# IMPORTANT:
#   - No model is trained.
#   - No hyperparameter tuning is performed.
#   - No threshold tuning is performed.
#   - Final Test data are NOT loaded.
#
# Selection hierarchy:
#
#   PRIMARY:
#       Validation Macro-F1
#
#   SUPPORTING:
#       Validation Balanced Accuracy
#       Warning Recall
#       Critical Recall
#       Warning F1
#       Critical F1
#       Train -> Validation Macro-F1 gap
#       Train -> Validation Balanced Accuracy gap
#
# Shortlist rule:
#   1. Rank all real classifiers by Validation Macro-F1.
#   2. Retain the TOP THREE real classifiers.
#   3. A shortlisted model must have non-zero Validation recall
#      for BOTH Warning and Critical.
#
# Why top three:
#   Classification has a stronger safety trade-off than regression.
#   Macro-F1 alone can hide materially different Warning/Critical
#   behavior. Keeping the top three lets Stage 28 investigate the
#   strongest overall model plus two competitive alternatives
#   without reopening weak model families.
#
# Stage 27 does NOT choose the final classifier.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    ROOT
    / "outputs"
    / "reports"
)

SPLIT_DIR = (
    ROOT
    / "data"
    / "splits"
)

ARTIFACT_DIR = (
    ROOT
    / "artifacts"
    / "classification"
    / "comparison"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TEST_PATH = (
    SPLIT_DIR
    / "Component3_Test.csv"
)


# ============================================================
# 2. INPUT REPORTS
# ============================================================

INPUTS = {
    "Baseline": {
        "metrics": (
            REPORT_DIR
            / "20_classification_baseline_metrics.csv"
        ),
        "per_class": (
            REPORT_DIR
            / "20_classification_baseline_per_class_metrics.csv"
        ),
    },
    "Logistic_Regression": {
        "metrics": (
            REPORT_DIR
            / "22_logistic_regression_metrics.csv"
        ),
        "per_class": (
            REPORT_DIR
            / "22_logistic_regression_per_class_metrics.csv"
        ),
    },
    "Random_Forest": {
        "metrics": (
            REPORT_DIR
            / "23_random_forest_classification_metrics.csv"
        ),
        "per_class": (
            REPORT_DIR
            / "23_random_forest_classification_per_class_metrics.csv"
        ),
    },
    "Extra_Trees": {
        "metrics": (
            REPORT_DIR
            / "24_extra_trees_classification_metrics.csv"
        ),
        "per_class": (
            REPORT_DIR
            / "24_extra_trees_classification_per_class_metrics.csv"
        ),
    },
    "Gradient_Boosting": {
        "metrics": (
            REPORT_DIR
            / "25_gradient_boosting_classification_metrics.csv"
        ),
        "per_class": (
            REPORT_DIR
            / "25_gradient_boosting_classification_per_class_metrics.csv"
        ),
    },
    "XGBoost": {
        "metrics": (
            REPORT_DIR
            / "26_xgboost_classification_metrics.csv"
        ),
        "per_class": (
            REPORT_DIR
            / "26_xgboost_classification_per_class_metrics.csv"
        ),
    },
}


# ============================================================
# 3. OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "27_classification_comparison_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "27_classification_comparison_checks.csv"
)

FULL_COMPARISON_PATH = (
    REPORT_DIR
    / "27_classification_full_comparison.csv"
)

REAL_RANKING_PATH = (
    REPORT_DIR
    / "27_classification_real_model_ranking.csv"
)

SHORTLIST_PATH = (
    REPORT_DIR
    / "27_classification_shortlist.csv"
)

CLASS_RECALL_PATH = (
    REPORT_DIR
    / "27_classification_class_recall_comparison.csv"
)

CLASS_F1_PATH = (
    REPORT_DIR
    / "27_classification_class_f1_comparison.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "27_classification_comparison_summary.json"
)

SHORTLIST_JSON_PATH = (
    ARTIFACT_DIR
    / "classification_shortlist.json"
)


# ============================================================
# 4. CONSTANTS
# ============================================================

REAL_MODELS = [
    "Logistic_Regression",
    "Random_Forest",
    "Extra_Trees",
    "Gradient_Boosting",
    "XGBoost",
]

BASELINE_MODELS = [
    "Dummy_Most_Frequent",
    "Dummy_Stratified",
]

CLASS_ORDER = [
    "Normal",
    "Warning",
    "Critical",
]

EXPECTED_VALIDATION_SUPPORT = {
    "Normal": 3973,
    "Warning": 168,
    "Critical": 18,
}


# ============================================================
# 5. START
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 27: CLASSIFICATION MODEL COMPARISON")
print("=" * 78)


# ============================================================
# 6. FILE EXISTENCE CHECKS
# ============================================================

for source_name, paths in INPUTS.items():
    for file_type, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {source_name} {file_type} file:\n{path}"
            )

if not TEST_PATH.exists():
    raise FileNotFoundError(
        f"Final Test file not found:\n{TEST_PATH}"
    )

print("\nAll Stage 20 and Stage 22-26 result files found.")
print("Final Test dataset existence verified, but contents were NOT loaded.")


# ============================================================
# 7. LOAD REPORT FILES ONLY
# ============================================================

baseline_metrics = pd.read_csv(
    INPUTS[
        "Baseline"
    ][
        "metrics"
    ]
)

baseline_per_class = pd.read_csv(
    INPUTS[
        "Baseline"
    ][
        "per_class"
    ]
)

real_metrics = {}

real_per_class = {}

for model_name in REAL_MODELS:
    real_metrics[
        model_name
    ] = pd.read_csv(
        INPUTS[
            model_name
        ][
            "metrics"
        ]
    )

    real_per_class[
        model_name
    ] = pd.read_csv(
        INPUTS[
            model_name
        ][
            "per_class"
        ]
    )


# ============================================================
# 8. HELPERS
# ============================================================

def get_metric_row(
    df,
    model_name,
    split_name,
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
            split_name
        )
    ]

    if len(
        row
    ) != 1:
        raise RuntimeError(
            f"Expected exactly one {model_name} / {split_name} metric row; "
            f"found {len(row)}."
        )

    return row.iloc[
        0
    ]


def get_class_row(
    df,
    model_name,
    split_name,
    class_name,
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
            split_name
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
            f"Expected exactly one {model_name} / {split_name} / "
            f"{class_name} per-class row; found {len(row)}."
        )

    return row.iloc[
        0
    ]


# ============================================================
# 9. BUILD BASELINE COMPARISON ROWS
# ============================================================

comparison_rows = []

for model_name in BASELINE_MODELS:
    train_row = get_metric_row(
        baseline_metrics,
        model_name,
        "Train",
    )

    val_row = get_metric_row(
        baseline_metrics,
        model_name,
        "Validation",
    )

    warning_row = get_class_row(
        baseline_per_class,
        model_name,
        "Validation",
        "Warning",
    )

    critical_row = get_class_row(
        baseline_per_class,
        model_name,
        "Validation",
        "Critical",
    )

    comparison_rows.append({
        "Model": model_name,
        "Model_Type": "Baseline",
        "Validation_Accuracy": float(
            val_row[
                "Accuracy"
            ]
        ),
        "Validation_Balanced_Accuracy": float(
            val_row[
                "Balanced_Accuracy"
            ]
        ),
        "Validation_Macro_Precision": float(
            val_row[
                "Macro_Precision"
            ]
        ),
        "Validation_Macro_Recall": float(
            val_row[
                "Macro_Recall"
            ]
        ),
        "Validation_Macro_F1": float(
            val_row[
                "Macro_F1"
            ]
        ),
        "Validation_Weighted_F1": float(
            val_row[
                "Weighted_F1"
            ]
        ),
        "Warning_Precision": float(
            warning_row[
                "Precision"
            ]
        ),
        "Warning_Recall": float(
            warning_row[
                "Recall"
            ]
        ),
        "Warning_F1": float(
            warning_row[
                "F1"
            ]
        ),
        "Critical_Precision": float(
            critical_row[
                "Precision"
            ]
        ),
        "Critical_Recall": float(
            critical_row[
                "Recall"
            ]
        ),
        "Critical_F1": float(
            critical_row[
                "F1"
            ]
        ),
        "Train_Macro_F1": float(
            train_row[
                "Macro_F1"
            ]
        ),
        "Macro_F1_Gap": float(
            train_row[
                "Macro_F1"
            ]
            -
            val_row[
                "Macro_F1"
            ]
        ),
        "Train_Balanced_Accuracy": float(
            train_row[
                "Balanced_Accuracy"
            ]
        ),
        "Balanced_Accuracy_Gap": float(
            train_row[
                "Balanced_Accuracy"
            ]
            -
            val_row[
                "Balanced_Accuracy"
            ]
        ),
    })


# ============================================================
# 10. BUILD REAL-MODEL COMPARISON ROWS
# ============================================================

validation_support_checks = []

for model_name in REAL_MODELS:
    metrics_df = real_metrics[
        model_name
    ]

    per_class_df = real_per_class[
        model_name
    ]

    train_row = get_metric_row(
        metrics_df,
        model_name,
        "Train",
    )

    val_row = get_metric_row(
        metrics_df,
        model_name,
        "Validation",
    )

    warning_row = get_class_row(
        per_class_df,
        model_name,
        "Validation",
        "Warning",
    )

    critical_row = get_class_row(
        per_class_df,
        model_name,
        "Validation",
        "Critical",
    )

    for class_name in CLASS_ORDER:
        class_row = get_class_row(
            per_class_df,
            model_name,
            "Validation",
            class_name,
        )

        validation_support_checks.append({
            "Model": model_name,
            "Class": class_name,
            "Support": int(
                class_row[
                    "Support"
                ]
            ),
            "Expected_Support": int(
                EXPECTED_VALIDATION_SUPPORT[
                    class_name
                ]
            ),
            "Matches": bool(
                int(
                    class_row[
                        "Support"
                    ]
                )
                ==
                EXPECTED_VALIDATION_SUPPORT[
                    class_name
                ]
            ),
        })

    comparison_rows.append({
        "Model": model_name,
        "Model_Type": "Real",
        "Validation_Accuracy": float(
            val_row[
                "Accuracy"
            ]
        ),
        "Validation_Balanced_Accuracy": float(
            val_row[
                "Balanced_Accuracy"
            ]
        ),
        "Validation_Macro_Precision": float(
            val_row[
                "Macro_Precision"
            ]
        ),
        "Validation_Macro_Recall": float(
            val_row[
                "Macro_Recall"
            ]
        ),
        "Validation_Macro_F1": float(
            val_row[
                "Macro_F1"
            ]
        ),
        "Validation_Weighted_F1": float(
            val_row[
                "Weighted_F1"
            ]
        ),
        "Warning_Precision": float(
            warning_row[
                "Precision"
            ]
        ),
        "Warning_Recall": float(
            warning_row[
                "Recall"
            ]
        ),
        "Warning_F1": float(
            warning_row[
                "F1"
            ]
        ),
        "Critical_Precision": float(
            critical_row[
                "Precision"
            ]
        ),
        "Critical_Recall": float(
            critical_row[
                "Recall"
            ]
        ),
        "Critical_F1": float(
            critical_row[
                "F1"
            ]
        ),
        "Train_Macro_F1": float(
            train_row[
                "Macro_F1"
            ]
        ),
        "Macro_F1_Gap": float(
            train_row[
                "Macro_F1"
            ]
            -
            val_row[
                "Macro_F1"
            ]
        ),
        "Train_Balanced_Accuracy": float(
            train_row[
                "Balanced_Accuracy"
            ]
        ),
        "Balanced_Accuracy_Gap": float(
            train_row[
                "Balanced_Accuracy"
            ]
            -
            val_row[
                "Balanced_Accuracy"
            ]
        ),
    })


# ============================================================
# 11. FULL COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df = (
    comparison_df
    .sort_values(
        by=[
            "Validation_Macro_F1",
            "Validation_Balanced_Accuracy",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)

comparison_df.insert(
    0,
    "Overall_Rank",
    np.arange(
        1,
        len(
            comparison_df
        )
        +
        1,
    ),
)

comparison_df.to_csv(
    FULL_COMPARISON_PATH,
    index=False,
)


# ============================================================
# 12. REAL-MODEL RANKING
# ============================================================

real_ranking_df = (
    comparison_df[
        comparison_df[
            "Model_Type"
        ]
        ==
        "Real"
    ]
    .copy()
    .sort_values(
        by=[
            "Validation_Macro_F1",
            "Validation_Balanced_Accuracy",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)

real_ranking_df[
    "Real_Model_Rank"
] = np.arange(
    1,
    len(
        real_ranking_df
    )
    +
    1,
)

# Put the real-model rank near the front.
rank_col = real_ranking_df.pop(
    "Real_Model_Rank"
)

real_ranking_df.insert(
    1,
    "Real_Model_Rank",
    rank_col,
)

real_ranking_df.to_csv(
    REAL_RANKING_PATH,
    index=False,
)


# ============================================================
# 13. FORMAL SHORTLIST
# ============================================================
#
# Top 3 by Validation Macro-F1, provided each detects both
# minority classes with non-zero Validation recall.

top_three = (
    real_ranking_df
    .head(
        3
    )
    .copy()
)

top_three[
    "Warning_Recall_Nonzero"
] = (
    top_three[
        "Warning_Recall"
    ]
    >
    0
)

top_three[
    "Critical_Recall_Nonzero"
] = (
    top_three[
        "Critical_Recall"
    ]
    >
    0
)

top_three[
    "Minority_Coverage_Pass"
] = (
    top_three[
        "Warning_Recall_Nonzero"
    ]
    &
    top_three[
        "Critical_Recall_Nonzero"
    ]
)

shortlist_df = (
    top_three[
        top_three[
            "Minority_Coverage_Pass"
        ]
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

shortlist_df[
    "Shortlist_Rank"
] = np.arange(
    1,
    len(
        shortlist_df
    )
    +
    1,
)

shortlist_rank_col = shortlist_df.pop(
    "Shortlist_Rank"
)

shortlist_df.insert(
    0,
    "Shortlist_Rank",
    shortlist_rank_col,
)

shortlist_df[
    "Stage28_Status"
] = (
    "SHORTLISTED_FOR_TUNING"
)

shortlist_df.to_csv(
    SHORTLIST_PATH,
    index=False,
)


# ============================================================
# 14. CLASS-SPECIFIC COMPARISON TABLES
# ============================================================

recall_rows = []

f1_rows = []

for _, row in real_ranking_df.iterrows():
    recall_rows.append({
        "Model": row[
            "Model"
        ],
        "Normal_Recall": float(
            get_class_row(
                real_per_class[
                    row[
                        "Model"
                    ]
                ],
                row[
                    "Model"
                ],
                "Validation",
                "Normal",
            )[
                "Recall"
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
    })

    f1_rows.append({
        "Model": row[
            "Model"
        ],
        "Normal_F1": float(
            get_class_row(
                real_per_class[
                    row[
                        "Model"
                    ]
                ],
                row[
                    "Model"
                ],
                "Validation",
                "Normal",
            )[
                "F1"
            ]
        ),
        "Warning_F1": float(
            row[
                "Warning_F1"
            ]
        ),
        "Critical_F1": float(
            row[
                "Critical_F1"
            ]
        ),
    })

recall_df = pd.DataFrame(
    recall_rows
)

f1_df = pd.DataFrame(
    f1_rows
)

recall_df.to_csv(
    CLASS_RECALL_PATH,
    index=False,
)

f1_df.to_csv(
    CLASS_F1_PATH,
    index=False,
)


# ============================================================
# 15. VALIDATION SUPPORT CONSISTENCY
# ============================================================

support_check_df = pd.DataFrame(
    validation_support_checks
)

all_supports_match = bool(
    support_check_df[
        "Matches"
    ].all()
)


# ============================================================
# 16. BASELINE IMPROVEMENT CHECK
# ============================================================

best_baseline_macro_f1 = float(
    comparison_df[
        comparison_df[
            "Model_Type"
        ]
        ==
        "Baseline"
    ][
        "Validation_Macro_F1"
    ].max()
)

best_real_macro_f1 = float(
    real_ranking_df.iloc[
        0
    ][
        "Validation_Macro_F1"
    ]
)

best_real_beats_baseline = bool(
    best_real_macro_f1
    >
    best_baseline_macro_f1
)


# ============================================================
# 17. SHORTLIST VALIDATION
# ============================================================

expected_shortlist_size = 3

shortlist_size_valid = bool(
    len(
        shortlist_df
    )
    ==
    expected_shortlist_size
)

shortlist_all_have_minority_recall = bool(
    (
        shortlist_df[
            "Warning_Recall"
        ]
        >
        0
    ).all()
    and
    (
        shortlist_df[
            "Critical_Recall"
        ]
        >
        0
    ).all()
)

shortlisted_models = (
    shortlist_df[
        "Model"
    ]
    .tolist()
)

best_model = str(
    real_ranking_df.iloc[
        0
    ][
        "Model"
    ]
)


# ============================================================
# 18. CHECKS
# ============================================================

numeric_cols = [
    "Validation_Accuracy",
    "Validation_Balanced_Accuracy",
    "Validation_Macro_Precision",
    "Validation_Macro_Recall",
    "Validation_Macro_F1",
    "Validation_Weighted_F1",
    "Warning_Precision",
    "Warning_Recall",
    "Warning_F1",
    "Critical_Precision",
    "Critical_Recall",
    "Critical_F1",
    "Train_Macro_F1",
    "Macro_F1_Gap",
    "Train_Balanced_Accuracy",
    "Balanced_Accuracy_Gap",
]

all_metrics_finite = bool(
    np.isfinite(
        comparison_df[
            numeric_cols
        ].to_numpy(
            dtype=float
        )
    ).all()
)

all_real_models_present = bool(
    set(
        real_ranking_df[
            "Model"
        ]
    )
    ==
    set(
        REAL_MODELS
    )
)

checks = pd.DataFrame(
    [
        {
            "Check": (
                "All five real classifiers present"
            ),
            "Status": (
                "PASS"
                if all_real_models_present
                else "FAIL"
            ),
            "Details": (
                ", ".join(
                    real_ranking_df[
                        "Model"
                    ].tolist()
                )
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
                "Baseline + real-model comparison metrics verified."
            ),
        },
        {
            "Check": (
                "Validation class supports consistent across models"
            ),
            "Status": (
                "PASS"
                if all_supports_match
                else "FAIL"
            ),
            "Details": (
                "Expected Normal=3973, Warning=168, Critical=18."
            ),
        },
        {
            "Check": (
                "Best real classifier beats best dummy baseline"
            ),
            "Status": (
                "PASS"
                if best_real_beats_baseline
                else "FAIL"
            ),
            "Details": (
                f"Best real Macro-F1={best_real_macro_f1:.6f}; "
                f"best baseline={best_baseline_macro_f1:.6f}"
            ),
        },
        {
            "Check": (
                "Shortlist contains exactly three models"
            ),
            "Status": (
                "PASS"
                if shortlist_size_valid
                else "FAIL"
            ),
            "Details": (
                ", ".join(
                    shortlisted_models
                )
            ),
        },
        {
            "Check": (
                "Shortlisted models detect both minority classes"
            ),
            "Status": (
                "PASS"
                if shortlist_all_have_minority_recall
                else "FAIL"
            ),
            "Details": (
                "Warning Recall > 0 and Critical Recall > 0."
            ),
        },
        {
            "Check": (
                "No model training performed"
            ),
            "Status": "PASS",
            "Details": (
                "Stage 27 reads saved development metrics only."
            ),
        },
        {
            "Check": (
                "No hyperparameter tuning performed"
            ),
            "Status": "PASS",
            "Details": (
                "Tuning reserved for Stage 28."
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
# 19. SHORTLIST ARTIFACT
# ============================================================

shortlist_artifact = {
    "stage": (
        "27_classification_model_comparison"
    ),
    "selection_primary_metric": (
        "Validation Macro-F1"
    ),
    "supporting_metrics": [
        "Validation Balanced Accuracy",
        "Warning Recall",
        "Critical Recall",
        "Warning F1",
        "Critical F1",
        "Train-Validation Macro-F1 Gap",
        "Train-Validation Balanced Accuracy Gap",
    ],
    "shortlist_rule": (
        "Top 3 real classifiers by Validation Macro-F1, "
        "provided each has non-zero Warning and Critical Validation recall."
    ),
    "best_untuned_model": (
        best_model
    ),
    "shortlisted_models": (
        shortlisted_models
    ),
    "stage28_action": (
        "Bounded hyperparameter tuning of shortlisted model families only."
    ),
    "test_loaded": False,
}

with open(
    SHORTLIST_JSON_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        shortlist_artifact,
        file,
        indent=2,
    )


# ============================================================
# 20. SUMMARY JSON
# ============================================================

summary = {
    "stage": (
        "27_classification_model_comparison"
    ),
    "overall_status": (
        overall_status
    ),
    "best_untuned_model": (
        best_model
    ),
    "best_validation_macro_f1": (
        best_real_macro_f1
    ),
    "best_baseline_macro_f1": (
        best_baseline_macro_f1
    ),
    "real_model_ranking": [
        {
            "rank": int(
                row[
                    "Real_Model_Rank"
                ]
            ),
            "model": str(
                row[
                    "Model"
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
            "macro_f1_gap": float(
                row[
                    "Macro_F1_Gap"
                ]
            ),
        }
        for _, row in (
            real_ranking_df.iterrows()
        )
    ],
    "shortlisted_models": (
        shortlisted_models
    ),
    "test_loaded": False,
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
# 21. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- REAL CLASSIFIER RANKING ---"
)

display_columns = [
    "Real_Model_Rank",
    "Model",
    "Validation_Macro_F1",
    "Validation_Balanced_Accuracy",
    "Warning_Recall",
    "Critical_Recall",
    "Warning_F1",
    "Critical_F1",
    "Macro_F1_Gap",
]

print(
    real_ranking_df[
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
    "\n--- BASELINE REFERENCE ---"
)

baseline_display = (
    comparison_df[
        comparison_df[
            "Model_Type"
        ]
        ==
        "Baseline"
    ][
        [
            "Model",
            "Validation_Accuracy",
            "Validation_Balanced_Accuracy",
            "Validation_Macro_F1",
            "Warning_Recall",
            "Critical_Recall",
        ]
    ]
)

print(
    baseline_display
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\n--- STAGE 28 SHORTLIST ---"
)

print(
    shortlist_df[
        [
            "Shortlist_Rank",
            "Model",
            "Validation_Macro_F1",
            "Validation_Balanced_Accuracy",
            "Warning_Recall",
            "Critical_Recall",
            "Macro_F1_Gap",
            "Stage28_Status",
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
    "\n--- FORMAL CONCLUSION ---"
)

print(
    f"Best untuned classifier          : "
    f"{best_model}"
)

print(
    f"Best Validation Macro-F1         : "
    f"{best_real_macro_f1:.4f}"
)

print(
    f"Best dummy baseline Macro-F1     : "
    f"{best_baseline_macro_f1:.4f}"
)

print(
    "Shortlisted for Stage 28        : "
    +
    ", ".join(
        shortlisted_models
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
# 22. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 27: "
        "CLASSIFICATION MODEL COMPARISON & SHORTLISTING\n"
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
        "Stage 27 formally compares the completed classification "
        "experiments from Stages 20 and 22-26. No classifier is "
        "trained or tuned here. Validation Macro-F1 is the primary "
        "ranking metric because the target is severely imbalanced. "
        "Balanced Accuracy and minority-class performance are used "
        "as supporting evidence.\n\n"
    )

    report.write(
        "SELECTION HIERARCHY\n"
        "-------------------\n"
    )

    report.write(
        "Primary metric: Validation Macro-F1.\n"
        "Supporting metrics: Validation Balanced Accuracy, Warning "
        "Recall, Critical Recall, Warning F1, Critical F1, and "
        "Train-to-Validation generalization gaps.\n\n"
    )

    report.write(
        "REAL CLASSIFIER RANKING\n"
        "-----------------------\n"
    )

    report.write(
        real_ranking_df[
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
        "BASELINE REFERENCE\n"
        "------------------\n"
    )

    report.write(
        baseline_display
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
        "STAGE-28 SHORTLIST RULE\n"
        "-----------------------\n"
    )

    report.write(
        "The top three real classifiers by Validation Macro-F1 are "
        "retained provided each has non-zero Warning and Critical "
        "Validation recall. This preserves the strongest overall "
        "candidates while keeping meaningful minority-class "
        "coverage for bounded tuning.\n\n"
    )

    report.write(
        "SHORTLIST\n"
        "---------\n"
    )

    report.write(
        shortlist_df[
            [
                "Shortlist_Rank",
                "Model",
                "Validation_Macro_F1",
                "Validation_Balanced_Accuracy",
                "Warning_Recall",
                "Critical_Recall",
                "Macro_F1_Gap",
            ]
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
        "INTERPRETATION\n"
        "--------------\n"
    )

    report.write(
        f"The strongest untuned development-stage classifier is "
        f"{best_model}, with Validation Macro-F1 "
        f"{best_real_macro_f1:.6f}. The shortlist does not represent "
        "final model selection. Stage 28 will perform bounded tuning "
        "of shortlisted model families only, and Stage 29 will make "
        "the final development-stage classifier selection.\n\n"
    )

    report.write(
        "METHODOLOGY PROTECTION\n"
        "----------------------\n"
    )

    report.write(
        "- No classifier was trained in Stage 27.\n"
        "- No hyperparameter tuning was performed.\n"
        "- No decision-threshold tuning was performed.\n"
        "- Ranking is based on saved development results only.\n"
        "- Validation remains development data.\n"
        "- Final Test rows were not loaded.\n"
        "- Final Test remains frozen until Stage 31.\n\n"
    )

    report.write(
        "NEXT STAGE\n"
        "----------\n"
    )

    report.write(
        "Stage 28 will perform bounded hyperparameter tuning on the "
        "three shortlisted classifier families only. The same Train "
        "and Validation split will be preserved, and Final Test will "
        "remain untouched.\n\n"
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
        f"STAGE 27 STATUS: "
        f"{overall_status}\n"
    )


# ============================================================
# 23. COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    f"STAGE 27 CLASSIFICATION MODEL COMPARISON: "
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
    f"\nShortlist artifact saved in:\n"
    f"{ARTIFACT_DIR}"
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 27 failed. Review "
        "27_classification_comparison_report.txt."
    )
