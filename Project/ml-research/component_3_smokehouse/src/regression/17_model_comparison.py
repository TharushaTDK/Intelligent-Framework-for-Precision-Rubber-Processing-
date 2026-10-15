from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# COMPONENT 3 - STAGE 17: REGRESSION MODEL COMPARISON
# ============================================================
#
# Purpose:
#   Consolidate and compare all completed regression experiments
#   from Stages 11-16.
#
# IMPORTANT:
#   - No new model is trained.
#   - No hyperparameter tuning is performed.
#   - Validation metrics are used for development comparison.
#   - Final Test remains completely untouched until Stage 31.
#
# Primary ranking metric:
#   Validation MAE (lower is better)
#
# Secondary metrics:
#   Validation RMSE (lower is better)
#   Validation R²   (higher is better)
#
# Tuning shortlist:
#   Top 2 NON-BASELINE candidate models by Validation MAE.
#
# Stage 18 will tune the shortlisted model families.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

# File location:
# component_3_smokehouse/src/regression/17_regression_comparison.py
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

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TEST_PATH = (
    SPLIT_DIR
    / "Component3_Test.csv"
)


# ============================================================
# 2. INPUT METRIC FILES
# ============================================================

STAGE11_PATH = (
    REPORT_DIR
    / "11_regression_baseline_metrics.csv"
)

STAGE12_PATH = (
    REPORT_DIR
    / "12_random_forest_regression_metrics.csv"
)

STAGE13_PATH = (
    REPORT_DIR
    / "13_gradient_boosting_regression_metrics.csv"
)

STAGE14_PATH = (
    REPORT_DIR
    / "14_extra_trees_regression_metrics.csv"
)

STAGE15_PATH = (
    REPORT_DIR
    / "15_xgboost_regression_metrics.csv"
)

STAGE16_PATH = (
    REPORT_DIR
    / "16_hist_gradient_boosting_regression_metrics.csv"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "17_regression_comparison_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "17_regression_comparison_checks.csv"
)

RANKING_PATH = (
    REPORT_DIR
    / "17_regression_validation_ranking.csv"
)

GENERALIZATION_PATH = (
    REPORT_DIR
    / "17_regression_generalization_comparison.csv"
)

PHYSICAL_PATH = (
    REPORT_DIR
    / "17_regression_physical_violation_comparison.csv"
)

SHORTLIST_PATH = (
    REPORT_DIR
    / "17_regression_tuning_shortlist.csv"
)

SUMMARY_JSON_PATH = (
    REPORT_DIR
    / "17_regression_comparison_summary.json"
)


# ============================================================
# 4. REQUIRED MODELS
# ============================================================

MODEL_SOURCES = {
    "Dummy_Mean": STAGE11_PATH,
    "Elapsed_Only_Linear": STAGE11_PATH,
    "Full_Linear": STAGE11_PATH,
    "Random_Forest": STAGE12_PATH,
    "Gradient_Boosting": STAGE13_PATH,
    "Extra_Trees": STAGE14_PATH,
    "XGBoost": STAGE15_PATH,
    "HistGradientBoosting": STAGE16_PATH,
}

BASELINE_MODELS = {
    "Dummy_Mean",
    "Elapsed_Only_Linear",
    "Full_Linear",
}

# Models eligible for Stage-18 tuning.
TUNABLE_CANDIDATES = {
    "Random_Forest",
    "Gradient_Boosting",
    "Extra_Trees",
    "XGBoost",
    "HistGradientBoosting",
}


# ============================================================
# 5. START
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 17: REGRESSION MODEL COMPARISON")
print("=" * 78)


# ============================================================
# 6. FILE EXISTENCE VALIDATION
# ============================================================

required_paths = [
    STAGE11_PATH,
    STAGE12_PATH,
    STAGE13_PATH,
    STAGE14_PATH,
    STAGE15_PATH,
    STAGE16_PATH,
    TEST_PATH,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("\nAll Stage 11-16 regression metric files found.")
print("Final Test dataset existence verified, but contents were NOT loaded.")


# ============================================================
# 7. LOAD ONLY MODEL METRIC FILES
# ============================================================

metrics_cache = {}

for path in {
    STAGE11_PATH,
    STAGE12_PATH,
    STAGE13_PATH,
    STAGE14_PATH,
    STAGE15_PATH,
    STAGE16_PATH,
}:
    metrics_cache[path] = pd.read_csv(
        path
    )


# ============================================================
# 8. METRIC EXTRACTION
# ============================================================

def extract_model_split(
    model_name,
    split_name,
):
    source_path = MODEL_SOURCES[
        model_name
    ]

    df = metrics_cache[
        source_path
    ]

    required_columns = {
        "Model",
        "Split",
        "MAE",
        "RMSE",
        "R2",
    }

    missing_columns = (
        required_columns
        -
        set(
            df.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            f"{source_path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

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
            f"Expected exactly one row for "
            f"{model_name} / {split_name}, found {len(row)}."
        )

    return row.iloc[0]


def optional_numeric(
    row,
    column,
):
    if column not in row.index:
        return np.nan

    value = row[
        column
    ]

    if pd.isna(
        value
    ):
        return np.nan

    return float(
        value
    )


# ============================================================
# 9. BUILD FULL COMPARISON TABLE
# ============================================================

comparison_rows = []

for model_name in MODEL_SOURCES:
    train_row = extract_model_split(
        model_name,
        "Train",
    )

    validation_row = extract_model_split(
        model_name,
        "Validation",
    )

    train_mae = float(
        train_row[
            "MAE"
        ]
    )

    train_rmse = float(
        train_row[
            "RMSE"
        ]
    )

    train_r2 = float(
        train_row[
            "R2"
        ]
    )

    val_mae = float(
        validation_row[
            "MAE"
        ]
    )

    val_rmse = float(
        validation_row[
            "RMSE"
        ]
    )

    val_r2 = float(
        validation_row[
            "R2"
        ]
    )

    val_negative = int(
        optional_numeric(
            validation_row,
            "Negative_Predictions",
        )
        if not math.isnan(
            optional_numeric(
                validation_row,
                "Negative_Predictions",
            )
        )
        else 0
    )

    val_above_126 = int(
        optional_numeric(
            validation_row,
            "Predictions_Above_126h",
        )
        if not math.isnan(
            optional_numeric(
                validation_row,
                "Predictions_Above_126h",
            )
        )
        else 0
    )

    comparison_rows.append({
        "Model": model_name,
        "Model_Type": (
            "Baseline"
            if model_name in BASELINE_MODELS
            else "Candidate"
        ),
        "Train_MAE": train_mae,
        "Validation_MAE": val_mae,
        "Train_RMSE": train_rmse,
        "Validation_RMSE": val_rmse,
        "Train_R2": train_r2,
        "Validation_R2": val_r2,
        "MAE_Generalization_Gap": (
            val_mae
            -
            train_mae
        ),
        "RMSE_Generalization_Gap": (
            val_rmse
            -
            train_rmse
        ),
        "R2_Generalization_Gap": (
            train_r2
            -
            val_r2
        ),
        "Negative_Predictions": (
            val_negative
        ),
        "Predictions_Above_126h": (
            val_above_126
        ),
        "Total_Physical_Violations": (
            val_negative
            +
            val_above_126
        ),
    })

comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# 10. VALIDATION RANKING
# ============================================================
#
# Primary:
#   Validation MAE ascending
#
# Tie-breakers:
#   Validation RMSE ascending
#   Validation R² descending

ranking_df = (
    comparison_df
    .sort_values(
        by=[
            "Validation_MAE",
            "Validation_RMSE",
            "Validation_R2",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)

ranking_df[
    "Overall_Validation_Rank"
] = np.arange(
    1,
    len(
        ranking_df
    )
    + 1,
)

ranking_df[
    "MAE_Rank"
] = ranking_df[
    "Validation_MAE"
].rank(
    method="min",
    ascending=True,
).astype(
    int
)

ranking_df[
    "RMSE_Rank"
] = ranking_df[
    "Validation_RMSE"
].rank(
    method="min",
    ascending=True,
).astype(
    int
)

ranking_df[
    "R2_Rank"
] = ranking_df[
    "Validation_R2"
].rank(
    method="min",
    ascending=False,
).astype(
    int
)

ranking_df = ranking_df[
    [
        "Overall_Validation_Rank",
        "Model",
        "Model_Type",
        "Validation_MAE",
        "Validation_RMSE",
        "Validation_R2",
        "MAE_Rank",
        "RMSE_Rank",
        "R2_Rank",
        "MAE_Generalization_Gap",
        "Total_Physical_Violations",
    ]
]

ranking_df.to_csv(
    RANKING_PATH,
    index=False,
)


# ============================================================
# 11. GENERALIZATION COMPARISON
# ============================================================

generalization_df = (
    comparison_df[
        [
            "Model",
            "Model_Type",
            "Train_MAE",
            "Validation_MAE",
            "MAE_Generalization_Gap",
            "Train_RMSE",
            "Validation_RMSE",
            "RMSE_Generalization_Gap",
            "Train_R2",
            "Validation_R2",
            "R2_Generalization_Gap",
        ]
    ]
    .sort_values(
        by="Validation_MAE"
    )
    .reset_index(
        drop=True
    )
)

generalization_df.to_csv(
    GENERALIZATION_PATH,
    index=False,
)


# ============================================================
# 12. PHYSICAL-RANGE COMPARISON
# ============================================================

VALIDATION_ROWS = 4159

physical_df = (
    comparison_df[
        [
            "Model",
            "Model_Type",
            "Negative_Predictions",
            "Predictions_Above_126h",
            "Total_Physical_Violations",
        ]
    ]
    .copy()
)

physical_df[
    "Physical_Violation_Rate_pct"
] = (
    physical_df[
        "Total_Physical_Violations"
    ]
    /
    VALIDATION_ROWS
    *
    100
)

physical_df = (
    physical_df
    .sort_values(
        by=[
            "Total_Physical_Violations",
            "Model",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)

physical_df.to_csv(
    PHYSICAL_PATH,
    index=False,
)


# ============================================================
# 13. IDENTIFY BEST CURRENT VALIDATION MODEL
# ============================================================

best_row = (
    ranking_df
    .iloc[
        0
    ]
)

BEST_MODEL = str(
    best_row[
        "Model"
    ]
)

BEST_MAE = float(
    best_row[
        "Validation_MAE"
    ]
)

BEST_RMSE = float(
    best_row[
        "Validation_RMSE"
    ]
)

BEST_R2 = float(
    best_row[
        "Validation_R2"
    ]
)


# ============================================================
# 14. BUILD STAGE-18 TUNING SHORTLIST
# ============================================================
#
# Baseline models are not tuned.
#
# Shortlist = top 2 candidate model families by Validation MAE.
#
# This does NOT declare the final regression model.
# Stage 18 performs tuning.
# Stage 19 performs final development-set selection.
# Stage 31 evaluates the selected model once on final Test.

candidate_df = (
    comparison_df[
        comparison_df[
            "Model"
        ].isin(
            TUNABLE_CANDIDATES
        )
    ]
    .sort_values(
        by=[
            "Validation_MAE",
            "Validation_RMSE",
            "Validation_R2",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)

shortlist_df = (
    candidate_df
    .head(
        2
    )
    .copy()
)

shortlist_df.insert(
    0,
    "Tuning_Priority",
    np.arange(
        1,
        len(
            shortlist_df
        )
        + 1,
    ),
)

shortlist_df[
    "Selection_Rationale"
] = (
    "Top candidate by Validation MAE; "
    "advance to Stage 18 tuning."
)

shortlist_df = shortlist_df[
    [
        "Tuning_Priority",
        "Model",
        "Validation_MAE",
        "Validation_RMSE",
        "Validation_R2",
        "MAE_Generalization_Gap",
        "Total_Physical_Violations",
        "Selection_Rationale",
    ]
]

shortlist_df.to_csv(
    SHORTLIST_PATH,
    index=False,
)


# ============================================================
# 15. BASELINE IMPROVEMENT OF CURRENT BEST
# ============================================================

full_linear_row = (
    comparison_df[
        comparison_df[
            "Model"
        ]
        ==
        "Full_Linear"
    ]
    .iloc[
        0
    ]
)

FULL_LINEAR_MAE = float(
    full_linear_row[
        "Validation_MAE"
    ]
)

FULL_LINEAR_RMSE = float(
    full_linear_row[
        "Validation_RMSE"
    ]
)

FULL_LINEAR_R2 = float(
    full_linear_row[
        "Validation_R2"
    ]
)

best_mae_improvement_vs_full_linear = (
    (
        FULL_LINEAR_MAE
        -
        BEST_MAE
    )
    /
    FULL_LINEAR_MAE
    *
    100
)

best_rmse_improvement_vs_full_linear = (
    (
        FULL_LINEAR_RMSE
        -
        BEST_RMSE
    )
    /
    FULL_LINEAR_RMSE
    *
    100
)


# ============================================================
# 16. CONSISTENCY CHECKS
# ============================================================

all_models_present = bool(
    set(
        ranking_df[
            "Model"
        ]
    )
    ==
    set(
        MODEL_SOURCES.keys()
    )
)

all_validation_metrics_finite = bool(
    np.isfinite(
        ranking_df[
            [
                "Validation_MAE",
                "Validation_RMSE",
                "Validation_R2",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all()
)

all_mae_nonnegative = bool(
    (
        ranking_df[
            "Validation_MAE"
        ]
        >=
        0
    ).all()
)

all_rmse_nonnegative = bool(
    (
        ranking_df[
            "Validation_RMSE"
        ]
        >=
        0
    ).all()
)

shortlist_size_correct = bool(
    len(
        shortlist_df
    )
    ==
    2
)

shortlist_only_candidates = bool(
    set(
        shortlist_df[
            "Model"
        ]
    ).issubset(
        TUNABLE_CANDIDATES
    )
)

ranking_unique = bool(
    ranking_df[
        "Model"
    ].is_unique
)


# ============================================================
# 17. PASS / FAIL CHECKS
# ============================================================

checks = pd.DataFrame(
    [
        {
            "Check": (
                "All required regression models present"
            ),
            "Status": (
                "PASS"
                if all_models_present
                else "FAIL"
            ),
            "Details": (
                f"Models={len(ranking_df)}; "
                f"expected={len(MODEL_SOURCES)}"
            ),
        },
        {
            "Check": (
                "One unique comparison row per model"
            ),
            "Status": (
                "PASS"
                if ranking_unique
                else "FAIL"
            ),
            "Details": (
                f"Unique models={ranking_df['Model'].nunique()}"
            ),
        },
        {
            "Check": (
                "Validation metrics finite"
            ),
            "Status": (
                "PASS"
                if all_validation_metrics_finite
                else "FAIL"
            ),
            "Details": (
                "MAE, RMSE and R² checked for all models."
            ),
        },
        {
            "Check": (
                "MAE/RMSE values non-negative"
            ),
            "Status": (
                "PASS"
                if (
                    all_mae_nonnegative
                    and
                    all_rmse_nonnegative
                )
                else "FAIL"
            ),
            "Details": (
                "Validation error metrics verified."
            ),
        },
        {
            "Check": (
                "Ranking based on Validation only"
            ),
            "Status": "PASS",
            "Details": (
                "Primary=MAE; tie-breakers=RMSE then R²."
            ),
        },
        {
            "Check": (
                "No new model training"
            ),
            "Status": "PASS",
            "Details": (
                "Stage 17 reads completed metric files only."
            ),
        },
        {
            "Check": (
                "No hyperparameter tuning"
            ),
            "Status": "PASS",
            "Details": (
                "Tuning is reserved for Stage 18."
            ),
        },
        {
            "Check": (
                "Tuning shortlist contains two candidates"
            ),
            "Status": (
                "PASS"
                if shortlist_size_correct
                else "FAIL"
            ),
            "Details": (
                f"Shortlist size={len(shortlist_df)}"
            ),
        },
        {
            "Check": (
                "Tuning shortlist excludes baseline models"
            ),
            "Status": (
                "PASS"
                if shortlist_only_candidates
                else "FAIL"
            ),
            "Details": (
                ", ".join(
                    shortlist_df[
                        "Model"
                    ].tolist()
                )
            ),
        },
        {
            "Check": (
                "Final Test not loaded"
            ),
            "Status": "PASS",
            "Details": (
                "Only Test file existence checked; "
                "contents were not read."
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
# 18. MACHINE-READABLE SUMMARY
# ============================================================

summary = {
    "stage": (
        "17_regression_model_comparison"
    ),
    "overall_status": (
        overall_status
    ),
    "primary_ranking_metric": (
        "Validation_MAE"
    ),
    "secondary_metrics": [
        "Validation_RMSE",
        "Validation_R2",
    ],
    "models_compared": (
        ranking_df[
            "Model"
        ].tolist()
    ),
    "current_best_validation_model": {
        "Model": (
            BEST_MODEL
        ),
        "MAE": (
            BEST_MAE
        ),
        "RMSE": (
            BEST_RMSE
        ),
        "R2": (
            BEST_R2
        ),
    },
    "improvement_vs_full_linear": {
        "MAE_percent": (
            best_mae_improvement_vs_full_linear
        ),
        "RMSE_percent": (
            best_rmse_improvement_vs_full_linear
        ),
        "R2_change": (
            BEST_R2
            -
            FULL_LINEAR_R2
        ),
    },
    "stage18_tuning_shortlist": (
        shortlist_df[
            "Model"
        ].tolist()
    ),
    "test_loaded": False,
    "final_model_selected": False,
    "final_model_selection_stage": 19,
    "final_test_evaluation_stage": 31,
}

with open(
    SUMMARY_JSON_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=2,
    )


# ============================================================
# 19. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- VALIDATION RANKING ---"
)

console_ranking = ranking_df[
    [
        "Overall_Validation_Rank",
        "Model",
        "Validation_MAE",
        "Validation_RMSE",
        "Validation_R2",
        "MAE_Generalization_Gap",
        "Total_Physical_Violations",
    ]
]

print(
    console_ranking
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\n--- CURRENT BEST UNTUNED MODEL ---"
)

print(
    f"Model                           : "
    f"{BEST_MODEL}"
)

print(
    f"Validation MAE                  : "
    f"{BEST_MAE:.4f} h"
)

print(
    f"Validation RMSE                 : "
    f"{BEST_RMSE:.4f} h"
)

print(
    f"Validation R2                   : "
    f"{BEST_R2:.4f}"
)

print(
    f"MAE improvement vs Full Linear  : "
    f"{best_mae_improvement_vs_full_linear:.2f}%"
)

print(
    f"RMSE improvement vs Full Linear : "
    f"{best_rmse_improvement_vs_full_linear:.2f}%"
)


print(
    "\n--- STAGE 18 TUNING SHORTLIST ---"
)

print(
    shortlist_df[
        [
            "Tuning_Priority",
            "Model",
            "Validation_MAE",
            "Validation_RMSE",
            "Validation_R2",
            "MAE_Generalization_Gap",
            "Total_Physical_Violations",
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
    "\n--- IMPORTANT INTERPRETATION ---"
)

print(
    "Stage 17 identifies the strongest DEVELOPMENT candidates only."
)

print(
    "It does NOT declare the final regression model."
)

print(
    "Stage 18 will tune the shortlisted model families."
)

print(
    "Stage 19 will perform final development-set model selection."
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
# 20. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 17: REGRESSION MODEL COMPARISON\n"
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
        "Stage 17 consolidates the completed regression "
        "experiments from Stages 11-16. No model is trained "
        "and no hyperparameter tuning is performed. Models are "
        "compared using the frozen Validation set only. "
        "Validation MAE is the primary ranking metric because "
        "it directly represents the average absolute error in "
        "remaining drying-time hours.\n\n"
    )

    report.write(
        "VALIDATION RANKING\n"
        "------------------\n"
    )

    report.write(
        console_ranking
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
        "CURRENT BEST UNTUNED DEVELOPMENT MODEL\n"
        "--------------------------------------\n"
    )

    report.write(
        f"Model: {BEST_MODEL}\n"
    )

    report.write(
        f"Validation MAE: "
        f"{BEST_MAE:.6f} h\n"
    )

    report.write(
        f"Validation RMSE: "
        f"{BEST_RMSE:.6f} h\n"
    )

    report.write(
        f"Validation R²: "
        f"{BEST_R2:.6f}\n"
    )

    report.write(
        f"MAE improvement vs Full Linear: "
        f"{best_mae_improvement_vs_full_linear:.6f}%\n"
    )

    report.write(
        f"RMSE improvement vs Full Linear: "
        f"{best_rmse_improvement_vs_full_linear:.6f}%\n\n"
    )

    report.write(
        "GENERALIZATION COMPARISON\n"
        "-------------------------\n"
    )

    report.write(
        generalization_df
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
        "PHYSICAL-RANGE DIAGNOSTICS\n"
        "--------------------------\n"
    )

    report.write(
        physical_df
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
        "Physical-range violations are diagnostic only at this "
        "stage. Predictions are not clipped because doing so "
        "would change the raw model comparison. Deployment-time "
        "constraints may be considered only after final model "
        "selection.\n\n"
    )

    report.write(
        "STAGE 18 TUNING SHORTLIST\n"
        "-------------------------\n"
    )

    report.write(
        shortlist_df
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
        "The shortlist contains the top two non-baseline "
        "candidate model families by Validation MAE. This is "
        "a tuning shortlist, not a final model declaration.\n\n"
    )

    report.write(
        "MODEL-SELECTION STATUS\n"
        "----------------------\n"
    )

    report.write(
        "Stage 17 identifies the strongest untuned development "
        "candidates. Stage 18 will tune the shortlisted model "
        "families. Stage 19 will select the final regression "
        "model using development evidence. The frozen Test set "
        "will be evaluated only once in Stage 31.\n\n"
    )

    report.write(
        "LEAKAGE PROTECTION\n"
        "------------------\n"
    )

    report.write(
        "- Only completed Train/Validation metric files were read.\n"
    )

    report.write(
        "- No model was trained in Stage 17.\n"
    )

    report.write(
        "- No hyperparameter tuning was performed.\n"
    )

    report.write(
        "- No Test rows were loaded.\n"
    )

    report.write(
        "- Final Test remains frozen until Stage 31.\n\n"
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
        f"STAGE 17 STATUS: "
        f"{overall_status}\n"
    )


# ============================================================
# 21. COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    f"STAGE 17 REGRESSION MODEL COMPARISON: "
    f"{overall_status}"
)

print(
    "=" * 78
)

print(
    f"\nReports saved in:\n"
    f"{REPORT_DIR}"
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 17 failed. Review "
        "17_regression_comparison_report.txt."
    )
