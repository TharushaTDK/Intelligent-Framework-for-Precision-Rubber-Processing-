from pathlib import Path
import hashlib
import json
import shutil

import joblib
import numpy as np
import pandas as pd


# ============================================================
# COMPONENT 3 - STAGE 19: FINAL REGRESSION SELECTION
# ============================================================
#
# Purpose:
#   Formally select the regression model to carry forward after:
#       Stage 11 - Baseline
#       Stage 12 - Random Forest
#       Stage 13 - Gradient Boosting
#       Stage 14 - Extra Trees
#       Stage 15 - XGBoost
#       Stage 16 - HistGradientBoosting
#       Stage 17 - Comparison
#       Stage 18 - Hyperparameter tuning
#
# IMPORTANT:
#   - No model is trained in Stage 19.
#   - No hyperparameter search is performed.
#   - Selection uses DEVELOPMENT evidence only.
#   - Final TEST remains untouched until Stage 31.
#
# Primary selection metric:
#   Validation MAE (lower is better)
#
# Supporting evidence:
#   Validation RMSE
#   Validation R²
#   Train-to-validation MAE generalization gap
#   Physical-range violations
#
# Stage 19 selects the development-stage final regression model
# artifact that will proceed to later robustness/explainability/
# export stages.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

# File location:
# component_3_smokehouse/src/regression/19_final_selection.py
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

TUNED_MODEL_DIR = (
    ROOT
    / "artifacts"
    / "models"
    / "regression"
    / "tuned"
)

FINAL_MODEL_DIR = (
    ROOT
    / "artifacts"
    / "models"
    / "regression"
    / "final"
)

FINAL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TEST_PATH = (
    SPLIT_DIR
    / "Component3_Test.csv"
)

STAGE17_RANKING_PATH = (
    REPORT_DIR
    / "17_regression_validation_ranking.csv"
)

STAGE18_BEST_PATH = (
    REPORT_DIR
    / "18_regression_tuning_best_results.csv"
)

STAGE18_IMPROVEMENT_PATH = (
    REPORT_DIR
    / "18_regression_tuning_improvement_comparison.csv"
)

STAGE18_SUMMARY_PATH = (
    REPORT_DIR
    / "18_regression_tuning_summary.json"
)

TUNED_GB_MODEL_PATH = (
    TUNED_MODEL_DIR
    / "gradient_boosting"
    / "tuned_gradient_boosting_regressor.joblib"
)

TUNED_GB_METADATA_PATH = (
    TUNED_MODEL_DIR
    / "gradient_boosting"
    / "tuned_gradient_boosting_metadata.json"
)

TUNED_XGB_MODEL_PATH = (
    TUNED_MODEL_DIR
    / "xgboost"
    / "tuned_xgboost_regressor.joblib"
)

TUNED_XGB_METADATA_PATH = (
    TUNED_MODEL_DIR
    / "xgboost"
    / "tuned_xgboost_metadata.json"
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "19_final_regression_selection_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "19_final_regression_selection_checks.csv"
)

DECISION_PATH = (
    REPORT_DIR
    / "19_final_regression_selection_decision.csv"
)

CANDIDATE_COMPARISON_PATH = (
    REPORT_DIR
    / "19_final_regression_candidate_comparison.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "19_final_regression_selection_summary.json"
)

FINAL_MODEL_PATH = (
    FINAL_MODEL_DIR
    / "selected_regression_model.joblib"
)

FINAL_METADATA_PATH = (
    FINAL_MODEL_DIR
    / "selected_regression_model_metadata.json"
)


# ============================================================
# 3. START
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 19: FINAL REGRESSION SELECTION")
print("=" * 78)


# ============================================================
# 4. REQUIRED FILE CHECKS
# ============================================================

required_files = [
    TEST_PATH,
    STAGE17_RANKING_PATH,
    STAGE18_BEST_PATH,
    STAGE18_IMPROVEMENT_PATH,
    STAGE18_SUMMARY_PATH,
    TUNED_GB_MODEL_PATH,
    TUNED_GB_METADATA_PATH,
    TUNED_XGB_MODEL_PATH,
    TUNED_XGB_METADATA_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

print("\nAll required Stage 17-18 evidence and tuned model artifacts found.")
print("Final Test dataset existence verified, but contents were NOT loaded.")


# ============================================================
# 5. LOAD DEVELOPMENT EVIDENCE ONLY
# ============================================================

stage17_ranking = pd.read_csv(
    STAGE17_RANKING_PATH
)

stage18_best = pd.read_csv(
    STAGE18_BEST_PATH
)

stage18_improvement = pd.read_csv(
    STAGE18_IMPROVEMENT_PATH
)

with open(
    STAGE18_SUMMARY_PATH,
    "r",
    encoding="utf-8",
) as file:
    stage18_summary = json.load(
        file
    )

with open(
    TUNED_GB_METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:
    gb_metadata = json.load(
        file
    )

with open(
    TUNED_XGB_METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:
    xgb_metadata = json.load(
        file
    )


# ============================================================
# 6. VALIDATE STAGE-18 CANDIDATES
# ============================================================

required_best_columns = {
    "Model_Family",
    "Trial_ID",
    "Validation_MAE",
    "Validation_RMSE",
    "Validation_R2",
    "Train_MAE",
    "MAE_Generalization_Gap",
    "Negative_Predictions",
    "Predictions_Above_126h",
}

missing_best_columns = (
    required_best_columns
    -
    set(
        stage18_best.columns
    )
)

if missing_best_columns:
    raise RuntimeError(
        "Stage-18 best-results file is missing columns: "
        f"{sorted(missing_best_columns)}"
    )

expected_families = {
    "Gradient_Boosting",
    "XGBoost",
}

actual_families = set(
    stage18_best[
        "Model_Family"
    ].tolist()
)

if actual_families != expected_families:
    raise RuntimeError(
        "Unexpected Stage-18 tuned model families.\n"
        f"Expected: {sorted(expected_families)}\n"
        f"Found: {sorted(actual_families)}"
    )

if len(
    stage18_best
) != 2:
    raise RuntimeError(
        "Stage-18 best-results file must contain exactly "
        "one best tuned model per shortlisted family."
    )


# ============================================================
# 7. FORMAL SELECTION RULE
# ============================================================
#
# Rank tuned finalists by:
#   1) Validation MAE ascending
#   2) Validation RMSE ascending
#   3) Validation R² descending
#
# Generalization gap and physical violations are retained as
# supporting evidence, but they do not override a clearly better
# primary validation result unless a serious methodological or
# physical-validity issue is detected.
#
# No such hard-disqualification condition is introduced here
# because the Stage-18 candidates are valid regression models and
# their small number of negative near-zero predictions were already
# treated as diagnostics rather than clipped evaluation values.

candidate_df = (
    stage18_best
    .copy()
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

candidate_df[
    "Selection_Rank"
] = np.arange(
    1,
    len(
        candidate_df
    )
    + 1,
)

candidate_df[
    "Total_Physical_Violations"
] = (
    candidate_df[
        "Negative_Predictions"
    ]
    +
    candidate_df[
        "Predictions_Above_126h"
    ]
)

candidate_df[
    "Selection_Status"
] = np.where(
    candidate_df[
        "Selection_Rank"
    ]
    ==
    1,
    "SELECTED",
    "NOT_SELECTED",
)

candidate_df.to_csv(
    CANDIDATE_COMPARISON_PATH,
    index=False,
)


# ============================================================
# 8. SELECT WINNER
# ============================================================

selected_row = (
    candidate_df
    .iloc[
        0
    ]
)

runner_up_row = (
    candidate_df
    .iloc[
        1
    ]
)

SELECTED_FAMILY = str(
    selected_row[
        "Model_Family"
    ]
)

SELECTED_TRIAL = str(
    selected_row[
        "Trial_ID"
    ]
)

SELECTED_MAE = float(
    selected_row[
        "Validation_MAE"
    ]
)

SELECTED_RMSE = float(
    selected_row[
        "Validation_RMSE"
    ]
)

SELECTED_R2 = float(
    selected_row[
        "Validation_R2"
    ]
)

SELECTED_TRAIN_MAE = float(
    selected_row[
        "Train_MAE"
    ]
)

SELECTED_MAE_GAP = float(
    selected_row[
        "MAE_Generalization_Gap"
    ]
)

SELECTED_NEGATIVE = int(
    selected_row[
        "Negative_Predictions"
    ]
)

SELECTED_ABOVE_126 = int(
    selected_row[
        "Predictions_Above_126h"
    ]
)

RUNNER_UP_FAMILY = str(
    runner_up_row[
        "Model_Family"
    ]
)

RUNNER_UP_MAE = float(
    runner_up_row[
        "Validation_MAE"
    ]
)

RUNNER_UP_RMSE = float(
    runner_up_row[
        "Validation_RMSE"
    ]
)

RUNNER_UP_R2 = float(
    runner_up_row[
        "Validation_R2"
    ]
)


# ============================================================
# 9. COMPARE WITH STAGE-17 BEST UNTUNED MODEL
# ============================================================

if "Overall_Validation_Rank" not in stage17_ranking.columns:
    raise RuntimeError(
        "Stage-17 ranking file is missing Overall_Validation_Rank."
    )

stage17_best_row = (
    stage17_ranking
    .sort_values(
        "Overall_Validation_Rank"
    )
    .iloc[
        0
    ]
)

STAGE17_BEST_MODEL = str(
    stage17_best_row[
        "Model"
    ]
)

STAGE17_BEST_MAE = float(
    stage17_best_row[
        "Validation_MAE"
    ]
)

STAGE17_BEST_RMSE = float(
    stage17_best_row[
        "Validation_RMSE"
    ]
)

STAGE17_BEST_R2 = float(
    stage17_best_row[
        "Validation_R2"
    ]
)

MAE_IMPROVEMENT_VS_STAGE17 = (
    (
        STAGE17_BEST_MAE
        -
        SELECTED_MAE
    )
    /
    STAGE17_BEST_MAE
    *
    100
)

RMSE_IMPROVEMENT_VS_STAGE17 = (
    (
        STAGE17_BEST_RMSE
        -
        SELECTED_RMSE
    )
    /
    STAGE17_BEST_RMSE
    *
    100
)

R2_CHANGE_VS_STAGE17 = (
    SELECTED_R2
    -
    STAGE17_BEST_R2
)


# ============================================================
# 10. COMPARE WITH RUNNER-UP
# ============================================================

MAE_ADVANTAGE_VS_RUNNER_UP = (
    RUNNER_UP_MAE
    -
    SELECTED_MAE
)

MAE_ADVANTAGE_PCT_VS_RUNNER_UP = (
    (
        RUNNER_UP_MAE
        -
        SELECTED_MAE
    )
    /
    RUNNER_UP_MAE
    *
    100
)

RMSE_ADVANTAGE_VS_RUNNER_UP = (
    RUNNER_UP_RMSE
    -
    SELECTED_RMSE
)

R2_ADVANTAGE_VS_RUNNER_UP = (
    SELECTED_R2
    -
    RUNNER_UP_R2
)


# ============================================================
# 11. RESOLVE SELECTED MODEL ARTIFACT
# ============================================================

if SELECTED_FAMILY == "Gradient_Boosting":
    source_model_path = (
        TUNED_GB_MODEL_PATH
    )
    source_metadata_path = (
        TUNED_GB_METADATA_PATH
    )
    selected_metadata = (
        gb_metadata
    )

elif SELECTED_FAMILY == "XGBoost":
    source_model_path = (
        TUNED_XGB_MODEL_PATH
    )
    source_metadata_path = (
        TUNED_XGB_METADATA_PATH
    )
    selected_metadata = (
        xgb_metadata
    )

else:
    raise RuntimeError(
        f"Unsupported selected model family: {SELECTED_FAMILY}"
    )


# ============================================================
# 12. ARTIFACT VALIDATION
# ============================================================

# Loading the selected JOBLIB artifact is allowed because it is
# a TRAIN-fitted model from Stage 18. No Test data are involved.

selected_model = joblib.load(
    source_model_path
)

if SELECTED_FAMILY == "Gradient_Boosting":
    expected_class_name = (
        "GradientBoostingRegressor"
    )
else:
    expected_class_name = (
        "XGBRegressor"
    )

actual_class_name = (
    selected_model
    .__class__
    .__name__
)

if actual_class_name != expected_class_name:
    raise RuntimeError(
        "Selected model artifact class mismatch.\n"
        f"Expected: {expected_class_name}\n"
        f"Found: {actual_class_name}"
    )

if str(
    selected_metadata.get(
        "best_trial_id"
    )
) != SELECTED_TRIAL:
    raise RuntimeError(
        "Selected Stage-18 model metadata Trial_ID does not "
        "match the Stage-18 comparison result."
    )


# ============================================================
# 13. COPY SELECTED ARTIFACT TO FINAL REGRESSION FOLDER
# ============================================================

shutil.copy2(
    source_model_path,
    FINAL_MODEL_PATH,
)


# ============================================================
# 14. HASHES
# ============================================================

def sha256_file(path):
    with open(
        path,
        "rb",
    ) as file:
        return hashlib.sha256(
            file.read()
        ).hexdigest()


source_model_sha256 = sha256_file(
    source_model_path
)

final_model_sha256 = sha256_file(
    FINAL_MODEL_PATH
)

artifact_copy_identical = bool(
    source_model_sha256
    ==
    final_model_sha256
)


# ============================================================
# 15. DECISION TABLE
# ============================================================

decision_df = pd.DataFrame(
    [
        {
            "Selected_Model_Family": (
                SELECTED_FAMILY
            ),
            "Selected_Trial_ID": (
                SELECTED_TRIAL
            ),
            "Validation_MAE": (
                SELECTED_MAE
            ),
            "Validation_RMSE": (
                SELECTED_RMSE
            ),
            "Validation_R2": (
                SELECTED_R2
            ),
            "Train_MAE": (
                SELECTED_TRAIN_MAE
            ),
            "MAE_Generalization_Gap": (
                SELECTED_MAE_GAP
            ),
            "Negative_Predictions": (
                SELECTED_NEGATIVE
            ),
            "Predictions_Above_126h": (
                SELECTED_ABOVE_126
            ),
            "Runner_Up": (
                RUNNER_UP_FAMILY
            ),
            "Runner_Up_MAE": (
                RUNNER_UP_MAE
            ),
            "MAE_Advantage_vs_Runner_Up_h": (
                MAE_ADVANTAGE_VS_RUNNER_UP
            ),
            "MAE_Advantage_vs_Runner_Up_pct": (
                MAE_ADVANTAGE_PCT_VS_RUNNER_UP
            ),
            "Stage17_Best_Untuned_Model": (
                STAGE17_BEST_MODEL
            ),
            "MAE_Improvement_vs_Stage17_Best_pct": (
                MAE_IMPROVEMENT_VS_STAGE17
            ),
            "Selection_Primary_Metric": (
                "Validation_MAE"
            ),
            "Final_Test_Used": False,
        }
    ]
)

decision_df.to_csv(
    DECISION_PATH,
    index=False,
)


# ============================================================
# 16. FINAL MODEL METADATA
# ============================================================

final_metadata = {
    "stage": (
        "19_final_regression_selection"
    ),
    "selection_scope": (
        "DEVELOPMENT_ONLY"
    ),
    "selected_model_family": (
        SELECTED_FAMILY
    ),
    "selected_trial_id": (
        SELECTED_TRIAL
    ),
    "model_class": (
        actual_class_name
    ),
    "best_parameters": (
        selected_metadata.get(
            "best_parameters",
            {}
        )
    ),
    "development_validation": {
        "MAE": (
            SELECTED_MAE
        ),
        "RMSE": (
            SELECTED_RMSE
        ),
        "R2": (
            SELECTED_R2
        ),
        "Train_MAE": (
            SELECTED_TRAIN_MAE
        ),
        "MAE_Generalization_Gap": (
            SELECTED_MAE_GAP
        ),
        "Negative_Predictions": (
            SELECTED_NEGATIVE
        ),
        "Predictions_Above_126h": (
            SELECTED_ABOVE_126
        ),
    },
    "runner_up": {
        "model_family": (
            RUNNER_UP_FAMILY
        ),
        "MAE": (
            RUNNER_UP_MAE
        ),
        "RMSE": (
            RUNNER_UP_RMSE
        ),
        "R2": (
            RUNNER_UP_R2
        ),
    },
    "selection_rule": {
        "primary": (
            "Validation_MAE_ascending"
        ),
        "tie_breaker_1": (
            "Validation_RMSE_ascending"
        ),
        "tie_breaker_2": (
            "Validation_R2_descending"
        ),
        "supporting_evidence": [
            "MAE_Generalization_Gap",
            "Physical_Range_Violations",
        ],
    },
    "stage17_best_untuned": {
        "Model": (
            STAGE17_BEST_MODEL
        ),
        "MAE": (
            STAGE17_BEST_MAE
        ),
        "RMSE": (
            STAGE17_BEST_RMSE
        ),
        "R2": (
            STAGE17_BEST_R2
        ),
    },
    "improvement_vs_stage17_best": {
        "MAE_percent": (
            MAE_IMPROVEMENT_VS_STAGE17
        ),
        "RMSE_percent": (
            RMSE_IMPROVEMENT_VS_STAGE17
        ),
        "R2_change": (
            R2_CHANGE_VS_STAGE17
        ),
    },
    "artifact": {
        "source_path": (
            str(
                source_model_path
            )
        ),
        "final_path": (
            str(
                FINAL_MODEL_PATH
            )
        ),
        "source_sha256": (
            source_model_sha256
        ),
        "final_sha256": (
            final_model_sha256
        ),
        "copy_identical": (
            artifact_copy_identical
        ),
    },
    "test_loaded": False,
    "test_evaluated": False,
    "group_cross_validation_stage": 30,
    "final_test_evaluation_stage": 31,
    "shap_stage": 32,
    "export_stage": 33,
}

with open(
    FINAL_METADATA_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_metadata,
        file,
        indent=2,
    )

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        final_metadata,
        file,
        indent=2,
    )


# ============================================================
# 17. PASS / FAIL CHECKS
# ============================================================

stage18_leader_family = str(
    stage18_summary[
        "stage18_validation_leader"
    ][
        "model_family"
    ]
)

stage18_leader_trial = str(
    stage18_summary[
        "stage18_validation_leader"
    ][
        "trial_id"
    ]
)

selection_matches_stage18_leader = bool(
    SELECTED_FAMILY
    ==
    stage18_leader_family
    and
    SELECTED_TRIAL
    ==
    stage18_leader_trial
)

selected_has_best_mae = bool(
    np.isclose(
        SELECTED_MAE,
        candidate_df[
            "Validation_MAE"
        ].min(),
    )
)

selection_metrics_finite = bool(
    np.isfinite(
        [
            SELECTED_MAE,
            SELECTED_RMSE,
            SELECTED_R2,
            SELECTED_TRAIN_MAE,
            SELECTED_MAE_GAP,
        ]
    ).all()
)

checks = pd.DataFrame(
    [
        {
            "Check": (
                "Exactly two tuned finalists available"
            ),
            "Status": (
                "PASS"
                if len(
                    candidate_df
                )
                ==
                2
                else "FAIL"
            ),
            "Details": (
                f"Finalists={len(candidate_df)}"
            ),
        },
        {
            "Check": (
                "Expected finalist families available"
            ),
            "Status": (
                "PASS"
                if actual_families
                ==
                expected_families
                else "FAIL"
            ),
            "Details": (
                ", ".join(
                    sorted(
                        actual_families
                    )
                )
            ),
        },
        {
            "Check": (
                "Selection metrics finite"
            ),
            "Status": (
                "PASS"
                if selection_metrics_finite
                else "FAIL"
            ),
            "Details": (
                "MAE, RMSE, R² and generalization values verified."
            ),
        },
        {
            "Check": (
                "Selected model has best tuned Validation MAE"
            ),
            "Status": (
                "PASS"
                if selected_has_best_mae
                else "FAIL"
            ),
            "Details": (
                f"{SELECTED_FAMILY}: "
                f"MAE={SELECTED_MAE:.6f}"
            ),
        },
        {
            "Check": (
                "Selection matches Stage-18 validation leader"
            ),
            "Status": (
                "PASS"
                if selection_matches_stage18_leader
                else "FAIL"
            ),
            "Details": (
                f"Stage18={stage18_leader_family}/"
                f"{stage18_leader_trial}; "
                f"Stage19={SELECTED_FAMILY}/"
                f"{SELECTED_TRIAL}"
            ),
        },
        {
            "Check": (
                "Selected model artifact class valid"
            ),
            "Status": (
                "PASS"
                if actual_class_name
                ==
                expected_class_name
                else "FAIL"
            ),
            "Details": (
                actual_class_name
            ),
        },
        {
            "Check": (
                "Selected artifact copied without modification"
            ),
            "Status": (
                "PASS"
                if artifact_copy_identical
                else "FAIL"
            ),
            "Details": (
                f"SHA256={final_model_sha256}"
            ),
        },
        {
            "Check": (
                "No model training performed"
            ),
            "Status": "PASS",
            "Details": (
                "Stage 19 loads and copies the selected "
                "Stage-18 artifact only."
            ),
        },
        {
            "Check": (
                "No hyperparameter tuning performed"
            ),
            "Status": "PASS",
            "Details": (
                "Stage 18 tuning evidence reused."
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
# 18. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- TUNED FINALIST COMPARISON ---"
)

display_columns = [
    "Selection_Rank",
    "Model_Family",
    "Trial_ID",
    "Validation_MAE",
    "Validation_RMSE",
    "Validation_R2",
    "Train_MAE",
    "MAE_Generalization_Gap",
    "Total_Physical_Violations",
    "Selection_Status",
]

print(
    candidate_df[
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
    "\n--- FINAL DEVELOPMENT-STAGE REGRESSION SELECTION ---"
)

print(
    f"Selected model family           : "
    f"{SELECTED_FAMILY}"
)

print(
    f"Selected tuning trial           : "
    f"{SELECTED_TRIAL}"
)

print(
    f"Validation MAE                  : "
    f"{SELECTED_MAE:.4f} h"
)

print(
    f"Validation RMSE                 : "
    f"{SELECTED_RMSE:.4f} h"
)

print(
    f"Validation R2                   : "
    f"{SELECTED_R2:.4f}"
)

print(
    f"Train MAE                       : "
    f"{SELECTED_TRAIN_MAE:.4f} h"
)

print(
    f"MAE generalization gap          : "
    f"{SELECTED_MAE_GAP:.4f} h"
)

print(
    f"Negative predictions            : "
    f"{SELECTED_NEGATIVE}"
)

print(
    f"Predictions above 126 h         : "
    f"{SELECTED_ABOVE_126}"
)


print(
    "\n--- ADVANTAGE OVER RUNNER-UP ---"
)

print(
    f"Runner-up                       : "
    f"{RUNNER_UP_FAMILY}"
)

print(
    f"MAE advantage                   : "
    f"{MAE_ADVANTAGE_VS_RUNNER_UP:.4f} h"
)

print(
    f"MAE advantage percentage        : "
    f"{MAE_ADVANTAGE_PCT_VS_RUNNER_UP:.2f}%"
)

print(
    f"RMSE advantage                  : "
    f"{RMSE_ADVANTAGE_VS_RUNNER_UP:.4f} h"
)

print(
    f"R2 advantage                    : "
    f"{R2_ADVANTAGE_VS_RUNNER_UP:+.4f}"
)


print(
    "\n--- IMPROVEMENT OVER STAGE-17 BEST UNTUNED MODEL ---"
)

print(
    f"Stage-17 best model             : "
    f"{STAGE17_BEST_MODEL}"
)

print(
    f"Stage-17 best MAE               : "
    f"{STAGE17_BEST_MAE:.4f} h"
)

print(
    f"Selected tuned MAE              : "
    f"{SELECTED_MAE:.4f} h"
)

print(
    f"MAE improvement                 : "
    f"{MAE_IMPROVEMENT_VS_STAGE17:.2f}%"
)


print(
    "\n--- SELECTED ARTIFACT ---"
)

print(
    f"Model class                     : "
    f"{actual_class_name}"
)

print(
    f"Final model path                : "
    f"{FINAL_MODEL_PATH}"
)

print(
    f"Artifact copy identical         : "
    f"{artifact_copy_identical}"
)


print(
    "\n--- IMPORTANT INTERPRETATION ---"
)

print(
    "Stage 19 selects the final regression model from DEVELOPMENT evidence."
)

print(
    "Its Validation metrics are not the final unbiased Test performance."
)

print(
    "Stage 30 will later assess group-aware robustness."
)

print(
    "Stage 31 will perform the one-time final Test evaluation."
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
# 19. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 19: FINAL REGRESSION SELECTION\n"
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
        "Stage 19 formally selects the regression model to carry "
        "forward using development evidence from Stages 11-18. "
        "No new model is trained and no hyperparameter search is "
        "performed. The final Test set remains untouched.\n\n"
    )

    report.write(
        "SELECTION RULE\n"
        "--------------\n"
    )

    report.write(
        "Primary criterion: lowest Validation MAE.\n"
        "Tie-breaker 1: lowest Validation RMSE.\n"
        "Tie-breaker 2: highest Validation R².\n"
        "Generalization gap and physical-range violations are "
        "recorded as supporting evidence.\n\n"
    )

    report.write(
        "TUNED FINALIST COMPARISON\n"
        "-------------------------\n"
    )

    report.write(
        candidate_df[
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
        "SELECTED REGRESSION MODEL\n"
        "-------------------------\n"
    )

    report.write(
        f"Model family: "
        f"{SELECTED_FAMILY}\n"
    )

    report.write(
        f"Tuning trial: "
        f"{SELECTED_TRIAL}\n"
    )

    report.write(
        f"Validation MAE: "
        f"{SELECTED_MAE:.6f} h\n"
    )

    report.write(
        f"Validation RMSE: "
        f"{SELECTED_RMSE:.6f} h\n"
    )

    report.write(
        f"Validation R²: "
        f"{SELECTED_R2:.6f}\n"
    )

    report.write(
        f"Train MAE: "
        f"{SELECTED_TRAIN_MAE:.6f} h\n"
    )

    report.write(
        f"MAE generalization gap: "
        f"{SELECTED_MAE_GAP:.6f} h\n"
    )

    report.write(
        f"Negative Validation predictions: "
        f"{SELECTED_NEGATIVE}\n"
    )

    report.write(
        f"Predictions above 126 h: "
        f"{SELECTED_ABOVE_126}\n\n"
    )

    report.write(
        "WHY IT WAS SELECTED\n"
        "-------------------\n"
    )

    report.write(
        f"{SELECTED_FAMILY} achieved the lowest Validation MAE "
        f"among the two tuned finalists ({SELECTED_MAE:.6f} h), "
        f"compared with {RUNNER_UP_MAE:.6f} h for "
        f"{RUNNER_UP_FAMILY}. It also achieved the stronger "
        f"Validation RMSE ({SELECTED_RMSE:.6f} h vs "
        f"{RUNNER_UP_RMSE:.6f} h) and stronger Validation R² "
        f"({SELECTED_R2:.6f} vs {RUNNER_UP_R2:.6f}). "
        "Therefore it is the strongest development-stage "
        "regression candidate according to the predefined "
        "selection rule.\n\n"
    )

    report.write(
        "ADVANTAGE OVER RUNNER-UP\n"
        "------------------------\n"
    )

    report.write(
        f"Runner-up: "
        f"{RUNNER_UP_FAMILY}\n"
    )

    report.write(
        f"MAE advantage: "
        f"{MAE_ADVANTAGE_VS_RUNNER_UP:.6f} h "
        f"({MAE_ADVANTAGE_PCT_VS_RUNNER_UP:.6f}%)\n"
    )

    report.write(
        f"RMSE advantage: "
        f"{RMSE_ADVANTAGE_VS_RUNNER_UP:.6f} h\n"
    )

    report.write(
        f"R² advantage: "
        f"{R2_ADVANTAGE_VS_RUNNER_UP:+.6f}\n\n"
    )

    report.write(
        "IMPROVEMENT AFTER TUNING\n"
        "------------------------\n"
    )

    report.write(
        f"Stage-17 best untuned model: "
        f"{STAGE17_BEST_MODEL}\n"
    )

    report.write(
        f"Stage-17 best untuned MAE: "
        f"{STAGE17_BEST_MAE:.6f} h\n"
    )

    report.write(
        f"Selected tuned MAE: "
        f"{SELECTED_MAE:.6f} h\n"
    )

    report.write(
        f"MAE improvement: "
        f"{MAE_IMPROVEMENT_VS_STAGE17:.6f}%\n"
    )

    report.write(
        f"RMSE improvement: "
        f"{RMSE_IMPROVEMENT_VS_STAGE17:.6f}%\n"
    )

    report.write(
        f"R² change: "
        f"{R2_CHANGE_VS_STAGE17:+.6f}\n\n"
    )

    report.write(
        "SELECTED ARTIFACT\n"
        "-----------------\n"
    )

    report.write(
        f"Source model: "
        f"{source_model_path}\n"
    )

    report.write(
        f"Final selected artifact: "
        f"{FINAL_MODEL_PATH}\n"
    )

    report.write(
        f"Model class: "
        f"{actual_class_name}\n"
    )

    report.write(
        f"SHA-256: "
        f"{final_model_sha256}\n"
    )

    report.write(
        f"Copied without modification: "
        f"{artifact_copy_identical}\n\n"
    )

    report.write(
        "METHODOLOGICAL STATUS\n"
        "---------------------\n"
    )

    report.write(
        "This is the final regression selection based on "
        "development evidence. Its Validation performance must "
        "not be described as unbiased final performance because "
        "Validation has been used for model comparison and "
        "hyperparameter tuning. Stage 30 will provide group-aware "
        "robustness evidence and Stage 31 will provide the "
        "one-time final Test evaluation.\n\n"
    )

    report.write(
        "LEAKAGE PROTECTION\n"
        "------------------\n"
    )

    report.write(
        "- No model training occurred in Stage 19.\n"
    )

    report.write(
        "- No hyperparameter tuning occurred in Stage 19.\n"
    )

    report.write(
        "- Only Stage-17/18 development evidence was used.\n"
    )

    report.write(
        "- Final Test rows were not loaded.\n"
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
        f"STAGE 19 STATUS: "
        f"{overall_status}\n"
    )


# ============================================================
# 20. COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    f"STAGE 19 FINAL REGRESSION SELECTION: "
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
    f"\nSelected model saved in:\n"
    f"{FINAL_MODEL_DIR}"
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 19 failed. Review "
        "19_final_regression_selection_report.txt."
    )
