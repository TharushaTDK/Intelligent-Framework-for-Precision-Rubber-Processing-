from pathlib import Path
import hashlib
import pandas as pd
import numpy as np


# ============================================================
# COMPONENT 3 - STAGE 02: TARGET VALIDATION
# ============================================================
#
# Purpose:
#   Validate the two prediction targets before preprocessing:
#
#   1) Regression:
#        Remaining_Drying_Time
#
#   2) Classification:
#        Fire_Risk_Label
#
# Important:
#   - This script DOES NOT modify the raw dataset.
#   - This script DOES NOT impute missing sensor values.
#   - This script DOES NOT split the dataset.
#   - This script DOES NOT engineer features.
#   - This script DOES NOT train models.
#
# Critical leakage rules:
#   - Actual_Total_Drying_Time is retrospective target-
#     construction/reference information and MUST NOT be used
#     as a deployed ML predictor.
#   - Fire_Risk_Assessment_Timestamp is target metadata and
#     MUST NOT be used as a physical ML predictor.
#   - Batch_ID is for grouping/splitting and MUST NOT be used
#     as a physical ML predictor.
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Component3_Dataset.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "02_target_validation_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "02_target_validation_checks.csv"
)

REGRESSION_SEQUENCE_PATH = (
    REPORT_DIR
    / "02_regression_sequence_validation.csv"
)

FIRE_ROW_DISTRIBUTION_PATH = (
    REPORT_DIR
    / "02_fire_risk_row_distribution.csv"
)

FIRE_ASSESSMENT_DISTRIBUTION_PATH = (
    REPORT_DIR
    / "02_fire_risk_assessment_distribution.csv"
)

FIRE_BY_LAYER_PATH = (
    REPORT_DIR
    / "02_fire_risk_by_layer.csv"
)

FIRE_BY_BATCH_PATH = (
    REPORT_DIR
    / "02_fire_risk_by_batch.csv"
)


# ============================================================
# 3. EXPECTED FROZEN DATASET IDENTITY
# ============================================================

EXPECTED_SHA256 = (
    "229139d2b4413fe15a8d322023b3b425"
    "74f9dccd5df6a91550bc72f05faaa5cb"
)

EXPECTED_FIRE_LABELS = {
    "Normal",
    "Warning",
    "Critical",
}

FORMULA_TOLERANCE_HOURS = 0.011


# ============================================================
# 4. LOAD RAW DATASET
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 02: TARGET VALIDATION")
print("=" * 78)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nFinal dataset not found:\n{DATA_PATH}\n\n"
        "Expected location:\n"
        "ml-research/component_3_smokehouse/data/raw/"
        "Component3_Dataset.csv"
    )

df = pd.read_csv(DATA_PATH)

print("\nRaw dataset loaded successfully.")


# ============================================================
# 5. DATASET FINGERPRINT
# ============================================================

with open(DATA_PATH, "rb") as file:
    dataset_sha256 = hashlib.sha256(
        file.read()
    ).hexdigest()

hash_matches = (
    dataset_sha256
    == EXPECTED_SHA256
)


# ============================================================
# 6. SAFE TYPE CONVERSION FOR VALIDATION
# ============================================================

validation_df = df.copy()

validation_df["Timestamp"] = pd.to_datetime(
    validation_df["Timestamp"],
    errors="coerce",
)

validation_df[
    "Fire_Risk_Assessment_Timestamp"
] = pd.to_datetime(
    validation_df[
        "Fire_Risk_Assessment_Timestamp"
    ],
    errors="coerce",
)

numeric_columns = [
    "Elapsed_Drying_Hours",
    "Actual_Total_Drying_Time",
    "Remaining_Drying_Time",
]

for column in numeric_columns:
    validation_df[column] = pd.to_numeric(
        validation_df[column],
        errors="coerce",
    )


# ============================================================
# 7. COMMON GROUPING KEY
# ============================================================

SEQUENCE_KEY = [
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
]

validation_df = validation_df.sort_values(
    SEQUENCE_KEY
    + ["Timestamp"]
).reset_index(drop=True)


# ============================================================
# 8. REGRESSION TARGET VALIDATION
# ============================================================

regression_target = (
    validation_df[
        "Remaining_Drying_Time"
    ]
)

regression_missing = int(
    regression_target
    .isna()
    .sum()
)

negative_remaining_count = int(
    (
        regression_target
        < 0
    ).sum()
)

formula_expected = (
    validation_df[
        "Actual_Total_Drying_Time"
    ]
    -
    validation_df[
        "Elapsed_Drying_Hours"
    ]
)

formula_absolute_error = (
    formula_expected
    -
    regression_target
).abs()

max_formula_error = float(
    formula_absolute_error.max()
)

formula_error_rows = int(
    (
        formula_absolute_error
        > FORMULA_TOLERANCE_HOURS
    ).sum()
)


# ============================================================
# 9. SEQUENCE-LEVEL REGRESSION VALIDATION
# ============================================================

sequence_rows = []

for key, sequence in validation_df.groupby(
    SEQUENCE_KEY,
    sort=True,
):
    sequence = sequence.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    smokehouse_id, batch_id, layer_id = key

    start_elapsed = float(
        sequence[
            "Elapsed_Drying_Hours"
        ].iloc[0]
    )

    end_elapsed = float(
        sequence[
            "Elapsed_Drying_Hours"
        ].iloc[-1]
    )

    start_remaining = float(
        sequence[
            "Remaining_Drying_Time"
        ].iloc[0]
    )

    end_remaining = float(
        sequence[
            "Remaining_Drying_Time"
        ].iloc[-1]
    )

    total_min = float(
        sequence[
            "Actual_Total_Drying_Time"
        ].min()
    )

    total_max = float(
        sequence[
            "Actual_Total_Drying_Time"
        ].max()
    )

    elapsed_diff = (
        sequence[
            "Elapsed_Drying_Hours"
        ]
        .diff()
        .dropna()
    )

    remaining_diff = (
        sequence[
            "Remaining_Drying_Time"
        ]
        .diff()
        .dropna()
    )

    elapsed_non_increasing_count = int(
        (
            elapsed_diff
            <= 0
        ).sum()
    )

    remaining_increase_count = int(
        (
            remaining_diff
            > FORMULA_TOLERANCE_HOURS
        ).sum()
    )

    sequence_rows.append({
        "Smokehouse_ID": smokehouse_id,
        "Batch_ID": batch_id,
        "Layer_ID": layer_id,
        "Records": len(sequence),
        "Start_Timestamp": (
            sequence[
                "Timestamp"
            ].iloc[0]
        ),
        "End_Timestamp": (
            sequence[
                "Timestamp"
            ].iloc[-1]
        ),
        "Start_Elapsed_Hours": (
            start_elapsed
        ),
        "End_Elapsed_Hours": (
            end_elapsed
        ),
        "Start_Remaining_Hours": (
            start_remaining
        ),
        "End_Remaining_Hours": (
            end_remaining
        ),
        "Total_Drying_Min_Hours": (
            total_min
        ),
        "Total_Drying_Max_Hours": (
            total_max
        ),
        "Total_Drying_Constant": (
            abs(
                total_max
                - total_min
            )
            <= FORMULA_TOLERANCE_HOURS
        ),
        "Starts_At_Elapsed_Zero": (
            abs(
                start_elapsed
            )
            <= FORMULA_TOLERANCE_HOURS
        ),
        "Ends_At_Remaining_Zero": (
            abs(
                end_remaining
            )
            <= FORMULA_TOLERANCE_HOURS
        ),
        "Final_Elapsed_Equals_Total": (
            abs(
                end_elapsed
                - total_min
            )
            <= FORMULA_TOLERANCE_HOURS
        ),
        "Elapsed_Strictly_Increases": (
            elapsed_non_increasing_count
            == 0
        ),
        "Remaining_Never_Increases": (
            remaining_increase_count
            == 0
        ),
        "Elapsed_Non_Increasing_Count": (
            elapsed_non_increasing_count
        ),
        "Remaining_Increase_Count": (
            remaining_increase_count
        ),
    })

sequence_validation = pd.DataFrame(
    sequence_rows
)

sequence_validation.to_csv(
    REGRESSION_SEQUENCE_PATH,
    index=False,
)

sequence_count = len(
    sequence_validation
)

start_elapsed_violations = int(
    (
        ~sequence_validation[
            "Starts_At_Elapsed_Zero"
        ]
    ).sum()
)

end_remaining_violations = int(
    (
        ~sequence_validation[
            "Ends_At_Remaining_Zero"
        ]
    ).sum()
)

final_elapsed_total_violations = int(
    (
        ~sequence_validation[
            "Final_Elapsed_Equals_Total"
        ]
    ).sum()
)

total_constant_violations = int(
    (
        ~sequence_validation[
            "Total_Drying_Constant"
        ]
    ).sum()
)

elapsed_monotonic_violations = int(
    (
        ~sequence_validation[
            "Elapsed_Strictly_Increases"
        ]
    ).sum()
)

remaining_monotonic_violations = int(
    (
        ~sequence_validation[
            "Remaining_Never_Increases"
        ]
    ).sum()
)


# ============================================================
# 10. CLASSIFICATION TARGET VALIDATION
# ============================================================

fire_target = (
    validation_df[
        "Fire_Risk_Label"
    ]
)

fire_missing = int(
    fire_target
    .isna()
    .sum()
)

actual_fire_labels = set(
    fire_target
    .dropna()
    .unique()
)

unexpected_fire_labels = sorted(
    actual_fire_labels
    -
    EXPECTED_FIRE_LABELS
)

missing_expected_fire_labels = sorted(
    EXPECTED_FIRE_LABELS
    -
    actual_fire_labels
)


# ============================================================
# 11. ROW-LEVEL FIRE-RISK DISTRIBUTION
# ============================================================

fire_row_distribution = (
    fire_target
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "Fire_Risk_Label"
    )
    .reset_index(
        name="Count"
    )
)

fire_row_distribution[
    "Percentage"
] = (
    fire_row_distribution[
        "Count"
    ]
    / len(validation_df)
    * 100
)

fire_row_distribution.to_csv(
    FIRE_ROW_DISTRIBUTION_PATH,
    index=False,
)


# ============================================================
# 12. FIRE-RISK ASSESSMENT TIMING VALIDATION
# ============================================================

assessment_missing = int(
    validation_df[
        "Fire_Risk_Assessment_Timestamp"
    ]
    .isna()
    .sum()
)

timestamp_missing = int(
    validation_df[
        "Timestamp"
    ]
    .isna()
    .sum()
)

assessment_age_minutes = (
    (
        validation_df[
            "Timestamp"
        ]
        -
        validation_df[
            "Fire_Risk_Assessment_Timestamp"
        ]
    )
    .dt
    .total_seconds()
    / 60
)

assessment_after_sensor_count = int(
    (
        assessment_age_minutes
        < 0
    ).sum()
)

assessment_older_than_window_count = int(
    (
        assessment_age_minutes
        >= 60
    ).sum()
)

assessment_window_invalid_count = (
    assessment_after_sensor_count
    +
    assessment_older_than_window_count
)


# ============================================================
# 13. LABEL CONSISTENCY INSIDE EACH HOURLY ASSESSMENT
# ============================================================

assessment_group_key = (
    SEQUENCE_KEY
    + [
        "Fire_Risk_Assessment_Timestamp"
    ]
)

assessment_label_nunique = (
    validation_df
    .groupby(
        assessment_group_key,
        dropna=False,
    )[
        "Fire_Risk_Label"
    ]
    .nunique(
        dropna=False
    )
)

assessment_label_inconsistencies = int(
    (
        assessment_label_nunique
        > 1
    ).sum()
)


# ============================================================
# 14. UNIQUE HOURLY ASSESSMENT EVENTS
# ============================================================

assessment_events = (
    validation_df
    .sort_values(
        SEQUENCE_KEY
        + [
            "Fire_Risk_Assessment_Timestamp",
            "Timestamp",
        ]
    )
    .drop_duplicates(
        subset=assessment_group_key,
        keep="first",
    )
)

unique_assessment_count = len(
    assessment_events
)

assessment_distribution = (
    assessment_events[
        "Fire_Risk_Label"
    ]
    .value_counts()
    .rename_axis(
        "Fire_Risk_Label"
    )
    .reset_index(
        name="Assessment_Count"
    )
)

assessment_distribution[
    "Percentage"
] = (
    assessment_distribution[
        "Assessment_Count"
    ]
    / unique_assessment_count
    * 100
)

assessment_distribution.to_csv(
    FIRE_ASSESSMENT_DISTRIBUTION_PATH,
    index=False,
)


# ============================================================
# 15. HOURLY ASSESSMENT CADENCE
# ============================================================

assessment_cadence_errors = 0

for _, sequence in validation_df.groupby(
    SEQUENCE_KEY,
    sort=True,
):
    assessment_times = (
        sequence[
            "Fire_Risk_Assessment_Timestamp"
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    assessment_differences = (
        assessment_times
        .diff()
        .dropna()
        .dt
        .total_seconds()
        / 60
    )

    assessment_cadence_errors += int(
        (
            assessment_differences
            != 60
        ).sum()
    )


# ============================================================
# 16. FIRE-RISK BY LAYER
# ============================================================

fire_by_layer = pd.crosstab(
    validation_df[
        "Layer_ID"
    ],
    validation_df[
        "Fire_Risk_Label"
    ],
)

for label in [
    "Normal",
    "Warning",
    "Critical",
]:
    if label not in fire_by_layer.columns:
        fire_by_layer[label] = 0

fire_by_layer = (
    fire_by_layer[
        [
            "Normal",
            "Warning",
            "Critical",
        ]
    ]
    .reset_index()
)

fire_by_layer.to_csv(
    FIRE_BY_LAYER_PATH,
    index=False,
)


# ============================================================
# 17. UNIQUE ASSESSMENT EVENTS BY BATCH
# ============================================================

fire_by_batch = pd.crosstab(
    [
        assessment_events[
            "Smokehouse_ID"
        ],
        assessment_events[
            "Batch_ID"
        ],
    ],
    assessment_events[
        "Fire_Risk_Label"
    ],
)

for label in [
    "Normal",
    "Warning",
    "Critical",
]:
    if label not in fire_by_batch.columns:
        fire_by_batch[label] = 0

fire_by_batch = (
    fire_by_batch[
        [
            "Normal",
            "Warning",
            "Critical",
        ]
    ]
    .reset_index()
)

fire_by_batch.to_csv(
    FIRE_BY_BATCH_PATH,
    index=False,
)

batches_with_warning = int(
    (
        fire_by_batch[
            "Warning"
        ]
        > 0
    ).sum()
)

batches_with_critical = int(
    (
        fire_by_batch[
            "Critical"
        ]
        > 0
    ).sum()
)


# ============================================================
# 18. TARGET-LEAKAGE / METADATA RULES
# ============================================================

# These are intentionally explicit so later stages do not
# accidentally use them as physical prediction features.

REGRESSION_TARGET = (
    "Remaining_Drying_Time"
)

CLASSIFICATION_TARGET = (
    "Fire_Risk_Label"
)

GROUP_ONLY_FIELDS = [
    "Batch_ID",
]

REFERENCE_OR_METADATA_FIELDS = [
    "Actual_Total_Drying_Time",
    "Fire_Risk_Assessment_Timestamp",
]

NEVER_USE_AS_PHYSICAL_PREDICTOR = [
    "Actual_Total_Drying_Time",
    "Fire_Risk_Assessment_Timestamp",
    "Batch_ID",
]


# ============================================================
# 19. PASS / FAIL CHECKS
# ============================================================

checks = [
    {
        "Check": "Frozen dataset SHA-256 matches",
        "Status": (
            "PASS"
            if hash_matches
            else "FAIL"
        ),
        "Details": dataset_sha256,
    },
    {
        "Check": "Regression target has no missing values",
        "Status": (
            "PASS"
            if regression_missing == 0
            else "FAIL"
        ),
        "Details": (
            f"Missing={regression_missing}"
        ),
    },
    {
        "Check": "Remaining drying time is never negative",
        "Status": (
            "PASS"
            if negative_remaining_count == 0
            else "FAIL"
        ),
        "Details": (
            f"Negative rows="
            f"{negative_remaining_count}"
        ),
    },
    {
        "Check": "Remaining-time construction formula is valid",
        "Status": (
            "PASS"
            if formula_error_rows == 0
            else "FAIL"
        ),
        "Details": (
            f"Rows above tolerance="
            f"{formula_error_rows}; "
            f"Max abs error="
            f"{max_formula_error:.6f} h"
        ),
    },
    {
        "Check": "All sequences start at elapsed time zero",
        "Status": (
            "PASS"
            if start_elapsed_violations == 0
            else "FAIL"
        ),
        "Details": (
            f"Violations="
            f"{start_elapsed_violations}"
        ),
    },
    {
        "Check": "All sequences end at remaining time zero",
        "Status": (
            "PASS"
            if end_remaining_violations == 0
            else "FAIL"
        ),
        "Details": (
            f"Violations="
            f"{end_remaining_violations}"
        ),
    },
    {
        "Check": "Final elapsed time matches total drying time",
        "Status": (
            "PASS"
            if final_elapsed_total_violations == 0
            else "FAIL"
        ),
        "Details": (
            f"Violations="
            f"{final_elapsed_total_violations}"
        ),
    },
    {
        "Check": "Actual total drying time is constant within sequence",
        "Status": (
            "PASS"
            if total_constant_violations == 0
            else "FAIL"
        ),
        "Details": (
            f"Violations="
            f"{total_constant_violations}"
        ),
    },
    {
        "Check": "Elapsed time strictly increases within sequence",
        "Status": (
            "PASS"
            if elapsed_monotonic_violations == 0
            else "FAIL"
        ),
        "Details": (
            f"Sequences with violations="
            f"{elapsed_monotonic_violations}"
        ),
    },
    {
        "Check": "Remaining time never increases within sequence",
        "Status": (
            "PASS"
            if remaining_monotonic_violations == 0
            else "FAIL"
        ),
        "Details": (
            f"Sequences with violations="
            f"{remaining_monotonic_violations}"
        ),
    },
    {
        "Check": "Fire-risk target has no missing values",
        "Status": (
            "PASS"
            if fire_missing == 0
            else "FAIL"
        ),
        "Details": (
            f"Missing={fire_missing}"
        ),
    },
    {
        "Check": "Fire-risk labels are exactly Normal/Warning/Critical",
        "Status": (
            "PASS"
            if (
                len(unexpected_fire_labels) == 0
                and
                len(missing_expected_fire_labels) == 0
            )
            else "FAIL"
        ),
        "Details": (
            f"Observed="
            f"{sorted(actual_fire_labels)}"
        ),
    },
    {
        "Check": "Sensor and assessment timestamps are valid",
        "Status": (
            "PASS"
            if (
                timestamp_missing == 0
                and
                assessment_missing == 0
            )
            else "FAIL"
        ),
        "Details": (
            f"Sensor invalid/missing="
            f"{timestamp_missing}; "
            f"Assessment invalid/missing="
            f"{assessment_missing}"
        ),
    },
    {
        "Check": "Each row uses the latest valid hourly assessment",
        "Status": (
            "PASS"
            if assessment_window_invalid_count == 0
            else "FAIL"
        ),
        "Details": (
            f"Invalid rows="
            f"{assessment_window_invalid_count}"
        ),
    },
    {
        "Check": "Fire-risk label is constant inside assessment window",
        "Status": (
            "PASS"
            if assessment_label_inconsistencies == 0
            else "FAIL"
        ),
        "Details": (
            f"Inconsistent assessment groups="
            f"{assessment_label_inconsistencies}"
        ),
    },
    {
        "Check": "Hourly assessment cadence is valid",
        "Status": (
            "PASS"
            if assessment_cadence_errors == 0
            else "FAIL"
        ),
        "Details": (
            f"Cadence errors="
            f"{assessment_cadence_errors}"
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

failed_checks = (
    checks_df[
        checks_df[
            "Status"
        ]
        == "FAIL"
    ]
)

overall_status = (
    "PASS"
    if len(
        failed_checks
    ) == 0
    else "FAIL"
)


# ============================================================
# 20. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- DATASET IDENTITY ---"
)

print(
    f"SHA-256              : "
    f"{dataset_sha256}"
)

print(
    f"Frozen hash matches  : "
    f"{hash_matches}"
)


print(
    "\n--- REGRESSION TARGET: "
    "Remaining_Drying_Time ---"
)

print(
    f"Missing target values          : "
    f"{regression_missing}"
)

print(
    f"Target range                   : "
    f"{regression_target.min():.2f} "
    f"to "
    f"{regression_target.max():.2f} h"
)

print(
    f"Negative target values         : "
    f"{negative_remaining_count}"
)

print(
    f"Max formula discrepancy        : "
    f"{max_formula_error:.6f} h"
)

print(
    f"Formula errors > "
    f"{FORMULA_TOLERANCE_HOURS:.3f} h     : "
    f"{formula_error_rows}"
)

print(
    f"Batch-Layer sequences          : "
    f"{sequence_count}"
)

print(
    f"Start elapsed-time violations  : "
    f"{start_elapsed_violations}"
)

print(
    f"End remaining-time violations  : "
    f"{end_remaining_violations}"
)

print(
    f"Final elapsed/total violations : "
    f"{final_elapsed_total_violations}"
)

print(
    f"Total-time consistency errors  : "
    f"{total_constant_violations}"
)

print(
    f"Elapsed monotonicity errors     : "
    f"{elapsed_monotonic_violations}"
)

print(
    f"Remaining monotonicity errors   : "
    f"{remaining_monotonic_violations}"
)


print(
    "\n--- CLASSIFICATION TARGET: "
    "Fire_Risk_Label ---"
)

print(
    f"Missing target values          : "
    f"{fire_missing}"
)

print(
    f"Observed labels                : "
    f"{sorted(actual_fire_labels)}"
)

print(
    f"Unique hourly assessments      : "
    f"{unique_assessment_count:,}"
)

print(
    f"Assessment-window errors       : "
    f"{assessment_window_invalid_count}"
)

print(
    f"Within-assessment inconsistencies: "
    f"{assessment_label_inconsistencies}"
)

print(
    f"Assessment cadence errors      : "
    f"{assessment_cadence_errors}"
)

print(
    f"Batches containing Warning     : "
    f"{batches_with_warning}"
    f"/12"
)

print(
    f"Batches containing Critical    : "
    f"{batches_with_critical}"
    f"/12"
)


print(
    "\n--- ROW-LEVEL FIRE-RISK DISTRIBUTION ---"
)

print(
    fire_row_distribution
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
    assessment_distribution
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
    "\n--- FUTURE MODEL EXCLUSION RULES ---"
)

print(
    "Regression target      : "
    "Remaining_Drying_Time"
)

print(
    "Classification target  : "
    "Fire_Risk_Label"
)

print(
    "Group-only field       : "
    "Batch_ID"
)

print(
    "Never physical predictors:"
)

for column in (
    NEVER_USE_AS_PHYSICAL_PREDICTOR
):
    print(
        f"  - {column}"
    )


# ============================================================
# 21. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 02: TARGET VALIDATION\n"
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
        "DATASET IDENTITY\n"
        "----------------\n"
    )

    report.write(
        f"SHA-256: "
        f"{dataset_sha256}\n"
    )

    report.write(
        f"Expected frozen hash match: "
        f"{hash_matches}\n\n"
    )

    report.write(
        "REGRESSION TARGET\n"
        "-----------------\n"
    )

    report.write(
        "Target: "
        "Remaining_Drying_Time\n"
    )

    report.write(
        f"Missing values: "
        f"{regression_missing}\n"
    )

    report.write(
        f"Range: "
        f"{regression_target.min():.2f} "
        f"to "
        f"{regression_target.max():.2f} h\n"
    )

    report.write(
        f"Negative values: "
        f"{negative_remaining_count}\n"
    )

    report.write(
        f"Maximum construction-formula "
        f"discrepancy: "
        f"{max_formula_error:.6f} h\n"
    )

    report.write(
        f"Rows above "
        f"{FORMULA_TOLERANCE_HOURS:.3f} h "
        f"tolerance: "
        f"{formula_error_rows}\n"
    )

    report.write(
        f"Batch-Layer sequences: "
        f"{sequence_count}\n"
    )

    report.write(
        f"Start elapsed-time violations: "
        f"{start_elapsed_violations}\n"
    )

    report.write(
        f"End remaining-time violations: "
        f"{end_remaining_violations}\n"
    )

    report.write(
        f"Final elapsed vs total violations: "
        f"{final_elapsed_total_violations}\n"
    )

    report.write(
        f"Total-time constancy violations: "
        f"{total_constant_violations}\n"
    )

    report.write(
        f"Elapsed monotonicity violations: "
        f"{elapsed_monotonic_violations}\n"
    )

    report.write(
        f"Remaining monotonicity violations: "
        f"{remaining_monotonic_violations}\n\n"
    )

    report.write(
        "CLASSIFICATION TARGET\n"
        "---------------------\n"
    )

    report.write(
        "Target: "
        "Fire_Risk_Label\n"
    )

    report.write(
        f"Missing values: "
        f"{fire_missing}\n"
    )

    report.write(
        f"Observed labels: "
        f"{sorted(actual_fire_labels)}\n"
    )

    report.write(
        f"Unique hourly assessment events: "
        f"{unique_assessment_count:,}\n"
    )

    report.write(
        f"Assessment-window errors: "
        f"{assessment_window_invalid_count}\n"
    )

    report.write(
        f"Within-assessment label "
        f"inconsistencies: "
        f"{assessment_label_inconsistencies}\n"
    )

    report.write(
        f"Hourly cadence errors: "
        f"{assessment_cadence_errors}\n"
    )

    report.write(
        f"Batches containing Warning: "
        f"{batches_with_warning}/12\n"
    )

    report.write(
        f"Batches containing Critical: "
        f"{batches_with_critical}/12\n\n"
    )

    report.write(
        "ROW-LEVEL FIRE-RISK DISTRIBUTION\n"
        "--------------------------------\n"
    )

    report.write(
        fire_row_distribution
        .round(
            {
                "Percentage": 4
            }
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
        assessment_distribution
        .round(
            {
                "Percentage": 4
            }
        )
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "MODEL-LEAKAGE RULES\n"
        "-------------------\n"
    )

    report.write(
        "- Remaining_Drying_Time is the "
        "regression target.\n"
    )

    report.write(
        "- Fire_Risk_Label is the "
        "classification target.\n"
    )

    report.write(
        "- Actual_Total_Drying_Time is "
        "retrospective target-construction/reference "
        "information and must never be a deployed "
        "prediction feature.\n"
    )

    report.write(
        "- Fire_Risk_Assessment_Timestamp is "
        "ground-truth assessment metadata and must "
        "not be used as a physical prediction feature.\n"
    )

    report.write(
        "- Batch_ID is required for grouping and "
        "future batch-aware splitting, but must not "
        "be used as a physical prediction feature.\n"
    )

    report.write(
        "- The row-level Fire_Risk_Label is the "
        "latest hourly keeper/expert assessment "
        "carried forward across the corresponding "
        "10-minute sensor rows. Therefore, repeated "
        "rows within an hourly assessment must not "
        "be interpreted as independent expert "
        "assessments.\n\n"
    )

    report.write(
        "PASS / FAIL CHECKS\n"
        "------------------\n"
    )

    report.write(
        checks_df
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        f"STAGE 02 STATUS: "
        f"{overall_status}\n"
    )


# ============================================================
# 22. COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    f"STAGE 02 TARGET VALIDATION: "
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
    "\nNo raw data was modified."
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 02 target validation failed. "
        "Review 02_target_validation_report.txt "
        "before continuing."
    )
