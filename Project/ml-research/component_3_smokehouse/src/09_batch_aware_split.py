from pathlib import Path
import hashlib
import pandas as pd
import numpy as np


# ============================================================
# COMPONENT 3 - STAGE 09: BATCH-AWARE TRAIN/VAL/TEST SPLIT
# ============================================================
#
# Purpose:
#   Split the Stage-07 feature-engineered dataset into
#   TRAIN / VALIDATION / TEST sets using COMPLETE batches.
#
# Critical rule:
#   NEVER randomly split individual 10-minute rows.
#
# Grouping unit:
#   Smokehouse_ID + Batch_ID
#
# Dataset hierarchy:
#   Smokehouse -> Batch -> Layer -> 10-minute observations
#
# Split design:
#   8 complete batches -> Train
#   2 complete batches -> Validation
#   2 complete batches -> Test
#
# Approximate independent-batch proportions:
#   Train      = 66.7%
#   Validation = 16.7%
#   Test       = 16.7%
#
# Why fixed group-stratified rather than purely chronological?
#   A purely chronological split would put both B05 batches
#   into validation, but both are all-Normal at the unique
#   hourly fire-risk assessment level. That would make
#   validation unsuitable for Warning/Critical evaluation.
#
# The fixed assignment below:
#   - keeps complete batches together
#   - includes both smokehouses in every split
#   - includes L1/L2/L3 in every split
#   - includes Normal/Warning/Critical assessment events in
#     every split
#   - gives the test set both a long-duration and a short-
#     duration batch for regression evaluation
#
# IMPORTANT:
#   Once created, the TEST split must remain untouched during
#   model comparison, tuning, feature decisions, threshold
#   decisions, or preprocessing fitting.
#
# Stage 10 will fit learned preprocessing on TRAIN only.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "Component3_Dataset_FeatureEngineered.csv"
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

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

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

REPORT_PATH = (
    REPORT_DIR
    / "09_batch_aware_split_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "09_batch_aware_split_checks.csv"
)

MANIFEST_PATH = (
    REPORT_DIR
    / "09_batch_split_manifest.csv"
)

SPLIT_SUMMARY_PATH = (
    REPORT_DIR
    / "09_split_summary.csv"
)

FIRE_SUMMARY_PATH = (
    REPORT_DIR
    / "09_fire_risk_split_summary.csv"
)

REGRESSION_SUMMARY_PATH = (
    REPORT_DIR
    / "09_regression_split_summary.csv"
)

MISSINGNESS_PATH = (
    REPORT_DIR
    / "09_selected_feature_missingness_by_split.csv"
)

REG_FEATURE_LIST_PATH = (
    REPORT_DIR
    / "08_regression_selected_features.csv"
)

CLS_FEATURE_LIST_PATH = (
    REPORT_DIR
    / "08_classification_selected_features.csv"
)


# ============================================================
# 3. EXPECTED DATASET STRUCTURE
# ============================================================

EXPECTED_ROWS = 24463

EXPECTED_BATCHES = {
    "SH01_B01",
    "SH01_B02",
    "SH01_B03",
    "SH01_B04",
    "SH01_B05",
    "SH01_B06",
    "SH02_B01",
    "SH02_B02",
    "SH02_B03",
    "SH02_B04",
    "SH02_B05",
    "SH02_B06",
}

SEQUENCE_KEY = [
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
]


# ============================================================
# 4. FIXED BATCH ASSIGNMENT
# ============================================================
#
# The assignment is now part of the reproducible methodology.
# Do not reshuffle these batches later because model results
# would otherwise no longer be directly comparable.
#
# TEST deliberately includes:
#   SH01_B03 -> long drying regime
#   SH02_B04 -> shorter drying regime + substantial Warning
#
# TRAIN retains the majority of Critical assessment events.

TRAIN_BATCHES = {
    "SH01_B01",
    "SH01_B04",
    "SH01_B05",
    "SH01_B06",

    "SH02_B01",
    "SH02_B03",
    "SH02_B05",
    "SH02_B06",
}

VALIDATION_BATCHES = {
    "SH01_B02",
    "SH02_B02",
}

TEST_BATCHES = {
    "SH01_B03",
    "SH02_B04",
}


# ============================================================
# 5. LOAD INPUT DATASET
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 09: BATCH-AWARE TRAIN/VAL/TEST SPLIT")
print("=" * 78)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"\nFeature-engineered dataset not found:\n{INPUT_PATH}\n\n"
        "Run Stage 07 first."
    )

if not REG_FEATURE_LIST_PATH.exists():
    raise FileNotFoundError(
        f"\nStage-08 regression feature list not found:\n"
        f"{REG_FEATURE_LIST_PATH}\n\n"
        "Run Stage 08 first."
    )

if not CLS_FEATURE_LIST_PATH.exists():
    raise FileNotFoundError(
        f"\nStage-08 classification feature list not found:\n"
        f"{CLS_FEATURE_LIST_PATH}\n\n"
        "Run Stage 08 first."
    )

with open(
    INPUT_PATH,
    "rb",
) as file:
    input_sha256 = hashlib.sha256(
        file.read()
    ).hexdigest()

df = pd.read_csv(
    INPUT_PATH
)

print("\nFeature-engineered dataset loaded successfully.")


# ============================================================
# 6. TYPE CONVERSION
# ============================================================

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="raise",
)

df[
    "Fire_Risk_Assessment_Timestamp"
] = pd.to_datetime(
    df[
        "Fire_Risk_Assessment_Timestamp"
    ],
    errors="raise",
)

df[
    "Remaining_Drying_Time"
] = pd.to_numeric(
    df[
        "Remaining_Drying_Time"
    ],
    errors="raise",
)

df = df.sort_values(
    SEQUENCE_KEY
    + ["Timestamp"]
).reset_index(
    drop=True
)


# ============================================================
# 7. READ STAGE-08 SELECTED FEATURE LISTS
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
    .tolist()
)

CLASSIFICATION_FEATURES = (
    cls_feature_df[
        "Feature"
    ]
    .tolist()
)

if len(
    REGRESSION_FEATURES
) != 17:
    raise RuntimeError(
        "Expected 17 Stage-08 regression features."
    )

if len(
    CLASSIFICATION_FEATURES
) != 17:
    raise RuntimeError(
        "Expected 17 Stage-08 classification features."
    )


# ============================================================
# 8. VERIFY OBSERVED BATCH SET
# ============================================================

observed_batches = set(
    df[
        "Batch_ID"
    ]
    .dropna()
    .unique()
)

if observed_batches != EXPECTED_BATCHES:
    raise RuntimeError(
        "Observed batch identifiers do not match the "
        "expected 12-batch final dataset."
    )

assignment_union = (
    TRAIN_BATCHES
    |
    VALIDATION_BATCHES
    |
    TEST_BATCHES
)

assignment_overlap_count = (
    len(
        TRAIN_BATCHES
        &
        VALIDATION_BATCHES
    )
    +
    len(
        TRAIN_BATCHES
        &
        TEST_BATCHES
    )
    +
    len(
        VALIDATION_BATCHES
        &
        TEST_BATCHES
    )
)

if assignment_union != EXPECTED_BATCHES:
    raise RuntimeError(
        "Fixed split assignment does not cover all 12 batches."
    )

if assignment_overlap_count != 0:
    raise RuntimeError(
        "A batch appears in more than one split."
    )


# ============================================================
# 9. CREATE SPLITS
# ============================================================

train_df = (
    df[
        df[
            "Batch_ID"
        ]
        .isin(
            TRAIN_BATCHES
        )
    ]
    .copy()
)

validation_df = (
    df[
        df[
            "Batch_ID"
        ]
        .isin(
            VALIDATION_BATCHES
        )
    ]
    .copy()
)

test_df = (
    df[
        df[
            "Batch_ID"
        ]
        .isin(
            TEST_BATCHES
        )
    ]
    .copy()
)

split_frames = {
    "Train": train_df,
    "Validation": validation_df,
    "Test": test_df,
}


# ============================================================
# 10. SAVE SPLIT DATASETS
# ============================================================

for split_df, path in [
    (
        train_df,
        TRAIN_PATH,
    ),
    (
        validation_df,
        VALIDATION_PATH,
    ),
    (
        test_df,
        TEST_PATH,
    ),
]:
    split_df.to_csv(
        path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )


# ============================================================
# 11. SPLIT FILE HASHES
# ============================================================

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


train_sha256 = sha256_file(
    TRAIN_PATH
)

validation_sha256 = sha256_file(
    VALIDATION_PATH
)

test_sha256 = sha256_file(
    TEST_PATH
)


# ============================================================
# 12. GROUP OVERLAP VALIDATION
# ============================================================

train_groups = set(
    train_df[
        [
            "Smokehouse_ID",
            "Batch_ID",
        ]
    ]
    .drop_duplicates()
    .itertuples(
        index=False,
        name=None,
    )
)

validation_groups = set(
    validation_df[
        [
            "Smokehouse_ID",
            "Batch_ID",
        ]
    ]
    .drop_duplicates()
    .itertuples(
        index=False,
        name=None,
    )
)

test_groups = set(
    test_df[
        [
            "Smokehouse_ID",
            "Batch_ID",
        ]
    ]
    .drop_duplicates()
    .itertuples(
        index=False,
        name=None,
    )
)

train_val_overlap = len(
    train_groups
    &
    validation_groups
)

train_test_overlap = len(
    train_groups
    &
    test_groups
)

val_test_overlap = len(
    validation_groups
    &
    test_groups
)


# ============================================================
# 13. OBSERVATION-KEY OVERLAP VALIDATION
# ============================================================

def observation_keys(
    frame,
):
    return set(
        frame[
            SEQUENCE_KEY
            + ["Timestamp"]
        ]
        .itertuples(
            index=False,
            name=None,
        )
    )


train_obs = observation_keys(
    train_df
)

validation_obs = observation_keys(
    validation_df
)

test_obs = observation_keys(
    test_df
)

observation_overlap_count = (
    len(
        train_obs
        &
        validation_obs
    )
    +
    len(
        train_obs
        &
        test_obs
    )
    +
    len(
        validation_obs
        &
        test_obs
    )
)


# ============================================================
# 14. SPLIT SUMMARY
# ============================================================

split_summary_rows = []

for split_name, split_df in (
    split_frames.items()
):
    split_summary_rows.append({
        "Split": split_name,
        "Rows": len(
            split_df
        ),
        "Row_Percentage": (
            len(
                split_df
            )
            / len(
                df
            )
            * 100
        ),
        "Batches": int(
            split_df[
                "Batch_ID"
            ]
            .nunique()
        ),
        "Batch_Percentage": (
            split_df[
                "Batch_ID"
            ]
            .nunique()
            / 12
            * 100
        ),
        "Smokehouses": int(
            split_df[
                "Smokehouse_ID"
            ]
            .nunique()
        ),
        "Layers": int(
            split_df[
                "Layer_ID"
            ]
            .nunique()
        ),
        "Batch_Layer_Sequences": int(
            split_df.groupby(
                SEQUENCE_KEY
            ).ngroups
        ),
    })

split_summary = pd.DataFrame(
    split_summary_rows
)

split_summary.to_csv(
    SPLIT_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 15. BATCH MANIFEST
# ============================================================

batch_manifest_rows = []

for split_name, split_df in (
    split_frames.items()
):
    assessment_rows = (
        split_df[
            split_df[
                "Timestamp"
            ]
            ==
            split_df[
                "Fire_Risk_Assessment_Timestamp"
            ]
        ]
    )

    for (
        smokehouse_id,
        batch_id,
    ), batch_df in split_df.groupby(
        [
            "Smokehouse_ID",
            "Batch_ID",
        ],
        sort=True,
    ):
        batch_assessments = (
            assessment_rows[
                assessment_rows[
                    "Batch_ID"
                ]
                ==
                batch_id
            ]
        )

        risk_counts = (
            batch_assessments[
                "Fire_Risk_Label"
            ]
            .value_counts()
        )

        sequence_total_hours = (
            batch_df.groupby(
                "Layer_ID"
            )[
                "Remaining_Drying_Time"
            ]
            .max()
        )

        batch_manifest_rows.append({
            "Split": split_name,
            "Smokehouse_ID": (
                smokehouse_id
            ),
            "Batch_ID": (
                batch_id
            ),
            "Rows": len(
                batch_df
            ),
            "Layers": int(
                batch_df[
                    "Layer_ID"
                ]
                .nunique()
            ),
            "Start_Timestamp": (
                batch_df[
                    "Timestamp"
                ]
                .min()
            ),
            "End_Timestamp": (
                batch_df[
                    "Timestamp"
                ]
                .max()
            ),
            "Mean_Sequence_Total_Drying_Hours": (
                float(
                    sequence_total_hours
                    .mean()
                )
            ),
            "Min_Sequence_Total_Drying_Hours": (
                float(
                    sequence_total_hours
                    .min()
                )
            ),
            "Max_Sequence_Total_Drying_Hours": (
                float(
                    sequence_total_hours
                    .max()
                )
            ),
            "Hourly_Normal_Assessments": int(
                risk_counts.get(
                    "Normal",
                    0,
                )
            ),
            "Hourly_Warning_Assessments": int(
                risk_counts.get(
                    "Warning",
                    0,
                )
            ),
            "Hourly_Critical_Assessments": int(
                risk_counts.get(
                    "Critical",
                    0,
                )
            ),
        })

batch_manifest = pd.DataFrame(
    batch_manifest_rows
)

batch_manifest.to_csv(
    MANIFEST_PATH,
    index=False,
)


# ============================================================
# 16. FIRE-RISK SPLIT SUMMARY
# ============================================================

fire_summary_rows = []

for split_name, split_df in (
    split_frames.items()
):
    assessment_df = (
        split_df[
            split_df[
                "Timestamp"
            ]
            ==
            split_df[
                "Fire_Risk_Assessment_Timestamp"
            ]
        ]
        .copy()
    )

    total_assessments = len(
        assessment_df
    )

    for label in [
        "Normal",
        "Warning",
        "Critical",
    ]:
        count = int(
            (
                assessment_df[
                    "Fire_Risk_Label"
                ]
                ==
                label
            )
            .sum()
        )

        fire_summary_rows.append({
            "Split": (
                split_name
            ),
            "Fire_Risk_Label": (
                label
            ),
            "Unique_Hourly_Assessment_Count": (
                count
            ),
            "Assessment_Percentage": (
                count
                / total_assessments
                * 100
            ),
        })

fire_summary = pd.DataFrame(
    fire_summary_rows
)

fire_summary.to_csv(
    FIRE_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 17. REGRESSION TARGET SPLIT SUMMARY
# ============================================================

regression_summary_rows = []

for split_name, split_df in (
    split_frames.items()
):
    sequence_duration = (
        split_df.groupby(
            SEQUENCE_KEY
        )[
            "Remaining_Drying_Time"
        ]
        .max()
    )

    regression_summary_rows.append({
        "Split": split_name,
        "Rows": len(
            split_df
        ),
        "Target_Min_Hours": float(
            split_df[
                "Remaining_Drying_Time"
            ]
            .min()
        ),
        "Target_Max_Hours": float(
            split_df[
                "Remaining_Drying_Time"
            ]
            .max()
        ),
        "Target_Mean_Hours": float(
            split_df[
                "Remaining_Drying_Time"
            ]
            .mean()
        ),
        "Target_SD_Hours": float(
            split_df[
                "Remaining_Drying_Time"
            ]
            .std()
        ),
        "Sequence_Count": len(
            sequence_duration
        ),
        "Mean_Sequence_Total_Drying_Hours": float(
            sequence_duration
            .mean()
        ),
        "Min_Sequence_Total_Drying_Hours": float(
            sequence_duration
            .min()
        ),
        "Max_Sequence_Total_Drying_Hours": float(
            sequence_duration
            .max()
        ),
    })

regression_summary = pd.DataFrame(
    regression_summary_rows
)

regression_summary.to_csv(
    REGRESSION_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 18. SELECTED-FEATURE MISSINGNESS BY SPLIT
# ============================================================

selected_union = list(
    dict.fromkeys(
        REGRESSION_FEATURES
        +
        CLASSIFICATION_FEATURES
    )
)

missingness_rows = []

for split_name, split_df in (
    split_frames.items()
):
    for feature in (
        selected_union
    ):
        missing_count = int(
            split_df[
                feature
            ]
            .isna()
            .sum()
        )

        missingness_rows.append({
            "Split": (
                split_name
            ),
            "Feature": (
                feature
            ),
            "Missing_Count": (
                missing_count
            ),
            "Missing_Percent": (
                missing_count
                / len(
                    split_df
                )
                * 100
            ),
        })

missingness = pd.DataFrame(
    missingness_rows
)

missingness.to_csv(
    MISSINGNESS_PATH,
    index=False,
)


# ============================================================
# 19. CRITICAL SPLIT VALIDATION
# ============================================================

row_total_preserved = (
    len(
        train_df
    )
    +
    len(
        validation_df
    )
    +
    len(
        test_df
    )
    ==
    len(
        df
    )
)

expected_batch_counts = (
    train_df[
        "Batch_ID"
    ]
    .nunique()
    == 8
    and
    validation_df[
        "Batch_ID"
    ]
    .nunique()
    == 2
    and
    test_df[
        "Batch_ID"
    ]
    .nunique()
    == 2
)

both_smokehouses_everywhere = all(
    split_df[
        "Smokehouse_ID"
    ]
    .nunique()
    == 2
    for split_df in (
        split_frames.values()
    )
)

all_layers_everywhere = all(
    set(
        split_df[
            "Layer_ID"
        ]
        .unique()
    )
    ==
    {
        "L1",
        "L2",
        "L3",
    }
    for split_df in (
        split_frames.values()
    )
)

all_risk_classes_everywhere = True

assessment_counts_for_console = {}

for split_name, split_df in (
    split_frames.items()
):
    assessment_df = (
        split_df[
            split_df[
                "Timestamp"
            ]
            ==
            split_df[
                "Fire_Risk_Assessment_Timestamp"
            ]
        ]
    )

    label_counts = (
        assessment_df[
            "Fire_Risk_Label"
        ]
        .value_counts()
    )

    assessment_counts_for_console[
        split_name
    ] = {
        "Normal": int(
            label_counts.get(
                "Normal",
                0,
            )
        ),
        "Warning": int(
            label_counts.get(
                "Warning",
                0,
            )
        ),
        "Critical": int(
            label_counts.get(
                "Critical",
                0,
            )
        ),
        "Total": len(
            assessment_df
        ),
    }

    if not {
        "Normal",
        "Warning",
        "Critical",
    }.issubset(
        set(
            assessment_df[
                "Fire_Risk_Label"
            ]
            .unique()
        )
    ):
        all_risk_classes_everywhere = False


# Regression target coverage
regression_target_complete = all(
    split_df[
        "Remaining_Drying_Time"
    ]
    .notna()
    .all()
    for split_df in (
        split_frames.values()
    )
)

# Ensure test remains ordinary raw Stage-07 rows, with no
# train-fitted transformations added.
split_columns_preserved = all(
    list(
        split_df.columns
    )
    ==
    list(
        df.columns
    )
    for split_df in (
        split_frames.values()
    )
)


# ============================================================
# 20. PASS / FAIL CHECKS
# ============================================================

checks = pd.DataFrame(
    [
        {
            "Check": (
                "All original rows assigned exactly once"
            ),
            "Status": (
                "PASS"
                if (
                    row_total_preserved
                    and
                    observation_overlap_count
                    == 0
                )
                else "FAIL"
            ),
            "Details": (
                f"Original={len(df):,}; "
                f"Train+Val+Test="
                f"{len(train_df)+len(validation_df)+len(test_df):,}; "
                f"Observation overlap={observation_overlap_count}"
            ),
        },
        {
            "Check": (
                "No batch-group overlap"
            ),
            "Status": (
                "PASS"
                if (
                    train_val_overlap
                    ==
                    0
                    and
                    train_test_overlap
                    ==
                    0
                    and
                    val_test_overlap
                    ==
                    0
                )
                else "FAIL"
            ),
            "Details": (
                f"Train-Val={train_val_overlap}; "
                f"Train-Test={train_test_overlap}; "
                f"Val-Test={val_test_overlap}"
            ),
        },
        {
            "Check": (
                "Expected 8/2/2 whole-batch split"
            ),
            "Status": (
                "PASS"
                if expected_batch_counts
                else "FAIL"
            ),
            "Details": (
                f"Train={train_df['Batch_ID'].nunique()}; "
                f"Validation={validation_df['Batch_ID'].nunique()}; "
                f"Test={test_df['Batch_ID'].nunique()}"
            ),
        },
        {
            "Check": (
                "Both smokehouses represented in every split"
            ),
            "Status": (
                "PASS"
                if both_smokehouses_everywhere
                else "FAIL"
            ),
            "Details": (
                "Each split must include SH01 and SH02."
            ),
        },
        {
            "Check": (
                "All three layers represented in every split"
            ),
            "Status": (
                "PASS"
                if all_layers_everywhere
                else "FAIL"
            ),
            "Details": (
                "Each split must include L1/L2/L3."
            ),
        },
        {
            "Check": (
                "All three fire-risk classes represented in every split"
            ),
            "Status": (
                "PASS"
                if all_risk_classes_everywhere
                else "FAIL"
            ),
            "Details": (
                "Checked on unique hourly keeper/expert assessments."
            ),
        },
        {
            "Check": (
                "Regression target complete in all splits"
            ),
            "Status": (
                "PASS"
                if regression_target_complete
                else "FAIL"
            ),
            "Details": (
                "Remaining_Drying_Time contains no missing target values."
            ),
        },
        {
            "Check": (
                "Stage-07 columns preserved without transformation"
            ),
            "Status": (
                "PASS"
                if split_columns_preserved
                else "FAIL"
            ),
            "Details": (
                "No scaling/imputation/encoding is fitted in Stage 09."
            ),
        },
        {
            "Check": (
                "Stage-08 regression feature list loaded"
            ),
            "Status": (
                "PASS"
                if len(
                    REGRESSION_FEATURES
                )
                ==
                17
                else "FAIL"
            ),
            "Details": (
                f"Features="
                f"{len(REGRESSION_FEATURES)}"
            ),
        },
        {
            "Check": (
                "Stage-08 classification feature list loaded"
            ),
            "Status": (
                "PASS"
                if len(
                    CLASSIFICATION_FEATURES
                )
                ==
                17
                else "FAIL"
            ),
            "Details": (
                f"Features="
                f"{len(CLASSIFICATION_FEATURES)}"
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
# 21. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- INPUT ---"
)

print(
    f"Rows                            : "
    f"{len(df):,}"
)

print(
    f"Batches                         : "
    f"{df['Batch_ID'].nunique()}"
)

print(
    f"Stage-07 SHA-256                : "
    f"{input_sha256}"
)


print(
    "\n--- FIXED BATCH ASSIGNMENT ---"
)

print(
    "TRAIN:"
)

for batch in sorted(
    TRAIN_BATCHES
):
    print(
        f"  - {batch}"
    )

print(
    "\nVALIDATION:"
)

for batch in sorted(
    VALIDATION_BATCHES
):
    print(
        f"  - {batch}"
    )

print(
    "\nTEST:"
)

for batch in sorted(
    TEST_BATCHES
):
    print(
        f"  - {batch}"
    )


print(
    "\n--- SPLIT SIZES ---"
)

print(
    split_summary[
        [
            "Split",
            "Rows",
            "Row_Percentage",
            "Batches",
            "Batch_Layer_Sequences",
        ]
    ]
    .round(
        {
            "Row_Percentage": 4,
        }
    )
    .to_string(
        index=False
    )
)


print(
    "\n--- UNIQUE HOURLY FIRE-RISK ASSESSMENTS ---"
)

for split_name in [
    "Train",
    "Validation",
    "Test",
]:
    counts = (
        assessment_counts_for_console[
            split_name
        ]
    )

    print(
        f"{split_name:<10} "
        f"Normal={counts['Normal']:,} | "
        f"Warning={counts['Warning']:,} | "
        f"Critical={counts['Critical']:,} | "
        f"Total={counts['Total']:,}"
    )


print(
    "\n--- REGRESSION TARGET SUMMARY ---"
)

print(
    regression_summary[
        [
            "Split",
            "Target_Min_Hours",
            "Target_Max_Hours",
            "Target_Mean_Hours",
            "Mean_Sequence_Total_Drying_Hours",
            "Min_Sequence_Total_Drying_Hours",
            "Max_Sequence_Total_Drying_Hours",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


print(
    "\n--- LEAKAGE / OVERLAP CHECK ---"
)

print(
    f"Train-Val batch overlap         : "
    f"{train_val_overlap}"
)

print(
    f"Train-Test batch overlap        : "
    f"{train_test_overlap}"
)

print(
    f"Val-Test batch overlap          : "
    f"{val_test_overlap}"
)

print(
    f"Observation-key overlap         : "
    f"{observation_overlap_count}"
)

print(
    f"All classes in every split      : "
    f"{all_risk_classes_everywhere}"
)

print(
    f"Both smokehouses in every split : "
    f"{both_smokehouses_everywhere}"
)


print(
    "\n--- SPLIT FILES ---"
)

print(
    f"Train:\n{TRAIN_PATH}"
)

print(
    f"\nValidation:\n{VALIDATION_PATH}"
)

print(
    f"\nTest:\n{TEST_PATH}"
)

print(
    "\nSplit SHA-256 fingerprints:"
)

print(
    f"  Train      : "
    f"{train_sha256}"
)

print(
    f"  Validation : "
    f"{validation_sha256}"
)

print(
    f"  Test       : "
    f"{test_sha256}"
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
        "COMPONENT 3 - STAGE 09: BATCH-AWARE TRAIN/VAL/TEST SPLIT\n"
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
        "METHODOLOGY\n"
        "-----------\n"
    )

    report.write(
        "The 24,463 rows are repeated 10-minute observations "
        "nested inside 12 complete drying batches. Therefore "
        "individual rows are not treated as independent units "
        "for splitting. Complete Smokehouse_ID + Batch_ID "
        "groups are assigned to Train, Validation, or Test.\n\n"
    )

    report.write(
        "A purely chronological 8/2/2 batch split was rejected "
        "because the corresponding B05 validation batches "
        "contain no Warning/Critical hourly assessment events. "
        "A fixed group-stratified split is therefore used to "
        "retain both smokehouses, all three layers, and all "
        "three fire-risk classes in every split while keeping "
        "whole batches isolated.\n\n"
    )

    report.write(
        "The TEST split is now frozen and must not be used for "
        "model selection, hyperparameter tuning, preprocessing "
        "fitting, class-weight decisions, feature decisions, or "
        "threshold decisions.\n\n"
    )

    report.write(
        "FIXED BATCH ASSIGNMENT\n"
        "----------------------\n"
    )

    report.write(
        "TRAIN:\n"
    )

    for batch in sorted(
        TRAIN_BATCHES
    ):
        report.write(
            f"- {batch}\n"
        )

    report.write(
        "\nVALIDATION:\n"
    )

    for batch in sorted(
        VALIDATION_BATCHES
    ):
        report.write(
            f"- {batch}\n"
        )

    report.write(
        "\nTEST:\n"
    )

    for batch in sorted(
        TEST_BATCHES
    ):
        report.write(
            f"- {batch}\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "SPLIT SUMMARY\n"
        "-------------\n"
    )

    report.write(
        split_summary
        .round(6)
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "UNIQUE HOURLY FIRE-RISK ASSESSMENTS\n"
        "-----------------------------------\n"
    )

    report.write(
        fire_summary
        .round(6)
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "REGRESSION TARGET COVERAGE\n"
        "--------------------------\n"
    )

    report.write(
        regression_summary
        .round(6)
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "LEAKAGE PROTECTION\n"
        "------------------\n"
    )

    report.write(
        "- No complete batch appears in more than one split.\n"
    )

    report.write(
        "- No Smokehouse+Batch+Layer+Timestamp observation "
        "appears in more than one split.\n"
    )

    report.write(
        "- Stage-09 does not impute engineered missing values.\n"
    )

    report.write(
        "- Stage-09 does not fit scaling or encoding.\n"
    )

    report.write(
        "- Stage-09 does not perform SMOTE/resampling.\n"
    )

    report.write(
        "- Stage-09 does not train any model.\n"
    )

    report.write(
        "- Stage-10 will fit preprocessing using TRAIN only "
        "and then apply those fitted transformations to "
        "Validation/Test.\n"
    )

    report.write(
        "- Final Test must remain untouched until Stage 31.\n\n"
    )

    report.write(
        "SPLIT FILE FINGERPRINTS\n"
        "-----------------------\n"
    )

    report.write(
        f"Train SHA-256: "
        f"{train_sha256}\n"
    )

    report.write(
        f"Validation SHA-256: "
        f"{validation_sha256}\n"
    )

    report.write(
        f"Test SHA-256: "
        f"{test_sha256}\n\n"
    )

    report.write(
        "PASS / FAIL CHECKS\n"
        "------------------\n"
    )

    report.write(
        checks
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        f"STAGE 09 STATUS: "
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
    f"STAGE 09 BATCH-AWARE SPLIT: "
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
    "\nNo preprocessing was fitted and no model was trained."
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 09 failed. Review 09_batch_aware_split_report.txt."
    )
