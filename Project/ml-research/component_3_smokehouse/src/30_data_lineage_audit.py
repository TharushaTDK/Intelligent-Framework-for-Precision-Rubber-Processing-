from pathlib import Path
import pandas as pd


# ============================================================
# COMPONENT 3 - STAGE 30:
# FULL DATA-LINEAGE AUDIT
# ============================================================
#
# Purpose:
#   Verify Fire_Risk_Label lineage across:
#
#   Stage 4 Preprocessed
#          ↓
#   Stage 7 Feature Engineered
#          ↓
#   Stage 9 Train + Validation
#
# IMPORTANT:
#   - Component3_Test.csv is NOT loaded.
#   - No data is modified.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESSED_PATH = (
    PROCESSED_DIR
    / "Component3_Dataset_Preprocessed.csv"
)

FEATURE_ENGINEERED_PATH = (
    PROCESSED_DIR
    / "Component3_Dataset_FeatureEngineered.csv"
)

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
    / "30_full_data_lineage_audit.txt"
)


# ============================================================
# 1. REQUIRED FILES
# ============================================================

required_files = [
    PREPROCESSED_PATH,
    FEATURE_ENGINEERED_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# 2. START + FINAL TEST PROTECTION
# ============================================================

print("=" * 78)
print(
    "COMPONENT 3 - STAGE 30: "
    "FULL DATA-LINEAGE AUDIT"
)
print("=" * 78)

final_test_content_read = False

print("\n--- TEST PROTECTION ---")
print(
    "Final Test exists:",
    TEST_PATH.exists(),
)
print(
    "Component3_Test.csv content read:",
    final_test_content_read,
)


# ============================================================
# 3. LOAD NON-TEST DATA ONLY
# ============================================================

preprocessed = pd.read_csv(
    PREPROCESSED_PATH
)

feature_engineered = pd.read_csv(
    FEATURE_ENGINEERED_PATH
)

train = pd.read_csv(
    TRAIN_PATH
)

validation = pd.read_csv(
    VALIDATION_PATH
)

development = pd.concat(
    [
        train.assign(_Split="Train"),
        validation.assign(_Split="Validation"),
    ],
    ignore_index=True,
)


# ============================================================
# 4. STRUCTURE CHECK
# ============================================================

KEY_COLUMNS = [
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
    "Timestamp",
]

REQUIRED_COLUMNS = (
    KEY_COLUMNS
    + ["Fire_Risk_Label"]
)

datasets = {
    "Preprocessed": preprocessed,
    "FeatureEngineered": feature_engineered,
    "Development": development,
}

for name, frame in datasets.items():
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{missing}"
        )

    for column in REQUIRED_COLUMNS:
        frame[column] = (
            frame[column]
            .astype(str)
            .str.strip()
        )


# ============================================================
# 5. CLASS DISTRIBUTIONS
# ============================================================

RISK_ORDER = [
    "Normal",
    "Warning",
    "Critical",
]


def risk_counts(frame):
    return (
        frame["Fire_Risk_Label"]
        .value_counts()
        .reindex(
            RISK_ORDER,
            fill_value=0,
        )
    )


preprocessed_counts = risk_counts(
    preprocessed
)

feature_counts = risk_counts(
    feature_engineered
)

development_counts = risk_counts(
    development
)

train_counts = risk_counts(
    train
)

validation_counts = risk_counts(
    validation
)

print("\n--- ROW COUNTS ---")
print(
    f"Preprocessed       : "
    f"{len(preprocessed):,}"
)
print(
    f"Feature Engineered : "
    f"{len(feature_engineered):,}"
)
print(
    f"Train              : "
    f"{len(train):,}"
)
print(
    f"Validation         : "
    f"{len(validation):,}"
)
print(
    f"Train + Validation : "
    f"{len(development):,}"
)

print("\n--- FIRE-RISK COUNTS ---")

print("\nPreprocessed:")
print(
    preprocessed_counts.to_string()
)

print("\nFeature Engineered:")
print(
    feature_counts.to_string()
)

print("\nTrain:")
print(
    train_counts.to_string()
)

print("\nValidation:")
print(
    validation_counts.to_string()
)

print("\nTrain + Validation:")
print(
    development_counts.to_string()
)


# ============================================================
# 6. DUPLICATE KEY CHECKS
# ============================================================

print("\n--- OBSERVATION-KEY CHECK ---")

duplicate_results = {}

for name, frame in datasets.items():
    duplicate_rows = int(
        frame.duplicated(
            subset=KEY_COLUMNS,
            keep=False,
        ).sum()
    )

    unique_keys = int(
        frame.drop_duplicates(
            subset=KEY_COLUMNS
        ).shape[0]
    )

    duplicate_results[name] = (
        duplicate_rows
    )

    print(
        f"{name:<18} "
        f"duplicate-key rows = "
        f"{duplicate_rows:,}; "
        f"unique keys = "
        f"{unique_keys:,}"
    )


# ============================================================
# 7. PREPROCESSED -> FEATURE ENGINEERED
# ============================================================

pre_ref = (
    preprocessed[
        KEY_COLUMNS
        + ["Fire_Risk_Label"]
    ]
    .rename(
        columns={
            "Fire_Risk_Label":
                "Preprocessed_Label"
        }
    )
)

feature_compare = (
    feature_engineered.merge(
        pre_ref,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
)

feature_missing_from_pre = int(
    feature_compare[
        "Preprocessed_Label"
    ].isna().sum()
)

feature_label_change_mask = (
    feature_compare[
        "Preprocessed_Label"
    ].notna()
    & (
        feature_compare[
            "Fire_Risk_Label"
        ]
        != feature_compare[
            "Preprocessed_Label"
        ]
    )
)

feature_label_changes = int(
    feature_label_change_mask.sum()
)

print(
    "\n--- PREPROCESSED -> "
    "FEATURE ENGINEERED ---"
)

print(
    "Feature-engineered rows not found "
    "in Preprocessed:",
    feature_missing_from_pre,
)

print(
    "Rows whose Fire_Risk_Label changed:",
    feature_label_changes,
)

if feature_label_changes > 0:
    pre_to_feature_transition = (
        pd.crosstab(
            feature_compare.loc[
                feature_label_change_mask,
                "Preprocessed_Label",
            ],
            feature_compare.loc[
                feature_label_change_mask,
                "Fire_Risk_Label",
            ],
        )
    )

    print(
        "\nLabel transitions:"
    )
    print(
        pre_to_feature_transition.to_string()
    )

else:
    pre_to_feature_transition = (
        pd.DataFrame()
    )


# ============================================================
# 8. FEATURE ENGINEERED -> TRAIN + VALIDATION
# ============================================================

feature_ref = (
    feature_engineered[
        KEY_COLUMNS
        + ["Fire_Risk_Label"]
    ]
    .rename(
        columns={
            "Fire_Risk_Label":
                "FeatureEngineered_Label"
        }
    )
)

development_compare = (
    development.merge(
        feature_ref,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
)

development_missing_from_feature = int(
    development_compare[
        "FeatureEngineered_Label"
    ].isna().sum()
)

development_label_change_mask = (
    development_compare[
        "FeatureEngineered_Label"
    ].notna()
    & (
        development_compare[
            "Fire_Risk_Label"
        ]
        != development_compare[
            "FeatureEngineered_Label"
        ]
    )
)

development_label_changes = int(
    development_label_change_mask.sum()
)

print(
    "\n--- FEATURE ENGINEERED -> "
    "TRAIN + VALIDATION ---"
)

print(
    "Development rows not found in "
    "Feature Engineered:",
    development_missing_from_feature,
)

print(
    "Rows whose Fire_Risk_Label changed:",
    development_label_changes,
)

if development_label_changes > 0:
    feature_to_split_transition = (
        pd.crosstab(
            development_compare.loc[
                development_label_change_mask,
                "FeatureEngineered_Label",
            ],
            development_compare.loc[
                development_label_change_mask,
                "Fire_Risk_Label",
            ],
        )
    )

    print(
        "\nLabel transitions:"
    )
    print(
        feature_to_split_transition.to_string()
    )

else:
    feature_to_split_transition = (
        pd.DataFrame()
    )


# ============================================================
# 9. DIRECT PREPROCESSED -> TRAIN + VALIDATION
# ============================================================

development_vs_pre = (
    development.merge(
        pre_ref,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
)

development_missing_from_pre = int(
    development_vs_pre[
        "Preprocessed_Label"
    ].isna().sum()
)

direct_label_change_mask = (
    development_vs_pre[
        "Preprocessed_Label"
    ].notna()
    & (
        development_vs_pre[
            "Fire_Risk_Label"
        ]
        != development_vs_pre[
            "Preprocessed_Label"
        ]
    )
)

direct_label_changes = int(
    direct_label_change_mask.sum()
)

print(
    "\n--- DIRECT PREPROCESSED -> "
    "TRAIN + VALIDATION ---"
)

print(
    "Development rows not found in "
    "Preprocessed:",
    development_missing_from_pre,
)

print(
    "Rows whose Fire_Risk_Label differs "
    "from Preprocessed:",
    direct_label_changes,
)


# ============================================================
# 10. INTERPRETATION
# ============================================================

if (
    feature_missing_from_pre == 0
    and feature_label_changes == 0
    and development_missing_from_feature == 0
    and development_label_changes == 0
):
    status = "PASS"

    interpretation = (
        "Fire_Risk_Label lineage is preserved from "
        "Preprocessed -> Feature Engineered -> "
        "Train/Validation. No label changes were detected."
    )

elif (
    feature_missing_from_pre == 0
    and feature_label_changes > 0
):
    status = (
        "LABEL CHANGES DETECTED DURING "
        "FEATURE ENGINEERING"
    )

    interpretation = (
        "The same observations are present, but "
        "Fire_Risk_Label changed between the "
        "Preprocessed and Feature Engineered datasets. "
        "Stage 7 must be reviewed before continuing."
    )

elif (
    development_missing_from_feature == 0
    and development_label_changes > 0
):
    status = (
        "LABEL CHANGES DETECTED DURING "
        "SPLITTING / LATER PREPARATION"
    )

    interpretation = (
        "Feature-engineered observations match, but "
        "Fire_Risk_Label changed before Train/Validation. "
        "Stage 8/9/10 must be reviewed."
    )

else:
    status = (
        "DATASET VERSION / LINEAGE MISMATCH"
    )

    interpretation = (
        "Some observations cannot be matched across "
        "the expected lifecycle datasets. A different "
        "dataset version may have been used."
    )


print("\n--- FINAL AUDIT RESULT ---")
print("Status:", status)
print(interpretation)


# ============================================================
# 11. REPORT
# ============================================================

with REPORT_PATH.open(
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 30 "
        "FULL DATA-LINEAGE AUDIT\n"
    )
    report.write(
        "=" * 78 + "\n\n"
    )

    report.write(
        "ROW COUNTS\n"
        "----------\n"
    )
    report.write(
        f"Preprocessed: {len(preprocessed):,}\n"
    )
    report.write(
        f"Feature Engineered: "
        f"{len(feature_engineered):,}\n"
    )
    report.write(
        f"Train: {len(train):,}\n"
    )
    report.write(
        f"Validation: "
        f"{len(validation):,}\n\n"
    )

    report.write(
        "FIRE-RISK COUNTS\n"
        "----------------\n"
    )
    report.write(
        "Preprocessed:\n"
        + preprocessed_counts.to_string()
        + "\n\n"
    )
    report.write(
        "Feature Engineered:\n"
        + feature_counts.to_string()
        + "\n\n"
    )
    report.write(
        "Train + Validation:\n"
        + development_counts.to_string()
        + "\n\n"
    )

    report.write(
        "PREPROCESSED -> FEATURE ENGINEERED\n"
        "----------------------------------\n"
    )
    report.write(
        "Rows not found: "
        f"{feature_missing_from_pre}\n"
    )
    report.write(
        "Changed labels: "
        f"{feature_label_changes}\n\n"
    )

    if not pre_to_feature_transition.empty:
        report.write(
            "Label transitions:\n"
        )
        report.write(
            pre_to_feature_transition.to_string()
            + "\n\n"
        )

    report.write(
        "FEATURE ENGINEERED -> "
        "TRAIN + VALIDATION\n"
        "--------------------------------------\n"
    )
    report.write(
        "Rows not found: "
        f"{development_missing_from_feature}\n"
    )
    report.write(
        "Changed labels: "
        f"{development_label_changes}\n\n"
    )

    if not feature_to_split_transition.empty:
        report.write(
            "Label transitions:\n"
        )
        report.write(
            feature_to_split_transition.to_string()
            + "\n\n"
        )

    report.write(
        "DIRECT PREPROCESSED -> "
        "TRAIN + VALIDATION\n"
        "-----------------------------------------\n"
    )
    report.write(
        "Rows not found: "
        f"{development_missing_from_pre}\n"
    )
    report.write(
        "Changed labels: "
        f"{direct_label_changes}\n\n"
    )

    report.write(
        f"FINAL STATUS: {status}\n"
    )
    report.write(
        interpretation + "\n\n"
    )

    report.write(
        "FINAL TEST PROTECTION\n"
        "---------------------\n"
    )
    report.write(
        "Component3_Test.csv contents "
        "loaded: False\n"
    )


# ============================================================
# 12. FINISH
# ============================================================

print("\n--- FINAL TEST PROTECTION ---")
print(
    "Component3_Test.csv content read:",
    final_test_content_read,
)

print("\nReport saved to:")
print(REPORT_PATH)
