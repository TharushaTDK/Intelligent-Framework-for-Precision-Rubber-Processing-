from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# COMPONENT 3 - STAGE 21: CLASS IMBALANCE ANALYSIS
# ============================================================
#
# Target:
#   Fire_Risk_Label = Normal / Warning / Critical
#
# Purpose:
#   Quantify class imbalance BEFORE training real classifiers
#   and formally define the imbalance-handling strategy.
#
# Key methodology:
#   - TRAIN determines class weights.
#   - VALIDATION is descriptive only.
#   - TEST is not loaded.
#   - No classifier is trained.
#   - No SMOTE / synthetic oversampling is performed.
#
# Chosen strategy for later stages:
#
#   Logistic Regression:
#       class_weight="balanced"
#
#   Random Forest:
#       class_weight="balanced"
#
#   Extra Trees:
#       class_weight="balanced"
#
#   Gradient Boosting:
#       use TRAIN-derived sample_weight during fit
#
#   XGBoost:
#       use TRAIN-derived sample_weight during fit
#
# Why not SMOTE:
#   The observations are grouped temporal measurements from only
#   8 Train batches, while fire-risk ground truth comes from
#   hourly keeper/expert assessments carried across 10-minute rows.
#   Synthetic row generation could create unrealistic temporal
#   combinations and weaken research validity.
#
# IMPORTANT:
#   These weights are derived from TRAIN only and must never be
#   recalculated using Validation or Test.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

# File location:
# component_3_smokehouse/src/classification/21_imbalance_analysis.py
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

ARTIFACT_DIR = (
    ROOT
    / "artifacts"
    / "classification"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. INPUT FILES
# ============================================================

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
    / "21_class_imbalance_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "21_class_imbalance_checks.csv"
)

ROW_DISTRIBUTION_PATH = (
    REPORT_DIR
    / "21_class_imbalance_row_distribution.csv"
)

ASSESSMENT_DISTRIBUTION_PATH = (
    REPORT_DIR
    / "21_class_imbalance_assessment_distribution.csv"
)

RATIOS_PATH = (
    REPORT_DIR
    / "21_class_imbalance_ratios.csv"
)

WEIGHTS_PATH = (
    REPORT_DIR
    / "21_class_weights.csv"
)

STRATEGY_PATH = (
    REPORT_DIR
    / "21_class_imbalance_strategy.json"
)

TRAIN_SAMPLE_WEIGHTS_PATH = (
    ARTIFACT_DIR
    / "train_classification_sample_weights.csv"
)


# ============================================================
# 4. CONSTANTS
# ============================================================

EXPECTED_TRAIN_ROWS = 16246
EXPECTED_VALIDATION_ROWS = 4159

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
print("COMPONENT 3 - STAGE 21: CLASS IMBALANCE ANALYSIS")
print("=" * 78)


# ============================================================
# 6. REQUIRED FILE CHECKS
# ============================================================

required_files = [
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

print("\nTrain and Validation target/metadata files found.")
print("Final Test dataset existence verified, but contents were NOT loaded.")


# ============================================================
# 7. LOAD DEVELOPMENT TARGETS + METADATA ONLY
# ============================================================

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
# 8. TARGET COLUMN RESOLUTION
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

if len(
    y_train
) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError(
        f"Unexpected Train row count: {len(y_train)}"
    )

if len(
    y_validation
) != EXPECTED_VALIDATION_ROWS:
    raise RuntimeError(
        f"Unexpected Validation row count: {len(y_validation)}"
    )

if len(
    meta_train
) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError(
        "Train metadata row count does not match target."
    )

if len(
    meta_validation
) != EXPECTED_VALIDATION_ROWS:
    raise RuntimeError(
        "Validation metadata row count does not match target."
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
        "Unexpected Train classes.\n"
        f"Expected: {sorted(EXPECTED_CLASSES)}\n"
        f"Found: {sorted(train_classes)}"
    )

if validation_classes != EXPECTED_CLASSES:
    raise RuntimeError(
        "Unexpected Validation classes.\n"
        f"Expected: {sorted(EXPECTED_CLASSES)}\n"
        f"Found: {sorted(validation_classes)}"
    )


# ============================================================
# 10. ROW-LEVEL DISTRIBUTIONS
# ============================================================

row_distribution_rows = []

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

        row_distribution_rows.append({
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
    row_distribution_rows
)

row_distribution_df.to_csv(
    ROW_DISTRIBUTION_PATH,
    index=False,
)


# ============================================================
# 11. UNIQUE HOURLY ASSESSMENT DISTRIBUTIONS
# ============================================================

assessment_key = [
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
    "Fire_Risk_Assessment_Timestamp",
]

for column in assessment_key:
    if column not in meta_train.columns:
        raise RuntimeError(
            f"Missing metadata column: {column}"
        )

    if column not in meta_validation.columns:
        raise RuntimeError(
            f"Missing Validation metadata column: {column}"
        )


def unique_assessment_table(
    metadata,
    y_values,
    split_name,
):
    temp = metadata[
        assessment_key
    ].copy()

    temp[
        "Fire_Risk_Label"
    ] = y_values

    consistency = (
        temp
        .groupby(
            assessment_key,
            dropna=False,
        )[
            "Fire_Risk_Label"
        ]
        .nunique()
    )

    inconsistent = int(
        (
            consistency
            >
            1
        ).sum()
    )

    if inconsistent != 0:
        raise RuntimeError(
            f"{split_name} contains {inconsistent} "
            "hourly assessment groups with inconsistent labels."
        )

    unique_assessments = (
        temp
        .drop_duplicates(
            subset=assessment_key,
            keep="first",
        )
    )

    rows = []

    total = len(
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

        rows.append({
            "Split": split_name,
            "Class": class_name,
            "Unique_Assessments": count,
            "Percentage": (
                count
                /
                total
                *
                100
            ),
        })

    return (
        pd.DataFrame(
            rows
        ),
        unique_assessments,
    )


train_assessment_distribution, train_unique_assessments = (
    unique_assessment_table(
        meta_train,
        y_train,
        "Train",
    )
)

validation_assessment_distribution, validation_unique_assessments = (
    unique_assessment_table(
        meta_validation,
        y_validation,
        "Validation",
    )
)

assessment_distribution_df = pd.concat(
    [
        train_assessment_distribution,
        validation_assessment_distribution,
    ],
    ignore_index=True,
)

assessment_distribution_df.to_csv(
    ASSESSMENT_DISTRIBUTION_PATH,
    index=False,
)


# ============================================================
# 12. TRAIN IMBALANCE RATIOS
# ============================================================

train_counts = {
    class_name: int(
        (
            y_train
            ==
            class_name
        ).sum()
    )
    for class_name in CLASS_ORDER
}

train_assessment_counts = {
    class_name: int(
        (
            train_unique_assessments[
                "Fire_Risk_Label"
            ]
            ==
            class_name
        ).sum()
    )
    for class_name in CLASS_ORDER
}

normal_rows = train_counts[
    "Normal"
]

warning_rows = train_counts[
    "Warning"
]

critical_rows = train_counts[
    "Critical"
]

normal_assessments = train_assessment_counts[
    "Normal"
]

warning_assessments = train_assessment_counts[
    "Warning"
]

critical_assessments = train_assessment_counts[
    "Critical"
]

ratio_rows = [
    {
        "Level": "Row",
        "Comparison": "Normal_to_Warning",
        "Majority_Count": normal_rows,
        "Minority_Count": warning_rows,
        "Ratio": (
            normal_rows
            /
            warning_rows
        ),
    },
    {
        "Level": "Row",
        "Comparison": "Normal_to_Critical",
        "Majority_Count": normal_rows,
        "Minority_Count": critical_rows,
        "Ratio": (
            normal_rows
            /
            critical_rows
        ),
    },
    {
        "Level": "Row",
        "Comparison": "Warning_to_Critical",
        "Majority_Count": warning_rows,
        "Minority_Count": critical_rows,
        "Ratio": (
            warning_rows
            /
            critical_rows
        ),
    },
    {
        "Level": "Hourly_Assessment",
        "Comparison": "Normal_to_Warning",
        "Majority_Count": normal_assessments,
        "Minority_Count": warning_assessments,
        "Ratio": (
            normal_assessments
            /
            warning_assessments
        ),
    },
    {
        "Level": "Hourly_Assessment",
        "Comparison": "Normal_to_Critical",
        "Majority_Count": normal_assessments,
        "Minority_Count": critical_assessments,
        "Ratio": (
            normal_assessments
            /
            critical_assessments
        ),
    },
    {
        "Level": "Hourly_Assessment",
        "Comparison": "Warning_to_Critical",
        "Majority_Count": warning_assessments,
        "Minority_Count": critical_assessments,
        "Ratio": (
            warning_assessments
            /
            critical_assessments
        ),
    },
]

ratios_df = pd.DataFrame(
    ratio_rows
)

ratios_df.to_csv(
    RATIOS_PATH,
    index=False,
)


# ============================================================
# 13. TRAIN-ONLY BALANCED CLASS WEIGHTS
# ============================================================
#
# sklearn balanced formula:
#
#   weight_c = n_samples / (n_classes * n_samples_c)
#
# These weights are computed ONLY from Train.

classes_array = np.array(
    CLASS_ORDER,
    dtype=object,
)

balanced_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes_array,
    y=y_train,
)

class_weight_map = {
    class_name: float(
        weight
    )
    for class_name, weight in zip(
        CLASS_ORDER,
        balanced_weights,
    )
}

weights_rows = []

for class_name in CLASS_ORDER:
    weights_rows.append({
        "Class": class_name,
        "Train_Row_Count": train_counts[
            class_name
        ],
        "Train_Row_Percentage": (
            train_counts[
                class_name
            ]
            /
            len(
                y_train
            )
            *
            100
        ),
        "Train_Unique_Assessments": train_assessment_counts[
            class_name
        ],
        "Balanced_Class_Weight": class_weight_map[
            class_name
        ],
    })

weights_df = pd.DataFrame(
    weights_rows
)

weights_df.to_csv(
    WEIGHTS_PATH,
    index=False,
)


# ============================================================
# 14. TRAIN ROW SAMPLE-WEIGHT ARTIFACT
# ============================================================
#
# Used later for estimators such as Gradient Boosting and
# XGBoost that accept sample_weight during fit.
#
# The output contains ONLY row index + Train-derived weight.
# It does not modify the dataset.

train_sample_weights = np.array(
    [
        class_weight_map[
            label
        ]
        for label in y_train
    ],
    dtype=float,
)

sample_weight_df = pd.DataFrame({
    "Train_Row_Index": np.arange(
        len(
            y_train
        )
    ),
    "Fire_Risk_Label": y_train,
    "Sample_Weight": train_sample_weights,
})

sample_weight_df.to_csv(
    TRAIN_SAMPLE_WEIGHTS_PATH,
    index=False,
)


# ============================================================
# 15. CHECK WEIGHT BALANCING EFFECT
# ============================================================

weighted_class_mass = {}

for class_name in CLASS_ORDER:
    mask = (
        y_train
        ==
        class_name
    )

    weighted_class_mass[
        class_name
    ] = float(
        train_sample_weights[
            mask
        ].sum()
    )

weighted_mass_values = np.array(
    list(
        weighted_class_mass.values()
    ),
    dtype=float,
)

weighted_masses_equal = bool(
    np.allclose(
        weighted_mass_values,
        weighted_mass_values[
            0
        ],
        rtol=1e-10,
        atol=1e-10,
    )
)


# ============================================================
# 16. COMPARE ROW VS ASSESSMENT IMBALANCE
# ============================================================

row_normal_warning_ratio = (
    normal_rows
    /
    warning_rows
)

assessment_normal_warning_ratio = (
    normal_assessments
    /
    warning_assessments
)

row_normal_critical_ratio = (
    normal_rows
    /
    critical_rows
)

assessment_normal_critical_ratio = (
    normal_assessments
    /
    critical_assessments
)

row_assessment_pattern_similar = bool(
    abs(
        row_normal_warning_ratio
        -
        assessment_normal_warning_ratio
    )
    <
    1.0
    and
    abs(
        row_normal_critical_ratio
        -
        assessment_normal_critical_ratio
    )
    <
    5.0
)


# ============================================================
# 17. FORMAL STRATEGY
# ============================================================

strategy = {
    "stage": (
        "21_class_imbalance_analysis"
    ),
    "target": (
        "Fire_Risk_Label"
    ),
    "classes": (
        CLASS_ORDER
    ),
    "weight_source": (
        "TRAIN_ROW_DISTRIBUTION_ONLY"
    ),
    "balanced_class_weights": (
        class_weight_map
    ),
    "primary_classification_metrics": [
        "Macro_F1",
        "Balanced_Accuracy",
        "Warning_Recall",
        "Critical_Recall",
    ],
    "supporting_metrics": [
        "Macro_Precision",
        "Macro_Recall",
        "Weighted_F1",
        "Accuracy",
        "Confusion_Matrix",
    ],
    "imbalance_strategy": {
        "Logistic_Regression": (
            "class_weight='balanced'"
        ),
        "Random_Forest": (
            "class_weight='balanced'"
        ),
        "Extra_Trees": (
            "class_weight='balanced'"
        ),
        "Gradient_Boosting": (
            "fit with TRAIN-derived sample_weight"
        ),
        "XGBoost": (
            "fit with TRAIN-derived sample_weight"
        ),
    },
    "smote_used": False,
    "smote_recommended": False,
    "smote_reason": (
        "Grouped temporal dataset with only 8 Train batches; "
        "hourly expert labels are carried over 10-minute rows. "
        "Synthetic row generation could violate temporal/process "
        "structure and create unrealistic minority observations."
    ),
    "validation_used_to_compute_weights": False,
    "test_loaded": False,
    "row_and_assessment_imbalance_pattern_similar": (
        row_assessment_pattern_similar
    ),
    "next_stage": (
        "22_logistic_regression"
    ),
}

with open(
    STRATEGY_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        strategy,
        file,
        indent=2,
    )


# ============================================================
# 18. PASS / FAIL CHECKS
# ============================================================

all_weights_positive = bool(
    (
        balanced_weights
        >
        0
    ).all()
)

weight_order_expected = bool(
    class_weight_map[
        "Normal"
    ]
    <
    class_weight_map[
        "Warning"
    ]
    <
    class_weight_map[
        "Critical"
    ]
)

all_train_rows_weighted = bool(
    len(
        sample_weight_df
    )
    ==
    EXPECTED_TRAIN_ROWS
)

no_missing_weights = bool(
    sample_weight_df[
        "Sample_Weight"
    ].notna().all()
)

checks = pd.DataFrame(
    [
        {
            "Check": (
                "Expected Train and Validation row counts"
            ),
            "Status": "PASS",
            "Details": (
                f"Train={len(y_train):,}; "
                f"Validation={len(y_validation):,}"
            ),
        },
        {
            "Check": (
                "Expected three classification classes"
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
                "Hourly assessment labels internally consistent"
            ),
            "Status": "PASS",
            "Details": (
                "One label per Smokehouse/Batch/Layer/"
                "assessment timestamp."
            ),
        },
        {
            "Check": (
                "Class weights derived from Train only"
            ),
            "Status": "PASS",
            "Details": (
                "Validation distribution not used in weight formula."
            ),
        },
        {
            "Check": (
                "All balanced class weights positive"
            ),
            "Status": (
                "PASS"
                if all_weights_positive
                else "FAIL"
            ),
            "Details": (
                str(
                    class_weight_map
                )
            ),
        },
        {
            "Check": (
                "Minority classes receive larger weights"
            ),
            "Status": (
                "PASS"
                if weight_order_expected
                else "FAIL"
            ),
            "Details": (
                "Normal < Warning < Critical."
            ),
        },
        {
            "Check": (
                "Balanced weighting equalizes class contribution"
            ),
            "Status": (
                "PASS"
                if weighted_masses_equal
                else "FAIL"
            ),
            "Details": (
                str(
                    weighted_class_mass
                )
            ),
        },
        {
            "Check": (
                "Every Train row assigned one sample weight"
            ),
            "Status": (
                "PASS"
                if all_train_rows_weighted
                and
                no_missing_weights
                else "FAIL"
            ),
            "Details": (
                f"Weighted rows={len(sample_weight_df):,}"
            ),
        },
        {
            "Check": (
                "No SMOTE or resampling performed"
            ),
            "Status": "PASS",
            "Details": (
                "Weighting selected to preserve temporal structure."
            ),
        },
        {
            "Check": (
                "No classifier trained"
            ),
            "Status": "PASS",
            "Details": (
                "Stage 21 performs analysis/strategy only."
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
# 19. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- ROW-LEVEL DISTRIBUTION ---"
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
    "\n--- TRAIN IMBALANCE RATIOS ---"
)

print(
    ratios_df
    .round(
        {
            "Ratio": 4
        }
    )
    .to_string(
        index=False
    )
)


print(
    "\n--- TRAIN-ONLY BALANCED CLASS WEIGHTS ---"
)

print(
    weights_df
    .round(
        {
            "Train_Row_Percentage": 4,
            "Balanced_Class_Weight": 6,
        }
    )
    .to_string(
        index=False
    )
)


print(
    "\n--- WEIGHTED CLASS MASS CHECK ---"
)

for class_name in CLASS_ORDER:
    print(
        f"{class_name:<10}: "
        f"{weighted_class_mass[class_name]:.4f}"
    )

print(
    f"Weighted masses equal           : "
    f"{weighted_masses_equal}"
)


print(
    "\n--- FORMAL IMBALANCE STRATEGY ---"
)

print(
    "Logistic Regression             : class_weight='balanced'"
)

print(
    "Random Forest                   : class_weight='balanced'"
)

print(
    "Extra Trees                     : class_weight='balanced'"
)

print(
    "Gradient Boosting               : TRAIN-derived sample_weight"
)

print(
    "XGBoost                         : TRAIN-derived sample_weight"
)

print(
    "SMOTE                           : NOT used"
)

print(
    "Reason                          : preserve grouped temporal structure"
)


print(
    "\n--- IMPORTANT METRICS FOR FUTURE CLASSIFIERS ---"
)

print(
    "Primary                         : Macro-F1"
)

print(
    "Also critical                   : Balanced Accuracy"
)

print(
    "Safety-sensitive                : Warning Recall, Critical Recall"
)

print(
    "Accuracy                        : supporting metric only"
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
        "COMPONENT 3 - STAGE 21: CLASS IMBALANCE ANALYSIS\n"
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
        "Stage 21 quantifies the imbalance in Fire_Risk_Label and "
        "defines the imbalance-handling strategy before training "
        "real classifiers. No classifier is trained in this stage.\n\n"
    )

    report.write(
        "ROW-LEVEL DISTRIBUTION\n"
        "----------------------\n"
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
        "TRAIN IMBALANCE RATIOS\n"
        "----------------------\n"
    )

    report.write(
        ratios_df
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
        "TRAIN-ONLY BALANCED CLASS WEIGHTS\n"
        "---------------------------------\n"
    )

    report.write(
        weights_df
        .round(
            8
        )
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "WEIGHT FORMULA\n"
        "--------------\n"
    )

    report.write(
        "Balanced weight for class c:\n"
        "    n_samples / (n_classes * n_samples_c)\n\n"
        "Weights are computed from Train only. Validation and Test "
        "must not influence class-weight calculation.\n\n"
    )

    report.write(
        "WHY CLASS WEIGHTING IS SELECTED\n"
        "-------------------------------\n"
    )

    report.write(
        "The Train data are strongly imbalanced, especially for the "
        "Critical class. However, the observations are grouped and "
        "temporal: only 8 complete batches form Train, and fire-risk "
        "labels originate from hourly keeper/expert assessments that "
        "are carried across 10-minute sensor rows. Synthetic "
        "oversampling such as SMOTE could generate artificial "
        "feature combinations that do not correspond to valid "
        "smokehouse temporal states. Therefore class/sample weighting "
        "is selected as the primary imbalance strategy.\n\n"
    )

    report.write(
        "MODEL-SPECIFIC STRATEGY\n"
        "-----------------------\n"
    )

    report.write(
        "Logistic Regression: class_weight='balanced'\n"
        "Random Forest: class_weight='balanced'\n"
        "Extra Trees: class_weight='balanced'\n"
        "Gradient Boosting: TRAIN-derived sample_weight\n"
        "XGBoost: TRAIN-derived sample_weight\n"
        "SMOTE / synthetic oversampling: not used\n\n"
    )

    report.write(
        "EVALUATION PRIORITY\n"
        "-------------------\n"
    )

    report.write(
        "Primary comparison metric: Macro-F1.\n"
        "Also important: Balanced Accuracy.\n"
        "Safety-sensitive metrics: Warning Recall and Critical Recall.\n"
        "Accuracy is retained only as a supporting metric because the "
        "Normal class dominates the dataset.\n\n"
    )

    report.write(
        "WEIGHTED CLASS MASS CHECK\n"
        "-------------------------\n"
    )

    for class_name in CLASS_ORDER:
        report.write(
            f"{class_name}: "
            f"{weighted_class_mass[class_name]:.8f}\n"
        )

    report.write(
        f"Equal weighted contribution: "
        f"{weighted_masses_equal}\n\n"
    )

    report.write(
        "METHODOLOGY PROTECTION\n"
        "----------------------\n"
    )

    report.write(
        "- Class weights were calculated from Train only.\n"
        "- Validation was used only for descriptive comparison.\n"
        "- No Test rows were loaded.\n"
        "- No SMOTE or other resampling was performed.\n"
        "- No classifier was trained.\n"
        "- Final Test remains frozen until Stage 31.\n\n"
    )

    report.write(
        "NEXT STAGE\n"
        "----------\n"
    )

    report.write(
        "Stage 22 will train weighted multinomial Logistic "
        "Regression using the Stage-10 model-ready features and "
        "the Train-only imbalance strategy defined here.\n\n"
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
        f"STAGE 21 STATUS: "
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
    f"STAGE 21 CLASS IMBALANCE ANALYSIS: "
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
    f"\nTrain sample-weight artifact saved in:\n"
    f"{ARTIFACT_DIR}"
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 21 failed. Review "
        "21_class_imbalance_report.txt."
    )
