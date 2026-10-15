from pathlib import Path
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# COMPONENT 3 - STAGE 08: FEATURE ANALYSIS & SELECTION
# ============================================================
#
# Purpose:
#   Analyze the 36 Stage-07 candidate features and freeze a
#   compact, deployment-safe feature set before batch splitting.
#
# CRITICAL METHODOLOGY RULE:
#   We DO NOT choose/drop features because of full-dataset
#   target performance. Doing that before the split would leak
#   target information into feature selection.
#
# Selection is based on:
#   - leakage safety
#   - runtime availability
#   - deterministic/algebraic redundancy
#   - severe predictor-predictor correlation
#   - near-constant availability indicators
#   - temporal/domain findings from Stages 05-06
#
# Target association tables are produced for DESCRIPTION ONLY.
# They are not used by the deterministic selection policy.
#
# No model is trained.
# No scaling/encoding/imputation is fitted.
# No train/test split is performed.
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

REPORT_DIR = (
    ROOT
    / "outputs"
    / "reports"
)

FIGURE_DIR = (
    ROOT
    / "outputs"
    / "figures"
    / "08_feature_analysis"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "08_feature_analysis_selection_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "08_feature_analysis_checks.csv"
)

FEATURE_SUMMARY_PATH = (
    REPORT_DIR
    / "08_feature_summary.csv"
)

HIGH_CORR_PATH = (
    REPORT_DIR
    / "08_high_correlation_pairs.csv"
)

REG_ASSOC_PATH = (
    REPORT_DIR
    / "08_regression_target_association_descriptive.csv"
)

CLS_ASSOC_PATH = (
    REPORT_DIR
    / "08_classification_numeric_association_descriptive.csv"
)

DECISIONS_PATH = (
    REPORT_DIR
    / "08_feature_selection_decisions.csv"
)

REG_SELECTED_PATH = (
    REPORT_DIR
    / "08_regression_selected_features.csv"
)

CLS_SELECTED_PATH = (
    REPORT_DIR
    / "08_classification_selected_features.csv"
)


# ============================================================
# 3. STAGE-07 SCHEMA
# ============================================================

EXPECTED_ROWS = 24463

METADATA_COLUMNS = [
    "Timestamp",
    "Batch_ID",
    "Fire_Risk_Assessment_Timestamp",
]

CATEGORICAL_CANDIDATES = [
    "Smokehouse_ID",
    "Layer_ID",
    "Temperature_Trend",
    "Humidity_Trend",
]

BASE_NUMERIC_CANDIDATES = [
    "Temperature_C",
    "Humidity_%",
    "Elapsed_Drying_Hours",
    "Number_of_Sheets",
    "Layer_Area_m2",
    "Temperature_Change",
    "Humidity_Change",
]

ENGINEERED_NUMERIC_CANDIDATES = [
    "Time_Since_Previous_Reading_Min",
    "Temperature_Rate_per_10min",
    "Humidity_Rate_per_10min",

    "Temperature_Lag_10m",
    "Humidity_Lag_10m",
    "Temperature_Change_10m",
    "Humidity_Change_10m",

    "Temperature_Lag_30m",
    "Humidity_Lag_30m",
    "Temperature_Change_30m",
    "Humidity_Change_30m",

    "Temperature_Lag_60m",
    "Humidity_Lag_60m",
    "Temperature_Change_60m",
    "Humidity_Change_60m",

    "Temperature_RollMean_30m",
    "Humidity_RollMean_30m",
    "Temperature_RollStd_30m",
    "Humidity_RollStd_30m",
    "Observation_Count_30m",

    "Temperature_RollMean_60m",
    "Humidity_RollMean_60m",
    "Temperature_RollStd_60m",
    "Humidity_RollStd_60m",
    "Observation_Count_60m",
]

TARGET_COLUMNS = [
    "Remaining_Drying_Time",
    "Fire_Risk_Label",
]

ALL_CANDIDATES = (
    CATEGORICAL_CANDIDATES
    + BASE_NUMERIC_CANDIDATES
    + ENGINEERED_NUMERIC_CANDIDATES
)

EXPECTED_COLUMNS = (
    METADATA_COLUMNS
    + ALL_CANDIDATES
    + TARGET_COLUMNS
)

NUMERIC_CANDIDATES = (
    BASE_NUMERIC_CANDIDATES
    + ENGINEERED_NUMERIC_CANDIDATES
)

SEQUENCE_KEY = [
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
]


# ============================================================
# 4. LOAD FEATURE-ENGINEERED DATA
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 08: FEATURE ANALYSIS & SELECTION")
print("=" * 78)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"\nStage-07 feature dataset not found:\n{INPUT_PATH}\n\n"
        "Run Stage 07 first."
    )

with open(INPUT_PATH, "rb") as file:
    input_sha256 = hashlib.sha256(
        file.read()
    ).hexdigest()

df = pd.read_csv(
    INPUT_PATH
)

print("\nFeature-engineered dataset loaded successfully.")

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Unexpected row count: {len(df):,}"
    )

if list(df.columns) != EXPECTED_COLUMNS:
    raise RuntimeError(
        "Unexpected Stage-07 schema."
    )


# ============================================================
# 5. TYPE CONVERSION
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

for column in (
    NUMERIC_CANDIDATES
    + [
        "Remaining_Drying_Time",
    ]
):
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# 6. FEATURE SUMMARY
# ============================================================

summary_rows = []

for column in ALL_CANDIDATES:

    series = df[
        column
    ]

    missing_count = int(
        series.isna().sum()
    )

    unique_count = int(
        series.nunique(
            dropna=True
        )
    )

    if column in NUMERIC_CANDIDATES:
        numeric_series = (
            series
            .dropna()
            .astype(float)
        )

        standard_deviation = (
            float(
                numeric_series.std()
            )
            if len(
                numeric_series
            )
            > 1
            else np.nan
        )

        value_counts = (
            numeric_series
            .value_counts(
                normalize=True
            )
        )

        dominant_ratio = (
            float(
                value_counts.iloc[0]
            )
            if len(
                value_counts
            )
            > 0
            else np.nan
        )

        feature_type = (
            "numeric"
        )

    else:
        standard_deviation = np.nan

        value_counts = (
            series
            .dropna()
            .value_counts(
                normalize=True
            )
        )

        dominant_ratio = (
            float(
                value_counts.iloc[0]
            )
            if len(
                value_counts
            )
            > 0
            else np.nan
        )

        feature_type = (
            "categorical"
        )

    summary_rows.append({
        "Feature": column,
        "Feature_Type": feature_type,
        "Missing_Count": missing_count,
        "Missing_Percent": (
            missing_count
            / len(df)
            * 100
        ),
        "Unique_Count": unique_count,
        "Standard_Deviation": (
            standard_deviation
        ),
        "Dominant_Value_Ratio": (
            dominant_ratio
        ),
    })

feature_summary = pd.DataFrame(
    summary_rows
)

feature_summary[
    "Near_Constant"
] = (
    feature_summary[
        "Dominant_Value_Ratio"
    ]
    >= 0.99
)

feature_summary.to_csv(
    FEATURE_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 7. NUMERIC REDUNDANCY / CORRELATION
# ============================================================

numeric_corr = (
    df[
        NUMERIC_CANDIDATES
    ]
    .corr(
        method="pearson"
    )
)

pair_rows = []

for i, feature_a in enumerate(
    NUMERIC_CANDIDATES
):
    for feature_b in (
        NUMERIC_CANDIDATES[
            i + 1:
        ]
    ):
        correlation = float(
            numeric_corr.loc[
                feature_a,
                feature_b,
            ]
        )

        if abs(
            correlation
        ) >= 0.95:
            pair_rows.append({
                "Feature_A": feature_a,
                "Feature_B": feature_b,
                "Pearson_Correlation": (
                    correlation
                ),
                "Absolute_Correlation": abs(
                    correlation
                ),
            })

high_corr_pairs = pd.DataFrame(
    pair_rows
)

if len(
    high_corr_pairs
) > 0:
    high_corr_pairs = (
        high_corr_pairs
        .sort_values(
            "Absolute_Correlation",
            ascending=False,
        )
    )

high_corr_pairs.to_csv(
    HIGH_CORR_PATH,
    index=False,
)


# ============================================================
# 8. DESCRIPTIVE REGRESSION TARGET ASSOCIATION
# ============================================================
#
# REPORT ONLY.
# Not used for deterministic feature selection.

regression_rows = []

for feature in NUMERIC_CANDIDATES:

    valid = (
        df[
            [
                feature,
                "Remaining_Drying_Time",
            ]
        ]
        .dropna()
    )

    pearson = float(
        valid[
            feature
        ]
        .corr(
            valid[
                "Remaining_Drying_Time"
            ],
            method="pearson",
        )
    )

    # Spearman correlation without requiring SciPy:
    # rank both variables, then calculate ordinary Pearson
    # correlation between the ranks.
    feature_rank = (
        valid[
            feature
        ]
        .rank(
            method="average"
        )
    )

    target_rank = (
        valid[
            "Remaining_Drying_Time"
        ]
        .rank(
            method="average"
        )
    )

    spearman = float(
        feature_rank.corr(
            target_rank,
            method="pearson",
        )
    )

    regression_rows.append({
        "Feature": feature,
        "Pearson_With_Remaining_Drying_Time": (
            pearson
        ),
        "Spearman_With_Remaining_Drying_Time": (
            spearman
        ),
        "Observation_Count": len(
            valid
        ),
        "Selection_Use": (
            "DESCRIPTIVE_ONLY_NOT_USED_FOR_PRE_SPLIT_SELECTION"
        ),
    })

regression_association = (
    pd.DataFrame(
        regression_rows
    )
    .sort_values(
        "Pearson_With_Remaining_Drying_Time",
        key=lambda s: s.abs(),
        ascending=False,
    )
)

regression_association.to_csv(
    REG_ASSOC_PATH,
    index=False,
)


# ============================================================
# 9. DESCRIPTIVE CLASSIFICATION ASSOCIATION
# ============================================================
#
# Use UNIQUE HOURLY assessment events rather than all
# carried-forward 10-minute labels.
#
# Eta-squared = between-class variance / total variance.
# REPORT ONLY. Not used for pre-split selection.

assessment_df = (
    df[
        df[
            "Timestamp"
        ]
        ==
        df[
            "Fire_Risk_Assessment_Timestamp"
        ]
    ]
    .copy()
)

classification_rows = []

for feature in NUMERIC_CANDIDATES:

    valid = (
        assessment_df[
            [
                feature,
                "Fire_Risk_Label",
            ]
        ]
        .dropna()
    )

    if len(
        valid
    ) == 0:
        eta_squared = np.nan

    else:
        grand_mean = float(
            valid[
                feature
            ].mean()
        )

        total_ss = float(
            (
                (
                    valid[
                        feature
                    ]
                    -
                    grand_mean
                )
                ** 2
            ).sum()
        )

        between_ss = 0.0

        for _, group in valid.groupby(
            "Fire_Risk_Label"
        ):
            between_ss += (
                len(
                    group
                )
                *
                (
                    float(
                        group[
                            feature
                        ].mean()
                    )
                    -
                    grand_mean
                )
                ** 2
            )

        eta_squared = (
            between_ss
            / total_ss
            if total_ss
            > 0
            else 0.0
        )

    classification_rows.append({
        "Feature": feature,
        "Eta_Squared_Across_Fire_Risk_Classes": (
            eta_squared
        ),
        "Unique_Hourly_Assessment_Count": len(
            valid
        ),
        "Selection_Use": (
            "DESCRIPTIVE_ONLY_NOT_USED_FOR_PRE_SPLIT_SELECTION"
        ),
    })

classification_association = (
    pd.DataFrame(
        classification_rows
    )
    .sort_values(
        "Eta_Squared_Across_Fire_Risk_Classes",
        ascending=False,
    )
)

classification_association.to_csv(
    CLS_ASSOC_PATH,
    index=False,
)


# ============================================================
# 10. DETERMINISTIC SELECTION POLICY
# ============================================================
#
# Selected core set:
#
# Keep:
#   - physical/context variables available at runtime
#   - current sensor values
#   - drying stage/load/context
#   - gap-aware short-term rates
#   - 30/60-minute exact changes
#   - 30/60-minute variability
#
# Drop:
#   - source change/trend fields because timestamp-aware
#     engineered behavior supersedes them
#   - exact lag values because:
#       Lag = Current - Change
#     so keeping Current + Change makes Lag algebraically
#     redundant
#   - exact 10-minute changes because gap-aware rates provide
#     the short-term signal more robustly
#   - 30/60-minute rolling means because they are extremely
#     correlated with the current sensor values
#   - observation-count and previous-interval availability
#     indicators because they are near-constant artifacts of
#     sequence warm-up / rare packet loss, not core physics
#
# This policy does NOT inspect target performance.

SELECTED_SHARED_FEATURES = [
    # Context
    "Smokehouse_ID",
    "Layer_ID",

    # Current state / drying context
    "Temperature_C",
    "Humidity_%",
    "Elapsed_Drying_Hours",
    "Number_of_Sheets",
    "Layer_Area_m2",

    # Short-term gap-aware rates
    "Temperature_Rate_per_10min",
    "Humidity_Rate_per_10min",

    # Exact recent changes
    "Temperature_Change_30m",
    "Humidity_Change_30m",
    "Temperature_Change_60m",
    "Humidity_Change_60m",

    # Recent variability at two time scales
    "Temperature_RollStd_30m",
    "Humidity_RollStd_30m",
    "Temperature_RollStd_60m",
    "Humidity_RollStd_60m",
]

# At this research stage the same physically justified core
# sensor/context set is retained for both tasks. The two tasks
# will learn different weights/interactions later.
REGRESSION_SELECTED = (
    SELECTED_SHARED_FEATURES.copy()
)

CLASSIFICATION_SELECTED = (
    SELECTED_SHARED_FEATURES.copy()
)


# ============================================================
# 11. FEATURE DECISION TABLE
# ============================================================

decision_reason = {}

for feature in ALL_CANDIDATES:
    decision_reason[
        feature
    ] = {
        "Decision": "DROP",
        "Reason": (
            "Not retained by deterministic Stage-08 policy."
        ),
    }

for feature in SELECTED_SHARED_FEATURES:
    decision_reason[
        feature
    ] = {
        "Decision": "KEEP",
        "Reason": (
            "Deployment-safe physical/context or temporal "
            "feature retained by Stage-08 policy."
        ),
    }


# More specific DROP explanations

for feature in [
    "Temperature_Change",
    "Humidity_Change",
]:
    decision_reason[
        feature
    ] = {
        "Decision": "DROP",
        "Reason": (
            "Source adjacent-change field is superseded by "
            "timestamp-aware rates and exact 30/60-minute changes."
        ),
    }

for feature in [
    "Temperature_Trend",
    "Humidity_Trend",
]:
    decision_reason[
        feature
    ] = {
        "Decision": "DROP",
        "Reason": (
            "Coarse discretized trend is redundant with richer "
            "numeric temporal rate/change features."
        ),
    }

for feature in [
    "Temperature_Lag_10m",
    "Humidity_Lag_10m",
    "Temperature_Lag_30m",
    "Humidity_Lag_30m",
    "Temperature_Lag_60m",
    "Humidity_Lag_60m",
]:
    decision_reason[
        feature
    ] = {
        "Decision": "DROP",
        "Reason": (
            "Exact lag value is algebraically redundant when "
            "current value and corresponding change are available."
        ),
    }

for feature in [
    "Temperature_Change_10m",
    "Humidity_Change_10m",
]:
    decision_reason[
        feature
    ] = {
        "Decision": "DROP",
        "Reason": (
            "Short-term signal retained through gap-aware "
            "per-10-minute rate; exact 10-minute change is "
            "unavailable across packet-loss gaps."
        ),
    }

for feature in [
    "Temperature_RollMean_30m",
    "Humidity_RollMean_30m",
    "Temperature_RollMean_60m",
    "Humidity_RollMean_60m",
]:
    decision_reason[
        feature
    ] = {
        "Decision": "DROP",
        "Reason": (
            "Rolling mean is highly redundant with the current "
            "sensor value in this smooth time series; recent "
            "history is retained through change and variability features."
        ),
    }

for feature in [
    "Time_Since_Previous_Reading_Min",
    "Observation_Count_30m",
    "Observation_Count_60m",
]:
    decision_reason[
        feature
    ] = {
        "Decision": "DROP",
        "Reason": (
            "Near-constant availability/warm-up indicator; "
            "not retained as a core physical predictor."
        ),
    }


decision_rows = []

for feature in ALL_CANDIDATES:

    decision_rows.append({
        "Feature": feature,
        "Stage08_Decision": (
            decision_reason[
                feature
            ][
                "Decision"
            ]
        ),
        "Regression_Selected": (
            feature
            in REGRESSION_SELECTED
        ),
        "Classification_Selected": (
            feature
            in CLASSIFICATION_SELECTED
        ),
        "Reason": (
            decision_reason[
                feature
            ][
                "Reason"
            ]
        ),
        "Selection_Basis": (
            "STRUCTURAL_DOMAIN_RUNTIME_ONLY_NOT_TARGET_PERFORMANCE"
        ),
    })

decisions = pd.DataFrame(
    decision_rows
)

decisions.to_csv(
    DECISIONS_PATH,
    index=False,
)


# ============================================================
# 12. SAVE TASK-SPECIFIC FEATURE LISTS
# ============================================================

regression_selected_df = pd.DataFrame({
    "Order": range(
        1,
        len(
            REGRESSION_SELECTED
        )
        + 1,
    ),
    "Feature": (
        REGRESSION_SELECTED
    ),
})

classification_selected_df = pd.DataFrame({
    "Order": range(
        1,
        len(
            CLASSIFICATION_SELECTED
        )
        + 1,
    ),
    "Feature": (
        CLASSIFICATION_SELECTED
    ),
})

regression_selected_df.to_csv(
    REG_SELECTED_PATH,
    index=False,
)

classification_selected_df.to_csv(
    CLS_SELECTED_PATH,
    index=False,
)


# ============================================================
# 13. SELECTED-FEATURE MISSINGNESS
# ============================================================

reg_selected_missing = {
    feature: int(
        df[
            feature
        ]
        .isna()
        .sum()
    )
    for feature in (
        REGRESSION_SELECTED
    )
}

cls_selected_missing = {
    feature: int(
        df[
            feature
        ]
        .isna()
        .sum()
    )
    for feature in (
        CLASSIFICATION_SELECTED
    )
}

reg_selected_missing_total = sum(
    reg_selected_missing.values()
)

cls_selected_missing_total = sum(
    cls_selected_missing.values()
)

# Missingness is expected only in history-dependent features.
# Stage 10 will fit imputation on training data only.


# ============================================================
# 14. LEAKAGE VALIDATION
# ============================================================

forbidden_predictors = {
    "Timestamp",
    "Batch_ID",
    "Fire_Risk_Assessment_Timestamp",
    "Actual_Total_Drying_Time",
    "Remaining_Drying_Time",
    "Fire_Risk_Label",
}

reg_forbidden = sorted(
    set(
        REGRESSION_SELECTED
    )
    &
    forbidden_predictors
)

cls_forbidden = sorted(
    set(
        CLASSIFICATION_SELECTED
    )
    &
    forbidden_predictors
)

reg_missing_columns = sorted(
    set(
        REGRESSION_SELECTED
    )
    -
    set(
        df.columns
    )
)

cls_missing_columns = sorted(
    set(
        CLASSIFICATION_SELECTED
    )
    -
    set(
        df.columns
    )
)


# ============================================================
# 15. HIGH-CORRELATION COUNT WITHIN SELECTED NUMERICS
# ============================================================

selected_numeric = [
    feature
    for feature in (
        SELECTED_SHARED_FEATURES
    )
    if feature
    in NUMERIC_CANDIDATES
]

selected_corr = (
    df[
        selected_numeric
    ]
    .corr()
)

selected_high_corr_rows = []

for i, a in enumerate(
    selected_numeric
):
    for b in (
        selected_numeric[
            i + 1:
        ]
    ):
        corr_value = float(
            selected_corr.loc[
                a,
                b,
            ]
        )

        if abs(
            corr_value
        ) >= 0.95:
            selected_high_corr_rows.append(
                (
                    a,
                    b,
                    corr_value,
                )
            )

selected_high_corr_count = len(
    selected_high_corr_rows
)


# ============================================================
# 16. VISUAL: ABSOLUTE CORRELATION OF SELECTED NUMERICS
# ============================================================

plt.figure(
    figsize=(12, 10)
)

plt.imshow(
    selected_corr.abs().to_numpy(),
    aspect="auto",
)

plt.xticks(
    range(
        len(
            selected_numeric
        )
    ),
    selected_numeric,
    rotation=90,
    fontsize=7,
)

plt.yticks(
    range(
        len(
            selected_numeric
        )
    ),
    selected_numeric,
    fontsize=7,
)

plt.title(
    "Absolute Correlation Among Stage-08 Selected Numeric Features"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "08_selected_numeric_correlation.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 17. PASS / FAIL CHECKS
# ============================================================

checks = pd.DataFrame(
    [
        {
            "Check": "Expected row count",
            "Status": (
                "PASS"
                if len(
                    df
                )
                == EXPECTED_ROWS
                else "FAIL"
            ),
            "Details": (
                f"Rows="
                f"{len(df):,}"
            ),
        },
        {
            "Check": "All 36 Stage-07 candidates present",
            "Status": (
                "PASS"
                if len(
                    ALL_CANDIDATES
                )
                == 36
                and
                set(
                    ALL_CANDIDATES
                )
                .issubset(
                    set(
                        df.columns
                    )
                )
                else "FAIL"
            ),
            "Details": (
                f"Candidates="
                f"{len(ALL_CANDIDATES)}"
            ),
        },
        {
            "Check": "Regression selected features exist",
            "Status": (
                "PASS"
                if len(
                    reg_missing_columns
                )
                == 0
                else "FAIL"
            ),
            "Details": (
                f"Missing="
                f"{reg_missing_columns}"
            ),
        },
        {
            "Check": "Classification selected features exist",
            "Status": (
                "PASS"
                if len(
                    cls_missing_columns
                )
                == 0
                else "FAIL"
            ),
            "Details": (
                f"Missing="
                f"{cls_missing_columns}"
            ),
        },
        {
            "Check": "Regression selected set has no forbidden leakage fields",
            "Status": (
                "PASS"
                if len(
                    reg_forbidden
                )
                == 0
                else "FAIL"
            ),
            "Details": (
                f"Forbidden="
                f"{reg_forbidden}"
            ),
        },
        {
            "Check": "Classification selected set has no forbidden leakage fields",
            "Status": (
                "PASS"
                if len(
                    cls_forbidden
                )
                == 0
                else "FAIL"
            ),
            "Details": (
                f"Forbidden="
                f"{cls_forbidden}"
            ),
        },
        {
            "Check": "Unique hourly fire-risk assessments preserved",
            "Status": (
                "PASS"
                if len(
                    assessment_df
                )
                == 4098
                else "FAIL"
            ),
            "Details": (
                f"Assessments="
                f"{len(assessment_df):,}"
            ),
        },
        {
            "Check": "Selection policy did not use target-ranking output",
            "Status": "PASS",
            "Details": (
                "Feature decisions are hard-coded from structural/"
                "runtime/redundancy rules; target tables are descriptive only."
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
        == "FAIL"
    ).any()
    else "FAIL"
)


# ============================================================
# 18. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- INPUT ---"
)

print(
    f"Rows / columns                    : "
    f"{len(df):,} / {len(df.columns)}"
)

print(
    f"Stage-07 SHA-256                  : "
    f"{input_sha256}"
)

print(
    f"Candidate features                : "
    f"{len(ALL_CANDIDATES)}"
)


print(
    "\n--- REDUNDANCY ANALYSIS ---"
)

print(
    f"Predictor pairs with |r| >= 0.95 : "
    f"{len(high_corr_pairs)}"
)

near_constant = (
    feature_summary[
        feature_summary[
            "Near_Constant"
        ]
    ][
        [
            "Feature",
            "Dominant_Value_Ratio",
        ]
    ]
)

print(
    "\nNear-constant candidates:"
)

if len(
    near_constant
) > 0:
    print(
        near_constant
        .round(4)
        .to_string(
            index=False
        )
    )
else:
    print(
        "None"
    )


print(
    "\n--- SELECTED FEATURE SETS ---"
)

print(
    f"Regression selected features     : "
    f"{len(REGRESSION_SELECTED)}"
)

print(
    f"Classification selected features : "
    f"{len(CLASSIFICATION_SELECTED)}"
)

print(
    f"Selected numeric |r|>=0.95 pairs : "
    f"{selected_high_corr_count}"
)


print(
    "\nRegression / Classification core:"
)

for feature in (
    SELECTED_SHARED_FEATURES
):
    print(
        f"  - {feature}"
    )


print(
    "\n--- EXPECTED SELECTED-FEATURE MISSINGNESS ---"
)

print(
    f"Regression selected missing cells : "
    f"{reg_selected_missing_total}"
)

print(
    f"Classification selected missing cells: "
    f"{cls_selected_missing_total}"
)

print(
    "These are history-related structural missing values; "
    "Stage 10 handles them after the batch split."
)


print(
    "\n--- LEAKAGE CHECK ---"
)

print(
    f"Regression forbidden fields      : "
    f"{reg_forbidden}"
)

print(
    f"Classification forbidden fields  : "
    f"{cls_forbidden}"
)

print(
    f"Hourly assessment events used for descriptive "
    f"class analysis: {len(assessment_df):,}"
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
        "COMPONENT 3 - STAGE 08: FEATURE ANALYSIS & SELECTION\n"
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
        "METHODOLOGY SAFEGUARD\n"
        "---------------------\n"
    )

    report.write(
        "Target-association statistics were calculated only "
        "for descriptive research reporting. They were NOT "
        "used to choose or drop features before the "
        "train/validation/test split. Feature selection was "
        "deterministic and based on leakage safety, runtime "
        "availability, redundancy, near-constant behavior, "
        "and findings from the earlier temporal/domain "
        "analysis.\n\n"
    )

    report.write(
        "INPUT\n"
        "-----\n"
    )

    report.write(
        f"Rows: {len(df):,}\n"
    )

    report.write(
        f"Columns: {len(df.columns)}\n"
    )

    report.write(
        f"Stage-07 SHA-256: "
        f"{input_sha256}\n"
    )

    report.write(
        f"Candidate features: "
        f"{len(ALL_CANDIDATES)}\n\n"
    )

    report.write(
        "REDUNDANCY ANALYSIS\n"
        "-------------------\n"
    )

    report.write(
        f"Numeric predictor pairs with |Pearson r| >= 0.95: "
        f"{len(high_corr_pairs)}\n\n"
    )

    if len(
        high_corr_pairs
    ) > 0:
        report.write(
            high_corr_pairs
            .round(6)
            .to_string(
                index=False
            )
        )

    report.write(
        "\n\n"
    )

    report.write(
        "FEATURE SELECTION POLICY\n"
        "------------------------\n"
    )

    report.write(
        "- Keep current temperature/humidity and known drying context.\n"
    )

    report.write(
        "- Keep gap-aware short-term rate features.\n"
    )

    report.write(
        "- Keep exact 30/60-minute changes because Stage 06 "
        "showed recent behavior is useful, especially for fire risk.\n"
    )

    report.write(
        "- Keep 30/60-minute variability because the two "
        "variability scales are not deterministically identical.\n"
    )

    report.write(
        "- Drop exact lag values because current + change "
        "already reconstruct the lag value algebraically.\n"
    )

    report.write(
        "- Drop raw source change/trend columns because the "
        "timestamp-aware engineered features supersede them.\n"
    )

    report.write(
        "- Drop near-constant availability indicators from the "
        "core predictor set.\n\n"
    )

    report.write(
        "SELECTED FEATURES\n"
        "-----------------\n"
    )

    report.write(
        f"Regression selected count: "
        f"{len(REGRESSION_SELECTED)}\n"
    )

    report.write(
        f"Classification selected count: "
        f"{len(CLASSIFICATION_SELECTED)}\n\n"
    )

    for feature in (
        SELECTED_SHARED_FEATURES
    ):
        report.write(
            f"- {feature}\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "TASK-SPECIFIC NOTE\n"
        "------------------\n"
    )

    report.write(
        "At Stage 08 the same physically justified core "
        "sensor/context feature set is retained for regression "
        "and classification. This is intentional: both outputs "
        "describe the same smokehouse process, while the later "
        "models learn different nonlinear relationships and "
        "feature importances. No target-driven pre-split "
        "selection is performed.\n\n"
    )

    report.write(
        "EXPECTED HISTORY-RELATED MISSINGNESS\n"
        "------------------------------------\n"
    )

    report.write(
        f"Regression selected missing cells: "
        f"{reg_selected_missing_total}\n"
    )

    report.write(
        f"Classification selected missing cells: "
        f"{cls_selected_missing_total}\n"
    )

    report.write(
        "These missing values occur in temporal features near "
        "sequence starts or unavailable exact histories. They "
        "remain untouched here. Stage 10 will fit missing-value "
        "handling on training data only.\n\n"
    )

    report.write(
        "DESCRIPTIVE TARGET ASSOCIATION\n"
        "------------------------------\n"
    )

    report.write(
        "Regression numeric target associations are saved to:\n"
        f"{REG_ASSOC_PATH.name}\n"
    )

    report.write(
        "Classification numeric associations use the 4,098 "
        "unique hourly keeper/expert assessment events and are "
        "saved to:\n"
        f"{CLS_ASSOC_PATH.name}\n"
    )

    report.write(
        "These association tables are for interpretation only "
        "and did not determine the feature list.\n\n"
    )

    report.write(
        "LEAKAGE PROTECTION\n"
        "------------------\n"
    )

    report.write(
        "- Actual_Total_Drying_Time is absent from the "
        "Stage-07 engineered dataset.\n"
    )

    report.write(
        "- Batch_ID is not selected as a predictor.\n"
    )

    report.write(
        "- Timestamp is not selected as a raw predictor.\n"
    )

    report.write(
        "- Fire_Risk_Assessment_Timestamp is not selected as "
        "a predictor.\n"
    )

    report.write(
        "- Neither prediction target is selected as a feature.\n"
    )

    report.write(
        "- No previous/future fire-risk label is used.\n\n"
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
        f"STAGE 08 STATUS: "
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
    f"STAGE 08 FEATURE ANALYSIS & SELECTION: "
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
    f"\nFigure saved in:\n"
    f"{FIGURE_DIR}"
)

print(
    "\nNo dataset was modified."
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 08 failed. Review the Stage-08 report."
    )
