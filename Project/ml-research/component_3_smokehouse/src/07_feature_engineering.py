from pathlib import Path
import hashlib
import pandas as pd
import numpy as np


# ============================================================
# COMPONENT 3 - STAGE 07: FEATURE ENGINEERING
# ============================================================
#
# Purpose:
#   Build deployment-safe candidate features for:
#
#   1) Regression
#      Target: Remaining_Drying_Time
#
#   2) Classification
#      Target: Fire_Risk_Label
#
# Core rule:
#   Every temporal feature uses ONLY the current observation
#   and/or past observations from the SAME
#   Smokehouse x Batch x Layer sequence.
#
# NO FUTURE DATA is used.
#
# Important:
#   - NO model is trained.
#   - NO feature is selected/dropped based on performance.
#   - NO scaling is fitted.
#   - NO encoding is fitted.
#   - NO SMOTE/resampling is performed.
#   - NO train/test split is performed.
#
# Leakage protection:
#   - Actual_Total_Drying_Time is deliberately EXCLUDED from
#     the engineered dataset.
#   - Batch_ID is retained ONLY for grouping/splitting.
#   - Fire_Risk_Assessment_Timestamp is retained ONLY as
#     target/assessment metadata.
#
# Stage 08 will analyze/select candidate features.
# Stage 09 will perform batch-aware splitting.
# Stage 10 will fit learned preprocessing on TRAINING DATA ONLY.
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "Component3_Dataset_Preprocessed.csv"
)

OUTPUT_PATH = (
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

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "07_feature_engineering_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "07_feature_engineering_checks.csv"
)

MANIFEST_PATH = (
    REPORT_DIR
    / "07_feature_manifest.csv"
)

MISSING_PATH = (
    REPORT_DIR
    / "07_engineered_feature_missingness.csv"
)

LAG_COVERAGE_PATH = (
    REPORT_DIR
    / "07_exact_lag_coverage.csv"
)


# ============================================================
# 3. EXPECTED INPUT STRUCTURE
# ============================================================

EXPECTED_ROWS = 24463

EXPECTED_COLUMNS = [
    "Smokehouse_ID",
    "Timestamp",
    "Batch_ID",
    "Layer_ID",
    "Temperature_C",
    "Humidity_%",
    "Elapsed_Drying_Hours",
    "Number_of_Sheets",
    "Layer_Area_m2",
    "Temperature_Change",
    "Humidity_Change",
    "Temperature_Trend",
    "Humidity_Trend",
    "Actual_Total_Drying_Time",
    "Remaining_Drying_Time",
    "Fire_Risk_Label",
    "Fire_Risk_Assessment_Timestamp",
]

SEQUENCE_KEY = [
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
]


# ============================================================
# 4. LOAD PREPROCESSED DATASET
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 07: FEATURE ENGINEERING")
print("=" * 78)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"\nPreprocessed dataset not found:\n{INPUT_PATH}\n\n"
        "Run Stage 04 before Stage 07."
    )

with open(INPUT_PATH, "rb") as file:
    input_sha256 = hashlib.sha256(
        file.read()
    ).hexdigest()

df = pd.read_csv(INPUT_PATH)

print("\nPreprocessed dataset loaded successfully.")

if len(df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Unexpected row count: {len(df):,}"
    )

if list(df.columns) != EXPECTED_COLUMNS:
    raise RuntimeError(
        "Unexpected input schema."
    )


# ============================================================
# 5. TYPE CONVERSION + ORDERING
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

for column in [
    "Temperature_C",
    "Humidity_%",
    "Elapsed_Drying_Hours",
    "Number_of_Sheets",
    "Layer_Area_m2",
    "Temperature_Change",
    "Humidity_Change",
    "Actual_Total_Drying_Time",
    "Remaining_Drying_Time",
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df = df.sort_values(
    SEQUENCE_KEY
    + ["Timestamp"]
).reset_index(drop=True)

input_missing_cells = int(
    df[
        EXPECTED_COLUMNS
    ]
    .isna()
    .sum()
    .sum()
)

if input_missing_cells != 0:
    raise RuntimeError(
        "Stage 07 expects the Stage 04 preprocessed dataset "
        "with no missing source values."
    )


# ============================================================
# 6. PRESERVE IMPORTANT ORIGINAL VALUES FOR VALIDATION
# ============================================================

preserved_reference = df[
    [
        "Smokehouse_ID",
        "Timestamp",
        "Batch_ID",
        "Layer_ID",
        "Remaining_Drying_Time",
        "Fire_Risk_Label",
        "Fire_Risk_Assessment_Timestamp",
    ]
].copy()


# ============================================================
# 7. PREVIOUS-READING / ACTUAL-INTERVAL FEATURES
# ============================================================
#
# These use the immediately preceding AVAILABLE observation.
# If a 10-minute packet was lost, the interval correctly
# becomes 20 minutes.

df["Previous_Timestamp"] = (
    df.groupby(
        SEQUENCE_KEY
    )[
        "Timestamp"
    ]
    .shift(1)
)

df["Time_Since_Previous_Reading_Min"] = (
    (
        df[
            "Timestamp"
        ]
        -
        df[
            "Previous_Timestamp"
        ]
    )
    .dt
    .total_seconds()
    / 60
)

df["Previous_Temperature_C"] = (
    df.groupby(
        SEQUENCE_KEY
    )[
        "Temperature_C"
    ]
    .shift(1)
)

df["Previous_Humidity_pct"] = (
    df.groupby(
        SEQUENCE_KEY
    )[
        "Humidity_%"
    ]
    .shift(1)
)

# Normalize the observed change to a 10-minute equivalent.
# This prevents a 20-minute packet-loss gap from being treated
# as if it were a normal 10-minute interval.

df[
    "Temperature_Rate_per_10min"
] = (
    (
        df[
            "Temperature_C"
        ]
        -
        df[
            "Previous_Temperature_C"
        ]
    )
    /
    df[
        "Time_Since_Previous_Reading_Min"
    ]
    * 10.0
)

df[
    "Humidity_Rate_per_10min"
] = (
    (
        df[
            "Humidity_%"
        ]
        -
        df[
            "Previous_Humidity_pct"
        ]
    )
    /
    df[
        "Time_Since_Previous_Reading_Min"
    ]
    * 10.0
)


# ============================================================
# 8. EXACT TIMESTAMP LAGS: 10 / 30 / 60 MINUTES
# ============================================================
#
# Exact timestamp joins are used instead of row shifts.
# Therefore the 12 packet-loss gaps cannot create fake lags.

lag_coverage_rows = []

base_lookup = df[
    SEQUENCE_KEY
    + [
        "Timestamp",
        "Temperature_C",
        "Humidity_%",
    ]
].copy()

for minutes in [
    10,
    30,
    60,
]:
    lookup = base_lookup.copy()

    lookup[
        "Timestamp"
    ] = (
        lookup[
            "Timestamp"
        ]
        +
        pd.to_timedelta(
            minutes,
            unit="m",
        )
    )

    temp_lag = (
        f"Temperature_Lag_{minutes}m"
    )

    humidity_lag = (
        f"Humidity_Lag_{minutes}m"
    )

    lookup = lookup.rename(
        columns={
            "Temperature_C": temp_lag,
            "Humidity_%": humidity_lag,
        }
    )

    df = df.merge(
        lookup,
        on=SEQUENCE_KEY
        + ["Timestamp"],
        how="left",
        validate="one_to_one",
    )

    df[
        f"Temperature_Change_{minutes}m"
    ] = (
        df[
            "Temperature_C"
        ]
        -
        df[
            temp_lag
        ]
    )

    df[
        f"Humidity_Change_{minutes}m"
    ] = (
        df[
            "Humidity_%"
        ]
        -
        df[
            humidity_lag
        ]
    )

    temp_available = int(
        df[
            temp_lag
        ]
        .notna()
        .sum()
    )

    humidity_available = int(
        df[
            humidity_lag
        ]
        .notna()
        .sum()
    )

    lag_coverage_rows.append({
        "Lag_Minutes": minutes,
        "Temperature_Pairs_Available": (
            temp_available
        ),
        "Temperature_Coverage_Percent": (
            temp_available
            / len(df)
            * 100
        ),
        "Humidity_Pairs_Available": (
            humidity_available
        ),
        "Humidity_Coverage_Percent": (
            humidity_available
            / len(df)
            * 100
        ),
    })

lag_coverage = pd.DataFrame(
    lag_coverage_rows
)

lag_coverage.to_csv(
    LAG_COVERAGE_PATH,
    index=False,
)


# ============================================================
# 9. TIMESTAMP-AWARE ROLLING WINDOWS
# ============================================================
#
# Windows include the CURRENT observation and historical
# observations inside the preceding 30/60 minutes.
#
# No future observations are included.
#
# We calculate:
#   - mean
#   - population standard deviation
#   - observation count
#
# Count is useful because real packet loss can reduce the
# number of observations available inside a time window.

rolling_parts = []

for key, group in df.groupby(
    SEQUENCE_KEY,
    sort=False,
):
    group = group.sort_values(
        "Timestamp"
    ).copy()

    indexed = (
        group
        .set_index(
            "Timestamp"
        )
    )

    for minutes in [
        30,
        60,
    ]:
        window = (
            f"{minutes}min"
        )

        rolling = (
            indexed[
                [
                    "Temperature_C",
                    "Humidity_%",
                ]
            ]
            .rolling(
                window=window,
                min_periods=1,
                closed="both",
            )
        )

        group[
            f"Temperature_RollMean_{minutes}m"
        ] = (
            rolling[
                "Temperature_C"
            ]
            .mean()
            .to_numpy()
        )

        group[
            f"Humidity_RollMean_{minutes}m"
        ] = (
            rolling[
                "Humidity_%"
            ]
            .mean()
            .to_numpy()
        )

        group[
            f"Temperature_RollStd_{minutes}m"
        ] = (
            rolling[
                "Temperature_C"
            ]
            .std(
                ddof=0
            )
            .to_numpy()
        )

        group[
            f"Humidity_RollStd_{minutes}m"
        ] = (
            rolling[
                "Humidity_%"
            ]
            .std(
                ddof=0
            )
            .to_numpy()
        )

        group[
            f"Observation_Count_{minutes}m"
        ] = (
            rolling[
                "Temperature_C"
            ]
            .count()
            .to_numpy()
        )

    rolling_parts.append(
        group
    )

df = (
    pd.concat(
        rolling_parts,
        axis=0,
        ignore_index=True,
    )
    .sort_values(
        SEQUENCE_KEY
        + ["Timestamp"]
    )
    .reset_index(drop=True)
)


# ============================================================
# 10. REMOVE TEMPORARY INTERNAL COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "Previous_Timestamp",
        "Previous_Temperature_C",
        "Previous_Humidity_pct",
    ]
)


# ============================================================
# 11. BUILD FINAL ENGINEERED OUTPUT SCHEMA
# ============================================================
#
# Actual_Total_Drying_Time is intentionally NOT included.
#
# Fire_Risk_Assessment_Timestamp stays only so later evaluation
# can correctly identify unique hourly ground-truth events.
#
# Batch_ID stays only for grouped splitting/CV.

metadata_columns = [
    "Timestamp",
    "Batch_ID",
    "Fire_Risk_Assessment_Timestamp",
]

categorical_candidate_features = [
    "Smokehouse_ID",
    "Layer_ID",
    "Temperature_Trend",
    "Humidity_Trend",
]

base_numeric_candidate_features = [
    "Temperature_C",
    "Humidity_%",
    "Elapsed_Drying_Hours",
    "Number_of_Sheets",
    "Layer_Area_m2",
    "Temperature_Change",
    "Humidity_Change",
]

engineered_numeric_candidate_features = [
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

target_columns = [
    "Remaining_Drying_Time",
    "Fire_Risk_Label",
]

output_columns = (
    metadata_columns
    + categorical_candidate_features
    + base_numeric_candidate_features
    + engineered_numeric_candidate_features
    + target_columns
)

engineered_df = df[
    output_columns
].copy()


# ============================================================
# 12. FEATURE MANIFEST
# ============================================================

manifest_rows = []


def add_manifest(
    column,
    role,
    data_type,
    runtime_available,
    description,
):
    manifest_rows.append({
        "Column": column,
        "Role": role,
        "Data_Type": data_type,
        "Available_At_Runtime": runtime_available,
        "Description": description,
    })


# Metadata
add_manifest(
    "Timestamp",
    "Metadata",
    "datetime",
    "Yes",
    "Sensor observation timestamp; used for ordering and temporal feature construction, not as a raw predictor by default.",
)

add_manifest(
    "Batch_ID",
    "Group_Only",
    "categorical",
    "Yes",
    "Batch grouping identifier for batch-aware splitting/CV; never a physical predictor.",
)

add_manifest(
    "Fire_Risk_Assessment_Timestamp",
    "Target_Metadata",
    "datetime",
    "No for prediction",
    "Hourly keeper/expert ground-truth assessment timestamp; used for evaluation bookkeeping only.",
)


# Categorical candidates
add_manifest(
    "Smokehouse_ID",
    "Candidate_Feature",
    "categorical",
    "Yes",
    "Known smokehouse/site identity; retained as a contextual candidate pending Stage 08 analysis.",
)

add_manifest(
    "Layer_ID",
    "Candidate_Feature",
    "categorical",
    "Yes",
    "Physical smokehouse layer L1/L2/L3.",
)

add_manifest(
    "Temperature_Trend",
    "Candidate_Feature",
    "categorical",
    "Yes",
    "Past/current temperature trend category from the source dataset.",
)

add_manifest(
    "Humidity_Trend",
    "Candidate_Feature",
    "categorical",
    "Yes",
    "Past/current humidity trend category from the source dataset.",
)


# Base numeric
base_descriptions = {
    "Temperature_C": "Current layer temperature.",
    "Humidity_%": "Current layer relative humidity.",
    "Elapsed_Drying_Hours": "Elapsed drying time known at runtime.",
    "Number_of_Sheets": "Batch sheet load.",
    "Layer_Area_m2": "Physical layer area.",
    "Temperature_Change": "Source short-term temperature change value.",
    "Humidity_Change": "Source short-term humidity change value.",
}

for column in base_numeric_candidate_features:
    add_manifest(
        column,
        "Candidate_Feature",
        "numeric",
        "Yes",
        base_descriptions[
            column
        ],
    )


# Engineered numeric
for column in engineered_numeric_candidate_features:

    if "Lag_" in column:
        description = (
            "Exact timestamp-matched historical sensor value; "
            "missing when the required past timestamp does not exist."
        )

    elif "Change_" in column:
        description = (
            "Current value minus exact historical value at the named lag; "
            "uses current/past data only."
        )

    elif "RollMean" in column:
        description = (
            "Timestamp-aware rolling mean using current and previous "
            "observations inside the named time window."
        )

    elif "RollStd" in column:
        description = (
            "Timestamp-aware rolling population standard deviation using "
            "current and previous observations inside the named time window."
        )

    elif "Observation_Count" in column:
        description = (
            "Number of actual available observations in the timestamp-aware "
            "rolling window; reflects sequence start and packet-loss availability."
        )

    elif "Rate_per_10min" in column:
        description = (
            "Change from previous available reading normalized to a "
            "10-minute equivalent using the actual timestamp interval."
        )

    elif column == "Time_Since_Previous_Reading_Min":
        description = (
            "Actual minutes since previous available observation in the same sequence."
        )

    else:
        description = (
            "Deployment-safe engineered temporal candidate."
        )

    add_manifest(
        column,
        "Candidate_Feature",
        "numeric",
        "Yes",
        description,
    )


# Targets
add_manifest(
    "Remaining_Drying_Time",
    "Regression_Target",
    "numeric",
    "No",
    "Target for remaining drying-time regression.",
)

add_manifest(
    "Fire_Risk_Label",
    "Classification_Target",
    "categorical",
    "No",
    "Target for Normal/Warning/Critical fire-risk classification.",
)


# Explicit excluded leakage field
add_manifest(
    "Actual_Total_Drying_Time",
    "EXCLUDED_LEAKAGE_REFERENCE",
    "numeric",
    "No",
    "Retrospective total drying duration used only for target construction/reference; intentionally absent from engineered dataset.",
)

manifest_df = pd.DataFrame(
    manifest_rows
)

manifest_df.to_csv(
    MANIFEST_PATH,
    index=False,
)


# ============================================================
# 13. ENGINEERED FEATURE MISSINGNESS
# ============================================================
#
# Missing values are EXPECTED in history-dependent features
# near the beginning of each sequence and after exact
# timestamp gaps.
#
# They are NOT imputed here.
# Learned/constant handling will be decided after splitting.

missing_rows = []

for column in output_columns:
    missing_count = int(
        engineered_df[
            column
        ]
        .isna()
        .sum()
    )

    missing_rows.append({
        "Column": column,
        "Missing_Count": missing_count,
        "Missing_Percent": (
            missing_count
            / len(
                engineered_df
            )
            * 100
        ),
    })

missing_summary = pd.DataFrame(
    missing_rows
)

missing_summary.to_csv(
    MISSING_PATH,
    index=False,
)


# ============================================================
# 14. LEAKAGE / PRESERVATION VALIDATION
# ============================================================

actual_total_absent = (
    "Actual_Total_Drying_Time"
    not in engineered_df.columns
)

target_missing = int(
    engineered_df[
        target_columns
    ]
    .isna()
    .sum()
    .sum()
)

metadata_target_changes = 0

for column in [
    "Timestamp",
    "Batch_ID",
    "Remaining_Drying_Time",
    "Fire_Risk_Label",
    "Fire_Risk_Assessment_Timestamp",
]:
    left = preserved_reference[
        column
    ].reset_index(
        drop=True
    )

    right = engineered_df[
        column
    ].reset_index(
        drop=True
    )

    metadata_target_changes += int(
        (
            left
            !=
            right
        ).sum()
    )


# ============================================================
# 15. NO-FUTURE-DATA AUDIT FOR EXACT LAGS
# ============================================================

future_lag_violations = 0

# By construction, each lookup uses:
# source_timestamp + lag = current_timestamp
# therefore source_timestamp = current - lag.
#
# Validate that lag values are unavailable for rows earlier
# than the requested history duration within each sequence.

sequence_start = (
    engineered_df.groupby(
        SEQUENCE_KEY
    )[
        "Timestamp"
    ]
    .transform(
        "min"
    )
)

minutes_since_sequence_start = (
    (
        engineered_df[
            "Timestamp"
        ]
        -
        sequence_start
    )
    .dt
    .total_seconds()
    / 60
)

for minutes in [
    10,
    30,
    60,
]:
    for column in [
        f"Temperature_Lag_{minutes}m",
        f"Humidity_Lag_{minutes}m",
    ]:
        invalid = (
            (
                minutes_since_sequence_start
                <
                minutes
            )
            &
            engineered_df[
                column
            ]
            .notna()
        )

        future_lag_violations += int(
            invalid.sum()
        )


# ============================================================
# 16. SAMPLING / PACKET LOSS PRESERVATION
# ============================================================

temp = engineered_df[
    SEQUENCE_KEY
    + ["Timestamp"]
].copy()

temp[
    "_Previous"
] = (
    temp.groupby(
        SEQUENCE_KEY
    )[
        "Timestamp"
    ]
    .shift(1)
)

temp[
    "_Interval"
] = (
    (
        temp[
            "Timestamp"
        ]
        -
        temp[
            "_Previous"
        ]
    )
    .dt
    .total_seconds()
    / 60
)

interval_counts = (
    temp[
        "_Interval"
    ]
    .dropna()
    .value_counts()
)

n10 = int(
    interval_counts.get(
        10.0,
        0,
    )
)

n20 = int(
    interval_counts.get(
        20.0,
        0,
    )
)

n_other = int(
    interval_counts[
        ~interval_counts.index.isin(
            [
                10.0,
                20.0,
            ]
        )
    ].sum()
)


# ============================================================
# 17. DUPLICATE CHECKS
# ============================================================

exact_duplicate_members = int(
    engineered_df.duplicated(
        keep=False
    ).sum()
)

duplicate_observation_keys = int(
    engineered_df.duplicated(
        subset=SEQUENCE_KEY
        + ["Timestamp"],
        keep="first",
    ).sum()
)


# ============================================================
# 18. SAVE ENGINEERED DATASET
# ============================================================

engineered_df.to_csv(
    OUTPUT_PATH,
    index=False,
    date_format="%Y-%m-%d %H:%M:%S",
)

with open(
    OUTPUT_PATH,
    "rb",
) as file:
    output_sha256 = hashlib.sha256(
        file.read()
    ).hexdigest()


# ============================================================
# 19. PASS / FAIL CHECKS
# ============================================================

candidate_feature_count = (
    len(
        categorical_candidate_features
    )
    +
    len(
        base_numeric_candidate_features
    )
    +
    len(
        engineered_numeric_candidate_features
    )
)

checks = pd.DataFrame(
    [
        {
            "Check": "Row count preserved",
            "Status": (
                "PASS"
                if len(
                    engineered_df
                )
                == EXPECTED_ROWS
                else "FAIL"
            ),
            "Details": (
                f"Rows="
                f"{len(engineered_df):,}"
            ),
        },
        {
            "Check": "Targets remain complete",
            "Status": (
                "PASS"
                if target_missing == 0
                else "FAIL"
            ),
            "Details": (
                f"Missing target cells="
                f"{target_missing}"
            ),
        },
        {
            "Check": "Actual_Total_Drying_Time excluded",
            "Status": (
                "PASS"
                if actual_total_absent
                else "FAIL"
            ),
            "Details": (
                "Leakage/reference field must not enter engineered dataset."
            ),
        },
        {
            "Check": "Metadata and targets preserved",
            "Status": (
                "PASS"
                if metadata_target_changes == 0
                else "FAIL"
            ),
            "Details": (
                f"Changed cells="
                f"{metadata_target_changes}"
            ),
        },
        {
            "Check": "No future exact-lag values",
            "Status": (
                "PASS"
                if future_lag_violations == 0
                else "FAIL"
            ),
            "Details": (
                f"Violations="
                f"{future_lag_violations}"
            ),
        },
        {
            "Check": "Sampling gaps preserved",
            "Status": (
                "PASS"
                if (
                    n10 == 24415
                    and
                    n20 == 12
                    and
                    n_other == 0
                )
                else "FAIL"
            ),
            "Details": (
                f"10-min={n10}; "
                f"20-min={n20}; "
                f"other={n_other}"
            ),
        },
        {
            "Check": "No duplicate observation keys",
            "Status": (
                "PASS"
                if duplicate_observation_keys == 0
                else "FAIL"
            ),
            "Details": (
                f"Extra duplicate keys="
                f"{duplicate_observation_keys}"
            ),
        },
        {
            "Check": "Expected 36 Batch-Layer sequences",
            "Status": (
                "PASS"
                if engineered_df.groupby(
                    SEQUENCE_KEY
                ).ngroups
                == 36
                else "FAIL"
            ),
            "Details": (
                f"Sequences="
                f"{engineered_df.groupby(SEQUENCE_KEY).ngroups}"
            ),
        },
        {
            "Check": "Candidate feature set created",
            "Status": (
                "PASS"
                if candidate_feature_count > 0
                else "FAIL"
            ),
            "Details": (
                f"Candidate features="
                f"{candidate_feature_count}"
            ),
        },
    ]
)

checks.to_csv(
    CHECKS_PATH,
    index=False,
)

failed_checks = (
    checks[
        checks[
            "Status"
        ]
        == "FAIL"
    ]
)

overall_status = (
    "PASS"
    if len(
        failed_checks
    )
    == 0
    else "FAIL"
)


# ============================================================
# 20. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- INPUT ---"
)

print(
    f"Rows / columns                 : "
    f"{len(df):,} / {len(EXPECTED_COLUMNS)}"
)

print(
    f"Input SHA-256                  : "
    f"{input_sha256}"
)


print(
    "\n--- CANDIDATE FEATURE SET ---"
)

print(
    f"Categorical candidates         : "
    f"{len(categorical_candidate_features)}"
)

print(
    f"Base numeric candidates        : "
    f"{len(base_numeric_candidate_features)}"
)

print(
    f"Engineered numeric candidates  : "
    f"{len(engineered_numeric_candidate_features)}"
)

print(
    f"Total candidate features       : "
    f"{candidate_feature_count}"
)

print(
    f"Output columns                 : "
    f"{len(engineered_df.columns)}"
)


print(
    "\n--- EXACT LAG COVERAGE ---"
)

print(
    lag_coverage
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\n--- ENGINEERED FEATURE MISSINGNESS ---"
)

missing_only = (
    missing_summary[
        missing_summary[
            "Missing_Count"
        ]
        > 0
    ]
)

if len(
    missing_only
) == 0:
    print(
        "No missing engineered feature values."
    )
else:
    print(
        missing_only
        .round(
            {
                "Missing_Percent": 4
            }
        )
        .to_string(
            index=False
        )
    )


print(
    "\n--- LEAKAGE / PRESERVATION ---"
)

print(
    f"Actual_Total_Drying_Time present: "
    f"{not actual_total_absent}"
)

print(
    f"Metadata/target changed cells   : "
    f"{metadata_target_changes}"
)

print(
    f"Future-lag violations           : "
    f"{future_lag_violations}"
)

print(
    f"Duplicate observation keys      : "
    f"{duplicate_observation_keys}"
)

print(
    f"10-minute / 20-minute intervals : "
    f"{n10:,} / {n20}"
)


print(
    "\n--- OUTPUT ---"
)

print(
    f"Feature-engineered dataset:\n"
    f"{OUTPUT_PATH}"
)

print(
    f"\nOutput SHA-256:\n"
    f"{output_sha256}"
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
        "COMPONENT 3 - STAGE 07: FEATURE ENGINEERING\n"
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
        "INPUT\n"
        "-----\n"
    )

    report.write(
        "Dataset: "
        "Component3_Dataset_Preprocessed.csv\n"
    )

    report.write(
        f"Rows: "
        f"{len(df):,}\n"
    )

    report.write(
        f"Input SHA-256: "
        f"{input_sha256}\n\n"
    )

    report.write(
        "FEATURE-ENGINEERING PRINCIPLE\n"
        "-----------------------------\n"
    )

    report.write(
        "All temporal features were created using only the "
        "current observation and/or earlier observations from "
        "the same Smokehouse x Batch x Layer sequence. "
        "No future observation was used.\n\n"
    )

    report.write(
        "CANDIDATE FEATURE COUNTS\n"
        "------------------------\n"
    )

    report.write(
        f"Categorical candidates: "
        f"{len(categorical_candidate_features)}\n"
    )

    report.write(
        f"Base numeric candidates: "
        f"{len(base_numeric_candidate_features)}\n"
    )

    report.write(
        f"Engineered numeric candidates: "
        f"{len(engineered_numeric_candidate_features)}\n"
    )

    report.write(
        f"Total candidate features: "
        f"{candidate_feature_count}\n\n"
    )

    report.write(
        "TEMPORAL FEATURES\n"
        "-----------------\n"
    )

    report.write(
        "- Actual time since previous available reading.\n"
    )

    report.write(
        "- Previous-reading temperature/humidity rate "
        "normalized to a 10-minute equivalent.\n"
    )

    report.write(
        "- Exact 10-, 30-, and 60-minute temperature/RH lags.\n"
    )

    report.write(
        "- Exact 10-, 30-, and 60-minute temperature/RH changes.\n"
    )

    report.write(
        "- Timestamp-aware 30- and 60-minute rolling means.\n"
    )

    report.write(
        "- Timestamp-aware 30- and 60-minute rolling standard "
        "deviations.\n"
    )

    report.write(
        "- Actual observation counts inside 30- and 60-minute "
        "windows.\n\n"
    )

    report.write(
        "EXACT-LAG COVERAGE\n"
        "------------------\n"
    )

    report.write(
        lag_coverage
        .round(6)
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "EXPECTED HISTORY-RELATED MISSINGNESS\n"
        "------------------------------------\n"
    )

    report.write(
        "Exact lag/change features are intentionally missing "
        "when the required historical timestamp does not exist "
        "(for example, at the beginning of a drying sequence "
        "or around a packet-loss gap). These values are NOT "
        "filled in Stage 07. Missing-value handling for model "
        "pipelines will be fitted on training data only in "
        "Stage 10.\n\n"
    )

    report.write(
        missing_only
        .round(6)
        .to_string(
            index=False
        )
        if len(
            missing_only
        )
        > 0
        else "No engineered missingness."
    )

    report.write(
        "\n\n"
    )

    report.write(
        "LEAKAGE PROTECTION\n"
        "------------------\n"
    )

    report.write(
        "- Actual_Total_Drying_Time is deliberately absent "
        "from the engineered dataset.\n"
    )

    report.write(
        "- Batch_ID remains only for grouped splitting/CV.\n"
    )

    report.write(
        "- Fire_Risk_Assessment_Timestamp remains only for "
        "ground-truth/evaluation bookkeeping.\n"
    )

    report.write(
        "- Remaining_Drying_Time is the regression target.\n"
    )

    report.write(
        "- Fire_Risk_Label is the classification target.\n"
    )

    report.write(
        "- No previous or future Fire_Risk_Label was engineered "
        "as a predictor.\n"
    )

    report.write(
        "- No global statistic was learned from the full data.\n"
    )

    report.write(
        "- No scaling, encoding, resampling, or model fitting "
        "was performed.\n\n"
    )

    report.write(
        "OUTPUT\n"
        "------\n"
    )

    report.write(
        f"Dataset: "
        f"{OUTPUT_PATH}\n"
    )

    report.write(
        f"Rows: "
        f"{len(engineered_df):,}\n"
    )

    report.write(
        f"Columns: "
        f"{len(engineered_df.columns)}\n"
    )

    report.write(
        f"Output SHA-256: "
        f"{output_sha256}\n\n"
    )

    report.write(
        "STAGE 08 NOTE\n"
        "-------------\n"
    )

    report.write(
        "All candidate features are retained at this stage. "
        "Stage 08 will analyze redundancy, target association, "
        "feature availability, multicollinearity/correlation, "
        "and task-specific usefulness before final candidate "
        "sets are frozen for the batch-aware split.\n\n"
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
        f"STAGE 07 STATUS: "
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
    f"STAGE 07 FEATURE ENGINEERING: "
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
    "\nNo source dataset was modified."
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 07 failed. Review 07_feature_engineering_report.txt."
    )
