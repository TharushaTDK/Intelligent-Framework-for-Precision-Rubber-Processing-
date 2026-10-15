from pathlib import Path
import hashlib
import pandas as pd
import numpy as np


# ============================================================
# COMPONENT 3 - STAGE 03: DATA QUALITY ANALYSIS
# ============================================================
#
# Purpose:
#   Perform a deep quality audit of the FINAL FROZEN raw
#   Component 3 dataset before preprocessing.
#
# This stage checks:
#   - frozen dataset identity
#   - missing-value patterns
#   - missing sensor-event distribution
#   - duplicate rows / duplicate observation keys
#   - timestamp validity and ordering
#   - 10-minute sampling and 20-minute packet-loss gaps
#   - sequence integrity
#   - categorical-value consistency
#   - numeric plausibility
#   - statistical IQR outliers for REVIEW ONLY
#
# Important:
#   - NO raw rows are deleted.
#   - NO missing values are imputed.
#   - NO outliers are removed.
#   - NO train/test split is performed.
#   - NO ML features are engineered.
#
# The purpose is to document quality issues so that Stage 04
# can make justified preprocessing decisions.
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
    / "03_data_quality_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "03_data_quality_checks.csv"
)

MISSING_SUMMARY_PATH = (
    REPORT_DIR
    / "03_missing_value_summary.csv"
)

MISSING_ROWS_PATH = (
    REPORT_DIR
    / "03_missing_sensor_rows.csv"
)

MISSING_BY_GROUP_PATH = (
    REPORT_DIR
    / "03_missing_by_group.csv"
)

GAP_REPORT_PATH = (
    REPORT_DIR
    / "03_sampling_gap_report.csv"
)

SEQUENCE_QUALITY_PATH = (
    REPORT_DIR
    / "03_sequence_quality.csv"
)

NUMERIC_QUALITY_PATH = (
    REPORT_DIR
    / "03_numeric_quality_summary.csv"
)

CATEGORICAL_QUALITY_PATH = (
    REPORT_DIR
    / "03_categorical_quality_summary.csv"
)

IQR_REVIEW_PATH = (
    REPORT_DIR
    / "03_iqr_outlier_review.csv"
)


# ============================================================
# 3. EXPECTED FINAL DATASET IDENTITY
# ============================================================

EXPECTED_SHA256 = (
    "229139d2b4413fe15a8d322023b3b425"
    "74f9dccd5df6a91550bc72f05faaa5cb"
)

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

EXPECTED_SMOKEHOUSES = {
    "SH01",
    "SH02",
}

EXPECTED_LAYERS = {
    "L1",
    "L2",
    "L3",
}

EXPECTED_TRENDS = {
    "Stable",
    "Increasing",
    "Decreasing",
}

EXPECTED_FIRE_LABELS = {
    "Normal",
    "Warning",
    "Critical",
}

SEQUENCE_KEY = [
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
]

OBSERVATION_KEY = (
    SEQUENCE_KEY
    + ["Timestamp"]
)


# ============================================================
# 4. LOAD RAW DATASET
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 03: DATA QUALITY ANALYSIS")
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
# 5. DATASET FINGERPRINT / SCHEMA
# ============================================================

with open(DATA_PATH, "rb") as file:
    dataset_sha256 = hashlib.sha256(
        file.read()
    ).hexdigest()

hash_matches = (
    dataset_sha256
    == EXPECTED_SHA256
)

row_count_matches = (
    len(df)
    == EXPECTED_ROWS
)

schema_matches = (
    list(df.columns)
    == EXPECTED_COLUMNS
)


# ============================================================
# 6. WORKING COPY + SAFE TYPE CONVERSION
# ============================================================

quality_df = df.copy()

quality_df["Timestamp"] = pd.to_datetime(
    quality_df["Timestamp"],
    errors="coerce",
)

quality_df[
    "Fire_Risk_Assessment_Timestamp"
] = pd.to_datetime(
    quality_df[
        "Fire_Risk_Assessment_Timestamp"
    ],
    errors="coerce",
)

numeric_columns = [
    "Temperature_C",
    "Humidity_%",
    "Elapsed_Drying_Hours",
    "Number_of_Sheets",
    "Layer_Area_m2",
    "Temperature_Change",
    "Humidity_Change",
    "Actual_Total_Drying_Time",
    "Remaining_Drying_Time",
]

for column in numeric_columns:
    quality_df[column] = pd.to_numeric(
        quality_df[column],
        errors="coerce",
    )


# ============================================================
# 7. MISSING-VALUE ANALYSIS
# ============================================================

missing_summary = pd.DataFrame({
    "Column": quality_df.columns,
    "Missing_Count": [
        int(
            quality_df[column]
            .isna()
            .sum()
        )
        for column in quality_df.columns
    ],
})

missing_summary[
    "Missing_Percent"
] = (
    missing_summary[
        "Missing_Count"
    ]
    / len(quality_df)
    * 100
)

missing_summary.to_csv(
    MISSING_SUMMARY_PATH,
    index=False,
)

temp_missing = (
    quality_df[
        "Temperature_C"
    ]
    .isna()
)

humidity_missing = (
    quality_df[
        "Humidity_%"
    ]
    .isna()
)

any_sensor_missing = (
    temp_missing
    |
    humidity_missing
)

temperature_only_missing = int(
    (
        temp_missing
        &
        ~humidity_missing
    ).sum()
)

humidity_only_missing = int(
    (
        ~temp_missing
        &
        humidity_missing
    ).sum()
)

both_sensor_missing = int(
    (
        temp_missing
        &
        humidity_missing
    ).sum()
)

rows_with_sensor_missing = int(
    any_sensor_missing.sum()
)

sensor_missing_percent = (
    rows_with_sensor_missing
    / len(quality_df)
    * 100
)

missing_rows = quality_df.loc[
    any_sensor_missing,
    [
        "Smokehouse_ID",
        "Batch_ID",
        "Layer_ID",
        "Timestamp",
        "Temperature_C",
        "Humidity_%",
        "Temperature_Change",
        "Humidity_Change",
        "Temperature_Trend",
        "Humidity_Trend",
        "Elapsed_Drying_Hours",
        "Fire_Risk_Label",
    ],
].copy()

missing_rows[
    "Missing_Pattern"
] = np.select(
    [
        missing_rows[
            "Temperature_C"
        ].isna()
        &
        missing_rows[
            "Humidity_%"
        ].isna(),

        missing_rows[
            "Temperature_C"
        ].isna()
        &
        missing_rows[
            "Humidity_%"
        ].notna(),

        missing_rows[
            "Temperature_C"
        ].notna()
        &
        missing_rows[
            "Humidity_%"
        ].isna(),
    ],
    [
        "Both",
        "Temperature only",
        "Humidity only",
    ],
    default="Unexpected",
)

missing_rows.to_csv(
    MISSING_ROWS_PATH,
    index=False,
)


# ============================================================
# 8. DERIVED-FIELD MISSINGNESS CONSISTENCY
# ============================================================

temp_change_missing_mismatch = int(
    (
        quality_df[
            "Temperature_C"
        ].isna()
        !=
        quality_df[
            "Temperature_Change"
        ].isna()
    ).sum()
)

temp_trend_missing_mismatch = int(
    (
        quality_df[
            "Temperature_C"
        ].isna()
        !=
        quality_df[
            "Temperature_Trend"
        ].isna()
    ).sum()
)

humidity_change_missing_mismatch = int(
    (
        quality_df[
            "Humidity_%"
        ].isna()
        !=
        quality_df[
            "Humidity_Change"
        ].isna()
    ).sum()
)

humidity_trend_missing_mismatch = int(
    (
        quality_df[
            "Humidity_%"
        ].isna()
        !=
        quality_df[
            "Humidity_Trend"
        ].isna()
    ).sum()
)

derived_missing_mismatches = (
    temp_change_missing_mismatch
    +
    temp_trend_missing_mismatch
    +
    humidity_change_missing_mismatch
    +
    humidity_trend_missing_mismatch
)


# ============================================================
# 9. MISSINGNESS BY SMOKEHOUSE / BATCH / LAYER
# ============================================================

missing_group_df = quality_df.loc[
    any_sensor_missing
].copy()

if len(missing_group_df) > 0:
    missing_by_group = (
        missing_group_df
        .groupby(
            [
                "Smokehouse_ID",
                "Batch_ID",
                "Layer_ID",
            ],
            dropna=False,
        )
        .agg(
            Rows_With_Any_Sensor_Missing=(
                "Timestamp",
                "size",
            ),
            Temperature_Missing=(
                "Temperature_C",
                lambda x: int(
                    x.isna().sum()
                ),
            ),
            Humidity_Missing=(
                "Humidity_%",
                lambda x: int(
                    x.isna().sum()
                ),
            ),
        )
        .reset_index()
    )
else:
    missing_by_group = pd.DataFrame(
        columns=[
            "Smokehouse_ID",
            "Batch_ID",
            "Layer_ID",
            "Rows_With_Any_Sensor_Missing",
            "Temperature_Missing",
            "Humidity_Missing",
        ]
    )

missing_by_group.to_csv(
    MISSING_BY_GROUP_PATH,
    index=False,
)

batches_with_any_sensor_missing = int(
    quality_df.loc[
        any_sensor_missing,
        [
            "Smokehouse_ID",
            "Batch_ID",
        ],
    ]
    .drop_duplicates()
    .shape[0]
)


# ============================================================
# 10. DUPLICATE ANALYSIS
# ============================================================

exact_duplicate_rows = int(
    quality_df.duplicated(
        keep=False
    ).sum()
)

# Count extra duplicated observations, not all members of duplicate groups.
duplicate_observation_keys = int(
    quality_df.duplicated(
        subset=OBSERVATION_KEY,
        keep="first",
    ).sum()
)


# ============================================================
# 11. IDENTIFIER / CATEGORICAL CONSISTENCY
# ============================================================

observed_smokehouses = set(
    quality_df[
        "Smokehouse_ID"
    ]
    .dropna()
    .unique()
)

observed_layers = set(
    quality_df[
        "Layer_ID"
    ]
    .dropna()
    .unique()
)

observed_temp_trends = set(
    quality_df[
        "Temperature_Trend"
    ]
    .dropna()
    .unique()
)

observed_humidity_trends = set(
    quality_df[
        "Humidity_Trend"
    ]
    .dropna()
    .unique()
)

observed_fire_labels = set(
    quality_df[
        "Fire_Risk_Label"
    ]
    .dropna()
    .unique()
)

categorical_rows = []

for column, observed, expected in [
    (
        "Smokehouse_ID",
        observed_smokehouses,
        EXPECTED_SMOKEHOUSES,
    ),
    (
        "Layer_ID",
        observed_layers,
        EXPECTED_LAYERS,
    ),
    (
        "Temperature_Trend",
        observed_temp_trends,
        EXPECTED_TRENDS,
    ),
    (
        "Humidity_Trend",
        observed_humidity_trends,
        EXPECTED_TRENDS,
    ),
    (
        "Fire_Risk_Label",
        observed_fire_labels,
        EXPECTED_FIRE_LABELS,
    ),
]:
    categorical_rows.append({
        "Column": column,
        "Observed_Values": " | ".join(
            sorted(
                str(value)
                for value in observed
            )
        ),
        "Unexpected_Values": " | ".join(
            sorted(
                str(value)
                for value in (
                    observed
                    -
                    expected
                )
            )
        ),
        "Missing_Expected_Values": " | ".join(
            sorted(
                str(value)
                for value in (
                    expected
                    -
                    observed
                )
            )
        ),
        "Valid": (
            observed
            == expected
        ),
    })

categorical_quality = pd.DataFrame(
    categorical_rows
)

categorical_quality.to_csv(
    CATEGORICAL_QUALITY_PATH,
    index=False,
)

categorical_errors = int(
    (
        ~categorical_quality[
            "Valid"
        ]
    ).sum()
)


# ============================================================
# 12. MANDATORY FIELD COMPLETENESS
# ============================================================

mandatory_columns = [
    "Smokehouse_ID",
    "Timestamp",
    "Batch_ID",
    "Layer_ID",
    "Elapsed_Drying_Hours",
    "Number_of_Sheets",
    "Layer_Area_m2",
    "Actual_Total_Drying_Time",
    "Remaining_Drying_Time",
    "Fire_Risk_Label",
    "Fire_Risk_Assessment_Timestamp",
]

mandatory_missing_count = int(
    quality_df[
        mandatory_columns
    ]
    .isna()
    .sum()
    .sum()
)


# ============================================================
# 13. NUMERIC PLAUSIBILITY
# ============================================================
#
# These are broad physical/domain sanity limits, NOT trimming
# thresholds. Statistical extremes inside these limits are not
# automatically errors.

plausibility_rules = {
    "Temperature_C": {
        "min_allowed": 0.0,
        "max_allowed": 100.0,
    },
    "Humidity_%": {
        "min_allowed": 0.0,
        "max_allowed": 100.0,
    },
    "Elapsed_Drying_Hours": {
        "min_allowed": 0.0,
        "max_allowed": 200.0,
    },
    "Number_of_Sheets": {
        "min_allowed": 1.0,
        "max_allowed": 1000.0,
    },
    "Layer_Area_m2": {
        "min_allowed": 0.01,
        "max_allowed": 1000.0,
    },
    "Actual_Total_Drying_Time": {
        "min_allowed": 1.0,
        "max_allowed": 200.0,
    },
    "Remaining_Drying_Time": {
        "min_allowed": 0.0,
        "max_allowed": 200.0,
    },
}

numeric_quality_rows = []

total_numeric_plausibility_errors = 0

for column, limits in plausibility_rules.items():
    series = quality_df[
        column
    ].dropna()

    below_allowed = int(
        (
            series
            <
            limits[
                "min_allowed"
            ]
        ).sum()
    )

    above_allowed = int(
        (
            series
            >
            limits[
                "max_allowed"
            ]
        ).sum()
    )

    invalid_count = (
        below_allowed
        +
        above_allowed
    )

    total_numeric_plausibility_errors += (
        invalid_count
    )

    numeric_quality_rows.append({
        "Column": column,
        "Observed_Min": float(
            series.min()
        ),
        "Observed_Max": float(
            series.max()
        ),
        "Observed_Mean": float(
            series.mean()
        ),
        "Allowed_Min": (
            limits[
                "min_allowed"
            ]
        ),
        "Allowed_Max": (
            limits[
                "max_allowed"
            ]
        ),
        "Below_Allowed_Count": (
            below_allowed
        ),
        "Above_Allowed_Count": (
            above_allowed
        ),
        "Invalid_Count": (
            invalid_count
        ),
    })

numeric_quality = pd.DataFrame(
    numeric_quality_rows
)

numeric_quality.to_csv(
    NUMERIC_QUALITY_PATH,
    index=False,
)


# ============================================================
# 14. IQR OUTLIER REVIEW
# ============================================================
#
# IMPORTANT:
# IQR outliers are reported only as statistical extremes.
# They are NOT automatically data errors and are NOT removed.

iqr_columns = [
    "Temperature_C",
    "Humidity_%",
    "Temperature_Change",
    "Humidity_Change",
    "Elapsed_Drying_Hours",
    "Number_of_Sheets",
    "Actual_Total_Drying_Time",
    "Remaining_Drying_Time",
]

iqr_rows = []

for column in iqr_columns:
    series = quality_df[
        column
    ].dropna()

    q1 = float(
        series.quantile(0.25)
    )

    q3 = float(
        series.quantile(0.75)
    )

    iqr = q3 - q1

    lower_bound = (
        q1
        -
        1.5 * iqr
    )

    upper_bound = (
        q3
        +
        1.5 * iqr
    )

    outlier_count = int(
        (
            (
                series
                <
                lower_bound
            )
            |
            (
                series
                >
                upper_bound
            )
        ).sum()
    )

    iqr_rows.append({
        "Column": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_IQR_Bound": (
            lower_bound
        ),
        "Upper_IQR_Bound": (
            upper_bound
        ),
        "IQR_Outlier_Count_For_Review": (
            outlier_count
        ),
        "Observed_Min": float(
            series.min()
        ),
        "Observed_Max": float(
            series.max()
        ),
    })

iqr_review = pd.DataFrame(
    iqr_rows
)

iqr_review.to_csv(
    IQR_REVIEW_PATH,
    index=False,
)


# ============================================================
# 15. SAMPLING / PACKET-LOSS ANALYSIS
# ============================================================

quality_df = quality_df.sort_values(
    SEQUENCE_KEY
    + ["Timestamp"]
).reset_index(drop=True)

quality_df[
    "Previous_Timestamp"
] = (
    quality_df
    .groupby(
        SEQUENCE_KEY
    )[
        "Timestamp"
    ]
    .shift(1)
)

quality_df[
    "Interval_Minutes"
] = (
    (
        quality_df[
            "Timestamp"
        ]
        -
        quality_df[
            "Previous_Timestamp"
        ]
    )
    .dt
    .total_seconds()
    / 60
)

interval_counts = (
    quality_df[
        "Interval_Minutes"
    ]
    .dropna()
    .value_counts()
    .sort_index()
)

ten_minute_intervals = int(
    interval_counts.get(
        10.0,
        0,
    )
)

twenty_minute_intervals = int(
    interval_counts.get(
        20.0,
        0,
    )
)

unexpected_interval_count = int(
    interval_counts[
        ~interval_counts.index.isin(
            [
                10.0,
                20.0,
            ]
        )
    ].sum()
)

packet_loss_rows = (
    quality_df[
        quality_df[
            "Interval_Minutes"
        ]
        == 20.0
    ]
    .copy()
)

gap_report = packet_loss_rows[
    [
        "Smokehouse_ID",
        "Batch_ID",
        "Layer_ID",
        "Previous_Timestamp",
        "Timestamp",
        "Interval_Minutes",
    ]
].copy()

gap_report[
    "Estimated_Missed_10min_Packets"
] = (
    (
        gap_report[
            "Interval_Minutes"
        ]
        / 10
    )
    .round()
    .astype(int)
    - 1
)

gap_report.to_csv(
    GAP_REPORT_PATH,
    index=False,
)

estimated_missed_packets = int(
    gap_report[
        "Estimated_Missed_10min_Packets"
    ].sum()
)


# ============================================================
# 16. MISSING SENSOR ROWS VS PACKET-LOSS ENDPOINTS
# ============================================================

gap_endpoint_keys = set()

for _, row in packet_loss_rows.iterrows():
    current_key = (
        row[
            "Smokehouse_ID"
        ],
        row[
            "Batch_ID"
        ],
        row[
            "Layer_ID"
        ],
        row[
            "Timestamp"
        ],
    )

    previous_key = (
        row[
            "Smokehouse_ID"
        ],
        row[
            "Batch_ID"
        ],
        row[
            "Layer_ID"
        ],
        row[
            "Previous_Timestamp"
        ],
    )

    gap_endpoint_keys.add(
        current_key
    )

    gap_endpoint_keys.add(
        previous_key
    )

missing_observation_keys = set(
    tuple(row)
    for row in quality_df.loc[
        (
            quality_df[
                "Temperature_C"
            ].isna()
            |
            quality_df[
                "Humidity_%"
            ].isna()
        ),
        OBSERVATION_KEY,
    ].itertuples(
        index=False,
        name=None,
    )
)

missing_at_gap_endpoints = len(
    missing_observation_keys
    &
    gap_endpoint_keys
)


# ============================================================
# 17. SEQUENCE-LEVEL QUALITY
# ============================================================

sequence_quality_rows = []

max_consecutive_sensor_missing = 0
sequence_timestamp_order_errors = 0

for key, sequence in quality_df.groupby(
    SEQUENCE_KEY,
    sort=True,
):
    sequence = sequence.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    smokehouse_id, batch_id, layer_id = key

    timestamp_diff = (
        sequence[
            "Timestamp"
        ]
        .diff()
        .dropna()
        .dt
        .total_seconds()
        / 60
    )

    non_positive_intervals = int(
        (
            timestamp_diff
            <= 0
        ).sum()
    )

    sequence_timestamp_order_errors += (
        non_positive_intervals
    )

    seq_missing = (
        sequence[
            [
                "Temperature_C",
                "Humidity_%",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
        .tolist()
    )

    current_run = 0
    longest_run = 0

    for is_missing in seq_missing:
        if is_missing:
            current_run += 1
            longest_run = max(
                longest_run,
                current_run,
            )
        else:
            current_run = 0

    max_consecutive_sensor_missing = max(
        max_consecutive_sensor_missing,
        longest_run,
    )

    sequence_quality_rows.append({
        "Smokehouse_ID": smokehouse_id,
        "Batch_ID": batch_id,
        "Layer_ID": layer_id,
        "Records": len(sequence),
        "Start_Timestamp": (
            sequence[
                "Timestamp"
            ].min()
        ),
        "End_Timestamp": (
            sequence[
                "Timestamp"
            ].max()
        ),
        "Sensor_Missing_Rows": int(
            sequence[
                [
                    "Temperature_C",
                    "Humidity_%",
                ]
            ]
            .isna()
            .any(
                axis=1
            )
            .sum()
        ),
        "Ten_Minute_Intervals": int(
            (
                timestamp_diff
                == 10
            ).sum()
        ),
        "Twenty_Minute_Intervals": int(
            (
                timestamp_diff
                == 20
            ).sum()
        ),
        "Unexpected_Intervals": int(
            (
                ~timestamp_diff.isin(
                    [
                        10,
                        20,
                    ]
                )
            ).sum()
        ),
        "Non_Positive_Intervals": (
            non_positive_intervals
        ),
        "Longest_Consecutive_Sensor_Missing_Run": (
            longest_run
        ),
    })

sequence_quality = pd.DataFrame(
    sequence_quality_rows
)

sequence_quality.to_csv(
    SEQUENCE_QUALITY_PATH,
    index=False,
)

sequence_count = len(
    sequence_quality
)


# ============================================================
# 18. BATCH / LAYER STRUCTURAL CONSISTENCY
# ============================================================

batch_structure = (
    quality_df.groupby(
        [
            "Smokehouse_ID",
            "Batch_ID",
        ],
        dropna=False,
    )
    .agg(
        Layer_Count=(
            "Layer_ID",
            "nunique",
        ),
    )
    .reset_index()
)

batches_not_three_layers = int(
    (
        batch_structure[
            "Layer_Count"
        ]
        != 3
    ).sum()
)

batch_count = int(
    batch_structure.shape[0]
)

smokehouse_batch_counts = (
    batch_structure
    .groupby(
        "Smokehouse_ID"
    )[
        "Batch_ID"
    ]
    .nunique()
)

smokehouses_not_six_batches = int(
    (
        smokehouse_batch_counts
        != 6
    ).sum()
)


# ============================================================
# 19. LAYER AREA CONSISTENCY
# ============================================================

area_unique_by_smokehouse = (
    quality_df.groupby(
        "Smokehouse_ID"
    )[
        "Layer_Area_m2"
    ]
    .nunique(
        dropna=False
    )
)

smokehouse_area_inconsistencies = int(
    (
        area_unique_by_smokehouse
        != 1
    ).sum()
)


# ============================================================
# 20. PASS / FAIL QUALITY CHECKS
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
        "Check": "Expected row count is preserved",
        "Status": (
            "PASS"
            if row_count_matches
            else "FAIL"
        ),
        "Details": (
            f"Rows={len(df):,}"
        ),
    },
    {
        "Check": "Expected 17-column schema is preserved",
        "Status": (
            "PASS"
            if schema_matches
            else "FAIL"
        ),
        "Details": (
            f"Columns={len(df.columns)}"
        ),
    },
    {
        "Check": "Mandatory identifiers/targets/metadata are complete",
        "Status": (
            "PASS"
            if mandatory_missing_count == 0
            else "FAIL"
        ),
        "Details": (
            f"Missing mandatory cells="
            f"{mandatory_missing_count}"
        ),
    },
    {
        "Check": "No exact duplicate rows",
        "Status": (
            "PASS"
            if exact_duplicate_rows == 0
            else "FAIL"
        ),
        "Details": (
            f"Duplicate-row members="
            f"{exact_duplicate_rows}"
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
        "Check": "Sensor missingness is low and documented",
        "Status": (
            "PASS"
            if (
                rows_with_sensor_missing
                == 24
                and
                sensor_missing_percent
                < 1.0
            )
            else "FAIL"
        ),
        "Details": (
            f"Rows={rows_with_sensor_missing}; "
            f"Percent={sensor_missing_percent:.4f}%"
        ),
    },
    {
        "Check": "Sensor/derived-field missingness is logically consistent",
        "Status": (
            "PASS"
            if derived_missing_mismatches == 0
            else "FAIL"
        ),
        "Details": (
            f"Mismatches="
            f"{derived_missing_mismatches}"
        ),
    },
    {
        "Check": "Sensor missing events are isolated",
        "Status": (
            "PASS"
            if max_consecutive_sensor_missing <= 1
            else "FAIL"
        ),
        "Details": (
            f"Longest consecutive missing run="
            f"{max_consecutive_sensor_missing}"
        ),
    },
    {
        "Check": "Missing sensor events are not packet-loss endpoints",
        "Status": (
            "PASS"
            if missing_at_gap_endpoints == 0
            else "FAIL"
        ),
        "Details": (
            f"Overlap rows="
            f"{missing_at_gap_endpoints}"
        ),
    },
    {
        "Check": "Timestamp order is strictly increasing by sequence",
        "Status": (
            "PASS"
            if sequence_timestamp_order_errors == 0
            else "FAIL"
        ),
        "Details": (
            f"Non-positive intervals="
            f"{sequence_timestamp_order_errors}"
        ),
    },
    {
        "Check": "Only expected 10/20-minute sampling intervals occur",
        "Status": (
            "PASS"
            if unexpected_interval_count == 0
            else "FAIL"
        ),
        "Details": (
            f"10-min={ten_minute_intervals}; "
            f"20-min={twenty_minute_intervals}; "
            f"other={unexpected_interval_count}"
        ),
    },
    {
        "Check": "Expected packet-loss count is preserved",
        "Status": (
            "PASS"
            if (
                twenty_minute_intervals == 12
                and
                estimated_missed_packets == 12
            )
            else "FAIL"
        ),
        "Details": (
            f"20-min gaps="
            f"{twenty_minute_intervals}; "
            f"estimated missed packets="
            f"{estimated_missed_packets}"
        ),
    },
    {
        "Check": "Expected 36 Batch-Layer sequences exist",
        "Status": (
            "PASS"
            if sequence_count == 36
            else "FAIL"
        ),
        "Details": (
            f"Sequences="
            f"{sequence_count}"
        ),
    },
    {
        "Check": "Each batch contains all 3 layers",
        "Status": (
            "PASS"
            if batches_not_three_layers == 0
            else "FAIL"
        ),
        "Details": (
            f"Violating batches="
            f"{batches_not_three_layers}"
        ),
    },
    {
        "Check": "Each smokehouse contains 6 batches",
        "Status": (
            "PASS"
            if (
                batch_count == 12
                and
                smokehouses_not_six_batches == 0
            )
            else "FAIL"
        ),
        "Details": (
            f"Total batches="
            f"{batch_count}; "
            f"violating smokehouses="
            f"{smokehouses_not_six_batches}"
        ),
    },
    {
        "Check": "Categorical values match expected domains",
        "Status": (
            "PASS"
            if categorical_errors == 0
            else "FAIL"
        ),
        "Details": (
            f"Invalid categorical domains="
            f"{categorical_errors}"
        ),
    },
    {
        "Check": "Numeric values pass broad physical/domain plausibility",
        "Status": (
            "PASS"
            if total_numeric_plausibility_errors == 0
            else "FAIL"
        ),
        "Details": (
            f"Impossible/out-of-domain values="
            f"{total_numeric_plausibility_errors}"
        ),
    },
    {
        "Check": "Layer area is internally consistent per smokehouse",
        "Status": (
            "PASS"
            if smokehouse_area_inconsistencies == 0
            else "FAIL"
        ),
        "Details": (
            f"Violating smokehouses="
            f"{smokehouse_area_inconsistencies}"
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

failed_checks = checks_df[
    checks_df[
        "Status"
    ]
    == "FAIL"
]

overall_status = (
    "PASS"
    if len(
        failed_checks
    ) == 0
    else "FAIL"
)


# ============================================================
# 21. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- DATASET IDENTITY ---"
)

print(
    f"SHA-256                       : "
    f"{dataset_sha256}"
)

print(
    f"Frozen hash matches           : "
    f"{hash_matches}"
)

print(
    f"Rows / columns                : "
    f"{len(df):,} / {len(df.columns)}"
)


print(
    "\n--- MISSING DATA ---"
)

print(
    f"Rows with any sensor missing  : "
    f"{rows_with_sensor_missing} "
    f"({sensor_missing_percent:.4f}%)"
)

print(
    f"Temperature only missing      : "
    f"{temperature_only_missing}"
)

print(
    f"Humidity only missing         : "
    f"{humidity_only_missing}"
)

print(
    f"Both Temp + RH missing        : "
    f"{both_sensor_missing}"
)

print(
    f"Temperature_C missing         : "
    f"{int(temp_missing.sum())}"
)

print(
    f"Humidity_% missing            : "
    f"{int(humidity_missing.sum())}"
)

print(
    f"Derived missing mismatches    : "
    f"{derived_missing_mismatches}"
)

print(
    f"Longest consecutive miss run  : "
    f"{max_consecutive_sensor_missing}"
)

print(
    f"Batches with missing sensors  : "
    f"{batches_with_any_sensor_missing}/12"
)


print(
    "\n--- DUPLICATES / STRUCTURE ---"
)

print(
    f"Exact duplicate row members   : "
    f"{exact_duplicate_rows}"
)

print(
    f"Duplicate observation keys    : "
    f"{duplicate_observation_keys}"
)

print(
    f"Batch-Layer sequences         : "
    f"{sequence_count}"
)

print(
    f"Batches not containing 3 layers: "
    f"{batches_not_three_layers}"
)


print(
    "\n--- SAMPLING / PACKET LOSS ---"
)

print(
    f"10-minute intervals           : "
    f"{ten_minute_intervals:,}"
)

print(
    f"20-minute intervals           : "
    f"{twenty_minute_intervals}"
)

print(
    f"Estimated missed packets      : "
    f"{estimated_missed_packets}"
)

print(
    f"Other interval lengths        : "
    f"{unexpected_interval_count}"
)

print(
    f"Missing/gap endpoint overlaps : "
    f"{missing_at_gap_endpoints}"
)


print(
    "\n--- OBSERVED NUMERIC RANGES ---"
)

print(
    numeric_quality[
        [
            "Column",
            "Observed_Min",
            "Observed_Max",
            "Invalid_Count",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


print(
    "\n--- IQR STATISTICAL EXTREMES (REVIEW ONLY) ---"
)

print(
    iqr_review[
        [
            "Column",
            "IQR_Outlier_Count_For_Review",
        ]
    ]
    .to_string(
        index=False
    )
)

print(
    "\nNOTE: IQR extremes are NOT automatically errors "
    "and nothing is removed in Stage 03."
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
        "COMPONENT 3 - STAGE 03: DATA QUALITY ANALYSIS\n"
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
        f"Frozen hash match: "
        f"{hash_matches}\n"
    )

    report.write(
        f"Rows: "
        f"{len(df):,}\n"
    )

    report.write(
        f"Columns: "
        f"{len(df.columns)}\n\n"
    )

    report.write(
        "MISSING-DATA QUALITY\n"
        "--------------------\n"
    )

    report.write(
        f"Rows with any sensor missing: "
        f"{rows_with_sensor_missing} "
        f"({sensor_missing_percent:.4f}%)\n"
    )

    report.write(
        f"Temperature only missing: "
        f"{temperature_only_missing}\n"
    )

    report.write(
        f"Humidity only missing: "
        f"{humidity_only_missing}\n"
    )

    report.write(
        f"Both temperature and humidity missing: "
        f"{both_sensor_missing}\n"
    )

    report.write(
        f"Temperature_C missing: "
        f"{int(temp_missing.sum())}\n"
    )

    report.write(
        f"Humidity_% missing: "
        f"{int(humidity_missing.sum())}\n"
    )

    report.write(
        f"Derived-field missingness mismatches: "
        f"{derived_missing_mismatches}\n"
    )

    report.write(
        f"Longest consecutive sensor-missing run: "
        f"{max_consecutive_sensor_missing}\n"
    )

    report.write(
        f"Batches containing at least one "
        f"missing sensor row: "
        f"{batches_with_any_sensor_missing}/12\n\n"
    )

    report.write(
        "DUPLICATES / STRUCTURE\n"
        "----------------------\n"
    )

    report.write(
        f"Exact duplicate row members: "
        f"{exact_duplicate_rows}\n"
    )

    report.write(
        f"Duplicate observation keys: "
        f"{duplicate_observation_keys}\n"
    )

    report.write(
        f"Batch-Layer sequences: "
        f"{sequence_count}\n"
    )

    report.write(
        f"Batches without exactly 3 layers: "
        f"{batches_not_three_layers}\n"
    )

    report.write(
        f"Smokehouses without exactly 6 batches: "
        f"{smokehouses_not_six_batches}\n\n"
    )

    report.write(
        "SAMPLING / PACKET LOSS\n"
        "----------------------\n"
    )

    report.write(
        f"10-minute intervals: "
        f"{ten_minute_intervals:,}\n"
    )

    report.write(
        f"20-minute intervals: "
        f"{twenty_minute_intervals}\n"
    )

    report.write(
        f"Estimated missed 10-minute packets: "
        f"{estimated_missed_packets}\n"
    )

    report.write(
        f"Other interval lengths: "
        f"{unexpected_interval_count}\n"
    )

    report.write(
        f"Missing sensor rows located directly at "
        f"20-minute gap endpoints: "
        f"{missing_at_gap_endpoints}\n\n"
    )

    report.write(
        "NUMERIC QUALITY\n"
        "---------------\n"
    )

    report.write(
        numeric_quality
        .round(4)
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "IQR OUTLIER REVIEW\n"
        "------------------\n"
    )

    report.write(
        iqr_review
        .round(4)
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\nIMPORTANT: IQR outliers are statistical "
        "extremes only. They are not automatically data errors "
        "and no rows are removed in Stage 03.\n\n"
    )

    report.write(
        "CATEGORICAL QUALITY\n"
        "-------------------\n"
    )

    report.write(
        categorical_quality
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "PREPROCESSING IMPLICATIONS FOR STAGE 04\n"
        "---------------------------------------\n"
    )

    report.write(
        "- Preserve the frozen raw CSV unchanged.\n"
    )

    report.write(
        "- Handle the very small amount of sensor "
        "missingness explicitly in a derived dataset.\n"
    )

    report.write(
        "- Preserve the 12 real 20-minute packet-loss gaps; "
        "do not invent raw observations.\n"
    )

    report.write(
        "- Timestamp-aware preprocessing must respect actual "
        "time differences when computing temporal features.\n"
    )

    report.write(
        "- Statistical IQR extremes should not be blindly "
        "deleted because high/low temperature and humidity "
        "conditions may carry meaningful drying/fire-risk "
        "information.\n"
    )

    report.write(
        "- Actual_Total_Drying_Time, "
        "Fire_Risk_Assessment_Timestamp and Batch_ID remain "
        "excluded from physical model predictors according "
        "to the Stage 02 leakage rules.\n"
    )

    report.write(
        "- Any transformation learned from data must later "
        "be fitted on training data only after the "
        "batch-aware split.\n\n"
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
        f"STAGE 03 STATUS: "
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
    f"STAGE 03 DATA QUALITY ANALYSIS: "
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
        "Stage 03 data quality analysis failed. "
        "Review 03_data_quality_report.txt before continuing."
    )
