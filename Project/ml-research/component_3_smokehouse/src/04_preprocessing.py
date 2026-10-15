from pathlib import Path
import hashlib
import pandas as pd
import numpy as np


# ============================================================
# COMPONENT 3 - STAGE 04: PREPROCESSING
# ============================================================
#
# Purpose:
#   Create a clean DERIVED dataset for EDA and later feature
#   engineering while preserving the frozen raw dataset.
#
# Missing-sensor strategy:
#   - Missing sensor events are extremely rare and isolated.
#   - Use LAST OBSERVATION CARRIED FORWARD (LOCF) only inside
#     the same Smokehouse x Batch x Layer sequence.
#   - The imputation uses PAST information only.
#   - NO future reading is used.
#
# Packet-loss strategy:
#   - Preserve the 12 genuine 20-minute gaps.
#   - DO NOT create synthetic raw sensor rows.
#
# Leakage strategy:
#   - NO scaling.
#   - NO one-hot encoding.
#   - NO global-statistic imputation.
#   - NO model fitting.
#   - NO train/test split.
#
# Those learned transformations belong after the batch-aware
# split in Stage 10.
#
# Raw-data rule:
#   Component3_Dataset.csv is NEVER modified.
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Component3_Dataset.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

VALIDATED_DIR = (
    PROJECT_ROOT
    / "data"
    / "validated"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VALIDATED_DIR.mkdir(
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

PROCESSED_DATA_PATH = (
    PROCESSED_DIR
    / "Component3_Dataset_Preprocessed.csv"
)

VALIDATED_DATA_PATH = (
    VALIDATED_DIR
    / "Component3_Dataset_Validated.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "04_preprocessing_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "04_preprocessing_checks.csv"
)

IMPUTATION_AUDIT_PATH = (
    REPORT_DIR
    / "04_imputation_audit.csv"
)

MISSING_BEFORE_AFTER_PATH = (
    REPORT_DIR
    / "04_missing_before_after.csv"
)


# ============================================================
# 3. FINAL DATASET IDENTITY
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

SEQUENCE_KEY = [
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
]

OBSERVATION_KEY = (
    SEQUENCE_KEY
    + ["Timestamp"]
)

TEMP_TREND_THRESHOLD = 0.35
HUMIDITY_TREND_THRESHOLD = 0.80


# ============================================================
# 4. LOAD RAW DATASET
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 04: PREPROCESSING")
print("=" * 78)

if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nFinal raw dataset not found:\n{RAW_DATA_PATH}\n\n"
        "Expected location:\n"
        "ml-research/component_3_smokehouse/data/raw/"
        "Component3_Dataset.csv"
    )

with open(RAW_DATA_PATH, "rb") as file:
    raw_sha256 = hashlib.sha256(
        file.read()
    ).hexdigest()

raw_df = pd.read_csv(
    RAW_DATA_PATH
)

if raw_sha256 != EXPECTED_SHA256:
    raise RuntimeError(
        "The raw dataset SHA-256 does not match the frozen "
        "final dataset. Stop preprocessing and verify the file."
    )

if len(raw_df) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Unexpected raw row count: {len(raw_df):,}"
    )

if list(raw_df.columns) != EXPECTED_COLUMNS:
    raise RuntimeError(
        "Unexpected raw schema. Stage 04 will not continue."
    )

print("\nFrozen raw dataset verified successfully.")


# ============================================================
# 5. CREATE WORKING COPY
# ============================================================

df = raw_df.copy()

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
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df = df.sort_values(
    SEQUENCE_KEY
    + ["Timestamp"]
).reset_index(drop=True)


# ============================================================
# 6. RECORD MISSINGNESS BEFORE PREPROCESSING
# ============================================================

missing_before = {
    column: int(
        df[column]
        .isna()
        .sum()
    )
    for column in EXPECTED_COLUMNS
}

rows_with_sensor_missing_before = int(
    df[
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
)


# ============================================================
# 6B. SAVE STRUCTURALLY VALIDATED (PRE-IMPUTATION) SNAPSHOT
# ============================================================
#
# This snapshot captures the dataset exactly at the point where:
#   - the frozen raw SHA-256/row-count/schema have been verified,
#   - Timestamp/Fire_Risk_Assessment_Timestamp/numeric columns have
#     been type-parsed,
#   - rows have been chronologically sorted by sequence + Timestamp,
#   - initial missingness has been recorded,
#
# but BEFORE any LOCF sensor imputation or derived-field (Change/
# Trend) reconstruction occurs. It therefore still contains the
# legitimate pre-imputation missing sensor values and is a pure
# traceability artifact: it is never read by Stage 07+ and must not
# be used as a model input.

validated_df = df[
    EXPECTED_COLUMNS
].copy()

validated_df.to_csv(
    VALIDATED_DATA_PATH,
    index=False,
    date_format="%Y-%m-%d %H:%M:%S",
)

with open(
    VALIDATED_DATA_PATH,
    "rb",
) as file:
    validated_sha256 = hashlib.sha256(
        file.read()
    ).hexdigest()

validated_missing_cells = int(
    validated_df.isna().sum().sum()
)

print(
    "\n--- VALIDATED (PRE-IMPUTATION) SNAPSHOT ---"
)

print(
    f"Validated rows / columns       : "
    f"{len(validated_df):,} / {len(validated_df.columns)}"
)

print(
    f"Validated missing cells        : "
    f"{validated_missing_cells}"
)

print(
    f"Validated SHA-256              : "
    f"{validated_sha256}"
)

print(
    f"Saved to:\n{VALIDATED_DATA_PATH}"
)


# ============================================================
# 7. PAST-ONLY SENSOR IMPUTATION
# ============================================================
#
# For each sequence, a missing sensor reading is replaced with
# the most recent earlier valid reading from the SAME sequence.
#
# This is causal and deployment-compatible:
# at prediction time the system can know the previous reading,
# but it cannot know a future reading.

audit_rows = []

for key, indices in df.groupby(
    SEQUENCE_KEY,
    sort=False,
).groups.items():

    sorted_indices = sorted(
        indices,
        key=lambda idx: df.at[
            idx,
            "Timestamp",
        ],
    )

    for sensor_column in [
        "Temperature_C",
        "Humidity_%",
    ]:

        last_valid_value = None
        last_valid_timestamp = None

        for idx in sorted_indices:
            current_value = df.at[
                idx,
                sensor_column,
            ]

            current_timestamp = df.at[
                idx,
                "Timestamp",
            ]

            if pd.isna(
                current_value
            ):
                if (
                    last_valid_value is None
                    or
                    last_valid_timestamp is None
                ):
                    raise RuntimeError(
                        "Past-only imputation is impossible because "
                        f"{sensor_column} is missing before any valid "
                        f"reading in sequence {key}."
                    )

                minutes_since_source = (
                    current_timestamp
                    -
                    last_valid_timestamp
                ).total_seconds() / 60

                original_value = current_value

                df.at[
                    idx,
                    sensor_column,
                ] = last_valid_value

                audit_rows.append({
                    "Smokehouse_ID": key[0],
                    "Batch_ID": key[1],
                    "Layer_ID": key[2],
                    "Timestamp": current_timestamp,
                    "Sensor_Column": sensor_column,
                    "Original_Value": original_value,
                    "Imputed_Value": last_valid_value,
                    "Source_Timestamp": last_valid_timestamp,
                    "Minutes_Since_Source": minutes_since_source,
                    "Method": "Past-only LOCF within sequence",
                })

            else:
                last_valid_value = float(
                    current_value
                )
                last_valid_timestamp = (
                    current_timestamp
                )

imputation_audit = pd.DataFrame(
    audit_rows
)

imputation_audit.to_csv(
    IMPUTATION_AUDIT_PATH,
    index=False,
)

temperature_imputations = int(
    (
        imputation_audit[
            "Sensor_Column"
        ]
        == "Temperature_C"
    ).sum()
)

humidity_imputations = int(
    (
        imputation_audit[
            "Sensor_Column"
        ]
        == "Humidity_%"
    ).sum()
)

max_source_age_minutes = float(
    imputation_audit[
        "Minutes_Since_Source"
    ].max()
)


# ============================================================
# 8. FILL ONLY THE DERIVED VALUES THAT WERE MISSING
# ============================================================
#
# Existing non-missing change/trend values are preserved.
# Only values that were missing because their source sensor
# reading was missing are reconstructed.
#
# No future observation is used.

def classify_trend(
    delta,
    positive_threshold,
):
    if delta >= positive_threshold:
        return "Increasing"
    elif delta <= -positive_threshold:
        return "Decreasing"
    else:
        return "Stable"


def get_past_trend_reference(
    sequence,
    current_position,
    value_column,
):
    current_time = sequence.loc[
        current_position,
        "Timestamp",
    ]

    candidates = []

    for previous_position in range(
        current_position - 1,
        -1,
        -1,
    ):
        previous_time = sequence.loc[
            previous_position,
            "Timestamp",
        ]

        minutes_back = (
            current_time
            -
            previous_time
        ).total_seconds() / 60

        if minutes_back > 40:
            break

        if (
            minutes_back >= 20
            and
            minutes_back <= 40
        ):
            candidates.append(
                (
                    abs(
                        minutes_back
                        - 30
                    ),
                    float(
                        sequence.loc[
                            previous_position,
                            value_column,
                        ]
                    ),
                )
            )

    if candidates:
        candidates.sort(
            key=lambda item: item[0]
        )
        return candidates[0][1]

    if current_position > 0:
        return float(
            sequence.loc[
                current_position - 1,
                value_column,
            ]
        )

    return None


for key, sequence_indices in df.groupby(
    SEQUENCE_KEY,
    sort=False,
).groups.items():

    sequence_indices = sorted(
        sequence_indices,
        key=lambda idx: df.at[
            idx,
            "Timestamp",
        ],
    )

    sequence = (
        df.loc[
            sequence_indices
        ]
        .copy()
        .reset_index()
        .rename(
            columns={
                "index": "_Original_Index"
            }
        )
    )

    for position in range(
        len(sequence)
    ):
        original_idx = int(
            sequence.loc[
                position,
                "_Original_Index",
            ]
        )

        # ---------------- Temperature change ----------------

        if pd.isna(
            df.at[
                original_idx,
                "Temperature_Change",
            ]
        ):
            if position == 0:
                temp_change = 0.0
            else:
                temp_change = (
                    float(
                        sequence.loc[
                            position,
                            "Temperature_C",
                        ]
                    )
                    -
                    float(
                        sequence.loc[
                            position - 1,
                            "Temperature_C",
                        ]
                    )
                )

            df.at[
                original_idx,
                "Temperature_Change",
            ] = round(
                temp_change,
                2,
            )

        # ---------------- Temperature trend ----------------

        if pd.isna(
            df.at[
                original_idx,
                "Temperature_Trend",
            ]
        ):
            reference = get_past_trend_reference(
                sequence,
                position,
                "Temperature_C",
            )

            if reference is None:
                temp_trend = "Stable"
            else:
                delta = (
                    float(
                        sequence.loc[
                            position,
                            "Temperature_C",
                        ]
                    )
                    -
                    reference
                )

                temp_trend = classify_trend(
                    delta,
                    TEMP_TREND_THRESHOLD,
                )

            df.at[
                original_idx,
                "Temperature_Trend",
            ] = temp_trend

        # ---------------- Humidity change ----------------

        if pd.isna(
            df.at[
                original_idx,
                "Humidity_Change",
            ]
        ):
            if position == 0:
                humidity_change = 0.0
            else:
                humidity_change = (
                    float(
                        sequence.loc[
                            position,
                            "Humidity_%",
                        ]
                    )
                    -
                    float(
                        sequence.loc[
                            position - 1,
                            "Humidity_%",
                        ]
                    )
                )

            df.at[
                original_idx,
                "Humidity_Change",
            ] = round(
                humidity_change,
                2,
            )

        # ---------------- Humidity trend ----------------

        if pd.isna(
            df.at[
                original_idx,
                "Humidity_Trend",
            ]
        ):
            reference = get_past_trend_reference(
                sequence,
                position,
                "Humidity_%",
            )

            if reference is None:
                humidity_trend = "Stable"
            else:
                delta = (
                    float(
                        sequence.loc[
                            position,
                            "Humidity_%",
                        ]
                    )
                    -
                    reference
                )

                humidity_trend = classify_trend(
                    delta,
                    HUMIDITY_TREND_THRESHOLD,
                )

            df.at[
                original_idx,
                "Humidity_Trend",
            ] = humidity_trend


# ============================================================
# 9. MISSINGNESS AFTER PREPROCESSING
# ============================================================

missing_after = {
    column: int(
        df[column]
        .isna()
        .sum()
    )
    for column in EXPECTED_COLUMNS
}

missing_before_after = pd.DataFrame({
    "Column": EXPECTED_COLUMNS,
    "Missing_Before": [
        missing_before[column]
        for column in EXPECTED_COLUMNS
    ],
    "Missing_After": [
        missing_after[column]
        for column in EXPECTED_COLUMNS
    ],
})

missing_before_after.to_csv(
    MISSING_BEFORE_AFTER_PATH,
    index=False,
)

remaining_missing_cells = int(
    df[
        EXPECTED_COLUMNS
    ]
    .isna()
    .sum()
    .sum()
)


# ============================================================
# 10. VERIFY OBSERVED SENSOR VALUES WERE NOT CHANGED
# ============================================================

raw_compare = raw_df.copy()

raw_compare["Timestamp"] = pd.to_datetime(
    raw_compare["Timestamp"],
    errors="raise",
)

raw_compare = raw_compare.sort_values(
    SEQUENCE_KEY
    + ["Timestamp"]
).reset_index(drop=True)

observed_temperature_mask = (
    raw_compare[
        "Temperature_C"
    ].notna()
)

observed_humidity_mask = (
    raw_compare[
        "Humidity_%"
    ].notna()
)

changed_observed_temperature = int(
    (
        pd.to_numeric(
            raw_compare.loc[
                observed_temperature_mask,
                "Temperature_C",
            ],
            errors="coerce",
        ).to_numpy()
        !=
        df.loc[
            observed_temperature_mask,
            "Temperature_C",
        ].to_numpy()
    ).sum()
)

changed_observed_humidity = int(
    (
        pd.to_numeric(
            raw_compare.loc[
                observed_humidity_mask,
                "Humidity_%",
            ],
            errors="coerce",
        ).to_numpy()
        !=
        df.loc[
            observed_humidity_mask,
            "Humidity_%",
        ].to_numpy()
    ).sum()
)


# ============================================================
# 11. VERIFY CRITICAL FIELDS ARE UNCHANGED
# ============================================================

preserved_columns = [
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

critical_field_changes = 0

for column in preserved_columns:

    if column in [
        "Timestamp",
        "Fire_Risk_Assessment_Timestamp",
    ]:
        original_series = pd.to_datetime(
            raw_compare[
                column
            ],
            errors="raise",
        )

        processed_series = pd.to_datetime(
            df[
                column
            ],
            errors="raise",
        )

    elif column in [
        "Elapsed_Drying_Hours",
        "Number_of_Sheets",
        "Layer_Area_m2",
        "Actual_Total_Drying_Time",
        "Remaining_Drying_Time",
    ]:
        original_series = pd.to_numeric(
            raw_compare[
                column
            ],
            errors="coerce",
        )

        processed_series = pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )

    else:
        original_series = (
            raw_compare[
                column
            ]
            .astype(str)
        )

        processed_series = (
            df[
                column
            ]
            .astype(str)
        )

    critical_field_changes += int(
        (
            original_series
            !=
            processed_series
        ).sum()
    )


# ============================================================
# 12. VERIFY SAMPLING GAPS ARE PRESERVED
# ============================================================

df[
    "_Previous_Timestamp"
] = (
    df.groupby(
        SEQUENCE_KEY
    )[
        "Timestamp"
    ]
    .shift(1)
)

df[
    "_Interval_Minutes"
] = (
    (
        df[
            "Timestamp"
        ]
        -
        df[
            "_Previous_Timestamp"
        ]
    )
    .dt
    .total_seconds()
    / 60
)

interval_counts = (
    df[
        "_Interval_Minutes"
    ]
    .dropna()
    .value_counts()
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

other_interval_count = int(
    interval_counts[
        ~interval_counts.index.isin(
            [
                10.0,
                20.0,
            ]
        )
    ].sum()
)

df = df.drop(
    columns=[
        "_Previous_Timestamp",
        "_Interval_Minutes",
    ]
)


# ============================================================
# 13. DUPLICATE / TARGET / CLASS CHECKS
# ============================================================

exact_duplicate_members = int(
    df.duplicated(
        keep=False
    ).sum()
)

duplicate_observation_keys = int(
    df.duplicated(
        subset=OBSERVATION_KEY,
        keep="first",
    ).sum()
)

regression_target_missing = int(
    df[
        "Remaining_Drying_Time"
    ]
    .isna()
    .sum()
)

classification_target_missing = int(
    df[
        "Fire_Risk_Label"
    ]
    .isna()
    .sum()
)

fire_distribution = (
    df[
        "Fire_Risk_Label"
    ]
    .value_counts()
    .to_dict()
)


# ============================================================
# 14. SAVE PREPROCESSED DATASET
# ============================================================

# Retain the same 17-column research schema.
df = df[
    EXPECTED_COLUMNS
].copy()

df.to_csv(
    PROCESSED_DATA_PATH,
    index=False,
    date_format="%Y-%m-%d %H:%M:%S",
)

with open(
    PROCESSED_DATA_PATH,
    "rb",
) as file:
    processed_sha256 = hashlib.sha256(
        file.read()
    ).hexdigest()


# ============================================================
# 15. PASS / FAIL CHECKS
# ============================================================

checks = [
    {
        "Check": "Frozen raw SHA-256 matches",
        "Status": (
            "PASS"
            if raw_sha256 == EXPECTED_SHA256
            else "FAIL"
        ),
        "Details": raw_sha256,
    },
    {
        "Check": "Row count preserved",
        "Status": (
            "PASS"
            if len(df) == EXPECTED_ROWS
            else "FAIL"
        ),
        "Details": (
            f"Rows={len(df):,}"
        ),
    },
    {
        "Check": "17-column schema preserved",
        "Status": (
            "PASS"
            if list(df.columns) == EXPECTED_COLUMNS
            else "FAIL"
        ),
        "Details": (
            f"Columns={len(df.columns)}"
        ),
    },
    {
        "Check": "All sensor missing values were causally imputed",
        "Status": (
            "PASS"
            if (
                temperature_imputations == 16
                and
                humidity_imputations == 16
                and
                df[
                    [
                        "Temperature_C",
                        "Humidity_%",
                    ]
                ]
                .isna()
                .sum()
                .sum()
                == 0
            )
            else "FAIL"
        ),
        "Details": (
            f"Temperature imputations="
            f"{temperature_imputations}; "
            f"Humidity imputations="
            f"{humidity_imputations}"
        ),
    },
    {
        "Check": "Imputation used only recent past readings",
        "Status": (
            "PASS"
            if max_source_age_minutes <= 10.0
            else "FAIL"
        ),
        "Details": (
            f"Maximum source age="
            f"{max_source_age_minutes:.1f} min"
        ),
    },
    {
        "Check": "All preprocessing-stage missing values are resolved",
        "Status": (
            "PASS"
            if remaining_missing_cells == 0
            else "FAIL"
        ),
        "Details": (
            f"Remaining missing cells="
            f"{remaining_missing_cells}"
        ),
    },
    {
        "Check": "Originally observed temperature values were preserved",
        "Status": (
            "PASS"
            if changed_observed_temperature == 0
            else "FAIL"
        ),
        "Details": (
            f"Changed observed values="
            f"{changed_observed_temperature}"
        ),
    },
    {
        "Check": "Originally observed humidity values were preserved",
        "Status": (
            "PASS"
            if changed_observed_humidity == 0
            else "FAIL"
        ),
        "Details": (
            f"Changed observed values="
            f"{changed_observed_humidity}"
        ),
    },
    {
        "Check": "Identifiers, targets and metadata were preserved",
        "Status": (
            "PASS"
            if critical_field_changes == 0
            else "FAIL"
        ),
        "Details": (
            f"Changed cells="
            f"{critical_field_changes}"
        ),
    },
    {
        "Check": "No duplicate rows introduced",
        "Status": (
            "PASS"
            if exact_duplicate_members == 0
            else "FAIL"
        ),
        "Details": (
            f"Duplicate members="
            f"{exact_duplicate_members}"
        ),
    },
    {
        "Check": "No duplicate observation keys introduced",
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
        "Check": "Original packet-loss gaps were preserved",
        "Status": (
            "PASS"
            if (
                ten_minute_intervals == 24415
                and
                twenty_minute_intervals == 12
                and
                other_interval_count == 0
            )
            else "FAIL"
        ),
        "Details": (
            f"10-min={ten_minute_intervals}; "
            f"20-min={twenty_minute_intervals}; "
            f"other={other_interval_count}"
        ),
    },
    {
        "Check": "Regression target remains complete",
        "Status": (
            "PASS"
            if regression_target_missing == 0
            else "FAIL"
        ),
        "Details": (
            f"Missing="
            f"{regression_target_missing}"
        ),
    },
    {
        "Check": "Classification target remains complete",
        "Status": (
            "PASS"
            if classification_target_missing == 0
            else "FAIL"
        ),
        "Details": (
            f"Missing="
            f"{classification_target_missing}"
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
# 16. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- RAW DATASET ---"
)

print(
    f"Rows / columns                 : "
    f"{len(raw_df):,} / {len(raw_df.columns)}"
)

print(
    f"Frozen SHA-256 verified        : "
    f"{raw_sha256 == EXPECTED_SHA256}"
)


print(
    "\n--- SENSOR IMPUTATION ---"
)

print(
    f"Rows with sensor missing before: "
    f"{rows_with_sensor_missing_before}"
)

print(
    f"Temperature values imputed     : "
    f"{temperature_imputations}"
)

print(
    f"Humidity values imputed        : "
    f"{humidity_imputations}"
)

print(
    f"Maximum source age             : "
    f"{max_source_age_minutes:.1f} minutes"
)

print(
    f"Remaining missing cells        : "
    f"{remaining_missing_cells}"
)

print(
    f"Observed temperature changed   : "
    f"{changed_observed_temperature}"
)

print(
    f"Observed humidity changed      : "
    f"{changed_observed_humidity}"
)


print(
    "\n--- PRESERVATION CHECKS ---"
)

print(
    f"Identifiers/targets changes    : "
    f"{critical_field_changes}"
)

print(
    f"Exact duplicate row members    : "
    f"{exact_duplicate_members}"
)

print(
    f"Duplicate observation keys     : "
    f"{duplicate_observation_keys}"
)

print(
    f"10-minute intervals            : "
    f"{ten_minute_intervals:,}"
)

print(
    f"20-minute packet-loss gaps     : "
    f"{twenty_minute_intervals}"
)

print(
    f"Other interval lengths         : "
    f"{other_interval_count}"
)


print(
    "\n--- TARGET PRESERVATION ---"
)

print(
    f"Regression target missing      : "
    f"{regression_target_missing}"
)

print(
    f"Classification target missing  : "
    f"{classification_target_missing}"
)

print(
    f"Fire-risk distribution         : "
    f"{fire_distribution}"
)


print(
    "\n--- OUTPUT ---"
)

print(
    f"Processed dataset:\n"
    f"{PROCESSED_DATA_PATH}"
)

print(
    f"\nProcessed SHA-256:\n"
    f"{processed_sha256}"
)


# ============================================================
# 17. HUMAN-READABLE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 04: PREPROCESSING\n"
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
        "RAW DATASET PRESERVATION\n"
        "------------------------\n"
    )

    report.write(
        "Raw dataset: "
        "Component3_Dataset.csv\n"
    )

    report.write(
        f"Raw SHA-256: "
        f"{raw_sha256}\n"
    )

    report.write(
        f"Raw rows: "
        f"{len(raw_df):,}\n"
    )

    report.write(
        f"Raw columns: "
        f"{len(raw_df.columns)}\n\n"
    )

    report.write(
        "VALIDATED (PRE-IMPUTATION) SNAPSHOT\n"
        "------------------------------------\n"
    )

    report.write(
        "A structurally validated snapshot is saved after raw-identity "
        "verification, type parsing, chronological sorting, and initial "
        "missingness recording, but BEFORE LOCF sensor imputation or "
        "derived-field reconstruction. It is a traceability artifact only "
        "and is not consumed by Stage 07+.\n"
    )

    report.write(
        f"Path: {VALIDATED_DATA_PATH}\n"
    )

    report.write(
        f"Validated rows: {len(validated_df):,}\n"
    )

    report.write(
        f"Validated columns: {len(validated_df.columns)}\n"
    )

    report.write(
        f"Validated missing cells (pre-imputation, expected > 0): "
        f"{validated_missing_cells}\n"
    )

    report.write(
        f"Validated SHA-256: {validated_sha256}\n\n"
    )

    report.write(
        "MISSING-SENSOR STRATEGY\n"
        "-----------------------\n"
    )

    report.write(
        "Method: last observation carried forward (LOCF) "
        "inside the same Smokehouse x Batch x Layer sequence.\n"
    )

    report.write(
        "Reason: missing sensor events are extremely rare, "
        "isolated, and the previous valid observation is "
        "available. This approach uses past information only "
        "and can be reproduced in deployment.\n"
    )

    report.write(
        "No future observation was used for imputation.\n"
    )

    report.write(
        f"Rows with any sensor missing before: "
        f"{rows_with_sensor_missing_before}\n"
    )

    report.write(
        f"Temperature values imputed: "
        f"{temperature_imputations}\n"
    )

    report.write(
        f"Humidity values imputed: "
        f"{humidity_imputations}\n"
    )

    report.write(
        f"Maximum age of imputation source: "
        f"{max_source_age_minutes:.1f} minutes\n"
    )

    report.write(
        f"Remaining missing cells after preprocessing: "
        f"{remaining_missing_cells}\n\n"
    )

    report.write(
        "DERIVED FIELDS\n"
        "--------------\n"
    )

    report.write(
        "Only Temperature_Change, Humidity_Change, "
        "Temperature_Trend and Humidity_Trend values that "
        "were missing because of a missing sensor reading "
        "were reconstructed. Existing non-missing derived "
        "values were preserved.\n\n"
    )

    report.write(
        "PACKET LOSS\n"
        "-----------\n"
    )

    report.write(
        f"10-minute intervals preserved: "
        f"{ten_minute_intervals:,}\n"
    )

    report.write(
        f"20-minute packet-loss gaps preserved: "
        f"{twenty_minute_intervals}\n"
    )

    report.write(
        f"Other interval lengths: "
        f"{other_interval_count}\n"
    )

    report.write(
        "No synthetic raw sensor rows were inserted.\n\n"
    )

    report.write(
        "LEAKAGE PROTECTION\n"
        "------------------\n"
    )

    report.write(
        "- No scaling was performed.\n"
    )

    report.write(
        "- No one-hot encoding was performed.\n"
    )

    report.write(
        "- No global-statistic imputation was performed.\n"
    )

    report.write(
        "- No model was fitted.\n"
    )

    report.write(
        "- No train/validation/test split was performed.\n"
    )

    report.write(
        "- Actual_Total_Drying_Time remains reference/"
        "target-construction information and is not a future "
        "physical predictor.\n"
    )

    report.write(
        "- Fire_Risk_Assessment_Timestamp remains ground-truth "
        "metadata and is not a future physical predictor.\n"
    )

    report.write(
        "- Batch_ID remains a grouping identifier rather than "
        "a physical model predictor.\n"
    )

    report.write(
        "- Learned preprocessing will be fitted on training "
        "data only after the batch-aware split in Stage 10.\n\n"
    )

    report.write(
        "OUTPUT DATASET\n"
        "--------------\n"
    )

    report.write(
        f"Path: "
        f"{PROCESSED_DATA_PATH}\n"
    )

    report.write(
        f"Rows: "
        f"{len(df):,}\n"
    )

    report.write(
        f"Columns: "
        f"{len(df.columns)}\n"
    )

    report.write(
        f"Processed SHA-256: "
        f"{processed_sha256}\n\n"
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
        f"STAGE 04 STATUS: "
        f"{overall_status}\n"
    )


# ============================================================
# 18. COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    f"STAGE 04 PREPROCESSING: "
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
    "\nFrozen raw data was not modified."
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 04 preprocessing failed. "
        "Review 04_preprocessing_report.txt before continuing."
    )
