from pathlib import Path
import hashlib
import pandas as pd


# ============================================================
# COMPONENT 3 - STAGE 01: DATA UNDERSTANDING
# ============================================================
#
# Purpose:
#   Understand the FINAL FROZEN raw dataset before any
#   validation, cleaning, preprocessing, feature engineering,
#   splitting or model training.
#
# Important:
#   - This script DOES NOT modify the raw dataset.
#   - This script DOES NOT fill missing values.
#   - This script DOES NOT remove duplicates.
#   - This script DOES NOT validate targets mathematically.
#   - This script DOES NOT create ML features.
#   - Actual_Total_Drying_Time is inspected here only as a
#     dataset field; it must not become a live model feature.
#   - Fire_Risk_Assessment_Timestamp is metadata for the
#     hourly keeper/expert assessment and must not become a
#     physical ML predictor.
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

OVERVIEW_PATH = (
    REPORT_DIR
    / "01_dataset_overview.txt"
)

COLUMN_SUMMARY_PATH = (
    REPORT_DIR
    / "01_column_summary.csv"
)

SMOKEHOUSE_SUMMARY_PATH = (
    REPORT_DIR
    / "01_smokehouse_summary.csv"
)

BATCH_SUMMARY_PATH = (
    REPORT_DIR
    / "01_batch_summary.csv"
)

LAYER_SUMMARY_PATH = (
    REPORT_DIR
    / "01_layer_summary.csv"
)

FIRE_RISK_PATH = (
    REPORT_DIR
    / "01_fire_risk_distribution.csv"
)

BATCH_LAYER_PATH = (
    REPORT_DIR
    / "01_batch_layer_structure.csv"
)


# ============================================================
# 3. EXPECTED FINAL SCHEMA
# ============================================================

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


# ============================================================
# 4. LOAD RAW DATASET
# ============================================================

print("=" * 78)
print("COMPONENT 3 - STAGE 01: DATA UNDERSTANDING")
print("=" * 78)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nFinal frozen dataset not found:\n{DATA_PATH}\n\n"
        "Place Component3_Dataset.csv inside:\n"
        "ml-research/component_3_smokehouse/data/raw/"
    )

df = pd.read_csv(
    DATA_PATH
)

print("\nRaw dataset loaded successfully.")


# ============================================================
# 5. DATASET FINGERPRINT
# ============================================================

with open(
    DATA_PATH,
    "rb",
) as file:
    dataset_sha256 = hashlib.sha256(
        file.read()
    ).hexdigest()


# ============================================================
# 6. BASIC STRUCTURE
# ============================================================

rows = len(df)
columns = len(df.columns)

actual_columns = list(
    df.columns
)

schema_matches = (
    actual_columns
    == EXPECTED_COLUMNS
)

missing_expected_columns = [
    column
    for column in EXPECTED_COLUMNS
    if column not in actual_columns
]

unexpected_columns = [
    column
    for column in actual_columns
    if column not in EXPECTED_COLUMNS
]


# ============================================================
# 7. SAFE TYPE CONVERSION FOR UNDERSTANDING
# ============================================================

# Work on a copy so the raw file is untouched.

analysis_df = df.copy()

analysis_df["Timestamp"] = pd.to_datetime(
    analysis_df["Timestamp"],
    errors="coerce",
)

analysis_df[
    "Fire_Risk_Assessment_Timestamp"
] = pd.to_datetime(
    analysis_df[
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
    analysis_df[column] = pd.to_numeric(
        analysis_df[column],
        errors="coerce",
    )


# ============================================================
# 8. CORE ENTITY COUNTS
# ============================================================

smokehouse_count = (
    analysis_df[
        "Smokehouse_ID"
    ].nunique(
        dropna=True
    )
)

batch_count = (
    analysis_df[
        "Batch_ID"
    ].nunique(
        dropna=True
    )
)

layer_count = (
    analysis_df[
        "Layer_ID"
    ].nunique(
        dropna=True
    )
)

batch_layer_count = (
    analysis_df.groupby(
        [
            "Smokehouse_ID",
            "Batch_ID",
            "Layer_ID",
        ],
        dropna=False,
    ).ngroups
)


# ============================================================
# 9. TIME COVERAGE
# ============================================================

timestamp_start = (
    analysis_df[
        "Timestamp"
    ].min()
)

timestamp_end = (
    analysis_df[
        "Timestamp"
    ].max()
)

assessment_start = (
    analysis_df[
        "Fire_Risk_Assessment_Timestamp"
    ].min()
)

assessment_end = (
    analysis_df[
        "Fire_Risk_Assessment_Timestamp"
    ].max()
)


# ============================================================
# 10. COLUMN SUMMARY
# ============================================================

column_rows = []

for column in analysis_df.columns:

    series = analysis_df[column]

    column_rows.append({
        "Column": column,
        "Pandas_Dtype": str(
            series.dtype
        ),
        "Non_Missing_Count": int(
            series.notna().sum()
        ),
        "Missing_Count": int(
            series.isna().sum()
        ),
        "Missing_Percent": round(
            series.isna().mean()
            * 100,
            6,
        ),
        "Unique_Count": int(
            series.nunique(
                dropna=True
            )
        ),
    })

column_summary = pd.DataFrame(
    column_rows
)

column_summary.to_csv(
    COLUMN_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 11. SMOKEHOUSE SUMMARY
# ============================================================

smokehouse_summary = (
    analysis_df.groupby(
        "Smokehouse_ID",
        dropna=False,
    )
    .agg(
        Records=(
            "Timestamp",
            "size",
        ),
        Batches=(
            "Batch_ID",
            "nunique",
        ),
        Layers=(
            "Layer_ID",
            "nunique",
        ),
        Start_Timestamp=(
            "Timestamp",
            "min",
        ),
        End_Timestamp=(
            "Timestamp",
            "max",
        ),
        Mean_Temperature_C=(
            "Temperature_C",
            "mean",
        ),
        Mean_Humidity_pct=(
            "Humidity_%",
            "mean",
        ),
    )
    .reset_index()
)

smokehouse_summary.to_csv(
    SMOKEHOUSE_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 12. BATCH SUMMARY
# ============================================================

batch_summary = (
    analysis_df.groupby(
        [
            "Smokehouse_ID",
            "Batch_ID",
        ],
        dropna=False,
    )
    .agg(
        Records=(
            "Timestamp",
            "size",
        ),
        Layers=(
            "Layer_ID",
            "nunique",
        ),
        Batch_Start=(
            "Timestamp",
            "min",
        ),
        Batch_End=(
            "Timestamp",
            "max",
        ),
        Min_Total_Drying_Hours=(
            "Actual_Total_Drying_Time",
            "min",
        ),
        Max_Total_Drying_Hours=(
            "Actual_Total_Drying_Time",
            "max",
        ),
    )
    .reset_index()
)

batch_summary.to_csv(
    BATCH_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 13. LAYER SUMMARY
# ============================================================

# Environmental means below are row-level descriptive statistics.
# Drying-duration statistics are calculated once per Batch x Layer
# sequence so longer sequences do not receive extra weight.

layer_environment_summary = (
    analysis_df.groupby(
        "Layer_ID",
        dropna=False,
    )
    .agg(
        Records=(
            "Timestamp",
            "size",
        ),
        Smokehouses=(
            "Smokehouse_ID",
            "nunique",
        ),
        Batches=(
            "Batch_ID",
            "nunique",
        ),
        Mean_Temperature_C=(
            "Temperature_C",
            "mean",
        ),
        Min_Temperature_C=(
            "Temperature_C",
            "min",
        ),
        Max_Temperature_C=(
            "Temperature_C",
            "max",
        ),
        Mean_Humidity_pct=(
            "Humidity_%",
            "mean",
        ),
        Min_Humidity_pct=(
            "Humidity_%",
            "min",
        ),
        Max_Humidity_pct=(
            "Humidity_%",
            "max",
        ),
    )
    .reset_index()
)

sequence_durations = (
    analysis_df[
        [
            "Smokehouse_ID",
            "Batch_ID",
            "Layer_ID",
            "Actual_Total_Drying_Time",
        ]
    ]
    .drop_duplicates(
        subset=[
            "Smokehouse_ID",
            "Batch_ID",
            "Layer_ID",
        ]
    )
)

layer_duration_summary = (
    sequence_durations.groupby(
        "Layer_ID",
        dropna=False,
    )
    .agg(
        Mean_Total_Drying_Hours=(
            "Actual_Total_Drying_Time",
            "mean",
        ),
        Min_Total_Drying_Hours=(
            "Actual_Total_Drying_Time",
            "min",
        ),
        Max_Total_Drying_Hours=(
            "Actual_Total_Drying_Time",
            "max",
        ),
    )
    .reset_index()
)

layer_summary = layer_environment_summary.merge(
    layer_duration_summary,
    on="Layer_ID",
    how="left",
)

layer_summary.to_csv(
    LAYER_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 14. BATCH x LAYER STRUCTURE
# ============================================================

batch_layer_summary = (
    analysis_df.groupby(
        [
            "Smokehouse_ID",
            "Batch_ID",
            "Layer_ID",
        ],
        dropna=False,
    )
    .agg(
        Records=(
            "Timestamp",
            "size",
        ),
        Start_Timestamp=(
            "Timestamp",
            "min",
        ),
        End_Timestamp=(
            "Timestamp",
            "max",
        ),
        Total_Drying_Hours=(
            "Actual_Total_Drying_Time",
            "first",
        ),
        Start_Remaining_Hours=(
            "Remaining_Drying_Time",
            "first",
        ),
        End_Remaining_Hours=(
            "Remaining_Drying_Time",
            "last",
        ),
    )
    .reset_index()
)

batch_layer_summary.to_csv(
    BATCH_LAYER_PATH,
    index=False,
)


# ============================================================
# 15. FIRE-RISK DISTRIBUTION
# ============================================================

fire_risk_distribution = (
    analysis_df[
        "Fire_Risk_Label"
    ]
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

fire_risk_distribution[
    "Percentage"
] = (
    fire_risk_distribution[
        "Count"
    ]
    / rows
    * 100
)

fire_risk_distribution.to_csv(
    FIRE_RISK_PATH,
    index=False,
)


# ============================================================
# 16. IMPORTANT NUMERIC RANGES
# ============================================================

important_numeric_ranges = {}

for column in [
    "Temperature_C",
    "Humidity_%",
    "Elapsed_Drying_Hours",
    "Actual_Total_Drying_Time",
    "Remaining_Drying_Time",
    "Number_of_Sheets",
    "Layer_Area_m2",
]:
    important_numeric_ranges[
        column
    ] = {
        "min": (
            analysis_df[
                column
            ].min()
        ),
        "max": (
            analysis_df[
                column
            ].max()
        ),
        "mean": (
            analysis_df[
                column
            ].mean()
        ),
    }


# ============================================================
# 17. CONSOLE OUTPUT
# ============================================================

print("\n--- DATASET STRUCTURE ---")

print(
    f"Rows                 : "
    f"{rows:,}"
)

print(
    f"Columns              : "
    f"{columns}"
)

print(
    f"Schema matches final : "
    f"{schema_matches}"
)

print(
    f"SHA-256              : "
    f"{dataset_sha256}"
)

print(
    f"Smokehouses          : "
    f"{smokehouse_count}"
)

print(
    f"Batches              : "
    f"{batch_count}"
)

print(
    f"Layers               : "
    f"{layer_count}"
)

print(
    f"Batch-Layer sequences: "
    f"{batch_layer_count}"
)


print("\n--- TIME COVERAGE ---")

print(
    f"Sensor data start     : "
    f"{timestamp_start}"
)

print(
    f"Sensor data end       : "
    f"{timestamp_end}"
)

print(
    f"Risk assessment start : "
    f"{assessment_start}"
)

print(
    f"Risk assessment end   : "
    f"{assessment_end}"
)


print(
    "\n--- FIRE-RISK DISTRIBUTION ---"
)

print(
    fire_risk_distribution
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
    "\n--- LAYER SUMMARY ---"
)

print(
    layer_summary[
        [
            "Layer_ID",
            "Records",
            "Mean_Temperature_C",
            "Mean_Humidity_pct",
            "Mean_Total_Drying_Hours",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


print(
    "\n--- MISSING VALUES OVERVIEW ---"
)

missing_display = (
    column_summary[
        column_summary[
            "Missing_Count"
        ] > 0
    ][
        [
            "Column",
            "Missing_Count",
            "Missing_Percent",
        ]
    ]
)

if len(
    missing_display
) == 0:
    print(
        "No missing values."
    )
else:
    print(
        missing_display
        .to_string(
            index=False
        )
    )


# ============================================================
# 18. HUMAN-READABLE OVERVIEW REPORT
# ============================================================

with open(
    OVERVIEW_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 01: DATA UNDERSTANDING\n"
    )

    report.write(
        "=" * 78
        + "\n\n"
    )

    report.write(
        "DATASET IDENTITY\n"
        "----------------\n"
    )

    report.write(
        "Dataset: "
        "Component3_Dataset.csv\n"
    )

    report.write(
        f"SHA-256: "
        f"{dataset_sha256}\n"
    )

    report.write(
        f"Rows: {rows:,}\n"
    )

    report.write(
        f"Columns: {columns}\n"
    )

    report.write(
        f"Final schema match: "
        f"{schema_matches}\n\n"
    )

    report.write(
        "CORE STRUCTURE\n"
        "--------------\n"
    )

    report.write(
        f"Smokehouses: "
        f"{smokehouse_count}\n"
    )

    report.write(
        f"Batches: "
        f"{batch_count}\n"
    )

    report.write(
        f"Layers: "
        f"{layer_count}\n"
    )

    report.write(
        f"Batch-Layer sequences: "
        f"{batch_layer_count}\n\n"
    )

    report.write(
        "TIME COVERAGE\n"
        "-------------\n"
    )

    report.write(
        f"Sensor data start: "
        f"{timestamp_start}\n"
    )

    report.write(
        f"Sensor data end: "
        f"{timestamp_end}\n"
    )

    report.write(
        f"Fire-risk assessment start: "
        f"{assessment_start}\n"
    )

    report.write(
        f"Fire-risk assessment end: "
        f"{assessment_end}\n\n"
    )

    report.write(
        "IMPORTANT NUMERIC RANGES\n"
        "------------------------\n"
    )

    for (
        column,
        statistics,
    ) in important_numeric_ranges.items():

        report.write(
            f"{column}: "
            f"min={statistics['min']:.4f}, "
            f"max={statistics['max']:.4f}, "
            f"mean={statistics['mean']:.4f}\n"
        )

    report.write(
        "\nFIRE-RISK DISTRIBUTION\n"
        "----------------------\n"
    )

    report.write(
        fire_risk_distribution
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
        "MISSING VALUES OVERVIEW\n"
        "-----------------------\n"
    )

    if len(
        missing_display
    ) == 0:

        report.write(
            "No missing values.\n"
        )

    else:

        report.write(
            missing_display
            .to_string(
                index=False
            )
        )

        report.write(
            "\n"
        )

    report.write(
        "\nMETHODOLOGY NOTES\n"
        "-----------------\n"
    )

    report.write(
        "- Stage 01 performs understanding only; "
        "no data cleaning or modification is performed.\n"
    )

    report.write(
        "- Actual_Total_Drying_Time is a retrospective "
        "reference/target-construction field and must not be "
        "used as a deployed predictor.\n"
    )

    report.write(
        "- Fire_Risk_Assessment_Timestamp records the hourly "
        "keeper/expert assessment time and should not be used "
        "as a physical ML predictor.\n"
    )

    report.write(
        "- Batch_ID is preserved for grouping and future "
        "batch-aware validation, not as a physical ML predictor.\n"
    )

    report.write(
        "- Missing values are only observed at this stage; "
        "handling decisions belong to later data-quality and "
        "preprocessing stages.\n"
    )

    report.write(
        "- The dataset is synthetic/simulated development data "
        "and must not be described as 24,463 physically "
        "field-collected sensor measurements.\n"
    )

    report.write(
        "\nSTAGE 01 STATUS: COMPLETE AFTER REVIEW\n"
    )


# ============================================================
# 19. COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "STAGE 01 DATA UNDERSTANDING COMPLETE"
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
