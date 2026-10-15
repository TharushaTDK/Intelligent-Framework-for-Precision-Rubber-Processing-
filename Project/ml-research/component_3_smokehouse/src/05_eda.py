from pathlib import Path
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# COMPONENT 3 - STAGE 05: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
#
# Purpose:
#   Explore the preprocessed Component 3 dataset before
#   temporal analysis and feature engineering.
#
# This stage investigates:
#   - numeric distributions
#   - layer and smokehouse differences
#   - sequence-level drying-duration patterns
#   - regression-target relationships
#   - fire-risk class imbalance
#   - environmental conditions associated with fire-risk labels
#
# Important:
#   - This stage is DESCRIPTIVE, not predictive.
#   - NO model is trained.
#   - NO feature selection is performed.
#   - NO rolling / lag features are created.
#   - NO train/test split is performed.
#   - NO causal claims are made from correlations.
#
# Leakage reminder:
#   Actual_Total_Drying_Time may be inspected descriptively
#   because it is part of the research dataset, but it MUST
#   NOT become a deployed predictor of Remaining_Drying_Time.
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Component3_Dataset_Preprocessed.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "05_eda"
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
    / "05_eda_report.txt"
)

CHECKS_PATH = (
    REPORT_DIR
    / "05_eda_checks.csv"
)

NUMERIC_SUMMARY_PATH = (
    REPORT_DIR
    / "05_numeric_summary.csv"
)

LAYER_SUMMARY_PATH = (
    REPORT_DIR
    / "05_layer_eda_summary.csv"
)

SMOKEHOUSE_SUMMARY_PATH = (
    REPORT_DIR
    / "05_smokehouse_eda_summary.csv"
)

SEQUENCE_SUMMARY_PATH = (
    REPORT_DIR
    / "05_sequence_summary.csv"
)

CORRELATION_PATH = (
    REPORT_DIR
    / "05_numeric_correlations.csv"
)

REGRESSION_RELATIONSHIP_PATH = (
    REPORT_DIR
    / "05_regression_target_relationships.csv"
)

FIRE_ROW_SUMMARY_PATH = (
    REPORT_DIR
    / "05_fire_risk_row_summary.csv"
)

FIRE_ASSESSMENT_SUMMARY_PATH = (
    REPORT_DIR
    / "05_fire_risk_assessment_summary.csv"
)

FIRE_ASSESSMENT_ENV_PATH = (
    REPORT_DIR
    / "05_fire_risk_assessment_environment.csv"
)


# ============================================================
# 3. EXPECTED STRUCTURE
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
print("COMPONENT 3 - STAGE 05: EXPLORATORY DATA ANALYSIS")
print("=" * 78)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nPreprocessed dataset not found:\n{DATA_PATH}\n\n"
        "Run Stage 04 first."
    )

with open(DATA_PATH, "rb") as file:
    processed_sha256 = hashlib.sha256(
        file.read()
    ).hexdigest()

df = pd.read_csv(DATA_PATH)

print("\nPreprocessed dataset loaded successfully.")


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


# ============================================================
# 6. INPUT INTEGRITY CHECKS
# ============================================================

row_count_ok = (
    len(df)
    == EXPECTED_ROWS
)

schema_ok = (
    list(df.columns)
    == EXPECTED_COLUMNS
)

missing_cells = int(
    df[
        EXPECTED_COLUMNS
    ]
    .isna()
    .sum()
    .sum()
)

duplicate_rows = int(
    df.duplicated(
        keep=False
    ).sum()
)

duplicate_observation_keys = int(
    df.duplicated(
        subset=SEQUENCE_KEY + ["Timestamp"],
        keep="first",
    ).sum()
)


# ============================================================
# 7. NUMERIC DESCRIPTIVE STATISTICS
# ============================================================

eda_numeric_columns = [
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

numeric_summary = (
    df[
        eda_numeric_columns
    ]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .T
    .reset_index()
    .rename(
        columns={
            "index": "Column"
        }
    )
)

numeric_summary.to_csv(
    NUMERIC_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 8. LAYER-LEVEL EDA
# ============================================================
#
# Environment = row-level summary.
# Drying duration = sequence-level summary so longer sequences
# do not receive additional weight.

layer_environment = (
    df.groupby(
        "Layer_ID"
    )
    .agg(
        Records=(
            "Timestamp",
            "size",
        ),
        Mean_Temperature_C=(
            "Temperature_C",
            "mean",
        ),
        SD_Temperature_C=(
            "Temperature_C",
            "std",
        ),
        Mean_Humidity_pct=(
            "Humidity_%",
            "mean",
        ),
        SD_Humidity_pct=(
            "Humidity_%",
            "std",
        ),
        Mean_Temperature_Change=(
            "Temperature_Change",
            "mean",
        ),
        Mean_Humidity_Change=(
            "Humidity_Change",
            "mean",
        ),
        Mean_Remaining_Drying_Time=(
            "Remaining_Drying_Time",
            "mean",
        ),
    )
    .reset_index()
)

sequence_duration_base = (
    df[
        SEQUENCE_KEY
        + [
            "Number_of_Sheets",
            "Layer_Area_m2",
            "Actual_Total_Drying_Time",
        ]
    ]
    .drop_duplicates(
        subset=SEQUENCE_KEY
    )
)

layer_duration = (
    sequence_duration_base
    .groupby(
        "Layer_ID"
    )
    .agg(
        Sequences=(
            "Actual_Total_Drying_Time",
            "size",
        ),
        Mean_Total_Drying_Hours=(
            "Actual_Total_Drying_Time",
            "mean",
        ),
        SD_Total_Drying_Hours=(
            "Actual_Total_Drying_Time",
            "std",
        ),
        Min_Total_Drying_Hours=(
            "Actual_Total_Drying_Time",
            "min",
        ),
        Max_Total_Drying_Hours=(
            "Actual_Total_Drying_Time",
            "max",
        ),
        Mean_Number_of_Sheets=(
            "Number_of_Sheets",
            "mean",
        ),
    )
    .reset_index()
)

layer_summary = layer_environment.merge(
    layer_duration,
    on="Layer_ID",
    how="left",
)

layer_summary.to_csv(
    LAYER_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 9. SMOKEHOUSE-LEVEL EDA
# ============================================================

smokehouse_environment = (
    df.groupby(
        "Smokehouse_ID"
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
        Mean_Temperature_C=(
            "Temperature_C",
            "mean",
        ),
        Mean_Humidity_pct=(
            "Humidity_%",
            "mean",
        ),
        Mean_Remaining_Drying_Time=(
            "Remaining_Drying_Time",
            "mean",
        ),
    )
    .reset_index()
)

sequence_smokehouse_duration = (
    sequence_duration_base
    .groupby(
        "Smokehouse_ID"
    )
    .agg(
        Sequences=(
            "Actual_Total_Drying_Time",
            "size",
        ),
        Mean_Total_Drying_Hours=(
            "Actual_Total_Drying_Time",
            "mean",
        ),
        SD_Total_Drying_Hours=(
            "Actual_Total_Drying_Time",
            "std",
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

smokehouse_summary = (
    smokehouse_environment
    .merge(
        sequence_smokehouse_duration,
        on="Smokehouse_ID",
        how="left",
    )
)

smokehouse_summary.to_csv(
    SMOKEHOUSE_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 10. BATCH x LAYER SEQUENCE SUMMARY
# ============================================================

sequence_summary = (
    df.groupby(
        SEQUENCE_KEY
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
        Mean_Temperature_C=(
            "Temperature_C",
            "mean",
        ),
        SD_Temperature_C=(
            "Temperature_C",
            "std",
        ),
        Mean_Humidity_pct=(
            "Humidity_%",
            "mean",
        ),
        SD_Humidity_pct=(
            "Humidity_%",
            "std",
        ),
        Number_of_Sheets=(
            "Number_of_Sheets",
            "first",
        ),
        Layer_Area_m2=(
            "Layer_Area_m2",
            "first",
        ),
        Total_Drying_Hours=(
            "Actual_Total_Drying_Time",
            "first",
        ),
    )
    .reset_index()
)

sequence_summary.to_csv(
    SEQUENCE_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 11. REGRESSION-TARGET RELATIONSHIPS
# ============================================================
#
# These are descriptive Pearson correlations only.
# Correlation does NOT imply causation.

regression_candidates = [
    "Temperature_C",
    "Humidity_%",
    "Elapsed_Drying_Hours",
    "Number_of_Sheets",
    "Layer_Area_m2",
    "Temperature_Change",
    "Humidity_Change",
]

regression_relationship_rows = []

for column in regression_candidates:

    valid = (
        df[
            [
                column,
                "Remaining_Drying_Time",
            ]
        ]
        .dropna()
    )

    pearson = float(
        valid[
            column
        ]
        .corr(
            valid[
                "Remaining_Drying_Time"
            ]
        )
    )

    regression_relationship_rows.append({
        "Predictor_For_Exploration": column,
        "Pearson_Correlation_With_Remaining_Drying_Time": pearson,
        "Absolute_Correlation": abs(
            pearson
        ),
        "Observation_Count": len(
            valid
        ),
    })

regression_relationships = (
    pd.DataFrame(
        regression_relationship_rows
    )
    .sort_values(
        "Absolute_Correlation",
        ascending=False,
    )
)

regression_relationships.to_csv(
    REGRESSION_RELATIONSHIP_PATH,
    index=False,
)


# ============================================================
# 12. GENERAL NUMERIC CORRELATION MATRIX
# ============================================================

correlation_columns = [
    "Temperature_C",
    "Humidity_%",
    "Elapsed_Drying_Hours",
    "Number_of_Sheets",
    "Layer_Area_m2",
    "Temperature_Change",
    "Humidity_Change",
    "Remaining_Drying_Time",
]

correlation_matrix = (
    df[
        correlation_columns
    ]
    .corr(
        method="pearson"
    )
)

correlation_matrix.to_csv(
    CORRELATION_PATH
)


# ============================================================
# 13. FIRE-RISK ROW-LEVEL DISTRIBUTION
# ============================================================

fire_row_summary = (
    df[
        "Fire_Risk_Label"
    ]
    .value_counts()
    .rename_axis(
        "Fire_Risk_Label"
    )
    .reset_index(
        name="Row_Count"
    )
)

fire_row_summary[
    "Row_Percentage"
] = (
    fire_row_summary[
        "Row_Count"
    ]
    / len(df)
    * 100
)

fire_row_summary.to_csv(
    FIRE_ROW_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 14. UNIQUE HOURLY FIRE-RISK ASSESSMENT EVENTS
# ============================================================
#
# Do not count the six carried-forward 10-minute rows as six
# independent keeper/expert assessments.

assessment_key = (
    SEQUENCE_KEY
    + [
        "Fire_Risk_Assessment_Timestamp"
    ]
)

assessment_events = (
    df.groupby(
        assessment_key,
        as_index=False,
    )
    .agg(
        Fire_Risk_Label=(
            "Fire_Risk_Label",
            "first",
        ),
        Window_Mean_Temperature_C=(
            "Temperature_C",
            "mean",
        ),
        Window_Min_Temperature_C=(
            "Temperature_C",
            "min",
        ),
        Window_Max_Temperature_C=(
            "Temperature_C",
            "max",
        ),
        Window_Mean_Humidity_pct=(
            "Humidity_%",
            "mean",
        ),
        Window_Min_Humidity_pct=(
            "Humidity_%",
            "min",
        ),
        Window_Max_Humidity_pct=(
            "Humidity_%",
            "max",
        ),
        Window_Mean_Elapsed_Hours=(
            "Elapsed_Drying_Hours",
            "mean",
        ),
        Window_Mean_Temperature_Change=(
            "Temperature_Change",
            "mean",
        ),
        Window_Mean_Humidity_Change=(
            "Humidity_Change",
            "mean",
        ),
        Sensor_Rows_In_Assessment=(
            "Timestamp",
            "size",
        ),
    )
)

fire_assessment_summary = (
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

fire_assessment_summary[
    "Assessment_Percentage"
] = (
    fire_assessment_summary[
        "Assessment_Count"
    ]
    / len(
        assessment_events
    )
    * 100
)

fire_assessment_summary.to_csv(
    FIRE_ASSESSMENT_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 15. FIRE-RISK ENVIRONMENTAL SUMMARY
# ============================================================

fire_assessment_environment = (
    assessment_events
    .groupby(
        "Fire_Risk_Label"
    )
    .agg(
        Assessment_Count=(
            "Fire_Risk_Label",
            "size",
        ),
        Mean_Temperature_C=(
            "Window_Mean_Temperature_C",
            "mean",
        ),
        Mean_Humidity_pct=(
            "Window_Mean_Humidity_pct",
            "mean",
        ),
        Mean_Elapsed_Hours=(
            "Window_Mean_Elapsed_Hours",
            "mean",
        ),
        Median_Elapsed_Hours=(
            "Window_Mean_Elapsed_Hours",
            "median",
        ),
        Mean_Temperature_Change=(
            "Window_Mean_Temperature_Change",
            "mean",
        ),
        Mean_Humidity_Change=(
            "Window_Mean_Humidity_Change",
            "mean",
        ),
    )
    .reset_index()
)

risk_order = pd.CategoricalDtype(
    [
        "Normal",
        "Warning",
        "Critical",
    ],
    ordered=True,
)

fire_assessment_environment[
    "Fire_Risk_Label"
] = (
    fire_assessment_environment[
        "Fire_Risk_Label"
    ]
    .astype(
        risk_order
    )
)

fire_assessment_environment = (
    fire_assessment_environment
    .sort_values(
        "Fire_Risk_Label"
    )
)

fire_assessment_environment.to_csv(
    FIRE_ASSESSMENT_ENV_PATH,
    index=False,
)


# ============================================================
# 16. FIGURE HELPERS
# ============================================================

def save_current_figure(
    filename,
):
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR
        / filename,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# 17. FIGURES - EACH FIGURE IS INDEPENDENT
# ============================================================

# Figure 1: Temperature distribution
plt.figure(
    figsize=(8, 5)
)
plt.hist(
    df[
        "Temperature_C"
    ],
    bins=35,
)
plt.xlabel(
    "Temperature (°C)"
)
plt.ylabel(
    "Frequency"
)
plt.title(
    "Temperature Distribution"
)
save_current_figure(
    "05_01_temperature_distribution.png"
)


# Figure 2: Humidity distribution
plt.figure(
    figsize=(8, 5)
)
plt.hist(
    df[
        "Humidity_%"
    ],
    bins=35,
)
plt.xlabel(
    "Relative Humidity (%)"
)
plt.ylabel(
    "Frequency"
)
plt.title(
    "Humidity Distribution"
)
save_current_figure(
    "05_02_humidity_distribution.png"
)


# Figure 3: Remaining drying time distribution
plt.figure(
    figsize=(8, 5)
)
plt.hist(
    df[
        "Remaining_Drying_Time"
    ],
    bins=35,
)
plt.xlabel(
    "Remaining Drying Time (hours)"
)
plt.ylabel(
    "Frequency"
)
plt.title(
    "Remaining Drying Time Distribution"
)
save_current_figure(
    "05_03_remaining_time_distribution.png"
)


# Figure 4: Temperature by layer
layer_order = [
    "L1",
    "L2",
    "L3",
]

temperature_by_layer = [
    df.loc[
        df[
            "Layer_ID"
        ]
        == layer,
        "Temperature_C",
    ]
    .to_numpy()
    for layer in layer_order
]

plt.figure(
    figsize=(8, 5)
)
plt.boxplot(
    temperature_by_layer,
    tick_labels=layer_order,
)
plt.xlabel(
    "Layer"
)
plt.ylabel(
    "Temperature (°C)"
)
plt.title(
    "Temperature by Smokehouse Layer"
)
save_current_figure(
    "05_04_temperature_by_layer.png"
)


# Figure 5: Humidity by layer
humidity_by_layer = [
    df.loc[
        df[
            "Layer_ID"
        ]
        == layer,
        "Humidity_%",
    ]
    .to_numpy()
    for layer in layer_order
]

plt.figure(
    figsize=(8, 5)
)
plt.boxplot(
    humidity_by_layer,
    tick_labels=layer_order,
)
plt.xlabel(
    "Layer"
)
plt.ylabel(
    "Relative Humidity (%)"
)
plt.title(
    "Humidity by Smokehouse Layer"
)
save_current_figure(
    "05_05_humidity_by_layer.png"
)


# Figure 6: Sequence-level total drying time by layer
duration_by_layer = [
    sequence_summary.loc[
        sequence_summary[
            "Layer_ID"
        ]
        == layer,
        "Total_Drying_Hours",
    ]
    .to_numpy()
    for layer in layer_order
]

plt.figure(
    figsize=(8, 5)
)
plt.boxplot(
    duration_by_layer,
    tick_labels=layer_order,
)
plt.xlabel(
    "Layer"
)
plt.ylabel(
    "Total Drying Time (hours)"
)
plt.title(
    "Total Drying Time by Layer (36 Sequences)"
)
save_current_figure(
    "05_06_total_drying_time_by_layer.png"
)


# Figure 7: Fire-risk row distribution
fire_row_plot = (
    fire_row_summary
    .set_index(
        "Fire_Risk_Label"
    )
    .reindex(
        [
            "Normal",
            "Warning",
            "Critical",
        ]
    )
)

plt.figure(
    figsize=(8, 5)
)
plt.bar(
    fire_row_plot.index,
    fire_row_plot[
        "Row_Count"
    ],
)
plt.xlabel(
    "Fire Risk Label"
)
plt.ylabel(
    "Sensor Rows"
)
plt.title(
    "Fire-Risk Distribution at 10-Minute Row Level"
)
save_current_figure(
    "05_07_fire_risk_row_distribution.png"
)


# Figure 8: Unique hourly fire-risk assessments
assessment_plot = (
    fire_assessment_summary
    .set_index(
        "Fire_Risk_Label"
    )
    .reindex(
        [
            "Normal",
            "Warning",
            "Critical",
        ]
    )
)

plt.figure(
    figsize=(8, 5)
)
plt.bar(
    assessment_plot.index,
    assessment_plot[
        "Assessment_Count"
    ],
)
plt.xlabel(
    "Fire Risk Label"
)
plt.ylabel(
    "Unique Hourly Assessments"
)
plt.title(
    "Fire-Risk Distribution at Hourly Assessment Level"
)
save_current_figure(
    "05_08_fire_risk_assessment_distribution.png"
)


# Figure 9: Temperature vs remaining drying time
plot_sample = (
    df.iloc[
        ::10
    ]
)

plt.figure(
    figsize=(8, 5)
)
plt.scatter(
    plot_sample[
        "Temperature_C"
    ],
    plot_sample[
        "Remaining_Drying_Time"
    ],
    alpha=0.25,
    s=12,
)
plt.xlabel(
    "Temperature (°C)"
)
plt.ylabel(
    "Remaining Drying Time (hours)"
)
plt.title(
    "Temperature vs Remaining Drying Time"
)
save_current_figure(
    "05_09_temperature_vs_remaining_time.png"
)


# Figure 10: Humidity vs remaining drying time
plt.figure(
    figsize=(8, 5)
)
plt.scatter(
    plot_sample[
        "Humidity_%"
    ],
    plot_sample[
        "Remaining_Drying_Time"
    ],
    alpha=0.25,
    s=12,
)
plt.xlabel(
    "Relative Humidity (%)"
)
plt.ylabel(
    "Remaining Drying Time (hours)"
)
plt.title(
    "Humidity vs Remaining Drying Time"
)
save_current_figure(
    "05_10_humidity_vs_remaining_time.png"
)


# ============================================================
# 18. KEY DESCRIPTIVE FINDINGS
# ============================================================

sequence_temp_duration_corr = float(
    sequence_summary[
        "Mean_Temperature_C"
    ]
    .corr(
        sequence_summary[
            "Total_Drying_Hours"
        ]
    )
)

sequence_humidity_duration_corr = float(
    sequence_summary[
        "Mean_Humidity_pct"
    ]
    .corr(
        sequence_summary[
            "Total_Drying_Hours"
        ]
    )
)

layer_duration_means = (
    layer_summary
    .set_index(
        "Layer_ID"
    )[
        "Mean_Total_Drying_Hours"
    ]
    .to_dict()
)

assessment_count = len(
    assessment_events
)

warning_assessments = int(
    (
        assessment_events[
            "Fire_Risk_Label"
        ]
        == "Warning"
    ).sum()
)

critical_assessments = int(
    (
        assessment_events[
            "Fire_Risk_Label"
        ]
        == "Critical"
    ).sum()
)


# ============================================================
# 19. PASS / FAIL CHECKS
# ============================================================

checks = [
    {
        "Check": "Expected row count",
        "Status": (
            "PASS"
            if row_count_ok
            else "FAIL"
        ),
        "Details": (
            f"Rows={len(df):,}"
        ),
    },
    {
        "Check": "Expected 17-column schema",
        "Status": (
            "PASS"
            if schema_ok
            else "FAIL"
        ),
        "Details": (
            f"Columns={len(df.columns)}"
        ),
    },
    {
        "Check": "Preprocessed dataset has no missing cells",
        "Status": (
            "PASS"
            if missing_cells == 0
            else "FAIL"
        ),
        "Details": (
            f"Missing cells={missing_cells}"
        ),
    },
    {
        "Check": "No exact duplicates",
        "Status": (
            "PASS"
            if duplicate_rows == 0
            else "FAIL"
        ),
        "Details": (
            f"Duplicate members={duplicate_rows}"
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
            f"Extra keys="
            f"{duplicate_observation_keys}"
        ),
    },
    {
        "Check": "Expected 36 Batch-Layer sequences",
        "Status": (
            "PASS"
            if len(sequence_summary) == 36
            else "FAIL"
        ),
        "Details": (
            f"Sequences="
            f"{len(sequence_summary)}"
        ),
    },
    {
        "Check": "Expected 4,098 unique hourly fire-risk assessments",
        "Status": (
            "PASS"
            if assessment_count == 4098
            else "FAIL"
        ),
        "Details": (
            f"Assessments="
            f"{assessment_count:,}"
        ),
    },
    {
        "Check": "Expected Warning assessment count",
        "Status": (
            "PASS"
            if warning_assessments == 156
            else "FAIL"
        ),
        "Details": (
            f"Warning assessments="
            f"{warning_assessments}"
        ),
    },
    {
        "Check": "Expected Critical assessment count",
        "Status": (
            "PASS"
            if critical_assessments == 26
            else "FAIL"
        ),
        "Details": (
            f"Critical assessments="
            f"{critical_assessments}"
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
# 20. CONSOLE OUTPUT
# ============================================================

print(
    "\n--- INPUT DATASET ---"
)

print(
    f"Rows / columns                   : "
    f"{len(df):,} / {len(df.columns)}"
)

print(
    f"Missing cells                    : "
    f"{missing_cells}"
)

print(
    f"Batch-Layer sequences            : "
    f"{len(sequence_summary)}"
)

print(
    f"Processed SHA-256                : "
    f"{processed_sha256}"
)


print(
    "\n--- LAYER SUMMARY ---"
)

print(
    layer_summary[
        [
            "Layer_ID",
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
    "\n--- SEQUENCE-LEVEL DRYING RELATIONSHIPS ---"
)

print(
    f"Mean temperature vs total drying : "
    f"{sequence_temp_duration_corr:.4f}"
)

print(
    f"Mean humidity vs total drying    : "
    f"{sequence_humidity_duration_corr:.4f}"
)


print(
    "\n--- REGRESSION TARGET RELATIONSHIPS ---"
)

print(
    regression_relationships[
        [
            "Predictor_For_Exploration",
            "Pearson_Correlation_With_Remaining_Drying_Time",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\n--- FIRE-RISK DISTRIBUTION ---"
)

print(
    f"Row-level records                : "
    f"{len(df):,}"
)

print(
    f"Unique hourly assessments        : "
    f"{assessment_count:,}"
)

print(
    fire_assessment_summary
    .round(
        {
            "Assessment_Percentage": 4
        }
    )
    .to_string(
        index=False
    )
)


print(
    "\n--- FIRE-RISK ENVIRONMENTAL SUMMARY ---"
)

print(
    fire_assessment_environment
    .round(3)
    .to_string(
        index=False
    )
)


print(
    "\n--- FIGURES CREATED ---"
)

print(
    f"Figure directory:\n"
    f"{FIGURE_DIR}"
)


# ============================================================
# 21. HUMAN-READABLE EDA REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "COMPONENT 3 - STAGE 05: EXPLORATORY DATA ANALYSIS\n"
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
        "INPUT DATASET\n"
        "-------------\n"
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
        f"Columns: "
        f"{len(df.columns)}\n"
    )

    report.write(
        f"Missing cells: "
        f"{missing_cells}\n"
    )

    report.write(
        f"Processed SHA-256: "
        f"{processed_sha256}\n\n"
    )

    report.write(
        "LAYER-LEVEL PATTERN\n"
        "-------------------\n"
    )

    report.write(
        layer_summary[
            [
                "Layer_ID",
                "Mean_Temperature_C",
                "Mean_Humidity_pct",
                "Mean_Total_Drying_Hours",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "SEQUENCE-LEVEL DRYING RELATIONSHIPS\n"
        "-----------------------------------\n"
    )

    report.write(
        f"Mean temperature vs total drying "
        f"time correlation: "
        f"{sequence_temp_duration_corr:.6f}\n"
    )

    report.write(
        f"Mean humidity vs total drying "
        f"time correlation: "
        f"{sequence_humidity_duration_corr:.6f}\n"
    )

    report.write(
        "Interpretation: these are descriptive associations "
        "across the 36 Batch x Layer sequences. They do not "
        "establish causation.\n\n"
    )

    report.write(
        "REGRESSION-TARGET RELATIONSHIPS\n"
        "-------------------------------\n"
    )

    report.write(
        regression_relationships
        .round(6)
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "FIRE-RISK CLASS DISTRIBUTION\n"
        "----------------------------\n"
    )

    report.write(
        "Row-level labels represent the latest hourly "
        "assessment carried across 10-minute sensor rows. "
        "Therefore, the unique hourly assessment distribution "
        "is the more appropriate description of independent "
        "keeper/expert ground-truth events.\n\n"
    )

    report.write(
        fire_assessment_summary
        .round(6)
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "FIRE-RISK ENVIRONMENTAL SUMMARY\n"
        "-------------------------------\n"
    )

    report.write(
        fire_assessment_environment
        .round(6)
        .to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "EDA INTERPRETATION NOTES\n"
        "------------------------\n"
    )

    report.write(
        "- L1 is generally warmer and less humid than L2/L3, "
        "while L3 shows the longest average sequence-level "
        "drying duration.\n"
    )

    report.write(
        "- Across the 36 sequences, higher mean temperature "
        "is associated with shorter total drying time, while "
        "higher mean humidity is associated with longer total "
        "drying time.\n"
    )

    report.write(
        "- Remaining_Drying_Time is strongly associated with "
        "Elapsed_Drying_Hours by construction, but elapsed "
        "time alone is not treated as sufficient evidence of "
        "final model performance.\n"
    )

    report.write(
        "- Fire_Risk_Label is strongly imbalanced. Accuracy "
        "alone will therefore be unsuitable for later model "
        "evaluation.\n"
    )

    report.write(
        "- Warning and Critical assessment conditions must not "
        "be interpreted using a single temperature or humidity "
        "threshold; later temporal analysis will examine recent "
        "behavior and trends.\n"
    )

    report.write(
        "- Actual_Total_Drying_Time is inspected descriptively "
        "only and remains excluded from deployed predictors.\n"
    )

    report.write(
        "- Stage 06 will analyze time dependence, layer-wise "
        "trajectories and temporal behavior. No rolling or lag "
        "features are created in Stage 05.\n\n"
    )

    report.write(
        "FIGURES\n"
        "-------\n"
    )

    for filename in sorted(
        path.name
        for path in FIGURE_DIR.glob(
            "*.png"
        )
    ):
        report.write(
            f"- {filename}\n"
        )

    report.write(
        "\nPASS / FAIL CHECKS\n"
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
        f"STAGE 05 STATUS: "
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
    f"STAGE 05 EXPLORATORY DATA ANALYSIS: "
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
    f"\nFigures saved in:\n"
    f"{FIGURE_DIR}"
)

print(
    "\nNo dataset was modified."
)

if overall_status != "PASS":
    raise RuntimeError(
        "Stage 05 EDA failed. "
        "Review 05_eda_report.txt before continuing."
    )
