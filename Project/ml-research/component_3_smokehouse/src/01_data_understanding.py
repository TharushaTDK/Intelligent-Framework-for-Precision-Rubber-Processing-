from pathlib import Path
import pandas as pd
import numpy as np


# COMPONENT 3 - STAGE 1: DATA UNDERSTANDING
# Smart Layer-Wise Rubber Smokehouse Drying-Time &
# Fire-Risk Prediction System


# 1. PROJECT PATHS


# Expected location of this script:
# ml-research/component_3_smokehouse/src/01_data_understanding.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Component3_Dataset.csv"
)

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = REPORT_DIR / "01_data_understanding_summary.txt"
BATCH_OVERVIEW_PATH = REPORT_DIR / "01_batch_overview.csv"


# 2. LOAD DATASET


print("=" * 72)
print("COMPONENT 3 - STAGE 1: DATA UNDERSTANDING")
print("=" * 72)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Expected location:\n"
        "ml-research/component_3_smokehouse/data/raw/Component3_Dataset.csv"
    )

df = pd.read_csv(DATA_PATH)
print("\nDataset loaded successfully.")


# 3. EXPECTED DATASET STRUCTURE


expected_columns = [
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
]

missing_columns = [c for c in expected_columns if c not in df.columns]
unexpected_columns = [c for c in df.columns if c not in expected_columns]


# 4. BASIC STRUCTURE


rows, columns = df.shape

print("\n--- DATASET SHAPE ---")
print(f"Rows    : {rows:,}")
print(f"Columns : {columns}")

print("\n--- COLUMN NAMES ---")
for index, column in enumerate(df.columns, start=1):
    print(f"{index:02d}. {column}")

print("\n--- EXPECTED COLUMN CHECK ---")
print(f"Missing expected columns : {missing_columns}")
print(f"Unexpected columns       : {unexpected_columns}")

if missing_columns:
    raise ValueError(
        "The dataset is missing required columns. "
        f"Missing: {missing_columns}"
    )

print("\n--- DATA TYPES ---")
print(df.dtypes)


# 5. TIMESTAMP UNDERSTANDING


# Converted only in memory for analysis. The raw CSV is not changed.
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

invalid_timestamps = int(df["Timestamp"].isna().sum())
start_time = df["Timestamp"].min()
end_time = df["Timestamp"].max()

if pd.notna(start_time) and pd.notna(end_time):
    collection_duration = end_time - start_time
    coverage_days = collection_duration.total_seconds() / 86400
else:
    coverage_days = np.nan

print("\n--- TIME COVERAGE ---")
print(f"Start              : {start_time}")
print(f"End                : {end_time}")
print(f"Calendar coverage  : {coverage_days:.2f} days")
print(f"Invalid timestamps : {invalid_timestamps}")


# 6. UNIQUE ENTITIES


smokehouses = sorted(
    df["Smokehouse_ID"].dropna().astype(str).unique().tolist())
layers = sorted(df["Layer_ID"].dropna().astype(str).unique().tolist())
risk_classes = sorted(
    df["Fire_Risk_Label"].dropna().astype(str).unique().tolist())

print("\n--- UNIQUE VALUES ---")
print(f"Smokehouses       : {smokehouses}")
print(f"Number of batches : {df['Batch_ID'].nunique()}")
print(f"Layers            : {layers}")
print(f"Fire-risk classes : {risk_classes}")


# 7. MISSING VALUES


missing_values = df.isnull().sum()
total_missing_values = int(missing_values.sum())

print("\n--- MISSING VALUES ---")
print(missing_values)
print(f"\nTotal missing values: {total_missing_values}")


# 8. DUPLICATE ANALYSIS


duplicate_rows = int(df.duplicated().sum())

observation_key_columns = [
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
    "Timestamp",
]

duplicate_observation_keys = int(
    df.duplicated(subset=observation_key_columns).sum()
)

print("\n--- DUPLICATES ---")
print(f"Duplicate complete rows    : {duplicate_rows}")
print(f"Duplicate observation keys : {duplicate_observation_keys}")


# 9. 10-MINUTE SAMPLING INTERVAL CHECK


interval_errors = 0
interval_values = []

group_columns = [
    "Smokehouse_ID",
    "Batch_ID",
    "Layer_ID",
]

for _, group in df.groupby(group_columns, dropna=False):
    group = group.sort_values("Timestamp")

    differences = (
        group["Timestamp"]
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60
    )

    interval_values.extend(differences.tolist())
    interval_errors += int((~np.isclose(differences, 10.0)).sum())

print("\n--- SAMPLING INTERVAL CHECK ---")
if interval_values:
    print(f"Minimum interval : {min(interval_values):.2f} minutes")
    print(f"Maximum interval : {max(interval_values):.2f} minutes")
else:
    print("No interval values available.")

print(f"Intervals different from 10 minutes: {interval_errors}")


# 10. INITIAL REGRESSION TARGET FORMULA CHECK


# Remaining_Drying_Time should equal:
# Actual_Total_Drying_Time - Elapsed_Drying_Hours
# A small tolerance is allowed for values rounded in the CSV.
calculated_remaining = (
    df["Actual_Total_Drying_Time"]
    - df["Elapsed_Drying_Hours"]
)

formula_difference = (
    calculated_remaining
    - df["Remaining_Drying_Time"]
).abs()

formula_tolerance = 0.011
formula_errors = int((formula_difference > formula_tolerance).sum())
maximum_formula_difference = float(formula_difference.max())

print("\n--- INITIAL REMAINING-TIME FORMULA CHECK ---")
print(
    "Maximum absolute difference : "
    f"{maximum_formula_difference:.6f} hours"
)
print(f"Rows outside tolerance      : {formula_errors}")


# 11. FIRE-RISK TARGET DISTRIBUTION


risk_counts = df["Fire_Risk_Label"].value_counts(dropna=False)
risk_percentages = (
    df["Fire_Risk_Label"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
)

risk_distribution = pd.DataFrame({
    "Count": risk_counts,
    "Percentage": risk_percentages,
})

print("\n--- FIRE-RISK CLASS DISTRIBUTION ---")
print(risk_distribution)


# 12. REGRESSION TARGET SUMMARY


remaining_time_summary = df["Remaining_Drying_Time"].describe()

print("\n--- REMAINING DRYING TIME TARGET ---")
print(remaining_time_summary)


# 13. IMPORTANT NUMERIC VARIABLES


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

numeric_summary = df[numeric_columns].describe().T

print("\n--- NUMERIC SUMMARY ---")
print(numeric_summary)


# 14. RECORD COUNTS


records_by_smokehouse = df["Smokehouse_ID"].value_counts().sort_index()
records_by_layer = df["Layer_ID"].value_counts().sort_index()
records_by_batch = df["Batch_ID"].value_counts().sort_index()

print("\n--- RECORDS BY SMOKEHOUSE ---")
print(records_by_smokehouse)

print("\n--- RECORDS BY LAYER ---")
print(records_by_layer)

print("\n--- RECORDS BY BATCH ---")
print(records_by_batch)


# 15. BATCH + LAYER OVERVIEW


batch_overview = (
    df
    .groupby(
        ["Smokehouse_ID", "Batch_ID", "Layer_ID"],
        dropna=False,
    )
    .agg(
        Start_Time=("Timestamp", "min"),
        End_Time=("Timestamp", "max"),
        Records=("Timestamp", "size"),
        Temp_Min=("Temperature_C", "min"),
        Temp_Max=("Temperature_C", "max"),
        Temp_Mean=("Temperature_C", "mean"),
        Humidity_Min=("Humidity_%", "min"),
        Humidity_Max=("Humidity_%", "max"),
        Humidity_Mean=("Humidity_%", "mean"),
        Actual_Total_Drying_Time=("Actual_Total_Drying_Time", "first"),
    )
    .reset_index()
)

batch_overview["Observed_Duration_Hours"] = (
    batch_overview["End_Time"]
    - batch_overview["Start_Time"]
).dt.total_seconds() / 3600


# 16. SAVE BATCH OVERVIEW


batch_overview.to_csv(BATCH_OVERVIEW_PATH, index=False)


# 17. SAVE TEXT SUMMARY


with open(SUMMARY_PATH, "w", encoding="utf-8") as report:
    report.write("COMPONENT 3 - DATA UNDERSTANDING SUMMARY\n")
    report.write("=" * 72 + "\n\n")

    report.write(f"Dataset: {DATA_PATH.name}\n\n")
    report.write(f"Rows: {rows:,}\n")
    report.write(f"Columns: {columns}\n\n")

    report.write("Columns:\n")
    for column in df.columns:
        report.write(f"- {column}\n")

    report.write("\nMissing expected columns:\n")
    report.write(str(missing_columns) + "\n")

    report.write("\nUnexpected columns:\n")
    report.write(str(unexpected_columns) + "\n")

    report.write(f"\nStart timestamp: {start_time}\n")
    report.write(f"End timestamp: {end_time}\n")
    report.write(f"Calendar coverage: {coverage_days:.2f} days\n")
    report.write(f"Invalid timestamps: {invalid_timestamps}\n")

    report.write(f"\nSmokehouses: {df['Smokehouse_ID'].nunique()}\n")
    report.write(f"Batches: {df['Batch_ID'].nunique()}\n")
    report.write(f"Layers: {df['Layer_ID'].nunique()}\n")

    report.write(f"\nTotal missing values: {total_missing_values}\n")
    report.write(f"Duplicate rows: {duplicate_rows}\n")
    report.write(
        f"Duplicate observation keys: {duplicate_observation_keys}\n"
    )
    report.write(f"10-minute interval errors: {interval_errors}\n")

    report.write("\nRemaining-time formula check:\n")
    report.write(
        "Maximum absolute difference: "
        f"{maximum_formula_difference:.6f} hours\n"
    )
    report.write(f"Rows outside tolerance: {formula_errors}\n")

    report.write("\nFire-Risk Distribution:\n")
    report.write(risk_distribution.to_string())

    report.write("\n\nRemaining Drying Time Summary:\n")
    report.write(remaining_time_summary.to_string())

    report.write("\n\nNumeric Summary:\n")
    report.write(numeric_summary.to_string())

    report.write("\n\nRecords by Smokehouse:\n")
    report.write(records_by_smokehouse.to_string())

    report.write("\n\nRecords by Layer:\n")
    report.write(records_by_layer.to_string())

    report.write("\n\nRecords by Batch:\n")
    report.write(records_by_batch.to_string())


# 18. COMPLETE


print("\n" + "=" * 72)
print("STAGE 1 COMPLETE")
print("=" * 72)

print(f"\nSummary saved to:\n{SUMMARY_PATH}")
print(f"\nBatch overview saved to:\n{BATCH_OVERVIEW_PATH}")
print("\nIMPORTANT: No raw dataset values were modified.")
