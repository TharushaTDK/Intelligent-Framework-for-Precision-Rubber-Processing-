from pathlib import Path
import pandas as pd

# Project directories
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = BASE_DIR / "data" / "raw" / "latex_coagulation_dataset_raw.csv"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Raw dataset not found: {RAW_FILE}\n"
        "Place the CSV inside component_2_coagulation/data/raw/."
    )

# Load the raw dataset without changing the original file.
df = pd.read_csv(RAW_FILE)

required_columns = [
    "batch_id",
    "record_date",
    "drc",
    "latex_volume_L",
    "water_amount_L",
    "prepared_1pct_formic_acid_solution_mL",
    "coagulation_time_hours",
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

# Dataset overview report
overview = [
    "LATEX COAGULATION DATASET OVERVIEW",
    "=" * 40,
    f"Rows: {df.shape[0]}",
    f"Columns: {df.shape[1]}",
    f"Duplicate batch IDs: {df['batch_id'].duplicated().sum()}",
    f"Duplicate complete rows: {df.duplicated().sum()}",
    f"Total missing values: {df.isna().sum().sum()}",
    "",
    "Column names and data types:",
    df.dtypes.to_string(),
    "",
    "First five records:",
    df.head().to_string(index=False),
]
(REPORT_DIR / "01_dataset_overview.txt").write_text(
    "\n".join(overview), encoding="utf-8"
)

# Column-level summary
column_summary = pd.DataFrame({
    "column_name": df.columns,
    "data_type": [str(df[col].dtype) for col in df.columns],
    "missing_values": [int(df[col].isna().sum()) for col in df.columns],
    "unique_values": [int(df[col].nunique(dropna=False)) for col in df.columns],
})
column_summary.to_csv(REPORT_DIR / "01_column_summary.csv", index=False)

# Descriptive summary
df.describe(include="all").transpose().to_csv(
    REPORT_DIR / "01_batch_summary.csv"
)

# Basic structural checks; these do not validate scientific correctness.
checks = pd.DataFrame([
    {
        "check": "At least 2,000 rows",
        "result": int(len(df)),
        "status": "PASS" if len(df) >= 2000 else "CHECK",
    },
    {
        "check": "Duplicate batch IDs",
        "result": int(df["batch_id"].duplicated().sum()),
        "status": "PASS" if not df["batch_id"].duplicated().any() else "CHECK",
    },
    {
        "check": "Missing values",
        "result": int(df.isna().sum().sum()),
        "status": "PASS" if not df.isna().any().any() else "CHECK",
    },
    {
        "check": "Duplicate complete rows",
        "result": int(df.duplicated().sum()),
        "status": "PASS" if not df.duplicated().any() else "CHECK",
    },
    {
        "check": "Required columns present",
        "result": len(required_columns),
        "status": "PASS" if not missing_columns else "CHECK",
    },
])
checks.to_csv(REPORT_DIR / "01_data_understanding_checks.csv", index=False)

print("Data understanding completed.")
print(f"Dataset shape: {df.shape}")
print(f"Reports saved to: {REPORT_DIR}")
print("Note: these checks assess dataset structure, not scientific validity.")
