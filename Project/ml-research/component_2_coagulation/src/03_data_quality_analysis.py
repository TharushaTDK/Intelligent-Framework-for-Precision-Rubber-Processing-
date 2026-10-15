"""
Step 3: Data quality analysis
Project: Estate-Level DRC Guided Smart Latex Coagulation Recommendation Component

This script describes data-quality issues without modifying or deleting any records.
The current dataset is synthetic demo data, so these checks do not prove that its
values are scientifically valid for real latex coagulation.
"""

from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "latex_coagulation_dataset_final.csv"
REPORT_DIR = PROJECT_DIR / "outputs" / "reports"

NUMERIC_COLUMNS = [
    "drc",
    "latex_volume_L",
    "water_amount_L",
    "prepared_1pct_formic_acid_solution_mL",
    "coagulation_time_hours",
]
DATE_COLUMN = "record_date"
ID_COLUMN = "batch_id"

def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}\n"
            "Place the CSV in component_2_coagulation/data/raw/."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_FILE)
    original_rows = len(df)

    # Parse dates into a separate series; the source CSV is never overwritten.
    parsed_dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce") if DATE_COLUMN in df else None

    # Numeric conversion is used only for analysis, not written back to the raw file.
    numeric_data = {}
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            numeric_data[col] = pd.to_numeric(df[col], errors="coerce")

    quality_rows = [
        {"check": "row_count", "column": "ALL", "result": original_rows,
         "status": "INFO", "details": "Number of records in the raw dataset."},
        {"check": "duplicate_full_rows", "column": "ALL", "result": int(df.duplicated().sum()),
         "status": "PASS" if not df.duplicated().any() else "REVIEW",
         "details": "Exact duplicate rows; records are not removed."},
    ]

    if ID_COLUMN in df.columns:
        missing_ids = int(df[ID_COLUMN].isna().sum())
        duplicate_ids = int(df[ID_COLUMN].duplicated().sum())
        quality_rows.extend([
            {"check": "missing_ids", "column": ID_COLUMN, "result": missing_ids,
             "status": "PASS" if missing_ids == 0 else "REVIEW",
             "details": "Missing batch identifiers."},
            {"check": "duplicate_ids", "column": ID_COLUMN, "result": duplicate_ids,
             "status": "PASS" if duplicate_ids == 0 else "REVIEW",
             "details": "Repeated batch identifiers."},
        ])

    if parsed_dates is not None:
        bad_dates = int(parsed_dates.isna().sum())
        quality_rows.append({
            "check": "unparseable_dates", "column": DATE_COLUMN, "result": bad_dates,
            "status": "PASS" if bad_dates == 0 else "REVIEW",
            "details": "Date values that could not be parsed.",
        })

    range_rows = []
    for col in NUMERIC_COLUMNS:
        if col not in numeric_data:
            quality_rows.append({
                "check": "column_missing", "column": col, "result": "NOT FOUND",
                "status": "REVIEW", "details": "Expected numeric column is absent.",
            })
            continue

        series = numeric_data[col]
        missing_or_non_numeric = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        quality_rows.append({
            "check": "missing_or_non_numeric", "column": col,
            "result": missing_or_non_numeric,
            "status": "PASS" if missing_or_non_numeric == 0 else "REVIEW",
            "details": "Includes blank values and values that cannot be converted to numbers.",
        })

        # IQR is a screening flag only. It does not establish that a value is wrong.
        valid = series.dropna()
        if len(valid):
            q1 = float(valid.quantile(0.25))
            q3 = float(valid.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            flagged = int(((valid < lower) | (valid > upper)).sum())
            min_val = float(valid.min())
            max_val = float(valid.max())
            mean_val = float(valid.mean())
            median_val = float(valid.median())
        else:
            lower = upper = min_val = max_val = mean_val = median_val = None
            flagged = 0

        range_rows.append({
            "column": col,
            "valid_count": int(valid.count()),
            "unique_values": unique_count,
            "minimum": min_val,
            "maximum": max_val,
            "mean": mean_val,
            "median": median_val,
            "iqr_lower_screening_bound": lower,
            "iqr_upper_screening_bound": upper,
            "iqr_flagged_count": flagged,
            "interpretation": "Review flagged values; do not delete automatically.",
        })

    # Simple physical/semantic screening checks. These are review flags, not corrections.
    # Note: drc is a numerical reading, not a bounded 0-100 percentage,
    # so no percent-range screening check is applied to it.
    if "latex_volume_L" in numeric_data:
        s = numeric_data["latex_volume_L"]
        count = int((s <= 0).sum())
        quality_rows.append({
            "check": "non_positive_latex_volume", "column": "latex_volume_L",
            "result": count, "status": "PASS" if count == 0 else "REVIEW",
            "details": "Latex batch volume should be positive.",
        })
    for col in ["water_amount_L", "prepared_1pct_formic_acid_solution_mL",
                "coagulation_time_hours"]:
        if col in numeric_data:
            s = numeric_data[col]
            count = int((s < 0).sum())
            quality_rows.append({
                "check": "negative_values", "column": col, "result": count,
                "status": "PASS" if count == 0 else "REVIEW",
                "details": "Negative target values need review.",
            })

    pd.DataFrame(quality_rows).to_csv(
        REPORT_DIR / "03_data_quality_checks.csv", index=False
    )
    pd.DataFrame(range_rows).to_csv(
        REPORT_DIR / "03_numeric_range_summary.csv", index=False
    )

    notes = [
        "STEP 3: DATA QUALITY ANALYSIS",
        f"Input file: {RAW_FILE.name}",
        f"Rows checked: {original_rows}",
        "",
        "Outputs:",
        "- 03_data_quality_checks.csv",
        "- 03_numeric_range_summary.csv",
        "",
        "Important:",
        "- The raw CSV was read only; this script does not edit it.",
        "- IQR flags identify unusual values for review, not confirmed errors.",
        "- No rows are automatically deleted or changed.",
        "- The dataset is synthetic demo data. These checks do not validate",
        "  real-world chemical amounts or prove scientific accuracy.",
    ]
    (REPORT_DIR / "03_data_quality_notes.txt").write_text(
        "\n".join(notes) + "\n", encoding="utf-8"
    )

    print("Step 3 completed.")
    print(f"Rows checked: {original_rows}")
    print(f"Reports saved in: {REPORT_DIR}")
    print("Raw dataset was not modified.")

if __name__ == "__main__":
    main()
