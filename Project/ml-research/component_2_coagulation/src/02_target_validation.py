from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = BASE_DIR / "data" / "raw" / "latex_coagulation_dataset_raw.csv"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
VALIDATED_DIR = BASE_DIR / "data" / "validated"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
VALIDATED_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_FILE.exists():
    raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

df = pd.read_csv(RAW_FILE)

targets = {
    "water_amount_L": "Model 1 target: water amount (L)",
    "prepared_1pct_formic_acid_solution_mL": "Model 1 target: prepared 1% formic acid solution amount (mL)",
    "coagulation_time_hours": "Model 2 target: coagulation time (hours)",
}
expected_inputs = [
    "drc",
    "latex_volume_L",
]

missing_targets = [c for c in targets if c not in df.columns]
missing_inputs = [c for c in expected_inputs if c not in df.columns]
if missing_targets or missing_inputs:
    raise ValueError(
        f"Missing target columns: {missing_targets}; "
        f"missing expected input columns: {missing_inputs}"
    )

rows = []
for column, description in targets.items():
    values = pd.to_numeric(df[column], errors="coerce")
    non_numeric = int(values.isna().sum() - df[column].isna().sum())
    missing = int(df[column].isna().sum())
    rows.append({
        "target_column": column,
        "description": description,
        "total_rows": int(len(df)),
        "missing_values": missing,
        "non_numeric_values": non_numeric,
        "unique_values": int(values.nunique(dropna=True)),
        "minimum": values.min(),
        "maximum": values.max(),
        "target_available_for_training": (
            "CHECK" if missing or non_numeric or values.nunique(dropna=True) < 2
            else "STRUCTURE_OK_NEEDS_DOMAIN_VALIDATION"
        ),
    })

target_report = pd.DataFrame(rows)
target_report.to_csv(REPORT_DIR / "02_target_validation.csv", index=False)

# Check whether expected numeric inputs can be read as numbers.
input_rows = []
for column in expected_inputs:
    original = df[column]
    numeric = pd.to_numeric(original, errors="coerce")
    input_rows.append({
        "input_column": column,
        "missing_values": int(original.isna().sum()),
        "non_numeric_values": int(numeric.isna().sum() - original.isna().sum()),
        "unique_values": int(numeric.nunique(dropna=True)),
        "status": "CHECK" if numeric.isna().any() else "STRUCTURE_OK_NEEDS_DOMAIN_VALIDATION",
    })
pd.DataFrame(input_rows).to_csv(
    REPORT_DIR / "02_input_column_checks.csv", index=False
)

notes = [
    "TARGET VALIDATION NOTES",
    "=" * 30,
    "Model 1 predicts water_amount_L and prepared_1pct_formic_acid_solution_mL.",
    "Model 2 predicts coagulation_time_hours.",
    "Expected available inputs in this dataset: drc, latex_volume_L, water_amount_L, prepared_1pct_formic_acid_solution_mL.",
    "",
    "This script checks column presence and basic numeric completeness only.",
    "It does not validate whether the target values are scientifically correct.",
    "The current dataset is synthetic demonstration data; do not treat model results from it as real-world evidence.",
]
(REPORT_DIR / "02_target_validation_notes.txt").write_text(
    "\n".join(notes), encoding="utf-8"
)

print("Target validation checks completed.")
print(f"Rows checked: {len(df)}")
print(f"Reports saved to: {REPORT_DIR}")
print("Review the CSV reports before continuing.")

# Save a validated copy without silently changing or deleting observations.
df.to_csv(VALIDATED_DIR / "Component2_Dataset_Validated.csv", index=False)
print("Validated dataset saved:", VALIDATED_DIR / "Component2_Dataset_Validated.csv")
