"""Step 4: Clean the raw dataset and save the final dataset used by the later steps."""
from pathlib import Path

import pandas as pd

from drc_ready_reckoner import drc_percent

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "latex_coagulation_dataset_raw.csv"
OUT = ROOT / "data" / "processed"
REP = ROOT / "outputs" / "reports"
FINAL = OUT / "latex_coagulation_dataset_final.csv"

KEEP = [
    "batch_id",
    "record_date",
    "metrolac_reading",
    "latex_temperature_C",
    "drc",
    "latex_volume_L",
    "water_amount_L",
    "prepared_1pct_formic_acid_solution_mL",
    "coagulation_time_minutes",
]
NUMERIC = KEEP[2:]
WHOLE_NUMBERS = ["metrolac_reading", "prepared_1pct_formic_acid_solution_mL", "coagulation_time_minutes"]
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"]


def parse_dates(values: pd.Series) -> pd.Series:
    # the raw file mixes three date formats, anything else stays empty
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        parsed = parsed.fillna(pd.to_datetime(values, format=fmt, errors="coerce"))
    return parsed


def invalid_reason(row: pd.Series) -> str:
    if pd.notna(row["drc"]) and not 0 < row["drc"] <= 100:
        return "drc must be above 0 and up to 100 %"
    if row["metrolac_reading"] <= 0:
        return "metrolac reading must be positive"
    if not 15 <= row["latex_temperature_C"] <= 35:
        return "temperature must be within the chart range 15-35 C"
    if row["latex_volume_L"] <= 0:
        return "latex volume must be positive"
    if row["water_amount_L"] < 0:
        return "water amount cannot be negative"
    if row["prepared_1pct_formic_acid_solution_mL"] <= 0:
        return "acid solution amount must be positive"
    if row["coagulation_time_minutes"] <= 0:
        return "coagulation time must be positive"
    return ""


def main():
    if not RAW.exists():
        raise FileNotFoundError(RAW)
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)

    # read everything as text so nothing is changed while loading
    df = pd.read_csv(RAW, dtype=str)
    steps = [{"step": "raw dataset", "rows": len(df), "removed": 0,
              "note": f"{df.shape[1]} columns"}]
    removed = []

    dropped_cols = [c for c in df.columns if c not in KEEP]
    df = df[KEEP].copy()
    steps.append({"step": "drop unwanted columns", "rows": len(df), "removed": 0,
                  "note": "dropped: " + ", ".join(dropped_cols)})

    # tidy ids, units written inside numbers and mixed date formats
    df["batch_id"] = df["batch_id"].str.strip().str.upper()
    for c in NUMERIC:
        text = df[c].str.replace(r"[^0-9.\-]", "", regex=True)
        df[c] = pd.to_numeric(text, errors="coerce")
    df["record_date"] = parse_dates(df["record_date"].str.strip())
    steps.append({"step": "fix formats", "rows": len(df), "removed": 0,
                  "note": "trimmed batch ids, removed units (%, L, mL, °C, min) from numbers, unified dates"})

    dup = df.duplicated()
    removed += [{"batch_id": b, "reason": "duplicate row"} for b in df.loc[dup, "batch_id"]]
    df = df[~dup]
    steps.append({"step": "remove duplicate rows", "rows": len(df), "removed": int(dup.sum()), "note": ""})

    missing = df.drop(columns="drc").isna().any(axis=1)
    removed += [{"batch_id": b, "reason": "missing or unreadable value"} for b in df.loc[missing, "batch_id"]]
    df = df[~missing]
    steps.append({"step": "remove rows with missing values", "rows": len(df),
                  "removed": int(missing.sum()), "note": "empty drc is filled later, other values are not filled in"})

    reasons = df.apply(invalid_reason, axis=1)
    bad = reasons != ""
    removed += [{"batch_id": b, "reason": r} for b, r in zip(df.loc[bad, "batch_id"], reasons[bad])]
    df = df[~bad]
    steps.append({"step": "remove impossible values", "rows": len(df), "removed": int(bad.sum()), "note": ""})

    # empty drc comes from the ready-reckoner chart
    empty = df["drc"].isna()
    df.loc[empty, "drc"] = [drc_percent(m, t) for m, t in
                            zip(df.loc[empty, "metrolac_reading"], df.loc[empty, "latex_temperature_C"])]
    steps.append({"step": "fill empty drc", "rows": len(df), "removed": 0,
                  "note": f"{int(empty.sum())} values from the temperature corrected Metrolac ready-reckoner (Kudaligama et al. 2014)"})

    if df["batch_id"].duplicated().any():
        raise ValueError("Some batch ids still appear more than once after cleaning.")

    df[WHOLE_NUMBERS] = df[WHOLE_NUMBERS].astype(int)
    df = df.sort_values("batch_id").reset_index(drop=True)
    df.to_csv(FINAL, index=False, date_format="%Y-%m-%d", lineterminator="\n")
    steps.append({"step": "final dataset", "rows": len(df), "removed": 0, "note": FINAL.name})

    pd.DataFrame(steps).to_csv(REP / "04_preprocessing_summary.csv", index=False)
    pd.DataFrame(removed).to_csv(REP / "04_removed_rows.csv", index=False)
    print(pd.DataFrame(steps).to_string(index=False))
    print("Step 4 complete. Final dataset:", FINAL)


if __name__ == "__main__":
    main()
