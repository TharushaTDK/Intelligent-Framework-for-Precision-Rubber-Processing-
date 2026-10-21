"""Stage 00: prepare the condition-matched farmer dataset for the Component 1 workflow.

Reads the supplied workbook (left unchanged in data/raw/) and writes one prepared
copy that the rest of the numbered workflow loads through ``common.load_raw()``.

What this stage does:
  * keeps every row, but marks which ones can train weather-based models. Only the
    farmer records have the Open-Meteo weather join; the 600 model-development rows
    are the earlier synthetic set and have no weather at all.
  * maps the workbook's column names to the names the models and backend use.
  * fills the latex price from data/reference/latex_farmgate_price_reference.csv:

        price per kg of field latex = farm-gate latex price (per kg dry rubber) x DRC / 100

    Field latex is bought on its dry rubber content, so watery (low-DRC) latex earns
    less per kg. The farm-gate level comes from the Rubber Development Department where
    a published figure exists, and is marked as an estimate where it does not.
  * adds a real rain forecast for the next 6 hours. The workbook's
    rainfall_next_6h_observed_mm is what actually fell afterwards, which nobody knows
    at tapping time; using it as an input made the tapping model look perfect. It is
    kept as the outcome the tapping label is judged on, and the model input becomes
    the Open-Meteo forecast for the same grid point that was issued a day earlier
    (data/reference/openmeteo_rain_forecast_issued_1day_before.csv, from the
    Previous Runs API, variable precipitation_previous_day1).

Nothing here invents weather, soil or DRC values.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "Combined_Rubber_Latex_Dataset_Condition_Matched.xlsx"
PRICE_REFERENCE = ROOT / "data" / "reference" / "latex_farmgate_price_reference.csv"
RAIN_FORECAST = ROOT / "data" / "reference" / "openmeteo_rain_forecast_issued_1day_before.csv"
RAIN_FORECAST_URL = (
    "https://previous-runs-api.open-meteo.com/v1/forecast?latitude=6.7838&longitude=80.1515"
    "&start_date=2026-05-29&end_date=2026-10-02&timezone=Asia%2FColombo"
    "&hourly=precipitation,precipitation_previous_day1"
)
PREPARED = ROOT / "data" / "validated" / "component1_realistic_prepared.csv"
REPORT = ROOT / "outputs" / "reports" / "00_dataset_preparation_report.md"

# workbook column -> name used by the models (and the backend request schemas)
COLUMN_MAP = {
    "estate_name": "estate_id",
    "weather_temperature_c": "temperature_c",
    "weather_relative_humidity_pct": "relative_humidity_pct",
    "weather_wind_speed_kmh": "wind_speed_kmh",
    # The farmer's own soil moisture reading is the model input; the Open-Meteo
    # estimate is kept alongside it for reference only.
    "soil_moisture_pct_farmer_recorded": "soil_moisture_pct",
    "openmeteo_soil_moisture_pct": "modelled_soil_moisture_pct",
}

# Workbook columns that duplicate others, or were renamed above.
DROP = [
    "soil_moisture_pct", "relative_humidity_pct", "wind_speed_kmh", "rain_mm",
    "soil_temperature_c", "price_lkr_per_kg", "price_lkr_per_litre",
]


def load_price_reference():
    ref = pd.read_csv(PRICE_REFERENCE, dtype={"month": str})
    if ref["month"].duplicated().any():
        raise ValueError("price reference has duplicate months")
    return ref.set_index("month")


def forecast_next_6h(timestamps):
    """Forecast rain for hours h+1..h+6 after each record's hour h.

    Same window and the same floored hour as the workbook's observed column, so the
    two differ only by forecast error.
    """
    fc = pd.read_csv(RAIN_FORECAST, parse_dates=["timestamp_local"])
    hourly = fc.set_index("timestamp_local")["forecast_rain_mm_issued_1day_before"]
    next_6h = sum(hourly.shift(-k) for k in range(1, 7)).round(2)
    return pd.to_datetime(timestamps).dt.floor("h").map(next_6h)


def main():
    raw = pd.read_excel(SOURCE, sheet_name="Combined_Dataset")
    df = raw.drop(columns=[c for c in DROP if c in raw.columns]).rename(columns=COLUMN_MAP)

    # Two kinds of rows, told apart by their data-quality note.
    is_farmer = df["data_quality_note"].astype(str).str.startswith("Farmer workbook record")
    df["record_group"] = np.where(is_farmer, "farmer", "model_development")
    weather_cols = ["temperature_c", "relative_humidity_pct", "rainfall_last_6h_mm", "wind_speed_kmh"]
    df["usable_for_weather_models"] = df[weather_cols].notna().all(axis=1)

    # Tidy categories without altering their meaning.
    df["tapping_system"] = df["tapping_system"].astype("string").str.strip().str.upper()
    df["visible_latex_condition"] = df["visible_latex_condition"].astype("string").str.strip()

    # Price from DRC and the month's farm-gate latex level.
    ref = load_price_reference()
    month = pd.to_datetime(df["timestamp_local"]).dt.strftime("%Y-%m")
    missing_months = sorted(set(month) - set(ref.index))
    if missing_months:
        raise ValueError(f"no farm-gate price for months: {missing_months}")
    df["farmgate_latex_lkr_per_kg_dry_rubber"] = month.map(ref["farmgate_latex_lkr_per_kg_dry_rubber"]).astype(float)
    df["price_basis_status"] = month.map(ref["status"])
    df["price_lkr_per_kg"] = (df["farmgate_latex_lkr_per_kg_dry_rubber"] * df["latex_drc_pct"] / 100).round(2)

    # What a farmer could actually see at tapping time.
    df["forecast_rain_next_6h_mm"] = forecast_next_6h(df["timestamp_local"])
    usable_fc = df["usable_for_weather_models"] & df["rainfall_next_6h_observed_mm"].notna()
    fc_corr = df.loc[usable_fc, ["forecast_rain_next_6h_mm", "rainfall_next_6h_observed_mm"]].corr().iloc[0, 1]

    PREPARED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREPARED, index=False)

    # ---------------- report ----------------
    usable = df[df["usable_for_weather_models"]]
    by_condition = usable.groupby("visible_latex_condition").agg(
        rows=("latex_drc_pct", "size"),
        mean_drc_pct=("latex_drc_pct", "mean"),
        mean_price_lkr_per_kg=("price_lkr_per_kg", "mean"),
    ).round(2)
    by_status = df.groupby("price_basis_status").size()

    lines = [
        "# Stage 00 - dataset preparation",
        "",
        f"Source: `{SOURCE.relative_to(ROOT)}` (sheet `Combined_Dataset`, unchanged).",
        f"Prepared copy: `{PREPARED.relative_to(ROOT)}`.",
        "",
        "## Rows",
        f"- Total rows: {len(df)}",
        f"- Farmer records (Open-Meteo weather joined): {int(is_farmer.sum())}",
        f"- Model-development records (earlier synthetic set, no weather): {int((~is_farmer).sum())}",
        f"- Usable for weather-based models: {int(df['usable_for_weather_models'].sum())}",
        "",
        "## Latex price",
        "Price per kg of field latex = farm-gate latex price (Rs per kg dry rubber) x DRC / 100.",
        "Field latex is paid on its dry rubber content, so watery (low-DRC) latex earns less per kg.",
        "",
        "Farm-gate level used, by source status:",
        "",
        "```",
        by_status.to_frame("rows").to_string(),
        "```",
        "",
        "Usable rows by visible latex condition:",
        "",
        "```",
        by_condition.to_string(),
        "```",
        "",
        "## Rain forecast for the next 6 hours",
        "The workbook's `rainfall_next_6h_observed_mm` is rain that fell afterwards, so it is kept only",
        "as the outcome the tapping label is judged on. The model input `forecast_rain_next_6h_mm` is",
        "the Open-Meteo forecast for the same grid point issued about a day earlier:",
        "",
        f"    {RAIN_FORECAST_URL}",
        "",
        f"Correlation between forecast and observed next-6 h rain on usable rows: {fc_corr:.2f}.",
        "A forecast issued a day ahead is a little less accurate than the one a farmer sees at dawn,",
        "so model scores here are, if anything, slightly pessimistic.",
        "",
        "## Caveats carried over from the workbook",
        "- DRC and Metrolac readings were normalized to valid Kudaligama chart entries (near 41%) for",
        "  model development; they are not original field measurements.",
        "- Visible latex condition was aligned with DRC by the dataset author (Watery lower, Normal 38-42%).",
        "- Soil moisture used as a model input is the farmer-recorded value; the Open-Meteo value is",
        "  modelled, not a sensor reading.",
        "- June-September 2026 farm-gate levels are estimates (see the price reference file).",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
