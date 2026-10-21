from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Prepared by src/00_prepare_realistic_dataset.py from the condition-matched farmer
# workbook in data/raw/ (which stays unchanged). The earlier synthetic workbook,
# RubberSense_RawStyle_Synthetic_Dataset.xlsx, is no longer used for training.
RAW = ROOT / "data" / "validated" / "component1_realistic_prepared.csv"
# Read-only project inputs stay in place regardless of training-only mode.
SPLITS = ROOT / "data" / "splits"

# Training-only mode. When RUBBERSENSE_TRAIN_ONLY_DIR is set, every generated
# processed copy, report, figure and model artifact is redirected under that
# directory so the project's final outputs/ and artifacts/ folders are never
# created or written. Without the env var the original behaviour is unchanged.
TRAIN_ONLY_DIR = os.environ.get("RUBBERSENSE_TRAIN_ONLY_DIR")
if TRAIN_ONLY_DIR:
    _OUT = Path(TRAIN_ONLY_DIR)
    PROCESSED = _OUT / "data" / "processed"
    REPORTS = _OUT / "outputs" / "reports"
    FIGURES = _OUT / "outputs" / "figures"
    ARTIFACTS = _OUT / "artifacts"
else:
    PROCESSED = ROOT / "data" / "processed"
    REPORTS = ROOT / "outputs" / "reports"
    FIGURES = ROOT / "outputs" / "figures"
    ARTIFACTS = ROOT / "artifacts"

REG_MODELS = ARTIFACTS / "models" / "regression"
CLS_MODELS = ARTIFACTS / "models" / "classification"
PIPELINES = ARTIFACTS / "pipelines"
PREPROCESSING = ARTIFACTS / "preprocessing"

# The four artefact roles shared by every task, mirroring the reference project.
MODEL_ROLES = ("baseline", "candidates", "tuned", "final")

# Task registry: regression tasks (target, feature list, tolerance) and the
# classification task. Single source of truth reused by every model stage.
#
# Inputs follow what the farmer records actually contain. Tree count, tapping system,
# days since last tap and previous yield are absent from every farmer record, so they
# are no longer features. The Metrolac reading and latex temperature are left out of
# the DRC model on purpose: together they give DRC exactly through the Kudaligama
# chart, so a model using them would only relearn the chart.
REG_TASKS = {
    "drc_prediction": (
        "latex_drc_pct",
        ["temperature_c", "relative_humidity_pct", "rainfall_last_6h_mm", "forecast_rain_next_6h_mm",
         "rainfall_last_3days_mm", "wind_speed_kmh", "soil_moisture_pct", "soil_ph",
         "visible_latex_condition", "estate_id", "month", "day_of_year"],
        1.0,
    ),
    # Price per kg of field latex = farm-gate price x DRC / 100. Given the true DRC a
    # model just relearns that formula, but in the app the DRC is itself an estimate.
    # So the latex_drc_pct input here is the DRC as the app would estimate it (see
    # src/04b_estimate_drc_for_price.py); the target still uses the true DRC.
    "price_estimation": (
        "price_lkr_per_kg",
        ["latex_drc_pct", "temperature_c", "relative_humidity_pct", "rainfall_last_6h_mm",
         "soil_moisture_pct", "soil_ph", "month", "day_of_year"],
        5.0,
    ),
}
CLS_TASK = "tapping_advisory"
CLS_TARGET = "tapping_suitability_proxy"
CLS_FEATURES = ["temperature_c", "relative_humidity_pct", "rainfall_last_6h_mm", "forecast_rain_next_6h_mm",
                "rainfall_last_3days_mm", "wind_speed_kmh", "soil_moisture_pct", "soil_ph",
                "month", "day_of_year"]
CLS_CLASSES = ["High", "Low", "Moderate"]

DATASET_PROVENANCE = ("condition-matched farmer records (5 estates, Jun-Sep 2026) with Open-Meteo "
                      "weather; DRC/Metrolac values normalized by the dataset author; latex price "
                      "derived from farm-gate levels (Jun-Sep 2026 levels estimated)")
TARGET_WARNING = ("Rule-derived proxy labels from rainfall thresholds, judged on rain that actually "
                  "fell; the model sees only the forecast issued beforehand (no added noise). "
                  "NOT expert-verified tapping advice.")

def ensure_dirs():
    for p in [PROCESSED, SPLITS, REPORTS, FIGURES, REG_MODELS, CLS_MODELS, PIPELINES, PREPROCESSING]:
        p.mkdir(parents=True, exist_ok=True)

def task_root(kind, task):
    """Return the per-task model root, e.g. artifacts/models/regression/drc_prediction."""
    base = REG_MODELS if kind == "regression" else CLS_MODELS
    return base / task

def role_dir(kind, task, role, name=None):
    """Return a role folder (baseline|candidates|tuned|final) and create it.

    For baseline/candidates/tuned a family ``name`` is required and becomes a
    sub-folder; for final the folder is used directly.
    """
    if role not in MODEL_ROLES:
        raise ValueError(f"unknown model role: {role!r}")
    path = task_root(kind, task) / role
    if role != "final":
        if not name:
            raise ValueError(f"role {role!r} requires a model name")
        path = path / name
    path.mkdir(parents=True, exist_ok=True)
    return path

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_frame(frame):
    """Deterministic hash of a DataFrame's values (column order respected)."""
    arr = np.asarray(frame)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

def load_raw():
    """Rows the weather-based models can train on (the farmer records).

    The index is reset so the stored split files keep pointing at the same rows on
    every run.
    """
    df = pd.read_csv(RAW)
    return df[df["usable_for_weather_models"].astype(bool)].reset_index(drop=True)

def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def clean_basic(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    # Keep invalid observations visible in the raw file; mark them missing only in processed copy.
    for col in ["rainfall_last_6h_mm", "forecast_rain_next_6h_mm", "rainfall_next_6h_observed_mm",
                "rainfall_last_3days_mm", "wind_speed_kmh"]:
        if col in df:
            df.loc[pd.to_numeric(df[col], errors="coerce") < 0, col] = np.nan
    if "soil_ph" in df:
        df.loc[~pd.to_numeric(df["soil_ph"], errors="coerce").between(0, 14), "soil_ph"] = np.nan
    for col in ["soil_moisture_pct", "relative_humidity_pct"]:
        if col in df:
            df.loc[~pd.to_numeric(df[col], errors="coerce").between(0, 100), col] = np.nan
    if "temperature_c" in df:
        df.loc[~pd.to_numeric(df["temperature_c"], errors="coerce").between(-5, 60), "temperature_c"] = np.nan
    if "latex_drc_pct" in df:
        df.loc[~pd.to_numeric(df["latex_drc_pct"], errors="coerce").between(0, 100), "latex_drc_pct"] = np.nan
    # Normalize category whitespace/case without changing source workbook.
    if "tapping_system" in df:
        df["tapping_system"] = df["tapping_system"].astype("string").str.strip().str.upper()
    for col in ["visible_latex_condition", "estate_id"]:
        if col in df:
            df[col] = df[col].astype("string").str.strip()
    # Feature engineering from timestamps.
    ts = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    df["tapping_suitability_proxy"] = tapping_proxy_label(df)
    return df


# Proxy tapping label, re-fitted to this wet-season data. Humidity is 95-100% in almost
# every record, so it no longer separates good and bad mornings and is not used. What
# matters for tapping is whether the bark is wet from recent rain, and whether heavy
# rain is coming soon enough to wash latex out of the cups.
#   Low:      >= 5 mm in the last 6 h (bark wet) or >= 5 mm in the next 6 h (wash-out)
#   Moderate: >= 1.5 mm in the last 6 h, >= 2 mm in the next 6 h, or >= 60 mm over 3 days
#   High:     otherwise
# 5 mm is roughly the wettest sixth of mornings here. No noise is added.
#
# "Next 6 h" is judged on the rain that actually fell (rainfall_next_6h_observed_mm),
# while the model only gets the forecast issued beforehand (forecast_rain_next_6h_mm).
# That gap is real forecast error, so the model cannot score perfectly by relearning
# the thresholds. The label is still a proxy, not expert advice.
TAPPING_LOW_RAIN_6H_MM = 5.0
TAPPING_LOW_NEXT_6H_MM = 5.0
TAPPING_MODERATE_RAIN_6H_MM = 1.5
TAPPING_MODERATE_NEXT_6H_MM = 2.0
TAPPING_MODERATE_RAIN_3D_MM = 60.0


def tapping_proxy_label(df):
    rain = pd.to_numeric(df["rainfall_last_6h_mm"], errors="coerce")
    upcoming = pd.to_numeric(df["rainfall_next_6h_observed_mm"], errors="coerce")
    rain_3d = pd.to_numeric(df["rainfall_last_3days_mm"], errors="coerce")
    low = (rain >= TAPPING_LOW_RAIN_6H_MM) | (upcoming >= TAPPING_LOW_NEXT_6H_MM)
    moderate = (~low) & (
        (rain >= TAPPING_MODERATE_RAIN_6H_MM)
        | (upcoming >= TAPPING_MODERATE_NEXT_6H_MM)
        | (rain_3d >= TAPPING_MODERATE_RAIN_3D_MM)
    )
    labels = np.select([low.fillna(False), moderate.fillna(False)], ["Low", "Moderate"], default="High")
    # Without the weather needed to judge, there is no label rather than a guess.
    no_weather = rain.isna() & upcoming.isna()
    return pd.Series(np.where(no_weather, None, labels), index=df.index, dtype="object")
