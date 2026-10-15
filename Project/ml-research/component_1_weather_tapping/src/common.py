from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "RubberSense_RawStyle_Synthetic_Dataset.xlsx"
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
REG_TASKS = {
    "drc_prediction": (
        "latex_drc_pct",
        ["temperature_c", "relative_humidity_pct", "rainfall_last_6h_mm", "forecast_rain_next_6h_mm",
         "rainfall_last_3days_mm", "wind_speed_kmh", "soil_moisture_pct", "soil_ph", "estate_id",
         "tree_count", "tapping_system", "days_since_last_tap", "previous_collection_yield_litres",
         "month", "day_of_year"],
        1.0,
    ),
    "price_estimation": (
        "price_lkr_per_kg",
        ["latex_drc_pct", "temperature_c", "relative_humidity_pct", "rainfall_last_6h_mm",
         "soil_moisture_pct", "soil_ph", "tree_count", "tapping_system", "month", "day_of_year"],
        9.0,
    ),
}
CLS_TASK = "tapping_advisory"
CLS_TARGET = "tapping_suitability_proxy"
CLS_FEATURES = ["temperature_c", "relative_humidity_pct", "rainfall_last_6h_mm", "forecast_rain_next_6h_mm",
                "rainfall_last_3days_mm", "wind_speed_kmh", "soil_moisture_pct", "soil_ph", "tree_count",
                "tapping_system", "days_since_last_tap", "month", "day_of_year"]
CLS_CLASSES = ["High", "Low", "Moderate"]

DATASET_PROVENANCE = "synthetic; not real-world evidence"
TARGET_WARNING = ("Rule-derived proxy labels with ~30% simulated label noise (seed 340); "
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
    return pd.read_excel(RAW, sheet_name="RawStyle_Data_600")

def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def clean_basic(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    # Keep invalid observations visible in the raw file; mark them missing only in processed copy.
    for col in ["rainfall_last_6h_mm", "forecast_rain_next_6h_mm", "rainfall_last_3days_mm", "wind_speed_kmh"]:
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
    # Feature engineering from timestamps.
    ts = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    # Illustrative labels only; not validated tapping advice.
    rain = pd.to_numeric(df["rainfall_last_6h_mm"], errors="coerce")
    forecast = pd.to_numeric(df["forecast_rain_next_6h_mm"], errors="coerce")
    humidity = pd.to_numeric(df["relative_humidity_pct"], errors="coerce")
    soil = pd.to_numeric(df["soil_moisture_pct"], errors="coerce")
    low = (rain >= 8) | (forecast >= 12) | (humidity >= 94)
    moderate = (~low) & ((rain >= 2.5) | (forecast >= 7) | (humidity >= 88) | (soil >= 58))
    base_labels = np.select(
        [low.fillna(False), moderate.fillna(False)],
        ["Low", "Moderate"], default="High"
    )
    # Simulate imperfect advisory annotations for demonstration only. A fixed seed
    # makes the synthetic benchmark reproducible; replace with expert estate labels.
    rng = np.random.default_rng(340)
    uncertain = rng.random(len(df)) < 0.30
    classes = np.array(["Low", "Moderate", "High"])
    for i in np.flatnonzero(uncertain):
        alternatives = classes[classes != base_labels[i]]
        base_labels[i] = rng.choice(alternatives)
    df["tapping_suitability_proxy"] = base_labels
    return df
