from pathlib import Path
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# COMPONENT 3 - STAGE 06: TEMPORAL & LAYER ANALYSIS
# ============================================================
# Analysis only:
# - no model training
# - no final ML feature dataset
# - no train/test split
# - no scaling / encoding / SMOTE
# - no future-data feature construction
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "Component3_Dataset_Preprocessed.csv"
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures" / "06_temporal_layer"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "06_temporal_layer_report.txt"
CHECKS_PATH = REPORT_DIR / "06_temporal_layer_checks.csv"
INTERVAL_PATH = REPORT_DIR / "06_sampling_interval_summary.csv"
LAG_PATH = REPORT_DIR / "06_exact_lag_analysis.csv"
PROGRESS_PATH = REPORT_DIR / "06_progress_layer_summary.csv"
SEQUENCE_PATH = REPORT_DIR / "06_sequence_temporal_summary.csv"
FIRE_CONTEXT_PATH = REPORT_DIR / "06_fire_risk_temporal_context.csv"
TRANSITION_PATH = REPORT_DIR / "06_fire_risk_transition_matrix.csv"

EXPECTED_ROWS = 24463
EXPECTED_COLUMNS = [
    "Smokehouse_ID", "Timestamp", "Batch_ID", "Layer_ID", "Temperature_C",
    "Humidity_%", "Elapsed_Drying_Hours", "Number_of_Sheets", "Layer_Area_m2",
    "Temperature_Change", "Humidity_Change", "Temperature_Trend", "Humidity_Trend",
    "Actual_Total_Drying_Time", "Remaining_Drying_Time", "Fire_Risk_Label",
    "Fire_Risk_Assessment_Timestamp",
]
KEY = ["Smokehouse_ID", "Batch_ID", "Layer_ID"]

print("=" * 78)
print("COMPONENT 3 - STAGE 06: TEMPORAL & LAYER ANALYSIS")
print("=" * 78)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessed dataset not found:\n{DATA_PATH}\nRun Stage 04 first.")

with open(DATA_PATH, "rb") as f:
    processed_sha256 = hashlib.sha256(f.read()).hexdigest()

df = pd.read_csv(DATA_PATH)
print("\nPreprocessed dataset loaded successfully.")

df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="raise")
df["Fire_Risk_Assessment_Timestamp"] = pd.to_datetime(
    df["Fire_Risk_Assessment_Timestamp"], errors="raise"
)
for c in [
    "Temperature_C", "Humidity_%", "Elapsed_Drying_Hours", "Number_of_Sheets",
    "Layer_Area_m2", "Temperature_Change", "Humidity_Change",
    "Actual_Total_Drying_Time", "Remaining_Drying_Time",
]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values(KEY + ["Timestamp"]).reset_index(drop=True)

row_ok = len(df) == EXPECTED_ROWS
schema_ok = list(df.columns) == EXPECTED_COLUMNS
missing_cells = int(df[EXPECTED_COLUMNS].isna().sum().sum())
sequence_count = df.groupby(KEY).ngroups

# ------------------------------------------------------------
# Sampling intervals
# ------------------------------------------------------------
df["_Prev_TS"] = df.groupby(KEY)["Timestamp"].shift(1)
df["_Interval_Min"] = (df["Timestamp"] - df["_Prev_TS"]
                       ).dt.total_seconds() / 60

interval_summary = (
    df["_Interval_Min"].dropna().value_counts().sort_index()
    .rename_axis("Interval_Minutes").reset_index(name="Count")
)
interval_summary["Percentage"] = (
    interval_summary["Count"] / interval_summary["Count"].sum() * 100
)
interval_summary.to_csv(INTERVAL_PATH, index=False)

n10 = int(
    interval_summary.loc[interval_summary["Interval_Minutes"] == 10.0, "Count"].sum())
n20 = int(
    interval_summary.loc[interval_summary["Interval_Minutes"] == 20.0, "Count"].sum())
n_other = int(interval_summary.loc[
    ~interval_summary["Interval_Minutes"].isin([10.0, 20.0]), "Count"
].sum())

# ------------------------------------------------------------
# Exact timestamp lag analysis: 10, 30, 60 min
# ------------------------------------------------------------
lagged = df.copy()
lag_records = []

base = df[KEY + ["Timestamp", "Temperature_C", "Humidity_%"]].copy()

for minutes in [10, 30, 60]:
    look = base.copy()
    look["Timestamp"] = look["Timestamp"] + pd.to_timedelta(minutes, unit="m")
    look = look.rename(columns={
        "Temperature_C": f"Temperature_Lag_{minutes}m",
        "Humidity_%": f"Humidity_Lag_{minutes}m",
    })
    lagged = lagged.merge(
        look, on=KEY + ["Timestamp"], how="left", validate="one_to_one")

    for variable, current, lagcol in [
        ("Temperature_C", "Temperature_C", f"Temperature_Lag_{minutes}m"),
        ("Humidity_%", "Humidity_%", f"Humidity_Lag_{minutes}m"),
    ]:
        valid = lagged[[current, lagcol]].dropna()
        lag_records.append({
            "Variable": variable,
            "Lag_Minutes": minutes,
            "Pairs_Available": len(valid),
            "Coverage_Percent": len(valid) / len(df) * 100,
            "Exact_Lag_Correlation": float(valid[current].corr(valid[lagcol])),
            "Mean_Absolute_Change": float((valid[current] - valid[lagcol]).abs().mean()),
        })

lag_analysis = pd.DataFrame(lag_records)
lag_analysis.to_csv(LAG_PATH, index=False)


def lag_metric(variable, minutes, field):
    return float(lag_analysis.loc[
        (lag_analysis["Variable"] == variable) &
        (lag_analysis["Lag_Minutes"] == minutes), field
    ].iloc[0])


# ------------------------------------------------------------
# Layer behavior through normalized drying progress
# ------------------------------------------------------------
df["Drying_Progress"] = (
    df["Elapsed_Drying_Hours"] / df["Actual_Total_Drying_Time"]
).clip(0, 1)

progress_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
df["Drying_Progress_Stage"] = pd.cut(
    df["Drying_Progress"],
    bins=[-1e-9, 0.20, 0.40, 0.60, 0.80, 1.000001],
    labels=progress_labels,
    include_lowest=True,
)

progress_summary = (
    df.groupby(["Layer_ID", "Drying_Progress_Stage"], observed=True)
    .agg(
        Records=("Timestamp", "size"),
        Mean_Temperature_C=("Temperature_C", "mean"),
        SD_Temperature_C=("Temperature_C", "std"),
        Mean_Humidity_pct=("Humidity_%", "mean"),
        SD_Humidity_pct=("Humidity_%", "std"),
        Mean_Remaining_Hours=("Remaining_Drying_Time", "mean"),
    ).reset_index()
)
progress_summary.to_csv(PROGRESS_PATH, index=False)

# ------------------------------------------------------------
# Sequence temporal slopes
# ------------------------------------------------------------
seq_records = []
for key, g in df.groupby(KEY, sort=True):
    g = g.sort_values("Timestamp")
    x = g["Elapsed_Drying_Hours"].to_numpy()
    temp = g["Temperature_C"].to_numpy()
    rh = g["Humidity_%"].to_numpy()

    seq_records.append({
        "Smokehouse_ID": key[0],
        "Batch_ID": key[1],
        "Layer_ID": key[2],
        "Records": len(g),
        "Total_Drying_Hours": float(g["Actual_Total_Drying_Time"].iloc[0]),
        "Mean_Temperature_C": float(g["Temperature_C"].mean()),
        "Mean_Humidity_pct": float(g["Humidity_%"].mean()),
        "Temperature_Slope_C_per_Hour": float(np.polyfit(x, temp, 1)[0]),
        "Humidity_Slope_pct_per_Hour": float(np.polyfit(x, rh, 1)[0]),
        "Temperature_SD": float(g["Temperature_C"].std()),
        "Humidity_SD": float(g["Humidity_%"].std()),
    })

sequence_summary = pd.DataFrame(seq_records)
sequence_summary.to_csv(SEQUENCE_PATH, index=False)

# ------------------------------------------------------------
# Fire-risk assessment events
# Each assessment timestamp aligns with a sensor timestamp.
# ------------------------------------------------------------
assessment = df[
    df["Timestamp"] == df["Fire_Risk_Assessment_Timestamp"]
].copy()
assessment = assessment.sort_values(KEY + ["Timestamp"]).reset_index(drop=True)
assessment_count = len(assessment)

context_cols = [
    "Temperature_Lag_30m", "Humidity_Lag_30m",
    "Temperature_Lag_60m", "Humidity_Lag_60m",
]
context_lookup = lagged[KEY + ["Timestamp"] + context_cols].copy()

fire = assessment.merge(
    context_lookup, on=KEY + ["Timestamp"], how="left", validate="one_to_one"
)

fire["Temperature_Change_30m"] = fire["Temperature_C"] - \
    fire["Temperature_Lag_30m"]
fire["Humidity_Change_30m"] = fire["Humidity_%"] - fire["Humidity_Lag_30m"]
fire["Temperature_Change_60m"] = fire["Temperature_C"] - \
    fire["Temperature_Lag_60m"]
fire["Humidity_Change_60m"] = fire["Humidity_%"] - fire["Humidity_Lag_60m"]

fire_context = (
    fire.groupby("Fire_Risk_Label")
    .agg(
        Assessment_Count=("Fire_Risk_Label", "size"),
        Current_Mean_Temperature_C=("Temperature_C", "mean"),
        Current_Mean_Humidity_pct=("Humidity_%", "mean"),
        Mean_Temperature_Change_30m=("Temperature_Change_30m", "mean"),
        Mean_Humidity_Change_30m=("Humidity_Change_30m", "mean"),
        Mean_Temperature_Change_60m=("Temperature_Change_60m", "mean"),
        Mean_Humidity_Change_60m=("Humidity_Change_60m", "mean"),
        Median_Elapsed_Hours=("Elapsed_Drying_Hours", "median"),
    ).reset_index()
)

order = pd.CategoricalDtype(["Normal", "Warning", "Critical"], ordered=True)
fire_context["Fire_Risk_Label"] = fire_context["Fire_Risk_Label"].astype(order)
fire_context = fire_context.sort_values("Fire_Risk_Label")
fire_context.to_csv(FIRE_CONTEXT_PATH, index=False)

# ------------------------------------------------------------
# Hour-to-hour fire-risk transition matrix
# ------------------------------------------------------------
assessment["Previous_Fire_Risk_Label"] = (
    assessment.groupby(KEY)["Fire_Risk_Label"].shift(1)
)
transitions = assessment[assessment["Previous_Fire_Risk_Label"].notna()].copy()

risk_labels = ["Normal", "Warning", "Critical"]
transition_matrix = pd.crosstab(
    transitions["Previous_Fire_Risk_Label"], transitions["Fire_Risk_Label"]
).reindex(index=risk_labels, columns=risk_labels, fill_value=0)
transition_matrix.index.name = "Previous_Label"
transition_matrix.to_csv(TRANSITION_PATH)

same_label = int(
    (transitions["Previous_Fire_Risk_Label"]
     == transitions["Fire_Risk_Label"]).sum()
)
total_transitions = len(transitions)
persistence_pct = same_label / total_transitions * 100

# ------------------------------------------------------------
# Figures
# ------------------------------------------------------------


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


temp_progress = progress_summary.pivot(
    index="Drying_Progress_Stage", columns="Layer_ID", values="Mean_Temperature_C"
).reindex(progress_labels)

plt.figure(figsize=(8, 5))
for layer in ["L1", "L2", "L3"]:
    plt.plot(temp_progress.index,
             temp_progress[layer], marker="o", label=layer)
plt.xlabel("Drying Progress Stage")
plt.ylabel("Mean Temperature (°C)")
plt.title("Layer Temperature Across Drying Progress")
plt.legend()
savefig("06_01_temperature_by_progress_layer.png")

rh_progress = progress_summary.pivot(
    index="Drying_Progress_Stage", columns="Layer_ID", values="Mean_Humidity_pct"
).reindex(progress_labels)

plt.figure(figsize=(8, 5))
for layer in ["L1", "L2", "L3"]:
    plt.plot(rh_progress.index, rh_progress[layer], marker="o", label=layer)
plt.xlabel("Drying Progress Stage")
plt.ylabel("Mean Relative Humidity (%)")
plt.title("Layer Humidity Across Drying Progress")
plt.legend()
savefig("06_02_humidity_by_progress_layer.png")

lag_plot = lag_analysis.pivot(
    index="Lag_Minutes", columns="Variable", values="Exact_Lag_Correlation"
).sort_index()

plt.figure(figsize=(8, 5))
for variable in ["Temperature_C", "Humidity_%"]:
    plt.plot(lag_plot.index, lag_plot[variable], marker="o", label=variable)
plt.xlabel("Exact Lag (minutes)")
plt.ylabel("Correlation with Current Value")
plt.title("Temporal Dependence of Temperature and Humidity")
plt.legend()
savefig("06_03_exact_lag_correlation.png")

plt.figure(figsize=(7, 5))
plt.imshow(transition_matrix.to_numpy(), aspect="auto")
plt.xticks(range(3), risk_labels)
plt.yticks(range(3), risk_labels)
plt.xlabel("Current Hourly Assessment")
plt.ylabel("Previous Hourly Assessment")
plt.title("Fire-Risk Hour-to-Hour Transition Matrix")
for i in range(3):
    for j in range(3):
        plt.text(
            j, i, int(transition_matrix.iloc[i, j]), ha="center", va="center")
savefig("06_04_fire_risk_transition_matrix.png")

box_data = [
    fire.loc[fire["Fire_Risk_Label"] == label,
             "Humidity_Change_60m"].dropna().to_numpy()
    for label in risk_labels
]
plt.figure(figsize=(8, 5))
plt.boxplot(box_data, tick_labels=risk_labels, showfliers=False)
plt.xlabel("Fire Risk Label")
plt.ylabel("Humidity Change Over Previous 60 Minutes (%)")
plt.title("Recent Humidity Change by Fire-Risk Assessment")
savefig("06_05_humidity_change_60m_by_fire_risk.png")

# ------------------------------------------------------------
# Pass/fail checks
# ------------------------------------------------------------
temp10_cov = lag_metric("Temperature_C", 10, "Coverage_Percent")
rh10_cov = lag_metric("Humidity_%", 10, "Coverage_Percent")

checks = pd.DataFrame([
    ["Expected row count", "PASS" if row_ok else "FAIL", f"Rows={len(df):,}"],
    ["Expected 17-column schema", "PASS" if schema_ok else "FAIL",
        f"Columns={len(EXPECTED_COLUMNS)}"],
    ["No missing cells in preprocessed input", "PASS" if missing_cells ==
        0 else "FAIL", f"Missing={missing_cells}"],
    ["Expected 36 Batch-Layer sequences", "PASS" if sequence_count ==
        36 else "FAIL", f"Sequences={sequence_count}"],
    ["Sampling intervals preserved", "PASS" if (n10 == 24415 and n20 == 12 and n_other == 0) else "FAIL",
     f"10-min={n10}; 20-min={n20}; other={n_other}"],
    ["Expected 4,098 hourly assessments", "PASS" if assessment_count == 4098 else "FAIL",
     f"Assessments={assessment_count:,}"],
    ["10-minute temperature lag coverage >99%", "PASS" if temp10_cov > 99 else "FAIL",
     f"Coverage={temp10_cov:.4f}%"],
    ["10-minute humidity lag coverage >99%", "PASS" if rh10_cov > 99 else "FAIL",
     f"Coverage={rh10_cov:.4f}%"],
], columns=["Check", "Status", "Details"])

checks.to_csv(CHECKS_PATH, index=False)
overall = "PASS" if not (checks["Status"] == "FAIL").any() else "FAIL"

# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------
print("\n--- INPUT DATASET ---")
print(
    f"Rows / columns                 : {len(df):,} / {len(EXPECTED_COLUMNS)}")
print(f"Missing cells                  : {missing_cells}")
print(f"Batch-Layer sequences          : {sequence_count}")
print(f"Processed SHA-256              : {processed_sha256}")

print("\n--- SAMPLING ---")
print(f"10-minute intervals            : {n10:,}")
print(f"20-minute intervals            : {n20}")
print(f"Other intervals                : {n_other}")

print("\n--- EXACT TEMPORAL DEPENDENCE ---")
print(
    f"Temperature corr @ 10 min      : {lag_metric('Temperature_C', 10, 'Exact_Lag_Correlation'):.4f}")
print(
    f"Temperature corr @ 30 min      : {lag_metric('Temperature_C', 30, 'Exact_Lag_Correlation'):.4f}")
print(
    f"Temperature corr @ 60 min      : {lag_metric('Temperature_C', 60, 'Exact_Lag_Correlation'):.4f}")
print(
    f"Humidity corr @ 10 min         : {lag_metric('Humidity_%', 10, 'Exact_Lag_Correlation'):.4f}")
print(
    f"Humidity corr @ 30 min         : {lag_metric('Humidity_%', 30, 'Exact_Lag_Correlation'):.4f}")
print(
    f"Humidity corr @ 60 min         : {lag_metric('Humidity_%', 60, 'Exact_Lag_Correlation'):.4f}")

print("\n--- LAYER × DRYING PROGRESS ---")
print(progress_summary[
    ["Layer_ID", "Drying_Progress_Stage", "Mean_Temperature_C", "Mean_Humidity_pct"]
].round(3).to_string(index=False))

print("\n--- FIRE-RISK TEMPORAL CONTEXT ---")
print(fire_context.round(3).to_string(index=False))

print("\n--- FIRE-RISK TRANSITIONS ---")
print(transition_matrix.to_string())
print(f"\nSame-label hourly persistence   : {persistence_pct:.2f}%")

print("\n--- STAGE 07 GUIDANCE ---")
print("Candidate temporal windows: current/10 min, 30 min, and 60 min.")
print("All Stage 07 features must use current or past observations only.")

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------
with open(REPORT_PATH, "w", encoding="utf-8") as r:
    r.write("COMPONENT 3 - STAGE 06: TEMPORAL & LAYER ANALYSIS\n")
    r.write("="*78 + "\n\n")
    r.write(f"OVERALL STATUS: {overall}\n\n")
    r.write("INPUT DATASET\n-------------\n")
    r.write(f"Rows: {len(df):,}\nColumns: {len(EXPECTED_COLUMNS)}\n")
    r.write(
        f"Sequences: {sequence_count}\nProcessed SHA-256: {processed_sha256}\n\n")

    r.write("SAMPLING\n--------\n")
    r.write(
        f"10-minute intervals: {n10:,}\n20-minute gaps: {n20}\nOther intervals: {n_other}\n")
    r.write("Exact timestamp matching was used for lag analysis.\n\n")

    r.write("EXACT LAG ANALYSIS\n------------------\n")
    r.write(lag_analysis.round(6).to_string(index=False) + "\n\n")

    r.write("LAYER × DRYING PROGRESS\n-----------------------\n")
    r.write(progress_summary.round(6).to_string(index=False) + "\n\n")

    r.write("SEQUENCE TEMPORAL SUMMARY\n-------------------------\n")
    r.write(sequence_summary.round(6).to_string(index=False) + "\n\n")

    r.write("FIRE-RISK TEMPORAL CONTEXT\n--------------------------\n")
    r.write(fire_context.round(6).to_string(index=False) + "\n\n")

    r.write("FIRE-RISK TRANSITIONS\n---------------------\n")
    r.write(transition_matrix.to_string() + "\n")
    r.write(f"Same-label hourly persistence: {persistence_pct:.4f}%\n\n")

    r.write("INTERPRETATION / STAGE 07 GUIDANCE\n----------------------------------\n")
    r.write("- Recent sensor history has measurable temporal dependence.\n")
    r.write("- 30- and 60-minute past-only windows are justified candidates for feature engineering.\n")
    r.write("- Warning and Critical should not be reduced to a single instantaneous T/RH threshold.\n")
    r.write("- Elapsed_Drying_Hours remains available because drying stage is known at runtime.\n")
    r.write("- Layer_ID remains a contextual physical feature candidate.\n")
    r.write("- Never use Actual_Total_Drying_Time, Batch_ID, or Fire_Risk_Assessment_Timestamp as physical predictors.\n")
    r.write("- Do not invent rows for the 12 packet-loss gaps.\n")
    r.write("- Stage 07 may create timestamp-aware lags/rolling summaries using current/past data only.\n\n")

    r.write("PASS / FAIL CHECKS\n------------------\n")
    r.write(checks.to_string(index=False) + "\n\n")
    r.write(f"STAGE 06 STATUS: {overall}\n")

print("\n" + "=" * 78)
print(f"STAGE 06 TEMPORAL & LAYER ANALYSIS: {overall}")
print("=" * 78)
print(f"\nReports saved in:\n{REPORT_DIR}")
print(f"\nFigures saved in:\n{FIGURE_DIR}")
print("\nNo dataset was modified.")

if overall != "PASS":
    raise RuntimeError("Stage 06 failed. Review 06_temporal_layer_report.txt.")
