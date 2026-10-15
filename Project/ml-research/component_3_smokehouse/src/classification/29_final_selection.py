from pathlib import Path
import json
import pandas as pd

# ============================================================
# COMPONENT 3 - STAGE 29: FINAL CLASSIFICATION SELECTION
# ============================================================
# Purpose:
#   Select the development-stage fire-risk classifier
#   configuration from Stage-28 validation results.
#
# Important:
#   - No model training is performed here.
#   - No tuned model artifact is required here.
#   - Final Test contents are NOT loaded.
#   - Stage 30 performs group-aware cross-validation.
#   - Stage 31 is the first locked Final-Test evaluation.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "outputs" / "reports"
FINAL_DIR = ROOT / "artifacts" / "models" / "classification" / "final"
SPLIT_DIR = ROOT / "data" / "splits"
FINAL_TEST_PATH = SPLIT_DIR / "Component3_Test.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_TABLE_PATH = REPORT_DIR / "29_classification_final_selection_table.csv"
SELECTED_CONFIG_PATH = REPORT_DIR / "29_selected_classification_configuration.csv"
REPORT_PATH = REPORT_DIR / "29_classification_final_selection_report.txt"
METADATA_PATH = FINAL_DIR / "29_selected_classifier_metadata.json"

MIN_WARNING_RECALL = 0.80
MIN_CRITICAL_RECALL = 0.50

REQUIRED_COLUMNS = {
    "Model_Family",
    "Trial",
    "Validation_Macro_F1",
    "Validation_Balanced_Accuracy",
    "Warning_Recall",
    "Critical_Recall",
}


def load_stage28_results():
    frames = []
    used_files = []

    for path in sorted(REPORT_DIR.glob("28*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue

        if REQUIRED_COLUMNS.issubset(frame.columns):
            frame = frame.copy()
            frame["_Stage28_Source_File"] = path.name
            frames.append(frame)
            used_files.append(path)

    if not frames:
        raise FileNotFoundError(
            "No Stage-28 CSV report with the required selection metrics was found.\n"
            f"Searched in: {REPORT_DIR}"
        )

    combined = pd.concat(frames, ignore_index=True)

    # Remove rows repeated in both all-trials and best-by-family reports.
    combined = combined.drop_duplicates(
        subset=[
            "Model_Family",
            "Trial",
            "Validation_Macro_F1",
            "Validation_Balanced_Accuracy",
            "Warning_Recall",
            "Critical_Recall",
        ],
        keep="first",
    ).reset_index(drop=True)

    return combined, used_files


def json_safe(value):
    if pd.isna(value):
        return None
    try:
        return value.item()
    except Exception:
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)


print("=" * 78)
print("COMPONENT 3 - STAGE 29: FINAL CLASSIFICATION SELECTION")
print("=" * 78)

# ============================================================
# 1. FINAL TEST PROTECTION
# ============================================================

if not FINAL_TEST_PATH.exists():
    raise FileNotFoundError(
        f"Final Test dataset not found:\n{FINAL_TEST_PATH}"
    )

final_test_content_read = False

print("\n--- TEST PROTECTION ---")
print("Final Test dataset existence verified, but contents were NOT loaded.")
print(f"Component3_Test.csv content read: {final_test_content_read}")
print("Final Test remains frozen until Stage 31.")

# ============================================================
# 2. LOAD STAGE-28 VALIDATION RESULTS
# ============================================================

stage28, source_files = load_stage28_results()

metric_columns = [
    "Validation_Macro_F1",
    "Validation_Balanced_Accuracy",
    "Warning_Recall",
    "Critical_Recall",
]

for column in metric_columns:
    stage28[column] = pd.to_numeric(stage28[column], errors="coerce")

if stage28[metric_columns].isna().any().any():
    raise ValueError(
        "Stage-28 required metrics contain missing/non-numeric values.")

print("\nStage-28 trial metrics loaded successfully.")
print(f"Candidate configurations: {len(stage28)}")
print("Source report(s): " + ", ".join(path.name for path in source_files))

# ============================================================
# 3. APPLY SAFETY-AWARE SELECTION POLICY
# ============================================================

selection = stage28.copy()
selection["Warning_Recall_Pass"] = selection["Warning_Recall"] >= MIN_WARNING_RECALL
selection["Critical_Recall_Pass"] = selection["Critical_Recall"] >= MIN_CRITICAL_RECALL
selection["Safety_Guardrails_Pass"] = (
    selection["Warning_Recall_Pass"] & selection["Critical_Recall_Pass"]
)

eligible = selection[selection["Safety_Guardrails_Pass"]].copy()

if eligible.empty:
    raise RuntimeError(
        "No Stage-28 candidate satisfies both selection guardrails:\n"
        f"Warning Recall >= {MIN_WARNING_RECALL:.2f}\n"
        f"Critical Recall >= {MIN_CRITICAL_RECALL:.2f}\n"
        "Do not weaken the criteria automatically."
    )

eligible = eligible.sort_values(
    by=[
        "Validation_Macro_F1",
        "Validation_Balanced_Accuracy",
        "Critical_Recall",
        "Warning_Recall",
    ],
    ascending=[False, False, False, False],
).reset_index(drop=True)

selected = eligible.iloc[0].copy()
selected_family = str(selected["Model_Family"])
selected_trial = str(selected["Trial"])

# ============================================================
# 4. GLOBAL MACRO-F1 LEADER FOR COMPARISON
# ============================================================

global_leader = stage28.sort_values(
    by=["Validation_Macro_F1", "Validation_Balanced_Accuracy"],
    ascending=[False, False],
).iloc[0]

global_family = str(global_leader["Model_Family"])
global_trial = str(global_leader["Trial"])
global_is_selected = (
    global_family == selected_family and global_trial == selected_trial
)

# ============================================================
# 5. SAVE SELECTION TABLE
# ============================================================

selection["Selection_Status"] = "Not Eligible"
selection.loc[selection["Safety_Guardrails_Pass"],
              "Selection_Status"] = "Eligible"
selection.loc[
    (selection["Model_Family"] == selected_family)
    & (selection["Trial"] == selected_trial),
    "Selection_Status",
] = "SELECTED"

status_order = {"SELECTED": 0, "Eligible": 1, "Not Eligible": 2}
selection["_Status_Order"] = selection["Selection_Status"].map(status_order)
selection = selection.sort_values(
    by=["_Status_Order", "Validation_Macro_F1", "Validation_Balanced_Accuracy"],
    ascending=[True, False, False],
).drop(columns=["_Status_Order"]).reset_index(drop=True)

selection.to_csv(SELECTION_TABLE_PATH, index=False)

# Preserve every Stage-28 column for the selected row. If the tuning report
# contains hyperparameter columns, Stage 30 can read them from this file.
pd.DataFrame([selected.to_dict()]).to_csv(SELECTED_CONFIG_PATH, index=False)

# ============================================================
# 6. SAVE METADATA
# ============================================================

selected_record = {
    str(key): json_safe(value)
    for key, value in selected.to_dict().items()
}

metadata = {
    "stage": 29,
    "task": "Development-stage final classification configuration selection",
    "target": "Fire_Risk_Label",
    "selection_policy": {
        "minimum_warning_recall": MIN_WARNING_RECALL,
        "minimum_critical_recall": MIN_CRITICAL_RECALL,
        "primary_metric_among_eligible": "Validation_Macro_F1",
        "tie_breaker": "Validation_Balanced_Accuracy",
        "note": "Project-specific safety-aware development selection guardrails.",
    },
    "selected_model_family": selected_family,
    "selected_trial": selected_trial,
    "selected_validation_metrics": {
        "macro_f1": float(selected["Validation_Macro_F1"]),
        "balanced_accuracy": float(selected["Validation_Balanced_Accuracy"]),
        "warning_recall": float(selected["Warning_Recall"]),
        "critical_recall": float(selected["Critical_Recall"]),
    },
    "selected_stage28_record": selected_record,
    "global_macro_f1_leader": {
        "model_family": global_family,
        "trial": global_trial,
        "macro_f1": float(global_leader["Validation_Macro_F1"]),
        "balanced_accuracy": float(global_leader["Validation_Balanced_Accuracy"]),
        "warning_recall": float(global_leader["Warning_Recall"]),
        "critical_recall": float(global_leader["Critical_Recall"]),
        "also_selected": bool(global_is_selected),
    },
    "model_artifact_created_in_stage_29": False,
    "reason_no_artifact_required": (
        "Stage 29 is a configuration-selection stage. Stage 30 will instantiate "
        "the selected configuration for group-aware cross-validation."
    ),
    "final_test_exists": True,
    "final_test_content_read": False,
    "next_stage": "30_group_cross_validation.py",
}

with METADATA_PATH.open("w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)

# ============================================================
# 7. CONSOLE OUTPUT
# ============================================================

display_columns = [
    "Selection_Status",
    "Model_Family",
    "Trial",
    "Validation_Macro_F1",
    "Validation_Balanced_Accuracy",
    "Warning_Recall",
    "Critical_Recall",
]

print("\n--- FORMAL SELECTION GUARDRAILS ---")
print(f"Minimum Warning Recall  : {MIN_WARNING_RECALL:.2f}")
print(f"Minimum Critical Recall : {MIN_CRITICAL_RECALL:.2f}")
print(f"Eligible configurations : {len(eligible)}")

print("\n--- CANDIDATE SELECTION TABLE ---")
print(selection[display_columns].round(4).to_string(index=False))

print("\n--- GLOBAL VALIDATION MACRO-F1 LEADER ---")
print(f"Family               : {global_family}")
print(f"Trial                : {global_trial}")
print(
    f"Validation Macro-F1  : {float(global_leader['Validation_Macro_F1']):.4f}")
print(
    f"Balanced Accuracy    : {float(global_leader['Validation_Balanced_Accuracy']):.4f}")
print(f"Warning Recall       : {float(global_leader['Warning_Recall']):.4f}")
print(f"Critical Recall      : {float(global_leader['Critical_Recall']):.4f}")

print("\n--- STAGE-29 SELECTED CONFIGURATION ---")
print(f"Family               : {selected_family}")
print(f"Trial                : {selected_trial}")
print(f"Validation Macro-F1  : {float(selected['Validation_Macro_F1']):.4f}")
print(
    f"Balanced Accuracy    : {float(selected['Validation_Balanced_Accuracy']):.4f}")
print(f"Warning Recall       : {float(selected['Warning_Recall']):.4f}")
print(f"Critical Recall      : {float(selected['Critical_Recall']):.4f}")

if not global_is_selected:
    print("\nNOTE:")
    print(
        "The highest-Macro-F1 configuration was not selected because it did "
        "not satisfy the formal minority-risk selection guardrails."
    )

print("\nNo model artifact is required in Stage 29.")
print(
    "The selected configuration will be instantiated in Stage 30 for "
    "group-aware cross-validation."
)

# ============================================================
# 8. HUMAN-READABLE REPORT
# ============================================================

with REPORT_PATH.open("w", encoding="utf-8") as report:
    report.write("COMPONENT 3 - STAGE 29: FINAL CLASSIFICATION SELECTION\n")
    report.write("=" * 78 + "\n\n")
    report.write("TARGET\n------\n")
    report.write("Fire_Risk_Label: Normal / Warning / Critical\n\n")

    report.write("SELECTION POLICY\n----------------\n")
    report.write(f"Warning Recall >= {MIN_WARNING_RECALL:.2f}\n")
    report.write(f"Critical Recall >= {MIN_CRITICAL_RECALL:.2f}\n")
    report.write("Among eligible candidates: highest Validation Macro-F1.\n")
    report.write("Tie-breaker: highest Validation Balanced Accuracy.\n\n")

    report.write("SELECTED CONFIGURATION\n----------------------\n")
    report.write(f"Model family: {selected_family}\n")
    report.write(f"Trial: {selected_trial}\n")
    report.write(
        f"Validation Macro-F1: {float(selected['Validation_Macro_F1']):.4f}\n")
    report.write(
        "Validation Balanced Accuracy: "
        f"{float(selected['Validation_Balanced_Accuracy']):.4f}\n"
    )
    report.write(f"Warning Recall: {float(selected['Warning_Recall']):.4f}\n")
    report.write(
        f"Critical Recall: {float(selected['Critical_Recall']):.4f}\n\n")

    report.write("GLOBAL MACRO-F1 LEADER\n----------------------\n")
    report.write(f"Model family: {global_family}\n")
    report.write(f"Trial: {global_trial}\n")
    report.write(
        f"Validation Macro-F1: {float(global_leader['Validation_Macro_F1']):.4f}\n")
    report.write(
        f"Critical Recall: {float(global_leader['Critical_Recall']):.4f}\n")
    report.write(f"Also selected: {global_is_selected}\n\n")

    if not global_is_selected:
        report.write("INTERPRETATION\n--------------\n")
        report.write(
            "The highest-Macro-F1 trial was not automatically selected because "
            "the fire-risk task also requires adequate Warning and Critical detection.\n\n"
        )

    report.write("MODEL ARTIFACT POLICY\n---------------------\n")
    report.write(
        "No classifier artifact is created or required in Stage 29.\n")
    report.write(
        "Stage 29 saves the selected configuration and metrics only.\n")
    report.write(
        "Stage 30 will instantiate the selected configuration for group-aware "
        "cross-validation.\n\n"
    )

    report.write("TEST PROTECTION\n---------------\n")
    report.write("Final Test dataset exists: True\n")
    report.write("Final Test contents loaded: False\n")
    report.write("Final Test remains frozen until Stage 31.\n\n")

    report.write("NEXT STEP\n---------\n")
    report.write(
        "Stage 30 - Group-aware cross-validation using complete Batch_ID groups.\n\n"
    )
    report.write("FINAL STATUS: STAGE 29 PASS\n")

# ============================================================
# 9. FINISH
# ============================================================

print("\n--- TEST PROTECTION ---")
print(f"Component3_Test.csv content read: {final_test_content_read}")
print("Final Test remains frozen until Stage 31.")

print("\n" + "=" * 78)
print("STAGE 29 FINAL CLASSIFICATION SELECTION: PASS")
print("=" * 78)
print(f"\nSelection report:\n{REPORT_PATH}")
print(f"\nSelection table:\n{SELECTION_TABLE_PATH}")
print(f"\nSelected configuration:\n{SELECTED_CONFIG_PATH}")
print(f"\nMetadata:\n{METADATA_PATH}")
