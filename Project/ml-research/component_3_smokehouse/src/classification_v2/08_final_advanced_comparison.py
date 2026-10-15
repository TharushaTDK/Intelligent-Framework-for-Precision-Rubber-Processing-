"""
CLASSIFICATION V2 - FINAL ROUND - FINAL COMPARISON
======================================================

Consolidates ALL candidates from both V2 rounds (reading their existing,
already-verified stored outputs - nothing is rerun here) into one final
comparison table with fold/batch stability statistics, classifies each
candidate (CLEARLY WORSE / TRADE-OFF-INCONCLUSIVE / PROMISING IMPROVEMENT /
CLEARLY SUPERIOR) against the XGB_03 baseline, and produces the required
figures.

Does NOT select a final model. Does NOT touch the Test split.
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def summarize(name, fold_df, pooled_row, critical_col="Critical_Recall"):
    crit = fold_df[critical_col] if critical_col in fold_df.columns else pd.Series(dtype=float)
    supported = crit.dropna()
    return {
        "Candidate": name,
        "Accuracy": pooled_row.get("Accuracy"),
        "Balanced_Accuracy": pooled_row.get("Supported_Balanced_Accuracy"),
        "Macro_F1": pooled_row.get("Supported_Macro_F1"),
        "Warning_Recall": pooled_row.get("Warning_Recall"),
        "Critical_Recall": pooled_row.get("Critical_Recall"),
        "Critical_Precision": pooled_row.get("Critical_Precision"),
        "Critical_F1": pooled_row.get("Critical_F1"),
        "Critical_F2": pooled_row.get("Critical_F2"),
        "False_Critical_Alerts": pooled_row.get("Predicted_Critical_Count", np.nan) - (
            pooled_row.get("_true_critical_predicted", np.nan)
        ) if "_true_critical_predicted" in pooled_row else np.nan,
        "Predicted_Critical_Count": pooled_row.get("Predicted_Critical_Count"),
        "Actual_Critical_Count": pooled_row.get("Actual_Critical_Count"),
        "Supported_Critical_Folds": int(len(supported)),
        "Total_Folds": int(len(fold_df)),
        "Mean_Fold_Critical_Recall": float(supported.mean()) if len(supported) else np.nan,
        "Std_Fold_Critical_Recall": float(supported.std(ddof=0)) if len(supported) else np.nan,
        "Min_Fold_Critical_Recall": float(supported.min()) if len(supported) else np.nan,
        "Max_Fold_Critical_Recall": float(supported.max()) if len(supported) else np.nan,
        "Median_Fold_Critical_Recall": float(supported.median()) if len(supported) else np.nan,
    }


def false_alerts_from_cm(cm_path):
    cm = pd.read_csv(cm_path, index_col=0)
    # rows Actual_Normal/Actual_Warning, column Predicted_Critical
    false_alerts = 0
    for row in ["Actual_Normal", "Actual_Warning"]:
        if row in cm.index and "Predicted_Critical" in cm.columns:
            false_alerts += int(cm.loc[row, "Predicted_Critical"])
    return false_alerts


def run():
    print("=" * 78)
    print("FINAL ROUND - COMPREHENSIVE CANDIDATE COMPARISON")
    print("=" * 78)

    rows = []
    false_alerts_map = {}

    # --- Baseline ---
    baseline_fold = pd.read_csv(common.OUT_DIR / "01_baseline_reproduction_fold_metrics.csv")
    with open(common.OUT_DIR / "01_baseline_reproduction_pooled_summary.json") as f:
        baseline_pooled = json.load(f)
    rows.append(summarize("Baseline_XGB_03", baseline_fold, baseline_pooled))
    false_alerts_map["Baseline_XGB_03"] = false_alerts_from_cm(
        common.OUT_DIR / "01_baseline_reproduction_confusion_matrix.csv"
    )

    # --- Previous V2 round: Cost-Sensitive x3.0 ---
    cs_fold_all = pd.read_csv(common.OUT_DIR / "02_cost_sensitive_fold_metrics.csv")
    cs_pooled_all = pd.read_csv(common.OUT_DIR / "02_cost_sensitive_pooled_summary.csv")
    cs3_fold = cs_fold_all[cs_fold_all["Critical_Multiplier"] == 3.0]
    cs3_pooled = cs_pooled_all[cs_pooled_all["Critical_Multiplier"] == 3.0].iloc[0].to_dict()
    rows.append(summarize("CostSensitive_x3.0", cs3_fold, cs3_pooled))
    false_alerts_map["CostSensitive_x3.0"] = false_alerts_from_cm(
        common.OUT_DIR / "02_cost_sensitive_confusion_matrix_mult_3.0.csv"
    )

    # --- Previous V2 round: Threshold-Tuned (Nested) ---
    thr_fold = pd.read_csv(common.OUT_DIR / "03_threshold_nested_outer_fold_metrics.csv")
    with open(common.OUT_DIR / "03_threshold_nested_pooled_summary.json") as f:
        thr_pooled = json.load(f)
    rows.append(summarize("Threshold_Tuned_Nested", thr_fold, thr_pooled))
    false_alerts_map["Threshold_Tuned_Nested"] = false_alerts_from_cm(
        common.OUT_DIR / "03_threshold_nested_confusion_matrix.csv"
    )

    # --- Previous V2 round: Two-Stage ---
    ts_fold = pd.read_csv(common.OUT_DIR / "04_two_stage_fold_metrics.csv")
    with open(common.OUT_DIR / "04_two_stage_pooled_summary.json") as f:
        ts_pooled = json.load(f)
    rows.append(summarize("Two_Stage_XGB", ts_fold, ts_pooled))
    false_alerts_map["Two_Stage_XGB"] = false_alerts_from_cm(
        common.OUT_DIR / "04_two_stage_confusion_matrix.csv"
    )

    # --- NEW: Critical-vs-Rest Combined ---
    det_fold = pd.read_csv(common.OUT_DIR / "06_critical_detector_fold_metrics.csv")
    with open(common.OUT_DIR / "06_critical_detector_pooled_metrics.json") as f:
        det_pooled = json.load(f)
    rows.append(summarize("Critical_vs_Rest_Combined", det_fold, det_pooled))
    false_alerts_map["Critical_vs_Rest_Combined"] = false_alerts_from_cm(
        common.OUT_DIR / "06_critical_detector_confusion_matrix.csv"
    )

    # --- NEW: Ordinal-Argmax ---
    ordA_fold = pd.read_csv(common.OUT_DIR / "07_ordinal_argmax_fold_metrics.csv")
    with open(common.OUT_DIR / "07_ordinal_argmax_pooled_metrics.json") as f:
        ordA_pooled = json.load(f)
    rows.append(summarize("Ordinal_Argmax", ordA_fold, ordA_pooled))
    false_alerts_map["Ordinal_Argmax"] = false_alerts_from_cm(
        common.OUT_DIR / "07_ordinal_argmax_confusion_matrix.csv"
    )

    # --- NEW: Ordinal-Threshold ---
    ordT_fold = pd.read_csv(common.OUT_DIR / "07_ordinal_threshold_fold_metrics.csv")
    with open(common.OUT_DIR / "07_ordinal_threshold_pooled_metrics.json") as f:
        ordT_pooled = json.load(f)
    rows.append(summarize("Ordinal_Threshold", ordT_fold, ordT_pooled))
    false_alerts_map["Ordinal_Threshold"] = false_alerts_from_cm(
        common.OUT_DIR / "07_ordinal_threshold_confusion_matrix.csv"
    )

    comparison_df = pd.DataFrame(rows)
    comparison_df["False_Critical_Alerts"] = comparison_df["Candidate"].map(false_alerts_map)

    # -------------------- CLASSIFICATION --------------------
    baseline_row = comparison_df[comparison_df["Candidate"] == "Baseline_XGB_03"].iloc[0]
    EPS = 1e-9
    WARNING_GUARDRAIL = 0.80

    def classify(row):
        if row["Candidate"] == "Baseline_XGB_03":
            return "REFERENCE"

        recall_gain = row["Critical_Recall"] - baseline_row["Critical_Recall"]
        precision_loss = baseline_row["Critical_Precision"] - row["Critical_Precision"]
        warning_loss = baseline_row["Warning_Recall"] - row["Warning_Recall"]
        macro_f1_gain = row["Macro_F1"] - baseline_row["Macro_F1"]
        below_warning_guardrail = row["Warning_Recall"] < WARNING_GUARDRAIL

        severe_collapse = (
            below_warning_guardrail
            or warning_loss > 0.10
            or precision_loss > 0.20
        )

        if recall_gain <= EPS and macro_f1_gain < -EPS:
            return "CLEARLY_WORSE"

        if severe_collapse:
            return "TRADEOFF_INCONCLUSIVE (severe Warning/Precision collapse)"

        if recall_gain > 0.02 and macro_f1_gain >= -0.01 and warning_loss <= 0.05:
            if macro_f1_gain > 0.01 and warning_loss <= 0.03 and precision_loss <= 0.05:
                return "CLEARLY_SUPERIOR (candidate)"
            return "PROMISING_IMPROVEMENT"

        return "TRADEOFF_INCONCLUSIVE"

    comparison_df["Classification"] = comparison_df.apply(classify, axis=1)
    comparison_df["Below_Warning_Guardrail_0.80"] = comparison_df["Warning_Recall"] < WARNING_GUARDRAIL

    comparison_df.to_csv(common.OUT_DIR / "08_final_advanced_candidate_comparison.csv", index=False)
    comparison_df.to_json(common.OUT_DIR / "08_final_advanced_candidate_comparison.json",
                           orient="records", indent=2)

    print("\n" + comparison_df[[
        "Candidate", "Critical_Recall", "Critical_Precision", "Critical_F2",
        "Warning_Recall", "Macro_F1", "Balanced_Accuracy",
        "False_Critical_Alerts", "Mean_Fold_Critical_Recall", "Std_Fold_Critical_Recall",
        "Min_Fold_Critical_Recall", "Max_Fold_Critical_Recall",
        "Below_Warning_Guardrail_0.80", "Classification",
    ]].to_string(index=False))

    # -------------------- PLOTS --------------------
    plt.figure(figsize=(8, 7))
    for _, row in comparison_df.iterrows():
        plt.scatter(row["Critical_Recall"], row["Critical_Precision"], s=80)
        plt.annotate(row["Candidate"], (row["Critical_Recall"], row["Critical_Precision"]),
                     fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Critical Recall (pooled OOF)")
    plt.ylabel("Critical Precision (pooled OOF)")
    plt.title("Critical Recall vs Critical Precision - All Candidates")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(common.FIG_DIR / "final_critical_recall_vs_precision.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 7))
    for _, row in comparison_df.iterrows():
        plt.scatter(row["Critical_Recall"], row["Warning_Recall"], s=80)
        plt.annotate(row["Candidate"], (row["Critical_Recall"], row["Warning_Recall"]),
                     fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.axhline(0.80, linestyle="--", alpha=0.5)
    plt.xlabel("Critical Recall (pooled OOF)")
    plt.ylabel("Warning Recall (pooled OOF)")
    plt.title("Critical Recall vs Warning Recall - All Candidates (dashed: 0.80 guardrail)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(common.FIG_DIR / "final_critical_recall_vs_warning_recall.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(comparison_df["Candidate"], comparison_df["Critical_F2"])
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Critical F2 (pooled OOF)")
    plt.title("Critical F2 by Candidate")
    plt.tight_layout()
    plt.savefig(common.FIG_DIR / "final_critical_f2_by_candidate.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(comparison_df["Candidate"], comparison_df["Macro_F1"])
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Macro F1 (pooled OOF)")
    plt.title("Macro F1 by Candidate")
    plt.tight_layout()
    plt.savefig(common.FIG_DIR / "final_macro_f1_by_candidate.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(comparison_df["Candidate"], comparison_df["False_Critical_Alerts"])
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("False Critical Alerts (pooled OOF)")
    plt.title("False Critical Alerts by Candidate")
    plt.tight_layout()
    plt.savefig(common.FIG_DIR / "final_false_critical_alerts_by_candidate.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 6))
    fold_sources = {
        "Baseline_XGB_03": baseline_fold,
        "CostSensitive_x3.0": cs3_fold,
        "Threshold_Tuned_Nested": thr_fold,
        "Two_Stage_XGB": ts_fold,
        "Critical_vs_Rest_Combined": det_fold,
        "Ordinal_Argmax": ordA_fold,
        "Ordinal_Threshold": ordT_fold,
    }
    for label, fdf in fold_sources.items():
        if "Fold" in fdf.columns and "Critical_Recall" in fdf.columns:
            plt.plot(fdf["Fold"], fdf["Critical_Recall"], marker="o", label=label, alpha=0.8)
    plt.xlabel("Outer Fold")
    plt.ylabel("Critical Recall")
    plt.title("Per-Fold Critical Recall - All Candidates")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(common.FIG_DIR / "final_per_fold_critical_recall_all.png", dpi=150)
    plt.close()

    # Per-batch Critical Recall DIFFERENCE vs baseline, for the two new candidates.
    base_batch = pd.read_csv(common.OUT_DIR / "01_baseline_reproduction_batch_metrics.csv")
    det_batch = pd.read_csv(common.OUT_DIR / "06_critical_detector_batch_metrics.csv")
    ordA_batch = pd.read_csv(common.OUT_DIR / "07_ordinal_argmax_batch_metrics.csv")

    crit_batches = base_batch[base_batch["Actual_Critical_Count"] > 0]["Batch_ID"].tolist()
    diff_rows = []
    for batch_id in crit_batches:
        b = base_batch[base_batch["Batch_ID"] == batch_id]["Critical_Recall"].iloc[0]
        d = det_batch[det_batch["Batch_ID"] == batch_id]["Critical_Recall"].iloc[0]
        o = ordA_batch[ordA_batch["Batch_ID"] == batch_id]["Critical_Recall"].iloc[0]
        diff_rows.append({"Batch_ID": batch_id, "Critical_vs_Rest_Combined": d - b, "Ordinal_Argmax": o - b})
    diff_df = pd.DataFrame(diff_rows).set_index("Batch_ID")
    diff_df.to_csv(common.OUT_DIR / "08_per_batch_critical_recall_difference_vs_baseline.csv")

    diff_df.plot(kind="bar", figsize=(9, 5))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Critical Recall difference vs baseline")
    plt.title("Per-Batch Critical Recall Difference vs Baseline (Critical-containing batches)")
    plt.tight_layout()
    plt.savefig(common.FIG_DIR / "final_per_batch_critical_recall_difference.png", dpi=150)
    plt.close()

    print(f"\nComparison table saved to: {common.OUT_DIR / '08_final_advanced_candidate_comparison.csv'}")
    print(f"Figures saved to: {common.FIG_DIR}")

    return comparison_df


if __name__ == "__main__":
    run()
