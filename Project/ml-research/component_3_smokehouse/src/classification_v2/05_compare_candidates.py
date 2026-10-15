"""
CLASSIFICATION V2 - CANDIDATE COMPARISON
==========================================

Consolidates every candidate produced by scripts 01-04 into one
comparison table (pooled OOF metrics + fold variability + batch
robustness), and produces the required diagnostic plots.

Does NOT select a final model. Does NOT touch the Test split.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def load_fold_df(name):
    return pd.read_csv(common.OUT_DIR / name)


def summarize_candidate(name, fold_df, pooled_row, critical_col="Critical_Recall"):
    crit = fold_df[critical_col]
    supported = crit.dropna()
    return {
        "Candidate": name,
        "Pooled_Critical_Recall": pooled_row.get("Critical_Recall"),
        "Pooled_Critical_Precision": pooled_row.get("Critical_Precision"),
        "Pooled_Critical_F1": pooled_row.get("Critical_F1"),
        "Pooled_Critical_F2": pooled_row.get("Critical_F2"),
        "Pooled_Warning_Recall": pooled_row.get("Warning_Recall"),
        "Pooled_Normal_Recall": pooled_row.get("Normal_Recall"),
        "Pooled_Macro_F1": pooled_row.get("Supported_Macro_F1"),
        "Pooled_Weighted_F1": pooled_row.get("Weighted_F1"),
        "Pooled_Balanced_Accuracy": pooled_row.get("Supported_Balanced_Accuracy"),
        "Pooled_Accuracy": pooled_row.get("Accuracy"),
        "Fold_Critical_Recall_Mean": float(supported.mean()) if len(supported) else np.nan,
        "Fold_Critical_Recall_Std": float(supported.std(ddof=0)) if len(supported) else np.nan,
        "Fold_Critical_Recall_Min": float(supported.min()) if len(supported) else np.nan,
        "Fold_Critical_Recall_Max": float(supported.max()) if len(supported) else np.nan,
        "N_Folds_With_Critical_Support": int(len(supported)),
        "N_Folds_Total": int(len(fold_df)),
        "Predicted_Critical_Count_Pooled": pooled_row.get("Predicted_Critical_Count"),
        "Actual_Critical_Count_Pooled": pooled_row.get("Actual_Critical_Count"),
    }


def run():
    print("=" * 78)
    print("CLASSIFICATION V2 - CANDIDATE COMPARISON")
    print("=" * 78)

    import json

    rows = []

    # --- Baseline (XGB_03 reproduction) ---
    baseline_fold = load_fold_df("01_baseline_reproduction_fold_metrics.csv")
    with open(common.OUT_DIR / "01_baseline_reproduction_pooled_summary.json") as f:
        baseline_pooled = json.load(f)
    rows.append(summarize_candidate("Baseline_XGB_03", baseline_fold, baseline_pooled))

    # --- Cost-sensitive candidates ---
    cs_fold_all = load_fold_df("02_cost_sensitive_fold_metrics.csv")
    cs_pooled_all = pd.read_csv(common.OUT_DIR / "02_cost_sensitive_pooled_summary.csv")
    for mult in common.COST_SENSITIVE_CRITICAL_MULTIPLIERS:
        fold_sub = cs_fold_all[cs_fold_all["Critical_Multiplier"] == mult]
        pooled_row = cs_pooled_all[cs_pooled_all["Critical_Multiplier"] == mult].iloc[0].to_dict()
        rows.append(summarize_candidate(f"CostSensitive_x{mult}", fold_sub, pooled_row))

    # --- Threshold-tuned (nested) ---
    thr_fold = load_fold_df("03_threshold_nested_outer_fold_metrics.csv")
    with open(common.OUT_DIR / "03_threshold_nested_pooled_summary.json") as f:
        thr_pooled = json.load(f)
    rows.append(summarize_candidate("Threshold_Tuned_Nested", thr_fold, thr_pooled))

    # --- Two-stage ---
    ts_fold = load_fold_df("04_two_stage_fold_metrics.csv")
    with open(common.OUT_DIR / "04_two_stage_pooled_summary.json") as f:
        ts_pooled = json.load(f)
    rows.append(summarize_candidate("Two_Stage_XGB", ts_fold, ts_pooled))

    comparison_df = pd.DataFrame(rows)

    # --- A/B/C classification (NOT automatic "highest recall wins") ---
    baseline_row = comparison_df[comparison_df["Candidate"] == "Baseline_XGB_03"].iloc[0]

    EPS = 1e-9  # float-noise tolerance (e.g. CSV round-trip precision)

    def classify(row):
        if row["Candidate"] == "Baseline_XGB_03":
            return "REFERENCE"
        if row["Candidate"] == "CostSensitive_x1.0":
            return "REFERENCE_EQUIVALENT_SANITY_CHECK"

        recall_gain = row["Pooled_Critical_Recall"] - baseline_row["Pooled_Critical_Recall"]
        precision_loss = baseline_row["Pooled_Critical_Precision"] - row["Pooled_Critical_Precision"]
        warning_loss = baseline_row["Pooled_Warning_Recall"] - row["Pooled_Warning_Recall"]
        macro_f1_loss = baseline_row["Pooled_Macro_F1"] - row["Pooled_Macro_F1"]

        if recall_gain <= EPS and macro_f1_loss > EPS:
            return "A_CLEARLY_WORSE"
        if warning_loss > 0.05 or precision_loss > 0.15:
            return "B_TRADEOFF_INCONCLUSIVE"
        if recall_gain > 0.02 and macro_f1_loss <= 0.01 and warning_loss <= 0.03:
            return "C_PROMISING"
        return "B_TRADEOFF_INCONCLUSIVE"

    comparison_df["Classification"] = comparison_df.apply(classify, axis=1)

    comparison_df.to_csv(common.OUT_DIR / "05_candidate_comparison_table.csv", index=False)

    print("\n" + comparison_df[[
        "Candidate", "Pooled_Critical_Recall", "Pooled_Critical_Precision", "Pooled_Critical_F2",
        "Pooled_Warning_Recall", "Pooled_Macro_F1", "Pooled_Balanced_Accuracy",
        "Fold_Critical_Recall_Mean", "Fold_Critical_Recall_Std",
        "Fold_Critical_Recall_Min", "Fold_Critical_Recall_Max",
        "N_Folds_With_Critical_Support", "Classification",
    ]].to_string(index=False))

    # -------------------- PLOTS --------------------
    plt.figure(figsize=(7, 6))
    for _, row in comparison_df.iterrows():
        plt.scatter(row["Pooled_Critical_Recall"], row["Pooled_Critical_Precision"], s=80)
        plt.annotate(row["Candidate"], (row["Pooled_Critical_Recall"], row["Pooled_Critical_Precision"]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Critical Recall (pooled OOF)")
    plt.ylabel("Critical Precision (pooled OOF)")
    plt.title("Critical Recall vs Critical Precision by Candidate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(common.FIG_DIR / "critical_recall_vs_precision.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 6))
    for _, row in comparison_df.iterrows():
        plt.scatter(row["Pooled_Critical_Recall"], row["Pooled_Warning_Recall"], s=80)
        plt.annotate(row["Candidate"], (row["Pooled_Critical_Recall"], row["Pooled_Warning_Recall"]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Critical Recall (pooled OOF)")
    plt.ylabel("Warning Recall (pooled OOF)")
    plt.title("Critical Recall vs Warning Recall by Candidate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(common.FIG_DIR / "critical_recall_vs_warning_recall.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(comparison_df["Candidate"], comparison_df["Pooled_Macro_F1"])
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Macro F1 (supported classes, pooled OOF)")
    plt.title("Candidate Macro F1")
    plt.tight_layout()
    plt.savefig(common.FIG_DIR / "candidate_macro_f1.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(comparison_df["Candidate"], comparison_df["Pooled_Balanced_Accuracy"])
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Balanced Accuracy (supported classes, pooled OOF)")
    plt.title("Candidate Balanced Accuracy")
    plt.tight_layout()
    plt.savefig(common.FIG_DIR / "candidate_balanced_accuracy.png", dpi=150)
    plt.close()

    # Per-fold Critical Recall across candidates
    plt.figure(figsize=(9, 6))
    fold_sources = {
        "Baseline_XGB_03": baseline_fold,
        "CostSensitive_x3.0": cs_fold_all[cs_fold_all["Critical_Multiplier"] == 3.0],
        "Threshold_Tuned_Nested": thr_fold,
        "Two_Stage_XGB": ts_fold,
    }
    for label, fdf in fold_sources.items():
        plt.plot(fdf["Fold"], fdf["Critical_Recall"], marker="o", label=label)
    plt.xlabel("Outer Fold")
    plt.ylabel("Critical Recall")
    plt.title("Per-Fold Critical Recall (selected candidates)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(common.FIG_DIR / "per_fold_critical_recall.png", dpi=150)
    plt.close()

    print(f"\nComparison table saved to: {common.OUT_DIR / '05_candidate_comparison_table.csv'}")
    print(f"Figures saved to: {common.FIG_DIR}")

    return comparison_df


if __name__ == "__main__":
    run()
