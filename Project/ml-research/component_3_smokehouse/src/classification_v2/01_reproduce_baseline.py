"""
CLASSIFICATION V2 - PHASE 3: BASELINE REPRODUCTION
====================================================

Purpose:
  Prove the V2 evaluation harness (common.py) reproduces the frozen
  Stage-30 pooled classification evidence for XGBoost XGB_03, using the
  SAME 5-fold GroupKFold-by-Batch_ID methodology, fold-local
  preprocessing, and fold-local balanced sample weights.

  This does NOT modify, re-save, or depend on src/30_group_cross_validation.py.
  It is an independent re-implementation that reads the same frozen inputs
  (Train+Validation splits, Stage-08 selected features, Stage-29 selected
  XGB_03 configuration) and must arrive at matching numbers.

Reference (frozen Stage-30 pooled OOF, from
outputs/reports/30_classification_group_cv_summary.csv):
  Accuracy          = 0.9558441558441558
  Balanced Accuracy = 0.7249685993484523
  Macro F1          = 0.6452671121687584
  Warning Recall    = 0.8564814814814815
  Critical Recall   = 0.35507246376811596

If pooled reproduction does not match within tolerance: STOP and report why.
Test split is never read.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


REFERENCE = {
    "Accuracy": 0.9558441558441558,
    "Balanced_Accuracy": 0.7249685993484523,
    "Macro_F1": 0.6452671121687584,
    "Warning_Recall": 0.8564814814814815,
    "Critical_Recall": 0.35507246376811596,
}
TOLERANCE = 1e-6


def run():
    print("=" * 78)
    print("CLASSIFICATION V2 - PHASE 3: BASELINE REPRODUCTION (XGB_03)")
    print("=" * 78)

    data = common.load_development()
    development = data["development"]
    selected_features = data["selected_features"]
    numeric_features = data["numeric_features"]
    categorical_features = data["categorical_features"]
    groups = data["groups"]

    print(f"\nDevelopment rows    : {len(development):,}")
    print(f"Development batches : {len(data['unique_batches'])}")
    print(f"Test file exists    : {data['test_exists']} (content read: {data['test_content_read']})")

    family, trial, xgb_params = common.recover_selected_xgb_configuration()
    print(f"\nRecovered selected classifier: {family} / {trial}")
    print("Parameters:")
    for k in sorted(xgb_params.keys()):
        print(f"  {k:<20}: {xgb_params[k]}")

    X_raw = development[selected_features].copy()
    y_cls = development["Fire_Risk_Int"].to_numpy(dtype=int)
    group_values = groups.to_numpy()

    group_kfold = GroupKFold(n_splits=common.N_SPLITS)

    fold_rows = []
    oof_records = []
    fold_overlap_count = 0

    for fold_number, (train_idx, val_idx) in enumerate(
        group_kfold.split(X_raw, y_cls, groups=group_values), start=1
    ):
        train_groups = set(group_values[train_idx])
        val_groups = set(group_values[val_idx])
        overlap = train_groups & val_groups
        if overlap:
            fold_overlap_count += 1
            raise RuntimeError(f"Fold {fold_number} batch overlap: {sorted(overlap)}")

        X_fold_train_raw = X_raw.iloc[train_idx].copy()
        X_fold_val_raw = X_raw.iloc[val_idx].copy()
        y_fold_train = y_cls[train_idx]
        y_fold_val = y_cls[val_idx]

        preprocessor = common.make_preprocessor(numeric_features, categorical_features)
        preprocessor.fit(X_fold_train_raw)
        X_fold_train = preprocessor.transform(X_fold_train_raw)
        X_fold_val = preprocessor.transform(X_fold_val_raw)

        weights = common.fold_sample_weight(y_fold_train, critical_multiplier=1.0)

        model = XGBClassifier(**xgb_params)
        model.fit(X_fold_train, y_fold_train, sample_weight=weights)

        y_pred = model.predict(X_fold_val).astype(int)

        fold_metrics = common.full_classification_metrics(y_fold_val, y_pred)
        fold_metrics["Fold"] = fold_number
        fold_metrics["Train_Rows"] = len(train_idx)
        fold_metrics["Validation_Rows"] = len(val_idx)
        fold_metrics["Validation_Batches"] = " | ".join(sorted(val_groups))
        fold_rows.append(fold_metrics)

        for local_pos, row_idx in enumerate(val_idx):
            oof_records.append({
                "Original_Row_Index": int(row_idx),
                "Fold": fold_number,
                "Batch_ID": str(group_values[row_idx]),
                "y_true_int": int(y_cls[row_idx]),
                "y_pred_int": int(y_pred[local_pos]),
            })

        print(f"\nFold {fold_number}/{common.N_SPLITS} | Validation batches: {', '.join(sorted(val_groups))}")
        print(f"  Critical Recall: {fold_metrics['Critical_Recall']}")
        print(f"  Warning Recall : {fold_metrics['Warning_Recall']}")

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_records).sort_values("Original_Row_Index").reset_index(drop=True)

    if len(oof_df) != len(development):
        raise RuntimeError("OOF row count does not equal development row count.")

    y_true_all = oof_df["y_true_int"].to_numpy()
    y_pred_all = oof_df["y_pred_int"].to_numpy()

    pooled = common.full_classification_metrics(y_true_all, y_pred_all)
    cm = common.confusion_df(y_true_all, y_pred_all)
    batch_df = common.batch_classification_metrics(oof_df)

    print("\n" + "=" * 78)
    print("POOLED OOF REPRODUCTION")
    print("=" * 78)
    for k in ["Accuracy", "Supported_Balanced_Accuracy", "Supported_Macro_F1", "Warning_Recall", "Critical_Recall"]:
        print(f"{k:<28}: {pooled[k]}")

    print("\nConfusion matrix:")
    print(cm.to_string())

    print("\n" + "=" * 78)
    print("REPRODUCTION VS FROZEN STAGE-30 REFERENCE")
    print("=" * 78)

    comparison_rows = []
    all_match = True
    mapping = {
        "Accuracy": "Accuracy",
        "Balanced_Accuracy": "Supported_Balanced_Accuracy",
        "Macro_F1": "Supported_Macro_F1",
        "Warning_Recall": "Warning_Recall",
        "Critical_Recall": "Critical_Recall",
    }
    for ref_name, pooled_key in mapping.items():
        ref_value = REFERENCE[ref_name]
        reproduced = pooled[pooled_key]
        diff = abs(reproduced - ref_value)
        match = diff < TOLERANCE
        all_match = all_match and match
        comparison_rows.append({
            "Metric": ref_name,
            "Stage30_Reference": ref_value,
            "V2_Reproduced": reproduced,
            "Abs_Difference": diff,
            "Match_Within_1e-6": match,
        })
        print(f"{ref_name:<20} reference={ref_value:.10f}  reproduced={reproduced:.10f}  diff={diff:.2e}  match={match}")

    comparison_df = pd.DataFrame(comparison_rows)

    status = "REPRODUCTION PASS" if all_match else "REPRODUCTION MISMATCH - STOP AND INVESTIGATE"
    print(f"\nSTATUS: {status}")

    # -------------------- SAVE OUTPUTS --------------------
    fold_df.to_csv(common.OUT_DIR / "01_baseline_reproduction_fold_metrics.csv", index=False)
    oof_df.to_csv(common.OUT_DIR / "01_baseline_reproduction_oof_predictions.csv", index=False)
    batch_df.to_csv(common.OUT_DIR / "01_baseline_reproduction_batch_metrics.csv", index=False)
    cm.to_csv(common.OUT_DIR / "01_baseline_reproduction_confusion_matrix.csv", index=True)
    comparison_df.to_csv(common.OUT_DIR / "01_baseline_reproduction_vs_stage30_reference.csv", index=False)

    pooled_out = dict(pooled)
    pooled_out["Fold_Overlap_Count"] = fold_overlap_count
    pooled_out["Reproduction_Status"] = status
    pd.Series(pooled_out).to_json(common.OUT_DIR / "01_baseline_reproduction_pooled_summary.json", indent=2)

    print(f"\nOutputs saved to: {common.OUT_DIR}")

    if not all_match:
        raise RuntimeError(
            "Baseline reproduction did not match frozen Stage-30 evidence within tolerance. "
            "Stopping before running improvement experiments, per protocol."
        )

    return {
        "pooled": pooled,
        "fold_df": fold_df,
        "oof_df": oof_df,
        "batch_df": batch_df,
        "cm": cm,
    }


if __name__ == "__main__":
    run()
