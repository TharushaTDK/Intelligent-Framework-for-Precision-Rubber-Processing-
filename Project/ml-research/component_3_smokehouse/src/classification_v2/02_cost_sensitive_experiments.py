"""
CLASSIFICATION V2 - PHASE 4: EXPERIMENT A - COST-SENSITIVE XGBOOST
====================================================================

Stage 30 already uses balanced sample weighting (class_weight="balanced",
recomputed per fold from fold-training labels only). This experiment does
NOT re-present that as new; it tests whether an ADDITIONAL predefined
Critical-specific weight multiplier on top of the existing balanced
recipe changes Critical-class generalization.

Multipliers (fixed BEFORE running, per protocol): 1.0, 1.5, 2.0, 3.0
  - 1.0 is the exact Stage-30 baseline (included as an internal
    consistency check against 01_reproduce_baseline.py output).

All weighting happens strictly inside each fold's training portion.
Test split is never read.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def run_one_multiplier(multiplier, X_raw, y_cls, group_values, numeric_features,
                        categorical_features, xgb_params):
    group_kfold = GroupKFold(n_splits=common.N_SPLITS)
    fold_rows = []
    oof_records = []

    for fold_number, (train_idx, val_idx) in enumerate(
        group_kfold.split(X_raw, y_cls, groups=group_values), start=1
    ):
        train_groups = set(group_values[train_idx])
        val_groups = set(group_values[val_idx])
        overlap = train_groups & val_groups
        if overlap:
            raise RuntimeError(f"Fold {fold_number} batch overlap: {sorted(overlap)}")

        X_fold_train_raw = X_raw.iloc[train_idx].copy()
        X_fold_val_raw = X_raw.iloc[val_idx].copy()
        y_fold_train = y_cls[train_idx]
        y_fold_val = y_cls[val_idx]

        preprocessor = common.make_preprocessor(numeric_features, categorical_features)
        preprocessor.fit(X_fold_train_raw)
        X_fold_train = preprocessor.transform(X_fold_train_raw)
        X_fold_val = preprocessor.transform(X_fold_val_raw)

        weights = common.fold_sample_weight(y_fold_train, critical_multiplier=multiplier)

        model = XGBClassifier(**xgb_params)
        model.fit(X_fold_train, y_fold_train, sample_weight=weights)
        y_pred = model.predict(X_fold_val).astype(int)

        fold_metrics = common.full_classification_metrics(y_fold_val, y_pred)
        fold_metrics["Fold"] = fold_number
        fold_metrics["Critical_Multiplier"] = multiplier
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

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_records).sort_values("Original_Row_Index").reset_index(drop=True)
    pooled = common.full_classification_metrics(
        oof_df["y_true_int"].to_numpy(), oof_df["y_pred_int"].to_numpy()
    )
    pooled["Critical_Multiplier"] = multiplier
    pooled["Candidate"] = f"CS_{multiplier}"
    batch_df = common.batch_classification_metrics(oof_df)
    batch_df["Critical_Multiplier"] = multiplier
    cm = common.confusion_df(oof_df["y_true_int"].to_numpy(), oof_df["y_pred_int"].to_numpy())

    return fold_df, oof_df, batch_df, cm, pooled


def run():
    print("=" * 78)
    print("CLASSIFICATION V2 - PHASE 4: EXPERIMENT A - COST-SENSITIVE XGBOOST")
    print("=" * 78)
    print(f"\nPredefined Critical multipliers: {common.COST_SENSITIVE_CRITICAL_MULTIPLIERS}")

    data = common.load_development()
    development = data["development"]
    selected_features = data["selected_features"]
    numeric_features = data["numeric_features"]
    categorical_features = data["categorical_features"]
    groups = data["groups"]

    _, _, xgb_params = common.recover_selected_xgb_configuration()

    X_raw = development[selected_features].copy()
    y_cls = development["Fire_Risk_Int"].to_numpy(dtype=int)
    group_values = groups.to_numpy()

    all_fold_rows = []
    all_oof_rows = []
    all_batch_rows = []
    all_pooled = []
    all_cm = {}

    for multiplier in common.COST_SENSITIVE_CRITICAL_MULTIPLIERS:
        print(f"\n--- Critical multiplier = {multiplier} ---")
        fold_df, oof_df, batch_df, cm, pooled = run_one_multiplier(
            multiplier, X_raw, y_cls, group_values, numeric_features, categorical_features, xgb_params
        )
        oof_df["Critical_Multiplier"] = multiplier
        all_fold_rows.append(fold_df)
        all_oof_rows.append(oof_df)
        all_batch_rows.append(batch_df)
        all_pooled.append(pooled)
        all_cm[multiplier] = cm

        print(f"  Pooled Critical Recall   : {pooled['Critical_Recall']}")
        print(f"  Pooled Critical Precision: {pooled['Critical_Precision']}")
        print(f"  Pooled Critical F2       : {pooled['Critical_F2']}")
        print(f"  Pooled Warning Recall    : {pooled['Warning_Recall']}")
        print(f"  Pooled Macro F1 (supp.)  : {pooled['Supported_Macro_F1']}")
        print(f"  Pooled Balanced Accuracy : {pooled['Supported_Balanced_Accuracy']}")
        print(f"  Predicted Critical count : {pooled['Predicted_Critical_Count']} "
              f"(actual: {pooled['Actual_Critical_Count']})")

        if multiplier == 1.0:
            ref_critical_recall = 0.35507246376811596
            diff = abs(pooled["Critical_Recall"] - ref_critical_recall)
            print(f"  [sanity check] multiplier=1.0 vs Stage-30 Critical Recall diff={diff:.2e} "
                  f"({'OK' if diff < 1e-6 else 'MISMATCH'})")

    combined_fold_df = pd.concat(all_fold_rows, ignore_index=True)
    combined_oof_df = pd.concat(all_oof_rows, ignore_index=True)
    combined_batch_df = pd.concat(all_batch_rows, ignore_index=True)
    pooled_df = pd.DataFrame(all_pooled)

    combined_fold_df.to_csv(common.OUT_DIR / "02_cost_sensitive_fold_metrics.csv", index=False)
    combined_oof_df.to_csv(common.OUT_DIR / "02_cost_sensitive_oof_predictions.csv", index=False)
    combined_batch_df.to_csv(common.OUT_DIR / "02_cost_sensitive_batch_metrics.csv", index=False)
    pooled_df.to_csv(common.OUT_DIR / "02_cost_sensitive_pooled_summary.csv", index=False)

    for multiplier, cm in all_cm.items():
        cm.to_csv(common.OUT_DIR / f"02_cost_sensitive_confusion_matrix_mult_{multiplier}.csv", index=True)

    print("\n" + "=" * 78)
    print("EXPERIMENT A SUMMARY (pooled OOF)")
    print("=" * 78)
    print(
        pooled_df[[
            "Candidate", "Critical_Multiplier", "Critical_Recall", "Critical_Precision",
            "Critical_F2", "Warning_Recall", "Supported_Macro_F1", "Supported_Balanced_Accuracy",
            "Predicted_Critical_Count", "Actual_Critical_Count",
        ]].to_string(index=False)
    )

    print(f"\nOutputs saved to: {common.OUT_DIR}")
    return pooled_df, combined_fold_df, combined_batch_df


if __name__ == "__main__":
    run()
