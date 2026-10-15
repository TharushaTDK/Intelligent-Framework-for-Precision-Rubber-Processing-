"""
CLASSIFICATION V2 - PHASE 6: EXPERIMENT C - TWO-STAGE CLASSIFICATION
========================================================================

Stage A: Normal (0) vs Abnormal (1 = Warning or Critical)
Stage B: Warning (0) vs Critical (1) - trained ONLY on rows whose TRUE
         label is Abnormal within the fold-training portion, and applied
         ONLY to rows Stage A predicts as Abnormal at evaluation time.

Both stages stay close to the existing XGBoost family (same core tree
hyperparameters as the frozen XGB_03 configuration), binarized:
objective="binary:logistic", eval_metric="logloss", num_class removed.
Balanced sample weights are computed independently per stage from that
stage's own fold-training labels only.

Preprocessing is fit ONCE per fold on the fold-training portion (identical
to the baseline/Stage-30 preprocessor) and reused for both stages - no
global pre-CV fitting.

Reports, in addition to the combined 3-class result: Stage-A binary
metrics and Stage-B metrics on the true-abnormal validation subset, so
failures can be localized to a specific stage.

Test split is never read.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def make_binary_xgb_params(base_xgb_params):
    params = dict(base_xgb_params)
    params.pop("num_class", None)
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    return params


def stage_a_binary_metrics(y_true_binary, y_pred_binary):
    """y=1 is Abnormal (positive class of interest)."""
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    return {
        "Abnormal_Recall": float(recall_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)),
        "Abnormal_Precision": float(precision_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)),
        "Abnormal_F1": float(f1_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)),
        "Normal_Recall_StageA": float(recall_score(y_true_binary, y_pred_binary, pos_label=0, zero_division=0)),
        "Confusion_TN_Normal_Normal": int(cm[0, 0]),
        "Confusion_FP_Normal_Abnormal": int(cm[0, 1]),
        "Confusion_FN_Abnormal_Normal": int(cm[1, 0]),
        "Confusion_TP_Abnormal_Abnormal": int(cm[1, 1]),
    }


def stage_b_binary_metrics(y_true_binary, y_pred_binary):
    """y=1 is Critical (positive class of interest), evaluated on TRUE abnormal rows only."""
    support_critical = int(np.sum(np.asarray(y_true_binary) == 1))
    if support_critical == 0:
        critical_recall = np.nan
    else:
        critical_recall = float(recall_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0))

    predicted_critical = int(np.sum(np.asarray(y_pred_binary) == 1))
    critical_precision = np.nan if predicted_critical == 0 else float(
        precision_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
    )
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    return {
        "StageB_Critical_Recall": critical_recall,
        "StageB_Critical_Precision": critical_precision,
        "StageB_Warning_Recall": float(recall_score(y_true_binary, y_pred_binary, pos_label=0, zero_division=0)),
        "StageB_Confusion_TN_Warning_Warning": int(cm[0, 0]),
        "StageB_Confusion_FP_Warning_Critical": int(cm[0, 1]),
        "StageB_Confusion_FN_Critical_Warning": int(cm[1, 0]),
        "StageB_Confusion_TP_Critical_Critical": int(cm[1, 1]),
    }


def run():
    print("=" * 78)
    print("CLASSIFICATION V2 - PHASE 6: EXPERIMENT C - TWO-STAGE CLASSIFICATION")
    print("=" * 78)

    data = common.load_development()
    development = data["development"]
    selected_features = data["selected_features"]
    numeric_features = data["numeric_features"]
    categorical_features = data["categorical_features"]
    groups = data["groups"]

    _, _, xgb_params_multiclass = common.recover_selected_xgb_configuration()
    xgb_params_binary = make_binary_xgb_params(xgb_params_multiclass)

    print("\nBinarized XGBoost parameters (both stages):")
    for k in sorted(xgb_params_binary.keys()):
        print(f"  {k:<20}: {xgb_params_binary[k]}")

    X_raw = development[selected_features].copy()
    y_cls = development["Fire_Risk_Int"].to_numpy(dtype=int)  # 0/1/2
    group_values = groups.to_numpy()

    group_kfold = GroupKFold(n_splits=common.N_SPLITS)

    fold_rows = []
    oof_records = []
    stage_a_pooled_records = []
    stage_b_pooled_records = []

    for fold_number, (train_idx, val_idx) in enumerate(
        group_kfold.split(X_raw, y_cls, groups=group_values), start=1
    ):
        train_groups = set(group_values[train_idx])
        val_groups = set(group_values[val_idx])
        if train_groups & val_groups:
            raise RuntimeError(f"Fold {fold_number} batch overlap.")

        X_fold_train_raw = X_raw.iloc[train_idx].copy()
        X_fold_val_raw = X_raw.iloc[val_idx].copy()
        y_fold_train = y_cls[train_idx]
        y_fold_val = y_cls[val_idx]

        preprocessor = common.make_preprocessor(numeric_features, categorical_features)
        preprocessor.fit(X_fold_train_raw)
        X_fold_train = preprocessor.transform(X_fold_train_raw)
        X_fold_val = preprocessor.transform(X_fold_val_raw)

        # ---------------- STAGE A: Normal(0) vs Abnormal(1) ----------------
        y_fold_train_a = (y_fold_train >= 1).astype(int)
        weights_a = compute_sample_weight(class_weight="balanced", y=y_fold_train_a)

        model_a = XGBClassifier(**xgb_params_binary)
        model_a.fit(X_fold_train, y_fold_train_a, sample_weight=weights_a)

        pred_a_val = model_a.predict(X_fold_val).astype(int)  # 0=Normal, 1=Abnormal
        y_val_a_true = (y_fold_val >= 1).astype(int)

        # ---------------- STAGE B: Warning(0) vs Critical(1) ----------------
        # Trained ONLY on rows whose TRUE label is Abnormal within fold-train.
        abnormal_train_mask = y_fold_train_a == 1
        X_fold_train_abnormal = X_fold_train[abnormal_train_mask]
        y_fold_train_b = (y_fold_train[abnormal_train_mask] == 2).astype(int)  # 1=Critical

        weights_b = compute_sample_weight(class_weight="balanced", y=y_fold_train_b)
        model_b = XGBClassifier(**xgb_params_binary)
        model_b.fit(X_fold_train_abnormal, y_fold_train_b, sample_weight=weights_b)

        # Apply Stage B only to rows Stage A predicted as Abnormal.
        final_pred = np.zeros_like(pred_a_val)  # default Normal = 0
        abnormal_val_mask = pred_a_val == 1
        if np.any(abnormal_val_mask):
            stage_b_pred = model_b.predict(X_fold_val[abnormal_val_mask]).astype(int)  # 0=Warning,1=Critical
            final_pred[abnormal_val_mask] = np.where(stage_b_pred == 1, 2, 1)  # 2=Critical, 1=Warning
        # rows predicted Normal by Stage A stay 0 (Normal)

        fold_metrics = common.full_classification_metrics(y_fold_val, final_pred)
        fold_metrics["Fold"] = fold_number
        fold_metrics["Validation_Batches"] = " | ".join(sorted(val_groups))
        fold_metrics.update(stage_a_binary_metrics(y_val_a_true, pred_a_val))

        # Stage-B evaluation restricted to rows whose TRUE label is abnormal
        # (shows Stage-B's own Warning-vs-Critical performance in isolation,
        # independent of Stage-A's errors).
        true_abnormal_val_mask = y_val_a_true == 1
        if np.any(true_abnormal_val_mask):
            # Re-predict Stage B on ALL true-abnormal validation rows
            # (not just those Stage A caught) to isolate Stage-B's own skill.
            stage_b_pred_on_true_abnormal = model_b.predict(
                X_fold_val[true_abnormal_val_mask]
            ).astype(int)
            y_true_b = (y_fold_val[true_abnormal_val_mask] == 2).astype(int)
            fold_metrics.update(stage_b_binary_metrics(y_true_b, stage_b_pred_on_true_abnormal))
            stage_b_pooled_records.append(pd.DataFrame({
                "y_true_b": y_true_b, "y_pred_b": stage_b_pred_on_true_abnormal, "Fold": fold_number,
            }))
        else:
            for key in ["StageB_Critical_Recall", "StageB_Critical_Precision", "StageB_Warning_Recall"]:
                fold_metrics[key] = np.nan

        fold_rows.append(fold_metrics)
        stage_a_pooled_records.append(pd.DataFrame({
            "y_true_a": y_val_a_true, "y_pred_a": pred_a_val, "Fold": fold_number,
        }))

        for local_pos, row_idx in enumerate(val_idx):
            oof_records.append({
                "Original_Row_Index": int(row_idx),
                "Fold": fold_number,
                "Batch_ID": str(group_values[row_idx]),
                "y_true_int": int(y_cls[row_idx]),
                "y_pred_int": int(final_pred[local_pos]),
                "StageA_Pred": int(pred_a_val[local_pos]),
            })

        print(f"\nFold {fold_number}/{common.N_SPLITS} | Validation batches: {', '.join(sorted(val_groups))}")
        print(f"  Stage-A Abnormal Recall/Precision: {fold_metrics['Abnormal_Recall']:.4f} / {fold_metrics['Abnormal_Precision']:.4f}")
        print(f"  Stage-B Critical Recall (on true-abnormal subset): {fold_metrics.get('StageB_Critical_Recall')}")
        print(f"  Combined 3-class Critical Recall: {fold_metrics['Critical_Recall']}")
        print(f"  Combined 3-class Warning Recall : {fold_metrics['Warning_Recall']}")

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_records).sort_values("Original_Row_Index").reset_index(drop=True)

    pooled = common.full_classification_metrics(
        oof_df["y_true_int"].to_numpy(), oof_df["y_pred_int"].to_numpy()
    )
    pooled["Candidate"] = "Two_Stage_XGB"
    cm = common.confusion_df(oof_df["y_true_int"].to_numpy(), oof_df["y_pred_int"].to_numpy())
    batch_df = common.batch_classification_metrics(oof_df)

    stage_a_all = pd.concat(stage_a_pooled_records, ignore_index=True)
    pooled_stage_a = stage_a_binary_metrics(stage_a_all["y_true_a"], stage_a_all["y_pred_a"])

    if stage_b_pooled_records:
        stage_b_all = pd.concat(stage_b_pooled_records, ignore_index=True)
        pooled_stage_b = stage_b_binary_metrics(stage_b_all["y_true_b"], stage_b_all["y_pred_b"])
    else:
        pooled_stage_b = {}

    print("\n" + "=" * 78)
    print("TWO-STAGE CANDIDATE - POOLED OOF (COMBINED 3-CLASS)")
    print("=" * 78)
    print(f"Critical Recall    : {pooled['Critical_Recall']}")
    print(f"Critical Precision : {pooled['Critical_Precision']}")
    print(f"Critical F2        : {pooled['Critical_F2']}")
    print(f"Warning Recall     : {pooled['Warning_Recall']}")
    print(f"Normal Recall      : {pooled['Normal_Recall']}")
    print(f"Macro F1 (supp.)   : {pooled['Supported_Macro_F1']}")
    print(f"Balanced Accuracy  : {pooled['Supported_Balanced_Accuracy']}")
    print("\nStage-A pooled (Normal vs Abnormal):")
    for k, v in pooled_stage_a.items():
        print(f"  {k}: {v}")
    print("\nStage-B pooled (Warning vs Critical, on true-abnormal rows only):")
    for k, v in pooled_stage_b.items():
        print(f"  {k}: {v}")
    print("\nConfusion matrix (combined 3-class):")
    print(cm.to_string())

    # -------------------- SAVE OUTPUTS --------------------
    fold_df.to_csv(common.OUT_DIR / "04_two_stage_fold_metrics.csv", index=False)
    oof_df.to_csv(common.OUT_DIR / "04_two_stage_oof_predictions.csv", index=False)
    batch_df.to_csv(common.OUT_DIR / "04_two_stage_batch_metrics.csv", index=False)
    cm.to_csv(common.OUT_DIR / "04_two_stage_confusion_matrix.csv", index=True)

    combined_summary = dict(pooled)
    combined_summary.update({f"StageA_{k}": v for k, v in pooled_stage_a.items()})
    combined_summary.update({f"StageB_Pooled_{k}": v for k, v in pooled_stage_b.items()})
    pd.Series(combined_summary).to_json(common.OUT_DIR / "04_two_stage_pooled_summary.json", indent=2)

    print(f"\nOutputs saved to: {common.OUT_DIR}")

    return {
        "pooled": pooled,
        "pooled_stage_a": pooled_stage_a,
        "pooled_stage_b": pooled_stage_b,
        "fold_df": fold_df,
        "batch_df": batch_df,
    }


if __name__ == "__main__":
    run()
