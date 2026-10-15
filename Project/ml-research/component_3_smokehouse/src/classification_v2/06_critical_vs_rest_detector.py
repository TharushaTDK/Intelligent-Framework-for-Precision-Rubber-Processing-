"""
CLASSIFICATION V2 - FINAL ROUND - EXPERIMENT D
DEDICATED CRITICAL-vs-REST DETECTOR + EXISTING MULTICLASS LOGIC
====================================================================

Research question: can a classifier trained specifically to distinguish
Critical (1) from Not-Critical (0 = Normal or Warning) identify the rare
Critical condition more effectively than the existing multiclass model,
while preserving acceptable Normal/Warning behavior?

Binary labels (Critical=1 / Rest=0) are created IN MEMORY ONLY, inside this
script, for model fitting and evaluation. They are never written back to
any dataset file.

Binary detector configuration (small, predefined deviation from XGB_03,
documented, not searched):
  - Same tree hyperparameters as XGB_03 (learning_rate, max_depth,
    n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda,
    min_child_weight, tree_method, random_state, n_jobs).
  - objective="binary:logistic" (num_class removed).
  - eval_metric="aucpr" instead of "mlogloss": fixed in advance because
    PR-AUC-oriented training is more appropriate than log-loss for a
    detector whose positive class is ~0.68% of development rows. This is
    the ONE documented, predefined parameter change - not a search.
  - Imbalance handling: scale_pos_weight = n_negative / n_positive,
    computed strictly from the applicable training partition (inner-train
    for threshold selection, full outer-train for the final fit). No
    SMOTE, no row duplication.

Combination rule under test:
  IF P_critical(binary detector) >= selected_threshold:
      combined prediction = Critical
  ELSE:
      combined prediction = argmax(multiclass XGB_03-configuration model)

The multiclass model is RETRAINED inside every fold (inner and outer) using
the exact Stage-30 leakage-safe recipe (fold-local preprocessing, fold-local
balanced sample weights) - the already-exported production
fire_risk_pipeline.pkl is never loaded or used for fold predictions, per
protocol (prevents development leakage through the frozen artifact).

Nested, group-safe threshold selection:
  Outer GroupKFold(5, Batch_ID) -> outer-test fold = unbiased evaluation.
  Inside each outer-training set: inner GroupKFold(<=4, Batch_ID) pools
  inner-validation predictions from both the binary detector and the
  retrained multiclass model. For each predefined grid threshold, the
  COMBINED system's pooled inner metrics are computed. Selection: maximize
  Critical F2, tie-break 1 = higher Critical Precision, tie-break 2 =
  higher combined Warning Recall, tie-break 3 = larger threshold.
  The chosen threshold is then applied, unchanged, to models refit on the
  FULL outer-training set and scored once on the untouched outer-test fold.

Grid (fixed before any outer-fold result was inspected):
  0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50

Test split is never read.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def make_binary_detector_params(base_xgb_params):
    params = dict(base_xgb_params)
    params.pop("num_class", None)
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "aucpr"
    return params


def fit_fold(X_train_raw, y_train_3class, X_eval_raw, numeric_features,
             categorical_features, multiclass_params, binary_params):
    """Fits (preprocessor, multiclass model, binary detector) on the given
    training partition only, returns predictions on X_eval_raw."""
    preprocessor = common.make_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_eval = preprocessor.transform(X_eval_raw)

    # Multiclass baseline, retrained with the exact Stage-30 recipe.
    multiclass_weights = compute_sample_weight(class_weight="balanced", y=y_train_3class)
    multiclass_model = XGBClassifier(**multiclass_params)
    multiclass_model.fit(X_train, y_train_3class, sample_weight=multiclass_weights)
    multiclass_pred_eval = multiclass_model.predict(X_eval).astype(int)

    # Binary Critical-vs-Rest detector (labels created in memory only).
    y_train_binary = (y_train_3class == common.CLASS_TO_INT["Critical"]).astype(int)
    spw = common.scale_pos_weight_from_labels(y_train_binary)
    binary_params_fold = dict(binary_params)
    binary_params_fold["scale_pos_weight"] = spw
    binary_model = XGBClassifier(**binary_params_fold)
    binary_model.fit(X_train, y_train_binary)
    p_critical_eval = binary_model.predict_proba(X_eval)[:, 1]

    return multiclass_pred_eval, p_critical_eval, spw


def combine(p_critical, multiclass_pred, threshold):
    return np.where(p_critical >= threshold, common.CLASS_TO_INT["Critical"], multiclass_pred).astype(int)


def combined_metrics_for_threshold(y_true, p_critical, multiclass_pred, threshold):
    pred = combine(p_critical, multiclass_pred, threshold)
    f2 = common.safe_fbeta(y_true, pred, common.CLASS_TO_INT["Critical"], beta=2.0)
    precision = common.safe_precision(y_true, pred, common.CLASS_TO_INT["Critical"])
    warning_recall = common.safe_recall(y_true, pred, common.CLASS_TO_INT["Warning"])
    return f2, precision, warning_recall


def select_threshold_inner_cv(outer_train_idx, X_raw, y_cls, group_values,
                               numeric_features, categorical_features,
                               multiclass_params, binary_params, grid, n_inner=4):
    pooled_rows = []
    n_inner_splits_run = 0

    for inner_train_idx, inner_val_idx in common.inner_group_kfold_splits(
        outer_train_idx, group_values, n_splits=n_inner
    ):
        inner_train_groups = set(group_values[inner_train_idx])
        inner_val_groups = set(group_values[inner_val_idx])
        if inner_train_groups & inner_val_groups:
            raise RuntimeError("Inner fold batch overlap detected (Experiment D).")

        n_inner_splits_run += 1
        multiclass_pred, p_critical, _ = fit_fold(
            X_raw.iloc[inner_train_idx], y_cls[inner_train_idx], X_raw.iloc[inner_val_idx],
            numeric_features, categorical_features, multiclass_params, binary_params,
        )
        for local_pos, row_idx in enumerate(inner_val_idx):
            pooled_rows.append({
                "y_true": int(y_cls[row_idx]),
                "p_critical": float(p_critical[local_pos]),
                "multiclass_pred": int(multiclass_pred[local_pos]),
            })

    if n_inner_splits_run == 0:
        return grid[-1], pd.DataFrame(), "FALLBACK_INSUFFICIENT_INNER_GROUPS"

    pooled = pd.DataFrame(pooled_rows)
    y_true_inner = pooled["y_true"].to_numpy()
    p_critical_inner = pooled["p_critical"].to_numpy()
    multiclass_pred_inner = pooled["multiclass_pred"].to_numpy()

    if int(np.sum(y_true_inner == common.CLASS_TO_INT["Critical"])) == 0:
        return grid[-1], pd.DataFrame(), "FALLBACK_NO_INNER_CRITICAL_SUPPORT"

    grid_rows = []
    for threshold in grid:
        f2, precision, warning_recall = combined_metrics_for_threshold(
            y_true_inner, p_critical_inner, multiclass_pred_inner, threshold
        )
        grid_rows.append({
            "Threshold": threshold,
            "Inner_Pooled_Critical_F2": f2,
            "Inner_Pooled_Critical_Precision": precision,
            "Inner_Pooled_Combined_Warning_Recall": warning_recall,
        })

    grid_df = pd.DataFrame(grid_rows)
    scored = grid_df.copy()
    scored["F2_key"] = scored["Inner_Pooled_Critical_F2"].fillna(-1.0)
    scored["Precision_key"] = scored["Inner_Pooled_Critical_Precision"].fillna(-1.0)
    scored["Warning_key"] = scored["Inner_Pooled_Combined_Warning_Recall"].fillna(-1.0)
    scored = scored.sort_values(
        by=["F2_key", "Precision_key", "Warning_key", "Threshold"],
        ascending=[False, False, False, False],
    )
    selected_threshold = float(scored.iloc[0]["Threshold"])
    return selected_threshold, grid_df, "SELECTED_VIA_INNER_CV"


def run():
    print("=" * 78)
    print("FINAL ROUND - EXPERIMENT D: CRITICAL-vs-REST DETECTOR + MULTICLASS")
    print("=" * 78)
    print(f"\nThreshold grid: {common.CRITICAL_THRESHOLD_GRID}")
    print("Selection: max inner pooled Critical F2 (combined system); "
          "tie-break Critical Precision; tie-break combined Warning Recall; "
          "tie-break larger threshold.")

    data = common.load_development()
    development = data["development"]
    selected_features = data["selected_features"]
    numeric_features = data["numeric_features"]
    categorical_features = data["categorical_features"]
    groups = data["groups"]

    _, _, xgb_params_multiclass = common.recover_selected_xgb_configuration()
    binary_params = make_binary_detector_params(xgb_params_multiclass)

    print("\nBinary detector parameters (differences from XGB_03 only):")
    print(f"  objective    : {binary_params['objective']} (was multi:softprob)")
    print(f"  eval_metric  : {binary_params['eval_metric']} (was mlogloss)")
    print("  scale_pos_weight: computed per-fold from that fold's training labels only")

    X_raw = development[selected_features].copy()
    y_cls = development["Fire_Risk_Int"].to_numpy(dtype=int)
    group_values = groups.to_numpy()

    group_kfold = GroupKFold(n_splits=common.N_SPLITS)

    fold_rows = []
    oof_records = []
    threshold_log = []
    all_inner_grids = []

    for fold_number, (outer_train_idx, outer_val_idx) in enumerate(
        group_kfold.split(X_raw, y_cls, groups=group_values), start=1
    ):
        outer_train_groups = set(group_values[outer_train_idx])
        outer_val_groups = set(group_values[outer_val_idx])
        if outer_train_groups & outer_val_groups:
            raise RuntimeError(f"Outer fold {fold_number} batch overlap.")

        print(f"\n--- Outer fold {fold_number}/{common.N_SPLITS} "
              f"(validation batches: {', '.join(sorted(outer_val_groups))}) ---")

        selected_threshold, inner_grid_df, selection_status = select_threshold_inner_cv(
            outer_train_idx, X_raw, y_cls, group_values,
            numeric_features, categorical_features,
            xgb_params_multiclass, binary_params,
            common.CRITICAL_THRESHOLD_GRID, n_inner=4,
        )
        inner_grid_df["Outer_Fold"] = fold_number
        all_inner_grids.append(inner_grid_df)
        print(f"  Inner-CV selected threshold: {selected_threshold} ({selection_status})")

        threshold_log.append({
            "Outer_Fold": fold_number,
            "Selected_Threshold": selected_threshold,
            "Selection_Status": selection_status,
            "Outer_Train_Batches": " | ".join(sorted(outer_train_groups)),
            "Outer_Validation_Batches": " | ".join(sorted(outer_val_groups)),
        })

        multiclass_pred_outer, p_critical_outer, outer_spw = fit_fold(
            X_raw.iloc[outer_train_idx], y_cls[outer_train_idx], X_raw.iloc[outer_val_idx],
            numeric_features, categorical_features, xgb_params_multiclass, binary_params,
        )
        y_true_outer = y_cls[outer_val_idx]
        combined_pred_outer = combine(p_critical_outer, multiclass_pred_outer, selected_threshold)

        # Standalone binary-detector decision (for separate binary reporting).
        y_true_binary_outer = (y_true_outer == common.CLASS_TO_INT["Critical"]).astype(int)
        binary_pred_outer = (p_critical_outer >= selected_threshold).astype(int)

        fold_metrics = common.full_classification_metrics(y_true_outer, combined_pred_outer)
        fold_metrics["Fold"] = fold_number
        fold_metrics["Selected_Threshold"] = selected_threshold
        fold_metrics["Outer_Scale_Pos_Weight"] = outer_spw
        fold_metrics["Validation_Batches"] = " | ".join(sorted(outer_val_groups))

        # Binary-only metrics for this fold.
        binary_support = int(np.sum(y_true_binary_outer == 1))
        fold_metrics["Binary_Critical_Recall"] = (
            np.nan if binary_support == 0 else
            float(np.sum((binary_pred_outer == 1) & (y_true_binary_outer == 1)) / binary_support)
        )
        predicted_pos = int(np.sum(binary_pred_outer == 1))
        fold_metrics["Binary_Critical_Precision"] = (
            np.nan if predicted_pos == 0 else
            float(np.sum((binary_pred_outer == 1) & (y_true_binary_outer == 1)) / predicted_pos)
        )
        if binary_support > 0 and len(set(y_true_binary_outer.tolist())) == 2:
            fold_metrics["Binary_ROC_AUC"] = float(roc_auc_score(y_true_binary_outer, p_critical_outer))
            fold_metrics["Binary_PR_AUC"] = float(average_precision_score(y_true_binary_outer, p_critical_outer))
        else:
            fold_metrics["Binary_ROC_AUC"] = np.nan
            fold_metrics["Binary_PR_AUC"] = np.nan

        fold_rows.append(fold_metrics)

        for local_pos, row_idx in enumerate(outer_val_idx):
            oof_records.append({
                "Original_Row_Index": int(row_idx),
                "Fold": fold_number,
                "Batch_ID": str(group_values[row_idx]),
                "y_true_int": int(y_true_outer[local_pos]),
                "y_pred_int": int(combined_pred_outer[local_pos]),
                "multiclass_pred_int": int(multiclass_pred_outer[local_pos]),
                "p_critical_binary": float(p_critical_outer[local_pos]),
                "binary_pred": int(binary_pred_outer[local_pos]),
                "Selected_Threshold": selected_threshold,
            })

        print(f"  Combined Critical Recall   : {fold_metrics['Critical_Recall']}")
        print(f"  Combined Critical Precision: {fold_metrics['Critical_Precision']}")
        print(f"  Combined Warning Recall    : {fold_metrics['Warning_Recall']}")
        print(f"  Binary PR-AUC / ROC-AUC    : {fold_metrics['Binary_PR_AUC']} / {fold_metrics['Binary_ROC_AUC']}")

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_records).sort_values("Original_Row_Index").reset_index(drop=True)
    threshold_log_df = pd.DataFrame(threshold_log)
    inner_grid_all_df = pd.concat(all_inner_grids, ignore_index=True)

    # -------------------- POOLED COMBINED 3-CLASS METRICS --------------------
    pooled = common.full_classification_metrics(
        oof_df["y_true_int"].to_numpy(), oof_df["y_pred_int"].to_numpy()
    )
    pooled["Candidate"] = "Critical_vs_Rest_Combined"
    cm = common.confusion_df(oof_df["y_true_int"].to_numpy(), oof_df["y_pred_int"].to_numpy())
    batch_df = common.batch_classification_metrics(oof_df)

    # -------------------- POOLED STANDALONE BINARY-DETECTOR METRICS --------------------
    y_true_binary_all = (oof_df["y_true_int"] == common.CLASS_TO_INT["Critical"]).astype(int).to_numpy()
    binary_pred_all = oof_df["binary_pred"].to_numpy()
    p_critical_all = oof_df["p_critical_binary"].to_numpy()

    binary_support_all = int(np.sum(y_true_binary_all == 1))
    binary_recall = float(np.sum((binary_pred_all == 1) & (y_true_binary_all == 1)) / binary_support_all)
    predicted_pos_all = int(np.sum(binary_pred_all == 1))
    binary_precision = (
        np.nan if predicted_pos_all == 0 else
        float(np.sum((binary_pred_all == 1) & (y_true_binary_all == 1)) / predicted_pos_all)
    )
    binary_f1 = (
        0.0 if (binary_recall + (binary_precision or 0)) == 0 or np.isnan(binary_precision) else
        2 * binary_precision * binary_recall / (binary_precision + binary_recall)
    )
    binary_f2 = (
        0.0 if np.isnan(binary_precision) or (4 * binary_precision + binary_recall) == 0 else
        5 * binary_precision * binary_recall / (4 * binary_precision + binary_recall)
    )
    binary_roc_auc = float(roc_auc_score(y_true_binary_all, p_critical_all))
    binary_pr_auc = float(average_precision_score(y_true_binary_all, p_critical_all))

    binary_summary = {
        "Candidate": "Critical_vs_Rest_Binary_Detector_Standalone",
        "Binary_Critical_Recall": binary_recall,
        "Binary_Critical_Precision": binary_precision,
        "Binary_Critical_F1": binary_f1,
        "Binary_Critical_F2": binary_f2,
        "Binary_ROC_AUC": binary_roc_auc,
        "Binary_PR_AUC": binary_pr_auc,
        "Predicted_Positive_Count": predicted_pos_all,
        "Actual_Positive_Count": binary_support_all,
        "False_Positives": int(np.sum((binary_pred_all == 1) & (y_true_binary_all == 0))),
        "False_Negatives": int(np.sum((binary_pred_all == 0) & (y_true_binary_all == 1))),
    }

    print("\n" + "=" * 78)
    print("EXPERIMENT D - POOLED RESULTS")
    print("=" * 78)
    print("\nStandalone binary detector (Critical-vs-Rest):")
    for k, v in binary_summary.items():
        print(f"  {k}: {v}")

    print("\nCombined 3-class system:")
    print(f"  Critical Recall   : {pooled['Critical_Recall']}")
    print(f"  Critical Precision: {pooled['Critical_Precision']}")
    print(f"  Critical F2       : {pooled['Critical_F2']}")
    print(f"  Warning Recall    : {pooled['Warning_Recall']}")
    print(f"  Macro F1 (supp.)  : {pooled['Supported_Macro_F1']}")
    print(f"  Balanced Accuracy : {pooled['Supported_Balanced_Accuracy']}")
    print("\nSelected thresholds per outer fold:")
    print(threshold_log_df.to_string(index=False))
    print("\nConfusion matrix (combined):")
    print(cm.to_string())

    # -------------------- SAVE OUTPUTS --------------------
    fold_df.to_csv(common.OUT_DIR / "06_critical_detector_fold_metrics.csv", index=False)
    oof_df.to_csv(common.OUT_DIR / "06_critical_detector_oof_predictions.csv", index=False)
    batch_df.to_csv(common.OUT_DIR / "06_critical_detector_batch_metrics.csv", index=False)
    cm.to_csv(common.OUT_DIR / "06_critical_detector_confusion_matrix.csv", index=True)
    threshold_log_df.to_csv(common.OUT_DIR / "06_critical_detector_thresholds.csv", index=False)
    inner_grid_all_df.to_csv(common.OUT_DIR / "06_critical_detector_inner_cv_grid_detail.csv", index=False)
    pd.Series(pooled).to_json(common.OUT_DIR / "06_critical_detector_pooled_metrics.json", indent=2)
    pd.Series(binary_summary).to_json(common.OUT_DIR / "06_critical_detector_binary_standalone_metrics.json", indent=2)

    # Combined summary CSV merging both views (one row) for the final comparison script.
    combined_row = dict(pooled)
    combined_row.update({f"StandaloneBinary_{k}": v for k, v in binary_summary.items() if k != "Candidate"})
    pd.DataFrame([combined_row]).to_csv(common.OUT_DIR / "06_critical_detector_pooled_metrics.csv", index=False)

    print(f"\nOutputs saved to: {common.OUT_DIR}")

    return {
        "pooled": pooled,
        "binary_summary": binary_summary,
        "fold_df": fold_df,
        "batch_df": batch_df,
        "threshold_log_df": threshold_log_df,
    }


if __name__ == "__main__":
    run()
