"""
CLASSIFICATION V2 - PHASE 5: EXPERIMENT B - CRITICAL PROBABILITY THRESHOLD
============================================================================

Decision rule under test (applied AFTER standard argmax-trained XGBoost):

    if P(Critical) >= critical_threshold:
        predict Critical
    else:
        predict argmax(P(Normal), P(Warning))

Leakage control (nested, group-safe):

    OUTER GroupKFold (5 splits, Batch_ID) -> held-out groups = unbiased
    candidate evaluation set, exactly as in Stage 30 / 01_reproduce_baseline.

    Inside EACH outer-training set only:
        INNER GroupKFold over the outer-training batches -> inner-validation
        predictions are pooled across inner folds, and the Critical
        threshold is selected using ONLY this pooled inner-OOF evidence.

    The selected threshold is then applied, UNCHANGED, to a model refit on
    the FULL outer-training set and evaluated on the untouched outer-test
    fold. The outer fold never influences its own threshold choice.

Threshold grid (fixed BEFORE any outer-fold result was inspected):
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50

Selection criterion (fixed BEFORE running): maximize pooled inner-OOF
Critical F2 (recall-weighted, consistent with the safety priority on
Critical detection). Tie-break 1: higher Critical Precision. Tie-break 2
(exact ties): prefer the LARGER threshold (more conservative, fewer
false Critical alerts).

A separate, clearly-labeled DESCRIPTIVE sweep (not a tuned candidate) is
also produced for the Critical Recall vs Precision plot: it applies every
grid threshold post-hoc to the standard (non-nested) 01_reproduce_baseline
OOF probabilities. This is leakage-safe to compute (threshold application
does not touch training) but must NOT be read as "the chosen threshold",
since picking the best point on that curve would use outer-fold evidence
for selection. Only the nested candidate below counts as a tuned result.

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


def apply_threshold(proba, threshold):
    """proba columns are [P(Normal), P(Warning), P(Critical)] (class id order)."""
    baseline_01 = np.argmax(proba[:, :2], axis=1)  # 0 or 1
    critical_mask = proba[:, 2] >= threshold
    return np.where(critical_mask, 2, baseline_01).astype(int)


def fit_predict_proba(X_train_raw, y_train, X_eval_raw, numeric_features, categorical_features, xgb_params):
    preprocessor = common.make_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_eval = preprocessor.transform(X_eval_raw)

    weights = common.fold_sample_weight(y_train, critical_multiplier=1.0)

    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train, sample_weight=weights)
    return model.predict_proba(X_eval)


def select_threshold_from_inner_cv(outer_train_idx, X_raw, y_cls, group_values,
                                    numeric_features, categorical_features, xgb_params,
                                    grid, inner_splits=4):
    inner_groups_all = group_values[outer_train_idx]
    unique_inner_groups = sorted(set(inner_groups_all))
    n_inner = min(inner_splits, len(unique_inner_groups))

    if n_inner < 2:
        # Not enough distinct batches to run inner CV; fall back to the
        # most conservative grid value and document it.
        return grid[-1], pd.DataFrame(), "FALLBACK_INSUFFICIENT_INNER_GROUPS"

    inner_kfold = GroupKFold(n_splits=n_inner)

    # Local index space: positions within outer_train_idx
    local_X = X_raw.iloc[outer_train_idx].reset_index(drop=True)
    local_y = y_cls[outer_train_idx]
    local_groups = group_values[outer_train_idx]

    pooled_proba_rows = []

    for inner_train_pos, inner_val_pos in inner_kfold.split(local_X, local_y, groups=local_groups):
        inner_train_groups = set(local_groups[inner_train_pos])
        inner_val_groups = set(local_groups[inner_val_pos])
        if inner_train_groups & inner_val_groups:
            raise RuntimeError("Inner fold batch overlap detected.")

        proba = fit_predict_proba(
            local_X.iloc[inner_train_pos], local_y[inner_train_pos],
            local_X.iloc[inner_val_pos],
            numeric_features, categorical_features, xgb_params,
        )

        for local_pos, pos in enumerate(inner_val_pos):
            pooled_proba_rows.append({
                "y_true": int(local_y[pos]),
                "P_Normal": float(proba[local_pos, 0]),
                "P_Warning": float(proba[local_pos, 1]),
                "P_Critical": float(proba[local_pos, 2]),
            })

    pooled_inner = pd.DataFrame(pooled_proba_rows)
    proba_matrix = pooled_inner[["P_Normal", "P_Warning", "P_Critical"]].to_numpy()
    y_true_inner = pooled_inner["y_true"].to_numpy()

    grid_rows = []
    for threshold in grid:
        pred = apply_threshold(proba_matrix, threshold)
        f2 = common.safe_fbeta(y_true_inner, pred, 2, beta=2.0)
        precision = common.safe_precision(y_true_inner, pred, 2)
        recall = common.safe_recall(y_true_inner, pred, 2)
        grid_rows.append({
            "Threshold": threshold,
            "Inner_Pooled_Critical_F2": f2,
            "Inner_Pooled_Critical_Precision": precision,
            "Inner_Pooled_Critical_Recall": recall,
        })

    grid_df = pd.DataFrame(grid_rows)

    # Selection: max F2, tie-break higher precision, tie-break larger threshold.
    scored = grid_df.copy()
    scored["F2_rank_key"] = scored["Inner_Pooled_Critical_F2"].fillna(-1.0)
    scored["Precision_rank_key"] = scored["Inner_Pooled_Critical_Precision"].fillna(-1.0)

    if scored["F2_rank_key"].max() <= -1.0 + 1e-12:
        # No inner Critical support anywhere in this outer-training set.
        return grid[-1], grid_df, "FALLBACK_NO_INNER_CRITICAL_SUPPORT"

    scored = scored.sort_values(
        by=["F2_rank_key", "Precision_rank_key", "Threshold"],
        ascending=[False, False, False],
    )
    selected_threshold = float(scored.iloc[0]["Threshold"])

    return selected_threshold, grid_df, "SELECTED_VIA_INNER_CV"


def run():
    print("=" * 78)
    print("CLASSIFICATION V2 - PHASE 5: EXPERIMENT B - CRITICAL THRESHOLD (NESTED CV)")
    print("=" * 78)
    print(f"\nThreshold grid: {common.CRITICAL_THRESHOLD_GRID}")
    print("Selection criterion: max pooled inner-OOF Critical F2, "
          "tie-break higher Critical Precision, tie-break larger threshold.")

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

    group_kfold = GroupKFold(n_splits=common.N_SPLITS)

    outer_fold_rows = []
    oof_records = []
    threshold_selection_log = []
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

        selected_threshold, inner_grid_df, selection_status = select_threshold_from_inner_cv(
            outer_train_idx, X_raw, y_cls, group_values,
            numeric_features, categorical_features, xgb_params,
            common.CRITICAL_THRESHOLD_GRID, inner_splits=4,
        )
        inner_grid_df["Outer_Fold"] = fold_number
        all_inner_grids.append(inner_grid_df)

        print(f"  Inner-CV selected threshold: {selected_threshold} ({selection_status})")

        threshold_selection_log.append({
            "Outer_Fold": fold_number,
            "Selected_Threshold": selected_threshold,
            "Selection_Status": selection_status,
            "Outer_Train_Batches": " | ".join(sorted(outer_train_groups)),
            "Outer_Validation_Batches": " | ".join(sorted(outer_val_groups)),
        })

        # Refit on the FULL outer-training set, apply the selected threshold
        # to the untouched outer-validation fold.
        proba_outer = fit_predict_proba(
            X_raw.iloc[outer_train_idx], y_cls[outer_train_idx],
            X_raw.iloc[outer_val_idx],
            numeric_features, categorical_features, xgb_params,
        )
        y_pred_outer = apply_threshold(proba_outer, selected_threshold)
        y_true_outer = y_cls[outer_val_idx]

        fold_metrics = common.full_classification_metrics(y_true_outer, y_pred_outer)
        fold_metrics["Fold"] = fold_number
        fold_metrics["Selected_Threshold"] = selected_threshold
        fold_metrics["Validation_Batches"] = " | ".join(sorted(outer_val_groups))
        outer_fold_rows.append(fold_metrics)

        for local_pos, row_idx in enumerate(outer_val_idx):
            oof_records.append({
                "Original_Row_Index": int(row_idx),
                "Fold": fold_number,
                "Batch_ID": str(group_values[row_idx]),
                "y_true_int": int(y_true_outer[local_pos]),
                "y_pred_int": int(y_pred_outer[local_pos]),
                "Selected_Threshold": selected_threshold,
            })

        print(f"  Outer Critical Recall   : {fold_metrics['Critical_Recall']}")
        print(f"  Outer Critical Precision: {fold_metrics['Critical_Precision']}")
        print(f"  Outer Warning Recall    : {fold_metrics['Warning_Recall']}")

    fold_df = pd.DataFrame(outer_fold_rows)
    oof_df = pd.DataFrame(oof_records).sort_values("Original_Row_Index").reset_index(drop=True)
    threshold_log_df = pd.DataFrame(threshold_selection_log)
    inner_grid_all_df = pd.concat(all_inner_grids, ignore_index=True)

    pooled = common.full_classification_metrics(
        oof_df["y_true_int"].to_numpy(), oof_df["y_pred_int"].to_numpy()
    )
    pooled["Candidate"] = "Threshold_Tuned_Nested"
    cm = common.confusion_df(oof_df["y_true_int"].to_numpy(), oof_df["y_pred_int"].to_numpy())
    batch_df = common.batch_classification_metrics(oof_df)

    print("\n" + "=" * 78)
    print("NESTED THRESHOLD-TUNED CANDIDATE - POOLED OOF")
    print("=" * 78)
    print(f"Critical Recall    : {pooled['Critical_Recall']}")
    print(f"Critical Precision : {pooled['Critical_Precision']}")
    print(f"Critical F2        : {pooled['Critical_F2']}")
    print(f"Warning Recall     : {pooled['Warning_Recall']}")
    print(f"Macro F1 (supp.)   : {pooled['Supported_Macro_F1']}")
    print(f"Balanced Accuracy  : {pooled['Supported_Balanced_Accuracy']}")
    print(f"False Critical alerts (Predicted Critical, Actual != Critical): "
          f"{pooled['Predicted_Critical_Count'] - cm.loc['Actual_Critical', 'Predicted_Critical']}")
    print("\nSelected thresholds per outer fold:")
    print(threshold_log_df.to_string(index=False))
    print("\nConfusion matrix:")
    print(cm.to_string())

    # -------------------- DESCRIPTIVE (NON-SELECTED) SWEEP --------------------
    # For the Recall-vs-Precision plot ONLY. Uses the standard (non-nested)
    # baseline OOF probabilities from a fresh, identically-structured run,
    # purely to show the shape of the trade-off curve. NOT used to choose
    # a threshold or rank candidates.
    print("\n" + "=" * 78)
    print("DESCRIPTIVE THRESHOLD SWEEP (for plotting only - not a tuned candidate)")
    print("=" * 78)

    sweep_rows = []
    sweep_oof_records = []
    for fold_number, (train_idx, val_idx) in enumerate(
        group_kfold.split(X_raw, y_cls, groups=group_values), start=1
    ):
        proba = fit_predict_proba(
            X_raw.iloc[train_idx], y_cls[train_idx], X_raw.iloc[val_idx],
            numeric_features, categorical_features, xgb_params,
        )
        for local_pos, row_idx in enumerate(val_idx):
            sweep_oof_records.append({
                "Original_Row_Index": int(row_idx),
                "y_true_int": int(y_cls[row_idx]),
                "P_Normal": float(proba[local_pos, 0]),
                "P_Warning": float(proba[local_pos, 1]),
                "P_Critical": float(proba[local_pos, 2]),
            })

    sweep_oof_df = pd.DataFrame(sweep_oof_records)
    sweep_proba = sweep_oof_df[["P_Normal", "P_Warning", "P_Critical"]].to_numpy()
    sweep_y_true = sweep_oof_df["y_true_int"].to_numpy()

    for threshold in common.CRITICAL_THRESHOLD_GRID:
        pred = apply_threshold(sweep_proba, threshold)
        metrics = common.full_classification_metrics(sweep_y_true, pred)
        metrics["Threshold"] = threshold
        sweep_rows.append(metrics)
        print(f"  threshold={threshold:.2f}  Critical Recall={metrics['Critical_Recall']:.4f}  "
              f"Critical Precision={metrics['Critical_Precision']}  "
              f"Warning Recall={metrics['Warning_Recall']:.4f}")

    sweep_df = pd.DataFrame(sweep_rows)

    # -------------------- SAVE OUTPUTS --------------------
    fold_df.to_csv(common.OUT_DIR / "03_threshold_nested_outer_fold_metrics.csv", index=False)
    oof_df.to_csv(common.OUT_DIR / "03_threshold_nested_oof_predictions.csv", index=False)
    batch_df.to_csv(common.OUT_DIR / "03_threshold_nested_batch_metrics.csv", index=False)
    cm.to_csv(common.OUT_DIR / "03_threshold_nested_confusion_matrix.csv", index=True)
    threshold_log_df.to_csv(common.OUT_DIR / "03_threshold_selected_per_outer_fold.csv", index=False)
    inner_grid_all_df.to_csv(common.OUT_DIR / "03_threshold_inner_cv_grid_detail.csv", index=False)
    pd.Series(pooled).to_json(common.OUT_DIR / "03_threshold_nested_pooled_summary.json", indent=2)
    sweep_df.to_csv(common.OUT_DIR / "03_threshold_descriptive_sweep_NOT_A_CANDIDATE.csv", index=False)

    print(f"\nOutputs saved to: {common.OUT_DIR}")

    return {
        "pooled": pooled,
        "fold_df": fold_df,
        "batch_df": batch_df,
        "threshold_log_df": threshold_log_df,
        "sweep_df": sweep_df,
    }


if __name__ == "__main__":
    run()
