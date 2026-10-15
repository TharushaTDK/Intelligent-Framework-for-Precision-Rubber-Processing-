"""
CLASSIFICATION V2 - FINAL ROUND - EXPERIMENT E
ORDINAL SEVERITY CLASSIFICATION (Normal < Warning < Critical)
====================================================================

Cumulative ordinal design:

  O1: P(Y >= Warning)   target: Normal=0, Warning=1, Critical=1
  O2: P(Y >= Critical)  target: Normal=0, Warning=0, Critical=1

Both are independent binary XGBoost classifiers (binarized XGB_03
hyperparameters: objective="binary:logistic", eval_metric="logloss",
num_class removed - kept consistent with the two-stage experiment's
binarization convention from the previous V2 round). Class weights for
each binary target are computed with compute_sample_weight("balanced")
strictly from that fold's own training labels - never globally, never
from the outer-validation/held-out partition.

Probability reconstruction (documented exactly, per protocol):

  p_ge_warning  = O1.predict_proba[:, 1]
  p_ge_critical = O2.predict_proba[:, 1]

  Ordinal consistency requires P(Y>=Critical) <= P(Y>=Warning). Any
  violation is corrected deterministically:

      p_ge_critical_corrected = min(p_ge_critical, p_ge_warning)

  Then:
      P(Normal)   = 1 - p_ge_warning
      P(Warning)  = p_ge_warning - p_ge_critical_corrected
      P(Critical) = p_ge_critical_corrected

  All three are clipped to >= 0 for floating-point safety and verified to
  sum to 1 within 1e-6 tolerance for every row.

Two candidates are reported:

  1. Ordinal-Argmax: argmax(P(Normal), P(Warning), P(Critical)). No tuning.
  2. Ordinal-Threshold (at most ONE nested threshold experiment, same
     grid/inner-CV/tie-break convention as Experiment D, applied to
     p_ge_critical_corrected): if p_ge_critical_corrected >= threshold,
     predict Critical; else argmax(P(Normal), P(Warning)) restricted to
     those two classes.

Test split is never read. Group-safe outer/inner GroupKFold throughout.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def make_ordinal_binary_params(base_xgb_params):
    params = dict(base_xgb_params)
    params.pop("num_class", None)
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    return params


def fit_ordinal_stage(X_train_raw, y_train_3class, X_eval_raw, numeric_features,
                       categorical_features, ordinal_params):
    """Fits the shared preprocessor + O1 + O2 on the given training
    partition only; returns reconstructed probabilities on X_eval_raw."""
    preprocessor = common.make_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_eval = preprocessor.transform(X_eval_raw)

    y_ge_warning = (y_train_3class >= common.CLASS_TO_INT["Warning"]).astype(int)
    y_ge_critical = (y_train_3class >= common.CLASS_TO_INT["Critical"]).astype(int)

    w1 = compute_sample_weight(class_weight="balanced", y=y_ge_warning)
    model_o1 = XGBClassifier(**ordinal_params)
    model_o1.fit(X_train, y_ge_warning, sample_weight=w1)

    w2 = compute_sample_weight(class_weight="balanced", y=y_ge_critical)
    model_o2 = XGBClassifier(**ordinal_params)
    model_o2.fit(X_train, y_ge_critical, sample_weight=w2)

    p_ge_warning = model_o1.predict_proba(X_eval)[:, 1]
    p_ge_critical_raw = model_o2.predict_proba(X_eval)[:, 1]

    return p_ge_warning, p_ge_critical_raw


def reconstruct_probabilities(p_ge_warning, p_ge_critical_raw):
    monotonicity_violations = int(np.sum(p_ge_critical_raw > p_ge_warning))
    p_ge_critical_corrected = np.minimum(p_ge_critical_raw, p_ge_warning)

    p_normal = np.clip(1.0 - p_ge_warning, 0.0, 1.0)
    p_critical = np.clip(p_ge_critical_corrected, 0.0, 1.0)
    p_warning = np.clip(p_ge_warning - p_ge_critical_corrected, 0.0, 1.0)

    total = p_normal + p_warning + p_critical
    # Renormalize only to absorb floating-point rounding (should be ~1 already).
    total_safe = np.where(total == 0, 1.0, total)
    p_normal, p_warning, p_critical = p_normal / total_safe, p_warning / total_safe, p_critical / total_safe

    max_sum_error = float(np.max(np.abs((p_normal + p_warning + p_critical) - 1.0)))
    if max_sum_error > 1e-6:
        raise RuntimeError(f"Ordinal probability reconstruction sum error too large: {max_sum_error}")

    return p_normal, p_warning, p_critical, p_ge_critical_corrected, monotonicity_violations


def argmax_decision(p_normal, p_warning, p_critical):
    stacked = np.stack([p_normal, p_warning, p_critical], axis=1)  # columns in class-id order 0,1,2
    return np.argmax(stacked, axis=1).astype(int)


def threshold_decision(p_normal, p_warning, p_critical_corrected, threshold):
    normal_vs_warning = np.where(p_normal >= p_warning, 0, 1)  # argmax among {Normal, Warning} only
    return np.where(p_critical_corrected >= threshold, common.CLASS_TO_INT["Critical"], normal_vs_warning).astype(int)


def select_ordinal_threshold_inner_cv(outer_train_idx, X_raw, y_cls, group_values,
                                       numeric_features, categorical_features,
                                       ordinal_params, grid, n_inner=4):
    pooled_rows = []
    n_inner_splits_run = 0

    for inner_train_idx, inner_val_idx in common.inner_group_kfold_splits(
        outer_train_idx, group_values, n_splits=n_inner
    ):
        inner_train_groups = set(group_values[inner_train_idx])
        inner_val_groups = set(group_values[inner_val_idx])
        if inner_train_groups & inner_val_groups:
            raise RuntimeError("Inner fold batch overlap detected (Experiment E).")

        n_inner_splits_run += 1
        p_ge_warning, p_ge_critical_raw = fit_ordinal_stage(
            X_raw.iloc[inner_train_idx], y_cls[inner_train_idx], X_raw.iloc[inner_val_idx],
            numeric_features, categorical_features, ordinal_params,
        )
        p_normal, p_warning, p_critical, p_critical_corr, _ = reconstruct_probabilities(
            p_ge_warning, p_ge_critical_raw
        )
        for local_pos, row_idx in enumerate(inner_val_idx):
            pooled_rows.append({
                "y_true": int(y_cls[row_idx]),
                "p_normal": float(p_normal[local_pos]),
                "p_warning": float(p_warning[local_pos]),
                "p_critical_corrected": float(p_critical_corr[local_pos]),
            })

    if n_inner_splits_run == 0:
        return grid[-1], pd.DataFrame(), "FALLBACK_INSUFFICIENT_INNER_GROUPS"

    pooled = pd.DataFrame(pooled_rows)
    y_true_inner = pooled["y_true"].to_numpy()

    if int(np.sum(y_true_inner == common.CLASS_TO_INT["Critical"])) == 0:
        return grid[-1], pd.DataFrame(), "FALLBACK_NO_INNER_CRITICAL_SUPPORT"

    grid_rows = []
    for threshold in grid:
        pred = threshold_decision(
            pooled["p_normal"].to_numpy(), pooled["p_warning"].to_numpy(),
            pooled["p_critical_corrected"].to_numpy(), threshold,
        )
        f2 = common.safe_fbeta(y_true_inner, pred, common.CLASS_TO_INT["Critical"], beta=2.0)
        precision = common.safe_precision(y_true_inner, pred, common.CLASS_TO_INT["Critical"])
        warning_recall = common.safe_recall(y_true_inner, pred, common.CLASS_TO_INT["Warning"])
        grid_rows.append({
            "Threshold": threshold,
            "Inner_Pooled_Critical_F2": f2,
            "Inner_Pooled_Critical_Precision": precision,
            "Inner_Pooled_Warning_Recall": warning_recall,
        })

    grid_df = pd.DataFrame(grid_rows)
    scored = grid_df.copy()
    scored["F2_key"] = scored["Inner_Pooled_Critical_F2"].fillna(-1.0)
    scored["Precision_key"] = scored["Inner_Pooled_Critical_Precision"].fillna(-1.0)
    scored["Warning_key"] = scored["Inner_Pooled_Warning_Recall"].fillna(-1.0)
    scored = scored.sort_values(
        by=["F2_key", "Precision_key", "Warning_key", "Threshold"],
        ascending=[False, False, False, False],
    )
    selected_threshold = float(scored.iloc[0]["Threshold"])
    return selected_threshold, grid_df, "SELECTED_VIA_INNER_CV"


def run():
    print("=" * 78)
    print("FINAL ROUND - EXPERIMENT E: ORDINAL SEVERITY CLASSIFICATION")
    print("=" * 78)

    data = common.load_development()
    development = data["development"]
    selected_features = data["selected_features"]
    numeric_features = data["numeric_features"]
    categorical_features = data["categorical_features"]
    groups = data["groups"]

    _, _, xgb_params_multiclass = common.recover_selected_xgb_configuration()
    ordinal_params = make_ordinal_binary_params(xgb_params_multiclass)

    print("\nOrdinal binary-stage parameters (differences from XGB_03 only):")
    print(f"  objective  : {ordinal_params['objective']} (was multi:softprob)")
    print(f"  eval_metric: {ordinal_params['eval_metric']} (was mlogloss)")
    print("  Both O1 (Y>=Warning) and O2 (Y>=Critical) use compute_sample_weight('balanced'),"
          " computed independently per fold from that fold's own binary target distribution.")

    X_raw = development[selected_features].copy()
    y_cls = development["Fire_Risk_Int"].to_numpy(dtype=int)
    group_values = groups.to_numpy()

    group_kfold = GroupKFold(n_splits=common.N_SPLITS)

    argmax_fold_rows = []
    argmax_oof_records = []
    thr_fold_rows = []
    thr_oof_records = []
    threshold_log = []
    all_inner_grids = []
    total_monotonicity_violations = 0
    total_eval_rows = 0

    for fold_number, (outer_train_idx, outer_val_idx) in enumerate(
        group_kfold.split(X_raw, y_cls, groups=group_values), start=1
    ):
        outer_train_groups = set(group_values[outer_train_idx])
        outer_val_groups = set(group_values[outer_val_idx])
        if outer_train_groups & outer_val_groups:
            raise RuntimeError(f"Outer fold {fold_number} batch overlap.")

        print(f"\n--- Outer fold {fold_number}/{common.N_SPLITS} "
              f"(validation batches: {', '.join(sorted(outer_val_groups))}) ---")

        # ---- Ordinal-Argmax candidate: fit once on full outer-train ----
        p_ge_warning, p_ge_critical_raw = fit_ordinal_stage(
            X_raw.iloc[outer_train_idx], y_cls[outer_train_idx], X_raw.iloc[outer_val_idx],
            numeric_features, categorical_features, ordinal_params,
        )
        p_normal, p_warning, p_critical, p_critical_corr, violations = reconstruct_probabilities(
            p_ge_warning, p_ge_critical_raw
        )
        total_monotonicity_violations += violations
        total_eval_rows += len(outer_val_idx)

        y_true_outer = y_cls[outer_val_idx]
        argmax_pred = argmax_decision(p_normal, p_warning, p_critical)

        argmax_metrics = common.full_classification_metrics(y_true_outer, argmax_pred)
        argmax_metrics["Fold"] = fold_number
        argmax_metrics["Validation_Batches"] = " | ".join(sorted(outer_val_groups))
        argmax_metrics["Monotonicity_Violations"] = violations
        argmax_metrics["Eval_Rows"] = len(outer_val_idx)
        argmax_fold_rows.append(argmax_metrics)

        for local_pos, row_idx in enumerate(outer_val_idx):
            argmax_oof_records.append({
                "Original_Row_Index": int(row_idx),
                "Fold": fold_number,
                "Batch_ID": str(group_values[row_idx]),
                "y_true_int": int(y_true_outer[local_pos]),
                "y_pred_int": int(argmax_pred[local_pos]),
                "p_normal": float(p_normal[local_pos]),
                "p_warning": float(p_warning[local_pos]),
                "p_critical": float(p_critical[local_pos]),
            })

        print(f"  [Ordinal-Argmax] Critical Recall: {argmax_metrics['Critical_Recall']}  "
              f"Warning Recall: {argmax_metrics['Warning_Recall']}  "
              f"Monotonicity violations: {violations}/{len(outer_val_idx)}")

        # ---- Ordinal-Threshold candidate: ONE nested threshold experiment ----
        selected_threshold, inner_grid_df, selection_status = select_ordinal_threshold_inner_cv(
            outer_train_idx, X_raw, y_cls, group_values,
            numeric_features, categorical_features, ordinal_params,
            common.CRITICAL_THRESHOLD_GRID, n_inner=4,
        )
        inner_grid_df["Outer_Fold"] = fold_number
        all_inner_grids.append(inner_grid_df)
        print(f"  [Ordinal-Threshold] selected threshold: {selected_threshold} ({selection_status})")

        threshold_log.append({
            "Outer_Fold": fold_number,
            "Selected_Threshold": selected_threshold,
            "Selection_Status": selection_status,
        })

        thr_pred = threshold_decision(p_normal, p_warning, p_critical_corr, selected_threshold)
        thr_metrics = common.full_classification_metrics(y_true_outer, thr_pred)
        thr_metrics["Fold"] = fold_number
        thr_metrics["Selected_Threshold"] = selected_threshold
        thr_metrics["Validation_Batches"] = " | ".join(sorted(outer_val_groups))
        thr_fold_rows.append(thr_metrics)

        for local_pos, row_idx in enumerate(outer_val_idx):
            thr_oof_records.append({
                "Original_Row_Index": int(row_idx),
                "Fold": fold_number,
                "Batch_ID": str(group_values[row_idx]),
                "y_true_int": int(y_true_outer[local_pos]),
                "y_pred_int": int(thr_pred[local_pos]),
                "Selected_Threshold": selected_threshold,
            })

        print(f"  [Ordinal-Threshold] Critical Recall: {thr_metrics['Critical_Recall']}  "
              f"Warning Recall: {thr_metrics['Warning_Recall']}")

    # -------------------- POOL: ORDINAL-ARGMAX --------------------
    argmax_fold_df = pd.DataFrame(argmax_fold_rows)
    argmax_oof_df = pd.DataFrame(argmax_oof_records).sort_values("Original_Row_Index").reset_index(drop=True)
    argmax_pooled = common.full_classification_metrics(
        argmax_oof_df["y_true_int"].to_numpy(), argmax_oof_df["y_pred_int"].to_numpy()
    )
    argmax_pooled["Candidate"] = "Ordinal_Argmax"
    argmax_cm = common.confusion_df(argmax_oof_df["y_true_int"].to_numpy(), argmax_oof_df["y_pred_int"].to_numpy())
    argmax_batch_df = common.batch_classification_metrics(argmax_oof_df)

    # -------------------- POOL: ORDINAL-THRESHOLD --------------------
    thr_fold_df = pd.DataFrame(thr_fold_rows)
    thr_oof_df = pd.DataFrame(thr_oof_records).sort_values("Original_Row_Index").reset_index(drop=True)
    thr_pooled = common.full_classification_metrics(
        thr_oof_df["y_true_int"].to_numpy(), thr_oof_df["y_pred_int"].to_numpy()
    )
    thr_pooled["Candidate"] = "Ordinal_Threshold"
    thr_cm = common.confusion_df(thr_oof_df["y_true_int"].to_numpy(), thr_oof_df["y_pred_int"].to_numpy())
    thr_batch_df = common.batch_classification_metrics(thr_oof_df)
    threshold_log_df = pd.DataFrame(threshold_log)
    inner_grid_all_df = pd.concat(all_inner_grids, ignore_index=True)

    print("\n" + "=" * 78)
    print("EXPERIMENT E - POOLED RESULTS")
    print("=" * 78)
    print(f"\nTotal monotonicity violations (P(Y>=Critical) > P(Y>=Warning) before correction): "
          f"{total_monotonicity_violations} / {total_eval_rows} rows "
          f"({100*total_monotonicity_violations/total_eval_rows:.2f}%)")

    print("\nOrdinal-Argmax pooled:")
    print(f"  Critical Recall   : {argmax_pooled['Critical_Recall']}")
    print(f"  Critical Precision: {argmax_pooled['Critical_Precision']}")
    print(f"  Critical F2       : {argmax_pooled['Critical_F2']}")
    print(f"  Warning Recall    : {argmax_pooled['Warning_Recall']}")
    print(f"  Macro F1 (supp.)  : {argmax_pooled['Supported_Macro_F1']}")
    print(f"  Balanced Accuracy : {argmax_pooled['Supported_Balanced_Accuracy']}")
    print("\nOrdinal-Threshold pooled:")
    print(f"  Critical Recall   : {thr_pooled['Critical_Recall']}")
    print(f"  Critical Precision: {thr_pooled['Critical_Precision']}")
    print(f"  Critical F2       : {thr_pooled['Critical_F2']}")
    print(f"  Warning Recall    : {thr_pooled['Warning_Recall']}")
    print(f"  Macro F1 (supp.)  : {thr_pooled['Supported_Macro_F1']}")
    print(f"  Balanced Accuracy : {thr_pooled['Supported_Balanced_Accuracy']}")
    print("\nSelected thresholds per outer fold (Ordinal-Threshold):")
    print(threshold_log_df.to_string(index=False))
    print("\nOrdinal-Argmax confusion matrix:")
    print(argmax_cm.to_string())
    print("\nOrdinal-Threshold confusion matrix:")
    print(thr_cm.to_string())

    # -------------------- SAVE OUTPUTS --------------------
    argmax_fold_df.to_csv(common.OUT_DIR / "07_ordinal_argmax_fold_metrics.csv", index=False)
    argmax_oof_df.to_csv(common.OUT_DIR / "07_ordinal_argmax_oof_predictions.csv", index=False)
    argmax_batch_df.to_csv(common.OUT_DIR / "07_ordinal_argmax_batch_metrics.csv", index=False)
    argmax_cm.to_csv(common.OUT_DIR / "07_ordinal_argmax_confusion_matrix.csv", index=True)
    pd.Series(argmax_pooled).to_json(common.OUT_DIR / "07_ordinal_argmax_pooled_metrics.json", indent=2)
    pd.DataFrame([argmax_pooled]).to_csv(common.OUT_DIR / "07_ordinal_pooled_metrics.csv", index=False)

    thr_fold_df.to_csv(common.OUT_DIR / "07_ordinal_threshold_fold_metrics.csv", index=False)
    thr_oof_df.to_csv(common.OUT_DIR / "07_ordinal_threshold_oof_predictions.csv", index=False)
    thr_batch_df.to_csv(common.OUT_DIR / "07_ordinal_threshold_batch_metrics.csv", index=False)
    thr_cm.to_csv(common.OUT_DIR / "07_ordinal_threshold_confusion_matrix.csv", index=True)
    pd.Series(thr_pooled).to_json(common.OUT_DIR / "07_ordinal_threshold_pooled_metrics.json", indent=2)
    pd.DataFrame([thr_pooled]).to_csv(
        common.OUT_DIR / "07_ordinal_pooled_metrics.csv", index=False, mode="a", header=False
    )

    threshold_log_df.to_csv(common.OUT_DIR / "07_ordinal_threshold_selected_per_outer_fold.csv", index=False)
    inner_grid_all_df.to_csv(common.OUT_DIR / "07_ordinal_threshold_inner_cv_grid_detail.csv", index=False)

    with open(common.OUT_DIR / "07_ordinal_monotonicity_summary.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Monotonicity violations (P(Y>=Critical) > P(Y>=Warning) before correction): "
            f"{total_monotonicity_violations} / {total_eval_rows} rows "
            f"({100*total_monotonicity_violations/total_eval_rows:.4f}%)\n"
            f"Correction applied: p_ge_critical_corrected = min(p_ge_critical, p_ge_warning)\n"
        )

    print(f"\nOutputs saved to: {common.OUT_DIR}")

    return {
        "argmax_pooled": argmax_pooled,
        "thr_pooled": thr_pooled,
        "argmax_fold_df": argmax_fold_df,
        "thr_fold_df": thr_fold_df,
        "argmax_batch_df": argmax_batch_df,
        "thr_batch_df": thr_batch_df,
        "monotonicity_violations": total_monotonicity_violations,
        "eval_rows": total_eval_rows,
    }


if __name__ == "__main__":
    run()
